import faiss
import timm
from timm.models.hub import push_to_hf_hub
import torch
import queue
import numpy as np
import threading
import json
import PIL
import os
from glob import glob
import re
import cv2
from PIL import Image
import torchvision

import torch
import torch.nn as nn
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
import logging
import os
from torchvision import transforms as T
import numpy as np
logging.getLogger().setLevel(logging.INFO)
from torch.optim import AdamW
import wandb
from collections import defaultdict
import shutil
from huggingface_hub import hf_hub_download, login, upload_file
from tqdm import tqdm

from ..utils import initialize_onnx_model
from ..utils import create_batches
from ..utils import get_transform

from ..utils.recognition.synth_crops import render_all_synth_in_parallel
from ..utils.recognition.datasets import create_dataset, create_render_dataset, create_hn_query_dataset
from ..utils.recognition.transforms import create_paired_transform, INV_NORMALIZE
from ..utils.recognition.custom_schedulers import CosineAnnealingDecWarmRestarts
from ..utils import get_path, dictmerge


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def str_to_ord_str(string):
    return '_'.join([str(ord(char)) for char in string])


def ord_str_to_word(ord_str):
    return ''.join([chr(int(ord_char)) for ord_char in re.split('[_|]', ord_str)])


def get_crop_embeddings(recognizer_engine, crops, num_streams=4):
    # Create batches of word crops
    crop_batches = create_batches(crops)

    input_queue = queue.Queue()
    for i, batch in enumerate(crop_batches):
        input_queue.put((i, batch))
    
    output_queue = queue.Queue()
    threads = []

    # for thread in range(num_streams):
    #     threads.append(RecognizerEngineExecutorThread(recognizer_engine, input_queue, output_queue))

    # for thread in threads:
    #     thread.start()

    # for thread in threads:
    #     thread.join()
    while not input_queue.empty():
        i, batch = input_queue.get()
        output = iteration(recognizer_engine, batch)
        output_queue.put((i, output))

    embeddings = [None] * len(crop_batches)
    while not output_queue.empty():
        i, result = output_queue.get()

        if isinstance(result, list):
            result = result[0]
        embeddings[i] = result 

    # TODO: we don't need to convert to a tensor just to normalize, right?
    embeddings = [torch.nn.functional.normalize(torch.from_numpy(embedding), p=2, dim=1) for embedding in embeddings]
    return embeddings


def iteration(model, input):
    output = model.run(input)
    return output


class RecognizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ):
        super(RecognizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue

    def run(self):
        while not self._input_queue.empty():
            i, batch = self._input_queue.get()
            output = iteration(self._model, batch)
            self._output_queue.put((i, output))


class RecognizerEngine:

    def __init__(self, model, backend, transform, input_name = None):
        self._model = model
        self._backend = backend
        self.transform = transform
        self.input_name = input_name

    def __call__(self, imgs):
        return self.run(imgs)

    def run(self, imgs):
        trans_imgs = []
        for img in imgs:
            try:
                trans_imgs.append(self.transform(img.astype(np.uint8))[0])
            except Exception as e:
                trans_imgs.append(torch.zeros((3, 224, 224)))

        input = torch.nn.functional.pad(torch.stack(trans_imgs), (0, 0, 0, 0, 0, 0, 0, 64 - len(imgs))).numpy()

        if self._backend == 'timm':
            embeddings = self._model(torch.from_numpy(input)).detach().numpy()
        elif self._backend == 'onnx':
            embeddings = self._model.run(None, {self.input_name: input})

        return embeddings


class Recognizer:


    def __init__(self, config, type = 'char'):

        self.config = config
        self.type = type
        self.transform = get_transform(type, self.config['Recognizer'][self.type]['training']['imsize'])
        self.initialize_model()


    def initialize_model(self):
        print('Initializing model...')
        if not self.config['Recognizer'][self.type]['hf_repo_id'] is None:

            print(f"Loading (HF) pretrained {self.type} recognizer model!")

            # Allow users to append to the huggingface hub repo any inner folders they need to naviagte
            # Works because hf hub repos will always be owner/repo_name, so anything after is a path
                
            # remove any leading or trailing slashes
            repo_id = self.config['Recognizer'][self.type]['hf_repo_id'].strip('/')
            # split on slashes
            if len(repo_id.split('/')) == 2:
                fn_prefix = ''
            elif len(repo_id.split('/')) < 2:
                raise ValueError('hf_repo_id must be in the format owner/repo_name, for example: dell-research-harvard/effocr_en')
            else:
                if self.config['Recognizer'][self.type]['model_backend'] == 'timm':
                    raise ValueError('huggingface models loaded by timm must follow standard conventions. Interior files are not allowed. See https://huggingface.co/docs/timm/hf_hub')
                else:
                    fn_prefix = '/'.join(repo_id.split('/')[2:]) + '/' # Careful, order matters here
                    repo_id = '/'.join(repo_id.split('/')[:2])

            if self.config['Recognizer'][self.type]['model_backend'] == 'timm':
                model_suffix = '.pth'
            elif self.config['Recognizer'][self.type]['model_backend'] == 'onnx':
                model_suffix = '.onnx'


            for fn in ['ref.txt', 'ref.index']:    
                if not os.path.exists(os.path.join(self.config['Recognizer'][self.type]['model_dir'], fn_prefix + fn)):            
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=fn_prefix+fn,
                        local_dir=self.config['Recognizer'][self.type]['model_dir'],
                        local_dir_use_symlinks=False,
                        token=self.config['Global']['hf_token_for_upload']
                    )
            
            self.config['Recognizer'][self.type]['model_dir'] = os.path.join(self.config['Recognizer'][self.type]['model_dir'], fn_prefix)
            self.index = faiss.read_index(get_path(self.config['Recognizer'][self.type]['model_dir'], ext="index"))

            with open(get_path(self.config['Recognizer'][self.type]['model_dir'], "txt"), 'r') as f:
                self.candidates = f.read().splitlines()

            if self.type == 'word':
                self.candidates = [ord_str_to_word(candidate) for candidate in self.candidates]

            if self.config['Recognizer'][self.type]['model_backend'] == 'timm':
                self.model = timm.create_model(f"hf-hub:{self.config['Recognizer'][self.type]['hf_repo_id']}", num_classes=0, pretrained=True)
                self.save_model(self.config['Recognizer'][self.type]["model_dir"], self.model, "pretrained", False)
                self.input_name = None
            elif self.config['Recognizer'][self.type]['model_backend'] == 'onnx':
                self.model, self.input_name, _ = initialize_onnx_model(
                    hf_hub_download(repo_id=repo_id,
                        filename=fn_prefix + 'enc_best' + model_suffix,
                        local_dir=self.config['Recognizer'][self.type]['model_dir'],
                        local_dir_use_symlinks=False), 
                    self.config['Recognizer'][self.type])

        elif not get_path(self.config['Recognizer'][self.type]['model_dir'], ext="index") is None:
        
            print(f"Loading (local) pretrained {self.type} recognizer model!")

            self.index = faiss.read_index(get_path(self.config['Recognizer'][self.type]['model_dir'], ext="index"))

            with open(get_path(self.config['Recognizer'][self.type]['model_dir'], "txt"), 'r') as f:
                self.candidates = f.read().splitlines()

            if self.type == 'word':
                self.candidates = [ord_str_to_word(candidate) for candidate in self.candidates]

            if self.config['Recognizer'][self.type]['model_backend'] == 'timm':
                self.model = timm.create_model(self.config['Recognizer'][self.type]['timm_model_name'], num_classes=0, pretrained=True)
                pretrained_dict = torch.load(get_path(self.config['Recognizer'][self.type]['model_dir'], ext="pth"), 
                                             map_location = torch.device(self.config['Recognizer'][self.type]['device']))
                pretrained_dict = {k.replace("net.", ""): v for k, v in pretrained_dict.items()}
                self.model.load_state_dict(pretrained_dict)
                self.input_name = None

            elif self.config['Recognizer'][self.type]['model_backend'] == 'onnx':
                self.model, self.input_name, _ = initialize_onnx_model(
                    get_path(self.config['Recognizer'][self.type]['model_dir'], ext="onnx"), 
                    self.config['Recognizer'][self.type])

        else:
            self.index = None
            self.candidates = None
            self.model = timm.create_model(self.config['Recognizer'][self.type]['timm_model_name'], num_classes=0, pretrained=True)
            self.input_name = None
        
        if self.config['Recognizer'][self.type]['model_backend'] == 'timm' and self.index is not None:
            self.model.eval()
            self.model.to(self.config['Recognizer'][self.type]['device'])
            self.knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
            self.infm = InferenceModel(self.model, knn_func=self.knn_func, data_device='cpu')
            self.infm.load_knn_func(get_path(self.config['Recognizer'][self.type]['model_dir'], ext="index"))


    def __call__(self, images):
        return self.run(images)
    

    def run_simple(self, results):

        # for a given line: [
        #   ((word_img, word_coords), [(char_img, char_coords), ...]), 
        # ...]
        # this function takes as input the first word tuple or the list of char tuples

        if not isinstance(results, list):
            results = [results]

        imgs = [self.transform(x[0].convert('RGB'))[0] for x in results]
        stacked_imgs = torch.stack(imgs)

        distances, indices = self.infm.get_nearest_neighbors(stacked_imgs, k=1)
        distances = [d.item() for d in distances]
        candidates = [self.candidates[index[0]] for index in indices]
       
        return distances, "".join(candidates)


    def run(self, images, cutoff = None):
        
        total_images = len(images)
        embeddings = get_crop_embeddings(RecognizerEngine(self.model, self.config['Recognizer'][self.type]['model_backend'], self.transform, self.input_name), images)
        embeddings = torch.cat(embeddings, dim=0)
        distances, indices = self.index.search(embeddings, 1)
        distances_and_indices = [(distance, index[0]) for distance, index in zip(distances, indices)]

        if cutoff:
            outputs = []
            for (distance, idx) in distances_and_indices[:total_images]:
                if distance > cutoff:
                    outputs.append(self.candidates[idx])
                else:
                    outputs.append(None)
            return outputs
        else:
            return [self.candidates[index] for _, index in distances_and_indices[:total_images]]
                

    def train(self, data_json, data_dir, **kwargs):

        if kwargs:
            config = dictmerge(config, kwargs)

        if not self.config['Recognizer'][self.type]['model_backend'] == 'timm':
            raise NotImplementedError('Training is only supported for timm models')
        
        if not os.path.isdir(self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"]):
            assert self.config['Recognizer'][self.type]['training']['render_dict'] is not None
            assert self.config['Recognizer'][self.type]['training']['font_dir_path'] is not None
        
        os.makedirs(self.config['Recognizer'][self.type]["model_dir"], exist_ok=True)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

        ## Create training data from input coco if not already created
        self._get_training_data(data_json, data_dir)
       
        ## Run training 
        self._train()

        ## Initialize newly trained model
        self.config['Recognizer'][self.type]['hf_repo_id'] = None
        self.initialize_model()


    def _get_training_data(self, data_json, data_dir):
        """
        Transcriptions are currently being passed along with file names
        """
        if not os.path.exists(self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"]):
            self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"] = \
                os.path.join(self.config['Recognizer'][self.type]["model_dir"], "ready_to_go_training_data")

        
        if not os.path.exists(os.path.join(self.config['Recognizer'][self.type]["model_dir"], "ready_to_go_training_data")) and \
            not os.path.exists(self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"]):

            # extract important metadata

            cat_catid_dict = {entry["name"]:entry["id"] for entry in data_json["categories"]}
            imageid_filename_dict = {x["id"]:x["file_name"] for x in data_json["images"]}

            try:
                type_catid = cat_catid_dict[self.type]
            except KeyError:
                print("The type of the model doesn't have a name that matches a category in the data json!")
                raise KeyError
            
            # crop annotations and save

            self.anno_crop_and_text_dict = defaultdict(list)

            print("Preparing training data...")
            os.makedirs(os.path.join(self.config['Recognizer'][self.type]["model_dir"], self.type), exist_ok=True)
            for anno in tqdm(data_json["annotations"]):
                if anno["category_id"] == type_catid:
                    image_containing_anno_filename = imageid_filename_dict[anno["image_id"]]
                    image_containing_anno_path = os.path.join(data_dir, image_containing_anno_filename)
                    anno_text = anno["text"]
                    image_containing_anno = PIL.Image.open(image_containing_anno_path)
                    ax, ay, aw, ah = anno["bbox"] # should be in xywh format in COCO, should do some checking for this
                    anno_crop = image_containing_anno.crop((ax, ay, ax+aw, ay+ah))
                    anno_crop_path = os.path.join(
                        os.path.join(self.config['Recognizer'][self.type]["model_dir"], self.type), 
                        self.encode_path_naming_convention(image_containing_anno_filename, anno_text, ax, ay)
                    )
                    anno_crop.save(anno_crop_path)
                    self.anno_crop_and_text_dict[str_to_ord_str(anno_text)].append(anno_crop_path)

            # create synthetic data

            if self.config['Recognizer'][self.type]['training']["render_dict"] is None:
                raise ValueError("render_dict "+self.type+" must be specified if existing training data is not specified")
            elif self.config['Recognizer'][self.type]['training']["font_dir_path"] is None:
                raise ValueError("font_dir_path must be specified if existing training data is not specified")

            render_all_synth_in_parallel(
                self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"], 
                self.config['Recognizer'][self.type]['training']["font_dir_path"], 
                glob(os.path.join(f"{self.config['Recognizer'][self.type]['training']['render_dict']}", "*")), 
                self.config['Recognizer'][self.type]['training']["ascender"]
            )

            # add in paired data

            print("Adding in paired data to synth data...")
            self.all_paired_image_paths = []
            for k, v in tqdm(self.anno_crop_and_text_dict.items()):
                for anno_img_path in v:
                    assert "PAIRED" in anno_img_path
                    shutil.copy(anno_img_path, os.path.join(self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"], k))
                    self.all_paired_image_paths.append(os.path.join(self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"], k, anno_img_path))

        else:

            self.all_paired_image_paths = glob(os.path.join(self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"], "**", "PAIRED*"), recursive=True)


    def _paths_from_coco_json(self, coco_json_path):
        with open(coco_json_path, 'r') as f:
            coco = json.load(f)
            coco_file_names = [os.path.splitext(x['file_name'])[0]  for x in coco['images']]
            paired_image_paths = \
                {"images": [{"file_name": x} for x in self.all_paired_image_paths if \
                            any(os.path.basename(x).startswith('PAIRED_'+y+"_") or \
                                os.path.basename(x).startswith('PAIRED-'+y+"_") or \
                                os.path.basename(x).startswith('PAIRED_'+y+"-") or \
                                os.path.basename(x).startswith('PAIRED-'+y+"-") for y in coco_file_names)]}
        return paired_image_paths


    def _get_train_splits(self, splitseed=99):

        os.makedirs(self.config['Recognizer'][self.type]["model_dir"], exist_ok=True)
        np.random.seed(splitseed)
        np.random.shuffle(self.all_paired_image_paths)

        train_pct, val_pct, test_pct = self.config['Recognizer'][self.type]['training']["train_val_test_split"]
        train_end_idx = int(len(self.all_paired_image_paths) * train_pct)
        val_end_idx = int(len(self.all_paired_image_paths) * (train_pct + val_pct))

        if not self.config['Recognizer'][self.type]['training']['few_shot'] is None:

            if not self.config['Recognizer'][self.type]['training']['train_set_from_coco_json'] is None:
                all_train_paths = [x['file_name'] for x in self._paths_from_coco_json(self.config['Recognizer'][self.type]['training']['train_set_from_coco_json'])['images']]
            else:
                all_train_paths = self.all_paired_image_paths[:train_end_idx]

            self.cat_path_dict = defaultdict(list)
            for tp in all_train_paths:
                cat = tp.split('/')[-2]
                self.cat_path_dict[cat].append(tp)
            unique_train = list(self.cat_path_dict.keys())
            print(f"Distinct paired characters: {len(unique_train)}")

            few_shot_paired_image_paths = []
            for k, v in self.cat_path_dict.items():
                few_shot_samples = np.random.choice(v, min(len(v), self.config['Recognizer'][self.type]['training']['few_shot']), replace=False).tolist()
                few_shot_paired_image_paths.extend([{"file_name": fss} for fss in few_shot_samples])

            train_paired_image_paths = {"images": few_shot_paired_image_paths}
            train_paired_image_json_path = os.path.join(self.config['Recognizer'][self.type]["model_dir"], f"train_paired_image_paths.json")
            with open(train_paired_image_json_path, "w") as f:
                json.dump(train_paired_image_paths, f)

        else:

            # train
            if not self.config['Recognizer'][self.type]['training']['train_set_from_coco_json'] is None:
                train_paired_image_paths = self._paths_from_coco_json(self.config['Recognizer'][self.type]['training']['train_set_from_coco_json'])
            else:
                train_paired_image_paths = {"images": [{"file_name": x} for x in self.all_paired_image_paths[:train_end_idx]]}
            train_paired_image_json_path = os.path.join(self.config['Recognizer'][self.type]["model_dir"], f"train_paired_image_paths.json")
            with open(train_paired_image_json_path, "w") as f:
                json.dump(train_paired_image_paths, f)
            
        # val
        if not self.config['Recognizer'][self.type]['training']['val_set_from_coco_json'] is None:
            val_paired_image_paths = self._paths_from_coco_json(self.config['Recognizer'][self.type]['training']['val_set_from_coco_json'])
        else:
            val_paired_image_paths = {"images": [{"file_name": x} for x in self.all_paired_image_paths[train_end_idx:val_end_idx]]}
        val_paired_image_json_path = os.path.join(self.config['Recognizer'][self.type]["model_dir"], f"val_paired_image_paths.json")
        with open(val_paired_image_json_path, "w") as f:
            json.dump(val_paired_image_paths, f)

        # test
        if not self.config['Recognizer'][self.type]['training']['test_set_from_coco_json'] is None:
            test_paired_image_paths = self._paths_from_coco_json(self.config['Recognizer'][self.type]['training']['test_set_from_coco_json'])
        else:
            test_paired_image_paths = {"images": [{"file_name": x} for x in self.all_paired_image_paths[val_end_idx:]]}
        test_paired_image_json_path = os.path.join(self.config['Recognizer'][self.type]["model_dir"], f"test_paired_image_paths.json")
        with open(test_paired_image_json_path, "w") as f:
            json.dump(test_paired_image_paths, f)

        # unique_test = [y.split('/')[-2] for y in [x['file_name'] for x in test_paired_image_paths['images']]]
        # print(f"Distinct train chars appearing in test: {len(set(unique_train).intersection(set(unique_test)))}/{len(set(unique_test))}")

        return train_paired_image_json_path, val_paired_image_json_path, test_paired_image_json_path


    def _train(self):

        # create splits
        train_paired_image_json_path, \
            val_paired_image_json_path, \
                test_paired_image_json_path = self._get_train_splits(splitseed=99)

        # setup

        if not self.config['Global']["wandb_project"] is None:
            wandb.init(project=self.config['Global']["wandb_project"], name=os.path.basename(self.config['Recognizer'][self.type]["model_dir"]))
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # data parallelism

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            datapara = True
            self.datapara = True
            self.model = nn.DataParallel(self.model)
        else:
            datapara = False
            self.datapara = False
        
        # create dataset

        train_dataset, val_dataset, test_dataset, \
                    train_loader, val_loader, test_loader, num_batches = \
            create_dataset(
                self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"], 
                train_paired_image_json_path,
                val_paired_image_json_path, 
                test_paired_image_json_path, 
                self.config['Recognizer'][self.type]['training']["batch_size"],
                hardmined_txt=self.config['Recognizer'][self.type]['training']["hns_txt_path"], 
                train_mode=self.type,
                m=self.config['Recognizer'][self.type]['training']["m"], 
                finetune=self.config['Recognizer'][self.type]['training']["finetune"],
                pretrain=self.config['Recognizer'][self.type]['training']["pretrain"],
                high_blur=self.config['Recognizer'][self.type]['training']["high_blur"],
                latin_suggested_augs=self.config['Recognizer'][self.type]['training']["latin_suggested_augs"],
                char_trans_version=self.config['Recognizer'][self.type]['training']["char_trans_version"],
                diff_sizes=self.config['Recognizer'][self.type]['training']["diff_sizes"],
                imsize=self.config['Recognizer'][self.type]['training']["imsize"],
                num_passes=self.config['Recognizer'][self.type]['training']["num_passes"],
                no_aug=self.config['Recognizer'][self.type]['training']["no_aug"],
                k=self.config['Recognizer'][self.type]['training']["hardneg_k"],
                aug_paired=self.config['Recognizer'][self.type]['training']["aug_paired"],
                expansion_factor=self.config['Recognizer'][self.type]['training']["expansion_factor"],
                tvt_split=self.config['Recognizer'][self.type]['training']["train_val_test_split"]
        )

        render_dataset = create_render_dataset(
            self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"],
            train_mode=self.type,
            font_name=self.config['Recognizer'][self.type]['training']['default_font_name'],
            imsize=self.config['Recognizer'][self.type]['training']["imsize"],
        )

        # optimizer and loss

        optimizer = AdamW(self.model.parameters(), lr=self.config['Recognizer'][self.type]['training']["lr"], 
                          weight_decay=self.config['Recognizer'][self.type]['training']["weight_decay"], 
                          betas=(self.config['Recognizer'][self.type]['training']["adamw_beta1"], 
                                 self.config['Recognizer'][self.type]['training']["adamw_beta2"]))
        loss_func = losses.SupConLoss(temperature = self.config['Recognizer'][self.type]['training']["temp"]) 

        # get zero-shot accuracy
        self.accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)

        print("Zero-shot accuracy:")
        best_acc = self.tester_knn(
            val_dataset, 
            render_dataset, 
            self.model, 
            split="zs", 
            log=not self.config['Global']["wandb_project"] is None
        )
        self.save_model(self.config['Recognizer'][self.type]["model_dir"], self.model, "best", datapara)
        
        ##Log 
        if not self.config['Global']["wandb_project"] is None:
            wandb.log({f"val/{self.type}/acc": best_acc})

        # set  schedule

        if self.config['Recognizer'][self.type]['training']["lr_schedule"]:
            scheduler = CosineAnnealingDecWarmRestarts(
                optimizer, T_0=1000 if num_batches is None else num_batches, T_mult=2, l_dec=0.9) 
        else:
            scheduler = None
        
        # warm start training

        print("Training...")

        if not self.config['Recognizer'][self.type]['training']["epoch_viz_dir"] is None: 
            os.makedirs(self.config['Recognizer'][self.type]['training']["epoch_viz_dir"], exist_ok=True)

        for epoch in range(
                self.config['Recognizer'][self.type]['training']["start_epoch"], 
                self.config['Recognizer'][self.type]['training']["num_epochs"]+self.config['Recognizer'][self.type]['training']["start_epoch"]):

            acc = self.trainer_knn_with_eval(
                val_dataset, render_dataset,
                self.model, loss_func, device, train_loader, 
                optimizer, epoch, self.config['Recognizer'][self.type]['training']["epoch_viz_dir"], 
                self.config['Recognizer'][self.type]['training']["diff_sizes"], scheduler,
                int_eval_steps=self.config['Recognizer'][self.type]['training']["int_eval_steps"],
                zs_accuracy=best_acc if best_acc != None else 0,
                wandb_log=not self.config['Global']["wandb_project"] is None
            )

            acc = self.tester_knn(
                val_dataset, 
                render_dataset, 
                self.model, 
                split="val",
                log=not self.config['Global']["wandb_project"] is None)
            
            ##Log
            if not self.config['Global']["wandb_project"] is None:
                wandb.log({f"val/{self.type}/acc": acc})

            if acc >= best_acc:
                best_acc = acc
                print("Saving model and index...")
                self.save_model(self.config['Recognizer'][self.type]["model_dir"], self.model, "best", datapara)
                print("Model and index saved.")

                if not scheduler is None:
                    scheduler.step()
                    ###Log on wandb
                    if not self.config['Global']["wandb_project"] is None:
                        wandb.log({f"train/{self.type}/lr": scheduler.get_last_lr()[0]})

        ## test with best encoder
        if self.datapara == False:
            self.model.load_state_dict(torch.load(os.path.join(self.config['Recognizer'][self.type]["model_dir"], "best.pth")))
        else:
            self.model.module.load_state_dict(torch.load(os.path.join(self.config['Recognizer'][self.type]["model_dir"], "best.pth")))

        self.save_ref_index(render_dataset, self.model, self.config['Recognizer'][self.type]["model_dir"])

        if self.config['Recognizer'][self.type]['training']["test_at_end"]:
            print("Testing on test set...")
            self.tester_knn(test_dataset, render_dataset, self.model, "test", log=not self.config['Global']["wandb_project"] is None)
            print("Test set testing complete.")

        # optionally infer hard negatives (turned on by default, highly recommend to facilitate hard negative training)

        if not self.config['Recognizer'][self.type]['training']["hardneg_k"] is None:
            if self.type == "word":
                query_paths = [x[0] for x in train_dataset.data if os.path.basename(x[0])]
                print("Number of query paths: ", len(query_paths))
                query_paths, query_dataset = self.prepare_hn_query_paths(
                    query_paths, train_dataset, paired_hn=True, image_size=self.config['Recognizer'][self.type]['training']["imsize"])
                print(f"Num hard neg paths: {len(query_paths)}")    
                transform = create_paired_transform(self.config['Recognizer'][self.type]['training']["imsize"])
                self.infer_hardneg_dataset(
                    query_dataset, 
                    train_dataset if self.config['Recognizer'][self.type]['training']["finetune"] else render_dataset, 
                    self.model, 
                    os.path.join(self.config['Recognizer'][self.type]["model_dir"], "ref.index"), 
                    os.path.join(self.config['Recognizer'][self.type]["model_dir"], "hns.txt"), 
                    k=self.config['Recognizer'][self.type]['training']["hardneg_k"]
                )
            else:
                ## LEGACY
                query_paths = [x[0] for x in train_dataset.data if os.path.basename(x[0]).startswith("PAIRED")]
                if len(query_paths) == 0:
                    print("No explicit training data... constructing hard neg from (unique) synth crops!")
                    query_path_char_map = defaultdict(list)
                    query_paths = []
                    for x in train_dataset.data:
                        query_path_char_map[os.path.basename(x[0]).split("_")[0]].append(x[0])
                    for k, v in query_path_char_map.items():
                        query_paths.append(np.random.choice(v))
                print(f"Num hard neg paths: {len(query_paths)}")
                transform = create_paired_transform(self.config['Recognizer'][self.type]['training']["imsize"])
                self.legacy_infer_hardneg(
                    query_paths, 
                    train_dataset if self.config['Recognizer'][self.type]['training']["finetune"] else render_dataset, 
                    self.model, 
                    os.path.join(self.config['Recognizer'][self.type]["model_dir"], "ref.index"), 
                    transform, os.path.join(self.config['Recognizer'][self.type]["model_dir"], "hns.txt"), 
                    k=self.config['Recognizer'][self.type]['training']["hardneg_k"], 
                    finetune=self.config['Recognizer'][self.type]['training']["finetune"])
            
        # save results of trained model

        if self.config['Global']['hf_token_for_upload'] is not None:

            login(token = self.config['Global']['hf_token_for_upload'])

            pt_config = self.model.pretrained_cfg
            if 'url' in pt_config:
                del pt_config['url']

            push_to_hf_hub(
                self.model,
                repo_id = os.path.basename(self.config['Recognizer'][self.type]["model_dir"]),
                commit_message = f'Add {self.type} recognizer model',
                token = self.config['Global']['hf_token_for_upload'],
                private = True,
                model_config = pt_config,
            )

            upload_file(
                path_or_fileobj=os.path.join(self.config['Recognizer'][self.type]["model_dir"], "ref.index"),
                path_in_repo="ref.index",
                repo_id=os.path.join(self.config['Global']['hf_username_for_upload'], os.path.basename(self.config['Recognizer'][self.type]["model_dir"])),
                token=self.config['Global']['hf_token_for_upload']
            )

            upload_file(
                path_or_fileobj=os.path.join(self.config['Recognizer'][self.type]["model_dir"], "ref.txt"),
                path_in_repo="ref.txt",
                repo_id=os.path.join(self.config['Global']['hf_username_for_upload'], os.path.basename(self.config['Recognizer'][self.type]["model_dir"])),
                token=self.config['Global']['hf_token_for_upload']
            )

    
    def tester_knn(self, test_set, ref_set, model, split, log=False):

        model.eval()

        test_embeddings, test_labels = self.get_all_embeddings(test_set, model)
        test_labels = test_labels.squeeze(1)
        ref_embeddings, ref_labels = self.get_all_embeddings(ref_set, model)
        ref_labels = ref_labels.squeeze(1)

        print("Computing accuracy...")
        accuracies = self.accuracy_calculator.get_accuracy(test_embeddings, 
            ref_embeddings,
            test_labels,
            ref_labels,
            embeddings_come_from_same_source=False)
        
        prec_1 = accuracies["precision_at_1"]

        ##Log the accuracy
        if log:
            wandb.log({f"{split}/{self.type}/accuracy": prec_1})
        print(f"Accuracy on {split} set (Precision@1) = {prec_1}")
        return prec_1
   
    
    def infer_hardneg_dataset(self, query_dataset, ref_dataset, model, index_path, inf_save_path, k=8):
        ###Now, embed the query_dataset
        query_embeddings, _ = self.get_all_embeddings(query_dataset, model)

        ##Convert to numpy
        query_embeddings = query_embeddings.cpu().numpy()

        index=faiss.read_index(index_path)

        ###ref dataset path dict
        ref_dataset_path_dict=ref_dataset.subsetted_path_dict
        ####Search the embeddings
        _, indices = index.search(query_embeddings, k=k)

        # ####Now, for each index in indices, get the word for the ref path 
        all_nns = []
        for i, idx in enumerate(tqdm(indices)):
            ###use path dict to get the path
            nn_paths = [ref_dataset_path_dict[j][0] for j in idx]
            nn_words = [os.path.basename(path).split("-word-")[1] for path in nn_paths]
            nn_words = [word.split(".")[0] for word in nn_words]
            all_nns.append("|".join(nn_words))

        with open(inf_save_path, 'w') as f:
            f.write("\n".join(all_nns))


    def prepare_hn_query_paths(
            self, 
            query_paths,
            train_dataset,
            paired_hn=True,
            font_paths=[],
            max_word_n=40,
            image_size=224
        ):

        if paired_hn:
            query_paths = [x[0] for x in train_dataset.data if "PAIRED" in os.path.basename(x[0])]
        else:
            query_paths = [x[0] for x in train_dataset.data]
            ###Keep only those paths that contain any of the fonts in font_paths
            query_paths = [x for x in query_paths if any([font in x for font in font_paths])]

        print("Number of query paths: ", len(query_paths))

        ###Get the list of directory names from the query_paths
        if paired_hn:
            ##Get paired paths
            query_paths = [x[0] for x in train_dataset.data if "PAIRED" in os.path.basename(x[0])]
            unpaired_paths=[x[0] for x in train_dataset.data if "PAIRED" not in os.path.basename(x[0]) and self.config['Recognizer'][self.type]['training']['default_font_name'] in os.path.basename(x[0])]
            ####Get only one unpaired path per word - dedup
            unpaired_paths_dedup = []
            unpaired_path_words = [os.path.basename(x).split("-word-")[1].split(".")[0] for x in unpaired_paths]
            unpaired_path_words_unique = list(set(unpaired_path_words))
            ###We only want one path per word from the unpaired_paths
            for word in unpaired_path_words_unique:
                unpaired_paths_dedup.append(unpaired_paths[unpaired_path_words.index(word)])

            unpaired_paths = unpaired_paths_dedup

            print(f"Num unpaired paths: {len(unpaired_paths)}")

        # ###Now, we want to take at most 10 paired paths per word
        ##First, let's make a dict of word to paths
        print("Preparing word paths dict")
        word_to_paths = defaultdict(list)
        for path in tqdm(query_paths):
            word_to_paths[os.path.basename(path).split("-word-")[1].split(".")[0]].append(path)
        
        ###Now, we want to take at most max_word_n paths per word using the word_to_paths dict
        max_word_n_paths = []
        for word in word_to_paths.keys():
            if len(word_to_paths[word]) <= max_word_n:
                max_word_n_paths.extend(word_to_paths[word])
            else:
                ##Shuffle the paths
                np.random.shuffle(word_to_paths[word])
                max_word_n_paths.extend(word_to_paths[word][:max_word_n])

        paired_paths = max_word_n_paths

        print(f"Num selected paths ({max_word_n} at max): {len(paired_paths)}")

        if paired_hn:            
            query_paths = list(set(paired_paths + unpaired_paths))
        else:
            query_paths = list(set(paired_paths))

        ###save query paths to file
        with open(os.path.join(self.config['Recognizer'][self.type]["model_dir"], f"query_paths.txt"), "w") as f:
            for path in query_paths:
                f.write(f"{path}\n")

        query_dataset = create_hn_query_dataset(self.config['Recognizer'][self.type]['training']["ready_to_go_data_dir_path"], imsize=image_size,hn_query_list=query_paths)

        print(f"Num hard neg paths: {len(query_paths)}")    
        return query_paths, query_dataset


    def trainer_knn_with_eval(
            self, val_dataset, render_dataset, model, loss_func, device, 
            train_loader, optimizer, epoch, epochviz=None, 
            diff_sizes=False,scheduler=None,int_eval_steps=None,
            zs_accuracy=0,wandb_log=False):

        model.train()

        for batch_idx, (data, labels) in enumerate(train_loader):

            labels = labels.to(device)
            data = [datum.to(device) for datum in data] if diff_sizes else data.to(device)

            optimizer.zero_grad()

            if diff_sizes:
                out_emb = []
                for datum in data:
                    emb = model(datum.unsqueeze(0)).squeeze(0)
                    out_emb.append(emb)
                embeddings = torch.stack(out_emb, dim=0)
            else:
                embeddings = model(data)

            loss = loss_func(embeddings, labels)
            loss.backward()
            optimizer.step()

            if wandb_log:
                wandb.log({f"train/{self.type}/loss": loss.item()})

            if not int_eval_steps is None:
                if batch_idx % int_eval_steps == 0:
                    acc = self.tester_knn(val_dataset, render_dataset, model, "val", log=wandb_log)
                    print("Intermediate accuracy: ",acc)
                    if wandb_log:
                        wandb.log({f"val/{self.type}/acc": acc})
                    if acc > zs_accuracy:
                        self.save_model(self.config['Recognizer'][self.type]["model_dir"], model, "best", self.datapara)
                        zs_accuracy=acc

            if batch_idx % 100 == 0:
                print("Epoch {} Iteration {}: Loss = {}".format(str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
                if not epochviz is None:
                    for i in range(10):
                        image = T.ToPILImage()(INV_NORMALIZE(data[i].cpu()))
                        image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))

            del embeddings
            del loss
            del labels
            if not scheduler is None:
                scheduler.step()
                if wandb_log:
                    wandb.log({f"train/{self.type}/lr": scheduler.get_lr()[0]})

        return zs_accuracy
   

    def legacy_infer_hardneg(
            self,
            query_paths, 
            ref_dataset, model, 
            index_path, transform, 
            inf_save_path, k=8, finetune=False
        ):

        knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
        infm = InferenceModel(model, knn_func=knn_func)
        infm.load_knn_func(index_path)
        
        all_nns = []
        for query_path in query_paths:
            im = PIL.Image.open(query_path).convert("RGB")
            query = transform(im).unsqueeze(0)
            _, indices = infm.get_nearest_neighbors(query, k=k)
            nn_chars = []
            for i in indices[0]:
                path_elements = os.path.basename(ref_dataset.data[i][0]).split("_")
                nn_chars.append(path_elements[-2] if finetune else path_elements[0])
            nn_chars = [chr(int(c, base=16)) if c.startswith("0x") else c for c in nn_chars]
            all_nns.append("".join(nn_chars))

        with open(inf_save_path, 'w') as f:
            f.write("\n".join(all_nns))


    @staticmethod
    def save_ref_index(ref_dataset, model, save_path, prefix=""):

        knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
        infm = InferenceModel(model, knn_func=knn_func)
        infm.train_knn(ref_dataset)
        infm.save_knn_func(os.path.join(save_path, "ref.index"))

        ref_data_file_names = []
        for x in ref_dataset.data:
            if os.path.basename(x[0]).startswith("0x"):
                ## LEGACY
                ref_data_file_names.append(chr(int(os.path.basename(x[0]).split("_")[0], base=16)))
            else:
                ref_data_file_names.append(os.path.basename(x[0]).split("-word-")[1].split(".")[0])
        with open(os.path.join(save_path, f"{prefix}ref.txt"), "w") as f:
            f.write("\n".join(ref_data_file_names))


    @staticmethod
    def save_model(model_folder, enc, epoch, datapara):

        if not os.path.exists(model_folder): os.makedirs(model_folder)

        if datapara:
            torch.save(enc.module.state_dict(), os.path.join(model_folder, f"{epoch}.pth"))
        else:
            torch.save(enc.state_dict(), os.path.join(model_folder, f"{epoch}.pth"))


    @staticmethod
    def get_all_embeddings(dataset, model, batch_size=128):

        tester = testers.BaseTester(batch_size=batch_size)
        
        return tester.get_all_embeddings(dataset, model)


    def encode_path_naming_convention(self, image_containing_anno_filename, anno_text, x, y):
        file_stem = os.path.splitext(image_containing_anno_filename)[0]
        if self.type == "char":
            return f"PAIRED-{file_stem}-{x}_{y}-char-{str_to_ord_str(anno_text)}.png"
        else:
            return f"PAIRED-{file_stem}-{x}_{y}-word-{str_to_ord_str(anno_text)}.png"

 
    def decode_path_naming_convention(self, path_name):
        if self.type == "char":
            return path_name.split("-char-")[1].split(".")[0]
        else:
            return path_name.split("-word-")[1].split(".")[0]

