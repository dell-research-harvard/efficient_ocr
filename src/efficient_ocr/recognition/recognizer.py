'''
Recognizer Class

Essentially holds an encoder, an faiss reference index, and a list of candidate words/characters corresponding to the reference index. 
'''
import faiss
import timm
import torch
import queue
import numpy as np
import threading
import json
import PIL
import os

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
from tqdm import tqdm

from ..utils import initialize_onnx_model
from ..utils import create_batches
from ..utils import get_transform

from ..utils.recognition.synth_crops import render_all_synth_in_parallel
from ..utils.recognition.datasets import create_dataset, create_render_dataset, create_hn_query_dataset
from ..utils.recognition.dataset_utils import create_paired_transform, INV_NORMALIZE
from ..utils.recognition.custom_schedulers import CosineAnnealingDecWarmRestarts
from ..utils.recognition.encoders import AutoEncoderFactory


DEFAULT_RECOGNIZER_CONFIG = {

    'wandb_project': None,
    'data_dir': "",
    'paired_train_anno_img_paths_json': "",
    'train_val_test_split': [0.7, 0.15, 0.15],

    'model_backend_word': 'onnx',
    'timm_model_name_word': None,
    'encoder_path_word': './models/word_recognizer/enc_best.onnx',
    'index_path_word': './models/word_recognizer/ref.index',
    'candidates_path_word': './models/word_recognizer/ref.txt',
    "img_save_dir_word": './intermediate_outputs/word_images',

    'model_backend_char': 'onnx',
    'timm_model_name_char': None,
    'encoder_path_char': './models/char_recognizer/enc_best.onnx',
    'index_path_char': './models/char_recognizer/ref.index',
    'candidates_path_char': './models/char_recognizer/ref.txt',
    "img_save_dir_char": './intermediate_outputs/word_images',

    "crop_dir_path": '',
    "font_dir_path": '',
    "word_dict": '',
    "ascender": True,
    "train_paired_image_paths_json": '',
    "val_paired_image_paths_json": '',
    "test_paired_image_paths_json": '',
    "train_mode": 'character',
    "run_name": '',
    "batch_size": 128,
    "lr": 2e-6,
    "weight_decay": 5e-4,
    "num_epochs": 5,
    "temp": 0.1,
    "start_epoch": 1,
    "m": 4,
    "imsize": 224,
    "hns_txt_path": '',
    "checkpoint": '',
    "finetune": False,
    "pretrain": False,
    "high_blur": False,
    "latin_suggested_augs": True,
    "char_trans_version": 4,
    "diff_sizes": False,
    "epoch_viz_dir": '',
    "infer_hardneg_k": 8,
    "test_at_end": True,
    "auto_model_hf": None,
    "auto_model_timm": None,
    "num_passes": 1,
    "no_aug": False,
    "lr_schedule": False,
    "k": 8,
    "dec_lr_factor": 0.9,
    "adamw_beta1": 0.9,
    "adamw_beta2": 0.999,
    "char_only_sampler": False,
    "aug_paired": False,
    "expansion_factor": 1,
    "int_eval_steps": None,
    "wandb_log": False,
    "default_font_name": "Noto",

}


def str_to_ord_str(string):
    return '_'.join([str(ord(char)) for char in string])


def ord_str_to_word(ord_str):
    return ''.join([chr(int(ord_char)) for ord_char in ord_str.split('_')])


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
        embeddings[i] = result[0]

    embeddings = [torch.nn.functional.normalize(torch.from_numpy(embedding), p=2, dim=1) for embedding in embeddings]
    return embeddings


def iteration(model, input):
    output = model.run(input)
    return output


'''Threaded Recognizer Inference'''
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
            embeddings = self._model.forward_features(torch.from_numpy(input)).numpy()
        elif self._backend == 'onnx':
            embeddings = self._model.run(None, {self.input_name: input})

        return embeddings


class Recognizer:


    def __init__(self, config, type = 'char', **kwargs):

        '''Set up the config'''
        self.config = config
        for key, value in DEFAULT_RECOGNIZER_CONFIG.items():
            if key not in self.config:
                self.config[key] = value

        for key, value in kwargs.items():
            self.config[key] = value

        self.type = type
        self.suffix = '_' + type
        self.initialize_model()
        self.transform = get_transform(type)


    def initialize_model(self):
        self.index = faiss.read_index(self.config['index_path' + self.suffix])
        with open(self.config['candidates_path' + self.suffix], 'r') as f:
            self.candidates = f.read().splitlines()

        if self.suffix == '_word':
            self.candidates = [ord_str_to_word(candidate) for candidate in self.candidates]

        if self.config['model_backend' + self.suffix] == 'timm':
            model = timm.create_model(self.config['timm_model_name' + self.suffix], num_classes=0, pretrained=True)
            self.model = model.load_state_dict(torch.load(self.config['encoder_path' + self.suffix]))
            self.input_name = None

        elif self.config['model_backend' + self.suffix] == 'onnx':
            self.model, self.input_name, _ = initialize_onnx_model(self.config['encoder_path' + self.suffix], self.config)


    def __call__(self, images):
        return self.run(images)
    

    def run(self, images, cutoff = None):
        
        total_images = len(images)
        embeddings = get_crop_embeddings(RecognizerEngine(self.model, self.config['model_backend' + self.suffix], self.transform, self.input_name), images)
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
        

    def _get_training_data(self, data_json, **kwargs):
        """
        Transcriptions are currently being passed along with file names
        """

        with open(data_json) as f:
            data_dict = json.load(f)

        cat_catid_dict = {entry["name"]:entry["id"] for entry in data_dict["categories"]}
        imageid_filename_dict = {x["id"]:x["file_name"] for x in data_dict["images"]}

        try:
            type_catid = cat_catid_dict[self.type]
        except KeyError:
            print("The type of the model doesn't have a name that matches a category in the data json!")
            raise KeyError
        
        self.anno_crop_and_text_dict = defaultdict(list)

        for anno in data_dict["annotations"]:
            if anno["category_id"] == type_catid:
                image_containing_anno_filename = imageid_filename_dict[anno["image_id"]]
                image_containing_anno_path = os.path.join(self.config["data_dir"], image_containing_anno_filename)
                anno_text = anno["text"]
                image_containing_anno = PIL.Image.open(image_containing_anno_path)
                ax, ay, aw, ah = anno["bbox"] # should be in xywh format in COCO, should do some checking for this
                anno_crop = image_containing_anno.crop((ax, ay, ax+aw, ay+ah))
                anno_crop_path = os.path.join(
                    self.config["img_save_dir" + self.suffix], 
                    self.encode_path_naming_convention(image_containing_anno_filename, anno_text)
                )
                anno_crop.save(anno_crop_path)
                self.anno_crop_and_text_dict[str_to_ord_str(anno_text)].append(anno_crop_path)
                

    def train(self, data_json, **kwargs):

        if not self.config['model_backend' + self.suffix] == 'timm':
            raise NotImplementedError('Training is only supported for timm models')

        ## Create training data from input coco
        self._get_training_data(data_json)
       
        ## Run training 
        self._train()

        ## Initialize newly trained model
        self.initialize_model()


    def _train(self, **kwargs):

        # create synthetic data

        render_all_synth_in_parallel(
            self.config["crop_dir_path"], 
            self.config["font_dir_path"], 
            self.config["word_dict"], 
            self.config["ascender"]
        )

        # preprocess paired data

        all_paired_image_paths = []
        for k, v in self.anno_crop_and_text_dict.items():
            for anno_img_path in v:
                shutil.copy(anno_img_path, os.path.join(self.config["crop_dir_path"], k))
                all_paired_image_paths.append(os.path.join(self.config["crop_dir_path"], k, anno_img_path))

        # create splits

        np.random.seed(99)
        np.random.shuffle(all_paired_image_paths)

        train_end_idx = len(all_paired_image_paths) * self.config["train_val_test_split"][0]
        val_end_idx = len(all_paired_image_paths) * (self.config["train_val_test_split"][0] + self.config["train_val_test_split"][1])

        train_paired_image_paths = {"images": [{"file_name": x} for x in all_paired_image_paths[:train_end_idx]]}
        with open(self.config["train_paired_image_paths_json"], "w") as f:
            json.dump(train_paired_image_paths, f)

        val_paired_image_paths = {"images": [{"file_name": x} for x in all_paired_image_paths[train_end_idx:val_end_idx]]}
        with open(self.config["val_paired_image_paths_json"], "w") as f:
            json.dump(val_paired_image_paths, f)

        test_paired_image_paths = {"images": [{"file_name": x} for x in all_paired_image_paths[val_end_idx:]]}
        with open(self.config["test_paired_image_paths_json"], "w") as f:
            json.dump(test_paired_image_paths, f)

        # setup

        if self.config["wandb_log"]:
            wandb.init(project=self.config["wandb_project"], name=self.config["run_name"])
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        os.makedirs(self.config["run_name"], exist_ok=True)

        # load encoder

        if self.config["auto_model_hf"] is None and self.config["auto_model_timm"] is None:
            raise NotImplementedError
        elif not self.config["auto_model_timm"] is None:
            encoder = AutoEncoderFactory("timm", self.config["auto_model_timm"])
        elif not self.config["auto_model_hf"] is None:
            encoder = AutoEncoderFactory("hf", self.config["auto_model_hf"])

        # init encoder

        if self.config["checkpoint"] is None:
            if not self.config["auto_model_timm"] is None:
                enc = encoder(self.config["auto_model_timm"])
            elif not self.config["auto_model_hf"] is None:
                enc = encoder(self.config["auto_model_hf"])
            else:
                enc = encoder()
        else:
            enc = encoder.load(self.config["checkpoint"])

        # data parallelism

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            datapara = True
            self.datapara = True
            enc = nn.DataParallel(enc)
        else:
            datapara = False
            self.datapara = False
        
        # create dataset

        train_dataset, val_dataset, test_dataset, \
                    train_loader, val_loader, test_loader, num_batches = \
            create_dataset(
                self.config["crop_dir_path"], 
                self.config["train_paired_image_paths_json"],
                self.config["val_paired_image_paths_json"], 
                self.config["test_paired_image_paths_json"], 
                self.config["batch_size"],
                hardmined_txt=self.config["hns_txt_path"], 
                train_mode=self.config["train_mode"],
                m=self.config["m"], 
                finetune=self.config["finetune"],
                pretrain=self.config["pretrain"],
                high_blur=self.config["high_blur"],
                latin_suggested_augs=self.config["latin_suggested_augs"],
                char_trans_version=self.config["char_trans_version"],
                diff_sizes=self.config["diff_sizes"],
                imsize=self.config["imsize"],
                num_passes=self.config["num_passes"],
                no_aug=self.config["no_aug"],
                k=self.config["k"],
                aug_paired=self.config["aug_paired"],
                expansion_factor=self.config["expansion_factor"],
        )
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        render_dataset = create_render_dataset(
            self.config["crop_dir_path"],
            train_mode=self.config["train_mode"],
            font_name=self.config["default_font_name"],
            imsize=self.config["imsize"],
        )

        self.render_dataset = render_dataset

        # optimizer and loss

        optimizer = AdamW(enc.parameters(), lr=self.config["lr"], 
                          weight_decay=self.config["weight_decay"], 
                          betas=(self.config["adamw_beta1"], self.config["adamw_beta2"]))
        loss_func = losses.SupConLoss(temperature = self.config["temp"]) 

        # get zero-shot accuracy

        accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)
        self.accuracy_calculator = accuracy_calculator

        if self.config["checkpoint"] is not None:
            print("Zero-shot accuracy:")
            best_acc = self.tester_knn(val_dataset, render_dataset, enc, 
                                  accuracy_calculator, "zs", log=self.config["wandb_log"])
            ##Log 
            if self.config["wandb_log"]:
                wandb.log({"val/acc": best_acc})

        # set  schedule

        if self.config["lr_schedule"]:
            scheduler = CosineAnnealingDecWarmRestarts(optimizer, T_0=1000 if num_batches is None else num_batches, 
                                                     T_mult=2, l_dec=0.9) 
        else:
            scheduler=None
        
        # warm start training

        print("Training...")
        if not self.config["epoch_viz_dir"] is None: os.makedirs(self.config["epoch_viz_dir"], exist_ok=True)
        for epoch in range(self.config["start_epoch"], self.config["num_epochs"]+self.config["start_epoch"]):
            acc = self.trainer_knn_with_eval(enc, loss_func, device, train_loader, 
                                      optimizer, epoch, self.config["epoch_viz_dir"], 
                                      self.config["diff_sizes"], scheduler,
                                      int_eval_steps=self.config["int_eval_steps"],
                                      zs_accuracy=best_acc if best_acc != None else 0,
                                      wandb_log=self.config["wandb_log"])
            acc = self.tester_knn(val_dataset, render_dataset, enc, accuracy_calculator, "val",log=self.config["wandb_log"])
            ##Log
            if self.config["wandb_log"]:
                wandb.log({"val/acc": acc})

            if acc >= best_acc:
                best_acc = acc
                print("Saving model and index...")
                self.save_model(self.config["run_name"], enc, "best", datapara)
                print("Model and index saved.")

                if scheduler!=None:
                    scheduler.step()
                    ###Log on wandb
                    if self.config["wandb_log"]:
                        wandb.log({"train/lr": scheduler.get_last_lr()[0]})

        del enc
        best_enc = encoder.load(os.path.join(self.config["run_name"], "enc_best.pth"))
        self.save_ref_index(render_dataset, best_enc, self.config["run_name"])

        if self.config["test_at_end"]:
            print("Testing on test set...")
            self.tester_knn(test_dataset, render_dataset, best_enc, accuracy_calculator, "test")
            print("Test set testing complete.")

        # optionally infer hard negatives (turned on by default, highly recommend to facilitate hard negative training)

        if not self.config["infer_hardneg_k"] is None:
            query_paths = [x[0] for x in train_dataset.data if os.path.basename(x[0])]
            print("Number of query paths: ", len(query_paths))
            query_paths, query_dataset=self.prepare_hn_query_paths(query_paths, train_dataset, paired_hn=True, image_size=self.config["imsize"])
            print(f"Num hard neg paths: {len(query_paths)}")    
            transform = create_paired_transform(self.config["imsize"])
            self.infer_hardneg_dataset(query_dataset, train_dataset if self.config["finetune"] else render_dataset, best_enc, 
                os.path.join(self.config["run_name"], "ref.index"), os.path.join(self.config["run_name"], "hns.txt"), 
                k=self.config["infer_hardneg_k"])
            
        # initialize trained model

        self.config['index_path' + self.suffix] = os.path.join(self.config["run_name"], "ref.index")
        self.config['candidates_path' + self.suffix] = os.path.join(self.config["run_name"], "ref.txt")
        self.config['encoder_path' + self.suffix] = os.path.join(self.config["run_name"], "enc_best.pth")


    @staticmethod
    def encode_path_naming_convention(self, image_containing_anno_filename, anno_text):
        file_stem = os.path.splitext(image_containing_anno_filename)[0]
        if self.type == "char":
            return f"PAIRED-{file_stem}-char-{str_to_ord_str(anno_text)}.png"
        else:
            return f"PAIRED-{file_stem}-word-{str_to_ord_str(anno_text)}.png"

 
    @staticmethod
    def decode_path_naming_convention(self, path_name):
        if self.type == "char":
            return path_name.split("-char-")[1].split(".")[0]
        else:
            return path_name.split("-word-")[1].split(".")[0]
        
    
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


    def prepare_hn_query_paths(self, query_paths,train_dataset,paired_hn=True,font_paths=[],max_word_n=40,image_size=224):
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
            unpaired_paths=[x[0] for x in train_dataset.data if "PAIRED" not in os.path.basename(x[0]) and self.config["default_font_name"] in os.path.basename(x[0])]
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
        print("preparing word paths dict")
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
        with open(os.path.join(self.config["run_name"], f"query_paths.txt"), "w") as f:
            for path in query_paths:
                f.write(f"{path}\n")

        query_dataset = create_hn_query_dataset(self.config["crop_dir_path"], imsize=image_size,hn_query_list=query_paths)

        print(f"Num hard neg paths: {len(query_paths)}")    
        return query_paths, query_dataset


    @staticmethod
    def save_ref_index(ref_dataset, model, save_path,prefix=""):

        os.makedirs("indices", exist_ok=True)
        knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
        infm = InferenceModel(model, knn_func=knn_func)
        infm.train_knn(ref_dataset)
        infm.save_knn_func(os.path.join(save_path, "ref.index"))

        ref_data_file_names = [os.path.basename(x[0]).split("-word-")[1].split(".")[0]  for x in ref_dataset.data]
        with open(os.path.join(save_path, f"{prefix}ref.txt"), "w") as f:
            f.write("\n".join(ref_data_file_names))


    @staticmethod
    def save_model(model_folder, enc, epoch, datapara):

        if not os.path.exists(model_folder): os.makedirs(model_folder)

        if datapara:
            torch.save(enc.module.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))
        else:
            torch.save(enc.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))


    @staticmethod
    def get_all_embeddings(dataset, model, batch_size=128):

        tester = testers.BaseTester(batch_size=batch_size)
        
        return tester.get_all_embeddings(dataset, model)

    
    def tester_knn(self, test_set, ref_set, model, accuracy_calculator, split, log=False):

        model.eval()

        test_embeddings, test_labels = self.get_all_embeddings(test_set, model)
        test_labels = test_labels.squeeze(1)
        ref_embeddings, ref_labels = self.get_all_embeddings(ref_set, model)
        ref_labels = ref_labels.squeeze(1)

        print("Computing accuracy...")
        accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
            ref_embeddings,
            test_labels,
            ref_labels,
            embeddings_come_from_same_source=False)
        

        prec_1 = accuracies["precision_at_1"]

        ##Log the accuracy
        if log:
            wandb.log({f"{split}/accuracy": prec_1})
        print(f"Accuracy on {split} set (Precision@1) = {prec_1}")
        return prec_1


    def trainer_knn_with_eval(
            self, model, loss_func, device, 
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
                wandb.log({"train/loss": loss.item()})

            if int_eval_steps!=None:
                if batch_idx % int_eval_steps == 0:
                    acc = self.tester_knn(self.val_dataset, self.render_dataset, model, 
                                          self.accuracy_calculator, "val",log=wandb_log)
                    print("Intermediate accuracy: ",acc)
                    if wandb_log:
                        wandb.log({"val/acc": acc})
                    if acc>zs_accuracy:
                        self.save_model(self.config["run_name"], model, "best_cer", self.datapara)
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
            if scheduler!=None:
                scheduler.step()
                if wandb_log:
                    wandb.log({"train/lr": scheduler.get_lr()[0]})

        return zs_accuracy
   
