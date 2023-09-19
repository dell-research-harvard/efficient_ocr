import os
import yolov5
import numpy as np
from collections import defaultdict
import threading
import torch
import queue
from huggingface_hub import hf_hub_download
import multiprocessing
import subprocess
from math import floor, ceil
import torchvision.ops.boxes as bops
# from mmdet.apis import init_detector, inference_detector


from ..utils import letterbox, yolov8_non_max_suppression, en_preprocess, initialize_onnx_model
from ..utils import DEFAULT_MEAN, DEFAULT_STD
from ..utils import create_yolo_training_data
from ..utils import get_path, dictmerge, dir_is_empty


PARA_WEIGHT_L = 3
PARA_WEIGHT_R = 1
PARA_THRESH = 5


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_inter_union_min(boxes1, boxes2):

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = rb - lt  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter
    minimum = torch.min(area1[:, None], area2)

    return inter, union, minimum


def torch_iom_iou(boxes1, boxes2):

    inter, union, minimum = _box_inter_union_min(boxes1, boxes2)
    iou = inter / union
    iom = inter / minimum
    return iom, iou


def iteration(model, input):
    output = model(input)
    return output


def onnx_iteration(model, input, input_name):
    output = model.run(None, {input_name: input})
    return output


def mmdet_iteration(model, input):
    output = inference_detector(model, input)
    return output


def blank_localizer_response():
    return {'words': [],
            'chars': [],
            'overlaps': [],
            'para_end': False }


def safe_coords(x0, y0, x1, y1, line_w, line_h):
    if x0 == x1 or y0 == y1 or x1 < 0 or y1 < 0 or x0 > line_w or y0 > line_h:
        return None
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(x1, line_w), min(y1, line_h)
    return x0, y0, x1, y1


class LocalizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        backend: str = 'yolov5',
        input_name: str = None
    ):
        super(LocalizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue
        self.backend = backend
        self.input_name = input_name

    def run(self):
        while not self._input_queue.empty():
            bbox_idx, line_idx, img = self._input_queue.get()
            if self.backend != 'mmdetection' and self.backend != 'onnx':
                output = iteration(self._model, img)
            elif self.backend == 'onnx':
                output = onnx_iteration(self._model, img, self.input_name)
            else:
                output = mmdet_iteration(self._model, img)
            
            self._output_queue.put((bbox_idx, line_idx, output))


class LocalizerModel:


    def __init__(self, config):

        self.config = config

        if self.config['Localizer']['hf_repo_id'] is not None:
            
            os.makedirs(self.config['Localizer']['model_dir'], exist_ok=True)
            
            # Parse out provided repo_id
            repo_id = self.config['Localizer']['hf_repo_id'].strip('/')
            # split on slashes
            
            if len(repo_id.split('/')) == 2:
                fn_prefix = ''
                fn = 'localizer.pt' if self.config['Localizer']['model_backend'] == 'yolov5' else 'localizer.onnx'

            elif len(repo_id.split('/')) < 2:
                raise ValueError('hf_repo_id must be in the format owner/repo_name, for example: dell-research-harvard/effocr_en')
            else:
                if repo_id.endswith('pt') or repo_id.endswith('onnx'):
                    fn = '/'.join(repo_id.split('/')[2:])
                    fn_prefix = ''
                else:
                    fn_prefix = '/'.join(repo_id.split('/')[2:]) + '/' # Careful, order matters here
                    fn = 'localizer.pt' if self.config['Localizer']['model_backend'] == 'yolov5' else 'localizer.onnx'
                repo_id = '/'.join(repo_id.split('/')[:2])

            if not os.path.exists(os.path.join(self.config['Localizer']['model_dir'], fn_prefix + fn)):
                hf_hub_download(repo_id = self.config['Localizer']['hf_repo_id'], 
                                filename = fn_prefix + fn, 
                                local_dir = self.config['Localizer']['model_dir'])          
        
        self.initialize_model()


    def initialize_model(self):
        """Initializes the model based on the model backend

        Returns:
            _type_: _description_
        """

        self.input_name = None
        if self.config['Localizer']['model_backend'] == 'yolov5':
            if get_path(self.config['Localizer']['model_dir'], ext='pt') is None:
                self.model = yolov5.load('yolov5s.pt')
            else:
                print("Loading pretrained localizer model!")
                self.model = yolov5.load(get_path(self.config['Localizer']['model_dir'], ext="pt"), device='cpu')
            self.model.conf = self.config['Localizer']['training']['conf_thresh']  # NMS confidence threshold
            self.model.iou = self.config['Localizer']['training']['iou_thresh']  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # NMS multiple labels per box
            self.model.max_det = self.config['Localizer']['training']['max_det']  # maximum number of detections per image
        elif self.config['Localizer']['model_backend'] == 'onnx' and not dir_is_empty(self.config['Localizer']['model_dir']):
            self.model, self.input_name, self.input_shape = \
                initialize_onnx_model(get_path(self.config['Localizer']['model_dir'], ext="onnx"), self.config["Localizer"])
        elif self.config['Localizer']['model_backend'] == 'mmdetection':
            if self.config['Localizer']['mmdet_config'] is None:
                raise ValueError('Must specify a mmdetection config file for mmdetection models!')
            loc_mmdet_config = {
                    "model.rpn_head.anchor_generator.scales":[2,8,32],
                    "classes":('char','word'), "data.train.classes":('char','word'), 
                    "data.val.classes":('char','word'), "data.test.classes":('char','word'),
                    "model.roi_head.bbox_head.0.num_classes": 2,
                    "model.roi_head.bbox_head.1.num_classes": 2,
                    "model.roi_head.bbox_head.2.num_classes": 2,
                    "model.roi_head.mask_head.num_classes": 2,
                }
            self.localizer = init_detector(
                self.config['Localizer']['mmdet_config'], 
                get_path(self.config['Localizer']['model_dir'], ext="pth"), 
                device='cpu', 
                cfg_options=loc_mmdet_config)
        else:
            raise ValueError('Invalid model backend specified and/or model directory empty')
        
        
    def __call__(self, line_results):
        return self.run(line_results)
    

    def run_simple(self, line_result_for_single_image, thresh=0.75):

        locl_result_for_single_image = []
        line_result_for_single_image = sorted(
            line_result_for_single_image, 
            key=lambda x: x[1][0] if self.config['Localizer']['vertical'] else x[1][1]
        )

        if self.config['Localizer']['model_backend'] == 'yolov5':

            for line_img, (lx0, ly0, lx1, ly1) in line_result_for_single_image:

                line_h = ly1 - ly0; line_w = lx1 - lx0
                result = self.model(line_img, augment=False)
                pred = result.pred[0]
                pred = pred[pred[:, 1].sort()[1]] if self.config['Localizer']['vertical'] else pred[pred[:, 0].sort()[1]]

                if pred.size(0) == 0:
                    continue

                bboxes, confs, labels = pred[:,:4], pred[:,4], pred[:,5]
                char_bboxes = bboxes[labels == 0]
                word_bboxes = bboxes[labels == 1]

                if not self.config['Global']['char_only']:

                    word_result_char_result_tuples_for_line = []
                    iom, iou = torch_iom_iou(word_bboxes, char_bboxes)

                    for i, wbbox in enumerate(word_bboxes):
                        wx0, wy0, wx1, wy1 = map(round, wbbox.numpy().tolist())
                        wx0, wy0, wx1, wy1 = safe_coords(wx0, wy0, wx1, wy1, line_w, line_h)
                        word_locl_img = line_img.crop((wx0, wy0, wx1, wy1))
                        word_result = (word_locl_img, (wx0, wy0, wx1, wy1))
                        char_results_for_word = []
                        for j, cbbox in enumerate(char_bboxes):
                            if iom[i,j] >= thresh:
                                cx0, cy0, cx1, cy1 = map(round, cbbox.numpy().tolist())
                                cx0, cy0, cx1, cy1 = safe_coords(cx0, cy0, cx1, cy1, line_w, line_h)
                                char_locl_img = line_img.crop((cx0, cy0, cx1, cy1))
                                char_locl_img.save(f"./saved_crops/saveim{j}.png")
                                char_results_for_word.append((char_locl_img, (cx0, cy0, cx1, cy1)))
                        word_result_char_result_tuples_for_line.append((word_result, char_results_for_word))

                        # for a given line: [
                        #   ((word_img, word_coords), [(char_img, char_coords), ...]), 
                        # ...]

                    locl_result_for_single_image.append(word_result_char_result_tuples_for_line)

                else:

                    char_result_tuples_for_line = []
                    for i, cbbox in enumerate(char_bboxes):
                        cx0, cy0, cx1, cy1 = map(round, cbbox.numpy().tolist())
                        cx0, cy0, cx1, cy1 = safe_coords(cx0, cy0, cx1, cy1, line_w, line_h)
                        char_locl_img = line_img.crop((cx0, cy0, cx1, cy1))
                        char_result_tuples_for_line.append((char_locl_img, (cx0, cy0, cx1, cy1)))
                
                    # for a given line: 
                    #   [(char_img, char_coords), ...]

                    locl_result_for_single_image.append(char_result_tuples_for_line)

        else:
            raise NotImplementedError
        
        return locl_result_for_single_image


    def run(self, line_results):

        if not isinstance(line_results, defaultdict):
            raise ValueError('line_results must be a defaultdict(list) with keys corresponding to box ids and lists of tuples as line results for those boxes, with (img, bounding box coordinates)')
        
        localizer_results = {}
        for bbox_idx in line_results.keys():
            # Set up localizer results as a mapping from bounding box index to a dictionary of line indices to a list of localizations
            localizer_results[bbox_idx] = {i: blank_localizer_response() for i in range(len(line_results[bbox_idx]))}
        
        # Set up the input queue
        input_queue = queue.Queue()
        for bbox_idx in line_results.keys():
            for line_idx, line_result in enumerate(line_results[bbox_idx]):
                input_queue.put((bbox_idx, line_idx, self.format_localizer_img(line_result[0])))
                localizer_results[bbox_idx][line_idx]['bbox'] = line_result[1]

        # Set up the output queue
        output_queue = queue.Queue()

        # Set up and run all the threads
        threads = []
        if self.config['Localizer']['num_cores'] is None:
            self.config['Localizer']['num_cores'] = multiprocessing.cpu_count()

        for _ in range(self.config['Localizer']['num_cores']):
            threads.append(
                LocalizerEngineExecutorThread(
                    self.model, input_queue, output_queue, 
                    backend = self.config['Localizer']['model_backend'], 
                    input_name = self.input_name
                )
            )
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()

        # Get the results from the output queue
        side_dists  = {bbox_idx: {'l_dists': [None] * len(line_results[bbox_idx]), 'r_dists': [None] * len(line_results[bbox_idx])} for bbox_idx in line_results.keys()}

        while not output_queue.empty():
            bbox_idx, im_idx, preds = output_queue.get()
            
            im = line_results[bbox_idx][im_idx][0]
            if self.config['Localizer']['model_backend'] == 'onnx':  
                preds = [torch.from_numpy(pred) for pred in preds]
                preds = [yolov8_non_max_suppression(
                    pred, conf_thres = self.config['Localizer']['training']['conf_thresh'], 
                    iou_thres=self.config['Localizer']['training']['iou_thresh'], 
                    max_det=self.config['Localizer']['training']['max_det'])[0] for pred in preds]
                preds = preds[0]
            else:
                preds = preds.pred[0]
            
            bboxes, confs, labels = preds[:, :4], preds[:, -2], preds[:, -1]
            if not self.config['Localizer']['vertical']:
                char_bboxes, word_bboxes = bboxes[labels == 0], bboxes[labels == 1]
                if word_bboxes.shape[0] > 0:
                    char_bboxes, word_bboxes, word_char_overlap = en_preprocess(char_bboxes, word_bboxes)
                else:
                    word_char_overlap = []

                if len(char_bboxes) > 0:
                    l_dist, r_dist = char_bboxes[0][0].item(), char_bboxes[-1][-2].item()
                    side_dists[bbox_idx]['l_dists'][im_idx] = l_dist # Store distances for paragraph detection
                    side_dists[bbox_idx]['r_dists'][im_idx] = r_dist
                else:
                    side_dists[bbox_idx]['l_dists'][im_idx] = None; side_dists[bbox_idx]['r_dists'][im_idx] = None
            else:
                char_bboxes = bboxes[labels == 0]
                char_bboxes = sorted(char_bboxes, key=lambda x: x[1])
            
            for i, bbox in enumerate(word_bboxes):
                x0, y0, x1, y1 = torch.round(bbox)
                
                # Verify the crop is not empty
                if x0 == x1 or y0 == y1 or x0 < 0:
                    # If so, eliminate the corresponding entry in the word_char_overlaps list
                    word_char_overlap.pop(i)
                else:
                    if self.config['Localizer']['model_backend'] == 'onnx':
                        im_width, im_height = im.shape[1], im.shape[0]
                        x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

                    else:
                        x0, y0, x1, y1 = int(x0.item()), int(y0.item()), int(x1.item()), int(y1.item())
                    
                    localizer_results[bbox_idx][im_idx]['words'].append((im[y0:y1, x0:x1, :], (y0, x0, y1, x1)))

            for i, bbox in enumerate(char_bboxes):
                x0, y0, x1, y1 = torch.round(bbox)

                if x0 == x1 or y0 == y1 or x0 < 0:
                    # If so, skip the entry
                    continue
                else:
                    # Rescale to correct coordinates if we're using ONNX models
                    if self.config['Localizer']['model_backend'] == 'onnx':
                        im_width, im_height = im.shape[1], im.shape[0]
                        x0, y0, x1, y1 = int(round(x0.item() * im_width / 640)), 0, int(round(x1.item() * im_width / 640)), im_height

                    # Note we go ahead and extend characters to the full height of the line
                    else:
                        x0, y0, x1, y1 = int(x0.item()), int(y0.item()), int(x1.item()), int(y1.item())
                    
                    localizer_results[bbox_idx][im_idx]['chars'].append((im[:, x0:x1, :], (y0, x0, y1, x1)))

            localizer_results[bbox_idx][im_idx]['overlaps'] = word_char_overlap

        # Compute paragraph ends-- TODO This is incredibly spaghetti-y, needs to be spun off into its own function
        for bbox_idx in line_results.keys():
            try:
                l_list = side_dists[bbox_idx]['l_dists']
                r_list = side_dists[bbox_idx]['r_dists']
                l_avg = sum(filter(None, l_list)) / (len(l_list) - l_list.count(None))
                r_avg = sum(filter(None, r_list)) / (len(r_list) - r_list.count(None))

                l_list = [l_avg if l is None else l for l in l_list]
                r_list = [r_avg if r is None else r for r in r_list]
                r_max = max(r_list)
                r_avg = r_max - r_avg

                l_list = [l / l_avg for l in l_list]
                try:
                    r_list = [(r_max - r) / r_avg for r in r_list]
                except ZeroDivisionError:
                    r_list = [0] * len(r_list)

                for i in range(len(l_list) - 1):
                    score = l_list[i + 1] * PARA_WEIGHT_L + r_list[i] * PARA_WEIGHT_R
                    if score > PARA_THRESH:
                        localizer_results[bbox_idx][i]['para_end'] = True

            except ZeroDivisionError:
                continue

        return localizer_results
    

    def format_localizer_img(self, img):

        if self.config['Line']['model_backend'] == 'onnx':
            im = letterbox(img, stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
            if im.ndim == 3:
                im = np.expand_dims(im, 0)
            return im
        
        elif self.config['Line']['model_backend'] == 'yolov5':
            im = img

        elif self.config['Line']['model_backend'] == 'mmdetection':
            raise NotImplementedError('Backend mmdetection is not implemented')
            
        else:
            raise NotImplementedError('Backend {} is not implemented'.format(self.config['model_backend']))
        
        return im

    def train(self, data_json, data_dir, **kwargs):

        if self.config['Localizer']['model_backend'] != 'yolov5':
            raise NotImplementedError('Only YOLO model backend is currently supported for training!')
        
        if kwargs:
            config = dictmerge(config, kwargs)

        os.makedirs(self.config['Localizer']['model_dir'], exist_ok=True)

        # Create yolo training data from coco
        if self.config['Localizer']['training']['training_data_dir'] is None:
            yaml_loc = create_yolo_training_data(
                data_json, data_dir, target='localizer', 
                output_dir=self.config["Localizer"]["model_dir"], 
                char_only=self.config["Global"]["char_only"])
        elif os.path.isfile(os.path.join(self.config['Localizer']['training']['training_data_dir'], 'data.yaml')):
            yaml_loc = os.path.join(self.config['Localizer']['training']['training_data_dir'], 'data.yaml')
        else:
            raise ValueError('Could not find training data yaml file! Please specify a valid training_data_dir in the config file (containing a data.yaml file) or a valid data_json.')
        
        train_weights = get_path(self.config['Localizer']['model_dir'], ext="pt")

        # Switching these all to subprocess.Popen calls to allow for streaming of outputs to the console in notebooks (esp colab)
        if self.config['Global']['hf_token_for_upload'] is None:
            p = subprocess.Popen(' '.join([
                "yolov5", "train",
                "--imgsz", str(self.config['Localizer']['training']['input_shape'][0]),
                "--data", yaml_loc,
                "--weights", train_weights if train_weights is not None else 'yolov5s.pt',
                "--epochs", str(self.config['Localizer']['training']['epochs']),
                "--batch_size", str(self.config['Localizer']['training']['batch_size']),
                "--device", self.config['Localizer']['training']['device'],
                "--project", self.config['Localizer']['model_dir'],
                "--name", "trained_localizer"]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True)
            
        else:
            assert self.config['Global']['hf_username_for_upload'] is not None, 'Must specify hf_username_for_upload in config file!'
            subprocess.Popen(" ".join([
                "huggingface-cli", "login", "--token", self.config['Global']['hf_token_for_upload'], 
                "&&",
                "yolov5", "train",
                "--imgsz", str(self.config['Localizer']['training']['input_shape'][0]),
                "--data", yaml_loc,
                "--weights", train_weights if train_weights is not None else 'yolov5s.pt',
                "--epochs", str(self.config['Localizer']['training']['epochs']),
                "--batch_size", str(self.config['Localizer']['training']['batch_size']),
                "--device", self.config['Localizer']['training']['device'],
                "--project", self.config['Localizer']['model_dir'],
                "--name", "trained_localizer",
                "--hf_model_id", os.path.join(self.config['Global']['hf_username_for_upload'], 
                                              os.path.basename(self.config['Localizer']['model_dir'])),
                "--hf_token", self.config['Global']['hf_token_for_upload'],
                "--hf_private"]), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True)
            
        # Stream the output to the console as we train
        for line in iter(p.stdout.readline, ''):
            if len(line) > 0:
                print(line.decode('utf-8').strip('\n'))
            elif not line:
                break
        p.wait()
                        
        self.initialize_model()

