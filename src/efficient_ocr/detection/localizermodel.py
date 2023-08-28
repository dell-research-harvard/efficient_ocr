'''
Class for EffOCR Localization Model
'''


import os
import json
import yolov5
from yolov5 import train
import numpy as np
from collections import defaultdict
import threading
import torch
import queue
from huggingface_hub import hf_hub_download, snapshot_download
import multiprocessing
# from mmdet.apis import init_detector, inference_detector


from ..utils import letterbox, yolov5_non_max_suppression, yolov8_non_max_suppression, en_preprocess, initialize_onnx_model
from ..utils import DEFAULT_MEAN, DEFAULT_STD
from ..utils import create_yolo_training_data, create_yolo_yaml
from ..utils import all_but_last_in_path, last_in_path, get_path, dictmerge, dir_is_empty


PARA_WEIGHT_L = 3
PARA_WEIGHT_R = 1
PARA_THRESH = 5


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
                output = onnx_iteration(self._model, self.onnx_format_img(img), self.input_name)
            else:
                output = mmdet_iteration(self._model, img)
            self._output_queue.put((bbox_idx, line_idx, output))

    def onnx_format_img(self, img):
        im = letterbox(img, stride=32, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
        if im.ndim == 3:
            im = np.expand_dims(im, 0)
        return im


class LocalizerModel:

    def __init__(self, config):
        """Instantiates the object, including setting up the wrapped ONNX InferenceSession"""

        '''Set up the config'''
        self.config = config
        if self.config['Localizer']['huggingface_model'] is not None:
            backend_ext = ".onnx" if self.config['Localizer']['model_backend'] == "onnx" else ".pt"
            snapshot_download(
                repo_id=self.config['Localizer']['huggingface_model'], 
                allow_patterns="*local*"+backend_ext,
                local_dir=self.config['Localizer']['model_dir'],
                local_dir_use_symlinks=False)
        self.initialize_model()

    def initialize_model(self):
        """Initializes the model based on the model backend

        Returns:
            _type_: _description_
        """

        os.makedirs(self.config['Localizer']['model_dir'], exist_ok=True)
        self.input_name = None
        if self.config['Localizer']['model_backend'] == 'yolov5':
            if dir_is_empty(self.config['Localizer']['model_dir']):
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            else:
                self.model = yolov5.load(get_path(self.config['Localizer']['model_dir'], ext="pt"), device='cpu')
            self.model.conf = self.config['Localizer']['conf_thresh']  # NMS confidence threshold
            self.model.iou = self.config['Localizer']['iou_thresh']  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # NMS multiple labels per box
            self.model.max_det = self.config['Localizer']['max_det']  # maximum number of detections per image
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
                input_queue.put((bbox_idx, line_idx, line_result[0]))
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
            print(output_queue.get())
            bbox_idx, im_idx, preds = output_queue.get()
            
            im = line_results[bbox_idx][im_idx][0]
            if self.config['Localizer']['model_backend'] == 'onnx':  
                preds = [torch.from_numpy(pred) for pred in preds]
                preds = [yolov5_non_max_suppression(
                    pred, conf_thres = self.config['Localizer']['conf_thresh'], 
                    iou_thres=self.config['Localizer']['iou_thresh'], 
                    max_det=self.config['Localizer']['max_det'])[0] for pred in preds]
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
                    x0, y0, x1, y1 = int(x0.item()), int(y0.item()), int(x1.item()), int(y1.item())
                    localizer_results[bbox_idx][im_idx]['words'].append((im[y0:y1, x0:x1, :], (y0, x0, y1, x1)))

            for i, bbox in enumerate(char_bboxes):
                x0, y0, x1, y1 = torch.round(bbox)

                if x0 == x1 or y0 == y1 or x0 < 0:
                    # If so, skip the entry
                    continue
                else:
                    # Note we go ahead and extend characters to the full height of the line
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
    
    def train(self, data_json, data_dir, **kwargs):
        if self.config['Localizer']['model_backend'] != 'yolov5':
            raise NotImplementedError('Only YOLO model backend is currently supported for training!')
        
        if kwargs:
            config = dictmerge(config, kwargs)

        """
        for key in TRAINING_REQUIRED_ARGS:
            if key not in self.config.keys():
                raise ValueError(f'Missing required argument {key} for training!')
        """

        # Create yolo training data from coco
        data_locs = create_yolo_training_data(data_json, data_dir, 'localizer', self.config["Localizer"]["model_dir"])

        # Create yaml with training data
        yaml_loc = create_yolo_yaml(data_locs, 'localizer')

        train.run(
            imgsz=self.config['Localizer']['input_shape'][0], 
            data=yaml_loc, 
            weights=get_path(self.config['Localizer']['model_dir'], ext="pt"), 
            epochs=self.config['Localizer']['epochs'], 
            batch_size=self.config['Localizer']['batch_size'], 
            device=self.config['Localizer']['device'],  
            name = self.config['Localizer']['model_dir'],
            exist_ok=True)
        
        self.initialize_model()
