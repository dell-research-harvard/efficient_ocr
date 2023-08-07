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
import multiprocessing
from mmdet.apis import init_detector, inference_detector


from ..utils import letterbox, yolov5_non_max_suppression, yolov8_non_max_suppression, en_preprocess, initialize_onnx_model
from ..utils import DEFAULT_MEAN, DEFAULT_STD
from ..utils import create_yolo_training_data, create_yolo_yaml

TRAINING_REQUIRED_ARGS = ['epochs', 'batch_size', 'localizer_training_name', 'device']

def iteration(model, input):
    output = model(input)
    return output

def mmdet_iteration(model, input):
    output = inference_detector(model, input)
    return output

''' Threaded Localizer Inference'''
class LocalizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        backend: str = 'yolo',
    ):
        super(LocalizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue
        self.backend = backend

    def run(self):
        while not self._input_queue.empty():
            bbox_idx, line_idx, img = self._input_queue.get()
            if self.backend != 'mmdetection':
                output = iteration(self._model, img)
            else:
                output = mmdet_iteration(self._model, img)
            self._output_queue.put((bbox_idx, line_idx, output))



def blank_localizer_response():
    return {'words': [],
            'chars': [],
            'overlaps': [],
            'para_end': False }

PARA_WEIGHT_L = 3
PARA_WEIGHT_R = 1
PARA_THRESH = 5

class LocalizerModel:

    def __init__(self, config, **kwargs):
        """Instantiates the object, including setting up the wrapped ONNX InferenceSession

        Args:
            model_path (str): Path to ONNX model that will be used
            iou_thresh (float, optional): IOU filter for line detection NMS. Defaults to 0.15.
            conf_thresh (float, optional): Confidence filter for line detection NMS. Defaults to 0.20.
            num_cores (_type_, optional): Number of cores to use during inference. Defaults to None, meaning no intra op thread limit.
            providers (_type_, optional): Any particular ONNX providers to use. Defaults to None, meaning results of ort.get_available_providers() will be used.
            input_shape (tuple, optional): Shape of input images. Defaults to (640, 640).
            model_backend (str, optional): Original model backend being used. Defaults to 'yolo'. Options are mmdetection, detectron2, yolo, yolov8.
        """

        '''Set up the config'''
        self.config = config
        for k, v in kwargs.items():
            self.config['Global'][k] = v

        self.initialize_model()

    def initialize_model(self):
        """Initializes the model based on the model backend

        Returns:
            _type_: _description_
        """
        if self.config['Localizer']['model_backend'] == 'yolo':
            self.model = yolov5.load(self.config['Localizer']['model_path'], device='cpu')
            self.model.conf = self.config['Localizer']['conf_thresh']  # NMS confidence threshold
            self.model.iou = self.config['Localizer']['iou_thresh']  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # NMS multiple labels per box
            self.model.max_det = self.config['Localizer']['max_det']  # maximum number of detections per image

        elif self.config['Localizer']['model_backend'] == 'onnx':
            self.model, self.input_name, self.input_shape = initialize_onnx_model(self.config['Localizer']['model_path'], self.config)

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
            self.localizer = init_detector(self.config['Localizer']['mmdet_config'], self.config['Localizer']['model_path'], device='cpu', cfg_options=loc_mmdet_config)

        else:
            raise ValueError('Invalid model backend specified! Must be one of yolo, onnx, or mmdetection')
        

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
            threads.append(LocalizerEngineExecutorThread(self.model, input_queue, output_queue))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()

        # Get the results from the output queue
        side_dists  = {bbox_idx: {'l_dists': [None] * len(line_results[bbox_idx]), 'r_dists': [None] * len(line_results[bbox_idx])} for bbox_idx in line_results.keys()}
        while not output_queue.empty():
            bbox_idx, im_idx, preds = output_queue.get()
            im = line_results[bbox_idx][im_idx][0]
            preds = preds.pred[0]
            bboxes, confs, labels = preds[:, :4], preds[:, -2], preds[:, -1]
            
            if self.config['Global']['language'] == 'en':
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
                raise NotImplementedError('Only English is currently supported!')
            
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
    
    def train(self, training_data, **kwargs):
        if self.config['Localizer']['model_backend'] != 'yolo':
            raise NotImplementedError('Only YOLO model backend is currently supported for training!')
        
        for key, val in kwargs.items():
            self.config['Localizer'][key] = val

        for key in TRAINING_REQUIRED_ARGS:
            if key not in self.config.keys():
                raise ValueError(f'Missing required argument {key} for training!')

        # Create yolo training data from coco
        data_locs = create_yolo_training_data(training_data, 'localizer')

        # Create yaml with training data
        yaml_loc = create_yolo_yaml(data_locs, 'localizer')


        train.run(imgsz=self.config['Localizer']['input_shape'][0], data=yaml_loc, weights=self.config['Localizer']['model_path'], epochs=self.config['Localizer']['epochs'], 
                     batch_size=self.config['Localizer']['batch_size'], device=self.config['Localizer']['device'], exist_ok=True, name = self.config['Localizer']['training_name'])
        

        self.config['Localizer']['model_path'] = os.path.join(self.config['Localizer']['training_name'], 'weights', 'best.pt')
        self.initialize_model()
