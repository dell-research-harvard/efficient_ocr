'''
Class for EffOCR Localization Model
'''


import os
import json
import yolov5
import numpy as np

from ..utils import letterbox, yolov5_non_max_suppression, yolov8_non_max_suppression, get_onnx_input_name, initialize_onnx_model
from ..utils import DEFAULT_MEAN, DEFAULT_STD

DEFAULT_LOCALIZER_CONFIG = { 'localizer_model_path': 'yolov5s.pt',
                        'iou_thresh': 0.15,
                        'conf_thresh': 0.20, 
                        'num_cores': None,
                        'providers': None, 
                        'input_shape': (640, 640),
                        'model_backend': 'yolo',
                        'visualize': None,
                        'num_cores': None,
                        'max_det': 200}

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
        print(config)
        self.config = config
        for key, value in DEFAULT_LOCALIZER_CONFIG.items():
            if key not in self.config:
                self.config[key] = value

        for key, value in kwargs.items():
            self.config[key] = value

        self.model = self.initialize_model()

    def initialize_model(self):
        """Initializes the model based on the model backend

        Returns:
            _type_: _description_
        """
        if self.config['model_backend'] == 'yolo':
            self.model = yolov5.load(self.config['line_model_path'], device='cpu')
            self.model.conf = self.config['conf_thresh']  # NMS confidence threshold
            self.model.iou = self.config['iou_thresh']  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # NMS multiple labels per box
            self.model.max_det = self.config['max_det']  # maximum number of detections per image

        elif self.config['model_backend'] == 'onnx':
            self.model, self.input_name, self.input_shape = initialize_onnx_model(self.config['line_model_path'], self.config)

        elif self.config['model_backend'] == 'mmdetection':
            raise NotImplementedError('mmdetection not yet implemented!')
        else:
            raise ValueError('Invalid model backend specified! Must be one of yolo, onnx, or mmdetection')

