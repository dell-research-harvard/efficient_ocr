'''
Class for EffOCR Line Detection Model
'''

import os
import sys
# import mmcv
import torch
import numpy as np
import torchvision
import cv2
from math import floor, ceil
import yolov5
from collections import defaultdict

from ..utils import letterbox, yolov5_non_max_suppression, yolov8_non_max_suppression, get_onnx_input_name, initialize_onnx_model
from ..utils import DEFAULT_MEAN, DEFAULT_STD
from ..utils import create_yolo_training_data, create_yolo_yaml


DEFAULT_LINE_CONFIG = { 'line_model_path': './models/yolo/line_model.pt',
                        'iou_thresh': 0.15,
                        'conf_thresh': 0.20, 
                        'num_cores': None,
                        'providers': None, 
                        'input_shape': (640, 640),
                        'model_backend': 'yolo',
                        'min_seg_ratio': 2,
                        'visualize': None,
                        'num_cores': None,
                        'max_det': 200}

class LineModel:
    """
    Class for running the EffOCR line detection model. Essentially a wrapper for the onnxruntime 
    inference session based on the model, wit some additional postprocessing, especially regarding splitting and 
    recombining especailly tall layout regions
    """

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
        for key, value in DEFAULT_LINE_CONFIG.items():
            if key not in self.config:
                self.config[key] = value

        for key, value in kwargs.items():
            self.config[key] = value

        self.initialize_model()


    def __call__(self, imgs):
        """Wraps the run method, allowing the object to be called directly

        Args:
            imgs (list or str or np.ndarray): List of image paths, list of images as np.ndarrays, or single image path, or single image as np.ndarray

        Returns:
            _type_: _description_
        """
        return self.run(imgs)
    
    
    def initialize_model(self):
        """Initializes the model based on the model backend

        Returns:
            _type_: _description_
        """
        if self.config['model_backend'] == 'yolo':
            self.model = yolov5.load(self.config['line_model_path'], device='cpu')
            print(type(self.model))
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
    
    def run(self, imgs):

        if isinstance(imgs, list):
            if all(isinstance(img, np.ndarray) for img in imgs):
                orig_images = [img.copy() for img in imgs]
                all_crops, num_img_crops, img_shapes = [], [], []
                for img in imgs:
                    crops = self.get_crops_from_layout_image(img)
                    all_crops.extend(crops)
                    num_img_crops.append(len(crops))
                    img_shapes.extend([crop.shape for crop in crops])

                imgs = [self.format_line_img(img) for img in imgs]
            else:
                raise ValueError('Invalid combination of input types in Line Detection list! Must be all np.ndarray')
        else:
            raise ValueError('Input type {} is not implemented'.format(type(imgs)))
        
        if self.config['model_backend'] == 'onnx':    
            results = [self.model.run(None, {self._input_name: img}) for img in imgs]
        elif self.config['model_backend'] == 'yolo':
            results = [self.model(img, augment=False) for img in imgs]
        elif self.config['model_backend'] == 'mmdetection':
            raise NotImplementedError('mmdetection not yet implemented!')
        
        return self._postprocess(results, imgs, num_img_crops, img_shapes, orig_images)

    
    def _postprocess(self, results, imgs, num_img_crops, img_shapes, orig_imgs):
        #YOLO NMS is carried out now, other backends will filter by bbox confidence score later
        if self.config['model_backend'] == 'onnx':  
            preds = [torch.from_numpy(pred[0]) for pred in results]
            preds = [yolov5_non_max_suppression(pred, conf_thres = self._conf_thresh, iou_thres=self._iou_thresh, max_det=100)[0] for pred in preds]

        elif self.config['model_backend'] == 'yolo':
            preds = [result.pred[0] for result in results]

        elif self._model_backend == 'mmdetection':
            raise NotImplementedError('mmdetection not yet implemented!')
    
        ### At this point preds is a list of
        start = 0
        final_preds = []
        for idxs in num_img_crops:
            preds = self.adjust_line_preds(preds[start:start+idxs], imgs[start:start+idxs], img_shapes[start:start+idxs])
            final_preds.append(self.readjust_line_predictions(preds, imgs[start].shape[1]))
            start += idxs

        line_results = defaultdict(list)
        for i, final_pred in enumerate(final_preds):
            for line_proj_crop in final_pred:
                x0, y0, x1, y1 = map(round, line_proj_crop)
                line_crop = orig_imgs[i][y0:y1, x0:x1]
                if line_crop.shape[0] == 0 or line_crop.shape[1] == 0:
                    continue

                # Line crops becomes a list of tuples (bbox_id, line_crop [the image itself], line_proj_crop [the coordinates of the line in the layout image])
                line_results[i].append((np.array(line_crop).astype(np.float32), (y0, x0, y1, x1)))
                            
        return line_results


    def adjust_line_preds(self, preds, imgs, orig_shapes):
        adjusted_preds = []

        for pred, shape in zip(preds, orig_shapes):
            line_predictions = pred[pred[:, 1].sort()[1]]
            line_bboxes, line_confs, line_labels = line_predictions[:, :4], line_predictions[:, -2], line_predictions[:, -1]

            im_width, im_height = shape[1], shape[0]
            if self.config['model_backend'] == 'onnx':
                if im_width > im_height:
                    h_ratio = (im_height / im_width) * 640
                    h_trans = 640 * ((1 - (im_height / im_width)) / 2)
                else:
                    h_trans = 0
                    h_ratio = 640

                line_proj_crops = []
                for line_bbox in line_bboxes:
                    x0, y0, x1, y1 = torch.round(line_bbox)
                    x0, y0, x1, y1 = 0, int(floor((y0.item() - h_trans) * im_height / h_ratio)), \
                                    im_width, int(ceil((y1.item() - h_trans) * im_height  / h_ratio))
                
                    line_proj_crops.append((x0, y0, x1, y1))

            # No need to rescale when using yolo natively
            elif self.config['model_backend'] == 'yolo':
                line_proj_crops = []
                for line_bbox in line_bboxes:
                    x0, y0, x1, y1 = torch.round(line_bbox)
                    x0, y0, x1, y1 = 0, y0, im_width, y1
                    line_proj_crops.append((x0, y0, x1, y1))

            elif self.config['model_backend'] == 'mmdetection':
                raise NotImplementedError('mmdetection not yet implemented!')
            
            adjusted_preds.append((line_proj_crops, line_confs, line_labels))

        return adjusted_preds
            
    def readjust_line_predictions(self, line_preds, orig_img_width):
        y0 = 0
        dif = int(orig_img_width * 1.5)
        all_preds, final_preds = [], []
        for j in range(len(line_preds)):
            preds, probs, labels = line_preds[j]
            for i, pred in enumerate(preds):
                all_preds.append((pred[0], pred[1] + y0, pred[2], pred[3] + y0, probs[i]))
            y0 += dif
        
        all_preds = torch.tensor(all_preds)
        if all_preds.dim() > 1:
            keep_preds = torchvision.ops.nms(all_preds[:, :4], all_preds[:, -1], iou_threshold=0.15)
            filtered_preds = all_preds[keep_preds, :4]
            filtered_preds = filtered_preds[filtered_preds[:, 1].sort()[1]]
            for pred in filtered_preds:
                x0, y0, x1, y1 = torch.round(pred)
                x0, y0, x1, y1 = x0.item(), y0.item(), x1.item(), y1.item()
                final_preds.append((x0, y0, x1, y1))
            return final_preds
        else:
            return []
             
    def format_line_img(self, img):
        if self.config['model_backend'] == 'onnx':
            im = letterbox(img, self.config['input_shape'], stride=32, auto=False)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous
            im = im.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
            if im.ndim == 3:
                im = np.expand_dims(im, 0)
        
        elif self.config['model_backend'] == 'yolo':
            im = img

        elif self.config['model_backend'] == 'mmdetection':
            raise NotImplementedError('Backend mmdetection is not implemented')
            
        else:
            raise NotImplementedError('Backend {} is not implemented'.format(self.config['model_backend']))
        
        return im

    def get_crops_from_layout_image(self, image):
        im_width, im_height = image.shape[0], image.shape[1]
        if im_height <= im_width * self.config['min_seg_ratio']:
            return [image]
        else:
            y0 = 0
            y1 = im_width * self.min_seg_ratio
            crops = []
            while y1 <= im_height:
                crops.append(image.crop((0, y0, im_width, y1)))
                y0 += int(im_width * self.min_seg_ratio * 0.75) # .75 factor ensures there is overlap between crops
                y1 += int(im_width * self.min_seg_ration * 0.75)
            
            crops.append(image.crop((0, y0, im_width, im_height)))
            return crops

    # TODO: Train
    def train(self, training_data, **kwargs):
        if not self.config['model_backend'] == 'yolo':
            raise NotImplementedError('Training is only implemented for yolo backend')
        
        # Create yolo training data from coco
        data_locs = create_yolo_training_data(training_data, 'localizer')

        # Create yaml with training data
        yaml_loc = create_yolo_yaml(data_locs, 'localizer')

        yolov5.train(imgsz=self.config['input_shape'][0], data=yaml_loc, weights=self.config['localizer_model_path'], epochs=self.config['epochs'], 
                     batch_size=self.config['batch_size'], device=self.config['device'], exist_ok=True, name = self.config['localizer_training_name'])
        

        self.config['localizer_model_path'] = os.path.join(self.config['localizer_training_name'], 'weights', 'best.pt')
        self.initialize_model()