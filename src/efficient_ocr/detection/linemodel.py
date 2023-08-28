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
from yolov5 import train
from huggingface_hub import hf_hub_download, snapshot_download
from collections import defaultdict

from ..utils import letterbox, yolov5_non_max_suppression, yolov8_non_max_suppression, get_onnx_input_name, initialize_onnx_model
from ..utils import DEFAULT_MEAN, DEFAULT_STD
from ..utils import create_yolo_training_data, create_yolo_yaml
from ..utils import all_but_last_in_path, last_in_path, get_path, dictmerge, dir_is_empty


# TRAINING_REQUIRED_ARGS = ['epochs', 'batch_size', 'line_training_name', 'device']


class LineModel:
    """
    Class for running the EffOCR line detection model. Essentially a wrapper for the onnxruntime 
    inference session based on the model, wit some additional postprocessing, especially regarding splitting and 
    recombining especailly tall layout regions
    """

    def __init__(self, config):

        self.config = config
        if self.config['Line']['huggingface_model'] is not None:
            backend_ext = ".onnx" if self.config['Line']['model_backend'] == "onnx" else ".pt"
            snapshot_download(
                repo_id=self.config['Line']['huggingface_model'], 
                allow_patterns="*line*"+backend_ext,
                local_dir=self.config['Line']['model_dir'],
                local_dir_use_symlinks=False)
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

        os.makedirs(self.config['Line']['model_dir'], exist_ok=True)

        if self.config['Line']['model_backend'] == 'yolov5':
            if get_path(self.config['Line']['model_dir'], ext='pt') is None:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            else:
                self.model = yolov5.load(get_path(self.config['Line']['model_dir'], ext="pt"), device='cpu')
            self.model.conf = self.config['Line']['conf_thresh']  # NMS confidence threshold
            self.model.iou = self.config['Line']['iou_thresh']  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # NMS multiple labels per box
            self.model.max_det = self.config['Line']['max_det']  # maximum number of detections per image

        elif self.config['Line']['model_backend'] == 'onnx' and not dir_is_empty(self.config['Line']['model_dir']):
            self.model, self._input_name, self._input_shape = \
                initialize_onnx_model(get_path(self.config['Line']['model_dir'], ext="onnx"), self.config['Line'])

        elif self.config['Line']['model_backend'] == 'mmdetection':
            raise NotImplementedError('mmdetection not yet implemented!')
        else:
            raise ValueError('Invalid model backend specified and/or model directory empty')
    
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

                imgs = [self.format_line_img(crop) for crop in all_crops]
            else:
                raise ValueError('Invalid combination of input types in Line Detection list! Must be all np.ndarray')
        else:
            raise ValueError('Input type {} is not implemented'.format(type(imgs)))
        
        if self.config['Line']['model_backend'] == 'onnx':    
            results = [self.model.run(None, {self._input_name: img}) for img in imgs]
        elif self.config['Line']['model_backend'] == 'yolov5':
            results = [self.model(img, augment=False) for img in imgs]
        elif self.config['Line']['model_backend'] == 'mmdetection':
            raise NotImplementedError('mmdetection not yet implemented!')
        
        return self._postprocess(results, imgs, num_img_crops, img_shapes, orig_images)

    
    def _postprocess(self, results, imgs, num_img_crops, img_shapes, orig_imgs):
        #YOLO NMS is carried out now, other backends will filter by bbox confidence score later
        if self.config['Line']['model_backend'] == 'onnx':  
            preds = [torch.from_numpy(pred[0]) for pred in results]
            preds = [yolov5_non_max_suppression(pred, conf_thres = self.config['Line']['conf_thresh'], iou_thres=self.config['Line']['iou_thresh'], max_det=self.config['Line']['max_det'])[0] for pred in preds]

        elif self.config['Line']['model_backend'] == 'yolov5':
            preds = [result.pred[0] for result in results]

        elif self._model_backend == 'mmdetection':
            raise NotImplementedError('mmdetection not yet implemented!')
    
        ### At this point preds is a list of
        start = 0
        final_preds = []
        for idxs in num_img_crops:
            adjusted_preds = self.adjust_line_preds(preds[start:start+idxs], imgs[start:start+idxs], img_shapes[start:start+idxs])
            final_preds.append(self.readjust_line_predictions(adjusted_preds, imgs[start].shape[1]))
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
            if self.config['Line']['model_backend'] == 'onnx':
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
            elif self.config['Line']['model_backend'] == 'yolov5':
                line_proj_crops = []
                for line_bbox in line_bboxes:
                    x0, y0, x1, y1 = torch.round(line_bbox)
                    x0, y0, x1, y1 = 0, y0, im_width, y1
                    line_proj_crops.append((x0, y0, x1, y1))

            elif self.config['Line']['model_backend'] == 'mmdetection':
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

    def get_crops_from_layout_image(self, image):
        im_width, im_height = image.shape[1], image.shape[0]
        if im_height <= im_width * self.config['Line']['min_seg_ratio']:
            return [image]
        else:
            y0 = 0
            y1 = im_width * self.config['Line']['min_seg_ratio']
            crops = []
            while y1 <= im_height:
                crops.append(image[y0:y1, 0:im_width])
                y0 += int(im_width * self.config['Line']['min_seg_ratio'] * 0.75) # .75 factor ensures there is overlap between crops
                y1 += int(im_width * self.config['Line']['min_seg_ratio'] * 0.75)
            
            crops.append(image[y0:im_height, 0:im_width])
            return crops

    # TODO: Train
    def train(self, data_json, data_dir, **kwargs):
        if not self.config['Line']['model_backend'] == 'yolov5':
            raise NotImplementedError('Training is only implemented for yolo backend')
        
        if kwargs:
            config = dictmerge(config, kwargs)

        """
        for key in TRAINING_REQUIRED_ARGS:
            if key not in self.config.keys():
                raise ValueError(f'Missing required argument {key} for training!')
        """
            
        # Create yolo training data from coco
        yaml_loc = create_yolo_training_data(
            data_json, data_dir, target='line', 
            output_dir=self.config["Line"]["model_dir"], 
            char_only=self.config["Global"]["char_only"])
        
        train_weights = get_path(self.config['Line']['model_dir'], ext="pt")

        train.run(
            imgsz=self.config['Line']['input_shape'][0], 
            data=yaml_loc,
            weights=train_weights if train_weights is not None else 'yolov5s.pt', 
            epochs=self.config['Line']['epochs'], 
            batch_size=self.config['Line']['batch_size'], 
            device=self.config['Line']['device'], 
            name = self.config['Line']['model_dir'],
            exist_ok=True)
        
        self.initialize_model()