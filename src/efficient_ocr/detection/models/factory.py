'''
Detector Model Factory Class

Generates model class based on config file. Model classes are subclasses 
'''
from onnx import YOLOv5ModelONNX, YOLOv8ModelONNX, MMDetectModelONNX
from yolo import YOLOv5Model, YOLOv8Model
from mmdetect import MMDetectModel

def DetectorFactory(config, target):

    if not config[target]['onnx']:
        if config[target]['model_backend'] == 'yolov5':
            return YOLOv5Model
        elif config[target]['model_backend'] == 'yolov8':
            return YOLOv8Model
        elif config[target]['model_backend'] == 'mmdetection':
            return MMDetectModel
    elif config[target]['onnx']:
        if config[target]['model_backend'] == 'yolov5':
            return YOLOv5ModelONNX
        elif config[target]['model_backend'] == 'yolov8':
            return YOLOv8ModelONNX
        elif config[target]['model_backend'] == 'mmdetection':
            return MMDetectModelONNX
    