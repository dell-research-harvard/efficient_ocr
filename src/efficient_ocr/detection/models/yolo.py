'''

'''
from huggingface_hub import hf_hub_download
from ...utils import create_yolo_training_data, create_yolo_yaml


'''
Base YOLO Model Class-- all YOLO models (v5, v8, any others implemented in the future)
should inherit from this class.

Methods children need to implement:
     __init__: Validates input and sets up model
     _run: Runs model on input, returning predictions in finished, nms-ed format
        
'''
class YOLOModel:


    def __init__(self, config, target):
        self.config = config
        self.target = target

    def __call__(self, imgs):
        return self._run(imgs)

    def train(self, training_data):
        
        # Create yolo training data from coco
        data_path = create_yolo_training_data(training_data, self.target)

        # Create yaml with training data
        yaml_loc = create_yolo_yaml(data_path, self.target)

        self._train(yaml_loc)


class YOLOv5Model(YOLOModel):
    import yolov5 # We only import yolov5 if we need it when the class is called
        
    def __init__(self, config, target):
        super().__init__(config, target)
        assert config[target]['model_backend'] == 'yolov5', 'YOLOv5Model only supports yolov5 backend'

        if self.config[target]['huggingface_model'] is not None:
            assert self.config[target]['huggingface_model'].endswith('.pt'), 'huggingface_model must be a .pt file for yolov5'
            self.config[target]['model_path'] = hf_hub_download('/'.join(self.config[target]['huggingface_model'].split('/')[:-1]), self.config[target]['huggingface_model'].split('/')[-1])

        self.model = self.yolov5.load(self.config[target]['model_path'], device='cpu') # TODO: Add GPU option
        self.model.conf = self.config['Line']['conf_thresh']  # NMS confidence threshold
        self.model.iou = self.config['Line']['iou_thresh']  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = self.config[target]['max_det']  # maximum number of detections per image


    def _run(self, imgs):
        return [self.model(img, augment=False) for img in imgs]
    
    def _train(self, yaml_path):
        from yolov5 import train

        train.run(imgsz=self.config[self.target]['input_shape'][0], data=yaml_path, weights= self.config[self.target]['model_path'], 
                    epochs=self.config[self.target]['epochs'], batch_size=self.config[self.target]['batch_size'], 
                    device=self.model.device, exist_ok=True, name = self.config[self.target]['training_name'])
        

class YOLOv8Model(YOLOModel):
    from ultralytics import YOLO

    def __init__(self, config, target):
        super().__init__(config, target)
        assert config[target]['model_backend'] == 'yolov8', 'YOLOv8Model only supports yolov8 backend'

        if self.config[target]['huggingface_model'] is not None:
            assert self.config[target]['huggingface_model'].endswith('.pt'), 'huggingface_model must be a .pt file for yolov8'
            self.config[target]['model_path'] = hf_hub_download('/'.join(self.config[target]['huggingface_model'].split('/')[:-1]), self.config[target]['huggingface_model'].split('/')[-1])

        self.model = self.YOLO(self.config[target]['model_path'], device='cpu') # TODO: Add GPU option
        self.conf = self.config['Line']['conf_thresh']
        self.iou = self.config['Line']['iou_thresh']
        self.max_det = self.config[target]['max_det']  # maximum number of detections per image

    def _run(self, imgs):
        return self.model(imgs, conf = self.conf, iou = self.iou, max_det = self.max_det)
    
    def _train(self, yaml_path):
        results = self.model.train(data=yaml_path, epochs=self.config[self.target]['epochs'], batch_size=self.config[self.target]['batch_size'], imgsz=self.config[self.target]['input_shape'][0])



