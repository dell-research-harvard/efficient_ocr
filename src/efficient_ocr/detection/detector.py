'''
This file initializes the BaseDetector object, which is the base class for all detection models.

The BaseDetector object has the following attributes:
    - model: the model object, which can be a torch.nn.Module or a ONNXRuntime InferenceSession
    - config: the configuration dictionary for the model
    - transform: the transform to be applied to the image before inference, which should be a object with a __call__ method that takes an image as input and returns a transformed image
    - sort_function: the function used to sort the model output, which should be a lambda function passed into sorted(preds, key=sort_function)
    - device: the device to run the model on TODO

The BaseDetector object has the following methods:
    - __call__: runs inference on an image or list of images
    - initliaze_model: initializes the model object from config
    - train: trains the model object
    - preprocess: runs any additional preprocessing steps-- implemented in child classes
    - postprocess: runs any additional postprocessing steps-- implemented in child classes
'''
import numpy as np
from .models import DetectorModelFactory

class BaseDetector:

    def __init__(self, config, target = 'line'):

        self.config = config
        self.model = self.initialize_model(config, target)
        self.transform = self.initialize_transform(config)

    def __call__(self, imgs):
        '''
        Runs inference on an image or list of images
        '''        
        return self.infer(imgs)
    
    def initialize_model(self, config, target):
        '''
        Initializes the model object from config
        '''
        self.model = DetectorModelFactory(config)(config, target)
        
    def infer(self, imgs):
        '''
        Runs inference on a list of images
        '''
        if not isinstance(imgs, list):
            imgs = [imgs]

        # Preprocess
        imgs = self.preprocess(imgs)

        # Run inference transform
        imgs = self.transform(imgs)

        # Run inference - preds will be a list with len(preds) == len(imgs)
        preds = self.model(imgs)

        # Postprocess
        preds = self.postprocess(preds)

        assert isinstance(preds[0], np.ndarray), 'preds must be a numpy array'
        assert preds[0].ndim == 2, 'preds must be a 2D numpy array'
        assert preds[0].shape[1] == 5, 'preds must have 4 columns: x0, y0, x1, y1, class'
        return preds

    



