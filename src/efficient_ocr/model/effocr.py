'''
EffOCR Main Class
'''
import json
import numpy as np
import cv2
# from .detection import infer_line # train_line, train_localizer, infer_line, infer_localizer
# from .recognition import train_word, train_char, infer_word, infer_char
from ..detection import LineModel, LocalizerModel # , word_model, char_model

class EffOCR:

    def __init__(self, data_json, config_json, **kwargs):

        self.training_funcs = {'line_detection': self._train_line,
                                'word_and_character_detection': self._train_localizer,
                                'word_recognition': self._train_word_recognizer,
                                'char_recognition': self._train_char_recognizer}
        
        
        with open(data_json, 'r') as f:
            self.data_json = json.load(f)

        self.config = self._load_config(config_json)
        print(self.config)

        self.line_model = self._initialize_line()
        self.localizer_model = self._initialize_localizer()
        self.word_model = self._initialize_word_recognizer()
        self.char_model = self._initialize_char_recognizer()

    def _load_config(self, config_json, **kwargs):
        if isinstance(config_json, str):
            with open(config_json, 'r') as f:
                config = json.load(f)
        elif isinstance(config_json, dict):
            config = config_json
        else:
            raise ValueError('config_json must be a path to a json file or a dictionary')
        
        if kwargs:
            for k, v in kwargs.items():
                config[k] = v
        
        return config
    
    def _load_and_format_images(self, imgs):
        if all([isinstance(img, str) for img in imgs]):
            imgs = [cv2.imread(img) for img in imgs]
        elif all([isinstance(img, np.ndarray) for img in imgs]):
            pass
        return imgs

    def train(self, target = None, **kwargs):
        
        if isinstance(target, str):
            target = [target]
        elif target is None:
            target = ['line_detection', 'word_and_character_detection', 'word_recognition', 'char_recognition']
        elif not isinstance(target, list):
            raise ValueError('target must be a single training procedure or a list of training procedures')

        for t in target:
            if t not in self.TRAINING_FUNCS:
                raise ValueError('target must be one of {}'.format(self.TRAINING_FUNCS.keys()))
            else:
                self.training_funcs[t](**kwargs)

    ### TOM
    def _train_line(self, **kwargs):
        self.line_model = train_line(self.line_model, self.data_json, self.config, **kwargs)

    ### TOM (and Jake)
    def _train_localizer(self, **kwargs):
        self.localizer_model = train_localizer(self.localizer_model, self.data_json, self.config, **kwargs)

    ### ABHISHEK (and Jake)
    def _train_word_recognizer(self, **kwargs):
        self.word_model = train_word(self.word_model, self.data_json, self.config, **kwargs)

    ### JAKE
    def _train_char_recognizer(self, **kwargs):
        self.char_model = train_char(self.char_model, self.data_json, self.config, **kwargs)

    
    ### TOM
    def infer(self, imgs, **kwargs):
        '''
        Inference pipeline has five steps:
        1. Loading and formatting images
        2. Line Detection
        3. Word and Character Detection
        4. Word Recognition
        5. Character Recognition

        Each input/output format is defined in the body
        '''
        
        '''
        Loading and Formatting Images:
            Input: images as one of:
                1. A single image path
                2. A single numpy array
                3. A list of image paths
                4. A list of numpy arrays
        '''
        if isinstance(imgs, str):
            imgs = [imgs]
        elif isinstance(imgs, np.ndarray):
            imgs = [imgs]
        elif not isinstance(imgs, list):
            raise ValueError('imgs must be a single image path/numpy array or a list of image paths/numpy arrays')
        elif not all([isinstance(img, str) for img in imgs]) or not all([isinstance(img, np.ndarray) for img in imgs]):
            raise ValueError('imgs must be a single image path/numpy array or a list of image paths/numpy arrays')
        
        imgs = self._load_and_format_images(imgs)

        '''
        Line Detection:
            Input: images as a list of numpy arrays
            Output: detections as defaultdict(list):
                mapping the index of the original image (in the order they were passed from the above function) to a list of tuples, 
                with each tuple in the format (textline img, (bounding box coordinates (y0, x0, y1, x1)))
        '''

        line_results = self.line_model(imgs, **kwargs) 

        '''
        Word and character localization:
            Input: detections as a defaultdict(list) as described above
            Output: 
        '''
        # localizer_results = infer_localizer(line_results, self.localizer_model, **kwargs) # Passes back detections and cropped images
        
        
        
        # word_results = infer_word(localizer_results, self.word_model, **kwargs) #Passes back predictions and chars to be recognized
        # char_results = infer_char(word_results, self.char_model, **kwargs) # Passes back predidctions

        return line_results
    
    '''
    Model Initialization Functions
    '''

    def _initialize_line(self):
        return LineModel(self.config)
    
    def _initialize_localizer(self):
        return LocalizerModel(self.config)
    
    def _initialize_word_recognizer(self):
        return None
    
    def _initialize_char_recognizer(self):
        return None
    

