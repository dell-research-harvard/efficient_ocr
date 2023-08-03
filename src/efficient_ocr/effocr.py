'''
EffOCR Main Class
'''
import json
import numpy as np
from .detection import train_line, train_localizer, infer_line, infer_localizer
from .recognition import train_word, train_char, infer_word, infer_char

class EffOCR:


    def __init__(self, data_json, config_json, **kwargs):

        self.training_funcs = {'line_detection': self.train_line,
                                'word_and_character_detection': self.train_localizer,
                                'word_recognition': self.train_word,
                                'char_recognition': self.train_char}
        
        with open(data_json, 'r') as f:
            self.data_json = json.load(f)

        self.config = self._load_config(config_json)

        self.line_model = self._initialize_line()
        self.localizer_model = self._initialize_localizer()
        self.word_model = self._initialize_word_recognizer()
        self.char_model = self._initialize_char_recognizer()

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
        
        if isinstance(imgs, str):
            imgs = [imgs]
        elif isinstance(imgs, np.ndarray):
            imgs = [imgs]
        elif not isinstance(imgs, list):
            raise ValueError('imgs must be a single image path/numpy array or a list of image paths/numpy arrays')
        elif not all([isinstance(img, str) for img in imgs]) or not all([isinstance(img, np.ndarray) for img in imgs]):
            raise ValueError('imgs must be a single image path/numpy array or a list of image paths/numpy arrays')
        
        imgs = self._load_and_format_images(imgs)

        line_results = infer_line(imgs, self.line_model, **kwargs) # Passes back detections and cropped images
        localizer_results = infer_localizer(line_results, self.localizer_model, **kwargs) # Passes back detections and cropped images
        word_results = infer_word(localizer_results, self.word_model, **kwargs) #Passes back predictions and chars to be recognized
        char_results = infer_char(word_results, self.char_model, **kwargs) # Passes back predidctions

        return char_results
    

