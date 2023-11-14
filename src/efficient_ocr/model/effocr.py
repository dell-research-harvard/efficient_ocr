'''
EffOCR Main Class
'''
import json
import numpy as np
import yaml
from collections import defaultdict
import os
import torch
import cv2
from PIL import Image

# from .detection import infer_line # train_line, train_localizer, infer_line, infer_localizer
from ..recognition import Recognizer, infer_last_chars, infer_words, infer_chars
from ..detection import LineModel, LocalizerModel # , word_model, char_model
from ..utils import make_coco_from_effocr_result, visualize_effocr_result, dictmerge, DEFAULT_CONFIG


class EffOCRResult:

    def __init__(self, full_text, result):
        self.text = full_text
        self.preds = result


class EffOCR:


    def __init__(
            self, config = dict(), 
            data_json = None, data_dir = None, 
            line_detector = None, localizer = None, 
            word_recognizer = None, char_recognizer = None,
            **kwargs
        ):

        print(f'GPU is available?: {torch.cuda.is_available()}')

        self.training_funcs = {
            'line_detector': self._train_line,
            'localizer': self._train_localizer,
            'word_recognizer': self._train_word_recognizer,
            'char_recognizer': self._train_char_recognizer
        }

        ## ingest
        
        if data_json is not None:
            with open(data_json, 'r') as f:
                self.data_json = json.load(f)
        else:
            self.data_json = {}

        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = os.getcwd()
            
        self.config = self._load_config(config, **kwargs)

        ## checks

        # if self.config['Line']['model_backend'] != "yolov5" or \
        #         self.config['Localizer']['model_backend'] != "yolov5" or \
                # self.config['Recognizer']['word']['model_backend'] != "timm" or \
                # self.config['Recognizer']['char']['model_backend'] != "timm":
            # raise NotImplementedError("Only the YOLOv5 and timm backends are currently supported!")
        
        ## load from args

        if not line_detector is None:
            if os.path.isdir(line_detector):
                self.config['Line']['model_dir'] = line_detector
            else:
                self.config['Line']['hf_repo_id'] = line_detector
        if not localizer is None:
            if os.path.isdir(localizer):
                self.config['Localizer']['model_dir'] = localizer
            else:
                self.config['Localizer']['hf_repo_id'] = localizer
        if not word_recognizer is None:
            if os.path.isdir(word_recognizer):
                self.config['Recognizer']['word']['model_dir'] = word_recognizer
            else:
                self.config['Recognizer']['word']['hf_repo_id'] = word_recognizer
        if not char_recognizer is None:
            if os.path.isdir(char_recognizer):
                self.config['Recognizer']['char']['model_dir'] = char_recognizer
            else:
                self.config['Recognizer']['char']['hf_repo_id'] = char_recognizer

        ## sensible naming conventions

        if self.config['Line']['hf_repo_id'] is not None and self.config['Line']['model_dir'] == "./line_model":
            self.config['Line']['model_dir'] = f"./{os.path.basename(self.config['Line']['hf_repo_id'])}"
        if self.config['Localizer']['hf_repo_id'] is not None and self.config['Localizer']['model_dir'] == "./localizer_model":
            self.config['Localizer']['model_dir'] = f"./{os.path.basename(self.config['Localizer']['hf_repo_id'])}"
        if self.config['Recognizer']['char']['hf_repo_id'] is not None and self.config['Recognizer']['char']['model_dir'] == "./char_model":
            self.config['Recognizer']['char']['model_dir'] = f"./{os.path.basename(self.config['Recognizer']['char']['hf_repo_id'])}"
        if self.config['Recognizer']['word']['hf_repo_id'] is not None and self.config['Recognizer']['word']['model_dir'] == "./word_model":
            self.config['Recognizer']['word']['model_dir'] = f"./{os.path.basename(self.config['Recognizer']['word']['hf_repo_id'])}"
        
        ## subset init
        if self.config['Global']['single_model_training'] is not None:
            to_train = self.config['Global']['single_model_training']
            if to_train == 'line_detector':
                self.line_model = self._initialize_line()
            elif to_train == 'localizer':
                self.localizer_model = self._initialize_localizer()
            elif to_train == 'word_recognizer':
                self.word_model = self._initialize_word_recognizer()
            elif to_train == 'char_recognizer':
                self.char_model = self._initialize_char_recognizer()
            else:
                raise ValueError('single_model_training must be one of: line_detector, localizer, word_recognizer, char_recognizer')

        ## full init
        elif self.config['Global']['char_only'] and self.config['Global']['recognition_only']:
            self.char_model = self._initialize_char_recognizer()
        elif self.config['Global']['recognition_only']:
            self.word_model = self._initialize_word_recognizer()
            self.char_model = self._initialize_char_recognizer()
        elif self.config['Global']['char_only']:
            self.char_model = self._initialize_char_recognizer()
            if not self.config['Global']['skip_line_detection']:
                self.line_model = self._initialize_line()
            self.localizer_model = self._initialize_localizer()
        else:
            self.word_model = self._initialize_word_recognizer()
            self.char_model = self._initialize_char_recognizer()
            if not self.config['Global']['skip_line_detection']:
                self.line_model = self._initialize_line()
            self.localizer_model = self._initialize_localizer()       
            

    def _load_config(self, config, **kwargs):
        
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        elif isinstance(config, dict):
            pass
        else:
            raise ValueError('config must be a path to a yaml file or a dictionary')
        
        config = dictmerge(DEFAULT_CONFIG, config)
        if kwargs:
            config = dictmerge(config, kwargs)
        
        return config
    

    def _load_and_format_images(self, imgs):
        if all([isinstance(img, str) for img in imgs]):
            imgs = [cv2.imread(img) for img in imgs]
        elif all([isinstance(img, np.ndarray) for img in imgs]):
            pass
        return imgs
    

    def _postprocess(self, results, **kwargs):
        full_results = [None] * len(results.keys())
        for bbox_idx in results.keys():
            if not self.config['Global']['char_only']:
                full_text = '\n'.join([' '.join(results[bbox_idx][i]['word_preds']) for i in range(len(results[bbox_idx]))])
            else:
                full_text = '\n'.join([results[bbox_idx][i]['char_preds'] for i in range(len(results[bbox_idx]))])
            full_results[bbox_idx] = EffOCRResult(full_text, results[bbox_idx])

        return full_results


    def train(self, target = None, **kwargs):
        if isinstance(target, str):
            target = [target]
        elif target is None and self.config['Global']['char_only']:
            target = ['line_detector', 'localizer', 'char_recognizer']
        elif target is None and self.config['Global']['recognition_only']:
            target = ['word_recognizer', 'char_recognizer']
        elif target is None and self.config['Global']['skip_line_detection']:
            target = ['localizer', 'word_recognizer', 'char_recognizer']
        elif target is None:
            target = ['line_detector', 'localizer', 'word_recognizer', 'char_recognizer']
        elif not isinstance(target, list):
            raise ValueError('target must be a single training procedure or a list of training procedures')
            
        for t in target:
            print(f"\n\n*** TRAINING: {t} ***\n\n")
            if t not in self.training_funcs.keys():
                raise ValueError('target must be one of {}'.format(self.training_funcs.keys()))
            else:
                self.training_funcs[t](**kwargs)


    def _train_line(self, **kwargs):
        self.line_model.train(self.data_json, self.data_dir, **kwargs)


    def _train_localizer(self, **kwargs):
        self.localizer_model.train(self.data_json, self.data_dir, **kwargs)


    def _train_word_recognizer(self, **kwargs):
        self.word_model.train(self.data_json, self.data_dir, **kwargs)


    def _train_char_recognizer(self, **kwargs):
        print(self.data_json)
        print(self.data_dir)
        self.char_model.train(self.data_json, self.data_dir, **kwargs)

    
    def infer(self, imgs, make_coco_annotations=None, visualize=None, save_crops = None, to_display = None, **kwargs):
        '''
        Inference pipeline has five steps:
        1. Loading and formatting images
        2. Line Detection
        3. Word and Character Detection
        4. Word Recognition
        5. Character Recognition

        Each input/output format is defined in the body

        Options:
            make_coco_annotations:
                None: no coco annotations are made
                True: coco annotations are made and saved to a default path
                str: coco annotations are made and saved to the path specified by the string

            TODO:
            visualize:
                None: Do not visualize
                save: Visualize in saved images
                display: Visualize in a window

            TODO:
            save_crops:
                None: Do not save
                line: Save crops of lines
                word: Save crops of words
                char: Save crops of characters
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
            if os.path.isdir(imgs):
                imgs = [os.path.join(imgs, img) for img in os.listdir(imgs)]
            else: # Assume that the file exists
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

        if not self.config['Global']['skip_line_detection']:
            line_results = self.line_model(imgs, **kwargs) 
        else:
            line_results = defaultdict(list)
            for i, img in enumerate(imgs):
                line_results[i].append((img, (0, 0, img.shape[0], img.shape[1])))

        '''
        Word and character localization:
            Input: detections as a defaultdict(list) as described above
            Output: detections as a dictionary with format:
                { bbox_idx: {
                        line_idx: {
                            'words': [(word_img, (y0, x0, y1, x1)), ...],
                            'chars': [(char_img, (y0, x0, y1, x1)), ...],
                            'overlaps': [[char_idx, char_idx, ...], ...],
                            'para_end': bool,
                            'bbox': (y0, x0, y1, x1)
                        },
                        ...
                    },
                    ...
                }}
        '''

        localizer_results = self.localizer_model(line_results, **kwargs) # Passes back detections and cropped images
        '''
        Last character recognition:
            Input: detections as a dictionary with format as described above
            Output: detections as a dictionary with similar format, but with:
                { bbox_idx: {
                        line_idx: {
                            'words': [(word_img, (y0, x0, y1, x1)), ...],
                            'chars': [(char_img, (y0, x0, y1, x1)), ...],
                            'overlaps': [[char_idx, char_idx, ...], ...],
                            'para_end': bool,
                            'bbox': (y0, x0, y1, x1),
                            'final_puncs': [word_end, ...]
                        },
                        ...
                    },
                    ...
                }}

                Where 'final_puncs' is a list with the same length as the word list for each entry, with the predicted final character of each word, if it is a punctuation mark. 
                If a punctuation mark was detected, all of the characters list, the overlaps object, and the word image and bounding boxes will be adjusted to reflect that detection. 
        '''

        if not self.config['Global']['char_only']:
            # TODO: skip on language from config
            last_char_results = infer_last_chars(localizer_results, self.char_model, **kwargs) # Passes back detections and cropped images

        '''
        Word Recognition: 
            Input: detections as a dictionary with format as described above under Last Char Recognition
            Output: detections as a dictionary with similar format, but with:
                
                { bbox_idx: {
                        line_idx: {
                            'words': [(word_img, (y0, x0, y1, x1)), ...],
                            'chars': [(char_img, (y0, x0, y1, x1)), ...],
                            'overlaps': [[char_idx, char_idx, ...], ...],
                            'para_end': bool,
                            'bbox': (y0, x0, y1, x1),
                            'final_puncs': [word_end, ...],
                            'word_preds': [word_pred, ...]
                        },
                        ...
                    },
                    ...
                }}

            Where 'word_preds' is a list with the same length as the word list for each entry, with the predicted text of that word, if the prediction was confident enough to 
            be over the cosine similarity threshold. If the prediction was not confident enough, the prediction will be "None".
            Regardless of predictions, the word image and bounding boxes stay the same.

        '''

        if not self.config['Global']['char_only']:
            # TODO: skip on language from config, work with skipped final punctuation step
            word_results = infer_words(last_char_results, self.word_model, **kwargs) 

        '''
        Character Recognition:
            Input: detections as a dictionary with format as described above under Word Recognition
            Output: detections as a dictionary with the exact same format, that is:
                
                { bbox_idx: {
                        line_idx: {
                            'words': [(word_img, (y0, x0, y1, x1)), ...],
                            'chars': [(char_img, (y0, x0, y1, x1)), ...],
                            'overlaps': [[char_idx, char_idx, ...], ...],
                            'para_end': bool,
                            'bbox': (y0, x0, y1, x1),
                            'final_puncs': [word_end, ...],
                            'word_preds': [word_pred, ...]
                        },
                        ...
                    },
                    ...
                }}

            Where 'word_preds' is fully filled out. For any word with a None prediction passed in, all overlapping characters have been recognized and combined to create the 
            word prediction. 
        '''

        # TODO: work with skipped word, last character on language from config
        char_results = infer_chars(word_results, self.char_model, self.config['Global']['char_only'], **kwargs) # Passes back predidctions

        # TBD what we do for postprocessing (likely will be combining all line predictions within a bounding box into a single text, then returning those texts as a list)
        # Passes through for now. 
        # TODO: Work with various inference combos
        final_results = self._postprocess(char_results, **kwargs)

        '''
        Output: EffOCRResult or list of EffOCRResults
            EffOCR Results store:
                text: the full predicted text
                preds: the full predictions dictionary, as described above
        '''

        if make_coco_annotations is not None or visualize is not None or save_crops is not None:
            make_coco_from_effocr_result(final_results, imgs, save_path=make_coco_annotations if isinstance(make_coco_annotations, str) else "./data/coco_annotations.json")

        if visualize is not None:
            visualize_effocr_result(imgs, 
                                    annotations_path = make_coco_annotations if isinstance(make_coco_annotations, str) else "./data/coco_annotations.json",
                                    save_path = visualize if isinstance(visualize, str) else "./data/visualized_effocr_result.jpg",
                                    to_display = to_display)
                                    # skip_lines = self.config['Global']['skip_line_detection'], char_only = self.config['Global']['char_only']


        return final_results
    

    def infer_simple(self, imgs):
        
        if isinstance(imgs, str):
            if os.path.isdir(imgs):
                imgs = [os.path.join(imgs, img) for img in os.listdir(imgs)]
            else:
                imgs = [imgs]

        elif isinstance(imgs, np.ndarray):
            imgs = [imgs]
        elif not isinstance(imgs, list):
            raise ValueError('imgs must be a single image path/numpy array or a list of image paths/numpy arrays')
        elif not all([isinstance(img, str) for img in imgs]) or not all([isinstance(img, np.ndarray) for img in imgs]):
            raise ValueError('imgs must be a single image path/numpy array or a list of image paths/numpy arrays')
        
        imgs = [Image.open(img) for img in imgs]
        img_texts = []

        for img in imgs:

            img_text = ""

            # for a given line: (np.array(line_crop).astype(np.float32), (x0, y0, x1, y1))
            if not self.config['Global']['skip_line_detection']:
                line_results = self.line_model.run_simple(img)
            else:
                line_results = [(img, (0, 0, img.shape[1], img.shape[0]))]

            localizer_results = self.localizer_model.run_simple(line_results)
            
            for line_level_locl_results in localizer_results:
                if not self.config['Global']['char_only']:
                    # for a given line: [((word_img, word_coords), [(char_img, char_coords), ...]), ...]
                    for word_result, char_results in line_level_locl_results:
                        word_dist, word_cand = self.word_model.run_simple(word_result)
                        if word_dist[0] >= self.config['Global']['min_word_sim']:
                            img_text += word_cand + " "
                        else:
                            char_dist, char_cand = self.char_model.run_simple(char_results)
                            img_text += char_cand + " "
                    img_text += "\n"
                else: 
                    # for a given line: [(char_img, char_coords), ...]
                    char_results = line_level_locl_results
                    char_dist, char_cand = self.char_model.run_simple(char_results)
                    img_text += char_cand + "\n"

            print(img_text)
            img_texts.append(img_text)

        return img_texts
    

    '''
    Model Initialization Functions
    '''

    def _initialize_line(self):
        return LineModel(self.config)
    
    def _initialize_localizer(self):
        return LocalizerModel(self.config)
    
    def _initialize_word_recognizer(self):
        return Recognizer(self.config, 'word')
    
    def _initialize_char_recognizer(self):
        return Recognizer(self.config, 'char')
    
