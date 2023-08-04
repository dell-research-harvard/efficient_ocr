'''
Recognizer Class

Essentially holds an encoder, an faiss reference index, and a list of candidate words/characters corresponding to the reference index. 
'''
import faiss
import timm
import torch
import queue
import numpy as np
import threading

from ..utils import initialize_onnx_model
from ..utils import create_batches
from ..utils import get_transform

DEFAULT_RECOGNIZER_CONFIG = {
    'model_backend_word': 'onnx',
    'timm_model_name_word': None,
    'encoder_path_word': './models/word_recognizer/enc_best.onnx',
    'index_path_word': './models/word_recognizer/ref.index',
    'candidates_path_word': './models/word_recognizer/ref.txt',
    'model_backend_char': 'onnx',
    'timm_model_name_char': None,
    'encoder_path_char': './models/char_recognizer/enc_best.onnx',
    'index_path_char': './models/char_recognizer/ref.index',
    'candidates_path_char': './models/char_recognizer/ref.txt',
}

def ord_str_to_word(ord_str):
    return ''.join([chr(int(ord_char)) for ord_char in ord_str.split('_')])

def get_crop_embeddings(recognizer_engine, crops, num_streams=4):
    # Create batches of word crops
    crop_batches = create_batches(crops)

    input_queue = queue.Queue()
    for i, batch in enumerate(crop_batches):
        input_queue.put((i, batch))
    
    output_queue = queue.Queue()
    threads = []

    # for thread in range(num_streams):
    #     threads.append(RecognizerEngineExecutorThread(recognizer_engine, input_queue, output_queue))

    # for thread in threads:
    #     thread.start()

    # for thread in threads:
    #     thread.join()
    while not input_queue.empty():
        i, batch = input_queue.get()
        output = iteration(recognizer_engine, batch)
        output_queue.put((i, output))

    embeddings = [None] * len(crop_batches)
    while not output_queue.empty():
        i, result = output_queue.get()
        embeddings[i] = result[0]

    embeddings = [torch.nn.functional.normalize(torch.from_numpy(embedding), p=2, dim=1) for embedding in embeddings]
    return embeddings

def iteration(model, input):
    output = model.run(input)
    return output

'''Threaded Recognizer Inference'''
class RecognizerEngineExecutorThread(threading.Thread):
    def __init__(
        self,
        model,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
    ):
        super(RecognizerEngineExecutorThread, self).__init__()
        self._model = model
        self._input_queue = input_queue
        self._output_queue = output_queue

    def run(self):
        while not self._input_queue.empty():
            i, batch = self._input_queue.get()
            output = iteration(self._model, batch)
            self._output_queue.put((i, output))

class RecognizerEngine:

    def __init__(self, model, backend, transform, input_name = None):
        self._model = model
        self._backend = backend
        self.transform = transform
        self.input_name = input_name

    def __call__(self, imgs):
        return self.run(imgs)

    def run(self, imgs):
        trans_imgs = []
        for img in imgs:
            try:
                trans_imgs.append(self.transform(img.astype(np.uint8))[0])
            except Exception as e:
                trans_imgs.append(torch.zeros((3, 224, 224)))

        input = torch.nn.functional.pad(torch.stack(trans_imgs), (0, 0, 0, 0, 0, 0, 0, 64 - len(imgs))).numpy()

        if self._backend == 'timm':
            embeddings = self._model.forward_features(torch.from_numpy(input)).numpy()
        elif self._backend == 'onnx':
            embeddings = self._model.run(None, {self.input_name: input})

        return embeddings

class Recognizer:

    def __init__(self, config, type = 'char', **kwargs):

        '''Set up the config'''
        self.config = config
        for key, value in DEFAULT_RECOGNIZER_CONFIG.items():
            if key not in self.config:
                self.config[key] = value

        for key, value in kwargs.items():
            self.config[key] = value

        self.suffix = '_' + type
        self.initialize_model()
        self.transform = get_transform(type)


    def initialize_model(self):
        self.index = faiss.read_index(self.config['index_path' + self.suffix])
        with open(self.config['candidates_path' + self.suffix], 'r') as f:
            self.candidates = f.read().splitlines()

        if self.suffix == '_word':
            self.candidates = [ord_str_to_word(candidate) for candidate in self.candidates]

        if self.config['model_backend' + self.suffix] == 'timm':
            model = timm.create_model(self.config['timm_model_name' + self.suffix], num_classes=0, pretrained=True)
            self.model = model.load_state_dict(torch.load(self.config['encoder_path' + self.suffix]))
            self.input_name = None

        elif self.config['model_backend' + self.suffix] == 'onnx':
            self.model, self.input_name, _ = initialize_onnx_model(self.config['encoder_path' + self.suffix], self.config)

    def __call__(self, images):
        return self.run(images)
    
    def run(self, images, cutoff = None):
        
        total_images = len(images)
        embeddings = get_crop_embeddings(RecognizerEngine(self.model, self.config['model_backend' + self.suffix], self.transform, self.input_name), images)
        embeddings = torch.cat(embeddings, dim=0)
        distances, indices = self.index.search(embeddings, 1)
        distances_and_indices = [(distance, index[0]) for distance, index in zip(distances, indices)]

        if cutoff:
            outputs = []
            for (distance, idx) in distances_and_indices[:total_images]:
                if distance > cutoff:
                    outputs.append(self.candidates[idx])
                else:
                    outputs.append(None)
            return outputs
        else:
            return [self.candidates[index] for _, index in distances_and_indices[:total_images]]
            
    def train(self, **kwargs):
        if not self.config['model_backend' + self.suffix] == 'timm':
            raise NotImplementedError('Training is only supported for timm models')

        # TODO: Call traiing script from /recognizer_training/train.py with approriate args