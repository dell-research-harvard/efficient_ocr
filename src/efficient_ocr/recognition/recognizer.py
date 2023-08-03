'''
Recognizer Class

Essentially holds an encoder, an faiss reference index, and a list of candidate words/characters corresponding to the reference index. 
'''
import faiss
import timm
import torch
import queue
import threading

from ..utils import initialize_onnx_model
from ..utils import create_batches

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

def get_crop_embeddings(recognizer_engine, crops, num_streams=4):
    # Create batches of word crops
    crop_batches = create_batches(crops)

    input_queue = queue.Queue()
    for i, batch in enumerate(crop_batches):
        input_queue.put((i, batch))
    output_queue = queue.Queue()
    threads = []

    for thread in range(num_streams):
        threads.append(RecognizerEngineExecutorThread(recognizer_engine, input_queue, output_queue))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    embeddings = [None] * len(crop_batches)
    while not output_queue.empty():
        i, result = output_queue.get()
        embeddings[i] = result[0][0]

    embeddings = [torch.nn.functional.normalize(torch.from_numpy(embedding), p=2, dim=1) for embedding in embeddings]
    return embeddings

def iteration(model, input):
    output = model.run(input)
    return output, output

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

    def __init__(self, model, backend, suffix = ''):
        self._model = model
        self._backend = backend
        self.suffix = '_' + suffix

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

        if self.backend == 'timm':
            embeddings = self.model.forward_features(torch.from_numpy(input)).numpy()
        elif self.backend == 'onnx':
            self.model.run(None, {self.input_name: input})

class Recognizer:

    def __init__(self, config, **kwargs):

        '''Set up the config'''
        self.config = config
        for key, value in DEFAULT_RECOGNIZER_CONFIG.items():
            if key not in self.config:
                self.config[key] = value

        for key, value in kwargs.items():
            self.config[key] = value


    def initialize_model(self):
        self.index = faiss.read_index(self.config['index_path' + self.suffix])
        with open(self.config['candidates_path' + self.suffix], 'r') as f:
            self.candidates = f.read().splitlines()

        if self.config['model_backend' + self.suffix] == 'timm':
            model = timm.create_model(self.config['timm_model_name' + self.suffix], num_classes=0, pretrained=True)
            self.model = model.load_state_dict(torch.load(self.config['encoder_path' + self.suffix]))

        elif self.config['model_backend'] == 'onnx':
            self.model, self.input_name, _ = initialize_onnx_model(self.config['encoder_path' + self.suffix], self.config)

    def __call__(self, images):
        return self.run(images)
    
    def run(self, images, cutoff = None):
        
        total_images = len(images)
        embeddings = get_crop_embeddings(RecognizerEngine(self.model, self.config['model_backend' + self.suffix]), images)
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