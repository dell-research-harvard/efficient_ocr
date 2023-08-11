'''
Provides Unit Tests for each of the EffOCR Models in various configurations and model sources, backends.

This test runs individual inference over each of the four models used in EffOCR. It runs a separate test
for each model backend available, meaning that it runs the following combinations:

# TODO: Continue to implement tests as each model backend comes online
    - Line Detection YOLOv5, YOLOv5 ONNX, YOLOv8, YOLOv8 ONNX, mmdetection, mmdetection ONNX
    - Localizer YOLOv5, YOLOv5 ONNX, YOLOv8, YOLOv8 ONNX, mmdetection, mmdetection ONNX
    - Char Recognition timm, timm ONNX
    - Word Recognition timm, timm ONNX

Each test has three parts:
1. Initialize the model, including model download from the model zoo
2. Run inference on a collection on images, pulled from /tests/fixtures/inference
3. Clean up model, including deleting downloaded weights

This tests tests model functionalities individually, meaning it tests the .__call__ method of each model, 
not the .infer method of the EffOCR object. .infer is tested separately in the test_effocr.py file.

Since OCR may vary based on provided models, we run with standardized models for predictable 
results. We use the following models:

# TODO: Create a huggingface model zoo for the test models here
# TODO: Fill in the test models here
'''

import os
import cv2
import numpy as np
import yaml
from collections import defaultdict
from huggingface_hub import hf_hub_download
from efficient_ocr.detection import LineModel, LocalizerModel
from efficient_ocr.recognition import Recognizer, infer_last_chars, infer_words, infer_chars

# TODO: These really should be pulled from a saved file, or from the model zoo directly...the keyed solution is temporary
TEST_COLLECTIONS = {
    'line_detection_en_yolov5_onnx': '/tests/fixtures/inference/line_det_en',
    'line_detection_jp': '/tests/fixtures/inference/line_det_jp',
    'localization_en': '/tests/fixtures/inference/localization_en',
    'localization_jp': '/tests/fixtures/inference/localization_jp',
    'word_recognition_en': '/tests/fixtures/inference/word_recognition_en',
    'char_recognition_en': '/tests/fixtures/inference/char_recognition_en',
    'char_recognition_jp': '/tests/fixtures/inference/char_recognition_jp'
}

TEST_CONFIGS = {
    'line_detection_en_yolov5_onnx': './tests/fixtures/configs/config_en_line_yolov5_onnx.yaml',
    'line_detection_jp': './tests/fixtures/configs/config_jp_full.yaml',
}

def load_inference_images(collection):
    '''
    Loads a collection of images for inference
    '''
    images = []
    for img in os.listdir(collection):
        if img.endswith('.png') or img.endswith('.jpg'):
            images.append(cv2.imread(os.path.join(collection, img)))
    return images

def load_config(self, config_yaml, **kwargs):
    if isinstance(config_yaml, str):
        with open(config_yaml, 'r') as f:
            config = yaml.safe_load(f)
    elif isinstance(config_yaml, dict):
        config = config_yaml
    else:
        raise ValueError('config_yaml must be a path to a yaml file or a dictionary')


def draw_rectangles_on_image(img, rectangles, color = (0, 255, 0), thickness = 2):
    for rect in rectangles:
        cv2.rectangle(img, (rect[1], rect[0]), (rect[3], rect[2]), color, thickness)
    return img

def get_test_data(test_name):
    '''
    Returns the test data for a given test name

    Args:
        test_name (str): The name of the test to run
    Returns:
        A tuple containing: (test_images [list np.ndarray], test_model_path [str], test_config [yaml dict])
    '''
    test_images = load_inference_images(TEST_COLLECTIONS[test_name])
    test_config = load_config(TEST_CONFIGS[test_name])
    return test_images, test_config

def test_line_detection_horizontal_yolov5_onnx():
    '''
    Test Line detection with a horizontal YOLOv5 ONNX model
    '''
    test_name = 'line_detection_en_yolov5_onnx'
    images, config = get_test_data(test_name)
    model = LineModel(config)
    line_results = model(images)

    # Manually check that the results are the right size and type

    # Check that the results are approximately matching the saved predictions

    cv2.imwrite(r'./tests/fixtures/test_line_det.jpg', line_results[0][0][0])
    # Draw and save the results
    cv2.imwrite(r'./tests/fixtures/test_line_det_boxes.jpg', test_img)

    # Draw and save all line regions
    # for i, line in enumerate(line_results[0]):
    #     cv2.imwrite(r'./tests/fixtures/lines/test_line_det_{}.jpg'.format(i), line[0])

    # test_imgs = [cv2.imread(r'./tests/fixtures/lines/test_line_det_{}.jpg'.format(i)) for i in range(len(os.listdir(r'./tests/fixtures/lines')))]
    config = {}
    model = LocalizerModel(config)
    localizer_results = model(line_results)
    assert isinstance(localizer_results, dict)
    assert isinstance(localizer_results[0], dict)
    assert isinstance(localizer_results[0][0], dict)
    assert all([k in localizer_results[0][0].keys() for k in ['words', 'chars', 'overlaps', 'para_end']])
    assert isinstance(localizer_results[0][0]['words'], list)
    assert isinstance(localizer_results[0][0]['chars'], list)
    assert isinstance(localizer_results[0][0]['overlaps'], list)
    assert isinstance(localizer_results[0][0]['para_end'], bool)

    assert len(localizer_results[0][0]['words']) == 6
    assert (len(localizer_results[0][0]['chars']) - 27) < 3
    assert localizer_results[0][9]['para_end'] == True
    assert localizer_results[0][0]['para_end'] == False

    # Draw and save the results
    for im_idx in range(1):
        words = [r[1] for r in localizer_results[0][im_idx]['words']]
        chars = [r[1] for r in localizer_results[0][im_idx]['chars']]

        img = line_results[0][im_idx][0].copy()
        img = draw_rectangles(img, chars, color = (0, 0, 255))
        cv2.imwrite(r'./tests/fixtures/localizer/test_localizer_chars_{}.jpg'.format(im_idx), img)

        img = line_results[0][im_idx][0].copy()
        words_img = draw_rectangles(img, words, color = (0, 255, 0))
        cv2.imwrite(r'./tests/fixtures/localizer/test_localizer_words_{}.jpg'.format(im_idx), img)

    char_recognizer = Recognizer(config, type='char')
    last_char_results = infer_last_chars(localizer_results, char_recognizer)
    assert len(last_char_results[0][0]['words']) == len(last_char_results[0][0]['final_puncs'])
    assert last_char_results[0][0]['final_puncs'] == [None, None, None, '.', None, None]
    assert last_char_results[0][9]['final_puncs'] == ['.']

    word_recognizer = Recognizer(config, type='word')
    word_results = infer_words(last_char_results, word_recognizer)
    assert len(word_results[0][0]['words']) == len(last_char_results[0][0]['word_preds'])
    assert word_results[0][0]['word_preds'] == ['The', 'tug', 'boat', 'Alice', 'with', 'Captain']
    assert word_results[0][1]['word_preds'][0] is None

    char_results = infer_chars(word_results, char_recognizer)
    assert char_results[0][0]['word_preds'] == ['The', 'tug', 'boat', 'Alice.', 'with', 'Captain']
    assert char_results[0][1]['word_preds'] == ['Rollie', 'Davis', 'and', 'Harry', 'Raymond', 'on']




    