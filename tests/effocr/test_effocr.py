'''
Basic Unit Tests for the EffOCR object
'''

from efficient_ocr import EffOCR
from efficient_ocr.utils import make_coco_from_effocr_result
import os

def test_effocr_init():
    data_json = r'.\data\coco_ex.json'
    config_json = r'.\config\config_ex.json'
    effocr = EffOCR(data_json, config_json, pretrained='en_locca')
    results = effocr.infer(r'.\tests\fixtures\test_locca_image.jpg', make_coco_annotations=True)

    assert results.text.startswith('The tug boat')


    