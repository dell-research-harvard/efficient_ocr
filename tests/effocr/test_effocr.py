'''
Basic Unit Tests for the EffOCR object
'''

from efficient_ocr import EffOCR
import os

def test_effocr_init():
    data_json = r'.\data\coco_ex.json'
    config_json = r'.\config\config_ex.json'
    effocr = EffOCR(data_json, config_json, pretrained='en_locca')
    results = effocr.infer(r'.\tests\fixtures\test_locca_image.jpg')
    assert effocr is not None