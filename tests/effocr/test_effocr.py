'''
Basic Unit Tests for the EffOCR object
'''

from efficient_ocr import EffOCR
from efficient_ocr.utils import make_coco_from_effocr_result
import os

def test_effocr_single():
    config_file = r'.\config\config_en_full.yaml'
    effocr = EffOCR(config_file)
    results = effocr.infer(r'.\tests\fixtures\test_locca_image.jpg', make_coco_annotations=True)
    print(results[0].text)
    assert results[0].text.startswith('The tug boat')

# def test_effocr_dir():
#     data_json = r'.\data\coco_ex.json'
#     config_json = r'.\config\config_ex.json'
#     effocr = EffOCR(data_json, config_json, pretrained='en_locca')
#     results = effocr.infer(r'C:\Users\bryan\Documents\NBER\datasets\paragraph_breaks\raw_imgs', make_coco_annotations=r'C:\Users\bryan\Documents\NBER\efficient_ocr\data\coco_annotations_multi.json')
    
