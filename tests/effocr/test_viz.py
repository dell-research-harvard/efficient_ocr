
import json
import os
import cv2

from efficient_ocr.utils import visualize_effocr_result

def test_viz():
    coco_json = r'.\data\coco_annotations.json'
    imgs = [cv2.imread(r'.\tests\fixtures\test_locca_image.jpg')]
    visualize_effocr_result(imgs, coco_json, save_path = r'C:\Users\bryan\Documents\NBER\efficient_ocr\tests\fixtures\results_viz')
