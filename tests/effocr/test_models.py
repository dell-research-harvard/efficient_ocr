'''
Develops Unit Tests for each of the EffOCR Models
'''

import os
import cv2
from effocr import EffOCR
from effocr.detection import LineModel, LocalizerModel

def test_line_det():
    test_img = cv2.imread('fixtures/test_locca_image.jpg')
    config = {}
    model = LineModel()
    line_results = model(test_img)
    assert isinstance(line_results, list)

    