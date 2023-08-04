'''
Develops Unit Tests for each of the EffOCR Models
'''

import os
import cv2
import numpy as np
from collections import defaultdict
from efficient_ocr import EffOCR
from efficient_ocr.detection import LineModel, LocalizerModel
from efficient_ocr.recognition import Recognizer, infer_last_chars, infer_words, infer_chars

def draw_rectangles(img, rectangles, color = (0, 255, 0), thickness = 2):
    for rect in rectangles:
        cv2.rectangle(img, (rect[1], rect[0]), (rect[3], rect[2]), color, thickness)
    return img

def test_line_det():
    test_img = cv2.imread(r'./tests/fixtures/test_locca_image.jpg')
    config = {}
    model = LineModel(config)
    line_results = model([test_img])

    assert isinstance(line_results, defaultdict)
    assert isinstance(line_results[0], list)
    assert isinstance(line_results[0][0], tuple)
    assert isinstance(line_results[0][0][0], np.ndarray)
    assert line_results[0][0][0].shape[1] == test_img.shape[1]
    assert len(line_results[0]) == 20

    cv2.imwrite(r'./tests/fixtures/test_line_det.jpg', line_results[0][0][0])
    # Draw and save the results
    test_img = draw_rectangles(test_img, [l[1] for l in line_results[0]])
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




    