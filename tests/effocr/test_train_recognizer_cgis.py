from efficient_ocr import EffOCR
import os
def test_recog_train():
    data_json = './data/coco_ex.json'
    data_dir = './data/images'
    config_yaml = "./config/config_jp_ex.yaml"
    
    effocr = EffOCR(
        config_yaml, data_json, data_dir
    )

    effocr.train(target=['char_recognition'])

if __name__ == '__main__':
    test_recog_train()