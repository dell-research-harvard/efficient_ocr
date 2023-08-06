from efficient_ocr import EffOCR

def test_recog_train():

    data_json = "../../data/coco_ex.json"
    data_dir = '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_datasets/locca/images'
    config_yaml = "../../config/config_ex.yaml"
    
    effocr = EffOCR(
        data_json, data_dir, config_yaml
    )

    effocr.train(target=['char_recognition', 'word_recognition'])

if __name__ == '__main__':
    test_recog_train()