from efficient_ocr import EffOCR

def test_recog_train():

    data_json = "../../data/coco_ex.json"
    data_dir = '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_datasets/locca/images'
    config_yaml = "../../config/config_jp_ex.yaml"
    
    effocr = EffOCR(
        config_yaml, data_json, data_dir
    )

    effocr.train(target=['char_recognition'])

if __name__ == '__main__':
    test_recog_train()