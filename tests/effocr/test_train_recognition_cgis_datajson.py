from efficient_ocr import EffOCR

def test_recog_train():

    data_json = "./coco_annotations_multi/all.json"
    data_dir = './coco_annotations_multi/images'
    config_yaml = "../../config/config_en_datajson.yaml"
    
    effocr = EffOCR(
        data_json, data_dir, config_yaml
    )

    effocr.train(target=['word_recognition', 'char_recognition'])

if __name__ == '__main__':
    test_recog_train()