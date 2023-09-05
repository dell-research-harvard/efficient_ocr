from efficient_ocr import EffOCR

def test_line_train():
    data_json = r'.\data\coco_annotations_multi.json'
    config_json = r'.\config\config_ex.json'
    effocr = EffOCR(data_json, config_json, pretrained='en_locca')
    effocr.train(target = 'line_detection', epochs = 1, batch_size = 1, device = 'cpu', line_training_name = 'line_test')

def test_line_train():
    data_json = r'.\data\coco_annotations_multi.json'
    config_json = r'.\config\config_ex.json'
    effocr = EffOCR(data_json, config_json, pretrained='en_locca')
    effocr.train(target = 'word_and_character_detection', epochs = 1, batch_size = 1, device = 'cpu', localizer_training_name = 'localizer_test')