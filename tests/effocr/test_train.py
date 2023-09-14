from efficient_ocr import EffOCR

def test_line_train():
    data_json = r'.\data\coco_annotations_multi.json'
    effocr = EffOCR()
    effocr.train(target = 'line_detection', epochs = 1, batch_size = 1, device = 'cpu', line_training_name = 'line_test')

def test_line_train():
    data_json = r'.\data\coco_annotations_multi.json'
    config_json = r'.\config\config_ex.json'
    effocr = EffOCR(data_json, config_json, pretrained='en_locca')
    effocr.train(target = 'word_and_character_detection', epochs = 1, batch_size = 1, device = 'cpu', localizer_training_name = 'localizer_test')

if __name__ == '__main__':
    print('Initializing')
    effocr = EffOCR(
        config={
            'Global': {
                'single_model_training': 'line_detector'
            },
            'Line': {
                'training': {'epochs': 1,
                             'batch_size': 1,
                             'device': 'cpu',
                             'training_data_dir': r'C:\Users\bryan\Downloads\line_detection\line_detection_en_articles'
                             },
                'model_dir': '/models',
                'model_backend': 'yolov5',
            },
        }
    )
    # print('Training')
    # effocr.train(target = 'line_detector')
