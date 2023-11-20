from efficient_ocr import EffOCR
import unittest

class TestTrainMethodsFromAnnotations(unittest.TestCase):
    def test_line_train(self):
        data_json = r'.\data\coco_annotations_multi.json'
        effocr = EffOCR()
        effocr.train(target = 'line_detection', epochs = 1, batch_size = 1, device = 'cpu', line_training_name = 'line_test')

    def test_line_train(self):
        data_json = r'.\data\coco_annotations_multi.json'
        config_json = r'.\config\config_ex.json'
        effocr = EffOCR(data_json, config_json, pretrained='en_locca')
        effocr.train(target = 'word_and_character_detection', epochs = 1, batch_size = 1, device = 'cpu', localizer_training_name = 'localizer_test')

    def test_char_train(self):
        effocr = EffOCR(
            config={
                'Global': {
                    'single_model_training': 'char_recognizer'
                },
                'Recognizer': { 'char': {
                    'training': {'num_epochs': 1,
                                'batch_size': 16,
                                'device': '0',
                                'ready_to_go_data_dir_path': '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_char_crops/locca'
                                },
                    'model_dir': './models/char_recognizer',
                    'model_backend': 'timm',
                }},
            },
            char_recognizer = './models/char_recognizer'
        )
        print('Training')
        effocr.train(target = 'char_recognizer')

class TestTrainMethodsPreparedData(unittest.TestCase):
    def test_line_train(self):
        data_json = r'.\data\coco_annotations_multi.json'
        effocr = EffOCR()
        effocr.train(target = 'line_detection', epochs = 1, batch_size = 1, device = 'cpu', line_training_name = 'line_test')

    def test_line_train(self):
        data_json = r'.\data\coco_annotations_multi.json'
        config_json = r'.\config\config_ex.json'
        effocr = EffOCR(data_json, config_json, pretrained='en_locca')
        effocr.train(target = 'word_and_character_detection', epochs = 1, batch_size = 1, device = 'cpu', localizer_training_name = 'localizer_test')

    def test_char_train(self):
        effocr = EffOCR(
            config={
                'Global': {
                    'single_model_training': 'char_recognizer'
                },
                'Recognizer': { 'char': {
                    'training': {'num_epochs': 1,
                                'batch_size': 16,
                                'device': '0',
                                'ready_to_go_data_dir_path': '/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/ocr_char_crops/locca'
                                },
                    'model_dir': './models/char_recognizer',
                    'model_backend': 'timm',
                }},
            },
            char_recognizer = './models/char_recognizer'
        )
        print('Training')
        effocr.train(target = 'char_recognizer')

if __name__ == '__main__':
    # effocr = EffOCR(
    #     config={
    #         'Global': {
    #             'single_model_training': 'line_detector'
    #         },
    #         'Line': {
    #             'training': {'epochs': 1,
    #                          'batch_size': 1,
    #                          'device': 'cpu',
    #                          'training_data_dir': r'C:\Users\bryan\Downloads\line_detection\line_detection_en_articles'
    #                          },
    #             'model_dir': './models',
    #             'model_backend': 'yolov5',
    #         },
    #     }
    # )
    # print('Training')
    # effocr.train(target = 'line_detector')
    pass
