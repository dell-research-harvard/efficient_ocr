from efficient_ocr import EffOCR
import cv2

if __name__ == '__main__':
    # effocr = EffOCR(
    #     data_json='tests/effocr/coco_annotations_multi/all.json',
    #     data_dir='tests/effocr/coco_annotations_multi/images',
    #     config={
    #         'Global': {
    #             # 'hf_username_for_upload': 'jcarlson',
    #             # 'hf_token_for_upload': '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',
    #             # 'char_only': True,
    #             # 'recognition_only': True,
    #         },
    #         'Recognizer': {
    #             'char': {
    #                 'model_dir': './char_model_test7',
    #                 # 'hf_repo_id': 'jcarlson/char_model_test6',
    #                 'training': {
    #                     'font_dir_path': 'fonts/english',
    #                     'render_dict': 'dicts/english',
    #                     'num_epochs': 1,
    #                     'hardneg_k': None,
    #                 }
    #             },
    #             'word': {
    #                 'model_dir': './word_model_test7',
    #                 # 'hf_repo_id': 'jcarlson/char_model_test6',
    #                 'training': {
    #                     'font_dir_path': 'fonts/english',
    #                     'render_dict': 'dicts/english',
    #                     'num_epochs': 1,
    #                     'hardneg_k': None,
    #                 }
    #             },
    #         },
    #         'Localizer': {
    #             'model_dir': './localizer_model_test7',
    #             # 'hf_repo_id': 'jcarlson/localizer_model_test6',
    #             'training': {'epochs': 1},
    #         },
    #         'Line': {
    #             'model_dir': './line_model_test7',
    #             # 'hf_repo_id': 'jcarlson/line_model_test6',
    #             'training': {'epochs': 1},
    #         },
    #     }
    # )
    # effocr.train()

    # timm backend for recognizer, yolov5 for localizer
    effocr = EffOCR(
        line_detector='./line_model_effocr_en',
        localizer='./locl_model_effocr_en',
        word_recognizer='./news_word_model',
        char_recognizer='./news_char_model',
    )
    results = effocr.infer('tests/effocr/test_locca_image.jpg')
    print(results[0].text)

    # ONNX backend for recognizer, yolov5 for localizer
    effocr = EffOCR(
        config={
            'Recognizer': {
                'char': {
                    'model_backend': 'onnx',
                    'model_dir': './char_model_test',
                    'timm_model_name': 'mobilenetv3_small_050.lamb_in1k',
                },
                'word': {
                    'model_backend': 'onnx',
                    'model_dir': './word_model_test',
                    'timm_model_name': 'mobilenetv3_small_050.lamb_in1k',
                },
            },
            'Localizer': {
                'model_backend': 'yolov5',
                'model_dir': './locl_model_effocr_en',
            },
            'Line': {
                'model_backend': 'yolov5',
                'model_dir': './line_model_effocr_en',
            },
        }
    )
    results = effocr.infer(r'.\tests\fixtures\test_locca_image.jpg')
    print(results[0].text)

    # ONNX all the way
    effocr = EffOCR(
        config={
            'Recognizer': {
                'char': {
                    'model_backend': 'onnx',
                    'model_dir': './char_model_test',
                    'timm_model_name': 'mobilenetv3_small_050.lamb_in1k',
                },
                'word': {
                    'model_backend': 'onnx',
                    'model_dir': './word_model_test',
                    'timm_model_name': 'mobilenetv3_small_050.lamb_in1k',
                },
            },
            'Localizer': {
                'model_backend': 'onnx',
                'model_dir': './localizer_model_test',
            },
            'Line': {
                'model_backend': 'onnx',
                'model_dir': './line_model_test',
            },
        }
    )
    results = effocr.infer(r'.\tests\fixtures\test_locca_image.jpg')
    print(results[0].text)
