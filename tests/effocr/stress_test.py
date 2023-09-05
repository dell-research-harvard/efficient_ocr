from efficient_ocr import EffOCR
import cv2
if __name__ == '__main__':
    effocr = EffOCR(
        data_json='tests/effocr/coco_annotations_multi/all.json',
        data_dir='tests/effocr/coco_annotations_multi/images',
        config={
            'Global': {
                # 'hf_username_for_upload': 'jcarlson',
                # 'hf_token_for_upload': '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',
                # 'char_only': True,
                # 'recognition_only': True,
            },
            'Recognizer': {
                'char': {
                    'model_dir': './char_model_test7',
                    'hf_repo_id': 'jcarlson/char_model_test6',
                    'training': {
                        'font_dir_path': 'fonts/english',
                        'render_dict': 'dicts/english',
                        'num_epochs': 1,
                        'hardneg_k': None,
                    }
                },
                'word': {
                    'model_dir': './word_model_test7',
                    'hf_repo_id': 'jcarlson/char_model_test6',
                    'training': {
                        'font_dir_path': 'fonts/english',
                        'render_dict': 'dicts/english',
                        'num_epochs': 1,
                        'hardneg_k': None,
                    }
                },
            },
            'Localizer': {
                'model_dir': './localizer_model_test7',
                'hf_repo_id': 'jcarlson/localizer_model_test6',
                'training': {'epochs': 1},
            },
            'Line': {
                'model_dir': './line_model_test7',
                'hf_repo_id': 'jcarlson/line_model_test6',
                'training': {'epochs': 1},
            },
        }
    )
    effocr.train()
    effocr = EffOCR(
        line_detector='./line_model_effocr_en',
        localizer='./locl_model_effocr_en',
        word_recognizer='./news_word_model',
        char_recognizer='./news_char_model',
    )
    results = effocr.infer_simple('tests/effocr/test_locca_image.jpg')