from efficient_ocr import EffOCR
import cv2

if __name__ == '__main__':
    effocr = EffOCR(
        config={
            'Recognizer': {
                'char': {
                    'model_backend': 'onnx',
                    'model_dir': './models',
                    'hf_repo_id': 'dell-research-harvard/effocr_en/char_recognizer',
                },
                'word': {
                    'model_backend': 'onnx',
                    'model_dir': './models',
                    'hf_repo_id': 'dell-research-harvard/effocr_en/word_recognizer',
                },
            },
            'Localizer': {
                'model_dir': '/models',
                'hf_repo_id': 'dell-research-harvard/effocr_en',
                'model_backend': 'onnx'            
            },
            'Line': {
                'model_dir': '/models',
                'hf_repo_id': 'dell-research-harvard/effocr_en',
                'model_backend': 'onnx',
            },
        }
    )
    results = effocr.infer('tests/effocr/test_locca_image.jpg')
    print(results[0].text)