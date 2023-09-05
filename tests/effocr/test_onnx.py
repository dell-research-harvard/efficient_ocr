from efficient_ocr import EffOCR

def test_onnx():
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
    results = effocr.infer(r'.\tests\fixtures\test_locca_image.jpg', make_coco_annotations=True)
    print(results[0].text)
    assert results[0].text.startswith('The tug boat')