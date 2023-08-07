import onnxruntime as ort
import onnx
from ..utils import get_onnx_input_name

def initialize_onnx_model(model_path, config):
    """
    Initializes an ONNX model from a path to an ONNX file.
    """
    
    model_path = model_path
    sess_options = ort.SessionOptions()
    if config['num_cores'] is not None:
        sess_options.intra_op_num_threads = config['num_cores']

    if config['providers'] is None:
        providers = ort.get_available_providers()
    
    model = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers,
        )
    
    base_model = onnx.load(model_path)
    input_name = get_onnx_input_name(base_model)
    model_input_shape = model.get_inputs()[0].shape

    return model, input_name, model_input_shape