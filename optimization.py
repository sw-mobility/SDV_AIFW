import os
import subprocess
import onnx
from flask import jsonify

INPUTS_FOLDER = 'Inputs'
MODELS_FOLDER = 'Models'

def optimize(request):
    if 'model' not in request.files:
        return jsonify({"error": "No model file uploaded."}), 400

    file = request.files['model']
    model_filename = file.filename

    if not (model_filename.endswith('.pt') or model_filename.endswith('.onnx')):
        return jsonify({"error": "Unsupported file format. Please upload a .pt or .onnx file."}), 400

    model_path = os.path.join(INPUTS_FOLDER, model_filename)
    file.save(model_path)

    if model_filename.endswith('.pt'):
        onnx_path = model_path.replace('.pt', '.onnx')
        try:
            convert_pt_to_onnx(model_path, onnx_path)
        except Exception as e:
            return jsonify({"error": f"ONNX conversion failed: {str(e)}"}), 500
        model_path = onnx_path

    engine_path = os.path.join(MODELS_FOLDER, model_filename.replace('.pt', '.engine').replace('.onnx', '.engine'))
    try:
        optimize_model(model_path, engine_path)
    except Exception as e:
        return jsonify({"error": f"Optimization failed: {str(e)}"}), 500

    return jsonify({"status": "Optimization completed", "engine_path": engine_path})

def convert_pt_to_onnx(pt_model_path, onnx_model_path):
    """Converts a YOLOv5 PyTorch (.pt) model to ONNX format."""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=pt_model_path, force_reload=True)
    model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)

    torch.onnx.export(
        model, dummy_input, onnx_model_path, export_params=True,
        input_names=['images'], output_names=['output']
    )

def optimize_model(onnx_path, engine_path):
    """Run trtexec to optimize ONNX model into TensorRT engine file."""
    trtexec_path = '/usr/src/tensorrt/bin/trtexec'
    command = [
        trtexec_path,
        '--onnx=' + onnx_path,
        '--saveEngine=' + engine_path,
        '--int8',
        '--useDLACore=0',
        '--allowGPUFallback',
        '--fp16'
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise Exception(f"TensorRT optimization failed: {result.stderr.decode()}")
