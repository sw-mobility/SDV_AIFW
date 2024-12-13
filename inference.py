import os
import torch
import shutil
from flask import jsonify, url_for

INPUTS_FOLDER = 'Inputs'
MODELS_FOLDER = 'Models'

def infer(request):
    selected_model = request.form.get('model')
    model_path = os.path.join(MODELS_FOLDER, selected_model)

    if not selected_model or not os.path.exists(model_path):
        return jsonify({"error": "Model not found."}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files['image']
    img_path = os.path.join(INPUTS_FOLDER, 'Images', file.filename)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    file.save(img_path)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    results = model(img_path)
    results.save()

    detect_folder = os.path.join('runs', 'detect')
    exp_folders = [f for f in os.listdir(detect_folder) if f.startswith('exp')]
    latest_exp = max(exp_folders, key=lambda x: os.path.getctime(os.path.join(detect_folder, x)))

    file_extension = file.filename.split('.')[-1]
    output_img_path = os.path.join(detect_folder, latest_exp, file.filename.replace(file_extension, 'jpg'))

    static_output_path = os.path.join('static', 'output', file.filename.replace(file_extension, 'jpg'))
    os.makedirs(os.path.dirname(static_output_path), exist_ok=True)
    shutil.copy(output_img_path, static_output_path)

    output_image_url = url_for('static', filename=f"output/{file.filename.replace(file_extension, 'jpg')}")
    return render_template('infer.html', models=[selected_model], output_image=output_image_url)
