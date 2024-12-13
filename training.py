import os
import subprocess
import shutil
from flask import jsonify
from sklearn.model_selection import train_test_split
import glob

INPUTS_FOLDER = 'Inputs'
MODELS_FOLDER = 'Models'

def get_unique_model_filename(directory, base_filename="best", extension=".pt"):
    counter = 1
    new_filename = f"{base_filename}{extension}"
    
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base_filename}({counter}){extension}"
        counter += 1

    return new_filename

def train(request):
    if 'images' not in request.files or 'labels' not in request.files:
        return jsonify({"error": "Images or labels not uploaded."}), 400

    image_files = request.files.getlist('images')
    label_files = request.files.getlist('labels')

    images_path = os.path.join(INPUTS_FOLDER, 'Images')
    labels_path = os.path.join(INPUTS_FOLDER, 'Labels')
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(labels_path, exist_ok=True)

    # Save uploaded images and labels
    image_paths = []
    for file in image_files:
        if file.filename != '':
            img_path = os.path.join(images_path, file.filename)
            file.save(img_path)
            image_paths.append(img_path)

    for file in label_files:
        if file.filename != '':
            label_path = os.path.join(labels_path, file.filename)
            file.save(label_path)

    if len(image_paths) > 1:
        train_imgs, val_imgs = train_test_split(image_paths, test_size=0.2, random_state=42)
    else:
        train_imgs = image_paths
        val_imgs = []

    custom_data_path = os.path.join(INPUTS_FOLDER, 'custom_data.yaml')
    with open(custom_data_path, 'w') as f:
        f.write(f"train: {images_path}\n")
        f.write(f"val: {images_path}\n")
        f.write(f"nc: 80\n")
        f.write(f"names: ['person', 'bicycle', 'car', ...]\n")  # Add the rest of the classes

    command = [
        'python3', 'train.py', '--img', '640', '--batch', '16', '--epochs', '1',
        '--data', custom_data_path,
        '--weights', 'yolov5/yolov5s.pt',
        '--cfg', 'models/yolov5s.yaml',
    ]
    process = subprocess.Popen(command, cwd='yolov5')
    process.wait()

    models_directory = MODELS_FOLDER
    unique_model_filename = get_unique_model_filename(models_directory)
    model_save_path = os.path.join(models_directory, unique_model_filename)

    runs_dir = os.path.join('yolov5', 'runs', 'train')
    exp_folders = glob.glob(os.path.join(runs_dir, 'exp*'))
    latest_exp_folder = max(exp_folders, key=os.path.getctime)

    original_best_model_path = os.path.join(latest_exp_folder, 'weights', 'best.pt')
    if os.path.exists(original_best_model_path):
        shutil.move(original_best_model_path, model_save_path)
    else:
        return jsonify({"error": "Best model not found."}), 500

    return jsonify({"status": "Training completed", "model_path": model_save_path})
