import logging
import os
import zipfile
import glob
import random
import shutil
import uuid
import yaml
import json
import requests
from fastapi import HTTPException
from ultralytics import YOLO

def get_yolo_model_filename(model_version, model_size, task_type):
    base = f"yolo{model_version}{model_size}"
    task_map = {
        "object_detection": "",
        "segmentation": "-seg",
        "pose": "-pose",
        "obb": "-obb",
        "classification": "-cls"
    }
    suffix = task_map.get(task_type, "")
    return f"{base}{suffix}.pt"

async def handle_yolov8_det_training(zip_path, project, model_name):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolov8_det_training] Start")
    workspace_dir = "/app/workspace"
    extract_dir = os.path.join(workspace_dir, f"dataset_{uuid.uuid4().hex}")
    os.makedirs(extract_dir, exist_ok=True)
    logger.info(f"[handle_yolov8_det_training] Extracting zip to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f"[handle_yolov8_det_training] Extraction complete")
    data_yaml_path = os.path.join(extract_dir, "data.yaml")
    if not os.path.exists(data_yaml_path):
        logger.error(f"[handle_yolov8_det_training] data.yaml not found in {extract_dir}")
        raise HTTPException(status_code=400, detail="data.yaml not found in dataset zip.")
    logger.info(f"[handle_yolov8_det_training] data.yaml found: {data_yaml_path}")
    split_ratio = project.get("split_ratio", 0.8)
    images_dir = os.path.join(extract_dir, "images")
    labels_dir = os.path.join(extract_dir, "labels")
    train_img_dir = os.path.join(images_dir, "train")
    val_img_dir = os.path.join(images_dir, "val")
    train_lbl_dir = os.path.join(labels_dir, "train")
    val_lbl_dir = os.path.join(labels_dir, "val")
    if not (os.path.exists(train_img_dir) and os.path.exists(val_img_dir)):
        logger.info("[handle_yolov8_det_training] Performing train/val split...")
        os.makedirs(train_img_dir, exist_ok=True)
        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(train_lbl_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)
        img_files = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
        random.shuffle(img_files)
        split_idx = int(len(img_files) * split_ratio)
        train_imgs = img_files[:split_idx]
        val_imgs = img_files[split_idx:]
        for img_path in train_imgs:
            shutil.move(img_path, train_img_dir)
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, base + ".txt")
            if os.path.exists(label_path):
                shutil.move(label_path, train_lbl_dir)
        for img_path in val_imgs:
            shutil.move(img_path, val_img_dir)
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, base + ".txt")
            if os.path.exists(label_path):
                shutil.move(label_path, val_lbl_dir)
        logger.info(f"[handle_yolov8_det_training] Split done: {len(train_imgs)} train, {len(val_imgs)} val")
    else:
        logger.info("[handle_yolov8_det_training] train/val directories already exist. Skip split.")
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f)
    data_yaml["train"] = "images/train"
    data_yaml["val"] = "images/val"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, allow_unicode=True)
    logger.info(f"[handle_yolov8_det_training] data.yaml updated: train/val set to images/train, images/val")
    epochs = project.get("epochs", 10)
    batch = project.get("batch")
    imgsz = project.get("imgsz")
    logger.info(f"[handle_yolov8_det_training] Training with model: {model_name}, epochs: {epochs}")
    try:
        model = YOLO(model_name)
        logger.info(f"[handle_yolov8_det_training] YOLO model loaded: {model_name}")
        train_kwargs = {
            "data": data_yaml_path,
            "epochs": epochs,
            "project": workspace_dir,
            "name": os.path.basename(extract_dir)
        }
        if batch is not None:
            train_kwargs["batch"] = batch
        if imgsz is not None:
            train_kwargs["imgsz"] = imgsz
        results = model.train(**train_kwargs)
        train_log = str(results)
        logger.info(f"[handle_yolov8_det_training] Training finished. Results: {train_log}")
    except Exception as e:
        train_log = str(e)
        logger.error(f"[handle_yolov8_det_training] Training failed: {e}")
    result_dir = None
    if 'results' in locals() and hasattr(results, 'save_dir'):
        result_dir = str(results.save_dir)
    else:
        exp_dirs = sorted(glob.glob(os.path.join(workspace_dir, 'runs', 'detect', 'exp*')), key=os.path.getmtime, reverse=True)
        if exp_dirs:
            result_dir = exp_dirs[0]
    if not result_dir or not os.path.exists(result_dir):
        raise RuntimeError('학습 결과 디렉토리를 찾을 수 없습니다.')
    project_name = project.get('name', project.get('project_id', os.path.basename(result_dir)))
    if not project_name:
        raise RuntimeError('project 정보에 name 또는 project_id가 필요합니다.')
    zip_output = os.path.join(os.path.dirname(result_dir), f"{project_name}.zip")
    shutil.make_archive(os.path.splitext(zip_output)[0], 'zip', result_dir)
    api_url = os.environ.get("API_UPLOAD_URL", "http://api-server:5002/api/v1/train/result/upload")
    files = {'file': (os.path.basename(zip_output), open(zip_output, 'rb'), 'application/zip')}
    data = {'project_id': project_name, 'params': json.dumps(project)}
    try:
        response = requests.post(api_url, files=files, data=data)
        upload_result = response.json() if response.ok else {"error": response.text}
    except Exception as e:
        upload_result = {"error": str(e)}
    finally:
        files['file'][1].close()
    try:
        shutil.rmtree(result_dir)
        os.remove(zip_output)
    except Exception:
        pass
    return {
        "extract_dir": extract_dir,
        "data_yaml": data_yaml_path,
        "train_log": train_log,
        "upload_result": upload_result
    }

async def handle_yolov8_seg_training(zip_path, project, model_name):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolov8_seg_training] Start")
    return "YOLO segmentation training started"

async def handle_yolov8_pose_training(zip_path, project, model_name):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolov8_pose_training] Start")
    return "YOLO pose training started"

async def handle_yolov8_obb_training(zip_path, project, model_name):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolov8_obb_training] Start")
    return "YOLO OBB training started"

async def handle_yolov8_cls_training(zip_path, project, model_name):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolov8_cls_training] Start")
    return "YOLO classification training started"
