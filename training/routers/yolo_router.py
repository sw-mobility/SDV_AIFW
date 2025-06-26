import os
import json
import zipfile
import shutil
import logging
import glob
import random
import uuid
import yaml
import requests

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.train_service import (
    get_yolo_model_filename,
    handle_yolov8_det_training,
    handle_yolov8_seg_training,
    handle_yolov8_pose_training,
    handle_yolov8_obb_training,
    handle_yolov8_cls_training
)

router = APIRouter()

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_router")

@router.post("/train/yolo")
async def execute_training(
    dataset_zip: UploadFile = File(...),
    project_info: str = Form(...)
):
    logger.info("[train/execute] Called. Saving zip file...")
    # 1. zip 파일 저장
    save_path = f"/app/workspace/{dataset_zip.filename}"
    with open(save_path, "wb") as f:
        content = await dataset_zip.read()
        f.write(content)
    logger.info(f"[train/execute] Zip file saved to {save_path}")

    # 2. project_info 파싱
    try:
        project = json.loads(project_info)
        logger.info(f"[train/execute] project_info loaded: {project}")
    except Exception as e:
        logger.error(f"[train/execute] Invalid project_info: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid project_info: {e}")

    # algorithm 필드가 없으면 무조건 YOLO로 간주
    algorithm = project.get("algorithm", None)
    if algorithm is None or algorithm == "yolo":
        task_type = project.get("task_type", "object_detection")
        model_version = project.get("model_version", "v8")
        model_size = project.get("model_size", "n")
        model_name = get_yolo_model_filename(model_version, model_size, task_type)
        logger.info(f"[train/execute] YOLO model_name: {model_name}, task_type: {task_type}")
        if task_type == "object_detection":
            result = await handle_yolov8_det_training(save_path, project, model_name)
        elif task_type == "segmentation":
            result = await handle_yolov8_seg_training(save_path, project, model_name)
        elif task_type == "pose":
            result = await handle_yolov8_pose_training(save_path, project, model_name)
        elif task_type == "obb":
            result = await handle_yolov8_obb_training(save_path, project, model_name)
        elif task_type == "classification":
            result = await handle_yolov8_cls_training(save_path, project, model_name)
        else:
            logger.error(f"[train/execute] Unknown YOLO task_type: {task_type}")
            raise HTTPException(status_code=400, detail=f"Unknown YOLO task_type: {task_type}")
    else:
        logger.error(f"[train/execute] Unknown algorithm: {algorithm}")
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {algorithm}")

    logger.info(f"[train/execute] Training result: {result}")
    return {"status": "received", "result": result}

yolo_router = router
