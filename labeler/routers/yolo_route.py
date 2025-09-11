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
import httpx # type: ignore

from fastapi import ( # type: ignore
    APIRouter, 
    UploadFile, 
    File, 
    Form, 
    HTTPException, 
    status, 
    BackgroundTasks
)
from services.yolo_service import (
    handle_yolov8_det_labeling,
    handle_yolov8_seg_labeling,
    handle_yolov8_pose_labeling,
    handle_yolov8_obb_labeling,
    handle_yolov8_cls_labeling
)
from models.yolo_labeling_model import (
    YoloDetLabelingInfo,
    YoloLabelingResult
)
from utils.cleanup import cleanup

router = APIRouter()
yolo_route = router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("label_router")


@router.post("/yolo", status_code=status.HTTP_200_OK)
async def execute_labeling(
    label_info: YoloDetLabelingInfo,
    background_tasks: BackgroundTasks
):
    # 1. 요청 정보 파싱
    task_type = label_info.task_type
    workdir = label_info.workdir
    workdir_clean = workdir.rstrip(os.sep)
    workdir_parent = os.path.dirname(workdir_clean)

    # 2. project_info 파싱
    if task_type == "detection":
        background_tasks.add_task(handle_yolov8_det_labeling, label_info)
        # background_tasks.add_task(handle_yolov8_det_labeling, label_info)

    # elif task_type == "segmentation":
    #     result = await handle_yolov8_seg_labeling(label_info)
    # elif task_type == "pose":
    #     result = await handle_yolov8_pose_labeling(label_info)
    # elif task_type == "obb":
    #     result = await handle_yolov8_obb_labeling(label_info)
    # elif task_type == "classification":
    #     result = await handle_yolov8_cls_labeling(label_info)
    else:
        logger.error(f"Coming soon: {task_type}")
        await cleanup(workdir_parent)
        raise HTTPException(status_code=400, detail=f"Coming soon: {task_type}")

    return {"message": "YOLO labeling started successfully"}