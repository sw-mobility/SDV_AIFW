import logging
import os
import zipfile
import glob
import random
import shutil
import uuid
import httpx # type: ignore
import yaml
import json
import requests
import sys
from fastapi import HTTPException # type: ignore
# from ultralytics import YOLO
import importlib
import asyncio

from models.yolo_labeling_model import (
    YoloLabelingResult,
    YoloDetLabelingParams,
    YoloDetLabelingInfo,
    YoloHandlingRequest
)

from utils.yolo_utils import *
from utils.time import get_current_time_kst

logger = logging.getLogger("label_router")


async def handle_yolov8_det_labeling(label_info: YoloDetLabelingInfo):
    for k in list(sys.modules.keys()):
        if k == "ultralytics" or k.startswith("ultralytics."):
            sys.modules.pop(k, None)
    importlib.invalidate_caches()

    uid = label_info.uid
    pid = label_info.pid
    parameters = label_info.parameters
    workdir = label_info.workdir
    did = label_info.did
    name = label_info.name
    classes = []

    logger.info("label service test1")

    workdir_clean = workdir.rstrip(os.sep)
    workdir_parent = os.path.dirname(workdir_clean)
    started_time = get_current_time_kst()
    
    artifact_path = os.path.join(workdir,did.replace("R","L"))
    os.makedirs(artifact_path, exist_ok=True)

    try:
        os.chdir(workdir)
        if workdir not in sys.path:
            sys.path.insert(0, workdir)

        # In case of dynamic workspace, "workspace" will be replaced with a variable that corresponds to uid.
        # yolo_path = 'ultralytics/'
        # yolo_path = f"workspace.labeling"
        logger.info("label service test2")
        logger.info(f"Workdir: {workdir}")

        yolo_module = importlib.import_module("ultralytics")
        logger.info("label service test3")

        YOLO = getattr(yolo_module, "YOLO")
        model = YOLO("yolo11n")
        logger.info("label service test4")

        if parameters is None:
            raise HTTPException(status_code=400, detail="Parameters are required.")

        logger.info("Executing YOLOv8 detection labeling")

        rawdata_txt_path = write_rawdata_txt(workdir, did, os.path.basename(artifact_path))
        logger.info("label service test6")

        shutil.copytree(did, os.path.basename(artifact_path), dirs_exist_ok=True)

        # 2. Prepare YOLO params
        label_kwargs = {
                            "model": "yolo11n",
                            "source": rawdata_txt_path,
                            "conf": parameters.conf,
                            "iou": parameters.iou,
                            "imgsz": parameters.imgsz,
                            "rect": parameters.rect,
                            "half": parameters.half,
                            "device": parameters.device,
                            "batch": parameters.batch,
                            "max_det": parameters.max_det,
                            "vid_stride": parameters.vid_stride,
                            "stream_buffer": parameters.stream_buffer,
                            "visualize": parameters.visualize,
                            "augment": parameters.augment,
                            "agnostic_nms": parameters.agnostic_nms,
                            "classes": parameters.classes,
                            "retina_masks": parameters.retina_masks,
                            "embed": parameters.embed,
                            "project": parameters.project,
                            "name": parameters.name,
                            "stream": parameters.stream,
                            "verbose": parameters.verbose,
                            "show": parameters.show,
                            "save": parameters.save,
                            "save_frames": parameters.save_frames,
                            "save_txt": parameters.save_txt,
                            "save_conf": parameters.save_conf,
                            "save_crop": parameters.save_crop,
                            "show_labels": parameters.show_labels,
                            "show_conf": parameters.show_conf,
                            "show_boxes": parameters.show_boxes,
                            "line_width": parameters.line_width
                    }
        # 3. Label the model
        logger.info(f"YOLO detection labeling started")
        label_kwargs = {k: v for k, v in label_kwargs.items() if v is not None}
        logger.info("label service test7")


       # model.label(**label_kwargs)
        results = model(rawdata_txt_path)
        logger.info("label service test8")

        os.remove(os.path.join(workdir, artifact_path, f"{did}.txt"))

        # save label files
        for result in results:
            # logger.info(f"label service testawjfeoweafoiweajfoiwajfe {result}")
            # logger.info(f"label service testawjfeoweafoiweajfoiwajfe {artifact_path}")
            # logger.info(f"label service testawjfeoweafoiweajfoiwajfe {did}")
            # logger.info(f"label service testawjfeoweafoiweajfoiwajfe {result.names}")

            '''
            기존 코드가 basename까지 바꿔버려 image/label 일치가 불가한 분제 수정함
            '''
            save_path = result.path
            # txt_path = save_path.replace("jpg","txt")
            # txt_path = txt_path.replace("jpeg","txt")
            # txt_path = txt_path.replace("png","txt")
            txt_path = save_path.rsplit('.', 1)[0] + '.txt'
            txt_path = txt_path.replace(did, os.path.basename(artifact_path))

            result.save_txt(txt_path)

            if isinstance(result.names, dict):
                classes = [result.names[i] for i in sorted(result.names.keys())]
            else:
                classes = list(result.names)
            # result.save(img_path)

        # Create data.yaml
        data_yaml_path = f"{artifact_path}/data.yaml"
        with open(data_yaml_path, 'w') as file:
            yaml.dump({"train": "train.txt", "val": "val.txt", "test": "test.txt", "nc": len(classes), "names": classes}, file)
        logger.info(f"Created {data_yaml_path}")

        # Exclude internal parameters from the result
        exclude = {"data", "classes", "project", "name", "exist_ok", "save_period", "save"}
        label_kwargs = {k: v for k, v in label_kwargs.items() if k not in exclude}

        logger.info(f"YOLO detection labeling completed")
        completed_time = get_current_time_kst()
        result = YoloLabelingResult(
            uid=uid,
            pid=pid,
            name=name,
            status="completed",
            type="image",
            task_type="detection",
            classes=classes,
            parameters=parameters.dict() if parameters is not None else {},
            started_time=started_time,
            completed_time=completed_time,
            label_format="YOLO",
            workdir=workdir,
            artifacts_path=os.path.join(workdir, artifact_path),
            codebase_id=label_info.cid,
            error_details=None,
            raw_dataset_id=did
            )

        logger.info(f"YOLO detection labeling completed: {result}")

    except Exception as e:
        logger.error(f"Error during YOLO labeling: {e}")
        completed_time = get_current_time_kst()    
        result = YoloLabelingResult(
            uid=uid,
            pid=pid,
            name=name,
            status="failed",
            type="image",
            task_type="detection",
            classes=classes,
            parameters=parameters.dict() if parameters is not None else {},
            started_time=started_time,
            completed_time=completed_time,
            label_format="YOLO",
            workdir=workdir,
            artifacts_path=None,
            codebase_id=label_info.cid,
            error_details=str(e),
            raw_dataset_id=did
            )
    
    async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                "http://api-server:5002/labeling/yolo/result",
                json=YoloHandlingRequest(workdir=workdir_parent, result=result).dict()
            )

    return result



async def handle_yolov8_seg_labeling(label_info):
    logger = logging.getLogger("label_router")
    logger.info("[handle_yolov8_seg_labeling] Start")
    return "YOLO segmentation labeling started"

async def handle_yolov8_pose_labeling(label_info):
    logger = logging.getLogger("label_router")
    logger.info("[handle_yolov8_pose_labeling] Start")
    return "YOLO pose labeling started"

async def handle_yolov8_obb_labeling(label_info):
    logger = logging.getLogger("label_router")
    logger.info("[handle_yolov8_obb_labeling] Start")
    return "YOLO OBB labeling started"

async def handle_yolov8_cls_labeling(label_info):
    logger = logging.getLogger("label_router")
    logger.info("[handle_yolov8_cls_labeling] Start")
    return "YOLO classification labeling started"
