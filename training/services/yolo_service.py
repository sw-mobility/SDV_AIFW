import logging
import os
import zipfile
import glob
import random
import shutil
import uuid
import httpx
import yaml
import json
import requests
import asyncio
import sys
from typing import Dict, Any
from fastapi import HTTPException
# from ultralytics import YOLO
import importlib

from models.yolo_training_model import (
    YoloTrainingResult,
    YoloDetTrainingParams,
    YoloTrainingInfo,
    YoloHandlingRequest
)

from utils.yolo_utils import (
    match_image_label_files,
    split_dataset,
    write_split_txt,
    dataset2model_mapping
)
from utils.time import get_current_time_kst

logger = logging.getLogger("train_router")


class YoloTrainingService:
    def __init__(self):
        self.active_trainings: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._max_stored_trainings = 100

    async def start_training(self, train_info: YoloTrainingInfo) -> str:
        """Training 시작"""
        # API 서버에서 전달받은 tid 사용
        tid = train_info.tid if hasattr(train_info, 'tid') and train_info.tid else f"train_{uuid.uuid4().hex[:8]}"
        
        self.active_trainings[tid] = {"status": "running"}
        
        task = asyncio.create_task(self._execute_training(tid, train_info))
        self._tasks[tid] = task
        task.add_done_callback(lambda t: self._tasks.pop(tid, None))
        
        logger.info(f"Started training {tid} for project {train_info.pid}")
        return tid

    async def _execute_training(self, tid: str, train_info: YoloTrainingInfo):
        """Training 실행 (기존 handle_yolo_det_training 로직 이용)"""
        try:
            result = await handle_yolo_det_training(train_info)
            self.active_trainings[tid] = {"status": "completed", "result": result}
            await self._auto_cleanup_if_needed()
            logger.info(f"Training {tid} completed successfully")
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Training {tid} failed: {error_message}", exc_info=True)
            self.active_trainings[tid] = {"status": "failed", "error": error_message}
            await self._auto_cleanup_if_needed()

    async def get_training_info(self, tid: str) -> Dict[str, Any]:
        """Training 정보 조회"""
        # 먼저 메모리에서 조회
        if tid in self.active_trainings:
            training_info = self.active_trainings[tid]
            response = {
                "tid": tid,
                "status": training_info["status"]
            }
            
            if training_info["status"] == "completed":
                result = training_info.get("result")
                if result:
                    response.update({
                        "artifacts_path": result.artifacts_path,
                        "started_time": result.started_time,
                        "completed_time": result.completed_time,
                        "classes": result.classes,
                        "task_type": result.task_type
                    })
            elif training_info["status"] == "failed":
                response["error"] = training_info.get("error")
            
            return response
        
        # 메모리에 없으면 MongoDB에서 조회 (validation과 동일한 방식)
        try:
            # MongoDB 연결 (간단한 방식)
            from pymongo import MongoClient
            client = MongoClient("mongodb://localhost:27017/")
            db = client["keti-aifw"]
            
            # training 정보 조회
            training_doc = db["trn_hst"].find_one({"tid": tid})
            
            if not training_doc:
                raise ValueError(f"Training {tid} not found")
            
            response = {
                "tid": tid,
                "status": training_doc.get("status", "unknown")
            }
            
            if training_doc.get("status") == "completed":
                response.update({
                    "artifacts_path": training_doc.get("artifacts_path"),
                    "started_time": training_doc.get("started_at"),
                    "completed_time": training_doc.get("completed_at"),
                    "classes": training_doc.get("classes", []),
                    "task_type": "detection"
                })
            elif training_doc.get("status") == "failed":
                response["error"] = training_doc.get("error_details")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to get training info from MongoDB: {e}")
            raise ValueError(f"Training {tid} not found")

    async def _auto_cleanup_if_needed(self):
        """메모리 사용량이 임계값을 초과하면 자동 정리"""
        if len(self.active_trainings) > self._max_stored_trainings:
            # 완료된 training들을 자동으로 정리
            to_remove = [
                tid for tid, info in self.active_trainings.items()
                if info["status"] in ["completed", "failed"]
            ]
            
            for tid in to_remove:
                del self.active_trainings[tid]
                logger.info(f"Auto cleaned up training: {tid}")

    async def shutdown(self):
        """Training 서비스 종료"""
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()
        
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        
        logger.info("YoloTrainingService shutdown completed")

async def handle_yolo_det_training(train_info: YoloTrainingInfo):
    for k in list(sys.modules.keys()):
        if k == "ultralytics" or k.startswith("ultralytics."):
            sys.modules.pop(k, None)
    importlib.invalidate_caches()

    uid = train_info.uid
    pid = train_info.pid
    parameters = train_info.parameters
    workdir = train_info.workdir
    did = train_info.did
    model_name = parameters["model"]
    origin_tid = train_info.origin_tid if train_info.origin_tid else model_name
    pretrained = False if model_name == "best.pt" else True
    user_classes = train_info.user_classes
    model_classes = train_info.model_classes
    dataset_classes = train_info.dataset_classes

    workdir_clean = workdir.rstrip(os.sep)
    workdir_parent = os.path.dirname(workdir_clean)
    started_time = get_current_time_kst()

    data_yaml_path = "data.yaml"
    image_path = "./images"
    label_path = "./labels"

    try:
        os.chdir(workdir)
        logger.info(f"Changed working directory to {workdir}, contents: {os.listdir('.')}")

        # 0) 로컬 커스텀 ultralytics가 최우선으로 import 되게 보장
        # (이미 CWD가 sys.path[0]인 경우가 많지만, 안전하게 한 번 더 넣음)
        if workdir not in sys.path:
            sys.path.insert(0, workdir)

        # 0-1) 실제로 어떤 ultralytics가 import 되었는지 경로 확인 로그
        yolo_module = importlib.import_module("ultralytics")
        YOLO = getattr(yolo_module, "YOLO")
        logger.info(f"Using ultralytics from: {yolo_module.__file__}")

        if parameters is None:
            raise HTTPException(status_code=400, detail="Parameters are required.")
        
        logger.info(f"model_name: {model_name}, best.pt exists: {os.path.exists('./best.pt')}")
        if model_name == "best.pt" and os.path.exists("./best.pt"):
            try:
                model = YOLO("./best.pt")
                logger.info(f"model type: {type(model)}")
                logger.info(f"model.model type: {type(model.model)}")
                logger.info(f"model summary: {getattr(model, 'info', lambda: 'no info method')()}")
                # best.pt가 실제로 로드된 모델의 가중치인지 확인 (예: 모델 구조, summary 등 체크)
                logger.info("YOLO model set to best.pt")
            except Exception as e:
                logger.error(f"Failed to load YOLO model from ./best.pt: {e}")
                raise HTTPException(status_code=500, detail=f"error: {e}")

        else:
            model = YOLO(model_name)
            logger.info(f"YOLO model file set to {model_name}")

        logger.info("Executing YOLOv8 detection training")

        # 1. Match dataset classes to model classes (If using custom model)
        await dataset2model_mapping(label_path, image_path, dataset_classes, user_classes)

        # 2. Reconstruct data.yaml
        if not os.path.exists(data_yaml_path):
            raise HTTPException(status_code=400, detail="data.yaml file is required for YOLO training.")

        matched_basenames = await match_image_label_files(image_path, label_path)
        # logger.info(f"Matched image and label files: {matched_basenames}")
        if not matched_basenames:
            raise HTTPException(status_code=400, detail="No matching image and label files found.")

        train_files, val_files, test_files = await split_dataset(matched_basenames, 
                                                        parameters["split_ratio"] if parameters["split_ratio"] else [0.8, 0.2, 0.0])

        if not train_files or not val_files:
            raise HTTPException(status_code=400, detail="Failed to split dataset.")

        await write_split_txt(image_path, train_files, "train", ".")
        await write_split_txt(image_path, val_files, "val", ".")
        await write_split_txt(image_path, test_files, "test", ".")

        with open(data_yaml_path, 'r') as file:
            data_yaml_dict = yaml.safe_load(file)
            data_yaml_dict['path'] = "."
            data_yaml_dict['train'] = "train.txt"
            data_yaml_dict['val'] = "val.txt"
            data_yaml_dict['test'] = "test.txt"
            # if user_classes:
            data_yaml_dict['nc'] = len(user_classes)
            data_yaml_dict['names'] = user_classes
            # else:
            #     pass

        with open(data_yaml_path, 'w') as file:
            yaml.dump(data_yaml_dict, file, default_flow_style=False)
        
        data_yaml_content = yaml.safe_load(open(data_yaml_path))
        logger.info(f"Loaded data.yaml")
        logger.info(f"data.yaml content: {data_yaml_content}")

        # 3. Prepare YOLO params
        train_kwargs = {
                        "data": data_yaml_path,
                        "epochs": parameters["epochs"],
                        "imgsz": parameters["imgsz"],
                        "batch": parameters["batch"],
                        "device": parameters["device"],
                        "project": workdir,
                        "name": "results",
                        "exist_ok": True,
                        "save_period": parameters["save_period"],
                        "save": True,
                        "workers": parameters["workers"],
                        "patience": parameters["patience"],
                        "optimizer": parameters["optimizer"],
                        "lr0": parameters["lr0"],
                        "lrf": parameters["lrf"],
                        "momentum": parameters["momentum"],
                        "weight_decay": parameters["weight_decay"],
                        "warmup_epochs": parameters["warmup_epochs"],
                        "warmup_momentum": parameters["warmup_momentum"],
                        "warmup_bias_lr": parameters["warmup_bias_lr"],
                        "seed": parameters["seed"] if parameters["seed"] is not None else 0,
                        "cache": parameters["cache"],
                        "dropout": parameters["dropout"],
                        "label_smoothing": parameters["label_smoothing"],
                        "rect": parameters["rect"],
                        "pretrained": pretrained,
                        "resume": False,
                        "amp": False,
                        "single_cls": False,
                        "cos_lr": parameters["cos_lr"],
                        "close_mosaic": parameters["close_mosaic"],
                        "overlap_mask": parameters["overlap_mask"],
                        "mask_ratio": parameters["mask_ratio"]
                    }
        # 4. Train the model
        logger.info(f"YOLO detection training started")
        train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
        await asyncio.to_thread(model.train, model=model_name, **train_kwargs)

        # 5. Prepare results to send
        # Exclude internal parameters from the result
        exclude = {"data", "project", "name", "exist_ok", "save_period", "save"}
        train_kwargs = {k: v for k, v in train_kwargs.items() if k not in exclude}

        logger.info(f"YOLO detection training completed")
        completed_time = get_current_time_kst()
        result = YoloTrainingResult(
            uid=uid,
            pid=pid,
            tid=train_info.tid if hasattr(train_info, 'tid') else None,
            status="completed",
            task_type="detection",
            classes=user_classes,
            parameters=parameters if parameters is not None else {},
            started_time=started_time,
            completed_time=completed_time, 
            workdir=workdir,
            artifacts_path=os.path.join(workdir, "results"),
            origin_tid=origin_tid,
            cid=train_info.cid,
            error_details=None,
            did=did
            )

    except Exception as e:
        logger.error(f"Error during YOLO training: {e}")
        completed_time = get_current_time_kst()
        result = YoloTrainingResult(
            uid=uid,
            pid=pid,
            tid=train_info.tid if hasattr(train_info, 'tid') else None,
            status="failed",
            task_type="detection",
            classes=user_classes,
            parameters=parameters if parameters is not None else {},
            started_time=started_time,
            completed_time=completed_time,
            workdir=workdir,
            artifacts_path=None,
            cid=train_info.cid,
            origin_tid=origin_tid,
            error_details=str(e),
            did=did
            )
    
    async with httpx.AsyncClient() as client:
            await client.post(
                "http://api-server:5002/training/yolo/result",
                json=YoloHandlingRequest(workdir=workdir_parent, result=result).dict()
            )

    return result



async def handle_yolo_seg_training(train_info):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolo_seg_training] Start")
    return "YOLO segmentation training started"

async def handle_yolo_pose_training(train_info):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolo_pose_training] Start")
    return "YOLO pose training started"

async def handle_yolo_obb_training(train_info):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolo_obb_training] Start")
    return "YOLO OBB training started"

async def handle_yolo_cls_training(train_info):
    logger = logging.getLogger("train_router")
    logger.info("[handle_yolo_cls_training] Start")
    return "YOLO classification training started"
