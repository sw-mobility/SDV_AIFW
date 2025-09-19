import os
import asyncio
import logging
import yaml
from typing import Dict, Any

import sys
import json
import httpx
import importlib

from models.yolo_validation_model import (
    YoloDetValidationServiceRequest,
    YoloValidationResult,
)
from config import settings as validation_settings
from utils import get_current_time_kst, extract_yolo_metrics, extract_essential_summary

API_SERVER_URL = "http://api-server:5002"

logger = logging.getLogger(__name__)


class YoloValidationService:
    def __init__(self):
        self._tasks: Dict[str, asyncio.Task] = {}

    async def _update_validation_status(self, vid: str, status: str):
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.patch(
                f"{API_SERVER_URL}/validation/yolo/{vid}/status",
                json={"status": status}
            )
        logger.info(f"Updated validation {vid} status to {status} via API server")

    async def start_validation(self, request: YoloDetValidationServiceRequest) -> str:
        vid = request.vid
        
        # MongoDB에 running 상태로 업데이트
        await self._update_validation_status(vid, "running")

        task = asyncio.create_task(self._execute_validation(vid, request))
        self._tasks[vid] = task
        task.add_done_callback(lambda t: self._tasks.pop(vid, None))

        logger.info(f"Started validation {vid} for project {request.pid}")
        return vid

    async def _execute_validation(self, vid: str, request: YoloDetValidationServiceRequest):
        for k in list(sys.modules.keys()):
            if k == "ultralytics" or k.startswith("ultralytics."):
                sys.modules.pop(k, None)
        importlib.invalidate_caches()

        """YOLO validation 실행"""
        uid = request.uid
        pid = request.pid
        parameters = request.parameters
        workdir = request.workdir
        did = request.did

        workdir_clean = workdir.rstrip(os.sep)
        workdir_parent = os.path.dirname(workdir_clean)
        started_time = get_current_time_kst()

        try:
            logger.info(f"Executing YOLO validation for {vid}")

            # 1. 작업 디렉토리 변경
            original_cwd = os.getcwd()
            os.chdir(workdir)

            # 2. ultralytics import
            if workdir not in sys.path:
                sys.path.insert(0, workdir)
            
            yolo_module = importlib.import_module("ultralytics")
            YOLO = getattr(yolo_module, "YOLO")
            logger.info(f"Using ultralytics from: {yolo_module.__file__}")

            # 3. 모델 경로 해결
            logger.info(f"Parameters type: {type(parameters)}")
            logger.info(f"Parameters content: {parameters}")

            if parameters and hasattr(parameters, 'model'):
                model_name = parameters.model
            else:
                model_name = "best.pt"

            if model_name == "best.pt":
                model_path = os.path.join(workdir, model_name)
            else:
                model_path = model_name

            # 4. 모델 로드 (YOLO가 자체적으로 검증)
            model = YOLO(model_path)
            logger.info(f"Loaded model: {model_path}")

            # 4. 검증 파라미터 준비
            data_yaml = os.path.join(workdir, "data.yaml")
            
            # data.yaml 파일 존재 확인
            if not os.path.exists(data_yaml):
                logger.error(f"data.yaml not found at: {data_yaml}")
                logger.error(f"Current working directory: {os.getcwd()}")
                logger.error(f"Workdir contents: {os.listdir(workdir) if os.path.exists(workdir) else 'Directory not found'}")
                raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

            with open(data_yaml, 'r') as file:
                data_yaml_dict = yaml.safe_load(file)
                data_yaml_dict['path'] = "."
                data_yaml_dict['train'] = "./images"
                data_yaml_dict['val'] = "./images"
                data_yaml_dict['test'] = "./images"

            with open(data_yaml, 'w') as file:
                yaml.dump(data_yaml_dict, file, default_flow_style=False)
            
            logger.info(f"Found data.yaml at: {data_yaml}")
            val_kwargs = self._prepare_validation_params(parameters, data_yaml, workdir)

            # 5. 검증 실행
            results = await asyncio.to_thread(model.val, **val_kwargs)

            # 6. 메트릭 추출
            metrics = extract_yolo_metrics(results, model)
            result_path = os.path.join(workdir, "validation_results")
            plots_path = str(getattr(results, "save_dir", os.path.join(workdir, "validation")))

            self._save_validation_results(vid, metrics, result_path)

            # 7. 결과 객체 생성
            completed_time = get_current_time_kst()
            result = YoloValidationResult(
                uid=uid,
                pid=pid,
                vid=vid,
                status="completed",
                task_type=request.task_type,
                parameters=parameters.dict() if parameters and hasattr(parameters, 'dict') else {},
                started_time=started_time,
                completed_time=completed_time,
                workdir=workdir,
                result_path=result_path,
                plots_path=plots_path,
                metrics=metrics,
                tid=request.tid,
                cid=request.cid,
                did=did,
                error_details=None
            )

            # 8. 콜백 전송
            await self._send_completion_callback(result, workdir_parent)

            logger.info(f"Validation {vid} completed successfully")

        except Exception as e:
            error_message = str(e)
            logger.error(f"Validation {vid} failed: {error_message}", exc_info=True)

            completed_time = get_current_time_kst()

            # 실패 결과 생성
            result = YoloValidationResult(
                uid=uid,
                pid=pid,
                vid=vid,
                status="failed",
                task_type=request.task_type,
                parameters=parameters.dict() if parameters and hasattr(parameters, 'dict') else {},
                started_time=started_time,
                completed_time=completed_time,
                workdir=workdir,
                result_path=None,
                plots_path=None,
                metrics={},
                tid=request.tid,
                cid=request.cid,
                did=did,
                error_details=error_message
            )

            # 콜백 전송
            await self._send_completion_callback(result, workdir_parent)
        finally:
            # 원래 디렉토리로 복원
            os.chdir(original_cwd)

    def _prepare_validation_params(self, parameters: Any, data_yaml: str, workdir: str) -> Dict[str, Any]:
        """Validation 파라미터 준비"""
        val_kwargs = {
            "data": data_yaml,
            "project": workdir,
            "name": "validation",
            "exist_ok": True,
            "save_json": True,
            "plots": True,
            "verbose": False,
        }

        # 사용자 파라미터 추가
        if parameters and hasattr(parameters, 'device') and parameters.device:
            val_kwargs["device"] = parameters.device
        if parameters and hasattr(parameters, 'batch') and parameters.batch:
            val_kwargs["batch"] = parameters.batch
        if parameters and hasattr(parameters, 'imgsz') and parameters.imgsz:
            val_kwargs["imgsz"] = parameters.imgsz
        if parameters and hasattr(parameters, 'conf') and parameters.conf is not None:
            val_kwargs["conf"] = parameters.conf
        if parameters and hasattr(parameters, 'iou') and parameters.iou is not None:
            val_kwargs["iou"] = parameters.iou

        return {k: v for k, v in val_kwargs.items() if v is not None}


    def _save_validation_results(self, vid: str, metrics: Dict[str, Any], result_path: str):
        """Validation 결과 저장"""
        os.makedirs(result_path, exist_ok=True)

        # 1. 필수 요약 저장
        summary = extract_essential_summary(metrics)
        summary.update({
            "vid": vid,
            "status": "completed"
        })

        summary_file = os.path.join(result_path, "summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # 2. 전체 메트릭 저장
        metrics_file = os.path.join(result_path, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # 3. 성능 통계 저장 (workdir에 )
        stats_text = f"""Validation Results Summary
=========================
Validation ID: {vid}
Total Images: {summary.get('total_images', 0)}

Performance Metrics:
- mAP@0.5: {summary.get('mAP_0.5', 0.0):.4f}
- mAP@0.5:0.95: {summary.get('mAP_0.5_0.95', 0.0):.4f}
- Precision: {summary.get('mean_precision', 0.0):.4f}
- Recall: {summary.get('mean_recall', 0.0):.4f}

Inference Speed:
{json.dumps(summary.get('inference_speed', {}), indent=2)}
"""

        stats_file = os.path.join(result_path, "validation_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(stats_text)

        logger.info(f"Validation results saved to: {result_path}")

    async def _send_completion_callback(self, result: YoloValidationResult, workdir_parent: str):
        callback_data = result.dict()
        callback_data["workdir"] = workdir_parent
        
        # metrics 추가
        if result.metrics:
            essential_metrics = extract_essential_summary(result.metrics)
            callback_data["metrics"] = essential_metrics

        async with httpx.AsyncClient(timeout=30.0) as client:
            await client.post(
                validation_settings.CALLBACK_URL,
                json=callback_data
            )
        logger.info(f"Callback sent for validation {result.vid}")


    async def cleanup_completed_validations(self):
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{API_SERVER_URL}/validation/yolo/cleanup")
            response.raise_for_status()
            result = response.json()
            logger.info(f"Cleaned up validations via API server: {result}")
            return result.get("deleted_count", 0)


    async def get_active_validations_count(self) -> int:
        """현재 실행 중인 validation 작업 수를 로컬에서 반환"""
        return len([task for task in self._tasks.values() if not task.done()])

    async def shutdown(self):
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        logger.info("YoloValidationService shutdown completed")