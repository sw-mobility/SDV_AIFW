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

logger = logging.getLogger(__name__)


class YoloValidationService:
    def __init__(self):
        self.active_validations: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._max_stored_validations = 100

    async def start_validation(self, request: YoloDetValidationServiceRequest) -> str:
        vid = request.vid

        self.active_validations[vid] = {"status": "running"}

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

            # 2. 로컬 커스텀 ultralytics import
            if workdir not in sys.path:
                sys.path.insert(0, workdir)
            
            #  실제로 어떤 ultralytics가 import 되었는지?
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

            # 8. 상태 업데이트 및 콜백 전송
            self.active_validations[vid] = {"status": "completed", "result": result}
            await self._send_completion_callback(result, workdir_parent)

            # 9. 메모리 정리
            await self._auto_cleanup_if_needed()

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

            # 상태 업데이트 및 콜백 전송
            self.active_validations[vid] = {"status": "failed", "error": error_message}
            await self._send_completion_callback(result, workdir_parent)

            # 메모리 정리
            await self._auto_cleanup_if_needed()
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

# _get_plots_path 함수 제거 - 인라인으로 단순화됨

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

        # 3. 성능 통계
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
        """Validation 완료 콜백"""
        # API 서버가 기대하는 구조로 변환
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

    async def get_validation_info(self, vid: str) -> Dict[str, Any]:
        """Validation 정보 조회"""
        if vid not in self.active_validations:
            raise ValueError(f"Validation {vid} not found")

        validation_info = self.active_validations[vid]
        response = {
            "vid": vid,
            "status": validation_info["status"]
        }

        if validation_info["status"] == "completed":
            result = validation_info.get("result")
            if result:
                # 핵심 메트릭만 추출
                essential_metrics = extract_essential_summary(result.metrics)
                response.update({
                    "metrics": essential_metrics,
                    "result_path": result.result_path,
                    "plots_path": result.plots_path
                })
        elif validation_info["status"] == "failed":
            response["error"] = validation_info.get("error")

        return response

    async def cleanup_completed_validations(self):
        """Validation 결과 정리"""
        to_remove = [
            vid for vid, info in self.active_validations.items()
            if info["status"] in ["completed", "failed"]
        ]

        for vid in to_remove:
            del self.active_validations[vid]
            logger.info(f"Cleaned up validation: {vid}")

        return len(to_remove)

    async def _auto_cleanup_if_needed(self):
        """메모리 사용량이 임계값을 초과하면 자동 정리"""
        if len(self.active_validations) > self._max_stored_validations:
            await self.cleanup_completed_validations()

    async def get_active_validations_count(self) -> int:
        """활성 Validation 수 조회"""
        return sum(1 for v in self.active_validations.values() if v.get("status") == "running")

    async def shutdown(self):
        """Validation 서비스 종료"""
        for task in list(self._tasks.values()):
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        logger.info("YoloValidationService shutdown completed")