from fastapi import HTTPException, status
from typing import Optional
import os
import yaml
import logging

from core.config import VALIDATION_WORKDIR

from utils.time import get_current_time_kst
from utils.init import init
from models.validation.yolo_validation_model import (
    YoloDetValidationRequest,
    YoloDetValidationInfo,
    ValidationHistory,
)

logger = logging.getLogger(__name__)

# 1) 요청 파싱 변환 + 사전 검증
async def parse_yolo_validation(uid: str, request: YoloDetValidationRequest) -> YoloDetValidationInfo:
    init_result = await init(uid)  # mongo minio 접근 준비
    mongo_client = init_result["mongo_client"]

    # 프로젝트 존재 여부 확인
    project = await mongo_client.db["projects"].find_one({"_id": uid + request.pid})
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    # workdir 계산
    workdir = f"{VALIDATION_WORKDIR}/{uid}/{request.pid}/"
    # 필수 파라미터 검증
    did = request.did
    task_type = request.task_type
    model = request.parameters.model if request.parameters and hasattr(request.parameters, 'model') else None

    return YoloDetValidationInfo(
        uid=uid,
        pid=request.pid,
        tid=request.tid,
        task_type=task_type,
        parameters=request.parameters,
        workdir=workdir,
        did=did,
        cid=request.cid,
    )


# 2) workspace 준비 - codebase 내려받기
async def prepare_codebase_to_workdir(uid: str, info: YoloDetValidationInfo) -> None:
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    os.makedirs(info.workdir, exist_ok=True)

    if info.cid == "yolo" or not info.cid:
        bucket = "keti-aifw"
        prefix = "codebases/yolo"
    else:
        codebase = await mongo_client.db["codebases"].find_one({"_id": uid + info.cid})
        if not codebase:
            raise HTTPException(status_code=404, detail="Codebase not found")
        bucket = uid
        prefix = codebase["path"]
    # 다운로드 실행
    ultralytics_dir = os.path.join(info.workdir, "ultralytics")
    logger.info(f"Downloading codebase from bucket: {bucket}, prefix: {prefix} to: {ultralytics_dir}")
    await minio_client.download_minio_directory(bucket, prefix, ultralytics_dir)
    logger.info(f"Codebase download completed. Ultralytics dir contents: {os.listdir(ultralytics_dir) if os.path.exists(ultralytics_dir) else 'Directory not found'}")

# 3. dataset 내려받기 , data.yaml rewrite
async def prepare_dataset(uid: str, info: YoloDetValidationInfo) -> None:
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    doc = await mongo_client.db["labeled_datasets"].find_one({"_id": uid + info.did})
    if not doc:
        raise HTTPException(status_code=404, detail="Dataset not found")

    key = doc.get("path")
    if not key:
        raise HTTPException(status_code=400, detail="Dataset path missing")

    # 다운로드
    os.makedirs(info.workdir, exist_ok=True)
    await minio_client.download_minio_directory(uid, key, info.workdir)

    # data.yaml 경로 수정
    data_yaml_path = os.path.join(info.workdir, "data.yaml")
    if os.path.exists(data_yaml_path):
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
            data_yaml['path'] = info.workdir
            with open(data_yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.warning(f"Failed to update data.yaml path: {e}")
    else:
        logger.warning("data.yaml not found in dataset")

# 4. best.pt 가져오기  (optional)
async def prepare_model(uid: str, info: YoloDetValidationInfo) -> None:
    if not info.tid:
        return

    if not (info.parameters and hasattr(info.parameters, 'model') and info.parameters.model == "best.pt"):
        return

    logger.info("Preparing training artifact model for validation")

    init_result = await init(uid)
    minio_client = init_result["minio_client"]

    try:
        # Training artifacts 경로에서 모델 다운로드
        origin_weight_path = f"artifacts/{info.pid}/training/{info.tid}/best.pt"

        # 저장 workdir/best.pt
        dest = os.path.join(info.workdir, "best.pt")
        await minio_client.download_minio_file(uid, origin_weight_path, dest)
        logger.info(f"Model downloaded successfully: {dest}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare model: {e}")


# 5.validation history 기록
async def create_validation_history(uid: str, pid: str, result: dict):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    did = result.get("did")
    doc = await mongo_client.db["labeled_datasets"].find_one({"_id": uid + did})
    dataset_name = doc.get("name", "Unknown Dataset") if doc else "Unknown Dataset"
    vid = result.get("vid")
    classes = []
    try:
        workdir = result.get("workdir", "")
        yaml_path = os.path.join(workdir, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                data_yaml = yaml.safe_load(f)
                classes = data_yaml.get('names', [])
    except Exception:
        classes = []

    artifacts_path = None
    if result.get("result_path"):
        artifacts_path = f"artifacts/{pid}/validation/{vid}"
    used_codebase = None
    cid = result.get("cid")
    if cid:
        cb_doc = await mongo_client.db["codebases"].find_one({"_id": uid + cid})
        used_codebase = cb_doc.get("name") if cb_doc else cid

    # metrics_summary 추출
    metrics_summary = None
    if result.get("status") == "completed" and result.get("metrics"):
        metrics_summary = result.get("metrics")
    elif result.get("status") == "completed" and result.get("result_path"):
        # result_path에서 summary.json 파일 읽기 시도
        try:
            summary_file = os.path.join(result.get("workdir", ""), "summary.json")
            if os.path.exists(summary_file):
                import json
                with open(summary_file, 'r') as f:
                    metrics_summary = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read metrics summary from file: {e}")

    history = ValidationHistory(
        _id=uid + pid + vid,
        uid=uid,
        pid=pid,
        vid=vid,
        did=did,
        dataset_name=dataset_name,
        parameters=result.get("parameters", {}),
        classes=classes,
        status=result.get("status", "completed"),
        created_at=get_current_time_kst(),
        used_codebase=used_codebase,
        artifacts_path=artifacts_path,
        error_details=result.get("error_details"),
        metrics_summary=metrics_summary,
    )

    await mongo_client.db["val_hst"].insert_one(history.dict(by_alias=True))
    return history, artifacts_path


# 6. 결과물 업로드 minio
async def upload_validation_artifacts(
        uid: str,
        pid: str,
        vid: str,
        workdir: str,
        result_path: Optional[str],
        plots_path: Optional[str],
) -> None:
    """Upload validation artifacts to MinIO"""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]

    try:
        # 1. 결과 디렉토리 업로드
        if result_path and os.path.exists(result_path):
            # 업로드할 파일 확장자 필터 (시각화 파일 제외)
            useful_extensions = {'.json', '.pt', '.onnx', '.yaml', '.yml'}

            for root, _, files in os.walk(result_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()

                    # labels txt 파일들과 시각화 파일들(png, jpg 등) 제외
                    if file_ext in useful_extensions and not file.startswith('labels'):
                        key = os.path.join("artifacts", pid, "validation", vid, file)
                        file_path = os.path.join(root, file)
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()
                        await minio_client.upload_files(uid, file_bytes, key)
                        logger.info(f"Uploaded {file_path} to {key} in MinIO")
                    else:
                        logger.debug(f"Skipped file (labels or visualization): {file}")

        # 2. 플롯 디렉토리 시각화 파일들만 업로드
        if plots_path and os.path.exists(plots_path):
            # 플롯에서 업로드할 파일 확장자
            plot_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.svg'}

            for root, _, files in os.walk(plots_path):
                for file in files:
                    file_ext = os.path.splitext(file)[1].lower()

                    if file_ext in plot_extensions:
                        file_path = os.path.join(root, file)
                        # plots 하위 구조 유지
                        relative_path = os.path.relpath(file_path, plots_path)
                        key = os.path.join("artifacts", pid, "validation", vid, "plots", relative_path)
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()
                        await minio_client.upload_files(uid, file_bytes, key)
                        logger.info(f"Uploaded {file_path} to {key} in MinIO")
                    else:
                        logger.debug(f"Skipped non-image file in plots: {file_path}")

        # 3. workdir에서 data.yaml 업로드
        for root, _, files in os.walk(workdir):
            for file in files:
                if file == "data.yaml":
                    key = os.path.join("artifacts", pid, "validation", vid, file)
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    await minio_client.upload_files(uid, file_bytes, key)
                    logger.info(f"Uploaded {file_path} to {key} in MinIO")

    except Exception as e:
        logger.error(f"Failed to upload validation artifacts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload validation artifacts: {e}")