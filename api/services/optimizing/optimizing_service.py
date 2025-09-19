# /api/services/optimizing/optimizing_service.py
# =========================================================
# 최적화 서비스 공용 유틸
# - API 컨테이너와 optimizing 컨테이너가 **공유 마운트**를 통해 같은 경로로 보이는
#   작업 디렉토리(/workspace/shared/jobs/UID/PID/OID)를 생성/정리합니다.
# - MinIO에서 입력 모델을 내려받아 OID 작업 디렉토리에 **로컬 파일**로 준비하고,
#   작업이 끝난 산출물(파일/디렉토리)을 MinIO의
#   artifacts/{pid}/optimizing/{oid}/... 로 업로드합니다.
# - Optimizing 컨테이너에 넘길 요청의 경로를 **로컬 경로**로 치환하여 전달합니다.
# - 작업 종료 후에는 OID 작업 디렉토리를 정리(shutil.rmtree)합니다.
# =========================================================

import os
import logging
import shutil
from typing import Tuple
from fastapi import HTTPException

from models.optimizing.optimizing_model import (
    OptimizingRequest,
    OptimizingResult,
    OptimizingInfo,
)
from core.minio import MinioStorageClient
from core.config import OPTIMIZING_WORKDIR as CONFIG_API_WORKDIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 공유 작업 루트 경로
# - API 컨테이너와 optimizing 컨테이너 모두에서 **동일한 마운트 경로**로 보여야 합니다.
# - 환경변수 API_WORKDIR > core.config.API_WORKDIR > 기본값 순으로 사용.
#   예: /workspace/shared/jobs
# ---------------------------------------------------------
API_WORKDIR = os.getenv("API_WORKDIR", CONFIG_API_WORKDIR or "/workspace/shared/jobs")


# ---------------- Paths & staging ----------------

def _jobs_root() -> str:
    """공유 작업 루트 절대경로를 반환."""
    return os.path.abspath(API_WORKDIR or "/workspace/shared/jobs")


def job_workdir(uid: str, pid: str, oid: str) -> str:
    """
    OID 작업 디렉토리 생성/반환.
    - 두 컨테이너에서 **같은 경로**로 접근 가능해야 함.
    - 예: /workspace/shared/jobs/{uid}/{pid}/{oid}
    """
    if not oid:
        # OID는 필수. 요청에 없으면 400
        raise HTTPException(status_code=400, detail="Missing oid for optimizing job")
    root = _jobs_root()
    wd = os.path.join(root, uid, pid, oid)
    os.makedirs(wd, exist_ok=True)
    return wd


def ensure_dir(path: str) -> None:
    """mkdir -p path (부모 포함 생성)"""
    os.makedirs(path, exist_ok=True)


# ---------------- MinIO I/O (API-owned) ----------------

async def prepare_model(uid: str, pid: str, oid: str, src_minio_key: str) -> Tuple[str, str]:
    """
    MinIO에서 입력 모델을 다운로드하여 **OID 작업 디렉토리**에 로컬 파일로 준비.
    - src_minio_key: 'artifacts/...' 형태의 MinIO key(앞 슬래시 제거)
    - 반환: (작업 디렉토리 경로, 로컬 모델 파일 절대경로)
    - *.pt / *.onnx / *.engine 은 **항상 best.* 로 저장**합니다.
    """
    minio = MinioStorageClient()
    wd = job_workdir(uid, pid, oid)          # e.g. /workspace/shared/jobs/0001/P0001/O0011
    ensure_dir(wd)

    src_minio_key = (src_minio_key or "").lstrip("/")
    if not src_minio_key:
        raise HTTPException(status_code=400, detail="parameters.input_path is required")

    # 입력 파일은 작업 디렉토리 바로 아래에 저장 (하위 폴더 없이)
    # *.pt / *.onnx / *.engine 는 항상 best.* 로 저장
    src_base = os.path.basename(src_minio_key)
    _, ext = os.path.splitext(src_base)
    ext = ext.lower()
    if ext in (".pt", ".onnx", ".engine"):
        best_name = f"best{ext}"
    else:
        # 예외 포맷은 원래 이름 유지
        best_name = src_base
    local_model_path = os.path.join(wd, best_name)

    try:
        logger.info("MinIO GetObject bucket=%s key=%s -> %s", uid, src_minio_key, local_model_path)
        await minio.download_minio_file(uid, src_minio_key, local_model_path)
        logger.info("Prepared input at %s (uid=%s pid=%s oid=%s)", local_model_path, uid, pid, oid)
    except Exception as e:
        logger.exception("Failed to download model from MinIO: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to prepare model: {e}")

    return wd, local_model_path


async def upload_optimized_artifacts(uid: str, pid: str, oid: str, path: str) -> None:
    """
    산출물 업로드(파일/디렉토리)
    - 업로드 대상:
        artifacts/{pid}/optimizing/{oid}/...
    - 파일이면 파일명 그대로 업로드
    - 디렉토리면 **상대 레이아웃 유지**하여 전체 트리 업로드
    """
    if not path:
        logger.warning("upload_optimized_artifacts called with empty path")
        return

    minio = MinioStorageClient()
    base_prefix = os.path.join("artifacts", pid, "optimizing", oid)

    if os.path.isfile(path):
        key = os.path.join(base_prefix, os.path.basename(path)).replace("\\", "/")
        with open(path, "rb") as f:
            await minio.upload_files(uid, f.read(), key)
        logger.info("Uploaded %s -> %s", path, key)
        return

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in files:
                abs_path = os.path.join(root, fname)
                rel = os.path.relpath(abs_path, start=path)
                key = os.path.join(base_prefix, rel).replace("\\", "/")
                with open(abs_path, "rb") as f:
                    await minio.upload_files(uid, f.read(), key)
        logger.info("Uploaded directory %s -> %s/*", path, base_prefix)
        return

    logger.warning("Nothing to upload; path not found: %s", path)


# ---------------- Request shaping for the Optimizing container ----------------

def build_localized_request(request: OptimizingRequest, local_input_path: str, desired_output_dir: str) -> OptimizingRequest:
    """
    Optimizing 컨테이너에 보낼 요청을 **로컬 경로 기준**으로 재작성.
    - input_path / output_path를 OID 작업 디렉토리 기준 절대경로로 치환
    - output_path 미지정 시 작업 타입에 따라 **항상 best.* 로 자동 설정**
    """
    try:
        req = request.model_copy(deep=True)
    except AttributeError:
        # (pydantic v1 호환)
        req = request.copy(deep=True)

    workdir = os.path.abspath(desired_output_dir)   # 이 디렉토리가 OID 작업 디렉토리
    info = req.info or OptimizingInfo(
        uid=req.uid, pid=req.pid, oid=req.oid, action=req.action, workdir=workdir
    )

    params = req.parameters
    p_updates = {}

    # 입력 파일: OID 디렉토리 내부의 로컬 절대경로
    if hasattr(params, "input_path"):
        p_updates["input_path"] = os.path.abspath(local_input_path)

    # 출력 파일: OID 디렉토리 바로 아래(서브폴더 X)
    if hasattr(params, "output_path"):
        ensure_dir(workdir)
        if getattr(params, "output_path", None):
            # 클라이언트가 파일명을 줬을 때는 **파일명만** 존중(경로 무시)
            base = os.path.basename(str(params.output_path))
            out_path = os.path.join(workdir, base)
        else:
            # 미지정 시, 작업 종류에 따라 항상 best.*
            exts = {
                "prune_unstructured": "pt",
                "prune_structured":   "pt",
                "pt_to_onnx":         "onnx",
                "onnx_to_trt":        "engine",
                "onnx_to_trt_int8":   "engine",
            }
            action_key = getattr(req.action, "value", req.action)
            ext = exts.get(action_key, "out")
            out_path = os.path.join(workdir, f"best.{ext}")
        p_updates["output_path"] = os.path.abspath(out_path)

    # info도 parameters에 주입하여 하위 로직에서 사용 가능하게 함
    new_params = params.model_copy(update={**p_updates, "info": info})
    return req.model_copy(update={"info": info, "parameters": new_params})


async def parse_optimizing_request(request: OptimizingRequest) -> OptimizingInfo:
    """
    OptimizingInfo만 필요할 때 사용: OID 작업 디렉토리 생성 후 info 구성.
    """
    uid, pid, oid = request.uid, request.pid, request.oid
    if not oid:
        raise HTTPException(status_code=400, detail="Missing oid in request")
    workdir = job_workdir(uid, pid, oid)
    return OptimizingInfo(uid=uid, pid=pid, oid=oid, action=request.action, workdir=workdir)


async def stage_for_optimizing(request: OptimizingRequest) -> OptimizingRequest:
    """
    최적화 실행 전 준비 단계:
    - OID 디렉토리 생성
    - input_path가 **절대경로로 로컬에 존재**하면 그대로 사용
    - 그 외에는 MinIO에서 파일을 다운로드하여 OID 디렉토리로 배치
    - calib_dir(있을 경우)은 문자열 타입만 통과(검증은 워커 측에서)
    - 결과: Optimizing 컨테이너에서 바로 사용할 수 있는 **로컬 경로** 기반 요청으로 변환
    """
    uid, pid, oid = request.uid, request.pid, request.oid
    if not oid:
        raise HTTPException(status_code=400, detail="Missing oid in request")

    wd = job_workdir(uid, pid, oid)  # OID dir
    src = getattr(request.parameters, "input_path", None)
    if not src:
        raise HTTPException(status_code=400, detail="parameters.input_path is required")

    # OptimizingInfo 부착 (workdir 포함)
    info = OptimizingInfo(uid=uid, pid=pid, oid=oid, action=request.action, workdir=wd)
    params_with_info = request.parameters.model_copy(update={"info": info})
    request = request.model_copy(update={"info": info, "parameters": params_with_info})

    # INT8용 calib_dir은 문자열만 허용(존재 유무/내용은 워커에서 처리)
    params = params_with_info
    if hasattr(params, "calib_dir"):
        calib_dir = getattr(params, "calib_dir", None)
        if calib_dir and not isinstance(calib_dir, str):
            raise HTTPException(status_code=400, detail="parameters.calib_dir must be a string")

    # 입력이 로컬 절대경로이고 실제로 존재하면 그대로 사용
    if os.path.isabs(src) and os.path.exists(src):
        return build_localized_request(request, src, wd)

    # 아니면 MinIO에서 받아서 OID 디렉토리에 저장 (여기서 best.* 규칙이 적용됨)
    _, local_model_path = await prepare_model(uid, pid, oid, src.replace("\\", "/"))
    return build_localized_request(request, local_model_path, wd)


# ---------------- Cleanup ----------------

async def cleanup_workdir(workdir: str):
    """OID 작업 디렉토리 삭제(디스크 정리). 실패해도 서비스는 계속."""
    try:
        if workdir and os.path.exists(workdir):
            shutil.rmtree(workdir)
            logger.info("Cleaned up workdir: %s", workdir)
    except Exception as e:
        logger.error("Failed to clean up workdir %s: %s", workdir, e)


async def create_optimizing_info(request: OptimizingRequest) -> OptimizingInfo:
    """
    (옵션) 단순히 OptimizingInfo를 만들고 싶을 때 사용.
    - OID 디렉토리까지 생성하여 workdir을 포함한 info 반환
    """
    uid, pid, oid = request.uid, request.pid, request.oid
    if not oid:
        raise HTTPException(status_code=400, detail="Missing oid in request")
    workdir = job_workdir(uid, pid, oid)
    return OptimizingInfo(uid=uid, pid=pid, oid=oid, action=request.action, workdir=workdir)


async def handle_optimizing_result(result: OptimizingResult):
    """
    최종 결과 처리 편의 함수:
    - artifacts_path가 있으면 MinIO에 업로드
      (artifacts/{pid}/optimizing/{oid}/...)
    - OID 작업 디렉토리 정리
    - 라우트 레이어에서 재사용 가능
    """
    uid, pid, oid = result.uid, result.pid, result.oid
    if result.artifacts_path:
        await upload_optimized_artifacts(uid, pid, oid, result.artifacts_path)
    else:
        logger.warning("No artifacts path provided in optimizing result; skipping upload.")

    # 항상 정리 시도
    wd = job_workdir(uid, pid, oid)
    await cleanup_workdir(wd)
    logger.info("Optimizing result processed successfully.")


async def validate_optimizing_request(request: OptimizingRequest):
    """
    요청 기본 검증:
    - uid/pid/oid 필수
    - parameters 필수
    """
    if not request.uid or not request.pid or not request.oid:
        raise HTTPException(status_code=400, detail="Missing required fields in request")
    if not request.parameters:
        raise HTTPException(status_code=400, detail="Missing parameters in request")
    logger.info("Optimizing request validated successfully.")
