from __future__ import annotations

# ==========================================================
# Optimizing API 라우터
# - 각 최적화 작업(프루닝, PT→ONNX, ONNX→TRT, 통계 추출)을
#   비동기 백그라운드 태스크로 실행하고 즉시 202 Accepted를 반환합니다.
# - 실제 완료/실패 결과는 API_RESULT_URL(콜백)로 POST 됩니다.
# ==========================================================

# ===== Imports =====
import asyncio, logging, os, httpx
from typing import cast
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from models.optimizing_model import (
    OptimizingRequest, OptimizingResult, OptimizingInfo,
    PruneUnstructuredParams, PruneStructuredParams, PtToOnnxParams,
    CheckModelStatsParams, OnnxToTrtParams, OnnxToTrtInt8Params,
)
from services.optimizing_service import (
    prune_unstructured_pt, prune_structured_pt,
    pt_to_onnx_fp32, pt_to_onnx_fp16, check_model_stats,
    onnx_to_trt as svc_onnx_to_trt,
    onnx_to_trt_int8 as svc_onnx_to_trt_int8,
)

# ===== Router & config =====
router = APIRouter(prefix="/optimizing", tags=["Optimizing"])
logger = logging.getLogger(__name__)

# BASE_DIR:
# - 클라이언트가 상대 경로를 보낸 경우, 컨테이너 내부의 작업 루트로 해석할 때 사용
BASE_DIR = "/app/workspace/optimizing"

# 작업 완료/실패 콜백을 보낼 API 엔드포인트 (env로 오버라이드 가능)
API_RESULT_URL = os.getenv("API_RESULT_URL", "http://api-server:5002/optimizing/result")

# ===== Helpers =====
def resolve_path(path: str) -> str:
    """
    입력 경로를 컨테이너 내부 절대경로로 표준화합니다.
    - 절대경로인 경우 그대로 사용
    - 상대경로인 경우 BASE_DIR 하위로 매핑
    - 파일명이 .pt/.onnx/.engine 이면 강제로 'best.*' 파일명으로 정규화
    """
    def _force_best_name(p):
        base = os.path.basename(p)
        ext = os.path.splitext(base)[1]
        if ext in [".pt", ".onnx", ".engine"]:
            return os.path.join(os.path.dirname(p), f"best{ext}")
        return p

    abs_path = path if os.path.isabs(path) else os.path.join(BASE_DIR, path)
    return _force_best_name(abs_path)

def resolve_calib_dir(path: str) -> str:
    """INT8 캘리브레이션 이미지 폴더 경로도 동일한 규칙으로 해석."""
    return resolve_path(path)

def _default_output(info: OptimizingInfo, ext: str) -> str:
    """output_path 미지정 시 info.workdir/best.<ext>로 기본값 설정"""
    return os.path.join(info.workdir, f"best.{ext}")

def _require_params(req: OptimizingRequest):
    """요청에 parameters가 없는 경우 400 + 'best.*' 규칙 검증."""
    if not getattr(req, "parameters", None):
        raise HTTPException(status_code=400, detail="Missing parameters")

    kind = getattr(req.parameters, "kind", None)
    input_path = getattr(req.parameters, "input_path", None)
    output_path = getattr(req.parameters, "output_path", None)
    input_base = os.path.basename(input_path) if input_path else None
    output_base = os.path.basename(output_path) if output_path else None

    if kind in ["prune_unstructured", "prune_structured"]:
        if input_base != "best.pt":
            raise HTTPException(status_code=400, detail="Pruning input must be 'best.pt'")
        if output_path is not None and output_base != "best.pt":
            raise HTTPException(status_code=400, detail="Pruning output must be 'best.pt'")

    elif kind == "pt_to_onnx":
        if input_base != "best.pt":
            raise HTTPException(status_code=400, detail="PT to ONNX input must be 'best.pt'")
        if output_path is not None and output_base != "best.onnx":
            raise HTTPException(status_code=400, detail="PT to ONNX output must be 'best.onnx'")

    elif kind in ["onnx_to_trt", "onnx_to_trt_int8"]:
        if input_base != "best.onnx":
            raise HTTPException(status_code=400, detail="ONNX to TensorRT input must be 'best.onnx'")
        if output_path is not None and output_base != "best.engine":
            raise HTTPException(status_code=400, detail="ONNX to TensorRT output must be 'best.engine'")

    elif kind == "check_model_stats":
        if input_base not in ["best.pt", "best.onnx", "best.engine"]:
            raise HTTPException(status_code=400, detail="Stats input must be 'best.pt', 'best.onnx', or 'best.engine'")

def _ensure_info(req: OptimizingRequest) -> OptimizingRequest:
    """
    OptimizingInfo(workdir 등) 보강.
    - 상위 미들웨어/인증에서 uid/pid/oid/action을 주입했다고 가정
    - info가 없으면 workdir 규칙에 맞게 생성하여 parameters에도 전파
    """
    info = req.info or OptimizingInfo(
        uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
        workdir=f"/app/workspace/optimizing/{req.uid}/{req.pid}/{req.oid}"
    )
    return req.model_copy(update={
        "info": info,
        "parameters": req.parameters.model_copy(update={"info": info})
    })

async def _post_back_async(res: OptimizingResult):
    """
    백그라운드 태스크 완료 후 API 서버로 결과를 콜백.
    - 실패해도 서버는 죽지 않고 로그만 남김(베스트 에포트)
    """
    payload = jsonable_encoder(res.model_dump(mode="json", exclude_none=True), exclude_none=True)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(API_RESULT_URL, json=payload)
            if r.status_code >= 400:
                logger.exception("API callback failed: %s %s", r.status_code, r.text)
    except Exception:
        logger.exception("Failed posting result to API")

async def _run_and_callback(func, params_model, request: OptimizingRequest):
    """
    실제 작업을 실행하고(동기/비동기 모두 지원), 콜백을 보내는 공통 래퍼.
    - 성공 시: status='completed', details에 서비스 쪽 metrics가 담김
    - 실패 시: status='failed', message에 에러 문자열 첨부
    - artifacts_path: 산출물 디렉토리(가능하면 채움; 없으면 info.workdir로 폴백)
    """
    logger.info(f"[BG TASK] Starting {request.action} (oid={request.oid})")
    status, error, artifacts_path = "completed", None, None
    details = {"kind": request.parameters.kind}
    try:
        ret = await func(params_model) if asyncio.iscoroutinefunction(func) else func(params_model)
        if isinstance(ret, dict):
            artifacts_path = ret.get("artifacts_path")
            details = ret.get("metrics") or details
        else:
            artifacts_path = getattr(params_model, "output_path", None)
    except Exception as e:
        status, error = "failed", str(e)
        logger.error(f"[BG TASK] failed for oid={request.oid}: {error}")

    info = params_model.info or request.info
    await _post_back_async(OptimizingResult(
        uid=request.uid, pid=request.pid, oid=request.oid, action=request.action,
        status=status, message=error,
        artifacts_path=(artifacts_path or (info.workdir if info else None)),
        info=info, details=details,
    ))

# ===== Routes =====
@router.post("/prune_structured", response_model=OptimizingResult, status_code=202)
async def prune_structured_route(request: OptimizingRequest):
    """
    구조적 프루닝(Ln) 실행.
    - 입력: pt 모델, amount/n/dim 등
    - 출력: pruned best.pt + <stem>_stats.txt
    - 요청 즉시 202 반환, 완료 결과는 콜백으로 통지
    """
    req = _ensure_info(request); _require_params(req)
    if not req.oid:
        raise HTTPException(status_code=400, detail="Missing oid")
    if req.parameters.kind != "prune_structured":
        raise HTTPException(status_code=400, detail="Wrong parameters kind for prune_structured")

    params = cast(PruneStructuredParams, req.parameters).model_copy(update={
        "input_path": resolve_path(req.parameters.input_path),
        "output_path": (
            resolve_path(req.parameters.output_path)
            if req.parameters.output_path else _default_output(req.info, "pt")
        ),
    })
    logger.info("[Route prune_structured] in=%s out=%s", params.input_path, params.output_path)
    asyncio.create_task(_run_and_callback(prune_structured_pt, params, req))
    return OptimizingResult(uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
                            status="started", info=req.info, details={"kind": params.kind})

@router.post("/prune_unstructured", response_model=OptimizingResult, status_code=202)
async def prune_unstructured_route(request: OptimizingRequest):
    """
    비구조적 프루닝(L1/Random) 실행.
    - Conv/Linear 전역 프루닝 → 저장 후 stats 생성
    """
    req = _ensure_info(request); _require_params(req)
    if not req.oid:
        raise HTTPException(status_code=400, detail="Missing oid")
    if req.parameters.kind != "prune_unstructured":
        raise HTTPException(status_code=400, detail="Wrong parameters kind for prune_unstructured")

    params = cast(PruneUnstructuredParams, req.parameters).model_copy(update={
        "input_path": resolve_path(req.parameters.input_path),
        "output_path": (
            resolve_path(req.parameters.output_path)
            if req.parameters.output_path else _default_output(req.info, "pt")
        ),
    })
    logger.info("[Route prune_unstructured] in=%s out=%s", params.input_path, params.output_path)
    asyncio.create_task(_run_and_callback(prune_unstructured_pt, params, req))
    return OptimizingResult(uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
                            status="started", info=req.info, details={"kind": params.kind})

@router.post("/pt_to_onnx_fp32", response_model=OptimizingResult, status_code=202)
async def pt_to_onnx_fp32_route(request: OptimizingRequest):
    """
    PT → ONNX FP32 내보내기.
    - parameters.input_size: [H, W] = Height(세로), Width(가로)
    - parameters.batch_size: 배치 크기
    """
    req = _ensure_info(request); _require_params(req)
    if not req.oid:
        raise HTTPException(status_code=400, detail="Missing oid")
    if req.parameters.kind != "pt_to_onnx":
        raise HTTPException(status_code=400, detail="Wrong parameters kind for pt_to_onnx (fp32)")

    params = cast(PtToOnnxParams, req.parameters).model_copy(update={
        "input_path": resolve_path(req.parameters.input_path),
        "output_path": (
            resolve_path(req.parameters.output_path)
            if req.parameters.output_path else _default_output(req.info, "onnx")
        ),
    })
    logger.info("[Route pt_to_onnx_fp32] in=%s out=%s", params.input_path, params.output_path)
    asyncio.create_task(_run_and_callback(pt_to_onnx_fp32, params, req))
    return OptimizingResult(uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
                            status="started", info=req.info, details={"kind": params.kind})

@router.post("/pt_to_onnx_fp16", response_model=OptimizingResult, status_code=202)
async def pt_to_onnx_fp16_route(request: OptimizingRequest):
    """
    PT → ONNX FP16 내보내기(half=True).
    - 나머지 옵션은 FP32와 동일
    """
    req = _ensure_info(request); _require_params(req)
    if not req.oid:
        raise HTTPException(status_code=400, detail="Missing oid")
    if req.parameters.kind != "pt_to_onnx":
        raise HTTPException(status_code=400, detail="Wrong parameters kind for pt_to_onnx (fp16)")

    params = cast(PtToOnnxParams, req.parameters).model_copy(update={
        "input_path": resolve_path(req.parameters.input_path),
        "output_path": (
            resolve_path(req.parameters.output_path)
            if req.parameters.output_path else _default_output(req.info, "onnx")
        ),
    })
    logger.info("[Route pt_to_onnx_fp16] in=%s out=%s", params.input_path, params.output_path)
    asyncio.create_task(_run_and_callback(pt_to_onnx_fp16, params, req))
    return OptimizingResult(uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
                            status="started", info=req.info, details={"kind": params.kind})

@router.post("/check_model_stats", response_model=OptimizingResult, status_code=202)
async def check_model_stats_route(request: OptimizingRequest):
    """
    모델 통계만 산출하여 <stem>_stats.txt 생성.
    - .pt / .onnx / .engine 모두 지원 (엔진은 TRT 메타데이터 포함 가능)
    """
    req = _ensure_info(request); _require_params(req)
    if not req.oid:
        raise HTTPException(status_code=400, detail="Missing oid")
    if req.parameters.kind != "check_model_stats":
        raise HTTPException(status_code=400, detail="Wrong parameters kind for check_model_stats")

    params = cast(CheckModelStatsParams, req.parameters).model_copy(update={
        "input_path": resolve_path(req.parameters.input_path),
    })
    logger.info("[Route check_model_stats] in=%s", params.input_path)
    asyncio.create_task(_run_and_callback(check_model_stats, params, req))
    return OptimizingResult(uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
                            status="started", info=req.info, details={"kind": params.kind})

@router.post("/onnx_to_trt", response_model=OptimizingResult, status_code=202)
async def onnx_to_trt_route(request: OptimizingRequest):
    """
    ONNX → TensorRT 엔진(FP32/FP16) 빌드(trtexec).
    - parameters.precision: "fp32" | "fp16"
    - parameters.device: "gpu" | "dla" | "dla0" | "dla1"
      * DLA 미지원 연산/사이즈는 자동 GPU 폴백(allowGPUFallback)
    - parameters.output_path: 미지정 시 info.workdir/best.engine 으로 저장
    """
    req = _ensure_info(request); _require_params(req)
    if not req.oid:
        raise HTTPException(status_code=400, detail="Missing oid")
    if req.parameters.kind != "onnx_to_trt":
        raise HTTPException(status_code=400, detail="Wrong parameters kind for onnx_to_trt")

    params = cast(OnnxToTrtParams, req.parameters).model_copy(update={
        "input_path": resolve_path(req.parameters.input_path),
        "output_path": (
            resolve_path(req.parameters.output_path)
            if req.parameters.output_path else _default_output(req.info, "engine")
        ),
    })
    logger.info("[Route onnx_to_trt] in=%s out=%s", params.input_path, params.output_path)
    asyncio.create_task(_run_and_callback(svc_onnx_to_trt, params, req))
    return OptimizingResult(uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
                            status="started", info=req.info, details={"kind": params.kind})

@router.post("/onnx_to_trt_int8", response_model=OptimizingResult, status_code=202)
async def onnx_to_trt_int8_route(request: OptimizingRequest):
    """
    ONNX → TensorRT INT8 양자화(Python API).
    - parameters.calib_dir: 컨테이너 내부 이미지 폴더(필수)
    - parameters.input_size: [H, W] (캘리브레이션 리사이즈 기준)
    - parameters.device: "gpu" | "dla"  (*스키마상 'dla0/1' 리터럴은 미허용)
    - parameters.mixed_fp16/sparse/int8_max_batches/workspace_mib: 선택 옵션
    - parameters.output_path: 미지정 시 info.workdir/best.engine 으로 저장
    """
    req = _ensure_info(request); _require_params(req)
    if not req.oid:
        raise HTTPException(status_code=400, detail="Missing oid")
    if req.parameters.kind != "onnx_to_trt_int8":
        raise HTTPException(status_code=400, detail="Wrong parameters kind for onnx_to_trt_int8")

    params = cast(OnnxToTrtInt8Params, req.parameters).model_copy(update={
        "input_path": resolve_path(req.parameters.input_path),
        "output_path": (
            resolve_path(req.parameters.output_path)
            if req.parameters.output_path else _default_output(req.info, "engine")
        ),
        "calib_dir": resolve_calib_dir(req.parameters.calib_dir),
    })

    # 캘리브레이션 폴더 존재 확인(이미지 최소 1장 요구는 서비스 레벨에서 검증)
    if not os.path.isdir(params.calib_dir):
        raise HTTPException(status_code=400, detail=f"calib_dir not found: {params.calib_dir}")

    logger.info("[Route onnx_to_trt_int8] in=%s out=%s calib_dir=%s",
                params.input_path, params.output_path, params.calib_dir)
    asyncio.create_task(_run_and_callback(svc_onnx_to_trt_int8, params, req))
    return OptimizingResult(uid=req.uid, pid=req.pid, oid=req.oid, action=req.action,
                            status="started", info=req.info, details={"kind": params.kind})
