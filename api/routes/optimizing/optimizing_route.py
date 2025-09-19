import os
import logging
import httpx
import os.path as _p

from fastapi import status, Depends, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from models.optimizing.optimizing_model import (
    OptimizingRequest,
    OptimizingResult,
    Action,
    OptimizingRequestBody,
    Status,
)
from models.optimizing.common_optimizing_model import OptimizingHistory
from services.optimizing.optimizing_service import stage_for_optimizing, cleanup_workdir

from core.minio import MinioStorageClient
from utils.init import init
from utils.counter import get_next_counter, get_next_hst_counter
from utils.time import get_current_time_kst
from utils.auth import get_uid

logger = logging.getLogger(__name__)

# 메인(정식) 라우터: 내부 prefix 없음 → 외부 래퍼(/optimizing)와 결합해 /optimizing/*
_router_main = APIRouter(tags=["Optimizing"])

OPT_BASE_URL = os.getenv("OPT_SERVICE_URL", "http://optimizing:5005/optimizing").rstrip("/")

# --- Helpers ---------------------------------------------------------------

def _to_jsonable(obj: BaseModel | dict) -> dict:
    if isinstance(obj, BaseModel):
        try:
            return obj.model_dump(mode="json", exclude_none=True)
        except Exception:
            return obj.dict(exclude_none=True)
    return jsonable_encoder(obj, exclude_none=True)

def _force_best_filename(path: str | None) -> str | None:
    """
    Enforce best.* naming based on extension if a filename is present.
    Works for MinIO keys (artifacts/...) or local paths.
    """
    if not path:
        return None
    base = _p.basename(path)
    root, ext = _p.splitext(base)
    if ext in (".pt", ".onnx", ".engine"):
        # keep parent directories; replace filename only
        return _p.join(_p.dirname(path), f"best{ext}").replace("\\", "/")
    return path

def _normalize_best_in_request(req: OptimizingRequest) -> OptimizingRequest:
    """
    Apply 'best.*' policy to input_path and (if present) output_path.
    If callers still send output_path (legacy), we'll normalize its name to best.*,
    but worker will also default it when omitted.
    """
    params = req.parameters
    in_path  = getattr(params, "input_path",  None)
    out_path = getattr(params, "output_path", None)

    # always force input filename to best.* when it has a known extension
    in_path = _force_best_filename(in_path)

    # allow output_path omission; if given, enforce best.* filename
    out_path = _force_best_filename(out_path)

    try:
        params = params.model_copy(update={"input_path": in_path, "output_path": out_path})
    except Exception:
        # pydantic v1 fallback
        for k, v in (("input_path", in_path), ("output_path", out_path)):
            if hasattr(params, k):
                setattr(params, k, v)

    return req.model_copy(update={"parameters": params})

async def _forward_to_worker(path: str, req: OptimizingRequest) -> OptimizingResult:
    url = f"{OPT_BASE_URL}{path}"
    payload = req.model_dump(mode="json", exclude_none=True)
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(url, json=payload)
    except httpx.RequestError:
        logger.exception("Optimizing service request failed")
        raise HTTPException(status_code=502, detail="Failed to reach optimizing service")
    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)
    return OptimizingResult.model_validate(resp.json())

async def _orchestrate_and_forward(*, action: Action, worker_path: str, request: OptimizingRequest) -> JSONResponse:
    uid, pid = request.uid, request.pid
    init_result = await init(uid)
    mongo = init_result["mongo_client"]

    project = await mongo.db["projects"].find_one({"uid": uid, "pid": pid})
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # 새 OID 발급
    oid = await get_next_hst_counter(mongo, "opt_hst", uid=uid, prefix="O", pid=pid, field="oid", width=4)

    # MinIO에 기본 prefix 예약(폴더 마커)
    minio = MinioStorageClient()
    base_prefix = f"artifacts/{pid}/optimizing/{oid}"
    await minio.upload_files(uid, b"", f"{base_prefix}/.init")

    # 'best.*' 네이밍 규칙 적용 + action/oid 주입
    req_with_oid = request.model_copy(update={"oid": oid, "action": action})
    req_with_oid = _normalize_best_in_request(req_with_oid)

    # 입력(Stage): MinIO key라면 workdir로 다운로드/정규화
    staged = await stage_for_optimizing(req_with_oid)

    # 이력 생성
    started_at = get_current_time_kst()
    hist = OptimizingHistory(
        _id=f"{uid}{pid}{oid}",
        uid=uid, pid=pid, oid=oid,
        status=Status.started,
        started_at=started_at,
        kind=staged.parameters.kind,
        parameters=staged.parameters.model_dump(mode="json"),
        artifacts_path=base_prefix,
    )
    await mongo.db["opt_hst"].insert_one(hist.model_dump(by_alias=True, mode="json"))

    # 워커 호출
    result = await _forward_to_worker(worker_path, staged)

    return JSONResponse(
        content={
            "message": "Optimizing started",
            "uid": uid, "pid": pid, "oid": oid,
            "action": staged.action.value,
            "kind": staged.parameters.kind,
            "service_response": result.model_dump(mode="json"),
        },
        status_code=status.HTTP_202_ACCEPTED,
    )

# =========================
# Endpoints (정식 + 호환 동시 등록)
# =========================
@_router_main.post("/onnx_to_trt", status_code=202)
async def onnx_to_trt_route(request: OptimizingRequestBody, uid: str = Depends(get_uid)):
    if request.parameters.kind != "onnx_to_trt":
        raise HTTPException(status_code=400, detail="parameters.kind must be 'onnx_to_trt'")
    full = OptimizingRequest(uid=uid, pid=request.pid, oid=request.oid, action=Action.onnx_to_trt,
                             parameters=request.parameters, info=None)
    return await _orchestrate_and_forward(action=Action.onnx_to_trt, worker_path="/onnx_to_trt", request=full)

@_router_main.post("/onnx_to_trt_int8", status_code=202)
async def onnx_to_trt_int8_route(request: OptimizingRequestBody, uid: str = Depends(get_uid)):
    if request.parameters.kind != "onnx_to_trt_int8":
        raise HTTPException(status_code=400, detail="parameters.kind must be 'onnx_to_trt_int8'")
    full = OptimizingRequest(uid=uid, pid=request.pid, oid=request.oid, action=Action.onnx_to_trt_int8,
                             parameters=request.parameters, info=None)
    return await _orchestrate_and_forward(action=Action.onnx_to_trt_int8, worker_path="/onnx_to_trt_int8", request=full)

@_router_main.post("/pt_to_onnx_fp32", status_code=202)
async def pt_to_onnx_fp32_route(request: OptimizingRequestBody, uid: str = Depends(get_uid)):
    if request.parameters.kind != "pt_to_onnx":
        raise HTTPException(status_code=400, detail="parameters.kind must be 'pt_to_onnx'")
    full = OptimizingRequest(uid=uid, pid=request.pid, oid=request.oid, action=Action.pt_to_onnx,
                             parameters=request.parameters, info=None)
    return await _orchestrate_and_forward(action=Action.pt_to_onnx, worker_path="/pt_to_onnx_fp32", request=full)

@_router_main.post("/pt_to_onnx_fp16", status_code=202)
async def pt_to_onnx_fp16_route(request: OptimizingRequestBody, uid: str = Depends(get_uid)):
    if request.parameters.kind != "pt_to_onnx":
        raise HTTPException(status_code=400, detail="parameters.kind must be 'pt_to_onnx'")
    full = OptimizingRequest(uid=uid, pid=request.pid, oid=request.oid, action=Action.pt_to_onnx,
                             parameters=request.parameters, info=None)
    return await _orchestrate_and_forward(action=Action.pt_to_onnx, worker_path="/pt_to_onnx_fp16", request=full)

@_router_main.post("/prune_unstructured", status_code=202)
async def prune_unstructured_route(request: OptimizingRequestBody, uid: str = Depends(get_uid)):
    if request.parameters.kind != "prune_unstructured":
        raise HTTPException(status_code=400, detail="parameters.kind must be 'prune_unstructured'")
    full = OptimizingRequest(uid=uid, pid=request.pid, oid=request.oid, action=Action.prune_unstructured,
                             parameters=request.parameters, info=None)
    return await _orchestrate_and_forward(action=Action.prune_unstructured, worker_path="/prune_unstructured", request=full)

@_router_main.post("/prune_structured", status_code=202)
async def prune_structured_route(request: OptimizingRequestBody, uid: str = Depends(get_uid)):
    if request.parameters.kind != "prune_structured":
        raise HTTPException(status_code=400, detail="parameters.kind must be 'prune_structured'")
    full = OptimizingRequest(uid=uid, pid=request.pid, oid=request.oid, action=Action.prune_structured,
                             parameters=request.parameters, info=None)
    return await _orchestrate_and_forward(action=Action.prune_structured, worker_path="/prune_structured", request=full)

@_router_main.post("/check_model_stats", status_code=202)
async def check_model_stats_route(request: OptimizingRequestBody, uid: str = Depends(get_uid)):
    if request.parameters.kind != "check_model_stats":
        raise HTTPException(status_code=400, detail="parameters.kind must be 'check_model_stats'")
    full = OptimizingRequest(uid=uid, pid=request.pid, oid=request.oid, action=Action.check_model_stats,
                             parameters=request.parameters, info=None)
    return await _orchestrate_and_forward(action=Action.check_model_stats, worker_path="/check_model_stats", request=full)

# 콜백(result)은 이미 정식+호환에 모두 등록되어 있어야 합니다.
@_router_main.post("/result", status_code=200)
async def optimizing_result(res: OptimizingResult):
    return await _optimizing_result_impl(res)

# 콜백 처리 구현
async def _optimizing_result_impl(res: OptimizingResult):
    if not (res.uid and res.pid and res.oid):
        raise HTTPException(status_code=400, detail="Invalid result: missing uid/pid/oid")

    uid, pid, oid = res.uid, res.pid, res.oid
    init_result = await init(uid)
    mongo = init_result["mongo_client"]
    minio = MinioStorageClient()

    base_prefix = f"artifacts/{pid}/optimizing/{oid}"

    # 워커가 파일 목록을 넘겨줬다면 우선 사용
    artifact_files = None
    if hasattr(res, "details") and isinstance(res.details, dict):
        artifact_files = res.details.get("artifact_files")

    # 업로드 소스 경로 선택 순서:
    # 1) res.artifacts_path (워커 콜백)
    # 2) /workspace/optimizing/{uid}/{pid}/{oid} (워크디렉터리 폴백)
    candidate = res.artifacts_path
    if not candidate:
        candidate = f"/workspace/optimizing/{uid}/{pid}/{oid}"

    if candidate and _p.exists(candidate):
        if artifact_files:
            for rel_path in artifact_files:
                fp = _p.join(candidate, rel_path)
                if _p.isfile(fp):
                    with open(fp, "rb") as f:
                        await minio.upload_files(uid, f.read(), f"{base_prefix}/{rel_path}")
        else:
            if _p.isfile(candidate):
                with open(candidate, "rb") as f:
                    await minio.upload_files(uid, f.read(), f"{base_prefix}/{_p.basename(candidate)}")
            else:
                for root, _, files in os.walk(candidate):
                    for name in files:
                        fp = _p.join(root, name)
                        rel = os.path.relpath(fp, start=candidate).replace("\\", "/")
                        with open(fp, "rb") as f:
                            await minio.upload_files(uid, f.read(), f"{base_prefix}/{rel}")

    completed_at = get_current_time_kst()
    update_doc = {
        "status": res.status.value if hasattr(res.status, "value") else str(res.status),
        "completed_at": completed_at,
        "error_details": res.message,
        "metrics": res.details,
        "artifacts_path": base_prefix,
    }
    await mongo.db["opt_hst"].update_one(
        {"_id": f"{uid}{pid}{oid}"},
        {"$set": update_doc},
        upsert=True,
    )

    if res.info and getattr(res.info, "workdir", None):
        await cleanup_workdir(res.info.workdir)

    return JSONResponse(
        content={"message": "Optimizing result processed", "uid": uid, "pid": pid, "oid": oid},
        status_code=status.HTTP_200_OK,
    )

# 최종 export: 두 라우터를 하나로 합쳐 외부에서 include할 수 있게 함
from fastapi import APIRouter as _AR
router = _AR()
router.include_router(_router_main)    # /optimizing/*
all = ["router"]
