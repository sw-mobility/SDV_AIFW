from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import logging
import httpx
from utils.init import init
from utils.counter import get_next_counter

from core.config import VALIDATION_WORKDIR
from services.validation.yolo_validation_service import (
    parse_yolo_validation,
    prepare_dataset,
    prepare_codebase_to_workdir,
    prepare_model,
    create_validation_history, #validation 결과를 mongo 에 기록
    upload_validation_artifacts, #결과 파일을 minio에 업로드
)
from models.validation.yolo_validation_model import YoloDetValidationRequest
from utils.auth import get_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yolo", tags=["Validation"])


@router.post("/validate/detection", status_code=status.HTTP_200_OK)
async def start_validation(request: YoloDetValidationRequest, uid: str = Depends(get_uid)):
    try:
        #1. 새로운 vid 생성
        init_result = await init(uid)
        mongo_client = init_result["mongo_client"]
        vid = await get_next_counter(mongo_client, "val_hst", uid=uid, prefix="V", field="vid", width=4)

        #2. 요청 내용을 내부적으로 사용하기 위한 info 로 변환, dataset codebase model file 준비
        info = await parse_yolo_validation(uid, request)
        await prepare_dataset(uid, info)
        await prepare_codebase_to_workdir(uid, info)
        # if info.parameters and info.parameters.model == "./best.pt":
        await prepare_model(uid, info)

        #3. payload 생성 (필드명 통일)
        payload = {
            "uid": info.uid,
            "pid": info.pid,
            "tid": info.tid,
            "vid": vid,
            "task_type": info.task_type,
            "cid": info.cid,
            "parameters": info.parameters.dict() if info.parameters else None,
            "did": info.did,
            "workdir": info.workdir
        }

        logger.info(f"Sending validation request to service - uid: {info.uid}, pid: {info.pid}, tid: {info.tid}, vid: {vid}")
        # 4. validation 서비스로 요청 전송
        async with httpx.AsyncClient() as client:
            resp = await client.post("http://validation:5004/yolo/validate/detection", json=payload)
            resp.raise_for_status()
        
        # validation 결과를 반환
        result = resp.json()
        return {
            "vid": vid,
            "status": result.get("status", "started"),
            "message": result.get("message", "YOLO validation started successfully")
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"[validation service error] {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=502, detail="Validation service error")
    except Exception as e:
        logger.error(f"Failed to start validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

"""
validation 상태 및 결과 조회 api 
특정 vid에 대한 검증 상태/결과
내부적으로는 validation 서비스(http://validation:5004/yolo/{vid})로 proxy 요청
"""
@router.get("/{vid}", status_code=status.HTTP_200_OK)
async def get_validation_info(vid: str):
    """Get validation status and results"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://validation:5004/yolo/{vid}")
            resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"[validation service error] {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=502, detail="Validation service error")
    except Exception as e:
        logger.error(f"Failed to get validation info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


"""
Validation 서비스가 콜백을 보낼 때 사용하는 endpoint
"""
@router.post("/result", status_code=status.HTTP_200_OK)
async def receive_result(result: dict):
    """Receive validation completion callback and persist to Mongo/MinIO."""
    try:
        uid = result.get("uid")
        pid = result.get("pid")
        if not (uid and pid):
            raise HTTPException(status_code=400, detail="uid/pid required")

        history, artifacts_path = await create_validation_history(uid, pid, result)

        workdir = result.get("workdir")
        if not workdir:
            workdir = f"{VALIDATION_WORKDIR}/{uid}/{pid}/"

        vid = history.vid
        await upload_validation_artifacts(
            uid, pid, vid,
            workdir=workdir,
            result_path=result.get("result_path"),
            plots_path=result.get("plots_path"),
        )

        return JSONResponse(
            content={"message": "Validation result processed successfully"},
            status_code=status.HTTP_200_OK
        )
    except Exception as e:
        logger.error(f"Failed to process validation result: {e}")
        raise HTTPException(status_code=500, detail=str(e))
