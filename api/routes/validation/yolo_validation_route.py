from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import logging
import httpx
from utils.init import init
from utils.counter import get_next_counter, get_next_hst_counter

from core.config import VALIDATION_WORKDIR
from services.validation.yolo_validation_service import (
    parse_yolo_validation,
    prepare_dataset,
    prepare_codebase_to_workdir,
    prepare_model,
    create_validation_history, #validation 결과를 mongo 에 기록
    upload_validation_artifacts, #결과 파일을 minio에 업로드
)
from models.validation.yolo_validation_model import YoloDetValidationRequest, ValidationHistory
from utils.auth import get_uid
from utils.time import get_current_time_kst

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yolo", tags=["Validation"])


@router.post("/validate/detection", status_code=status.HTTP_200_OK)
async def start_validation(request: YoloDetValidationRequest, uid: str = Depends(get_uid)):
    try:
        #1. 새로운 vid 생성
        init_result = await init(uid)
        mongo_client = init_result["mongo_client"]
        vid = await get_next_hst_counter(mongo_client, "val_hst", uid=uid, pid=request.pid, prefix="V", field="vid", width=4)

        #2. MongoDB에 초기 상태 저장 (started)
        initial_history = ValidationHistory(
            _id=uid + request.pid + vid,
            uid=uid,
            pid=request.pid,
            vid=vid,
            did=request.did,
            dataset_name="Unknown Dataset",  #나중 callback 에서 update
            parameters=request.parameters.dict() if request.parameters else {},
            classes=[],  #나중 callback 에서 update
            status="started",
            created_at=get_current_time_kst(),
            used_codebase=request.cid,
            artifacts_path=None,
            error_details=None,
            metrics_summary=None,
        )
        
        await mongo_client.db["val_hst"].insert_one(initial_history.dict(by_alias=True))
        logger.info(f"Initial validation history saved for {vid} with status: started")

        #3. 요청 내용을 내부적으로 사용하기 위한 info 로 변환, dataset codebase model file 준비
        info = await parse_yolo_validation(uid, request)
        await prepare_dataset(uid, info)
        await prepare_codebase_to_workdir(uid, info)
        # if info.parameters and info.parameters.model == "./best.pt":
        await prepare_model(uid, info)

        #4. payload 생성 (필드명 통일)
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
        # 5. validation 서비스로 요청 전송
        async with httpx.AsyncClient() as client:
            resp = await client.post("http://validation:5004/yolo/validate/detection", json=payload)
            resp.raise_for_status()
        
        # validation 결과를 반환
        result = resp.json()
        return {
            "vid": vid,
            "status": "started",
            "message": "YOLO validation started successfully"
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"[validation service error] {e.response.status_code}: {e.response.text}")
        # error 발생 시 mongodb 상태 update
        if 'vid' in locals():
            await mongo_client.db["val_hst"].update_one(
                {"vid": vid},
                {"$set": {"status": "failed", "error_details": f"Validation service error: {e.response.text}"}}
            )
        raise HTTPException(status_code=502, detail="Validation service error")
    except Exception as e:
        logger.error(f"Failed to start validation: {e}")
        # error 발생 시 mongodb 상태 update
        if 'vid' in locals():
            await mongo_client.db["val_hst"].update_one(
                {"vid": vid},
                {"$set": {"status": "failed", "error_details": str(e)}}
            )
        raise HTTPException(status_code=500, detail=str(e))

"""
validation 상태 및 결과 조회 api 
특정 vid에 대한 검증 상태/결과
MongoDB에서 직접 조회
"""
@router.get("/{vid}", status_code=status.HTTP_200_OK)
async def get_validation_info(vid: str, uid: str = Depends(get_uid)):
    """Get validation status and results directly from MongoDB"""
    try:
        init_result = await init(uid)
        mongo_client = init_result["mongo_client"]
        
        # MongoDB에서 validation 정보 조회
        validation_doc = await mongo_client.db["val_hst"].find_one({"vid": vid})
        
        if not validation_doc:
            raise HTTPException(status_code=404, detail=f"Validation {vid} not found")
        
        # _id 필드 제거하고 반환
        validation_doc.pop("_id", None)
        return validation_doc
        
    except HTTPException:
        # 이미 HTTPException인 경우 그대로 re-raise
        raise
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

@router.patch("/{vid}/status", status_code=status.HTTP_200_OK)
async def update_validation_status(vid: str, status_data: dict):
    """Update validation status - for validation service to call"""
    try:
        status = status_data.get("status")
        if not status:
            raise HTTPException(status_code=400, detail="status required")
        
        init_result = await init("0001")  # 시스템 레벨 접근
        mongo_client = init_result["mongo_client"]
        
        result = await mongo_client.db["val_hst"].update_one(
            {"vid": vid},
            {"$set": {"status": status}},
            upsert=False
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Validation {vid} not found")
        
        logger.info(f"Updated validation {vid} status to {status}")
        return {"message": f"Validation {vid} status updated to {status}"}
        
    except Exception as e:
        logger.error(f"Failed to update validation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cleanup", status_code=status.HTTP_200_OK)
async def cleanup_validations():
    """Cleanup completed validations - for validation service to call"""
    try:
        init_result = await init("0001")  
        mongo_client = init_result["mongo_client"]
        
        result = await mongo_client.db["val_hst"].delete_many(
            {"status": {"$in": ["completed", "failed"]}}
        )
        deleted_count = result.deleted_count
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} completed/failed validations")
        
        return {"deleted_count": deleted_count, "message": f"Cleaned up {deleted_count} validations"}
        
    except Exception as e:
        logger.error(f"Failed to cleanup validations: {e}")
        raise HTTPException(status_code=500, detail=str(e))
