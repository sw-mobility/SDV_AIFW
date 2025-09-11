from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile,
    Response,
    File, 
    Path, 
    Query, 
    status, 
    Body, 
    Depends,
    Form,
    BackgroundTasks
)
from fastapi.responses import JSONResponse
from typing import (
    List, 
    Optional
)
from core.config import (
    MONGODB_URL, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTIONS,
    API_WORKDIR,
    TRAINING_WORKDIR,
    FRONTEND_WORKDIR,
    YOLO_MODELS
)
from services.training.yolo_training_service import (
    parse_yolo_det_training,
    prepare_dataset,
    create_training_history,
    upload_artifacts,
    prepare_model,
    prepare_codebase,
    upload_preprocessed_dataset
)
from models.training.yolo_training_model import (
    YoloDetTrainingRequest,
    YoloDetTrainingParams,
    YoloTrainingResult,
    YoloHandlingRequest
)
from models.training.common_training_model import (
    TrainingHistory
)
import logging
import requests
import httpx
import os
from utils.init import init
from utils.cleanup import cleanup_workdir
from utils.auth import get_uid
from utils.counter import get_next_counter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yolo", tags=["YOLO Training"])

# 1. YOLO Detection Training 요청
@router.post("/detection", status_code=status.HTTP_200_OK)
async def yolo_training(request: YoloDetTrainingRequest, uid: str = Depends(get_uid)):
    """Start YOLO training with the provided parameters."""
    try:
        #1. 새로운 tid 생성 (validation과 동일한 방식)
        init_result = await init(uid)
        mongo_client = init_result["mongo_client"]
        tid = await get_next_counter(mongo_client, "trn_hst", uid=uid, prefix="T", field="tid", width=4)

        train_info = await parse_yolo_det_training(uid, request)
        if not train_info or not train_info.parameters or not train_info.parameters["model"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid training request.")

        logger.info(f"train_info.parameters['model']: {train_info.parameters['model']}")
        if train_info.parameters["model"] == "best.pt":
            await prepare_model(uid, train_info)
        else:
            pass
        await prepare_codebase(uid, train_info)
        await prepare_dataset(uid, train_info)

        #2. payload 생성 (validation과 동일한 구조)
        payload = {
            "uid": train_info.uid,
            "pid": train_info.pid,
            "origin_tid": train_info.origin_tid,
            "tid": tid,
            "task_type": train_info.task_type,
            "cid": train_info.cid,
            "parameters": train_info.parameters,
            "user_classes": train_info.user_classes,
            "model_classes": train_info.model_classes,
            "dataset_classes": train_info.dataset_classes,
            "workdir": train_info.workdir,
            "did": train_info.did
        }

        logger.info(f"Sending training request to service - uid: {train_info.uid}, pid: {train_info.pid}, tid: {tid}")
        #3. Training 마이크로서비스로 요청 전달
        async with httpx.AsyncClient() as client:
            response = await client.post("http://training:5003/yolo/train/detection", json=payload)
            response.raise_for_status()
        
        #4. Training 결과를 반환 (validation과 동일한 구조)
        result = response.json()
        return {
            "tid": tid,
            "status": result.get("status", "started"),
            "message": result.get("message", "YOLO training started successfully")
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"[training service error] {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=502, detail="Training service error")
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# 2. YOLO Training 결과 핸들러

@router.post("/result", status_code=status.HTTP_200_OK)
async def yolo_training_result(result: YoloHandlingRequest, background_tasks: BackgroundTasks):
    """Handle the result of YOLO training (background upload)."""
    to_parse = result.result
    try:
        if not to_parse:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid training result format.")
        logger.info(f"YOLO Training Result: {result.dict()}")

        trn_hst = await create_training_history(to_parse)

        tid = trn_hst.tid
        artifacts_path = to_parse.artifacts_path
        workdir = result.workdir
        dataset_basepath = os.path.join(workdir, to_parse.pid)
        is_success = to_parse.status

        # 백그라운드 태스크로 업로드 등록
        if is_success == "completed":
            background_tasks.add_task(
                upload_preprocessed_dataset,
                uid=to_parse.uid,
                origin_did=trn_hst.origin_did,
                processed_did=trn_hst.processed_did,
                pid=to_parse.pid,
                tid=tid,
                task_type=to_parse.task_type,
                name=trn_hst.processed_dataset_name,
                dataset_basepath=dataset_basepath,
                classes=to_parse.classes
            )

        if artifacts_path is not None:
            background_tasks.add_task(
                upload_artifacts,
                uid=to_parse.uid,
                pid=to_parse.pid,
                tid=tid,
                workdir=workdir,
                path=artifacts_path,
            )
        else:
            logger.warning("No artifacts path provided in training history, skipping upload.")

        background_tasks.add_task(cleanup_workdir, workdir)

        return JSONResponse(
            content={"message": "Training result processing started in background."},
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        logger.error("Error handling YOLO training result: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{tid}", status_code=status.HTTP_200_OK)
async def get_training_info(tid: str):
    """Get training status and results"""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"http://training:5003/yolo/{tid}")
            resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"[training service error] {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=502, detail="Training service error")
    except Exception as e:
        logger.error(f"Failed to get training info: {e}")
        raise HTTPException(status_code=500, detail=str(e))






    
# # 1. yolo 기본 코드베이스 로드
# @router.get("/default-codebase", status_code=status.HTTP_200_OK)
# async def get_default_assets(uid: str, pid: str):
#     """Get default assets for YOLO training."""
#     try:
#         await prepare_default_codebase(uid, pid)
#         # monaco에 수정 가능한 코드 파싱해서 전송하는 기능 추가
#     except Exception as e:
#         logger.error(f"Failed to get default assets: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get default assets")


# # 2. 선택한 데이터셋의 기본 YAML 로드
# @router.get("/default-yaml", status_code=status.HTTP_200_OK)
# async def get_default_yaml(uid: str, pid: str, dataset_id: str):
#     """Get default YAML configuration for YOLO training."""
#     try:
#         await prepare_default_yaml(uid, pid, dataset_id)
#     except Exception as e:
#         logger.error(f"Failed to get default YAML: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get default YAML")


# # 3. 커스텀 모델 로드
# @router.get("/custom-model", status_code=status.HTTP_200_OK)
# async def get_yolo_model(uid: str, pid: str, tid: str):
#     """Get the custom YOLO model for training."""
#     try:
#         await prepare_model_and_yaml(uid, pid, origin_tid=tid)
#     except Exception as e:
#         logger.error(f"Failed to get YOLO model: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get YOLO model")


# # 4. 커스텀 코드베이스 로드
# @router.get("/custom-codebase", status_code=status.HTTP_200_OK)
# async def get_codebase(uid: str, pid: str, codebase_id: str):
#     """Get the custom codebase for YOLO training."""
#     try:
#         await prepare_codebase(uid, pid, codebase_id)
#     except Exception as e:
#         logger.error(f"Failed to get codebase: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get codebase")
    

# # 5. 스냅샷 생성
# @router.post("/snapshot", status_code=status.HTTP_200_OK)
# async def create_snapshot(uid: str, pid: str, name: str, algorithm: str, task_type: str, 
#                           description: Optional[str]):
#     """Create a snapshot of the current training state."""
#     try:
#         # monaco에서 받은 정보를 frontend 볼륨 안의 codebase에 dump하는 기능 추가
#         await create_training_snapshot(uid, pid, name, algorithm, task_type, description)
#     except Exception as e:
#         logger.error(f"Failed to create snapshot: {e}")
#         raise HTTPException(status_code=500, detail="Failed to create snapshot")