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
    LABELING_WORKDIR
)
from services.labeling.yolo_labeling_service import (
    parse_yolo_labeling,
    prepare_dataset,
    prepare_model,
    create_labeling_history,
    upload_artifacts,
    prepare_default_codebase,
    prepare_codebase,
    prepare_default_yaml,
    prepare_model_and_yaml,
    create_labeling_snapshot,
    deploy_frontend_files
)
from models.labeling.yolo_labeling_model import (
    YoloDetLabelingRequest,
    YoloDetLabelingParams,
    YoloLabelingResult,
    YoloHandlingRequest
)
from models.labeling.common_labeling_model import (
    LabelingHistory
)
import logging
import requests
import httpx
from utils.init import init
from utils.cleanup import cleanup_workdir
from utils.auth import get_uid
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yolo", tags=["Labeling"])


# 1. YOLO Detection Labeling 요청
@router.post("/detection", status_code=status.HTTP_200_OK)
async def yolo_labeling(request: YoloDetLabelingRequest, uid: str = Depends(get_uid)):
    """Start YOLO labeling with the provided parameters."""
    try:
        label_info = await parse_yolo_labeling(uid, request)
        if not label_info or not label_info.parameters:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid labeling request.")

        await prepare_codebase(uid, label_info)
        await prepare_dataset(uid, label_info)

        async with httpx.AsyncClient() as client:
            await client.post("http://labeler:5006/yolo", json=label_info.dict())

    except Exception as e:
        logger.error("Error starting YOLO training: %s", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return JSONResponse(content={"message": "YOLO training started successfully", "data": label_info.dict()}, status_code=status.HTTP_200_OK)


# 2. YOLO Labeling 결과 핸들러
@router.post("/result", status_code=status.HTTP_200_OK)
async def yolo_labeling_result(result: YoloHandlingRequest):
    """Handle the result of YOLO labeling."""
    to_parse = result.result

    try:
        if not to_parse:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid labeling result format.")
        logger.info(f"YOLO Labeling Result: {result.dict()}")

        artifacts_path = to_parse.artifacts_path
        workdir = result.workdir

        if artifacts_path is not None:
            await upload_artifacts(to_parse
            )
        else:
            logger.warning("No artifacts path provided in labeling history, skipping upload.")

        await cleanup_workdir(workdir)

        return JSONResponse(
            content={"message": "Labeling result processed successfully"},
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        logger.error("Error handling YOLO labeling result: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    



# 1. yolo 기본 코드베이스 로드
# @router.get("/default-codebase", status_code=status.HTTP_200_OK)
# async def get_default_assets(uid: str, pid: str):
#     """Get default assets for YOLO labeling."""
#     try:
#         await prepare_default_codebase(uid, pid)
#         # monaco에 수정 가능한 코드 파싱해서 전송하는 기능 추가
#     except Exception as e:
#         logger.error(f"Failed to get default assets: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get default assets")


# # 2. 선택한 데이터셋의 기본 YAML 로드
# @router.get("/default-yaml", status_code=status.HTTP_200_OK)
# async def get_default_yaml(uid: str, pid: str, dataset_id: str):
#     """Get default YAML configuration for YOLO labeling."""
#     try:
#         await prepare_default_yaml(uid, pid, dataset_id)
#     except Exception as e:
#         logger.error(f"Failed to get default YAML: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get default YAML")


# # 3. 커스텀 모델 로드
# @router.get("/custom-model", status_code=status.HTTP_200_OK)
# async def get_yolo_model(uid: str, pid: str, tid: str):
#     """Get the custom YOLO model for labeling."""
#     try:
#         await prepare_model_and_yaml(uid, pid, origin_tid=tid)
#     except Exception as e:
#         logger.error(f"Failed to get YOLO model: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get YOLO model")


# # 4. 커스텀 코드베이스 로드
# @router.get("/custom-codebase", status_code=status.HTTP_200_OK)
# async def get_codebase(uid: str, pid: str, codebase_id: str):
#     """Get the custom codebase for YOLO labeling."""
#     try:
#         await prepare_codebase(uid, pid, codebase_id)
#     except Exception as e:
#         logger.error(f"Failed to get codebase: {e}")
#         raise HTTPException(status_code=500, detail="Failed to get codebase")
    

# # 5. 스냅샷 생성
# @router.post("/snapshot", status_code=status.HTTP_200_OK)
# async def create_snapshot(uid: str, pid: str, name: str, algorithm: str, task_type: str, 
#                           description: Optional[str]):
#     """Create a snapshot of the current labeling state."""
#     try:
#         # monaco에서 받은 정보를 frontend 볼륨 안의 codebase에 dump하는 기능 추가
#         await create_labeling_snapshot(uid, pid, name, algorithm, task_type, description)
#     except Exception as e:
#         logger.error(f"Failed to create snapshot: {e}")
#         raise HTTPException(status_code=500, detail="Failed to create snapshot")


# # 6. YOLO Labeling 요청
# @router.post("/label", status_code=status.HTTP_200_OK)
# async def yolo_labeling(request: YoloDetLabelingRequest):
#     """Start YOLO labeling with the provided parameters."""
#     try:
        
#         logger.info("test1")
#         label_info = await parse_yolo_labeling(request)
#         dataset_id = label_info.dataset_id
#         workdir = label_info.workdir #f"{LABELING_WORKDIR}/{uid}/{pid}/"
#         logger.info(f"test2 {label_info}")
        
#         await prepare_dataset(request.uid, dataset_id, workdir)
        
#         logger.info("test3")
#         await deploy_frontend_files(request.uid, request.pid, workdir)
        
#         logger.info("test4")

#         async with httpx.AsyncClient() as client:
#             # logger.info("test5")
            
#             # logger.info("test6: %s", label_info.dict())
#             await client.post("http://labeler:5004/yolo", json=label_info.dict())
#         logger.info("YOLO labeling started successfully with parameters: %s", label_info.dict())

#     except Exception as e:
#         logger.error("Error starting YOLO labeling: %s", exc_info=True)
#         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

#     return JSONResponse(content={"message": "Dataset successfully loaded to workspace"}, status_code=status.HTTP_200_OK)