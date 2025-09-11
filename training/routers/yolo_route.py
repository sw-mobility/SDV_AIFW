import os
import logging

from fastapi import (
    APIRouter, 
    HTTPException, 
    status
)
from services.yolo_service import (
    YoloTrainingService,
    handle_yolo_det_training,
    handle_yolo_seg_training,
    handle_yolo_pose_training,
    handle_yolo_obb_training,
    handle_yolo_cls_training
)
from models.yolo_training_model import (
    YoloTrainingInfo,
    YoloTrainingResult,
    YoloTrainingResponse
)
from utils.cleanup import cleanup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_router")

router = APIRouter(prefix="/yolo", tags=["Yolo Training"])

training_service = YoloTrainingService()

@router.post("/train/detection", response_model=YoloTrainingResponse, status_code=status.HTTP_200_OK)
async def execute_training(train_info: YoloTrainingInfo):
    """Start YOLO training process"""
    try:
        # 1. 요청 정보 파싱
        task_type = train_info.task_type
        workdir = train_info.workdir
        workdir_clean = workdir.rstrip(os.sep)
        workdir_parent = os.path.dirname(workdir_clean)

        # 2. 지원되는 task_type 확인
        if task_type == "detection":
            tid = await training_service.start_training(train_info)
            return YoloTrainingResponse(
                tid=tid,
                status="started",
                message="YOLO training started successfully"
            )
        else:
            logger.error(f"Coming soon: {task_type}")
            await cleanup(workdir_parent)
            raise HTTPException(status_code=400, detail=f"Coming soon: {task_type}")
            
    except Exception as e:
        logger.error(f"Error starting YOLO training: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{tid}", status_code=status.HTTP_200_OK)
async def get_training_info(tid: str):
    """Get training status and results (unified endpoint)"""
    try:
        training_info = await training_service.get_training_info(tid)
        
        return training_info
    except Exception as e:
        logger.error(f"Error getting training info: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

