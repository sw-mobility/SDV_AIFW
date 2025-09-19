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
    TRAINING_WORKDIR
)
from services.training.yolo_training_service import (
    parse_yolo_det_training,
    prepare_dataset,
    create_training_history,
    upload_artifacts
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
from utils.init import init
from utils.cleanup import cleanup_workdir
from utils.auth import get_uid

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Training"])

@router.get("/list", status_code=status.HTTP_200_OK)
async def list_training_histories(uid: str = Depends(get_uid)):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    histories = await mongo_client.db["trn_hst"].find({"uid": uid}).to_list(length=None)
    for history in histories:
        history.pop("_id", None)
        history.pop("uid", None)

    return histories