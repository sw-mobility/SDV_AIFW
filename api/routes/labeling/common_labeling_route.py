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
from services.labeling.yolo_labeling_service import (
    parse_yolo_labeling,
    prepare_dataset,
    create_labeling_history,
    upload_artifacts
)
from models.labeling.yolo_labeling_model import (
    YoloDetLabelingRequest,
    YoloDetLabelingParams,
    YoloLabelingResult,
    YoloHandlingRequest
)
import logging
import requests
import httpx
from utils.init import init
from utils.cleanup import cleanup_workdir

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Labeling"])