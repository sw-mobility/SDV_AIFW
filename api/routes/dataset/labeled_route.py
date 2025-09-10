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
    Form
)
from fastapi.responses import JSONResponse
from typing import (
    List, 
    Optional
)
from core.config import (
    MONGODB_URL, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTIONS
)
from services.dataset.labeled_service import (
    create_labeled_dataset,
    list_labeled_datasets,
    get_labeled_dataset,
    update_labeled_dataset,
    upload_labeled_data
)
from models.dataset.labeled_model import (
    LabeledDatasetCreate,
    LabeledDatasetUpdate,
    LabeledDataUpload,
    LabeledDatasetInfo,
    LabeledDataInfo
)
from utils.init import init
from utils.auth import get_uid

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/labeled", tags=["Labeled Dataset"])

# 1. 생성 (POST /labeled/)
@router.post("/", status_code=status.HTTP_201_CREATED)
async def create(LabeledDatasetCreate: LabeledDatasetCreate, uid: str = Depends(get_uid)):
    """Create a new labeled dataset."""
    try:
        dataset_info = await create_labeled_dataset(uid, LabeledDatasetCreate)
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=dataset_info)
    except Exception as e:
        logger.error("Error creating labeled dataset: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 2. 정보 업데이트 (PUT /labeled/)
@router.put("/", status_code=status.HTTP_200_OK)
async def update(dataset: LabeledDatasetUpdate, uid: str = Depends(get_uid)):
    """Update an existing labeled dataset."""
    updated_dataset = await update_labeled_dataset(uid, dataset)
    return JSONResponse(status_code=status.HTTP_200_OK, content=updated_dataset)


# 3. 전체 목록 조회 (GET /labeled/)
@router.get("/", status_code=status.HTTP_200_OK)
async def list(uid: str = Depends(get_uid)):
    """List all labeled datasets for a given user."""
    try:
        datasets = await list_labeled_datasets(uid)
        return JSONResponse(status_code=status.HTTP_200_OK, content=datasets)
    except Exception as e:
        logger.error("Error listing labeled datasets: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 4. 파일 업로드 (POST /labeled/upload)
@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload(
    labeled_data_upload: LabeledDataUpload = Depends(LabeledDataUpload.as_form),  # as_form 유틸 필요
    files: List[UploadFile] = File(...),
    uid: str = Depends(get_uid)
):
    try:
        docs = await upload_labeled_data(uid, labeled_data_upload, files)
        return docs
    except Exception as e:
        logger.error("Error uploading labeled data: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 5. 단일 조회 (GET /labeled/single)
@router.get("/single", status_code=status.HTTP_200_OK)
async def get(did: str, uid: str = Depends(get_uid)):
    """
    Get details of a specific labeled dataset by its ID.
    """
    dataset, data_list = await get_labeled_dataset(uid, did)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "dataset": dataset,
            "data": data_list
        }
    )