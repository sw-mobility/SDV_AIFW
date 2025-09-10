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
from services.dataset.raw_service import (
    create_raw_dataset,
    list_raw_datasets,
    get_raw_dataset,
    update_raw_dataset,
    upload_raw_data
)
from models.dataset.raw_model import (
    RawDatasetCreate,
    RawDatasetUpdate,
    RawDataUpload,
    RawDatasetInfo,
    RawDataInfo
)
import logging
from utils.init import init
from utils.auth import get_uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/raw", tags=["Raw Dataset"])


# 1. 생성 (POST /raw/)
@router.post("/", status_code=status.HTTP_201_CREATED)
async def create(dataset: RawDatasetCreate, uid: str = Depends(get_uid)):
    """Create a new raw dataset."""
    try:
        dataset_info = await create_raw_dataset(uid, dataset)
        return JSONResponse(status_code=status.HTTP_201_CREATED, content=dataset_info)
    except Exception as e:
        logger.error("Error creating raw dataset: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# 2. 정보 업데이트 (PUT /raw/)
@router.put("/", status_code=status.HTTP_200_OK)
async def update(dataset: RawDatasetUpdate, uid: str = Depends(get_uid)):
    """Update an existing raw dataset."""
    try:
        updated_dataset = await update_raw_dataset(uid, dataset)
        return JSONResponse(status_code=status.HTTP_200_OK, content=updated_dataset)
    except Exception as e:
        logger.error("Error updating raw dataset: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# 3. 전체 목록 조회 (GET /raw/)
@router.get("/", status_code=status.HTTP_200_OK)
async def list(uid: str = Depends(get_uid)):
    """List all raw datasets for a given user."""
    try:
        datasets = await list_raw_datasets(uid)
        return JSONResponse(status_code=status.HTTP_200_OK, content=datasets)
    except Exception as e:
        logger.error("Error listing raw datasets: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 4. 파일 업로드 (POST /raw/data)
@router.post("/upload", status_code=status.HTTP_200_OK)
async def upload(
    raw_data_upload: RawDataUpload = Depends(RawDataUpload.as_form),  # as_form 유틸 필요
    files: List[UploadFile] = File(...),
    uid: str = Depends(get_uid)
):
    try:
        uploaded = await upload_raw_data(uid, raw_data_upload, files)
        return JSONResponse(status_code=status.HTTP_200_OK, content=uploaded)
    except Exception as e:
        logger.error("Error uploading raw data: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 5. 단일 조회 (GET /raw/single)
@router.get("/single", status_code=status.HTTP_200_OK)
async def get(did: str, uid: str = Depends(get_uid)):
    """
    Get details of a specific raw dataset by its ID.
    """
    await init(uid)
    dataset, data_list = await get_raw_dataset(uid, did)
    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    return JSONResponse(status_code=status.HTTP_200_OK, content={"dataset": dataset, "data": data_list})