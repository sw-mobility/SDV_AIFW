"""Raw Image Dataset API Router"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Path, Query, status
from fastapi.responses import JSONResponse
from typing import List
from utils.logging import logger
from core.mongodb import MongoDB, get_dataset_collection

from models.dataset.raw.images.mongodb import (
    Dataset, DatasetCreate, 
    ImageFile,
    convert_mongo_document
)
from storage.managers.raw.images.manager import RawImageStorageManager
from .file_handler import RawImageFileHandler

router = APIRouter(prefix="/images", tags=["Raw Images"])

# Singleton instances
_storage_manager = None
_file_handler = None

def get_storage_manager():
    """Get or create storage manager instance"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = RawImageStorageManager()
    return _storage_manager

def get_file_handler():
    """Get or create file handler instance"""
    global _file_handler
    if _file_handler is None:
        _file_handler = RawImageFileHandler()
    return _file_handler

def get_dataset_collection_raw():
    return get_dataset_collection(dataset_type="raw_images")

# Health check endpoint
@router.get("/health")
async def health_check():
    """서비스 상태를 체크합니다."""
    return {"status": "healthy"}

# Dataset endpoints
@router.post("/datasets", response_model=Dataset, status_code=201)
async def create_dataset(dataset: DatasetCreate):
    """
    새로운 원본 이미지 데이터셋을 생성합니다.
    """
    try:
        dataset_collection = get_dataset_collection_raw()
        existing = await dataset_collection.find_one({"name": dataset.name})
        if existing:
            raise HTTPException(status_code=409, detail="Dataset with this name already exists.")
        return await get_storage_manager().create_dataset(dataset)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets", response_model=List[Dataset])
async def list_datasets():
    """
    모든 원본 이미지 데이터셋 목록을 조회합니다.
    """
    try:
        dataset_collection = get_dataset_collection_raw()
        cursor = dataset_collection.find({"type": "raw_images"})
        datasets = []
        async for document in cursor:
            document = convert_mongo_document(document)
            datasets.append(document)
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}", response_model=Dataset)
async def get_dataset(
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름")
):
    """
    특정 원본 이미지 데이터셋의 정보를 조회합니다.
    """
    try:
        dataset = await get_storage_manager().get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset.model_dump()  # dict로 반환
    except Exception as e:
        logger.error(f"Failed to get dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File endpoints
@router.post("/datasets/{dataset_id}/upload", status_code=201)
async def upload_files(
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름"),
    files: List[UploadFile] = File(..., description="업로드할 이미지 파일들")
):
    """
    원본 이미지 카테고리에 파일들을 업로드합니다.
    """
    try:
        # 먼저 업로드할 파일들의 파일명 중복 검사
        upload_filenames = [file.filename for file in files]
        existing_files = await get_storage_manager().list_files(dataset_id)
        existing_filenames = [f.filename for f in existing_files]
        
        # 디버깅용 로그 추가
        logger.info(f"Upload filenames: {upload_filenames}")
        logger.info(f"Existing filenames: {existing_filenames}")
        
        # 중복되는 파일명 찾기
        duplicate_filenames = []
        for filename in upload_filenames:
            if filename in existing_filenames:
                duplicate_filenames.append(filename)
                logger.info(f"Duplicate found: {filename}")
        
        # 중복 파일이 있으면 에러 반환
        if duplicate_filenames:
            return JSONResponse(
                status_code=409,  # Conflict
                content={
                    "message": f"Duplicate files detected: {', '.join(duplicate_filenames)}",
                    "duplicate_files": duplicate_filenames,
                    "error_type": "duplicate_files"
                }
            )
        
        results = []
        filenames = []
        failures = []
        for filename in upload_filenames:
            if filename in existing_filenames:
                duplicate_filenames.append(filename)
                logger.info(f"Duplicate found: {filename}")
        
        # 중복 파일이 있으면 에러 반환
        if duplicate_filenames:
            return JSONResponse(                status_code=409,  # Conflict
                content={
                    "message": f"Duplicate files detected: {', '.join(duplicate_filenames)}",
                    "duplicate_files": duplicate_filenames,
                    "error_type": "duplicate_files"
                }
            )
        
        results = []
        filenames = []
        failures = []
        
        for file in files:
            try:
                file_info = await get_file_handler().upload_file(dataset_id, file)
                results.append(file_info)
                filenames.append(file.filename)
            except Exception as e:
                # Log the error and track failures
                error_msg = str(e)
                logger.error(f"Error uploading {file.filename}: {error_msg}")
                failures.append({"filename": file.filename, "error": error_msg})
        
        # Determine appropriate status code and message based on results
        if len(failures) == 0:
            # All uploads succeeded
            return JSONResponse(
                status_code=201,
                content={
                    "message": f"Successfully uploaded {len(results)} files", 
                    "filenames": filenames
                }
            )
        elif len(results) == 0:
            # All uploads failed
            raise HTTPException(
                status_code=500, 
                detail={"message": "All file uploads failed", "failures": failures}
            )
        else:
            # Partial success (some files uploaded, some failed)
            return JSONResponse(
                status_code=207,  # Multi-Status
                content={
                    "message": f"Partially successful: {len(results)} files uploaded, {len(failures)} files failed",
                    "successful": filenames,
                    "failures": failures
                }
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}/files", response_model=List[ImageFile])
async def list_files(
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름")
):
    """
    원본 이미지 카테고리의 모든 파일 목록을 조회합니다.
    """
    try:
        return await get_storage_manager().list_files(dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))