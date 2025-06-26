"""Labeled Image Dataset API Router (COCO/YOLO flat 구조, no category)"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Path, Query, Form
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
from utils.logging import logger

from models.dataset.labeled.images.mongodb import Dataset, DatasetCreate, LabeledImageFile, Annotation
from models.dataset.labeled.images.api import AnnotationCreate, AnnotationResponse
from storage.managers.labeled.images.manager import LabeledImageStorageManager
from .file_handler import LabeledImageFileHandler

router = APIRouter(tags=["Generic Labeled Images"])

# Singleton instances
_storage_manager = None
_file_handler = None

def get_storage_manager():
    """Get or create storage manager instance"""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = LabeledImageStorageManager()
    return _storage_manager

def get_file_handler():
    """Get or create file handler instance"""
    global _file_handler
    if _file_handler is None:
        _file_handler = LabeledImageFileHandler()
    return _file_handler

# Health check endpoint
@router.get("/health")
async def health_check():
    """서비스 상태를 체크합니다."""
    return {"status": "healthy"}

# Dataset endpoints
@router.post("/datasets", response_model=Dataset, status_code=201)
async def create_dataset(dataset: DatasetCreate):
    """
    새로운 레이블링 이미지 데이터셋을 생성합니다.
    """
    try:
        return await get_storage_manager().create_dataset(dataset)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets", response_model=List[Dataset])
async def list_datasets():
    """
    모든 레이블링 이미지 데이터셋 목록을 조회합니다.
    """
    try:
        return await get_storage_manager().list_datasets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}", response_model=Dataset)
async def get_dataset(
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름")
):
    """
    특정 레이블링 이미지 데이터셋의 정보를 조회합니다.
    """
    try:
        dataset = await get_storage_manager().get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return dataset.model_dump()  # dict로 반환
    except Exception as e:
        logger.error(f"Failed to get dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름")
):
    """
    레이블링 이미지 데이터셋을 삭제합니다.
    """
    try:
        result = await get_storage_manager().delete_dataset(dataset_id)
        if not result:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# File endpoints
@router.post("/datasets/{dataset_id}/files", status_code=201)
async def upload_file(
    file: UploadFile = File(...),
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름"),
    annotations: Optional[List[AnnotationCreate]] = None
):
    """
    새로운 이미지 파일을 업로드하고 애노테이션을 추가합니다.
    """
    try:
        result, _ = await get_file_handler().upload_file(file, dataset_id, annotations)
        if hasattr(result, 'status') and result.status == "error":
            raise HTTPException(status_code=400, detail=result.message)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}/files", response_model=List[LabeledImageFile])
async def list_files(
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름"),
    skip: int = Query(0, ge=0, description="건너뛸 항목 수"),
    limit: int = Query(100, ge=1, le=1000, description="반환할 최대 항목 수")
):
    """
    데이터셋의 파일 목록을 조회합니다.
    """
    try:
        return await get_storage_manager().list_files(dataset_id, skip, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{file_id}", response_model=LabeledImageFile)
async def get_file(
    file_id: str = Path(..., description="파일 ID")
):
    """
    특정 파일의 정보를 조회합니다.
    """
    try:
        file = await get_storage_manager().get_file(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        return file
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/files/{file_id}", status_code=204)
async def delete_file(
    file_id: str = Path(..., description="파일 ID")
):
    """
    파일을 삭제합니다.
    """
    try:
        result = await get_file_handler().delete_file(file_id)
        if not result:
            raise HTTPException(status_code=404, detail="File not found")
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Annotation endpoints
@router.post("/files/{file_id}/annotations", response_model=AnnotationResponse, status_code=201)
async def create_annotation(
    annotation: AnnotationCreate,
    file_id: str = Path(..., description="파일 ID")
):
    """
    파일에 새로운 애노테이션을 추가합니다.
    """
    try:
        file = await get_storage_manager().get_file(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
            
        # 애노테이션 생성
        annotation_model = Annotation(
            image_id=file_id,
            type=annotation.type,
            coordinates=annotation.coordinates,
            label=annotation.label,
            attributes=annotation.attributes or {}
        )
        
        # 애노테이션 저장
        saved = await get_storage_manager().annotation_manager.create_annotation(annotation_model)
        
        # 파일에 애노테이션 ID 추가
        await get_storage_manager().file_manager.add_annotation_to_file(file_id, saved.id)
        
        # AnnotationResponse 형태로 변환하여 반환
        return AnnotationResponse(
            id=saved.id,
            image_id=saved.image_id,
            type=saved.type,
            coordinates=saved.coordinates,
            label=saved.label,
            attributes=saved.attributes,
            created_at=saved.created_at.isoformat(),
            updated_at=saved.updated_at.isoformat()
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/files/{file_id}/annotations", response_model=List[AnnotationResponse])
async def list_annotations(
    file_id: str = Path(..., description="파일 ID")
):
    """
    파일의 모든 애노테이션을 조회합니다.
    """
    try:
        file = await get_storage_manager().get_file(file_id)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
            
        annotations = await get_storage_manager().annotation_manager.get_annotations_by_image(file_id)
        
        # AnnotationResponse 형태로 변환하여 반환
        return [
            AnnotationResponse(
                id=ann.id,
                image_id=ann.image_id,
                type=ann.type,
                coordinates=ann.coordinates,
                label=ann.label,
                attributes=ann.attributes,
                created_at=ann.created_at.isoformat(),
                updated_at=ann.updated_at.isoformat()
            )
            for ann in annotations
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/annotations/{annotation_id}", status_code=204)
async def delete_annotation(
    annotation_id: str = Path(..., description="애노테이션 ID")
):
    """
    애노테이션을 삭제합니다.
    """
    try:
        # 애노테이션 조회
        annotation = await get_storage_manager().annotation_manager.get_annotation(annotation_id)
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")
            
        # 파일에서 애노테이션 ID 제거
        await get_storage_manager().file_manager.remove_annotation_from_file(
            annotation.image_id, annotation_id
        )
        
        # 애노테이션 삭제
        result = await get_storage_manager().annotation_manager.delete_annotation(annotation_id)
        if not result:
            raise HTTPException(status_code=404, detail="Annotation not found")
            
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
