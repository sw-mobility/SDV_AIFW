"""이미지 레이블링 데이터셋 API 모델 정의 (COCO/YOLO flat 구조, no category)"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class AnnotationCreate(BaseModel):
    """어노테이션 생성 요청 모델"""
    type: str
    coordinates: Dict[str, Any]
    label: str
    attributes: Optional[Dict[str, Any]] = None

class DatasetCreate(BaseModel):
    """데이터셋 생성 요청 모델"""
    name: str
    description: Optional[str] = None
    annotation_types: Optional[List[str]] = Field(default_factory=lambda: ["bbox"])

class FileUploadResponse(BaseModel):
    """파일 업로드 응답 모델 (flat 구조)"""
    filename: str
    path: str
    status: str
    message: str
    # category_id 등 카테고리 관련 필드 제거

class FileUploadError(BaseModel):
    """파일 업로드 에러 모델 (flat 구조)"""
    filename: str
    status: str = "error"
    message: str
    type: str

class BatchUploadResponse(BaseModel):
    """일괄 업로드 응답 모델 (flat 구조)"""
    dataset: str
    successful_uploads: List[FileUploadResponse]
    failed_uploads: List[FileUploadError]
    total_files: int
    successful_count: int
    failed_count: int
    # category_id 등 카테고리 관련 필드 제거

class AnnotationResponse(BaseModel):
    """어노테이션 응답 모델 (flat 구조)"""
    id: str
    image_id: str
    type: str
    coordinates: Dict[str, Any]
    label: str
    attributes: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
