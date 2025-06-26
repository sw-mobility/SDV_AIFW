"""API 요청/응답 모델 정의"""
from typing import List, Optional
from pydantic import BaseModel

class DatasetCreate(BaseModel):
    """데이터셋 생성 요청 모델"""
    name: str
    description: Optional[str] = None

class CategoryCreate(BaseModel):
    """카테고리 생성 요청 모델"""
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None

class FileUploadResponse(BaseModel):
    """파일 업로드 응답 모델"""
    filename: str
    path: str
    category_id: str
    status: str
    message: str

class FileUploadError(BaseModel):
    """파일 업로드 에러 모델"""
    filename: str
    status: str = "error"
    message: str
    type: str

class BatchUploadResponse(BaseModel):
    """일괄 업로드 응답 모델"""
    dataset: str
    category_id: str
    successful_uploads: List[FileUploadResponse]
    failed_uploads: List[FileUploadError]
    total_files: int
    successful_count: int
    failed_count: int
