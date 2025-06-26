"""이미지 데이터셋 API 모델 정의"""
from typing import List, Optional
from pydantic import BaseModel

class CategoryCreate(BaseModel):
    """카테고리 생성 요청 모델"""
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None

class DatasetCreate(BaseModel):
    """데이터셋 생성 요청 모델"""
    name: str
    description: Optional[str] = None

class FileUploadResponse(BaseModel):
    """파일 업로드 응답 모델"""
    filename: str
    path: str
    status: str
    message: str
    category_id: Optional[str] = None
    label_path: Optional[str] = None

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
