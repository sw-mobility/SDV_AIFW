"""MongoDB 데이터 모델 (COCO/YOLO flat 구조, no category)"""
from typing import Optional, List, Any, Dict, ClassVar
from pydantic import (
    BaseModel, Field, ConfigDict, field_validator, 
    model_validator, computed_field, TypeAdapter, ValidationInfo
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from bson import ObjectId, errors as bson_errors

from models.dataset.raw.images.mongodb import MongoBaseModel

class Annotation(MongoBaseModel):
    """이미지 어노테이션 모델 (flat 구조)"""
    collection_name: ClassVar[str] = "labeled_image_annotations"
    
    image_id: str
    type: str  # 'bbox', 'polygon', 'segmentation' 등
    coordinates: Dict[str, Any]
    label: str
    attributes: Dict[str, Any] = Field(default_factory=dict)
    # category_id 등 카테고리 관련 필드 제거

    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """애노테이션 타입 검증"""
        allowed_types = {'bbox', 'polygon', 'segmentation', 'keypoint', 'line', 'point'}
        if v not in allowed_types:
            raise ValueError(f"Annotation type must be one of {allowed_types}")
        return v
    
    @field_validator('coordinates')
    @classmethod
    def validate_coordinates(cls, v, info: ValidationInfo):
        """좌표 데이터 검증"""
        if info.data.get('type') == 'bbox':
            required = {'x', 'y', 'width', 'height'}
            if not all(k in v for k in required):
                raise ValueError(f"Bounding box must contain {required}")
        elif info.data.get('type') == 'polygon':
            if 'points' not in v or not isinstance(v['points'], list):
                raise ValueError("Polygon must contain 'points' array")
        # 다른 타입에 대한 검증 로직 추가
        return v

class LabeledImageFile(MongoBaseModel):
    """레이블링된 이미지 파일 모델 (flat 구조)"""
    collection_name: ClassVar[str] = "labeled_image_files"
    
    dataset_id: str  # 항상 문자열로 강제
    filename: str
    path: str
    size: int
    width: Optional[int] = None
    height: Optional[int] = None
    mime_type: str
    hash: Optional[str] = None
    annotations: List[str] = Field(default_factory=list)
    source_image_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # category_id 등 카테고리 관련 필드 제거

    def to_mongo(self) -> Dict[str, Any]:
        data = super().to_mongo()
        # dataset_id를 항상 문자열로 저장
        if "dataset_id" in data and not isinstance(data["dataset_id"], str):
            data["dataset_id"] = str(data["dataset_id"])
        return data

class DatasetCreate(BaseModel):
    """데이터셋 생성 요청 모델"""
    name: str
    description: Optional[str] = None
    annotation_types: List[str] = Field(default_factory=lambda: ["bbox"])

class Dataset(MongoBaseModel):
    """이미지 레이블링 데이터셋 모델"""
    collection_name: ClassVar[str] = "labeled_image_datasets"
    
    name: str
    description: Optional[str] = None
    total_images: int = 0
    annotation_types: List[str] = Field(default_factory=lambda: ["bbox"])  # 지원하는 애노테이션 타입
    
    @field_validator('annotation_types')
    @classmethod
    def validate_annotation_types(cls, v):
        """애노테이션 타입 검증"""
        allowed_types = {'bbox', 'polygon', 'segmentation', 'keypoint', 'line', 'point'}
        for t in v:
            if t not in allowed_types:
                raise ValueError(f"Annotation type must be one of {allowed_types}")
        return v
