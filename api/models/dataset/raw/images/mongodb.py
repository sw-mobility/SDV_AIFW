"""MongoDB 데이터 모델"""
from typing import Optional, List, Any, Dict, ClassVar, Union, Type, TypeVar, Annotated
from datetime import datetime
from pydantic import (
    BaseModel, Field, ConfigDict, field_validator, 
    model_validator, computed_field, TypeAdapter, ValidationInfo
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from bson import ObjectId, errors as bson_errors

T = TypeVar('T', bound='MongoBaseModel')

class MongoBaseModel(BaseModel):
    """MongoDB 기본 모델"""
    model_config = ConfigDict(
        populate_by_name=True,  # alias를 통한 필드 접근 허용
        arbitrary_types_allowed=True,  # MongoDB ObjectId 같은 임의 타입 허용
        json_encoders={
            ObjectId: str,  # ObjectId를 문자열로 직렬화
            datetime: lambda dt: dt.isoformat()  # datetime을 ISO 형식 문자열로 직렬화
        },
        validate_assignment=True,  # 값 할당 시에도 유효성 검증
        extra="ignore",  # 추가 필드 무시
        ser_json_bytes="utf8",  # JSON 직렬화 인코딩
        ser_json_timedelta="iso8601",  # 시간 간격 직렬화 형식
    )

    # 컬렉션 이름 (상속 클래스에서 재정의)
    collection_name: ClassVar[str] = ""
    
    # PyObjectId 대신 일반 문자열 사용
    id: Optional[str] = Field(default=None, alias="_id", description="MongoDB 문서 ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="생성 시간")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="수정 시간")
    
    @model_validator(mode="before")
    @classmethod
    def update_timestamps(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """업데이트 시간 갱신"""
        if isinstance(data, dict) and "_id" in data:
            data["updated_at"] = datetime.utcnow()
        return data
        
    @classmethod
    def from_mongo(cls: Type[T], data: Optional[Dict[str, Any]]) -> Optional[T]:
        """MongoDB 문서를 모델로 변환합니다."""
        if not data:
            return None
            
        # 입력 데이터의 복사본을 만들어 원본 데이터를 변경하지 않도록 함
        data = data.copy()
            
        # MongoDB ObjectId를 문자열로 변환하여 모델에 적용
        if "_id" in data:
            if isinstance(data["_id"], ObjectId):
                data["_id"] = str(data["_id"])
            
        # 중첩된 ObjectId 변환
        data = cls._convert_nested_object_ids(data)
        
        try:
            return cls.model_validate(data)
        except Exception as e:
            import logging
            logging.error(f"Failed to convert MongoDB document to {cls.__name__}: {str(e)}")
            logging.error(f"Data: {data}")
            # 디버깅을 위해 구체적인 오류 메시지 포함
            raise ValueError(f"Failed to convert MongoDB document to {cls.__name__}: {str(e)}, Data: {data}")
    
    @staticmethod
    def _convert_nested_object_ids(data: Dict[str, Any]) -> Dict[str, Any]:
        """중첩된 ObjectId를 문자열로 변환"""
        result = {}
        for key, value in data.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, dict):
                result[key] = MongoBaseModel._convert_nested_object_ids(value)
            elif isinstance(value, list):
                result[key] = [
                    MongoBaseModel._convert_nested_object_ids(item) if isinstance(item, dict)
                    else (str(item) if isinstance(item, ObjectId) else item)
                    for item in value
                ]
            else:                result[key] = value
        return result
    
    def to_mongo(self) -> Dict[str, Any]:
        """Pydantic 모델을 MongoDB 문서로 변환합니다."""
        # 모델을 딕셔너리로 변환
        data = self.model_dump(by_alias=True)
        
        # 문자열 ID를 ObjectId로 변환하거나 null인 경우 제거
        if "_id" in data:
            if data["_id"] is None:
                # _id가 null이면 MongoDB가 생성하도록 필드를 제거
                del data["_id"]
            elif isinstance(data["_id"], str):
                try:
                    data["_id"] = ObjectId(data["_id"])
                except bson_errors.InvalidId:
                    # 유효한 ObjectId 형식이 아니면 삭제 (새 문서 생성시)
                    del data["_id"]
            
        # 모든 None 값을 제거 (MongoDB에 불필요한 null 필드 방지)
        return {k: v for k, v in data.items() if v is not None}

# Request Models
class DatasetCreate(BaseModel):
    """데이터셋 생성 요청 모델"""
    name: str = Field(..., description="데이터셋 이름")
    description: Optional[str] = Field(None, description="데이터셋 설명")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "자율주행",
                "description": "자율주행 이미지 데이터셋"
            }
        }
    )

# Database Models
class Dataset(MongoBaseModel):
    """데이터셋 정보 (category_count 제거)"""
    collection_name: ClassVar[str] = "datasets"
    
    name: str = Field(..., index=True, description="데이터셋 이름")
    description: Optional[str] = Field(None, description="데이터셋 설명")
    type: str = Field("raw_images", description="데이터셋 타입")
    file_count: int = Field(0, ge=0, description="파일 수")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "5f7a9c5d2d9f4c8f7e2d9f4c",
                "name": "자율주행",
                "description": "자율주행 이미지 데이터셋",
                "type": "raw_images",
                "file_count": 120,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-02T00:00:00"
            }
        }
    )
    
    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str, info: ValidationInfo) -> str:
        """데이터셋 이름 유효성 검증"""
        if not v or not v.strip():
            raise ValueError("Dataset name cannot be empty")
        return v.strip()
    
    @model_validator(mode="after")
    def validate_counts(self) -> "Dataset":
        """파일 수 유효성 검증"""
        if self.file_count < 0:
            raise ValueError("File count cannot be negative")
        return self

class ImageFile(MongoBaseModel):
    """이미지 파일 정보 (flat 구조, category_path 제거)"""
    collection_name: ClassVar[str] = "files"
    
    filename: str = Field(..., description="파일 이름")
    path: str = Field(..., index=True, description="파일 전체 경로")
    dataset_name: str = Field(..., index=True, description="소속 데이터셋 이름")
    content_type: str = Field(..., description="콘텐츠 타입(MIME)")
    size: int = Field(..., gt=0, description="파일 크기(바이트)")    
    metadata: Optional[Dict[str, Any]] = Field(None, description="메타데이터")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "5f7a9c5d2d9f4c8f7e2d9f4c",
                "filename": "car001.jpg",
                "path": "datasets/raw/images/자율주행/car001.jpg",
                "dataset_name": "자율주행",
                "content_type": "image/jpeg",
                "size": 1024000,
                "created_at": "2023-01-01T00:00:00",
                "updated_at": "2023-01-01T00:00:00"
            }
        }
    )
    
    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v: str) -> str:
        """파일명 유효성 검증"""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        # 파일명에 경로 구분자가 포함되면 안됨
        if '/' in v or '\\' in v:
            raise ValueError("Filename cannot contain path separators")
        return v.strip()
    
    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        """콘텐츠 타입 유효성 검증"""
        if not v or not v.strip():
            raise ValueError("Content type cannot be empty")
        # 이미지 타입 검증
        if not v.startswith("image/"):
            raise ValueError(f"Invalid image content type: {v}")
        return v.strip()
    
    @property
    def file_extension(self) -> str:
        """파일 확장자"""
        if "." in self.filename:
            return self.filename.split(".")[-1].lower()
        return ""

class DatasetFile(BaseModel):
    """데이터셋 파일 정보 (이미지/레이블 구분)"""
    filename: str
    dataset_name: str
    content_type: str
    size: int
    path: str
    file_type: str  # 'image' 또는 'label'

# MongoDB 및 Pydantic 통합 유틸리티

class MongoQueryBuilder:
    """MongoDB 쿼리 생성 도우미"""
    
    @staticmethod
    def id_query(id_value: Union[str, ObjectId]) -> Dict[str, ObjectId]:
        """ID로 검색하는 쿼리 생성"""
        if isinstance(id_value, str):
            try:
                return {"_id": ObjectId(id_value)}
            except bson_errors.InvalidId:
                raise ValueError(f"Invalid ObjectId format: {id_value}")
        elif isinstance(id_value, ObjectId):
            return {"_id": id_value}
        raise TypeError(f"id_value must be str or ObjectId, got {type(id_value)}")
    
    @staticmethod
    def pagination_params(page: int = 1, limit: int = 50) -> Dict[str, int]:
        """페이지네이션 파라미터 생성"""
        if page < 1:
            page = 1
        if limit < 1:
            limit = 50
        elif limit > 100:
            limit = 100
        
        skip = (page - 1) * limit
        return {"skip": skip, "limit": limit}


async def parse_mongo_results(cursor, model_cls: Type[T]) -> List[T]:
    """MongoDB 쿼리 결과를 Pydantic 모델로 변환"""
    results = []
    async for doc in cursor:
        try:
            model_instance = model_cls.from_mongo(doc)
            if model_instance:
                results.append(model_instance)
        except Exception as e:
            # 오류가 있는 문서는 건너뜁니다
            import logging
            logging.error(f"Failed to convert document to {model_cls.__name__}: {str(e)}")
    return results


class MongoAggregation:
    """MongoDB 집계 파이프라인 도우미"""
    
    @staticmethod
    def lookup_pipeline(from_collection: str, local_field: str, 
                       foreign_field: str, as_field: str) -> Dict:
        """$lookup 스테이지 생성"""
        return {
            "$lookup": {
                "from": from_collection,
                "localField": local_field,
                "foreignField": foreign_field,
                "as": as_field
            }
        }
    
    @staticmethod
    def match_pipeline(query: Dict) -> Dict:
        """$match 스테이지 생성"""
        return {"$match": query}
    
    @staticmethod
    def project_pipeline(fields: Dict) -> Dict:
        """$project 스테이지 생성"""
        return {"$project": fields}

# MongoDB 유틸리티 함수들
def fix_object_id(obj_id) -> str:
    """ObjectId를 문자열로 변환"""
    if obj_id is None:
        return None
    return str(obj_id)

def convert_mongo_document(data: Dict[str, Any]) -> Dict[str, Any]:
    """MongoDB 문서의 ObjectId를 문자열로 변환"""
    if not data:
        return {}
        
    result = {}
    for key, value in data.items():
        if key == "_id" and value is None:
            # None 값을 그대로 유지 - 새 문서 생성 시 필요
            result[key] = None
        elif key == "_id" and (isinstance(value, ObjectId) or hasattr(value, "__str__")):
            result[key] = str(value)
        elif isinstance(value, ObjectId):
            result[key] = str(value)
        elif isinstance(value, dict):
            result[key] = convert_mongo_document(value)
        elif isinstance(value, list):
            result[key] = [
                convert_mongo_document(item) if isinstance(item, dict)
                else (str(item) if isinstance(item, ObjectId) else item)
                for item in value
            ]
        else:
            result[key] = value
    return result

def safe_mongo_convert(cls: Type[T], data: Dict[str, Any]) -> Optional[T]:
    """MongoDB 문서를 안전하게 변환"""
    if not data:
        return None
    
    # 데이터 복사 및 ObjectId 변환
    converted_data = convert_mongo_document(data)
    
    # _id가 있지만 None이면 제거
    if "_id" in converted_data and converted_data["_id"] is None:
        del converted_data["_id"]
    
    try:
        # 모델 유효성 검증
        return cls.model_validate(converted_data)
    except Exception as e:
        # 오류 로깅
        import logging
        logging.error(f"Failed to convert MongoDB document to {cls.__name__}: {str(e)}")
        logging.error(f"Data: {converted_data}")
        raise ValueError(f"문서 변환 실패: {str(e)}")
