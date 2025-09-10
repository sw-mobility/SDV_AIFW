from fastapi import Form
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

#API 요청 모델
class LabeledDatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: str
    task_type: str
    label_format: str

class LabeledDatasetUpdate(BaseModel):
    did: str
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None
    task_type: Optional[str] = None
    label_format: Optional[str] = None

class LabeledDataUpload(BaseModel):
    did: str

    @classmethod
    def as_form(
        cls,
        did: str = Form(...),
    ):
        return cls(did=did)



#DB 저장 모델
class LabeledDatasetInfo(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    did: str
    name: str
    description: Optional[str] = None
    classes: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    type: str
    task_type: str
    label_format: str
    total: Optional[int] = 0  # 총 데이터 수
    origin_raw: Optional[str] = None  # 원본 Raw Dataset ID
    path: str
    created_at: str  # ISODate

class P_LabeledDatasetInfo(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    did: str
    pid: str
    name: str
    description: Optional[str] = None
    classes: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    type: str
    task_type: str
    label_format: str
    total: Optional[int] = 0  # 총 데이터 수
    origin_raw: Optional[str] = None  # 원본 Raw Dataset ID
    path: str
    created_at: str  # ISODate

class LabeledDataInfo(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    did: str
    dataset: str
    name: str
    type: str
    file_format: str
    origin_raw: Optional[str] = None  # 원본 Raw Dataset ID
    path: str  # MinIO 버킷 경로
    created_at: str  # ISODate