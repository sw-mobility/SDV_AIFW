from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import Form

# API 요청 모델
class RawDatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    type: str

class RawDatasetUpdate(BaseModel):
    did: str
    name: Optional[str] = None
    description: Optional[str] = None
    type: Optional[str] = None

class RawDataUpload(BaseModel):
    did: str

    @classmethod
    def as_form(
        cls,
        did: str = Form(...)
    ):
        return cls(did=did)


# DB 저장 모델
class RawDatasetInfo(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    did: str
    name: str
    description: Optional[str] = None
    type: str
    total: Optional[int] = 0  # 총 데이터 수
    path: str
    created_at: str  # ISODate


class RawDataInfo(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    did: str 
    dataset: str
    name: str
    type: str
    file_format: str
    path: str
    created_at: str  # ISODate