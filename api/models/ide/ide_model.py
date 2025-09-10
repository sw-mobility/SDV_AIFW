from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import Form


# API 요청 모델
class CodebaseCreateRequest(BaseModel):
    name: str
    algorithm: str
    stage: str = Field(..., description="Stage of the training snapshot (e.g., 'Training', 'Validating', 'Optimizing')")
    task_type: str
    description: Optional[str] = None

class CodebaseUpdateRequest(BaseModel):
    cid: str
    name: Optional[str]
    algorithm: Optional[str]
    stage: Optional[str] = Field(..., description="Stage of the training snapshot (e.g., 'Training', 'Validating', 'Optimizing')")
    task_type: Optional[str]
    description: Optional[str] = None


# DB 저장 모델
class CodebaseInfo(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    cid: str
    name: str
    stage: str = Field(..., description="Stage of the training snapshot (e.g., 'Training', 'Validating', 'Optimizing')")
    algorithm: str
    task_type: str
    description: Optional[str] = None
    path: str
    last_modified: Optional[str] = Field(..., description="Timestamp when the codebase was last modified")
    created_at: str = Field(..., description="Timestamp when the training snapshot was created")