from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import Form

# API 요청 모델
class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None

class ProjectUpdate(BaseModel):
    pid: str
    name: Optional[str] = None
    description: Optional[str] = None


# DB 저장 모델
class ProjectInfo(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    pid: str  # 프로젝트 ID
    name: str
    description: Optional[str] = None
    status: str  # 프로젝트 상태 (예: 'active', 'archived')
    # last_trn_snap: Optional[str] = None  # 마지막 트레이닝 스냅샷 ID
    # last_opt_snap: Optional[str] = None  # 마지막 최적화 스냅샷 ID
    # last_val_snap: Optional[str] = None  # 마지막 검증 스냅샷 ID
    created_at: str  # ISODate