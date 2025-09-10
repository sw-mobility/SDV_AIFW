from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import Form

class LabelingHistory(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    pid: str
    tid: str
    origin_tid: Optional[str] = None
    dataset_id: str
    dataset_name: str
    parameters: Dict[str, Any] = Field(..., description="Labeling parameters used for the training run")
    classes: List[str] = Field(..., description="List of classes used in the training")
    status: str
    created_at: str = Field(..., description="Timestamp when the training history was created")
    used_codebase: Optional[str] = None  # 사용된 코드베이스 ID
    artifacts_path: Optional[str] = None  # 학습된 모델의 경로
    error_details: Optional[str] = None  # 오류 발생 시 상세 정보


class LabelingSnapshot(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    cid: str
    name: str
    algorithm: str
    task_type: str
    description: Optional[str] = None
    path: str
    created_at: str = Field(..., description="Timestamp when the training snapshot was created")