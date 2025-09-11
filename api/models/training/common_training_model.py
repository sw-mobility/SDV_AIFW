from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import Form

class TrainingHistory(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    pid: str
    tid: str
    origin_tid: Optional[str] = None
    origin_did: str
    origin_dataset_name: str
    processed_did: str
    processed_dataset_name: str
    parameters: Dict[str, Any] = Field(..., description="Training parameters used for the training run")
    classes: List[str] = Field(..., description="List of classes used in the training")
    status: str
    started_at: str = Field(..., description="Timestamp when the training was started")
    completed_at: str = Field(..., description="Timestamp when the training was completed")
    used_codebase: Optional[str] = None  # 사용된 코드베이스 ID
    artifacts_path: Optional[str] = None  # 학습된 모델의 경로
    error_details: Optional[str] = None  # 오류 발생 시 상세 정보