from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from .optimizing_model import Status
from pydantic import ConfigDict
from datetime import datetime

class OptimizingHistory(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    uid: str
    pid: str
    oid: str

    origin_oid: Optional[str] = None
    input_model_id: Optional[str] = None
    input_model_name: Optional[str] = None

    status: Status = Status.started
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    kind: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None

    artifacts_path: Optional[str] = None
    error_details: Optional[str] = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _ensure_id(self):
        if not self.id:
            self.id = f"{self.uid}{self.pid}{self.oid}"
        return self
