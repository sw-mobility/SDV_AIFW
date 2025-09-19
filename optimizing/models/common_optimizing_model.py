# api/models/optimizing/common_optimizing_model.py
from __future__ import annotations
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
from .optimizing_model import Status  # reuse your enum

class OptimizingHistory(BaseModel):
    # Mongo primary key
    id: str = Field(..., alias="_id")            # e.g. f"{uid}{pid}{oid}"
    uid: str
    pid: str
    oid: str                                     # O0001, O0002, ...

    # Lineage / inputs
    origin_oid: Optional[str] = None             # optimize-from previous OID (if any)
    input_model_id: Optional[str] = None         # reference into your own table, if you have one
    input_model_name: Optional[str] = None       # e.g. best.pt / yolov8n.pt

    # Execution
    status: Status                               # started | completed | failed
    started_at: Optional[str] = None             # ISO string KST or UTC
    completed_at: Optional[str] = None           # may be null on insert

    # What was run + params/metrics
    kind: Optional[str] = None                   # e.g. "pt_to_onnx", "onnx_to_trt_int8"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None     # size_before/after, latency, etc

    # Artifacts & errors
    artifacts_path: Optional[str] = None         # "artifacts/{pid}/optimizing/{oid}"
    error_details: Optional[str] = None

    # Allow using either "id" or "_id" when constructing
    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _ensure_id(self):
        # auto-derive _id if caller didn't set it
        if not self.id:
            object.__setattr__(self, "id", f"{self.uid}{self.pid}{self.oid}")
        return self
