from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class YoloDetValidationParams(BaseModel):
    model: Optional[str] = Field(default="best.pt")
    imgsz: Optional[int] = Field(default=640)
    batch: Optional[int] = Field(default=32)
    device: Optional[str] = Field(default="cpu")
    conf: Optional[float] = Field(default=0.001)
    iou: Optional[float] = Field(default=0.6)


class YoloDetValidationServiceRequest(BaseModel):
    uid: str
    pid: str
    tid: Optional[str] = None
    vid: Optional[str] = None
    task_type: str = Field(default="detection", description="Task type for YOLO validation (always detection)")
    cid: Optional[str] = None
    parameters: Optional[YoloDetValidationParams]
    did: str
    workdir: str


class YoloValidationResult(BaseModel):
    uid: str
    pid: str
    tid: Optional[str] = None
    vid: Optional[str] = None
    status: str
    task_type: str
    workdir: str
    started_time: Optional[str] = None
    completed_time: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    did: Optional[str] = None
    cid: Optional[str] = None
    result_path: Optional[str] = None
    plots_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None


class YoloValidationResponse(BaseModel):
    vid: str
    status: str
    message: str