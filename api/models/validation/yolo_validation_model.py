from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ValidationHistory(BaseModel):
    id: str = Field(..., alias="_id")
    uid: str
    pid: str
    vid: str
    did: str
    dataset_name: str
    parameters: Dict[str, Any] = Field(..., description="Validation parameters used for the run")
    classes: List[str] = Field(..., description="List of classes used in the validation")
    status: str
    created_at: str = Field(..., description="Timestamp when the validation history was created")
    used_codebase: Optional[str] = None
    artifacts_path: Optional[str] = None
    error_details: Optional[str] = None
    metrics_summary: Optional[Dict[str, Any]] = Field(None, description="Validation metrics summary")


class YoloDetValidationParams(BaseModel):
    model: Optional[str] = Field(default="best.pt")
    imgsz: Optional[int] = Field(default=640)
    batch: Optional[int] = Field(default=32)
    device: Optional[str] = Field(default="cpu")
    workers: Optional[int] = Field(default=8)
    conf: Optional[float] = Field(default=0.001)
    iou: Optional[float] = Field(default=0.6)
    max_det: Optional[int] = Field(default=300)
    save_json: Optional[bool] = Field(default=True)
    save_txt: Optional[bool] = Field(default=True)
    save_conf: Optional[bool] = Field(default=True)
    plots: Optional[bool] = Field(default=True)
    verbose: Optional[bool] = Field(default=True)
    half: Optional[bool] = Field(default=False)
    dnn: Optional[bool] = Field(default=False)
    agnostic_nms: Optional[bool] = Field(default=False)
    augment: Optional[bool] = Field(default=False)
    rect: Optional[bool] = Field(default=False)


class YoloDetValidationRequest(BaseModel):
    pid: str
    tid: str
    cid: Optional[str] = None
    did: str
    task_type: str = Field(..., description="Task type for YOLO validation (detection/segmentation/pose/obb/classification)")
    parameters: Optional[YoloDetValidationParams]
    

class YoloDetValidationInfo(BaseModel):
    uid: str
    pid: str
    tid: Optional[str] = None
    task_type: str = Field(..., description="Task type for YOLO validation (detection/segmentation/pose/obb/classification)")
    parameters: Optional[YoloDetValidationParams]
    workdir: str
    did: str
    cid: Optional[str] = None