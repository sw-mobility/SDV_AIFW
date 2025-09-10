from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import Form

class YoloDetTrainingParams(BaseModel):
    model: Optional[str] = None
    split_ratio: Optional[List[float]] = Field(default_factory=lambda: [0.8, 0.2, 0.0])
    epochs: Optional[int] = None
    batch: Optional[int] = None
    imgsz: Optional[int] = None
    device: Optional[str] = None
    save_period: Optional[int] = None
    workers: Optional[int] = None
    pretrained: Optional[bool] = None
    optimizer: Optional[str] = None
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    momentum: Optional[float] = None
    weight_decay: Optional[float] = None
    patience: Optional[int] = None
    augment: Optional[bool] = None
    warmup_epochs: Optional[int] = None
    warmup_momentum: Optional[float] = None
    warmup_bias_lr: Optional[float] = None
    seed: Optional[int] = None
    cache: Optional[bool] = None
    dropout: Optional[float] = None
    label_smoothing: Optional[float] = None
    rect: Optional[bool] = None
    pretrained: Optional[bool] = None
    resume: Optional[bool] = None
    amp: Optional[bool] = None
    single_cls: Optional[bool] = None
    cos_lr: Optional[bool] = None
    close_mosaic: Optional[int] = None
    overlap_mask: Optional[bool] = None
    mask_ratio: Optional[float] = None

class YoloDetTrainingRequest(BaseModel):
    pid: str
    cid: Optional[str] = None
    did: str
    origin_tid: Optional[str] = None
    user_classes: List[str]
    parameters: YoloDetTrainingParams
    # task_type: str = Field(..., description="Task type for YOLO training (e.g., object_detection, segmentation, pose, obb, classification)")

# --------------------------------------------------------detection--------------------------------------------------------

class YoloTrainingInfo(BaseModel):
    uid: str
    pid: str
    origin_tid: Optional[str] = None
    tid: Optional[str] = None
    task_type: str = Field(..., description="Task type for YOLO training (e.g., object_detection, segmentation, pose, obb, classification)")
    parameters: dict
    user_classes: List[str]
    model_classes: List[str]
    dataset_classes: List[str]
    workdir: str
    did: str
    cid: Optional[str] = None 

class YoloTrainingResult(BaseModel):
    uid: str
    pid: str
    origin_tid: Optional[str] = None
    status: str
    task_type: str
    classes: List[str]
    workdir: str  # 작업 디렉토리 경로
    started_time: str
    completed_time: str
    parameters: Optional[Dict[str, Any]] = None  # 사용된 파라미터 정보
    did: Optional[str] = None  # 사용된 데이터셋 did
    cid: Optional[str] = None  # 사용된 코드베이스 ID
    artifacts_path: Optional[str] = None  # 학습된 모델의 경로
    error_details: Optional[str] = None  # 오류 발생 시 상세 정보

class YoloHandlingRequest(BaseModel):
    workdir: str
    result: YoloTrainingResult


class YoloTrainingResponse(BaseModel):
    tid: str
    status: str
    message: str