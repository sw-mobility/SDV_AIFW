from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field # type: ignore
from fastapi import Form # type: ignore

class YoloDetLabelingParams(BaseModel):
    model: str
    source: Optional[str] = None
    conf: Optional[float] = None  # batch_size → batch
    iou	: Optional[float] = None
    imgsz: Optional[int] = None
    rect: Optional[bool] = None  # save_period 필드 추가
    half: Optional[bool] = None
    device: Optional[str] = None
    batch: Optional[int] = None
    max_det	: Optional[int] = None
    vid_stride: Optional[int] = None
    stream_buffer: Optional[bool] = None
    visualize: Optional[bool] = None
    augment: Optional[bool] = None
    agnostic_nms: Optional[bool] = None
    classes	: Optional[List[int]] = None
    retina_masks: Optional[bool] = None
    embed: Optional[List[int]] = None
    project: Optional[str] = None
    name: Optional[str] = None
    stream: Optional[bool] = None
    verbose: Optional[bool] = None

    ########
    show: Optional[bool] = None
    save	: Optional[bool] = None
    save_frames: Optional[bool] = None
    save_txt: Optional[bool] = None
    save_conf: Optional[bool] = None
    save_crop: Optional[bool] = None
    show_labels: Optional[bool] = None
    show_conf: Optional[bool] = None
    show_boxes: Optional[bool] = None
    line_width: Optional[int] = None

class YoloDetLabelingRequest(BaseModel):
    pid: str
    did: str
    cid: Optional[str] = None
    name: str
    parameters: Optional[YoloDetLabelingParams]

class YoloDetLabelingInfo(BaseModel):
    uid: str
    pid: str
    did: str
    cid: Optional[str] = None
    name: str
    task_type: str
    parameters: Optional[YoloDetLabelingParams]
    workdir: str

# --------------------------------------------------------detection--------------------------------------------------------


class YoloLabelingResult(BaseModel):
    uid: str
    pid: str
    name: str
    status: str
    type: str
    task_type: str
    classes: List[str]  # 모델이 인식하는 클래스 목록
    parameters: Optional[Dict[str, Any]] = None  # 사용된 파라미터 정보
    started_time: str
    completed_time: str
    label_format: str = "COCO"
    workdir: str  # 작업 디렉토리 경로
    artifacts_path: Optional[str] = None  # 학습된 모델의 경로
    codebase_id: Optional[str] = None  # 사용된 코드베이스 ID
    error_details: Optional[str] = None  # 오류 발생 시 상세 정보
    raw_dataset_id: Optional[str] = None  # 오류 발생 시 상세 정보

class YoloHandlingRequest(BaseModel):
    workdir: str
    result: YoloLabelingResult