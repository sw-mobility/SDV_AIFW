from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class TrainingProject(BaseModel):
    name: str = Field(..., description="프로젝트 이름")
    description: Optional[str] = Field("", description="프로젝트 설명")
    created_at: Optional[str] = Field(None, description="생성일시")
    algorithm: str = Field(..., description="사용할 알고리즘 (예: YOLO, Faster R-CNN 등)")
    model_version: str = Field(..., description="모델 버전 (예: v8, v7 등)")
    model_size: str = Field(..., description="모델 사이즈 (예: n, s, m, l, x)")
    task_type: str = Field(..., description="태스크 타입 (예: object_detection, classification 등)")
    dataset_path: str = Field(None, description="데이터셋 경로")
    split_ratio: Optional[float] = Field(0.8, description="train/val split 비율 (예: 0.8이면 80% train, 20% val)")
    # YOLO 학습 관련 파라미터 (필수 제외 모두 optional)
    epochs: Optional[int] = Field(None, description="학습 epoch 수")
    batch: Optional[int] = Field(None, description="배치 크기")
    imgsz: Optional[int] = Field(None, description="입력 이미지 크기")
    lr0: Optional[float] = Field(None, description="초기 learning rate")
    lrf: Optional[float] = Field(None, description="최종 learning rate fraction")
    momentum: Optional[float] = Field(None, description="SGD momentum")
    weight_decay: Optional[float] = Field(None, description="가중치 감쇠")
    warmup_epochs: Optional[float] = Field(None, description="워밍업 epoch 수")
    warmup_momentum: Optional[float] = Field(None, description="워밍업 momentum")
    warmup_bias_lr: Optional[float] = Field(None, description="워밍업 bias learning rate")
    box: Optional[float] = Field(None, description="box loss gain")
    cls: Optional[float] = Field(None, description="cls loss gain")
    dfl: Optional[float] = Field(None, description="dfl loss gain")
    fl_gamma: Optional[float] = Field(None, description="focal loss gamma")
    label_smoothing: Optional[float] = Field(None, description="라벨 스무딩")
    nbs: Optional[int] = Field(None, description="nominal batch size")
    optimizer: Optional[str] = Field(None, description="최적화 알고리즘")
    dropout: Optional[float] = Field(None, description="드롭아웃 비율")
    # 기타 YOLO 파라미터는 필요시 추가
    task_configs: Optional[List[dict]] = None  # <-- Make this optional
