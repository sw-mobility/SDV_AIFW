from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class BaseCategory(BaseModel):
    """기본 카테고리 모델"""
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    image_count: int = 0

class BaseDataset(BaseModel):
    """기본 데이터셋 모델"""
    name: str
    description: Optional[str] = None
    categories: List[BaseCategory] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    total_images: int = 0
