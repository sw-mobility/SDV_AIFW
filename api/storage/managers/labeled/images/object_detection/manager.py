"""Object Detection 이미지 데이터셋 스토리지 관리자 (COCO/YOLO flat 구조, no model/version dir)"""
from typing import List, Optional, Dict, Any
from datetime import datetime

from core.storage import storage_client
from storage.base import BaseStorageManager, DatasetOperationError
from utils.logging import logger

class ObjectDetectionStorageManager(BaseStorageManager):
    """Object Detection 이미지 데이터셋 저장소 관리자 (flat 구조, no model/version dir)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True

    @property
    def storage(self):
        return storage_client

    def _get_dataset_path(self, name: str) -> str:
        return f"datasets/labeled/images/object_detection/{name}"

    def _get_file_path(self, dataset_name: str, filename: str, subfolder: str) -> str:
        return f"{self._get_dataset_path(dataset_name)}/{subfolder}/{filename}"
