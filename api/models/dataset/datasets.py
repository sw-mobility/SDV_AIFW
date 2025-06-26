"""데이터셋 통합 모듈"""
from typing import Dict, Any, List, Union, Optional
from models.dataset.raw.images.mongodb import Dataset as RawImageDataset
from models.dataset.labeled.images.mongodb import Dataset as LabeledImageDataset

class DatasetRegistry:
    """데이터셋 레지스트리 - 시스템 내 모든 데이터셋 관리"""
    
    @staticmethod
    async def list_all_datasets() -> Dict[str, List[Dict[str, Any]]]:
        """시스템의 모든 데이터셋을 유형별로 목록화"""
        from storage.managers.raw.images.manager import RawImageStorageManager
        from storage.managers.labeled.images.yolo.v8.det.manager import YOLOv8DetectionStorageManager
        
        raw_manager = RawImageStorageManager()
        labeled_manager = YOLOv8DetectionStorageManager()
        
        # 각 데이터셋 매니저에서 데이터셋 목록 가져오기
        raw_datasets = await raw_manager.list_datasets()
        labeled_datasets = await labeled_manager.list_datasets()
        
        # 결과 구성
        return {
            "raw_images": [ds.model_dump() for ds in raw_datasets],
            "labeled_images": [ds.model_dump() for ds in labeled_datasets]
        }
    
    @staticmethod
    async def get_dataset_stats() -> Dict[str, Any]:
        """시스템 데이터셋 통계"""
        from storage.managers.raw.images.manager import RawImageStorageManager
        from storage.managers.labeled.images.yolo.v8.det.manager import YOLOv8DetectionStorageManager
        
        raw_manager = RawImageStorageManager()
        labeled_manager = YOLOv8DetectionStorageManager()
        
        raw_datasets = await raw_manager.list_datasets()
        labeled_datasets = await labeled_manager.list_datasets()
        
        return {
            "total_datasets": len(raw_datasets) + len(labeled_datasets),
            "raw_image_datasets": len(raw_datasets),
            "labeled_image_datasets": len(labeled_datasets),
            "total_raw_images": sum(ds.total_images for ds in raw_datasets),
            "total_labeled_images": sum(ds.total_images for ds in labeled_datasets)
        }
    @staticmethod
    async def find_dataset(name_or_id: str) -> Optional[Dict[str, Any]]:
        """이름 또는 ID로 데이터셋 검색"""
        from storage.managers.raw.images.manager import RawImageStorageManager
        from storage.managers.labeled.images.yolo.v8.det.manager import YOLOv8DetectionStorageManager
        
        # 각 매니저에서 데이터셋 검색
        raw_manager = RawImageStorageManager()
        labeled_manager = YOLOv8DetectionStorageManager()
        
        # 원본 이미지 데이터셋 확인
        raw_dataset = await raw_manager.get_dataset(name_or_id)
        if raw_dataset:
            return {
                "type": "raw_image",
                "dataset": raw_dataset.model_dump()
            }
        
        # 레이블링 이미지 데이터셋 확인
        labeled_dataset = await labeled_manager.get_dataset(name_or_id)
        if labeled_dataset:
            return {
                "type": "labeled_image",
                "dataset": labeled_dataset.model_dump()
            }
        
        return None
