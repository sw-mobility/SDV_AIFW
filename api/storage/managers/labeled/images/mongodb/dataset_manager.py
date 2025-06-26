"""Labeled 이미지 데이터셋 MongoDB 관리자 (COCO/YOLO flat 구조, no category)"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

from core.mongodb import MongoDB
from models.dataset.labeled.images.mongodb import Dataset
from utils.logging import logger

class DatasetMongoManager:
    """레이블링된 이미지 데이터셋 MongoDB 관리자 (flat 구조)"""
    def __init__(self):
        self.collection = MongoDB.db[Dataset.collection_name]
    
    async def create_dataset(self, dataset: Dataset) -> Dataset:
        """데이터셋을 생성합니다."""
        # MongoDB 문서로 변환
        document = dataset.to_mongo()
        
        # 결과 삽입
        result = await self.collection.insert_one(document)
        
        # 삽입된 ID 설정
        dataset.id = str(result.inserted_id)
        
        return dataset
    
    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """데이터셋 ID 또는 이름으로 데이터셋을 조회합니다."""
        # ID로 조회 시도
        try:
            if ObjectId.is_valid(dataset_id):
                document = await self.collection.find_one({"_id": ObjectId(dataset_id)})
                if document:
                    return Dataset.from_mongo(document)
        except Exception as e:
            logger.error(f"Error retrieving dataset by ID: {str(e)}")
        
        # 이름으로 조회 시도
        document = await self.collection.find_one({"name": dataset_id})
        return Dataset.from_mongo(document)
    
    async def list_datasets(self) -> List[Dataset]:
        """모든 데이터셋 목록을 조회합니다."""
        cursor = self.collection.find()
        datasets = []
        
        async for document in cursor:
            datasets.append(Dataset.from_mongo(document))
            
        return datasets
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """데이터셋을 삭제합니다."""
        # ID 또는 이름으로 삭제
        if ObjectId.is_valid(dataset_id):
            result = await self.collection.delete_one({"_id": ObjectId(dataset_id)})
        else:
            result = await self.collection.delete_one({"name": dataset_id})
            
        return result.deleted_count > 0
    
    async def update_dataset(self, dataset_id: str, update_data: Dict[str, Any]) -> bool:
        """데이터셋을 업데이트합니다."""
        # 업데이트 데이터에 updated_at 추가
        update_data["updated_at"] = datetime.utcnow()
        
        # ID로 업데이트
        if ObjectId.is_valid(dataset_id):
            result = await self.collection.update_one(
                {"_id": ObjectId(dataset_id)},
                {"$set": update_data}
            )
        else:
            # 이름으로 업데이트
            result = await self.collection.update_one(
                {"name": dataset_id},
                {"$set": update_data}
            )
            
        return result.modified_count > 0
