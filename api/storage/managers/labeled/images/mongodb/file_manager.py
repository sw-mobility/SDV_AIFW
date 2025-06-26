"""Labeled 이미지 파일 MongoDB 관리자 (COCO/YOLO flat 구조, no category)"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

from core.mongodb import MongoDB
from models.dataset.labeled.images.mongodb import LabeledImageFile
from utils.logging import logger

class ImageFileMongoManager:
    """레이블링된 이미지 파일 MongoDB 관리자 (flat 구조)"""
    def __init__(self):
        self.collection = MongoDB.db[LabeledImageFile.collection_name]
    
    async def create_file(self, file: LabeledImageFile) -> LabeledImageFile:
        """파일을 생성합니다."""
        # MongoDB 문서로 변환
        document = file.to_mongo()
        
        # 결과 삽입
        result = await self.collection.insert_one(document)
        
        # 삽입된 ID 설정
        file.id = str(result.inserted_id)
        
        return file
    
    async def get_file(self, file_id: str) -> Optional[LabeledImageFile]:
        """파일 ID로 파일을 조회합니다."""
        if not ObjectId.is_valid(file_id):
            return None
            
        document = await self.collection.find_one({"_id": ObjectId(file_id)})
        return LabeledImageFile.from_mongo(document)
    
    async def list_files(
        self, 
        dataset_id: str, 
        skip: int = 0,
        limit: int = 100
    ) -> List[LabeledImageFile]:
        """데이터셋의 파일 목록을 조회합니다."""
        query = {"dataset_id": dataset_id}
        
        cursor = self.collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        files = []
        
        async for document in cursor:
            files.append(LabeledImageFile.from_mongo(document))
            
        return files
    
    async def delete_file(self, file_id: str) -> bool:
        """파일을 삭제합니다."""
        if not ObjectId.is_valid(file_id):
            return False
            
        result = await self.collection.delete_one({"_id": ObjectId(file_id)})
        return result.deleted_count > 0
    
    async def update_file(self, file_id: str, update_data: Dict[str, Any]) -> bool:
        """파일을 업데이트합니다."""
        if not ObjectId.is_valid(file_id):
            return False
            
        # 업데이트 데이터에 updated_at 추가
        update_data["updated_at"] = datetime.utcnow()
        
        result = await self.collection.update_one(
            {"_id": ObjectId(file_id)},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    async def update_file_annotations(self, file_id: str, annotation_ids: List[str]) -> bool:
        """파일의 애노테이션 목록을 업데이트합니다."""
        if not ObjectId.is_valid(file_id):
            return False
            
        result = await self.collection.update_one(
            {"_id": ObjectId(file_id)},
            {
                "$set": {
                    "annotations": annotation_ids,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
    
    async def add_annotation_to_file(self, file_id: str, annotation_id: str) -> bool:
        """파일에 애노테이션을 추가합니다."""
        if not ObjectId.is_valid(file_id) or not ObjectId.is_valid(annotation_id):
            return False
            
        result = await self.collection.update_one(
            {"_id": ObjectId(file_id)},
            {
                "$addToSet": {"annotations": annotation_id},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0
    
    async def remove_annotation_from_file(self, file_id: str, annotation_id: str) -> bool:
        """파일에서 애노테이션을 제거합니다."""
        if not ObjectId.is_valid(file_id) or not ObjectId.is_valid(annotation_id):
            return False
            
        result = await self.collection.update_one(
            {"_id": ObjectId(file_id)},
            {
                "$pull": {"annotations": annotation_id},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0
