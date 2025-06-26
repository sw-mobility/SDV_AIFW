"""Labeled 이미지 애노테이션 MongoDB 관리자"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

from core.mongodb import MongoDB
from models.dataset.labeled.images.mongodb import Annotation
from utils.logging import logger

class AnnotationMongoManager:
    """레이블링된 이미지 애노테이션 MongoDB 관리자"""
    def __init__(self):
        self.collection = MongoDB.db[Annotation.collection_name]
    
    async def create_annotation(self, annotation: Annotation) -> Annotation:
        """애노테이션을 생성합니다."""
        # MongoDB 문서로 변환
        document = annotation.to_mongo()
        
        # 결과 삽입
        result = await self.collection.insert_one(document)
        
        # 삽입된 ID 설정
        annotation.id = str(result.inserted_id)
        
        return annotation
    
    async def get_annotation(self, annotation_id: str) -> Optional[Annotation]:
        """애노테이션 ID로 애노테이션을 조회합니다."""
        if not ObjectId.is_valid(annotation_id):
            return None
            
        document = await self.collection.find_one({"_id": ObjectId(annotation_id)})
        return Annotation.from_mongo(document)
    
    async def get_annotations_by_image(self, image_id: str) -> List[Annotation]:
        """이미지 ID로 모든 애노테이션을 조회합니다."""
        query = {"image_id": image_id}
            
        cursor = self.collection.find(query).sort("created_at", -1)
        annotations = []
        
        async for document in cursor:
            annotations.append(Annotation.from_mongo(document))
            
        return annotations
    
    async def get_annotations_by_label(self, label: str) -> List[Annotation]:
        """레이블로 모든 애노테이션을 조회합니다."""
        query = {"label": label}
            
        cursor = self.collection.find(query).sort("created_at", -1)
        annotations = []
        
        async for document in cursor:
            annotations.append(Annotation.from_mongo(document))
            
        return annotations
    
    async def delete_annotation(self, annotation_id: str) -> bool:
        """애노테이션을 삭제합니다."""
        if not ObjectId.is_valid(annotation_id):
            return False
            
        result = await self.collection.delete_one({"_id": ObjectId(annotation_id)})
        return result.deleted_count > 0
    
    async def update_annotation(self, annotation_id: str, update_data: Dict[str, Any]) -> bool:
        """애노테이션을 업데이트합니다."""
        if not ObjectId.is_valid(annotation_id):
            return False
            
        # 업데이트 데이터에 updated_at 추가
        update_data["updated_at"] = datetime.utcnow()
        
        result = await self.collection.update_one(
            {"_id": ObjectId(annotation_id)},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
