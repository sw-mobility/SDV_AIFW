"""Labeled 이미지 카테고리 MongoDB 관리자"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

from core.mongodb import MongoDB
from models.dataset.labeled.images.mongodb import Category
from utils.logging import logger

class CategoryMongoManager:
    """레이블링된 이미지 카테고리 MongoDB 관리자"""
    def __init__(self):
        self.collection = MongoDB.db[Category.collection_name]
    
    async def create_category(self, category: Category) -> Category:
        """카테고리를 생성합니다."""
        # MongoDB 문서로 변환
        document = category.to_mongo()
        
        # 결과 삽입
        result = await self.collection.insert_one(document)
        
        # 삽입된 ID 설정
        category.id = str(result.inserted_id)
        
        return category
    
    async def get_category(self, category_id: str) -> Optional[Category]:
        """카테고리 ID로 카테고리를 조회합니다."""
        if not ObjectId.is_valid(category_id):
            return None
            
        document = await self.collection.find_one({"_id": ObjectId(category_id)})
        return Category.from_mongo(document)
    
    async def list_categories(
        self, dataset_id: str, parent_id: Optional[str] = None
    ) -> List[Category]:
        """데이터셋의 카테고리 목록을 조회합니다."""
        query = {"dataset_id": dataset_id}
        
        # 부모 ID가 지정된 경우 필터링
        if parent_id is not None:
            query["parent_id"] = parent_id
            
        cursor = self.collection.find(query).sort("path", 1)
        categories = []
        
        async for document in cursor:
            categories.append(Category.from_mongo(document))
            
        return categories
    
    async def get_categories_by_path(self, dataset_id: str, path: str) -> List[Category]:
        """경로로 카테고리를 조회합니다."""
        query = {
            "dataset_id": dataset_id,
            "path": {"$regex": f"^{path}(/|$)"}
        }
        
        cursor = self.collection.find(query)
        categories = []
        
        async for document in cursor:
            categories.append(Category.from_mongo(document))
            
        return categories
    
    async def delete_category(self, category_id: str) -> bool:
        """카테고리를 삭제합니다."""
        if not ObjectId.is_valid(category_id):
            return False
            
        result = await self.collection.delete_one({"_id": ObjectId(category_id)})
        return result.deleted_count > 0
    
    async def update_category(self, category_id: str, update_data: Dict[str, Any]) -> bool:
        """카테고리를 업데이트합니다."""
        if not ObjectId.is_valid(category_id):
            return False
            
        # 업데이트 데이터에 updated_at 추가
        update_data["updated_at"] = datetime.utcnow()
        
        result = await self.collection.update_one(
            {"_id": ObjectId(category_id)},
            {"$set": update_data}
        )
        
        return result.modified_count > 0
    
    async def increment_image_count(self, category_id: str) -> bool:
        """카테고리의 이미지 수를 증가시킵니다."""
        if not ObjectId.is_valid(category_id):
            return False
            
        result = await self.collection.update_one(
            {"_id": ObjectId(category_id)},
            {"$inc": {"image_count": 1}}
        )
        
        return result.modified_count > 0
    
    async def decrement_image_count(self, category_id: str) -> bool:
        """카테고리의 이미지 수를 감소시킵니다."""
        if not ObjectId.is_valid(category_id):
            return False
            
        result = await self.collection.update_one(
            {"_id": ObjectId(category_id)},
            {"$inc": {"image_count": -1}}
        )
        
        return result.modified_count > 0
