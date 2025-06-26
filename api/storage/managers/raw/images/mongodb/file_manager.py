"""이미지 파일 MongoDB 관리자"""
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from core.mongodb import MongoDB, NotInitializedError
from models.dataset.raw.images.mongodb import ImageFile
from utils.logging import logger

class ImageFileMongoManager:
    """이미지 파일 MongoDB 관리자"""
    
    @property
    def collection(self):
        """파일 컬렉션을 반환합니다."""
        try:
            return MongoDB.db.files
        except NotInitializedError as e:
            logger.error("Attempting to access MongoDB before initialization")
            raise

    async def create_file(self, dataset_name: str, filename: str, path: str, content_type: str, size: int, metadata: Optional[dict] = None) -> ImageFile:
        """
        새로운 파일 정보를 생성합니다 (flat 구조, category_path 제거)
        """
        try:
            # 파일이 이미 존재하는지 확인
            existing = await self.collection.find_one({
                "dataset_name": dataset_name,
                "path": path
            })
            if existing:
                raise ValueError(f"File with path '{path}' already exists in dataset '{dataset_name}'")
            file = ImageFile(
                filename=filename,
                path=path,
                dataset_name=dataset_name,
                content_type=content_type,
                size=size,
                metadata=metadata
            )
            doc = file.to_mongo()
            if '_id' in doc:
                del doc['_id']
            result = await self.collection.insert_one(doc)
            file.id = str(result.inserted_id)
            logger.info(f"File uploaded successfully: {path} in dataset {dataset_name}")
            logger.info(f"Created file in MongoDB: {path} in dataset {dataset_name}")
            return file
        except Exception as e:
            logger.error(f"Error creating file in MongoDB: {e}")
            raise

    async def get_file(self, dataset_name: str, file_identifier: str) -> Optional[ImageFile]:
        """파일 정보를 조회합니다. file_identifier는 ObjectId, 파일 경로, 또는 파일명이 될 수 있습니다."""
        if self.collection is None:
            raise RuntimeError("MongoDB is not initialized")

        # 여러 방법으로 파일 검색 시도
        query_options = []
        
        # 1. ObjectId로 검색 (가장 정확한 방법)
        if ObjectId.is_valid(file_identifier):
            query_options.append({"dataset_name": dataset_name, "_id": ObjectId(file_identifier)})
        
        # 2. 전체 경로로 검색
        query_options.append({"dataset_name": dataset_name, "path": file_identifier})
        
        # 3. 파일명으로 검색
        query_options.append({"dataset_name": dataset_name, "filename": file_identifier})

        for query in query_options:
            doc = await self.collection.find_one(query)
            if doc:
                return ImageFile.from_mongo(doc)

        return None

    async def list_files(self, dataset_name: str) -> List[ImageFile]:
        """
        데이터셋 내 모든 파일을 반환합니다 (flat 구조, category_path 제거)
        """
        try:
            cursor = self.collection.find({"dataset_name": dataset_name})
            files = []
            async for doc in cursor:
                files.append(ImageFile.from_mongo(doc))
            return files
        except Exception as e:
            logger.error(f"Error listing files in MongoDB: {e}")
            raise

    async def delete_file(self, dataset_name: str, file_identifier: str) -> bool:
        """파일 정보를 삭제합니다. file_identifier는 ObjectId, 파일 경로, 또는 파일명이 될 수 있습니다."""
        if self.collection is None:
            raise RuntimeError("MongoDB is not initialized")

        # 먼저 파일 정보를 찾아서 정확한 _id와 path를 얻습니다
        file_info = await self.get_file(dataset_name, file_identifier)
        if not file_info:
            return False

        # 파일 삭제 - ObjectId로 삭제
        result = await self.collection.delete_one({
            "_id": ObjectId(file_info.id)
        })
        
        if result.deleted_count == 0:
            return False
            
        # 간단하게 유지 - 카운트 업데이트는 나중에 구현
        logger.info(f"Deleted file from MongoDB: {file_info.path} in dataset {dataset_name}")
        return True
