"""레이블링된 이미지 데이터셋 스토리지 통합 관리자 (COCO/YOLO flat 구조, no category)"""
import os
import uuid
from typing import BinaryIO, Optional, List, Tuple, Any, Dict
from datetime import datetime
from fastapi import UploadFile

from core.storage import storage_client
from storage.base import (
    BaseStorageManager, StorageError, FileUploadError as BaseFileUploadError, 
    FileDownloadError, DatasetOperationError
)
from storage.managers.labeled.images.mongodb.dataset_manager import DatasetMongoManager
from storage.managers.labeled.images.mongodb.file_manager import ImageFileMongoManager
from storage.managers.labeled.images.mongodb.annotation_manager import AnnotationMongoManager
from models.dataset.labeled.images.api import FileUploadResponse, FileUploadError
from models.dataset.labeled.images.mongodb import Dataset, DatasetCreate, LabeledImageFile, Annotation
from config.settings import SUPPORTED_IMAGE_EXTENSIONS
from utils.logging import logger
from urllib.parse import unquote

class LabeledImageStorageManager(BaseStorageManager[Dataset, LabeledImageFile, DatasetCreate]):
    """레이블링된 이미지 데이터셋 저장소 통합 관리자 (flat 구조)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize all managers
            cls._instance.dataset_manager = DatasetMongoManager()
            cls._instance.file_manager = ImageFileMongoManager()
            cls._instance.annotation_manager = AnnotationMongoManager()
        return cls._instance

    @property
    def storage(self):
        return storage_client

    def _get_dataset_path(self, name: str) -> str:
        """데이터셋 경로를 생성합니다."""
        return f"datasets/labeled/images/{name}"

    def _get_file_path(self, dataset_name: str, filename: str) -> str:
        """파일 경로를 생성합니다."""
        return f"{self._get_dataset_path(dataset_name)}/{filename}"

    async def create_dataset(self, dataset_create: DatasetCreate) -> Dataset:
        """새로운 레이블링 이미지 데이터셋을 생성합니다."""
        result = None
        try:
            # 요청으로부터 데이터셋 모델 생성
            dataset = Dataset(
                name=dataset_create.name,
                description=dataset_create.description,
                annotation_types=dataset_create.annotation_types,
                # id 필드를 명시적으로 None으로 설정
                id=None
            )
            
            # MongoDB에 데이터셋 생성
            result = await self.dataset_manager.create_dataset(dataset)
            
            # MinIO에 데이터셋 디렉토리 및 메타데이터 생성
            await self.storage.put_object(
                key=f"{self._get_dataset_path(dataset.name)}/.keep",
                body=b""
            )
            
            logger.info(f"Created labeled dataset: {dataset.name}")
            return result
        except Exception as e:
            # 실패 시 롤백
            try:
                if result:
                    await self.dataset_manager.delete_dataset(dataset.name)
            except:
                pass
            raise DatasetOperationError(f"Failed to create dataset: {str(e)}")

    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """데이터셋을 조회합니다."""
        try:
            return await self.dataset_manager.get_dataset(dataset_id)
        except Exception as e:
            logger.error(f"Failed to retrieve dataset {dataset_id}: {str(e)}")
            raise

    async def list_datasets(self) -> List[Dataset]:
        """모든 데이터셋 목록을 조회합니다."""
        try:
            return await self.dataset_manager.list_datasets()
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            raise

    async def delete_dataset(self, dataset_id: str) -> bool:
        """데이터셋을 삭제합니다."""
        try:
            dataset = await self.dataset_manager.get_dataset(dataset_id)
            if not dataset:
                logger.warning(f"Dataset not found for deletion: {dataset_id}")
                return False
                
            # MinIO에서 데이터셋 디렉토리 삭제
            try:
                objects = await self.storage.list_objects(
                    prefix=self._get_dataset_path(dataset.name)
                )
                for obj in objects:
                    await self.storage.remove_object(obj.object_name)
            except Exception as e:
                logger.error(f"Failed to delete objects from storage: {str(e)}")
                
            # MongoDB에서 데이터셋 삭제
            return await self.dataset_manager.delete_dataset(dataset_id)
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_id}: {str(e)}")
            raise

    async def get_file(self, file_id: str) -> Optional[LabeledImageFile]:
        """파일 정보를 조회합니다."""
        try:
            return await self.file_manager.get_file(file_id)
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {str(e)}")
            raise

    async def list_files(
        self, dataset_id: str, 
        skip: int = 0, limit: int = 100
    ) -> List[LabeledImageFile]:
        """파일 목록을 조회합니다."""
        try:
            return await self.file_manager.list_files(dataset_id, skip, limit)
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise

    async def delete_file(self, dataset_id: str, file_id: str) -> bool:
        """
        파일을 삭제합니다. (MongoDB + MinIO)
        """
        try:
            # 파일 정보 조회
            file = await self.get_file(file_id)
            if not file:
                logger.warning(f"File not found for deletion: {file_id}")
                return False
            # MinIO에서 파일 삭제
            try:
                await self.storage.remove_object(file.path)
            except Exception as e:
                logger.error(f"Failed to delete file from storage: {str(e)}")
            # MongoDB에서 파일 삭제
            return await self.file_manager.delete_file(file_id)
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {str(e)}")
            raise

    async def upload_file(self, dataset_id: str, file: UploadFile):
        """
        파일 업로드 (구현 필요 시 실제 로직 작성)
        """
        raise NotImplementedError("upload_file is not implemented yet")
