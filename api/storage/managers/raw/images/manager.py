"""원본 이미지 데이터셋 저장소 통합 관리자 (COCO/YOLO flat 구조, no category)"""
import os
import uuid
from typing import BinaryIO, Optional, List, Tuple, Any
from datetime import datetime
from botocore.exceptions import ClientError
from fastapi import UploadFile

from core.storage import storage_client
from core.mongodb import MongoDB, get_dataset_collection
from storage.base import (
    BaseStorageManager, StorageError, FileUploadError as BaseFileUploadError, 
    FileDownloadError, DatasetOperationError
)
from storage.managers.raw.images.mongodb.dataset_manager import DatasetMongoManager
from storage.managers.raw.images.mongodb.file_manager import ImageFileMongoManager
from models.dataset.raw.images.api import FileUploadResponse, FileUploadError
from models.dataset.raw.images.mongodb import Dataset, DatasetCreate, ImageFile
from config.settings import SUPPORTED_IMAGE_EXTENSIONS
from utils.logging import logger
from urllib.parse import unquote

def get_dataset_collection_raw():
    return get_dataset_collection(dataset_type="raw_images")

class RawImageStorageManager(BaseStorageManager[Dataset, ImageFile, DatasetCreate]):
    """원본 이미지 데이터셋 저장소 통합 관리자 (flat 구조)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.dataset_manager = DatasetMongoManager()
            cls._instance.file_manager = ImageFileMongoManager()
        return cls._instance

    @property
    def storage(self):
        return storage_client

    def _get_dataset_path(self, name: str) -> str:
        return f"datasets/raw/images/{name}"

    def _get_file_path(self, dataset_name: str, filename: str) -> str:
        return f"{self._get_dataset_path(dataset_name)}/{filename}"

    async def create_dataset(self, dataset_create: DatasetCreate) -> Dataset:
        result = None
        try:
            dataset = Dataset(
                name=dataset_create.name,
                description=dataset_create.description,
                id=None
            )
            result = await self.dataset_manager.create_dataset(dataset)
            await self.storage.create_directory(
                path=self._get_dataset_path(dataset.name)
            )
            logger.info(f"Created dataset: {dataset.name}")
            return result
        except Exception as e:
            try:
                if result:
                    await self.dataset_manager.delete_dataset(dataset.name)
            except:
                pass
            raise DatasetOperationError(f"Failed to create dataset: {str(e)}")

    async def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        return await self.dataset_manager.get_dataset(dataset_id)

    async def list_datasets(self) -> List[Dataset]:
        return await self.dataset_manager.list_datasets()

    async def delete_dataset(self, dataset_id: str) -> bool:
        return await self.dataset_manager.delete_dataset(dataset_id)

    async def upload_file(self, dataset_id: str, file: UploadFile) -> ImageFile:
        """이미지를 업로드합니다."""
        file_path = None
        file_content = None
        try:
            # MinIO에 파일 업로드
            file_content = await file.read()
            file_path = self._get_file_path(dataset_id, file.filename)
            
            await self.storage.put_object(
                key=file_path,
                body=file_content,
                content_type=file.content_type
            )            # 업로드 메타데이터 생성
            upload_metadata = {
                "upload_timestamp": datetime.now().isoformat(),
                "original_filename": file.filename,
                "content_type": file.content_type
            }

            # MongoDB에 파일 정보 저장 - 원본 파일명과 메타데이터 포함
            file_info = ImageFile(
                filename=file.filename,  # 원본 파일명 저장
                path=file_path,          # 저장 경로 (이제 원본 파일명과 동일)
                dataset_name=dataset_id,
                content_type=file.content_type,
                size=len(file_content),
                metadata=upload_metadata
            )
            
            result = await self.file_manager.create_file(
                dataset_name=dataset_id,
                filename=file.filename,
                path=file_path,
                content_type=file.content_type,
                size=len(file_content),
                metadata=upload_metadata
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Upload failed: {str(e)}")
            # 실패 시 롤백
            if file_path:
                try:
                    await self.storage.delete_object(file_path)
                    logger.info(f"Rolled back file upload: {file_path}")
                except:
                    logger.warning(f"Failed to rollback file upload: {file_path}")
            raise BaseFileUploadError(f"Failed to upload file: {str(e)}")

    async def list_files(self, dataset_id: str) -> List[ImageFile]:
        """모든 파일 목록을 조회합니다."""
        try:
            return await self.file_manager.list_files(dataset_id)
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise FileDownloadError(f"Failed to list files: {str(e)}")
    
    async def get_file(self, dataset_id: str, file_id: str) -> Tuple[BinaryIO, ImageFile]:
        """파일을 조회합니다."""
        try:
            # 파일 정보 조회
            file_info = await self.file_manager.get_file(dataset_id, file_id)
            if not file_info:
                raise ValueError("File not found")

            # MinIO에서 파일 다운로드
            file_obj = await self.storage.get_object(file_info.path)
            return file_obj["Body"], file_info
            
        except Exception as e:
            logger.error(f"Download failed: {str(e)}")
            raise FileDownloadError(f"Failed to download file: {str(e)}")

    async def delete_file(self, dataset_id: str, file_id: str) -> bool:
        """파일을 삭제합니다."""
        try:
            # 파일 정보 조회
            file_info = await self.file_manager.get_file(dataset_id, file_id)
            if not file_info:
                return False

            # MinIO에서 파일 삭제
            await self.storage.delete_object(file_info.path)

            # MongoDB에서 파일 정보 삭제
            result = await self.file_manager.delete_file(dataset_id, file_id)
            
            logger.info(f"Deleted file: {file_info.path} from dataset {dataset_id}")
            return result
            
        except Exception as e:
            logger.error(f"Delete file failed: {str(e)}")
            raise StorageError(f"Failed to delete file: {str(e)}")
