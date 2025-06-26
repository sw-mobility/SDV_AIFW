"""데이터셋 저장소 관리를 위한 추상 기본 클래스 (COCO/YOLO flat 구조, no category)"""
from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, List, TypeVar, Generic, Dict, Any, Tuple
from fastapi import UploadFile
from pydantic import BaseModel

class StorageError(Exception):
    """저장소 작업 중 발생하는 예외의 기본 클래스"""
    pass

class FileUploadError(StorageError):
    """파일 업로드 중 발생하는 예외"""
    pass

class FileDownloadError(StorageError):
    """파일 다운로드 중 발생하는 예외"""
    pass

class DatasetOperationError(StorageError):
    """데이터셋 작업 중 발생하는 예외"""
    pass

# 일반적인 데이터셋과 카테고리를 위한 타입 변수
DatasetType = TypeVar('DatasetType', bound=BaseModel)
FileType = TypeVar('FileType', bound=BaseModel)
CreateType = TypeVar('CreateType', bound=BaseModel)

class BaseStorageManager(ABC, Generic[DatasetType, FileType, CreateType]):
    """모든 데이터 타입(이미지, LIDAR 등)에 대한 저장소 관리자의 기본 클래스 (flat 구조)"""
    
    @abstractmethod
    async def create_dataset(self, dataset: CreateType) -> DatasetType:
        """새로운 데이터셋을 생성합니다."""
        pass

    @abstractmethod
    async def get_dataset(self, dataset_id: str) -> Optional[DatasetType]:
        """데이터셋을 조회합니다."""
        pass

    @abstractmethod
    async def list_datasets(self) -> List[DatasetType]:
        """모든 데이터셋 목록을 조회합니다."""
        pass

    @abstractmethod
    async def delete_dataset(self, dataset_id: str) -> bool:
        """데이터셋을 삭제합니다."""
        pass

    @abstractmethod
    async def upload_file(self, dataset_id: str, file: UploadFile) -> FileType:
        """데이터셋에 새로운 파일을 업로드합니다."""
        pass

    @abstractmethod
    async def list_files(self, dataset_id: str) -> List[FileType]:
        """데이터셋의 모든 파일 목록을 조회합니다."""
        pass

    @abstractmethod
    async def get_file(self, dataset_id: str, file_id: str) -> Tuple[BinaryIO, FileType]:
        """파일을 다운로드합니다."""
        pass

    @abstractmethod
    async def delete_file(self, dataset_id: str, file_id: str) -> bool:
        """파일을 삭제합니다."""
        pass
