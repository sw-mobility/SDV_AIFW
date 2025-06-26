"""이미지 파일 처리 핸들러"""
from typing import List
import aiofiles
from fastapi import HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from routes.base import BaseFileHandler
from core.storage import storage_client
from storage.managers.raw.images.manager import RawImageStorageManager
from storage.managers.raw.images.mongodb.file_manager import ImageFileMongoManager
from storage.base import FileUploadError, FileDownloadError
from utils.logging import logger
from models.dataset.raw.images.mongodb import ImageFile

class RawImageFileHandler(BaseFileHandler):
    """Raw image file handling operations (category 개념 제거)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init__()
        return cls._instance

    def __init__(self):
        super().__init__(storage_client)
        self.storage_manager = RawImageStorageManager()

    async def upload_file(self, dataset_id: str, file: UploadFile) -> ImageFile:
        """Upload an image file (flat 구조)"""
        try:
            return await self.storage_manager.upload_file(dataset_id, file)
        except FileUploadError as e:
            logger.error(f"Upload file failed: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file: {str(e)}")
        except ValueError as e:
            logger.error(f"Upload validation failed: {str(e)}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file: {str(e)}")
        except Exception as e:
            logger.error(f"Upload file failed with unexpected error: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload file: {str(e)}")

    async def download_file(self, dataset_id: str, file_id: str) -> StreamingResponse:
        """Download an image file (flat 구조)"""
        try:
            file_stream, file_info = await self.storage_manager.get_file(dataset_id, file_id)
            return StreamingResponse(
                file_stream,
                media_type=file_info.content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="{file_info.filename}"',
                    "Content-Length": str(file_info.size)
                }
            )
        except FileDownloadError as e:
            logger.error(f"Download file failed: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to download file: {str(e)}")
        except ValueError as e:
            logger.error(f"Download validation failed: {str(e)}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid file: {str(e)}")
        except Exception as e:
            logger.error(f"Download file failed with unexpected error: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to download file: {str(e)}")

    async def list_files(self, dataset_id: str) -> List[ImageFile]:
        """List all files in a dataset (flat 구조)"""
        try:
            return await self.storage_manager.list_files(dataset_id)
        except Exception as e:
            logger.error(f"List files failed: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to list files: {str(e)}")

    async def delete_file(self, dataset_id: str, file_id: str) -> bool:
        """Delete an image file (flat 구조)"""
        try:
            return await self.storage_manager.delete_file(dataset_id, file_id)
        except Exception as e:
            logger.error(f"Delete file failed: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to delete file: {str(e)}")
