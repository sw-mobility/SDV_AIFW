"""기본 파일 처리 기능"""
from typing import BinaryIO, Optional, Tuple
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from botocore.exceptions import ClientError
from storage.clients.minio import MinioStorageClient
from utils.logging import logger

class BaseFileHandler:
    def __init__(self, storage_client: MinioStorageClient):
        self.storage = storage_client

    async def download_file(self, key: str, filename: str = None) -> StreamingResponse:
        """파일을 다운로드합니다."""
        try:
            response = await self.storage.get_object(key)
            filename = filename or key.split('/')[-1]
            return StreamingResponse(
                response['Body'],
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise HTTPException(status_code=404, detail="File not found")
            logger.error(f"Download failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Download failed")
        except Exception as e:
            logger.error(f"Unexpected download error: {str(e)}")
            raise HTTPException(status_code=500, detail="Download failed")

    async def check_health(self) -> dict:
        """스토리지 서버의 상태를 확인합니다."""
        try:
            is_healthy = await self.storage.check_health()
            if is_healthy:
                return {"status": "healthy", "bucket": "accessible"}
            return {"status": "unhealthy", "error": "Bucket not accessible"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
