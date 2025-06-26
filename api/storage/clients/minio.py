from typing import Optional, AsyncGenerator, Dict, Any
import aioboto3
from botocore.exceptions import ClientError
from config.settings import MINIO_CONFIG, BUCKET_NAME
from utils.logging import logger

class MinioStorageClient:
    """MinIO 스토리지 클라이언트"""
    
    def __init__(self):
        self.session = aioboto3.Session()
        self.config = MINIO_CONFIG
        self.bucket_name = BUCKET_NAME

    async def _get_client(self):
        """S3 클라이언트를 생성합니다."""
        return self.session.client('s3', **self.config)

    async def init_bucket(self):
        """버킷을 초기화합니다."""
        async with await self._get_client() as s3:
            try:
                await s3.head_bucket(Bucket=self.bucket_name)
                logger.info(f"Bucket {self.bucket_name} exists")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    await s3.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created bucket {self.bucket_name}")
                else:
                    raise

    async def put_object(self, key: str, body: bytes, content_type: Optional[str] = None) -> Dict[str, Any]:
        """객체를 저장합니다."""
        params = {
            'Bucket': self.bucket_name,
            'Key': key,
            'Body': body
        }
        if content_type:
            params['ContentType'] = content_type

        async with await self._get_client() as s3:
            response = await s3.put_object(**params)
            return response

    async def get_object(self, key: str) -> Dict[str, Any]:
        """객체를 조회합니다."""
        async with await self._get_client() as s3:
            return await s3.get_object(Bucket=self.bucket_name, Key=key)

    async def list_objects(self, prefix: str = "", delimiter: Optional[str] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """객체 목록을 조회합니다."""
        async with await self._get_client() as s3:
            paginator = s3.get_paginator('list_objects_v2')
            params = {
                'Bucket': self.bucket_name,
                'Prefix': prefix
            }
            if delimiter:
                params['Delimiter'] = delimiter

            async for page in paginator.paginate(**params):
                yield page

    async def check_health(self) -> bool:
        """스토리지 상태를 확인합니다."""
        try:
            async with await self._get_client() as s3:
                await s3.head_bucket(Bucket=self.bucket_name)
                return True
        except Exception as e:
            logger.error(f"Storage health check failed: {str(e)}")
            return False

    async def delete_object(self, key: str) -> Dict[str, Any]:
        """객체를 삭제합니다."""
        async with await self._get_client() as s3:
            return await s3.delete_object(Bucket=self.bucket_name, Key=key)

    async def head_object(self, key: str) -> Dict[str, Any]:
        """객체의 메타데이터를 조회합니다."""
        async with await self._get_client() as s3:
            return await s3.head_object(Bucket=self.bucket_name, Key=key)

    async def create_directory(self, path: str) -> Dict[str, Any]:
        """
        객체 스토리지에서 디렉토리 구조를 생성합니다.
        (MinIO는 실제 디렉토리가 아닌 경로가 포함된 객체로 디렉토리를 표현)
        
        Args:
            path: 생성할 디렉토리 경로
        
        Returns:
            S3 응답 객체
        """
        # 경로 끝에 / 추가
        if not path.endswith('/'):
            path = f"{path}/"
            
        # 빈 객체 생성으로 '디렉토리' 표현
        return await self.put_object(
            key=path,
            body=b'',
            content_type='application/x-directory'
        )
