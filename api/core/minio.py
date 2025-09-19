from importlib.resources import files
import os
import shutil
from typing import Optional, AsyncGenerator, Dict, Any, List
import aioboto3 # type: ignore
from botocore.exceptions import ClientError # type: ignore
from core.config import (
    MINIO_CONFIG,
    MINIO_CORE_BUCKET,
    API_WORKDIR,
    BATCH_SIZE
)
from math import ceil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MinioStorageClient:
    """MinIO 스토리지 클라이언트"""
    
    def __init__(self):
        self.session = aioboto3.Session()
        self.config = MINIO_CONFIG

    async def _get_client(self):
        """S3 클라이언트를 생성합니다."""
        return self.session.client('s3', **self.config)
    
    async def init_core_bucket(self):
        """코어 버킷을 초기화합니다."""
        async with await self._get_client() as s3:
            try:
                logger.info(f"Checking core bucket: {MINIO_CORE_BUCKET}")
                await s3.head_bucket(Bucket=MINIO_CORE_BUCKET)
                logging.info(f"Core bucket exists")
            except ClientError as e:
                if e.response['Error']['Code'] in ['404', 'NoSuchBucket']:
                    await s3.create_bucket(Bucket=MINIO_CORE_BUCKET)
                    logging.info(f"Created core bucket")
                else:
                    raise

    async def init_bucket(self, uid):
        """사용자 버킷을 초기화합니다."""
        async with await self._get_client() as s3:
            try:
                await s3.head_bucket(Bucket=uid)
                logging.info(f"Bucket {uid} exists")

            except ClientError as e:
                if e.response['Error']['Code'] in ['404', 'NoSuchBucket']:
                    await s3.create_bucket(Bucket=uid)
                    logging.info(f"Created bucket {uid}")
                else:
                    raise

    async def core_default_assets_init(self):
        """코어 기본 에셋 파일 읽어오기"""
        base_dir = "default_assets"
        files: list[tuple[str, bytes]] = []

        for root, _, filenames in os.walk(base_dir):
            for file in filenames:
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'rb') as f:
                        file_bytes = f.read()
                        key = os.path.relpath(file_path, base_dir)
                        files.append((key, file_bytes))
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {e}")
                    raise

        """코어 파일들을 업로드합니다."""
        async with await self._get_client() as s3:
            for key, file_bytes in files:
                try:
                    await s3.put_object(Bucket=MINIO_CORE_BUCKET, Key=key, Body=file_bytes)
                    logging.info(f"Uploaded {key} to core bucket")
                except ClientError as e:
                    logging.error(f"Error uploading {key} to core bucket: {e}")
                    raise


    async def upload_directory(self, bucket: str, base_dir: str, prefix: str = ""):
        """
        base_dir 내부의 모든 파일을 MinIO에 prefix 경로로 디렉토리 구조를 유지하며 업로드합니다.
        """
        async with await self._get_client() as s3:
            for root, _, files in os.walk(base_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, base_dir)  # base_dir 기준 상대 경로
                    s3_key = os.path.join(prefix, rel_path).replace("\\", "/")  # prefix + 상대경로
                    with open(abs_path, "rb") as f:
                        file_bytes = f.read()
                    await s3.put_object(Bucket=bucket, Key=s3_key, Body=file_bytes)
                    logging.info(f"Uploaded {s3_key} to bucket {bucket}")

  
    async def upload_files(self, uid: str, file_bytes: bytes, key: str):
        MULTIPART_THRESHOLD = 5 * 1024 * 1024  # 5MB
        CHUNK_SIZE = 5 * 1024 * 1024  # 5MB
        async with await self._get_client() as s3:

            if len(file_bytes) < MULTIPART_THRESHOLD:
                await s3.put_object(Bucket=uid, Key=key, Body=file_bytes)
                logging.info(f"Uploaded {key} to bucket {uid} (single part)")
            else:
                mpu = await s3.create_multipart_upload(Bucket=uid, Key=key)
                upload_id = mpu['UploadId']
                parts = []
                try:
                    num_parts = ceil(len(file_bytes) / CHUNK_SIZE)
                    for i in range(num_parts):
                        part_number = i + 1
                        start = i * CHUNK_SIZE
                        end = min(start + CHUNK_SIZE, len(file_bytes))
                        chunk = file_bytes[start:end]
                        part = await s3.upload_part(
                            Bucket=uid,
                            Key=key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk
                        )
                        parts.append({
                            'ETag': part['ETag'],
                            'PartNumber': part_number
                        })
                    await s3.complete_multipart_upload(
                        Bucket=uid,
                        Key=key,
                        UploadId=upload_id,
                        MultipartUpload={'Parts': parts}
                    )
                    logging.info(f"Uploaded {key} to bucket {uid} (multipart)")
                    
                except Exception as e:
                    await s3.abort_multipart_upload(Bucket=uid, Key=key, UploadId=upload_id)
                    logging.error(f"Multipart upload aborted for {key}: {e}")
                    raise


    async def delete_dataset(
        self,
        uid: str,
        target_path_list: List[str]
    ):
        """
        MinIO에서 데이터셋을 삭제합니다.
        """
        async with await self._get_client() as s3:

            if not target_path_list:
                logger.info("target_path_list not specified, skipping deletion")
                return

            for path in target_path_list:

                paginator = s3.get_paginator('list_objects_v2')
                contents = []
                
                async for result in paginator.paginate(Bucket=uid, Prefix=path):
                    page_contents = result.get('Contents') or []
                    contents.extend(page_contents)

                if not contents:
                    continue

                delete_keys = [{'Key': obj['Key']} for obj in contents]

                # 배치로 1000개씩 삭제
                for i in range(0, len(delete_keys), BATCH_SIZE):
                    batch = delete_keys[i:i+BATCH_SIZE]
                    try:
                        await s3.delete_objects(
                            Bucket=uid,
                            Delete={'Objects': batch}
                        )
                        for obj in batch:
                            logging.info(f"Deleted {obj['Key']} from bucket {uid}")
                    except ClientError as e:
                        logging.error(f"Error deleting objects with prefix {path} from bucket {uid}: {e}")
                        raise


    async def delete_data(
            self,
            uid: str,
            target_path_list: List[str]
    ):
        """
        MinIO에서 여러 파일을 배치로 삭제합니다.
        """
        async with await self._get_client() as s3:
            for i in range(0, len(target_path_list), BATCH_SIZE):
                batch = target_path_list[i:i+BATCH_SIZE]
                delete_keys = [{'Key': path} for path in batch]
                try:
                    await s3.delete_objects(
                        Bucket=uid,
                        Delete={'Objects': delete_keys}
                    )
                    for obj in delete_keys:
                        logging.info(f"Deleted {obj['Key']} from bucket {uid}")
                except ClientError as e:
                    logging.error(f"Error deleting objects from bucket {uid}: {e}")
                    raise

        
    async def compress_dataset_to_download(
            self, 
            uid: str, 
            target_name: str,
            basedir: str,
            target_path_list: List[str]
            ):

            dataset_path = f"{basedir}/{uid}/{target_name}"
            zip_path = f"{basedir}/{uid}/{target_name}"
            os.makedirs(f"{dataset_path}", exist_ok=True)

            async with await self._get_client() as s3:

                for path in target_path_list:
                    try:
                        split = path.split('/')[3:]
                        save_path = os.path.join(dataset_path, *split)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        s3_object = await s3.get_object(Bucket=uid, Key=path)
                        stream = s3_object['Body']

                        try:
                            data = await stream.read()
                            with open(save_path, 'wb') as f:
                                f.write(data)
                        except Exception as e:
                            logging.error(f"Error reading stream for {path}: {e}")
                    except Exception as e:
                        logging.error(f"Error downloading {path}: {e}")
                        raise

            zip = shutil.make_archive(zip_path, 'zip', dataset_path)
            return zip
    
    async def compress_data_to_download(
            self,
            uid: str,
            basedir: str,
            target_path_list: List[str]
            ):

            random = os.urandom(8).hex()
            base_dir = f"{basedir}/{uid}/{uid}+_+{random}"
            zip_path = base_dir
            os.makedirs(base_dir, exist_ok=True)
            logging.info(f"target_path_list: {target_path_list}")

            async with await self._get_client() as s3:
                for path in target_path_list:
                    try:
                        # 파일명 추출 (예: S3 키의 마지막 부분)
                        filename = os.path.basename(path)
                        file_save_path = os.path.join(base_dir, filename)
                        s3_object = await s3.get_object(Bucket=uid, Key=path)
                        stream = s3_object['Body']
                        try:
                            data = await stream.read()
                            with open(file_save_path, 'wb') as f:
                                f.write(data)
                        except Exception as e:
                            logging.error(f"Error reading stream for {path}: {e}")
                    except Exception as e:
                        logging.error(f"Error downloading {path}: {e}")
                        raise

            zip = shutil.make_archive(zip_path, 'zip', base_dir)
            logging.info(f"Compressed data saved to: {zip}")
            return zip


    async def get_dataset(self, uid: str, key: str):
        async with await self._get_client() as s3:
            try:
                response = await s3.get_object(
                    Bucket=uid,
                    Key=key,
                )
                data = await response['Body'].read()
                return data
            except Exception as e:
                logging.error(f"Failed to get dataset: {e}")
                return None
            

    async def download_minio_directory(self, bucket, prefix, dest_dir):
        async with await self._get_client() as s3:
            # 1. MinIO에서 prefix로 모든 파일 리스트 가져오기
            response = await s3.list_objects(Bucket=bucket, Prefix=prefix)
            objects = response.get('Contents', [])
            for obj in objects:
                rel_path = obj['Key'][len(prefix):].lstrip("/")  # prefix 이후 경로
                local_path = os.path.join(dest_dir, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3_object = await s3.get_object(Bucket=bucket, Key=obj['Key'])
                data = await s3_object['Body'].read()
                with open(local_path, "wb") as f:
                    f.write(data)


    async def download_minio_file(self, bucket: str, key: str, dest_path: str):
        """
        MinIO에서 단일 파일을 다운로드합니다.
        :param bucket: 버킷명
        :param key: S3 파일 경로(키)
        :param dest_path: 저장할 로컬 경로
        """
        async with await self._get_client() as s3:
            try:
                s3_object = await s3.get_object(Bucket=bucket, Key=key)
                data = await s3_object['Body'].read()
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(data)
                logging.info(f"Downloaded {key} from bucket {bucket} to {dest_path}")
            except Exception as e:
                logging.error(f"Error downloading file {key} from bucket {bucket}: {e}")
                raise


    async def list_objects(self, bucket: str, prefix: str) -> AsyncGenerator[Dict[str, Any], None]:
        async with await self._get_client() as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    yield obj


    async def get_object(self, Bucket: str, Key: str):
        async with await self._get_client() as s3:
            response = await s3.get_object(Bucket=Bucket, Key=Key)
            data = await response['Body'].read()
            return data