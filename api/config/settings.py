from typing import List, Dict
import os
import aioboto3
from botocore.config import Config
import json
from datetime import datetime

# Custom JSON encoder to handle datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

# MongoDB 설정
MONGODB_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME", "admin")
MONGODB_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "password")
MONGODB_HOST = os.getenv("MONGODB_HOST", "mongodb")
MONGODB_PORT = os.getenv("MONGODB_PORT", "27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "labeled_images")

MONGODB_URL = f"mongodb://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}/?authSource=admin"

# MinIO 설정
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
MINIO_USE_SSL = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'

MINIO_CONFIG = {
    'endpoint_url': f"http://{MINIO_ENDPOINT}:{MINIO_PORT}",
    'aws_access_key_id': MINIO_ACCESS_KEY,
    'aws_secret_access_key': MINIO_SECRET_KEY,
    'region_name': 'us-east-1',
    'use_ssl': MINIO_USE_SSL,
    'config': Config(
        s3={
            'addressing_style': 'path'
        },
        signature_version='s3v4',
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        }
    )
}

# 버킷 이름 설정
BUCKET_NAME = 'jwlee'

# 지원하는 이미지 형식
SUPPORTED_IMAGE_EXTENSIONS: List[str] = ['.jpg', '.jpeg', '.png']
# 지원하는 레이블 형식
SUPPORTED_LABEL_EXTENSIONS: List[str] = ['text/plain', '.txt', '.json']

# 스토리지 구조
STORAGE_STRUCTURE: Dict = {
    'datasets': {
        'images': {}  # 카테고리는 동적으로 생성됨
    }
}
