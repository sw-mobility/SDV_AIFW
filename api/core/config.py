from typing import List, Dict
import os
import aioboto3
from botocore.config import Config
import json


# MongoDB 설정
MONGODB_USER = os.getenv("MONGO_INITDB_ROOT_USERNAME", "admin")
MONGODB_PASSWORD = os.getenv("MONGO_INITDB_ROOT_PASSWORD", "password")
MONGODB_HOST = os.getenv("MONGODB_HOST", "mongodb")
MONGODB_PORT = os.getenv("MONGODB_PORT", "27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "keti_aifw")
MONGODB_COLLECTIONS= [
    "default", "users", "raw_datasets", "raw_data", "labeled_datasets", "labeled_data",
    "projects", "codebases", "trn_hst", "opt_hst", "val_hst"
]

MONGODB_URL = f"mongodb://{MONGODB_USER}:{MONGODB_PASSWORD}@{MONGODB_HOST}:{MONGODB_PORT}/?authSource=admin"


# MinIO 설정
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio')
MINIO_PORT = os.getenv('MINIO_PORT', '9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ROOT_USER', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin123')
MINIO_USE_SSL = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'
MINIO_CORE_BUCKET = os.getenv('MINIO_CORE_BUCKET', 'keti-aifw')

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

MIME_TYPES = {
    # 지원하는 이미지 형식
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    # 지원하는 yaml 형식
    'application/x-yaml': '.yaml',
    'application/x-yml': '.yml',
    # 지원하는 레이블 형식
    'text/plain': '.txt',
    'application/json': '.json',
}

BATCH_SIZE = 1000
API_WORKDIR = "/app/workspace"
TRAINING_WORKDIR = "/app/workspace/training"
FRONTEND_WORKDIR = "/app/workspace/frontend"
VALIDATION_WORKDIR = "/app/workspace/validation"
LABELING_WORKDIR = "/app/workspace/labeling"
OPTIMIZING_WORKDIR = "/app/workspace/optimizing"


class YOLO_MODELS:

    DET = [
        # YOLOv5u (Detection)
        "yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt", "yolov5xu.pt",
        "yolov5n6u.pt", "yolov5s6u.pt", "yolov5m6u.pt", "yolov5l6u.pt", "yolov5x6u.pt",
        "yolov5nu", "yolov5su", "yolov5mu", "yolov5lu", "yolov5xu",
        "yolov5n6u", "yolov5s6u", "yolov5m6u", "yolov5l6u", "yolov5x6u",

        # YOLOv8 (Detection)
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",

        # YOLO11 (Detection)
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
    ]

    CLS = [
        # YOLOv8 (Classification)
        "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt", "yolov8l-cls.pt", "yolov8x-cls.pt",
        "yolov8n-cls", "yolov8s-cls", "yolov8m-cls", "yolov8l-cls", "yolov8x-cls",

        # YOLO11 (Classification)
        "yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt",
        "yolo11n-cls", "yolo11s-cls", "yolo11m-cls", "yolo11l-cls", "yolo11x-cls",
    ]

    SEG = [
        # YOLOv8 (Segmentation)
        "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
        "yolov8n-seg", "yolov8s-seg", "yolov8m-seg", "yolov8l-seg", "yolov8x-seg",

        # YOLO11 (Segmentation)
        "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
        "yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg",
    ]

    POSE = [
        # YOLOv8 (Pose)
        "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt",
        "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
        # YOLO11 (Pose)
        "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
        "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
    ]

    OBB = [
        # YOLO11 (OBB)
        "yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt",
        "yolo11n-obb", "yolo11s-obb", "yolo11m-obb", "yolo11l-obb", "yolo11x-obb"
    ]