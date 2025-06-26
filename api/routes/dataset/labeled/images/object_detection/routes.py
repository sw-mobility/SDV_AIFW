"""Object Detection API Router (COCO/YOLO flat structure, no model/version dir)"""
import os
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Path, Query, Form, Depends, Body, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional, Dict, Any
from utils.logging import logger

from core.storage import storage_client
from core.mongodb import MongoDB, get_dataset_collection
from models.dataset.raw.images.mongodb import Dataset, DatasetCreate, ImageFile, DatasetFile, convert_mongo_document
from models.dataset.raw.images.api import FileUploadResponse, FileUploadError
from .file_handler import ObjectDetectionFileHandler
# from .fix_dataset_type import router as admin_router  # 필요시

router = APIRouter(tags=["Object Detection"])

# 파일 핸들러 인스턴스
_file_handler = None

def get_dataset_collection_object_detection():
    """Object Detection 데이터셋 컬렉션 반환"""
    return get_dataset_collection(dataset_type="labeled_images", task_type="object_detection")

def get_file_collection():
    return MongoDB.db["labeled_image_files"]

def get_file_handler():
    global _file_handler
    if _file_handler is None:
        _file_handler = ObjectDetectionFileHandler()
    return _file_handler

def join_path(*args):
    return "/".join([str(a) for a in args if a])

def resolve_dataset_id(dataset_id_or_name: str):
    """dataset_id 또는 name을 받아 실제 ObjectId(str)와 name을 반환"""
    from bson import ObjectId
    dataset_collection = get_dataset_collection_object_detection()
    async def _resolve():
        if ObjectId.is_valid(dataset_id_or_name):
            dataset_obj = await dataset_collection.find_one({"_id": ObjectId(dataset_id_or_name)})
        else:
            dataset_obj = await dataset_collection.find_one({"name": dataset_id_or_name})
        if not dataset_obj:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return str(dataset_obj["_id"]), dataset_obj["name"]
    return _resolve

@router.get("/datasets", response_model=List[Dataset])
async def list_datasets():
    """
    모든 Object Detection 데이터셋 목록을 조회합니다.
    """
    try:
        dataset_collection = get_dataset_collection_object_detection()
        filter_query = {"type": "object_detection"}
        cursor = dataset_collection.find(filter_query)
        datasets = []
        async for document in cursor:
            document = convert_mongo_document(document)
            datasets.append(document)
        return datasets
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets", response_model=Dataset, status_code=201)
async def create_dataset(dataset: DatasetCreate):
    """
    새로운 Object Detection 데이터셋을 생성합니다.
    """
    try:
        dataset_collection = get_dataset_collection_object_detection()
        existing = await dataset_collection.find_one({"name": dataset.name})
        if existing:
            raise HTTPException(status_code=409, detail="Dataset with this name already exists.")
        document = dataset.dict()
        document["created_at"] = document["updated_at"] = datetime.utcnow()
        document["type"] = "object_detection"
        result = await dataset_collection.insert_one(document)
        dataset_id = str(result.inserted_id)
        dataset_path = f"datasets/labeled/images/object_detection/{dataset.name}"
        await storage_client.create_directory(dataset_path)
        await storage_client.create_directory(f"{dataset_path}/images")
        await storage_client.create_directory(f"{dataset_path}/labels")
        return convert_mongo_document(document)
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_id}", response_model=Dataset)
async def get_dataset(dataset_id: str = Path(..., description="데이터셋 ID 또는 이름")):
    """
    특정 Object Detection 데이터셋 정보를 조회합니다.
    """
    try:
        dataset_collection = get_dataset_collection_object_detection()
        file_collection = get_file_collection()
        from bson import ObjectId
        if ObjectId.is_valid(dataset_id):
            document = await dataset_collection.find_one({"_id": ObjectId(dataset_id)})
            if document:
                document = convert_mongo_document(document)
        else:
            document = await dataset_collection.find_one({"name": dataset_id})
            if not document:
                raise HTTPException(status_code=404, detail="Dataset not found")
            document = convert_mongo_document(document)
        dataset_id_str = str(document["_id"])
        file_count = await file_collection.count_documents({"dataset_id": dataset_id_str})
        document["file_count"] = file_count
        return document  # dict로 반환
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/datasets/{dataset_id}/upload", status_code=201)
async def upload_multiple_files(
    dataset_id: str = Path(..., description="데이터셋 ID 또는 이름"),
    files: List[UploadFile] = File(..., description="이미지 파일들"),
    label_files: Optional[List[UploadFile]] = File(None, description="레이블 텍스트 파일들 (YOLO 형식)"),
    data_yaml: Optional[UploadFile] = File(None, description="data.yaml 파일")
):
    handler = get_file_handler()
    dataset_id_str, dataset_name = await resolve_dataset_id(dataset_id)()
    results = []
    for image_file in files:
        label_file = None
        if label_files:
            base_name = os.path.splitext(image_file.filename)[0]
            for lf in label_files:
                if os.path.splitext(lf.filename)[0] == base_name:
                    label_file = lf
                    break
        upload_result, _ = await handler.upload_file(dataset_id_str, dataset_name, image_file, label_file)
        results.append(upload_result)
    # data.yaml 업로드 처리
    if data_yaml:
        yaml_path = f"datasets/labeled/images/object_detection/{dataset_name}/data.yaml"
        contents = await data_yaml.read()
        await storage_client.put_object(
            key=yaml_path,
            body=contents,
            content_type=data_yaml.content_type or "text/yaml"
        )
        # (선택) MongoDB에 메타데이터 저장
        # await handler.upload_data_yaml_metadata(dataset_id_str, dataset_name, yaml_path, len(contents))
        results.append({
            "filename": data_yaml.filename,
            "path": yaml_path,
            "status": "success",
            "message": "data.yaml uploaded successfully"
        })
    return results

@router.get("/datasets/{dataset_id}/files", response_model=List[DatasetFile])
async def list_files(dataset_id: str = Path(..., description="데이터셋 ID 또는 이름")):
    handler = get_file_handler()
    dataset_id_str, _ = await resolve_dataset_id(dataset_id)()
    return await handler.list_files(dataset_id_str)
