from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile, 
    File, 
    Path, 
    Query, 
    status, 
    Body, 
    Depends
)
from fastapi.responses import (
    JSONResponse
)
from typing import (
    List, 
    Optional
)
from core.minio import (
    MinioStorageClient
)
from core.mongodb import (
    MongoDBClient
)
from core.config import (
    MONGODB_URL, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTIONS,
    MIME_TYPES
)
from models.dataset.raw_model import (
    RawDatasetCreate,
    RawDatasetUpdate,
    RawDataUpload,
    RawDatasetInfo,
    RawDataInfo
)
from utils.counter import (
    get_next_counter
)
from utils.time import (
    get_current_time_kst
)
from utils.init import (
    init
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_raw_dataset(uid: str, dataset: RawDatasetCreate):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    did = await get_next_counter(mongo_client, "raw_datasets", uid=uid, prefix="R", field="did", width=4)

    # 데이터셋 정보
    dataset_info = RawDatasetInfo(
        _id = uid + did,  # ex. 0001R0001
        uid = uid,
        did = did,
        name = dataset.name,
        description = dataset.description,
        type = dataset.type,
        path = f"datasets/raw/{did}",
        total = 0,
        created_at = get_current_time_kst()  # 문자열로 변환
    )

    # MongoDB에 데이터셋 정보 저장
    await mongo_client.db["raw_datasets"].insert_one(dataset_info.dict(by_alias=True))
    logger.info(f"Created raw dataset: {dataset_info}")

    doc = dataset_info.dict(by_alias=True)
    doc.pop("_id", None)
    doc.pop("uid", None)

    return doc


async def update_raw_dataset(uid: str, update: RawDatasetUpdate):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    did = update.did
    
    check = await mongo_client.db["raw_datasets"].find_one({"_id": uid+did})
    if not check:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    update_fields = {}
    if update.name is not None:
        update_fields["name"] = update.name
    if update.description is not None:
        update_fields["description"] = update.description
    if update.type is not None:
        update_fields["type"] = update.type
    if update_fields:
        await mongo_client.db["raw_datasets"].update_one(
            {"_id": uid+did},
            {"$set": update_fields}
        )
    updated_dataset = await mongo_client.db["raw_datasets"].find_one({"_id": uid+did})
    if not updated_dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    
    mongo_client.db["raw_data"].update_many(
        {"did": did, "uid": uid},
        {"$set": {"dataset": update.name}}
    )

    doc = updated_dataset
    doc.pop("_id", None)
    doc.pop("uid", None)

    return doc


async def list_raw_datasets(uid: str):

    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    datasets = await mongo_client.db["raw_datasets"].find({"uid": uid}).to_list(length=None)
    for dataset in datasets:
        dataset.pop("_id", None)
        dataset.pop("uid", None)

    return datasets


async def get_raw_dataset(uid: str, did: str):

    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    dataset = await mongo_client.db["raw_datasets"].find_one({"_id": uid+did})
    dataset.pop("_id", None)
    dataset.pop("uid", None)

    data_list = await mongo_client.db["raw_data"].find({"did": did, "uid": uid}).to_list(length=None)
    for data in data_list:
        data.pop("_id", None)
        data.pop("uid", None)

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    # 데이터셋 정보와 데이터 리스트를 RawDatasetInfoWithData로 변환하여 반환
    return dataset, data_list


async def upload_raw_data(uid, raw_data_upload: RawDataUpload, files: List[UploadFile]):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]
    
    did = raw_data_upload.did

    check = await mongo_client.db["raw_datasets"].find_one({"_id": uid+did})
    if not check:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    documents = []
    time = get_current_time_kst()

    try:
        # 파일을 (파일명, 바이트) 튜플 리스트로 변환 (filename이 None일 경우 "unnamed"로 대체)
        files_to_upload = [((file.filename or "unnamed"), await file.read()) for file in files]

        dataset_info = await mongo_client.db["raw_datasets"].find_one({"_id": uid+did})
        dataset_name = dataset_info.get("name") if dataset_info else "Unnamed Raw Dataset"
        data_collection = "raw_data"
        dataset_collection = "raw_datasets"

        for filename, file_bytes in files_to_upload:
            filename = filename or "unnamed"
            file_ext = filename.split('.')[-1].lower()  # 확장자만 추출

            if f".{file_ext}" in MIME_TYPES.values():
                key = f"datasets/raw/{did}/{filename}"
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            doc = {
                "_id": uid + did + filename,
                "uid": uid,
                "did": did,
                "dataset": dataset_name,
                "name": filename,
                "type": "application/octet-stream",
                "file_format": file_ext,
                "path": f"{key}",
                "created_at": time
            }
            documents.append(doc)

            await mongo_client.upload_data(uid=uid, 
                                           did=did, 
                                           data_collection=data_collection,
                                           dataset_collection=dataset_collection,
                                           doc=doc)

            await minio_client.upload_files(uid, file_bytes, key)

        docs = documents
        for doc in docs:
            doc.pop("_id", None)
            doc.pop("uid", None)

        return docs

    except Exception as e:
        logger.error("Error uploading raw data: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )