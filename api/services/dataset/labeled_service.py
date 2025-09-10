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
from models.dataset.labeled_model import (
    LabeledDatasetCreate,
    LabeledDatasetUpdate,
    LabeledDataUpload,
    LabeledDatasetInfo,
    LabeledDataInfo
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
import yaml

logger = logging.getLogger(__name__)


async def create_labeled_dataset(uid: str, dataset: LabeledDatasetCreate):

    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    did = await get_next_counter(mongo_client, "labeled_datasets", uid=uid, prefix="L", field="did", width=4)

    # 데이터셋 정보
    dataset_info = LabeledDatasetInfo(
        _id = uid + did,  # ex. 0001L0001
        uid = uid,
        did = did,
        name = dataset.name,
        description = dataset.description,
        classes = [],
        type = dataset.type,
        task_type = dataset.task_type,
        label_format = dataset.label_format,
        path = f"datasets/labeled/{did}",
        total = 0,
        created_at = get_current_time_kst()  # 문자열로 변환
    )

    # MongoDB에 데이터셋 정보 저장
    await mongo_client.db["labeled_datasets"].insert_one(dataset_info.dict(by_alias=True))
    logger.info(f"Created labeled dataset: {dataset_info}")

    doc = dataset_info.dict(by_alias=True)
    doc.pop("_id", None)
    doc.pop("uid", None)

    return doc


async def update_labeled_dataset(uid: str, update: LabeledDatasetUpdate):
    did = update.did

    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    check = await mongo_client.db["labeled_datasets"].find_one({"_id": uid+did})
    if not check:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    update_fields = {}
    if update.name is not None:
        update_fields["name"] = update.name
    if update.description is not None:
        update_fields["description"] = update.description
    if update.type is not None:
        update_fields["type"] = update.type
    if update.task_type is not None:
        update_fields["task_type"] = update.task_type
    if update.label_format is not None:
        update_fields["label_format"] = update.label_format
    if update_fields:
        await mongo_client.db["labeled_datasets"].update_one(
            {"_id": uid+did},
            {"$set": update_fields}
        )
    updated_dataset = await mongo_client.db["labeled_datasets"].find_one({"_id": uid+did})
    if not updated_dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
    
    mongo_client.db["labeled_data"].update_many(
        {"uid": uid, "did": did},
        {"$set": {"dataset": update.name}}
    )

    doc = updated_dataset
    doc.pop("_id", None)
    doc.pop("uid", None)

    return doc


async def list_labeled_datasets(uid: str) -> List[LabeledDatasetInfo]:
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    datasets = await mongo_client.db["labeled_datasets"].find({"uid": uid}).to_list(length=None)
    for dataset in datasets:
        dataset.pop("_id", None)
        dataset.pop("uid", None)

    return datasets


async def get_labeled_dataset(uid: str, did: str):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    dataset = await mongo_client.db["labeled_datasets"].find_one({"_id": uid+did})
    dataset.pop("_id", None)
    dataset.pop("uid", None)

    data_list = await mongo_client.db["labeled_data"].find({"did": did, "uid": uid}).to_list(length=None)
    for data in data_list:
        data.pop("_id", None)
        data.pop("uid", None)

    if not dataset:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    return dataset, data_list


async def upload_labeled_data(uid: str, labeled_data_upload: LabeledDataUpload, files: List[UploadFile]):
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    did = labeled_data_upload.did

    check = await mongo_client.db["labeled_datasets"].find_one({"_id": uid+did})
    if not check:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")

    uploaded_files = []
    documents = []
    names = []

    try:
        # 파일을 (파일명, 바이트) 튜플 리스트로 변환 (filename이 None일 경우 "unnamed"로 대체)
        files_to_upload = [((file.filename or "unnamed"), await file.read()) for file in files]

        data_collection = "labeled_data"
        dataset_collection = "labeled_datasets"
        dataset_info = await mongo_client.db[dataset_collection].find_one({"_id": uid+did})
        dataset_path = dataset_info.get("path") if dataset_info else f"datasets/labeled/{did}"
        origin_raw = dataset_info.get("origin_raw") if dataset_info else None 
        dataset = dataset_info.get("name") if dataset_info else "Unnamed Labeled Dataset"

        dataset_path = f"datasets/labeled/{did}"

        for filename, file_bytes in files_to_upload:
            filename = filename or "unnamed"
            file_ext = filename.split('.')[-1]

            if file_ext in ["jpg", "png", "jpeg"]:
                key = f"{dataset_path}/images/{filename}"
                type = "image"

            elif file_ext in ["json", "txt"]:
                key = f"{dataset_path}/labels/{filename}"
                type = "label"

            elif file_ext in ["yaml", "yml"]:
                key = f"{dataset_path}/{filename}"
                type = "yaml"
                data = yaml.safe_load(file_bytes)
                names = data.get("names", [])

            # 지원하지 않는 확장자에 대한 처리, 추후 다른 data type이 생기면 여기에 추가
            else:
                key = f"{dataset_path}/others/{filename}"
                type = "other"

            doc = {
                "_id": uid+did+filename,
                "uid": uid,
                "did": did,
                "dataset": dataset,
                "name": filename,
                "type": type,
                "origin_raw": origin_raw,
                "file_format": file_ext,
                "path": f"{key}",
                "created_at": get_current_time_kst()
            }
            documents.append(doc)

            await mongo_client.upload_data(uid=uid, 
                                        did=did,
                                        data_collection=data_collection,
                                        dataset_collection=dataset_collection,
                                        doc=doc
                                        )
            if names:
                await mongo_client.db["labeled_datasets"].update_one(
                    {"_id": uid+did}, 
                    {"$set": {"classes": names}}
                    )

            await minio_client.upload_files(uid, file_bytes, key)

        docs = documents
        for doc in docs:
            doc.pop("_id", None)
            doc.pop("uid", None)

        return docs

    except Exception as e:
        logger.error("Error uploading data: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=str(e)
        )