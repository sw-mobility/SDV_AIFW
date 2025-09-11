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
from fastapi.responses import JSONResponse
from models.dataset.common_model import (
    DatasetDelete,
    DataDelete,
    DatasetDownload,
    DataDownload
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
    MIME_TYPES,
    API_WORKDIR,
    BATCH_SIZE
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
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def delete_dataset(uid: str, dataset_delete: DatasetDelete):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    target_did_list = dataset_delete.target_did_list
    target_path_list = dataset_delete.target_path_list

    await minio_client.delete_dataset(uid, target_path_list)
    await mongo_client.delete_dataset(uid, target_did_list=target_did_list)


async def delete_data(uid: str, data_delete: DataDelete):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    target_did = data_delete.target_did
    target_path_list = data_delete.target_path_list
    target_name_list = data_delete.target_name_list

    target_id_list = []
    for target_name in target_name_list:
        target_id_list.append(uid+target_did+target_name)

    await minio_client.delete_data(uid, target_path_list)
    await mongo_client.delete_data(uid, target_id_list=target_id_list)


async def compress_dataset(uid: str, download_dataset: DatasetDownload):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    target_did = download_dataset.target_did
    basedir = API_WORKDIR

    if target_did[0] == "R":
        target_collection = "raw_datasets"
        target_data_collection = "raw_data"
    elif target_did[0] == "L":
        target_collection = "labeled_datasets"
        target_data_collection = "labeled_data"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid target_id format"
        )

    target_info = await mongo_client.db[target_collection].find_one({"_id": uid+target_did})
    target_name = target_info.get("name")
    target_path_list = await mongo_client.data_listup_to_download(uid, target_did, target_data_collection)
    logging.info(f"Target paths for compression: {target_path_list}")

    zip_path = await minio_client.compress_dataset_to_download(uid, target_name, basedir,target_path_list)
    logging.info(f"Compressed dataset saved to: {zip_path}")

    return zip_path, target_name  # Return the path of the zip file for download


async def compress_data(uid: str, download_data: DataDownload):
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    
    target_path_list = download_data.target_path_list
    basedir = API_WORKDIR

    zip_path = await minio_client.compress_data_to_download(uid, basedir, target_path_list)

    return zip_path  # Return the path of the zip file for download