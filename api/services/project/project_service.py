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
from models.project.project_model import (
    ProjectCreate,
    ProjectUpdate,
    ProjectInfo
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


async def create_project(
    uid: str,
    project: ProjectCreate
):
    
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    pid = await get_next_counter(mongo_client, "projects", uid=uid, prefix="P", field="pid", width=4)

    project_info = ProjectInfo(
        _id = uid + pid,
        uid = uid,
        pid = pid,
        name = project.name,
        description = project.description,
        status = "Active",
        created_at = get_current_time_kst()
    )
    
    await mongo_client.db["projects"].insert_one(project_info.dict(by_alias=True))

    return project_info


async def update_project(
        uid: str,
        update: ProjectUpdate
    ):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    id = uid + update.pid  # Combine uid and pid to form the full project ID
    project = await mongo_client.db["projects"].find_one({"_id": id})

    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project not found: {id}"
        )

    update_data = update.dict(exclude_unset=True)
    await mongo_client.db["projects"].update_one(
        {"_id": id},
        {"$set": update_data}
    )

    updated_project = await mongo_client.db["projects"].find_one({"_id": id})

    return ProjectInfo(**updated_project)  # Pydantic 모델로 변환하여 반환


async def list_projects(uid: str) -> List[ProjectInfo]:

    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    projects = await mongo_client.db["projects"].find({"uid": uid}).to_list(length=None)
    return [ProjectInfo(**project) for project in projects]


async def delete_project(
        uid: str,
        pid: str
    ):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    project_id = uid+pid   # Extract pid from project_id

    project = await mongo_client.db["projects"].find_one({"_id": project_id})
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

    try:

        trn_hst = await mongo_client.db["trn_hst"].find({"uid": uid, "pid": pid}).to_list(length=None)
        opt_hst = await mongo_client.db["opt_hst"].find({"uid": uid, "pid": pid}).to_list(length=None)
        val_hst = await mongo_client.db["val_hst"].find({"uid": uid, "pid": pid}).to_list(length=None)
        p_datasets_list = await mongo_client.db["labeled_datasets"].find({"uid": uid, "pid": pid}).to_list(length=None)
        logger.info(f"{p_datasets_list}")

        trn_artifact_path_list = []
        opt_artifact_path_list = []
        val_artifact_path_list = []
        p_datasets_path_list = []
        p_datasets_did_list = []

        if trn_hst is not None:
            trn_artifact_path_list = [doc["artifacts_path"] for doc in trn_hst if "artifacts_path" in doc]
        if opt_hst is not None:
            opt_artifact_path_list = [doc["artifacts_path"] for doc in opt_hst if "artifacts_path" in doc]
        if val_hst is not None:
            val_artifact_path_list = [doc["artifacts_path"] for doc in val_hst if "artifacts_path" in doc]
        if p_datasets_list is not None:
            p_datasets_did_list = [doc["did"] for doc in p_datasets_list if "did" in doc]
            p_datasets_path_list = [doc["path"] for doc in p_datasets_list if "path" in doc]
        logger.info(f"{p_datasets_did_list}")

        all_paths = trn_artifact_path_list + opt_artifact_path_list + val_artifact_path_list + p_datasets_path_list
        logger.info(f"All paths to delete: {all_paths}")

        await mongo_client.db["projects"].delete_one({"_id": project_id})
        await mongo_client.db["trn_hst"].delete_many({"uid": uid, "pid": pid})
        await mongo_client.db["opt_hst"].delete_many({"uid": uid, "pid": pid})
        await mongo_client.db["val_hst"].delete_many({"uid": uid, "pid": pid})

        if p_datasets_did_list:
            await mongo_client.delete_dataset(uid, p_datasets_did_list)

        if all_paths:
            await minio_client.delete_dataset(uid, all_paths)

        return {"message": "Project deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete project")