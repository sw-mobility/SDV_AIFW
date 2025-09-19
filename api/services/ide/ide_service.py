from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile, 
    File, 
    Path, 
    Query, 
    status, 
    Body, 
    Depends,
    BackgroundTasks
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
    MIME_TYPES,
    API_WORKDIR,
    TRAINING_WORKDIR,
    FRONTEND_WORKDIR,
    YOLO_MODELS
)
from models.ide.ide_model import (
    CodebaseInfo,
    CodebaseCreateRequest,
    CodebaseUpdateRequest
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
from utils.counter import (
    get_next_counter
)
from utils.cleanup import (
    cleanup_workdir
)
from utils.model_id import (
    is_custom_model_id
)
import os
import logging
import yaml
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def prepare_codebase(uid: str, cid: str):

    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    codebase_id = uid+cid

    try:
        # Prepare the codebase for the given user ID and codebase ID
        # Branches for more default codebases will be added later
        if cid == "yolo":
            bucket = "keti-aifw"
            prefix = "codebases/yolo"
        else:
            codebase = await mongo_client.db["codebases"].find_one({"_id": codebase_id})
            bucket = uid
            prefix = codebase["path"]

        keys = []
        files = {}
        tree = []

        async for obj in minio_client.list_objects(bucket, prefix):
            key = obj["Key"]
            rel_key = key[len(prefix):].lstrip("/")
            keys.append(rel_key)
            data = await minio_client.get_object(Bucket=bucket, Key=key)
            files[rel_key] = data.decode("utf-8", errors="ignore")

        logger.info(f"prepare_codebase keys: {keys}")
        logger.info(f"prepare_codebase files: {list(files.keys())}")

        # Build tree structure from keys
        for key in keys:
            parts = key.split('/')
            current = tree
            for i, name in enumerate(parts):
                is_file = (i == len(parts) - 1)
                # 현재 레벨에서 이미 존재하는지 확인
                found = None
                for node in current:
                    if node['name'] == name:
                        found = node
                        break
                if found:
                    if not is_file:
                        current = found.setdefault('children', [])
                    # 파일이면 더 이상 진행하지 않음
                else:
                    node = {'name': name, 'type': 'file' if is_file else 'directory'}
                    if not is_file:
                        node['children'] = []
                        current.append(node)
                        current = node['children']
                    else:
                        current.append(node)
        logger.info(f"prepare_codebase tree: {tree}")
        # 예시: return 또는 저장/응답
        return {"tree": tree, "files": files}

    except Exception as e:
        logger.error(f"Failed to prepare codebase: {e}")
        raise HTTPException(status_code=500, detail="Failed to prepare codebase")


async def create_codebase(uid: str, data: dict, request: CodebaseCreateRequest):

    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    cid = await get_next_counter(mongo_client, "codebases", uid=uid, prefix="C", field="cid", width=4)
    prefix = f"codebases/{request.algorithm}/{cid}"
    
    tree = data.get("tree", [])
    paths = []
    stack = [(tree, "")]  # (현재 노드 리스트, 현재 경로)

    # Convert tree structure to flat list of paths
    while stack:
        nodes, parent_path = stack.pop()
        for node in nodes:
            current_path = os.path.join(parent_path, node["name"]) if parent_path else node["name"]
            if node["type"] == "file":
                paths.append(current_path)
            elif node["type"] == "directory" and "children" in node:
                stack.append((node["children"], current_path))

    # Update MinIO objects with context of matching paths
    async with await minio_client._get_client() as s3:
        for path in paths:
            content = data["files"].get(path)
            # Update the object with the new context
            await s3.put_object(Bucket=uid, Key=prefix+"/"+path, Body=content.encode("utf-8"))
            logger.info(f"Uploaded {path} to MinIO bucket {uid} with prefix {prefix}")

    try:
        codebase_info = CodebaseInfo(
            _id=uid+cid,
            uid=uid,
            cid=cid,
            name=request.name,
            algorithm=request.algorithm,
            stage=request.stage,
            task_type=request.task_type,
            description=request.description,
            path=prefix,
            last_modified=None,  # This will be set later
            created_at=get_current_time_kst()
        )

        await mongo_client.db["codebases"].insert_one(codebase_info.dict(by_alias=True))
    except Exception as e:
        logger.error(f"Failed to create snapshot: {e}")
        raise HTTPException(status_code=500, detail="Failed to create snapshot")


async def update_codebase(uid: str, data: dict, request: CodebaseUpdateRequest):
    """Update the codebase with the provided data."""
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    codebase_id = uid+request.cid
    codebase_info = await mongo_client.db["codebases"].find_one({"_id": codebase_id})
    prefix = codebase_info["path"]

    tree = data.get("tree", [])
    paths = []
    stack = [(tree, "")]  # (현재 노드 리스트, 현재 경로)

    try:
        # Convert tree structure to flat list of paths
        while stack:
            nodes, parent_path = stack.pop()
            for node in nodes:
                current_path = os.path.join(parent_path, node["name"]) if parent_path else node["name"]
                if node["type"] == "file":
                    paths.append(current_path)
                elif node["type"] == "directory" and "children" in node:
                    stack.append((node["children"], current_path))

        # Update MinIO objects with context of matching paths
        async with await minio_client._get_client() as s3:
            for path in paths:
                content = data["files"].get(path)
                # Update the object with the new context
                await s3.put_object(Bucket=uid, Key=prefix+"/"+path, Body=content.encode("utf-8"))
                logger.info(f"Uploaded {path} to MinIO bucket {uid} with prefix {prefix}")

        doc = {
            "name": codebase_info["name"] if request.name is None else request.name,
            "algorithm": codebase_info["algorithm"] if request.algorithm is None else request.algorithm,
            "stage": codebase_info["stage"] if request.stage is None else request.stage,
            "task_type": codebase_info["task_type"] if request.task_type is None else request.task_type,
            "description": codebase_info["description"] if request.description is None else request.description,
            "last_modified": get_current_time_kst()
        }

        # Update the codebase in MongoDB
        await mongo_client.db["codebases"].update_one(
            {"_id": codebase_id},
            {"$set": doc}
        )
        return {"status": "success", "message": "Codebase updated successfully"}
    
    except Exception as e:
        logger.error(f"Failed to update codebase: {e}")
        raise HTTPException(status_code=500, detail="Failed to update codebase")