from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile,
    Response,
    File, 
    Path, 
    Query, 
    status, 
    Body, 
    Depends,
    Form,
    BackgroundTasks
)
from fastapi.responses import JSONResponse
from typing import (
    List, 
    Optional
)
from core.config import (
    MONGODB_URL, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTIONS,
    API_WORKDIR,
    TRAINING_WORKDIR
)
from services.ide.ide_service import (
    prepare_codebase,
    create_codebase,
    update_codebase
)
from models.ide.ide_model import (
    CodebaseInfo,
    CodebaseCreateRequest,
    CodebaseUpdateRequest
)
import logging
import requests
import httpx
from utils.init import init
from utils.cleanup import cleanup_workdir
from utils.auth import get_uid

logger = logging.getLogger(__name__)

router = APIRouter()


# 1. 코드베이스 로드
@router.get("/codebase", status_code=status.HTTP_200_OK)
async def get_codebase(cid: str, uid: str = Depends(get_uid)):
    """Get codebase to edit."""
    try:
        # monaco에 수정 가능한 코드 파싱해서 전송
        json = await prepare_codebase(uid, cid)
        return JSONResponse(content=json, status_code=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Failed to get codebase: {e}")
        raise HTTPException(status_code=500, detail="Failed to get codebase")


# 2. 코드베이스 생성
@router.post("/codebase/create", status_code=status.HTTP_200_OK)
async def create_cb(data: dict, request: CodebaseCreateRequest = Body(...), uid: str = Depends(get_uid)):
    """Create a snapshot of the current training state."""
    try:
        # monaco에서 받은 정보를 minio 안의 codebase에 dump
        result = await create_codebase(uid, data, request)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Failed to create codebase: {e}")
        raise HTTPException(status_code=500, detail="Failed to create codebase")
    

# 3. 코드베이스 업데이트
@router.post("/codebase/update", status_code=status.HTTP_200_OK)
async def update_cb(data: dict, request: CodebaseUpdateRequest = Body(...), uid: str = Depends(get_uid)):
    """Update a snapshot of the current training state."""
    try:
        # monaco에서 받은 정보를 minio 안의 codebase에 dump
        result = await update_codebase(uid, data, request)
        return JSONResponse(content=result, status_code=status.HTTP_200_OK)
    
    except Exception as e:
        logger.error(f"Failed to update codebase: {e}")
        raise HTTPException(status_code=500, detail="Failed to update codebase")
    

# 4. 코드베이스 리스트업
@router.get("/codebases", status_code=status.HTTP_200_OK)
async def list_codebases(uid: str = Depends(get_uid)):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    histories = await mongo_client.db["codebases"].find({"uid": uid}).to_list(length=None)
    for history in histories:
        history.pop("_id", None)
        history.pop("uid", None)

    return histories

# 5. 코드베이스 삭제
@router.delete("/codebase", status_code=status.HTTP_200_OK)
async def delete_codebase(cid: str, uid: str = Depends(get_uid)):
    """Delete a codebase."""
    try:
        init_result = await init(uid)
        mongo_client = init_result["mongo_client"]
        minio_client = init_result["minio_client"]

        path = []

        # Check if the codebase exists
        codebase = await mongo_client.db["codebases"].find_one({"uid": uid, "cid": cid})
        cbpath = codebase["path"] if codebase else None
        path.append(cbpath)
        if not codebase:
            raise HTTPException(status_code=404, detail="Codebase not found")

        # Delete the codebase entry from MongoDB
        await mongo_client.db["codebases"].delete_one({"uid": uid, "cid": cid})

        # Delete associated files from storage (e.g., MinIO)
        await minio_client.delete_dataset(uid, path)
        # This part depends on how files are stored and should be implemented accordingly.

        return JSONResponse(content={"detail": "Codebase deleted successfully"}, status_code=status.HTTP_200_OK)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to delete codebase: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete codebase")