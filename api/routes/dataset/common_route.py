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
    Response,
    BackgroundTasks
)
from fastapi.responses import (
    JSONResponse,
    FileResponse
)
from models.dataset.common_model import (
    DatasetDelete,
    DataDelete,
    DatasetDownload,
    DataDownload
)
from services.dataset.common_service import (
    delete_dataset,
    delete_data,
    compress_dataset,
    compress_data
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
from utils.counter import (
    get_next_counter
)
from utils.time import (
    get_current_time_kst
)
from utils.init import (
    init
)
from utils.cleanup import cleanup_zip
from utils.auth import get_uid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Common Functions"])


# 1. 데이터셋 삭제 (DELETE /common)
@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
async def route_delete_dataset(dataset_delete: DatasetDelete, uid: str = Depends(get_uid)):
    await init(uid)
    try:
        await delete_dataset(uid, dataset_delete)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        logger.error("Error deleting dataset: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 2. 데이터 삭제 (DELETE /common/data)
@router.delete("/data", status_code=status.HTTP_204_NO_CONTENT)
async def route_delete_data(data_delete: DataDelete, uid: str = Depends(get_uid)):
    await init(uid)
    try:
        await delete_data(uid, data_delete)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except Exception as e:
        logger.error("Error deleting data: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 3. 데이터셋 다운로드 (POST /common/download-dataset)
@router.post("/download-dataset", status_code=status.HTTP_200_OK)
async def route_download_dataset(download_dataset: DatasetDownload, background_tasks: BackgroundTasks, uid: str = Depends(get_uid)):
    await init(uid)
    try:
        zip_path, target_name = await compress_dataset(uid, download_dataset)
        folder_path = zip_path.replace(".zip", "")

        background_tasks.add_task(
            cleanup_zip, 
            zip_path, 
            folder_path
        )

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename=target_name
        )
    except Exception as e:
        logger.error("Error downloading dataset: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 4. 데이터 다운로드 (GET /common/download-data)
@router.post("/download-data", status_code=status.HTTP_200_OK)
async def route_download_data(data_download: DataDownload, background_tasks: BackgroundTasks, uid: str = Depends(get_uid)):
    await init(uid)
    try:
        zip_path = await compress_data(uid, data_download)
        folder_path = zip_path.replace(".zip", "")

        background_tasks.add_task(
            cleanup_zip,
            zip_path,
            folder_path
        )

        return FileResponse(
            path=zip_path,
            media_type="application/zip",
            filename="archive.zip"
        )
    
    except Exception as e:
        logger.error("Error downloading data: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))