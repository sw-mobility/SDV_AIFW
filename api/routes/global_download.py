from fastapi import APIRouter, Query, HTTPException, Response
from fastapi.responses import StreamingResponse
from typing import List, Optional
import io
import zipfile
from core.storage import storage_client

router = APIRouter(prefix="/global_download", tags=["Global Download"])

@router.get("/files/zip")
async def download_files_zip(
    bucket: str = Query(..., description="MinIO bucket name (e.g. datasets/labeled/images/object_detection)"),
    prefix: str = Query(..., description="Prefix/folder path inside the bucket (e.g. dataset_name/images/)"),
    filenames: Optional[List[str]] = Query(None, description="List of filenames to include. If omitted, all files under prefix are included."),
):
    """
    Download multiple files or a folder as a zip from MinIO.
    """
    minio_client = storage_client
    try:
        # List all objects under the prefix (async generator)
        file_objs = []
        async for page in minio_client.list_objects(prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    if filenames and obj["Key"].split("/")[-1] not in filenames:
                        continue
                    file_objs.append(obj)
        if not file_objs:
            raise HTTPException(status_code=404, detail="No files found to download.")
        # Create zip in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
            for obj in file_objs:
                data = await minio_client.get_object(obj["Key"])
                arcname = obj["Key"][len(prefix):] if obj["Key"].startswith(prefix) else obj["Key"]
                zipf.writestr(arcname, data)
        zip_buffer.seek(0)
        return StreamingResponse(zip_buffer, media_type="application/x-zip-compressed", headers={"Content-Disposition": f"attachment; filename=download.zip"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
