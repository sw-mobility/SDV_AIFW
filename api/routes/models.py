from fastapi import APIRouter, HTTPException, Response, Path, BackgroundTasks
from core.storage import storage_client
from core.mongodb import MongoDB
from fastapi.responses import FileResponse
import tempfile
import os
import io

router = APIRouter(prefix="/models", tags=["Model Artifacts"])

@router.get("/{project_name}/list")
async def list_model_files(project_name: str):
    """
    List all model artifact files for a given project from MinIO.
    """
    prefix = f"models/{project_name}/"
    minio_client = storage_client
    files = []
    try:
        async for page in minio_client.list_objects(prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    # Only list files, not directories
                    key = obj["Key"]
                    if not key.endswith("/"):
                        files.append(key[len(prefix):])
        if not files:
            raise HTTPException(status_code=404, detail="No model artifacts found for this project.")
        return {"project_name": project_name, "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def remove_file(path: str):
    import os
    try:
        os.remove(path)
    except Exception:
        pass

@router.get("/{project_name}/{filename:path}")
async def download_model_file(project_name: str, filename: str = Path(...), background: BackgroundTasks = None):
    """
    Download a specific model artifact file from MinIO for a given project.
    """
    prefix = f"models/{project_name}/"
    minio_key = prefix + filename
    minio_client = storage_client
    import tempfile, os
    try:
        async with await minio_client._get_client() as s3:
            obj = await s3.get_object(Bucket=minio_client.bucket_name, Key=minio_key)
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                while True:
                    chunk = await obj["Body"].read(1024 * 1024)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name
        if background is not None:
            background.add_task(remove_file, tmp_file_path)
        return FileResponse(
            tmp_file_path,
            media_type="application/octet-stream",
            filename=filename.split('/')[-1],
            headers={"Content-Disposition": f"attachment; filename={filename.split('/')[-1]}"},
            background=background
        )
    except Exception as e:
        import logging
        logging.error(f"Download error for {minio_key}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"File not found: {filename} ({str(e)})")
