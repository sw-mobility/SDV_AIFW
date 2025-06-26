# take dataset from minio and info from frontend to labeler container
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.storage import storage_client  # Use the async Minio client
import os
import tempfile
import requests
import aiofiles
import asyncio
from typing import Optional

router = APIRouter(prefix="/labeler/images", tags=["Labeler Images"])

@router.post("/send-to-labeler")
async def send_to_labeler(
    dataset_name: str = Form(...),
    labeler_url: str = Form("http://labeler:8001/images/label"),
    model: str = Form("yolov8n"),
    result_name: str = Form(None)
):
    """
    MinIO에서 이미지들을 zip으로 묶어 labeler 컨테이너로 전송
    """
    import io
    import zipfile
    import asyncio
    import json

    prefix = f"datasets/raw/images/{dataset_name}/"
    # 1. 이미지 파일 키 목록 수집 (project.py 방식)
    async def list_objects_recursive(path):
        result = []
        try:
            async with await storage_client._get_client() as s3:
                paginator = s3.get_paginator('list_objects_v2')
                async for page in paginator.paginate(Bucket=storage_client.bucket_name, Prefix=path):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            key = obj["Key"]
                            if key.lower().endswith((".jpg", ".jpeg", ".png")):
                                result.append(key)
            return result
        except Exception as e:
            print(f"Error listing objects: {str(e)}")
            return []

    image_files = await list_objects_recursive(prefix)
    if not image_files:
        raise HTTPException(status_code=404, detail="No images found in MinIO for this dataset.")

    # 2. zip 파일 메모리 생성 및 이미지 추가 (재시도 포함)
    zip_buffer = io.BytesIO()
    async with await storage_client._get_client() as s3:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in image_files:
                relative_path = file_path.replace(f"{prefix}", "")
                if not relative_path or relative_path.endswith('/'):
                    continue
                for attempt in range(3):
                    try:
                        obj = await s3.get_object(Bucket=storage_client.bucket_name, Key=file_path)
                        async with obj["Body"] as stream:
                            data = await stream.read()
                            zip_file.writestr(relative_path, data)
                        await asyncio.sleep(0.2)
                        break
                    except Exception as e:
                        if attempt < 2:
                            await asyncio.sleep(0.5)
                            continue
                        print(f"Error adding file to ZIP: {file_path} - {str(e)}")
    zip_buffer.seek(0)

    # 3. labeler로 multipart/form-data 전송
    data = {
        'model': model,
        'dataset_name': dataset_name,
        'result_name': result_name,
        'do_split': 'false',
        'train_percent': '100'
    }
    if result_name:
        data['result_name'] = result_name
    if not result_name or not str(result_name).strip():
        raise HTTPException(status_code=400, detail="result_name must not be empty or whitespace.")

    files = {
        "files": ("dataset.zip", zip_buffer.getvalue(), "application/zip"),
    }
    try:
        response = requests.post(labeler_url, files=files, data=data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error sending files to labeler: {str(e)}")
    if not response.ok:
        raise HTTPException(status_code=500, detail=f"Labeler error: {response.text}")
    return response.json()