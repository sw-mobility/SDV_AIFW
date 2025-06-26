import os
import shutil
import zipfile
from datetime import datetime
import asyncio
from typing import List, Optional

import aiofiles
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from core.storage import StorageManager
from core.mongodb import MongoDB

router = APIRouter()


@router.post("/api/v1/train/result/upload")
async def upload_training_result(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    params: Optional[str] = Form(None),
):
    """
    학습 결과 압축파일을 수신하여 임시 디렉토리에 저장 후, 압축 해제, MinIO 업로드, MongoDB 저장까지 처리합니다.
    """
    try:
        # 1. 임시 저장 경로
        save_dir = f"/tmp/train_results/{project_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        os.makedirs(save_dir, exist_ok=True)
        zip_path = os.path.join(save_dir, file.filename)
        async with aiofiles.open(zip_path, "wb") as buffer:
            content = await file.read()
            await buffer.write(content)

        # 2. 압축 해제
        extract_dir = os.path.join(save_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # 3. MinIO 업로드
        storage_client = StorageManager.get_instance()
        minio_prefix = f"models/{project_id}/"
        uploaded_files = []
        for root, _, files in os.walk(extract_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, extract_dir)
                minio_key = minio_prefix + rel_path.replace("\\", "/")
                async with aiofiles.open(fpath, "rb") as f:
                    data = await f.read()
                await storage_client.put_object(key=minio_key, body=data)
                uploaded_files.append(minio_key)

        # 4. MongoDB 저장
        await MongoDB.connect_to_mongo()
        db = MongoDB.db
        doc = {
            "project_id": project_id,
            "timestamp": datetime.utcnow().isoformat(),
            "minio_files": uploaded_files,
            "params": params,
        }
        await db["training_results"].insert_one(doc)

        # 5. 응답
        return JSONResponse(
            {
                "status": "success",
                "project_id": project_id,
                "uploaded_files": uploaded_files,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
