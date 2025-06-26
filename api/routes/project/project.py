from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Any
from models.project.training_project import TrainingProject
from core.mongodb import MongoDB
from core.storage import storage_client
import io
import zipfile
import httpx
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/project", tags=["Project"])

COLLECTION_NAME = "projects"

@router.post("/", response_model=TrainingProject, status_code=201)
async def create_project(project: TrainingProject = Body(...)):
    await MongoDB.connect_to_mongo()
    collection = MongoDB.db[COLLECTION_NAME]
    existing = await collection.find_one({"name": project.name})
    if existing:
        raise HTTPException(status_code=409, detail="Project with this name already exists.")
    if not getattr(project, "task_configs", None):
        project.task_configs = {"training": {}}
    await collection.insert_one(project.dict())
    return project

@router.post("/{project_name}/optimize", response_model=Dict[str, Any])
async def run_optimization(project_name: str, params: Dict[str, Any] = Body(...)):
    await MongoDB.connect_to_mongo()
    collection = MongoDB.db[COLLECTION_NAME]
    doc = await collection.find_one({"name": project_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    doc.setdefault("task_configs", {})
    doc["task_configs"].setdefault("optimization_runs", []).append(params)
    await collection.update_one({"name": project_name}, {"$set": {"task_configs": doc["task_configs"]}})
    return {"status": "optimization started (mock)", "params": params}

@router.post("/{project_name}/label", response_model=Dict[str, Any])
async def run_labeling(project_name: str, params: Dict[str, Any] = Body(...)):
    await MongoDB.connect_to_mongo()
    collection = MongoDB.db[COLLECTION_NAME]
    doc = await collection.find_one({"name": project_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    doc.setdefault("task_configs", {})
    doc["task_configs"].setdefault("labeling_runs", []).append(params)
    await collection.update_one({"name": project_name}, {"$set": {"task_configs": doc["task_configs"]}})
    return {"status": "labeling started (mock)", "params": params}

@router.get("/{project_name}", response_model=TrainingProject)
async def get_project(project_name: str):
    await MongoDB.connect_to_mongo()
    collection = MongoDB.db[COLLECTION_NAME]
    doc = await collection.find_one({"name": project_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    return TrainingProject(**doc)

@router.get("/", response_model=List[TrainingProject])
async def list_projects():
    await MongoDB.connect_to_mongo()
    collection = MongoDB.db[COLLECTION_NAME]
    projects = []
    async for doc in collection.find():
        projects.append(TrainingProject(**doc))
    return projects

@router.post("/train/{project_name}/execute", response_model=Dict[str, Any])
async def export_and_send(project_name: str, params: Dict[str, Any] = Body(...)):
    # 1. 프로젝트 조회 및 dataset_path 획득
    await MongoDB.connect_to_mongo()
    collection = MongoDB.db[COLLECTION_NAME]
    doc = await collection.find_one({"name": project_name})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found.")
    dataset_path = doc.get("dataset_path")
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required in project dict.")

    # MinIO에서 dataset_path 하위 모든 파일을 zip으로 묶기
    import json
    async def list_objects_recursive(path):
        result = []
        try:
            async with await storage_client._get_client() as s3:
                paginator = s3.get_paginator('list_objects_v2')
                async for page in paginator.paginate(Bucket=storage_client.bucket_name, Prefix=path):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            result.append(obj["Key"])
            return result
        except Exception as e:
            print(f"Error listing objects: {str(e)}")
            return []

    all_files = await list_objects_recursive(dataset_path)

    # zip 파일 메모리 생성
    import io
    import zipfile
    import asyncio
    zip_buffer = io.BytesIO()
    async with await storage_client._get_client() as s3:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in all_files:
                relative_path = file_path.replace(f"{dataset_path}/", "")
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

    # project info json 생성
    project_info = {k: v for k, v in doc.items() if k != '_id'}
    project_json = json.dumps(project_info, default=str, ensure_ascii=False)

    # training 컨테이너로 multipart/form-data 전송 (zip + json)
    if doc.get("algorithm") == "yolo":
        training_url = "http://training:5003/train/yolo"   
    else:
        training_url = "http://training:5003/train/yolo" # Room for other algorithms, e.g., "faster_rcnn"
    files = {
        "dataset_zip": ("dataset.zip", zip_buffer.getvalue(), "application/zip"),
        "project_info": (None, project_json, "application/json"),
    }
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(
            training_url,
            files=files
            # 쿼리 파라미터 사용 금지: params=params 제거
        )
        result = response.json()
        return {"status": "sent", "result": result, "dataset_path": dataset_path}

# @router.get("/{project_name}/dataset/zip")
# async def download_dataset_zip(project_name: str):
#     """
#     해당 프로젝트의 dataset_path에 해당하는 데이터셋을 zip 파일로 만들어 스트리밍 다운로드합니다.
#     """
#     import asyncio
#     await MongoDB.connect_to_mongo()
#     collection = MongoDB.db[COLLECTION_NAME]
#     doc = await collection.find_one({"name": project_name})
#     if not doc:
#         raise HTTPException(status_code=404, detail="Project not found.")
#     dataset_path = doc.get("dataset_path")
#     if not dataset_path:
#         raise HTTPException(status_code=400, detail="dataset_path is required in project dict.")
#     import io, zipfile
#     async def list_objects_recursive(path):
#         result = []
#         try:
#             async with await storage_client._get_client() as s3:
#                 paginator = s3.get_paginator('list_objects_v2')
#                 async for page in paginator.paginate(Bucket=storage_client.bucket_name, Prefix=path):
#                     if "Contents" in page:
#                         for obj in page["Contents"]:
#                             result.append(obj["Key"])
#             return result
#         except Exception as e:
#             print(f"Error listing objects: {str(e)}")
#             return []
#     all_files = await list_objects_recursive(dataset_path)
#     if not all_files:
#         raise HTTPException(status_code=404, detail="No files found for this dataset_path.")
#     zip_buffer = io.BytesIO()
#     with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
#         for file_path in all_files:
#             # Skip directory objects (S3/MinIO convention: endswith '/' or empty relative path)
#             relative_path = file_path.replace(f"{dataset_path}/", "")
#             if not relative_path or relative_path.endswith('/'):
#                 continue
#             for attempt in range(3):
#                 try:
#                     obj = await storage_client.get_object(file_path)
#                     async with obj["Body"] as stream:
#                         data = await stream.read()
#                         zip_file.writestr(relative_path, data)
#                     await asyncio.sleep(0.2)  # 더 긴 sleep
#                     break  # 성공 시 루프 탈출
#                 except Exception as e:
#                     if attempt < 2:
#                         await asyncio.sleep(0.5)  # 실패 시 더 길게 대기 후 재시도
#                         continue
#                     print(f"Error adding file to ZIP: {file_path} - {str(e)}")
#     zip_buffer.seek(0)
#     return StreamingResponse(zip_buffer, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=dataset.zip"})
