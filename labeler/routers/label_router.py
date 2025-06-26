from fastapi import APIRouter, UploadFile, File, Form, Request
from typing import List, Optional
from services.label_service import label_images_service, get_script_service, save_script_service

router = APIRouter()

@router.get("/api/script")
async def get_script(path: str):
    return await get_script_service(path)

@router.post("/api/script")
async def save_script(request: Request):
    return await save_script_service(request)

@router.post("/images/label")
async def label_images(
    files: List[UploadFile] = File(...),
    model: str = Form("yolov8n"),
    dataset_name: str = Form("dataset"),
    do_split: bool = Form(False),
    train_percent: float = Form(0.8),
    result_name: Optional[str] = Form(None)
):
    return await label_images_service(
        files=files,
        model=model,
        dataset_name=dataset_name,
        do_split=do_split,
        train_percent=train_percent,
        result_name=result_name
    )