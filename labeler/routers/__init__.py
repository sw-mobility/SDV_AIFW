from fastapi import APIRouter
from .yolo_route import router as yolo_router

router = APIRouter(prefix="/label", tags=["Labeling"])

router.include_router(yolo_router)