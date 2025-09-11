from fastapi import APIRouter
from .yolo_labeling_route import router as labeling_router

router = APIRouter(prefix="/labeling", tags=["Labeling"])
router.include_router(labeling_router)