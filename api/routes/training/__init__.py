from fastapi import APIRouter
from .yolo_training_route import router as training_router
from .common_training_route import router as common_router

router = APIRouter(prefix="/training", tags=["Training"])
router.include_router(training_router)
router.include_router(common_router)