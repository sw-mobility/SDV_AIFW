from fastapi import APIRouter
from .yolo_validation_route import router as yolo_router
from .common_validation_route import router as common_router

router = APIRouter(prefix="/validation", tags=["Validation"])
router.include_router(yolo_router)
router.include_router(common_router)



