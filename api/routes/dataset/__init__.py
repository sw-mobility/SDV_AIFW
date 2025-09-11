from fastapi import APIRouter
from .raw_route import router as raw_dataset_router
from .labeled_route import router as labeled_dataset_router
from .common_route import router as common_dataset_router

router = APIRouter(prefix="/datasets", tags=["Dataset Management"])

router.include_router(raw_dataset_router)
router.include_router(labeled_dataset_router)
router.include_router(common_dataset_router)