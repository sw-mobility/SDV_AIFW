# api/routes/optimizing/__init__.py
from fastapi import APIRouter
from .optimizing_route import router as optimizing_router
from .common_optimizing_route import router as common_optimizing_router

router = APIRouter(prefix="/optimizing", tags=["Optimizing"])
router.include_router(optimizing_router)
router.include_router(common_optimizing_router)
