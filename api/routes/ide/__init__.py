from fastapi import APIRouter
from .ide_route import router as ide_route

router = APIRouter(prefix="/IDE", tags=["Code Editing"])
router.include_router(ide_route)