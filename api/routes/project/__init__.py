from fastapi import APIRouter
from .project_route import router as project_router

router = APIRouter(prefix="/projects", tags=["Project Management"])

router.include_router(project_router)