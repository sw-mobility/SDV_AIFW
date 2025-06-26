"""API 라우트 패키지"""
from fastapi import APIRouter
from .dataset.raw import router as raw_router
from .dataset.labeled import router as labeled_router
from .training import router as training_router
from .files import router as files_router
from .project import router as project_router
from .labeler.labeler_handler import router as labeler_handle_router
from .labeler.labeler_images import router as labeler_images_router

# 최상위 API 라우터
router = APIRouter()

# 하위 라우터 등록
router.include_router(raw_router)
router.include_router(labeled_router)
router.include_router(training_router)
router.include_router(files_router)
router.include_router(project_router)
router.include_router(labeler_handle_router)
router.include_router(labeler_images_router)

__all__ = [
    'router', 'raw_router', 'labeled_router', 'training_router',
    'files_router', 'project_router', 'labeler_handle_router', 'labeler_images_router'
]