"""원본 데이터셋 라우트"""
from fastapi import APIRouter
from .images.routes import router as images_router

# 모든 원본 데이터 API에 대한 통합 라우터
router = APIRouter(prefix="/raw")

# 이미지 라우터 포함
router.include_router(images_router)

__all__ = ['router', 'images_router']