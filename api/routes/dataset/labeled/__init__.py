"""Labeled 데이터셋 라우트 (object_detection 구조)"""
from fastapi import APIRouter
from .images import router as images_router

# 모든 레이블링 데이터 API에 대한 통합 라우터
router = APIRouter(prefix="/labeled")

# 이미지 라우터 포함 (구조화된 이미지 관련 모든 라우트를 포함)
router.include_router(images_router)

__all__ = ['router']
