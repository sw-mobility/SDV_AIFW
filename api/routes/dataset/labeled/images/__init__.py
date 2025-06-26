"""Labeled Images 라우터 패키지 (object_detection 기준)"""
from fastapi import APIRouter
from .object_detection import routes as object_detection_routes

router = APIRouter(prefix="/images")

# Object Detection 라우터 포함
router.include_router(object_detection_routes.router)

__all__ = ['router']
