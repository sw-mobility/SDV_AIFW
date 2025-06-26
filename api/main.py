"""FastAPI Main Application"""
from contextlib import asynccontextmanager
from datetime import datetime
import json

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder

from config.settings import DateTimeEncoder
from core.storage import storage_client
from core.mongodb import MongoDB, initialize_db
from routes import router as api_router
from routes.global_delete import router as global_delete_router
from routes.global_download import router as global_download_router
from routes.training.result_upload import router as result_upload_router
from routes.models import router as models_router
from utils.logging import logger

# 서버 시작/종료 시 실행될 로직 정의
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting server...")
    try:
        # MinIO 초기화
        await storage_client.init_bucket()
        logger.info("Storage initialized successfully")        # MongoDB 연결 및 전역 db 객체 초기화
        await initialize_db()
        logger.info("MongoDB connected successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise
    
    yield  # 서버 실행 중
    
    # 서버 종료 시 리소스 정리
    logger.info("Shutting down server...")
    await MongoDB.close_mongo_connection()
    logger.info("MongoDB connection closed")

# FastAPI 애플리케이션 인스턴스 생성 (lifespan 적용)
app = FastAPI(lifespan=lifespan)

# 커스텀 JSON 인코더 설정
def custom_json_serializer(obj):
    """커스텀 JSON 직렬화 함수"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj

# 기본 JSON 인코더 설정
json._default_encoder = DateTimeEncoder()

# 라우터 등록
app.include_router(api_router)
app.include_router(global_delete_router)
app.include_router(global_download_router)
app.include_router(result_upload_router)
app.include_router(models_router)

# API 문서화 설정
app.title = "Dataset Management API"
app.description = "API for managing raw and labeled image datasets"
app.version = "1.1.0"