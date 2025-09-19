from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from core.minio import MinioStorageClient
from core.mongodb import MongoDBClient
from core.config import MONGODB_URL, MONGODB_DB_NAME, MONGODB_COLLECTIONS
from routes.dataset import router as dataset_router
from routes.project import router as project_router
from routes.training import router as training_router
from routes.validation import router as validation_router
from routes.optimizing import router as optimizing_router
from routes.labeling import router as labeling_router
from routes.ide import router as ide_router
import logging
import asyncio
from utils.auth import get_uid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(dataset_router)
app.include_router(project_router)
app.include_router(ide_router)
app.include_router(labeling_router)
app.include_router(training_router)
app.include_router(validation_router)
app.include_router(optimizing_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

'''
라우트를 통해 초기화하는 방식은 테스트용으로만 사용해야 함
uid는 헤더에서 자동으로 읽히도록 변경해야 하며
시스템의 모든 uid 관련 함수들도 똑같이 수정해야 함
'''

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    logger.info("Waiting for DBs...")
    await asyncio.sleep(5)  # Simulate some startup delay
    logger.info("Initializing Default Assets...")
    for i in range(5):
        try:
            await MinioStorageClient().init_core_bucket()
            logger.info("MinIO core bucket initialized successfully.")
        except Exception as e:
            logger.error(f"MinIO initialization error (try {i+1}): {str(e)}")
            await asyncio.sleep(5)

        try:
            await MinioStorageClient().core_default_assets_init()
            logger.info("Default assets initialized successfully.")
            break
        except Exception as e:
            logger.error(f"Default assets initialization error (try {i+1}): {str(e)}")
            await asyncio.sleep(5)

    else:
        logger.error("MinIO initialization failed after 5 attempts.")

@app.post("/")
async def init(uid: str = Depends(get_uid)):
    try:
        await MinioStorageClient().init_bucket(uid)
        
    except Exception as e:
        logger.error(f"MinIO init error: {str(e)}")
        return {"error": f"MinIO init error: {str(e)}", "uid": uid}
    
    try:
        mongo_client = MongoDBClient(MONGODB_URL, MONGODB_DB_NAME)
        await mongo_client.init_collections(MONGODB_COLLECTIONS)
        existing = await mongo_client.db["users"].find_one({"uid": uid})

        if not existing:
            result = await mongo_client.db["users"].insert_one({"uid": uid})
            logger.info(f"UID added: {result.inserted_id}")
        else:
            logger.info("UID already exists, skipping insert.")

        return {"message": f"DB initialized"}
    
    except Exception as e:
        logger.error(f"MongoDB init error: {str(e)}")
        return {"error": f"MongoDB init error: {str(e)}", "uid": uid}

