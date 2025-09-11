from core.minio import (
    MinioStorageClient
)
from core.mongodb import (
    MongoDBClient
)
from core.config import (
    MONGODB_URL, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTIONS,
    MIME_TYPES
)
from fastapi import HTTPException, status

async def init(uid: str):

    mongo_client = MongoDBClient(MONGODB_URL, MONGODB_DB_NAME)
    await mongo_client.init_collections(MONGODB_COLLECTIONS)

    # Initialize MinIO client
    minio_client = MinioStorageClient()
    await minio_client.init_bucket(uid)

    check = await mongo_client.db["users"].find_one({"uid": uid})

    if not check:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown User: {uid}"
        )

    return {"mongo_client": mongo_client, "minio_client": minio_client, "message": "Initialization complete"}