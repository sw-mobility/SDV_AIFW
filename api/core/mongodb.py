from config.settings import MONGODB_URL, MONGODB_DB_NAME
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure

"""MongoDB 클라이언트 설정"""

class NotInitializedError(Exception):
    """MongoDB 연결이 초기화되지 않았을 때 발생하는 예외"""
    def __init__(self, message="MongoDB is not initialized. Call connect_to_mongo() first."):
        self.message = message
        super().__init__(self.message)

class MongoDB:
    """MongoDB 연결 관리자"""
    _client: AsyncIOMotorClient = None
    _db: AsyncIOMotorDatabase = None
    _initialized: bool = False

    @classmethod
    async def connect_to_mongo(cls):
        """MongoDB에 연결합니다."""
        try:
            if not cls._initialized:
                cls._client = AsyncIOMotorClient(MONGODB_URL)
                cls._db = cls._client[MONGODB_DB_NAME]
                await cls._client.admin.command('ping')  # 연결 테스트
                cls._initialized = True
                return cls._client
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

    @classmethod
    async def close_mongo_connection(cls):
        """MongoDB 연결을 종료합니다."""
        if cls._client:
            cls._client.close()
            cls._client = None
            cls._db = None
            cls._initialized = False    @classmethod
    @property
    def client(cls) -> AsyncIOMotorClient:
        """MongoDB 클라이언트를 반환합니다."""
        if not cls._initialized:
            raise NotInitializedError()
        return cls._client
        
    @classmethod
    @property
    def db(cls) -> AsyncIOMotorDatabase:
        """MongoDB 데이터베이스를 반환합니다."""
        if not cls._initialized:
            raise NotInitializedError()
        return cls._db

# 직접 db 객체 참조를 위한 전역 변수
# 애플리케이션 시작 시 MongoDB.connect_to_mongo() 이후 초기화됨
db = None

# MongoDB 연결 시 db 변수 초기화를 위한 함수
async def initialize_db():
    """전역 db 객체를 초기화합니다."""
    global db
    await MongoDB.connect_to_mongo()
    db = MongoDB.db

def get_dataset_collection(dataset_type: str, model_type: str = None, task_type: str = None):
    """
    데이터 타입/모델/태스크별로 컬렉션을 분리해서 반환
    """
    if dataset_type == "raw_images":
        return MongoDB.db["raw_images_datasets"]
    elif dataset_type == "labeled_images" and model_type == "YOLOv8-det":
        return MongoDB.db["YOLOv8_ObjectDetection_datasets"]
    # flat 구조: object_detection task는 datasets 컬렉션 사용
    elif dataset_type == "labeled_images" and (task_type == "object_detection" or task_type is not None):
        return MongoDB.db["datasets"]
    else:
        return MongoDB.db["datasets"]
