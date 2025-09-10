from motor.motor_asyncio import ( # type: ignore
    AsyncIOMotorClient, 
    AsyncIOMotorDatabase
)
from pymongo.errors import (
    ConnectionFailure
)
from typing import (
    List, 
    Optional
)
from utils.time import (
    get_current_time_kst
)


class MongoDBClient:
    """MongoDB 클라이언트 구현 (저수준)"""
    def __init__(self, uri: str, db_name: str):
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[db_name]

    @property
    def client(self) -> AsyncIOMotorClient:
        return self._client

    @property
    def db(self) -> AsyncIOMotorDatabase:
        return self._db

    async def test_connection(self):
        try:
            await self._client.admin.command('ping')
        except ConnectionFailure as e:
            raise ConnectionError(f"Failed to connect to MongoDB: {str(e)}")

    async def init_collections(self, collections: Optional[List[str]] = None):
        """
        MONGODB_COLLECTIONS 리스트에 있는 컬렉션들을 생성합니다.
        이미 존재하는 컬렉션은 건너뜁니다.
        """
        if collections is None:
            collections = MONGODB_COLLECTIONS # type: ignore
        existing = await self.db.list_collection_names()
        for name in collections: # type: ignore
            if name not in existing:
                await self.db.create_collection(name)


    async def upload_data(
        self,
        uid: str,
        did: str,
        data_collection: str,
        dataset_collection: str,
        doc: dict
    ):

        if doc:
            # 데이터 정보 저장
            await self.db[data_collection].insert_one(doc)
            # 기존 total 값 읽기
            dataset = await self.db[dataset_collection].find_one({"did": did, "uid": uid})
            prev_count = dataset.get("total", 0) if dataset else 0
            new_count = prev_count + 1
            # 데이터셋의 파일 수 업데이트
            await self.db[dataset_collection].update_one(
                {"did": did, "uid": uid},
                {"$set": {"total": new_count}}
            )

        return doc


    async def delete_dataset(self, 
                             uid: str,
                             target_did_list: List[str]
                             ):
        """
        MongoDB에서 데이터셋을 삭제합니다.
        target_did_list에 있는 id를 가진 데이터셋을 삭제합니다.
        """

        if not target_did_list:
            raise ValueError("target_did_list가 지정되지 않았습니다.")

        result = {"deleted_count": 0, "details": []}

        for did in target_did_list:

            if did[0] == "R":
                data_collection = "raw_data"
                dataset_collection = "raw_datasets"
            elif did[0] == "L":
                data_collection = "labeled_data"
                dataset_collection = "labeled_datasets"
            else:
                raise ValueError(f"Unknown dataset type: {id}")

            deleted1 = await self.db[data_collection].delete_many({"uid": uid, "did": did})
            deleted2 = await self.db[dataset_collection].delete_many({"_id": uid+did})
            result["deleted_count"] += deleted1.deleted_count + deleted2.deleted_count
            result["details"].append({
                "did": did,
                f"{data_collection}_deleted": deleted1.deleted_count,
                f"{dataset_collection}_deleted": deleted2.deleted_count
            })

        # 삭제가 하나도 안 됐으면 예외 발생
        if result["deleted_count"] == 0:
            raise ValueError("삭제된 문서가 없습니다.")

        return result
    

    async def delete_data(self, 
                          uid: str,
                          target_id_list: List[str]
                          ):
        """
        MongoDB에서 데이터 문서를 삭제합니다.
        target_id_list에 있는 id를 가진 데이터 문서를 삭제합니다.
        """

        if not target_id_list:
            raise ValueError("target_id_list가 지정되지 않았습니다.")

        result = {"deleted_count": 0, "details": []}

        for id in target_id_list:
            uid = id[:4]
            did = id[4:9]

            if id[4] == "R":
                data_collection = "raw_data"
                dataset_collection = "raw_datasets"
            elif id[4] == "L":
                data_collection = "labeled_data"
                dataset_collection = "labeled_datasets"
            else:
                continue

            deleted1 = await self.db[data_collection].delete_many({"_id": id})
            deleted2 = await self.db[dataset_collection].update_one(
                {"did": did, "uid": uid},
                {"$inc": {"total": -deleted1.deleted_count}}
            )
            
            result["deleted_count"] += deleted1.deleted_count
            result["details"].append({
                "id": id,
                "deleted": deleted1.deleted_count,
                f"{data_collection}_updated": deleted2.modified_count
            })

        # 삭제가 하나도 안 됐으면 예외 발생
        if result["deleted_count"] == 0:
            raise ValueError("삭제된 문서가 없습니다.")

        return result

    async def data_listup_to_download(
        self,
        uid: str,
        target_did: str,
        target_data_collection: str
        ):

        target_path_list = []

        cursor = self.db[target_data_collection].find(
            {"uid": uid, "did": target_did},
            {"path": 1}
        )

        async for doc in cursor:
            
            if "path" in doc:
                target_path_list.append(doc["path"])

        return target_path_list


    def close(self):
        self._client.close()