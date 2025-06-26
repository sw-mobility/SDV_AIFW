"""데이터셋 MongoDB 관리자"""
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from core.mongodb import MongoDB, NotInitializedError, get_dataset_collection
from models.dataset.raw.images.mongodb import Dataset, safe_mongo_convert, convert_mongo_document
from utils.logging import logger

class DatasetMongoManager:
    """데이터셋 MongoDB 관리자"""
    
    @property
    def collection(self):
        """데이터셋 컬렉션을 가져옵니다."""
        try:
            return get_dataset_collection(dataset_type="raw_images")
        except NotInitializedError as e:
            logger.error("Attempting to access MongoDB before initialization")
            raise

    async def create_dataset(self, dataset: Dataset) -> Dataset:
        """새로운 데이터셋을 생성합니다."""
        try:
            # 이미 존재하는 데이터셋인지 확인
            existing = await self.collection.find_one({"name": dataset.name})
            if existing:
                raise ValueError(f"Dataset '{dataset.name}' already exists")

            # 모델을 딕셔너리로 변환
            data_dict = dataset.model_dump(by_alias=True)
            
            # id가 None이면 _id 필드 제거 (MongoDB가 자동 생성하도록)
            if "_id" in data_dict and data_dict["_id"] is None:
                del data_dict["_id"]

            result = await self.collection.insert_one(data_dict)
            dataset.id = str(result.inserted_id)
            
            logger.info(f"Created dataset in MongoDB: {dataset.name}")
            return dataset
            
        except NotInitializedError:
            raise
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            raise    
        
    async def get_dataset(self, name: str) -> Optional[Dataset]:
        """데이터셋을 조회합니다."""
        try:
            doc = await self.collection.find_one({"name": name})
            if not doc:
                return None

            # MongoDB 문서를 모델로 변환
            try:
                # 안전한 변환 함수 사용
                return safe_mongo_convert(Dataset, doc)
            except Exception as e:
                logger.error(f"Failed to convert MongoDB document to Dataset model: {str(e)}")
                # 디버깅을 위한 추가 정보 로깅
                logger.error(f"Document: {doc}")
                raise
            
        except NotInitializedError:
            raise
        except Exception as e:
            logger.error(f"Failed to get dataset: {str(e)}")
            raise    
        
    async def list_datasets(self) -> List[Dataset]:
        """모든 데이터셋 목록을 조회합니다."""
        try:
            datasets = []
            async for doc in self.collection.find():
                try:
                    # 안전한 변환 함수 사용
                    dataset = safe_mongo_convert(Dataset, doc)
                    if dataset:
                        datasets.append(dataset)
                except Exception as conversion_error:
                    logger.error(f"Failed to convert dataset document: {str(conversion_error)}")
                    logger.error(f"Document: {doc}")
                    # 변환 실패한 문서는 건너뜁니다

            return datasets
            
        except NotInitializedError:
            raise
        except Exception as e:
            logger.error(f"Failed to list datasets: {str(e)}")
            raise

    async def delete_dataset(self, name: str) -> bool:
        """데이터셋을 삭제합니다."""
        try:
            # 데이터셋 삭제
            result = await self.collection.delete_one({"name": name})
            if result.deleted_count == 0:
                return False

            # 관련된 파일 정보도 삭제
            await MongoDB.db.files.delete_many({"dataset_name": name})

            logger.info(f"Deleted dataset from MongoDB: {name}")
            return True
            
        except NotInitializedError:
            raise
        except Exception as e:
            logger.error(f"Failed to delete dataset: {str(e)}")
            raise

    async def update_counts(self, name: str, files: int = None):
        """데이터셋의 파일 수를 업데이트합니다."""
        try:
            update = {}
            if files is not None:
                update["file_count"] = files
            
            if update:
                update["updated_at"] = datetime.utcnow()
                await self.collection.update_one(
                    {"name": name},
                    {"$set": update}
                )
        except NotInitializedError:
            raise
        except Exception as e:
            logger.error(f"Failed to update dataset counts: {str(e)}")
            raise
