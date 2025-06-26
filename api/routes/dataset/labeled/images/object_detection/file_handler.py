"""Object Detection 파일 핸들러 (COCO/YOLO flat 구조, no model/version dir)"""
import os
import uuid
from typing import Optional, List, Dict, Any, Tuple
from fastapi import UploadFile, HTTPException, status
from fastapi.responses import StreamingResponse
from bson import ObjectId

from core.storage import storage_client
from core.mongodb import MongoDB
from utils.logging import logger
from utils.file_hash import calculate_file_hash
from config.settings import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_LABEL_EXTENSIONS
from models.dataset.raw.images.mongodb import ImageFile, convert_mongo_document
from models.dataset.raw.images.api import FileUploadResponse, FileUploadError
from ._mongo_manager import get_mongo_manager
from models.dataset.labeled.images.mongodb import LabeledImageFile
from storage.base import FileDownloadError

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    logger.warning("PIL/Pillow library not installed. Image dimension detection will be disabled.")
    HAS_PIL = False
import io

class ObjectDetectionFileHandler:
    """Object Detection 파일 핸들러 (flat 구조, no model/version dir)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.storage_client = storage_client

    def _get_dataset_path(self, name: str) -> str:
        return f"datasets/labeled/images/object_detection/{name}"

    def _get_file_path(self, dataset_name: str, filename: str, subfolder: str) -> str:
        return f"datasets/labeled/images/object_detection/{dataset_name}/{subfolder}/{filename}"

    async def upload_file(self, dataset_id: str, dataset_name: str, image_file: UploadFile, label_file: Optional[UploadFile] = None
                         ) -> Tuple[FileUploadResponse, Dict[str, Any]]:
        try:
            _, ext = os.path.splitext(image_file.filename)
            if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                raise HTTPException(status_code=400, detail="이미지 파일은 반드시 jpg, jpeg, png 형식이어야 합니다.")
            unique_filename = image_file.filename
            base_filename = os.path.splitext(unique_filename)[0]
            image_path = self._get_file_path(dataset_name, unique_filename, "images")
            try:
                await self.storage_client.stat_object(image_path)
                raise HTTPException(status_code=409, detail=f"이미 존재하는 파일명입니다: {unique_filename}")
            except Exception:
                pass
            contents = await image_file.read()
            file_size = len(contents)
            file_hash = calculate_file_hash(contents)
            width, height = None, None
            if HAS_PIL:
                try:
                    img = Image.open(io.BytesIO(contents))
                    width, height = img.size
                except Exception as e:
                    logger.warning(f"Failed to get image dimensions: {str(e)}")
            await self.storage_client.put_object(
                key=image_path,
                body=contents,
                content_type=image_file.content_type
            )
            file_metadata = {
                "dataset_id": dataset_id,  # 문자열로만 저장 (ObjectId 아님)
                "filename": unique_filename,
                "path": image_path,
                "size": file_size,
                "width": width,
                "height": height,
                "mime_type": image_file.content_type or "application/octet-stream",
                "hash": file_hash,
                "metadata": {"original_filename": unique_filename}
            }
            # MongoDB에 이미지 파일 메타데이터 저장 (rawdata와 동일하게 명시적 필드)
            image_file_doc = LabeledImageFile(**file_metadata)
            doc = image_file_doc.model_dump(by_alias=True)
            if '_id' in doc:
                del doc['_id']
            await MongoDB.db["labeled_image_files"].insert_one(doc)
            label_path = None
            if label_file:
                if label_file.content_type not in SUPPORTED_LABEL_EXTENSIONS:
                    raise HTTPException(status_code=400, detail="레이블 파일은 반드시 .txt(text/plain) 형식이어야 합니다.")
                label_filename = f"{base_filename}.txt"
                label_contents = await label_file.read()
                label_path = self._get_file_path(dataset_name, label_filename, "labels")
                await self.storage_client.put_object(
                    key=label_path,
                    body=label_contents,
                    content_type="text/plain"
                )
                # MongoDB에 레이블 파일 메타데이터 저장
                label_file_metadata = {
                    "dataset_id": dataset_id,
                    "filename": label_filename,
                    "path": label_path,
                    "size": len(label_contents),
                    "width": None,
                    "height": None,
                    "mime_type": "text/plain",
                    "hash": None,
                    "metadata": {"original_filename": label_filename}
                }
                label_file_doc = LabeledImageFile(**label_file_metadata)
                label_doc = label_file_doc.model_dump(by_alias=True)
                if '_id' in label_doc:
                    del label_doc['_id']
                await MongoDB.db["labeled_image_files"].insert_one(label_doc)
            response = FileUploadResponse(
                filename=unique_filename,
                path=image_path,
                status="success",
                message="File uploaded successfully",
                label_path=label_path
            )
            return response, file_metadata
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload file: {str(e)}"
            )

    async def list_files(self, dataset_id: str) -> List[Dict[str, Any]]:
        """
        특정 Object Detection 데이터셋의 이미지, 레이블(txt), data.yaml 파일 목록을 반환합니다.
        """
        dataset_path = self._get_dataset_path(dataset_id)
        files = []
        # images, labels 폴더 모두 조회
        for subfolder, type_map, file_type in [
            ("images", {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}, "image"),
            ("labels", {"txt": "text/plain"}, "label")
        ]:
            folder_path = f"{dataset_path}/{subfolder}"
            collection = MongoDB.db["labeled_image_files"]
            async for doc in collection.find({"dataset_id": dataset_id, "path": {"$regex": f"/{subfolder}/"}}):
                doc = convert_mongo_document(doc)
                file_info = {
                    "_id": doc["_id"],
                    "filename": doc.get("filename", ""),
                    "dataset_name": dataset_id,
                    "content_type": type_map.get(doc.get("filename", "").split(".")[-1].lower(), "application/octet-stream"),
                    "size": doc.get("size", 0),
                    "path": doc.get("path", ""),
                    "file_type": file_type,
                }
                files.append(file_info)
        # data.yaml 파일 존재 시 목록에 추가
        yaml_path = f"{dataset_path}/data.yaml"
        try:
            stat = await self.storage_client.stat_object(yaml_path)
            files.append({
                "_id": None,
                "filename": "data.yaml",
                "dataset_name": dataset_id,
                "content_type": "text/yaml",
                "size": stat.size if hasattr(stat, 'size') else None,
                "path": yaml_path,
                "file_type": "config",
            })
        except Exception:
            pass  # 없으면 무시
        return files
