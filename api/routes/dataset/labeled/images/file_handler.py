"""레이블링된 이미지 파일 핸들러"""
import os
import uuid
from typing import Optional, List, Dict, Any, BinaryIO, Tuple
from fastapi import UploadFile, HTTPException
from datetime import datetime

from core.storage import storage_client
from storage.managers.labeled.images.manager import LabeledImageStorageManager
from models.dataset.labeled.images.mongodb import LabeledImageFile, Annotation
from models.dataset.labeled.images.api import FileUploadResponse, FileUploadError, AnnotationCreate
from config.settings import SUPPORTED_IMAGE_EXTENSIONS
from utils.file_hash import calculate_file_hash
from utils.logging import logger
# Pillow 라이브러리가 없는 경우를 대비한 조건부 임포트
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    logger.warning("PIL/Pillow library not installed. Image dimension detection will be disabled.")
    HAS_PIL = False
import io

class LabeledImageFileHandler:
    """레이블링된 이미지 파일 처리 클래스"""
    
    def __init__(self):
        self.storage_manager = LabeledImageStorageManager()
    
    async def upload_file(
        self, 
        file: UploadFile, 
        dataset_id: str, 
        category_id: str,
        annotations: Optional[List[AnnotationCreate]] = None
    ) -> Tuple[FileUploadResponse, Optional[List[str]]]:
        """이미지 파일 업로드 및 애노테이션 처리"""
        try:
            # 파일 확장자 확인
            _, ext = os.path.splitext(file.filename)
            if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported file format: {ext}")
                
            # 중복 방지를 위한 고유 파일명 생성
            unique_filename = f"{uuid.uuid4()}{ext}"
            
            # 파일 내용 읽기
            contents = await file.read()
            file_size = len(contents)
            
            # 파일 해시 계산
            file_hash = calculate_file_hash(contents)
              # 이미지 크기 정보 추출
            width, height = None, None
            if HAS_PIL:
                try:
                    img = Image.open(io.BytesIO(contents))
                    width, height = img.size
                except Exception as e:
                    logger.warning(f"Failed to get image dimensions: {str(e)}")
            else:
                logger.info("Skipping image dimension detection due to missing PIL/Pillow library.")
            
            # 스토리지에 파일 저장 (파일 경로는 스토리지 매니저에서 생성)
            dataset = await self.storage_manager.get_dataset(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset not found: {dataset_id}")
                
            category = await self.storage_manager.get_category(category_id)
            if not category:
                raise ValueError(f"Category not found: {category_id}")
            
            # 파일 경로 생성
            file_path = self.storage_manager._get_file_path(
                dataset.name, 
                category.path, 
                unique_filename
            )
            
            # 스토리지에 파일 저장
            await storage_client.put_object(
                key=file_path,
                body=contents,
                content_type=file.content_type
            )
            
            # 데이터베이스에 파일 정보 저장
            image_file = LabeledImageFile(
                dataset_id=dataset_id,
                category_id=category_id,
                filename=unique_filename,
                path=file_path,
                size=file_size,
                width=width,
                height=height,
                mime_type=file.content_type or "application/octet-stream",
                hash=file_hash,
                annotations=[]  # 초기에는 빈 목록
            )
            
            # MongoDB에 이미지 파일 정보 저장
            saved_file = await self.storage_manager.file_manager.create_file(image_file)
            
            # 애노테이션 처리
            annotation_ids = []
            if annotations:
                for ann in annotations:
                    annotation = Annotation(
                        image_id=saved_file.id,
                        type=ann.type,
                        coordinates=ann.coordinates,
                        label=ann.label,
                        category_id=category_id,
                        attributes=ann.attributes or {}
                    )
                    saved_ann = await self.storage_manager.annotation_manager.create_annotation(annotation)
                    annotation_ids.append(saved_ann.id)
                
                # 이미지 파일에 애노테이션 ID 목록 업데이트
                await self.storage_manager.file_manager.update_file_annotations(
                    saved_file.id, annotation_ids
                )
            
            # 응답 생성
            return (
                FileUploadResponse(
                    filename=unique_filename,
                    path=file_path,
                    category_id=category_id,
                    status="success",
                    message="File uploaded successfully"
                ),
                annotation_ids
            )
                
        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            return (
                FileUploadError(
                    filename=file.filename,
                    message=f"Upload failed: {str(e)}",
                    type="error"
                ),
                None
            )
            
    async def delete_file(self, file_id: str) -> bool:
        """이미지 파일 삭제"""
        try:
            # 파일 정보 조회
            file = await self.storage_manager.file_manager.get_file(file_id)
            if not file:
                raise ValueError(f"File not found: {file_id}")
                
            # 관련 애노테이션 삭제
            for ann_id in file.annotations:
                await self.storage_manager.annotation_manager.delete_annotation(ann_id)
                
            # 스토리지에서 파일 삭제
            await storage_client.remove_object(file.path)
            
            # 데이터베이스에서 파일 정보 삭제
            result = await self.storage_manager.file_manager.delete_file(file_id)
            
            return result
        except Exception as e:
            logger.error(f"File deletion error: {str(e)}")
            raise
