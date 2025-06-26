"""글로벌 파일 삭제 라우터"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# from routes.labeled.images.yolo.v8.det.routes import get_file_handler as get_yolov8_file_handler  # removed
# from routes.raw.images.file_handler import get_file_handler as get_raw_handler

from fastapi import APIRouter, HTTPException, Path
from utils.logging import logger

router = APIRouter(prefix="/files", tags=["Files"])

@router.delete("/{file_id}", status_code=204)
async def delete_file(file_id: str = Path(..., description="파일 ID")):
    """
    파일 ID로 모든 유형의 파일(labeled/raw/yolo 등)을 삭제합니다.
    """
    try:
        # YOLOv8 Detection 시도 (제거)
        # try:
        #     result = await get_yolov8_file_handler().delete_file(file_id)
        #     if result:
        #         return None
        # except Exception:
        #     pass
        # raw 시도 (dataset/category 정보가 필요할 수 있음)
        # try:
        #     result = await get_raw_handler().delete_file(...)
        #     if result:
        #         return None
        # except Exception:
        #     pass
        raise HTTPException(status_code=404, detail="File not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Global file deletion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
