import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup(path: str):
    """
    압축 파일을 삭제합니다.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")