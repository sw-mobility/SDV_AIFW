import os
import shutil
import logging    

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup_zip(zip_path: str, folder_path: str):
    """
    압축 파일을 삭제합니다.
    """
    try:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")


async def cleanup_workdir(path: str):
    try:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            logging.warning(f"경로가 존재하지 않음: {path}")
    except Exception as e:
        logging.error(f"Error cleaning up workdir {path}: {e}")
        raise