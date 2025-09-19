import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def cleanup(path: str):
    """
    Cleans up the specified file or directory.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
