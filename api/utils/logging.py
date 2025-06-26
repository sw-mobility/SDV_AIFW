import logging

def setup_logging():
    """로깅 설정을 초기화합니다."""
    logging.basicConfig(level=logging.DEBUG)
    # pymongo DEBUG 로그 숨기기
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    return logging.getLogger(__name__)

logger = setup_logging()
