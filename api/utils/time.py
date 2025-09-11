from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time_kst() -> str:
    """
    현재 시간을 한국 표준시(KST)로 ISO 포맷 문자열로 반환합니다.
    """
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()