from datetime import datetime
from zoneinfo import ZoneInfo

def get_current_time_kst() -> str:
    """
    Returns the current time in Korea Standard Time (KST) as an ISO format string.
    """
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
