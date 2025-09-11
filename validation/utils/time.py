from datetime import datetime

def get_current_time_kst() -> str:
    """현재 시간을 한국 표준시(KST)로 ISO 포맷 문자열로 반환"""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
    except ImportError:
        # zoneinfo가 없는 경우 UTC 사용
        return datetime.now().isoformat()

