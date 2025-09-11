# main.py 또는 별도 utils/auth.py 등에 작성
from fastapi import Header, HTTPException, Depends

def get_uid(uid: str = Header("0001")):
    if not uid:
        raise HTTPException(status_code=401, detail="No uid header")
    return uid