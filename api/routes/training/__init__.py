"""Training service router (object_detection only)"""

from fastapi import APIRouter

# Training API 라우터
router = APIRouter(prefix="/training", tags=["Training"])
