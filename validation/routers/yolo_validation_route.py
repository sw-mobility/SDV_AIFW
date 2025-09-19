from fastapi import APIRouter, HTTPException, status
from models import YoloDetValidationServiceRequest, YoloValidationResponse
from services import YoloValidationService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/yolo", tags=["Yolo Validation"])

validation_service = YoloValidationService()

@router.post("/validate/detection", response_model=YoloValidationResponse, status_code=status.HTTP_200_OK)
async def start_yolo_validation(request: YoloDetValidationServiceRequest):
    """Start YOLO validation process"""
    try:
        result = await validation_service.start_validation(request)
        return YoloValidationResponse(
            vid=result,
            status="started",
            message="YOLO validation started successfully",
        )
    except Exception as e:
        logger.error(f"Error starting YOLO validation: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



@router.post("/cleanup", status_code=status.HTTP_200_OK)
async def cleanup_validations():
    """Cleanup completed validations"""
    try:
        cleaned_count = await validation_service.cleanup_completed_validations()
        return {
            "message": f"Cleaned up {cleaned_count} validations",
            "cleaned_count": cleaned_count
        }
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) 