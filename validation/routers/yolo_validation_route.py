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

@router.get("/{vid}", status_code=status.HTTP_200_OK)
async def get_validation_info(vid: str):
    """Get validation status and results (unified endpoint)"""
    try:
        validation_info = await validation_service.get_validation_info(vid)
        
        return validation_info
    except Exception as e:
        logger.error(f"Error getting validation info: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    try:
        active_count = await validation_service.get_active_validations_count()
        return {
            "status": "healthy",
            "active_validations": active_count,
            "service": "yolo_validation"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

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