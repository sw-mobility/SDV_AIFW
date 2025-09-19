from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import JSONResponse
import logging
import httpx
from utils.init import init
from utils.auth import get_uid

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Validation"])

@router.get("/list", status_code=status.HTTP_200_OK)
async def list_validation_histories(uid: str = Depends(get_uid)):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    histories = await mongo_client.db["val_hst"].find({"uid": uid}).to_list(length=None)
    for history in histories:
        history.pop("_id", None)
        history.pop("uid", None)

    return histories