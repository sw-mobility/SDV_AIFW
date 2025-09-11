from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile, 
    File, 
    Path, 
    Query, 
    status, 
    Body, 
    Depends
)
from fastapi.responses import (
    JSONResponse
)
from typing import (
    List, 
    Optional
)
from core.minio import (
    MinioStorageClient
)
from core.mongodb import (
    MongoDBClient
)
from core.config import (
    MONGODB_URL, 
    MONGODB_DB_NAME, 
    MONGODB_COLLECTIONS,
    MIME_TYPES
)
from models.project.project_model import (
    ProjectCreate,
    ProjectUpdate,
    ProjectInfo
)
from services.project.project_service import (
    create_project,
    update_project,
    list_projects,
    delete_project
)
from utils.counter import (
    get_next_counter
)
from utils.time import (
    get_current_time_kst
)
from utils.init import init
from utils.auth import get_uid
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/projects", tags=["Project Management"])

# 1. 프로젝트 생성 (POST /projects/create)
@router.post("/create", response_model=ProjectCreate, status_code=status.HTTP_201_CREATED)
async def create(project: ProjectCreate, uid: str = Depends(get_uid)):
    """Create a new project."""
    try:
        project_info = await create_project(uid, project)
        logger.info(f"Project created successfully: {project_info.name}")
        return JSONResponse(status_code=status.HTTP_201_CREATED,
                            content={
                                "message": "Project created successfully",
                                "project": project_info.model_dump()  # 또는 .model_dump() (Pydantic v2)
                            }
                        )
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create project")

# 2. 정보 업데이트 (PUT /projects/update)
@router.put("/update", response_model=ProjectInfo)
async def update(pid: str, project: ProjectUpdate, uid: str = Depends(get_uid)):
    """Update an existing project."""
    try:
        updated_project = await update_project(uid, project)
        logger.info(f"Project updated successfully: {updated_project.name}")
        return JSONResponse(status_code=status.HTTP_200_OK,
                            content={
                                "message": "Project updated successfully",
                                "project": updated_project.model_dump()
                            }
                        )
    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update project")
    

# 3. 전체 목록 조회 (GET /projects/)
@router.get("/", response_model=List[ProjectInfo], status_code=status.HTTP_200_OK)
async def list(uid: str = Depends(get_uid)):
    """List all projects for a given user."""
    try:
        projects = await list_projects(uid)
        return projects
    except Exception as e:
        logger.error("Error listing projects: %s", str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    

# 4. 프로젝트 삭제 (DELETE /projects/)
@router.delete("/", status_code=status.HTTP_200_OK)
async def delete(pid: str, uid: str = Depends(get_uid)):
    """Delete a project by ID."""
    try:
        await delete_project(uid, pid)
        logger.info(f"Project deleted successfully: {pid}")
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Project deleted successfully"})
    
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete project")