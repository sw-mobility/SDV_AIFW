"""글로벌 삭제 API: 데이터셋/파일/어노테이션 등 모든 도메인에 대해 일관된 삭제 제공"""
from fastapi import APIRouter, HTTPException, Path
from storage.managers.labeled.images.manager import LabeledImageStorageManager
from storage.managers.raw.images.manager import RawImageStorageManager
#from storage.managers.labeled.lidar.manager import LabeledLidarStorageManager  # 필요시
#from storage.managers.raw.lidar.manager import RawLidarStorageManager  # 필요시

router = APIRouter(tags=["Global Delete"])

# 데이터셋 삭제 (labeled/raw 자동 판별)
@router.delete("/delete/dataset/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: str = Path(..., description="데이터셋 ID 또는 이름")):
    """데이터셋 ID로 labeled/raw 구분 없이 삭제"""
    # 우선 labeled → raw 순으로 시도
    for manager in [LabeledImageStorageManager(), RawImageStorageManager()]:
        try:
            result = await manager.delete_dataset(dataset_id)
            if result:
                return None
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Dataset not found")

# 파일 삭제 (labeled/raw 자동 판별)
@router.delete("/delete/file/{file_id}", status_code=204)
async def delete_file(file_id: str = Path(..., description="파일 ID")):
    """파일 ID로 labeled/raw 구분 없이 삭제"""
    for manager in [LabeledImageStorageManager(), RawImageStorageManager()]:
        try:
            result = await manager.delete_file(file_id)
            if result:
                return None
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="File not found")

# 어노테이션 삭제 (labeled만 예시)
@router.delete("/delete/annotation/{annotation_id}", status_code=204)
async def delete_annotation(annotation_id: str = Path(..., description="어노테이션 ID")):
    """어노테이션 ID로 삭제 (labeled만 예시)"""
    try:
        result = await LabeledImageStorageManager().annotation_manager.delete_annotation(annotation_id)
        if result:
            return None
    except Exception:
        pass
    raise HTTPException(status_code=404, detail="Annotation not found")
