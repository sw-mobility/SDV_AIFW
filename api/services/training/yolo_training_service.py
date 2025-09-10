from fastapi import (
    APIRouter, 
    HTTPException, 
    UploadFile, 
    File, 
    Path, 
    Query, 
    status, 
    Body, 
    Depends,
    BackgroundTasks
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
    MIME_TYPES,
    API_WORKDIR,
    TRAINING_WORKDIR,
    FRONTEND_WORKDIR,
    YOLO_MODELS
)
from models.training.yolo_training_model import (
    YoloDetTrainingRequest,
    YoloTrainingInfo,
    YoloTrainingResult,
    YoloDetTrainingParams,
)
from models.training.common_training_model import (
    TrainingHistory
)
from models.dataset.labeled_model import (
    LabeledDatasetCreate,
    LabeledDataInfo,
    P_LabeledDatasetInfo,
    LabeledDatasetInfo
)
from utils.counter import (
    get_next_counter
)
from utils.time import (
    get_current_time_kst
)
from utils.init import (
    init
)
from utils.counter import (
    get_next_counter
)
from utils.cleanup import (
    cleanup_workdir
)
from utils.model_id import (
    is_custom_model_id
)
import os
import logging
import yaml
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def type_check(task_type: str):
    if task_type == "detection":
        return "image"
    else:
        return "idk"

async def parse_yolo_det_training(uid: str, request: YoloDetTrainingRequest):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    pid = request.pid

    project = await mongo_client.db["projects"].find_one({"uid": uid, "pid": pid})
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    workdir = f"{TRAINING_WORKDIR}/{uid}/{pid}/"
    did = request.did
    cid = request.cid
    model = request.parameters.model
    parameters = request.parameters

    model_doc = await mongo_client.db["trn_hst"].find_one({"_id": f"{uid}{pid}{request.origin_tid}"})
    if model_doc:
        model_classes = model_doc.get("classes", [])
    else:
        model_classes = []
        
    dataset_doc = await mongo_client.db["labeled_datasets"].find_one({"_id": f"{uid}{did}"})
    dataset_classes = dataset_doc.get("classes", [])

    if parameters is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Training parameters are required.")
    
    if did is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset is required for YOLO training."
        )

    # 1. Parse model file name
    if not model:
        parameters.model = "best.pt"

    else:
        if model not in YOLO_MODELS.DET:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model name for detection task."
            )
    
    data = YoloTrainingInfo(
        uid=uid,
        pid=pid,
        origin_tid=request.origin_tid,
        task_type="detection",
        parameters=parameters.dict(),
        user_classes=request.user_classes,
        model_classes=model_classes,
        dataset_classes=dataset_classes,
        workdir=workdir,
        did=did,
        cid=cid,
    )

    return data

async def prepare_model(uid: str, train_info: YoloTrainingInfo):
    """Prepare the model for training by downloading it from MinIO."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    workdir = train_info.workdir
    model_path = os.path.join(workdir, "best.pt")
    tid = train_info.origin_tid
    pid = train_info.pid

    try:
        # Create workdir (ignore if exists)
        os.makedirs(workdir, exist_ok=True)
        logger.info(f"Created workdir: {workdir}")

        artifact = await mongo_client.db["trn_hst"].find_one({"_id": uid+pid+tid})
        key = artifact.get("artifacts_path")

        # Download the model from MinIO
        await minio_client.download_minio_file(uid, key+"/best.pt", model_path)
        if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
            raise HTTPException(status_code=500, detail="Downloaded model is empty or missing.")
        
        logger.info(f"Model for YOLO training prepared")

    except Exception as e:
        logger.error(f"Failed to prepare model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare model: {e}")
    

async def prepare_dataset(uid: str, train_info: YoloTrainingInfo):
    """Prepare the dataset for training by downloading it from MinIO."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    did = train_info.did
    workdir = train_info.workdir

    try:
        # Create workdir (ignore if exists)
        os.makedirs(workdir, exist_ok=True)
        logger.info(f"Created workdir: {workdir}")

        doc = await mongo_client.db["labeled_datasets"].find_one({"_id": uid+did})
        target_path = doc.get("path")
        if not target_path:
            raise HTTPException(status_code=404, detail=f"Dataset not found in MongoDB.")

        await minio_client.download_minio_directory(uid, target_path, workdir)
        logger.info(f"Dataset for YOLO training deployed: {workdir}")

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare dataset: {e}")


async def prepare_codebase(uid: str, train_info: YoloTrainingInfo):
    """Prepare the codebase for YOLO training."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]
    
    cid = train_info.cid
    logger.info(f"Preparing codebase for YOLO training with cid: {cid}")
    workdir = train_info.workdir
    codebase_dir = os.path.join(workdir, "ultralytics")

    if os.path.exists(codebase_dir):
        await cleanup_workdir(codebase_dir)

    try:
        # Create workdir (ignore if exists)
        os.makedirs(workdir, exist_ok=True)
        logger.info(f"Created workdir: {workdir}")

        if not cid:
            bucket = "keti-aifw"
            prefix = "codebases/yolo"

        else:
            codebase = await mongo_client.db["codebases"].find_one({"_id": uid+cid})
            if not codebase:
                raise HTTPException(status_code=404, detail="Codebase not found")
            bucket = uid
            prefix = codebase["path"]

        # Download the codebase from MinIO
        await minio_client.download_minio_directory(bucket, prefix, codebase_dir)

        logger.info(f"Codebase for YOLO training deployed: {codebase_dir}")

    except Exception as e:
        logger.error(f"Failed to prepare codebase: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare codebase: {e}")


async def create_training_history(result):
    """Create a training history entry in MongoDB."""
    init_result = await init(result.uid)
    mongo_client = init_result["mongo_client"]

    # 전달받은 tid 사용 (새로 생성하지 않음)
    tid = result.tid if hasattr(result, 'tid') and result.tid else await get_next_counter(mongo_client, "trn_hst", uid=result.uid, prefix="T", field="tid", width=4)
    doc = await mongo_client.db["labeled_datasets"].find_one({"_id": f"{result.uid}{result.did}"})
    origin_dataset_name = doc.get("name", "Unknown Dataset")
    processed_dataset_name = result.did + "_preprocessed_" + result.origin_tid
    processed_did = await get_next_counter(mongo_client, "labeled_datasets", uid=result.uid, prefix="L", field="did", width=4)
    classes = result.classes

    artifacts_path = None
    if result.artifacts_path is None:
        artifacts_path = None
    else:
        artifacts_path = f"artifacts/{result.pid}/training/{tid}"

    codebase_name = None
    if result.cid:
        codebase_doc = await mongo_client.db["trn_cb"].find_one({"_id": result.cid})
        codebase_name = codebase_doc.get("name") if codebase_doc else result.cid

    if result.parameters is None:
        result.parameters = {}

    history = TrainingHistory(
        _id = result.uid + result.pid + tid,
        uid=result.uid,
        pid=result.pid,
        tid=tid,
        origin_tid=result.origin_tid,
        origin_did=result.did,
        origin_dataset_name=origin_dataset_name,
        processed_did=processed_did,
        processed_dataset_name=processed_dataset_name,
        artifacts_path=artifacts_path,
        used_codebase=codebase_name,
        error_details=result.error_details,
        parameters=result.parameters,
        started_at=result.started_time,
        completed_at=result.completed_time,
        classes=classes,
        status=result.status,
    )

    try:
        await mongo_client.db["trn_hst"].insert_one(history.dict(by_alias=True))
        logger.info(f"Training history created")
        return history
    except Exception as e:
        logger.error(f"Failed to create training history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create training history: {e}")


async def upload_preprocessed_dataset(uid: str, origin_did: str, processed_did: str, pid: str, tid: str, task_type: str, name: str, dataset_basepath: str, classes: list):

    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    # 프로젝트 정보
    project = await mongo_client.db["projects"].find_one({"_id": uid+pid})
    project_name = project.get("name", "Unknown Project") if project else "Unknown Project"

    # 데이터셋 경로 리스트
    path_list = [os.path.join(dataset_basepath, "images"), os.path.join(dataset_basepath, "labels")]
    documents = []
    names = []

    # 데이터셋 정보 객체 생성
    dataset_info_obj = P_LabeledDatasetInfo(
        _id = uid + processed_did,
        uid = uid,
        did = processed_did,
        pid = pid,
        name = name,
        description = f"This is a preprocessed dataset of {origin_did} for training run {tid[1:]} in project {project_name}.",
        classes = classes,
        type = await type_check(task_type),
        task_type = task_type,
        label_format = "YOLO",
        path = f"datasets/labeled/{processed_did}",
        total = 0,
        created_at = get_current_time_kst()
    )

    try:
        # 데이터셋 정보 등록
        await mongo_client.db["labeled_datasets"].insert_one(dataset_info_obj.dict(by_alias=True))
        logger.info(f"Created labeled dataset: {dataset_info_obj.name}")

        data_collection = "labeled_data"
        dataset_collection = "labeled_datasets"
        dataset_info = await mongo_client.db[dataset_collection].find_one({"_id": uid+processed_did})
        dataset_path = dataset_info.get("path") if dataset_info else f"datasets/labeled/{processed_did}"
        origin_raw = dataset_info.get("origin_raw") if dataset_info else None
        dataset = dataset_info.get("name") if dataset_info else "Unnamed Labeled Dataset"

        for path in path_list:
            if not os.path.exists(path):
                logger.warning(f"Path does not exist: {path}")
                continue
            for root, dirs, files in os.walk(path, topdown=True):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = file.split('.')[-1].lower()
                    key = None
                    doc_type = None
                    # 파일 유형 및 key 결정
                    if file_ext in ["jpg", "png", "jpeg"]:
                        key = f"{dataset_path}/images/{file}"
                        doc_type = "image"
                    elif file_ext in ["json", "txt"]:
                        key = f"{dataset_path}/labels/{file}"
                        doc_type = "label"
                    else:
                        key = f"{dataset_path}/others/{file}"
                        doc_type = "other"

                    doc = {
                        "_id": uid+processed_did+file,
                        "uid": uid,
                        "did": processed_did,
                        "dataset": dataset,
                        "name": file,
                        "type": doc_type,
                        "origin_raw": origin_raw,
                        "file_format": file_ext,
                        "path": f"{key}",
                        "created_at": get_current_time_kst()
                    }
                    documents.append(doc)

                    await mongo_client.upload_data(
                        uid=uid,
                        did=processed_did,
                        data_collection=data_collection,
                        dataset_collection=dataset_collection,
                        doc=doc
                    )

                    try:
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()
                        await minio_client.upload_files(uid, file_bytes, key)
                    except Exception as fe:
                        logger.error(f"Failed to upload file {file_path}: {fe}")
                        continue

        yaml_path = os.path.join(dataset_basepath, "data.yaml")
        if os.path.exists(yaml_path):
            with open(yaml_path, "rb") as yf:
                file_bytes = yf.read()
            key = f"{dataset_path}/data.yaml"
            await minio_client.upload_files(uid, file_bytes, key)

            yaml_doc = {
                        "_id": uid+processed_did+"data.yaml",
                        "uid": uid,
                        "did": processed_did,
                        "dataset": dataset,
                        "name": f"data.yaml",
                        "type": "yaml",
                        "origin_raw": origin_raw,
                        "file_format": "yaml",
                        "path": f"{key}",
                        "created_at": get_current_time_kst()
                    }
            documents.append(yaml_doc)

            await mongo_client.upload_data(
                    uid=uid,
                    did=processed_did,
                    data_collection=data_collection,
                    dataset_collection=dataset_collection,
                    doc=yaml_doc
                )

        # 문서 리스트 반환
        return documents

    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {e}")
        

async def upload_artifacts(uid: str, pid: str, tid: str, path: str, workdir: str):
    """Upload the training artifacts to MinIO."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    artifacts_dir = path

    try:
        # Upload model weights and data.yaml
        if os.path.exists(artifacts_dir):
            os.walk(artifacts_dir, topdown=True)
            for root, _, files in os.walk(artifacts_dir):
                for file in files:
                    key = os.path.join("artifacts", pid, "training", tid, file)
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    await minio_client.upload_files(uid, file_bytes, key)
                    logger.info(f"Uploaded {file_path} to {key} in MinIO")
            for root, _, files in os.walk(workdir):
                for file in files:
                    key = os.path.join("artifacts", pid, "training", tid, file)
                    if file == "data.yaml":
                        file_path = os.path.join(root, file)
                        with open(file_path, "rb") as f:
                            file_bytes = f.read()
                        await minio_client.upload_files(uid, file_bytes, key)
                        logger.info(f"Uploaded {file_path} to {key} in MinIO")

    except Exception as e:
        logger.error(f"Failed to upload artifacts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload artifacts: {e}")



# async def prepare_model(uid: str, train_info: YoloDetTrainingInfo):
#     """Prepare the model for training by downloading it from MinIO."""
#     init_result = await init(uid)
#     minio_client = init_result["minio_client"]
#     mongo_client = init_result["mongo_client"]

#     uid = train_info.uid
#     tid = train_info.origin_tid
#     pid = train_info.pid
#     workdir = train_info.workdir

#     path = mongo_client.db["trn_hst"].find_one({"_id": uid+pid+tid})

#     try:
#         # Create workdir (ignore if exists)
#         os.makedirs(workdir, exist_ok=True)
#         logger.info(f"Created workdir: {workdir}")

#         # Download the model from MinIO
#         await minio_client.download_minio_file(uid, path+"/best.pt", os.path.join(workdir, "best.pt"))

#         logger.info(f"Model for YOLO training prepared")

#     except Exception as e:
#         logger.error(f"Failed to prepare model: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to prepare model: {e}")



#     uid = download_dataset.uid

#     init_result = await init(uid)
#     mongo_client = init_result["mongo_client"]
#     minio_client = init_result["minio_client"]

#     target_id = download_dataset.target_id
#     basedir = API_WORKDIR

#     if target_id[4] == "R":
#         target_collection = "raw_datasets"
#         target_data_collection = "raw_data"
#     elif target_id[4] == "L":
#         target_collection = "labeled_datasets"
#         target_data_collection = "labeled_data"
#     else:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Invalid target_id format"
#         )

#     target_info = await mongo_client.db[target_collection].find_one({"_id": target_id})
#     target_name = target_info.get("name")
#     target_path_list = await mongo_client.data_listup_to_download(target_id, target_data_collection)
#     logging.info(f"Target paths for compression: {target_path_list}")

#     zip_path = await minio_client.compress_dataset_to_download(uid, target_name, basedir,target_path_list)
#     logging.info(f"Compressed dataset saved to: {zip_path}")

#    return zip_path



# async def create_training_snapshot(uid: str, pid: str, name: str, algorithm: str, task_type: str, 
#                           description: Optional[str]):
#     """Create a snapshot of the current training state."""
#     init_result = await init(uid)
#     mongo_client = init_result["mongo_client"]
#     minio_client = init_result["minio_client"]

#     cid = await get_next_counter(mongo_client, 'trn_cb', uid=uid, prefix='C', field='cid', width=4)
#     codebase_path = f"{FRONTEND_WORKDIR}/{uid}/{pid}/training/codebase"

#     if not os.path.exists(codebase_path):
#         raise HTTPException(status_code=404, detail="Codebase does not exist")

#     # Create a new training history entry
#     doc = TrainingSnapshot(
#         _id=f"{uid}{cid}",
#         uid=uid,
#         cid=cid,
#         name=name,
#         algorithm=algorithm,
#         task_type=task_type,
#         description=description,
#         path = f"codebase/training/{cid}",
#         created_at=get_current_time_kst()
#     )

#     prefix = f"codebase/training/{cid}"

#     try:
#         await minio_client.upload_directory(uid, base_dir=codebase_path, prefix=prefix)
#         await mongo_client.db["trn_cb"].insert_one(doc.dict(by_alias=True))
#         return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Snapshot created successfully", "info": doc.dict()})
#     except Exception as e:
#         logger.error(f"Failed to create training snapshot: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to create training snapshot: {e}")






















