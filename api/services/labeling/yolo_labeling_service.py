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
    ####################################################
    LABELING_WORKDIR,
    #####################################################
    FRONTEND_WORKDIR,
    YOLO_MODELS
)
from models.labeling.yolo_labeling_model import (
    YoloDetLabelingRequest,
    YoloDetLabelingInfo,
    YoloLabelingResult,
    YoloDetLabelingParams,
)
from models.dataset.labeled_model import (
    LabeledDatasetInfo,
    LabeledDataInfo
)
from models.labeling.common_labeling_model import (
    LabelingHistory
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

async def parse_yolo_labeling(uid: str, request: YoloDetLabelingRequest):
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]

    pid = request.pid
    did = request.did
    cid = request.cid
    name = request.name
    workdir = f"{LABELING_WORKDIR}/{uid}/{pid}/"

    project = await mongo_client.db["projects"].find_one({"uid": uid, "pid": pid})
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    
    # 1. Parse parameters
    parameters = request.parameters
    if parameters is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Labeling parameters are required.")

    labeling_params = parameters

    if did is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset ID is required for YOLO labeling."
        )

    data = YoloDetLabelingInfo(
        uid=uid,
        pid=pid,
        did=did,
        cid=cid,
        name=name,
        task_type="detection",
        parameters=labeling_params,
        workdir=workdir,
    )

    return data

async def prepare_model(uid: str, train_info: YoloDetLabelingInfo):
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


async def prepare_dataset(uid: str, label_info: YoloDetLabelingInfo):
    """Prepare the dataset for labeling by downloading it from MinIO."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    did = label_info.did
    workdir = os.path.join(label_info.workdir, did)

    try:
        # Create workdir (ignore if exists)
        os.makedirs(workdir, exist_ok=True)
        logger.info(f"Created workdir: {workdir}")

        cursor = mongo_client.db["raw_datasets"].find({"_id": uid+did})
        target_list = await cursor.to_list(length=None)
        if not target_list:
            raise HTTPException(status_code=404, detail=f"Dataset with ID {did} not found in MongoDB.")

        target_path = target_list[0]['path']

        await minio_client.download_minio_directory(uid, target_path, workdir)

    except Exception as e:
        logger.error(f"Failed to prepare dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare dataset: {e}")
    

async def prepare_default_codebase(uid: str, pid: str):
    """Prepare default codebase for YOLO labeling."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    workdir = f"{FRONTEND_WORKDIR}/{uid}/{pid}/labeling/"
    codebase_dir = os.path.join(workdir, "codebase/")
    logger.info(f"API workdir: {workdir}")

    if os.path.exists(codebase_dir):
        await cleanup_workdir(codebase_dir)

    try:
        # Create workdir (ignore if exists)
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(codebase_dir, exist_ok=True)
        logger.info(f"Created workdir: {workdir}")

        # Download the default assets from MinIO

        await minio_client.download_minio_directory("keti-aifw", "codebases/yolo", codebase_dir)

        logger.info(f"Default codebase for YOLO deployed")

    except Exception as e:
        logger.error(f"Failed to prepare default assets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare default assets: {e}")
    

async def prepare_default_yaml(uid: str, pid: str, dataset_id: str):
    """Prepare default YAML configuration for YOLO labeling."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    workdir = f"{FRONTEND_WORKDIR}/{uid}/{pid}/labeling/"
    yaml_path = os.path.join(workdir, "data.yaml")

    info = await mongo_client.db["raw_datasets"].find_one({"_id": dataset_id})
    key = info["path"]
    name = info["name"]
    print(key)
    print(name)
    if os.path.exists(yaml_path):
        await cleanup_workdir(yaml_path)

    try:
        # Create workdir (ignore if exists)
        os.makedirs(workdir, exist_ok=True)
        logger.info(f"Created workdir: {workdir}")

        await minio_client.download_minio_file(uid, f"{key}/data.yaml", yaml_path)

        logger.info(f"Default YAML for Dataset {name} deployed")

    except Exception as e:
        logger.error(f"Failed to prepare default YAML: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare default YAML: {e}")


async def prepare_model_and_yaml(uid: str, pid: str, origin_tid: str):
    """Upload the labeling artifacts to MinIO."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    workdir = f"{FRONTEND_WORKDIR}/{uid}/{pid}/labeling/"
    weight_path = os.path.join(workdir, "best.pt")
    yaml_path = os.path.join(workdir, "data.yaml")

    origin_weight_path = f"artifacts/{pid}/labeling/{origin_tid}/best.pt"
    origin_yaml_path = f"artifacts/{pid}/labeling/{origin_tid}/data.yaml"
    
    try:
        if os.path.exists(weight_path):
            await cleanup_workdir(weight_path)
        if os.path.exists(yaml_path):
            await cleanup_workdir(yaml_path)

        await minio_client.download_minio_file(uid, origin_yaml_path, yaml_path)
        await minio_client.download_minio_file(uid, origin_weight_path, weight_path)

    except Exception as e:
        logger.error(f"Failed to prepare model and YAML: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare model and YAML: {e}")
    
    
async def prepare_codebase(uid: str, label_info: YoloDetLabelingInfo):
    """Prepare the codebase for YOLO labeling."""
    init_result = await init(uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    cid = label_info.cid
    pid = label_info.pid

    workdir = f"{LABELING_WORKDIR}/{uid}/{pid}/"
    codebase_dir = os.path.join(workdir, "ultralytics/")

    if os.path.exists(codebase_dir):
        await cleanup_workdir(codebase_dir)

    try:
        # Create workdir (ignore if exists)
        os.makedirs(workdir, exist_ok=True)
        logger.info(f"Created workdir: {workdir}")

        if cid:
        # Download the codebase from MinIO
            doc = await mongo_client.db["codebases"].find_one({"_id": uid+cid})
            if not doc:
                raise HTTPException(status_code=404, detail="Codebase not found")
            bucket = uid
            key = doc.get("path")
            if not key:
                raise HTTPException(status_code=400, detail="Codebase path is missing")
        
        else:
            bucket = "keti-aifw"
            key = "codebases/yolo"

        await minio_client.download_minio_directory(bucket, key, codebase_dir)

        logger.info(f"Codebase for YOLO labeling deployed")

    except Exception as e:
        logger.error(f"Failed to prepare codebase: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to prepare codebase: {e}")


async def create_labeling_snapshot(uid: str, pid: str, name: str, algorithm: str, task_type: str, 
                          description: Optional[str]):
    """Create a snapshot of the current labeling state."""
    init_result = await init(uid)
    mongo_client = init_result["mongo_client"]
    minio_client = init_result["minio_client"]

    cid = await get_next_counter(mongo_client, 'trn_cb', uid=uid, prefix='C', field='cid', width=4)
    codebase_path = f"{FRONTEND_WORKDIR}/{uid}/{pid}/labeling/codebase"

    if not os.path.exists(codebase_path):
        raise HTTPException(status_code=404, detail="Codebase does not exist")

    # Create a new labeling history entry
    doc = LabelingSnapshot(
        _id=f"{uid}{cid}",
        uid=uid,
        cid=cid,
        name=name,
        algorithm=algorithm,
        task_type=task_type,
        description=description,
        path = f"codebase/labeling/{cid}",
        created_at=get_current_time_kst()
    )

    prefix = f"codebase/labeling/{cid}"

    try:
        await minio_client.upload_directory(uid, base_dir=codebase_path, prefix=prefix)
        await mongo_client.db["trn_cb"].insert_one(doc.dict(by_alias=True))
        return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Snapshot created successfully", "info": doc.dict()})
    except Exception as e:
        logger.error(f"Failed to create labeling snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create labeling snapshot: {e}")
    

async def deploy_frontend_files(uid: str, pid: str, workdir: str):
    """Deploy frontend files to the labeling workspace."""
    await init(uid)
    #workdir = os.path.join(workdir, dataset_id) #f"{LABELING_WORKDIR}/{uid}/{pid}/"

    try:
        src_dir = f"{FRONTEND_WORKDIR}/{uid}/{pid}/labeling"
        dst_dir = os.path.join(workdir, 'ultralytics')  #f"{LABELING_WORKDIR}/{uid}/{pid}/"
        logger.info(f"source dir: {src_dir}")
        logger.info(f"destination dir: {dst_dir}")

        os.makedirs(dst_dir, exist_ok=True)

        if os.path.exists(os.path.join(src_dir, "best.pt")):
            shutil.copy2(os.path.join(src_dir, "best.pt"), os.path.join(dst_dir, "best.pt"))
        else:
            logger.info(f"Custom weight does not exist, skipping copy.")

        # shutil.copy2(os.path.join(src_dir, "data.yaml"), os.path.join(dst_dir, "data.yaml"))
        shutil.copytree(os.path.join(src_dir, "codebase"), os.path.join(dst_dir), dirs_exist_ok=True)

        logger.info(f"Data for YOLO labeling deployed")

    except Exception as e:
        logger.error(f"Failed to deploy frontend files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy frontend files: {e}")


async def create_labeling_history(result):
    """Create a labeling history entry in MongoDB."""
    init_result = await init(result.uid)
    mongo_client = init_result["mongo_client"]

    doc = await mongo_client.db["labeled_datasets"].find_one({"_id": result.dataset_id})
    dataset_name = doc.get("name", "Unknown Dataset") if doc else "Unknown Dataset"
    tid = await get_next_counter(mongo_client, "lab_hst", uid=result.uid, prefix="L", field="lid", width=4)
    classes = []
    with open(os.path.join(result.workdir, "data.yaml"), 'r') as file:
        data_yaml = yaml.safe_load(file)
        classes = data_yaml.get('names', [])

    artifacts_path = None
    if result.artifacts_path is None:
        artifacts_path = None
    else:
        artifacts_path = f"artifacts/{result.pid}/labeling/{tid}"

    codebase_name = None
    if result.codebase_id:
        codebase_doc = await mongo_client.db["trn_cb"].find_one({"_id": result.codebase_id})
        codebase_name = codebase_doc.get("name", "Unknown Codebase") if codebase_doc else "Unknown Codebase"

    if result.parameters is None:
        result.parameters = {}

    history = LabelingHistory(
        _id = result.uid + result.pid + tid,
        uid=result.uid,
        pid=result.pid,
        tid=tid,
        origin_tid=result.origin_tid,
        dataset_id=result.dataset_id,
        dataset_name=dataset_name,
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
        logger.info(f"Labeling history created")
        return history
    except Exception as e:
        logger.error(f"Failed to create labeling history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create labeling history: {e}")


async def upload_artifacts(params):
    """Upload the labeling artifacts to MinIO."""
    init_result = await init(params.uid)
    minio_client = init_result["minio_client"]
    mongo_client = init_result["mongo_client"]

    artifacts_dir = params.artifacts_path
    uid = params.uid
    name = params.name
    type = None
    did= await get_next_counter(mongo_client, "labeled_datasets", uid=uid, prefix="L", field="did", width=4)
    rawdid = params.raw_dataset_id
    logger.info(f"YOLO Labeling Result: {params}")
    logger.info(f"YOLO Labeling Result: {params}")
    logger.info(f"YOLO Labeling Result: {params}")
    logger.info(f"YOLO Labeling Result: {params}")
    logger.info(f"YOLO Labeling Result: {params}")
    logger.info(f"YOLO Labeling Result: {params}")

    try:
        dataset_info = LabeledDatasetInfo(
        _id = uid+did,
        uid = uid,
        did = did,
        name = name,
        description= None,
        classes = params.classes,
        parameters = params.parameters,
        type = params.type,
        task_type = params.task_type,
        label_format = params.label_format,
        total = 0, # 총 데이터 수
        origin_raw = rawdid, # 원본 Raw Dataset ID
        path = f"datasets/labeled/{did}",
        created_at = params.completed_time # ISODate  # ISODate
    )
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")
        logger.info(f"YOLO Labeling Result: {dataset_info}")

        # Upload model weights and data.yaml
        if os.path.exists(artifacts_dir):
            os.walk(artifacts_dir, topdown=True)
            for root, _, files in os.walk(artifacts_dir):

                await mongo_client.db["labeled_datasets"].insert_one(dataset_info.dict(by_alias=True))

                for file in files:
                    file_ext = file.split('.')[-1]

                    if file_ext in ["jpg", "png", "jpeg"]:
                        key = f"datasets/labeled/{did}/images/{file}"
                        type = "image"

                    elif file_ext in ["json", "txt"]:
                        key = f"datasets/labeled/{did}/labels/{file}"
                        type = "label"

                    elif file_ext in ["yaml", "yml"]:
                        key = f"datasets/labeled/{did}/{file}"
                        type = "yaml"

                    else:
                        key = f"datasets/labeled/{did}/others/{file}"
                        type = "other"

                    file_path = os.path.join(root, file)

                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                        
                        await minio_client.upload_files(uid, file_bytes, key)

                    data_info = LabeledDataInfo(
                    _id = uid+did+file,
                    uid = uid,
                    did = did,
                    name=file,
                    dataset = name,
                    type = type,
                    file_format = file.split(".")[1],
                    origin_raw = rawdid, # 원본 Raw Dataset ID
                    path = key,# MinIO 버킷 경로
                    created_at = params.completed_time # ISODate
                    )

                    await mongo_client.db["labeled_data"].insert_one(data_info.dict(by_alias=True))
                    await mongo_client.db["labeled_datasets"].update_one(
                        {"_id": uid+did},
                        {"$inc": {"total": 1}}
                    )

                    logger.info(f"Uploaded {file} to {key} in MinIO")
            # for root, _, files in os.walk(workdir):
            #     for file in files:
            #         key = os.path.join("artifacts", pid, "labeling", tid, file)
            #         if file == "data.yaml":
            #             file_path = os.path.join(root, file)
            #             with open(file_path, "rb") as f:
            #                 file_bytes = f.read()
            #             await minio_client.upload_files(uid, file_bytes, key)
            #             logger.info(f"Uploaded {file_path} to {key} in MinIO")

    except Exception as e:
        logger.error(f"Failed to upload artifacts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload artifacts: {e}")







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
























