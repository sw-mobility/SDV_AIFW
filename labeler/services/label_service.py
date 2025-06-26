from fastapi import UploadFile, File, Form, Request
from fastapi.responses import PlainTextResponse
from typing import List, Optional
from datetime import datetime
import os
import shutil
import cv2
import yaml
from ultralytics import YOLO
import glob
import requests
import httpx

EXPORT_DIR = "/app/workspace"
CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.1))

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# os.makedirs(EXPORT_DIR, exist_ok=True)

def get_model(model_name):
    # Always resolve the model path relative to this file's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, '../models')
    model_path = os.path.join(models_dir, f"{model_name}.pt")
    model_path = os.path.abspath(model_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Current working directory: {os.getcwd()}")
    return YOLO(model_path)

async def get_script_service(path: str):
    if not os.path.isfile(path):
        return PlainTextResponse("File not found", status_code=404)
    with open(path, "r") as f:
        return PlainTextResponse(f.read())

async def save_script_service(request: Request):
    data = await request.json()
    path = data["path"]
    code = data["code"]
    with open(path, "w") as f:
        f.write(code)
    return {"status": "ok"}

async def label_images_service(
    files: List[UploadFile] = File(...),
    model: str = Form(...),
    dataset_name: str = Form(...),
    do_split: bool = Form(False),
    train_percent: float = Form(0.8),
    result_name: Optional[str] = Form(None)
):
    zip_file = files[0]
    extract_dir = await extract_zip_to_dir(zip_file, EXPORT_DIR)

    # 2. 이미지 파일 경로 수집 (모든 하위 폴더 포함)
    image_paths = [
        y for x in [glob.glob(os.path.join(root, '*')) for root, _, _ in os.walk(extract_dir)]
        for y in x if y.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_paths:
        return {"status": "no images found", "output_dir": extract_dir}

    # 3. 결과 디렉토리 생성
    if result_name:
        batch_dir = os.path.join(EXPORT_DIR, result_name)
    else:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_dir = os.path.join(EXPORT_DIR, f"{dataset_name}_{now_str}")
    images_dir = os.path.join(batch_dir, "images")
    labels_dir = os.path.join(batch_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # 4. YOLO 추론 및 라벨링 (이하 기존 로직)
    model_instance = get_model(model)
    results = model_instance(image_paths)

    print(f"[DEBUG] CONFIDENCE_THRESHOLD: {CONFIDENCE_THRESHOLD} (type: {type(CONFIDENCE_THRESHOLD)})")
    # Save images and labels, collect used class IDs
    used_class_ids = set()
    img_label_pairs = []
    for img_path, result in zip(image_paths, results):
        boxes = result.boxes
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] cv2.imread failed for {img_path}")
            continue
        base = os.path.splitext(os.path.basename(img_path))[0]
        txt_filename = base + '.txt'
        h, w = img.shape[:2]
        lines = []
        # Robust YOLO detection loop (like label_cars.py)
        for box in boxes:
            conf = float(box.conf)
            print(f"[DEBUG] DETECTION: conf={conf}, threshold={CONFIDENCE_THRESHOLD}")
            if conf >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                used_class_ids.add(class_id)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        if not lines:
            print(f"[WARNING] No detections found for {img_path}.")
        print(f"[DEBUG] {img_path}: {len(lines)} detections, used_class_ids so far: {used_class_ids}, lines: {lines}")
        img_label_pairs.append((img_path, img, txt_filename, lines))

    # Remap class IDs to 0...N-1 and update label files
    used_class_ids = sorted(used_class_ids)
    class_id_map = {old: new for new, old in enumerate(used_class_ids)}
    class_names = [COCO_NAMES[i] for i in used_class_ids]
    print(f"[DEBUG] Final used_class_ids: {used_class_ids}")
    print(f"[DEBUG] Final class_names: {class_names}")
    if not used_class_ids:
        print("[WARNING] No classes detected in any image. No labels will be written and data.yaml will be empty.")
    # Save images and labels in YOLOv8 format (no split)
    for img_path, img, txt_filename, lines in img_label_pairs:
        cv2.imwrite(os.path.join(images_dir, os.path.basename(img_path)), img)
        txt_path = os.path.join(labels_dir, txt_filename)
        new_lines = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if int(parts[0]) in class_id_map:
                    parts[0] = str(class_id_map[int(parts[0])])
                new_lines.append(' '.join(parts))
        print(f"[DEBUG] Writing to {txt_path}: {new_lines}")
        with open(txt_path, 'w') as f:
            if new_lines:
                f.write('\n'.join(new_lines) + '\n')

    # Write data.yaml
    data_yaml = {
        'train': 'images',
        'val': 'images',
        'nc': len(class_names),
        # 'names' will be written separately for flat list
    }
    yaml_path = os.path.join(batch_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
        # Write 'names' as a flat list (flow style)
        f.write('names: ')
        yaml.dump(class_names, f, default_flow_style=True)

    print(f"[INFO] data.yaml written at: {yaml_path}")
    # Zip the output folder with the user-given dataset name or result_name
    zip_base = os.path.join(EXPORT_DIR, result_name or dataset_name)
    zip_path = shutil.make_archive(zip_base, 'zip', batch_dir)
    print(f"[INFO] Zip file created: {zip_path}")
    print(f"[INFO] Zip file exists: {os.path.exists(zip_path)}, size: {os.path.getsize(zip_path) if os.path.exists(zip_path) else 'N/A'} bytes")
    print(f"[INFO] Notifying API to upload zip: {zip_path}, result_name: {result_name}")
    # --- AUTOMATION: Notify API to upload zip to MinIO ---
    notify_result = await notify_api_upload_zip(
        zip_path=zip_path,  # Make sure this path is accessible by the API container (use shared volume if needed)
        dataset_name=dataset_name,
        result_name=result_name,  # 추가: result_name 전달
        bucket_name="jwlee",         # <-- Use your actual bucket name
        minio_access_key="minioadmin",   # <-- Use your actual MinIO credentials
        minio_secret_key="minioadmin123" # <-- Use your actual MinIO credentials
    )
    print(f"[INFO] API upload result: {notify_result}")
    # -----------------------------------------------------

    return {
        "status": "success",
        "output_dir": batch_dir,
        "zip_path": zip_path,
        "minio_upload": notify_result
    }

async def extract_zip_to_dir(files: UploadFile, export_dir: str):
    """
    zip 파일을 임시 디렉토리에 저장하고 압축 해제하여, 압축 해제된 경로를 반환합니다.
    """
    import tempfile, zipfile
    temp_dir = tempfile.mkdtemp(dir=export_dir)
    filesname = files.filename or "uploaded.zip"
    zip_path = os.path.join(temp_dir, filesname)
    with open(zip_path, "wb") as f:
        shutil.copyfileobj(files.file, f)
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return extract_dir

async def notify_api_upload_zip(
    zip_path, dataset_name, result_name, bucket_name,
    minio_access_key, minio_secret_key,
    api_url="http://api-server:5002/labeler/handle/upload-zip-to-minio"
):
    # Send the actual zip file as multipart/form-data, not just the path
    async with httpx.AsyncClient(timeout=60.0) as client:
        with open(zip_path, "rb") as f:
            files = {"files": (os.path.basename(zip_path), f, "application/zip")}
            data = {
                "result_name": result_name,
            }
            response = await client.post(api_url, files=files, data=data)
    return response.json()