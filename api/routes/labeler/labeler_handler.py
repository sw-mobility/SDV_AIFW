# sends request from labeler container to api to send and save in minio.
from fastapi import APIRouter, Form, HTTPException, UploadFile, File
import os
import zipfile
import tempfile
from core.storage import storage_client  # Use the async Minio client

router = APIRouter(prefix="/labeler/handle", tags=["Labeler Handler"])

@router.post("/upload-zip-to-minio")
async def upload_zip_to_minio(
    files: UploadFile = File(...),  # 'files'로 변경
    result_name: str = Form(...), # 'result_name'으로 변경
):
    print(f"[DEBUG] /labeler/handle/upload-zip-to-minio called. result_name={result_name}, files.filename={files.filename}")
    """
    Extract the zip output from the labeler and upload images, labels, and data.yaml to MinIO (async).
    """
    if not result_name or not result_name.strip():
        raise HTTPException(status_code=400, detail="result_name must not be empty or whitespace.")

    # Save the uploaded zip file to a temp location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await files.read()
        print(f"[DEBUG] Received zip file size: {len(content)} bytes")
        tmp.write(content)
        tmp_path = tmp.name

    if not os.path.exists(tmp_path):
        raise HTTPException(status_code=404, detail="Zip file not found.")

    uploaded_files = []
    try:
        with tempfile.TemporaryDirectory() as extract_dir:
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print(f"[DEBUG] Zip extracted to {extract_dir}")
            images_dir = os.path.join(extract_dir, "images")
            labels_dir = os.path.join(extract_dir, "labels")
            data_yaml = os.path.join(extract_dir, "data.yaml")

            minio_dir = result_name

            # Upload images
            if os.path.exists(images_dir):
                for root, _, files_ in os.walk(images_dir):
                    for file in files_:
                        file_path = os.path.join(root, file)
                        object_name = f"datasets/labeled/images/object_detection/{minio_dir}/images/{file}"
                        with open(file_path, "rb") as f:
                            await storage_client.put_object(
                                key=object_name,
                                body=f.read(),
                                content_type=None
                            )
                        uploaded_files.append(object_name)

            # Upload labels
            if os.path.exists(labels_dir):
                for root, _, files_ in os.walk(labels_dir):
                    for file in files_:
                        file_path = os.path.join(root, file)
                        object_name = f"datasets/labeled/images/object_detection/{minio_dir}/labels/{file}"
                        with open(file_path, "rb") as f:
                            await storage_client.put_object(
                                key=object_name,
                                body=f.read(),
                                content_type=None
                            )
                        uploaded_files.append(object_name)

            # Upload data.yaml
            if os.path.exists(data_yaml):
                object_name = f"datasets/labeled/images/object_detection/{minio_dir}/data.yaml"
                with open(data_yaml, "rb") as f:
                    await storage_client.put_object(
                        key=object_name,
                        body=f.read(),
                        content_type="text/yaml"
                    )
                uploaded_files.append(object_name)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"status": "uploaded", "minio_path": f"datasets/labeled/images/object_detection/{minio_dir}/", "files": uploaded_files}