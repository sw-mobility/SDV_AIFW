import os
import random
import shutil
import logging

logger = logging.getLogger("yolo_utils")

yolo11_default_parameters = {
    "model": "yolo11n",
    "epochs": 5,
    "img_size": 416,
    "batch_size": 16,
    "learning_rate": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "optimizer": "SGD",
    "device": "cuda",
    "workers": 8,
    "project": "runs/train",
    "name": "exp",
    "exist_ok": False,
    "patience": 100,
    "seed": 42,
    "rect": False,
    "resume": False,
    "nosave": False,
    "noval": False,
    "noautoanchor": False,
    "noplots": False,
    "cache": False,
    "image_weights": False,
    "multi_scale": False,
    "single_cls": False,
    "adam": False,
    "sync_bn": False,
    "half": False,
    "label_smoothing": 0.0,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "fl_gamma": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0
}
async def match_image_label_files(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}
    matched_basenames = image_basenames & label_basenames
    return list(matched_basenames)

async def split_dataset(basenames, split_ratio=(0.8, 0.2, 0.0)):
    random.shuffle(basenames)
    n_total = len(basenames)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    train = basenames[:n_train]
    val = basenames[n_train:n_train+n_val]
    test = basenames[n_train+n_val:]
    return train, val, test

async def write_split_txt(images_dir, basenames, split_name, output_dir):
    txt_path = os.path.join(output_dir, f"{split_name}.txt")
    with open(txt_path, "w") as f:
        for name in basenames:
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = name + ext
                img_path = os.path.join(images_dir, img_file)  # 상대경로로 기록
                if os.path.exists(os.path.join(images_dir, img_file)):
                    f.write(img_path + "\n")
                    break

async def dataset2model_mapping(label_path, image_path, dataset_classes, user_classes):
    """
    user_classes에 없는 클래스 라인은 삭제, 남은 라벨이 하나도 없으면 라벨 파일과 매칭 이미지까지 삭제
    label_path: 라벨 파일들이 있는 디렉토리
    images_path: 이미지 파일들이 있는 디렉토리
    dataset_classes: 데이터셋 클래스 리스트
    user_classes: 실제 사용할 클래스 리스트
    """
    user_classes_lower = [c.lower() for c in user_classes]
    dataset_classes_lower = [c.lower() for c in dataset_classes]
    class_map = {}
    for i, cname in enumerate(dataset_classes_lower):
        if cname in user_classes_lower:
            class_map[i] = user_classes_lower.index(cname)

    for dirpath, dirnames, filenames in os.walk(label_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            new_lines = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    src_idx = int(parts[0])
                    if src_idx not in class_map:
                        continue  # user_classes에 없는 클래스는 완전히 삭제
                    dst_idx = class_map[src_idx]
                    parts[0] = str(dst_idx)
                    new_lines.append(" ".join(parts))
            if not new_lines:
                # 라벨이 하나도 남지 않으면 라벨 파일과 매칭 이미지 삭제
                os.remove(file_path)
                # 이미지 파일명 추정: 라벨 파일명에서 확장자만 바꿔서 찾기
                base = os.path.splitext(filename)[0]
                for ext in [".jpg", ".jpeg", ".png"]:
                    img_file = base + ext
                    img_path = os.path.join(image_path, img_file)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        break
            else:
                with open(file_path, "w", encoding="utf-8") as f:
                    for line in new_lines:
                        f.write(line + "\n")

# async def model2dataset_mapping(model_classes, dataset_classes):
#     """
#     모델 인덱스 → 데이터셋 인덱스 매핑 dict 반환
#     """
#     mapping = {}
#     for i, cname in enumerate(model_classes):
#         if cname in dataset_classes:
#             mapping[i] = dataset_classes.index(cname)
#         else:
#             mapping[i] = -1
#     return mapping

# async def dataset2model_mapping(user_classes, dataset_classes, model_classes):
#     """
#     데이터셋 인덱스 → 모델 인덱스 매핑 dict 반환
#     (model_classes에 없는 클래스는 자동 추가)
#     """
#     mapping = {}
#     # 비교용: model_classes를 소문자로 변환
#     model_classes_lower = [c.lower() for c in model_classes]
#     for i, cname in enumerate(dataset_classes):
#         cname_lower = cname.lower()
#         if cname_lower not in model_classes_lower:
#             model_classes.append(cname)
#             model_classes_lower.append(cname_lower)
#         mapping[i] = model_classes_lower.index(cname_lower)
#     return mapping, model_classes

# async def convert_yolo_label_file(label_path, mapping):
#     """
#     YOLO 라벨 파일의 클래스 인덱스를 매핑에 따라 변환
#     """
#     for dirpath, dirnames, filenames in os.walk(label_path):
#         for filename in filenames:
#             file_path = os.path.join(dirpath, filename)
#             new_lines = []  # 파일마다 초기화
#             with open(file_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if not parts:
#                         continue
#                     src_idx = int(parts[0])
#                     dst_idx = mapping.get(src_idx, -1)
#                     if dst_idx == -1:
#                         continue  # 매핑 안 되는 클래스는 건너뜀
#                     parts[0] = str(dst_idx)
#                     new_lines.append(" ".join(parts))
#             with open(file_path, "w", encoding="utf-8") as f:
#                 for line in new_lines:
#                     f.write(line + "\n")
#             logger.info(f"Converted labels in {file_path}")
