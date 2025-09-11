import os
import random
import shutil
import logging

logger = logging.getLogger("yolo_utils")

def match_image_label_files(images_dir, labels_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}
    matched_basenames = image_basenames & label_basenames
    return list(matched_basenames)

def split_dataset(basenames, split_ratio=(0.8, 0.2, 0.0)):
    random.shuffle(basenames)
    n_total = len(basenames)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    train = basenames[:n_train]
    val = basenames[n_train:n_train+n_val]
    test = basenames[n_train+n_val:]
    return train, val, test

def write_split_txt(images_dir, basenames, split_name, output_dir):
    txt_path = os.path.join(output_dir, f"{split_name}.txt")
    with open(txt_path, "w") as f:
        for name in basenames:
            for ext in ['.jpg', '.png', '.jpeg']:
                img_file = name + ext
                img_path = os.path.join(images_dir, img_file)  # 상대경로로 기록
                if os.path.exists(os.path.join(images_dir, img_file)):
                    f.write(img_path + "\n")
                    break


def write_rawdata_txt(workdir, dataset_id,  artifact_path):
    image_files = [f for f in os.listdir(dataset_id) if f.endswith(('.jpg', '.png', '.jpeg'))]

    txt_path = os.path.join(workdir, dataset_id,f"{dataset_id}.txt")
    with open(txt_path, "w") as f:
        for name in image_files:

            img_path = os.path.join(dataset_id, name)  # 상대경로로 기록
            f.write(img_path + "\n")
    return txt_path
