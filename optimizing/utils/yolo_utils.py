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
