import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from load_data import get_bbox_df


IMG_SIZE = 128
NUM_SAMPLES = 1#2000


bbox_df = get_bbox_df()

images = []
labels = []

for i in range(NUM_SAMPLES):
    image_bbox = bbox_df.iloc[i]
    img_path = os.path.join("dataset\celeba\img_align_celeba\img_align_celeba",image_bbox["image_id"])

    if not os.path.exists(img_path):
        print("!!! Image not Found !!!")
        print(image_bbox)
        continue

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalizing bbox
    x = image_bbox["x_1"] / img_width
    y = image_bbox["y_1"] / img_height
    width = image_bbox["width"] / img_width
    height = image_bbox["height"] / img_height
    conf = 1.0

    # adding data
    images.append(img / 255)
    labels.append((x, y, width, height, conf))

X = np.array(images, dtype=np.float32)
y_ = np.array(labels, dtype=np.float32)

print("=== DATA PROCESSED ===")