import pandas as pd
import os
import cv2

DATASET_PATH = "dataset/celeba"

def get_bbox_df():
    bbox_df = pd.read_csv(os.path.join(DATASET_PATH,"list_bbox_celeba.csv"))
    bbox_df.head()
    return bbox_df

def prep_image(img_path, IMG_SIZE=128):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape

    return cv2.resize(img, (IMG_SIZE, IMG_SIZE)), img_height, img_width


if __name__ == "__main__":
    print(get_bbox_df())