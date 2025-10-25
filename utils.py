import pandas as pd
import pickle
import os
import cv2
from model_setup import IMG_SIZE, CELEBA_DATASET_PATH, CIFAR_DATASET_PATH



def get_bbox_df():
    bbox_df = pd.read_csv(os.path.join(CELEBA_DATASET_PATH,"list_bbox_celeba.csv"))
    bbox_df.head()
    return bbox_df

def prep_image(img_path, IMG_SIZE=IMG_SIZE):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = img.shape

    return cv2.resize(img, (IMG_SIZE, IMG_SIZE)), img_height, img_width

def load_cifar10_batch(batch_name='data_batch_1'):
    with open(os.path.join(CIFAR_DATASET_PATH, batch_name), 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']        # shape: (10000, 3072)
    labels = batch['labels']    # list of 10000 labels
    # Reshape: each image is 32x32x3
    images = data.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)  # shape → (N, 32, 32, 3)
    return images, labels


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # print(get_bbox_df())
    img = load_cifar10_batch()[0][0]  # first image
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)           # convert color channels
    img_resized = cv2.resize(img_bgr, (32, 32))
    cv2.imshow('CIFAR-10 Image', img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()