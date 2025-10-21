import pandas as pd

DATASET_PATH = "dataset/celeba"

def get_bbox_df():
    bbox_df = pd.read_csv(DATASET_PATH+"/list_bbox_celeba.csv")
    bbox_df.head()
    return bbox_df


if __name__ == "__main__":
    print(get_bbox_df())