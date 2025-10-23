import os

IMG_SIZE = 128
NUM_SAMPLES = 20000
SAMPLE_OFFSET = 10000

MODEL_FOLDER = "model"
TRAINED_MODEL_PATH = os.path.join(MODEL_FOLDER, "face_detector_model.keras")
DATASET_PATH = os.path.join("dataset", "celeba")
IMAGE_PATH = os.path.join(DATASET_PATH, "img_align_celeba", "img_align_celeba")