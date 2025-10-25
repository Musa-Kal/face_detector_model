import os

IMG_SIZE = 128
NUM_SAMPLES = 20000
SAMPLE_OFFSET = 0
FACE_IMAGE_SPLIT_RATIO = .45

MODEL_FOLDER = "model"
TRAINED_MODEL_PATH = None#os.path.join(MODEL_FOLDER, "face_detector_model.keras")
CELEBA_DATASET_PATH = os.path.join("dataset", "celeba")
CIFAR_DATASET_PATH = os.path.join("dataset", "cifar10_data")
IMAGE_PATH = os.path.join(CELEBA_DATASET_PATH, "img_align_celeba", "img_align_celeba")