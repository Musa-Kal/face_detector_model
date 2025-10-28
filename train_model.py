import numpy as np
import os
from tensorflow.keras import models
from model import FaceBBoxModel
from utils import get_bbox_df, prep_image
from tqdm import tqdm
from model_setup import NUM_SAMPLES, MODEL_FOLDER, IMAGE_PATH, SAMPLE_OFFSET, TRAINED_MODEL_PATH, FACE_IMAGE_SPLIT_RATIO, IMG_SIZE


bbox_df = get_bbox_df()

images = []
labels = []

for i in tqdm(range(NUM_SAMPLES)):

    if np.random.rand() < FACE_IMAGE_SPLIT_RATIO:
        random_img = np.random.randint(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        images.append(random_img / 255)
        labels.append((0, 0, 0, 0, 0))  # no face
        continue

    image_bbox = bbox_df.iloc[i+SAMPLE_OFFSET]
    img_path = os.path.join(IMAGE_PATH, f"{image_bbox['image_id']}")

    if not os.path.exists(img_path):
        print("!!! Image not Found !!!")
        print(img_path)
        continue

    prepared_img, original_img_height, original_img_width = prep_image(img_path)

    # normalizing bbox
    x = image_bbox["x_1"] / original_img_width
    y = image_bbox["y_1"] / original_img_height
    width = image_bbox["width"] / original_img_width
    height = image_bbox["height"] / original_img_height
    conf = 1.0

    # adding data
    images.append(prepared_img / 255)
    labels.append((x, y, width, height, conf))

X = np.array(images, dtype=np.float32)
y_ = np.array(labels, dtype=np.float32)

print("=== DATA PROCESSED ===")
print(len(X), "samples")
print(len(y_), "labels")


if (TRAINED_MODEL_PATH and os.path.exists(TRAINED_MODEL_PATH)):
    print("=== LOADING EXISTING MODEL ===")
    model = models.load_model(TRAINED_MODEL_PATH)
else:
    print("=== BUIILDING MODEL ===")
    model = FaceBBoxModel(input_size=(IMG_SIZE, IMG_SIZE, 3)).model
    model.compile(optimizer="adam", loss='mse')
    model.summary()

print("=== TRAINING MODEL ===")

history = model.fit(X, y_, epochs=15, batch_size=32, validation_split=0.2)

# Save weights
model.save_weights(os.path.join(MODEL_FOLDER,"model_weights.weights.h5"))

# Save full model
model.save(os.path.join(MODEL_FOLDER,"face_detector_model.keras"))

# Save training history
import json
with open(os.path.join(MODEL_FOLDER,"training_history.json"), 'w') as f:
    json.dump(history.history, f)

print(f"TRAINING DONE - MODEL SAVED IN ./{MODEL_FOLDER}")