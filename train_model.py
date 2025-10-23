import numpy as np
import os
from tensorflow.keras import layers, models
from utils import get_bbox_df, prep_image
from tqdm import tqdm
from model_setup import NUM_SAMPLES, MODEL_FOLDER, IMAGE_PATH


bbox_df = get_bbox_df()

images = []
labels = []

for i in tqdm(range(NUM_SAMPLES)):
    image_bbox = bbox_df.iloc[i]
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
print("=== BUIILDING MODEL ===")

def build_model(input_size=(128,128,3)):
    # shaped like this cause why not 
    inputs = layers.Input(shape=input_size)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(5, activation='sigmoid')(x)
    return models.Model(inputs, x)


model = build_model()
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
with open(os.path.join(MODEL_FOLDER,"training_history,json"), 'w') as f:
    json.dump(history.history, f)

print(f"TRAINING DONE - MODEL SAVED IN ./{MODEL_FOLDER}")