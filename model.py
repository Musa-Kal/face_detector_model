from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from utils import prep_image
from model_setup import MODEL_FOLDER, IMAGE_PATH


def load_model_from_file(path):
    if not os.path.exists(path):
        raise FileExistsError(f"Model not found at {path}")
    return load_model(path)


def visualize_prediction(model, img_path):

    if not os.path.exists(img_path):
        print("!!! Image not Found !!!")
        return
    
    img = cv2.imread(img_path)
    
    prepared_img, original_img_height, original_img_width = prep_image(img_path)

    pred = model.predict(np.expand_dims(prepared_img / 255, axis=0))[0]

    print(" Predicted (normalized): ", pred)

    x, yb, bw, bh, conf = pred * [original_img_width, original_img_height, original_img_width, original_img_height, 1]

    img_drawn = img.copy()
    if conf > 0.5:
        cv2.rectangle(img_drawn, (int(x), int(yb)), (int(x+bw), int(yb+bh)), (0,255,0), 2)
        cv2.putText(img_drawn, f"conf={conf:.2f}", (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    else:
        cv2.putText(img_drawn, "no face", (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    plt.imshow(img_drawn)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print("loading...")
    model = load_model_from_file(os.path.join(MODEL_FOLDER, "face_detector_model.keras"))
    print("=== MODEL LOADED ===")
    visualize_prediction(model, os.path.join(IMAGE_PATH, "000001.jpg"))