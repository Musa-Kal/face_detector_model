from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from utils import prep_image, get_bbox_df
from model_setup import MODEL_FOLDER, IMAGE_PATH, IMG_SIZE


def load_model_from_file(path):
    if not os.path.exists(path):
        raise FileExistsError(f"Model not found at {path}")
    return load_model(path)


def visualize_prediction(model, img_path, box_df=None):

    if not os.path.exists(img_path):
        print(f"!!! Image not Found at {img_path} !!!")
        return
    
    img = cv2.imread(img_path)
    
    prepared_img, original_img_height, original_img_width = prep_image(img_path)

    pred = model.predict(np.expand_dims(prepared_img / 255, axis=0))[0]

    print(" Predicted (normalized): ", pred)

    x, yb, bw, bh, conf = pred * [original_img_width, original_img_height, original_img_width, original_img_height, 1]

    img_drawn = img.copy()
    if conf > 0.5:
        if (box_df is not None):
                true_x = box_df['x_1']
                true_y = box_df['y_1']
                true_w = box_df['width']
                true_h = box_df['height']
                cv2.rectangle(img_drawn, (int(true_x), int(true_y)), (int(true_x+true_w), int(true_y+true_h)), (255,0,0), 2)
        cv2.rectangle(img_drawn, (int(x), int(yb)), (int(x+bw), int(yb+bh)), (0,255,0), 2)
        cv2.putText(img_drawn, f"conf={conf:.2f}", (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    else:
        cv2.putText(img_drawn, "no face", (10,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    # cv2.imshow('CIFAR-10 Image', img_drawn)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.imshow(img_drawn)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    print("loading...")
    model = load_model_from_file(os.path.join(MODEL_FOLDER, "face_detector_model.keras"))
    print("=== MODEL LOADED ===")
    data = get_bbox_df().iloc[1]
    #visualize_prediction(model, os.path.join(IMAGE_PATH, data["image_id"]), data)
    img = np.expand_dims(np.random.randint(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) / 255, axis=0)
    print(model.predict(img))