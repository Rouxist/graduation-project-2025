import os
import cv2
import pandas as pd
from ultralytics import YOLO

def masking(model_path, csv_path, input_path, output_path, target_class, threshold=0.5):
  model = YOLO(model_path, verbose=False)
  df = pd.read_csv(csv_path)

  img_list = df['image_path'].unique()
  
  for img in img_list:
    full_input_path = os.path.join(input_path, img)
    full_output_path = os.path.join(output_path, img)

    if not os.path.exists(full_input_path):
        continue
    
    if os.path.exists(full_output_path):
        continue

    img = cv2.imread(full_input_path)

    results = model(img)[0]
    img_update = results.orig_img.copy()
    names = results.names

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = names[cls]

        if label in target_classes and conf > threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            img_update[y1:y2, x1:x2] = 0

if __name__ == "__main__":
    model_path = "models/yolov5su.pt"
    csv_path = "data/quilt_1m.csv" #replace with test, train and val
    input_dir = "images/quilt_1m"
    output_dir = "images/masked_images"
    target_classes = ['person']

    mask_person_in_images(
        model_path=model_path,
        csv_path=csv_path,
        input_dir=input_dir,
        output_dir=output_dir,
        target_classes=target_classes
    )
