import cv2
import numpy as np
import pandas as pd
import os
import shutil

def avg_color(img_path, lower_range, upper_range, threshold = 0.01):
  
  img = cv2.imread(img_path)
  img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  mask = cv2.inRange(img_hsv, lower_range, upper_range)
  ratio = np.sum(mask > 0) / mask.size

  return ratio > threshold

def img_path_checker(csv_path, input_folder):
  
  df = pd.read_csv(csv_path)  
  img_paths = []
    
  for relative_path in df['image_path']:
    full_path = os.path.join(input_folder, str(relative_path))
    if os.path.exists(full_path):
      img_paths.append(full_path)

  return img_paths

def is_textured(img_path, variance_threshold=800):
  
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  if img is None:
    return False
  variance = cv2.Laplacian(img, cv2.CV_64F).var()
  
  return variance > variance_threshold

def filter_images(image_paths, output_folder, lower_hsv, upper_hsv, variance_threshold=800):

  os.makedirs(output_folder, exist_ok=True)

  for img_path in image_paths:
    if avg_color(img_path, lower_hsv, upper_hsv) and is_textured(img_path, variance_threshold):
      filename = os.path.basename(img_path)
      shutil.copy2(img_path, os.path.join(output_folder, filename))

if __name__ == "__main__":
  csv_path = "data/quilt_1m.csv" #replace with test, train and val
  input_folder = "data/masked_images"
  output_folder = "data/purple_images"

  #adjust according to desired color value
  lower_hsv = np.array([35, 100, 100])
  upper_hsv = np.array([85, 255, 255])

valid_paths = img_path_checker(csv_path, input_folder)
filter_images(valid_paths, output_folder, lower_hsv, upper_hsv)
