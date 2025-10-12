# HistoClipCap

## Description

This repository contains the work for the 2025 graduation project at the Department of Data Science, Hanyang University.

This project was conducted by [Ammar Rais](https://www.linkedin.com/in/ammar-rais/) and [Yongjin Kang](https://www.linkedin.com/in/yongjin-kang/).

## Environment Setup

```
git clone https://github.com/Rouxist/graduation-project-2025
cd graduation-project-2025
conda env create -f environment.yml
conda activate histo_clipcap
```

## Dataset preprocessing

In the original dataset, some images are nearly identical with slight difference (e.g. a screenshot with/without a mouse cursor). The script `preprocess_dedup_same_images.py` removes such duplicates and merge their corresponding captions. It results a CSV file in which each row contains an "image_path" and a "medical_text". The "medical_text" is a list of single or multiple captions associated with the same image.

The output CSV file from `preprocess_dedup_same_images.py` is further processed by `preprocess_json_converter.py`, which separates the multiple captions of each image using the pandas .explode() function. It generates final .json file, which is later used by `parse_image.py`.

## Training

## Examples

## Others

Final report will be added.
