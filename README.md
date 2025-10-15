# HistoClipCap

## Description

This repository contains the work for the 2025 graduation project at the Department of Data Science, Hanyang University.

\*\*This work applies [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) on histopathology domain.

\*\*We used [Quilt1M](https://github.com/wisdomikezogwo/quilt1m) dataset to utilize large amount of histopathology image-text pair.

\*\*It combines [Quilt-B-32](https://huggingface.co/wisdomik/QuiltNet-B-32) and [BioGPT](https://github.com/microsoft/BioGPT) to handle histopathology domain with ClipCap.

This project was conducted by [Ammar Rais](https://www.linkedin.com/in/ammar-rais/) and [Yongjin Kang](https://www.linkedin.com/in/yongjin-kang/).

## Environment Setup

\*\* This environment covers libraries required for `parse_image.py` and `train.py'. Other scripts may require additional libraries.

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

\*\*First, encode images using `parse_image.py`.

```
python parse_image.py
```

\*\*Then, fine-tune like this.

```
python train.py --data_train ./Data/quilt1m/oscar_split_train.pkl --out_dir ./quilt1m_train/
```

## Examples

## Others

Final report will be added.
