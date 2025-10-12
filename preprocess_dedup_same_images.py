import pandas as pd
from PIL import Image
import numpy as np
from ast import literal_eval

def images_are_same(img1_path, img2_path, fraction_threshold=0.01, pixel_threshold=20):
    """
    Compare two images to check if they are 'the same' (ignoring small differences).
    """
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    if img1.size != img2.size:
        return False  # different size => not same

    arr1 = np.array(img1)
    arr2 = np.array(img2)

    diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16))
    diff_gray = np.max(diff, axis=2)
    differing_pixels = np.count_nonzero(diff_gray >= pixel_threshold)
    total_pixels = diff_gray.size
    diff_fraction = differing_pixels / total_pixels

    return diff_fraction <= fraction_threshold


def deduplicate_images(csv_path, out_csv_path, img_dir, fraction_threshold=0.01, pixel_threshold=20):
    """
    Deduplicate images in a CSV grouped by video_id, keeping one representative image
    and merging 'medical_text' lists for duplicates.
    """
    df = pd.read_csv(csv_path)

    # Convert medical_text column from string to Python list (if stored as string)
    if isinstance(df.loc[0, "medical_text"], str):
        df["medical_text"] = df["medical_text"].apply(literal_eval)

    results = []

    for video_id, group in df.groupby("video_id"):
        group = group.reset_index(drop=True)
        used = [False] * len(group)

        for i in range(len(group)):
            if used[i]:
                continue

            base_img = group.loc[i, "image_path"]
            print("Current base_img =", base_img)
            merged_texts = set(group.loc[i, "medical_text"])
            used[i] = True

            for j in range(i + 1, len(group)):
                if used[j]:
                    continue

                other_img = group.loc[j, "image_path"]
                if images_are_same(img_dir+base_img, img_dir+other_img,
                                   fraction_threshold=fraction_threshold,
                                   pixel_threshold=pixel_threshold):
                    merged_texts.update(group.loc[j, "medical_text"])
                    used[j] = True

            results.append({
                "video_id": video_id,
                "image_path": base_img,
                "medical_text": list(merged_texts)
            })

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv_path, index=False)
    return out_df


if __name__ == "__main__":
    INPUT_FILE_DIR  = "example_input.csv"
    OUTPUT_FILE_DIR = "example_output.csv"
    IMAGE_DIR       = "example_images/"
    
    deduped = deduplicate_images(INPUT_FILE_DIR, OUTPUT_FILE_DIR,
                                 IMAGE_DIR,
                                 fraction_threshold=0.05,  # tolerate 0.5% pixels
                                 pixel_threshold=20)        # ignore tiny differences
    print("Done")