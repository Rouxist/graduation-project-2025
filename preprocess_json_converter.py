import pandas as pd
import ast
import os
import json


def main(file_name):
    df = pd.read_csv(file_name + ".csv")
    df = df[['image_path', 'medical_text']]

    # convert 'medical_text' from string to list
    df["medical_text"] = df["medical_text"].apply(ast.literal_eval)

    # explode into multiple rows
    df_exploded = df.explode("medical_text", ignore_index=True)

    # rename columns
    df_exploded = df_exploded.rename(columns={
        "image_path": "image_id",
        "medical_text": "caption"
    })

    # strip file extension from image_id
    df_exploded["image_id"] = df_exploded["image_id"].apply(lambda x: os.path.splitext(x)[0]) # it removes the string '.jpg' at the end

    # assign a unique id for each row (starting from 1)
    df_exploded["id"] = range(1, len(df_exploded) + 1)

    # reorder columns
    df_exploded = df_exploded[["image_id", "id", "caption"]]

    # save as JSON
    result = df_exploded.to_dict(orient="records")
    with open(FILE_NAME + "_exploded.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("[Preview] first 5 rows")
    print(json.dumps(result[:5], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    FILE_NAME = "example"

    main(FILE_NAME)