import os
import torch
import open_clip
from PIL import Image
import skimage.io as io
import json
from tqdm import tqdm
import pickle

def main():
    out_path_train = f"./data/quilt1m/oscar_split_train.pkl"
    out_path_val = f"./data/quilt1m/oscar_split_val.pkl"
    out_path_test = f"./data/quilt1m/oscar_split_test.pkl"
    
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    ## Train dataset
    with open('./data/quilt1m/train_caption.json', 'r') as f:
        data_train = json.load(f)

    all_embeddings_train = []
    all_captions_train = []
    for i in tqdm(range(len(data_train))):
        d = data_train[i]
        img_id = d["image_id"]
        filename = f"./data/quilt1m/train/{img_id}.jpg"
        image = io.imread(filename)
        image = preprocess_train(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings_train.append(prefix)
        all_captions_train.append(d)
        if (i + 1) % 5000 == 0:
            with open(out_path_train, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings_train, dim=0), "captions": all_captions_train}, f)

    with open(out_path_train, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings_train, dim=0), "captions": all_captions_train}, f)

    print("(Train set) Done, %0d embeddings are saved.\n" % len(all_embeddings_train))


    ## Validation dataset
    with open('./data/quilt1m/val_caption.json', 'r') as f:
        data_val = json.load(f)

    all_embeddings_val = []
    all_captions_val = []
    for i in tqdm(range(len(data_val))):
        d = data_val[i]
        img_id = d["image_id"]
        filename = f"./data/quilt1m/val/{img_id}.jpg"
        image = io.imread(filename)
        image = preprocess_val(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings_val.append(prefix)
        all_captions_val.append(d)
        if (i + 1) % 5000 == 0:
            with open(out_path_val, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings_val, dim=0), "captions": all_captions_val}, f)

    with open(out_path_val, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings_val, dim=0), "captions": all_captions_val}, f)

    print("(Validation set) Done, %0d embeddings are saved.\n" % len(all_embeddings_val))


    ## Test dataset
    with open('./data/quilt1m/test_caption.json', 'r') as f:
        data_test = json.load(f)

    all_embeddings_test = []
    all_captions_test = []
    for i in tqdm(range(len(data_test))):
        d = data_test[i]
        img_id = d["image_id"]
        filename = f"./data/quilt1m/test/{img_id}.jpg"
        image = io.imread(filename)
        image = preprocess_val(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = model.encode_image(image).cpu()
        d["clip_embedding"] = i
        all_embeddings_test.append(prefix)
        all_captions_test.append(d)
        if (i + 1) % 5000 == 0:
            with open(out_path_test, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings_test, dim=0), "captions": all_captions_test}, f)

    with open(out_path_test, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings_test, dim=0), "captions": all_captions_test}, f)

    print("(Test set) Done, %0d embeddings are saved." % len(all_embeddings_test))

    return 0

if __name__ == '__main__':
    main()
