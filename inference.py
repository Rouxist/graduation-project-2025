import open_clip
import os
import pickle
import warnings
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
from typing import Tuple, Optional
from transformers import BioGptTokenizer, BioGptForCausalLM
from tqdm import tqdm, trange
import skimage.io as io
import PIL.Image
import argparse
import time

T = torch.Tensor
D = torch.device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


## Model definition
class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class HistoClipCap(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.biogpt.embed_tokens(tokens) # replaced 'wte'/'transformer' of GPT2 with 'embed_tokens'/'biogpt' of BioGPT
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(HistoClipCap, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.gpt_embedding_size = self.gpt.biogpt.embed_tokens.weight.shape[1] # replaced 'transformer.wte' of GPT2 with 'biogpt.embed_tokens' of BioGPT
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))
    

def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67, # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.biogpt.embed_tokens(tokens) # replaced 'wte'/'transformer' of GPT2 with 'embed_tokens'/'biogpt' of BioGPT

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.biogpt.embed_tokens(next_token) # replaced 'wte'/'transformer' of GPT2 with 'embed_tokens'/'biogpt' of BioGPT
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                
                generated = torch.cat((generated, next_token_embed), dim=1) # This is input of gpt at next iteration

                if stop_token_index == next_token.item():
                    break

            print("tokens:",tokens)
            # output_list = list(tokens.squeeze().cpu().numpy())
            output_list = tokens.cpu().numpy().flatten().tolist() # to handle the cases where first predicted token is EOS
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


## Setup checkpoints
current_directory = os.getcwd()
save_path = os.path.join(current_directory, "pretrained_models")
os.makedirs(save_path, exist_ok=True)
model_path = os.path.join(save_path, 'histo_clipcap-008.pt')


if not os.path.isfile(model_path):
    raise FileNotFoundError("Checkpoints not found:", model_path)


## Setup model
clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:wisdomik/QuiltNet-B-32')

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")

prefix_length = 10

model = HistoClipCap(prefix_length)

checkpoints = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoints, strict=False) 
model = model.eval() 
model = model.to(device)


## Load image from test set
NAME = "path/to/image"

images_path = os.path.join(current_directory, "data/quilt1m")
os.makedirs(images_path, exist_ok=True)
UPLOADED_FILE = os.path.join(images_path, NAME)

if not os.path.isfile(UPLOADED_FILE):
    raise FileNotFoundError("Image not found:", UPLOADED_FILE)


## Main inference procedure
###### Using encoder ############################################################
# image = io.imread(UPLOADED_FILE)
# pil_image = PIL.Image.fromarray(image)
# pil_image.show()
# display(pil_image)
# image = preprocess_val(pil_image).unsqueeze(0).to(device)

# with torch.no_grad():
#     prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
#     prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
#################################################################################

###### Using encoded data #######################################################
# Since images were already encoded, it is enough to fetch from the .pkl file
pkl_path = "data/quilt1m/oscar_split_train.pkl"

with open(pkl_path, 'rb') as f:
    converted_images = pickle.load(f)

clip_embedding = converted_images["clip_embedding"]
captions = converted_images["captions"]

for i in captions:
    if i['image_id']+".jpg"== NAME:
        prefix = clip_embedding[i['clip_embedding']].unsqueeze(0)

with torch.no_grad():
    prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
#################################################################################

use_beam_search = False

if use_beam_search:
    pass
else:
    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed, entry_length=5)

print('\n\n======= Genearted caption ===========================================================================')
print(generated_text_prefix)
print('=====================================================================================================')
