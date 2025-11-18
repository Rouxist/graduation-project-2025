import argparse
import csv
import json
import os
import pickle
import time
from typing import Optional
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as nnf

from transformers import BioGptTokenizer, BioGptForCausalLM
from model import MLP, TransformerMapper

T = torch.Tensor
D = torch.device

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

    def __init__(self, prefix_length: int, prefix_size: int = 512, mapping_type: str = 'transformer'):
        super(HistoClipCap, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
        self.gpt_embedding_size = self.gpt.biogpt.embed_tokens.weight.shape[1] # replaced 'transformer.wte' of GPT2 with 'biogpt.embed_tokens' of BioGPT

        if mapping_type == "transformer":
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length=10, num_layers=8)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))

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

        for entry_idx in range(entry_count):
            if embed is not None:
                # Manually putting bos
                embed = embed.to(device)
                bos_id = model.gpt.config.bos_token_id or tokenizer.bos_token_id or 0
                bos_tensor = torch.tensor([[bos_id]], device=device)
                bos_embed = model.gpt.biogpt.embed_tokens(bos_tensor)

                generated = torch.cat((embed, bos_embed), dim=1)
                tokens = bos_tensor
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

            output_list = tokens[0,1:].cpu().numpy().flatten().tolist() # to handle the cases where first predicted token is EOS
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='./pretrained_models/mlp_trial_id006-030.pt')
    parser.add_argument('--test_pkl_dir', default='./data/quilt1m/oscar_split_test.pkl')
    parser.add_argument('--caption_length', type=int, default=128, help='max # of tokens of the generated caption')
    parser.add_argument('--out_dir', default='./evaluation')
    parser.add_argument('--out_json_name', default='quilt_1m_test_small_pred.json')
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prefix_length = args.prefix_length
    pkl_path = args.test_pkl_dir
    caption_length = args.caption_length
    out_dir = args.out_dir
    out_json_name = args.out_json_name

    ## Setup model
    tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
    model = HistoClipCap(prefix_length)

    checkpoints = torch.load(args.ckpt_dir, map_location=device, weights_only=True)
    model.load_state_dict(checkpoints, strict=False) 
    model = model.eval() 
    model = model.to(device)

    ## Load image embedding vectors
    with open(pkl_path, 'rb') as f:
        converted_images = pickle.load(f)

    clip_embedding = converted_images["clip_embedding"]
    captions = converted_images["captions"]

    image_id_seen = set()
    unique_captions = []
    for item in captions:
        if item['image_id'] not in image_id_seen:
            unique_captions.append(item)
            image_id_seen.add(item['image_id'])

    ## Iterate test dataset info .csv file to generate caption of each image
    num_data = len(unique_captions)
    out_json_rows = []

    start_time = time.time()
    
    for image in tqdm(unique_captions, total=num_data):
        image_id = image["image_id"]
        idx = image['clip_embedding']
        prefix = clip_embedding[idx:idx+1].to(device)

        with torch.no_grad():
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
            generated_caption = generate2(model, tokenizer, embed=prefix_embed, entry_length=caption_length)

        out_json_rows.append({
        "image_id": image_id,
        "caption": generated_caption
        })

    print(f"\nEvaluation took {time.time()-start_time} seconds.")

    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, out_json_name), "w") as f:
        json.dump(out_json_rows, f, indent=2)

    print("Generated captions are successfully saved to:", os.path.join(out_dir, out_json_name))

if __name__ == '__main__':
    main()
