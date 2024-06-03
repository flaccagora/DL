import os
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from model_kan import GPT as KAN
from model_kan import GPTConfig as KANConfig

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\nHello, how are you" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
architecture = 'KAN'
attn = 'Linear_Attn'
#exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------
architecture == 'KAN'
GPTConfig = KANConfig
GPT = KAN
out_dir = 'checkpoints'
# -----------------------------------------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
#model.load_state_dict(state_dict)





rename_map = {
    "transformer.h.0.mlp.c_fc.layers.0.base_weight": "transformer.h.0.mlp.c_fc.act_fun.0.scale_base",
    "transformer.h.0.mlp.c_fc.layers.0.grid": "transformer.h.0.mlp.c_fc.act_fun.0.grid",
    "transformer.h.0.mlp.c_fc.layers.0.spline_weight": "transformer.h.0.mlp.c_fc.act_fun.0.coef",
    "transformer.h.0.mlp.c_fc.layers.0.spline_scaler": "transformer.h.0.mlp.c_fc.act_fun.0.scale_sp",
    
    "transformer.h.0.mlp.c_proj.layers.0.base_weight": "transformer.h.0.mlp.c_proj.act_fun.0.scale_base",
    "transformer.h.0.mlp.c_proj.layers.0.grid": "transformer.h.0.mlp.c_proj.act_fun.0.grid",
    "transformer.h.0.mlp.c_proj.layers.0.spline_weight": "transformer.h.0.mlp.c_proj.act_fun.0.coef",
    "transformer.h.0.mlp.c_proj.layers.0.spline_scaler": "transformer.h.0.mlp.c_proj.act_fun.0.scale_sp",
    
    "transformer.h.1.mlp.c_fc.layers.0.base_weight": "transformer.h.1.mlp.c_fc.act_fun.0.scale_base",
    "transformer.h.1.mlp.c_fc.layers.0.grid": "transformer.h.1.mlp.c_fc.act_fun.0.grid",
    "transformer.h.1.mlp.c_fc.layers.0.spline_weight": "transformer.h.1.mlp.c_fc.act_fun.0.coef",
    "transformer.h.1.mlp.c_fc.layers.0.spline_scaler": "transformer.h.1.mlp.c_fc.act_fun.0.scale_sp",
    
    "transformer.h.1.mlp.c_proj.layers.0.base_weight": "transformer.h.1.mlp.c_proj.act_fun.0.scale_base",
    "transformer.h.1.mlp.c_proj.layers.0.grid": "transformer.h.1.mlp.c_proj.act_fun.0.grid",
    "transformer.h.1.mlp.c_proj.layers.0.spline_weight": "transformer.h.1.mlp.c_proj.act_fun.0.coef",
    "transformer.h.1.mlp.c_proj.layers.0.spline_scaler": "transformer.h.1.mlp.c_proj.act_fun.0.scale_sp",

    # Add other attribute mappings here
}


reshape_params = {
    "transformer.h.1.mlp.c_fc.layers.0.base_weight": [-1],
    "transformer.h.1.mlp.c_fc.layers.0.spline_weight": [3072*768, 6],
    "transformer.h.1.mlp.c_fc.layers.0.spline_scaler": [-1],
    "transformer.h.0.mlp.c_fc.layers.0.base_weight": [-1],
    "transformer.h.0.mlp.c_fc.layers.0.spline_weight": [3072*768, 6],
    "transformer.h.0.mlp.c_fc.layers.0.spline_scaler":[-1],
    "transformer.h.1.mlp.c_proj.layers.0.base_weight": [-1],
    "transformer.h.1.mlp.c_proj.layers.0.spline_weight": [3072*768, 6],
    "transformer.h.1.mlp.c_proj.layers.0.spline_scaler": [-1],
    "transformer.h.0.mlp.c_proj.layers.0.base_weight": [-1],
    "transformer.h.0.mlp.c_proj.layers.0.spline_weight": [3072*768, 6],
    "transformer.h.0.mlp.c_proj.layers.0.spline_scaler": [-1],
}


def update_state_dict_keys(state_dict, rename_map, reshape_params):
    """
    Update state dictionary keys according to the rename map.
    Rename specified keys and copy over all other keys unchanged.

    Args:
        state_dict (dict): The state dictionary with original parameter names.
        rename_map (dict): A dictionary mapping original parameter names to new parameter names.

    Returns:
        dict: A new state dictionary with updated keys.
    """
    updated_state_dict = {}
    for old_key, value in state_dict.items():
        if old_key == "transformer.h.1.mlp.c_fc.layers.0.grid" or old_key=="transformer.h.0.mlp.c_fc.layers.0.grid" or old_key=="transformer.h.1.mlp.c_proj.layers.0.grid" or old_key=="transformer.h.0.mlp.c_proj.layers.0.grid":  # Exclude the "grid" parameter
            continue
        new_key = rename_map.get(old_key, old_key)  # Rename if in map, otherwise keep original key
        if old_key in reshape_params:
            new_shape = reshape_params[old_key]
            value = value.view(*new_shape)
        updated_state_dict[new_key] = value
    return updated_state_dict



new_state_dict = update_state_dict_keys(state_dict, rename_map, reshape_params)
model.load_state_dict(new_state_dict, strict=False)

import tiktoken
enc = tiktoken.get_encoding("gpt2")

encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

x = 'Hello, how are you doing today?\nI am doing well'
encoded_x = encode(x)
new_x=(torch.tensor(encoded_x, dtype=torch.long, device=device)[None, ...])


model(new_x)