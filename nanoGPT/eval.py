from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
import torch
import os
import tiktoken
from contextlib import nullcontext

#sys.path.append("../../../nanokan")  # Add the path of the specific folder to import modules from
#sys.path.append("./")
from model import GPT, GPTConfig
from model_kan import GPT as KAN_GPT
from model_kan import GPTConfig as KAN_GPTConfig
import torch.nn.functional as F

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 10 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
architecture = 'mlp'
task = 'hellaswag'
limit = 100
batch_size = 1
bootstrap_iters = 0

exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


#################################

if architecture == 'Kan':
    GPT = KAN_GPT
    GPTConfig = KAN_GPTConfig
    out_dir = 'out_kan'
    k = 3
    grid = 5
    print("Using KAN architecture\n")

print("Loading checkpoint")
ckpt_path = os.path.join(out_dir, './ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])

#@register_model("MyCustom")
class MyCustomLM(LM):
    
    def __init__(self, model, batch_size) -> None:
        super().__init__()
        self.batch_size = batch_size
       
       
        self.model = model
       
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
           if k.startswith(unwanted_prefix):
              state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.model.to(device)
        if compile:
           self.model = torch.compile(self.model) 
        
        print("Loading tokenizer")
        # ok let's assume gpt-2 encodings by default
        self.enc = tiktoken.get_encoding("gpt2")
        self.encode = lambda s: self.enc.encode(s, allowed_special={"<|endoftext|>"})
        self.decode = lambda l: self.enc.decode(l)
   
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
     
        print("Computing loglikelihoods")
        loglikelihoods = []
        count = 0
        for instance in requests:
            print("Len :", len(requests))
            input_str, target_str = instance.arguments
            input_ids = torch.tensor([self.encode(input_str)], device=device)
            target_ids = torch.tensor([self.encode(target_str)], device=device)
        
    
        
            with torch.no_grad():
                outputs = self.model(input_ids)
                predictions = outputs[0] #logits

            # Calcola la log-verosimiglianza
            log_probs = F.log_softmax(predictions, dim=-1)
            print("Log_probs: ", log_probs)
            target_log_likelihood = log_probs[0, -1, target_ids].sum().item()
            print("Target_log_likelihood: ", target_log_likelihood)
        
        
            # Aggiungi la log-verosimiglianza alla lista
            loglikelihoods.append(tuple([target_log_likelihood, False]))
        
         
        print("Log_probs shape: ", log_probs.shape)
        print("Logits shape: ", predictions.shape)
        print("Input_ids shape: ", input_ids.shape)
        print("loglikelihoods: ", loglikelihoods)
        
        print(len(loglikelihoods))
        print(len([(0.0, False) for _ in requests]))
        return loglikelihoods
        
        
        
        


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
     
            print("Computing loglikelihoods")
            loglikelihoods = []
            for instance in requests:
               input_str = instance.arguments[0]
               input_ids = torch.tensor([self.encode(input_str)], device=device)



               with torch.no_grad():
                   outputs = self.model(input_ids)
                   predictions = outputs[0] #logits

               # Calcola la log-verosimiglianza
               
               log_probs = F.log_softmax(predictions, dim=-1)
               print("Log_probs: ", log_probs)
               print("Input_ids size: ", input_ids.size(1))
               input_log_likelihood = 0.0
            
               for i in range(1, input_ids.size(1)):  # Skip the initial token (usually <BOS>)
                       token_id = input_ids[0, i]
                       token_log_prob = log_probs[0, i - 1, token_id].item()
                       input_log_likelihood += token_log_prob
                       loglikelihoods.append(tuple([input_log_likelihood, ]))
               
               print("Log_probs shape: ", log_probs.shape)
               print("Logits shape: ", predictions.shape)
               print("Input_ids shape: ", input_ids.shape)
            
            # Aggiungi la log-verosimiglianza alla lista
            

            return loglikelihoods
            

        


    def generate_until(self, requests: list[Instance]) -> list[str]:
        
        str_list = []
        for instance in requests:
            input_str = instance.arguments[0]
            params = instance.arguments[1]
            input_ids = self.encode(input_str)
            x = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])
            with torch.no_grad():
                with ctx:
                        y = self.model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            
            
                        str_list.append( self.decode(y[0].tolist()))
        return str_list
                        
        
    






import lm_eval




my_model = GPT(GPTConfig()) # create your model (could be running finetuning with some custom modeling code)
...
# instantiate an LM subclass that takes your initialized model and can run
# - `Your_LM.loglikelihood()`
# - `Your_LM.loglikelihood_rolling()`
# - `Your_LM.generate_until()`
lm_obj = MyCustomLM(model = my_model, batch_size=1)

# indexes all tasks from the `lm_eval/tasks` subdirectory.
# Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
# to include a set of tasks in a separate directory.
task_manager = lm_eval.tasks.TaskManager()

# Setting `task_manager` to the one above is optional and should generally be done
# if you want to include tasks from paths other than ones in `lm_eval/tasks`.
# `simple_evaluate` will instantiate its own task_manager if it is set to None here.
results = lm_eval.simple_evaluate( # call simple_evaluate
    model=lm_obj,
    tasks=[task],
    task_manager=task_manager,
    limit=limit,
    device=device,
    batch_size=batch_size,
    predict_only=False,
    bootstrap_iters=bootstrap_iters
    
    
    
    
    
)

# import json
# # Convert results to JSON
# results_json = json.dumps(results)

# # Save results to a JSON file
# with open('results.json', 'w') as f:
#     f.write(results_json)

import wandb

wandb.login()

from lm_eval.logging_utils import WandbLogger

wandb_logger = WandbLogger(
    project="gpt2-eval", job_type="eval"
)  # or empty if wandb.init(...) already called before
wandb_logger.post_init(results)
wandb_logger.log_eval_result()
wandb_logger.log_eval_samples(results["samples"])