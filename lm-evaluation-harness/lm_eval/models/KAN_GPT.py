from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
import torch
import os
import tiktoken
from contextlib import nullcontext
import sys
# Add the path of the specific folder to import modules from
#sys.path.append("../../../nanokan/")
sys.path.append("../")
from nanoGPT.model_kan import GPT, GPTConfig
# from model_kan import GPT as KAN_GPT
#from model_kan import GPTConfig as KAN_GPTConfig
import torch.nn.functional as F
from tqdm import tqdm

# -----------------------------------------------------------------------------
# init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
# out_dir = './' # ignored if init_from is not 'resume'
# start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
# num_samples = 1 # number of samples to draw
# max_new_tokens = 10 # number of tokens generated in each sample
# temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
# top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
# seed = 1337
# device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
# compile = False # use PyTorch 2.0 to compile the model to be faster

#exec(open('../configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------




#################################




from lm_eval.api.registry import register_model

@register_model("KAN_GPT")
class KAN_GPT(LM):
    
    def __init__(self,  batch_size, model=GPT(GPTConfig()), max_length = 1024,
                 init_from = 'resume',
                 out_dir = './', 
                 start = "\n", 
                 num_samples = 1,
                 max_new_tokens = 10, 
                 temperature = 0.8, 
                 top_k = 200, 
                 seed = 1337,
                 device = 'cpu', 
                 dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
                 compile = False
                 
                 
                 ) -> None:
    
        
        
        
        
        self.eot_token_id = 50256
        self.max_length = max_length
        self.init_from = init_from # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
        self.out_dir = out_dir # ignored if init_from is not 'resume'
        self.start = start # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
        self.num_samples = num_samples # number of samples to draw
        self.max_new_tokens = max_new_tokens # number of tokens generated in each sample
        self.temperature = temperature # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
        self.top_k = top_k # retain only the top_k most likely tokens, clamp others to have 0 probability
        self.seed = seed
        self.device = device # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        self.dtype = dtype # 'float32' or 'bfloat16' or 'float16'
        self.compile = compile # use PyTorch 2.0 to compile the model to be faster
        
        super().__init__()
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        self.device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)
        
        
        print("Loading KAN GPT checkpoint")
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        self.batch_size = batch_size
       
       
       
        self.model = model
        self.model = GPT(gptconf)
       
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
     
        # print("Computing loglikelihoods")
        loglikelihoods = []
        count = 0
        for instance in tqdm(requests):
            count += 1
            #print("Count :", count, "Requests :", len(requests))
            input_str, target_str = instance.arguments
            input_ids = torch.tensor([self.encode(input_str)], device=self.device)
            target_ids = torch.tensor([self.encode(target_str)], device=self.device)

            
            if input_ids.size(1) > self.max_length:
                input_ids = input_ids[:, -self.max_length:]
        
            with torch.no_grad():
                outputs = self.model(input_ids)
                predictions = outputs[0] #logits

            # Calcola la log-verosimiglianza
            log_probs = F.log_softmax(predictions, dim=-1)
            target_log_likelihood = log_probs[0, -1, target_ids].sum().item()
            
            # Determina se la stringa target sarebbe generata dal campionamento greedy
            is_greedy = int(target_str == self.decode(predictions[0].argmax(dim=-1).tolist()))
           
            
        
            # Aggiungi la log-verosimiglianza alla lista
            loglikelihoods.append(tuple([target_log_likelihood, is_greedy]))
        
         
    
        
        
        return loglikelihoods
    
    # def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
    #     print("The loglikelihood function is run.")
        
    #     new_reqs = []
        
    #     for context, continuation in [req.args for req in requests]:
    #         if context == "":
    #             # end of text as context
    #             context_enc, continuation_enc = [self.eot_token_id], self.encode(continuation)
    #         else:
    #             context_enc, continuation_enc = self.encode(context), self.encode(continuation)

    #         new_reqs.append(((context, continuation), context_enc, continuation_enc))

    #     results = [] # (logprobs, is_greedily_generated)

    #     pbar = tqdm(total=len(new_reqs), desc="loglikelihood")
    #     with torch.no_grad():
    #         for (context, continuation), context_enc, continuation_enc in new_reqs:

    #             """
    #             V: Vocab size
    #             C: Context length
    #             L: Continuation length
    #             """

    #             input_ids = (torch.tensor(context_enc + continuation_enc)
    #                         [:-1].unsqueeze(0).to(self.device)) # [1, C+L-1]
    #             mode_output = self.model(input_ids)[0] # [1, C+L-1, V]
                
    #             logits = mode_output[:,len(context_enc)-1:, :].squeeze(0) # only get the logits for the continuation [L, V]
    #             labels = torch.tensor(continuation_enc).to(self.device) # [L]
    #             is_match = (logits.argmax(-1) == labels).all().item()
    #             # logprobs = F.cross_entropy(logits, labels) # [1]
    #             log_probs_per_target_token = F.log_softmax(logits, dim=-1) # [L, V]
    #             loglikelihood_val = torch.gather(log_probs_per_target_token, -1, labels.unsqueeze(0)).sum() # [1]
                

    #             results.append((loglikelihood_val.detach().cpu().item(), is_match))
    #             pbar.update(1)

    #     pbar.close()

    #     return results
        
        
        
        


    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
     
            count = 0
            loglikelihoods = []
            for instance in tqdm(requests):
               count += 1
               input_str = instance.arguments[0]
               input_ids = torch.tensor([self.encode(input_str)], device=self.device)
              
               if input_ids.size(1) > self.max_length:
                    input_ids = input_ids[:, -self.max_length:]

               with torch.no_grad():
                   outputs = self.model(input_ids)
                   predictions = outputs[0] #logits

               # Calculate the log-likelihoods
               log_probs = F.log_softmax(predictions, dim=-1)
               input_log_likelihood = 0.0
            
               for i in range(1, input_ids.size(1)):  # Skip the initial token (usually <BOS>)
                       token_id = input_ids[0, i]
                       token_log_prob = log_probs[0, i - 1, token_id].item()
                       input_log_likelihood += token_log_prob
                       loglikelihoods.append(tuple([input_log_likelihood, ]))
               

            return loglikelihoods
            

        


    def generate_until(self, requests: list[Instance]) -> list[str]:
        
        count = 0
        str_list = []
        for instance in tqdm(requests):
            count += 1
            input_str = instance.arguments[0]
            params = instance.arguments[1]
            input_ids = self.encode(input_str)
            
            # if input_ids.size(1) > self.max_length:
            #         input_ids = input_ids[:, -self.max_length:]
                    
            x = (torch.tensor(input_ids, dtype=torch.long, device=self.device)[None, ...])
            with torch.no_grad():
                with self.ctx:
                        y = self.model.generate(x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)
            
            
                        str_list.append( self.decode(y[0].tolist()))
        return str_list
                        
        
    






