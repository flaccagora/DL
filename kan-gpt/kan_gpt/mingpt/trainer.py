"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader

import tqdm as tqdm
from tqdm import trange
from kan_gpt.mingpt.utils import CfgNode as CN

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


class Trainer:

    @staticmethod
    def get_default_config():
        C = CN()
        # device to train on
        C.device = "auto"
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = None
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1  # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, train_dataset, gpu_id):
        self.config = config
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.optimizer = model.configure_optimizers(config)

        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.model = DDP(model, device_ids=[gpu_id])

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.epoch = 0


    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)
   
    def prepare_dataloader(self):
        return  DataLoader(
            self.train_dataset,
            sampler=DistributedSampler(self.train_dataset),
            shuffle=False,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def _save_checkpoint(self):


        ckpt = {        'epoch': self.epoch,
                        'step': self.iter_num,
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        # 'scheduler': self.scheduler.state_dict(),
                        'config': self.config
                        }
        
        torch.save(ckpt, f"checkpoint_{self.iter_num}.pt")

        print(f"Saved checkpoint to checkpoint_{self.iter_num}.pt")

    def _load_checkpoint(self, PATH):

        print(f"Loading checkpoint from {PATH}")

        ckpt = torch.load(PATH, map_location='cuda')

        self.epoch = ckpt['epoch']
        self.iter_num = ckpt['step']
        self.config = ckpt['config']
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        print("Checkpoint config:\n ", self.config)
        
        # scheduler.load_state_dict(ckpt['scheduler'])
        del ckpt
        torch.cuda.empty_cache()


    def run(self):
        model, config = self.model, self.config

        # setup the dataloader
        train_loader = self.prepare_dataloader()

        model.train()
        self.iter_num = 0
        data_iter = iter(train_loader)

        training_iters = tqdm.tqdm(
                range(len(train_loader)),
                total=len(train_loader),
                dynamic_ncols=True
                )
            
        training_iters.update(1)

        
        for self.epoch in range(config.max_iters):

            b_sz = len(next(iter(train_loader))[0])
            print(f"[GPU{self.gpu_id}] Epoch {self.iter_num} | Batchsize (perGPU): {b_sz} | Steps (perGPU): {len(train_loader)}")

            ## batch size * steps * numgpus = len(train_dataset)
            
            train_loader.sampler.set_epoch(self.epoch)
            
            for i,(x, y) in enumerate(train_loader):
                
                if i < self.iter_num - (len(train_loader) * self.epoch):
                    training_iters.update(1)
                    continue

                
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)
                
                # if(self.gpu_id == 0):
                #     print("0\n", torch.cuda.memory_summary(device=self.gpu_id))
                
                # forward the model
                logits, self.loss = model(x, y)
               

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_norm_clip
                )
                self.optimizer.step()
                
                self.iter_num += 1
                training_iters.update(1)
               

                if self.gpu_id == 0 and self.iter_num % config.save == 0 and self.iter_num!=0:
                    self._save_checkpoint()
                
                # wait for all processes to synchronize
                torch.distributed.barrier()

                if self.iter_num % config.eval == 0 and self.iter_num!=0:
                    self.trigger_callbacks("on_batch_end")

            
            
