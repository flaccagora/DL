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
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)
        self.model = DDP(model, device_ids=[gpu_id])

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

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

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def run(self):
        model, config = self.model, self.config

        # setup the optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
        )

        # setup the dataloader
        train_loader = self.prepare_dataloader()

        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        data_iter = iter(train_loader)
        
        for self.iter_num in trange(config.max_iters):

            b_sz = len(next(iter(train_loader))[0])
            print(f"[GPU{self.gpu_id}] Epoch {self.iter_num} | Batchsize: {b_sz} | Steps: {len(train_loader)}")
            
            train_loader.sampler.set_epoch(self.iter_num)
            
            for x, y in train_loader:
                
                x = x.to(self.gpu_id)
                y = y.to(self.gpu_id)
                
                # forward the model
                logits, self.loss = model(x, y)

                # backprop and update the parameters
                model.zero_grad(set_to_none=True)
                self.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_norm_clip
                )
                self.optimizer.step()

                if self.gpu_id == 0:
                    self.trigger_callbacks("on_batch_end")
                
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow

                # training_iters.update(1)
            if self.gpu_id == 0 and self.iter_num % config.save == 0:
                self._save_checkpoint(self.iter_num)