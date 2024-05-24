"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import time
from collections import defaultdict

import torch
import torch.distributed
from torch.utils.data.dataloader import DataLoader

import tqdm as tqdm
from kan_gpt.mingpt.utils import CfgNode as CN

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F
import wandb
import numpy as np


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

    def __init__(self, config, model, train_dataset,test_dataset, gpu_id):
        self.config = config
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.optimizer, self.scheduler = model.configure_optimizers(config)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.callbacks = defaultdict(list)
        self.model = DDP(model, device_ids=[gpu_id])

        # variables that will be assigned to trainer class later for logging and etc
        self.iter_num = 0
        self.start_epoch = 0


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
            shuffle=True,
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

        self.start_epoch = ckpt['epoch']
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

        training_iters = tqdm.tqdm(
                range(len(train_loader)),
                total=len(train_loader),
                dynamic_ncols=True
                )
            
        training_iters.update(1)

        
        for self.epoch in range(self.start_epoch, config.max_iters):
            train_running_loss = 0.0
            b_sz = len(next(iter(train_loader))[0])
            print(f"[GPU{self.gpu_id}] Epoch {self.iter_num} | Batchsize (perGPU): {b_sz} | Steps (perGPU): {len(train_loader)}")

            ## batch size * steps * numgpus = len(train_dataset)
            
            train_loader.sampler.set_epoch(self.epoch)
            
            for i,(x, y) in enumerate(train_loader):
                
                if i <= self.iter_num - (len(train_loader) * self.epoch):
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
                self.scheduler.step()
                
                self.iter_num += 1
                training_iters.update(1)
               

                if self.gpu_id == 0 and self.iter_num % config.save == 0 and self.iter_num!=0:
                    self._save_checkpoint()
                
                # wait for all processes to synchronize
                torch.distributed.barrier()

                if self.iter_num % config.eval == 0 and self.iter_num!=0:
                    #self.trigger_callbacks("on_batch_end")
                    self.batch_end_callback()

                train_running_loss += self.loss.item()
                # print(train_running_loss / (i + 1))
                wandb.log(
                    {
                        "train_running_loss": train_running_loss / (i + 1),
                    })
            
            
    def batch_end_callback(self):
        # TODO: Add W&B Hooks
        if self.iter_num % self.config.eval == 0 and self.iter_num !=0:

            print(
                # f"iter_dt {self.iter_dt * 1000:.2f}ms; "
                f"iter {self.iter_num}: "
                f"train loss {self.loss.item():.5f}"
            )

            print("=" * 20)
            print("EVAL")
            print("=" * 20)

            self.model.eval()
            with torch.no_grad():
                train_loss = self.eval_split(
                    "train",
                    max_batches=10
                )

                test_loss = self.eval_split(
                    "test",
                    max_batches=1000
                )

                # (
                #     train_loss,
                #     # train_perplexity,
                #     # train_f1,
                #     # train_precision,
                #     # train_recall,
                #     # train_cross_entropy,
                # ) = train_score
                # (
                #     test_loss,
                #     # test_perplexity,
                #     # test_f1,
                #     # test_precision,
                #     # test_recall,
                #     # test_cross_entropy,
                # ) = test_score

            self.model.train()
            print("train_loss: ", train_loss)
            # print("train_perplexity: ", train_perplexity)
            # print("train_f1: ", train_f1)
            # print("train_precision: ", train_precision)
            # print("train_recall: ", train_recall)
            # print("train_cross_entropy: ", train_cross_entropy)

            print("test_loss: ", test_loss)
            # print("test_perplexity: ", test_perplexity)
            # print("test_f1: ", test_f1)
            # print("test_precision: ", test_precision)
            # print("test_recall: ", test_recall)
            # print("test_cross_entropy: ", test_cross_entropy)

            wandb.log(
                {
                    "train_loss": train_loss,
                    #"train_perplexity": train_perplexity,
                    # "train_f1": train_f1,
                    # "train_precision": train_precision,
                    # "train_recall": train_recall,
                    # "train_cross_entropy": train_cross_entropy,
                    # "test_loss": test_loss,
                    # "test_perplexity": test_perplexity,
                    # "test_f1": test_f1,
                    # "test_precision": test_precision,
                    # "test_recall": test_recall,
                    # "test_cross_entropy": test_cross_entropy,
                }
            )

            print("=" * 20)

            # print("reduced loss: ",torch.distributed.all_reduce(train_loss))


    def eval_split(self, split, max_batches=5):
        # dataset = {"train": self.train_dataset, "test": self.test_dataset}[split]
        dataset = self.train_dataset if split == "train" else self.test_dataset
        results = []

        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, num_workers=0, drop_last=False,
        )
        tot : int = max_batches if max_batches is not None else len(loader)
        training_iters = tqdm.tqdm(
                range(tot),
                total=tot,
                dynamic_ncols=True
                )
            
        training_iters.update(1)

        running_loss = 0.0            

        for b, (x, y) in enumerate(loader):
            x = x.to(self.gpu_id)
            y = y.to(self.gpu_id)

            # block_size = y.shape[1]

            _ , loss = self.model(x, y)

            running_loss += loss.item()
            training_iters.update(1)

            if max_batches is not None and b + 1 >= max_batches:
                break
            

        #rt = torch.tensor(results, dtype=torch.float)
        #print("%s loss: %.2f" % (split, rt.mean(dim=0)[0]))
        
        return running_loss / (b + 1)   

        

    def metrics(self, y, y_pred):
        """
        y: (B, T) INT - True labels
        y_pred: (B, T, C) FLOAT - Predicted probabilities

        Returns:
        - Perplexity
        - F1 Score
        - Precision
        - Recall
        - Cross Entropy
        """

        # Make sure y_pred is between 0 and 1
        if not (np.all(y_pred >= 0) and np.all(y_pred <= 1)):
            # Softmax
            y_pred = np.exp(y_pred) / np.sum(
                np.exp(y_pred), axis=-1, keepdims=True
            )

        assert np.all(y_pred >= 0) and np.all(
            y_pred <= 1
        ), "y_pred must be between 0 and 1"

        # Add a small epsilon for numerical stability
        epsilon = 1e-9
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Cross Entropy
        y_one_hot = np.eye(y_pred.shape[-1])[y]
        cross_entropy = -np.mean(np.sum(y_one_hot * np.log(y_pred), axis=-1))

        # Perplexity
        perplexity = 2**cross_entropy

        # # Predicted classes
        # y_pred_class = np.argmax(y_pred, axis=-1)

        # # True Positives, False Positives, and False Negatives
        # TP = np.sum(y == y_pred_class)
        # FP = np.sum(y != y_pred_class)
        # FN = FP  # Binary setup, false positives and false negatives are equivalent

        # # Precision, Recall
        # precision = TP / (TP + FP)
        # recall = TP / (TP + FN)

        # # F1 Score
        # f1 = 2 * (precision * recall) / (precision + recall)

        return perplexity # , f1, precision, recall, cross_entropy
