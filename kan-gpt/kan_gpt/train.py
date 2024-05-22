import os
from datetime import datetime
from typing import Union

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

import wandb
from kan_gpt.dataset import TinyShakespeareDataset, WebTextDataset
from kan_gpt.mingpt.model import GPT as MLP_GPT
from kan_gpt.mingpt.trainer import Trainer
from kan_gpt.model import GPT as KAN_GPT
from kan_gpt.settings import settings

from torch.distributed import init_process_group, destroy_process_group
import os
import torch.multiprocessing as mp
import tqdm as tqdm


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def metrics(y, y_pred):
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


def eval_split(
    trainer, split, max_batches, batch_size, model, train_dataset, test_dataset
):
    dataset = {"train": train_dataset, "test": test_dataset}[split]
    results = []

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0, drop_last=False
    )
    length = max_batches if max_batches is not None else len(loader)
    training_iters = tqdm.tqdm(
            range(length),
            total=length,
            dynamic_ncols=True
            )
        
    training_iters.update(1)


    for b, (x, y) in enumerate(loader):
        x = x.to(trainer.gpu_id)
        y = y.to(trainer.gpu_id)

        block_size = y.shape[1]

        logits, loss = model(x, y)

        probs = F.softmax(logits, dim=-1)

        _, y_pred = torch.topk(probs, k=block_size, dim=-1)

        #perplexity, f1, precision, recall, cross_entropy = metrics(
        #    y=y.cpu().numpy(), y_pred=probs.cpu().numpy()
        #)

        perplexity = metrics(y=y.cpu().numpy(), y_pred=probs.cpu().numpy())
        f1 = 0
        precision = 0
        recall = 0
        cross_entropy = 0

        results.append(
            (loss, perplexity)#, f1, precision, recall, cross_entropy)
        )
        
        training_iters.update(1)

        if max_batches is not None and b + 1 >= max_batches:
            break
        

    rt = torch.tensor(results, dtype=torch.float)
    print("%s loss: %.2f" % (split, rt.mean(dim=0)[0]))
    return rt.mean(dim=0)


def save_model(
    model: torch.nn.Module, run: Union[Run, RunDisabled, None] = None
):
    os.makedirs(settings.train.WEIGHTS_PATH, exist_ok=True)
    id = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(
        settings.train.WEIGHTS_PATH,
        "model_{id}.pth".format(
            id=id,
        ),
    )
    torch.save(model.state_dict(), save_path)
    print("Model saved: {}".format(save_path))
    if run is not None and isinstance(run, Run):
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(save_path)
        run.log_artifact(artifact)
    else:
        print("Model NOT uploaded to wandb")

    return save_path


def main(rank, args):
    
    ddp_setup(rank, args.num_gpus)

    config = {
        "model_type": args.model_type,
        "batch_size": args.batch_size,
        "dummy_dataset": args.dummy_dataset,
        "learning_rate": args.learning_rate,
        "max_iters": args.max_iters,
        "num_workers": args.num_workers,
        "dataset": args.dataset,
        "architecture": args.architecture,
        "device": args.device,
    }

    model_type = args.model_type

    if args.dataset == "webtext":
        Dataset = WebTextDataset
    elif args.dataset == "tinyshakespeare":
        Dataset = TinyShakespeareDataset

    # print an example instance of the dataset
    if args.dummy_dataset:
        train_dataset = Dataset("test", "gpt2")
    else:
        train_dataset = Dataset("train", "gpt2")

    test_dataset = Dataset("test", "gpt2")

    if rank==0:
        print("test_dataset: ", len(test_dataset))
        print("train_dataset: ", len(train_dataset))

    if args.architecture == "KAN":
        GPT = KAN_GPT
    else:
        GPT = MLP_GPT

    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)

    # create a Trainer object
    train_config = Trainer.get_default_config()
    train_config.learning_rate = float(
        args.learning_rate
    )  # the model we're using is so small that we can go a bit faster
    train_config.max_iters = int(args.max_iters)
    train_config.num_workers = int(args.num_workers)
    train_config.batch_size = int(args.batch_size)
    train_config.device = args.device
    train_config.save = args.save
    train_config.eval = args.eval
    trainer = Trainer(train_config, model, train_dataset, rank)
    
    if args.from_checkpoint:
        trainer._load_checkpoint(PATH=args.from_checkpoint)

    
    # run = None
    # if run is None:
    #     run = wandb.init(project="KAN-GPT", config=config)
    # wandb.watch(model)

    def batch_end_callback(trainer):
        # TODO: Add W&B Hooks
        if trainer.iter_num % args.eval == 0 and trainer.iter_num !=0:

            print(
                # f"iter_dt {trainer.iter_dt * 1000:.2f}ms; "
                f"iter {trainer.iter_num}: "
                f"train loss {trainer.loss.item():.5f}"
            )

            print("=" * 20)
            print("EVAL")
            print("=" * 20)

            trainer.model.eval()
            with torch.no_grad():
                train_score = eval_split(
                    trainer,
                    "train",
                    max_batches=5,
                    batch_size=int(args.batch_size),
                    model=trainer.model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                )
                test_score = eval_split(
                    trainer,
                    "test",
                    max_batches=5,
                    batch_size=int(args.batch_size),
                    model=trainer.model,
                    train_dataset=train_dataset,
                    test_dataset=test_dataset,
                )

                (
                    train_loss,
                    train_perplexity,
                    train_f1,
                    train_precision,
                    train_recall,
                    train_cross_entropy,
                ) = train_score
                (
                    test_loss,
                    test_perplexity,
                    test_f1,
                    test_precision,
                    test_recall,
                    test_cross_entropy,
                ) = test_score

            trainer.model.train()
            print("train_loss: ", train_loss)
            print("train_perplexity: ", train_perplexity)
            print("train_f1: ", train_f1)
            print("train_precision: ", train_precision)
            print("train_recall: ", train_recall)
            print("train_cross_entropy: ", train_cross_entropy)

            print("test_loss: ", test_loss)
            print("test_perplexity: ", test_perplexity)
            print("test_f1: ", test_f1)
            print("test_precision: ", test_precision)
            print("test_recall: ", test_recall)
            print("test_cross_entropy: ", test_cross_entropy)

            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_perplexity": train_perplexity,
                    "train_f1": train_f1,
                    "train_precision": train_precision,
                    "train_recall": train_recall,
                    "train_cross_entropy": train_cross_entropy,
                    "test_loss": test_loss,
                    "test_perplexity": test_perplexity,
                    "test_f1": test_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_cross_entropy": test_cross_entropy,
                }
            )

            print("=" * 20)

    trainer.set_callback("on_batch_end", batch_end_callback)

    trainer.run()

    # if rank==0:
    #     save_model(model=model, run=run)

    #wandb.finish()

    destroy_process_group()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("KAN-GPT Trainer")
    parser.add_argument("--model_type", default="gpt-mini")
    parser.add_argument("--dummy_dataset", action="store_true")
    parser.add_argument("--learning_rate", default=5e-3)
    parser.add_argument("--max_iters", default=2000)
    parser.add_argument("--num_workers", default=0)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--eval", type=int, default=50)
    parser.add_argument("--save", type=int, default=1000)
    parser.add_argument("--num_gpus",type=int, default=4)
    parser.add_argument("--from_checkpoint",type=str, default=None)

    parser.add_argument(
        "--dataset",
        choices=["webtext", "tinyshakespeare"],
        default="webtext",
    )
    parser.add_argument(
        "--architecture", choices=["MLP", "KAN"], default="KAN"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    args = parser.parse_args()

    args.learning_rate = float(args.learning_rate)
    args.max_iters = int(args.max_iters)
    args.num_workers = int(args.num_workers)
    args.batch_size = int(args.batch_size)

    world_size = torch.cuda.device_count()

    if int(args.num_gpus) < world_size:
        world_size = args.num_gpus 
        print("\n\nNumber of GPUs is less than available. only using ", args.num_gpus, " GPUs.\n\n")

    mp.spawn(main, args= (args,), nprocs=world_size)

    # salloc -p boost_usr_prod -A ict24_dssc_gpu -N1 -n4 -t 00:30:00 --gres=gpu:4 --qos=boost_qos_dbg
    # WANDB_MODE=offline  python3 -m kan_gpt.train --architecture MLP --batch_size 32 --dataset webtext --max_iters 100 --num_gpus=4 --save 100 --from_checkpoint checkpoint_50.pt