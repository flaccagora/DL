import os

import torch
from transformers import GPT2Tokenizer

from kan_gpt.mingpt.model import GPT as MLP_GPT
from kan_gpt.model import GPT as KAN_GPT


@torch.no_grad()
def main(args):

    model_type = args.model_type

    print("Model type: ", model_type)
    print("Model architecture: ", args.architecture)
    if args.architecture == "KAN":
        GPT = KAN_GPT
    else:
        GPT = MLP_GPT

    # create a GPT instance
    model_config = GPT.get_default_config()
    model_config.model_type = model_type
    model_config.vocab_size = 50257
    model_config.block_size = 64
    model = GPT(model_config)

    if args.model_path is not None:
        ckpt = torch.load(args.model_path, map_location=torch.device("cpu"))
        print(ckpt.keys())


        start_epoch = ckpt['epoch']
        iter_num = ckpt['step']
        config = ckpt['config']

        print("Checkpoint config:\n ", config)
        print("Loaded model from epoch: ", start_epoch)
        print("Loaded model from iteration: ", iter_num)

        model.load_state_dict(ckpt['model'])

        assert os.path.isfile(args.model_path)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    prompt_encoded = tokenizer.encode(
        text=args.prompt, add_special_tokens=False
    )

    x = torch.tensor(prompt_encoded).unsqueeze(0)

    x = x.to(device=args.device)
    model = model.to(device=args.device)

    model.eval()
    y = model.generate(x, args.max_tokens)

    y_np = y.squeeze(0).cpu().detach().numpy()

    result = tokenizer.decode(y_np)

    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("KAN-GPT Trainer")
    parser.add_argument("--model_type", default="gpt-mini")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--max_tokens", default=10)

    parser.add_argument(
        "--prompt", default="Out of thy sleep. What is it thou didst say?"
    )
    parser.add_argument(
        "--architecture", choices=["MLP", "KAN"], default="KAN"
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    args = parser.parse_args()

    args.max_tokens = int(args.max_tokens)

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"


    main(args)
