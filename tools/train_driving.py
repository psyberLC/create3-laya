#!/usr/bin/env python3
"""Fine-tune a Laya checkpoint to drive, training only the top encoder layers and the head.

`laya-train` offers a full fine-tune (about 6.3 GB of weights, gradients and optimizer state
for the 395M-parameter encoder; on an 8 GB laptop GPU next to a desktop it spills into
shared memory and slows to about 1 item/s) or a frozen encoder (fast, but the head alone
does not learn this task: 32% -> 45% on held-out states). This wraps laya's own trainer
and freezes the embeddings and lower layers, so only the top layers and head train.

Run with the Laya venv's Python (the GPU server must be stopped while this trains):
  .venv-laya\\Scripts\\python.exe tools\\train_driving.py --base <typed-decisions dir>
"""

import argparse
import sys
from pathlib import Path

import laya.train as laya_train

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--base', required=True, help='local typed-decisions checkpoint directory')
    p.add_argument('--data', default=str(ROOT / 'data' / 'driving_train.jsonl'))
    p.add_argument('--eval', default=str(ROOT / 'data' / 'driving_eval.jsonl'))
    p.add_argument('--out', default=str(ROOT / 'models' / 'create3-driving'))
    p.add_argument('--train-layers', type=int, default=12,
                   help='how many of the top encoder layers to train (ModernBERT-large has 28)')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--micro-batch', type=int, default=8)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--encoder-lr', type=float, default=1e-4)
    p.add_argument('--head-lr', type=float, default=3e-4)
    p.add_argument('--loss', choices=['soft-ce', 'rlcd'], default='soft-ce',
                   help="soft-ce: plain cross-entropy; rlcd: laya's default calibration loss")
    p.add_argument('--seed', type=int, default=1)
    args = p.parse_args()

    original = laya_train.train_model

    def train_top_layers(model, *a, **kw):
        encoder = model.encoder
        layers = encoder.layers
        keep = max(0, len(layers) - args.train_layers)
        for param in encoder.embeddings.parameters():
            param.requires_grad_(False)
        for layer in layers[:keep]:
            for param in layer.parameters():
                param.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f'training top {len(layers) - keep} of {len(layers)} encoder layers + head: '
              f'{trainable / 1e6:.0f}M of {total / 1e6:.0f}M parameters', flush=True)
        return original(model, *a, **kw)

    laya_train.train_model = train_top_layers
    config = laya_train.TrainConfig(
        epochs=args.epochs, micro_batch=args.micro_batch, grad_accum=args.grad_accum,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, loss=args.loss,
        shuffle_options=('choice',),
        eval_data=args.eval, seed=args.seed)
    laya_train.finetune(args.data, args.base, args.out, config=config, device='cuda')
    return 0


if __name__ == '__main__':
    sys.exit(main())
