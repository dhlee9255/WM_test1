"""Train the diffusion world model.

  python prepare_data.py --raw data/raw --out data/packed                 # once
  torchrun --nproc_per_node 4 train.py --data data/packed --out runs/wm_v2   # 4 GPUs
  python train.py --data data/packed --out runs/wm_v2                        # 1 GPU

Re-running the same command resumes from <out>/last.pt. Every --eval-every steps an
autoregressive rollout on validation episodes is saved to <out>/samples/ (top: real, bottom: model).
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from wm.data import PackedData, WindowDataset, to_model
from wm.model import ModelConfig, WorldModel
from wm.rollout import comparison_strip, rollout


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/packed")
    ap.add_argument("--out", default="runs/wm_v2")
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=16, help="per GPU")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--context", type=int, default=8, help="past frames the model sees (8 = ~0.5 s)")
    ap.add_argument("--channels", default="64,64,128,256,256")
    ap.add_argument("--workers", type=int, default=8, help="dataloader workers per GPU")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--keep-every", type=int, default=25000, help="also keep step_XXXXXX.pt snapshots")
    ap.add_argument("--rollout-len", type=int, default=45)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse()
    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main_proc = rank == 0
    out = Path(args.out)
    if main_proc:
        (out / "samples").mkdir(parents=True, exist_ok=True)

    data = PackedData(args.data)
    num_actions = data.actions.shape[1]
    chs = tuple(int(c) for c in args.channels.split(","))
    cfg = ModelConfig(context=args.context, num_actions=num_actions, channels=chs,
                      attn=tuple(i >= len(chs) - 2 for i in range(len(chs))))
    model = WorldModel(cfg).to(dev)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    step = 0
    ckpt_path = out / "last.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if main_proc:
            print(f"resumed from step {step}")
    if main_proc:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model: {n_params / 1e6:.1f}M params, config {cfg.to_dict()}")
        (out / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "args": vars(args)}, indent=2))

    net = model
    if args.compile:
        net = torch.compile(net)
    if ddp:
        net = DDP(net, device_ids=[local])

    train_ds = WindowDataset(data, args.context, val=False)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if ddp else None
    loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    val_eps = data.split(val=True)
    length = args.context + args.rollout_len
    eval_eps = [e for e in val_eps if e[1] >= length + 100][:3]
    if main_proc:
        print(f"train windows {len(train_ds)}, val episodes {len(val_eps)}, "
              f"global batch {args.batch * world}")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = (s - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    def save(path):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "config": cfg.to_dict(), "args": vars(args)}, path)

    epoch, t0, loss_acc, n_acc = 0, time.time(), 0.0, 0
    it = iter(())
    while step < args.steps:
        try:
            frames, actions = next(it)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(epoch)
            epoch += 1
            it = iter(loader)
            continue
        x = to_model(frames.to(dev, non_blocking=True))
        acts = actions.to(dev, non_blocking=True).float()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = net(x[:, :args.context], acts, x[:, args.context])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - args.ema)
        step += 1
        loss_acc += loss.item()
        n_acc += 1

        if main_proc and step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:7d}  loss {loss_acc / n_acc:.4f}  lr {lr_at(step):.2e}  "
                  f"{n_acc * args.batch * world / dt:.0f} samples/s", flush=True)
            t0, loss_acc, n_acc = time.time(), 0.0, 0
        if main_proc and (step % args.save_every == 0 or step == args.steps):
            save(out / "tmp.pt")
            os.replace(out / "tmp.pt", ckpt_path)
            if step % args.keep_every == 0:
                save(out / f"step_{step:06d}.pt")
        if main_proc and step % args.eval_every == 0 and eval_eps:
            strips = []
            for s, n in eval_eps:
                fr = data.frames[s + 100:s + 100 + length]
                pred = rollout(ema, fr, data.actions[s + 100:s + 100 + length])
                strips.append(comparison_strip(fr[args.context:], pred))
            Image.fromarray(np.concatenate(strips)).save(out / "samples" / f"step_{step:06d}.png")
            print(f"  saved rollout samples/step_{step:06d}.png", flush=True)
        if ddp and step % args.eval_every == 0:
            dist.barrier()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
