"""Autoregressive rollouts of the world model with recorded (ground-truth) actions."""
import numpy as np
import torch

from .data import to_image, to_model


@torch.no_grad()
def rollout(model, frames_u8, actions_u8, steps=3, ctx_sigma=0.0):
    """frames_u8 (T,H,W,3) and actions_u8 (T,A) from one episode. Uses the first N frames as
    context, then predicts T-N frames feeding its own outputs back. Returns uint8 (T-N,H,W,3)."""
    n = model.cfg.context
    dev = next(model.parameters()).device
    x = to_model(torch.from_numpy(np.array(frames_u8))[None].to(dev))[0]
    acts = torch.from_numpy(np.array(actions_u8)).float().to(dev)
    ctx = x[:n].clone()
    preds = []
    for t in range(n - 1, len(x) - 1):
        a = acts[t - n + 1:t + 1]
        nxt = model.sample(ctx[None], a[None], steps=steps, ctx_sigma=ctx_sigma)[0]
        preds.append(nxt)
        ctx = torch.cat([ctx[1:], nxt[None]])
    return to_image(torch.stack(preds))


def comparison_strip(gt_u8, pred_u8, every=3, max_cols=10):
    """Two rows (ground truth over prediction), every k-th frame."""
    idx = list(range(0, len(pred_u8), every))[:max_cols]
    top = np.concatenate([gt_u8[i] for i in idx], 1)
    bot = np.concatenate([pred_u8[i] for i in idx], 1)
    return np.concatenate([top, bot], 0)
