"""Action-conditioned diffusion world model (DIAMOND-style EDM denoiser in pixel space).

Given the last N frames and the N actions taken at those frames, denoise frame t+1.
Frames are float tensors in [-1, 1], shape (B, 3, H, W). Actions are multi-hot (B, N, A).
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    context: int = 4
    num_actions: int = 6
    channels: tuple = (64, 64, 128, 256, 256)
    attn: tuple = (False, False, False, True, True)
    blocks: int = 2
    cond_dim: int = 256
    sigma_data: float = 0.5
    max_ctx_noise: float = 0.3  # noise augmentation on context frames (autoregressive robustness)

    def to_dict(self):
        return asdict(self)


def _groups(ch):
    return max(1, min(32, ch // 8))


class Fourier(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("w", torch.randn(1, dim // 2))

    def forward(self, x):
        f = 2 * math.pi * x[:, None] * self.w
        return torch.cat([f.cos(), f.sin()], -1)


class AdaGN(nn.Module):
    def __init__(self, ch, cond):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch, affine=False)
        self.proj = nn.Linear(cond, 2 * ch)

    def forward(self, x, c):
        scale, shift = self.proj(c)[:, :, None, None].chunk(2, 1)
        return self.norm(x) * (1 + scale) + shift


class SelfAttention(nn.Module):
    def __init__(self, ch, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.heads = heads

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).transpose(-1, -2).unbind(1)
        y = F.scaled_dot_product_attention(q, k, v)
        return x + self.out(y.transpose(-1, -2).reshape(b, c, h, w))


class ResBlock(nn.Module):
    def __init__(self, cin, cout, cond, attn):
        super().__init__()
        self.n1, self.c1 = AdaGN(cin, cond), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = AdaGN(cout, cond), nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.attn = SelfAttention(cout) if attn else nn.Identity()

    def forward(self, x, c):
        h = self.c1(F.silu(self.n1(x, c)))
        h = self.c2(F.silu(self.n2(h, c)))
        return self.attn(self.skip(x) + h)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        chs, cd = cfg.channels, cfg.cond_dim
        self.noise_emb = Fourier(cd)
        self.aug_emb = Fourier(cd)
        self.act_emb = nn.Linear(cfg.context * cfg.num_actions, cd)
        self.cond_mlp = nn.Sequential(nn.Linear(cd, cd), nn.SiLU(), nn.Linear(cd, cd))
        self.conv_in = nn.Conv2d(3 * (cfg.context + 1), chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        skip_ch, ch = [chs[0]], chs[0]
        for i, c in enumerate(chs):
            for _ in range(cfg.blocks):
                self.downs.append(ResBlock(ch, c, cd, cfg.attn[i]))
                ch = c
                skip_ch.append(ch)
            if i < len(chs) - 1:
                self.downs.append(Down(ch))
                skip_ch.append(ch)
        self.mid = nn.ModuleList([ResBlock(ch, ch, cd, True), ResBlock(ch, ch, cd, False)])
        self.ups = nn.ModuleList()
        for i, c in reversed(list(enumerate(chs))):
            for _ in range(cfg.blocks + 1):
                self.ups.append(ResBlock(ch + skip_ch.pop(), c, cd, cfg.attn[i]))
                ch = c
            if i > 0:
                self.ups.append(Up(ch))
        self.norm_out = nn.GroupNorm(_groups(ch), ch)
        self.conv_out = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, c_noise, c_aug, actions):
        c = self.noise_emb(c_noise) + self.aug_emb(c_aug) + self.act_emb(actions.flatten(1))
        c = self.cond_mlp(c)
        h = self.conv_in(x)
        hs = [h]
        for m in self.downs:
            h = m(h, c) if isinstance(m, ResBlock) else m(h)
            hs.append(h)
        for m in self.mid:
            h = m(h, c)
        for m in self.ups:
            h = m(torch.cat([h, hs.pop()], 1), c) if isinstance(m, ResBlock) else m(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class WorldModel(nn.Module):
    """EDM-preconditioned denoiser D(x_noisy; sigma, context frames, actions)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = UNet(cfg)

    def _forward(self, x_noisy, sigma, ctx, actions, ctx_sigma):
        """Raw network output plus the EDM skip/out scalings."""
        sd = self.cfg.sigma_data
        s = sigma[:, None, None, None]
        c_skip = sd ** 2 / (s ** 2 + sd ** 2)
        c_out = s * sd / (s ** 2 + sd ** 2).sqrt()
        c_in = 1 / (s ** 2 + sd ** 2).sqrt()
        inp = torch.cat([c_in * x_noisy, ctx.flatten(1, 2) / sd], 1)
        return self.net(inp, sigma.log() / 4, ctx_sigma * 4, actions), c_skip, c_out

    def denoise(self, x_noisy, sigma, ctx, actions, ctx_sigma):
        f, c_skip, c_out = self._forward(x_noisy, sigma, ctx, actions, ctx_sigma)
        return c_skip * x_noisy + c_out * f

    def loss(self, ctx, actions, target):
        """ctx (B,N,3,H,W), actions (B,N,A), target (B,3,H,W), all frames in [-1,1]."""
        b = target.shape[0]
        sigma = (torch.randn(b, device=target.device) * 1.2 - 0.4).exp().clamp(2e-3, 20)
        ctx_sigma = torch.rand(b, device=target.device) * self.cfg.max_ctx_noise
        ctx = ctx + ctx_sigma[:, None, None, None, None] * torch.randn_like(ctx)
        x_noisy = target + sigma[:, None, None, None] * torch.randn_like(target)
        f, c_skip, c_out = self._forward(x_noisy, sigma, ctx, actions, ctx_sigma)
        return F.mse_loss(f.float(), ((target - c_skip * x_noisy) / c_out).float())

    forward = loss  # DDP only syncs gradients for calls that go through forward()

    @torch.no_grad()
    def sample(self, ctx, actions, steps=3, sigma_min=2e-3, sigma_max=5.0, rho=7.0, ctx_sigma=0.0):
        """Euler sampler over a Karras schedule. Returns next frame in [-1,1]."""
        b, _, _, h, w = ctx.shape
        ramp = torch.linspace(0, 1, steps, device=ctx.device)
        sigmas = (sigma_max ** (1 / rho) + ramp * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])
        cs = torch.full((b,), float(ctx_sigma), device=ctx.device)
        if ctx_sigma > 0:  # match training: context frames are actually noised at the level we report
            ctx = ctx + ctx_sigma * torch.randn_like(ctx)
        x = torch.randn(b, 3, h, w, device=ctx.device) * sigmas[0]
        for i in range(steps):
            s = sigmas[i].expand(b)
            den = self.denoise(x, s, ctx, actions, cs)
            x = x + (x - den) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
        return x.clamp(-1, 1)


def load_checkpoint(path, device="cpu", ema=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in ck["config"].items()})
    model = WorldModel(cfg)
    model.load_state_dict(ck["ema" if ema and "ema" in ck else "model"])
    return model.to(device).eval(), ck
