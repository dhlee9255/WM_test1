"""Play inside the learned world model. The real game renders the first N context frames,
then every frame comes from the diffusion model conditioned on your keys.

python play_model.py --ckpt checkpoints/last.pt --fp16
Keys: WASD move, Space attack, E pick up, R reset (re-seed from the real game), Esc quit.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
import torch

sys.path.insert(0, str(Path(__file__).parent / "wm_server"))
from wm.data import to_image, to_model  # noqa: E402
from wm.model import load_checkpoint  # noqa: E402

from wmgame.core import TICK_HZ  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=3, help="denoising steps per frame")
    ap.add_argument("--display", default="960x720")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fp16", action="store_true", help="half precision (tensor cores on RTX GPUs)")
    args = ap.parse_args()
    use_fp16 = args.fp16 and args.device.startswith("cuda")

    model, ck = load_checkpoint(args.ckpt, args.device)
    n = model.cfg.context
    print(f"loaded step {ck['step']} on {args.device}")

    from wmgame.app import Session
    sess = Session(320, 240, offscreen=True)

    def real_context(seed):
        frames = [sess.reset(seed)]
        for _ in range(n - 1):
            frames.append(sess.step((0,) * 6)[0])
        x = to_model(torch.from_numpy(np.stack(frames))[None].to(args.device))[0]
        return x, torch.zeros(n, model.cfg.num_actions, device=args.device)

    ctx, acts = real_context(args.seed)
    dw, dh = map(int, args.display.lower().split("x"))
    pygame.init()
    screen = pygame.display.set_mode((dw, dh))
    pygame.key.stop_text_input()
    clock = pygame.time.Clock()
    seed, attack, pickup, running = args.seed, False, False, True
    fps = 0.0
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    attack = True
                elif ev.key == pygame.K_e:
                    pickup = True
                elif ev.key == pygame.K_r:
                    seed += 1
                    ctx, acts = real_context(seed)
        pressed = pygame.key.get_pressed()
        keys = [pressed[k] for k in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d)] + [attack, pickup]
        attack = pickup = False
        acts = torch.cat([acts[1:], torch.tensor([keys], dtype=torch.float32, device=args.device)])
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            nxt = model.sample(ctx[None], acts[None], steps=args.steps)[0].float()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        fps = 0.9 * fps + 0.1 / max(time.time() - t0, 1e-6)
        ctx = torch.cat([ctx[1:], nxt[None]])

        img = to_image(nxt)
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        scale = min(dw // img.shape[1], dh // img.shape[0])
        sw, sh = img.shape[1] * scale, img.shape[0] * scale
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.smoothscale(surf, (sw, sh)), ((dw - sw) // 2, (dh - sh) // 2))
        pygame.display.flip()
        pygame.display.set_caption(f"World model  |  {fps:.1f} fps  |  {args.steps} steps  |  {args.device}"
                                   + ("  fp16" if use_fp16 else ""))
        clock.tick(TICK_HZ)
    pygame.quit()


if __name__ == "__main__":
    main()
