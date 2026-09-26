"""Play with the keyboard. WASD move, Space attack, E pick up potion, R restart, Esc quit.

The game renders offscreen at --width x --height (same frames the bot records) and is
shown upscaled (smooth) in a --display sized window.

python play.py                 # just play
python play.py --record        # also save episodes to data/human (same format as the bot)
"""
import argparse
import os
import random

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from wmgame.app import Recorder, Session
from wmgame.core import TICK_HZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--width", type=int, default=320, help="render (and saved frame) width")
    ap.add_argument("--height", type=int, default=240, help="render (and saved frame) height")
    ap.add_argument("--display", default="960x720", help="window size (4:3, 3x of 320x240)")
    ap.add_argument("--out", default="data/human")
    args = ap.parse_args()

    sess = Session(args.width, args.height, offscreen=True)
    rec = Recorder(args.out, args.width, args.height) if args.record else None

    dw, dh = map(int, args.display.lower().split("x"))
    scale = max(1, min(dw // args.width, dh // args.height))
    sw, sh = args.width * scale, args.height * scale
    pygame.init()
    screen = pygame.display.set_mode((dw, dh))
    pygame.key.stop_text_input()  # keep IME (e.g. Korean input) from eating WASD
    clock = pygame.time.Clock()

    seed = args.seed if args.seed is not None else random.randrange(1 << 30)
    frame = sess.reset(seed)
    if rec:
        rec.start(frame, sess.game, "human")
    attack = pickup = restart = False
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    attack = True
                elif ev.key == pygame.K_e:
                    pickup = True
                elif ev.key == pygame.K_r:
                    restart = True
        if not running:
            break
        if restart or sess.game.done:
            if rec:
                print("saved", rec.save(sess.game.done))
            seed += 1
            restart = False
            frame = sess.reset(seed)
            if rec:
                rec.start(frame, sess.game, "human")

        pressed = pygame.key.get_pressed()
        keys = tuple(int(pressed[k]) for k in (pygame.K_w, pygame.K_a, pygame.K_s, pygame.K_d)) + (int(attack), int(pickup))
        attack = pickup = False
        frame, info = sess.step(keys)
        if rec:
            rec.add(keys, frame, info, sess.game)

        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.smoothscale(surf, (sw, sh)), ((dw - sw) // 2, (dh - sh) // 2))
        pygame.display.flip()
        pygame.display.set_caption(f"WorldModel Game  |  kills {sess.game.kills}  |  seed {seed}"
                                   + ("  |  REC" if rec else ""))
        clock.tick(TICK_HZ)

    if rec:
        print("saved", rec.save(sess.game.done))
    pygame.quit()


if __name__ == "__main__":
    main()
