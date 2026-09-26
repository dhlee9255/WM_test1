"""Run the scripted bot and record (frame, keys) episodes for world-model training.

python collect.py --episodes 330 --workers 4 --smooth-frac 0.3 --out wm_server/data/raw   # ~300k frames
python collect.py --episodes 48 --passive --seed 20000 --workers 4 --out wm_server/data/raw  # deaths
python collect.py --episodes 1 --show --no-save                          # watch the bot
"""
import argparse
import random
import re
import subprocess
import sys
import time

from wmgame.core import TICK_HZ


def run_worker(args, shard, nshards):
    from wmgame.app import Recorder, Session
    from wmgame.bot import PassiveBot, ScriptedBot

    sess = Session(args.width, args.height, offscreen=not args.show, title="WorldModel Game - bot")
    rec = None if args.no_save else Recorder(args.out, args.width, args.height)
    done_seeds = set()
    if args.skip_existing:
        from pathlib import Path
        done_seeds = {int(m.group(1)) for f in Path(args.out).glob("*.npz") if (m := re.search(r"_s(\d+)\.npz$", f.name))}
    total_steps, t_start = 0, time.time()
    for ep in range(shard, args.episodes, nshards):
        seed = args.seed + ep
        if seed in done_seeds:
            continue
        smooth = random.Random(seed * 7919 + 1).random() < args.smooth_frac
        frame = sess.reset(seed)
        bot = PassiveBot() if args.passive else ScriptedBot(seed, smooth=smooth)
        style = "passive" if args.passive else "smooth" if smooth else "normal"
        if rec:
            rec.start(frame, sess.game, "passive" if args.passive else "bot")
        for _ in range(args.max_steps):
            t0 = time.time()
            keys = bot.act(sess.game)
            frame, info = sess.step(keys)
            if rec:
                rec.add(keys, frame, info, sess.game)
            if args.show:
                time.sleep(max(0.0, 1 / TICK_HZ - (time.time() - t0)))
            if sess.game.done:
                break
        total_steps += sess.game.tick
        path = rec.save(sess.game.done) if rec else None
        print(f"[w{shard}] ep {ep} seed {seed} {style}: steps {sess.game.tick} kills {sess.game.kills} "
              f"died {sess.game.done} -> {path}  ({total_steps / (time.time() - t_start):.0f} steps/s)",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--width", type=int, default=320, help="saved frame width")
    ap.add_argument("--height", type=int, default=240, help="saved frame height")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/bot")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--smooth-frac", type=float, default=0.0, help="fraction of episodes with the smooth bot")
    ap.add_argument("--skip-existing", action="store_true", help="skip seeds already saved in --out")
    ap.add_argument("--passive", action="store_true", help="bot stands still until enemies kill it")
    ap.add_argument("--shard", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--show", action="store_true", help="render in a window at real-time speed")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.shard is not None:
        k, n = map(int, args.shard.split("/"))
        run_worker(args, k, n)
    elif args.workers <= 1 or args.show:
        run_worker(args, 0, 1)
    else:
        procs = [subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--shard", f"{k}/{args.workers}"])
                 for k in range(args.workers)]
        sys.exit(max(p.wait() for p in procs))


if __name__ == "__main__":
    main()
