"""
play.py -- watch a trained controller
=====================================
Loads the network a lab saved in outputs/ and drives the environment with it. No learning, no noise.

    python play.py --lab 2                  # DQN car, writes outputs/play_lab2.mp4
    python play.py --lab 3                  # DDPG pendulum, writes outputs/play_lab3.mp4
    python play.py --lab 4                  # DDPG car that learned to climb (the "fixed" run of Lab 4)
    python play.py --lab 4 --variant naive  # the car that learned to stand still
    python play.py --lab 3 --window         # live window instead of a video
    python play.py --lab 3 --checkpoint outputs/lab3_td3_seed2.pt --episodes 3

Without --window the frames are written to an mp4 file and nothing opens, which works on any machine.
With --window a viewer opens. On macOS the MuJoCo viewer (labs 3) needs to be started with `mjpython`
instead of `python`: `mjpython play.py --lab 3 --window`.
"""
import argparse

import numpy as np
import torch
import gymnasium as gym

from common import mlp, out_dir


def load_dqn(path):
    """Rebuild the Lab 2 network from its checkpoint. Returns (policy_fn, env_name)."""
    ck = torch.load(path, weights_only=False)
    net = mlp([2, *ck["hidden"], 3])
    net.load_state_dict(ck["state_dict"])
    env = gym.make(ck["env"])
    low, high = env.observation_space.low.astype(np.float32), env.observation_space.high.astype(np.float32)

    @torch.no_grad()
    def act(obs):
        x = np.asarray(obs, dtype=np.float32)
        if ck["normalise"]:
            x = 2.0 * (x - low) / (high - low) - 1.0
        return int(net(torch.as_tensor(x).unsqueeze(0)).argmax(dim=1).item())
    return act, ck["env"]


def default_checkpoint(lab, variant):
    return {2: "lab2_dqn.pt", 3: "lab3_ddpg.pt", 4: f"lab4_{variant}.pt"}[lab]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--lab", type=int, choices=[2, 3, 4], required=True)
    ap.add_argument("--variant", choices=["fixed", "naive"], default="fixed", help="Lab 4 only")
    ap.add_argument("--checkpoint", default=None, help="path to a .pt file (default: the one the lab wrote)")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--window", action="store_true", help="open a live window instead of writing an mp4")
    ap.add_argument("--seed", type=int, default=10_000)
    ap.add_argument("--max-steps", type=int, default=500, help="stop an episode after this many steps (video length)")
    args = ap.parse_args()

    path = args.checkpoint or str(out_dir() / default_checkpoint(args.lab, args.variant))
    try:
        if args.lab == 2:
            act, env_name = load_dqn(path)
        else:
            from lab3_ddpg_inverted_pendulum import load_actor
            act, env_name = load_actor(path)
    except FileNotFoundError:
        raise SystemExit(f"No checkpoint at {path}. Run the lab first (python lab{args.lab}_*.py).")

    env = gym.make(env_name, render_mode="human" if args.window else "rgb_array")
    frames, fps = [], env.metadata.get("render_fps", 30)
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total, done, steps = 0.0, False, 0
        while not done and steps < args.max_steps:
            obs, r, terminated, truncated, _ = env.step(act(obs))
            total += float(r)
            steps += 1
            done = terminated or truncated
            if not args.window:
                frames.append(env.render())
        print(f"episode {ep + 1}: {steps} steps, return {total:.1f}"
              f"{'  (reached the end of the episode)' if done else '  (video stopped at --max-steps)'}")
    env.close()

    if not args.window:
        import imageio
        target = out_dir() / f"play_lab{args.lab}{'_' + args.variant if args.lab == 4 else ''}.mp4"
        imageio.mimsave(target, frames, fps=fps)
        print(f"Saved {target}  ({len(frames)} frames at {fps} fps)")


if __name__ == "__main__":
    main()
