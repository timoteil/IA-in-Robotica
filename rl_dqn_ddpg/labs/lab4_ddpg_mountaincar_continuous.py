"""
Lab 4 -- DDPG and the exploration problem: MountainCarContinuous
================================================================
The hill from Labs 1 and 2 again, but now the push is a real number a in [-1, 1] and the reward is

        r  =  -0.1 * a²   at every step          (the engine costs energy)
        r  =  +100        once, when the car reaches the flag      (and the episode ends)

The agent in this lab is the DDPG of Lab 3, imported unchanged. Nothing in the algorithm differs. Only the
environment does, and DDPG fails on it with the settings that worked on the pendulum.

Why it fails. Rewards come from only two places: a small penalty for pushing, and a huge prize that the
car cannot reach without first rocking back and forth. Until the car has been to the flag once, the critic
has seen NOTHING but penalties, and the best thing a critic can learn from penalties is that pushing is
costly. The actor, which climbs the critic, learns to push with force zero. Then the policy is
"stand still", the exploration noise around zero is far too small to carry the car up the hill, the car never
reaches the flag, and the critic never hears about the prize. It is a closed loop with a stable fixed point.

It is the continuous-action version of the greedy agent in Lab 2 of the 1-hour course: an agent that only
does what already looks best, in a world where it has not yet seen the best.

The cure is not a change of algorithm, it is a change of the DATA. Gaussian noise with a new, independent draw
at every step averages out: push left, push right, push left, and the car stays where it was. To climb, the car
has to push the same way for many steps in a row. Ornstein-Uhlenbeck noise is correlated in time:

        x  <-  x + theta * (0 - x) + sigma * N(0, 1)        small theta  =  long memory

so it hands the car a run of pushes in one direction. One episode that reaches the flag puts +100 in the replay
buffer, and the critic and then the actor can learn from it.

Two presets, selected with --run:
    naive    Gaussian noise, sigma = 0.1, warm-up 1000 steps       (what worked on the pendulum)
    fixed    Ornstein-Uhlenbeck noise, sigma = 0.5, theta = 0.15, warm-up 1000 steps

Run it
------
    python lab4_ddpg_mountaincar_continuous.py                        # both presets, about 3 minutes
    python lab4_ddpg_mountaincar_continuous.py --run naive            # only the one that fails
    python lab4_ddpg_mountaincar_continuous.py --run fixed
    python lab4_ddpg_mountaincar_continuous.py --run naive --sigma 0.5          # louder, but still independent
    python lab4_ddpg_mountaincar_continuous.py --run fixed --theta 1.0          # OU with no memory: white noise
    python lab4_ddpg_mountaincar_continuous.py --run fixed --warmup 0
    python lab4_ddpg_mountaincar_continuous.py --seed 3
    python lab4_ddpg_mountaincar_continuous.py --show
    python play.py --lab 4                                             # watch the car that learned

Exercises
---------
 1. Run the default. What is the greedy return of the naive run, exactly? What does its actor output? (The script
    prints the mean force |π(s)| over every state the run visited.) Why is that return a stable result, not a
    bad luck?
 2. Look at the coverage plots: the states visited in training. How much of the (position, velocity) plane did
    the naive run see, and how much did the fixed run? Where is the flag, and where does the naive run's data end?
 3. Open the CSV of the naive run and look at the `mean_q` column. What is the critic's value for a state in an
    episode where nothing but -0.1 * a² is received? Why can the critic not know that the flag exists?
 4. Run --run naive --sigma 0.5: more noise, still Gaussian. Does it reach the flag? Then --run fixed --theta 1.0, which
    is OU noise with no memory. Explain both outcomes with the one word "independent".
 5. Run --run fixed --warmup 0. Is the warm-up needed here? In which of the two labs would you have expected it to
    matter more, and why?
 6. The trained car reaches the flag with a return of about 90, not 100. Where do the missing 10 points go? Estimate the
    number of steps and the average force from the CSV.
 7. The Lab 3 pendulum trained with Gaussian noise of 0.1. Run `python lab3_ddpg_inverted_pendulum.py --noise ou`.
    What does correlated noise cost there?
"""
import argparse

import numpy as np
import gymnasium as gym

from common import (AMBER, BLUE, CORAL, GREEN, INK, SLATE, evaluate, get_pyplot, out_dir, smooth, style_axes)
from lab3_ddpg_inverted_pendulum import save_checkpoint, train

ENV_NAME = "MountainCarContinuous-v0"
PRESETS = {
    "naive": dict(noise="gaussian", sigma=0.1, theta=0.15, warmup=1000),
    "fixed": dict(noise="ou", sigma=0.5, theta=0.15, warmup=1000),
}
LABELS = {"naive": "naive: Gaussian σ=0.1", "fixed": "fixed: OU σ=0.5, θ=0.15"}
COLOURS = {"naive": CORAL, "fixed": GREEN}


def describe(name, cfg):
    if cfg["noise"] == "gaussian":
        return f"{name}: Gaussian noise sigma {cfg['sigma']}, warm-up {cfg['warmup']} steps"
    return f"{name}: OU noise sigma {cfg['sigma']}, theta {cfg['theta']}, warm-up {cfg['warmup']} steps"


def summarise(name, res):
    """Greedy evaluation and the quantities the exercises ask about."""
    env = gym.make(ENV_NAME)
    agent = res["agent"]
    ev = evaluate(env, agent.policy(), episodes=20)
    vis = res["visited"]
    forces = np.abs(np.array([agent.act(s)[0] for s in vis[::10].astype(np.float32)]))
    reach_train = int(np.sum(res["returns"] > 0))
    first = next((i + 1 for i, x in enumerate(res["returns"]) if x > 0), None)
    print(f"  {name:6} training episodes that reached the flag: {reach_train} of {len(res['returns'])} "
          f"(first one: episode {first if first else 'never'})")
    print(f"         states visited: position range [{vis[:, 0].min():+.2f}, {vis[:, 0].max():+.2f}] "
          f"(flag at +0.45 .. +0.50)")
    print(f"         actor's mean |force| over those states: {forces.mean():.3f}")
    print(f"         greedy return over 20 starts: mean {ev.mean():.1f}  (min {ev.min():.1f}, max {ev.max():.1f}),  "
          f"{np.mean(ev > 50):.0%} reach the flag\n")
    return ev


def make_figure(results, args):
    plt = get_pyplot(args.show)
    names = list(results)
    fig, axes = plt.subplots(1, 3 + (1 if "fixed" in results else 0), figsize=(5.2 * (3 + ("fixed" in results)), 4.3))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for name in names:
        res = results[name]
        ax.plot(res["eval"][:, 0], res["eval"][:, 1], color=COLOURS.get(name, SLATE), lw=2, marker="o", ms=3,
                label=LABELS.get(name, name))
    ax.axhline(0, color=INK, lw=1, ls="--")
    ax.text(2500, 6, "stand still: return 0", fontsize=8, color=INK)
    style_axes(ax, "Greedy evaluation return (5 episodes)", "environment step", "return")
    ax.legend(frameon=False, fontsize=8, loc="center right")

    for k, name in enumerate(names):
        ax = axes[1 + k]
        vis = results[name]["visited"]
        sub = vis[:: max(1, len(vis) // 6000)]
        ax.scatter(sub[:, 0], sub[:, 1], s=2, color=COLOURS.get(name, SLATE), alpha=0.35)
        ax.axvline(0.45, color=GREEN, lw=2)
        ax.text(0.44, 0.066, "flag", color=GREEN, ha="right", fontsize=8, va="top")
        ax.axvspan(-0.6, -0.4, ymin=0.0, ymax=1.0, color=AMBER, alpha=0.12)
        ax.text(-0.5, -0.066, "start", color=INK, ha="center", fontsize=8)
        ax.set_xlim(-1.25, 0.65), ax.set_ylim(-0.075, 0.075)
        style_axes(ax, f"states visited in training: {LABELS.get(name, name)}", "position", "velocity")

    if "fixed" in results:
        ax = axes[-1]
        agent = results["fixed"]["agent"]
        import torch
        n = 100
        xs, vs = np.linspace(-1.2, 0.6, n), np.linspace(-0.07, 0.07, n)
        grid = np.array([[x, v] for v in vs for x in xs], dtype=np.float32)
        with torch.no_grad():
            force = agent.actor(torch.as_tensor(grid))[:, 0].numpy().reshape(n, n)
        im = ax.imshow(force, origin="lower", extent=[-1.2, 0.6, -0.07, 0.07], aspect="auto", cmap="coolwarm",
                       vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        env = gym.make(ENV_NAME)
        obs, _ = env.reset(seed=10_000)
        path, done = [obs], False
        while not done:
            obs, _, te, tr, _ = env.step(agent.act(obs))
            path.append(obs)
            done = te or tr
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], color=INK, lw=1.3)
        style_axes(ax, "fixed run: the actor's force, and one greedy drive", "position", "velocity")
    fig.suptitle(f"DDPG on {ENV_NAME}, seed {args.seed}: same agent, different exploration", color=INK)
    fig.tight_layout()
    path = out_dir() / "lab4_ddpg_mountaincar_continuous.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if args.show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", choices=["naive", "fixed", "both"], default="both")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--noise", choices=["gaussian", "ou"], default=None, help="override the preset")
    ap.add_argument("--sigma", type=float, default=None, help="override the preset")
    ap.add_argument("--theta", type=float, default=None, help="override the preset (OU only)")
    ap.add_argument("--warmup", type=int, default=None, help="override the preset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="open the plot window as well as saving the PNG")
    args = ap.parse_args()

    env = gym.make(ENV_NAME)
    print(f"{ENV_NAME}: state (position, velocity), action in [{env.action_space.low[0]:.0f}, "
          f"{env.action_space.high[0]:.0f}]; reward -0.1 a² per step and +100 at the flag; "
          f"at most {env.spec.max_episode_steps} steps.\n")
    names = ["naive", "fixed"] if args.run == "both" else [args.run]
    results = {}
    for name in names:
        cfg = dict(PRESETS[name])
        for key in ("noise", "sigma", "theta", "warmup"):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        print(f"--- {describe(name, cfg)}")
        results[name] = train(env_name=ENV_NAME, seed=args.seed, steps=args.steps, eval_every=2_500,
                              tag=f"lab4_{name}", **cfg)
        print()
    print("Summary")
    for name, res in results.items():
        summarise(name, res)
        save_checkpoint(res, out_dir() / f"lab4_{name}_seed{args.seed}.pt")
        if args.seed == 0:
            save_checkpoint(res, out_dir() / f"lab4_{name}.pt")      # the files play.py loads
    make_figure(results, args)


if __name__ == "__main__":
    main()
