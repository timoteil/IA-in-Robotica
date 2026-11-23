"""
Lab 2 -- DQN: replace the table by a neural network
===================================================
Same car, same three actions, same -1 per step as Lab 1. The only thing that changes is where Q lives.
Lab 1 cut the plane into boxes and kept one row per box. Now Q is a function with weights θ:

        Q_θ(s, a)       a network   (position, velocity)  ->  [Q(left), Q(nothing), Q(right)]

There are no boxes and no rows. Nearby states share weights, so what is learned at one state moves the
estimate at its neighbours. That is the gain (it generalises and the size does not depend on a grid) and the
danger (an update at one state can damage the value at another).

The update is Lab 1's update with the table write replaced by a gradient step on the squared TD error:

        y        =  r + gamma * (1 - terminated) * max_a' Q_θ⁻(s', a')           the target
        loss     =  mean over a minibatch of ( y - Q_θ(s, a) )²
        θ       <-  θ - lr * ∇_θ loss                                           (Adam does the stepping)

Fitting a moving target with a network that generalises is unstable, and two additions make DQN work. Each is
a few lines in this file, and each can be switched off to see why it is there:

  REPLAY BUFFER    Consecutive transitions of one episode are almost the same point, so training on them in order
                   is like fitting a curve to one tiny corner of the plane, and then forgetting it. Instead every
                   transition is stored, and each update trains on a random minibatch of old ones.   --no-replay
  TARGET NETWORK   If y is computed from the network being trained, every step moves the goalpost the network
                   chases. Instead y comes from a frozen copy Q_θ⁻, refreshed every `--target-every` steps.
                                                                                                    --no-target

Exploration is epsilon-greedy (as in Lab 1), with epsilon falling linearly from 1 to 0.05.

Training episodes start at random places on the hill (`--no-random-starts` turns this off). A network that
always starts at the bottom and pushes at random never sees the flag, and then has nothing to learn from; the
exercises come back to this. The evaluation episodes always start at the bottom, as in Lab 1.

The network sees the state scaled to [-1, 1]. Position is about 1 and velocity about 0.05, and a network
cannot learn the effect of two inputs of such different size without help.

Run it
------
    python lab2_dqn_mountaincar.py                  # about a minute or two
    python lab2_dqn_mountaincar.py --trace          # print one minibatch update, to check by hand
    python lab2_dqn_mountaincar.py --no-replay      # learn from the latest transition only
    python lab2_dqn_mountaincar.py --no-target      # the target comes from the network being trained
    python lab2_dqn_mountaincar.py --compare        # default, --no-replay, --no-target on one plot
    python lab2_dqn_mountaincar.py --no-random-starts   # training episodes all start at the bottom
    python lab2_dqn_mountaincar.py --seed 3         # same code, other random numbers
    python lab2_dqn_mountaincar.py --show           # also open the plot window

Exercises
---------
 1. Run --trace. Take one row of the printed minibatch and check by hand that y = r + gamma * max Q_target(s')
    (and that y = r when d = 1). What would y be for that row if Q_target were the online network?
 2. Compare the default run with --no-replay and --no-target (or run --compare). What happens to each learning
    curve? Which one is more harmful here? Look at the "mean Q" column of the CSV in outputs/: what does it do?
 3. The network never saw most of the states in the plane, yet it has a value for all of them. Compare its value
    map with the table of Lab 1 (the right-hand panel). Where do they agree, and where is the network guessing?
 4. Change --target-every to 10, 100, 500, 5000. What is lost when the target is refreshed too often? Too rarely?
 5. Change --buffer to 500. The buffer then holds only the last two or three episodes. What does the curve do?
 6. Remove the line that scales the state to [-1, 1] (set `NORMALISE = False`) and run again.
 7. Run --no-random-starts. The table of Lab 1 found the flag from the bottom of the hill and the network does
    not. Print the training returns in outputs/lab2_dqn-bottom_seed0.csv: how many episodes ever reached the flag?
    Why does a network that generalises have less of Lab 1's optimism, and what does the replay buffer then contain?
"""
import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

from common import (AMBER, BLUE, CORAL, GREEN, INK, SLATE, CSVLog, ReplayBuffer, evaluate, get_pyplot, mlp,
                    n_params, out_dir, read_csv, seed_everything, smooth, soft_update, style_axes)

ENV_NAME = "MountainCar-v0"
NORMALISE = True                       # exercise 6 turns this off
HIDDEN = (128, 128)
ACTION_NAMES = ["left", "nothing", "right"]


# ======================================================================= THE STATE SCALER
class Scaler:
    """Maps the state box [low, high] onto [-1, 1] in every coordinate."""

    def __init__(self, env):
        self.low = env.observation_space.low.astype(np.float32)
        self.high = env.observation_space.high.astype(np.float32)

    def __call__(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if not NORMALISE:
            return obs
        return 2.0 * (obs - self.low) / (self.high - self.low) - 1.0


# ======================================================================= THE AGENT
class DQN:
    """Online network, frozen target network, and the three lines that make it learn."""

    def __init__(self, obs_dim, n_actions, lr, gamma, use_target):
        self.n_actions = n_actions
        self.gamma = gamma
        self.use_target = use_target
        self.q = mlp([obs_dim, *HIDDEN, n_actions])             # Q_θ
        self.q_target = copy.deepcopy(self.q)                  # Q_θ⁻ starts as an exact copy
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs_n, epsilon, rng):
        """epsilon-greedy: with probability epsilon a random action, otherwise argmax_a Q_θ(s, a)."""
        if rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        return int(self.q(torch.as_tensor(obs_n).unsqueeze(0)).argmax(dim=1).item())

    @torch.no_grad()
    def targets(self, r, s2, d):
        """y = r + gamma * (1 - terminated) * max_a' Q⁻(s', a')."""
        net = self.q_target if self.use_target else self.q       # --no-target: chase our own tail
        return r + self.gamma * (1.0 - d) * net(s2).max(dim=1, keepdim=True).values

    def update(self, batch, trace=False):
        s, a, r, s2, d = batch
        a = a.long()                                              # stored as a float column; an index here
        y = self.targets(r, s2, d)                                # a constant: no gradient flows into it
        q_all = self.q(s)
        q_sa = q_all.gather(1, a)                                 # Q_θ(s, a) of the action actually taken
        loss = ((y - q_sa) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if trace:
            print("One minibatch update (first 6 rows of the batch). State shown scaled to [-1, 1].")
            print(f"  gamma = {self.gamma}, target network {'ON' if self.use_target else 'OFF'}")
            print("   row   position  velocity   action   r    d   max Q(s')   y = r + γ(1-d)·max Q(s')   Q(s,a)   TD error")
            with torch.no_grad():
                net = self.q_target if self.use_target else self.q
                mx = net(s2).max(dim=1).values
            for i in range(min(6, len(s))):
                print(f"   {i:3d}   {s[i, 0]:+8.3f}  {s[i, 1]:+8.3f}   {ACTION_NAMES[int(a[i, 0])]:7}  "
                      f"{r[i, 0]:+.0f}  {d[i, 0]:.0f}   {mx[i]:+9.3f}   {y[i, 0]:+24.3f}   "
                      f"{q_sa[i, 0].item():+7.3f}   {(y[i, 0] - q_sa[i, 0]).item():+7.3f}")
            print(f"  loss = mean of (y - Q)² over the {len(s)} rows = {loss.item():.4f}\n")
        return float(loss.item()), float(q_sa.mean().item())

    def greedy(self, scaler):
        """A function obs -> action with no exploration, for evaluation and for play.py."""
        @torch.no_grad()
        def act(obs):
            return int(self.q(torch.as_tensor(scaler(obs)).unsqueeze(0)).argmax(dim=1).item())
        return act


# ======================================================================= TRAINING LOOP
def reset_for_training(env, rng, seed=None, random_starts=True):
    """Start an episode. With random_starts the car begins at a random (position, velocity) on the hill.

    A car that always starts at the bottom, driven by random pushes, essentially never reaches the flag,
    so the replay buffer would hold only -1 rewards and there would be nothing to learn from. In Lab 1 the
    table's optimism found the flag by itself; a network generalises too smoothly to do that (see the
    exercises). Starting at random places is the simplest way to put a few flag transitions in the buffer.
    Evaluation always uses the normal start.
    """
    obs, _ = env.reset(seed=seed)
    if random_starts:
        low, high = env.observation_space.low, env.observation_space.high
        x = rng.uniform(low[0], 0.45)                              # anywhere on the hill, short of the flag
        v = rng.uniform(low[1], high[1])
        env.unwrapped.state = np.array([x, v])
        obs = np.array([x, v], dtype=np.float32)
    return obs


def train(seed=0, steps=100_000, lr=1e-3, gamma=0.99, batch=64, buffer=50_000, learn_start=1_000,
          target_every=500, eps_final=0.05, eps_fraction=0.5, use_replay=True, use_target=True, random_starts=True,
          eval_every=5_000, trace=False, tag="dqn", verbose=True):
    """DQN on MountainCar-v0. Returns a dict with the agent, the scaler, and the logged curves."""
    rng = seed_everything(seed)
    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    scaler = Scaler(env)
    agent = DQN(2, env.action_space.n, lr, gamma, use_target)
    buf = ReplayBuffer(2, 1, buffer, rng)
    log = CSVLog(out_dir() / f"lab2_{tag}_seed{seed}.csv",
                 ["step", "episode", "return", "epsilon", "loss", "mean_q", "eval_return"])

    obs = reset_for_training(env, rng, seed, random_starts)
    s = scaler(obs)
    ep, ep_ret, losses, qs = 1, 0.0, [], []
    best_eval, best_state = -np.inf, None
    episode_returns, eval_points = [], []
    t0 = time.time()
    traced = False
    for step in range(1, steps + 1):
        # ---- act: epsilon falls linearly from 1 to eps_final over the first eps_fraction of training
        eps = max(eps_final, 1.0 - (1.0 - eps_final) * step / (eps_fraction * steps))
        a = agent.act(s, eps, rng)
        obs2, r, terminated, truncated, _ = env.step(a)
        s2 = scaler(obs2)
        buf.add(s, a, r, s2, terminated)                      # terminated only. A time-out is not "the end"
        s, ep_ret = s2, ep_ret + r

        # ---- learn
        if step >= learn_start:
            if use_replay:
                batch_data = buf.sample(batch)                # random old transitions
            else:
                batch_data = buf.latest()                     # --no-replay: the transition we just saw
            do_trace = trace and not traced
            loss, mean_q = agent.update(batch_data, trace=do_trace)
            traced = traced or do_trace
            losses.append(loss), qs.append(mean_q)
            if step % target_every == 0:
                soft_update(agent.q_target, agent.q, tau=1.0)   # hard copy: Q_θ⁻ <- Q_θ

        # ---- end of an episode
        if terminated or truncated:
            episode_returns.append(ep_ret)
            log.write(step=step, episode=ep, **{"return": ep_ret}, epsilon=eps,
                      loss=np.mean(losses) if losses else "", mean_q=np.mean(qs) if qs else "")
            ep, ep_ret, losses, qs = ep + 1, 0.0, [], []
            obs = reset_for_training(env, rng, None, random_starts)
            s = scaler(obs)

        # ---- periodic evaluation of the greedy policy
        if step % eval_every == 0:
            ev = evaluate(eval_env, agent.greedy(scaler), episodes=5)
            eval_points.append((step, float(ev.mean())))
            log.write(step=step, eval_return=float(ev.mean()))
            if ev.mean() > best_eval:
                best_eval = float(ev.mean())
                best_state = copy.deepcopy(agent.q.state_dict())
            if verbose:
                recent = np.mean(episode_returns[-20:]) if episode_returns else float("nan")
                print(f"  step {step:7d}  epsilon {eps:.2f}  train return (last 20) {recent:7.1f}  "
                      f"eval return {ev.mean():7.1f}  ({time.time() - t0:.0f}s)")
    log.close()
    final_state = copy.deepcopy(agent.q.state_dict())
    return {"agent": agent, "scaler": scaler, "returns": np.array(episode_returns), "eval": np.array(eval_points),
            "best_state": best_state, "final_state": final_state, "best_eval": best_eval, "seed": seed,
            "csv": log.path}


def save_checkpoint(result, path):
    torch.save({"state_dict": result["final_state"], "hidden": HIDDEN, "normalise": NORMALISE,
                "env": ENV_NAME}, path)


# ======================================================================= PLOTTING
def value_and_policy(agent, scaler, env, n=120):
    """The network's max_a Q and argmax_a Q on an n x n grid of (position, velocity)."""
    lo, hi = env.observation_space.low, env.observation_space.high
    xs, vs = np.linspace(lo[0], hi[0], n), np.linspace(lo[1], hi[1], n)
    grid = np.array([[x, v] for v in vs for x in xs], dtype=np.float32)
    with torch.no_grad():
        q = agent.q(torch.as_tensor(np.stack([scaler(g) for g in grid]))).numpy()
    return q.max(axis=1).reshape(n, n), q.argmax(axis=1).reshape(n, n)


def make_figure(result, args, comparison=None, name="lab2_dqn_mountaincar.png"):
    from matplotlib.colors import ListedColormap
    plt = get_pyplot(args.show)
    env = gym.make(ENV_NAME)
    agent, scaler = result["agent"], result["scaler"]
    table_path = out_dir() / "lab1_q_table.npy"
    have_table = table_path.exists()
    ncols = 4 if have_table else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.1), gridspec_kw={"width_ratios": [1.15] + [1] * (ncols - 1)})

    ax = axes[0]
    if comparison:
        colours = {"dqn": GREEN, "no-replay": CORAL, "no-target": AMBER}
        for name, res in comparison.items():
            ax.plot(res["eval"][:, 0], res["eval"][:, 1], color=colours.get(name, SLATE), lw=2, marker="o", ms=3, label=name)
        style_axes(ax, "Greedy evaluation return (5 episodes)", "environment step", "return")
    else:
        ax.plot(np.cumsum(np.ones(len(result["returns"]))), result["returns"], color=SLATE, lw=0.6, alpha=0.5)
        ax.plot(smooth(result["returns"], 20), color=GREEN, lw=2.0, label="training return, mean of last 20")
        ax.set_xlabel("episode")
        ax2 = ax.twiny()
        ax2.plot(result["eval"][:, 0], result["eval"][:, 1], color=AMBER, lw=2, marker="o", ms=3, label="greedy evaluation")
        ax2.set_xlabel("environment step", color=AMBER)
        ax2.spines[["right"]].set_visible(False)
        style_axes(ax, "Learning curve", None, "return  (-steps; time out = -200)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower right")
    if comparison:
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.axhline(-200, color=CORAL, lw=1, ls="--")

    lo, hi = env.observation_space.low, env.observation_space.high
    extent = [lo[0], hi[0], lo[1], hi[1]]
    value, policy = value_and_policy(agent, scaler, env)
    path = []
    obs, _ = env.reset(seed=10_000)
    path.append(obs)
    done = False
    act = agent.greedy(scaler)
    while not done:
        obs, _, te, tr, _ = env.step(act(obs))
        path.append(obs)
        done = te or tr
    path = np.array(path)
    vmin, vmax = -100, 0
    if have_table:
        vmin = min(vmin, float(np.load(table_path).max(axis=2).min()))
    im = axes[1].imshow(value, origin="lower", extent=extent, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].plot(path[:, 0], path[:, 1], color="white", lw=1.4)
    style_axes(axes[1], "network: max Q over actions", "position", "velocity")
    axes[2].imshow(policy, origin="lower", extent=extent, aspect="auto", vmin=0, vmax=2,
                   cmap=ListedColormap([BLUE, "#DDDDDD", AMBER]))
    axes[2].plot(path[:, 0], path[:, 1], color=INK, lw=1.2)
    axes[2].axvline(0.5, color=GREEN, lw=2)
    style_axes(axes[2], "network: greedy action (blue left, amber right)", "position", "velocity")
    if have_table:
        Q1 = np.load(table_path)
        seen = np.any(Q1 != 0, axis=2).T
        im = axes[3].imshow(np.where(seen, Q1.max(axis=2).T, np.nan), origin="lower", extent=extent, aspect="auto",
                            cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=axes[3], fraction=0.046)
        style_axes(axes[3], f"Lab 1 table: max Q  ({Q1.shape[0]}x{Q1.shape[1]} boxes)", "position", "velocity")
    fig.suptitle(f"DQN on MountainCar-v0, seed {result['seed']}: network with {n_params(agent.q):,} weights "
                 f"vs table with {np.load(table_path).size if have_table else 1200:,} entries" if not comparison else
                 "DQN ablations on MountainCar-v0 (same seed, same hyper-parameters)", color=INK)
    fig.tight_layout()
    name = "lab2_dqn_compare.png" if comparison else name
    fig.savefig(out_dir() / name, dpi=150)
    print(f"Saved {out_dir() / name}")
    if args.show:
        plt.show()


# ======================================================================= MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--buffer", type=int, default=50_000, help="replay buffer capacity")
    ap.add_argument("--target-every", type=int, default=500, help="steps between copies of Q into the target network")
    ap.add_argument("--no-replay", action="store_true", help="train on the latest transition only")
    ap.add_argument("--no-target", action="store_true", help="compute y from the network being trained")
    ap.add_argument("--no-random-starts", action="store_true",
                    help="always start training episodes at the bottom of the hill (see exercise 7)")
    ap.add_argument("--compare", action="store_true", help="run default, --no-replay and --no-target and plot all three")
    ap.add_argument("--trace", action="store_true", help="print the first minibatch update")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="open the plot window as well as saving the PNG")
    args = ap.parse_args()

    common = dict(seed=args.seed, steps=args.steps, lr=args.lr, gamma=args.gamma, batch=args.batch,
                  buffer=args.buffer, target_every=args.target_every,
                  random_starts=not args.no_random_starts)
    print(f"{ENV_NAME}: state (position, velocity), 3 actions.  Network {2}-{HIDDEN[0]}-{HIDDEN[1]}-3, "
          f"{n_params(mlp([2, *HIDDEN, 3])):,} weights.\n")

    if args.compare:
        results = {}
        for name, kw in [("dqn", {}), ("no-replay", {"use_replay": False}), ("no-target", {"use_target": False})]:
            print(f"--- {name}")
            results[name] = train(tag=name, **common, **kw)
        make_figure(results["dqn"], args, comparison=results)
        for name, res in results.items():
            print(f"{name:10} final eval {res['eval'][-1, 1]:7.1f}   best eval {res['best_eval']:7.1f}")
        return

    use_replay, use_target = not args.no_replay, not args.no_target
    tag = "dqn" if use_replay and use_target else ("no-replay" if not use_replay else "no-target")
    if args.no_random_starts:
        tag += "-bottom"                                           # keep the default run's files intact
    print(f"replay buffer {'ON' if use_replay else 'OFF'} (capacity {args.buffer}),  "
          f"target network {'ON' if use_target else 'OFF'} (refreshed every {args.target_every} steps)\n")
    result = train(trace=args.trace, tag=tag, use_replay=use_replay, use_target=use_target, **common)
    agent, scaler = result["agent"], result["scaler"]
    ev = evaluate(gym.make(ENV_NAME), agent.greedy(scaler), episodes=20)
    print(f"\nGreedy policy over 20 starts: reaches the flag in {np.mean(ev > -200):.0%} of runs, "
          f"mean return {ev.mean():.1f}  (Lab 1's table: about -150; best possible about -100).")
    save_checkpoint(result, out_dir() / f"lab2_{tag}_seed{args.seed}.pt")
    print(f"Saved the network to {out_dir() / f'lab2_{tag}_seed{args.seed}.pt'} and the log to {result['csv']}")
    if tag == "dqn":
        save_checkpoint(result, out_dir() / "lab2_dqn.pt")      # the file play.py loads
    make_figure(result, args, name="lab2_dqn_mountaincar.png" if tag == "dqn" else f"lab2_{tag}.png")


if __name__ == "__main__":
    main()
