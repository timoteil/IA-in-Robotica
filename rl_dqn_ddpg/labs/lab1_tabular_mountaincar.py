"""
Lab 1 -- Tabular Q-learning when the state is a pair of real numbers
====================================================================
The car in MountainCar-v0 is under-powered: it cannot drive straight up the right hill. It has to
rock back and forth to build up speed. The state is two real numbers, s = (position, velocity),
and there are three actions: push left, do nothing, push right. Every step costs -1, and the episode
ends when the car reaches the flag (position 0.5) or after 200 steps.

Lab 2 of the 1-hour course filled in a table Q[state, action]. Here the state is not one of 24 cells,
it is a point in the plane, and a table needs a finite list of rows. The simplest fix is to cut the
plane into an N x N grid of boxes and give each box one row:

        box(s)   =  (which position bin, which velocity bin)
        Q[box, a] <- Q[box, a] + alpha * ( r + gamma * (1 - terminated) * max_a' Q[box(s'), a'] - Q[box, a] )

This is the SAME update as in the 1-hour course, with alpha < 1 because the boxes are shared by many
different states, so the world looks random from the table's point of view. Nothing else changed.

Two things to watch.
  * The table works. With 20 x 20 boxes it has 400 x 3 = 1200 numbers and the car learns to climb.
  * The table grows as N^d. A 2-dimensional state needs N^2 rows. The inverted pendulum of Lab 3 has 4
    state numbers, Ant has 105: you cannot write down N^105 rows. That is the reason for Lab 2.

Exploration here is epsilon-greedy: with probability epsilon the agent pushes at random, otherwise it takes
argmax_a Q. Epsilon starts high and decays linearly. (The Q-table starts at 0 and every reward is -1, so
every action the agent has tried looks WORSE than the ones it has not: the table itself pushes the agent
to try something new. This is called optimistic initialisation.)

Run it
------
    python lab1_tabular_mountaincar.py                    # 20 x 20 boxes
    python lab1_tabular_mountaincar.py --bins 8           # coarse boxes
    python lab1_tabular_mountaincar.py --bins 40          # fine boxes
    python lab1_tabular_mountaincar.py --trace 8          # print the first 8 updates, to check by hand
    python lab1_tabular_mountaincar.py --seed 3           # same code, other random numbers
    python lab1_tabular_mountaincar.py --show             # also open the plot window

Exercises
---------
 1. Run with --trace 8. For one printed line, check by hand that target = r + gamma * max Q(box') and that
    new Q = old Q + alpha * TD error. Why is the TD error so small, and why is every new Q slightly negative?
 2. Run with --bins 5, 10, 20, 40. For each, note the table size and the fraction of evaluation runs that
    reach the flag. Why can very fine boxes be WORSE than medium ones with the same number of episodes?
 3. Look at the value map. Why does it have a spiral shape? (Hint: what does the trajectory do in the
    position/velocity plane?) Which part of the plane has the lowest value, and why?
 4. Run --epsilon 0 (never explores at random) and compare with the default. Does the agent still find the
    flag, and how soon? What do the zeros in the table have to do with it? Then run --epsilon 0 --q0 -150
    (a table that starts pessimistic) and explain what changes. Why is -100 the borderline? (Compute the
    value of a state in which every step costs -1 forever, with gamma = 0.99.)
 5. Count the rows a table would need for InvertedPendulum (4 numbers) with 20 boxes per number, and for
    Ant (105 numbers). Then read the first lines of Lab 2.
"""
import argparse
import time

import numpy as np
import gymnasium as gym

from common import (AMBER, BLUE, CORAL, GREEN, INK, SLATE, evaluate, get_pyplot, out_dir,
                    seed_everything, smooth, style_axes)

ACTION_NAMES = ["push left", "do nothing", "push right"]


# ======================================================================= THE DISCRETISER
class Boxes:
    """Cuts [low, high] of every state variable into `bins` equal intervals.

    box(s) returns a tuple of integers, one per state variable, and that tuple indexes the table.
    """

    def __init__(self, env, bins: int):
        self.bins = bins
        self.low = env.observation_space.low.astype(float)
        self.width = (env.observation_space.high.astype(float) - self.low) / bins

    def __call__(self, obs):
        idx = ((np.asarray(obs, dtype=float) - self.low) / self.width).astype(int)
        return tuple(int(i) for i in np.clip(idx, 0, self.bins - 1))

    def centre(self, i, j):
        """The (position, velocity) in the middle of box (i, j). Used for the plots."""
        return self.low[0] + (i + 0.5) * self.width[0], self.low[1] + (j + 0.5) * self.width[1]


# ======================================================================= CORE ALGORITHM
def q_learning(env, boxes, episodes=10000, gamma=0.99, alpha=0.1, epsilon=1.0, epsilon_final=0.01,
               decay_fraction=0.6, q0=0.0, seed=0, trace=0, verbose=True):
    """Tabular Q-learning with epsilon-greedy exploration. Returns (Q, returns, first_goal).

    returns[i]     the total reward of episode i+1 (that is -(number of steps), at most -200)
    first_goal     the first episode in which the car reached the flag (None if never)
    trace=n        print the first n updates, one line each
    """
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = np.full((boxes.bins, boxes.bins, n_actions), q0)        # the whole "brain": a table
    returns, first_goal = [], None
    decay_episodes = max(1, int(decay_fraction * episodes))
    printed = 0
    t0 = time.time()
    for ep in range(1, episodes + 1):
        # epsilon falls linearly from `epsilon` to `epsilon_final` over the first decay_fraction of training
        frac = min(1.0, (ep - 1) / decay_episodes)
        eps = epsilon + frac * (epsilon_final - epsilon)
        obs, _ = env.reset(seed=seed + ep)
        b = boxes(obs)
        total, done = 0.0, False
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(n_actions))                  # explore
            else:
                a = int(np.flatnonzero(Q[b] == Q[b].max())[0])    # exploit (ties -> first action)
            obs2, r, terminated, truncated, _ = env.step(a)
            b2 = boxes(obs2)

            # ---- the Q-learning update. `terminated` cuts the bootstrap, `truncated` (time out) does not.
            target = r + gamma * (0.0 if terminated else Q[b2].max())
            td_error = target - Q[b + (a,)]
            if printed < trace:
                print(f"  ep {ep} box {b} {ACTION_NAMES[a]:10} r={r:+.0f}  box' {b2}  "
                      f"target {target:+8.4f}   TD error {td_error:+8.4f}   "
                      f"Q {Q[b + (a,)]:+8.4f} -> {Q[b + (a,)] + alpha * td_error:+8.4f}")
                printed += 1
            Q[b + (a,)] += alpha * td_error

            b = b2
            total += r
            done = terminated or truncated
            if terminated and first_goal is None:
                first_goal = ep
        returns.append(total)
        if verbose and ep % max(1, episodes // 10) == 0:
            recent = np.mean(returns[-100:])
            print(f"  episode {ep:5d}   epsilon {eps:.2f}   mean return of last 100: {recent:7.1f}   "
                  f"({time.time() - t0:.0f}s)")
    return Q, np.array(returns), first_goal


def greedy_action(Q, boxes, obs):
    return int(np.argmax(Q[boxes(obs)]))


def rollout(env, act_fn, seed):
    """One episode with act_fn; returns the (position, velocity) path and whether the flag was reached."""
    obs, _ = env.reset(seed=seed)
    path, done, reached = [obs.copy()], False, False
    while not done:
        obs, _, terminated, truncated, _ = env.step(act_fn(obs))
        path.append(obs.copy())
        reached = reached or terminated
        done = terminated or truncated
    return np.array(path), reached


# ======================================================================= MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--bins", type=int, default=20, help="boxes per state variable")
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.1, help="step size")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--epsilon", type=float, default=1.0, help="starting epsilon (decays linearly)")
    ap.add_argument("--q0", type=float, default=0.0, help="initial value of every table entry")
    ap.add_argument("--trace", type=int, default=0, help="print the first N updates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="open the plot window as well as saving the PNG")
    args = ap.parse_args()

    seed_everything(args.seed)
    env = gym.make("MountainCar-v0")
    boxes = Boxes(env, args.bins)
    n_rows = args.bins ** 2
    print(f"MountainCar-v0: state = (position, velocity), 3 actions.")
    print(f"{args.bins} x {args.bins} boxes = {n_rows} rows x 3 actions = {n_rows * 3} numbers in the table.")
    print(f"The same grid on a 4-number state (InvertedPendulum) would need {args.bins ** 4:,} rows; "
          f"on Ant's 105 numbers, {float(args.bins) ** 105:.1e}.\n")
    if args.trace:
        print(f"First {args.trace} updates (alpha = {args.alpha}, gamma = {args.gamma}):")

    Q, returns, first_goal = q_learning(env, boxes, args.episodes, args.gamma, args.alpha, args.epsilon,
                                        q0=args.q0, seed=args.seed, trace=args.trace)
    print(f"\nThe flag was first reached in episode {first_goal if first_goal else 'never'}.")

    # ---- evaluation: greedy policy, 20 fresh starting positions
    eval_env = gym.make("MountainCar-v0")
    act = lambda obs: greedy_action(Q, boxes, obs)
    ev = evaluate(eval_env, act, episodes=20)
    reach = float(np.mean(ev > -200))
    print(f"Greedy policy over 20 starts: reaches the flag in {reach:.0%} of runs, "
          f"mean return {ev.mean():.1f}  (best possible is about -100).")
    visited = int(np.sum(np.any(Q != args.q0, axis=2)))
    print(f"Boxes whose row was ever updated: {visited} of {n_rows}.")

    # ---- save the table: Lab 2 draws it next to the neural network's value map
    np.save(out_dir() / "lab1_q_table.npy", Q)

    # ---- figure
    plt = get_pyplot(args.show)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    ax.plot(returns, color=SLATE, lw=0.6, alpha=0.5)
    ax.plot(smooth(returns, 100), color=GREEN, lw=2.2, label="mean of last 100")
    ax.axhline(-200, color=CORAL, lw=1, ls="--")
    ax.text(len(returns) * 0.02, -196, "time out", color=CORAL, fontsize=8, va="top")
    style_axes(ax, "Learning curve", "episode", "return  (= minus the number of steps)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    lo, hi = boxes.low, boxes.low + boxes.width * args.bins
    extent = [lo[0], hi[0], lo[1], hi[1]]
    seen = np.any(Q != args.q0, axis=2).T                      # boxes the agent ever updated
    value = Q.max(axis=2).T
    im = axes[1].imshow(np.where(seen, value, np.nan), origin="lower", extent=extent, aspect="auto",
                        cmap="viridis")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    path, reached = rollout(eval_env, act, seed=10_000)
    axes[1].plot(path[:, 0], path[:, 1], color="white", lw=1.4)
    style_axes(axes[1], f"max Q over actions   ({args.bins}x{args.bins} boxes)", "position", "velocity")

    policy = Q.argmax(axis=2).T.astype(float)
    policy[~seen] = np.nan
    from matplotlib.colors import ListedColormap
    axes[2].imshow(policy, origin="lower", extent=extent, aspect="auto", vmin=0, vmax=2,
                   cmap=ListedColormap([BLUE, "#DDDDDD", AMBER]))
    axes[2].plot(path[:, 0], path[:, 1], color=INK, lw=1.2)
    axes[2].axvline(0.5, color=GREEN, lw=2)
    style_axes(axes[2], "greedy action   (blue: left, amber: right)", "position", "velocity")
    fig.suptitle(f"Tabular Q-learning, seed {args.seed}: the white / black line is one greedy run "
                 f"({'reached the flag' if reached else 'timed out'} after {len(path) - 1} steps)", color=INK)
    fig.tight_layout()
    fig.savefig(out_dir() / "lab1_tabular_mountaincar.png", dpi=150)
    print(f"Saved {out_dir() / 'lab1_tabular_mountaincar.png'}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
