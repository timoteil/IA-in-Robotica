"""
Lab 1 -- Planning with Value Iteration
======================================
The agent KNOWS the system equations (env.model). Nothing is learned from experience:
we compute the best behaviour from Bellman's optimality equation.

        V(s)  <-  max over a  of  [ r(s,a) + gamma * V( F(s,a) ) ]

Start with V = 0 everywhere and apply that line to every state, again and again ("sweeps").
The values stop changing; the best action in each state is then simply the one that
achieves the max.

Run it
------
    python lab1_value_iteration.py                  # gamma = 0.95: head for the far goal
    python lab1_value_iteration.py --sweeps 3       # stop after 3 sweeps: watch the wave of value
    python lab1_value_iteration.py --gamma 0.5      # a short-sighted agent
    python lab1_value_iteration.py --show           # also open the plot window

Exercises
---------
 1. Run the default. Which exit does the agent choose? How many sweeps did it need?
    Compare that number with the number of moves from S to G.
 2. Run with --sweeps 1, 2, 3, ... Where is the value non-zero after k sweeps? Why there?
 3. Try --gamma 0.95, 0.8, 0.75, 0.5, 0.1. At which gamma does the agent switch to the near
    exit? Work it out by hand first: compare  -0.02 + gamma*0.3  (near exit, 2 moves)  with the
    discounted sum along the 6-move route to the goal.
 4. Change the cost of walking: --step-reward 0, -0.1, -0.5. What changes, and why?
 5. Edit LAYOUT in gridworld.py (move the exit, add a wall, change a reward). Predict the route and the
    number of sweeps before you run it.
"""
import argparse
from pathlib import Path

import numpy as np

from gridworld import CORAL, GREEN, INK, make_env, plot_grid

OUT = Path(__file__).parent / "outputs"


# ======================================================================= CORE ALGORITHM
def one_step_lookahead(env, V, s, gamma):
    """q(s,a) = r(s,a) + gamma * V(s')   with  s' = F(s,a),  for every action a.

    Needs the model: to evaluate action a we must know where it leads (s') and what it pays (r).
    Terminal states are worth 0 afterwards, so we never bootstrap from them.
    """
    q = np.zeros(env.n_actions)
    for a in range(env.n_actions):
        s2, r, done = env.model(s, a)
        q[a] = r + gamma * V[s2] * (not done)
    return q


def value_iteration(env, gamma, theta=1e-8, max_sweeps=10_000, stop_after=None, snapshots=None):
    """Value iteration.  Returns the values and the list of changes max_s |V_new(s) - V(s)|.

    Synchronous: every new value is computed from the complete table of the previous sweep.
    `stop_after=k` stops after k sweeps; `snapshots` (a list) collects V after every sweep.
    """
    V = np.zeros(env.n_states)
    changes = []
    if snapshots is not None:
        snapshots.append(V.copy())
    for sweep in range(1, max_sweeps + 1):
        V_new = V.copy()
        for s in range(env.n_states):
            if env.terminal[s]:
                continue
            V_new[s] = one_step_lookahead(env, V, s, gamma).max()     # Bellman optimality backup
        change = np.abs(V_new - V).max()
        changes.append(change)
        V = V_new
        if snapshots is not None:
            snapshots.append(V.copy())
        if change < theta or sweep == stop_after:
            break
    return V, changes


def greedy_policy(env, V, gamma, tol=1e-9):
    """Read the policy off the values:  pi(s) = argmax_a [ r(s,a) + gamma * V(F(s,a)) ].

    Ties are broken by taking the FIRST best action (up, down, left, right).
    The small tolerance stops floating-point noise from flipping equal actions.
    """
    actions = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        q = one_step_lookahead(env, V, s, gamma)
        actions[s] = np.flatnonzero(q >= q.max() - tol)[0]
    return actions


# ======================================================================= DEMO
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    ap.add_argument("--theta", type=float, default=1e-8, help="stop when the biggest change is below this")
    ap.add_argument("--step-reward", type=float, default=-0.02, help="reward for every move")
    ap.add_argument("--sweeps", type=int, default=None, help="stop after this many sweeps")
    ap.add_argument("--show", action="store_true", help="open the plot window as well as saving the PNG")
    args = ap.parse_args()

    env = make_env(step_reward=args.step_reward)
    print(f"GridWorld: {env.n_states} states, {env.n_actions} actions, gamma={args.gamma}, "
          f"step reward {args.step_reward}\n")
    print(env.render(), "\n")

    V, changes = value_iteration(env, args.gamma, args.theta, stop_after=args.sweeps)
    actions = greedy_policy(env, V, args.gamma)
    if args.sweeps is not None and len(changes) == args.sweeps and changes[-1] >= args.theta:
        print(f"Stopped after {len(changes)} sweeps (largest change in the last sweep: {changes[-1]:.3f}).\n")
    else:
        print(f"Value iteration converged after {len(changes)} sweeps "
              f"(largest change in the last sweep: {changes[-1]:.1e}).\n")
    print("Values V(s) and the greedy policy:")
    print(env.render(values=V, actions=actions), "\n")

    path = env.follow_policy(actions)
    route = " -> ".join(f"({env.cells[s][0]},{env.cells[s][1]})" for s in path)
    end = {"G": "the far GOAL", "E": "the near EXIT"}.get(env.kind[path[-1]], "no exit (stuck)")
    print(f"Following the policy from S: {len(path) - 1} moves, ending at {end}.")
    print(f"  {route}")
    print(f"  discounted return of that route : {env.path_return(path, args.gamma):+.4f}")
    print(f"  V(S) from value iteration       : {V[env.start]:+.4f}   (the same number: V(S) IS the return of the best route)\n")

    # --- Plots -----------------------------------------------------------------
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    plot_grid(env, axes[0], values=V, actions=actions, path=path,
              title=f"Values V and greedy policy  (gamma={args.gamma})")
    ax = axes[1]
    ax.semilogy(range(1, len(changes) + 1), np.maximum(changes, 1e-16), color=GREEN, lw=2.5, marker="o", ms=4)
    ax.axhline(args.theta, color=CORAL, ls="--", lw=1)
    ax.text(len(changes) * .98, args.theta * 1.8, "theta", color=CORAL, ha="right", fontsize=9)
    ax.set_xlabel("sweep"), ax.set_ylabel("largest change  (log scale)")
    ax.set_title("The values settle down", fontsize=11, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lab1_value_iteration.png", dpi=150)
    print(f"Saved {OUT / 'lab1_value_iteration.png'}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
