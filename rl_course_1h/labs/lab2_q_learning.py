"""
Lab 2 -- Learning without a model: tabular Q-learning
=====================================================
In Lab 1 the agent was handed the system equations (env.model) and computed the answer.
Here the agent is NOT told how the world works. It can only

        s = env.reset()            and            s2, r, done = env.step(a)

so all it ever sees is a stream of data (s, a, r, s'). From that stream it fills in a table
Q[state, action]  (a plain numpy array: no function approximation anywhere).

        TD error   e      =  r + gamma * max_a' Q(s', a')  -  Q(s, a)
        update     Q(s,a) <-  Q(s, a) + alpha * e

 * Q(s,a) is the value of doing a in s and then acting well. The best action is argmax_a Q(s,a):
   no model needed to act.
 * In this deterministic world alpha = 1 is enough: the new estimate simply REPLACES the old one,
   and the update is value iteration done with samples instead of the model.
 * The table only learns what the agent tries. The agent therefore has to PROBE: try every
   action in every state, whatever the table currently says. Compare with a greedy agent that
   always does what looks best.

Value iteration from Lab 1 is imported only to CHECK the answer; the learner never uses it.

Run it
------
    python lab2_q_learning.py                       # probing agent: learns the exact table
    python lab2_q_learning.py --behaviour greedy    # greedy agent: what goes wrong?
    python lab2_q_learning.py --compare             # both, side by side
    python lab2_q_learning.py --alpha 0.5           # a smaller step: still converges, more slowly
    python lab2_q_learning.py --show                # also open the plot window

Exercises
---------
 1. Run the default and read the trace of the first episode. Pick one line and check it by hand:
    does the printed target equal r + gamma * max Q(s')?
 2. In which episode does the agent first reach the goal? Why can the table not improve before that?
 3. Run --behaviour greedy. Which exit does it end up using, and what is the value of its start state
    compared with the best possible? Why does it never find out?
 4. Try --alpha 1.0, 0.5, 0.1 with --episodes 500. Compare the number of exact states and the
    largest error. In this deterministic world, what does a smaller step buy you? What does it cost?
 5. Set --gamma 0.5 and compare with Lab 1 at the same gamma: do the learned table and value
    iteration still agree?
 6. In `q_learning` replace the probing rule by a random action. It works too, but what do you give up?
"""
import argparse
from pathlib import Path

import numpy as np

from gridworld import AMBER, BLUE, GREEN, INK, SLATE, ACTION_NAMES, make_env, plot_grid
from lab1_value_iteration import greedy_policy, value_iteration

OUT = Path(__file__).parent / "outputs"


# ======================================================================= CORE ALGORITHM
def choose_action(Q, tried, s, behaviour):
    """How does the agent pick the next action in state s?

    probing : try the actions in turn, whatever Q says (first time: up, then down, left, right, up, ...).
    greedy  : always do the best action in the table (ties -> the first one).
    """
    if behaviour == "probing":
        a = tried[s] % Q.shape[1]
        tried[s] += 1
        return int(a)
    return int(np.flatnonzero(Q[s] == Q[s].max())[0])


def q_learning(env, episodes=150, gamma=0.95, alpha=1.0, behaviour="probing", max_steps=100, trace=0):
    """Tabular Q-learning. Returns (Q, history, reached).

    history[i] is a copy of the table after episode i+1; reached[kind] is the first episode
    in which the agent stepped onto a G / E cell. `trace=n` prints every transition of the
    first n episodes. Uses only env.reset() and env.step().
    """
    Q = np.zeros((env.n_states, env.n_actions))              # the whole "brain": a table
    tried = np.zeros(env.n_states, dtype=int)
    history, reached = [], {}
    for ep in range(1, episodes + 1):
        s = env.reset()
        for _ in range(max_steps):
            a = choose_action(Q, tried, s, behaviour)        # 1. BEHAVE: probe (or be greedy)
            s2, r, done = env.step(a)                        # 2. ACT: one observed transition (s, a, r, s')
            target = r + gamma * Q[s2].max() * (not done)    # 3. TARGET: reward + discounted best next value
            td_error = target - Q[s, a]                      # 4. TD ERROR: prediction vs. what we observed
            if ep <= trace:
                (r0, c0), (r1, c1) = env.cells[s], env.cells[s2]
                print(f"  ep {ep:>2}  s=({r0},{c0}) a={ACTION_NAMES[a]:<5} -> r={r:+.2f} s'=({r1},{c1})"
                      f"   target={target:+.3f}  error={td_error:+.3f}  Q(s,a): {Q[s, a]:+.3f} -> "
                      f"{Q[s, a] + alpha * td_error:+.3f}")
            Q[s, a] += alpha * td_error                      # 5. LEARN: move Q towards the target
            s = s2
            if done:
                reached.setdefault(env.kind[s], ep)
                break
        history.append(Q.copy())
    return Q, history, reached


# ======================================================================= EVALUATION HELPERS
def errors_vs_value_iteration(env, history, V_star):
    """Largest |max_a Q(s,a) - V*(s)| over the non-terminal states, after every episode."""
    live = ~env.terminal
    return np.array([np.abs(Q.max(axis=1) - V_star)[live].max() for Q in history])


def exact_states(env, history, V_star, tol=1e-6):
    """How many non-terminal states already have max_a Q(s,a) = V*(s), after every episode."""
    live = ~env.terminal
    return np.array([(np.abs(Q.max(axis=1) - V_star)[live] < tol).sum() for Q in history])


def describe_run(env, Q, history, reached, V_star, gamma, behaviour):
    err = errors_vs_value_iteration(env, history, V_star)[-1]
    n_exact, n_live = exact_states(env, history, V_star)[-1], int((~env.terminal).sum())
    path = env.follow_policy(Q.argmax(axis=1))
    end = {"G": "the far GOAL", "E": "the near EXIT"}.get(env.kind[path[-1]], "no exit (stuck)")
    print(f"Episodes in which each exit was first reached: "
          f"goal G = {reached.get('G', 'never')},  near exit E = {reached.get('E', 'never')}")
    print(f"States whose value already equals value iteration's: {n_exact} of {n_live}   "
          f"(largest |max Q - V| = {err:.4f})")
    print(f"Greedy route from S: {len(path) - 1} moves, ending at {end}")
    print(f"  value of the start state under the learned table : {env.path_return(path, gamma):+.4f}")
    print(f"  best possible (value iteration)                  : {V_star[env.start]:+.4f}")
    return err, path


# ======================================================================= EXPERIMENTS
def run_single(args):
    env = make_env()
    V_star, _ = value_iteration(env, args.gamma)
    print(f"Q-learning, behaviour = {args.behaviour}, alpha = {args.alpha}, gamma = {args.gamma}, "
          f"{args.episodes} episodes (at most {args.max_steps} steps each)\n")
    if args.trace:
        print(f"Every transition observed in the first {args.trace} episode(s):")
    Q, history, reached = q_learning(env, args.episodes, args.gamma, args.alpha, args.behaviour,
                                     args.max_steps, args.trace)
    if args.trace:
        print()
    err, path = describe_run(env, Q, history, reached, V_star, args.gamma, args.behaviour)
    print("\nLearned values max_a Q(s,a) and greedy policy:")
    print(env.render(values=Q.max(axis=1), actions=Q.argmax(axis=1)))
    print(f"\nQ-table row of the start state (up, down, left, right): {np.round(Q[env.start], 3)}")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exact = exact_states(env, history, V_star)
    n_live = int((~env.terminal).sum())
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})
    ax = axes[0]
    ax.plot(np.arange(1, len(exact) + 1), exact, color=GREEN, lw=2.5)
    ax.axhline(n_live, color=SLATE, ls=":", lw=1)
    if "G" in reached:
        ax.axvline(reached["G"], color=AMBER, ls="--", lw=1.5)
        ax.text(reached["G"] + 2, 0.5, "goal reached\nfor the first time", color=INK, fontsize=9)
    ax.set_ylim(-0.6, n_live + 1)
    ax.set_xlabel("episode"), ax.set_ylabel(f"states with the exact value (of {n_live})")
    ax.set_title("Agreement with value iteration", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    plot_grid(env, axes[1], values=Q.max(axis=1), actions=Q.argmax(axis=1), path=path,
              title=f"Learned: max Q and greedy policy ({args.behaviour})")
    plot_grid(env, axes[2], values=V_star, actions=greedy_policy(env, V_star, args.gamma),
              title="Exact (value iteration, Lab 1)")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lab2_q_learning.png", dpi=150)
    print(f"\nSaved {OUT / 'lab2_q_learning.png'}")
    if args.show:
        plt.show()


def run_compare(args):
    env = make_env()
    V_star, _ = value_iteration(env, args.gamma)
    results = {}
    for behaviour in ["probing", "greedy"]:
        print(f"--- behaviour = {behaviour} " + "-" * 40)
        Q, history, reached = q_learning(env, args.episodes, args.gamma, args.alpha, behaviour, args.max_steps)
        err, path = describe_run(env, Q, history, reached, V_star, args.gamma, behaviour)
        results[behaviour] = (Q, exact_states(env, history, V_star), path)
        print()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.3]})
    colours = {"probing": GREEN, "greedy": SLATE}
    n_live = int((~env.terminal).sum())
    for behaviour, (_, exact, _) in results.items():
        axes[0].plot(np.arange(1, len(exact) + 1), exact, color=colours[behaviour], lw=2.5, label=behaviour)
    axes[0].axhline(n_live, color=SLATE, ls=":", lw=1)
    axes[0].set_ylim(-0.6, n_live + 1)
    axes[0].set_xlabel("episode"), axes[0].set_ylabel(f"states with the exact value (of {n_live})")
    axes[0].legend(frameon=False, loc="center right"), axes[0].set_title("Agreement with value iteration", color=INK)
    axes[0].spines[["top", "right"]].set_visible(False)
    for ax, behaviour in zip(axes[1:], results):
        Q, _, path = results[behaviour]
        plot_grid(env, ax, values=Q.max(axis=1), actions=Q.argmax(axis=1), path=path,
                  title=f"{behaviour}: where the agent ends up")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lab2_probing_vs_greedy.png", dpi=150)
    print(f"Saved {OUT / 'lab2_probing_vs_greedy.png'}")
    if args.show:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--behaviour", choices=["probing", "greedy"], default="probing")
    ap.add_argument("--compare", action="store_true", help="run probing and greedy side by side")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--max-steps", type=int, default=100, help="steps after which an episode is cut off")
    ap.add_argument("--alpha", type=float, default=1.0, help="step size (1.0 = replace the old estimate)")
    ap.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    ap.add_argument("--trace", type=int, default=1, help="print every transition of the first N episodes")
    ap.add_argument("--show", action="store_true", help="open the plot window as well as saving the PNG")
    args = ap.parse_args()
    (run_compare if args.compare else run_single)(args)


if __name__ == "__main__":
    main()
