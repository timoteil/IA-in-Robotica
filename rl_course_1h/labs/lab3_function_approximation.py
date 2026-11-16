"""
Lab 3 -- Function approximation: a neural Q and a neural controller
===================================================================
Labs 1 and 2 stored every number in a table. This lab stores a vector of weights and
computes the number:

        Q(s, a)  ≈  Q_θ(s, a)          the critic, a small neural network
        a        ≈  π_φ(s)             the controller (actor), a second network

The world is the same deterministic grid. The data are the same kind of transitions
(s, a, r, s') the probing agent of Lab 2 would see. Nothing probabilistic is added.

Two approximators are trained on those transitions.

  1. LINEAR.    Q(s, a) = φ(s) · w_a    with    φ(s) = [1, x, y]
                Twelve weights. Each sweep replaces w by the least-squares fit of the
                Bellman targets. A plane cannot bend around the wall.

  2. NEURAL Q.  A network (x, y) → 16 ReLU → 16 ReLU → 4 numbers, one Q per action.
                Each sweep freezes the targets y = r + γ max_a' Q_θ(s', a') and then
                fits θ to those targets. This is fitted Q-iteration: value iteration
                in which the table write has been replaced by a regression.

The controller is a second, smaller network. It does not see rewards. It is trained to
imitate the greedy action of the critic, argmax_a Q_θ(s, a), so the actor is only as
good as the critic it copies. Acting with argmax Q_θ directly is the other controller:
it needs no second network, but it has to score every action.

Run it
------
    python lab3_function_approximation.py            # seed 0: the run the lecture quotes
    python lab3_function_approximation.py --seed 6   # same code, a different initialisation
    python lab3_function_approximation.py --seeds 10 # how many of 10 seeds reach the goal
    python lab3_function_approximation.py --show     # also open the plot window

Exercises
---------
 1. Run the default. How many weights does the critic have, and how many entries would
    the Q-table of Lab 2 have had? Which routes reach G: linear, critic, actor?
 2. Read the printed linear gradient step. Check by hand that the new Q at S equals
    α (target − Q) φ(s) · φ(S). Why did the start cell move, when Lab 2 would have
    changed only the cell that was updated?
 3. Run --seed 6, then --seeds 10. The update rule did not change. What did, and which
    controller inherits the mistake?
 4. In `linear_features`, append x², y² and xy. Predict whether the linear route can
    now reach G, then run it.
 5. The actor never receives a reward. Where does its notion of a good action come from?
    What would go wrong if you trained the actor from a critic that had not converged?
"""
import argparse
import os
from pathlib import Path

# Keep the matmuls single-threaded so a rerun prints the same decimals.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from gridworld import AMBER, GREEN, INK, SLATE, make_env, plot_grid
from lab1_value_iteration import greedy_policy, one_step_lookahead, value_iteration

OUT = Path(__file__).parent / "outputs"

GAMMA = 0.95
CRITIC_DIMS = (2, 16, 16, 4)
ACTOR_DIMS = (2, 16, 4)
CRITIC_SWEEPS = 25
CRITIC_PASSES = 30          # passes over the data inside one sweep, fitting a frozen target
CRITIC_LR = 0.02
ACTOR_STEPS = 200
ACTOR_LR = 0.05
LINEAR_SWEEPS = 30
STEP_ALPHA = 0.1            # the single linear gradient step printed at the start


# ======================================================================= FEATURES
def coordinates(env, states):
    """(x, y) of each state, scaled to [-1, 1]. Row 0 is the top of the map."""
    states = np.atleast_1d(states)
    rc = np.array([env.cells[int(s)] for s in states], dtype=float)
    x = rc[:, 1] / (env.n_cols - 1) * 2.0 - 1.0
    y = rc[:, 0] / (env.n_rows - 1) * 2.0 - 1.0
    return np.column_stack([x, y])


def linear_features(env, states):
    """φ(s) = [1, x, y]. Twelve weights in all: one 3-vector per action."""
    xy = coordinates(env, states)
    return np.column_stack([np.ones(len(xy)), xy])


# ======================================================================= DATA
def collect_transitions(env, episodes=80, max_steps=40):
    """Every distinct (s, a) the probing agent can see, via reset() and step() only.

    Probing tries action k on the k-th visit to a state, exactly as in Lab 2.
    The world is deterministic, so each pair is stored once.
    """
    tried = np.zeros(env.n_states, dtype=int)
    last = {}
    reached = {}
    for ep in range(1, episodes + 1):
        s = env.reset()
        for _ in range(max_steps):
            a = int(tried[s] % env.n_actions)
            tried[s] += 1
            s2, r, done = env.step(a)
            last[(s, a)] = (int(s), int(a), float(r), int(s2), bool(done))
            s = s2
            if done:
                reached.setdefault(env.kind[s], ep)
                break
    rows = list(last.values())
    data = {
        "s": np.array([t[0] for t in rows]),
        "a": np.array([t[1] for t in rows]),
        "r": np.array([t[2] for t in rows]),
        "s2": np.array([t[3] for t in rows]),
        "done": np.array([t[4] for t in rows]),
        "reached": reached,
    }
    return data


def exact_q(env, gamma):
    """Q* from value iteration: the table the approximators are scored against.

    Training never reads this. It is the same check Lab 2 uses.
    """
    V, _ = value_iteration(env, gamma)
    Q = np.zeros((env.n_states, env.n_actions))
    for s in range(env.n_states):
        if not env.terminal[s]:
            Q[s] = one_step_lookahead(env, V, s, gamma)
    actions = greedy_policy(env, V, gamma)
    return V, Q, actions


# ======================================================================= A SMALL MLP
class MLP:
    """A ReLU network in a few dozen lines, so the gradient is visible.

    forward:  h = ReLU(h W + b) on every layer except the last, which stays linear.
    grads:    back-propagation of a batch of output derivatives.
    adam:     the step actually taken. The direction is the gradient of the squared
              error; Adam only rescales it so one learning rate serves every layer.
    """

    def __init__(self, dims, seed):
        rng = np.random.default_rng(seed)
        self.dims = tuple(dims)
        self.W, self.b = [], []
        for i, (d0, d1) in enumerate(zip(dims[:-1], dims[1:])):
            # He scale on the ReLU layers, a smaller scale on the linear output.
            scale = np.sqrt(2.0 / d0) if i < len(dims) - 2 else np.sqrt(1.0 / d0)
            self.W.append(rng.normal(0.0, scale, size=(d0, d1)))
            self.b.append(np.zeros(d1))
        self._mw = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vw = [np.zeros_like(w) for w in self.W]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    @property
    def n_params(self):
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def forward(self, x):
        """Return [input, h1, h2, ..., output]. Hidden layers are ReLU."""
        hs = [np.asarray(x, dtype=float)]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            if i < len(self.W) - 1:
                z = np.maximum(z, 0.0)
            hs.append(z)
        return hs

    def grads(self, hs, delta):
        """delta is ∂L/∂(output), shape (batch, n_out). Returns grads of W and b."""
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.append(hs[i].T @ delta)
            gb.append(delta.sum(axis=0))
            if i == 0:
                break
            delta = (delta @ self.W[i].T) * (hs[i] > 0.0)     # ReLU: closed units pass nothing
        return gW[::-1], gb[::-1]

    def adam(self, gW, gb, lr):
        """One adaptive step along the gradient. Moments are kept across sweeps."""
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self._mw[i] = b1 * self._mw[i] + (1.0 - b1) * gW[i]
            self._mb[i] = b1 * self._mb[i] + (1.0 - b1) * gb[i]
            self._vw[i] = b2 * self._vw[i] + (1.0 - b2) * gW[i] ** 2
            self._vb[i] = b2 * self._vb[i] + (1.0 - b2) * gb[i] ** 2
            mw = self._mw[i] / (1.0 - b1 ** self._t)
            mb = self._mb[i] / (1.0 - b1 ** self._t)
            vw = self._vw[i] / (1.0 - b2 ** self._t)
            vb = self._vb[i] / (1.0 - b2 ** self._t)
            self.W[i] -= lr * mw / (np.sqrt(vw) + eps)
            self.b[i] -= lr * mb / (np.sqrt(vb) + eps)


def q_of(env, net, states):
    """Network outputs for a list of states. Input is (x, y), output is one Q per action."""
    return net.forward(coordinates(env, states))[-1]


# ======================================================================= SCORING
class Fit:
    """What an approximator believes, and how that compares with value iteration."""

    def __init__(self, name, n_params, q, actions, rmse_curve, env, Q_star, gamma, score_values=True):
        self.name = name
        self.n_params = int(n_params)
        self.q = q
        self.actions = actions
        self.rmse_curve = list(rmse_curve)
        live = ~env.terminal
        self.n_live = int(live.sum())
        if score_values:
            self.rmse = float(np.sqrt(np.mean((q[live] - Q_star[live]) ** 2)))
        else:
            self.rmse = None                                   # the actor emits actions, not Q values
        chosen = np.take_along_axis(Q_star[live], actions[live, None], 1)
        optimal = Q_star[live].max(axis=1, keepdims=True)
        self.n_optimal = int(np.sum(chosen >= optimal - 1e-6))
        self.agree = self.n_optimal / self.n_live
        self.path = env.follow_policy(actions)
        self.moves = len(self.path) - 1
        self.end = env.kind[self.path[-1]]
        self.ret = float(env.path_return(self.path, gamma))
        self.reached_goal = self.end == "G"


def _pack_q(env, values_live, actions_live, live):
    q = np.zeros((env.n_states, env.n_actions))
    actions = np.zeros(env.n_states, dtype=int)
    q[live] = values_live
    actions[live] = actions_live
    return q, actions


def _rmse_live(env, q, Q_star, live):
    return float(np.sqrt(np.mean((q[live] - Q_star[live]) ** 2)))


# ======================================================================= LINEAR Q
def fit_linear(env, data, Q_star, gamma=GAMMA, sweeps=LINEAR_SWEEPS):
    """Projected fitted Q. Each sweep solves min_W ||Φ W − targets|| exactly.

    Φ has one row φ(s) = [1, x, y] per non-terminal state, and W is 3 × 4.
    """
    live_idx = np.flatnonzero(~env.terminal)
    index = {int(s): i for i, s in enumerate(live_idx)}
    Phi = linear_features(env, live_idx)                          # (n_live, 3)
    W = np.zeros((Phi.shape[1], env.n_actions))
    curve = []
    for _ in range(sweeps):
        nxt = (linear_features(env, data["s2"]) @ W).max(axis=1)
        y = data["r"] + gamma * nxt * (~data["done"])
        T = Phi @ W                                                # start from the current prediction
        for s, a, yi in zip(data["s"], data["a"], y):
            T[index[int(s)], int(a)] = yi                          # then overwrite with the Bellman target
        W, *_ = np.linalg.lstsq(Phi, T, rcond=None)
        q, _ = _pack_q(env, Phi @ W, (Phi @ W).argmax(1), live_idx)
        curve.append(_rmse_live(env, q, Q_star, live_idx))
    q_live = Phi @ W
    q, actions = _pack_q(env, q_live, q_live.argmax(1), live_idx)
    fit = Fit("linear", W.size, q, actions, curve, env, Q_star, gamma)
    fit.W = W
    return fit


def linear_gradient_step(env, alpha=STEP_ALPHA):
    """One hand-checkable step of the linear model, starting from w = 0.

    The transition is the one Lab 2 prints first: from the cell left of S, step left
    onto the near exit E and receive +0.30. A table would write 0.30 into that one
    entry. The linear model steps w_left, and Q_left changes in every cell.
    """
    s = env.pos2s[(3, 1)]
    a = 2                                                          # left
    s2, r, done = env.model(s, a)                                  # only to name the transition; w starts at 0
    assert done and abs(r - 0.30) < 1e-9
    phi = linear_features(env, [s])[0]
    # w = 0, so Q = 0 and the target is just r.  w <- w - α (Q − target) φ = α target φ
    target = r
    w = alpha * target * phi
    live = np.flatnonzero(~env.terminal)
    Phi = linear_features(env, live)
    q_left = Phi @ w                                               # every cell's new Q for "left"
    moved = int(np.sum(np.abs(q_left) > 1e-3))
    s_start = env.start
    return {
        "s": s, "a": a, "r": r, "s2": s2,
        "phi": phi, "alpha": alpha, "target": target,
        "q_here": float(phi @ w),
        "q_start": float(linear_features(env, [s_start])[0] @ w),
        "moved": moved, "n_live": int(len(live)),
        "w": w,
    }


# ======================================================================= NEURAL Q
def _regress(env, net, states, actions, targets, passes, lr, batch=18):
    """Fit the taken-action outputs to a FROZEN target vector. Several passes of Adam.

    The target must not depend on the weights being updated. That is the semi-gradient:
    we differentiate Q_θ(s, a), and we treat r + γ max Q(s', ·) as a constant.
    """
    X = coordinates(env, states)
    y = np.asarray(targets, dtype=float)
    take = np.asarray(actions)
    n = len(y)
    rng = np.random.default_rng(0)                                 # same shuffle every sweep; only the init is random
    order = np.arange(n)
    for _ in range(passes):
        rng.shuffle(order)
        for start in range(0, n, batch):
            b = order[start:start + batch]
            hs = net.forward(X[b])
            pred = hs[-1][np.arange(len(b)), take[b]]
            delta = np.zeros_like(hs[-1])
            # ∂/∂Q of the mean squared error (pred − y)², and 0 for the actions we did not take
            delta[np.arange(len(b)), take[b]] = (2.0 / len(b)) * (pred - y[b])
            net.adam(*net.grads(hs, delta), lr)


def fit_critic(env, data, Q_star, seed, gamma=GAMMA, sweeps=CRITIC_SWEEPS, passes=CRITIC_PASSES, lr=CRITIC_LR):
    """Fitted Q-iteration. Sweep k freezes the Bellman targets, then regresses onto them.

    Same skeleton as value iteration: read a backup from the previous approximation,
    write a new approximation. The write is a fit, not a table assignment.
    """
    net = MLP(CRITIC_DIMS, seed)
    live = np.flatnonzero(~env.terminal)
    curve = []
    for _ in range(sweeps):
        nxt = q_of(env, net, data["s2"]).max(axis=1)
        y = data["r"] + gamma * nxt * (~data["done"])              # frozen for the whole sweep
        _regress(env, net, data["s"], data["a"], y, passes, lr)
        q_live = q_of(env, net, live)
        q, _ = _pack_q(env, q_live, q_live.argmax(1), live)
        curve.append(_rmse_live(env, q, Q_star, live))
    q_live = q_of(env, net, live)
    q, actions = _pack_q(env, q_live, q_live.argmax(1), live)
    fit = Fit("critic", net.n_params, q, actions, curve, env, Q_star, gamma)
    fit.net = net
    return fit


# ======================================================================= CONTROLLER
def fit_actor(env, critic, Q_star, seed, gamma=GAMMA, steps=ACTOR_STEPS, lr=ACTOR_LR):
    """A second network, π_φ(s), trained to imitate argmax_a Q_θ(s, a).

    Cross-entropy on the 18 non-terminal cells. The actor never sees a reward: the
    critic is the only teacher. Its greedy action is the label.
    """
    live = np.flatnonzero(~env.terminal)
    labels = critic.actions[live]
    net = MLP(ACTOR_DIMS, seed)
    X = coordinates(env, live)
    for _ in range(steps):
        hs = net.forward(X)
        logits = hs[-1] - hs[-1].max(axis=1, keepdims=True)        # stable softmax
        prob = np.exp(logits)
        prob /= prob.sum(axis=1, keepdims=True)
        delta = prob.copy()
        delta[np.arange(len(live)), labels] -= 1.0                 # ∂ cross-entropy / ∂ logits
        delta /= len(live)
        net.adam(*net.grads(hs, delta), lr)
    pred = net.forward(X)[-1].argmax(axis=1)
    q = np.zeros((env.n_states, env.n_actions))                    # unused; the actor emits actions, not values
    actions = np.zeros(env.n_states, dtype=int)
    actions[live] = pred
    fit = Fit("actor", net.n_params, q, actions, [], env, Q_star, gamma, score_values=False)
    fit.match = float(np.mean(pred == labels))                     # agreement with the critic, not with Q*
    fit.n_copied = int(np.sum(pred == labels))
    fit.net = net
    return fit


# ======================================================================= ONE FULL RUN
def run(env, Q_star, data, seed, gamma=GAMMA):
    """Linear Q, neural Q and the actor, for one initialisation of the networks."""
    linear = fit_linear(env, data, Q_star, gamma)
    critic = fit_critic(env, data, Q_star, seed, gamma)
    actor = fit_actor(env, critic, Q_star, seed, gamma)
    return linear, critic, actor


def scan_seeds(env, Q_star, data, n, gamma=GAMMA):
    """How many of n initialisations put the agent on the route to G."""
    rows = []
    for seed in range(n):
        _, critic, actor = run(env, Q_star, data, seed, gamma)
        rows.append({"seed": seed, "critic_goal": critic.reached_goal, "critic_rmse": critic.rmse,
                     "critic_agree": critic.agree, "n_optimal": critic.n_optimal, "n_live": critic.n_live,
                     "actor_goal": actor.reached_goal, "actor_match": actor.match,
                     "actor_ret": actor.ret, "critic_ret": critic.ret, "critic_end": critic.end,
                     "critic_moves": critic.moves})
    return rows


# ======================================================================= DEMO
def _describe(env, fit):
    end = {"G": "the far GOAL", "E": "the near EXIT"}.get(fit.end, "no exit (stuck)")
    rmse = "     —" if fit.rmse is None else f"{fit.rmse:.4f}"
    print(f"{fit.name:8}  {fit.n_params:4d} weights   RMSE vs Q* = {rmse}   "
          f"optimal actions {fit.n_optimal} of {fit.n_live}   {fit.moves} moves, ending at {end}")
    if fit.reached_goal or fit.end == "E":
        print(f"          return of that route {fit.ret:+.4f}")
    else:
        print("          the route never reaches an exit, so it has no return to compare with V(S)")
    if hasattr(fit, "match"):
        print(f"          actions copied from the critic: {fit.n_copied} of {fit.n_live}")
    route = " -> ".join(f"({env.cells[s][0]},{env.cells[s][1]})" for s in fit.path)
    print(f"          {route}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=0, help="initialisation of the two networks")
    ap.add_argument("--seeds", type=int, default=0, help="also train this many seeds and tally who reaches G")
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--show", action="store_true", help="open the plot window as well as saving the PNG")
    args = ap.parse_args()

    env = make_env()
    V, Q_star, act_star = exact_q(env, args.gamma)
    data = collect_transitions(env)
    n_live = int((~env.terminal).sum())
    n_pairs = n_live * env.n_actions
    print(f"GridWorld: {n_live} non-terminal states × {env.n_actions} actions = {n_pairs} table entries")
    print(f"Probing collected {len(data['s'])} distinct transitions "
          f"(goal G first reached in episode {data['reached'].get('G', 'never')}, "
          f"near exit E in episode {data['reached'].get('E', 'never')})\n")

    step = linear_gradient_step(env, STEP_ALPHA)
    (r0, c0), (r1, c1) = env.cells[step["s"]], env.cells[step["s2"]]
    print("One linear gradient step, weights starting at 0, on the transition into the near exit.")
    print(f"  s=({r0},{c0}) a=left -> r={step['r']:+.2f} s'=({r1},{c1})   "
          f"target = {step['target']:+.3f}   Q before = 0")
    print(f"  φ(s) = [{step['phi'][0]:.0f}, {step['phi'][1]:.1f}, {step['phi'][2]:.1f}]    α = {step['alpha']}")
    print(f"  w_left  <-  α · target · φ(s)")
    print(f"  Q_left at the updated cell : {step['q_here']:+.4f}    "
          f"(a table would have written {step['target']:+.3f} and stopped)")
    print(f"  Q_left at the start S      : {step['q_start']:+.4f}    "
          f"(the table cell for S would still be 0)")
    print(f"  cells whose Q_left moved   : {step['moved']} of {step['n_live']}\n")

    print(f"Training seed {args.seed}. The critic is fitted Q-iteration "
          f"({CRITIC_SWEEPS} sweeps). The actor imitates the critic.\n")
    linear, critic, actor = run(env, Q_star, data, args.seed, args.gamma)
    exact_path = env.follow_policy(act_star)
    print(f"{'exact':8}  {n_pairs:4d} entries   RMSE vs Q* = 0         "
          f"this is value iteration (Lab 1)")
    print(f"          return of that route {env.path_return(exact_path, args.gamma):+.4f}    "
          f"V(S) = {V[env.start]:+.4f}\n")
    for fit in (linear, critic, actor):
        _describe(env, fit)
        print()
    s_exit = env.pos2s[(3, 1)]
    print(f"Q(left) on the cell beside E, the transition from the gradient step:")
    print(f"  Q*        {Q_star[s_exit, 2]:+.4f}")
    print(f"  linear    {linear.q[s_exit, 2]:+.4f}")
    print(f"  neural Q  {critic.q[s_exit, 2]:+.4f}\n")

    if args.seeds:
        print(f"Repeating the networks for seeds 0..{args.seeds - 1}:")
        rows = scan_seeds(env, Q_star, data, args.seeds, args.gamma)
        n_c = sum(r["critic_goal"] for r in rows)
        n_a = sum(r["actor_goal"] for r in rows)
        print(f"  critic reaches G in {n_c} of {args.seeds} seeds")
        print(f"  actor  reaches G in {n_a} of {args.seeds} seeds  (it copies the critic)")
        for r in rows:
            if not r["critic_goal"]:
                print(f"  seed {r['seed']}: critic misses G "
                      f"(RMSE {r['critic_rmse']:.3f}, optimal actions {r['n_optimal']} of {r['n_live']}), "
                      f"actor copies {r['actor_match']:.0%}, actor return {r['actor_ret']:+.3f}")
        print()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    ax = axes[0]
    ax.plot(np.arange(1, len(linear.rmse_curve) + 1), linear.rmse_curve, color=SLATE, lw=2.2, label="linear, 12 weights")
    ax.plot(np.arange(1, len(critic.rmse_curve) + 1), critic.rmse_curve, color=GREEN, lw=2.2, label="neural Q")
    ax.set_xlabel("sweep"), ax.set_ylabel("RMSE against Q*  (18 states × 4 actions)")
    ax.set_title("Fitting the Bellman targets", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plot_grid(env, axes[1], values=Q_star.max(1), actions=act_star, path=exact_path,
              title="Exact  Q*   (Lab 1)")
    lin_title = "Linear   stuck" if not linear.reached_goal else f"Linear   return {linear.ret:+.2f}"
    neu_title = f"Neural Q   return {critic.ret:+.2f}" if critic.reached_goal else "Neural Q   misses G"
    plot_grid(env, axes[2], values=linear.q.max(1), actions=linear.actions, path=linear.path, title=lin_title)
    plot_grid(env, axes[3], values=critic.q.max(1), actions=critic.actions, path=critic.path, title=neu_title)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "lab3_function_approximation.png", dpi=150)
    print(f"Saved {OUT / 'lab3_function_approximation.png'}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
