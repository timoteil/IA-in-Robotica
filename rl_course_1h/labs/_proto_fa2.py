"""Fitted Q-iteration: does a sweep-then-fit network recover the gridworld?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gridworld import make_env
from lab1_value_iteration import greedy_policy, one_step_lookahead, value_iteration

G = 0.95
env = make_env()
V, _ = value_iteration(env, G)
Qstar = np.zeros((env.n_states, 4))
for s in range(env.n_states):
    if not env.terminal[s]:
        Qstar[s] = one_step_lookahead(env, V, s, G)
live = np.flatnonzero(~env.terminal)
STAR = greedy_policy(env, V, G)


def coords(states):
    xy = np.array([env.cells[int(s)] for s in np.atleast_1d(states)], dtype=float)
    x = xy[:, 1] / (env.n_cols - 1) * 2 - 1
    y = xy[:, 0] / (env.n_rows - 1) * 2 - 1
    return np.column_stack([x, y])


class MLP:
    def __init__(self, dims, seed):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            scale = 1.0 if d1 == dims[-1] and d0 != dims[0] else np.sqrt(2 / d0)
            # output layer smaller
            if d1 == dims[-1]:
                scale = np.sqrt(1 / d0)
            self.W.append(rng.normal(0, scale, size=(d0, d1)))
            self.b.append(np.zeros(d1))

    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            if i < len(self.W) - 1:
                z = np.maximum(z, 0.0)
            hs.append(z)
        return hs

    def grads(self, hs, delta):
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.append(hs[i].T @ delta)
            gb.append(delta.sum(0))
            if i == 0:
                break
            delta = (delta @ self.W[i].T) * (hs[i] > 0)
        return gW[::-1], gb[::-1]


def nparams(net):
    return int(sum(w.size + b.size for w, b in zip(net.W, net.b)))


FEAT = coords


def predict(net, states):
    return net.forward(FEAT(states))[-1]


def rmse(net):
    return float(np.sqrt(np.mean((predict(net, live) - Qstar[live]) ** 2)))


def agree_and_route(net):
    actions = np.zeros(env.n_states, dtype=int)
    actions[live] = predict(net, live).argmax(1)
    ok = 0
    for s in live:
        a = actions[s]
        if Qstar[s, a] >= Qstar[s].max() - 1e-6:
            ok += 1
    path = env.follow_policy(actions)
    return ok / len(live), env.kind[path[-1]], len(path) - 1, env.path_return(path, G), actions


def collect(episodes=60, seed=0):
    tried = np.zeros(env.n_states, dtype=int)
    last = {}
    reached = {}
    for ep in range(1, episodes + 1):
        s = env.reset()
        for _ in range(50):
            a = int(tried[s] % 4)
            tried[s] += 1
            s2, r, done = env.step(a)
            last[(s, a)] = (s, a, r, s2, float(done))
            s = s2
            if done:
                reached.setdefault(env.kind[s], ep)
                break
    rows = list(last.values())
    return (np.array([t[0] for t in rows]),
            np.array([t[1] for t in rows]),
            np.array([t[2] for t in rows]),
            np.array([t[3] for t in rows]),
            np.array([t[4] for t in rows], dtype=bool),
            reached)


def fit_targets(net, X, take, y, steps, lr, seed):
    """Regress the taken-action outputs onto frozen targets y. Plain SGD + momentum."""
    mom_w = [np.zeros_like(w) for w in net.W]
    mom_b = [np.zeros_like(b) for b in net.b]
    mu = 0.9
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    for step in range(steps):
        rng.shuffle(idx)
        # one pass in minibatches of 18
        for start in range(0, n, 18):
            b = idx[start:start + 18]
            hs = net.forward(X[b])
            pred = hs[-1][np.arange(len(b)), take[b]]
            delta = np.zeros_like(hs[-1])
            delta[np.arange(len(b)), take[b]] = (2.0 / len(b)) * (pred - y[b])
            gW, gb = net.grads(hs, delta)
            for i in range(len(net.W)):
                mom_w[i] = mu * mom_w[i] + gW[i]
                mom_b[i] = mu * mom_b[i] + gb[i]
                net.W[i] -= lr * mom_w[i]
                net.b[i] -= lr * mom_b[i]


def fitted_q(dims, data, sweeps, fit_steps, lr, seed):
    S, A, R, S2, D = data
    net = MLP(dims, seed)
    X = FEAT(S)
    curve = []
    for sweep in range(sweeps):
        nxt = predict(net, S2).max(1)
        y = R + G * nxt * (~D)
        fit_targets(net, X, A, y, fit_steps, lr, seed + sweep)
        curve.append(rmse(net))
    return net, curve


def supervised(dims, steps, lr, seed, holdout=()):
    net = MLP(dims, seed)
    states = [s for s in live if s not in holdout]
    # one row per (s, a)
    S, A, Y = [], [], []
    for s in states:
        for a in range(4):
            S.append(s)
            A.append(a)
            Y.append(Qstar[s, a])
    fit_targets(net, FEAT(S), np.array(A), np.array(Y), steps, lr, seed)
    return net


def main():
    S, A, R, S2, D, reached = collect()
    print(f"unique transitions {len(S)} states {len(set(S.tolist()))} reached {reached}")

    print("\nSUPERVISED coords")
    for dims in ([2, 8, 4], [2, 16, 4], [2, 16, 16, 4]):
        net = supervised(dims, steps=40, lr=0.05, seed=0)
        ag, end, moves, ret, _ = agree_and_route(net)
        print(f"  {dims} p={nparams(net)} rmse={rmse(net):.4f} agree={ag:.2f} {end} {moves} {ret:+.3f}")

    # hold out 4 states
    hold = [env.pos2s[p] for p in [(0, 2), (2, 1), (3, 4), (0, 4)]]
    net = supervised([2, 16, 16, 4], steps=60, lr=0.05, seed=0, holdout=hold)
    pred = predict(net, hold)
    err = np.sqrt(np.mean((pred - Qstar[hold]) ** 2))
    train_states = [s for s in live if s not in hold]
    train_err = float(np.sqrt(np.mean((predict(net, train_states) - Qstar[train_states]) ** 2)))
    print(f"  holdout train rmse {train_err:.4f} held-out rmse {err:.4f}")
    for s, p in zip(hold, pred):
        print(f"    {env.cells[s]} pred {np.round(p, 3)} true {np.round(Qstar[s], 3)}")

    print("\nFITTED Q")
    for seed in range(5):
        for dims, steps, lr, sweeps in (
            ([2, 16, 16, 4], 25, 0.08, 12),
            ([2, 16, 16, 4], 40, 0.05, 15),
            ([2, 32, 4], 40, 0.05, 15),
        ):
            net, curve = fitted_q(dims, (S, A, R, S2, D), sweeps, steps, lr, seed)
            ag, end, moves, ret, _ = agree_and_route(net)
            print(f"  seed {seed} {dims} steps {steps} lr {lr} sw {sweeps} "
                  f"rmse {rmse(net):.3f} agree {ag:.2f} {end} m{moves} {ret:+.3f} "
                  f"curve {[round(c, 3) for c in curve]}")

    print("\nACTOR imitating exact policy and imitating fitted critic")
    # train actor
    def train_actor(labels, seed=0, steps=30, lr=0.1):
        actor = MLP([2, 16, 4], seed)
        X = coords(live)
        y = labels
        mom_w = [np.zeros_like(w) for w in actor.W]
        mom_b = [np.zeros_like(b) for b in actor.b]
        for step in range(steps):
            hs = actor.forward(X)
            logits = hs[-1] - hs[-1].max(1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(1, keepdims=True)
            delta = prob.copy()
            delta[np.arange(len(live)), y] -= 1
            delta /= len(live)
            gW, gb = actor.grads(hs, delta)
            for i in range(len(actor.W)):
                mom_w[i] = 0.9 * mom_w[i] + gW[i]
                mom_b[i] = 0.9 * mom_b[i] + gb[i]
                actor.W[i] -= lr * mom_w[i]
                actor.b[i] -= lr * mom_b[i]
        pred_a = actor.forward(X)[-1].argmax(1)
        actions = np.zeros(env.n_states, dtype=int)
        actions[live] = pred_a
        path = env.follow_policy(actions)
        match = np.mean(pred_a == y)
        # optimal-action match
        ok = np.mean([Qstar[s, a] >= Qstar[s].max() - 1e-6 for s, a in zip(live, pred_a)])
        return match, ok, env.kind[path[-1]], len(path) - 1, env.path_return(path, G), nparams(actor)

    m, ok, end, moves, ret, p = train_actor(STAR[live])
    print(f"  actor|pi* match {m:.2f} optimal {ok:.2f} {end} m{moves} {ret:+.3f} params {p}")


if __name__ == "__main__":
    main()
