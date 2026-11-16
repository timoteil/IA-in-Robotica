"""Find a seed-stable fitted-Q recipe and a closed-form linear baseline."""
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


def xy_of(states):
    xy = np.array([env.cells[int(s)] for s in np.atleast_1d(states)], dtype=float)
    x = xy[:, 1] / (env.n_cols - 1) * 2 - 1
    y = xy[:, 0] / (env.n_rows - 1) * 2 - 1
    return np.column_stack([x, y])


def collect():
    tried = np.zeros(env.n_states, dtype=int)
    last = {}
    for _ep in range(80):
        s = env.reset()
        for _ in range(40):
            a = int(tried[s] % 4)
            tried[s] += 1
            s2, r, done = env.step(a)
            last[(s, a)] = (s, a, r, s2, bool(done))
            s = s2
            if done:
                break
    rows = list(last.values())
    return [np.array([t[i] for t in rows]) for i in range(5)]


S, A, R, S2, D = collect()
D = D.astype(bool)
print("n", len(S))


def score(q_of_live, actions_live):
    rmse = float(np.sqrt(np.mean((q_of_live - Qstar[live]) ** 2)))
    ok = 0
    for s, a in zip(live, actions_live):
        if Qstar[s, a] >= Qstar[s].max() - 1e-8:
            ok += 1
    full = np.zeros(env.n_states, dtype=int)
    full[live] = actions_live
    path = env.follow_policy(full)
    return rmse, ok / len(live), env.kind[path[-1]], len(path) - 1, float(env.path_return(path, G))


# ----- linear least squares, phi = [1, x, y], separate weights per action
PHI = np.column_stack([np.ones(len(live)), xy_of(live)])  # (18, 3)


def linear_predict(W, states):
    P = np.column_stack([np.ones(len(np.atleast_1d(states))), xy_of(states)])
    return P @ W  # (n, 4) if W is (3, 4)


def fit_linear(targets_live):
    # targets_live: (18, 4)
    W, *_ = np.linalg.lstsq(PHI, targets_live, rcond=None)
    return W


print("\nLINEAR supervised")
W = fit_linear(Qstar[live])
q = linear_predict(W, live)
rmse, ag, end, moves, ret = score(q, q.argmax(1))
print(f"  rmse {rmse:.4f} agree {ag:.2f} {end} m{moves} {ret:+.3f}  W shape {W.shape} params {W.size}")

print("LINEAR fitted Q")
W = np.zeros((3, 4))
for sweep in range(30):
    # targets for every live state, every action, from the model-free dataset
    q2 = linear_predict(W, S2).max(1)
    y = R + G * q2 * (~D)
    # scatter y into a (18, 4) target matrix; unseen stay at current prediction
    T = linear_predict(W, live)
    index = {int(s): i for i, s in enumerate(live)}
    for s, a, yi in zip(S, A, y):
        T[index[int(s)], int(a)] = yi
    W = fit_linear(T)
    if sweep in (0, 4, 9, 19, 29):
        q = linear_predict(W, live)
        rmse, ag, end, moves, ret = score(q, q.argmax(1))
        print(f"  sweep {sweep+1:2d} rmse {rmse:.3f} agree {ag:.2f} {end} m{moves} {ret:+.3f}")


class Net:
    def __init__(self, dims, seed, act="relu"):
        rng = np.random.default_rng(seed)
        self.act = act
        self.W, self.b = [], []
        for i, (d0, d1) in enumerate(zip(dims[:-1], dims[1:])):
            scale = np.sqrt(2 / d0) if i < len(dims) - 2 else np.sqrt(1 / d0)
            self.W.append(rng.normal(0, scale, (d0, d1)))
            self.b.append(np.zeros(d1))
        self.mw = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vw = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    def forward(self, x):
        hs = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = hs[-1] @ W + b
            if i < len(self.W) - 1:
                if self.act == "relu":
                    z = np.maximum(z, 0)
                else:
                    z = np.tanh(z)
            hs.append(z)
        return hs

    def predict(self, states):
        return self.forward(xy_of(states))[-1]

    def adam(self, gW, gb, lr):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i in range(len(self.W)):
            self.mw[i] = b1 * self.mw[i] + (1 - b1) * gW[i]
            self.mb[i] = b1 * self.mb[i] + (1 - b1) * gb[i]
            self.vw[i] = b2 * self.vw[i] + (1 - b2) * gW[i] ** 2
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * gb[i] ** 2
            self.W[i] -= lr * (self.mw[i] / (1 - b1 ** self.t)) / (np.sqrt(self.vw[i] / (1 - b2 ** self.t)) + eps)
            self.b[i] -= lr * (self.mb[i] / (1 - b1 ** self.t)) / (np.sqrt(self.vb[i] / (1 - b2 ** self.t)) + eps)

    def grads(self, hs, delta):
        gW, gb = [], []
        for i in reversed(range(len(self.W))):
            gW.append(hs[i].T @ delta)
            gb.append(delta.sum(0))
            if i == 0:
                break
            dh = hs[i] > 0 if self.act == "relu" else (1 - hs[i] ** 2)
            delta = (delta @ self.W[i].T) * dh
        return gW[::-1], gb[::-1]


def regress(net, X, take, y, passes, lr, batch=18):
    n = len(y)
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    for _ in range(passes):
        rng.shuffle(idx)
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            hs = net.forward(X[b])
            pred = hs[-1][np.arange(len(b)), take[b]]
            delta = np.zeros_like(hs[-1])
            delta[np.arange(len(b)), take[b]] = (2 / len(b)) * (pred - y[b])
            net.adam(*net.grads(hs, delta), lr)


def fitted(dims, seed, sweeps, passes, lr, act="relu"):
    net = Net(dims, seed, act)
    X = xy_of(S)
    for _ in range(sweeps):
        y = R + G * net.predict(S2).max(1) * (~D)
        regress(net, X, A, y, passes, lr)
    q = net.predict(live)
    return score(q, q.argmax(1)) + (sum(w.size + b.size for w, b in zip(net.W, net.b)),)


print("\nMLP FITTED")
configs = []
for act in ("relu", "tanh"):
    for dims in ([2, 32, 4], [2, 16, 16, 4]):
        for sweeps, passes, lr in ((20, 20, 0.01), (25, 30, 0.02), (12, 40, 0.02)):
            goods = []
            for seed in range(6):
                rmse, ag, end, moves, ret, npar = fitted(dims, seed, sweeps, passes, lr, act)
                goods.append(end == "G" and ag >= 0.9)
            print(f"  {act:4} {dims} sw{sweeps} pass{passes} lr{lr}  G {sum(goods)}/6")
