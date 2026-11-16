"""Prototype: can a small MLP learn Q and a controller on the course gridworld?"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gridworld import make_env
from lab1_value_iteration import greedy_policy, one_step_lookahead, value_iteration

G = 0.95
env = make_env()
V, _ = value_iteration(env, G)
Qstar = np.zeros((env.n_states, env.n_actions))
for s in range(env.n_states):
    if not env.terminal[s]:
        Qstar[s] = one_step_lookahead(env, V, s, G)
act_star = greedy_policy(env, V, G)
live = np.flatnonzero(~env.terminal)


def coords(states):
    xy = np.array([env.cells[int(s)] for s in np.atleast_1d(states)], dtype=float)
    x = xy[:, 1] / (env.n_cols - 1) * 2 - 1
    y = xy[:, 0] / (env.n_rows - 1) * 2 - 1
    return np.column_stack([x, y])


def poly(states):
    xy = coords(states)
    x, y = xy[:, 0], xy[:, 1]
    return np.column_stack([np.ones(len(x)), x, y, x * y, x ** 2, y ** 2])


FEATURES = coords


class MLP:
    def __init__(self, dims, seed):
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for d0, d1 in zip(dims[:-1], dims[1:]):
            self.W.append(rng.normal(0, np.sqrt(2 / d0), size=(d0, d1)))
            self.b.append(np.zeros(d1))

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

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
            gb.append(delta.sum(axis=0))
            if i == 0:
                break
            delta = (delta @ self.W[i].T) * (hs[i] > 0)
        return gW[::-1], gb[::-1]


def nparams(net):
    return sum(w.size + b.size for w, b in zip(net.W, net.b))


def predict(net, states):
        return net.forward(FEATURES(states))[-1]


def rmse(net):
    pred = predict(net, live)
    return np.sqrt(np.mean((pred - Qstar[live]) ** 2))


def max_abs(net):
    pred = predict(net, live)
    return np.max(np.abs(pred.max(1) - V[live]))


def policy_agree(actions):
    # an action agrees if its Q* is within 1e-6 of the best
    ok = 0
    for s, a in zip(live, actions):
        if Qstar[s, a] >= Qstar[s].max() - 1e-6:
            ok += 1
    return ok / len(live)


def fit_supervised(dims, seed, steps, lr):
    net = MLP(dims, seed)
    X = FEATURES(live)
    Y = Qstar[live]
    m = np.zeros_like  # adam state
    mw = [np.zeros_like(w) for w in net.W]
    mb = [np.zeros_like(b) for b in net.b]
    vw = [np.zeros_like(w) for w in net.W]
    vb = [np.zeros_like(b) for b in net.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    idx = np.arange(len(live))
    rng = np.random.default_rng(seed)
    for step in range(steps):
        batch = rng.choice(idx, size=min(16, len(idx)), replace=False)
        hs = net.forward(X[batch])
        err = hs[-1] - Y[batch]
        gW, gb = net.grads(hs, (2.0 / len(batch)) * err)
        t += 1
        for i in range(len(net.W)):
            mw[i] = b1 * mw[i] + (1 - b1) * gW[i]
            mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
            vw[i] = b2 * vw[i] + (1 - b2) * gW[i] ** 2
            vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
            mw_h = mw[i] / (1 - b1 ** t)
            mb_h = mb[i] / (1 - b1 ** t)
            vw_h = vw[i] / (1 - b2 ** t)
            vb_h = vb[i] / (1 - b2 ** t)
            net.W[i] -= lr * mw_h / (np.sqrt(vw_h) + eps)
            net.b[i] -= lr * mb_h / (np.sqrt(vb_h) + eps)
    return net


def collect(episodes, seed):
    rng = np.random.default_rng(seed)
    tried = np.zeros(env.n_states, dtype=int)
    data = []
    reached = {}
    for ep in range(1, episodes + 1):
        s = env.reset()
        for _ in range(40):
            a = int(tried[s] % env.n_actions)  # probing
            tried[s] += 1
            s2, r, done = env.step(a)
            data.append((s, a, r, s2, done))
            s = s2
            if done:
                reached.setdefault(env.kind[s], ep)
                break
    return data, reached


def train_td(dims, data, seed, epochs, lr, batch, target_every):
    online = MLP(dims, seed)
    target = MLP(dims, seed + 1)
    target.copy_from(online)
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    R = np.array([d[2] for d in data], dtype=float)
    S2 = np.array([d[3] for d in data])
    D = np.array([d[4] for d in data])
    rng = np.random.default_rng(seed)
    # adam
    mw = [np.zeros_like(w) for w in online.W]
    mb = [np.zeros_like(b) for b in online.b]
    vw = [np.zeros_like(w) for w in online.W]
    vb = [np.zeros_like(b) for b in online.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    n = len(data)
    history = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            b = order[start:start + batch]
            q = predict(online, S[b])
            with_target = predict(target, S2[b]).max(1)
            y = R[b] + G * with_target * (~D[b])
            pred = q[np.arange(len(b)), A[b]]
            delta = np.zeros_like(q)
            delta[np.arange(len(b)), A[b]] = (2.0 / len(b)) * (pred - y)
            hs = online.forward(FEATURES(S[b]))
            gW, gb = online.grads(hs, delta)
            t += 1
            for i in range(len(online.W)):
                mw[i] = b1 * mw[i] + (1 - b1) * gW[i]
                mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
                vw[i] = b2 * vw[i] + (1 - b2) * (gW[i] ** 2)
                vb[i] = b2 * vb[i] + (1 - b2) * (gb[i] ** 2)
                online.W[i] -= lr * (mw[i] / (1 - b1 ** t)) / (np.sqrt(vw[i] / (1 - b2 ** t)) + eps)
                online.b[i] -= lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
            if t % target_every == 0:
                target.copy_from(online)
        history.append(rmse(online))
    return online, history


def actor_from_q(qnet, seed, steps, lr):
    actor = MLP([FEATURES(live[:1]).shape[1], 16, env.n_actions], seed)
    # labels: greedy action of the critic on live states
    labels = predict(qnet, live).argmax(1)
    X = FEATURES(live)
    rng = np.random.default_rng(seed)
    mw = [np.zeros_like(w) for w in actor.W]
    mb = [np.zeros_like(b) for b in actor.b]
    vw = [np.zeros_like(w) for w in actor.W]
    vb = [np.zeros_like(b) for b in actor.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    for step in range(steps):
        hs = actor.forward(X)
        logits = hs[-1]
        logits = logits - logits.max(1, keepdims=True)
        exp = np.exp(logits)
        prob = exp / exp.sum(1, keepdims=True)
        delta = prob.copy()
        delta[np.arange(len(live)), labels] -= 1.0
        delta /= len(live)
        gW, gb = actor.grads(hs, delta)
        t += 1
        for i in range(len(actor.W)):
            mw[i] = b1 * mw[i] + (1 - b1) * gW[i]
            mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
            vw[i] = b2 * vw[i] + (1 - b2) * gW[i] ** 2
            vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
            actor.W[i] -= lr * (mw[i] / (1 - b1 ** t)) / (np.sqrt(vw[i] / (1 - b2 ** t)) + eps)
            actor.b[i] -= lr * (mb[i] / (1 - b1 ** t)) / (np.sqrt(vb[i] / (1 - b2 ** t)) + eps)
    actions = actor.forward(X)[-1].argmax(1)
    full = np.zeros(env.n_states, dtype=int)
    full[live] = actions
    path = env.follow_policy(full)
    end = env.kind[path[-1]]
    return policy_agree(actions), end, env.path_return(path, G), len(path) - 1


def train_td_sgd(dims, data, seed, epochs, lr, batch, target_every):
    online = MLP(dims, seed)
    target = MLP(dims, seed + 1)
    target.copy_from(online)
    S = np.array([d[0] for d in data])
    A = np.array([d[1] for d in data])
    R = np.array([d[2] for d in data], dtype=float)
    S2 = np.array([d[3] for d in data])
    D = np.array([d[4] for d in data])
    rng = np.random.default_rng(seed)
    n = len(data)
    t = 0
    history = []
    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            b = order[start:start + batch]
            q = predict(online, S[b])
            y = R[b] + G * predict(target, S2[b]).max(1) * (~D[b])
            pred = q[np.arange(len(b)), A[b]]
            delta = np.zeros_like(q)
            delta[np.arange(len(b)), A[b]] = (2.0 / len(b)) * (pred - y)
            hs = online.forward(FEATURES(S[b]))
            gW, gb = online.grads(hs, delta)
            for i in range(len(online.W)):
                online.W[i] -= lr * gW[i]
                online.b[i] -= lr * gb[i]
            t += 1
            if t % target_every == 0:
                target.copy_from(online)
        history.append((rmse(online), max_abs(online)))
    return online, history


def report(tag, net):
    actions = predict(net, live).argmax(1)
    full = np.zeros(env.n_states, dtype=int)
    full[live] = actions
    path = env.follow_policy(full)
    print(f"{tag:22} params {nparams(net):4d}  rmse {rmse(net):.4f}  max|V| {max_abs(net):.4f}  "
          f"agree {policy_agree(actions):.2f}  end {env.kind[path[-1]]}  "
          f"moves {len(path)-1}  ret {env.path_return(path, G):+.4f}")
    return net


def unique_data(data):
    last = {}
    for row in data:
        last[(row[0], row[1])] = row
    return list(last.values())


def main():
    global FEATURES
    FEATURES = coords
    data, reached = collect(80, 0)
    uniq = unique_data(data)
    covered = {s for s, a, r, s2, d in uniq}
    print(f"transitions {len(data)} unique {len(uniq)} states {len(covered)}/{len(live)} reached {reached}")
    missing = [env.cells[s] for s in live if s not in covered]
    print("missing states", missing)

    print("--- linear [1,x,y] supervised ---")
    def lin(states):
        xy = coords(states)
        return np.column_stack([np.ones(len(xy)), xy])
    FEATURES = lin
    net = fit_supervised([3, 4], 0, 4000, 0.05)
    report("linear xy", net)
    actions = predict(net, live).argmax(1)
    bad = [env.cells[s] for s, a in zip(live, actions) if Qstar[s, a] < Qstar[s].max() - 1e-6]
    print("  bad cells", bad)

    FEATURES = poly
    net = fit_supervised([6, 4], 0, 4000, 0.05)
    report("poly", net)
    actions = predict(net, live).argmax(1)
    bad = [env.cells[s] for s, a in zip(live, actions) if Qstar[s, a] < Qstar[s].max() - 1e-6]
    print("  bad cells", bad)

    FEATURES = coords
    print("--- seeds TD full-batch unique, adam ---")
    for seed in range(6):
        net, hist = train_td([2, 16, 16, 4], uniq, seed, epochs=80, lr=0.01, batch=len(uniq), target_every=20)
        report(f"seed {seed}", net)
    print("--- actor on last critic, and on exact policy ---")
    ag, end, ret, moves = actor_from_q(net, 0, 600, 0.05)
    print("actor from critic", round(ag, 3), end, moves, round(ret, 4))

    # actor imitating exact greedy policy: temporarily swap predict labels by fitting a net
    # reuse actor_from_q by a fake q net whose predict matches Qstar — just call the body
    class Exact:
        pass
    # monkeypatch predict via a wrapper net: easiest to call actor training inline
    actor = MLP([2, 16, 4], 0)
    labels = act_star[live]
    X = coords(live)
    # few full-batch steps with adam copied poorly; use the function by making predict return Qstar
    real_predict = predict

    def predict_star(net, states):
        out = np.zeros((len(np.atleast_1d(states)), 4))
        for i, s in enumerate(np.atleast_1d(states)):
            out[i] = Qstar[int(s)]
        return out

    globals()["predict"] = predict_star
    ag, end, ret, moves = actor_from_q(Exact(), 0, 600, 0.05)
    globals()["predict"] = real_predict
    print("actor from exact", round(ag, 3), end, moves, round(ret, 4))


if __name__ == "__main__":
    main()
