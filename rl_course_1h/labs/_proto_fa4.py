"""Confirm the winning fitted-Q recipe, the actor, and a hold-out."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import _proto_fa3 as P

DIMS = [2, 16, 16, 4]
SW, PASSES, LR = 25, 30, 0.02


def actor_imitate(labels, steps=80, lr=0.02, seed=0):
    net = P.Net([2, 16, 4], seed, "relu")
    X = P.xy_of(P.live)
    y = np.asarray(labels)
    for _ in range(steps):
        hs = net.forward(X)
        logits = hs[-1] - hs[-1].max(1, keepdims=True)
        prob = np.exp(logits)
        prob /= prob.sum(1, keepdims=True)
        delta = prob.copy()
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)
        net.adam(*net.grads(hs, delta), lr)
    pred = net.forward(X)[-1].argmax(1)
    match = float(np.mean(pred == y))
    actions = np.zeros(P.env.n_states, dtype=int)
    actions[P.live] = pred
    path = P.env.follow_policy(actions)
    ok = np.mean([P.Qstar[s, a] >= P.Qstar[s].max() - 1e-8 for s, a in zip(P.live, pred)])
    return match, float(ok), P.env.kind[path[-1]], len(path) - 1, float(P.env.path_return(path, P.G)), pred


print("10 seeds of relu 2-16-16-4")
routes = []
for seed in range(10):
    rmse, ag, end, moves, ret, npar = P.fitted(DIMS, seed, SW, PASSES, LR, "relu")
    print(f"  seed {seed} rmse {rmse:.4f} agree {ag:.2f} {end} m{moves} {ret:+.4f}")
    routes.append((seed, rmse, ag, end, moves, ret))

# rebuild seed 0 critic to train the actor (fitted() doesn't return the net)
# duplicate the loop once
net = P.Net(DIMS, 0, "relu")
X = P.xy_of(P.S)
for _ in range(SW):
    y = P.R + P.G * net.predict(P.S2).max(1) * (~P.D)
    P.regress(net, X, P.A, y, PASSES, LR)
labels = net.predict(P.live).argmax(1)
print("seed0 critic", P.score(net.predict(P.live), labels)[:5])
print("actor from critic", actor_imitate(labels))
star = P.greedy_policy(P.env, P.V, P.G)[P.live]
print("actor from pi*", actor_imitate(star))

# hold out four states: drop their outgoing transitions
hold_cells = [(0, 2), (2, 1), (3, 4), (0, 4)]
hold = {P.env.pos2s[c] for c in hold_cells}
keep = np.array([int(s) not in hold for s in P.S])
print("kept", keep.sum(), "of", len(P.S))
# monkeypatch globals used by fitted/regress — easier to inline
net = P.Net(DIMS, 0, "relu")
Sk, Ak, Rk, S2k, Dk = P.S[keep], P.A[keep], P.R[keep], P.S2[keep], P.D[keep]
Xk = P.xy_of(Sk)
for _ in range(SW):
    y = Rk + P.G * net.predict(S2k).max(1) * (~Dk)
    P.regress(net, Xk, Ak, y, PASSES, LR)
q_h = net.predict(list(hold))
print("HOLDOUT predictions vs Q*")
for s, q in zip(hold, q_h):
    a = int(q.argmax())
    good = P.Qstar[s, a] >= P.Qstar[s].max() - 1e-8
    print(f"  {P.env.cells[s]} pred {np.round(q, 3)} true {np.round(P.Qstar[s], 3)} action_ok {good}")
train_states = [s for s in P.live if s not in hold]
print("  train rmse", float(np.sqrt(np.mean((net.predict(train_states) - P.Qstar[train_states]) ** 2))))
print("  hold rmse", float(np.sqrt(np.mean((q_h - P.Qstar[list(hold)]) ** 2))))
print("params", npar, "linear", 12)
