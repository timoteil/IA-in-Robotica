"""
common.py -- the small pieces every lab shares
==============================================

Nothing in this file is reinforcement learning "magic". Each piece is a few lines you can read:

    seed_everything   one call makes a run repeatable
    ReplayBuffer      a ring buffer of transitions (s, a, r, s', terminated) and random minibatches
    mlp               a stack of nn.Linear layers
    soft_update       target <- (1 - tau) * target + tau * source     (tau = 1 is a plain copy)
    evaluate          run the policy with NO exploration noise and report the returns
    CSVLog            one row per episode, so a learning curve is a file you can open
    smooth            moving average for the plots

PyTorch is used for the networks and for autograd (the gradient of a loss with respect to the
weights). Everything that makes it *reinforcement learning* is written in the lab files.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

# One thread: a rerun prints the same decimals, and tiny networks do not gain from more.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

OUT = Path(__file__).parent / "outputs"

# Same palette as the 1-hour course and its slides.
INK, GREEN, AMBER, CORAL, BLUE, SLATE, LIGHT = (
    "#14202B", "#0B6E4F", "#F2A900", "#D1495B", "#2E86AB", "#8D99A6", "#EEF3F1")


# ======================================================================= REPEATABILITY
def seed_everything(seed: int) -> np.random.Generator:
    """Seed python, numpy and torch. Returns a numpy Generator the lab uses for its own draws."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    return np.random.default_rng(seed)


def out_dir() -> Path:
    OUT.mkdir(exist_ok=True)
    return OUT


# ======================================================================= REPLAY BUFFER
class ReplayBuffer:
    """A ring buffer holding the last `capacity` transitions (s, a, r, s', terminated).

    `terminated` is the ONLY flag the learning target may use to cut the bootstrap:
        y = r + gamma * (1 - terminated) * Q(s', .)
    A time-limit cut ("truncated") is not a property of the world, so it is NOT stored here.
    (Lab 3, exercise 4, asks you to put it in and watch what happens.)

    Discrete actions are stored as one float in a column of width 1; the DQN lab casts them
    back to integers. Continuous actions use one column per action dimension.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, s, a, r, s2, terminated) -> None:
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity           # when full, overwrite the oldest
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Uniformly random transitions, as float32 tensors: s, a, r, s2, d (each of shape (B, .))."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        return tuple(torch.as_tensor(x[idx]) for x in (self.s, self.a, self.r, self.s2, self.d))

    def latest(self):
        """The newest transition alone, as a batch of one. Used by the --no-replay ablation."""
        i = (self.ptr - 1) % self.capacity
        return tuple(torch.as_tensor(x[i:i + 1]) for x in (self.s, self.a, self.r, self.s2, self.d))


# ======================================================================= NETWORKS
def mlp(sizes, activation=nn.ReLU, out_activation=None) -> nn.Sequential:
    """sizes = [in, h1, h2, ..., out]. Linear + activation, and the output layer is linear
    unless `out_activation` is given."""
    layers = []
    for i, (n0, n1) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(n0, n1))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif out_activation is not None:
            layers.append(out_activation())
    return nn.Sequential(*layers)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """target <- (1 - tau) * target + tau * source.   tau = 1 copies, tau = 0.005 follows slowly."""
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.mul_(1.0 - tau).add_(tau * ps)


def n_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


# ======================================================================= EVALUATION
def evaluate(env, act_fn, episodes: int = 10, seed: int = 10_000) -> np.ndarray:
    """Return of each of `episodes` runs of act_fn(obs) -> action, with no exploration.

    Evaluation uses its own environment instance and its own seeds, so it never disturbs
    the training stream of data.
    """
    returns = []
    for k in range(episodes):
        obs, _ = env.reset(seed=seed + k)
        total, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, _ = env.step(act_fn(obs))
            total += float(r)
            done = terminated or truncated
        returns.append(total)
    return np.array(returns)


# ======================================================================= LOGGING
class CSVLog:
    """One row per call to `write`. Open the file in a spreadsheet to inspect a run."""

    def __init__(self, path: Path, fields):
        self.path = Path(path)
        self.fields = list(fields)
        self.path.parent.mkdir(exist_ok=True, parents=True)
        self._f = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.fields)
        self._w.writeheader()

    def write(self, **row) -> None:
        self._w.writerow({k: row.get(k, "") for k in self.fields})
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def read_csv(path: Path) -> dict:
    """Read a CSVLog file back as {column: np.array of floats}."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in rows[0].keys():
        vals = [float(r[k]) if r[k] != "" else np.nan for r in rows]
        cols[k] = np.array(vals)
    return cols


# ======================================================================= PLOTTING
def smooth(y, window: int = 20) -> np.ndarray:
    """Trailing moving average (shorter windows at the start)."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.nanmean(y[max(0, i - window + 1): i + 1])
    return out


def style_axes(ax, title=None, xlabel=None, ylabel=None) -> None:
    if title:
        ax.set_title(title, color=INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def get_pyplot(show: bool):
    """matplotlib, with the file-only backend unless the user asked for a window."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
