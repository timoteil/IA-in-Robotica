"""
Lab 3 -- DDPG: Q-learning when the action is a real number
==========================================================
InvertedPendulum-v5 (MuJoCo): a pole is hinged on a cart. The state is 4 real numbers
(cart position, pole angle, cart velocity, pole angular velocity) and the action is ONE real number, the
force on the cart, anywhere in [-3, 3]. The reward is +1 for every step the pole stays up. The episode ends
if the pole tips over, and is cut at 1000 steps, so the best possible return is 1000.

DQN needed   max_a Q(s', a)   for the target and   argmax_a Q(s, a)   to act. With three actions that is a
loop over three numbers. With a real-valued action it is an optimisation problem to be solved at every
step. DDPG (Deep Deterministic Policy Gradient) removes it by learning a second network that answers it:

        critic   Q_φ(s, a)           how good is it to push with force a in state s, then follow π?
        actor    π_θ(s)  ->  a       the force to apply:  the actor's guess for argmax_a Q_φ(s, a)

        critic target   y = r + gamma * (1 - terminated) * Q_φ⁻(s', π_θ⁻(s'))     (π⁻ replaces the max over a')
        critic loss     mean ( y - Q_φ(s, a) )²                                    (same as DQN)
        actor loss     -mean Q_φ(s, π_θ(s))                                        (move the action uphill on Q)

The actor loss is the new idea. Q_φ(s, a) is a network, so we can ask for ∂Q/∂a, and the chain rule carries
it back to the actor weights:   ∇_θ  Q_φ(s, π_θ(s))  =  ∂Q/∂a · ∂π/∂θ.   In Lab 3 of the 1-hour course the
actor copied the critic's greedy action. Here it climbs the critic's gradient instead.

DDPG is OFF-POLICY and MODEL-FREE: it learns from a replay buffer of old transitions (collected by older
versions of the policy, plus exploration noise), and it never uses the equations of the pendulum.

Pieces, all in this file:
  * target networks for BOTH actor and critic, moved by Polyak averaging  θ⁻ <- (1 - τ) θ⁻ + τ θ
  * exploration noise added to the actor's action (Gaussian here, Ornstein-Uhlenbeck with --noise ou),
    needed because the actor itself is deterministic
  * a warm-up: the first `--warmup` steps use uniformly random actions and no learning
  * `terminated` cuts the bootstrap; `truncated` (the 1000-step cut) does not

Three one-line repairs known as TD3 can be switched on one at a time:
  --twin    two critics, and the target uses the smaller of the two (overestimation is DDPG's usual disease)
  --delay   update the actor and the targets only every 2nd critic update
  --smooth  add small clipped noise to the target action, so the target does not peak on a single sharp action
  --td3     all three

Run it
------
    python lab3_ddpg_inverted_pendulum.py                  # about 1-2 minutes
    python lab3_ddpg_inverted_pendulum.py --trace          # print one update, with the gradient dQ/da, to check by hand
    python lab3_ddpg_inverted_pendulum.py --td3            # the three repairs
    python lab3_ddpg_inverted_pendulum.py --noise ou       # correlated exploration noise
    python lab3_ddpg_inverted_pendulum.py --seeds 5        # how many of 5 seeds reach return 1000
    python lab3_ddpg_inverted_pendulum.py --show           # also open the plot window
    python play.py --lab 3                                 # watch the trained controller

Exercises
---------
 1. Run --trace. For one row, check y = r + gamma * Q_target(s', actor_target(s')). Then read the printed dQ/da:
    which way will the actor move its action for that state, and is it the way you would push the cart?
 2. In the Q-slice figure, find the action at which each curve peaks. Is the black dot (the actor's action) on
    the peak? Why is it not exactly there, even for a trained actor?
 3. Look at the "mean Q" panel. With reward +1 per step and gamma = 0.99, what is the largest value Q can have
    in truth (the pole stays up for ever)? Where does the curve sit relative to that line, and why is it still
    climbing at the end of the run? Then run --steps 100000 and watch whether it overshoots; an estimate above the
    true maximum is overestimation, the thing --twin is there to cure.
 4. In `train`, store `terminated or truncated` in the replay buffer instead of `terminated`. Run again. At the
    1000-step cut the pole is still up. What does the critic now believe about the state at step 1000?
 5. Set --tau 1.0 (the target is a plain copy of the network, refreshed every step) and compare with the default
    0.005. Then try --tau 0.0005.
 6. Run --sigma 0 --warmup 0, then --sigma 0.1 --warmup 1000. The actor is deterministic. What data does the
    critic see when there is no noise, and what can it say about actions the actor would never take?
 7. Run --seeds 5, then --td3 --seeds 5. All seeds use the same code and hyper-parameters. Do they all learn?
    What does that say about reading a single learning curve? Which of --twin, --delay, --smooth matters most?
    (Run each alone with --seeds 5 and count.)
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

ENV_NAME = "InvertedPendulum-v5"
HIDDEN = (256, 256)


# ======================================================================= THE NETWORKS
class Actor(nn.Module):
    """π_θ(s): state in, one force out. The tanh keeps the output in (-1, 1); the scale stretches it to the
    action bounds [-scale, scale]."""

    def __init__(self, obs_dim, act_dim, scale):
        super().__init__()
        self.net = mlp([obs_dim, *HIDDEN, act_dim], out_activation=nn.Tanh)
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, s):
        return self.scale * self.net(s)


class Critic(nn.Module):
    """Q_φ(s, a): state and action in, one number out."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *HIDDEN, 1])

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))


# ======================================================================= EXPLORATION NOISE
class GaussianNoise:
    """Independent N(0, (sigma * scale)²) at every step."""

    def __init__(self, act_dim, scale, sigma, rng):
        self.act_dim, self.scale, self.sigma, self.rng = act_dim, np.asarray(scale), sigma, rng

    def reset(self):
        pass

    def __call__(self):
        return self.sigma * self.scale * self.rng.standard_normal(self.act_dim)


class OUNoise:
    """Ornstein-Uhlenbeck noise: x <- x + theta * (0 - x) + sigma * N(0, 1).

    Successive values are correlated (a small theta keeps x near where it was), so the push has momentum:
    it can hold in one direction for many steps, which a car on a hill needs and a coin flip never gives.
    Reset at the start of each episode.
    """

    def __init__(self, act_dim, scale, sigma, rng, theta=0.15):
        self.act_dim, self.scale, self.sigma, self.theta, self.rng = act_dim, np.asarray(scale), sigma, theta, rng
        self.x = np.zeros(act_dim)

    def reset(self):
        self.x = np.zeros(self.act_dim)

    def __call__(self):
        self.x = self.x + self.theta * (0.0 - self.x) + self.sigma * self.rng.standard_normal(self.act_dim)
        return self.scale * self.x


# ======================================================================= THE AGENT
class DDPG:
    """Actor, critic, their target copies, and the two update rules. `twin`, `delay`, `smooth` add TD3."""

    def __init__(self, obs_dim, act_dim, scale, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99, tau=0.005,
                 twin=False, delay=1, smooth=False, target_noise=0.2, noise_clip=0.5):
        self.gamma, self.tau = gamma, tau
        self.twin, self.delay, self.smooth = twin, delay, smooth
        self.target_noise, self.noise_clip = target_noise, noise_clip
        self.scale = torch.as_tensor(scale, dtype=torch.float32)

        self.actor = Actor(obs_dim, act_dim, scale)
        self.actor_target = copy.deepcopy(self.actor)
        self.critics = [Critic(obs_dim, act_dim) for _ in range(2 if twin else 1)]
        self.critic_targets = [copy.deepcopy(c) for c in self.critics]
        for net in [self.actor_target, *self.critic_targets]:
            for p in net.parameters():
                p.requires_grad_(False)                       # targets are never trained, only averaged into
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        critic_params = [p for c in self.critics for p in c.parameters()]
        self.critic_opt = torch.optim.Adam(critic_params, lr=lr_critic)
        self.updates = 0

    @property
    def critic(self):
        return self.critics[0]

    @torch.no_grad()
    def act(self, s):
        """The actor's action for one state (a numpy vector), without noise."""
        return self.actor(torch.as_tensor(s, dtype=torch.float32).unsqueeze(0))[0].numpy()

    @torch.no_grad()
    def targets(self, r, s2, d):
        """y = r + gamma * (1 - terminated) * Q⁻(s', π⁻(s')).   TD3: min of two critics, smoothed action."""
        a2 = self.actor_target(s2)
        if self.smooth:                                       # small, clipped noise on the target action
            eps = (self.target_noise * self.scale * torch.randn_like(a2))
            eps = torch.max(torch.min(eps, self.noise_clip * self.scale), -self.noise_clip * self.scale)
            a2 = torch.max(torch.min(a2 + eps, self.scale), -self.scale)
        q2 = torch.cat([c(s2, a2) for c in self.critic_targets], dim=1).min(dim=1, keepdim=True).values
        return r + self.gamma * (1.0 - d) * q2

    def update(self, batch, trace=False):
        """One critic step, and (every `delay` critic steps) one actor step and one target step."""
        s, a, r, s2, d = batch
        self.updates += 1

        # ---- 1. critic: regress Q_φ(s, a) onto the frozen target y
        y = self.targets(r, s2, d)
        q_values = [c(s, a) for c in self.critics]
        critic_loss = sum(((y - q) ** 2).mean() for q in q_values)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {"critic_loss": float(critic_loss.item() / len(q_values)), "mean_q": float(q_values[0].mean().item()),
                "actor_loss": np.nan}

        if trace:
            self._print_trace(s, a, r, s2, d, y, q_values[0])

        # ---- 2. actor: move π_θ(s) uphill on Q_φ(s, ·). The loss is minus Q, so gradient DESCENT = ascent on Q.
        #         The gradient also reaches the critic's weights, but only the actor's optimiser steps here.
        if self.updates % self.delay == 0:
            actor_loss = -self.critic(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            info["actor_loss"] = float(actor_loss.item())
            # ---- 3. targets follow the networks slowly
            soft_update(self.actor_target, self.actor, self.tau)
            for ct, c in zip(self.critic_targets, self.critics):
                soft_update(ct, c, self.tau)
        return info

    def _print_trace(self, s, a, r, s2, d, y, q_sa):
        n = min(5, len(s))
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q2 = self.critic_targets[0](s2, a2)
        # dQ/da at the action the actor would choose now: the direction the actor update will push
        s_g = s[:n].clone()
        a_g = self.actor(s_g).detach().requires_grad_(True)
        self.critic(s_g, a_g).sum().backward()
        dqda = a_g.grad
        print(f"One DDPG update, first {n} rows of a minibatch of {len(s)}.   gamma = {self.gamma}, tau = {self.tau}")
        print("   row  angle  ang.vel |  a (buffer)  r   d |  a' = π⁻(s')  Q⁻(s',a')   y = r + γ(1-d)Q⁻ |  Q(s,a)  TD err |  π(s)   dQ/da at π(s)")
        for i in range(n):
            print(f"   {i:3d}  {s[i, 1]:+.3f}  {s[i, 3]:+.3f} |  {a[i, 0]:+9.3f}  {r[i, 0]:+.0f}  {d[i, 0]:.0f} |  "
                  f"{a2[i, 0]:+10.3f}  {q2[i, 0]:+9.3f}  {y[i, 0]:+16.3f} |  {q_sa[i, 0].item():+6.3f}  {(y[i, 0] - q_sa[i, 0]).item():+6.3f} |  "
                  f"{a_g[i, 0].item():+5.2f}  {dqda[i, 0].item():+12.4f}")
        self.critic.zero_grad()                               # discard the gradient we only wanted to read
        print("  A positive dQ/da means Q grows with the force there, so the actor update will push π(s) up; negative, down.\n")

    def policy(self):
        """obs -> action, no noise. play.py and evaluation use it."""
        return lambda obs: self.act(obs)


# ======================================================================= TRAINING LOOP
def train(env_name=ENV_NAME, seed=0, steps=30_000, warmup=1_000, batch=256, buffer=100_000, noise="gaussian",
          sigma=0.1, theta=0.15, eval_every=2_000, trace=False, tag="ddpg", verbose=True, stop_at=None, **agent_kw):
    """DDPG on any Box-action Gymnasium env. Returns the agent, the curves and every state the agent visited.

    stop_at: end the run early once the greedy evaluation reaches this return (used by --seeds).
    """
    rng = seed_everything(seed)
    env, eval_env = gym.make(env_name), gym.make(env_name)
    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    low, high = env.action_space.low, env.action_space.high
    scale = high                                                # actions are symmetric: [-scale, scale]
    agent = DDPG(obs_dim, act_dim, scale, **agent_kw)
    buf = ReplayBuffer(obs_dim, act_dim, buffer, rng)
    noise_fn = (GaussianNoise(act_dim, scale, sigma, rng) if noise == "gaussian"
                else OUNoise(act_dim, scale, sigma, rng, theta))
    log = CSVLog(out_dir() / f"{tag}_{env_name}_seed{seed}.csv",
                 ["step", "episode", "return", "noise_sigma", "critic_loss", "mean_q", "actor_loss", "eval_return"])

    obs, _ = env.reset(seed=seed)
    ep, ep_ret = 1, 0.0
    crit_l, mean_q, act_l = [], [], []
    episode_returns, eval_points, qcurve = [], [], []
    visited = []
    t0 = time.time()
    for step in range(1, steps + 1):
        # ---- act. Warm-up: uniformly random. Afterwards: the actor, plus exploration noise, clipped to the bounds.
        if step <= warmup:
            a = rng.uniform(low, high)
        else:
            a = np.clip(agent.act(obs) + noise_fn(), low, high)
        obs2, r, terminated, truncated, _ = env.step(a)
        buf.add(obs, a, r, obs2, terminated)                    # terminated only: a time-out is not a real ending
        visited.append(obs2.copy())
        obs, ep_ret = obs2, ep_ret + r

        # ---- learn (one gradient step per environment step, once the warm-up is over)
        if step > warmup:
            info = agent.update(buf.sample(batch), trace=trace and not agent.updates)
            crit_l.append(info["critic_loss"]), mean_q.append(info["mean_q"])
            if not np.isnan(info["actor_loss"]):
                act_l.append(info["actor_loss"])

        # ---- end of an episode
        if terminated or truncated:
            episode_returns.append(ep_ret)
            log.write(step=step, episode=ep, **{"return": ep_ret}, noise_sigma=sigma,
                      critic_loss=np.mean(crit_l) if crit_l else "", mean_q=np.mean(mean_q) if mean_q else "",
                      actor_loss=np.mean(act_l) if act_l else "")
            qcurve.append((step, np.mean(mean_q) if mean_q else np.nan))
            ep, ep_ret, crit_l, mean_q, act_l = ep + 1, 0.0, [], [], []
            obs, _ = env.reset()
            noise_fn.reset()

        if step % eval_every == 0:
            ev = evaluate(eval_env, agent.policy(), episodes=5)
            eval_points.append((step, float(ev.mean())))
            log.write(step=step, eval_return=float(ev.mean()))
            if verbose:
                recent = np.mean(episode_returns[-10:]) if episode_returns else float("nan")
                print(f"  step {step:7d}  train return (last 10) {recent:8.1f}   eval return {ev.mean():8.1f}   "
                      f"({time.time() - t0:.0f}s)")
            if stop_at is not None and ev.mean() >= stop_at:
                break
    log.close()
    return {"agent": agent, "returns": np.array(episode_returns), "eval": np.array(eval_points),
            "q": np.array(qcurve), "visited": np.array(visited), "seed": seed, "csv": log.path,
            "env_name": env_name, "steps": step, "scale": scale, "seconds": time.time() - t0}


def save_checkpoint(result, path, extra=None):
    agent = result["agent"]
    torch.save({"actor": agent.actor.state_dict(), "critic": agent.critic.state_dict(), "env": result["env_name"],
                "scale": np.asarray(result["scale"]), "hidden": HIDDEN, "obs_dim": agent.actor.net[0].in_features,
                "act_dim": agent.actor.net[-2].out_features, **(extra or {})}, path)


def load_actor(path):
    """Rebuild a trained actor from a checkpoint written by save_checkpoint. Returns (policy_fn, env_name)."""
    global HIDDEN
    ck = torch.load(path, weights_only=False)
    HIDDEN = tuple(ck["hidden"])
    actor = Actor(ck["obs_dim"], ck["act_dim"], ck["scale"])
    actor.load_state_dict(ck["actor"])
    actor.eval()

    @torch.no_grad()
    def act(obs):
        return actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
    return act, ck["env"]


# ======================================================================= PLOTTING
def q_slice(agent, angles, grid):
    """Q_φ(s, a) over a grid of forces a, for states with the cart at rest in the middle and the given angles."""
    out = {}
    with torch.no_grad():
        for ang in angles:
            s = torch.tensor([[0.0, ang, 0.0, 0.0]], dtype=torch.float32).repeat(len(grid), 1)
            out[ang] = agent.critic(s, torch.as_tensor(grid, dtype=torch.float32).unsqueeze(1))[:, 0].numpy()
    return out


def actor_map(agent, n=80):
    """π(s) over a grid of pole angle and angular velocity (cart centred and at rest)."""
    angs, vels = np.linspace(-0.2, 0.2, n), np.linspace(-1.5, 1.5, n)
    s = np.array([[0.0, th, 0.0, w] for w in vels for th in angs], dtype=np.float32)
    with torch.no_grad():
        a = agent.actor(torch.as_tensor(s))[:, 0].numpy().reshape(n, n)
    return angs, vels, a


def make_figure(result, args, name="lab3_ddpg_inverted_pendulum.png"):
    plt = get_pyplot(args.show)
    agent = result["agent"]
    scale = float(np.asarray(result["scale"]).max())
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})

    ax = axes[0]
    ret = result["returns"]
    ax.plot(ret, color=SLATE, lw=0.6, alpha=0.5)
    ax.plot(smooth(ret, 10), color=GREEN, lw=2, label="training return, mean of last 10")
    ax.set_xlabel("episode")
    ax2 = ax.twiny()
    ax2.plot(result["eval"][:, 0], result["eval"][:, 1], color=AMBER, lw=2, marker="o", ms=3, label="greedy evaluation")
    ax2.set_xlabel("environment step", color=AMBER)
    ax2.spines[["right"]].set_visible(False)
    style_axes(ax, "Learning curve", None, "return  (1000 = never falls)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")

    ax = axes[1]
    q = result["q"]
    ax.plot(q[:, 0], q[:, 1], color=CORAL, lw=1.8, label="mean Q(s, a) over the minibatches")
    ax.axhline(1.0 / (1.0 - agent.gamma), color=INK, lw=1, ls="--")
    ax.text(q[0, 0], 1.0 / (1.0 - agent.gamma), " largest true value: 1 / (1 - γ)", va="bottom", fontsize=8, color=INK)
    style_axes(ax, "What the critic believes", "environment step", "Q")
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    ax = axes[2]
    grid = np.linspace(-scale, scale, 121)
    angles = [-0.1, 0.0, 0.1]
    sl = q_slice(agent, angles, grid)
    for ang, col in zip(angles, [BLUE, SLATE, AMBER]):
        ax.plot(grid, sl[ang], color=col, lw=2, label=f"pole angle {ang:+.1f}")
        with torch.no_grad():
            a_star = agent.act(np.array([0.0, ang, 0.0, 0.0], dtype=np.float32))[0]
        ax.plot([a_star], [np.interp(a_star, grid, sl[ang])], "o", color=INK, ms=7)
    style_axes(ax, "Q(s, a) against the force a  (black dot: π(s))", "force a", "Q")
    ax.legend(frameon=False, fontsize=8, loc="best")

    angs, vels, a_map = actor_map(agent)
    im = axes[3].imshow(a_map, origin="lower", extent=[angs[0], angs[-1], vels[0], vels[-1]], aspect="auto",
                        cmap="coolwarm", vmin=-scale, vmax=scale)
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    style_axes(axes[3], "the actor π(s): force on the cart", "pole angle (rad)", "pole angular velocity (rad/s)")
    fig.suptitle(f"{'TD3' if (agent.twin and agent.smooth and agent.delay == 2) else 'DDPG'} on {result['env_name']}, seed {result['seed']}: "
                 f"actor {n_params(agent.actor):,} weights, critic {n_params(agent.critic):,} weights", color=INK)
    fig.tight_layout()
    path = out_dir() / name
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")
    if args.show:
        plt.show()


# ======================================================================= MAIN
def agent_options(args):
    if args.td3:
        args.twin = args.delay_flag = args.smooth = True
    return dict(tau=args.tau, lr_actor=args.lr_actor, lr_critic=args.lr_critic, gamma=args.gamma, twin=args.twin,
                delay=2 if args.delay_flag else 1, smooth=args.smooth)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--warmup", type=int, default=1_000, help="initial steps with random actions and no learning")
    ap.add_argument("--noise", choices=["gaussian", "ou"], default="gaussian")
    ap.add_argument("--sigma", type=float, default=0.1, help="noise size as a fraction of the action bound")
    ap.add_argument("--tau", type=float, default=0.005, help="Polyak averaging rate of the target networks")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr-actor", type=float, default=1e-3)
    ap.add_argument("--lr-critic", type=float, default=1e-3)
    ap.add_argument("--twin", action="store_true", help="TD3: two critics, take the minimum in the target")
    ap.add_argument("--delay", dest="delay_flag", action="store_true", help="TD3: actor/targets every 2nd update")
    ap.add_argument("--smooth", action="store_true", help="TD3: noise on the target action")
    ap.add_argument("--td3", action="store_true", help="all three TD3 repairs")
    ap.add_argument("--trace", action="store_true", help="print the first update, with dQ/da")
    ap.add_argument("--seeds", type=int, default=0, help="run this many seeds and report who reaches return 1000")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="open the plot window as well as saving the PNG")
    args = ap.parse_args()
    kw = agent_options(args)
    tag = "td3" if (kw["twin"] and kw["delay"] == 2 and kw["smooth"]) else "ddpg"
    common = dict(steps=args.steps, warmup=args.warmup, noise=args.noise, sigma=args.sigma, tag=f"lab3_{tag}", **kw)

    env = gym.make(ENV_NAME)
    print(f"{ENV_NAME}: {env.observation_space.shape[0]} state numbers, 1 action in "
          f"[{env.action_space.low[0]:.0f}, {env.action_space.high[0]:.0f}], at most 1000 steps (return 1000).")
    print(f"{tag.upper()}: actor + critic networks {HIDDEN}, gamma {args.gamma}, tau {args.tau}, "
          f"{args.noise} noise sigma {args.sigma}, warm-up {args.warmup} steps\n")

    if args.seeds:
        rows = []
        for sd in range(args.seeds):
            res = train(seed=sd, verbose=False, **common)
            final = res["eval"][-1, 1] if len(res["eval"]) else float("nan")
            first = next((int(st) for st, v in res["eval"] if v >= 999.0), None)
            rows.append((sd, final, first))
            print(f"  seed {sd}: final greedy return {final:7.1f}   first reached 1000 at step {first if first else 'never'}")
        n_ok = sum(1 for _, f, _ in rows if f >= 999.0)
        print(f"\n{n_ok} of {args.seeds} seeds end at return 1000.")
        return

    result = train(seed=args.seed, trace=args.trace, **common)
    ev = evaluate(gym.make(ENV_NAME), result["agent"].policy(), episodes=20)
    print(f"\nGreedy controller over 20 episodes: mean return {ev.mean():.1f}, "
          f"{np.mean(ev >= 999):.0%} of them keep the pole up for all 1000 steps.")
    ck = out_dir() / f"lab3_{tag}_seed{args.seed}.pt"
    save_checkpoint(result, ck)
    print(f"Saved the networks to {ck} and the log to {result['csv']}")
    if tag == "ddpg" and args.seed == 0:
        save_checkpoint(result, out_dir() / "lab3_ddpg.pt")      # the file play.py loads
    make_figure(result, args, name="lab3_ddpg_inverted_pendulum.png" if tag == "ddpg" else f"lab3_{tag}_inverted_pendulum.png")


if __name__ == "__main__":
    main()
