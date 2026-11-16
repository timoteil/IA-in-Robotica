"""
gridworld.py -- the tiny deterministic world shared by both labs
================================================================

The world is a discrete-time system, exactly as in control theory:

        s_{k+1} = F(s_k, a_k)          reward   r_k = r(s_k, a_k)

The same action in the same state ALWAYS gives the same result. There are no
probabilities anywhere in this course.

The world is offered in two ways:

  1. THE MODEL      env.model(s, a) -> (next_state, reward, done)
                    You may ask "what would happen if I did a in state s?" for ANY
                    s and a. That is knowing the equations of the system. Used by
                    dynamic programming (Lab 1: value iteration).

  2. THE SIMULATOR  env.reset() / env.step(a)
                    You can only ACT from where you currently are and observe what
                    happens. That is learning from data. Used by Q-learning (Lab 2),
                    which must NEVER call env.model -- that is the whole point.

Map legend
----------
    S  start            .  empty cell          #  wall
    G  goal  (+goal_reward, episode ends)
    E  near exit (+exit_reward, episode ends)

Four actions: UP, DOWN, LEFT, RIGHT. Bumping into a wall or the border leaves you where
you are. Every move costs `step_reward` (a small negative reward); stepping onto G or E pays
that cell's reward instead and ends the episode.

Only numpy is needed here. matplotlib is imported lazily by `plot_grid`.
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
N_ACTIONS = 4
ACTION_NAMES = ["up", "down", "left", "right"]
ARROWS = ["↑", "↓", "←", "→"]
_DELTAS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# ------------------------------------------------------------------ palette (same as the slides)
INK, GREEN, AMBER, CORAL, BLUE, SLATE, LIGHT = (
    "#14202B", "#0B6E4F", "#F2A900", "#D1495B", "#2E86AB", "#8D99A6", "#EEF3F1")

# ------------------------------------------------------------------ the map
# A near exit (E, two moves away, +0.3) and a far goal (G, six moves away, +1.0).
# Rows are written with spaces between cells for readability; spaces are ignored.
LAYOUT = [
    ". . . . . G",
    ". # # # # .",
    ". . . . . .",
    "E . S . . .",
]


def make_env(step_reward: float = -0.02, goal_reward: float = 1.0, exit_reward: float = 0.3,
             layout=None) -> "GridWorld":
    """Build the course gridworld. Any reward can be overridden."""
    return GridWorld(layout or LAYOUT, step_reward, goal_reward, exit_reward)


class GridWorld:
    def __init__(self, layout, step_reward=-0.02, goal_reward=1.0, exit_reward=0.3):
        self.layout = [row.replace(" ", "") for row in layout]
        self.n_rows, self.n_cols = len(self.layout), len(self.layout[0])
        self.step_reward = step_reward
        self._terminal_reward = {"G": goal_reward, "E": exit_reward}

        # States are the non-wall cells, numbered row by row.
        self.cells, self.pos2s, self.kind = [], {}, []
        for r, row in enumerate(self.layout):
            for c, ch in enumerate(row):
                if ch != "#":
                    self.pos2s[(r, c)] = len(self.cells)
                    self.cells.append((r, c))
                    self.kind.append(ch)
        self.n_states = len(self.cells)
        self.n_actions = N_ACTIONS
        self.terminal = np.array([k in self._terminal_reward for k in self.kind])
        self.start = self.kind.index("S")

        self._table = [[self._transition(s, a) for a in range(N_ACTIONS)]
                       for s in range(self.n_states)]
        self.s = self.start

    # ------------------------------------------------------------ the MODEL
    def _move(self, s, a):
        """Where does action a take us from s?  Walls and borders leave us in place."""
        r, c = self.cells[s]
        dr, dc = _DELTAS[a]
        return self.pos2s.get((r + dr, c + dc), s)

    def _transition(self, s, a):
        if self.terminal[s]:                              # absorbing: nothing more happens
            return s, 0.0, True
        s2 = self._move(s, a)
        r = self._terminal_reward.get(self.kind[s2], self.step_reward)
        return s2, r, bool(self.terminal[s2])

    def model(self, s, a):
        """The system equations:  (s, a)  ->  (s', r, done).   Dynamic programming only."""
        return self._table[s][a]

    # ------------------------------------------------------------ the SIMULATOR
    def reset(self):
        self.s = self.start
        return self.s

    def step(self, a):
        """Act from the current state. Returns (next_state, reward, done)."""
        self.s, r, done = self._table[self.s][a]
        return self.s, r, done

    # ------------------------------------------------------------ helpers
    def follow_policy(self, actions, max_steps=200):
        """Cells visited when the agent starts at S and always does actions[state]."""
        path, s = [self.start], self.start
        for _ in range(max_steps):
            if self.terminal[s]:
                break
            s = self._move(s, int(actions[s]))
            path.append(s)
            if path.count(s) > 2:                         # stuck in a loop
                break
        return path

    def path_return(self, path, gamma):
        """Discounted sum of rewards collected along a path of states: the 'cost-to-go'."""
        total = 0.0
        for k in range(len(path) - 1):
            r = self._terminal_reward.get(self.kind[path[k + 1]], self.step_reward)
            total += gamma ** k * r
        return total

    def render(self, values=None, actions=None):
        """Plain-text picture of the grid, with optional values and/or greedy arrows."""
        lines = []
        for r in range(self.n_rows):
            row = []
            for c in range(self.n_cols):
                s = self.pos2s.get((r, c))
                if s is None:
                    row.append(" ##### ")
                elif self.terminal[s]:
                    row.append({"G": " GOAL  ", "E": " EXIT  "}[self.kind[s]])
                else:
                    txt = f"{values[s]:+.2f}" if values is not None else ("  S " if s == self.start else "  . ")
                    if actions is not None:
                        txt += ARROWS[int(actions[s])]
                    row.append(f" {txt:^6}")
            lines.append("|".join(row))
        return "\n".join(lines)


# ------------------------------------------------------------------ plotting
def plot_grid(env, ax, values=None, actions=None, path=None, title="", fmt="{:.2f}"):
    """Draw the grid on a matplotlib axis: value heat-map, greedy arrows and/or a path."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import Rectangle

    cmap = LinearSegmentedColormap.from_list("rl", [CORAL, "#FFFFFF", GREEN])
    nonterm = [s for s in range(env.n_states) if not env.terminal[s]]
    vmax = max(abs(np.asarray(values)[nonterm]).max(), 1e-9) if values is not None else 1.0
    norm = Normalize(-vmax, vmax)

    ax.set_xlim(0, env.n_cols), ax.set_ylim(env.n_rows, 0)
    ax.set_aspect("equal"), ax.axis("off")
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = env.pos2s.get((r, c))
            if s is None:
                face, label = INK, ""
            elif env.terminal[s]:
                k = env.kind[s]
                face = {"G": AMBER, "E": "#F7D488"}[k]
                label = k
            else:
                face = cmap(norm(values[s])) if values is not None else LIGHT
                label = ""
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=face, edgecolor="white", lw=2))
            if label:
                ax.text(c + .5, r + .5, label, ha="center", va="center", fontsize=16,
                        fontweight="bold", color=INK)
            elif s is not None:
                if values is not None:
                    ax.text(c + .5, r + (.78 if actions is not None else .5), fmt.format(values[s]),
                            ha="center", va="center", fontsize=8, color=INK)
                if actions is not None:
                    ax.text(c + .5, r + .38, ARROWS[int(actions[s])], ha="center", va="center",
                            fontsize=17, color=INK)
                if s == env.start:
                    ax.text(c + .08, r + .14, "S", ha="left", va="center", fontsize=8,
                            fontweight="bold", color=BLUE)
    if path:
        xs = [env.cells[s][1] + .5 for s in path]
        ys = [env.cells[s][0] + .5 for s in path]
        ax.plot(xs, ys, "-o", color=BLUE, lw=2.5, ms=5, alpha=.9)
    ax.set_title(title, fontsize=11, color=INK, pad=6)
