"""Build RL_in_One_Hour.pptx. Every number on the slides is computed from the lab code."""
import contextlib
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ["COURSE_ROOT"]                       # .../rl_course_1h
LABS = os.path.join(ROOT, "labs")
sys.path.insert(0, LABS)
sys.path.insert(0, HERE)

import numpy as np

import deckkit
import lab1_value_iteration as L1
import lab2_q_learning as L2
from deckkit import *
from gridworld import ARROWS, make_env

deckkit.COURSE = "Reinforcement Learning in One Hour"
G = 0.95
prs = new_presentation()
SECTIONS = []          # (name, first_slide_index)


def begin(name):
    SECTIONS.append((name, len(prs.slides)))


# ============================================================================ DATA FROM THE LABS
env = make_env()
STEP, GOAL_R, EXIT_R = env.step_reward, env._terminal_reward["G"], env._terminal_reward["E"]
N_LIVE = int((~env.terminal).sum())

snap = []
V, changes = L1.value_iteration(env, G, snapshots=snap)
act = L1.greedy_policy(env, V, G)
path_G = env.follow_policy(act)
V5, changes5 = L1.value_iteration(env, 0.5)
act5 = L1.greedy_policy(env, V5, 0.5)
path_E = env.follow_policy(act5)
assert env.kind[path_G[-1]] == "G" and env.kind[path_E[-1]] == "E"
ret_G, ret_E = env.path_return(path_G, G), env.path_return(path_E, G)
assert abs(ret_G - V[env.start]) < 1e-9

# the discount at which the agent switches from the near exit to the far goal
switch_gamma = None
for g_ in np.arange(0.50, 0.99, 0.01):
    Vg, _ = L1.value_iteration(env, g_)
    if env.kind[env.follow_policy(L1.greedy_policy(env, Vg, g_))[-1]] == "G":
        switch_gamma = round(float(g_), 2)
        break

# Q-learning: probing versus greedy
EPISODES = 150
env_p, env_g = make_env(), make_env()
Q_p, hist_p, reached_p = L2.q_learning(env_p, EPISODES, G, 1.0, "probing", 100)
Q_g, hist_g, reached_g = L2.q_learning(env_g, EPISODES, G, 1.0, "greedy", 100)
exact_p, exact_g = L2.exact_states(env, hist_p, V), L2.exact_states(env, hist_g, V)
all_exact_ep = int(np.argmax(exact_p == N_LIVE)) + 1
act_p, act_g = Q_p.argmax(1), Q_g.argmax(1)
path_p, path_g = env.follow_policy(act_p), env.follow_policy(act_g)
assert env.kind[path_p[-1]] == "G" and env.kind[path_g[-1]] == "E"
assert exact_p[-1] == N_LIVE and exact_g[-1] == 0
ret_p, ret_g = env.path_return(path_p, G), env.path_return(path_g, G)

# the worked update on slide 16 comes from the real trace of Lab 2
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    L2.q_learning(make_env(), 3, G, 1.0, "probing", 100, trace=3)
trace = buf.getvalue()
assert "s=(3,1) a=left  -> r=+0.30 s'=(3,0)   target=+0.300" in trace
assert "s=(3,2) a=left  -> r=-0.02 s'=(3,1)   target=+0.265" in trace

S = env.start
S_cell = env.cells[S]


def cell_s(r, c):
    return env.pos2s[(r, c)]


# ============================================================================ SMALL REUSABLE PIECES
def table(slide, x, y, col_w, rows, size=15, row_h=0.55, head_h=0.55, header_fills=None, first_bold=True,
          fills=None, fonts=None, aligns=None, name="Table"):
    nr, nc = len(rows), len(col_w)
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(head_h + row_h * (nr - 1)))
    gf.name = name
    tbl = gf.table
    tbl._tbl.tblPr.set("bandRow", "0")
    for j, wj in enumerate(col_w):
        tbl.columns[j].width = Inches(wj)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(head_h if i == 0 else row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.12)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[(aligns or {}).get(j, "l")]
            head = (i == 0)
            add_runs(p, str(val), size, "paper" if head else "ink", bold=head or (first_bold and j == 0),
                     font=(fonts or {}).get(j) if not head else None)
            c.fill.solid()
            if head:
                setc(c.fill.fore_color, (header_fills or {}).get(j, "pine"))
            else:
                setc(c.fill.fore_color, (fills or {}).get((i, j), "mist" if i % 2 else "paper"))
    return gf


def tile(slide, x, y, w, h, title, body, fill="mist", num=None, title_pt=18, body_pt=15, title_color="ink", name=None):
    box(slide, x, y, w, h, fill=fill, radius=0.1, name=name or ("Card: " + title))
    off = 0.2
    if num:
        badge(slide, x + 0.2, y + 0.2, 0.5, str(num), fill="green", size=16)
        off = 0.85
    text(slide, x + 0.2, y + off, w - 0.4, 0.5, title, size=title_pt, bold=True, color=title_color)
    text(slide, x + 0.2, y + off + 0.45, w - 0.4, h - off - 0.55, body, size=body_pt, color="ink")


def callout(slide, x, y, w, h, content, fill="amber_t", size=16, name="Callout", color="ink"):
    box(slide, x, y, w, h, fill=fill, radius=0.1, name=name)
    text(slide, x + 0.25, y + 0.1, w - 0.5, h - 0.2, content, size=size, color=color, anchor="m")


def panel_label(slide, x, w, y, content, size=16):
    text(slide, x, y, w, 0.4, content, size=size, bold=True, color="ink", align="c")


def lab_slide(title, intro, commands, exercises, notes, outcome):
    s = content_slide(prs, title, module="Lab", notes=notes, chip_color="amber")
    text(s, 0.6, 1.55, 6.0, 0.45, "What you will do", size=20, bold=True)
    text(s, 0.6, 2.05, 6.0, 2.6, intro, size=16, bullets=True, space_after=7)
    text(s, 0.6, 4.75, 6.0, 0.4, "Run it", size=20, bold=True)
    code_block(s, 0.6, 5.2, 6.0, len(commands) * 0.34 + 0.4, commands, size=14, name="Commands")
    box(s, 7.0, 1.55, 5.75, 3.55, fill="mist", radius=0.1, name="Exercises card")
    text(s, 7.3, 1.7, 5.2, 0.45, "Exercises", size=20, bold=True)
    text(s, 7.3, 2.2, 5.2, 2.9, exercises, size=15, bullets=True, space_after=6)
    callout(s, 7.0, 5.3, 5.75, 1.45, outcome, fill="green_t", size=15, name="Expected outcome")
    return s


def link_runs(tb, url):
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            r.hyperlink.address = url
            setc(r.font.color, "blue")
            r.font.underline = True


def f3(x):
    return f"{x:+.3f}"


def mark_start(slide, x, y, cell):
    """A readable 'S' in the start cell (draw_grid's own marker is tiny)."""
    r_, c_ = env.cells[env.start]
    text(slide, x + c_ * cell, y + r_ * cell, cell, cell, "S", size=max(12, int(cell * 28)), color="blue", bold=True,
         align="c", anchor="m", name="Start marker")


def neg(txt):
    return txt.replace("-", "−")


# ============================================================================ INTRODUCTION
begin("Introduction")

# ---- 1 title
s = prs.slides.add_slide(prs.slide_layouts[5])
dark_background(s)
t = s.shapes.title
t.text_frame.text = "Reinforcement Learning in One Hour"
t.left, t.top, t.width, t.height = Inches(0.8), Inches(1.7), Inches(6.9), Inches(2.6)
t.text_frame.vertical_anchor = MSO_ANCHOR.TOP
for r in t.text_frame.paragraphs[0].runs:
    r.font.size = Pt(48)
    setc(r.font.color, "paper")
text(s, 0.8, 4.3, 6.7, 1.0, "From Bellman's equation to Q-learning: a deterministic, tabular introduction",
     size=20, color="mint", name="Subtitle")
text(s, 0.8, 5.55, 6.9, 1.2, ["Based on David Silver's UCL / DeepMind course and on Lewis & Vrabie, "
                              "“Reinforcement Learning and Adaptive Dynamic Programming for Feedback Control”, IEEE CAS Magazine, 2009",
                              "Two short Python labs · no probabilities · no function approximators"],
     size=14, color="code_dim", name="Credits", space_after=6)
draw_grid(s, env, 8.1, 1.7, 0.78, actions=act, dark=True, show_start=False, arrow_pt=22, name="Title gridworld")
text(s, 8.1, 5.75, 4.7, 0.6, "The optimal policy for one gridworld: this is what you will compute, then learn.",
     size=14, color="code_dim", name="Grid caption")
s.notes_slide.notes_text_frame.text = (
    "Welcome. This is a one-hour introduction. Timing: about 3 minutes of introduction, 11 on the problem, 17 on planning "
    "with a model (including a 5-minute live demo of Lab 1), 19 on learning without a model (including a 5-minute live demo "
    "of Lab 2), 7 to wrap up and 3 for questions. The world is deterministic: the same action in the same state always gives the "
    "same result, so there are no probabilities to track. Everything is stored in tables you can print. The gridworld on the right "
    "is the running example: by the end you will have computed its optimal policy with value iteration and then learned it "
    "from experience with Q-learning. The framing of control versus learning comes from Lewis and Vrabie's 2009 paper.")

# ---- 2 the hour at a glance
s = content_slide(prs, "The hour at a glance", notes=(
    "Walk along the bar. The amber blocks are the two live demos of the labs: run the script, read the output together, and leave "
    "the exercises for afterwards. The four tiles are the contract of the hour. Say out loud what is not in the hour: policy "
    "iteration, bandits and exploration strategies, probabilistic worlds and function approximators. Those are natural follow-ups, "
    "not prerequisites."))
segs = [("Intro", 3, "slate", "3"), ("The problem", 11, "green", "11 min"), ("Planning", 12, "blue", "12 min"),
        ("Lab 1", 5, "amber", "5"), ("Learning", 14, "plum", "14 min"), ("Lab 2", 5, "amber", "5"),
        ("Wrap-up", 7, "slate", "7 min"), ("Q&A", 3, "line", "3")]
x0, per_min = 0.6, 12.1 / 60.0
for name, mins, col, lab in segs:
    w_ = mins * per_min
    b_ = box(s, x0, 1.7, w_ - 0.04, 0.75, fill=col, radius=0.06, name="Time block: " + name)
    label_in(b_, lab, size=15, bold=True, color="ink" if col in ("amber", "line") else "paper")
    text(s, x0 - 0.1, 2.52, w_ + 0.16, 0.35, name, size=13, color="muted", align="c", name="Time label: " + name)
    x0 += w_
text(s, 0.6, 3.0, 12.1, 0.4, "60 minutes · 22 slides · two live demos", size=15, color="muted", italic=True)
goals = [("Describe a problem", "as a system with states, actions, a reward and a discount: a deterministic MDP"),
         ("Plan with Bellman", "compute optimal values and a policy with value iteration (Lab 1)"),
         ("Learn from data", "fill a Q-table from experience with no model of the world (Lab 2)"),
         ("Explain exploration", "why the data must cover every state–action pair")]
for i, (h_, b_) in enumerate(goals):
    tile(s, 0.6 + i * 3.09, 3.55, 2.85, 2.55, h_, b_, num=i + 1, title_pt=18, body_pt=15)
text(s, 0.6, 6.4, 12.1, 0.5, "Not in this hour: policy iteration, bandits, exploration strategies, probabilistic worlds, function approximators.",
     size=15, color="muted", italic=True)

# ============================================================================ PART 1  THE PROBLEM
begin("1 · The problem")

# ---- 3 learning by acting
s = content_slide(prs, "Learning by acting", "1", notes=(
    "Start with the loop. At every step the agent looks at the state, chooses an action, and the world answers with a reward and "
    "a new state. That is the whole interface, and it is exactly what env.step(a) does in Lab 2. Three things make this different "
    "from learning with labels: nobody tells the agent which action was right, the feedback can arrive long after the decision "
    "that caused it, and the goal is the total reward over time, not the next reward. Lewis and Vrabie open their paper with the "
    "same idea: living organisms improve their behaviour by acting, observing the reward and adjusting. About 3 minutes."))
box(s, 0.6, 1.65, 2.8, 1.85, fill="green_t", line="green", radius=0.12, name="Agent")
text(s, 0.6, 1.8, 2.8, 0.5, "Agent", size=26, bold=True, align="c", color="green")
text(s, 0.75, 2.4, 2.5, 1.0, "picks actions with its policy", size=15, align="c")
box(s, 5.6, 1.65, 2.8, 1.85, fill="blue_t", line="blue", radius=0.12, name="Environment")
text(s, 5.6, 1.8, 2.8, 0.5, "Environment", size=26, bold=True, align="c", color="blue")
text(s, 5.75, 2.4, 2.5, 1.0, "the world; its rules may be unknown", size=15, align="c")
math(s, 3.45, 1.62, 2.1, 0.4, "action a_{k}", size=18, align="c")
line(s, 3.45, 2.05, 5.55, 2.05, color="ink", w=2.5, arrow=True)
math(s, 3.45, 2.3, 2.1, 0.4, "state s_{k+1}", size=18, align="c")
line(s, 5.55, 2.72, 3.45, 2.72, color="ink", w=2.5, arrow=True)
math(s, 3.45, 2.85, 2.1, 0.4, "reward r_{k}", size=18, align="c", color="coral")
line(s, 5.55, 3.3, 3.45, 3.3, color="coral", w=2.5, arrow=True)
text(s, 9.0, 1.65, 3.7, 0.45, "At every step k", size=18, bold=True)
text(s, 9.0, 2.15, 3.7, 1.5, ["The agent sees the state s_{k}", "It chooses an action a_{k}", "The world replies with r_{k} and s_{k+1}"],
     size=16, bullets=True, space_after=6)
cards = [("No teacher", "Nobody says which action was right. Only a number comes back: the reward.", "mist"),
         ("Late feedback", "A move may pay off many steps later. The agent has to connect cause and effect.", "mist"),
         ("A goal, not a label", "Maximise the total reward over time, not just the next reward.", "mist")]
for i, (h_, d_, c_) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 4.05, 3.9, 2.0, fill=c_, radius=0.1, name="Card: " + h_)
    text(s, x + 0.25, 4.2, 3.4, 0.5, h_, size=20, bold=True)
    text(s, x + 0.25, 4.75, 3.4, 1.2, d_, size=16)
text(s, 0.6, 6.4, 12.1, 0.5, "Nature does the same: animals adjust their behaviour from reward and punishment (Lewis & Vrabie, 2009).",
     size=15, color="muted", italic=True)

# ---- 4 vocabularies
s = content_slide(prs, "One loop, two vocabularies", "1", notes=(
    "Lewis and Vrabie write the same loop in the language of control: the environment is the system or plant, the policy is a "
    "feedback controller, reward is the negative of a cost, and the value is the cost-to-go. Keep this table in mind if you come "
    "from control or circuits: the maths is the same, only the names and one sign differ. We use the left-hand words in this "
    "course and maximise reward; the paper minimises cost. A discrete-time system x(k+1) = F(x(k), u(k)) already has the "
    "Markov property: the next state depends only on the current state and input. About 2 minutes."))
rows = [["In this course", "Symbol", "In control theory", "Symbol"],
        ["Environment", "s_{k+1} = F(s_{k}, a_{k})", "System, plant", "x_{k+1} = F(x_{k}, u_{k})"],
        ["State", "s", "State", "x"],
        ["Action", "a", "Control input", "u"],
        ["Policy", "a = π(s)", "Feedback controller", "u = h(x)"],
        ["Reward (maximise)", "r(s, a)", "Utility, cost (minimise)", "r(x, u)"],
        ["Value", "V(s)", "Cost-to-go", "V^{h}(x)"],
        ["Q-value", "Q(s, a)", "Action-dependent value", "Q^{h}(x, u)"],
        ["TD error", "e = r + γ V(s′) − V(s)", "Bellman residual (Hamiltonian)", "e_{k} = r + γ V(x_{k+1}) − V(x_{k})"]]
table(s, 0.6, 1.5, [2.4, 3.3, 3.2, 3.2], rows, size=15, row_h=0.5, head_h=0.5, fonts={1: "Cambria", 3: "Cambria"},
      header_fills={0: "green", 1: "green", 2: "slate", 3: "slate"}, name="RL and control vocabulary")
callout(s, 0.6, 6.15, 12.1, 0.7, "**Sign convention:** reward = −cost. Maximising reward and minimising cost are the same problem.",
        fill="amber_t", size=16, name="Sign convention")

# ---- 5 the world
s = content_slide(prs, "The world: a deterministic system", "1", notes=(
    "This is the running example for the whole hour: a 4 by 6 grid with 18 free cells plus two exits. The agent starts at S. The "
    "near exit E is two moves away and pays 0.3. The far goal G is six moves away and pays 1.0. Every move costs 0.02, so "
    "wandering is penalised. Walking into a wall leaves you where you are. This is a Markov decision process without the "
    "probabilities: the same action in the same state always gives the same result, as in a discrete-time control system. "
    "Real problems are noisy; handling that is the natural next course. About 3 minutes."))
draw_grid(s, env, 0.7, 1.7, 0.85, name="Course gridworld", start_label=False)
mark_start(s, 0.7, 1.7, 0.85)
legend = [("S", "start", "mist", "blue"), ("E", "near exit, +%.1f" % EXIT_R, "amber_l", "ink"),
          ("G", "far goal, +%.1f" % GOAL_R, "amber", "ink"), ("", "wall", "ink", "paper")]
for (sym, desc, fill, tcol), lx, lw in zip(legend, (0.7, 1.7, 3.7, 5.35), (0.9, 1.9, 1.5, 0.6)):
    b_ = box(s, lx, 5.25, 0.38, 0.38, fill=fill, radius=0.04, line="line", line_w=1, name="Legend " + desc)
    label_in(b_, sym, size=13, bold=True, color=tcol)
    text(s, lx + 0.45, 5.25, lw, 0.38, desc, size=12, anchor="m")
cards = [("State s", "the cell the agent is in: %d cells, %d where it can act and 2 exits" % (env.n_states, N_LIVE)),
         ("Action a", "move ↑ ↓ ← →"),
         ("Transition F", "s′ = F(s, a): the cell you move to; a wall leaves you where you are"),
         ("Reward r", "%.2f for every move, +%.1f at E, +%.1f at G; reaching E or G ends the episode" % (STEP, EXIT_R, GOAL_R))]
for i, (h_, d_) in enumerate(cards):
    y = 1.65 + i * 0.98
    box(s, 6.6, y, 6.1, 0.88, fill="mist", radius=0.1, name="Card: " + h_)
    text(s, 6.8, y, 1.7, 0.88, h_, size=17, bold=True, color="green", anchor="m")
    text(s, 8.5, y, 4.1, 0.88, d_, size=14, anchor="m")
callout(s, 0.6, 5.95, 12.1, 0.85, "**Deterministic:** the same state and the same action always give the same result. "
        "There are no probabilities anywhere in this course.", fill="amber_t", size=17, name="Deterministic")

# ---- 6 policy and return
s = content_slide(prs, "Policy and return", "1", notes=(
    "A policy is a rule that picks an action in every state: here, one arrow per cell. To compare policies we need one number "
    "per policy: the return, or cost-to-go in Lewis and Vrabie's words. It is the sum of the rewards collected, each multiplied "
    "by gamma to the power of how many steps away it is. Both policies on the slide are legitimate: A heads for the near exit, "
    "B for the far goal. With gamma = 0.95 the far goal wins: %s against %s. The value V of a state is the return you get by "
    "starting there and following the policy. About 3 minutes." % (f3(ret_G), f3(ret_E))))
text(s, 0.6, 1.5, 5.8, 1.1, ["**Policy π:** a rule that picks an action in every state. A table with one arrow per cell."],
     size=17)
text(s, 6.9, 1.5, 5.8, 1.2, ["**Return:**  G = r_{0} + γ r_{1} + γ^{2} r_{2} + …", "V_{π}(s) is the return of starting in s and following π."],
     size=17, space_after=4)
for i, (lab, path_, ret_, calc, col) in enumerate([
        ("Policy A: head for the near exit E", path_E, ret_E,
         "%.2f + γ × %.2f = %s" % (STEP, EXIT_R, f3(ret_E)), "amber"),
        ("Policy B: head for the far goal G", path_G, ret_G,
         "%.2f (1 + γ + γ² + γ³ + γ⁴) + γ⁵ × %.1f = %s" % (STEP, GOAL_R, f3(ret_G)), "green")]):
    x = 0.9 + i * 6.1
    panel_label(s, x - 0.3, 4.3, 2.75, lab, size=17)
    draw_grid(s, env, x, 3.2, 0.62, path=path_, name="Route " + lab, start_label=False)
    mark_start(s, x, 3.2, 0.62)
    text(s, x - 0.3, 5.75, 4.3, 0.4, "%d moves" % (len(path_) - 1), size=15, color="muted", align="c")
    math(s, x - 1.0, 6.1, 5.7, 0.5, neg(calc), size=16, align="c", color="ink")

# ============================================================================ PART 2  PLANNING
begin("2 · Planning with a model")

# ---- 7 Bellman's principle
s = content_slide(prs, "Bellman's principle of optimality", "2", notes=(
    "Which policy is best? Trying them all is hopeless: even this tiny map has 4 to the power 18, about 69 billion, deterministic "
    "policies. Bellman's principle breaks the problem up: whatever the first move was, the rest of the route must be the best "
    "one from wherever you land. In an equation: the best value of a state is the best, over actions, of the reward now plus "
    "gamma times the best value of the next state. The next state is F(s, a): we are using the model. This equation is the "
    "thread through the rest of the hour: Lab 1 solves it with the model, Lab 2 solves it from data. About 3 minutes."))
box(s, 0.6, 1.6, 4.0, 3.5, fill="coral_t", radius=0.12, name="Stat: brute force")
math(s, 0.6, 1.75, 4.0, 1.3, "6.9 × 10^{10}", size=44, color="coral", align="c", bold=True)
text(s, 0.85, 3.1, 3.5, 1.9, ["deterministic policies on this little map (4^{%d})." % N_LIVE,
                              "Comparing them one by one is hopeless."], size=17, align="c", space_after=6)
text(s, 5.0, 1.6, 7.7, 1.2, "Whatever the first move, the rest of the route must be the best one from wherever you land.",
     size=22, bold=True, font="Cambria")
math(s, 5.0, 3.0, 7.7, 0.8, "V_{*}(s) = max_{a} [ r(s,a) + γ V_{*}(F(s,a)) ]", size=28)
for i, (h_, d_, col) in enumerate([("V_{*}(s)", "the best total reward from s", "green"),
                                   ("r(s,a)", "the reward now", "coral"), ("γ V_{*}(F(s,a))", "discounted best value of the next state", "blue")]):
    x = 5.0 + i * 2.6
    box(s, x, 3.95, 2.45, 1.15, fill="mist", radius=0.1, name="Term: " + d_)
    math(s, x, 4.0, 2.45, 0.45, h_, size=17, color=col, align="c", bold=True)
    text(s, x + 0.1, 4.45, 2.25, 0.6, d_, size=13, align="c")
box(s, 0.6, 5.5, 12.1, 1.2, fill="pine", radius=0.12, name="Plan")
text(s, 0.95, 5.5, 11.4, 1.2, "So: compute V_{*} for every cell, then act greedily with respect to it. "
     "F is the model: this is planning.", size=19, color="paper", anchor="m")

# ---- 8 one Bellman step
s_cell = cell_s(2, 5)
qv = L1.one_step_lookahead(env, V, s_cell, G)
assert int(np.argmax(qv)) == 0 and abs(qv.max() - V[s_cell]) < 1e-9
s = content_slide(prs, "One Bellman step, with numbers", "2", notes=(
    "Take the highlighted cell, in the right-hand column just below the goal's neighbour. For each of the four actions, look up "
    "where it leads and what it pays, then add gamma times the value of that next cell. Going up reaches the cell next to the goal "
    "with value 1.0: -0.02 + 0.95 x 1.00 = 0.930. Going down or left leads to cells worth %.3f, so they score %.3f. Going right bumps "
    "into the border and leaves you in the same cell, worth %.3f itself, so it scores %.3f. The max is %.3f, which is also the value "
    "already in the table: the table satisfies the equation. About 2 minutes." % (V[env.model(s_cell, 1)[0]], qv[1], V[s_cell], qv[3], qv.max())))
draw_grid(s, env, 0.7, 1.8, 0.85, values=V, vmax=1.0, value_pt=12, path=[s_cell], name="V* with one cell highlighted")
text(s, 0.7, 5.3, 5.1, 0.7, "Optimal values V_{*} for γ = %.2f. The highlighted cell is the one we check." % G, size=14, color="muted")
rows = [["Action", "Lands in", "r + γ V_{*}(next)", "Value"]]
names = ["↑ up", "↓ down", "← left", "→ right"]
for a_ in range(4):
    s2, r_, done_ = env.model(s_cell, a_)
    land = "(%d,%d)" % env.cells[s2] if s2 != s_cell else "same cell (wall)"
    rows.append([names[a_], land, neg("%.2f + %.2f × %.3f" % (r_, G, V[s2])), "%.3f" % qv[a_]])
best_row = 1 + int(np.argmax(qv))
table(s, 6.3, 1.8, [1.3, 1.7, 2.3, 1.1], rows, size=15, row_h=0.6, head_h=0.55, fonts={2: "Cambria"},
      fills={(best_row, j): "green_t" for j in range(4)}, aligns={3: "c"}, name="Four candidate actions")
math(s, 6.3, 4.7, 6.4, 0.6, "V_{*}(%d,%d) = max{ %s } = %.3f" % (env.cells[s_cell] + (", ".join("%.3f" % q for q in qv), qv.max())),
     size=18)
callout(s, 6.3, 5.4, 6.4, 1.0, "Best action: **↑**. Its value is already the one stored in the table: the table satisfies the equation.",
        fill="green_t", size=16, name="Best action")

# ---- 9 value iteration
s = content_slide(prs, "Value iteration", "2", notes=(
    "The equation has V* on both sides. Lewis and Vrabie point out that it is a fixed-point equation: apply its right-hand side "
    "again and again and the table converges, starting from any table. That is value iteration. Start from all zeros, and in each "
    "sweep replace every value by the best one-step look-ahead computed from the previous table. A nice reading: after k sweeps, V "
    "holds the best total reward you can collect in k moves. It needs the model, F and r, to do the look-ahead. Compare with the "
    "code in lab1_value_iteration.py: it is the same six lines. About 3 minutes."))
math(s, 0.6, 1.55, 12.1, 0.8, "V_{k+1}(s) = max_{a} [ r(s,a) + γ V_{k}(F(s,a)) ]", size=28, align="c")
code_block(s, 0.6, 2.6, 6.4, 3.15, ["V ← 0 for every cell", "repeat:", "  for each non-terminal cell s:", "    for each action a:",
                                   "      q[a] ← r(s,a) + γ V[F(s,a)]", "    V_new[s] ← max over a of q[a]", "  V ← V_new      # old table on the right",
                                   "until V stops changing", "π(s) ← the a with the largest q[a]"], size=15, name="Value iteration pseudo-code")
text(s, 7.4, 2.6, 5.3, 0.45, "What to notice", size=20, bold=True)
text(s, 7.4, 3.15, 5.3, 3.5, ["Any starting table works: zeros are fine", "Each sweep is one look-ahead per cell, using the previous table",
                              "After k sweeps, V holds the best reward collectable in k moves",
                              "It needs the model: F and r"], size=16, bullets=True, space_after=9)

# ---- 10 VI in action
s = content_slide(prs, "Value iteration in action", "2", notes=(
    "At k = 0 every value is zero. After one sweep only the cells next to an exit know about its reward. Each further sweep pushes "
    "that information one cell further back, shrinking it by gamma and by the step cost. After %d sweeps nothing changes: these "
    "are the optimal values. Information flows backwards from the reward. The nearly white cells are not yet informed: their value "
    "is just the cost of the moves taken so far. About 2 minutes." % len(changes)))
vi_panels = [("k = 1", snap[1]), ("k = 2", snap[2]), ("k = 4", snap[4]), ("k = %d  (V_{*})" % len(changes), snap[len(changes)])]
assert np.allclose(snap[len(changes)], V)
for i, (lab, vals) in enumerate(vi_panels):
    x = 0.7 + i * 3.08
    panel_label(s, x, 2.82, 1.6, lab, size=18)
    draw_grid(s, env, x, 2.1, 0.47, values=vals, vmax=1.0, value_pt=10, fmt="{:.2f}", name="Value iteration " + lab, start_label=False)
math(s, 0.6, 4.6, 12.1, 0.75, "V_{k+1}(s) = max_{a} [ r(s,a) + γ V_{k}(F(s,a)) ]", size=24, align="c")
text(s, 0.6, 5.5, 12.1, 1.0, ["Reward information flows backwards from the exits, one cell per sweep, shrinking by γ each time.",
                              "γ = %.2f. It converges after %d sweeps; the longest route is 6 moves." % (G, len(changes))],
     size=16, color="muted", align="c", space_after=4)

# ---- 11 values to policy, gamma
s = content_slide(prs, "Read the policy off the values", "2", notes=(
    "Once V is known, the policy is one more look-ahead: in every cell take the action with the best r + gamma V of the next cell. "
    "Follow the arrows from S and you get the blue route. The return of that route equals V(S): %s on the left. On the right, the "
    "same code and map with gamma = 0.5: the agent cares much less about the future, so the far goal is not worth the walk and "
    "it takes the near exit instead. The switch happens at gamma of about %.2f. Gamma is the planning horizon. This is the "
    "myopic versus far-sighted planner of Lewis and Vrabie. About 3 minutes." % (f3(V[S]), switch_gamma)))
math(s, 0.6, 1.5, 12.1, 0.7, "π(s) = argmax_{a} [ r(s,a) + γ V_{*}(F(s,a)) ]", size=26, align="c")
for i, (lab, vals, acts, path_, tag) in enumerate([
        ("γ = %.2f: far-sighted" % G, V, act, path_G, "%d moves to G · V(S) = %s" % (len(path_G) - 1, f3(V[S]))),
        ("γ = 0.5: short-sighted", V5, act5, path_E, "%d moves to E · V(S) = %s" % (len(path_E) - 1, f3(V5[S])))]):
    x = 0.7 + i * 4.3
    panel_label(s, x, 3.72, 2.3, lab, size=17)
    draw_grid(s, env, x, 2.75, 0.62, values=vals, actions=acts, path=path_, vmax=1.0, arrow_pt=16, value_pt=9, name="Policy " + lab)
    text(s, x, 5.3, 3.72, 0.4, tag, size=15, color="blue", bold=True, align="c")
callout(s, 9.3, 2.75, 3.45, 2.95, ["**γ is the horizon:** how far ahead the agent cares.",
                                   "Same code, same map: the route switches near γ = %.2f." % switch_gamma],
        fill="amber_t", size=17, name="Takeaway")
text(s, 0.6, 6.15, 12.1, 0.6, "Numbers: optimal values V_{*}. Arrows: the greedy policy. Blue outline: the route from S.",
     size=15, color="muted", italic=True)

# ---- 12 Lab 1
lab_slide("Lab 1: value iteration on a gridworld",
          ["Read gridworld.py: env.model(s, a) is the model", "Run value iteration: values, policy, route",
           "Stop early with --sweeps k and watch the wave of value", "Change γ and the step cost; explain the new route"],
          ["python lab1_value_iteration.py", "python lab1_value_iteration.py --sweeps 3", "python lab1_value_iteration.py --gamma 0.5"],
          ["How many sweeps? Compare with the 6 moves from S to G.", "After k sweeps, where is the value non-zero? Why there?",
           "At which γ does the agent switch to the near exit?", "Make walking free, then expensive. What changes?"],
          "Live demo, about 5 minutes. Run the first command and read the output together: the grid of values with arrows, the route "
          "with its return, and the line saying V(S) equals the return of the best route. Then run --sweeps 3 to show the wave, and "
          "--gamma 0.5 to show the switch. Leave the exercises for after the session. Everything is in about 100 lines.",
          "You should see: %d sweeps, a %d-move route to G and V(S) = %s. With --gamma 0.5: %d moves to the near exit, V(S) = %s."
          % (len(changes), len(path_G) - 1, f3(V[S]), len(path_E) - 1, f3(V5[S])))

# ============================================================================ PART 3  LEARNING
begin("3 · Learning without a model")

# ---- 13 planning vs learning
s = content_slide(prs, "The catch: planning needs the model", "3", notes=(
    "Value iteration asked the model a what-if question for every cell and every action. In most real problems we do not have that "
    "model: we only have the real system or a simulator, and we can only act from where we are and watch what happens. "
    "Lewis and Vrabie make the same contrast: dynamic programming works backwards through the known equations, offline; reinforcement "
    "learning works forward in time, online, from measured data (state, next state, reward). The goal is the same, V* and the optimal "
    "policy. Can we reach it from data alone? About 2 minutes."))
for i, (h_, tag, c_, col, code, bullets) in enumerate([
        ("Planning", "Lab 1", "blue_t", "blue", ["s2, r, done = env.model(s, a)", "  # any state, any action"],
         ["Knows F and r", "Works backwards through the model, offline", "Sweeps every cell"]),
        ("Learning", "Lab 2", "green_t", "green", ["s2, r, done = env.step(a)", "  # act, then observe"],
         ["Does not know F or r", "Learns forward in time from data (s, a, r, s′)", "Only learns about what it visits"])]):
    x = 0.6 + i * 6.2
    box(s, x, 1.6, 5.9, 4.0, fill=c_, radius=0.12, name="Card: " + h_)
    text(s, x + 0.3, 1.75, 4.0, 0.5, h_, size=24, bold=True, color=col)
    text(s, x + 4.2, 1.8, 1.4, 0.45, tag, size=16, bold=True, color=col, align="r")
    code_block(s, x + 0.3, 2.4, 5.3, 1.0, code, size=15, name="Interface: " + h_)
    text(s, x + 0.3, 3.65, 5.3, 1.9, bullets, size=17, bullets=True, space_after=7, bullet_color=col)
box(s, 0.6, 5.95, 12.1, 0.95, fill="pine", radius=0.12, name="Question")
text(s, 0.95, 5.95, 11.4, 0.95, "Same goal: V_{*} and the optimal policy. Can we get there from data alone?", size=20, color="paper", anchor="m")

# ---- 14 why Q
s = content_slide(prs, "Why a table of Q, not of V", "3", notes=(
    "To act using V you need the model, because choosing an action means asking where each action leads. The trick, due to Watkins "
    "and presented in the paper as action-dependent value learning, is to store the look-ahead itself: Q(s, a) is the value of "
    "doing a in s and then acting well. Now the best action is just the largest entry in the row, and V is the largest entry too. "
    "No model needed to act. The table on the right is the real table Lab 2 learns, for three cells. Look at the cell next to the "
    "near exit: going left looks attractive at 0.3, but going up is worth 0.63. The price: one entry per state-action pair, here "
    "%d x 4 = %d numbers. About 3 minutes." % (N_LIVE, N_LIVE * 4)))
math(s, 0.6, 1.55, 6.0, 0.55, "V(s) → needs F to choose:", size=18, color="coral", bold=True)
math(s, 0.6, 2.1, 6.0, 0.6, "π(s) = argmax_{a} [ r(s,a) + γ V(F(s,a)) ]", size=20)
math(s, 0.6, 3.0, 6.0, 0.55, "Q(s,a) → stores the look-ahead:", size=18, color="green", bold=True)
math(s, 0.6, 3.55, 6.0, 0.6, "Q(s,a) = r(s,a) + γ V(F(s,a))", size=20)
math(s, 0.6, 4.2, 6.0, 0.6, "π(s) = argmax_{a} Q(s,a)    V(s) = max_{a} Q(s,a)", size=20)
callout(s, 0.6, 5.2, 6.0, 1.3, "**No model needed to act:** the best action is the largest entry in the row.", fill="green_t", size=17,
        name="No model needed")
cells_q = [("S = (%d,%d)" % S_cell, S), ("(3,1)", cell_s(3, 1)), ("(2,5)", cell_s(2, 5))]
rows = [["State", "↑", "↓", "←", "→"]]
fills = {}
for i, (lab, st) in enumerate(cells_q, start=1):
    q = Q_p[st]
    rows.append([lab] + ["%.3f" % v_ for v_ in q])
    for j in range(4):
        if q[j] >= q.max() - 1e-9:
            fills[(i, j + 1)] = "green_t"
table(s, 7.1, 1.75, [1.6, 1.0, 1.0, 1.0, 1.0], rows, size=16, row_h=0.7, head_h=0.6, fills=fills,
      aligns={1: "c", 2: "c", 3: "c", 4: "c"}, name="Learned Q-values")
text(s, 7.1, 4.55, 5.6, 0.9, "Three rows of the table Lab 2 learns. Green: the best action(s). Ties are real: from S, up and right are equally good.",
     size=14, color="muted")
text(s, 7.1, 5.55, 5.6, 0.9, "Price: one entry per state–action pair: %d × 4 = %d numbers." % (N_LIVE, N_LIVE * 4), size=16)

# ---- 15 TD error
s = content_slide(prs, "Learning from one transition: the TD error", "3", notes=(
    "The agent observes one transition: it was in s, did a, got r and landed in s'. Before that, the table predicted Q(s, a). "
    "After it, the agent has a better estimate: the reward it just got plus gamma times the best it thinks it can do from s'. The "
    "difference is the temporal-difference error. If it is zero, the table is already consistent with the Bellman equation along "
    "this transition and there is nothing to learn. Otherwise we move the entry towards the observation by alpha times the error. "
    "In this deterministic world alpha = 1 is enough: we replace the old estimate. Lewis and Vrabie call the TD error a prediction "
    "error between predicted and observed performance, and note that dopamine neurons seem to carry the same signal. About 3 minutes."))
box(s, 0.6, 1.55, 12.1, 0.85, fill="mist", radius=0.1, name="Data")
text(s, 0.85, 1.55, 3.2, 0.85, "One observed transition", size=16, bold=True, color="muted", anchor="m")
math(s, 4.0, 1.55, 8.6, 0.85, "( s,  a,  r,  s′ )", size=28, align="c")
b = box(s, 0.6, 2.9, 3.6, 1.4, fill="blue_t", line="blue", radius=0.12, name="Predicted")
label_in(b, ["**Predicted**", "Q(s, a)"], size=20, font="Cambria")
math(s, 4.3, 3.15, 0.5, 0.9, "−", size=36, align="c")
b = box(s, 4.9, 2.9, 4.1, 1.4, fill="green_t", line="green", radius=0.12, name="Observed")
label_in(b, ["**Observed**", "r + γ max_{a′} Q(s′, a′)"], size=20, font="Cambria")
line(s, 9.1, 3.6, 9.7, 3.6, color="ink", w=3, arrow=True)
b = box(s, 9.8, 2.9, 2.9, 1.4, fill="amber_t", line="amber", radius=0.12, name="TD error")
label_in(b, ["**TD error**", "e"], size=20, font="Cambria")
box(s, 0.6, 4.65, 12.1, 0.95, fill="pine", radius=0.12, name="Update")
math(s, 0.6, 4.65, 12.1, 0.95, "Q(s, a)  ←  Q(s, a) + α · e", size=28, color="paper", align="c")
for i, (h_, d_) in enumerate([("e = 0", "the table is consistent with Bellman's equation here: nothing to learn"),
                              ("α = 1", "deterministic world: the new estimate simply replaces the old one"),
                              ("Brain", "dopamine neurons seem to signal this prediction error (Lewis & Vrabie, 2009)")]):
    x = 0.6 + i * 4.1
    box(s, x, 5.85, 3.9, 1.0, fill="mist", radius=0.1, name="Note: " + h_)
    math(s, x + 0.15, 5.85, 0.95, 1.0, h_, size=15, color="green", bold=True, align="c")
    text(s, x + 1.1, 5.85, 2.7, 1.0, d_, size=12, anchor="m")

# ---- 16 Q-learning = sampled VI
s = content_slide(prs, "Q-learning is value iteration from data", "3", notes=(
    "Put the two updates side by side. Both do one look-ahead and take a max. The only difference is where the next state and the "
    "reward come from: value iteration asks the model, Q-learning uses what it just observed. That is why Q-learning reaches "
    "the same answer. The two lines at the bottom are real lines from the trace in Lab 2. In episode 1 the agent, standing next to "
    "the near exit, happens to step onto it and sees +0.30: that entry jumps from 0 to 0.300. In episode 3 it is back in S, "
    "does the same move, lands on that cell and observes -0.02: the target is -0.02 + 0.95 x 0.300 = 0.265. The good news "
    "about the exit has travelled one cell back. About 3 minutes."))
for i, (h_, tag, c_, col, formula, src) in enumerate([
        ("Value iteration", "model", "blue_t", "blue", "V(s) ← max_{a} [ r(s,a) + γ V(F(s,a)) ]", "s′ and r come from the model F"),
        ("Q-learning", "data", "green_t", "green", "Q(s,a) ← r + γ max_{a′} Q(s′,a′)", "s′ and r come from what was observed")]):
    x = 0.6 + i * 6.2
    box(s, x, 1.6, 5.9, 2.35, fill=c_, radius=0.12, name="Card: " + h_)
    text(s, x + 0.3, 1.72, 3.5, 0.5, h_, size=22, bold=True, color=col)
    text(s, x + 4.2, 1.76, 1.4, 0.45, tag, size=16, bold=True, color=col, align="r")
    math(s, x + 0.2, 2.35, 5.5, 0.7, formula, size=22, align="c")
    text(s, x + 0.3, 3.2, 5.3, 0.6, src, size=16, color="muted", align="c")
text(s, 0.6, 4.25, 12.1, 0.45, "A real update from the trace of Lab 2  (γ = %.2f, α = 1)" % G, size=18, bold=True)
rows = [["Episode", "State, action", "Observed r, s′", "Target = r + γ max Q(s′, ·)", "Q(s, a) before → after"],
        ["1", "(3,1), ←", "+0.30, exit E", "+0.30  (E is terminal)", "0 → +0.300"],
        ["3", "(3,2), ←", "−0.02, (3,1)", "−0.02 + 0.95 × 0.300 = +0.265", "−0.020 → +0.265"]]
table(s, 0.6, 4.75, [1.2, 2.0, 2.3, 3.9, 2.7], rows, size=15, row_h=0.6, head_h=0.5, fonts={3: "Cambria"},
      aligns={0: "c", 4: "c"}, name="Worked updates")
text(s, 0.6, 6.5, 12.1, 0.4, "The news about the exit travels one cell back each time a neighbour is updated.", size=15, color="muted", italic=True)

# ---- 17 exploration
s = content_slide(prs, "The table only learns what the agent tries", "3", notes=(
    "Every update touches only the state-action pair that was tried. So what the agent tries decides what it can learn. Compare two "
    "agents on the same map and the same Q-learning rule. The greedy agent always does what the table says looks best. It stumbles "
    "onto the near exit in episode 1, the table now says that route is good, and it never tries anything else: it ends with V(S) = %s "
    "instead of %s, and not one of the %d values is right. The probing agent tries every action in every state in turn, whatever the "
    "table says: it finds the far goal in episode %d and all %d values match value iteration by episode %d. Lewis and Vrabie call "
    "this persistent excitation: the data must cover every state-action pair. There are smarter strategies, such as epsilon-greedy "
    "or UCB; they are beyond this hour. About 4 minutes." % (f3(ret_g), f3(ret_p), N_LIVE, reached_p["G"], N_LIVE, all_exact_ep)))
xs = list(range(1, EPISODES + 1))
xy_chart(s, 0.6, 1.55, 6.2, 3.9, [("probing", xs, exact_p, "green"), ("greedy", xs, exact_g, "slate")],
         xtitle="episode", ytitle="cells with the exact value (of %d)" % N_LIVE, ymin=0, ymax=N_LIVE, ymajor=3, xmin=0, xmax=EPISODES,
         xmajor=30, legend=True, description="Line chart: probing reaches all %d exact values; greedy stays at zero" % N_LIVE,
         name="Exact cells per episode")
for i, (lab, Qm, acts, path_, tag) in enumerate([
        ("Greedy: the best known action", Q_g, act_g, path_g, "ends at E · V(S) = %s" % f3(ret_g)),
        ("Probing: try every action in turn", Q_p, act_p, path_p, "ends at G · V(S) = %s" % f3(ret_p))]):
    x = 7.1 + i * 2.85
    text(s, x - 0.05, 1.55, 2.75, 0.7, lab, size=14, bold=True, align="c")
    draw_grid(s, env, x, 2.35, 0.45, values=Qm.max(1), actions=acts, path=path_, vmax=1.0, arrow_pt=12, value_pt=7,
              name="Learned " + lab, start_label=False)
    text(s, x - 0.05, 4.2, 2.75, 0.4, tag, size=13, color="blue", bold=True, align="c")
text(s, 7.1, 4.65, 5.6, 0.8, "Numbers: learned max Q. Arrows: greedy policy. Probing = the k-th visit to a state tries action k mod 4.",
     size=13, color="muted")
callout(s, 0.6, 5.7, 12.1, 1.15, "**Probing is persistent excitation:** the data must cover every state–action pair. "
        "Smarter strategies exist (ε-greedy, UCB, …): not needed today.", fill="amber_t", size=17, name="Probing")

# ---- 18 Lab 2
lab_slide("Lab 2: Q-learning without a model",
          ["The agent only calls env.reset() and env.step(a)", "Read the trace of the first episode: check one update by hand",
           "Compare the probing agent with a greedy one", "Check the learned table against Lab 1"],
          ["python lab2_q_learning.py", "python lab2_q_learning.py --behaviour greedy", "python lab2_q_learning.py --compare"],
          ["In which episode is G first reached? Why can the table not improve before?",
           "Greedy agent: which exit, and what is V(S) against the best?", "Try --alpha 0.5. What does a smaller step cost?",
           "Try --gamma 0.5. Does Q still agree with Lab 1?"],
          "Live demo, about 5 minutes. Run the default and read the trace of episode 1 together: each line is one observed transition with "
          "its target and TD error. Then show that all %d values end up equal to value iteration's. Run --behaviour greedy to show the lock-in "
          "on the near exit, or --compare to see both side by side. The learner never calls env.model: value iteration is imported only to "
          "check the answer." % N_LIVE,
          "You should see: probing reaches G in episode %d and all %d of %d values match value iteration, with V(S) = %s. Greedy ends at E "
          "with V(S) = %s." % (reached_p["G"], int(exact_p[-1]), N_LIVE, f3(ret_p), f3(ret_g)))

# ============================================================================ WRAP-UP
begin("Wrap-up")

# ---- 19 side by side
s = content_slide(prs, "Planning and learning, side by side", notes=(
    "Summarise the hour in one table. Both algorithms solve the same Bellman optimality equation and end with the same table. The "
    "difference is the source of information: a model that can be queried anywhere, or data gathered by acting. That is the "
    "difference Lewis and Vrabie draw between offline dynamic programming and online reinforcement learning. About 2 minutes."))
rows = [["", "Value iteration (Lab 1)", "Q-learning (Lab 2)"],
        ["Knows the model F, r?", "Yes: env.model(s, a)", "No: only env.step(a)"],
        ["Information comes from", "queries to the model, for every cell", "observed transitions (s, a, r, s′)"],
        ["Table", "V(s): %d numbers" % N_LIVE, "Q(s, a): %d numbers" % (N_LIVE * 4)],
        ["Update", "V ← max_{a} [ r + γ V(F(s,a)) ]", "Q ← Q + α [ r + γ max Q′ − Q ]"],
        ["Exploration", "not needed", "needed: probing"],
        ["Result", "V_{*} and π_{*}", "the same V_{*} and π_{*}  (%d of %d cells match)" % (int(exact_p[-1]), N_LIVE)]]
table(s, 0.6, 1.5, [2.9, 4.6, 4.6], rows, size=17, row_h=0.62, head_h=0.55, fonts={1: "Cambria", 2: "Cambria"},
      header_fills={0: "pine", 1: "blue", 2: "green"}, fills={(6, 1): "blue_t", (6, 2): "green_t"}, name="Value iteration versus Q-learning")
callout(s, 0.6, 6.15, 12.1, 0.7, "Same fixed point of the Bellman optimality equation, different source of information.",
        fill="mist", size=17, name="Summary")

# ---- 20 where tables stop
s = content_slide(prs, "Where tables stop", notes=(
    "A table is exact and transparent, but it needs one entry per state-action pair. Our gridworld has 72. Backgammon has about "
    "10 to the 20 states, a figure from Silver's lectures. And a continuous state, such as the position and velocity of a motor or a "
    "robot, cannot be tabulated at all. Lewis and Vrabie's answer is to replace the table by a parametric approximator for the value "
    "(the critic) and for the policy (the actor): Approximate or Adaptive Dynamic Programming. That is the natural next step, and "
    "outside this hour. About 2 minutes."))
for i, (big, h_, d_, c_, col) in enumerate([("%d" % (N_LIVE * 4), "Our gridworld", "%d cells × 4 actions: a table you can print" % N_LIVE, "green_t", "green"),
                                            ("10^{20}", "Backgammon", "states, according to Silver's lectures", "amber_t", "amber"),
                                            ("∞", "Continuous state", "x in R^{n}: a motor, a robot arm, a vehicle", "coral_t", "coral")]):
    x = 0.6 + i * 4.1
    box(s, x, 1.7, 3.9, 3.3, fill=c_, radius=0.12, name="Stat: " + h_)
    math(s, x, 1.85, 3.9, 1.5, big, size=60, color="ink" if i == 1 else col, align="c", bold=True)
    text(s, x + 0.25, 3.4, 3.4, 0.5, h_, size=21, bold=True, align="c")
    text(s, x + 0.25, 3.95, 3.4, 0.9, d_, size=15, align="c", color="muted")
box(s, 0.6, 5.3, 12.1, 1.5, fill="pine", radius=0.12, name="Next step")
text(s, 0.95, 5.3, 11.4, 1.5, "Next step: replace the table by a function approximator for the value (the critic) and the policy (the actor). "
     "That is Approximate or Adaptive Dynamic Programming, the subject of Lewis & Vrabie, and outside this hour.",
     size=18, color="paper", anchor="m")

# ---- 21 cheat sheet
s = content_slide(prs, "Cheat sheet", notes=(
    "A one-page reference for the hour. Ask the audience to explain each row in their own words. If they can, they have met the "
    "objectives. About 1 minute."))
rows = [["Concept", "In one line", "Key formula"],
        ["Reward, return", "what we want; the discounted total", "G = Σ γ^{k} r_{k}"],
        ["Policy", "how the agent acts: one action per state", "a = π(s)"],
        ["Optimal value", "the best return from s: Bellman's principle", "V_{*}(s) = max_{a} [ r + γ V_{*}(F(s,a)) ]"],
        ["Value iteration", "the equation as an update; needs the model", "V_{k+1}(s) = max_{a} [ r + γ V_{k}(F(s,a)) ]"],
        ["Q-function", "the look-ahead stored; gives the policy without a model", "Q(s,a) = r + γ V(F(s,a)),  π = argmax_{a} Q"],
        ["TD error", "prediction against observation", "e = r + γ max_{a′} Q(s′,a′) − Q(s,a)"],
        ["Q-learning", "value iteration from data", "Q(s,a) ← Q(s,a) + α e"],
        ["Exploration", "the data must cover every state–action pair", "probing"]]
table(s, 0.6, 1.5, [2.3, 4.9, 4.9], rows, size=15, row_h=0.56, head_h=0.5, fonts={2: "Cambria"}, name="Cheat sheet")

# ---- 22 questions and resources
s = content_slide(prs, "Check yourself, then go further", notes=(
    "Use the questions as a quick exit ticket or as homework. Suggested answers: 1 the near exit is closer, so it is discounted less; "
    "with a large gamma the far, bigger reward wins, with a small gamma the near one does. 2 The best value of a state is the best "
    "reward now plus the discounted best value of where you land. 3 Q already contains the look-ahead through the model, so the "
    "argmax over Q needs no model; V does not. 4 The prediction already agrees with the observation: the table is consistent here. "
    "5 It found the near exit first, the table then said that route was good, and it never tried other actions, so it never reached "
    "the far goal. The resources are where this course comes from."))
text(s, 0.6, 1.5, 6.4, 0.45, "Check your understanding", size=20, bold=True)
qs = ["Why does the near exit win at γ = 0.5 but lose at γ = 0.95?", "What does Bellman's optimality equation say, in one sentence?",
      "Why can you act with Q without a model, but not with V?", "What does a TD error of zero mean?",
      "Why did the greedy agent never find the far goal?"]
for i, q in enumerate(qs):
    y = 2.05 + i * 0.96
    box(s, 0.6, y, 6.4, 0.84, fill="mist", radius=0.1, name="Question %d" % (i + 1))
    badge(s, 0.75, y + 0.15, 0.55, str(i + 1), size=16)
    text(s, 1.5, y, 5.4, 0.84, q, size=15, anchor="m")
text(s, 7.4, 1.5, 5.3, 0.45, "Go further", size=20, bold=True)
res = [("Silver: course page and slides", "Lectures 1–5 are this hour in depth", "http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html"),
       ("Silver: lecture videos (YouTube)", "Reinforcement Learning, UCL / DeepMind", "https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLMZdRRhAoLnKFxZlmFoFp0uHVvN2PSE9T"),
       ("Sutton & Barto, 2nd edition", "The standard textbook, free online", "http://incompleteideas.net/book/the-book-2nd.html"),
       ("Lewis & Vrabie, IEEE CAS Magazine, 2009", "RL and ADP for feedback control", "https://doi.org/10.1109/MCAS.2009.933854")]
for i, (h_, d_, url) in enumerate(res):
    y = 2.05 + i * 1.2
    box(s, 7.4, y, 5.3, 1.05, fill="mist", radius=0.1, name="Resource %d" % (i + 1))
    tb = text(s, 7.6, y + 0.12, 4.9, 0.45, h_, size=15, bold=True)
    text(s, 7.6, y + 0.57, 4.9, 0.4, d_, size=13, color="muted")
    link_runs(tb, url)

# ============================================================================ SAVE
sections = []
for i, (name, first) in enumerate(SECTIONS):
    last = SECTIONS[i + 1][1] if i + 1 < len(SECTIONS) else len(prs.slides)
    sections.append((name, list(range(first, last))))
add_sections(prs, sections)
out = os.path.join(ROOT, "RL_in_One_Hour.pptx")
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
print("sections:", [(n, len(i)) for n, i in sections])
print(f"VI sweeps {len(changes)}  V(S) {V[S]:.4f}  gamma .5 -> E {V5[S]:.4f}  switch {switch_gamma}  all exact at episode {all_exact_ep}")
