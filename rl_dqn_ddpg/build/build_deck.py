"""Build RL_Bridge_DQN_DDPG.pptx: the algorithms of the four labs, with the figures the labs produced.

    cd rl_bridge_dqn_ddpg
    pip install python-pptx pillow numpy
    python build/build_deck.py

Run the labs first (they write the PNG figures in labs/outputs/ that the result slides show).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                           # .../rl_bridge_dqn_ddpg
OUTPUTS = os.path.join(ROOT, "labs", "outputs")
sys.path.insert(0, HERE)

import numpy as np
from PIL import Image

import deckkit
from deckkit import *

deckkit.COURSE = "From Q-tables to DDPG"
prs = new_presentation()
SECTIONS = []
CHIP = {"1": "green", "2": "blue", "3": "coral", "4": "amber"}


def begin(name):
    SECTIONS.append((name, len(prs.slides)))


# ============================================================================ SMALL PIECES
def table(slide, x, y, col_w, rows, size=14, row_h=0.5, head_h=0.5, aligns=None, fonts=None, fills=None,
          first_bold=True, name="Table"):
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
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[(aligns or {}).get(j, "l")]
            head = i == 0
            add_runs(p, str(val), size, "paper" if head else "ink", bold=head or (first_bold and j == 0),
                     font=(fonts or {}).get(j) if not head else None)
            c.fill.solid()
            setc(c.fill.fore_color, "pine" if head else (fills or {}).get((i, j), "mist" if i % 2 else "paper"))
    return gf


def tile(slide, x, y, w, h, title, body, fill="mist", title_pt=17, body_pt=14, bullets=False, space_after=4):
    box(slide, x, y, w, h, fill=fill, radius=0.1, name="Card: " + title)
    text(slide, x + 0.2, y + 0.15, w - 0.4, 0.45, title, size=title_pt, bold=True)
    text(slide, x + 0.2, y + 0.65, w - 0.4, h - 0.75, body, size=body_pt, bullets=bullets, space_after=space_after)


def callout(slide, x, y, w, h, content, fill="amber_t", size=15, name="Callout"):
    box(slide, x, y, w, h, fill=fill, radius=0.1, name=name)
    text(slide, x + 0.25, y + 0.08, w - 0.5, h - 0.16, content, size=size, anchor="m")


def picture(slide, filename, x, y, w, name="Figure"):
    path = os.path.join(OUTPUTS, filename)
    wpx, hpx = Image.open(path).size
    pic = slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    pic.name = name
    return pic, w * hpx / wpx


def slide(title, module=None, notes="", chip=None):
    return content_slide(prs, title, module=module, notes=notes, chip_color=chip or CHIP.get(module, "green"))


def section_slide(number, title, subtitle, colour, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[5])
    dark_background(s)
    t = s.shapes.title
    t.text_frame.text = title
    t.left, t.top, t.width, t.height = Inches(1.9), Inches(2.5), Inches(10.5), Inches(1.2)
    for r in t.text_frame.paragraphs[0].runs:
        r.font.size = Pt(44)
        setc(r.font.color, "paper")
    badge(s, 0.8, 2.6, 0.9, number, fill=colour, size=32, color="ink" if colour == "amber" else "paper")
    text(s, 1.9, 3.75, 10.2, 1.4, subtitle, size=20, color="mint")
    footer(s, dark=True)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def flow_box(s, x, y, w, h, label, fill="mist", line_c=None, size=15, color="ink", bold=True, font=None):
    b = box(s, x, y, w, h, fill=fill, line=line_c, radius=0.1)
    label_in(b, label, size=size, color=color, bold=bold, font=font)
    return b


# ============================================================================ DATA USED ON SLIDES
BINS = 20
rows_1d = BINS ** 2
growth = {"2 numbers (MountainCar)": BINS ** 2, "4 numbers (InvertedPendulum)": BINS ** 4}
rng = np.random.default_rng(0)
T_NOISE = 200
gauss = np.clip(0.5 * rng.standard_normal(T_NOISE), -1, 1)
x_ou, ou = 0.0, []
for _ in range(T_NOISE):
    x_ou = x_ou + 0.15 * (0.0 - x_ou) + 0.5 * rng.standard_normal()
    ou.append(x_ou)
ou = np.clip(np.array(ou), -1, 1)
cum_g, cum_o = np.cumsum(gauss), np.cumsum(ou)

# ============================================================================ 1. TITLE
begin("Introduction")
s = prs.slides.add_slide(prs.slide_layouts[5])
dark_background(s)
t = s.shapes.title
t.text_frame.text = "From Q-tables to DDPG"
t.left, t.top, t.width, t.height = Inches(0.8), Inches(1.9), Inches(11.5), Inches(1.4)
for r in t.text_frame.paragraphs[0].runs:
    r.font.size = Pt(54)
    setc(r.font.color, "paper")
text(s, 0.8, 3.4, 11.5, 1.0, "The algorithms behind four short labs: tabular Q-learning, DQN, DDPG and the exploration problem",
     size=22, color="mint")
text(s, 0.8, 5.0, 11.5, 1.2, ["Model-free, off-policy, value-based learning on MuJoCo and classic-control tasks",
                              "Every algorithm is written by hand in Python · PyTorch only provides networks and autograd"],
     size=15, color="code_dim", space_after=6)
for i, (num, col) in enumerate([("1", "green"), ("2", "blue"), ("3", "coral"), ("4", "amber")]):
    badge(s, 0.8 + i * 0.85, 6.15, 0.65, num, fill=col, size=20, color="ink" if col == "amber" else "paper")
s.notes_slide.notes_text_frame.text = (
    "This deck explains the algorithms used in the four labs of the bridge module. It sits between the 1-hour tabular course "
    "and the PPO robot-arm exercise. The labs are short: each runs on a laptop CPU in a few seconds to two minutes.")

# ============================================================================ 2. WHERE IT FITS
s = slide("Where this module sits",
          notes="Walk the chain left to right. The one-hour course ends with a table and a small numpy network on a 24-cell world. "
                "The robot-arm PPO exercise uses a library. These four labs fill the gap, changing one thing at a time.")
flow_box(s, 0.6, 1.55, 2.5, 1.0, "rl_course_1h\ntabular Q, gridworld", fill="slate_t", size=14)
for k, (n, lab, col) in enumerate([("1", "Tabular Q on\nboxes", "green"), ("2", "DQN", "blue"),
                                   ("3", "DDPG", "coral"), ("4", "Exploration", "amber")]):
    x = 3.55 + k * 1.92
    flow_box(s, x, 1.55, 1.7, 1.0, f"Lab {n}\n{lab}", fill=col, size=14, color="ink" if col == "amber" else "paper")
    line(s, x - 0.43, 2.05, x - 0.05, 2.05, color="slate", arrow=True)
flow_box(s, 11.2, 1.55, 1.55, 1.0, "SO101\nPPO", fill="slate_t", size=14)
line(s, 11.2 - 0.43 + 0.0, 2.05, 11.15, 2.05, color="slate", arrow=True)
rows = [["Lab", "Task", "What changes", "Algorithm"],
        ["1", "MountainCar-v0, 3 actions", "state = 2 real numbers; boxes make table rows", "tabular Q-learning, ε-greedy"],
        ["2", "MountainCar-v0, 3 actions", "the table becomes a network", "DQN: replay buffer + target network"],
        ["3", "InvertedPendulum-v5, 1 real action", "argmax over actions cannot be enumerated", "DDPG (+ TD3 repairs)"],
        ["4", "MountainCarContinuous-v0", "only the environment: sparse reward", "DDPG + correlated (OU) noise"]]
table(s, 0.6, 3.0, [0.8, 3.5, 4.6, 3.25], rows, size=14, row_h=0.62, head_h=0.5, aligns={0: "c"}, name="Labs table")
callout(s, 0.6, 6.0, 12.15, 0.8,
        "**Model-free** (no equations of the world), **off-policy** (learns from old data in a replay buffer), **value-based** "
        "(learns Q, and an actor that follows it).", size=15, fill="green_t")

# ============================================================================ 3. COMMON INGREDIENTS
s = slide("Shared ingredients",
          notes="These are the pieces every lab uses. The agent only ever sees transitions. The terminated/truncated distinction is easy "
                "to get wrong and is the subject of a Lab 3 exercise.")
text(s, 0.6, 1.5, 6.0, 0.4, "The loop every algorithm lives in", size=19, bold=True)
flow_box(s, 0.7, 2.2, 2.2, 1.1, "Agent", fill="green", size=18, color="paper")
flow_box(s, 4.3, 2.2, 2.2, 1.1, "Environment", fill="pine", size=18, color="paper")
line(s, 2.9, 2.5, 4.3, 2.5, color="ink", arrow=True, w=2.5)
text(s, 2.9, 2.1, 1.4, 0.35, "action a", size=13, align="c", color="muted")
line(s, 4.3, 3.0, 2.9, 3.0, color="ink", arrow=True, w=2.5)
text(s, 2.7, 3.07, 1.8, 0.55, ["next state s′", "reward r"], size=13, align="c", color="muted")
math(s, 0.7, 3.9, 5.8, 0.6, "(s, a, r, s′, terminated)", size=24, align="c")
text(s, 0.7, 4.5, 5.8, 1.0,
     "Everything an algorithm learns from is a stream of these tuples. No model s′ = F(s, a) is ever used.",
     size=15, color="muted", align="c")
text(s, 7.0, 1.5, 5.7, 0.4, "Four things to keep straight", size=19, bold=True)
items = [("Return and discount", "G = r₀ + γ r₁ + γ² r₂ + …   with γ = 0.99, so reward 100 steps away is worth 0.37 of its face value."),
         ("Q-function", "Q(s, a): the return from doing a in s and then acting well. Acting needs only argmax_a Q(s, a)."),
         ("TD target", "y = r + γ (1 − terminated) · max Q(s′, ·). Bootstrapping: the estimate is built from another estimate."),
         ("terminated ≠ truncated", "Only a true end of the task (flag reached, pole fell) cuts the bootstrap. A 200-step time-out does not.")]
for i, (h, b) in enumerate(items):
    y = 2.0 + i * 1.18
    box(s, 7.0, y, 5.75, 1.05, fill="mist", radius=0.1, name="Item " + h)
    text(s, 7.2, y + 0.07, 5.35, 0.35, h, size=15, bold=True)
    text(s, 7.2, y + 0.42, 5.35, 0.62, b, size=12.5, color="ink")

# ============================================================================ LAB 1
begin("Lab 1")
section_slide("1", "Tabular Q-learning on boxes", "The 1-hour course's update, unchanged, on a state of two real numbers.", "green",
              notes="Lab 1 is the bridge from the previous course. The only new idea is how a real-valued state becomes a table row.")

s = slide("Q-learning with a discretised state", module="1",
          notes="Same update as the gridworld Q-learning. Alpha is below one because many different states share a box, so from the table's "
                "point of view the world looks noisy. Epsilon-greedy replaces the probing rule of the earlier course.")
text(s, 0.6, 1.5, 6.1, 0.4, "Step 1: cut the plane into boxes", size=19, bold=True)
math(s, 0.6, 1.95, 6.1, 0.55, "box(s) = ⌊ (s − low) / width ⌋  in each coordinate", size=19)
text(s, 0.6, 2.7, 6.1, 0.4, "Step 2: the same update as before", size=19, bold=True)
math(s, 0.6, 3.15, 6.1, 0.55, "y = r + γ (1 − terminated) max_{a′} Q[box(s′), a′]", size=19)
math(s, 0.6, 3.75, 6.1, 0.55, "Q[box(s), a] ← Q[box(s), a] + α (y − Q[box(s), a])", size=19)
text(s, 0.6, 4.55, 6.1, 0.4, "Step 3: explore with ε-greedy", size=19, bold=True)
text(s, 0.6, 5.0, 6.1, 1.8,
     ["with probability ε: a random action", "otherwise: argmax_a Q[box(s), a]",
      "ε falls linearly from 1 to 0.01 over the first 60% of 10,000 episodes"], size=15, bullets=True, space_after=4)
code_block(s, 7.1, 1.55, 5.65, 2.15,
           ["target = r + gamma * (0.0 if terminated", "                     else Q[b2].max())",
            "td_error = target - Q[b + (a,)]", "Q[b + (a,)] += alpha * td_error"], size=14, name="Update code")
text(s, 7.1, 3.85, 5.65, 0.4, "Settings in the lab", size=17, bold=True)
table(s, 7.1, 4.3, [2.6, 3.05],
      [["Setting", "Value"], ["boxes per variable", "20  (400 rows × 3 actions)"], ["step size α", "0.1"],
       ["discount γ", "0.99"], ["table starts at", "0 (optimistic: all rewards are −1)"]],
      size=13, row_h=0.42, head_h=0.42, name="Lab 1 settings")

s = slide("Lab 1 result: it works, and it does not scale", module="1",
          notes="Read the figure left to right: learning curve, value map with the greedy trajectory spiralling out, and the greedy action map. "
                "The table starts at zero and every reward is negative, so tried actions look worse than untried ones. That optimism does the exploring.")
pic, ph = picture(s, "lab1_tabular_mountaincar.png", 0.6, 1.45, 12.1, name="Lab 1 figure")
y0 = 1.45 + ph + 0.2
tile(s, 0.6, y0, 3.9, 6.85 - y0, "Boxes matter",
     ["20×20 boxes: 100% of 20 starts reach the flag, mean return −157", "8×8: 0%.  40×40: 25%"], body_pt=13, bullets=True)
tile(s, 4.7, y0, 3.9, 6.85 - y0, "Optimism explores",
     ["Zeros under a −1 reward make tried actions look worse than untried ones", "A table starting at −150 never finds the flag"],
     body_pt=13, bullets=True, fill="green_t")
tile(s, 8.8, y0, 3.95, 6.85 - y0, "The table grows as N^d",
     [f"2 numbers: {BINS**2:,} rows", f"4 numbers: {BINS**4:,} rows", f"Ant, 105 numbers: {float(BINS)**105:.1e}"],
     body_pt=13, bullets=True, fill="amber_t")

# ============================================================================ LAB 2
begin("Lab 2")
section_slide("2", "DQN: replace the table by a network", "Same car, same actions, same reward. Only the place where Q lives changes.", "blue",
              notes="Lab 2 changes exactly one thing: Q is a network. Everything hard about DQN is about making that change stable.")

s = slide("Q as a function: the semi-gradient update", module="2",
          notes="The table write becomes a gradient step on the squared TD error. Stress that y is treated as a constant: no gradient flows into it. "
                "That is the semi-gradient. In the code y is computed under torch.no_grad().")
text(s, 0.6, 1.5, 6.3, 0.4, "The network", size=19, bold=True)
xs = [0.7, 2.55, 4.4, 6.0]
labels = [("state\n(pos, vel)\n2 inputs", "slate_t"), ("hidden\n128 ReLU", "blue_t"), ("hidden\n128 ReLU", "blue_t"),
          ("Q(s,·)\n3 outputs", "green_t")]
widths = [1.5, 1.5, 1.5, 1.2]
for x, (lab, fill), w in zip(xs, labels, widths):
    flow_box(s, x, 2.05, w, 1.25, lab, fill=fill, size=13, bold=False)
for a, b, wa in zip(xs[:-1], xs[1:], widths[:-1]):
    line(s, a + wa, 2.67, b, 2.67, color="ink", arrow=True, w=2)
text(s, 0.7, 3.5, 6.3, 0.7,
     "17,283 weights, against the 1,200 entries of the Lab 1 table. Input scaled to [−1, 1]: position is ~1 and velocity ~0.05.",
     size=14, color="muted")
text(s, 0.6, 4.4, 6.3, 0.4, "Why a plain regression is not enough", size=19, bold=True)
text(s, 0.6, 4.85, 6.3, 2.0,
     ["The target contains the thing being learned", "Consecutive samples are almost the same point", "Updating one state moves its neighbours"],
     size=15, bullets=True, space_after=5)
box(s, 7.3, 1.55, 5.45, 5.2, fill="mist", radius=0.1, name="DQN equations")
text(s, 7.55, 1.7, 5.0, 0.4, "One update", size=19, bold=True)
math(s, 7.55, 2.2, 5.0, 0.6, "y = r + γ (1 − d) max_{a′} Q_{θ⁻}(s′, a′)", size=20)
math(s, 7.55, 2.95, 5.0, 0.6, "L(θ) = mean ( y − Q_θ(s, a) )²", size=20)
math(s, 7.55, 3.7, 5.0, 0.6, "θ ← θ − lr · ∇_θ L(θ)", size=20)
text(s, 7.55, 4.5, 5.0, 2.2,
     ["Q_θ(s, a) is the output for the action actually taken", "y is a constant: no gradient flows through it",
      "Adam supplies the step; lr = 10⁻³", "d = terminated, never truncated"],
     size=14, bullets=True, space_after=4)

s = slide("Two stabilisers: replay buffer and target network", module="2",
          notes="Left: the ring buffer. When it is full, the newest transition overwrites the oldest. A minibatch is 64 random transitions. "
                "Right: the online and target networks. The target is a frozen copy refreshed every 500 steps. Ablation flags in the lab turn each off.")
text(s, 0.6, 1.5, 6.0, 0.4, "Replay buffer (ring buffer)", size=19, bold=True)
n_cells = 12
for i in range(n_cells):
    fill = "blue_t" if i not in (5, 6) else "amber_l"
    c = box(s, 0.65 + i * 0.47, 2.15, 0.42, 0.55, fill=fill, radius=0, line="paper", name="Buffer slot")
text(s, 0.65, 2.75, 5.7, 0.35, "slots hold (s, a, r, s′, d); new data overwrites the oldest", size=12, color="muted")
line(s, 0.65 + 5 * 0.47 + 0.2, 3.5, 0.65 + 5 * 0.47 + 0.2, 2.75, color="coral", arrow=True, w=2)
text(s, 2.2, 3.5, 3.5, 0.35, "write pointer", size=12, color="coral")
text(s, 0.6, 4.05, 6.0, 2.8,
     ["Training on consecutive steps is fitting a curve to one tiny corner of the plane, then forgetting it",
      "Instead: store every transition, train on 64 random old ones",
      "Capacity 50,000; learning starts after 1,000 transitions",
      "Remove it with --no-replay: the car never learns (return −200 throughout)"],
     size=14.5, bullets=True, space_after=5)
text(s, 7.0, 1.5, 5.7, 0.4, "Target network", size=19, bold=True)
flow_box(s, 7.0, 2.1, 2.5, 1.0, "Online Q_θ\ntrained by gradient", fill="blue", size=14, color="paper")
flow_box(s, 10.2, 2.1, 2.5, 1.0, "Target Q_θ⁻\nfrozen copy", fill="slate_t", size=14)
line(s, 9.5, 2.45, 10.2, 2.45, color="ink", arrow=True, w=2)
text(s, 9.45, 2.0, 0.9, 0.4, "copy", size=12, align="c", color="muted")
text(s, 7.0, 3.2, 5.7, 0.4, "every 500 steps:  θ⁻ ← θ", size=15, align="c", color="muted", font="Cambria")
line(s, 11.45, 3.1, 11.45, 3.75, color="ink", arrow=True, w=2)
flow_box(s, 8.2, 3.75, 4.5, 0.75, "y = r + γ (1 − d) max Q_{θ⁻}(s′, ·)", fill="green_t", size=14, font="Cambria", bold=False)
text(s, 7.0, 4.75, 5.7, 2.1,
     ["Without it, every update moves the goalpost the network chases",
      "On this small problem the effect is milder than for the buffer: --no-target still learns, with a more erratic evaluation curve"],
     size=14.5, bullets=True, space_after=5)

s = slide("The DQN algorithm", module="2",
          notes="This is the whole training loop of lab2_dqn_mountaincar.py. Note the order: act, store, learn, refresh the target. Evaluation uses a separate environment.")
code_block(s, 0.6, 1.5, 7.2, 5.3,
           ["for step in 1 .. 100,000:",
            "  ε = max(0.05, 1 - 0.95 * step / 50,000)   # linear decay",
            "  a = random  if rand() < ε  else argmax Q(s, ·)",
            "  s2, r, terminated, truncated = env.step(a)",
            "  buffer.add(s, a, r, s2, terminated)",
            "  if step >= 1,000:",
            "    s, a, r, s2, d = buffer.sample(64)",
            "    y = r + γ (1 - d) max Q⁻(s2, ·)      # no grad",
            "    loss = mean( (y - Q(s, a))² )",
            "    Adam step on θ",
            "    every 500 steps:  Q⁻ ← Q",
            "  if terminated or truncated: reset env"], size=14, name="DQN pseudo-code")
text(s, 8.2, 1.5, 4.5, 0.4, "Hyper-parameters", size=19, bold=True)
table(s, 8.2, 2.0, [2.3, 2.25],
      [["Setting", "Value"], ["network", "2-128-128-3"], ["learning rate", "10⁻³ (Adam)"], ["discount γ", "0.99"],
       ["minibatch", "64"], ["buffer", "50,000"], ["target refresh", "500 steps"], ["steps", "100,000"]],
      size=13, row_h=0.45, head_h=0.45, name="DQN settings")
text(s, 8.2, 5.8, 4.5, 1.0, "About 30 seconds on a laptop CPU.", size=14, color="muted")

s = slide("Lab 2 result: what each stabiliser is worth", module="2",
          notes="Left panel: greedy evaluation of the three runs with the same seed. Without replay the car never learns. Without the target network it learns but is "
                "less steady. The middle panels are the network's value and policy maps; the right panel is the Lab 1 table for comparison. "
                "Training episodes start at random places on the hill because a car that starts at the bottom and pushes at random never sees the flag.")
pic, ph = picture(s, "lab2_dqn_compare.png", 0.6, 1.45, 12.1, name="Lab 2 comparison figure")
y0 = 1.45 + ph + 0.25
tile(s, 0.6, y0, 3.9, 6.85 - y0, "DQN vs the table",
     ["20 of 20 starts reach the flag, mean return −113", "Lab 1 table: −157, with 1,200 entries"], body_pt=13, bullets=True, fill="green_t")
tile(s, 4.7, y0, 3.9, 6.85 - y0, "The ablations",
     ["--no-replay: −200 at every evaluation", "--no-target: learns, but a noisier curve"], body_pt=13, bullets=True)
tile(s, 8.8, y0, 3.95, 6.85 - y0, "Exploration is still needed",
     ["Training starts at random positions on the hill", "--no-random-starts: no flag transition in 100,000 steps, return −200"],
     body_pt=13, bullets=True, fill="amber_t")

# ============================================================================ LAB 3
begin("Lab 3")
section_slide("3", "DDPG: a real-valued action", "A second network answers the question “which action is best?”", "coral",
              notes="The step from DQN to DDPG is forced by a single fact: you cannot loop over a continuum of actions.")

s = slide("Why DQN breaks, and the actor-critic answer", module="3",
          notes="DQN needs a max over actions twice: in the target and to act. With one real action that is an optimisation problem at every step. "
                "DDPG learns a network pi that outputs the maximiser directly. The critic is a network over (s, a), so dQ/da exists and can train the actor.")
text(s, 0.6, 1.5, 5.9, 0.4, "The problem", size=19, bold=True)
text(s, 0.6, 1.95, 5.9, 1.5,
     ["InvertedPendulum: the action is a force in [−3, 3]", "DQN needs max_a Q(s, a) for the target and for acting",
      "With 3 actions: a loop.  With a real action: an optimisation at every step"],
     size=15, bullets=True, space_after=5)
text(s, 0.6, 3.75, 5.9, 0.4, "The answer: learn the maximiser", size=19, bold=True)
math(s, 0.6, 4.2, 5.9, 0.55, "π_θ(s) ≈ argmax_a Q_φ(s, a)", size=22)
text(s, 0.6, 4.95, 5.9, 1.9,
     ["Q_φ(s, a) is a network, so ∂Q/∂a exists", "The chain rule carries it back to the actor's weights",
      "In the 1-hour course the actor copied the critic's greedy action; here it climbs the critic's gradient"],
     size=14.5, bullets=True, space_after=4)
flow_box(s, 7.2, 1.75, 1.7, 1.0, "state s", fill="slate_t", size=15)
flow_box(s, 9.3, 1.75, 1.7, 1.0, "Actor π_θ", fill="coral", size=15, color="paper")
flow_box(s, 11.15, 1.75, 1.6, 1.0, "action a", fill="slate_t", size=15)
line(s, 8.9, 2.25, 9.3, 2.25, color="ink", arrow=True, w=2)
line(s, 11.0, 2.25, 11.15, 2.25, color="ink", arrow=True, w=2)
flow_box(s, 9.3, 3.55, 3.45, 1.0, "Critic Q_φ(s, a)", fill="blue", size=16, color="paper")
line(s, 11.95, 2.75, 11.95, 3.55, color="ink", arrow=True, w=2)
line(s, 8.05, 2.75, 8.05, 4.05, color="ink", w=2)
line(s, 8.05, 4.05, 9.3, 4.05, color="ink", arrow=True, w=2)
text(s, 9.3, 4.75, 3.45, 0.4, "output: one number", size=13, align="c", color="muted")
callout(s, 7.2, 5.45, 5.55, 1.3, "**Off-policy and model-free.** DDPG learns from a replay buffer of older transitions and never uses the equations of the pendulum.",
        size=14.5, fill="green_t")

s = slide("The two updates", module="3",
          notes="The critic update is DQN's, with the max over actions replaced by the target actor's action. The actor update is the new idea: the loss is minus Q, "
                "so gradient descent on the actor is gradient ascent on Q. The gradient also reaches the critic weights but only the actor optimiser steps in that phase.")
box(s, 0.6, 1.55, 6.05, 5.25, fill="blue_t", radius=0.1, name="Critic card")
text(s, 0.85, 1.7, 5.6, 0.4, "Critic: regress onto the target", size=19, bold=True)
math(s, 0.85, 2.3, 5.6, 0.6, "y = r + γ (1 − d) Q_{φ⁻}(s′, π_{θ⁻}(s′))", size=19)
math(s, 0.85, 3.1, 5.6, 0.6, "L(φ) = mean ( y − Q_φ(s, a) )²", size=19)
text(s, 0.85, 3.95, 5.6, 2.8,
     ["The same as DQN, with π⁻(s′) in place of the max", "Both target networks are used: π⁻ chooses a′, Q⁻ scores it",
      "y is a constant: computed under no_grad"], size=14.5, bullets=True, space_after=5)
box(s, 6.9, 1.55, 5.85, 5.25, fill="coral_t", radius=0.1, name="Actor card")
text(s, 7.15, 1.7, 5.4, 0.4, "Actor: climb the critic", size=19, bold=True)
math(s, 7.15, 2.3, 5.4, 0.6, "L(θ) = − mean Q_φ(s, π_θ(s))", size=19)
math(s, 7.15, 3.1, 5.4, 0.6, "∇_θ Q_φ(s, π_θ(s)) = ∂Q/∂a · ∂π/∂θ", size=19)
text(s, 7.15, 3.95, 5.4, 2.8,
     ["The loss is minus Q: descent on the loss is ascent on Q", "Positive ∂Q/∂a at π(s) pushes the force up; negative, down",
      "Only the actor's optimiser steps in this phase; the critic is not changed by it"], size=14.5, bullets=True, space_after=5)

s = slide("What keeps DDPG stable and exploring", module="3",
          notes="Four details that matter. Polyak averaging is a smooth version of DQN's periodic copy. The actor is deterministic, so exploration noise is added by hand. "
                "The warm-up fills the buffer with varied data. The terminated flag is the only one stored.")
tile(s, 0.6, 1.55, 6.0, 2.55, "Polyak-averaged targets",
     ["θ⁻ ← (1 − τ) θ⁻ + τ θ, with τ = 0.005", "Both actor and critic have a target copy",
      "τ = 1 is DQN's hard copy; try --tau 1.0"], bullets=True, fill="blue_t")
tile(s, 6.75, 1.55, 6.0, 2.55, "Exploration noise",
     ["a = clip(π(s) + noise, −3, 3)", "Gaussian: N(0, (0.1 · 3)²), a new draw each step",
      "Ornstein-Uhlenbeck with --noise ou (Lab 4)"], bullets=True, fill="coral_t")
tile(s, 0.6, 4.3, 6.0, 2.5, "Warm-up",
     ["First 1,000 steps: uniform random actions, no learning", "Gives the critic data from outside the actor's habits"],
     bullets=True, fill="amber_t")
tile(s, 6.75, 4.3, 6.0, 2.5, "terminated vs truncated",
     ["The buffer stores terminated only", "The pole is still up at step 1,000: that is a time-out, so the target keeps bootstrapping"],
     bullets=True, fill="green_t")

s = slide("The DDPG algorithm", module="3",
          notes="This is train() in lab3_ddpg_inverted_pendulum.py with the agent's update inlined. One gradient step per environment step after the warm-up.")
code_block(s, 0.6, 1.5, 7.5, 5.3,
           ["for step in 1 .. 30,000:",
            "  a = uniform()  if step <= 1,000",
            "      else clip(π(s) + noise)",
            "  s2, r, terminated, truncated = env.step(a)",
            "  buffer.add(s, a, r, s2, terminated)",
            "  if step > 1,000:",
            "    s, a, r, s2, d = buffer.sample(256)",
            "    y = r + γ (1 - d) Q⁻(s2, π⁻(s2))     # no grad",
            "    critic step: minimise (y - Q(s, a))²",
            "    actor step:  minimise -Q(s, π(s))",
            "    π⁻ ← (1-τ) π⁻ + τ π;  Q⁻ likewise",
            "  if terminated or truncated:",
            "      reset env and noise"], size=13.5, name="DDPG pseudo-code")
text(s, 8.45, 1.5, 4.3, 0.4, "Hyper-parameters", size=19, bold=True)
table(s, 8.45, 2.0, [2.2, 2.1],
      [["Setting", "Value"], ["actor / critic", "256-256 each"], ["learning rates", "10⁻³ both"], ["discount γ", "0.99"],
       ["τ", "0.005"], ["minibatch", "256"], ["buffer", "100,000"], ["noise σ", "0.1 × bound"]],
      size=13, row_h=0.45, head_h=0.45, name="DDPG settings")
text(s, 8.45, 5.8, 4.3, 1.0, "About 40 seconds for 30,000 steps on a laptop CPU.", size=14, color="muted")

s = slide("TD3: three one-line repairs", module="3",
          notes="DDPG's usual disease is overestimation: the actor seeks out actions where the critic is too optimistic. TD3 fixes it with three small changes, each a flag in the lab. "
                "In the seed experiment, plain DDPG fails on one seed in five and TD3 on none.")
tile(s, 0.6, 1.55, 4.0, 3.5, "--twin: two critics",
     ["Train Q₁ and Q₂", "Target uses the smaller one"], body_pt=14, bullets=True, fill="blue_t")
math(s, 0.8, 3.35, 3.6, 0.9, "y = r + γ (1 − d) min(Q⁻₁, Q⁻₂)(s′, a′)", size=15, anchor="t")
tile(s, 4.75, 1.55, 4.0, 3.5, "--delay: slower actor",
     ["Update actor and targets only every 2nd critic update", "The critic settles before the actor follows it"],
     body_pt=14, bullets=True, fill="coral_t")
tile(s, 8.9, 1.55, 3.85, 3.5, "--smooth: smoothed target",
     ["Add clipped noise to the target action", "The target no longer peaks on a single sharp action"],
     body_pt=14, bullets=True, fill="amber_t")
math(s, 9.1, 3.65, 3.5, 0.9, "a′ = clip(π⁻(s′) + clip(ε, ±0.5 · bound))", size=14, anchor="t")
callout(s, 0.6, 5.3, 12.15, 1.5,
        ["**--td3** switches all three on. On InvertedPendulum, 5 seeds each, 30,000 steps:", "plain DDPG: 4 of 5 seeds end at return 1000     ·     TD3: 5 of 5"],
        size=17, fill="green_t")

s = slide("Lab 3 result: the pendulum stays up", module="3",
          notes="Four panels. Learning curve; the critic's mean Q against its largest possible value 1/(1-gamma) = 100; Q over the force at three pole angles, with the actor's choice marked; "
                "and the actor as a map. Counting seeds is the real lesson: one learning curve is one sample.")
pic, ph = picture(s, "lab3_ddpg_inverted_pendulum.png", 0.6, 1.45, 12.1, name="Lab 3 figure")
y0 = 1.45 + ph + 0.25
tile(s, 0.6, y0, 3.9, 6.85 - y0, "Seed 0",
     ["Return 1000 (pole never falls) from step 14,000", "20 of 20 greedy episodes at 1000"], body_pt=13, bullets=True, fill="green_t")
tile(s, 4.7, y0, 3.9, 6.85 - y0, "Across seeds",
     ["DDPG: 4 of 5 seeds reach 1000", "TD3: 5 of 5", "With OU noise: 2 of 3"], body_pt=13, bullets=True)
tile(s, 8.8, y0, 3.95, 6.85 - y0, "Reading the critic",
     ["Mean Q climbs toward the ceiling 1/(1 − γ) = 100", "The actor is close to bang-bang: ±3 over most of the map"],
     body_pt=13, bullets=True, fill="amber_t")

# ============================================================================ LAB 4
begin("Lab 4")
section_slide("4", "Exploration: a harder world", "The DDPG of Lab 3, imported unchanged, learns to stand still.", "amber",
              notes="Lab 4 changes no algorithm. It changes the environment and the data the agent collects.")

s = slide("Why DDPG learns to stand still", module="4",
          notes="Reward is a small penalty for pushing and a large prize at the flag that the car cannot reach without first rocking. Until the car has been to the flag once, "
                "the critic sees only penalties. This is a stable closed loop, not bad luck. It is the continuous-action version of the greedy agent in the first course.")
text(s, 0.6, 1.5, 5.6, 0.4, "The reward", size=19, bold=True)
math(s, 0.6, 1.95, 5.6, 0.6, "r = −0.1 a²   each step", size=21)
math(s, 0.6, 2.55, 5.6, 0.6, "r = +100       once, at the flag", size=21)
text(s, 0.6, 3.4, 5.6, 3.4,
     ["The car cannot drive straight up: it must rock back and forth", "Until it reaches the flag once, the critic has seen only penalties",
      "The best it can learn from penalties: pushing is costly", "So the actor learns to output force 0"],
     size=15, bullets=True, space_after=6)
steps = ["Buffer holds only penalties", "Critic: pushing costs", "Actor outputs force ≈ 0", "Noise ±0.1 too small to climb"]
cols = ["slate_t", "blue_t", "coral_t", "amber_t"]
for i, (lab, col) in enumerate(zip(steps, cols)):
    flow_box(s, 6.8, 1.6 + i * 1.12, 3.3, 0.85, lab, fill=col, size=14)
    if i < 3:
        line(s, 8.45, 2.45 + i * 1.12, 8.45, 2.72 + i * 1.12, color="ink", arrow=True, w=2)
line(s, 10.1, 4.97, 10.9, 4.97, color="coral", w=2.5)
line(s, 10.9, 4.97, 10.9, 2.03, color="coral", w=2.5)
line(s, 10.9, 2.03, 10.1, 2.03, color="coral", arrow=True, w=2.5)
text(s, 11.0, 3.2, 1.8, 1.2, ["no flag,", "no +100,", "and round again"], size=13, color="coral")
callout(s, 6.8, 6.1, 5.95, 0.75, "A stable fixed point of the learning loop, not bad luck.", size=14.5, fill="coral_t")

s = slide("The cure is in the data: correlated noise", module="4",
          notes="The chart shows the running sum of the exploration push over 200 steps, with the same sigma. Independent Gaussian draws cancel; the sum stays within a few units. "
                "OU noise has memory, so it keeps pushing one way and the sum drifts far. A car needs sustained pushes in one direction to build speed.")
text(s, 0.6, 1.5, 5.4, 0.4, "Gaussian: independent each step", size=18, bold=True)
math(s, 0.6, 1.95, 5.4, 0.55, "x_t = σ · N(0, 1)", size=20)
text(s, 0.6, 2.7, 5.4, 0.4, "Ornstein-Uhlenbeck: with memory", size=18, bold=True)
math(s, 0.6, 3.15, 5.4, 0.55, "x ← x + θ (0 − x) + σ N(0, 1)", size=20)
text(s, 0.6, 3.95, 5.4, 2.9,
     ["Small θ = long memory; θ = 1 erases the memory and gives back white noise", "OU hands the car a run of pushes in one direction",
      "One episode that reaches the flag puts +100 in the buffer; then the critic and the actor can learn from it",
      "Preset: σ = 0.5, θ = 0.15, reset every episode"],
     size=14, bullets=True, space_after=5)
xs_t = list(range(1, T_NOISE + 1))
xy_chart(s, 6.3, 1.5, 6.5, 5.35,
         [("Gaussian σ=0.5 (independent)", xs_t, cum_g, "coral"), ("OU σ=0.5, θ=0.15 (memory)", xs_t, cum_o, "green")],
         xtitle="step", ytitle="running sum of the push (clipped to ±1)", description="Running sum of two exploration noise sequences",
         name="Noise chart")

s = slide("Lab 4 result: same agent, different data", module="4",
          notes="Left: greedy evaluation. The naive run is exactly zero throughout. Middle: the states visited in training. The naive run never leaves a small blob near the start; the fixed run covers the plane out to the flag. "
                "Right: the actor of the fixed run as a map of force, with one greedy drive. The table lists variants that were run.")
pic, ph = picture(s, "lab4_ddpg_mountaincar_continuous.png", 0.6, 1.45, 12.1, name="Lab 4 figure")
y0 = 1.45 + ph + 0.15
rows = [["Exploration", "Training episodes reaching the flag", "Greedy return"],
        ["Gaussian σ=0.1 (Lab 3 settings)", "0 of 30", "0 (stands still)"],
        ["Gaussian σ=0.5", "none", "0"],
        ["OU σ=0.5, θ=1.0 (no memory)", "none", "0"],
        ["OU σ=0.5, θ=0.15", "170 of 171", "≈ 89 to 93"]]
table(s, 0.6, y0, [4.6, 4.5, 3.0], rows, size=13, row_h=0.4, head_h=0.4, aligns={1: "c", 2: "c"},
      fills={(4, 0): "green_t", (4, 1): "green_t", (4, 2): "green_t"}, name="Variants table")
text(s, 0.6, y0 + 2.05, 12.1, 0.45,
     "What matters is that the noise is correlated in time, not how large it is. The ≈ 90 (not 100) is the energy cost −0.1 a² of a near full-force policy.",
     size=13, color="muted")

# ============================================================================ WRAP UP
begin("Wrap-up")
s = slide("Four algorithms side by side",
          notes="Use this slide to fix the vocabulary. Each column adds machinery only because the previous one broke.")
rows = [["", "Tabular Q (Lab 1)", "DQN (Lab 2)", "DDPG (Lab 3)", "DDPG + OU (Lab 4)"],
        ["Q lives in", "table, 1,200 entries", "network, 17,283 weights", "critic network", "critic network"],
        ["Actions", "3, discrete", "3, discrete", "1 real number", "1 real number"],
        ["How it acts", "argmax over table row", "argmax over 3 outputs", "actor π_θ(s)", "actor π_θ(s)"],
        ["Exploration", "ε-greedy", "ε-greedy", "Gaussian noise + warm-up", "correlated OU noise"],
        ["Stabilisers", "none needed", "replay buffer, target network", "replay, Polyak targets (TD3: twin, delay, smooth)", "same, unchanged"],
        ["Family", "model-free, off-policy, value-based", "same", "same, actor-critic", "same"]]
table(s, 0.6, 1.55, [1.8, 2.5, 2.6, 3.05, 2.2], rows, size=13, row_h=0.74, head_h=0.5, name="Comparison table")
callout(s, 0.6, 6.15, 12.15, 0.7, "Each step down the table adds machinery only because the step before it broke.", size=15, fill="green_t")

s = slide("Where each algorithm lives in the code",
          notes="Everything students need to read is in labs/. The shared pieces are in common.py. Lab 4 imports the DDPG class from Lab 3.")
rows = [["Piece", "Where", "What to look for"],
        ["Box discretiser, table update", "lab1  Boxes, q_learning", "the three-line update, ε-greedy"],
        ["Replay buffer", "common.py  ReplayBuffer", "ring buffer, terminated only"],
        ["Polyak / hard copy", "common.py  soft_update", "τ = 1 copies, τ = 0.005 averages"],
        ["DQN target and update", "lab2  DQN.targets, DQN.update", "no_grad target, gather(Q of taken action)"],
        ["Actor, critic networks", "lab3  Actor, Critic", "tanh output scaled to the bounds"],
        ["DDPG / TD3 update", "lab3  DDPG.update", "critic step, actor step, target step"],
        ["Noise", "lab3  GaussianNoise, OUNoise", "one line each"],
        ["Training loop and evaluation", "lab3 train, common.evaluate", "warm-up, terminated vs truncated"]]
table(s, 0.6, 1.55, [3.9, 4.2, 4.05], rows, size=13.5, row_h=0.56, head_h=0.5, fonts={1: "Courier New"}, name="Code map")
text(s, 0.6, 6.55, 12.1, 0.4, "Run any lab with --trace to print one update with the real numbers, and check it by hand.", size=14, color="muted")

s = slide("Where this leads: from off-policy to PPO",
          notes="Close the loop with the next exercise. The students arrive knowing what a critic, a bootstrapped target and exploration noise are, and can ask what PPO does about each.")
tile(s, 0.6, 1.55, 6.0, 3.3, "What students now know",
     ["A critic and a bootstrapped target", "Why targets must be frozen or averaged", "Why exploration is a data problem, not an algorithm problem",
      "How to read a learning curve: seeds, evaluation, ablations"], bullets=True, fill="green_t", body_pt=15)
tile(s, 6.75, 1.55, 6.0, 3.3, "What PPO does differently",
     ["On-policy: uses each batch once, then discards it", "No replay buffer and no target networks", "No critic over actions: a value function of the state",
      "Stability comes from clipping the policy change"], bullets=True, fill="blue_t", body_pt=15)
text(s, 0.6, 5.1, 12.1, 0.4, "Run it", size=19, bold=True)
code_block(s, 0.6, 5.55, 12.15, 1.3,
           ["python lab1_tabular_mountaincar.py   |   python lab2_dqn_mountaincar.py --compare",
            "python lab3_ddpg_inverted_pendulum.py --td3 --seeds 5   |   python lab4_ddpg_mountaincar_continuous.py",
            "python play.py --lab 3      # watch the trained controller"], size=13, name="Commands")

# ============================================================================ SAVE
sections = []
for i, (name, first) in enumerate(SECTIONS):
    last = SECTIONS[i + 1][1] if i + 1 < len(SECTIONS) else len(prs.slides)
    sections.append((name, list(range(first, last))))
add_sections(prs, sections)
out = os.path.join(ROOT, "RL_Bridge_DQN_DDPG.pptx")
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
