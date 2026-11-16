"""Build RL_in_One_Hour_RO.pptx: the Romanian version of the deck.

Same layout and the same numbers as build_deck_1h.py (every number is computed from the lab code).
Technical terms stay in English (RL, MDP, Agent, Environment, State, Action, Reward, Return, Policy, Q-learning,
Q-table, Q-value, TD error, Value iteration, discount, probing, greedy, sweep, ...); the rest is translated.
Decimals use the Romanian comma on the slides; code blocks and command lines keep the dot.
"""
import contextlib
import io
import os
import re
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

# ---------------------------------------------------------------- Romanian number format
_DEC = re.compile(r"(?<!--alpha )(?<!--gamma )(?<![\w.])(\d)\.(\d)")


def D(txt):
    """Decimal point -> decimal comma (command-line flags such as '--alpha 0.5' are left alone)."""
    return _DEC.sub(r"\1,\2", txt)


class RoFmt:
    """Grid-cell number format: two decimals, comma, real minus sign."""

    def format(self, v):
        return f"{v:.2f}".replace(".", ",").replace("-", "−")


_orig_add_runs, _orig_draw_grid = deckkit.add_runs, deckkit.draw_grid


def _add_runs_ro(p, txt, *a, **k):
    return _orig_add_runs(p, D(txt), *a, **k)


def _draw_grid_ro(*a, **k):
    k["fmt"] = RoFmt()
    return _orig_draw_grid(*a, **k)


deckkit.add_runs, deckkit.draw_grid = _add_runs_ro, _draw_grid_ro
from deckkit import *                                   # noqa: E402  (picks up the patched helpers)
from gridworld import ARROWS, make_env                  # noqa: E402

deckkit.COURSE = "Reinforcement Learning într-o oră"
G = 0.95
prs = new_presentation()
SECTIONS = []          # (name, first_slide_index)
_content_slide = deckkit.content_slide


def content_slide(prs_, title, module=None, notes="", number=None, chip_color="green"):
    return _content_slide(prs_, title, module, D(notes), number, chip_color)


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
    text(s, 0.6, 1.55, 6.0, 0.45, "Ce veți face", size=20, bold=True)
    text(s, 0.6, 2.05, 6.0, 2.6, intro, size=16, bullets=True, space_after=7)
    text(s, 0.6, 4.75, 6.0, 0.4, "Rulați", size=20, bold=True)
    code_block(s, 0.6, 5.2, 6.0, len(commands) * 0.34 + 0.4, commands, size=14, name="Commands")
    box(s, 7.0, 1.55, 5.75, 3.55, fill="mist", radius=0.1, name="Exercises card")
    text(s, 7.3, 1.7, 5.2, 0.45, "Exerciții", size=20, bold=True)
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


def numere(n):
    """'18 numere', '72 de numere'."""
    return f"{n} numere" if 1 <= n % 100 <= 19 else f"{n} de numere"


# ============================================================================ INTRODUCERE
begin("Introducere")

# ---- 1 title
s = prs.slides.add_slide(prs.slide_layouts[5])
dark_background(s)
t = s.shapes.title
t.text_frame.text = "Reinforcement Learning într-o oră"
t.left, t.top, t.width, t.height = Inches(0.8), Inches(1.7), Inches(6.9), Inches(2.6)
t.text_frame.vertical_anchor = MSO_ANCHOR.TOP
for r in t.text_frame.paragraphs[0].runs:
    r.font.size = Pt(48)
    setc(r.font.color, "paper")
text(s, 0.8, 4.3, 6.7, 1.0, "De la ecuația lui Bellman la Q-learning: o introducere deterministă, cu tabele",
     size=20, color="mint", name="Subtitle")
text(s, 0.8, 5.55, 6.9, 1.2, ["Pe baza cursului lui David Silver (UCL / DeepMind) și a lucrării Lewis & Vrabie, "
                              "„Reinforcement Learning and Adaptive Dynamic Programming for Feedback Control”, IEEE CAS Magazine, 2009",
                              "Două laboratoare scurte în Python · fără probabilități · fără aproximatori de funcții"],
     size=14, color="code_dim", name="Credits", space_after=6)
draw_grid(s, env, 8.1, 1.7, 0.78, actions=act, dark=True, show_start=False, arrow_pt=22, name="Title gridworld")
text(s, 8.1, 5.75, 4.7, 0.6, "Policy-ul optim pentru un gridworld: asta veți calcula, iar apoi veți învăța.",
     size=14, color="code_dim", name="Grid caption")
s.notes_slide.notes_text_frame.text = D(
    "Bun venit. Aceasta este o introducere de o oră. Timpii: circa 3 minute de introducere, 11 pentru problemă, 17 pentru "
    "planificarea cu model (inclusiv o demonstrație live de 5 minute a Lab 1), 19 pentru învățarea fără model (inclusiv o "
    "demonstrație live de 5 minute a Lab 2), 7 pentru încheiere și 3 pentru întrebări. Lumea este deterministă: același action "
    "în același state dă mereu același rezultat, deci nu avem probabilități de urmărit. Totul este stocat în tabele pe care le "
    "puteți tipări. Gridworld-ul din dreapta este exemplul de lucru: la final veți calcula policy-ul optim cu Value iteration, "
    "apoi îl veți învăța din experiență cu Q-learning. Cadrul control versus learning vine din lucrarea lui Lewis și Vrabie din 2009.")

# ---- 2 the hour at a glance
s = content_slide(prs, "Ora pe scurt", notes=(
    "Parcurgeți bara de timp. Blocurile galbene sunt cele două demonstrații live ale laboratoarelor: rulați scriptul, citiți "
    "împreună rezultatul și lăsați exercițiile pentru după. Cele patru plăcuțe sunt contractul orei. Spuneți cu voce tare ce nu "
    "intră în această oră: policy iteration, bandits și strategiile de exploration, lumile probabilistice și aproximatorii de "
    "funcții. Sunt continuări firești, nu cerințe prealabile."))
segs = [("Intro", 3, "slate", "3"), ("Problema", 11, "green", "11 min"), ("Planificare", 12, "blue", "12 min"),
        ("Lab 1", 5, "amber", "5"), ("Învățare", 14, "plum", "14 min"), ("Lab 2", 5, "amber", "5"),
        ("Încheiere", 7, "slate", "7 min"), ("Q&A", 3, "line", "3")]
x0, per_min = 0.6, 12.1 / 60.0
for name, mins, col, lab in segs:
    w_ = mins * per_min
    b_ = box(s, x0, 1.7, w_ - 0.04, 0.75, fill=col, radius=0.06, name="Time block: " + name)
    label_in(b_, lab, size=15, bold=True, color="ink" if col in ("amber", "line") else "paper")
    text(s, x0 - 0.1, 2.52, w_ + 0.16, 0.35, name, size=13, color="muted", align="c", name="Time label: " + name)
    x0 += w_
text(s, 0.6, 3.0, 12.1, 0.4, "60 de minute · 22 de slide-uri · două demonstrații live", size=15, color="muted", italic=True)
goals = [("Descrieți o problemă", "ca un sistem cu State, Action, Reward și discount: un MDP determinist"),
         ("Planificați cu Bellman", "calculați valorile optime și un policy cu Value iteration (Lab 1)"),
         ("Învățați din date", "completați un Q-table din experiență, fără niciun model al lumii (Lab 2)"),
         ("Explicați exploration", "de ce datele trebuie să acopere fiecare pereche State–Action")]
for i, (h_, b_) in enumerate(goals):
    tile(s, 0.6 + i * 3.09, 3.55, 2.85, 2.55, h_, b_, num=i + 1, title_pt=18, body_pt=15)
text(s, 0.6, 6.4, 12.1, 0.5, "Nu intră în această oră: policy iteration, bandits, strategii de exploration, lumi probabilistice, "
     "aproximatori de funcții.", size=15, color="muted", italic=True)

# ============================================================================ PARTEA 1  PROBLEMA
begin("1 · Problema")

# ---- 3 learning by acting
s = content_slide(prs, "Învățarea prin interacțiune", "1", notes=(
    "Începem cu bucla. La fiecare pas agent-ul se uită la state, alege un action, iar lumea răspunde cu un reward și cu un state "
    "nou. Aceasta este toată interfața, și exact asta face env.step(a) în Lab 2. Trei lucruri o deosebesc de învățarea cu "
    "etichete: nimeni nu-i spune agent-ului care action a fost corect, feedback-ul poate sosi mult după decizia care l-a cauzat, "
    "iar scopul este reward-ul total în timp, nu următorul reward. Lewis și Vrabie încep lucrarea cu aceeași idee: organismele "
    "vii își îmbunătățesc comportamentul acționând, observând reward-ul și ajustându-se. Circa 3 minute."))
box(s, 0.6, 1.65, 2.8, 1.85, fill="green_t", line="green", radius=0.12, name="Agent")
text(s, 0.6, 1.8, 2.8, 0.5, "Agent", size=26, bold=True, align="c", color="green")
text(s, 0.75, 2.4, 2.5, 1.0, "alege action-uri cu policy-ul său", size=15, align="c")
box(s, 5.6, 1.65, 2.8, 1.85, fill="blue_t", line="blue", radius=0.12, name="Environment")
text(s, 5.6, 1.8, 2.8, 0.5, "Environment", size=26, bold=True, align="c", color="blue")
text(s, 5.75, 2.4, 2.5, 1.0, "lumea; regulile ei pot fi necunoscute", size=15, align="c")
math(s, 3.45, 1.62, 2.1, 0.4, "action a_{k}", size=18, align="c")
line(s, 3.45, 2.05, 5.55, 2.05, color="ink", w=2.5, arrow=True)
math(s, 3.45, 2.3, 2.1, 0.4, "state s_{k+1}", size=18, align="c")
line(s, 5.55, 2.72, 3.45, 2.72, color="ink", w=2.5, arrow=True)
math(s, 3.45, 2.85, 2.1, 0.4, "reward r_{k}", size=18, align="c", color="coral")
line(s, 5.55, 3.3, 3.45, 3.3, color="coral", w=2.5, arrow=True)
text(s, 9.0, 1.65, 3.7, 0.45, "La fiecare pas k", size=18, bold=True)
text(s, 9.0, 2.15, 3.7, 1.5, ["Agent-ul vede state-ul s_{k}", "Alege un action a_{k}", "Lumea răspunde cu r_{k} și s_{k+1}"],
     size=16, bullets=True, space_after=6)
cards = [("Fără profesor", "Nimeni nu spune care action a fost corect. Se întoarce doar un număr: reward-ul.", "mist"),
         ("Feedback întârziat", "O mutare poate da roade abia după mulți pași. Agent-ul trebuie să lege cauza de efect.", "mist"),
         ("Un scop, nu o etichetă", "Maximizați reward-ul total în timp, nu doar următorul reward.", "mist")]
for i, (h_, d_, c_) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 4.05, 3.9, 2.0, fill=c_, radius=0.1, name="Card: " + h_)
    text(s, x + 0.25, 4.2, 3.4, 0.5, h_, size=20, bold=True)
    text(s, x + 0.25, 4.75, 3.4, 1.2, d_, size=16)
text(s, 0.6, 6.4, 12.1, 0.5, "Natura face la fel: animalele își ajustează comportamentul din reward și pedeapsă (Lewis & Vrabie, 2009).",
     size=15, color="muted", italic=True)

# ---- 4 vocabularies
s = content_slide(prs, "O singură buclă, două vocabulare", "1", notes=(
    "Lewis și Vrabie scriu aceeași buclă în limbajul controlului: environment-ul este sistemul sau procesul (plant), policy-ul "
    "este un regulator cu feedback, reward-ul este opusul unui cost, iar value este cost-to-go. Păstrați acest tabel în minte "
    "dacă veniți din control sau din circuite: matematica este aceeași, diferă doar numele și un semn. În acest curs folosim "
    "termenii din stânga și maximizăm reward-ul; lucrarea minimizează costul. Un sistem în timp discret x(k+1) = F(x(k), u(k)) "
    "are deja proprietatea Markov: următorul state depinde doar de state-ul curent și de intrare. Circa 2 minute."))
rows = [["În acest curs", "Simbol", "În teoria controlului", "Simbol"],
        ["Environment", "s_{k+1} = F(s_{k}, a_{k})", "Sistem (plant)", "x_{k+1} = F(x_{k}, u_{k})"],
        ["State", "s", "Stare", "x"],
        ["Action", "a", "Mărime de comandă", "u"],
        ["Policy", "a = π(s)", "Regulator cu feedback", "u = h(x)"],
        ["Reward (se maximizează)", "r(s, a)", "Utilitate, cost (se minimizează)", "r(x, u)"],
        ["Value", "V(s)", "Cost-to-go", "V^{h}(x)"],
        ["Q-value", "Q(s, a)", "Valoare dependentă de acțiune", "Q^{h}(x, u)"],
        ["TD error", "e = r + γ V(s′) − V(s)", "Reziduul Bellman (Hamiltonian)", "e_{k} = r + γ V(x_{k+1}) − V(x_{k})"]]
table(s, 0.6, 1.5, [2.6, 3.1, 3.3, 3.1], rows, size=15, row_h=0.5, head_h=0.5, fonts={1: "Cambria", 3: "Cambria"},
      header_fills={0: "green", 1: "green", 2: "slate", 3: "slate"}, name="Vocabular RL și control")
callout(s, 0.6, 6.15, 12.1, 0.7, "**Convenția de semn:** reward = −cost. A maximiza reward-ul și a minimiza costul este aceeași problemă.",
        fill="amber_t", size=16, name="Convenția de semn")

# ---- 5 the world
s = content_slide(prs, "Lumea: un sistem determinist", "1", notes=(
    "Acesta este exemplul de lucru pentru toată ora: o grilă de 4 pe 6 cu 18 celule libere plus două ieșiri. Agent-ul pornește din "
    "S. Ieșirea apropiată E este la două mutări și plătește 0.3. Ținta îndepărtată G este la șase mutări și plătește 1.0. Fiecare "
    "mutare costă 0.02, deci rătăcirea este penalizată. Dacă te lovești de un perete, rămâi pe loc. Este un Markov decision process "
    "fără probabilități: același action în același state dă mereu același rezultat, ca într-un sistem de control în timp discret. "
    "Problemele reale sunt zgomotoase; tratarea lor este următorul curs firesc. Circa 3 minute."))
draw_grid(s, env, 0.7, 1.7, 0.85, name="Gridworld-ul cursului", start_label=False)
mark_start(s, 0.7, 1.7, 0.85)
legend = [("S", "start", "mist", "blue"), ("E", "ieșire, +%.1f" % EXIT_R, "amber_l", "ink"),
          ("G", "țintă, +%.1f" % GOAL_R, "amber", "ink"), ("", "perete", "ink", "paper")]
for (sym, desc, fill, tcol), lx, lw in zip(legend, (0.7, 1.75, 3.4, 4.85), (0.6, 1.2, 1.2, 0.8)):
    b_ = box(s, lx, 5.25, 0.38, 0.38, fill=fill, radius=0.04, line="line", line_w=1, name="Legendă " + desc)
    label_in(b_, sym, size=13, bold=True, color=tcol)
    text(s, lx + 0.45, 5.25, lw, 0.38, desc, size=12, anchor="m")
cards = [("State s", "celula în care se află agent-ul: %d celule, %d în care poate acționa și 2 ieșiri" % (env.n_states, N_LIVE)),
         ("Action a", "mișcare ↑ ↓ ← →"),
         ("Transition F", "s′ = F(s, a): celula în care ajungi; un perete te lasă pe loc"),
         ("Reward r", neg("%.2f" % STEP) + " la fiecare mutare, +%.1f la E, +%.1f la G; ajungerea la E sau G încheie episodul" % (EXIT_R, GOAL_R))]
for i, (h_, d_) in enumerate(cards):
    y = 1.65 + i * 0.98
    box(s, 6.6, y, 6.1, 0.88, fill="mist", radius=0.1, name="Card: " + h_)
    text(s, 6.8, y, 1.7, 0.88, h_, size=17, bold=True, color="green", anchor="m")
    text(s, 8.5, y, 4.1, 0.88, d_, size=14, anchor="m")
callout(s, 0.6, 5.95, 12.1, 0.85, "**Determinist:** același state și același action dau mereu același rezultat. "
        "Nu există probabilități nicăieri în acest curs.", fill="amber_t", size=17, name="Determinist")

# ---- 6 policy and return
s = content_slide(prs, "Policy și return", "1", notes=(
    "Un policy este o regulă care alege un action în fiecare state: aici, o săgeată pe celulă. Ca să comparăm policy-uri avem "
    "nevoie de un singur număr per policy: return-ul, sau cost-to-go în cuvintele lui Lewis și Vrabie. Este suma reward-urilor "
    "colectate, fiecare înmulțit cu gamma la puterea numărului de pași la care se află. Ambele policy-uri de pe slide sunt "
    "legitime: A merge către ieșirea apropiată, B către ținta îndepărtată. Cu gamma = 0.95 câștigă ținta îndepărtată: %s față de %s. "
    "Valoarea unui state este return-ul obținut pornind de acolo și urmând policy-ul. Circa 3 minute." % (f3(ret_G), f3(ret_E))))
text(s, 0.6, 1.5, 5.8, 1.1, ["**Policy π:** o regulă care alege un action în fiecare state. Un tabel cu o săgeată pe celulă."],
     size=17)
text(s, 6.9, 1.5, 5.8, 1.2, ["**Return:**  G = r_{0} + γ r_{1} + γ^{2} r_{2} + …", "V_{π}(s) este return-ul obținut pornind din s și urmând π."],
     size=17, space_after=4)
for i, (lab, path_, ret_, calc, col) in enumerate([
        ("Policy A: către ieșirea apropiată E", path_E, ret_E,
         "%.2f + γ × %.2f = %s" % (STEP, EXIT_R, f3(ret_E)), "amber"),
        ("Policy B: către ținta îndepărtată G", path_G, ret_G,
         "%.2f (1 + γ + γ² + γ³ + γ⁴) + γ⁵ × %.1f = %s" % (STEP, GOAL_R, f3(ret_G)), "green")]):
    x = 0.9 + i * 6.1
    panel_label(s, x - 0.7, 5.3, 2.75, lab, size=17)
    draw_grid(s, env, x, 3.2, 0.62, path=path_, name="Traseu " + lab, start_label=False)
    mark_start(s, x, 3.2, 0.62)
    text(s, x - 0.3, 5.75, 4.3, 0.4, "%d mutări" % (len(path_) - 1), size=15, color="muted", align="c")
    math(s, x - 1.0, 6.1, 5.7, 0.5, neg(calc), size=16, align="c", color="ink")

# ============================================================================ PARTEA 2  PLANIFICARE
begin("2 · Planificare cu model")

# ---- 7 Bellman's principle
s = content_slide(prs, "Principiul optimalității al lui Bellman", "2", notes=(
    "Care policy este cel mai bun? Să le încercăm pe toate este fără speranță: chiar și această hartă mică are 4 la puterea 18, "
    "adică circa 69 de miliarde de policy-uri deterministe. Principiul lui Bellman rupe problema în bucăți: oricare ar fi fost "
    "prima mutare, restul traseului trebuie să fie cel mai bun din locul în care ajungi. În ecuație: cea mai bună valoare a unui "
    "state este maximul, peste action-uri, al reward-ului de acum plus gamma ori cea mai bună valoare a următorului state. "
    "Următorul state este F(s, a): folosim modelul. Această ecuație este firul roșu al restului orei: Lab 1 o rezolvă cu modelul, "
    "Lab 2 o rezolvă din date. Circa 3 minute."))
box(s, 0.6, 1.6, 4.0, 3.5, fill="coral_t", radius=0.12, name="Stat: forță brută")
math(s, 0.6, 1.75, 4.0, 1.3, "6.9 × 10^{10}", size=44, color="coral", align="c", bold=True)
text(s, 0.85, 3.1, 3.5, 1.9, ["policy-uri deterministe pe această hartă mică (4^{%d})." % N_LIVE,
                              "Compararea lor una câte una este fără speranță."], size=17, align="c", space_after=6)
text(s, 5.0, 1.6, 7.7, 1.2, "Oricare ar fi prima mutare, restul traseului trebuie să fie cel mai bun din locul în care ajungi.",
     size=22, bold=True, font="Cambria")
math(s, 5.0, 3.0, 7.7, 0.8, "V_{*}(s) = max_{a} [ r(s,a) + γ V_{*}(F(s,a)) ]", size=28)
for i, (h_, d_, col) in enumerate([("V_{*}(s)", "cel mai bun reward total din s", "green"),
                                   ("r(s,a)", "reward-ul de acum", "coral"),
                                   ("γ V_{*}(F(s,a))", "cea mai bună valoare a următorului state, cu discount", "blue")]):
    x = 5.0 + i * 2.6
    box(s, x, 3.95, 2.45, 1.15, fill="mist", radius=0.1, name="Termen: " + d_)
    math(s, x, 4.0, 2.45, 0.45, h_, size=17, color=col, align="c", bold=True)
    text(s, x + 0.1, 4.45, 2.25, 0.6, d_, size=13, align="c")
box(s, 0.6, 5.5, 12.1, 1.2, fill="pine", radius=0.12, name="Plan")
text(s, 0.95, 5.5, 11.4, 1.2, "Deci: calculăm V_{*} pentru fiecare celulă, apoi acționăm greedy în raport cu ea. "
     "F este modelul: aceasta este planificare.", size=19, color="paper", anchor="m")

# ---- 8 one Bellman step
s_cell = cell_s(2, 5)
qv = L1.one_step_lookahead(env, V, s_cell, G)
assert int(np.argmax(qv)) == 0 and abs(qv.max() - V[s_cell]) < 1e-9
s = content_slide(prs, "Un pas Bellman, cu numere", "2", notes=(
    "Luați celula evidențiată, în coloana din dreapta, chiar sub vecina țintei. Pentru fiecare dintre cele patru action-uri, "
    "căutați unde duce și cât plătește, apoi adăugați gamma ori valoarea acelei celule următoare. Mersul în sus ajunge în celula "
    "de lângă țintă, cu valoarea 1.0: -0.02 + 0.95 x 1.00 = 0.930. Mersul în jos sau la stânga duce în celule cu valoarea %.3f, "
    "deci obțin %.3f. Mersul la dreapta se lovește de margine și te lasă în aceeași celulă, care are ea însăși valoarea %.3f, "
    "deci obține %.3f. Maximul este %.3f, adică exact valoarea deja scrisă în tabel: tabelul satisface ecuația. Circa 2 minute."
    % (V[env.model(s_cell, 1)[0]], qv[1], V[s_cell], qv[3], qv.max())))
draw_grid(s, env, 0.7, 1.8, 0.85, values=V, vmax=1.0, value_pt=12, path=[s_cell], name="V* cu o celulă evidențiată")
text(s, 0.7, 5.3, 5.1, 0.7, "Valorile optime V_{*} pentru γ = %.2f. Celula evidențiată este cea pe care o verificăm." % G,
     size=14, color="muted")
rows = [["Action", "Ajunge în", "r + γ V_{*}(următorul)", "Valoare"]]
names = ["↑ sus", "↓ jos", "← stânga", "→ dreapta"]
for a_ in range(4):
    s2, r_, done_ = env.model(s_cell, a_)
    land = "(%d,%d)" % env.cells[s2] if s2 != s_cell else "aceeași (perete)"
    rows.append([names[a_], land, neg("%.2f + %.2f × %.3f" % (r_, G, V[s2])), "%.3f" % qv[a_]])
best_row = 1 + int(np.argmax(qv))
table(s, 6.3, 1.8, [1.4, 1.9, 2.1, 1.0], rows, size=15, row_h=0.6, head_h=0.55, fonts={2: "Cambria"},
      fills={(best_row, j): "green_t" for j in range(4)}, aligns={3: "c"}, name="Cele patru action-uri candidate")
math(s, 6.3, 4.7, 6.4, 0.6, "V_{*}(%d,%d) = max{ %s } = %.3f" % (env.cells[s_cell] + (", ".join("%.3f" % q for q in qv), qv.max())),
     size=18)
callout(s, 6.3, 5.4, 6.4, 1.0, "Cel mai bun action: **↑**. Valoarea lui este deja cea din tabel: tabelul satisface ecuația.",
        fill="green_t", size=16, name="Cel mai bun action")

# ---- 9 value iteration
s = content_slide(prs, "Value iteration", "2", notes=(
    "Ecuația îl are pe V* în ambele părți. Lewis și Vrabie observă că este o ecuație de punct fix: aplicați din nou și din nou "
    "partea dreaptă și tabelul converge, pornind de la orice tabel. Aceasta este value iteration. Pornim de la zerouri și, la "
    "fiecare sweep, înlocuim fiecare valoare cu cea mai bună căutare pe un pas, calculată din tabelul anterior. O lectură utilă: "
    "după k sweep-uri, V conține cel mai bun reward total pe care îl poți strânge în k mutări. Are nevoie de model, F și r, ca să "
    "facă această căutare. Comparați cu codul din lab1_value_iteration.py: sunt aceleași șase linii. Circa 3 minute."))
math(s, 0.6, 1.55, 12.1, 0.8, "V_{k+1}(s) = max_{a} [ r(s,a) + γ V_{k}(F(s,a)) ]", size=28, align="c")
code_block(s, 0.6, 2.6, 6.4, 3.15, ["V ← 0 pentru fiecare celulă", "repetă:", "  pentru fiecare celulă neterminală s:",
                                    "    pentru fiecare action a:", "      q[a] ← r(s,a) + γ V[F(s,a)]",
                                    "    V_nou[s] ← max peste a din q[a]", "  V ← V_nou      # tabelul vechi în dreapta",
                                    "până când V nu se mai schimbă", "π(s) ← a cu cel mai mare q[a]"],
           size=15, name="Pseudo-cod Value iteration")
text(s, 7.4, 2.6, 5.3, 0.45, "Ce observăm", size=20, bold=True)
text(s, 7.4, 3.15, 5.3, 3.5, ["Orice tabel de start merge: zerourile sunt bune",
                              "Fiecare sweep este o căutare pe un pas pentru fiecare celulă, din tabelul anterior",
                              "După k sweep-uri, V conține cel mai bun reward strâns în k mutări",
                              "Are nevoie de model: F și r"], size=16, bullets=True, space_after=9)

# ---- 10 VI in action
s = content_slide(prs, "Value iteration în acțiune", "2", notes=(
    "La k = 0 toate valorile sunt zero. După un sweep, doar celulele de lângă o ieșire știu de reward-ul ei. Fiecare sweep "
    "următor împinge informația cu o celulă mai departe, micșorând-o cu gamma și cu costul pasului. După %d sweep-uri nimic nu "
    "se mai schimbă: acestea sunt valorile optime. Informația curge înapoi dinspre reward. Celulele aproape albe nu sunt încă "
    "informate: valoarea lor este doar costul mutărilor făcute până acum. Circa 2 minute." % len(changes)))
vi_panels = [("k = 1", snap[1]), ("k = 2", snap[2]), ("k = 4", snap[4]), ("k = %d  (V_{*})" % len(changes), snap[len(changes)])]
assert np.allclose(snap[len(changes)], V)
for i, (lab, vals) in enumerate(vi_panels):
    x = 0.7 + i * 3.08
    panel_label(s, x, 2.82, 1.6, lab, size=18)
    draw_grid(s, env, x, 2.1, 0.47, values=vals, vmax=1.0, value_pt=10, fmt="{:.2f}", name="Value iteration " + lab, start_label=False)
math(s, 0.6, 4.6, 12.1, 0.75, "V_{k+1}(s) = max_{a} [ r(s,a) + γ V_{k}(F(s,a)) ]", size=24, align="c")
text(s, 0.6, 5.5, 12.1, 1.0, ["Informația despre reward curge înapoi dinspre ieșiri, o celulă pe sweep, scăzând cu γ de fiecare dată.",
                              "γ = %.2f. Converge după %d sweep-uri; cel mai lung traseu are 6 mutări." % (G, len(changes))],
     size=16, color="muted", align="c", space_after=4)

# ---- 11 values to policy, gamma
s = content_slide(prs, "Citim policy-ul din valori", "2", notes=(
    "Odată ce V este cunoscut, policy-ul este încă o căutare pe un pas: în fiecare celulă alegeți action-ul cu cel mai bun "
    "r + gamma V al celulei următoare. Urmați săgețile din S și obțineți traseul albastru. Return-ul acelui traseu este egal cu "
    "V(S): %s în stânga. În dreapta, același cod și aceeași hartă cu gamma = 0.5: agent-ul se preocupă mult mai puțin de viitor, "
    "deci ținta îndepărtată nu merită drumul și alege ieșirea apropiată. Schimbarea are loc la un gamma de circa %.2f. Gamma "
    "este orizontul de planificare. Este planificatorul miop față de cel cu viziune lungă din Lewis și Vrabie. "
    "Circa 3 minute." % (f3(V[S]), switch_gamma)))
math(s, 0.6, 1.5, 12.1, 0.7, "π(s) = argmax_{a} [ r(s,a) + γ V_{*}(F(s,a)) ]", size=26, align="c")
for i, (lab, vals, acts, path_, tag) in enumerate([
        ("γ = %.2f: viziune pe termen lung" % G, V, act, path_G, "%d mutări către G · V(S) = %s" % (len(path_G) - 1, f3(V[S]))),
        ("γ = 0.5: viziune pe termen scurt", V5, act5, path_E, "%d mutări către E · V(S) = %s" % (len(path_E) - 1, f3(V5[S])))]):
    x = 0.7 + i * 4.3
    panel_label(s, x - 0.2, 4.12, 2.3, lab, size=17)
    draw_grid(s, env, x, 2.75, 0.62, values=vals, actions=acts, path=path_, vmax=1.0, arrow_pt=16, value_pt=9, name="Policy " + lab)
    text(s, x - 0.2, 5.3, 4.12, 0.4, tag, size=15, color="blue", bold=True, align="c")
callout(s, 9.3, 2.75, 3.45, 2.95, ["**γ este orizontul:** cât de departe se uită agent-ul.",
                                   "Același cod, aceeași hartă: traseul se schimbă în jurul lui γ\u00a0=\u00a0%.2f." % switch_gamma],
        fill="amber_t", size=17, name="Concluzie")
text(s, 0.6, 6.15, 12.1, 0.6, "Numere: valorile optime V_{*}. Săgeți: policy-ul greedy. Contur albastru: traseul din S.",
     size=15, color="muted", italic=True)

# ---- 12 Lab 1
lab_slide("Lab 1: value iteration pe un gridworld",
          ["Citiți gridworld.py: env.model(s, a) este modelul", "Rulați value iteration: valori, policy, traseu",
           "Opriți-l devreme cu --sweeps k și urmăriți unda de valoare", "Schimbați γ și costul pasului; explicați noul traseu"],
          ["python lab1_value_iteration.py", "python lab1_value_iteration.py --sweeps 3", "python lab1_value_iteration.py --gamma 0.5"],
          ["Câte sweep-uri? Comparați cu cele 6 mutări de la S la G.", "După k sweep-uri, unde este valoarea nenulă? De ce acolo?",
           "La ce γ trece agent-ul la ieșirea apropiată?", "Faceți mersul gratuit, apoi scump. Ce se schimbă?"],
          "Demonstrație live, circa 5 minute. Rulați prima comandă și citiți împreună rezultatul: grila de valori cu săgeți, traseul "
          "cu return-ul lui și linia care spune că V(S) este egal cu return-ul celui mai bun traseu. Apoi rulați --sweeps 3 ca să "
          "arătați unda, și --gamma 0.5 ca să arătați schimbarea. Lăsați exercițiile pentru după sesiune. Totul are circa 100 de linii.",
          "Ar trebui să vedeți: %d sweep-uri, un traseu de %d mutări către G și V(S) = %s. Cu --gamma 0.5: %d mutări către ieșirea "
          "apropiată, V(S) = %s." % (len(changes), len(path_G) - 1, f3(V[S]), len(path_E) - 1, f3(V5[S])))

# ============================================================================ PARTEA 3  ÎNVĂȚARE
begin("3 · Învățare fără model")

# ---- 13 planning vs learning
s = content_slide(prs, "Capcana: planificarea are nevoie de model", "3", notes=(
    "Value iteration a pus modelului o întrebare „ce-ar fi dacă” pentru fiecare celulă și fiecare action. În majoritatea "
    "problemelor reale nu avem acest model: avem doar sistemul real sau un simulator și putem doar să acționăm de unde suntem și "
    "să vedem ce se întâmplă. Lewis și Vrabie fac același contrast: programarea dinamică lucrează înapoi prin ecuațiile "
    "cunoscute, offline; reinforcement learning lucrează înainte în timp, online, din date măsurate (state, state următor, "
    "reward). Scopul este același, V* și policy-ul optim. Putem ajunge acolo doar din date? Circa 2 minute."))
for i, (h_, tag, c_, col, code, bullets) in enumerate([
        ("Planificare", "Lab 1", "blue_t", "blue", ["s2, r, done = env.model(s, a)", "  # orice state, orice action"],
         ["Cunoaște F și r", "Lucrează înapoi prin model, offline", "Parcurge fiecare celulă"]),
        ("Învățare", "Lab 2", "green_t", "green", ["s2, r, done = env.step(a)", "  # acționează, apoi observă"],
         ["Nu cunoaște F sau r", "Învață înainte în timp din date (s, a, r, s′)", "Învață doar despre ce vizitează"])]):
    x = 0.6 + i * 6.2
    box(s, x, 1.6, 5.9, 4.0, fill=c_, radius=0.12, name="Card: " + h_)
    text(s, x + 0.3, 1.75, 4.0, 0.5, h_, size=24, bold=True, color=col)
    text(s, x + 4.2, 1.8, 1.4, 0.45, tag, size=16, bold=True, color=col, align="r")
    code_block(s, x + 0.3, 2.4, 5.3, 1.0, code, size=15, name="Interfață: " + h_)
    text(s, x + 0.3, 3.65, 5.3, 1.9, bullets, size=17, bullets=True, space_after=7, bullet_color=col)
box(s, 0.6, 5.95, 12.1, 0.95, fill="pine", radius=0.12, name="Întrebare")
text(s, 0.95, 5.95, 11.4, 0.95, "Același scop: V_{*} și policy-ul optim. Putem ajunge acolo doar din date?", size=20, color="paper", anchor="m")

# ---- 14 why Q
s = content_slide(prs, "De ce un tabel de Q, nu de V", "3", notes=(
    "Ca să acționezi folosind V ai nevoie de model, pentru că a alege un action înseamnă a întreba unde duce fiecare action. "
    "Trucul, datorat lui Watkins și prezentat în lucrare ca învățare a valorii dependente de action, este să stocăm chiar "
    "căutarea pe un pas: Q(s, a) este valoarea lui a în s, urmat de o comportare bună. Acum cel mai bun action este pur și simplu "
    "cea mai mare intrare din rând, iar V este tot cea mai mare intrare. Nu este nevoie de model pentru a acționa. Tabelul din "
    "dreapta este tabelul real pe care îl învață Lab 2, pentru trei celule. Priviți celula de lângă ieșirea apropiată: mersul la "
    "stânga pare atractiv cu 0.3, dar mersul în sus valorează 0.63. Prețul: o intrare pentru fiecare pereche state-action, aici "
    "%d x 4 = %s. Circa 3 minute." % (N_LIVE, numere(N_LIVE * 4))))
math(s, 0.6, 1.55, 6.0, 0.55, "V(s) → are nevoie de F pentru a alege:", size=18, color="coral", bold=True)
math(s, 0.6, 2.1, 6.0, 0.6, "π(s) = argmax_{a} [ r(s,a) + γ V(F(s,a)) ]", size=20)
math(s, 0.6, 3.0, 6.0, 0.55, "Q(s,a) → stochează căutarea pe un pas:", size=18, color="green", bold=True)
math(s, 0.6, 3.55, 6.0, 0.6, "Q(s,a) = r(s,a) + γ V(F(s,a))", size=20)
math(s, 0.6, 4.2, 6.0, 0.6, "π(s) = argmax_{a} Q(s,a)    V(s) = max_{a} Q(s,a)", size=20)
callout(s, 0.6, 5.2, 6.0, 1.3, "**Nu e nevoie de model pentru a acționa:** cel mai bun action este cea mai mare intrare din rând.",
        fill="green_t", size=17, name="Fără model")
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
      aligns={1: "c", 2: "c", 3: "c", 4: "c"}, name="Valori Q învățate")
text(s, 7.1, 4.55, 5.6, 0.9, "Trei rânduri din tabelul învățat în Lab 2. Verde: cel mai bun action (sau cele mai bune). "
     "Egalitățile sunt reale: din S, sus și dreapta sunt la fel de bune.", size=14, color="muted")
text(s, 7.1, 5.55, 5.6, 0.9, "Prețul: o intrare pentru fiecare pereche State–Action: %d\u00a0×\u00a04\u00a0=\u00a0%s." % (N_LIVE, numere(N_LIVE * 4)), size=16)

# ---- 15 TD error
s = content_slide(prs, "Învățarea dintr-o tranziție: TD error", "3", notes=(
    "Agent-ul observă o tranziție: era în s, a făcut a, a primit r și a ajuns în s'. Înainte de asta, tabelul prezicea Q(s, a). "
    "După ea, agent-ul are o estimare mai bună: reward-ul tocmai primit plus gamma ori cel mai bun rezultat pe care crede că îl "
    "poate obține din s'. Diferența este temporal-difference error. Dacă este zero, tabelul este deja consistent cu ecuația "
    "Bellman pe această tranziție și nu e nimic de învățat. Altfel mutăm intrarea spre observație cu alfa ori eroarea. În această "
    "lume deterministă alfa = 1 este suficient: înlocuim vechea estimare. Lewis și Vrabie numesc TD error o eroare de predicție "
    "între performanța prezisă și cea observată și notează că neuronii dopaminergici par să poarte același semnal. Circa 3 minute."))
box(s, 0.6, 1.55, 12.1, 0.85, fill="mist", radius=0.1, name="Date")
text(s, 0.85, 1.55, 3.2, 0.85, "O tranziție observată", size=16, bold=True, color="muted", anchor="m")
math(s, 4.0, 1.55, 8.6, 0.85, "( s,  a,  r,  s′ )", size=28, align="c")
b = box(s, 0.6, 2.9, 3.6, 1.4, fill="blue_t", line="blue", radius=0.12, name="Prezis")
label_in(b, ["**Prezis**", "Q(s, a)"], size=20, font="Cambria")
math(s, 4.3, 3.15, 0.5, 0.9, "−", size=36, align="c")
b = box(s, 4.9, 2.9, 4.1, 1.4, fill="green_t", line="green", radius=0.12, name="Observat")
label_in(b, ["**Observat**", "r + γ max_{a′} Q(s′, a′)"], size=20, font="Cambria")
line(s, 9.1, 3.6, 9.7, 3.6, color="ink", w=3, arrow=True)
b = box(s, 9.8, 2.9, 2.9, 1.4, fill="amber_t", line="amber", radius=0.12, name="TD error")
label_in(b, ["**TD error**", "e"], size=20, font="Cambria")
box(s, 0.6, 4.65, 12.1, 0.95, fill="pine", radius=0.12, name="Actualizare")
math(s, 0.6, 4.65, 12.1, 0.95, "Q(s, a)  ←  Q(s, a) + α · e", size=28, color="paper", align="c")
for i, (h_, d_) in enumerate([("e = 0", "tabelul este consistent cu ecuația Bellman aici: nu mai e nimic de învățat"),
                              ("α = 1", "lume deterministă: noua estimare o înlocuiește pur și simplu pe cea veche"),
                              ("Creier", "neuronii dopaminergici par să semnaleze această eroare de predicție (Lewis & Vrabie, 2009)")]):
    x = 0.6 + i * 4.1
    box(s, x, 5.85, 3.9, 1.0, fill="mist", radius=0.1, name="Notă: " + h_)
    math(s, x + 0.15, 5.85, 0.95, 1.0, h_, size=15, color="green", bold=True, align="c")
    text(s, x + 1.1, 5.85, 2.7, 1.0, d_, size=12, anchor="m")

# ---- 16 Q-learning = sampled VI
s = content_slide(prs, "Q-learning este value iteration din date", "3", notes=(
    "Puneți cele două actualizări una lângă alta. Ambele fac o căutare pe un pas și iau un maxim. Singura diferență este de unde "
    "vin state-ul următor și reward-ul: value iteration întreabă modelul, Q-learning folosește ce tocmai a observat. De aceea "
    "Q-learning ajunge la același răspuns. Cele două linii de jos sunt linii reale din trace-ul din Lab 2. În episodul 1 agent-ul, "
    "aflat lângă ieșirea apropiată, calcă întâmplător pe ea și vede +0.30: acea intrare sare de la 0 la 0.300. În episodul 3 este "
    "din nou în S, face aceeași mutare, ajunge pe acea celulă și observă -0.02: target-ul este -0.02 + 0.95 x 0.300 = 0.265. Vestea "
    "bună despre ieșire a călătorit cu o celulă înapoi. Circa 3 minute."))
for i, (h_, tag, c_, col, formula, src) in enumerate([
        ("Value iteration", "model", "blue_t", "blue", "V(s) ← max_{a} [ r(s,a) + γ V(F(s,a)) ]", "s′ și r vin din modelul F"),
        ("Q-learning", "date", "green_t", "green", "Q(s,a) ← r + γ max_{a′} Q(s′,a′)", "s′ și r vin din ceea ce a fost observat")]):
    x = 0.6 + i * 6.2
    box(s, x, 1.6, 5.9, 2.35, fill=c_, radius=0.12, name="Card: " + h_)
    text(s, x + 0.3, 1.72, 3.5, 0.5, h_, size=22, bold=True, color=col)
    text(s, x + 4.2, 1.76, 1.4, 0.45, tag, size=16, bold=True, color=col, align="r")
    math(s, x + 0.2, 2.35, 5.5, 0.7, formula, size=22, align="c")
    text(s, x + 0.3, 3.2, 5.3, 0.6, src, size=16, color="muted", align="c")
text(s, 0.6, 4.25, 12.1, 0.45, "O actualizare reală din trace-ul Lab 2  (γ = %.2f, α = 1)" % G, size=18, bold=True)
rows = [["Episod", "State, action", "r, s′ observate", "Target = r + γ max Q(s′, ·)", "Q(s, a) înainte → după"],
        ["1", "(3,1), ←", "+0.30, ieșirea E", "+0.30  (E este terminal)", "0 → +0.300"],
        ["3", "(3,2), ←", "−0.02, (3,1)", "−0.02 + 0.95 × 0.300 = +0.265", "−0.020 → +0.265"]]
table(s, 0.6, 4.75, [1.2, 2.0, 2.3, 3.9, 2.7], rows, size=15, row_h=0.6, head_h=0.5, fonts={3: "Cambria"},
      aligns={0: "c", 4: "c"}, name="Actualizări exemplificate")
text(s, 0.6, 6.5, 12.1, 0.4, "Vestea despre ieșire se întoarce cu o celulă de fiecare dată când un vecin este actualizat.",
     size=15, color="muted", italic=True)

# ---- 17 exploration
s = content_slide(prs, "Tabelul învață doar ce încearcă agent-ul", "3", notes=(
    "Fiecare actualizare atinge doar perechea state-action care a fost încercată. Deci ceea ce încearcă agent-ul decide ce poate "
    "învăța. Comparați doi agenți pe aceeași hartă, cu aceeași regulă Q-learning. Agent-ul greedy face mereu ce pare mai bun în "
    "tabel. Dă peste ieșirea apropiată în episodul 1, tabelul spune acum că traseul acela e bun, și nu mai încearcă nimic altceva: "
    "ajunge la V(S) = %s în loc de %s, și nici una dintre cele %d valori nu este corectă. Agent-ul cu probing încearcă fiecare "
    "action în fiecare state pe rând, indiferent ce spune tabelul: găsește ținta îndepărtată în episodul %d, iar toate cele %d "
    "valori coincid cu value iteration până în episodul %d. Lewis și Vrabie numesc aceasta persistent excitation: datele "
    "trebuie să acopere fiecare pereche state-action. Există strategii mai deștepte, precum epsilon-greedy sau UCB; ele "
    "depășesc această oră. Circa 4 minute." % (f3(ret_g), f3(ret_p), N_LIVE, reached_p["G"], N_LIVE, all_exact_ep)))
xs = list(range(1, EPISODES + 1))
xy_chart(s, 0.6, 1.55, 6.2, 3.9, [("probing", xs, exact_p, "green"), ("greedy", xs, exact_g, "slate")],
         xtitle="episod", ytitle="celule cu valoarea exactă (din %d)" % N_LIVE, ymin=0, ymax=N_LIVE, ymajor=3, xmin=0, xmax=EPISODES,
         xmajor=30, legend=True, description="Grafic: probing ajunge la toate cele %d valori exacte; greedy rămâne la zero" % N_LIVE,
         name="Celule exacte pe episod")
for i, (lab, Qm, acts, path_, tag) in enumerate([
        ("Greedy: cel mai bun action cunoscut", Q_g, act_g, path_g, "se oprește la E · V(S) = %s" % f3(ret_g)),
        ("Probing: încearcă fiecare action pe rând", Q_p, act_p, path_p, "se oprește la G · V(S) = %s" % f3(ret_p))]):
    x = 7.1 + i * 2.85
    text(s, x - 0.05, 1.55, 2.75, 0.7, lab, size=14, bold=True, align="c")
    draw_grid(s, env, x, 2.35, 0.45, values=Qm.max(1), actions=acts, path=path_, vmax=1.0, arrow_pt=12, value_pt=7,
              name="Învățat " + lab, start_label=False)
    text(s, x - 0.05, 4.2, 2.75, 0.4, tag, size=13, color="blue", bold=True, align="c")
text(s, 7.1, 4.65, 5.6, 0.8, "Numere: max Q învățat. Săgeți: policy-ul greedy. Probing = a k-a vizită într-un state încearcă action-ul k mod 4.",
     size=13, color="muted")
callout(s, 0.6, 5.7, 12.1, 1.15, "**Probing înseamnă persistent excitation:** datele trebuie să acopere fiecare pereche State–Action. "
        "Există strategii mai deștepte (ε-greedy, UCB, …): nu sunt necesare azi.", fill="amber_t", size=17, name="Probing")

# ---- 18 Lab 2
lab_slide("Lab 2: Q-learning fără model",
          ["Agent-ul apelează doar env.reset() și env.step(a)", "Citiți trace-ul primului episod: verificați o actualizare de mână",
           "Comparați agent-ul cu probing cu unul greedy", "Verificați tabelul învățat față de Lab 1"],
          ["python lab2_q_learning.py", "python lab2_q_learning.py --behaviour greedy", "python lab2_q_learning.py --compare"],
          ["În ce episod este atinsă G prima dată? De ce nu se poate îmbunătăți tabelul înainte?",
           "Agent-ul greedy: ce ieșire folosește și cât este V(S) față de cel mai bun?", "Încercați --alpha 0.5. Ce costă un pas mai mic?",
           "Încercați --gamma 0.5. Mai este Q de acord cu Lab 1?"],
          "Demonstrație live, circa 5 minute. Rulați varianta implicită și citiți împreună trace-ul episodului 1: fiecare linie este o "
          "tranziție observată, cu target-ul și TD error-ul ei. Apoi arătați că toate cele %d valori ajung egale cu cele din value "
          "iteration. Rulați --behaviour greedy ca să arătați blocarea pe ieșirea apropiată, sau --compare ca să-i vedeți unul lângă "
          "altul. Learner-ul nu apelează niciodată env.model: value iteration este importat doar pentru a verifica răspunsul." % N_LIVE,
          "Ar trebui să vedeți: probing atinge G în episodul %d și toate cele %d din %d valori coincid cu value iteration, cu V(S) = %s. "
          "Greedy se oprește la E cu V(S) = %s." % (reached_p["G"], int(exact_p[-1]), N_LIVE, f3(ret_p), f3(ret_g)))

# ============================================================================ ÎNCHEIERE
begin("Încheiere")

# ---- 19 side by side
s = content_slide(prs, "Planificare și învățare, una lângă alta", notes=(
    "Rezumați ora într-un tabel. Ambii algoritmi rezolvă aceeași ecuație de optimalitate Bellman și ajung la același tabel. "
    "Diferența este sursa informației: un model care poate fi interogat oriunde, sau date adunate acționând. Aceasta este "
    "diferența pe care o fac Lewis și Vrabie între programarea dinamică offline și reinforcement learning online. Circa 2 minute."))
rows = [["", "Value iteration (Lab 1)", "Q-learning (Lab 2)"],
        ["Cunoaște modelul F, r?", "Da: env.model(s, a)", "Nu: doar env.step(a)"],
        ["Informația vine din", "interogări ale modelului, pentru fiecare celulă", "tranziții observate (s, a, r, s′)"],
        ["Tabel", "V(s): %s" % numere(N_LIVE), "Q(s, a): %s" % numere(N_LIVE * 4)],
        ["Actualizare", "V ← max_{a} [ r + γ V(F(s,a)) ]", "Q ← Q + α [ r + γ max Q′ − Q ]"],
        ["Exploration", "nu e necesar", "necesar: probing"],
        ["Rezultat", "V_{*} și π_{*}", "aceleași V_{*} și π_{*}  (%d din %d celule coincid)" % (int(exact_p[-1]), N_LIVE)]]
table(s, 0.6, 1.5, [2.9, 4.6, 4.6], rows, size=17, row_h=0.62, head_h=0.55, fonts={1: "Cambria", 2: "Cambria"},
      header_fills={0: "pine", 1: "blue", 2: "green"}, fills={(6, 1): "blue_t", (6, 2): "green_t"}, name="Value iteration versus Q-learning")
callout(s, 0.6, 6.15, 12.1, 0.7, "Același punct fix al ecuației de optimalitate Bellman, altă sursă de informație.",
        fill="mist", size=17, name="Rezumat")

# ---- 20 where tables stop
s = content_slide(prs, "Unde se opresc tabelele", notes=(
    "Un tabel este exact și transparent, dar are nevoie de o intrare pentru fiecare pereche state-action. Gridworld-ul nostru are "
    "72. Backgammon are circa 10 la puterea 20 de state-uri, o cifră din cursurile lui Silver. Iar un state continuu, precum "
    "poziția și viteza unui motor sau a unui robot, nu poate fi tabelat deloc. Răspunsul lui Lewis și Vrabie este să înlocuiască "
    "tabelul cu un aproximator parametric pentru valoare (critic) și pentru policy (actor): Approximate sau Adaptive Dynamic "
    "Programming. Acesta este pasul următor firesc și depășește această oră. Circa 2 minute."))
for i, (big, h_, d_, c_, col) in enumerate([("%d" % (N_LIVE * 4), "Gridworld-ul nostru", "%d celule × 4 action-uri: un tabel pe care îl poți tipări" % N_LIVE, "green_t", "green"),
                                            ("10^{20}", "Backgammon", "de state-uri, conform cursurilor lui Silver", "amber_t", "amber"),
                                            ("∞", "State continuu", "x în R^{n}: un motor, un braț robotic, un vehicul", "coral_t", "coral")]):
    x = 0.6 + i * 4.1
    box(s, x, 1.7, 3.9, 3.3, fill=c_, radius=0.12, name="Stat: " + h_)
    math(s, x, 1.85, 3.9, 1.5, big, size=60, color="ink" if i == 1 else col, align="c", bold=True)
    text(s, x + 0.25, 3.4, 3.4, 0.5, h_, size=21, bold=True, align="c")
    text(s, x + 0.25, 3.95, 3.4, 0.9, d_, size=15, align="c", color="muted")
box(s, 0.6, 5.3, 12.1, 1.5, fill="pine", radius=0.12, name="Pasul următor")
text(s, 0.95, 5.3, 11.4, 1.5, "Pasul următor: înlocuiți tabelul cu un aproximator de funcții pentru valoare (critic) și pentru policy (actor). "
     "Aceasta este Approximate sau Adaptive Dynamic Programming, tema lucrării Lewis & Vrabie, și depășește această oră.",
     size=18, color="paper", anchor="m")

# ---- 21 cheat sheet
s = content_slide(prs, "Fișă recapitulativă", notes=(
    "O referință pe o pagină pentru această oră. Rugați audiența să explice fiecare rând cu propriile cuvinte. Dacă pot, și-au "
    "atins obiectivele. Circa 1 minut."))
rows = [["Concept", "Pe scurt", "Formula-cheie"],
        ["Reward, return", "ce ne dorim; totalul cu discount", "G = Σ γ^{k} r_{k}"],
        ["Policy", "cum acționează agent-ul: un action pe state", "a = π(s)"],
        ["Valoarea optimă", "cel mai bun return din s: principiul lui Bellman", "V_{*}(s) = max_{a} [ r + γ V_{*}(F(s,a)) ]"],
        ["Value iteration", "ecuația ca regulă de actualizare; are nevoie de model", "V_{k+1}(s) = max_{a} [ r + γ V_{k}(F(s,a)) ]"],
        ["Q-function", "căutarea pe un pas stocată; dă policy-ul fără model", "Q(s,a) = r + γ V(F(s,a)),  π = argmax_{a} Q"],
        ["TD error", "predicție față de observație", "e = r + γ max_{a′} Q(s′,a′) − Q(s,a)"],
        ["Q-learning", "value iteration din date", "Q(s,a) ← Q(s,a) + α e"],
        ["Exploration", "datele trebuie să acopere fiecare pereche State–Action", "probing"]]
table(s, 0.6, 1.5, [2.3, 4.9, 4.9], rows, size=15, row_h=0.56, head_h=0.5, fonts={2: "Cambria"}, name="Fișă recapitulativă")

# ---- 22 questions and resources
s = content_slide(prs, "Verificați-vă, apoi mergeți mai departe", notes=(
    "Folosiți întrebările ca bilet de ieșire rapid sau ca temă. Răspunsuri sugerate: 1 ieșirea apropiată este mai aproape, deci "
    "este redusă mai puțin; cu un gamma mare câștigă reward-ul mai mare și mai îndepărtat, cu un gamma mic câștigă cel apropiat. "
    "2 Cea mai bună valoare a unui state este cel mai bun reward de acum plus cea mai bună valoare, cu discount, a locului în care "
    "ajungi. 3 Q conține deja căutarea prin model, deci argmax peste Q nu are nevoie de model; V nu o conține. 4 Predicția este "
    "deja de acord cu observația: tabelul este consistent aici. 5 A găsit întâi ieșirea apropiată, tabelul a spus apoi că "
    "traseul acela e bun, și nu a mai încercat alte action-uri, așa că n-a ajuns niciodată la ținta îndepărtată. Resursele sunt "
    "sursele acestui curs."))
text(s, 0.6, 1.5, 6.4, 0.45, "Verificați-vă înțelegerea", size=20, bold=True)
qs = ["De ce câștigă ieșirea apropiată la γ = 0.5, dar pierde la γ = 0.95?", "Ce spune ecuația de optimalitate Bellman, într-o propoziție?",
      "De ce poți acționa cu Q fără model, dar nu cu V?", "Ce înseamnă un TD error egal cu zero?",
      "De ce n-a găsit niciodată agent-ul greedy ținta îndepărtată?"]
for i, q in enumerate(qs):
    y = 2.05 + i * 0.96
    box(s, 0.6, y, 6.4, 0.84, fill="mist", radius=0.1, name="Întrebarea %d" % (i + 1))
    badge(s, 0.75, y + 0.15, 0.55, str(i + 1), size=16)
    text(s, 1.5, y, 5.4, 0.84, q, size=15, anchor="m")
text(s, 7.4, 1.5, 5.3, 0.45, "Mergeți mai departe", size=20, bold=True)
res = [("Silver: pagina cursului și slide-urile", "Lecțiile 1–5 sunt această oră în profunzime", "http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html"),
       ("Silver: videoclipurile cursului (YouTube)", "Reinforcement Learning, UCL / DeepMind", "https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLMZdRRhAoLnKFxZlmFoFp0uHVvN2PSE9T"),
       ("Sutton & Barto, ediția a 2-a", "Manualul de referință, gratuit online", "http://incompleteideas.net/book/the-book-2nd.html"),
       ("Lewis & Vrabie, IEEE CAS Magazine, 2009", "RL și ADP pentru controlul cu feedback", "https://doi.org/10.1109/MCAS.2009.933854")]
for i, (h_, d_, url) in enumerate(res):
    y = 2.05 + i * 1.2
    box(s, 7.4, y, 5.3, 1.05, fill="mist", radius=0.1, name="Resursa %d" % (i + 1))
    tb = text(s, 7.6, y + 0.12, 4.9, 0.45, h_, size=15, bold=True)
    text(s, 7.6, y + 0.57, 4.9, 0.4, d_, size=13, color="muted")
    link_runs(tb, url)

# ============================================================================ SAVE
sections = []
for i, (name, first) in enumerate(SECTIONS):
    last = SECTIONS[i + 1][1] if i + 1 < len(SECTIONS) else len(prs.slides)
    sections.append((name, list(range(first, last))))
add_sections(prs, sections)
out = os.path.join(ROOT, "RL_in_One_Hour_RO.pptx")
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
print("sections:", [(n, len(i)) for n, i in sections])
