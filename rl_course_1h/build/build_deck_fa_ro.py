"""Build RL_Function_Approximation_RO.pptx.

The second hour of the course, in Romanian. Technical terms stay in English
(neural network, weight, hidden layer, ReLU, gradient, feature, critic, actor,
controller, target, one-hot, fitted Q-iteration, semi-gradient, softmax, RMSE,
TD error, Q-function, Policy, Value iteration, sweep, probing, discount).
Every number is computed by labs/lab3_function_approximation.py.
Decimals use a comma on the slides; command lines keep the dot.
"""
import os
import re
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("COURSE_ROOT", os.path.dirname(HERE))
LABS = os.path.join(ROOT, "labs")
sys.path.insert(0, LABS)
sys.path.insert(0, HERE)

import numpy as np

import deckkit
import lab3_function_approximation as L3

_DEC = re.compile(r"(?<!--alpha )(?<!--gamma )(?<![\w.])(\d)\.(\d)")


def D(txt):
    """Decimal point -> decimal comma. Flags such as '--alpha 0.5' are left alone."""
    return _DEC.sub(r"\1,\2", txt)


class RoFmt:
    def format(self, v):
        return f"{v:.2f}".replace(".", ",").replace("-", "−")


_orig_add_runs, _orig_draw_grid = deckkit.add_runs, deckkit.draw_grid


def _add_runs_ro(p, txt, *a, **k):
    return _orig_add_runs(p, D(txt), *a, **k)


def _draw_grid_ro(*a, **k):
    k["fmt"] = RoFmt()
    return _orig_draw_grid(*a, **k)


deckkit.add_runs, deckkit.draw_grid = _add_runs_ro, _draw_grid_ro
from deckkit import *  # noqa: E402
from gridworld import make_env  # noqa: E402

deckkit.COURSE = "Aproximarea funcțiilor în RL"
G = L3.GAMMA
prs = new_presentation()
SECTIONS = []
_content_slide = deckkit.content_slide


def content_slide(prs_, title, module=None, notes="", number=None, chip_color="green"):
    return _content_slide(prs_, title, module, D(notes), number, chip_color)


def begin(name):
    SECTIONS.append((name, len(prs.slides)))


# ============================================================================ DATA FROM THE LAB
env = make_env()
V, Q_star, act_star = L3.exact_q(env, G)
data = L3.collect_transitions(env)
N_LIVE = int((~env.terminal).sum())
N_PAIRS = N_LIVE * env.n_actions
assert len(data["s"]) == N_PAIRS

linear, critic, actor = L3.run(env, Q_star, data, seed=0, gamma=G)
step = L3.linear_gradient_step(env, L3.STEP_ALPHA)
rows = L3.scan_seeds(env, Q_star, data, n=10, gamma=G)
bad = next(r for r in rows if not r["critic_goal"])
critic_bad = L3.fit_critic(env, data, Q_star, bad["seed"], G)
actor_bad = L3.fit_actor(env, critic_bad, Q_star, bad["seed"], G)
n_goal = sum(r["critic_goal"] for r in rows)
n_actor_goal = sum(r["actor_goal"] for r in rows)

assert critic.reached_goal and critic.n_optimal == N_LIVE
assert actor.reached_goal and actor.n_copied == N_LIVE
assert not linear.reached_goal and linear.n_optimal == 14
assert step["moved"] == N_LIVE
assert critic.n_params == 388 and actor.n_params == 116 and linear.n_params == 12
assert n_goal == 9 and bad["seed"] == 6 and n_actor_goal == 9
assert abs(critic.ret - V[env.start]) < 1e-9
assert abs(actor.ret - V[env.start]) < 1e-9

phi = step["phi"]
phi_S = L3.linear_features(env, [env.start])[0]
assert abs(step["q_start"] - step["alpha"] * step["target"] * float(phi @ phi_S)) < 1e-9

S_EXIT = env.pos2s[(3, 1)]
LEFT = 2
peak = max(critic.rmse_curve)
peak_at = critic.rmse_curve.index(peak) + 1

exact_path = env.follow_policy(act_star)


def f3(x):
    return f"{x:+.3f}"


def f4(x):
    return f"{x:+.4f}"


def numere(n):
    """'18 numere', '72 de numere' — Romanian counting."""
    return f"{n} numere" if 1 <= n % 100 <= 19 else f"{n} de numere"


def phi_txt(v):
    return "[ " + ",  ".join(f"{x:.1f}" for x in v) + " ]"


def table(slide, x, y, col_w, rows_, size=15, row_h=0.55, head_h=0.55, header_fills=None, first_bold=True,
          fills=None, fonts=None, aligns=None, name="Table"):
    nr, nc = len(rows_), len(col_w)
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(col_w)), Inches(head_h + row_h * (nr - 1)))
    gf.name = name
    tbl = gf.table
    tbl._tbl.tblPr.set("bandRow", "0")
    for j, wj in enumerate(col_w):
        tbl.columns[j].width = Inches(wj)
    for i, row in enumerate(rows_):
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


def lab_slide(title, intro, commands, exercises, notes, outcome):
    s = content_slide(prs, title, module="Lab", notes=notes, chip_color="amber")
    text(s, 0.6, 1.55, 6.0, 0.45, "Ce veți face", size=20, bold=True)
    text(s, 0.6, 2.05, 6.0, 2.5, intro, size=16, bullets=True, space_after=7)
    text(s, 0.6, 4.7, 6.0, 0.4, "Rulați", size=20, bold=True)
    code_block(s, 0.6, 5.15, 6.0, len(commands) * 0.32 + 0.42, commands, size=13, name="Commands")
    box(s, 7.0, 1.55, 5.75, 3.5, fill="mist", radius=0.1, name="Exercises card")
    text(s, 7.3, 1.7, 5.2, 0.45, "Exerciții", size=20, bold=True)
    text(s, 7.3, 2.2, 5.2, 2.7, exercises, size=14, bullets=True, space_after=5)
    callout(s, 7.0, 5.2, 5.75, 1.55, outcome, fill="green_t", size=15, name="Expected outcome")
    return s


def link_runs(tb, url):
    for p in tb.text_frame.paragraphs:
        for r in p.runs:
            r.hyperlink.address = url
            setc(r.font.color, "blue")
            r.font.underline = True


def mark_start(slide, x, y, cell):
    r_, c_ = env.cells[env.start]
    text(slide, x + c_ * cell, y + r_ * cell, cell, cell, "S", size=max(12, int(cell * 26)), color="blue", bold=True,
         align="c", anchor="m", name="Start marker")


# ============================================================================ INTRODUCERE
begin("Introducere")

# ---- 1 title
s = prs.slides.add_slide(prs.slide_layouts[5])
dark_background(s)
t = s.shapes.title
t.text_frame.text = "Aproximarea funcțiilor în Reinforcement Learning"
t.left, t.top, t.width, t.height = Inches(0.7), Inches(1.35), Inches(7.3), Inches(2.5)
t.text_frame.word_wrap = True
t.text_frame.vertical_anchor = MSO_ANCHOR.TOP
for r in t.text_frame.paragraphs[0].runs:
    r.font.size = Pt(36)
    setc(r.font.color, "paper")
text(s, 0.7, 4.15, 7.1, 1.15, "Cum un neural network aproximează Q-function-ul și controller-ul. "
     "A doua oră, după tabele.", size=20, color="mint", name="Subtitle")
text(s, 0.7, 5.5, 7.1, 1.2, ["Continuarea cursului „Reinforcement Learning într-o oră”.",
                             "Sutton & Barto, cap. 9 · Silver, Lecture 6 · Lewis & Vrabie, critic și actor"],
     size=14, color="code_dim", name="Credits", space_after=4)
draw_grid(s, env, 8.35, 1.55, 0.72, actions=critic.actions, path=critic.path, dark=True, show_start=False,
          arrow_pt=20, name="Title policy")
text(s, 8.35, 4.6, 4.4, 0.9, "Policy-ul critic-ului, seed 0: aceeași rută de 6 mutări către G pe care Value iteration o calcula din tabel.",
     size=14, color="code_dim", name="Grid caption")
s.notes_slide.notes_text_frame.text = D(
    "Bun venit la a doua oră. Prima oră s-a oprit la tabel: 72 de numere, unul pentru fiecare pereche State–Action, "
    "și fraza „pasul următor este un aproximator”. Azi facem pasul. Criticul este un neural network care calculează Q. "
    "Controller-ul este fie argmax peste acel Q, fie o a doua rețea, actorul, care emite action-ul direct. "
    "Lumea este același gridworld determinist, ca să puteți compara fiecare număr cu Lab 1 și Lab 2. "
    "Timpii: 6 minute de introducere, 12 pentru reprezentare, 14 pentru critic, 6 pentru demonstrația Lab 3, "
    "12 pentru controller, 7 pentru încheiere și 3 pentru întrebări. "
    "Grid-ul din dreapta este policy-ul învățat, nu cel din tabel.")

# ---- 2 the hour
s = content_slide(prs, "Ora pe scurt", notes=(
    "Parcurgeți bara. Blocul galben este demonstrația live: rulați Lab 3, citiți pasul de gradient și cele trei rute, "
    "lăsați exercițiile pentru după. Spuneți ce nu intră: antrenarea unui policy gradient, DQN pe pixeli, un sistem continuu "
    "rulat efectiv, PPO. Le numim, ca să se vadă unde se află ideea, și ne oprim la rețelele mici de pe grid. Circa 3 minute."))
segs = [("Intro", 6, "slate", "6"), ("Reprezentare", 12, "green", "12 min"), ("Critic", 14, "blue", "14 min"),
        ("Lab 3", 6, "amber", "6"), ("Controller", 12, "plum", "12 min"), ("Încheiere", 7, "slate", "7"),
        ("Q&A", 3, "line", "3")]
x0, per_min = 0.55, 12.2 / 60.0
for name, mins, col, lab in segs:
    w_ = mins * per_min
    b_ = box(s, x0, 1.65, max(w_ - 0.05, 0.3), 0.7, fill=col, radius=0.06, name="Time block: " + name)
    label_in(b_, lab, size=13, bold=True, color="ink" if col in ("amber", "line") else "paper")
    text(s, x0 - 0.08, 2.42, w_ + 0.1, 0.4, name, size=12, color="muted", align="c", name="Time label: " + name)
    x0 += w_
text(s, 0.6, 2.9, 12.1, 0.35, "60 de minute · aceeași hartă ca în prima oră · o demonstrație live", size=15, color="muted", italic=True)
goals = [("Înlocuiți tabelul", "Q(s, a) devine Q_{θ}(s, a): un vector de weight-uri, nu o intrare pe pereche"),
         ("Antrenați criticul", "Fitted Q-iteration: backup-ul Bellman, scris prin regresie, nu prin atribuire"),
         ("Construiți controller-ul", "Întâi argmax Q, apoi o a doua rețea care emite action-ul"),
         ("Vedeți când greșește", "Un model liniar rămâne în perete; un seed nefericit pierde ruta")]
for i, (h_, b_) in enumerate(goals):
    tile(s, 0.55 + i * 3.15, 3.45, 3.0, 2.55, h_, b_, num=i + 1, title_pt=16, body_pt=14)

# ---- 3 what stays
s = content_slide(prs, "Ecuația rămâne, tabelul nu", "1", notes=(
    "Nu reluați prima oră. Trei lucruri rămân: tranziția (s, a, r, s′), backup-ul Bellman și TD error-ul. "
    "Se schimbă un singur lucru: unde se scrie numărul. În Lab 2, Q[s, a] era o căsuță. Azi Q_θ(s, a) este ieșirea unei "
    "funcții. Datele sunt tot 72 de tranziții adunate prin probing, cu env.step, fără model. "
    f"Probing-ul atinge E în episodul {data['reached'].get('E')} și G în episodul {data['reached'].get('G')}, ca în Lab 2. "
    "Circa 3 minute."))
rows_ = [["Rămâne din prima oră", "Se schimbă azi"],
         ["Tranziția observată (s, a, r, s′)", "Numărul nu mai stă într-o căsuță Q[s, a]"],
         ["Backup-ul  r + γ max_{a′} Q(s′, a′)", "Îl produce o funcție Q_{θ}, cu weight-urile θ"],
         ["TD error: predicție față de backup", "Gradientul lui modifică weight-uri comune"],
         ["Probing: fiecare pereche State–Action", "Aceeași funcție este și controller, sau antrenează unul"]]
table(s, 0.6, 1.6, [6.05, 6.05], rows_, size=16, row_h=0.72, head_h=0.55,
      header_fills={0: "green", 1: "plum"}, name="Ce rămâne")
callout(s, 0.6, 5.9, 12.1, 0.95,
        f"Harta nu s-a mărit: {numere(N_PAIRS)} ar încăpea într-un tabel. "
        "O ținem mică ca să puteți verifica aproximarea față de Q* din Lab 1. "
        "Câștigul apare când State-ul este un vector pe care nu îl puteți enumera.",
        fill="amber_t", size=16)

# ============================================================================ REPREZENTAREA
begin("1 · Reprezentarea")

# ---- 4 Q as a function
s = content_slide(prs, "Q devine o funcție cu weight-uri", "1", notes=(
    "Scrieți cele două forme una sub alta. Tabelul este o funcție și el, doar că are un parametru separat pentru fiecare "
    "intrare și nu poate răspunde pentru un state pe care nu l-a văzut ca indice. Aproximatorul are un vector θ mult mai "
    "scurt decât tabelul atunci când state-urile sunt multe, și este definit pentru orice intrare din forma pe care o "
    f"așteaptă. Pe harta noastră θ al criticului are {critic.n_params} de componente, iar tabelul are {N_PAIRS}. "
    "Rețeaua este mai mare decât tabelul. Spuneți asta tare, ca să nu rămână impresia că orice neural network comprimă. "
    "Compresia nu este lecția de azi. Lecția este că intrarea poate fi un vector. Circa 3 minute."))
box(s, 0.6, 1.6, 5.9, 2.3, fill="mist", radius=0.12, name="Tabel")
text(s, 0.85, 1.75, 5.4, 0.4, "Lab 2 · tabel", size=16, bold=True, color="slate")
math(s, 0.85, 2.2, 5.4, 0.7, "Q[s, a]   ∈   R", size=28, align="c")
text(s, 0.9, 3.0, 5.3, 0.7, f"O intrare pentru fiecare pereche. Aici, {numere(N_PAIRS)}.", size=16, align="c")
box(s, 6.8, 1.6, 5.9, 2.3, fill="green_t", radius=0.12, name="Functie")
text(s, 7.05, 1.75, 5.4, 0.4, "Azi · funcție", size=16, bold=True, color="green")
math(s, 7.05, 2.2, 5.4, 0.7, "Q_{θ}(s, a)   ∈   R", size=28, align="c")
text(s, 7.1, 3.0, 5.3, 0.7, "θ este un vector de weight-uri. Aceeași formulă pentru orice s.", size=16, align="c")
for i, (h_, d_) in enumerate([
        ("Parametri puțini sau mulți", "Pe o hartă uriașă, |θ| ≪ |S|·|A|. Pe harta noastră, criticul are mai multe weight-uri decât tabelul."),
        ("Definită peste tot", "Dai un vector de intrare, primești un Q. Nu există „căsuță lipsă”."),
        ("Eroare de aproximare", "Familia de funcții poate să nu conțină Q*. Rămâne un rest. RMSE-ul din Lab 3 este restul acela.")]):
    x = 0.6 + i * 4.15
    box(s, x, 4.2, 3.95, 2.15, fill="mist", radius=0.1, name="Card: " + h_)
    text(s, x + 0.2, 4.35, 3.55, 0.7, h_, size=16, bold=True)
    text(s, x + 0.2, 5.1, 3.55, 1.05, d_, size=14)

# ---- 5 one-hot is a table
s = content_slide(prs, "One-hot este chiar tabelul", "1", notes=(
    "Merită văzut, pentru că altfel rețeaua pare o specie nouă. Dacă intrarea este vectorul one-hot e_s, un layer liniar "
    "are exact un weight pentru fiecare pereche (s, a): Q(s, a) = θ_{s,a}. Gradientul erorii pătratice pe action-ul "
    "încercat este (Q − target) pe weight-ul acela și zero în rest. Actualizarea este Q-learning din Lab 2, cu α. "
    "Deci Lab 2 era deja un neural network, fără hidden layer și fără weight-uri comune. Nu derivați pe tablă mai mult "
    "de atât. Circa 3 minute."))
math(s, 0.6, 1.55, 12.1, 0.6, "Intrare one-hot e_{s}    ⇒    Q(s, a) = θ_{s, a}", size=26, align="c")
rows_ = [["", "Unde se scrie", "Cine se mișcă la un update"],
         ["Lab 2", "Q[s, a] ← Q[s, a] + α · TD error", "O singură căsuță"],
         ["Layer liniar, intrare one-hot", "θ_{s,a} ← θ_{s,a} + α · (target − Q)", "Un singur weight"],
         ["Cu feature-uri comune", "θ ← θ − α ∇_{θ} ½ (Q_{θ}(s, a) − target)^{2}", "Toate weight-urile, deci toate state-urile"]]
table(s, 0.6, 2.35, [3.3, 5.5, 3.3], rows_, size=15, row_h=0.7, head_h=0.55, fonts={1: "Cambria"},
      fills={(3, 1): "green_t", (3, 2): "green_t"}, name="One-hot versus features")
callout(s, 0.6, 5.7, 12.1, 1.1,
        "**Semi-gradient:** target-ul r + γ max Q(s′, ·) este tratat ca o constantă. "
        "Derivăm doar predicția Q_{θ}(s, a), nu și backup-ul. Altfel gradientul trece prin maximul de la pasul următor "
        "și actualizarea nu mai este cea din Q-learning.",
        fill="amber_t", size=16)

# ---- 6 coordinates
s = content_slide(prs, "(x, y) în loc de un indice", "1", notes=(
    "One-hot nu generalizează: o celulă nouă este un indice nou, cu weight-uri noi, la zero. Coordonatele fac opusul. "
    "Două celule apropiate au vectori (x, y) apropiați, deci împart aceleași weight-uri și primesc Q-uri apropiate. "
    "Asta este generalizarea. Reversul ei se numește interferență: un update făcut pentru o celulă mișcă predicția "
    "celorlalte. Calculul cu un singur pas, pe modelul liniar, este la tranziția în E. Scalarea în [−1, 1] este doar "
    "ca weight-urile să nu trebuiască să compenseze un 0…5. Circa 3 minute."))
box(s, 0.6, 1.6, 5.9, 3.5, fill="mist", radius=0.12, name="One-hot card")
text(s, 0.85, 1.8, 5.4, 0.45, "One-hot", size=22, bold=True)
text(s, 0.9, 2.4, 5.3, 2.4, ["Un 1 pe poziția s, restul 0",
                             "Nicio legătură între celula (2, 3) și celula (2, 4)",
                             "Un state care nu e în listă nu are intrare",
                             "Update-ul nu poate atinge alt state"], size=16, bullets=True, space_after=8)
box(s, 6.8, 1.6, 5.9, 3.5, fill="green_t", radius=0.12, name="Coordinates card")
text(s, 7.05, 1.8, 5.4, 0.45, "Coordonate (x, y)", size=22, bold=True, color="green")
text(s, 7.1, 2.4, 5.3, 2.4, ["x, y ∈ [−1, 1], colțul stânga-sus este (−1, −1)",
                             "Celulele vecine au intrări vecine",
                             "Poți evalua și un punct care nu e centrul unei celule",
                             "Un update mișcă Q pe toată harta"], size=16, bullets=True, space_after=8)
callout(s, 0.6, 5.35, 12.1, 1.4,
        "Generalizarea și interferența sunt același fapt. Weight-urile nu știu „pentru ce celulă” au fost atinse. "
        "Știu doar în ce direcție din spațiul de intrare să crească Q. "
        "Lab 3 folosește coordonatele și pentru critic, și pentru actor.",
        fill="mist", size=18)

# ---- 7 linear fails
s = content_slide(prs, "Un model liniar nu ocolește peretele", "1", notes=(
    f"Acesta este aproximatorul pe care îl puteți rezolva cu mâna. φ(s) = [1, x, y], douăsprezece weight-uri, "
    f"patru drepte. La fiecare sweep înlocuim W cu soluția de least squares a target-urilor Bellman: nu rămâne "
    f"eroare de optimizare, rămâne doar ce un plan nu poate reprezenta. După {L3.LINEAR_SWEEPS} de sweep-uri, "
    f"RMSE față de Q* este {linear.rmse:.3f}. Doar {linear.n_optimal} din {N_LIVE} săgeți sunt optime, iar din S "
    "agentul urcă în perete și rămâne acolo. Q* nu este o funcție liniară de (x, y): peretele rupe harta în două. "
    "Circa 4 minute."))
math(s, 0.6, 1.5, 7.2, 0.55, "Q(s, a) = φ(s) · w_{a}", size=26)
math(s, 0.6, 2.1, 7.2, 0.5, "φ(s) = [ 1,  x,  y ]", size=22, color="muted")
text(s, 0.6, 2.75, 7.2, 1.5, [f"**{linear.n_params} weight-uri**, câte 3 pentru fiecare action",
                              "Fiecare sweep este un least squares exact, nu un pas de gradient",
                              f"RMSE final **{linear.rmse:.3f}** · săgeți optime **{linear.n_optimal} din {N_LIVE}**",
                              "Din S: sus, apoi în perete. Ruta nu se termină."],
     size=16, bullets=True, space_after=6)
draw_grid(s, env, 8.5, 1.55, 0.62, values=linear.q.max(1), actions=linear.actions, path=linear.path,
          vmax=1.0, arrow_pt=16, value_pt=9, name="Linear policy")
text(s, 8.3, 4.15, 4.5, 0.7, "max_{a} Q liniar. Săgeata din coridorul de jos intră în perete.", size=13, color="muted", align="c")
callout(s, 0.6, 5.15, 12.1, 1.6,
        f"**Nu este un eșec de antrenare.** Target-ul fiecărui sweep este potrivit exact. "
        f"Familia de funcții nu conține Q*. Lângă E, action-ul „left” plătește {f3(Q_star[S_EXIT, LEFT])}; "
        f"planul spune {f3(linear.q[S_EXIT, LEFT])}. Nu are cum să fie mic lângă ieșire și, în același timp, "
        f"să crească de-a lungul coridorului către G.",
        fill="coral_t", size=16)

# ---- 8 anatomy
s = content_slide(prs, "Criticul: două hidden layer-uri ReLU", "1", notes=(
    f"Desenați cutiile. Intrarea are 2 numere. Primul hidden layer are 16 unități ReLU, al doilea încă 16, "
    f"ieșirea are 4 numere, câte un Q pentru ↑ ↓ ← →. Numărați weight-urile cu studenții: "
    f"2·16+16 și 16·16+16 și 16·4+4 fac {critic.n_params}. ReLU(z) = max(0, z). O unitate închisă are gradient zero: "
    "nu învață din exemplul acela. De aceea inițializarea contează, și de aceea seed-ul 6, mai târziu, nu repetă seed-ul 0. "
    "Nu deschideți discuția despre alte activări. Circa 3 minute."))
layers = [("input", "2", "(x, y)", "blue_t", "blue"),
          ("h1", "16", "ReLU", "green_t", "green"),
          ("h2", "16", "ReLU", "green_t", "green"),
          ("out", "4", "Q pe action", "amber_t", "amber")]
for i, (nm, n, sub, fill, col) in enumerate(layers):
    x = 0.7 + i * 3.15
    box(s, x, 1.6, 2.5, 1.7, fill=fill, line=col, radius=0.12, name="Layer " + nm)
    text(s, x, 1.75, 2.5, 0.7, n, size=32, bold=True, align="c", color=col)
    text(s, x + 0.1, 2.5, 2.3, 0.55, sub, size=16, align="c")
    if i < 3:
        line(s, x + 2.5, 2.45, x + 3.15, 2.45, color="ink", w=2.0, arrow=True)
text(s, 0.7, 3.5, 12.0, 0.7,
     f"Weight-uri:  (2·16 + 16)  +  (16·16 + 16)  +  (16·4 + 4)  =  **{critic.n_params}**.   "
     f"Tabelul are **{N_PAIRS}**.",
     size=18, align="c")
for i, (h_, d_, fill) in enumerate([
        ("Forward", "h = ReLU(h W + b) pe hidden layer-uri. Ultimul layer rămâne liniar: Q poate fi negativ.", "mist"),
        ("De ce două layer-uri", "Cu un singur hidden layer, ruta nu iese stabil pe harta asta. Cu două, la seed 0, iese.", "mist"),
        ("Mai mare decât tabelul", "388 față de 72. Aceeași rețea, fără nicio modificare, primește și un state continuu.", "green_t")]):
    x = 0.6 + i * 4.15
    box(s, x, 4.4, 3.95, 2.0, fill=fill, radius=0.1, name="Note: " + h_)
    text(s, x + 0.2, 4.55, 3.55, 0.45, h_, size=16, bold=True)
    text(s, x + 0.2, 5.1, 3.55, 1.1, d_, size=14)

# ============================================================================ CRITICUL
begin("2 · Criticul")

# ---- 9 fitted Q
s = content_slide(prs, "Fitted Q-iteration", "2", notes=(
    "Puneți pseudocodul lângă value iteration și lăsați-i să vadă că diferă o singură linie: scrierea. "
    "Value iteration scrie maximul în V[s]. Fitted Q-iteration îngheață target-urile y pentru toate cele 72 de tranziții, "
    "apoi potrivește weight-urile ca Q_θ(s, a) să se apropie de y. Înghețarea este importantă. Dacă target-ul s-ar mișca "
    "la fiecare minibatch, ați deriva prin el sau ați urmări o țintă care fuge. DQN numește copia înghețată target network; "
    "aici copia ține un sweep întreg. Înăuntrul sweep-ului, pasul este Adam pe eroarea pătratică: direcția este gradientul, "
    "α se scalează singur pe fiecare weight. Nu derivați Adam. Circa 4 minute."))
for i, (h_, col, fill, formula, foot) in enumerate([
        ("Value iteration", "blue", "blue_t",
         "V(s) ← max_{a} [ r + γ V(s′) ]", "scriere într-o căsuță, din model"),
        ("Fitted Q-iteration", "green", "green_t",
         "y ← r + γ max_{a′} Q_{θ}(s′, a′)", "y înghețat, apoi regresie pe θ")]):
    x = 0.6 + i * 6.35
    box(s, x, 1.55, 6.05, 2.15, fill=fill, radius=0.12, name="Card: " + h_)
    text(s, x + 0.25, 1.7, 5.55, 0.4, h_, size=20, bold=True, color=col)
    math(s, x + 0.15, 2.2, 5.75, 0.7, formula, size=20, align="c")
    text(s, x + 0.25, 3.0, 5.55, 0.45, foot, size=15, color="muted", align="c")
code_block(s, 0.6, 3.95, 7.5, 2.55, [
    "for sweep in range(25):",
    "    y = r + gamma * max(Q(s2)) * (not done)  # înghețat",
    "    # câțiva pași de Adam pe (Q(s, a) - y)^2",
    "    fit(theta, transitions, y)",
], size=15, name="Pseudocod fitted Q")
box(s, 8.35, 3.95, 4.35, 2.55, fill="amber_t", radius=0.1, name="De ce inghetat")
text(s, 8.55, 4.1, 4.0, 0.4, "De ce se îngheață y", size=16, bold=True)
text(s, 8.55, 4.6, 4.0, 1.7, "Semi-gradientul nu trece prin max-ul de la s′. Target-ul este o etichetă, ca la regresie, până la sweep-ul următor.",
     size=15)

# ---- 10 one gradient step
s = content_slide(prs, "Un pas pe care îl verificați cu mâna", "2", notes=(
    "Acesta este singurul calcul pe care cereți să-l refacă pe hârtie. Weight-urile liniare pornesc de la zero, deci Q este 0. "
    f"Tranziția este cea din trace-ul Lab 2: din (3, 1), left, reward {step['r']:+.2f}, ajunge pe E, care e terminal, "
    f"deci target-ul este chiar reward-ul, {step['target']:+.3f}. Cu α = {step['alpha']}, w_left devine α·target·φ. "
    f"Q în celula actualizată urcă doar la {step['q_here']:+.4f}, nu la 0.300: un pas nu este o scriere. "
    f"În S, pe care pasul acesta nu l-a vizitat, Q_left devine {step['q_start']:+.4f}, pentru că φ(S)·φ(s) nu este zero. "
    f"Toate cele {N_LIVE} celule se mișcă. Un tabel ar fi scris 0.300 într-o căsuță și atât. Circa 4 minute."))
box(s, 0.55, 1.5, 4.3, 2.35, fill="blue_t", radius=0.1, name="Tranzitie")
text(s, 0.75, 1.6, 3.95, 0.4, "Tranziția, ca în Lab 2", size=15, bold=True, color="blue")
text(s, 0.75, 2.1, 3.95, 1.5, ["s = (3, 1),  a = left", f"r = {step['r']:+.2f},  s′ = E, terminal",
                               f"target = {step['target']:+.3f}", "Q înainte = 0, pentru că w = 0"],
     size=15, space_after=2)
box(s, 5.05, 1.5, 7.7, 2.35, fill="mist", radius=0.1, name="Formula pas")
text(s, 5.25, 1.6, 7.3, 0.35, "Pasul, doar pe weight-urile action-ului left", size=15, bold=True)
math(s, 5.2, 2.0, 7.4, 0.55, "w  ←  w + α (target − Q) φ(s)", size=22, align="c")
math(s, 5.2, 2.6, 7.4, 0.5, f"φ(s) = {phi_txt(phi)}", size=18, align="c", color="muted")
math(s, 5.2, 3.15, 7.4, 0.5, f"α = {step['alpha']}", size=18, align="c", color="muted")
for i, (h_, val, d_) in enumerate([
        ("În celula atinsă", f4(step["q_here"]), "un tabel ar fi scris +0.300"),
        ("În S, neatins", f4(step["q_start"]), f"φ(S) = {phi_txt(phi_S)}"),
        ("Celule mișcate", f"{step['moved']} din {step['n_live']}", "un tabel ar fi mișcat 1")]):
    x = 0.55 + i * 4.2
    box(s, x, 4.1, 4.0, 1.55, fill="green_t" if i == 2 else "mist", radius=0.1, name="Efect: " + h_)
    text(s, x + 0.15, 4.2, 3.7, 0.35, h_, size=14, bold=True, color="muted")
    math(s, x + 0.15, 4.5, 3.7, 0.55, val, size=26, align="c")
    text(s, x + 0.15, 5.1, 3.7, 0.4, d_, size=13, align="c", color="muted")
callout(s, 0.55, 5.85, 12.2, 0.95,
        "Verificare: ΔQ(s̃) = α · target · φ(s) · φ(s̃). "
        f"Pentru S, produsul scalar dă {f4(step['q_start'])}. Același mecanism, prin ReLU, mișcă și criticul.",
        fill="amber_t", size=16)

# ---- 11 three grids
s = content_slide(prs, "Aceeași hartă, trei Q-uri", "2", notes=(
    f"Lăsați grila să vorbească. Stânga este Q* din Value iteration, return {f3(V[env.start])} în 6 mutări. "
    f"Mijlocul este planul: {linear.n_optimal} din {N_LIVE} săgeți coincid cu o acțiune optimă, dar din S ruta intră în perete. "
    f"Dreapta este criticul, seed 0: toate cele {N_LIVE} săgeți sunt optime, ruta are {critic.moves} mutări și return-ul "
    f"{f3(critic.ret)}, același număr cu V(S). Culorile folosesc aceeași scală, de la −1 la +1, ca un Q mai mic să se vadă mai palid. "
    "Circa 3 minute."))
panels = [("Q*   ·   Value iteration", Q_star.max(1), act_star, exact_path),
          (f"Liniar   ·   {linear.n_optimal}/{N_LIVE} săgeți", linear.q.max(1), linear.actions, linear.path),
          (f"Critic   ·   return {f3(critic.ret)}", critic.q.max(1), critic.actions, critic.path)]
for i, (cap, vals, acts, path_) in enumerate(panels):
    x = 0.45 + i * 4.25
    text(s, x, 1.45, 3.7, 0.4, cap, size=14, bold=True, align="c")
    draw_grid(s, env, x + 0.15, 1.95, 0.58, values=vals, actions=acts, path=path_, vmax=1.0,
              arrow_pt=15, value_pt=8, name="Compare " + cap)
callout(s, 0.45, 5.55, 12.4, 1.2,
        f"Criticul are {critic.n_params} de weight-uri și RMSE {critic.rmse:.3f}. "
        f"Planul are {linear.n_params} și RMSE {linear.rmse:.3f}, și nu ajunge la G. "
        "Return-ul rutei critic-ului este egal cu V(S): pe această inițializare, aproximarea nu a costat nimic din rută.",
        fill="green_t", size=16)

# ---- 12 curves
s = content_slide(prs, "Target-ul se mută între sweep-uri", "2", notes=(
    f"Curba gri coboară și se oprește: least squares nu mai are ce să scadă, planul și-a atins limita, RMSE {linear.rmse:.3f}. "
    f"Curba verde urcă întâi, cu un maxim de {peak:.3f} la sweep-ul {peak_at}, apoi coboară la {critic.rmse:.3f}. "
    "Urcarea nu este un bug. Target-ul sweep-ului k+1 este construit din rețeaua sweep-ului k, care încă nu a adus reward-ul "
    "lui G înapoi pe toată harta. Aceleași sweep-uri, în Value iteration, umpleau harta dinspre ieșiri. "
    f"La final, pe celula de lângă E, Q*(left) = {f4(Q_star[S_EXIT, LEFT])}, criticul dă {f4(critic.q[S_EXIT, LEFT])}, "
    f"planul dă {f4(linear.q[S_EXIT, LEFT])}. Circa 3 minute."))
xs_l = list(range(1, len(linear.rmse_curve) + 1))
xs_c = list(range(1, len(critic.rmse_curve) + 1))
xy_chart(s, 0.4, 1.45, 7.3, 4.5,
         [("liniar, 12 weight-uri", xs_l, linear.rmse_curve, "slate"),
          ("critic, neural network", xs_c, critic.rmse_curve, "green")],
         xtitle="sweep", ytitle="RMSE față de Q*", ymin=0, ymax=0.8, ymajor=0.2,
         xmin=1, xmax=L3.LINEAR_SWEEPS, legend=True,
         description="RMSE-ul planului se oprește; RMSE-ul criticului coboară sub el după o urcare",
         name="RMSE pe sweep")
box(s, 7.9, 1.6, 4.8, 4.3, fill="mist", radius=0.1, name="Q langa E")
text(s, 8.1, 1.75, 4.4, 0.7, "Q(left) lângă E, după antrenare", size=16, bold=True)
q_rows = [("Q*", f4(Q_star[S_EXIT, LEFT]), "green"),
          ("Critic", f4(critic.q[S_EXIT, LEFT]), "green"),
          ("Liniar", f4(linear.q[S_EXIT, LEFT]), "coral")]
for i, (nm, val, col) in enumerate(q_rows):
    y = 2.6 + i * 0.95
    text(s, 8.15, y, 1.6, 0.7, nm, size=16, bold=True, anchor="m", color=col)
    math(s, 9.7, y, 2.7, 0.7, val, size=24, color=col)
callout(s, 0.55, 6.05, 12.15, 0.75,
        "Target-ul corect pe tranziția în E este +0.300. Planul nu îl poate nimeri fără să strice restul hărții.",
        fill="mist", size=16)

# ---- 13 lab
lab_slide(
    "Lab 3: Q și controller-ul, fără tabel",
    ["Citiți pasul liniar: Q se mișcă și în S",
     "Comparați ruta liniară cu ruta criticului",
     "Actorul copiază săgețile criticului, nu reward-ul",
     "Schimbați seed-ul și numărați cine mai ajunge la G"],
    ["python lab3_function_approximation.py",
     "python lab3_function_approximation.py --seed 6",
     "python lab3_function_approximation.py --seeds 10"],
    ["De ce se schimbă Q(S) la un pas făcut în altă celulă?",
     "Planul are 12 weight-uri și pierde. Ce îi lipsește, de fapt?",
     "La --seed 6, actorul repară ruta sau copiază greșeala?",
     "Adăugați x², y², xy în φ. Ajunge planul la G?"],
    "Demonstrație live, circa 6 minute. Rulați varianta implicită. Citiți împreună pasul de gradient: "
    f"target {step['target']:+.3f}, Q în celula atinsă {step['q_here']:+.4f}, Q în S {step['q_start']:+.4f}, "
    f"{step['moved']} din {step['n_live']} celule mișcate. Apoi cele trei rute. "
    f"Criticul și actorul ajung la G cu return {f3(critic.ret)}. Planul rămâne în perete, "
    f"cu {linear.n_optimal} din {N_LIVE} săgeți optime. Dacă aveți un minut, --seed 6.",
    f"Seed 0: critic RMSE {critic.rmse:.3f}, {critic.n_optimal}/{N_LIVE} săgeți, return {f3(critic.ret)}. "
    f"Actorul copiază {actor.n_copied}/{N_LIVE}. Liniarul nu ajunge la nicio ieșire.")

# ============================================================================ CONTROLLER-UL
begin("3 · Controller-ul")

# ---- 14 argmax
s = content_slide(prs, "Controller-ul ascuns în argmax", "3", notes=(
    "Prima formă de controller nu are weight-uri noi. Policy-ul este π(s) = argmax_a Q_θ(s, a). "
    "Criticul de la seed 0 dă deja ruta optimă, deci acest controller este gata în momentul în care criticul s-a potrivit. "
    "Prețul: ca să alegeți action-ul, evaluați Q pentru fiecare action. La patru săgeți, prețul este nimic. "
    "La un u continuu, nu aveți o listă pe care să o parcurgeți. De aceea există actorul, pe slide-ul următor. "
    "Lewis și Vrabie numesc legea u = h(x) regulator; argmax este regulatorul cât timp action-ul este discret. Circa 3 minute."))
math(s, 0.6, 1.55, 12.1, 0.7, "π(s)  =  argmax_{a}  Q_{θ}(s, a)", size=32, align="c")
for i, (h_, d_, fill, col) in enumerate([
        ("Nu învață nimic nou", "Săgeata este un index în vectorul de 4 ieșiri. Niciun weight în plus.", "mist", "ink"),
        ("Are nevoie de toate action-urile", "Pentru a alege, criticul trebuie evaluat pe fiecare a. Aici sunt 4.", "mist", "ink"),
        ("Pe seed 0 este deja optim", f"{critic.moves} mutări până la G, return {f3(critic.ret)}, egal cu V(S).", "green_t", "green")]):
    x = 0.6 + i * 4.15
    box(s, x, 2.5, 3.95, 2.15, fill=fill, radius=0.1, name="Argmax: " + h_)
    text(s, x + 0.2, 2.65, 3.55, 0.7, h_, size=16, bold=True, color=col)
    text(s, x + 0.2, 3.4, 3.55, 1.05, d_, size=15)
callout(s, 0.6, 4.95, 12.1, 1.75,
        "Acesta este controller-ul pe care îl aveți și în Lab 2, unde argmax se făcea pe un rând din tabel. "
        "Neural network-ul nu schimbă regula de decizie. Schimbă doar de unde vine numărul din interiorul argmax-ului. "
        "Când action-ul nu mai este o săgeată dintr-o listă, argmax-ul nu se mai poate calcula, și controller-ul trebuie să fie el însuși o funcție.",
        fill="amber_t", size=16)

# ---- 15 actor
s = content_slide(prs, "Controller-ul ca a doua rețea", "3", notes=(
    f"Actorul este un neural network mai mic: (x, y) → 16 ReLU → 4 logits, {actor.n_params} weight-uri. "
    "Softmax-ul face din logits o distribuție, iar action-ul executat este argmax-ul logit-ului, nu o extragere aleatoare: "
    "lumea rămâne deterministă. Eticheta nu este un reward. Eticheta este action-ul greedy al criticului. "
    f"Pe seed 0 copiază {actor.n_copied} din {N_LIVE} și return-ul este din nou {f3(actor.ret)}. "
    "Spuneți explicit: actorul nu poate fi mai bun decât criticul pe care îl imită. Loss-ul este cross-entropy. "
    "Policy gradient, formula cu avantajul, este ce ați folosi dacă nimeni nu v-ar da eticheta. Nu o rulăm. Circa 4 minute."))
text(s, 0.6, 1.5, 7.3, 0.4, "Actorul imită argmax-ul criticului", size=18, bold=True)
math(s, 0.6, 2.0, 7.3, 0.6, "a*(s) = argmax_{a} Q_{θ}(s, a)", size=22, align="c")
math(s, 0.6, 2.65, 7.3, 0.6, "φ  ←  φ − α ∇_{φ} [ −log π_{φ}(a*(s) | s) ]", size=18, align="c")
text(s, 0.6, 3.4, 7.3, 1.3, [f"Arhitectură: 2 → 16 ReLU → 4 logits, **{actor.n_params} weight-uri**",
                             f"Seed 0: copiază **{actor.n_copied} din {N_LIVE}**, ruta până la G",
                             "Nu primește r. Profesorul lui este criticul."],
     size=16, bullets=True, space_after=5)
draw_grid(s, env, 8.55, 1.55, 0.62, actions=actor.actions, path=actor.path, vmax=1.0,
          values=critic.q.max(1), arrow_pt=16, value_pt=8, name="Actor policy")
text(s, 8.4, 4.15, 4.5, 0.55, "Săgeți: actorul. Numere: Q-ul criticului.", size=13, color="muted", align="c")
callout(s, 0.6, 4.9, 12.1, 1.8,
        "Fără etichetă, același actor s-ar antrena cu policy gradient: "
        "φ ← φ + α ∇_{φ} log π_{φ}(a | s) · (Q(s, a) − V(s)). "
        "Avantajul spune dacă action-ul încercat a fost mai bun decât media. "
        "Azi eticheta există, pentru că action-ul este discret și criticul știe argmax-ul. Varianta cu avantaj rămâne pentru cursul următor.",
        fill="mist", size=16)

# ---- 16 ADP loop
s = content_slide(prs, "Critic și actor, bucla din ADP", "3", notes=(
    "Lewis și Vrabie despart exact așa: criticul aproximează cost-to-go, actorul aproximează legea de comandă. "
    "La ei semnul este invers, pentru că minimizează un cost; la noi criticul aproximează Q și actorul copiază argmax-ul. "
    "Parcurgeți cele patru căsuțe în cerc. Datele vin din probing, nu din actor: dacă ați colecta doar cu actorul greedy, "
    "ați repeta blocajul din Lab 2. Probing-ul este persistent excitation, același cuvânt ca în prima oră. "
    "Actorul se antrenează după critic, pe state-urile vii, nu de-a lungul unei singure traiectorii. Circa 3 minute."))
steps = [("1", "Date", " probing dă (s, a, r, s′) pentru toate cele 72 de perechi", "blue"),
         ("2", "Critic", "fitted Q-iteration scrie backup-ul în weight-urile θ", "green"),
         ("3", "Actor", "cross-entropy către argmax_{a} Q_{θ}(s, a)", "plum"),
         ("4", "Controller", "la rulare, a = π_{φ}(s). Criticul poate fi dat deoparte.", "amber")]
for i, (n, h_, d_, col) in enumerate(steps):
    x = 0.5 + (i % 4) * 3.2
    y = 1.6
    box(s, x, y, 3.0, 2.7, fill="mist", radius=0.1, name="Bucla " + h_)
    badge(s, x + 0.2, y + 0.2, 0.5, n, fill=col, size=16)
    text(s, x + 0.85, y + 0.25, 1.95, 0.45, h_, size=18, bold=True, anchor="m")
    text(s, x + 0.2, y + 1.0, 2.6, 1.45, d_, size=15)
callout(s, 0.5, 4.55, 12.3, 2.15,
        "Ordinea contează. Actorul antrenat pe un critic neterminat învață săgețile greșite la fel de conștiincios "
        f"ca pe cele bune: pe seed {bad['seed']} copiază {actor_bad.n_copied} din {N_LIVE} etichete, "
        "și rata lui este rata criticului, nu ruta optimă. "
        "Controller-ul nu corectează criticul. Îl execută.",
        fill="amber_t", size=17)

# ---- 17 continuous
s = content_slide(prs, "Când action-ul este un număr real", "3", notes=(
    "Nu rulați nimic aici. Slide-ul există ca să nu rămână impresia că argmax-ul este forma generală a controller-ului. "
    "Dacă u este o forță, u ∈ [−1, 1], nu aveți patru coloane în Q. Două variante, ambele în literatura pe care o citați: "
    "criticul primește și u ca intrare, Q_θ(x, u), iar actorul μ_φ(x) este antrenat să urce Q, prin lanțul de derivate; "
    "sau actorul este singur și învață din return, policy gradient. Lewis și Vrabie scriu a doua rețea direct ca legea de "
    "feedback. Lab-ul de azi este cazul discret al aceleiași despărțiri. Circa 3 minute."))
box(s, 0.6, 1.6, 6.0, 3.3, fill="mist", radius=0.12, name="Discret")
text(s, 0.85, 1.8, 5.5, 0.45, "Azi · action discret", size=18, bold=True, color="green")
math(s, 0.85, 2.4, 5.5, 0.7, "a ∈ {↑, ↓, ←, →}", size=24, align="c")
text(s, 0.9, 3.2, 5.4, 1.4, "Argmax-ul este o comparare de 4 numere. Actorul este opțional: imită compararea, ca să existe o rețea care emite action-ul.",
     size=16)
box(s, 6.9, 1.6, 5.8, 3.3, fill="plum_t", radius=0.12, name="Continuu")
text(s, 7.15, 1.8, 5.3, 0.45, "Următorul pas · u continuu", size=18, bold=True, color="plum")
math(s, 7.15, 2.4, 5.3, 0.7, "u = μ_{φ}(x)  ∈  R", size=24, align="c")
text(s, 7.2, 3.2, 5.2, 1.4, "Nu există listă de parcurs. Actorul este singurul controller. Se antrenează urcând Q_{θ}(x, μ_{φ}(x)).",
     size=16)
callout(s, 0.6, 5.15, 12.1, 1.55,
        "Actualizarea actorului continuu, deterministic policy gradient:   "
        "φ ← φ + α ∇_{φ} Q_{θ}( x, μ_{φ}(x) ).   "
        "Gradientul trece prin critic, apoi prin actor. Nu o implementăm azi; formula spune doar de ce a doua rețea nu este un moft.",
        fill="mist", size=16)

# ---- 18 fragility
s = content_slide(prs, "Aceeași ecuație, alt seed", "3", notes=(
    f"Din {len(rows)} de inițializări, {n_goal} critici ajung la G și {n_actor_goal} actori la fel, pentru că actorul copiază. "
    f"Seed {bad['seed']} este excepția: RMSE-ul criticului rămâne {critic_bad.rmse:.3f}, aproape de {critic.rmse:.3f} al seed-ului 0, "
    f"dar doar {critic_bad.n_optimal} din {N_LIVE} săgeți sunt optime și ruta din S nu atinge G. "
    f"Actorul copiază {actor_bad.n_copied} din {N_LIVE} etichete, inclusiv pe cele greșite. "
    "Trei ingrediente poartă un nume, triada mortală: function approximation, bootstrapping și date off-policy. "
    "Le avem pe toate. Înghețarea target-ului și setul complet de perechi le țin în frâu în 9 cazuri din 10, nu în 10. "
    "Un tabel cu α = 1 nu depinde de un seed. Circa 4 minute."))
text(s, 0.55, 1.5, 6.2, 0.4, f"Seed {bad['seed']} · criticul pierde ruta", size=16, bold=True, align="c")
draw_grid(s, env, 1.15, 2.0, 0.7, values=critic_bad.q.max(1), actions=critic_bad.actions, path=critic_bad.path,
          vmax=1.0, arrow_pt=18, value_pt=10, name="Seed rau")
text(s, 0.55, 4.95, 6.2, 0.7,
     f"RMSE {critic_bad.rmse:.3f} · săgeți optime {critic_bad.n_optimal} din {N_LIVE} · actorul copiază greșeala",
     size=14, align="c", color="coral")
box(s, 7.0, 1.55, 5.7, 4.15, fill="coral_t", radius=0.12, name="Triada")
text(s, 7.25, 1.7, 5.2, 0.45, "Triada care poate diverge", size=18, bold=True, color="coral")
text(s, 7.25, 2.3, 5.2, 3.1, ["**Function approximation** — weight-uri comune",
                              "**Bootstrapping** — target-ul folosește propriul Q",
                              "**Off-policy** — datele vin din probing, nu din actor",
                              "Împreună, actualizarea poate să nu conveargă. Aici ratează 1 seed din 10."],
     size=15, bullets=True, space_after=8)
callout(s, 0.55, 5.8, 12.15, 0.95,
        f"Cele {n_goal} seed-uri reușite nu anulează seed-ul {bad['seed']}. "
        "Un tabel terminat rămâne terminat. O aproximare bună pe RMSE poate avea săgeata greșită exact pe rută.",
        fill="mist", size=16)

# ============================================================================ ÎNCHEIERE
begin("Încheiere")

# ---- 19 four ways
s = content_slide(prs, "Patru feluri de a ține minte Q și π", notes=(
    "Citiți tabelul pe rândul de jos, de la stânga la dreapta. Tabelul și criticul de la seed 0 au același return. "
    "Planul, cu de douăsprezece ori mai puține numere decât tabelul, nu are rută. Actorul nu stochează Q deloc: "
    "stochează săgețile pe care criticul i le-a dictat. Circa 2 minute."))
rows_ = [["", "Tabel (Lab 1–2)", "Liniar", "Critic", "Actor"],
         ["Câte numere", numere(N_PAIRS), f"{linear.n_params} weight-uri", f"{critic.n_params} weight-uri",
          f"{actor.n_params} weight-uri"],
         ["Intrarea", "indicele s", "φ = [1, x, y]", "(x, y)", "(x, y)"],
         ["Ce se potrivește", "backup-ul, într-o căsuță", "target-ul, least squares", "target-ul, Adam",
          "eticheta argmax Q"],
         ["Din S, seed 0", f"G,  {f3(V[env.start])}", "rămâne în perete", f"G,  {f3(critic.ret)}", f"G,  {f3(actor.ret)}"],
         ["Săgeți optime", f"{N_LIVE} din {N_LIVE}", f"{linear.n_optimal} din {N_LIVE}",
          f"{critic.n_optimal} din {N_LIVE}", f"{actor.n_optimal} din {N_LIVE}"]]
table(s, 0.45, 1.5, [2.15, 2.45, 2.45, 2.4, 2.45], rows_, size=13, row_h=0.72, head_h=0.5,
      header_fills={0: "pine", 1: "blue", 2: "slate", 3: "green", 4: "plum"},
      fills={(4, 1): "blue_t", (4, 2): "coral_t", (4, 3): "green_t", (4, 4): "plum_t"},
      aligns={1: "c", 2: "c", 3: "c", 4: "c"}, name="Patru reprezentari")

# ---- 20 cheat sheet
s = content_slide(prs, "Fișă recapitulativă", notes=(
    "O pagină pentru această oră. Rugați pe cineva să explice diferența dintre rândul „semi-gradient” și o atribuire în tabel. "
    "Dacă poate, a prins lecția. Circa 2 minute."))
rows_ = [["", "Pe scurt", "Formula"],
         ["Aproximare", "Q nu mai este o căsuță", "Q(s, a) ≈ Q_{θ}(s, a)"],
         ["One-hot", "un weight pe pereche: chiar tabelul", "Q(s, a) = θ_{s, a}"],
         ["Feature liniar", "un plan; nu ocolește peretele", "Q(s, a) = φ(s) · w_{a}"],
         ["Semi-gradient", "target-ul se tratează ca o constantă", "θ ← θ + α (y − Q_{θ}) ∇_{θ} Q_{θ}"],
         ["Fitted Q", "un sweep: îngheață y, apoi regresie", "y = r + γ max Q_{θ}(s′, ·)"],
         ["Controller implicit", "fără weight-uri noi", "π(s) = argmax_{a} Q_{θ}(s, a)"],
         ["Actor", "a doua rețea, imită criticul", "a*(s) = argmax Q_{θ},   învață π_{φ}"],
         ["Interferență", "un pas mișcă și celulele nevizitate", "ΔQ(s̃) ∝ φ(s) · φ(s̃)"]]
table(s, 0.5, 1.45, [2.5, 4.7, 4.9], rows_, size=14, row_h=0.55, head_h=0.42, fonts={2: "Cambria"},
      name="Fisa")

# ---- 21 questions
s = content_slide(prs, "Verificați-vă, apoi mergeți mai departe", notes=(
    "Răspunsuri, pe scurt. 1. φ(S) nu este ortogonal pe φ(s), deci pasul pe w schimbă și Q(S); un tabel nu are acest produs scalar. "
    "2. Least squares-ul este exact, deci nu antrenarea e de vină: Q* nu stă într-un plan peste (x, y), peretele îl rupe. "
    f"3. Nimic la numărătoare, {critic.n_params} față de {N_PAIRS}. Câștigul este intrarea: același cod primește un vector, "
    "nu un indice dintr-o listă finită. "
    f"4. Din argmax-ul criticului. Pe seed {bad['seed']} actorul copiază săgețile greșite și ratează G la fel. "
    "5. Ca să nu derivăm prin backup și ca target-ul să stea locului cât îl potrivim; altfel urmărim o țintă care se mută la fiecare minibatch. "
    "Resursele continuă prima oră: Silver 6 și 7, Sutton și Barto cap. 9, DQN, și lucrarea ADP."))
text(s, 0.55, 1.45, 6.5, 0.4, "Verificați-vă înțelegerea", size=18, bold=True)
qs = ["De ce un pas liniar făcut la (3, 1) schimbă Q și în S?",
      "De ce 12 weight-uri potrivite exact tot nu ajung la G?",
      "Criticul are mai multe weight-uri decât tabelul. Ce am câștigat?",
      "De unde primește actorul eticheta, și ce copiază pe seed 6?",
      "De ce target-ul rămâne înghețat pe durata unui sweep?"]
for i, q in enumerate(qs):
    y = 1.95 + i * 0.92
    box(s, 0.55, y, 6.5, 0.82, fill="mist", radius=0.08, name="Întrebarea %d" % (i + 1))
    badge(s, 0.7, y + 0.14, 0.52, str(i + 1), size=15)
    text(s, 1.4, y, 5.45, 0.82, q, size=14, anchor="m")
text(s, 7.3, 1.45, 5.4, 0.4, "Mergeți mai departe", size=18, bold=True)
res = [("Silver, Lecture 6 — Value Function Approximation", "De la feature-uri liniare la neural network-uri",
        "https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-6-value-function-approximation-.pdf"),
       ("Silver, Lecture 7 — Policy Gradient", "Controller-ul învățat fără o etichetă de argmax",
        "https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-7-policy-gradient-methods.pdf"),
       ("Sutton & Barto, ediția a 2-a, cap. 9", "Aproximarea on-policy, cu demonstrațiile de convergență",
        "http://incompleteideas.net/book/the-book-2nd.html"),
       ("Mnih et al., Nature, 2015  ·  Lewis & Vrabie, 2009", "DQN: replay și target network. ADP: critic și actor.",
        "https://doi.org/10.1038/nature14236")]
for i, (h_, d_, url) in enumerate(res):
    y = 1.95 + i * 1.15
    box(s, 7.3, y, 5.45, 1.05, fill="mist", radius=0.08, name="Resursa %d" % (i + 1))
    tb = text(s, 7.5, y + 0.08, 5.1, 0.5, h_, size=13, bold=True)
    text(s, 7.5, y + 0.55, 5.1, 0.4, d_, size=12, color="muted")
    link_runs(tb, url)

# ============================================================================ SAVE
sections = []
for i, (name, first) in enumerate(SECTIONS):
    last = SECTIONS[i + 1][1] if i + 1 < len(SECTIONS) else len(prs.slides)
    sections.append((name, list(range(first, last))))
add_sections(prs, sections)
out = os.path.join(ROOT, "RL_Function_Approximation_RO.pptx")
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))
print("sections:", [(n, len(ix)) for n, ix in sections])
print("seed 0 critic RMSE", round(critic.rmse, 4), "actor", actor.n_copied, "linear optimal", linear.n_optimal)
print("goals", n_goal, "of", len(rows), "bad seed", bad["seed"], "bad RMSE", round(critic_bad.rmse, 4))
