"""Tema 2 — Regresie liniară de la zero: estimarea prețului unei case.

Model: ŷ = w * x + b.   Loss: MSE.
Completează doar locurile marcate cu `...` și comentariile `# TODO`.

    python 02_regresie_liniara.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["axes.unicode_minus"] = False

FIGURI = Path(__file__).resolve().parent / "figures"
FIGURI.mkdir(exist_ok=True)


def salveaza(nume):
    plt.tight_layout()
    plt.savefig(FIGURI / nume, dpi=140)
    if plt.get_backend().lower() != "agg":
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Sarcina 2.1 — Generează datele
# ---------------------------------------------------------------------------

np.random.seed(0)
n_houses = 100

suprafata_mp = np.random.uniform(30, 150, n_houses)
zgomot = np.random.normal(0, 15, n_houses)
pret_mii_eur = 1.5 * suprafata_mp + 20 + zgomot

plt.figure(figsize=(7, 4))
plt.scatter(suprafata_mp, pret_mii_eur, s=18, alpha=0.85)
plt.xlabel("Suprafața (m²)")
plt.ylabel("Prețul (mii €)")
plt.title("Case vândute: suprafață și preț")
plt.grid(True, alpha=0.3)
salveaza("02_date.png")

raspuns_21 = ""  # TODO: ce relație observi între suprafață și preț? Poți trasa o dreaptă prin nor?
print("2.1:", raspuns_21)


# ---------------------------------------------------------------------------
# Sarcina 2.2 — Rescalare și model
# ---------------------------------------------------------------------------

x = suprafata_mp / 100  # sute de m²
y = pret_mii_eur / 100  # sute de mii €

print("x min/max:", float(x.min()), float(x.max()))
print("y min/max:", float(y.min()), float(y.max()))


def predict(x, w, b):
    # TODO: ŷ = w * x + b
    ...


# ---------------------------------------------------------------------------
# Sarcina 2.3 — Funcția de pierdere (MSE)
# ---------------------------------------------------------------------------

def mse(y_pred, y_true):
    # TODO: (1/N) * suma (ŷ - y)^2. Indiciu: np.mean
    ...


# ---------------------------------------------------------------------------
# Sarcina 2.4 — Gradienții dL/dw și dL/db, deduși de mână
# ---------------------------------------------------------------------------

def gradiente(x, y_true, w, b):
    # TODO: întoarce dL/dw și dL/db.
    # Eroarea unei case: r = ŷ - y. Media lui 2*r*x, respectiv 2*r.
    ...


# Opțional: pune True după ce ai implementat gradientul numeric de mai jos.
VERIFICA_GRADIENT_NUMERIC = False


def dL_dw_numeric(w, b, epsilon=1e-6):
    # TODO opțional: [L(w+ε, b) - L(w-ε, b)] / (2ε)
    ...


def dL_db_numeric(w, b, epsilon=1e-6):
    # TODO opțional: [L(w, b+ε) - L(w, b-ε)] / (2ε)
    ...


if VERIFICA_GRADIENT_NUMERIC:
    dw_an, db_an = gradiente(x, y, 0.0, 0.0)
    print("dL/dw analitic", dw_an, "numeric", dL_dw_numeric(0.0, 0.0))
    print("dL/db analitic", db_an, "numeric", dL_db_numeric(0.0, 0.0))


# ---------------------------------------------------------------------------
# Sarcina 2.5 — Antrenează modelul
# ---------------------------------------------------------------------------

w_init, b_init = 0.0, 0.0
w, b = w_init, b_init
learning_rate = 0.1
epochs = 500

loss_history = []

for epoch in range(epochs):
    y_pred = ...  # TODO: predicția pentru toate casele
    loss = ...  # TODO: MSE

    dw = ...  # TODO: dL/dw
    db = ...  # TODO: dL/db

    w = ...  # TODO: actualizează w
    b = ...  # TODO: actualizează b

    loss_history.append(loss)

print(f"w învățat = {w:.4f} (generare: 1.5)")
print(f"b învățat = {b:.4f} (generare: 0.2)")
print(f"MSE inițial = {loss_history[0]:.4f}, MSE final = {loss_history[-1]:.4f}")

plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="C1")
plt.xlabel("Epocă")
plt.ylabel("MSE")
plt.title("Loss-ul de antrenare")
plt.grid(True, alpha=0.3)
salveaza("02_loss.png")

x_line = np.linspace(x.min(), x.max(), 100)
plt.figure(figsize=(7, 4))
plt.scatter(x, y, s=18, alpha=0.85, label="date")
plt.plot(x_line, predict(x_line, w_init, b_init), label="înainte de antrenare")
plt.plot(x_line, predict(x_line, w, b), linewidth=2, label="după antrenare")
plt.plot(x_line, predict(x_line, 1.5, 0.2), linestyle="--", label="formula de generare")
plt.xlabel("x (sute de m²)")
plt.ylabel("y (sute de mii €)")
plt.title("Dreapta învățată")
plt.grid(True, alpha=0.3)
plt.legend()
salveaza("02_dreapta.png")

raspuns_de_ce_nu_exact = ""  # TODO: de ce w și b nu sunt exact 1.5 și 0.2?
print("2.5:", raspuns_de_ce_nu_exact)


# ---------------------------------------------------------------------------
# Sarcina 2.6 — Predicții pentru case noi
# ---------------------------------------------------------------------------

case_noi_mp = np.array([50.0, 80.0, 120.0])
x_nou = ...  # TODO: suprafața în unitățile modelului
y_nou = ...  # TODO: predicția modelului
pret_eur = ...  # TODO: prețul în euro
cost_m2 = ...  # TODO: euro pentru un metru pătrat în plus

for mp, eur in zip(case_noi_mp, pret_eur):
    print(f"{mp:6.0f} m²  ->  {eur:10.0f} euro")
print(f"Un m² în plus costă aproximativ {cost_m2:.0f} euro")

raspuns_26 = ""  # TODO: pare o valoare realistă pentru un metru pătrat?
print("2.6:", raspuns_26)


# ---------------------------------------------------------------------------
# Sarcina 2.7 — Interpretare
# ---------------------------------------------------------------------------

raspuns_27_unde = ""  # TODO: unde este stocată cunoașterea învățată? La ce mai folosesc datele după antrenare?
raspuns_27_factori = ""  # TODO: ce factori nu vede modelul? Cum l-ai îmbunătăți?
print("2.7 unde:", raspuns_27_unde)
print("2.7 factori:", raspuns_27_factori)
