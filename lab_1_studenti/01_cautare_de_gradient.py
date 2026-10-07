"""Tema 1 — Căutare de gradient pentru un singur parametru.

Funcția obiectiv: f(w) = (w - 3)^2.
Completează doar locurile marcate cu `...` și comentariile `# TODO`.

    python 01_căutare_de_gradient.py
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
# Sarcina 1.1 — Definește și vizualizează funcția obiectiv
# ---------------------------------------------------------------------------

def f(w):
    # TODO: întoarce (w - 3)^2
    ...


w_values = np.linspace(-5, 10, 400)

plt.figure(figsize=(7, 4))
plt.plot(w_values, f(w_values), color="C0", label="f(w)")
# TODO: marchează minimul cu plt.axvline(valoare, linestyle="--")
plt.xlabel("w")
plt.ylabel("f(w)")
plt.title("Funcția obiectiv f(w) = (w − 3)²")
plt.grid(True, alpha=0.3)
plt.legend()
salveaza("01_functie_obiectiv.png")

raspuns_11 = ""  # TODO: în ce punct pare minimul și cât este f acolo?
print("1.1:", raspuns_11)


# ---------------------------------------------------------------------------
# Sarcina 1.2 — Deduce gradientul de mână, apoi scrie-l aici
# ---------------------------------------------------------------------------

def df(w):
    # TODO: df/dw. Indiciu: derivata lui (expresie)^2 este 2 * expresie * derivata expresiei.
    ...


# ---------------------------------------------------------------------------
# Sarcina 1.3 — Căutare de gradient
# ---------------------------------------------------------------------------

# Calculează mai întâi pe hârtie, pentru w = -4 și eta = 0.1.
gradient_pe_hartie = ...  # TODO: df/dw în acest punct
w_pe_hartie = ...  # TODO: w după un singur pas

w = -4.0
learning_rate = 0.1

w_history = []
loss_history = []

for iteration in range(50):
    loss = ...  # TODO
    grad = ...  # TODO

    w_history.append(...)  # TODO: valoarea curentă a lui w
    loss_history.append(...)  # TODO: valoarea curentă a loss-ului

    w = ...  # TODO: w <- w - eta * df/dw

print("Pe hârtie: gradient =", gradient_pe_hartie, ", w după un pas =", w_pe_hartie)
print("În cod, w după prima actualizare =", w_history[1])
print("w final =", w, ", f(w final) =", f(w))

plt.figure(figsize=(7, 4))
plt.plot(w_history, color="C0")
plt.axhline(3, linestyle="--", color="crimson", label="w = 3")
plt.xlabel("Iterație")
plt.ylabel("w")
plt.title("Căutare de gradient: parametrul w")
plt.grid(True, alpha=0.3)
plt.legend()
salveaza("01_w.png")

plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="C1")
plt.xlabel("Iterație")
plt.ylabel("loss  f(w)")
plt.title("Căutare de gradient: loss-ul")
plt.grid(True, alpha=0.3)
salveaza("01_loss.png")

raspuns_13_pasi = ""  # TODO: ce se întâmplă cu mărimea pașilor când w se apropie de minim? De ce?
raspuns_13_final = ""  # TODO: de ce valoarea finală a lui w este rezonabilă?
print("1.3 pași:", raspuns_13_pasi)
print("1.3 final:", raspuns_13_final)


# ---------------------------------------------------------------------------
# Sarcina 1.4 — Experiment cu rata de învățare
# ---------------------------------------------------------------------------

def gradient_descent(learning_rate, w_start=-4.0, n_iter=50):
    w = w_start
    w_history = []
    for _ in range(n_iter):
        # TODO: aceeași logică ca în sarcina 1.3
        loss = ...
        grad = ...
        w_history.append(...)
        w = ...
    return w_history


# Completează după ce te uiți la curbe. Folosește exact una dintre:
# convergență lentă, convergență rapidă, convergență oscilatorie, divergență
CLASIFICARE = {
    0.001: "",
    0.05: "",
    0.1: "",
    0.9: "",
    1.1: "",
}

plt.figure(figsize=(8, 4.5))
for eta, comportament in CLASIFICARE.items():
    istoric = np.asarray(gradient_descent(eta), dtype=float)
    eticheta = f"η = {eta}"
    if comportament:
        eticheta += f" ({comportament})"
    print(f"η = {eta:<5}  w_final ≈ {istoric[-1]:.4f}  {comportament}")
    # Curba care diverge ar umple graficul cu linii verticale. O tăiem la ieșirea din cadru.
    iesire = np.flatnonzero(np.abs(istoric) > 12)
    if iesire.size:
        istoric = istoric[: iesire[0]]
    plt.plot(istoric, label=eticheta)

plt.axhline(3, linestyle="--", color="black", linewidth=1, label="w = 3")
plt.ylim(-8, 12)
plt.xlabel("Iterație")
plt.ylabel("w")
plt.title("Efectul ratei de învățare (η = 1.1 iese din grafic)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
salveaza("01_rate_invatare.png")


# ---------------------------------------------------------------------------
# Sarcina 1.5 — Gradient numeric (diferențe finite centrale)
# ---------------------------------------------------------------------------

def numerical_gradient(w, epsilon=1e-6):
    # TODO: [f(w + epsilon) - f(w - epsilon)] / (2 * epsilon)
    ...


print(f"{'w':>6} {'epsilon':>10} {'analitic':>12} {'numeric':>12} {'|dif|':>12}")
for epsilon in (1e-1, 1e-3, 1e-6):
    for valoare in (-4, 0, 2.5, 3, 7):
        analitic = df(valoare)
        numeric = numerical_gradient(valoare, epsilon=epsilon)
        diferenta = abs(analitic - numeric)
        print(
            f"{valoare:6.1f} {epsilon:10.0e} {analitic:12.6f} "
            f"{numeric:12.6f} {diferenta:12.3e}"
        )

raspuns_15 = ""  # TODO: sunt diferențe între gradientul numeric și cel analitic? Ce observi când ε scade?
print("1.5:", raspuns_15)
