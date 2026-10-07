"""Tema 3 — De la un model liniar la un neuron.

Completează doar locurile marcate cu `...` și comentariile `# TODO`.

    python 03_neuron_si_activari.py
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
# Sarcina 3.1 — Două straturi liniare, fără activare
# ---------------------------------------------------------------------------
# h = W1·x + b1
# y = W2·h + b2
# Înlocuiește h în a doua ecuație și desfă parantezele.

raspuns_31 = ""  # TODO: de ce modelul nu este mai expresiv decât un singur strat liniar?
print("3.1:", raspuns_31)


# ---------------------------------------------------------------------------
# Sarcina 3.2 — Graficele ReLU, sigmoid și tanh
# ---------------------------------------------------------------------------

z = np.linspace(-6, 6, 400)

relu = np.maximum(0, z)
sigmoid = ...  # TODO: 1 / (1 + e^(-z)); indiciu: np.exp
tanh = ...  # TODO: indiciu: np.tanh

plt.figure(figsize=(7, 4))
plt.plot(z, relu, label="ReLU")
plt.plot(z, sigmoid, label="sigmoid")
plt.plot(z, tanh, label="tanh")
plt.xlabel("z")
plt.ylabel("activare")
plt.title("Funcții de activare")
plt.grid(True, alpha=0.3)
plt.legend()
salveaza("03_activari.png")

print(f"ReLU     min={relu.min(): .3f}  max={relu.max(): .3f}")
print(f"sigmoid  min={sigmoid.min(): .3f}  max={sigmoid.max(): .3f}")
print(f"tanh     min={tanh.min(): .3f}  max={tanh.max(): .3f}")

raspuns_32 = ""  # TODO: intervalele, cine produce valori negative, unde este graficul plat și ce înseamnă asta pentru gradient?
print("3.2:", raspuns_32)
