"""Tema 4 — O problemă pe care un clasificator liniar nu o rezolvă bine.

Completează doar locurile marcate cu `...` și comentariile `# TODO`.
Funcția care desenează frontiera este dată: nu trebuie modificată, în afară de titlurile axelor.

    python 04_clasificare_moons.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
# Sarcina 4.1 — Setul de date „moons"
# ---------------------------------------------------------------------------

X, y = make_moons(n_samples=500, noise=0.15, random_state=0)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Două clase în formă de semilună")
salveaza("04_moons.png")

raspuns_41 = ""  # TODO: descrie forma celor două clase. Le poți separa cu o singură dreaptă?
print("4.1:", raspuns_41)


# ---------------------------------------------------------------------------
# Sarcina 4.2 — Antrenare și test
# ---------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("X_train", X_train.shape)
print("X_test", X_test.shape)

raspuns_42 = ""  # TODO: câte exemple are fiecare set?
print("4.2:", raspuns_42)


# ---------------------------------------------------------------------------
# Sarcina 4.3 — Regresie logistică (baseline liniar)
# ---------------------------------------------------------------------------

linear_model = LogisticRegression()
linear_model.fit(X_train, y_train)

train_acc = ...  # TODO: linear_model.score(X_train, y_train)
test_acc = ...  # TODO

print("train accuracy:", train_acc)
print("test accuracy:", test_acc)


def plot_decision_boundary_sklearn(model, X, y, step=0.02):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=20, cmap="coolwarm")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Frontiera de decizie — regresie logistică")
    salveaza("04_frontiera_liniara.png")


plot_decision_boundary_sklearn(linear_model, X_test, y_test)


# ---------------------------------------------------------------------------
# Sarcina 4.4 — Interpretare
# ---------------------------------------------------------------------------

raspuns_44 = ""  # TODO: de ce o frontieră dreaptă se potrivește slab? Ce frontieră ar fi potrivită?
print("4.4:", raspuns_44)
