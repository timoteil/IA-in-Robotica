"""Tema 6 — Completează bucla de antrenare a rețelei.

Aceeași buclă ca în temele 1 și 2: predicție → loss → gradient → actualizare.
Datele și modelul sunt recreate aici, ca scriptul să ruleze singur.
Completează doar locurile marcate cu `...` și comentariile `# TODO`.

    python 06_bucla_de_antrenare.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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


np.random.seed(0)
torch.manual_seed(0)

X, y = make_moons(n_samples=500, noise=0.15, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# Sarcina 6.1 — Tensori
# ---------------------------------------------------------------------------

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

for nume, tensor in (
    ("X_train", X_train_t),
    ("y_train", y_train_t),
    ("X_test", X_test_t),
    ("y_test", y_test_t),
):
    print(f"{nume:8} shape={tuple(tensor.shape)} dtype={tensor.dtype}")


# ---------------------------------------------------------------------------
# Sarcina 6.2 — Același model ca în tema 5, plus loss și optimizator
# ---------------------------------------------------------------------------

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2),
)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


# ---------------------------------------------------------------------------
# Sarcina 6.3 — Cei cinci pași ai antrenării
# ---------------------------------------------------------------------------

loss_history = []

for epoch in range(1000):
    # 1. Propagare înainte
    predictions = ...  # TODO

    # 2. Loss
    loss = ...  # TODO

    # 3. Șterge gradienții anteriori
    ...  # TODO

    # 4. Backpropagation
    ...  # TODO

    # 5. Actualizarea parametrilor
    ...  # TODO

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(epoch, loss.item())

plt.figure(figsize=(7, 4))
plt.plot(loss_history, color="C1")
plt.xlabel("Epocă")
plt.ylabel("Cross-entropy")
plt.title("Loss-ul rețelei")
plt.grid(True, alpha=0.3)
salveaza("06_loss.png")

raspuns_curba = ""  # TODO: cum arată curba de loss? Scade? Se stabilizează?
print("6.3 curbă:", raspuns_curba)


def accuracy(model, X, y):
    with torch.no_grad():
        logits = ...  # TODO: scorurile modelului pentru X
        predicted_class = ...  # TODO: indiciu logits.argmax(dim=1)
        return ...  # TODO: fracția de predicții corecte


print("train accuracy:", accuracy(model, X_train_t, y_train_t))
print("test accuracy:", accuracy(model, X_test_t, y_test_t))

baseline = LogisticRegression()
baseline.fit(X_train, y_train)
print("test accuracy liniar:", baseline.score(X_test, y_test))

raspuns_pasi = ""  # TODO: ce înseamnă fiecare dintre cei cinci pași și de ce este necesar?
raspuns_comparatie = ""  # TODO: compară acuratețea pe test a rețelei cu cea a modelului liniar. De ce diferă?
print("6.3 pași:", raspuns_pasi)
print("6.3 comparație:", raspuns_comparatie)


def _contur(predict_clase, X, y, ax):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict_clase(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", s=16, cmap="coolwarm")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def _clase_liniare(grid):
    return baseline.predict(grid)


def _clase_retea(grid):
    with torch.no_grad():
        logits = model(torch.tensor(grid, dtype=torch.float32))
        return logits.argmax(dim=1).numpy()


fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
_contur(_clase_liniare, X_test, y_test, axes[0])
axes[0].set_title("Regresie logistică")
_contur(_clase_retea, X_test, y_test, axes[1])
axes[1].set_title("Rețea neuronală")
salveaza("06_frontiere.png")
