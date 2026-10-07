"""Tema 5 — Rețea neuronală feed-forward 2 → 16 → 16 → 2.

Arhitectura este dată. Tu identifici componentele și numeri parametrii de mână.
Completează variabilele marcate cu `...` și răspunsurile goale.

    python 05_retea_neuronala.py
"""

import torch

torch.manual_seed(0)

model = torch.nn.Sequential(
    torch.nn.Linear(2, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 2),
)

print(model)


# ---------------------------------------------------------------------------
# Sarcina 5.1 — Identifică componentele
# ---------------------------------------------------------------------------

raspuns_51 = ""  # TODO: intrare, straturi ascunse, ieșire, activări, rolul celor două valori de ieșire
print("5.1:", raspuns_51)


# ---------------------------------------------------------------------------
# Sarcina 5.2 — Numără parametrii de mână, apoi verifică cu PyTorch
# ---------------------------------------------------------------------------
# Un strat Linear(in, out) are in * out ponderi și out bias-uri.

p_2_16 = ...  # TODO: ponderi 2 → 16
b_2_16 = ...  # TODO: bias-uri 2 → 16
p_16_16 = ...  # TODO: ponderi 16 → 16
b_16_16 = ...  # TODO: bias-uri 16 → 16
p_16_2 = ...  # TODO: ponderi 16 → 2
b_16_2 = ...  # TODO: bias-uri 16 → 2
total_manual = ...  # TODO: suma tuturor

print("2 → 16:  ponderi =", p_2_16, " bias =", b_2_16)
print("16 → 16: ponderi =", p_16_16, " bias =", b_16_16)
print("16 → 2:  ponderi =", p_16_2, " bias =", b_16_2)
print("Total manual =", total_manual)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total PyTorch =", n_params)

for name, param in model.named_parameters():
    print(name, tuple(param.shape))

raspuns_52 = ""  # TODO: coincide totalul? Ce formă are matricea de ponderi a primului strat și de ce?
print("5.2:", raspuns_52)
