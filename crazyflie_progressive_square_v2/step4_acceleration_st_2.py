#!/usr/bin/env python3
"""Etapa 4. O limită de accelerație și înclinarea corespunzătoare.

Se scriu ecuațiile (12), (13) și (14) în step(). Programul de zbor cere în
continuare viteza din etapa 3 și yaw-ul din etapa 2. Aici se limitează doar
cât de repede se poate schimba acea viteză și se raportează înclinarea pe
care o cere variația.

    python3 step4_acceleration.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C aterizează. Linia de stare afișează înclinarea o dată pe secundă.
"""


def step(velocity_wanted, velocity_sent, dt, A_MAX, G):
    """Un apel pentru limita din etapa 4. Ecuațiile (12), (13) și (14).

    Pătratul nu se recalculează aici. velocity_wanted este (v_x, v_y) ales
    de etapa 3 în acest apel. velocity_sent se apropie de el cu cel mult
    a_max în același apel. Programul de zbor trimite velocity_sent returnat
    prin ecuațiile (1) și (2) și afișează tilt. Nici accelerația, nici
    înclinarea nu ajung la motoare.

    Intrări
    -------
    velocity_wanted : array, forma (2,)
        (v_x, v_y), m/s, în sistemul de coordonate absolut. Comanda din etapa 3 pentru acest
        apel, înainte de limită.
    velocity_sent : array, forma (2,)
        (v_x, v_y), m/s, în sistemul de coordonate absolut. Comanda returnată la apelul
        anterior. Este v_previous din ecuația (12). La primul apel este
        (0, 0). Nu se modifică vectorul primit. Se returnează un vector nou.
    dt : float
        Δt al acestui apel, în secunde.
    A_MAX : float
        a_max, m/s². Ecuația (12).
    G : float
        g, m/s². Ecuația (14).

    Returnează
    ----------
    velocity_sent : array, forma (2,)
        (v_x, v_y) limitat, m/s, în sistemul de coordonate absolut. Ecuația (12). Variația
        față de velocity_sent primit are lungimea cel mult A_MAX * dt.
    acceleration : array, forma (2,)
        (a_x, a_y), m/s². Ecuația (13), (v_sent − v_previous) / Δt. Este
        accelerația comandată, nu o măsurătoare.
    tilt : float
        Radiani. Ecuația (14), atan(‖a_xy‖ / g). Programul de zbor o
        convertește în grade pentru linia de stare.
    """
    # Etapa 3 a ales deja velocity_wanted. Nu se recalculează pătratul și nu se
    # cere yaw-ul. velocity_sent primit este v_previous. Nu se modifică acel vector.

    # Ecuațiile (12) și (13).
    # delta este velocity_wanted minus velocity_sent primit.
    # Dacă lungimea lui delta depășește A_MAX * dt, se scalează delta la acea
    # lungime. Direcția rămâne aceeași. Un delta mai scurt rămâne neschimbat.
    # velocity_sent returnat este vectorul primit plus delta.
    # acceleration este delta împărțit la dt. Este vectorul (a_x, a_y), nu un scalar.

    # Ecuația (14).
    # tilt este atan dintre lungimea lui acceleration și G.
    # În radiani. Programul de zbor îl afișează. Nu este o comandă către motoare.

    # Se returnează velocity_sent, acceleration, tilt.
    raise NotImplementedError(
        "step4_acceleration.py: scrie ecuațiile (12), (13) și (14)."
    )


if __name__ == "__main__":
    from fly_square import main

    main(stage=4)
