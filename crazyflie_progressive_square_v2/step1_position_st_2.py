#!/usr/bin/env python3
"""Etapa 1. O referință de poziție și o staționare la fiecare colț.

Se scriu ecuațiile (3) și (4) în step(). Programul de zbor o apelează o dată
la fiecare interval dt. El citește deja OptiTrack, urcă, aterizează și
transformă viteza din reperul lumii în reperul corpului, cu ecuațiile (1) și (2).

    python3 step1_position.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C aterizează. Când ți se cere, orientează drona spre peretele de calibrare.
"""


def step(
    WAYPOINTS,
    index,
    position,
    v_x_measured,
    v_y_measured,
    dt,
    hold,
    KP,
    V_MAX,
    R_ACCEPT,
    V_ARRIVE,
    ARRIVE_HOLD,
):
    """Un apel pentru etapa 1. Ecuațiile (3) și (4).

    Nu se parcurge tot pătratul în acest apel. Un apel calculează o singură
    comandă. Programul de zbor apelează din nou funcția cu index și hold
    returnate aici.

    Pătratul din note pornește din origine. Acest zbor construiește același
    pătrat pe punctul de decolare, deci colțurile se citesc din WAYPOINTS.
    Toate cele cinci rânduri au aceeași înălțime. Ultimul rând repetă primul,
    ca drona să aibă unde să staționeze. LAST = len(WAYPOINTS) - 1.
    Staționarea este rândul 0, iar index pornește de la 1.

    Intrări
    -------
    WAYPOINTS : array, forma (5, 3)
        Metri, reperul lumii. Rândul index este ținta p*. Nu se scrie în el.
    index : int
        Rândul țintei. Este indicele din note.
    position : array, forma (3,)
        Poziția măsurată (x, y, z), în metri. p este position[:2].
    v_x_measured, v_y_measured : float
        Viteza orizontală măsurată, m/s, în reperul lumii. Ecuația (4)
        folosește viteza măsurată, hypot(v_x_measured, v_y_measured).
        Nu este viteza comandată.
    dt : float
        Δt al acestui apel, în secunde.
    hold : float
        De câte secunde este deja îndeplinită ecuația (4). Programul de zbor
        reține valoarea returnată și o trimite înapoi la apelul următor.
    KP : float
        K_p, din gains.py. Ecuația (3), în 1/s.
    V_MAX : float
        v_max, m/s. Ecuația (3).
    R_ACCEPT : float
        r_accept, în metri. Ecuația (4).
    V_ARRIVE : float
        v_arrive, m/s. Ecuația (4).
    ARRIVE_HOLD : float
        t_hold, în secunde. Se avansează numai după ce ecuația (4) a rămas
        îndeplinită acest interval, ca un eșantion zgomotos să nu sară peste
        staționare.

    Returnează
    ----------
    velocity : array, forma (2,)
        (v_x, v_y), m/s, în reperul lumii. Ecuația (3). Se scalează întregul
        vector astfel încât lungimea lui să fie cel mult V_MAX. Un vector mai
        scurt rămâne neschimbat. Nu se rotește în reperul corpului. Ecuațiile
        (1) și (2) sunt deja în programul de zbor. geometry.clip limitează un
        scalar, nu acest vector.
    yaw_rate : float
        rad/s. În etapa 1 este 0. Yaw-ul rămâne la orientarea de la staționare.
    height : float
        Metri. Coordonata z a țintei din acest apel, goal[2]. Se citește
        înainte de modificarea lui index.
    index : int
        Rândul țintei pentru apelul următor. Se adaugă unu numai când hold a
        atins ARRIVE_HOLD și index < LAST. La ultimul punct nu există un punct
        următor, deci index rămâne neschimbat.
    hold : float
        Secunde. Se adaugă dt cât timp drona este în interiorul lui r_accept
        și viteza măsurată este sub v_arrive. Se returnează 0 de îndată ce
        unul dintre teste nu mai este îndeplinit. Se returnează 0 și la avans,
        pentru ca staționarea colțului următor să pornească de la zero.
    """
    # p este position[:2]. Ținta p* este WAYPOINTS[index].
    # LAST = len(WAYPOINTS) - 1. Nu se scrie în WAYPOINTS.

    # Ecuația (3).
    # Eroarea orizontală este ținta minus p.
    # Viteza este această eroare înmulțită cu KP.
    # Dacă lungimea vectorului depășește V_MAX, se scalează vectorul întreg
    # astfel încât lungimea lui să fie V_MAX. Un vector mai scurt rămâne neschimbat.

    # yaw_rate este 0.
    # height este coordonata z a țintei. Se citește înainte de modificarea lui index.

    # Ecuația (4).
    # Distanța orizontală până la țintă, și viteza măsurată
    # hypot(v_x_measured, v_y_measured). Nu se folosește viteza comandată.
    # hold crește cu dt cât timp distanța este sub R_ACCEPT și viteza măsurată
    # este sub V_ARRIVE. Altfel hold revine la 0.
    # Când hold a atins ARRIVE_HOLD și index < LAST, index crește cu 1 și hold
    # repornește de la 0.
    # La ultimul punct index rămâne neschimbat. Programul de zbor aterizează
    # când index este LAST și hold a atins ARRIVE_HOLD.

    # Se returnează velocity, yaw_rate, height, index, hold.
    raise NotImplementedError("step1_position.py: scrie ecuațiile (3) și (4).")


if __name__ == "__main__":
    from fly_square import main

    main(stage=1)
