#!/usr/bin/env python3
"""Etapa 3. O viteză de croazieră care traversează colțurile intermediare.

Se scriu ecuațiile (10) și (11) în step(), iar ultimul colț se încheie cu
ecuația (4). Yaw-ul rămâne în step2_small_yaw.py. Programul de zbor îl
apelează cu n = k. Această funcție înlocuiește numai viteza pe care etapa 1
o construia din eroarea de poziție.

    python3 step3_cruise.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C aterizează.
"""


def step(
    WAYPOINTS,
    k,
    position,
    v_x_measured,
    v_y_measured,
    dt,
    hold,
    R_DISK,
    V_CRUISE,
    K_CROSS,
    R_PASS,
    D_SLOW,
    R_ACCEPT,
    V_ARRIVE,
    ARRIVE_HOLD,
):
    """Un apel pentru viteza din etapa 3. Ecuațiile (10) și (11).

    Nu se parcurge tot pătratul în acest apel. k este pasul de colț, nu
    numărul etapei. Pornește de la 1 după staționare. N = len(WAYPOINTS) - 1,
    adică 4 pe acest pătrat. k crește abia la un apel ulterior, când testul
    de avans din ecuația (10) este îndeplinit.

    P_k = WAYPOINTS[k - 1] este colțul tocmai părăsit.
    P_(k+1) = WAYPOINTS[k] este ținta.
    Cât timp k < N, P_(k+2) = WAYPOINTS[k + 1]. Când k = N, nu se citește
    P_(k+2). În note, traseul pornește din origine. Acest zbor construiește
    același pătrat pe punctul de decolare, deci se folosesc WAYPOINTS.

    Numerele din note sunt acești parametri. În ecuația (10): 0.45 m este
    R_DISK, raza discului din ecuația (6); 0.35 m/s este V_CRUISE; 0.8 este
    K_CROSS, coeficientul lui e_cross; 0.20 m este R_PASS. În ecuația (11),
    0.50 m este D_SLOW, distanța pe care viteza scade până la oprire.
    Se folosesc numele. unit din geometry.py este normalize:
    direcție, lungime = unit(vector). Un vector mai scurt de 1e-6 nu are
    direcție. clip, tot în geometry.py, limitează un scalar.

    Intrări
    -------
    WAYPOINTS : array, forma (5, 3)
        Metri, sistemul de coordonate absolut. Nu se scrie în el.
    k : int
        Colțul tocmai părăsit și rândul țintei. Același număr pe care etapa 2
        îl numește n.
    position : array, forma (3,)
        Poziția măsurată (x, y, z), în metri. p este position[:2].
    v_x_measured, v_y_measured : float
        Viteza orizontală măsurată, m/s, în sistemul de coordonate absolut. Se folosește numai
        când k = N, în ecuația (4). Nu este comanda.
    dt : float
        Δt al acestui apel, în secunde. Ecuația (4), la ultimul colț.
    hold : float
        Secunde de când ecuația (4) rămâne îndeplinită. Cât timp k < N, se
        returnează neschimbat, sau 0 dacă k a avansat. Când k = N, se
        actualizează ca în etapa 1. Programul de zbor reține valoarea
        returnată.
    R_DISK : float
        R, în metri. Ecuația (6), folosită din nou ca α în ecuația (10).
    V_CRUISE : float
        v_cruise, m/s. Valoarea 0.35 din ecuațiile (10) și (11).
    K_CROSS : float
        K_cross, în 1/s. Coeficientul 0.8 al lui e_cross.
    R_PASS : float
        r_pass, în metri. Testul de 0.20 m din ecuația (10). Se măsoară
        distanța rămasă de-a lungul laturii, |leg| − s, nu distanța în linie
        dreaptă până la punct.
    D_SLOW : float
        d_slow, în metri. Valoarea 0.50 m din ecuația (11).
    R_ACCEPT, V_ARRIVE, ARRIVE_HOLD : float
        r_accept în metri, v_arrive în m/s, t_hold în secunde. Ecuația (4),
        și numai când k = N. Nu se folosesc pentru a traversa un colț
        intermediar.

    Returnează
    ----------
    v : array, forma (2,)
        (v_x, v_y), m/s, în sistemul de coordonate global. Ecuația (10) cât timp k < N.
        Ecuația (11) când k = N. Nu se rotește în coordonatele dronei.
    k : int
        Pasul de colț pentru apelul următor. Cât timp k < N, se adaugă unu
        când |leg| − s < R_PASS. Când k = N, nu se avansează.
    hold : float
        Secunde, ca mai sus. Ultimul colț s-a încheiat când valoarea a atins
        ARRIVE_HOLD și k a rămas N. Programul de zbor aterizează atunci.
    """
    # N = len(WAYPOINTS) - 1. k pornește de la 1.
    # P_k = WAYPOINTS[k - 1], colțul tocmai părăsit.
    # P_(k+1) = WAYPOINTS[k], ținta.
    # p = position[:2].
    # d este distanța orizontală de la p până la țintă.
    # Latura este ținta minus P_k. unit dă direcția și lungimea.
    # s este proiecția lui (p − P_k) pe direcția laturii.
    # e_cross este abaterea laterală față de latură: poziția relativă minus s * direcție.
    # Yaw-ul nu se calculează aici. Programul de zbor apelează etapa 2 cu n = k.

    # alpha, același disc ca în ecuația (6), cu raza R_DISK și distanța d.
    # O latură mai scurtă de 1e-6 nu are direcție.
    # Cât timp k < N, direcția următoare este WAYPOINTS[k + 1] minus ținta.
    # Când k = N, nu se citește punctul de după țintă, iar alpha este 0.

    # Ecuația (10), cât timp k < N.
    # Direcția comandată amestecă latura curentă cu latura următoare, după alpha.
    # Viteza are lungimea V_CRUISE pe direcția amestecată.
    # Se scade corecția laterală (1 − alpha) * K_CROSS * e_cross.
    # Avansul nu folosește distanța în linie dreaptă. Se adaugă 1 la k când
    # lungimea laturii minus s este sub R_PASS, iar hold repornește de la 0.
    # Altfel k și hold rămân cele primite.

    # Ecuația (11), când k = N.
    # Direcția este de la p către țintă.
    # Viteza are lungimea V_CRUISE, înmulțită cu distanța d limitată la intervalul
    # D_SLOW, apoi se scade K_CROSS * e_cross.
    # k nu avansează.

    # Ecuația (4), numai când k = N.
    # hold crește cu dt cât timp d < R_ACCEPT și viteza măsurată
    # hypot(v_x_measured, v_y_measured) < V_ARRIVE. Altfel hold este 0.
    # R_ACCEPT, V_ARRIVE și ARRIVE_HOLD nu se folosesc la colțurile intermediare.
    # Programul de zbor aterizează când k a rămas N și hold a atins ARRIVE_HOLD.

    # Se returnează v, k, hold.
    raise NotImplementedError(
        "step3_cruise.py: scrie ecuația (10) cât timp k < N și ecuația (11) când k = N."
    )


if __name__ == "__main__":
    from fly_square import main

    main(stage=3)
