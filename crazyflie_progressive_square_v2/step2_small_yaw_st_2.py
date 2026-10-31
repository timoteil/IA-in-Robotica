#!/usr/bin/env python3
"""Etapa 2. O referință mică de yaw la apropierea de un colț.

Se scriu ecuațiile (5)–(9) în step(). Viteza din etapa 1 rămâne în
step1_position.py, inclusiv staționarea de la fiecare colț. Această funcție
înlocuiește numai yaw_rate = 0.

    python3 step2_small_yaw.py 10.10.10.100 --client 10.10.10.2 --body-id 4 --address 0xE7E7E7E7E7

Ctrl+C aterizează.
"""


def step(
    WAYPOINTS,
    n,
    distance,
    yaw,
    dt,
    psi_set,
    R_DISK,
    BETA,
    OMEGA_SET,
    KY,
    R_MAX,
):
    """Un apel pentru yaw-ul din etapa 2. Ecuațiile (5)–(9).

    Nu se parcurg colțurile în acest apel. Programul de zbor cere întâi
    viteza din etapa 1, apoi apelează această funcție pentru colțul încă în
    zbor și abia după aceea folosește indicele returnat de etapa 1.

    N = len(WAYPOINTS) - 1. Pe acest pătrat, N este 4. Colțul n este colțul
    tocmai părăsit, WAYPOINTS[n - 1]. Ținta, colțul n+1, este WAYPOINTS[n].
    Cât timp n < N, colțul n+2 este WAYPOINTS[n + 1]. Pe ultima latură,
    n = N, nu există o orientare următoare: se pune α = 0 și
    ψ_target,n = ψ_leg,n.

    O latură ale cărei colțuri au același x și același y nu are orientare.
    Se păstrează yaw-ul anterior. wrap și clip sunt în geometry.py. wrap aduce
    un unghi în (−π, π]. clip limitează un scalar.

    Intrări
    -------
    WAYPOINTS : array, forma (5, 3)
        Metri, reperul lumii. Pătratul este construit pe punctul de decolare,
        nu pe originea din note. Nu se scrie în el.
    n : int
        Colțul tocmai părăsit și rândul țintei. Este indicele din etapa 1,
        înainte ca ecuația (4) să îl incrementeze. Pornește de la 1.
    distance : float
        d, în metri. Distanța orizontală de la dronă până la țintă, colțul
        n+1. Ecuația (6). Programul de zbor a măsurat-o deja.
    yaw : float
        ψ, în radiani. Yaw-ul măsurat.
    dt : float
        Δt al acestui apel, în secunde. Ecuația (8).
    psi_set : float
        ψ_set, în radiani, reținut de la apelul anterior. La primul apel este
        orientarea de la staționare. Ecuația (8) îl apropie de ψ_target,n cu
        cel mult ω_set. Se returnează valoarea de la care pornește apelul
        următor. Programul de zbor o reține.
    R_DISK : float
        R, în metri. Discul de apropiere din ecuația (6).
    BETA : float
        β, adimensional. Fracția din colț comandată în timpul apropierii.
        Ecuația (7).
    OMEGA_SET : float
        ω_set, rad/s. Rata maximă cu care se poate deplasa ψ_set. Ecuația (8).
    KY : float
        K_y, în 1/s. Ecuația (9).
    R_MAX : float
        r_max, rad/s. Limita ratei de yaw din ecuația (9).

    Returnează
    ----------
    yaw_rate : float
        rad/s. Ecuația (9). Este yaw_rate transmis regulatorului de viteză.
        Nu este o orientare.
    psi_set : float
        Radiani. Valoarea scrisă de ecuația (8), pentru apelul următor.
    """
    # N = len(WAYPOINTS) - 1.
    # Colțul tocmai părăsit este WAYPOINTS[n - 1]. Ținta este WAYPOINTS[n].
    # Nu se recalculează viteza. Ecuațiile (3) și (4) rămân în etapa 1.
    # distance este deja măsurată de programul de zbor.

    # Ecuațiile (5)–(7).
    # Latura curentă este ținta minus colțul părăsit.
    # O latură cu lungime sub 1e-6 nu are orientare: se păstrează psi_set primit,
    # iar alpha este 0.
    # Altfel, orientarea laturii este atan2 din componentele ei.
    # Cât timp n < N, latura următoare pornește din țintă către WAYPOINTS[n + 1].
    # alpha crește de la 0 la 1 în interiorul discului R_DISK, după distance.
    # În afara discului, alpha este 0.
    # Pe ultima latură, n = N, nu există orientare următoare: alpha este 0 și
    # ținta de yaw rămâne orientarea laturii curente.
    # psi_target combină orientarea curentă cu fracția BETA din diferența până
    # la orientarea următoare. Diferența de unghi trece prin wrap.

    # Ecuația (8).
    # psi_set primit se apropie de psi_target.
    # Pasul maxim într-un apel este OMEGA_SET * dt, în ambele sensuri.
    # wrap aduce diferența în (−π, π]. clip limitează pasul.
    # Se returnează psi_set nou. Programul de zbor îl trimite la apelul următor.

    # Ecuația (9).
    # yaw_rate pornește din diferența dintre psi_set și yaw-ul măsurat,
    # înmulțită cu KY, apoi limitată la ±R_MAX.

    # Se returnează yaw_rate, psi_set.
    raise NotImplementedError(
        "step2_small_yaw.py: scrie ecuațiile (5)–(9)."
    )


if __name__ == "__main__":
    from fly_square import main

    main(stage=2)
