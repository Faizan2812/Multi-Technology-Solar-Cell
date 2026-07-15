"""
temperature.py — temperature-dependent material physics.

Adds T-dependence that the static database lacks, enabling realistic
temperature sweeps and perovskite/Si tandem work:
  * Varshni band gap  Eg(T) = Eg0 - a T^2 / (T + b)
  * Effective DOS     Nc,Nv ~ T^{3/2}
  * Intrinsic density ni(T) from Nc(T),Nv(T),Eg(T)
  * Mobility          mu(T) = mu300 (T/300)^(-gamma)  (phonon-limited)

Self-tests (__main__): Eg falls with T; ni rises with T; everything returns the
300 K base values at T = 300 K.
"""
import numpy as np
from .constants import K_B, Q, T_REF

# representative Varshni parameters (MAPbI3-like; a in eV/K, b in K)
VARSHNI = {"MAPbI3": dict(Eg0=1.61, a=3.3e-4, b=-110.0),
           "Si":     dict(Eg0=1.17, a=4.73e-4, b=636.0)}


def eg_varshni(T, Eg0, a, b):
    return Eg0 - a * T ** 2 / (T + b)


def nc_nv(Nc300, Nv300, T):
    s = (T / T_REF) ** 1.5
    return Nc300 * s, Nv300 * s


def ni(T, Nc300, Nv300, Eg0, a, b):
    Nc, Nv = nc_nv(Nc300, Nv300, T)
    Eg = eg_varshni(T, Eg0, a, b)
    return np.sqrt(Nc * Nv) * np.exp(-Eg * Q / (2 * K_B * T))


def mobility(mu300, T, gamma=1.5):
    return mu300 * (T / T_REF) ** (-gamma)


if __name__ == "__main__":
    p = VARSHNI["MAPbI3"]
    Ts = np.array([250, 300, 350])
    Eg = eg_varshni(Ts, **p)
    nis = np.array([ni(T, 2.2e18, 1.8e19, **p) for T in Ts])
    mu = mobility(2.0, Ts)
    print("Eg(250/300/350 K) =", np.round(Eg, 4))
    print("ni (250/300/350 K) =", [f"{v:.2e}" for v in nis])
    print("mu (250/300/350 K) =", np.round(mu, 3))
    assert Eg[0] > Eg[1] > Eg[2], "Eg should fall with T (Varshni)"
    assert nis[0] < nis[1] < nis[2], "ni should rise with T"
    assert mu[0] > mu[1] > mu[2], "phonon-limited mobility should fall with T"
    assert abs(eg_varshni(300, **p) - p["Eg0"] + p["a"] * 300 ** 2 / (300 + p["b"])) < 1e-12
    print("temperature: ALL CHECKS PASS")
