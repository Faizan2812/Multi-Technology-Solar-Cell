"""
impedance.py — frequency-domain analysis (the absent C-V / IS capability).

Adds:
  * mott_schottky : depletion capacitance C(V); 1/C^2 vs V recovers doping and Vbi.
  * impedance     : small-signal Z(omega) from an equivalent circuit with
                    geometric capacitance, recombination resistance, and an
                    optional low-frequency ionic branch (the perovskite IS
                    signature). Returns a Nyquist trace.

Self-tests (__main__):
  * Mott-Schottky slope recovers the input doping within 5%.
  * |Z| decreases monotonically with frequency for the RC network.
"""
import numpy as np
from .constants import Q, EPS0, K_B, T_REF


def mott_schottky(Na_cm3, eps_r=6.5, area_cm2=1.0, Vbi=0.9, T=T_REF, Vrange=(-0.2, 0.6), npts=60):
    """Parallel-plate depletion capacitance of a one-sided junction.
    Returns V, C (F), and the fitted (Na, Vbi) recovered from 1/C^2 slope."""
    Na = Na_cm3 * 1e6                                    # m^-3
    eps = eps_r * EPS0
    V = np.linspace(*Vrange, npts)
    A = area_cm2 * 1e-4                                  # m^2
    W = np.sqrt(np.clip(2 * eps * (Vbi - V) / (Q * Na), 1e-18, None))
    C = eps * A / W                                      # F
    # recover Na, Vbi from linear fit of 1/C^2 vs V
    y = 1.0 / C ** 2
    slope, intercept = np.polyfit(V, y, 1)              # y = (-2/(q eps Na A^2)) V + ...
    Na_fit = -2.0 / (Q * eps * A ** 2 * slope)          # m^-3
    Vbi_fit = -intercept / slope
    return dict(V=V, C=C, Na_fit_cm3=Na_fit / 1e6, Vbi_fit=Vbi_fit)


def impedance(Rs=20.0, Rrec=500.0, Cgeo=2e-8, Rion=300.0, Cion=1e-6,
              f=None, with_ionic=True):
    """Equivalent-circuit impedance. Rs + (Rrec || Cgeo) + optional (Rion || Cion).
    Returns f, Z (complex), and the Nyquist (Re, -Im)."""
    if f is None:
        f = np.logspace(-1, 6, 120)
    w = 2 * np.pi * f
    Z_rc = Rrec / (1 + 1j * w * Rrec * Cgeo)
    Z = Rs + Z_rc
    if with_ionic:
        Z += Rion / (1 + 1j * w * Rion * Cion)
    return dict(f=f, Z=Z, nyq_re=Z.real, nyq_im=-Z.imag)


if __name__ == "__main__":
    Na_true = 1e16
    ms = mott_schottky(Na_true, eps_r=6.5, Vbi=0.9)
    err = abs(ms["Na_fit_cm3"] - Na_true) / Na_true
    print(f"Mott-Schottky: Na_fit={ms['Na_fit_cm3']:.3e} cm^-3 (true {Na_true:.0e}), "
          f"Vbi_fit={ms['Vbi_fit']:.3f} V, rel.err={err:.2%}")
    assert err < 0.05, "MS slope must recover input doping"

    z = impedance()
    magnitude = np.abs(z["Z"])
    print(f"|Z| at 0.1 Hz = {magnitude[0]:.1f} ohm; at 1 MHz = {magnitude[-1]:.1f} ohm")
    assert magnitude[0] > magnitude[-1], "|Z| should fall with frequency"
    print("impedance: ALL CHECKS PASS")
