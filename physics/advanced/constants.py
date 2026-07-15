"""Shared physical constants and a small AM1.5G helper (SI unless noted)."""
import numpy as np
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
Q = 1.602176634e-19; K_B = 1.380649e-23; EPS0 = 8.854187817e-12
H = 6.62607015e-34; C = 2.99792458e8
T_REF = 300.0

def vt(T=T_REF):
    return K_B * T / Q

def am15g(nw=200):
    """Coarse but monotone AM1.5G proxy (W m^-2 nm^-1) on 300-1100 nm,
    normalised to ~1000 W/m^2. Real spectra/n,k can be plugged in; this keeps
    the module self-contained and is used only for relative generation
    profiles and conservation tests, not absolute Jsc claims."""
    wl = np.linspace(300, 1100, nw)
    irr = 1.5 * np.exp(-((wl - 600) / 250) ** 2) + 0.2 * np.exp(-((wl - 450) / 60) ** 2)
    irr = np.clip(irr, 1e-6, None)
    irr *= 1000.0 / _trapz(irr, wl)
    return wl, irr

def photon_flux(wl_nm, irr):
    """Spectral photon flux (m^-2 s^-1 nm^-1) from spectral irradiance."""
    E = H * C / (wl_nm * 1e-9)
    return irr / E
