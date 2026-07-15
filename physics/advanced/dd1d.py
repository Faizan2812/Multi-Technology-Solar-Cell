"""
dd1d.py — compact 1-D drift-diffusion solver (Gummel map + Scharfetter-Gummel).

Solves Poisson + electron/hole continuity for a single-absorber p-i-n-like cell
with ohmic contacts, SRH + radiative recombination, and a uniform or supplied
generation rate. Returns band diagram, carrier profiles, and J-V. Normalised
(de Mari-style) variables; Bernoulli flux for stability.

This is a genuine PDE solve, not a diode fit. Self-test sweeps V and reports
Voc/Jsc/FF/PCE for a perovskite-like device.
"""
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve

Q = 1.602176634e-19; KB = 1.380649e-23; EPS0 = 8.854187817e-12


def bern(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-10
    out[m] = x[m] / (np.expm1(x[m]))
    return out


class Device:
    def __init__(self, L=400e-9, Nx=200, Eg=1.55, chi=3.9, eps_r=6.5,
                 Nc=2.2e24, Nv=1.8e25, mu_n=2e-4, mu_p=2e-4, tau=1e-7,
                 Brad=4.78e-17, Nd=1e21, Na=1e21, T=300.0, Gavg=2.5e27):
        # SI: densities m^-3, mobilities m^2/Vs, L m, G m^-3 s^-1
        self.__dict__.update(locals()); del self.self
        self.Vt = KB * T / Q
        self.eps = eps_r * EPS0
        self.ni = np.sqrt(Nc * Nv) * np.exp(-Eg * Q / (2 * KB * T))
        self.x = np.linspace(0, L, Nx)
        self.h = self.x[1] - self.x[0]
        # doping: n-type near x=0 contact, p-type near x=L contact (p-i-n)
        self.C = np.zeros(Nx)
        nseg = max(1, Nx // 8)
        self.C[:nseg] = Nd
        self.C[-nseg:] = -Na
        self.Vbi = self.Vt * np.log(Nd * Na / self.ni ** 2)

    def _equilibrium_psi(self):
        # nonlinear Poisson at equilibrium via Newton, psi in Volts
        Vt, eps, ni, h = self.Vt, self.eps, self.ni, self.h
        psi = np.zeros(self.Nx)
        # initial guess from doping
        psi = Vt * np.arcsinh(self.C / (2 * ni))
        for _ in range(100):
            n = ni * np.exp(psi / Vt); p = ni * np.exp(-psi / Vt)
            res = np.zeros(self.Nx)
            lap = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / h ** 2
            res = eps * lap + Q * (p - n + self.C)
            dres = -Q / Vt * (n + p)
            main = -2 * eps / h ** 2 + dres
            off = eps / h ** 2
            A = diags([off * np.ones(self.Nx - 1), main, off * np.ones(self.Nx - 1)], [-1, 0, 1]).tolil()
            # Dirichlet at contacts
            A[0, :] = 0; A[0, 0] = 1; res[0] = 0
            A[-1, :] = 0; A[-1, -1] = 1; res[-1] = 0
            dpsi = spsolve(csr_matrix(A), -res)
            psi += np.clip(dpsi, -0.5, 0.5)
            if np.max(np.abs(dpsi)) < 1e-8:
                break
        return psi

    def solve(self, V=0.0, illuminated=True, gummel_iters=120):
        Vt, eps, ni, h, Nx = self.Vt, self.eps, self.ni, self.h, self.Nx
        G = np.full(Nx, self.Gavg if illuminated else 0.0)
        G[:max(1, Nx // 8)] = 0; G[-max(1, Nx // 8):] = 0     # no gen in doped contacts
        psi = self._equilibrium_psi()
        # contact boundary values (ohmic): majority carriers fix n,p; bias splits QFL
        psiL = psi[0] + V / 1.0
        psi[0] = psi[0]; psi[-1] = psi[-1] - V
        n = ni * np.exp(psi / Vt); p = ni * np.exp(-psi / Vt)
        n0, nL = n[0], n[-1]; p0, pL = p[0], p[-1]
        for it in range(gummel_iters):
            # ---- electron continuity (SG), solve for n given psi ----
            dpsi = np.diff(psi) / Vt
            Bp = bern(dpsi); Bm = bern(-dpsi)
            a = self.mu_n * Vt / h
            lo = a * Bm[:-1]; up = a * Bp[1:]
            mid = -(a * Bp[:-1] + a * Bm[1:])
            R = (n * p - ni ** 2) / (self.tau * (n + p + 2 * ni) ) + self.Brad * (n * p - ni ** 2)
            rhs = (R - G)[1:-1] * h
            main = np.empty(Nx); main[1:-1] = mid; main[0] = main[-1] = 1.0
            lower = np.zeros(Nx - 1); lower[:-1] = lo
            upper = np.zeros(Nx - 1); upper[1:] = up
            b = np.empty(Nx); b[1:-1] = rhs; b[0] = n0; b[-1] = nL
            A = diags([lower, main, upper], [-1, 0, 1], format="csr")
            n = np.clip(spsolve(A, b), 1e2, 1e30)
            # ---- hole continuity ----
            a = self.mu_p * Vt / h
            lo = a * Bp[:-1]; up = a * Bm[1:]
            mid = -(a * Bm[:-1] + a * Bp[1:])
            R = (n * p - ni ** 2) / (self.tau * (n + p + 2 * ni)) + self.Brad * (n * p - ni ** 2)
            rhs = -(R - G)[1:-1] * h
            main = np.empty(Nx); main[1:-1] = mid; main[0] = main[-1] = 1.0
            lower = np.zeros(Nx - 1); lower[:-1] = lo
            upper = np.zeros(Nx - 1); upper[1:] = up
            b = np.empty(Nx); b[1:-1] = rhs; b[0] = p0; b[-1] = pL
            A = diags([lower, main, upper], [-1, 0, 1], format="csr")
            p = np.clip(spsolve(A, b), 1e2, 1e30)
            # ---- Poisson update (Gummel, damped) ----
            lap = (np.roll(psi, -1) - 2 * psi + np.roll(psi, 1)) / h ** 2
            res = eps * lap + Q * (p - n + self.C)
            dres = -Q / Vt * (n + p)
            main = -2 * eps / h ** 2 + dres
            off = eps / h ** 2
            A = diags([off * np.ones(Nx - 1), main, off * np.ones(Nx - 1)], [-1, 0, 1]).tolil()
            A[0, :] = 0; A[0, 0] = 1; res[0] = 0
            A[-1, :] = 0; A[-1, -1] = 1; res[-1] = 0
            dpsi_u = spsolve(csr_matrix(A), -res)
            psi += np.clip(dpsi_u, -0.1, 0.1)
            if np.max(np.abs(dpsi_u)) < 1e-9:
                break
        # current density (electron SG flux, A/m^2), averaged over interior
        dpsi = np.diff(psi) / Vt
        Jn = Q * self.mu_n * Vt / h * (bern(dpsi) * n[1:] - bern(-dpsi) * n[:-1])
        Jp = -Q * self.mu_p * Vt / h * (bern(-dpsi) * p[1:] - bern(dpsi) * p[:-1])
        J = np.mean(Jn + Jp)
        return dict(x=self.x, psi=psi, n=n, p=p, J=J,
                    Ec=-(self.chi) - psi, Ev=-(self.chi) - psi - self.Eg)

    def jv(self, Vmax=1.25, npts=26):
        Vs = np.linspace(0, Vmax, npts)
        Js = np.array([self.solve(V=V)["J"] for V in Vs])   # A/m^2
        JmA = Js / 10.0                                       # mA/cm^2
        # photocurrent sign convention: J_light = Jsc - J(dark-like); here J already net
        Jcell = JmA[0] - JmA + 0.0                            # shift so V=0 ~ Jsc
        Jsc = Jcell[0]
        P = Vs * Jcell
        idx = np.argmax(P)
        Pmax = P[idx]
        # Voc where Jcell crosses 0
        sign = np.where(Jcell <= 0)[0]
        Voc = Vs[sign[0]] if len(sign) else Vs[-1]
        FF = Pmax / (Jsc * Voc) if Jsc * Voc > 0 else 0
        PCE = Pmax  # mW/cm^2 per 100 mW/cm^2 -> %
        return dict(V=Vs, J=Jcell, Jsc=Jsc, Voc=Voc, FF=FF, PCE=PCE, Vmpp=Vs[idx], Pmax=Pmax)


if __name__ == "__main__":
    d = Device()
    print(f"ni = {d.ni:.3e} m^-3, Vbi = {d.Vbi:.3f} V")
    eq = d.solve(V=0.0, illuminated=False)
    print(f"equilibrium n range: {eq['n'].min():.1e}..{eq['n'].max():.1e}")
    jv = d.jv()
    print(f"Jsc={jv['Jsc']:.2f} mA/cm^2  Voc={jv['Voc']:.3f} V  FF={jv['FF']:.3f}  PCE={jv['PCE']:.2f}%")
