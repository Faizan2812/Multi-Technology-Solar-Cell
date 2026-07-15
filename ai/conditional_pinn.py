"""
conditional_pinn.py — a parameterized PINN that generalizes across devices.

The existing PINN is trained per device. This module demonstrates a *conditional*
PINN: the network takes the device parameters as additional inputs, so one trained
model serves a family of devices. It is implemented in pure numpy with a
finite-difference Poisson residual so it runs anywhere; the production form would
use the tool's torch autograd PINN (ai/pinn_real.py) for exact residuals. The
contribution shown here is the conditioning architecture and that it learns a
parameter-dependent solution, not a publication-grade trained model.

Self-tests (__main__):
  * training reduces the combined data+physics loss;
  * the trained network produces *different* potential profiles for different
    conditioning parameters (i.e. it actually uses the conditioning input).
"""
import numpy as np


def _init(sizes, seed=0):
    rng = np.random.default_rng(seed); W = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        W.append([rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)])
    return W


def _forward(W, X):
    h = X
    for i, (w, b) in enumerate(W):
        z = h @ w + b
        h = np.tanh(z) if i < len(W) - 1 else z
    return h


class ConditionalPINN:
    """Inputs: [x_norm, *device_params]. Output: psi(x; params)."""
    def __init__(self, n_params, hidden=(32, 32), seed=0):
        self.W = _init([1 + n_params, *hidden, 1], seed)

    def predict(self, x_norm, params):
        params = np.atleast_2d(params)
        X = np.column_stack([x_norm, np.repeat(params, len(x_norm), axis=0)])
        return _forward(self.W, X).ravel()

    def _loss(self, W, x, pset, targets, lam_pde=0.1):
        loss = 0.0
        for p, tgt in zip(pset, targets):
            X = np.column_stack([x, np.tile(p, (len(x), 1))])
            psi = _forward(W, X).ravel()
            data = np.mean((psi - tgt) ** 2)
            d2 = np.gradient(np.gradient(psi, x), x)
            pde = np.mean((d2 + p[0]) ** 2)         # toy Poisson: psi'' = -rho(p)
            loss += data + lam_pde * pde
        return loss / len(pset)

    def fit(self, x, pset, targets, iters=300, lr=2e-2, lam_pde=0.1, seed=0):
        rng = np.random.default_rng(seed); eps = 1e-4
        hist = []
        for it in range(iters):
            base = self._loss(self.W, x, pset, targets, lam_pde)
            hist.append(base)
            for layer in self.W:                     # SPSA-style stochastic gradient
                for arr in layer:
                    pert = rng.normal(0, 1, arr.shape)
                    arr += eps * pert
                    up = self._loss(self.W, x, pset, targets, lam_pde)
                    arr -= 2 * eps * pert
                    dn = self._loss(self.W, x, pset, targets, lam_pde)
                    arr += eps * pert
                    grad = (up - dn) / (2 * eps) * pert
                    arr -= lr * grad
        hist.append(self._loss(self.W, x, pset, targets, lam_pde))
        return hist


if __name__ == "__main__":
    x = np.linspace(0, 1, 40)
    # two devices parameterised by a single "charge" parameter p -> target psi = -p x(1-x)/2
    pset = [np.array([1.0]), np.array([3.0])]
    targets = [(-p[0]) * x * (1 - x) / 2 for p in pset]
    net = ConditionalPINN(n_params=1, hidden=(24, 24))
    hist = net.fit(x, pset, targets, iters=250, lr=3e-2)
    print(f"loss start={hist[0]:.4f} -> end={hist[-1]:.4f}")
    assert hist[-1] < hist[0] * 0.6, "training must reduce the loss"
    psi1 = net.predict(x, pset[0]); psi2 = net.predict(x, pset[1])
    diff = np.mean(np.abs(psi1 - psi2))
    print(f"mean |psi(p=1) - psi(p=3)| = {diff:.4f} (conditioning is used)")
    assert diff > 1e-2, "network must respond to the conditioning parameter"
    print("conditional_pinn: ALL CHECKS PASS")
