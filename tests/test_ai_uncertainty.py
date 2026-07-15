"""Tests for uncertainty quantification and conditional PINN."""
import numpy as np
from ai import uncertainty as u
from ai import conditional_pinn as c

def test_uq_widens_for_low_confidence():
    f = lambda dct: 25 - 8*(dct["x"]-0.5)**2
    hi = u.propagate_confidence(f, {"x":0.5}, {"x":"HIGH"})
    lo = u.propagate_confidence(f, {"x":0.5}, {"x":"LOW"})
    assert (lo["ci95"][1]-lo["ci95"][0]) > (hi["ci95"][1]-hi["ci95"][0])

def test_conditional_pinn_learns():
    x = np.linspace(0,1,30); pset=[np.array([1.0]),np.array([3.0])]
    tgt=[(-p[0])*x*(1-x)/2 for p in pset]
    net=c.ConditionalPINN(n_params=1,hidden=(20,20))
    hist=net.fit(x,pset,tgt,iters=200,lr=3e-2)
    assert hist[-1] < hist[0]*0.7
