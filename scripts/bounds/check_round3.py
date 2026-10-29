"""Deterministic/synthetic checks; never reads future corpora or measurements."""
import numpy as np
import conjecture_future as model
from scipy.special import ndtr
from scipy.stats import norm


def main():
    rng=np.random.default_rng(7)
    x=rng.standard_normal((4096,16))+np.array([2]+[0]*15)
    st=model.statistics(x)
    w=st['bin_weight']
    assert abs(w.sum()-1)<1e-12
    assert abs(w@st['bin_mean'])<1e-12
    assert abs(w@st['bin_scale']-1)<1e-12
    assert abs(w@(st['bin_var']+st['bin_mean']**2)-st['var_l'])<1e-12
    # One Gaussian component reconstructs the complete directional covariance.
    t=rng.standard_normal(16); t/=np.linalg.norm(t)
    a=t@st['mean_eig']; h=t@st['regression_eig']; var=(t*t)@st['eig']
    resid=var-st['var_l']*h*h
    reconstructed=resid*(w@st['bin_scale'])+h*h*(w@(st['bin_var']+st['bin_mean']**2))
    assert abs(reconstructed-var)<1e-12
    # Gaussian maximum inverse CDF, including numerical tail stability.
    for n in (2000,70000):
        p=.00125
        z=norm.isf(-np.expm1(np.log(p)/n))
        assert abs(np.exp(n*np.log1p(-ndtr(-z)))-p)<1e-12
    # Degenerate isotropic centred model: exact maximum distribution independent of T.
    st=dict(d=8,n=2000,r=.01,eig=np.ones(8)*.1,
            mean_eig=np.array([.01]+[0.]*7),regression_eig=np.array([1.]+[0.]*7),
            var_l=.1,bin_mean=np.array([0.]),bin_var=np.array([.1]),
            bin_scale=np.array([1.]),bin_weight=np.array([1.]))
    saved_power=model.POWER
    model.POWER=11
    try:
        row=model.evaluate(st,[(50,20000)])[0]
    finally:
        model.POWER=saved_power
    assert abs(row['corrected']-row['gaussian'])<1e-8
    assert abs(row['constant']-row['gaussian'])<1e-8
    print('PASS: mixture moment identities, stable maximum CDF, Gaussian/isotropic reductions')

if __name__=='__main__':
    main()
