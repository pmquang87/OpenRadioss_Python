import numpy as np
from scipy import stats
from scipy.optimize import brentq

def _solve_pair_rho_t_copula(target, h3_1, h4_1, kappa_1, h3_2, h4_2, kappa_2, nu, n_samples=100000, seed=42):
    rng = np.random.default_rng(seed)
    Z1_norm = rng.standard_normal(n_samples)
    Z2_indep = rng.standard_normal(n_samples)
    W = rng.chisquare(nu, size=n_samples)
    sqrt_nu_W = np.sqrt(nu / W)
    
    def obj(rho_U):
        Z2_norm = rho_U * Z1_norm + np.sqrt(1.0 - rho_U**2) * Z2_indep
        X1 = Z1_norm * sqrt_nu_W
        X2 = Z2_norm * sqrt_nu_W
        Z1 = stats.norm.ppf(stats.t.cdf(X1, df=nu))
        Z2 = stats.norm.ppf(stats.t.cdf(X2, df=nu))
        q1 = kappa_1 * (Z1 + h3_1*(Z1**2 - 1.0) + h4_1*(Z1**3 - 3.0*Z1))
        q2 = kappa_2 * (Z2 + h3_2*(Z2**2 - 1.0) + h4_2*(Z2**3 - 3.0*Z2))
        return np.corrcoef(q1, q2)[0, 1] - target
        
    try:
        return brentq(obj, -0.999, 0.999)
    except ValueError:
        return np.sign(target) * 0.999

def _induced_projection_moments_t_copula(R, proj_a, h3f, h4f, kappaf, nu, n_samples=100000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(proj_a)
    L = np.linalg.cholesky(R)
    Z_norm = rng.standard_normal((n_samples, n))
    Y = Z_norm @ L.T
    
    W = rng.chisquare(nu, size=(n_samples, 1))
    X = Y * np.sqrt(nu / W)
    
    U = stats.t.cdf(X, df=nu)
    Z = stats.norm.ppf(U)
    
    Z2 = Z**2
    Z3 = Z**3
    q = proj_a * kappaf * (Z + h3f * (Z2 - 1.0) + h4f * (Z3 - 3.0 * Z))
    
    s = np.sum(q, axis=1)
    
    var = np.var(s)
    std = np.sqrt(var)
    if var > 0:
        skew = np.mean((s - np.mean(s))**3) / (std**3)
        kurt = np.mean((s - np.mean(s))**4) / (std**4)
    else:
        skew = 0.0
        kurt = 3.0
    return float(var), float(skew), float(kurt)

