import numpy as np
from scipy import stats

def induced_projection_moments_mc(R, proj_a, h3f, h4f, kappaf, copula='gaussian', copula_params=None, n_samples=1000000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(proj_a)
    
    # Generate underlying Gaussian Y ~ N(0, R)
    L = np.linalg.cholesky(R)
    Z_norm = rng.standard_normal((n_samples, n))
    Y = Z_norm @ L.T
    
    if copula == 't':
        nu = copula_params if copula_params is not None else 4.0
        # Generate W ~ chi2(nu)
        W = rng.chisquare(nu, size=(n_samples, 1))
        # Multivariate t
        X = Y * np.sqrt(nu / W)
        # Transform margins to uniform then to standard normal
        # U = stats.t.cdf(X, df=nu)
        # Z = stats.norm.ppf(U)
        # But wait, scipy.stats.t.cdf is slow for 1M samples.
        # Alternatively, we can just use scipy!
        U = stats.t.cdf(X, df=nu)
        Z = stats.norm.ppf(U)
    elif copula == 'gaussian':
        Z = Y
    else:
        raise ValueError(f"Unknown copula: {copula}")
        
    # Hermite transform
    # q_c = a_c * kappa_c * (Z_c + h3_c(Z_c^2 - 1) + h4_c(Z_c^3 - 3Z_c))
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
    return var, skew, kurt

