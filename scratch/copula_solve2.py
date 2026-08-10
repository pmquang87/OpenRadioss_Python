import numpy as np
from scipy import stats
from scipy.optimize import brentq
import time

def solve_pair_mc(target, h3_1, h4_1, kappa_1, h3_2, h4_2, kappa_2, nu, n_samples=100000, seed=42):
    rng = np.random.default_rng(seed)
    Z1_norm = rng.standard_normal(n_samples)
    Z2_indep = rng.standard_normal(n_samples)
    W = rng.chisquare(nu, size=n_samples)
    sqrt_nu_W = np.sqrt(nu / W)
    
    def obj(rho_U):
        Z2_norm = rho_U * Z1_norm + np.sqrt(1 - rho_U**2) * Z2_indep
        X1 = Z1_norm * sqrt_nu_W
        X2 = Z2_norm * sqrt_nu_W
        Z1 = stats.norm.ppf(stats.t.cdf(X1, df=nu))
        Z2 = stats.norm.ppf(stats.t.cdf(X2, df=nu))
        q1 = kappa_1 * (Z1 + h3_1*(Z1**2 - 1.0) + h4_1*(Z1**3 - 3.0*Z1))
        q2 = kappa_2 * (Z2 + h3_2*(Z2**2 - 1.0) + h4_2*(Z2**3 - 3.0*Z2))
        return np.corrcoef(q1, q2)[0, 1] - target
        
    try:
        # We assume monotonicity
        return brentq(obj, -0.999, 0.999)
    except ValueError:
        return np.sign(target) * 0.999

t0 = time.time()
print(solve_pair_mc(0.5, 0.0, 0.02, 0.95, 0.0, 0.02, 0.95, 4.0))
print(time.time() - t0)
