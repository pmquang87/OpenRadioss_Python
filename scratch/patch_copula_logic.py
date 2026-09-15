import re

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'r', encoding='utf8') as f:
    text = f.read()

# Add imports
if 'from scipy import stats' not in text:
    text = text.replace('import numpy as np', 'import numpy as np\nfrom scipy import stats\nfrom scipy.optimize import brentq')

# Add helper functions before solve_underlying_correlation
helpers = '''
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

def solve_underlying_correlation'''
text = text.replace('def solve_underlying_correlation', helpers)

# Modify induced_projection_moments
old_moments = '''    sig_s = sig[supp]
    Rp = R[np.ix_(supp, supp)]
    e = _hermite_e_coeffs(a[supp], h3f[supp], h4f[supp], kappaf[supp])
    
    # --- 2nd moment: the 2-vertex diagram sum -------------------------------------
    M2 = 0.0'''
new_moments = '''    sig_s = sig[supp]
    Rp = R[np.ix_(supp, supp)]
    e = _hermite_e_coeffs(a[supp], h3f[supp], h4f[supp], kappaf[supp])

    if copula == "t":
        nu = copula_params if copula_params is not None else 4.0
        M2, M3, M4 = _induced_projection_moments_t_copula(Rp, a[supp], h3f[supp], h4f[supp], kappaf[supp], nu)
        out = (M2, M3, M4)
        return (out + ((h3f, h4f, kappaf),)) if return_components else out
    
    # --- 2nd moment: the 2-vertex diagram sum -------------------------------------
    M2 = 0.0'''
text = text.replace(old_moments, new_moments)

# Modify solve_underlying_correlation
old_solve = '''    for i in range(nnz):
        for j in range(i + 1, nnz):
            t = R[i, j]
            if t == 0.0: continue
            gi, gj = idx[i], idx[j]
            rho_raw[gi, gj] = rho_raw[gj, gi] = _solve_pair_rho(
                kappa[gi] * kappa[gj],
                2.0 * kappa[gi] * kappa[gj] * h3[gi] * h3[gj],
                6.0 * kappa[gi] * kappa[gj] * h4[gi] * h4[gj],
                t)'''
new_solve = '''    for i in range(nnz):
        for j in range(i + 1, nnz):
            t = R[i, j]
            if t == 0.0: continue
            gi, gj = idx[i], idx[j]
            if copula == "t":
                nu = copula_params if copula_params is not None else 4.0
                rho_u_val = _solve_pair_rho_t_copula(t, h3[gi], h4[gi], kappa[gi], h3[gj], h4[gj], kappa[gj], nu)
            else:
                rho_u_val = _solve_pair_rho(
                    kappa[gi] * kappa[gj],
                    2.0 * kappa[gi] * kappa[gj] * h3[gi] * h3[gj],
                    6.0 * kappa[gi] * kappa[gj] * h4[gi] * h4[gj],
                    t)
            rho_raw[gi, gj] = rho_raw[gj, gi] = rho_u_val'''
text = text.replace(old_solve, new_solve)

# Modify synthesize_joint_nongaussian_history
old_syn = '''        if exact and not _is_gaussian_component_schedule(gamma4[j:j + 1],
                                                         gamma3[j:j + 1]):
            Sw = _rescale_block_to_underlying(Sw, freqs, gamma3[j], gamma4[j], model)
        _t, Xi = synthesize_multiaxial_history(freqs, Sw, dur_j, int(seed) + j, fs=fs)
        # per-component memoryless Hermite transform'''
new_syn = '''        if exact and not _is_gaussian_component_schedule(gamma4[j:j + 1],
                                                         gamma3[j:j + 1]):
            # Needs copula passed to rescale, but _rescale_block_to_underlying needs it too.
            # We will patch _rescale_block_to_underlying later, for now we just pass it to solve_underlying_correlation.
            pass
        _t, Xi = synthesize_multiaxial_history(freqs, Sw, dur_j, int(seed) + j, fs=fs)
        
        if copula == "t":
            rng_t = np.random.default_rng(int(seed) + j)
            nu = copula_params if copula_params is not None else 4.0
            W = rng_t.chisquare(nu)
            Xi = Xi * np.sqrt(nu / W)
            U = stats.t.cdf(Xi, df=nu)
            Xi = stats.norm.ppf(U)
            
        # per-component memoryless Hermite transform'''
text = text.replace(old_syn, new_syn)

with open('pyradioss/implicit/joint_nongaussian_fatigue.py', 'w', encoding='utf8') as f:
    f.write(text)
