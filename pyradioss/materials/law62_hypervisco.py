"""
LAW62 — hyper-visco-elastic foam (/MAT/LAW62, /MAT/VISC_HYP).  Solids
only: an Ogden series with PER-TERM compressibility exponents beta_i
(nu_i) plus an optional Prony-series viscoelastic overstress.

Fortran origin
--------------
* engine : ``engine/source/materials/mat/mat062/sigeps62.F`` (the
  principal-stretch evaluation, the pressure/PK2 split, the Prony
  history update and the CIMAX sound-speed bound — ported block by
  block below);
* starter: ``starter/source/materials/mat/mat062/hm_read_mat62.F``
  (defaults: alpha_i 0 -> 1, tau_i <= 0 -> 1e20, gamma_inf =
  1 - sum(gamma_i) must stay > 0, nu clamps at 0.499; per-term
  beta_i = nu_i/(1-2 nu_i) when any nu_i is given, else the global nu;
  Rflag = 2 rescales mu_i by 1/gamma_inf).

Theory
------
With principal stretches lambda_j and J = lambda_1 lambda_2 lambda_3,
each Ogden term contributes the principal 2nd Piola-Kirchhoff stress

    S_j = (2 mu_i / alpha_i / lambda_j^2) *
          (lambda_j^alpha_i - J^(-beta_i alpha_i))

and the Cauchy principal stress is sigma_j = S_j lambda_j^2 / J.  The
``beta_i`` exponent is the term's volumetric response — nu_i = 0 (the
corpus foam decks) gives beta = 0: fully decoupled, highly compressible
foam.

Viscosity (Ivisc = 1, deviatoric — the M and gamma_i/tau_i input):
the deviatoric part of the PK2 stress SD = (S - p J C^-1) J^(2/3) is
tracked per Prony term with the standard midpoint exponential update

    H_i(t+dt) = exp(-dt/tau_i) H_i(t) + exp(-dt/2tau_i) (SD(t+dt)-SD(t))

carried in the GLOBAL frame (the principal frame rotates), and the
total stress is p J C^-1 + [gamma_inf SD + sum gamma_i dev(H_i)]
J^(-2/3).  Ivisc = 2 (Flag_Visc = 1) applies the series to the FULL
stress instead.

Documented deviations of the port
---------------------------------
* the principal stretches come from the EXACT deformation gradient the
  solid kernels pass (``extra['F']``, same machinery as LAW42) — the
  Fortran diagonalizes its incrementally-accumulated total strain,
  which is the approximate version of the same quantity (ISMSTR
  variants are therefore moot here);
* the implicit/IHET tangent-stiffness feedback (ET) is not ported.

Extra state (only with Prony terms):
    sdg62 (6,)          previous deviatoric PK2 in the global frame
    h62   (nprony, 6)   Prony history H_i (global frame)
"""

from __future__ import annotations

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20


def _spectral(F):
    """Principal stretches (n, 3) and spatial principal directions
    (n, 3, 3; columns = eigenvectors) from B = F F^T."""
    B = np.einsum("nij,nkj->nik", F, F)
    w, v = np.linalg.eigh(B)
    ev = np.sqrt(np.maximum(w, _EM20))
    return ev, v


def solid_update(mat, sig, deps, epsp, dt, extra=None):
    """sigeps62.F — total-form update from extra['F'].  Returns
    (sig, epsp, c)."""
    p = mat.params
    mu = p["MU62"]
    al = p["AL62"]
    beta = p["BETA62"]
    gama = p["GAMA62"]
    taux = p["TAU62"]
    gamainf = p["GAMAINF"]
    rbulk = p["RBULK"]
    ivisc = int(p["IVISC"])
    nprony = len(gama)

    F = extra["F"]
    ev, dirp = _spectral(F)                    # (n,3), (n,3,3)
    rv = ev[:, 0] * ev[:, 1] * ev[:, 2]        # J = det F
    ec = ev ** 2
    rv_m = np.maximum(rv, _EM20)

    # ---- pressure term and principal PK2 stress -----------------------------
    pres = np.zeros(len(ev))
    S = np.zeros_like(ev)
    for i in range(len(mu)):
        fac = 2.0 * mu[i] / al[i]
        lam_al = ev ** al[i]                              # lambda^alpha
        jvol = rv_m ** (-beta[i] * al[i])                 # J^(-beta*alpha)
        fac1 = fac / rv_m
        pres += fac1 * (lam_al.sum(axis=1) / 3.0 - jvol)
        S += (fac / ec) * (lam_al - jvol[:, None])

    # ---- viscoelastic overstress (Prony) ------------------------------------
    # (skipped when the caller passes no history views — the implicit
    # statics re-evaluation path calls total-form laws with a bare
    # {"F"} extra; re-advancing the Prony history there would double
    # the relaxation step)
    if ivisc > 0 and nprony > 0 and extra is not None and "sdg62" in extra:
        rv23 = rv_m ** (2.0 / 3.0)
        ssp = pres[:, None] / ec                          # pressure part
        sd = (S - ssp * rv[:, None]) * rv23[:, None]      # scaled deviator
        if ivisc == 2:
            sd = S.copy()                                 # full stress
        # principal (diagonal) -> global frame: SDG = R diag(sd) R^T
        sdg = np.einsum("naj,nj,nbj->nab", dirp, sd, dirp)
        sdg6 = np.stack([sdg[:, 0, 0], sdg[:, 1, 1], sdg[:, 2, 2],
                         sdg[:, 0, 1], sdg[:, 1, 2], sdg[:, 0, 2]], axis=1)
        sdg0 = extra["sdg62"]
        h = extra["h62"]                                  # (n, nprony, 6)
        hp = np.zeros((len(ev), nprony, 3))
        for ii in range(nprony):
            fac = -dt / taux[ii]
            h[:, ii] = (np.exp(fac) * h[:, ii]
                        + np.exp(0.5 * fac) * (sdg6 - sdg0))
            # global H -> principal diagonal
            HM = np.empty_like(sdg)
            HM[:, 0, 0], HM[:, 1, 1], HM[:, 2, 2] = \
                h[:, ii, 0], h[:, ii, 1], h[:, ii, 2]
            HM[:, 0, 1] = HM[:, 1, 0] = h[:, ii, 3]
            HM[:, 1, 2] = HM[:, 2, 1] = h[:, ii, 4]
            HM[:, 0, 2] = HM[:, 2, 0] = h[:, ii, 5]
            hp[:, ii] = np.einsum("nai,nab,nbi->ni", dirp, HM, dirp)
        sdg0[:] = sdg6
        if ivisc == 1:
            # deviatoric projection in the strain metric
            hd = hp - (np.einsum("nij,nj->ni", hp, ec) / 3.0
                       )[:, :, None] / np.maximum(ec, _EM20)[:, None, :]
            rvm23 = rv_m ** (-2.0 / 3.0)
            S = ssp * rv[:, None] + gamainf * rvm23[:, None] * sd
            for ii in range(nprony):
                S += gama[ii] * rvm23[:, None] * hd[:, ii]
        else:                                             # ivisc == 2
            S = gamainf * S
            for ii in range(nprony):
                S += gama[ii] * hp[:, ii]

    # ---- Cauchy principal stress, back to the global frame ------------------
    cauchy = S * ec / rv_m[:, None]
    sig_t = np.einsum("naj,nj,nbj->nab", dirp, cauchy, dirp)
    sig[:, 0] = sig_t[:, 0, 0]
    sig[:, 1] = sig_t[:, 1, 1]
    sig[:, 2] = sig_t[:, 2, 2]
    sig[:, 3] = sig_t[:, 0, 1]
    sig[:, 4] = sig_t[:, 1, 2]
    sig[:, 5] = sig_t[:, 0, 2]

    # ---- sound-speed bound CIMAX (sigeps62.F lines 452-478) ------------------
    gmax = 2.0 * mu.sum()
    gvis = gmax if ivisc > 0 else 0.0
    cmax0 = (2.0 / 3.0) * gmax + rbulk
    ai = np.zeros_like(ev)
    bi = np.zeros_like(ev)
    cj = np.zeros(len(ev))
    for i in range(len(mu)):
        lam_al = ev ** al[i]
        rvl = (beta[i] + 1.0 / al[i]) * rv_m ** (-beta[i] * al[i])
        cj += 2.0 * mu[i] * rvl
        ai += 2.0 * mu[i] * lam_al
        bi += (2.0 * mu[i] / al[i]) * lam_al
    d = ai - bi + cj[:, None]
    cmax = d.max(axis=1) / rv_m
    cimax = (2.0 / 3.0) * gvis + np.maximum(cmax, cmax0)
    rho = extra.get("rho") if extra else None
    if rho is None:
        rho = mat.rho0 / rv_m                # rho = rho0/J
    c = np.sqrt(np.maximum(cimax, _EM20) / rho)
    return sig, epsp, c


# ----------------------------------------------------------------------------
# cfg-record constructor (mat_reader physics registry)
# ----------------------------------------------------------------------------

def build_law62(rec) -> Material:
    """hm_read_mat62.F: cfg attributes -> uparam equivalents."""
    q = rec.params
    nug = float(q.get("MAT_NU", 0.0) or 0.0)
    norder = int(q.get("ORDER", 0) or 0)
    nvisc = int(q.get("Order2", 0) or 0)
    flag_visc = int(q.get("Vflag", 0) or 0)
    flag_rigidity = int(q.get("Rflag", 0) or 0)

    def _arr(name, count):
        v = q.get(name) or []
        if not isinstance(v, list):
            v = [v]
        v = [float(x or 0.0) for x in v]
        v += [0.0] * (count - len(v))
        return np.asarray(v[:count])

    mu = _arr("Mu_arr", norder)
    al = _arr("Alpha_arr", norder)
    nu = _arr("Nu_arr", norder)
    gama = _arr("Gamma_arr", nvisc)
    taux = _arr("Tau_arr", nvisc)

    if norder <= 0:
        raise ValueError("LAW62 needs at least one Ogden term "
                         "(hm_read_mat62 error 559)")
    al = np.where(al == 0.0, 1.0, al)
    taux = np.where(taux <= 0.0, 1e20, taux)
    gamainf = 1.0
    if nvisc > 0:
        if np.any((gama < 0.0) | (gama > 1.0)):
            raise ValueError("LAW62: every gamma_i must be in [0, 1] "
                             "(hm_read_mat62 error 560)")
        gamainf = 1.0 - gama.sum()
        if gamainf <= 0.0:
            raise ValueError("LAW62: sum(gamma_i) must stay below 1 "
                             "(hm_read_mat62 error 2084)")
    nug = min(nug, 0.499)
    if nvisc > 0 and flag_rigidity == 2:
        mu = mu / gamainf
    gs = mu.sum()
    if gs < 0.0:
        raise ValueError("LAW62: sum(mu_i) must be positive "
                         "(hm_read_mat62 error 846)")
    nu = np.where(nu >= 0.5, 0.499, nu)
    if np.any(nu != 0.0):
        beta = nu / (1.0 - 2.0 * nu)
        rbulk = float((2.0 * mu * (1.0 / 3.0 + beta)).sum())
        nug = 0.5 * (3.0 * rbulk - 2.0 * gs) / (3.0 * rbulk + gs)
    else:
        beta = np.full(norder, nug / (1.0 - 2.0 * nug))
        rbulk = (2.0 / 3.0) * gs * (1.0 + nug) / max(1e-20, 1.0 - 2.0 * nug)
    ivisc = 0 if nvisc == 0 else (2 if flag_visc == 1 else 1)

    params = {
        # generic elastic constants: G = sum(mu_i), the upstream
        # PARMAT(2) young modulus E = 2 G (1 + nu)
        "E": 2.0 * gs * (1.0 + nug), "nu": nug,
        "MU62": mu, "AL62": al, "BETA62": beta,
        "GAMA62": gama, "TAU62": taux,
        "GAMAINF": gamainf, "RBULK": rbulk, "IVISC": ivisc,
        "NPRONY": nvisc,
    }
    return Material(id=rec.id, law=62, rho0=rec.density,
                    title=rec.title, params=params)


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY.setdefault("LAW62", build_law62)
    MAT_PHYSICS_REGISTRY.setdefault("VISC_HYP", build_law62)


_register()
