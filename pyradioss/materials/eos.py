"""
/EOS — equations of state for solid elements (M6, deferred from M3).

Fortran origin: ``starter/source/materials/eos/hm_read_eos.F`` (readers,
one per EOS type) and ``engine/source/materials/eos/eosmain.F`` +
``eos/polynomial.F``, ``eos/idealgas.F`` (the engine solve). The M3 note
in the PORTING_GUIDE explains why this waited: an energy-dependent
pressure is a SOLVER-loop change, not just a material — the element must
carry a relative-volume state and integrate its internal energy
implicitly coupled to the pressure.

Theory — the polynomial EOS and the E-p coupling
------------------------------------------------
The ported form is the compaction polynomial (the Radioss
/EOS/POLYNOMIAL, also LS-DYNA's *EOS_LINEAR_POLYNOMIAL):

    p(mu, E) = C0 + C1 mu + C2 mubar^2 + C3 mu^3 + (C4 + C5 mu) E

with mu = rho/rho0 - 1 = 1/J - 1 the compression, mubar = max(mu, 0)
(the quadratic term is dropped in tension — the Radioss convention), and
E the internal energy PER UNIT REFERENCE VOLUME. The ideal gas is the
special case C0..C3 = 0, C4 = C5 = gamma - 1:

    p = (gamma - 1)(1 + mu) E = (gamma - 1) rho e      (e specific)

**The energy equation is solved implicitly with the pressure.** Over a
cycle the element's energy (per V0) changes by the deviator work and the
pdV work of the pressure itself,

    E_new = E_old + dE_dev - 1/2 (p_old + p_new) dv,     dv = J_new-J_old

and p_new depends on E_new. Because p is LINEAR in E, the coupled pair
has a closed form (this is exactly eosmain's update):

    write p_new = A(mu) + B(mu) E_new,
        A = C0 + C1 mu + C2 mubar^2 + C3 mu^3,     B = C4 + C5 mu
    =>  E_new = [E_old + dE_dev - 1/2 dv (p_old + A)] / (1 + 1/2 B dv)
        p_new = A + B E_new

Integrating E explicitly instead (p from E_old) drifts the isentrope at
first order in dv per cycle; the implicit form reproduces the ideal-gas
adiabat p V^gamma = const to second order (checked against the closed
form by the M6 unit tests).

**Shock heating.** The bulk-viscosity work (the artificial viscosity
that spreads a shock over a few elements) heats the element: the kernels
feed their q-work into E as well, which is what puts the computed shock
states on the RANKINE-HUGONIOT curve rather than the isentrope — without
it a strong shock's temperature (and hence pressure, through the E term)
is badly underestimated. (The port books the q work into E with the same
half-step-deferred trapezoidal split as the eint ledger — one cycle of
lag for the p evaluation, second-order and non-secular.)

**Sound speed.** The time step must see the EOS stiffness. Along the
adiabat (dE = -p dv per V0, dv = -dmu/(1+mu)^2):

    rho0 c^2 = dp/dmu|_E + B(mu) p / (1 + mu)^2

For the ideal gas this collapses to the textbook c^2 = gamma p / rho.
The kernels add the constitutive shear stiffness (4G/3)/rho on top and
feed the result into the Courant condition — the same 'the law feeds its
own sound speed' lesson as LAW42 (M3).

The EOS attaches to a material like /FAIL does (``/EOS/type/mat_ID``)
and REPLACES the law's own pressure: the law keeps its deviatoric
response (exact for the ported isotropic-elastic / J2-plastic laws 1, 2
and 36, whose deviator never depends on pressure), and the kernel
rebuilds sigma = s_dev - p_eos I every cycle.
"""

from __future__ import annotations

import numpy as np


def _coefficients_gruneisen(eos, mu: np.ndarray):
    """A(mu), B(mu) for Mie-Grüneisen EOS (common_source/eos/gruneisen.F)."""
    p = eos.params
    c = p.get("c", 0.0)
    s1 = p.get("s1", 0.0)
    s2 = p.get("s2", 0.0)
    s3 = p.get("s3", 0.0)
    gamma0 = p.get("gamma0", 0.0)
    a = p.get("a", 0.0)
    rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

    mu_pos = np.maximum(mu, 0.0)
    eta = 1.0 + mu
    xx = np.where(mu > 0.0, mu / np.maximum(eta, 1e-12), 0.0)
    ff = 1.0 + (1.0 - 0.5 * gamma0) * mu - 0.5 * a * (mu_pos ** 2)
    fg = 1.0 - (s1 - 1.0 + s2 * xx + s3 * (xx ** 2)) * mu
    fg_safe = np.where(np.abs(fg) < 1e-12, 1e-12, fg)
    fac = np.where(mu > 0.0, ff / (fg_safe ** 2), 1.0)

    A = fac * rho0 * (c ** 2) * mu
    B = gamma0 + a * mu
    return A, B


def _coefficients_tillotson(eos, mu: np.ndarray, e: np.ndarray | float | None = None):
    """A(mu), B(mu, e) for Tillotson EOS (common_source/eos/tillotson.F)."""
    p = eos.params
    c1 = p.get("c1", 0.0)
    c2 = p.get("c2", 0.0)
    a = p.get("a", 0.0)
    b = p.get("b", 0.0)
    er = p.get("er", p.get("ezero", p.get("e0_ref", 1.0)))
    es = p.get("es", p.get("esubl", 0.0))
    vs = p.get("vs", p.get("vsubl", 1.0))
    alpha = p.get("alpha", 0.0)
    beta = p.get("beta", 0.0)

    if e is None:
        e = p.get("e0", 0.0)
    e = np.asarray(e, dtype=float)

    eta = 1.0 + mu
    df = 1.0 / np.maximum(eta, 1e-12)
    xx = np.where(np.abs(eta) > 1e-12, mu / eta, 0.0)
    expa = np.exp(-alpha * (xx ** 2))
    expb = np.exp(beta * xx)

    hot = (mu < 0.0) & ((df > vs) | ((df <= vs) & (e >= es)))
    facc1 = np.where(hot, expa * expb, 1.0)
    facc2 = np.where(mu >= 0.0, 1.0, 0.0)
    facpb = np.where(hot, expa, 1.0)

    omega = 1.0 + e / np.maximum(er * (eta ** 2), 1e-15)
    A = facc1 * c1 * mu + facc2 * c2 * (mu ** 2)
    B = (a + facpb * b / omega) * eta
    return A, B


def _coefficients_jwl(eos, mu: np.ndarray):
    """A(mu), B(mu) for JWL EOS (common_source/eos/jwl.F)."""
    p = eos.params
    a = p.get("a", 0.0)
    b = p.get("b", 0.0)
    r1 = p.get("r1", 0.0)
    r2 = p.get("r2", 0.0)
    omega = p.get("omega", 0.0)
    psh = p.get("psh", 0.0)

    eta = np.maximum(1.0 + mu, 1e-12)
    df = 1.0 / eta
    r1df = r1 * df
    r2df = r2 * df
    er1df = np.exp(-r1df)
    er2df = np.exp(-r2df)

    term1 = a * (1.0 - omega / np.maximum(r1df, 1e-12)) * er1df
    term2 = b * (1.0 - omega / np.maximum(r2df, 1e-12)) * er2df
    A = term1 + term2 - psh
    B = omega * eta
    return A, B


def _coefficients_murnaghan(eos, mu: np.ndarray):
    """A(mu), B(mu) for Murnaghan EOS (common_source/eos/murnaghan.F)."""
    p = eos.params
    k0 = p.get("k0", 0.0)
    k1 = p.get("k1", 1.0)
    p0 = p.get("p0", 0.0)
    psh = p.get("psh", 0.0)

    eta = np.maximum(1.0 + mu, 1e-12)
    k1_safe = k1 if abs(k1) > 1e-12 else 1.0
    A = (k0 / k1_safe) * (np.power(eta, k1) - 1.0) + p0 - psh
    B = np.zeros_like(A)
    return A, B


def _coefficients_noble_abel(eos, mu: np.ndarray):
    """A(mu), B(mu) for Noble-Abel EOS (common_source/eos/noble_abel.F)."""
    p = eos.params
    b = p.get("b", 0.0)
    gamma = p.get("gamma", 1.4)
    psh = p.get("psh", 0.0)
    rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

    eta = 1.0 + mu
    denom = 1.0 - b * rho0 * eta
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    A = np.full_like(mu, -psh, dtype=float)
    B = (gamma - 1.0) * eta / denom_safe
    return A, B


def _coefficients_nasg(eos, mu: np.ndarray):
    """A(mu), B(mu) for NASG EOS (common_source/eos/nasg.F)."""
    p = eos.params
    b = p.get("b", 0.0)
    gamma = p.get("gamma", 1.4)
    p_star = p.get("p_star", 0.0)
    q = p.get("q", 0.0)
    psh = p.get("psh", 0.0)
    rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

    eta = 1.0 + mu
    denom = 1.0 - b * rho0 * eta
    denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

    B = (gamma - 1.0) * eta / denom_safe
    A = -B * (rho0 * q) - gamma * p_star - psh
    return A, B


def _coefficients_puff(eos, mu: np.ndarray, e: np.ndarray | float | None = None):
    """A(mu), B(mu, e) for PUFF EOS (common_source/eos/puff.F)."""
    p = eos.params
    c1 = p.get("c1", 0.0)
    c2 = p.get("c2", 0.0)
    c3 = p.get("c3", 0.0)
    t1 = p.get("t1", 0.0)
    t2 = p.get("t2", 0.0)
    esubl = p.get("es", p.get("esubl", 0.0))
    gamma0 = p.get("gamma0", p.get("g0", 0.0))
    h = p.get("h", p.get("hh", 0.0))
    psh = p.get("psh", 0.0)

    if e is None:
        e = p.get("e0", 0.0)
    e_arr = np.asarray(e, dtype=float)

    eta = 1.0 + mu
    xx = np.where(np.abs(eta) > 1e-12, mu / eta, 0.0)
    gx = 1.0 - 0.5 * gamma0 * xx

    aa_comp = ((c1 + c3 * (mu ** 2)) * mu + c2 * (mu ** 2)) * gx
    aa_cold = ((t1 + t2 * mu) * mu) * gx

    ee = np.sqrt(np.maximum(eta, 1e-12))
    bb_hot = (h + (gamma0 - h) * ee) * eta
    cc = np.where(np.abs(gamma0 * esubl) > 1e-12, c1 / (gamma0 * esubl), 0.0)
    expa = np.exp(cc * xx)
    aa_hot = bb_hot * esubl * (expa - 1.0)

    is_comp = (mu >= 0.0)
    is_cold = (mu < 0.0) & (e_arr < esubl)

    A_unscaled = np.where(is_comp, aa_comp, np.where(is_cold, aa_cold, aa_hot))
    A = A_unscaled - psh
    B = np.where(is_comp | is_cold, gamma0, bb_hot)
    return A, B


def coefficients(eos, mu: np.ndarray, e: np.ndarray | float | None = None):
    """A(mu), B(mu) of p = A + B E (see module docstring)."""
    if eos.kind == "GRUNEISEN":
        return _coefficients_gruneisen(eos, mu)
    if eos.kind == "TILLOTSON":
        return _coefficients_tillotson(eos, mu, e)
    if eos.kind == "JWL":
        return _coefficients_jwl(eos, mu)
    if eos.kind == "MURNAGHAN":
        return _coefficients_murnaghan(eos, mu)
    if eos.kind == "NOBLE-ABEL":
        return _coefficients_noble_abel(eos, mu)
    if eos.kind == "NASG":
        return _coefficients_nasg(eos, mu)
    if eos.kind == "PUFF":
        return _coefficients_puff(eos, mu, e)
    if eos.kind == "STIFF-GAS":
        p = eos.params
        gamma = p["gamma"]
        p_star = p["p_star"]
        psh = p.get("psh", 0.0)
        A = -gamma * p_star - psh
        B = (gamma - 1.0) * (1.0 + mu)
        return A, B

    p = eos.params
    mubar = np.maximum(mu, 0.0)
    A = p["c0"] + p["c1"] * mu + p["c2"] * mubar ** 2 + p["c3"] * mu ** 3
    B = p["c4"] + p["c5"] * mu
    return A, B


def update(eos, mu: np.ndarray, dv: np.ndarray, e_old: np.ndarray,
           p_old: np.ndarray, de_other: np.ndarray):
    """One implicit E-p update (eosmain). All arrays per element slice.

    mu       : compression 1/J - 1 at the END of the increment
    dv       : relative-volume change J_new - J_old (per V0)
    e_old    : energy per V0 entering the cycle
    p_old    : pressure entering the cycle
    de_other : non-pdV energy added this cycle per V0 (deviator work,
               deferred viscous heating)

    Returns (p_new, e_new, c2_bulk) with c2_bulk the EOS part of the
    squared sound speed (>= 0; the kernel adds the shear stiffness).
    """
    p = eos.params
    if eos.kind == "STIFF-GAS":
        gamma = p["gamma"]
        p_star = p["p_star"]
        psh = p["psh"]
        A = -gamma * p_star - psh
        B = (gamma - 1.0) * (1.0 + mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A + 2.0 * psh)) \
            / np.maximum(denom, 1e-6)
        p_new = A + B * e_new
        p_new = np.maximum(p_new, -psh)
        dpdmu = (gamma - 1.0) * e_new
        c2 = (dpdmu + B * (p_new + psh) / (1.0 + mu) ** 2) / eos.rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if eos.kind == "GRUNEISEN":
        c = p.get("c", 0.0)
        s1 = p.get("s1", 0.0)
        s2 = p.get("s2", 0.0)
        s3 = p.get("s3", 0.0)
        gamma0 = p.get("gamma0", 0.0)
        a = p.get("a", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

        A, B = _coefficients_gruneisen(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A)) / np.maximum(denom, 1e-6)
        p_new = A + B * e_new
        p_tot = np.maximum(p_new, pmin)
        p_new = p_tot - psh

        mu_pos = np.maximum(mu, 0.0)
        eta = 1.0 + mu
        xx = np.where(mu > 0.0, mu / np.maximum(eta, 1e-12), 0.0)
        ff = 1.0 + (1.0 - 0.5 * gamma0) * mu - 0.5 * a * (mu_pos ** 2)
        fg = 1.0 - (s1 - 1.0 + s2 * xx + s3 * (xx ** 2)) * mu
        fg_safe = np.where(np.abs(fg) < 1e-12, 1e-12, fg)
        ff_safe = np.where(np.abs(ff) < 1e-12, 1e-12, ff)
        fac = np.where(mu > 0.0, ff / (fg_safe ** 2), 1.0)
        dff = 1.0 - 0.5 * gamma0 - a * mu
        dfg = 1.0 - s1 + xx * (-2.0 * s2 + xx * (s2 - 3.0 * s3) + 2.0 * s3 * (xx ** 2))
        fac1 = np.where(mu > 0.0, fac * (1.0 + mu * (dff / ff_safe - 2.0 * dfg / fg_safe)), 1.0)

        dpdmu = fac1 * rho0 * (c ** 2) + a * e_new
        dpdm = dpdmu + B * p_tot / (np.maximum(eta, 1e-12) ** 2)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if eos.kind == "TILLOTSON":
        c1 = p.get("c1", 0.0)
        c2 = p.get("c2", 0.0)
        a = p.get("a", 0.0)
        b = p.get("b", 0.0)
        er = p.get("er", p.get("ezero", p.get("e0_ref", 1.0)))
        es = p.get("es", p.get("esubl", 0.0))
        vs = p.get("vs", p.get("vsubl", 1.0))
        alpha = p.get("alpha", 0.0)
        beta = p.get("beta", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

        eta = 1.0 + mu
        df = 1.0 / np.maximum(eta, 1e-12)
        xx = np.where(np.abs(eta) > 1e-12, mu / eta, 0.0)
        expa = np.exp(-alpha * (xx ** 2))
        expb = np.exp(beta * xx)

        # Predictor step using e_old
        hot0 = (mu < 0.0) & ((df > vs) | ((df <= vs) & (e_old >= es)))
        facc1_0 = np.where(hot0, expa * expb, 1.0)
        facc2_0 = np.where(mu >= 0.0, 1.0, 0.0)
        facpb_0 = np.where(hot0, expa, 1.0)
        A0 = facc1_0 * c1 * mu + facc2_0 * c2 * (mu ** 2)
        omega0 = 1.0 + e_old / np.maximum(er * (eta ** 2), 1e-15)
        B0 = (a + facpb_0 * b / omega0) * eta

        denom0 = 1.0 + 0.5 * B0 * dv
        e_pred = (e_old + de_other - 0.5 * dv * (p_old + A0)) / np.maximum(denom0, 1e-6)

        # Corrector step using e_pred
        hot1 = (mu < 0.0) & ((df > vs) | ((df <= vs) & (e_pred >= es)))
        facc1_1 = np.where(hot1, expa * expb, 1.0)
        facc2_1 = np.where(mu >= 0.0, 1.0, 0.0)
        facpb_1 = np.where(hot1, expa, 1.0)
        A1 = facc1_1 * c1 * mu + facc2_1 * c2 * (mu ** 2)
        omega1 = 1.0 + e_pred / np.maximum(er * (eta ** 2), 1e-15)
        B1 = (a + facpb_1 * b / omega1) * eta

        denom1 = 1.0 + 0.5 * B1 * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A1)) / np.maximum(denom1, 1e-6)
        p_new = A1 + B1 * e_new
        p_tot = np.maximum(p_new, pmin)
        p_new = p_tot - psh

        b_unscaled = a + facpb_1 * b / omega1
        dpdm = (facc1_1 * c1 + 2.0 * facc2_1 * c2 * mu
                + B1 * p_tot / (np.maximum(eta, 1e-12) ** 2)
                + e_new * (b_unscaled + (2.0 * e_new / np.maximum(eta, 1e-12) - p_tot / (np.maximum(eta, 1e-12) ** 2))
                           * b * facpb_1 / (er * np.maximum(eta, 1e-12) * (omega1 ** 2))))
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if eos.kind == "JWL":
        p = eos.params
        a = p.get("a", 0.0)
        b = p.get("b", 0.0)
        r1 = p.get("r1", 0.0)
        r2 = p.get("r2", 0.0)
        omega = p.get("omega", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -psh)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

        A, B = _coefficients_jwl(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A)) / np.maximum(denom, 1e-6)
        p_raw = A + B * e_new
        p_new = np.maximum(p_raw, pmin)

        eta = np.maximum(1.0 + mu, 1e-12)
        df = 1.0 / eta
        r1df = r1 * df
        r2df = r2 * df
        er1df = np.exp(-r1df)
        er2df = np.exp(-r2df)

        dpde = omega * eta
        dpdmu = (-a * omega * er1df / np.maximum(r1, 1e-12)
                 + a * (1.0 - omega / np.maximum(r1df, 1e-12)) * (r1df ** 2) * er1df
                 - b * omega * er2df / np.maximum(r2, 1e-12)
                 + b * (1.0 - omega / np.maximum(r2df, 1e-12)) * (r2df ** 2) * er2df
                 + omega * e_new)
        dpdm = dpdmu + (p_new + psh) * (df ** 2) * dpde
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if eos.kind == "MURNAGHAN":
        p = eos.params
        k0 = p.get("k0", 0.0)
        k1 = p.get("k1", 1.0)
        p0 = p.get("p0", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

        A, B = _coefficients_murnaghan(eos, mu)
        p_tot = np.maximum(A + psh, pmin)
        p_new = p_tot - psh
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)

        eta = np.maximum(1.0 + mu, 1e-12)
        dpdm = k0 * np.power(eta, k1 - 1.0)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if eos.kind == "NOBLE-ABEL":
        p = eos.params
        b = p.get("b", 0.0)
        gamma = p.get("gamma", 1.4)
        psh = p.get("psh", 0.0)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

        A, B = _coefficients_noble_abel(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A)) / np.maximum(denom, 1e-6)
        e_new = np.maximum(e_new, 0.0)
        p_new = A + B * e_new

        eta = 1.0 + mu
        df = 1.0 / np.maximum(eta, 1e-12)
        denom_cov = 1.0 - b * rho0 * eta
        denom_safe = np.where(np.abs(denom_cov) < 1e-12, 1e-12, denom_cov)
        pp = p_new + psh

        dpde = (gamma - 1.0) * eta / denom_safe
        dpdm = (gamma - 1.0) * e_new / denom_safe + (pp / denom_safe) * (b * rho0) + pp * (df ** 2) * dpde
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if eos.kind == "NASG":
        p = eos.params
        b = p.get("b", 0.0)
        gamma = p.get("gamma", 1.4)
        p_star = p.get("p_star", 0.0)
        q = p.get("q", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

        A, B = _coefficients_nasg(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A + 2.0 * psh)) / np.maximum(denom, 1e-6)
        p_raw = A + B * e_new
        p_lim = np.maximum(p_raw + psh, np.maximum(pmin, -gamma * p_star))
        p_new = p_lim - psh

        eta = 1.0 + mu
        denom_cov = 1.0 - b * rho0 * eta
        denom_safe = np.where(np.abs(denom_cov) < 1e-12, 1e-12, denom_cov)
        num = e_new - rho0 * q

        dpde = (gamma - 1.0) * eta / denom_safe
        dpdm = (gamma - 1.0) * num / (denom_safe ** 2) + dpde * (p_new + psh) / (eta ** 2)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if eos.kind == "PUFF":
        p = eos.params
        c1 = p.get("c1", 0.0)
        c2 = p.get("c2", 0.0)
        c3 = p.get("c3", 0.0)
        t1 = p.get("t1", 0.0)
        t2 = p.get("t2", 0.0)
        esubl = p.get("es", p.get("esubl", 0.0))
        gamma0 = p.get("gamma0", p.get("g0", 0.0))
        h = p.get("h", p.get("hh", 0.0))
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

        # Predictor step with e_old
        A0, B0 = _coefficients_puff(eos, mu, e_old)
        denom0 = 1.0 + 0.5 * B0 * dv
        e_pred = (e_old + de_other - 0.5 * dv * (p_old + A0)) / np.maximum(denom0, 1e-6)

        # Corrector step with e_pred
        A1, B1 = _coefficients_puff(eos, mu, e_pred)
        denom1 = 1.0 + 0.5 * B1 * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A1)) / np.maximum(denom1, 1e-6)
        p_raw = A1 + B1 * e_new
        p_tot = np.maximum(p_raw + psh, pmin)
        p_new = p_tot - psh

        eta = 1.0 + mu
        df = 1.0 / np.maximum(eta, 1e-12)
        xx = np.where(np.abs(eta) > 1e-12, mu / eta, 0.0)
        gx = 1.0 - 0.5 * gamma0 * xx
        is_comp = (mu >= 0.0)
        is_cold = (mu < 0.0) & (e_new < esubl)

        aa_raw_comp = (c1 + c3 * (mu ** 2)) * mu + c2 * (mu ** 2)
        aa_raw_cold = (t1 + t2 * mu) * mu

        ee = np.sqrt(np.maximum(eta, 1e-12))
        bb_hot = (h + (gamma0 - h) * ee) * eta
        cc = np.where(np.abs(gamma0 * esubl) > 1e-12, c1 / (gamma0 * esubl), 0.0)
        expa = np.exp(cc * xx)

        dpdm_comp = (c1 + 2.0 * c2 * mu + 3.0 * c3 * (mu ** 2)) * gx + gamma0 * (df ** 2) * (p_tot - 0.5 * aa_raw_comp)
        dpdm_cold = (t1 + 2.0 * t2 * mu) * gx + gamma0 * (df ** 2) * (p_tot - 0.5 * aa_raw_cold)
        dpdm_hot = bb_hot * (df ** 2) * (p_tot + esubl * expa * cc) + (e_new + esubl * (expa - 1.0)) * (h + 1.5 * ee * (gamma0 - h))

        dpdm = np.where(is_comp, dpdm_comp, np.where(is_cold, dpdm_cold, dpdm_hot))
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    A, B = coefficients(eos, mu)
    denom = 1.0 + 0.5 * B * dv
    # denom <= 0 would need a catastrophic single-cycle expansion of a
    # stiff gas — guard like the original (floors the denominator)
    e_new = (e_old + de_other - 0.5 * dv * (p_old + A)) \
        / np.maximum(denom, 1e-6)
    if eos.kind == "IDEAL-GAS":
        # a gas carries no tension and no negative energy
        e_new = np.maximum(e_new, 0.0)
    p_new = A + B * e_new

    mubar = np.maximum(mu, 0.0)
    dpdmu = p["c1"] + 2.0 * p["c2"] * mubar + 3.0 * p["c3"] * mu ** 2 \
        + p["c5"] * e_new
    c2 = (dpdmu + B * p_new / (1.0 + mu) ** 2) / eos.rho0
    return p_new, e_new, np.maximum(c2, 0.0)


def pressure(eos, mu: np.ndarray | float, e: np.ndarray | float) -> np.ndarray | float:
    """Evaluate EOS pressure p(mu, E) at given compression mu and internal energy E."""
    is_scalar = np.isscalar(mu) and np.isscalar(e)
    mu_arr = np.asarray(mu, dtype=float)
    e_arr = np.asarray(e, dtype=float)

    if eos.kind == "STIFF-GAS":
        p = eos.params
        gamma = p["gamma"]
        p_star = p["p_star"]
        psh = p.get("psh", 0.0)
        A = -gamma * p_star - psh
        B = (gamma - 1.0) * (1.0 + mu_arr)
        p_val = np.maximum(A + B * e_arr, -psh)
        return float(p_val) if is_scalar else p_val

    if eos.kind == "GRUNEISEN":
        A, B = _coefficients_gruneisen(eos, mu_arr)
        p_val = A + B * e_arr
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(p_val, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if eos.kind == "TILLOTSON":
        A, B = _coefficients_tillotson(eos, mu_arr, e_arr)
        p_val = A + B * e_arr
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(p_val, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if eos.kind == "JWL":
        A, B = _coefficients_jwl(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        p_val = np.maximum(A + B * e_arr, -psh)
        return float(p_val) if is_scalar else p_val

    if eos.kind == "MURNAGHAN":
        A, B = _coefficients_murnaghan(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A + psh, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if eos.kind == "NOBLE-ABEL":
        A, B = _coefficients_noble_abel(eos, mu_arr)
        p_val = A + B * e_arr
        return float(p_val) if is_scalar else p_val

    if eos.kind == "NASG":
        A, B = _coefficients_nasg(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        gamma = eos.params.get("gamma", 1.4)
        p_star = eos.params.get("p_star", 0.0)
        p_val = np.maximum(A + B * e_arr + psh, np.maximum(pmin, -gamma * p_star)) - psh
        return float(p_val) if is_scalar else p_val

    if eos.kind == "PUFF":
        A, B = _coefficients_puff(eos, mu_arr, e_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A + B * e_arr + psh, pmin) - psh
        return float(p_val) if is_scalar else p_val

    A, B = coefficients(eos, mu_arr)
    p_val = A + B * e_arr
    if eos.kind == "IDEAL-GAS":
        p_val = np.maximum(p_val, 0.0)
    return float(p_val) if is_scalar else p_val


def sound_speed(eos, mu: np.ndarray | float, e: np.ndarray | float) -> np.ndarray | float:
    """Evaluate bulk sound speed c = sqrt(max(c2_bulk, 0)) at given mu and energy E."""
    is_scalar = np.isscalar(mu) and np.isscalar(e)
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
    e_arr = np.atleast_1d(np.asarray(e, dtype=float))
    dv = np.zeros_like(mu_arr)
    p_arr = np.atleast_1d(np.asarray(pressure(eos, mu_arr, e_arr), dtype=float))
    de = np.zeros_like(mu_arr)
    _, _, c2 = update(eos, mu_arr, dv, e_arr, p_arr, de)
    c = np.sqrt(np.maximum(c2, 0.0))
    return float(c[0]) if is_scalar else c


def initial_state(eos):
    """(e0, p0) at mu = 0 — p0 evaluated from the coefficients so the
    stored pair is always consistent with the polynomial."""
    if eos.kind == "STIFF-GAS":
        p = eos.params
        gamma = p["gamma"]
        e0 = (p["p0"] + gamma * p["p_star"]) / (gamma - 1.0)
        p0 = p["p0"] - p.get("psh", 0.0)
        return e0, p0

    if eos.kind == "GRUNEISEN":
        p = eos.params
        e0 = p.get("e0", 0.0)
        gamma0 = p.get("gamma0", 0.0)
        p0_param = p.get("p0", 0.0)
        if p0_param > 0.0 and e0 == 0.0 and gamma0 > 0.0:
            e0 = p0_param / gamma0
        psh = p.get("psh", 0.0)
        p0 = gamma0 * e0 - psh
        return e0, p0

    if eos.kind == "TILLOTSON":
        p = eos.params
        e0 = p.get("e0", 0.0)
        a = p.get("a", 0.0)
        b = p.get("b", 0.0)
        er = p.get("er", p.get("ezero", p.get("e0_ref", 1.0)))
        omega = 1.0 + e0 / er if er > 0.0 else 1.0
        psh = p.get("psh", 0.0)
        p0 = (a + b / omega) * e0 - psh
        return e0, p0

    if eos.kind == "JWL":
        p = eos.params
        e0 = p.get("e0", 0.0)
        p0 = pressure(eos, 0.0, e0)
        return e0, p0

    if eos.kind == "MURNAGHAN":
        p = eos.params
        e0 = 0.0
        p0 = p.get("p0", 0.0) - p.get("psh", 0.0)
        return e0, p0

    if eos.kind == "NOBLE-ABEL":
        p = eos.params
        e0 = p.get("e0", 0.0)
        p0 = pressure(eos, 0.0, e0)
        return e0, p0

    if eos.kind == "NASG":
        p = eos.params
        b = p.get("b", 0.0)
        gamma = p.get("gamma", 1.4)
        p_star = p.get("p_star", 0.0)
        q = p.get("q", 0.0)
        p0_param = p.get("p0", 0.0)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)
        e0 = p.get("e0", None)
        if e0 is None or e0 == 0.0:
            e0 = (p0_param + gamma * p_star) * (1.0 - rho0 * b) / (gamma - 1.0) + rho0 * q
        p0 = p0_param - p.get("psh", 0.0)
        return e0, p0

    if eos.kind == "PUFF":
        p = eos.params
        e0 = p.get("e0", 0.0)
        p0 = pressure(eos, 0.0, e0)
        return e0, p0

    e0 = eos.params.get("e0", 0.0)
    p0 = eos.params.get("c0", 0.0) + eos.params.get("c4", 0.0) * e0
    return e0, p0
