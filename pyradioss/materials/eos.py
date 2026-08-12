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


def coefficients(eos, mu: np.ndarray):
    """A(mu), B(mu) of p = A + B E (see module docstring)."""
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


def initial_state(eos):
    """(e0, p0) at mu = 0 — p0 evaluated from the coefficients so the
    stored pair is always consistent with the polynomial."""
    if eos.kind == "STIFF-GAS":
        p = eos.params
        gamma = p["gamma"]
        e0 = (p["p0"] + gamma * p["p_star"]) / (gamma - 1.0)
        p0 = p["p0"] - p["psh"]
        return e0, p0
        
    e0 = eos.params.get("e0", 0.0)
    p0 = eos.params["c0"] + eos.params["c4"] * e0
    return e0, p0
