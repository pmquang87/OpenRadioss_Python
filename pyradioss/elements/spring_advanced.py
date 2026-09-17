"""
Advanced spring element formulations:
- /PROP/TYPE19 (SPR_TORS): 2-node Torsion Spring (Fortran r1tors.F)
- /PROP/TYPE44 (SPR_CRUS): Crushing Spring with energy absorption (Fortran ruser44.F)
- /PROP/TYPE46 (SPR_MUSCLE): Active Muscle Spring with Hill-type dynamics (Fortran ruser46.F)

Fortran origin
--------------
* engine : ``engine/source/elements/spring/rforc3.F``
  - TYPE19: Torsion spring relative angle theta about line of nodes, rotational stiffness K_theta,
    damping C_theta, applied opposite moments M = K_theta * theta + C_theta * dtheta/dt.
  - TYPE44: ``engine/source/elements/spring/ruser44.F`` (crushable frame spring with non-recoverable
    plastic crushing displacement, compression plateau, and elastic unloading).
  - TYPE46: ``engine/source/elements/spring/ruser46.F`` (Hill-type muscle spring with activation level
    alpha(t), maximum isometric force F_max, force-length relationship f_L(L/L_0), and force-velocity
    relationship f_v(v/v_max)).
* starter:
  - ``starter/source/properties/spring/hm_read_prop19.F``
  - ``starter/source/properties/spring/hm_read_prop44.F``
  - ``starter/source/properties/spring/hm_read_prop46.F``
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple
import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3

#: Advanced spring property type numbers
ADVANCED_SPRING_PROP_TYPES = frozenset({19, 44, 46})


def _safe_param(params: dict, key: str, default: float = 0.0) -> float:
    """Safely extract float parameter from prop.params with fallback."""
    val = params.get(key)
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _safe_int_param(params: dict, key: str, default: int = 0) -> int:
    """Safely extract int parameter from prop.params with fallback."""
    val = params.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# ============================================================================
# TYPE19: Torsion Spring
# ============================================================================

def init_torsion_type19(group, model, log, idx19, massn, inertn):
    """Initialize state arrays for TYPE19 torsion springs.
    
    TYPE19 is a 2-node torsional spring acting about the relative line of nodes.
    Parameters:
      - k_theta / k: torsional stiffness [torque / rad]
      - c_theta / c: torsional damping [torque * s / rad]
      - inertia: lumped rotational inertia
      - mass: lumped translational mass
    """
    st = group.state
    n = group.n
    m19 = len(idx19)
    if m19 == 0:
        return

    if "t19_k_theta" not in st:
        st["t19_k_theta"] = np.zeros(n)
        st["t19_c_theta"] = np.zeros(n)
        st["t19_inertia"] = np.zeros(n)
        st["t19_mass"] = np.zeros(n)
        st["t19_theta"] = np.zeros(n)
        st["t19_moment"] = np.zeros(n)
        st["t19_moment_old"] = np.zeros(n)

    pos = {int(e): j for j, e in enumerate(idx19)}
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 0) != 19:
            continue
        rng = np.arange(group.n)[sl]
        local = [e for e in rng if int(e) in pos]
        if not len(local):
            continue
        p = getattr(prop, "params", {}) or {}
        kt = _safe_param(p, "k_theta", _safe_param(p, "k", 0.0))
        ct = _safe_param(p, "c_theta", _safe_param(p, "c", 0.0))
        iner = _safe_param(p, "inertia", 0.0)
        ms = _safe_param(p, "mass", 0.0)

        st["t19_k_theta"][local] = kt
        st["t19_c_theta"][local] = ct
        st["t19_inertia"][local] = iner
        st["t19_mass"][local] = ms
        st["k"][local] = kt
        st["cdamp"][local] = ct
        st["mass"][local] = ms

        if massn is not None and ms > 0.0:
            for e in local:
                if 2 * e + 1 < len(massn):
                    massn[2 * e] += ms / 2.0
                    massn[2 * e + 1] += ms / 2.0

        if inertn is not None and iner > 0.0:
            for e in local:
                if 2 * e + 1 < len(inertn):
                    inertn[2 * e] += iner / 2.0
                    inertn[2 * e + 1] += iner / 2.0


def forces_torsion_type19(group, x, v, vr, dt, fint, mint, idx19):
    """Compute forces and moments for TYPE19 torsion springs.

    Moment M = K_theta * theta + C_theta * dtheta/dt
    Applied to node 1 as +M * a, and to node 2 as -M * a.
    Work is integrated into eint.
    """
    if idx19 is None or len(idx19) == 0:
        return np.empty(0)
    st = group.state
    conn = group.conn[idx19]
    n1, n2 = conn[:, 0], conn[:, 1]

    dx = x[n2] - x[n1]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    if vr is None:
        theta_dot = np.zeros(len(conn))
    else:
        d_omega = vr[n2] - vr[n1]
        theta_dot = np.einsum("nb,nb->n", d_omega, a)

    dt_val = max(float(dt), 0.0) if dt is not None else 0.0
    st["t19_theta"][idx19] += theta_dot * dt_val

    k_theta = st["t19_k_theta"][idx19]
    c_theta = st["t19_c_theta"][idx19]
    theta = st["t19_theta"][idx19]

    M_old = st["t19_moment"][idx19].copy()
    M = k_theta * theta + c_theta * theta_dot
    st["t19_moment_old"][idx19] = M_old
    st["t19_moment"][idx19] = M

    # Opposite moments along line of nodes: M1 = +M*a, M2 = -M*a
    mvec = M[:, None] * a
    if mint is not None:
        np.add.at(mint, n1, mvec)
        np.add.at(mint, n2, -mvec)

    # Internal energy accounting (work done by torsional moment)
    if dt_val > 0.0:
        st["eint"][idx19] += 0.5 * (M_old + M) * theta_dot * dt_val

    # Critical time step for rotational oscillator
    inertia = np.maximum(st["t19_inertia"][idx19], 0.0)
    kt = np.maximum(k_theta, 0.0)
    ct = np.maximum(c_theta, 0.0)

    pos = (kt > 0.0) & (inertia > 0.0)
    pure_c = (kt <= 0.0) & (ct > 0.0) & (inertia > 0.0)

    # dt_rot = inertia / (sqrt(C^2 + inertia * K) + C)
    dt_crit = np.full(len(idx19), EP30)
    if np.any(pos):
        in_p = inertia[pos]
        k_p = kt[pos]
        c_p = ct[pos]
        denom = np.sqrt(c_p * c_p + in_p * k_p) + c_p
        dt_crit[pos] = in_p / np.maximum(denom, EM20)

    if np.any(pure_c):
        in_c = inertia[pure_c]
        c_c = ct[pure_c]
        dt_crit[pure_c] = 0.5 * in_c / np.maximum(c_c, EM20)

    return dt_crit


# ============================================================================
# TYPE44: Crushing Spring with Energy Absorption
# ============================================================================

def init_crushing_type44(group, model, log, idx44, massn, inertn):
    """Initialize state arrays for TYPE44 crushing springs.

    TYPE44 features plastic crushing in compression:
      - Yield force F_yield
      - Crushing plastic displacement delta_crush
      - Elastic unloading stiffness K_unload
      - Permanent non-recoverable deformation
      - Plastic energy dissipation booking into E_int
    """
    st = group.state
    n = group.n
    m44 = len(idx44)
    if m44 == 0:
        return

    if "t44_k_unload" not in st:
        st["t44_k_unload"] = np.zeros(n)
        st["t44_f_yield"] = np.zeros(n)
        st["t44_f_yield_tensile"] = np.zeros(n)
        st["t44_delta_crush_max"] = np.full(n, 1e20)
        st["t44_delta_p"] = np.zeros(n)
        st["t44_e_plastic"] = np.zeros(n)
        st["t44_cdamp"] = np.zeros(n)
        st["t44_mass"] = np.zeros(n)
        st["t44_fct_yield"] = np.zeros(n, dtype=np.int64)

    pos = {int(e): j for j, e in enumerate(idx44)}
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 0) != 44:
            continue
        rng = np.arange(group.n)[sl]
        local = [e for e in rng if int(e) in pos]
        if not len(local):
            continue
        p = getattr(prop, "params", {}) or {}

        # Stiffness: k_unload, k11, k, or stiff1
        k_un = _safe_param(p, "k_unload", _safe_param(p, "k11", _safe_param(p, "k", _safe_param(p, "stiff1", 0.0))))
        # Yield force: f_yield, f_max, or from cards
        fy = _safe_param(p, "f_yield", _safe_param(p, "f_max", 0.0))
        fy_t = _safe_param(p, "f_yield_tensile", 0.0)
        d_crush = _safe_param(p, "delta_crush_max", _safe_param(p, "delta_crush", 1e20))
        c = _safe_param(p, "c", _safe_param(p, "cdamp", _safe_param(p, "dscale_x", 0.0)))
        ms = _safe_param(p, "mass", 0.0)
        fct_y = _safe_int_param(p, "fun_b1", _safe_int_param(p, "fct_yield", 0))

        st["t44_k_unload"][local] = k_un
        st["t44_f_yield"][local] = fy
        st["t44_f_yield_tensile"][local] = fy_t
        st["t44_delta_crush_max"][local] = d_crush
        st["t44_cdamp"][local] = c
        st["t44_mass"][local] = ms
        st["t44_fct_yield"][local] = fct_y
        st["k"][local] = k_un
        st["cdamp"][local] = c
        st["mass"][local] = ms

        if massn is not None and ms > 0.0:
            for e in local:
                if 2 * e + 1 < len(massn):
                    massn[2 * e] += ms / 2.0
                    massn[2 * e + 1] += ms / 2.0


def forces_crushing_type44(group, x, v, dt, fint, idx44):
    """Compute forces for TYPE44 crushing springs.

    Trial force: F_trial = K_unload * (delta - delta_p)
    Compression yield: when F_trial < -F_yield (crushing plateau)
      F_elastic = -F_yield
      delta_p = delta + F_yield / K_unload
    Elastic unloading:
      when unloading, delta_p is fixed and F = K_unload * (delta - delta_p)
    Energy accounting:
      Incremental work = 0.5 * (F_old + F) * Ldot * dt booked into eint.
    """
    if idx44 is None or len(idx44) == 0:
        return np.empty(0)
    st = group.state
    conn = group.conn[idx44]
    n1, n2 = conn[:, 0], conn[:, 1]
    model = st.get("model")

    dx = x[n2] - x[n1]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    if v is None:
        Ldot = np.zeros(len(conn))
    else:
        Ldot = np.einsum("nb,nb->n", v[n2] - v[n1], a)

    dt_val = max(float(dt), 0.0) if dt is not None else 0.0

    L0 = st["L0"][idx44]
    delta = L - L0  # delta < 0 is compression, delta > 0 is tension

    k_unload = st["t44_k_unload"][idx44]
    f_yield = st["t44_f_yield"][idx44].copy()
    f_yield_t = st["t44_f_yield_tensile"][idx44]
    delta_p = st["t44_delta_p"][idx44]
    c_damp = st["t44_cdamp"][idx44]
    fct_y = st["t44_fct_yield"][idx44]

    # Evaluate yield force from functions if specified
    if model is not None and hasattr(model, "functions"):
        for i, fid in enumerate(fct_y):
            if fid > 0 and fid in model.functions:
                func = model.functions[fid]
                f_yield[i] = abs(func.eval(abs(delta[i])))

    # Elastic trial force: F_trial = K_unload * (delta - delta_p)
    F_trial = k_unload * (delta - delta_p)

    F_elastic = F_trial.copy()
    delta_p_new = delta_p.copy()

    # Compressive crushing yield: F_trial < -f_yield
    has_crush_yield = f_yield > 0.0
    crush_mask = has_crush_yield & (F_trial < -f_yield)
    if np.any(crush_mask):
        k_m = np.maximum(k_unload[crush_mask], EM20)
        fy_m = f_yield[crush_mask]
        d_m = delta[crush_mask]
        F_elastic[crush_mask] = -fy_m
        delta_p_new[crush_mask] = d_m + fy_m / k_m

    # Tensile yield: F_trial > f_yield_t (if f_yield_t > 0)
    has_tens_yield = f_yield_t > 0.0
    tens_mask = has_tens_yield & (F_trial > f_yield_t)
    if np.any(tens_mask):
        k_m = np.maximum(k_unload[tens_mask], EM20)
        fyt_m = f_yield_t[tens_mask]
        d_m = delta[tens_mask]
        F_elastic[tens_mask] = fyt_m
        delta_p_new[tens_mask] = d_m - fyt_m / k_m

    # Plastic energy dissipation update
    d_delta_p = np.abs(delta_p_new - delta_p)
    st["t44_e_plastic"][idx44] += f_yield * d_delta_p
    st["t44_delta_p"][idx44] = delta_p_new

    # Viscous damping force
    F_damp = c_damp * Ldot
    F = F_elastic + F_damp

    F_old = st["force"][idx44].copy()
    st["force"][idx44] = F

    # Apply forces to nodes: tension pulls together, compression pushes apart
    fvec = F[:, None] * a
    if fint is not None:
        np.add.at(fint, n1, fvec)
        np.add.at(fint, n2, -fvec)

    # Internal energy work booking
    if dt_val > 0.0:
        st["eint"][idx44] += 0.5 * (F_old + F) * Ldot * dt_val

    # Time step calculation
    mass = np.maximum(st["t44_mass"][idx44], EM20)
    k_dt = np.maximum(k_unload, 0.0)
    pos_k = (k_dt > 0.0) & (st["t44_mass"][idx44] > 0.0)
    pure_c = (k_dt <= 0.0) & (c_damp > 0.0) & (st["t44_mass"][idx44] > 0.0)

    omega = 2.0 * np.sqrt(np.where(pos_k, k_dt / mass, 1.0))
    xi = np.where(pos_k, c_damp / np.sqrt(np.maximum(k_dt * mass, EM20)), 0.0)
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    dt_c = np.where(pure_c, 0.5 * mass / np.maximum(c_damp, EM20), EP30)
    return np.where(pos_k, dt_crit, dt_c)


# ============================================================================
# TYPE46: Active Muscle Spring
# ============================================================================

def init_muscle_type46(group, model, log, idx46, massn, inertn):
    """Initialize state arrays for TYPE46 active muscle springs.

    Hill-type muscle model with:
      - Activation alpha(t) in [0, 1]
      - Maximum isometric force F_max
      - Maximum velocity v_max
      - Force-length relationship f_L(L / L0)
      - Force-velocity relationship f_v(v / v_max)
      - Passive elastic resistance K_pe
    """
    st = group.state
    n = group.n
    m46 = len(idx46)
    if m46 == 0:
        return

    if "t46_f_max" not in st:
        st["t46_f_max"] = np.zeros(n)
        st["t46_v_max"] = np.full(n, 10.0)
        st["t46_k_pe"] = np.zeros(n)
        st["t46_l_opt"] = np.zeros(n)
        st["t46_activation"] = np.ones(n)
        st["t46_fct_act"] = np.zeros(n, dtype=np.int64)
        st["t46_fct_fl"] = np.zeros(n, dtype=np.int64)
        st["t46_fct_fv"] = np.zeros(n, dtype=np.int64)
        st["t46_fct_pass"] = np.zeros(n, dtype=np.int64)
        st["t46_cdamp"] = np.zeros(n)
        st["t46_mass"] = np.zeros(n)

    pos = {int(e): j for j, e in enumerate(idx46)}
    xe = model.x0[group.conn[idx46]]
    l0_calc = norm3(xe[:, 1] - xe[:, 0])

    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 0) != 46:
            continue
        rng = np.arange(group.n)[sl]
        local = [e for e in rng if int(e) in pos]
        if not len(local):
            continue
        p = getattr(prop, "params", {}) or {}

        # Active force parameters
        f_max = _safe_param(p, "f_max", _safe_param(p, "nforce", _safe_param(p, "xlim2", 0.0)))
        v_max = _safe_param(p, "v_max", _safe_param(p, "vel_max", _safe_param(p, "vel_x", _safe_param(p, "xlim1", 10.0))))
        k_pe = _safe_param(p, "k_pe", _safe_param(p, "stiff0", _safe_param(p, "stiff1", 0.0)))
        l_opt = _safe_param(p, "l_opt", 0.0)
        act_const = _safe_param(p, "activation", _safe_param(p, "act_init", 1.0))

        # Function IDs
        fct_act = _safe_int_param(p, "fun_a1", _safe_int_param(p, "ifunc_act", 0))
        fct_fl = _safe_int_param(p, "fun_b1", _safe_int_param(p, "ifunc_ce", 0))
        fct_fv = _safe_int_param(p, "fun_c1", 0)
        fct_pass = _safe_int_param(p, "fun_d1", _safe_int_param(p, "ifunc_pe", 0))

        c = _safe_param(p, "damp1", _safe_param(p, "c", _safe_param(p, "cdamp", 0.0)))
        ms = _safe_param(p, "mass", _safe_param(p, "prop_mass", 0.0))

        st["t46_f_max"][local] = f_max
        st["t46_v_max"][local] = v_max if v_max > 0.0 else 10.0
        st["t46_k_pe"][local] = k_pe
        st["t46_l_opt"][local] = np.where(l_opt > 0.0, l_opt, st["L0"][local])
        st["t46_activation"][local] = act_const
        st["t46_fct_act"][local] = fct_act
        st["t46_fct_fl"][local] = fct_fl
        st["t46_fct_fv"][local] = fct_fv
        st["t46_fct_pass"][local] = fct_pass
        st["t46_cdamp"][local] = c
        st["t46_mass"][local] = ms
        st["k"][local] = k_pe if k_pe > 0.0 else (f_max / np.maximum(st["L0"][local], EM20))
        st["cdamp"][local] = c
        st["mass"][local] = ms

        if massn is not None and ms > 0.0:
            for e in local:
                if 2 * e + 1 < len(massn):
                    massn[2 * e] += ms / 2.0
                    massn[2 * e + 1] += ms / 2.0


def forces_muscle_type46(group, x, v, dt, fint, idx46):
    """Compute forces for TYPE46 active muscle springs.

    Hill muscle mechanics:
      F_active = F_max * alpha(t) * f_L(L / L0) * f_v(v / v_max)
      F_passive = K_pe * max(0, L - L0)
      F_damp = C * v
      Total tensile force F = max(0, F_active + F_passive + F_damp)
    """
    if idx46 is None or len(idx46) == 0:
        return np.empty(0)
    st = group.state
    conn = group.conn[idx46]
    n1, n2 = conn[:, 0], conn[:, 1]
    model = st.get("model")
    t = getattr(model, "t", 0.0) if model is not None else 0.0

    dx = x[n2] - x[n1]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    if v is None:
        Ldot = np.zeros(len(conn))
    else:
        Ldot = np.einsum("nb,nb->n", v[n2] - v[n1], a)

    dt_val = max(float(dt), 0.0) if dt is not None else 0.0

    L0 = np.maximum(st["t46_l_opt"][idx46], EM20)
    lambda_L = L / L0  # normalized stretch ratio
    v_max = np.maximum(st["t46_v_max"][idx46], EM20)
    v_norm = Ldot / v_max  # shortening is negative Ldot

    f_max = st["t46_f_max"][idx46]
    k_pe = st["t46_k_pe"][idx46]
    c_damp = st["t46_cdamp"][idx46]
    act_const = st["t46_activation"][idx46]

    fct_act = st["t46_fct_act"][idx46]
    fct_fl = st["t46_fct_fl"][idx46]
    fct_fv = st["t46_fct_fv"][idx46]
    fct_pass = st["t46_fct_pass"][idx46]

    # 1. Activation level alpha(t)
    alpha = act_const.copy()
    if model is not None and hasattr(model, "functions"):
        for i, fid in enumerate(fct_act):
            if fid > 0 and fid in model.functions:
                func = model.functions[fid]
                alpha[i] = np.clip(func.eval(t), 0.0, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 2. Force-Length relationship f_L(L / L0)
    f_L = np.maximum(0.0, 1.0 - ((lambda_L - 1.0) / 0.5) ** 2)
    if model is not None and hasattr(model, "functions"):
        for i, fid in enumerate(fct_fl):
            if fid > 0 and fid in model.functions:
                func = model.functions[fid]
                f_L[i] = max(0.0, func.eval(lambda_L[i]))

    # 3. Force-Velocity relationship f_v(v / v_max)
    # Contraction velocity vc = -v_norm (positive for shortening)
    vc = -v_norm
    a_rel = 0.25
    f_v = np.ones(len(idx46))
    shortening = vc >= 0.0
    # Hill hyperbolic for shortening: (1 - vc) / (1 + vc / a_rel)
    vc_s = np.clip(vc[shortening], 0.0, 1.0)
    f_v[shortening] = np.maximum(0.0, (1.0 - vc_s) / (1.0 + vc_s / a_rel))
    # Lengthening / eccentric stretch: plateau above 1.0
    lengthening = ~shortening
    if np.any(lengthening):
        vc_l = -vc[lengthening]
        f_v[lengthening] = 1.0 + 0.5 * (vc_l / (1.0 + vc_l / a_rel))

    if model is not None and hasattr(model, "functions"):
        for i, fid in enumerate(fct_fv):
            if fid > 0 and fid in model.functions:
                func = model.functions[fid]
                f_v[i] = max(0.0, func.eval(v_norm[i]))

    # Active contractile force
    F_active = f_max * alpha * f_L * f_v

    # Passive elastic force
    F_passive = k_pe * np.maximum(0.0, L - L0)
    if model is not None and hasattr(model, "functions"):
        for i, fid in enumerate(fct_pass):
            if fid > 0 and fid in model.functions:
                func = model.functions[fid]
                F_passive[i] = max(0.0, func.eval(lambda_L[i] - 1.0))

    # Viscous damping force
    Ldot_clamped = np.clip(Ldot, -v_max, v_max)
    F_damp = c_damp * Ldot_clamped

    # Muscle force (unidirectional tension)
    F = np.maximum(0.0, F_active + F_passive + F_damp)

    F_old = st["force"][idx46].copy()
    st["force"][idx46] = F

    fvec = F[:, None] * a
    if fint is not None:
        np.add.at(fint, n1, fvec)
        np.add.at(fint, n2, -fvec)

    # Internal energy work booking
    if dt_val > 0.0:
        st["eint"][idx46] += 0.5 * (F_old + F) * Ldot * dt_val

    # Time step calculation
    mass = np.maximum(st["t46_mass"][idx46], EM20)
    k_eff = np.maximum(k_pe, f_max / L0)
    pos_k = (k_eff > 0.0) & (st["t46_mass"][idx46] > 0.0)

    omega = 2.0 * np.sqrt(np.where(pos_k, k_eff / mass, 1.0))
    xi = np.where(pos_k, c_damp / np.sqrt(np.maximum(k_eff * mass, EM20)), 0.0)
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    return np.where(pos_k, dt_crit, EP30)


# ============================================================================
# Combined Dispatch for spring.py and Standalone Element Kernel
# ============================================================================

def init_advanced(group, model, log, idx19, idx44, idx46, massn, inertn):
    """Dispatcher called by spring.py init_group for advanced spring types."""
    if idx19 is not None and len(idx19):
        init_torsion_type19(group, model, log, idx19, massn, inertn)
    if idx44 is not None and len(idx44):
        init_crushing_type44(group, model, log, idx44, massn, inertn)
    if idx46 is not None and len(idx46):
        init_muscle_type46(group, model, log, idx46, massn, inertn)


def init_group(group, model, log):
    """Standalone element kernel init_group for advanced spring groups."""
    from . import spring
    return spring.init_group(group, model, log)


def forces(group, x, v, vr, dt, fint, mint):
    """Standalone element kernel forces for advanced spring groups."""
    from . import spring
    return spring.forces(group, x, v, vr, dt, fint, mint)
