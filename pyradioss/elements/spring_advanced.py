"""
Advanced spring element formulations:
- /PROP/TYPE19 (SPR_TORS): 2-node Torsion Spring (Fortran r1tors.F)
- /PROP/TYPE25 (SPR_AXI): Axisymmetric Nonlinear Spring (Fortran r6def3.F, redef3.F90, rforc3.F)
- /PROP/TYPE44 (SPR_CRUS): Crushing Spring with energy absorption (Fortran ruser44.F)
- /PROP/TYPE46 (SPR_MUSCLE): Active Muscle Spring with Hill-type dynamics (Fortran ruser46.F)

Fortran origin
--------------
* engine : ``engine/source/elements/spring/rforc3.F``
  - TYPE19: Torsion spring relative angle theta about line of nodes, rotational stiffness K_theta,
    damping C_theta, applied opposite moments M = K_theta * theta + C_theta * dtheta/dt.
  - TYPE25: ``engine/source/elements/spring/r6def3.F``, ``redef3.F90`` (nonlinear axial and shear spring
    with hardening, rate dependency, and displacement/force rupture).
  - TYPE44: ``engine/source/elements/spring/ruser44.F`` (crushable frame spring with non-recoverable
    plastic crushing displacement, compression plateau, and elastic unloading).
  - TYPE46: ``engine/source/elements/spring/ruser46.F`` (Hill-type muscle spring with activation level
    alpha(t), maximum isometric force F_max, force-length relationship f_L(L/L_0), and force-velocity
    relationship f_v(v/v_max)).
* starter:
  - ``starter/source/properties/spring/hm_read_prop19.F``
  - ``starter/source/properties/spring/hm_read_prop25.F``
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
ADVANCED_SPRING_PROP_TYPES = frozenset({19, 25, 26, 27, 44, 46})


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
# TYPE25: Axisymmetric Nonlinear Spring (/PROP/TYPE25, /PROP/SPR_AXI)
# ============================================================================

def init_axi_type25(group, model, log, idx25, massn, inertn):
    """Initialize state arrays for TYPE25 axisymmetric nonlinear springs.

    Fortran origin:
      - Starter: ``starter/source/properties/spring/hm_read_prop25.F``
      - Engine:  ``engine/source/elements/spring/r6def3.F``,
                 ``engine/source/elements/spring/redef3.F90``

    TYPE25 features independent nonlinear axial and shear behavior:
      - Axial tension/compression stiffness K_ax, damping C_ax, nonlinear curve fun_a,
        hardening flag hflag (0=elastic, 1=isotropic hardening, 2=decoupled),
        rate dependency A, B, D, and rupture limits min_rup (comp) / max_rup (tens).
      - Shear stiffness K_sh, damping C_sh, nonlinear curve fun_a, and shear rupture.
      - ileng flag: 0 for displacement Delta L, 1 for engineering strain Delta L / L0.
      - ifail flag: 0 for uniaxial failure, 1 for multiaxial failure interaction.
      - ifail2 flag: 0 for displacement rupture, 2 for force rupture.
    """
    st = group.state
    n = group.n
    m25 = len(idx25)
    if m25 == 0:
        return

    if "t25_k_ax" not in st:
        st["t25_k_ax"] = np.zeros(n)
        st["t25_c_ax"] = np.zeros(n)
        st["t25_fun_a_ax"] = np.zeros(n, dtype=np.int64)
        st["t25_hflag_ax"] = np.zeros(n, dtype=np.int64)
        st["t25_a_ax"] = np.zeros(n)
        st["t25_b_ax"] = np.zeros(n)
        st["t25_d_ax"] = np.zeros(n)
        st["t25_min_rup_ax"] = np.zeros(n)
        st["t25_max_rup_ax"] = np.zeros(n)

        st["t25_k_sh"] = np.zeros(n)
        st["t25_c_sh"] = np.zeros(n)
        st["t25_fun_a_sh"] = np.zeros(n, dtype=np.int64)
        st["t25_hflag_sh"] = np.zeros(n, dtype=np.int64)
        st["t25_min_rup_sh"] = np.zeros(n)
        st["t25_max_rup_sh"] = np.zeros(n)

        st["t25_ileng"] = np.zeros(n, dtype=np.int64)
        st["t25_ifail"] = np.zeros(n, dtype=np.int64)
        st["t25_ifail2"] = np.zeros(n, dtype=np.int64)
        st["t25_skew_id"] = np.zeros(n, dtype=np.int64)

        st["t25_mass"] = np.zeros(n)
        st["t25_inertia"] = np.zeros(n)

        st["t25_delta_sh"] = np.zeros((n, 3))
        st["t25_delta_p_ax"] = np.zeros(n)
        st["t25_fxep"] = np.zeros(n)
        st["t25_dx_old"] = np.zeros(n)

    pos = {int(e): j for j, e in enumerate(idx25)}
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 0) != 25:
            continue
        rng = np.arange(group.n)[sl]
        local = [e for e in rng if int(e) in pos]
        if not len(local):
            continue
        p = getattr(prop, "params", {}) or {}
        p25_obj = None
        if hasattr(model, "prop_type25s"):
            p25_obj = model.prop_type25s.get(getattr(prop, "id", 0))

        tens = p.get("tension", {})
        if not tens and p25_obj is not None:
            tens = getattr(p25_obj, "tension", {})
        shear = p.get("shear", {})
        if not shear and p25_obj is not None:
            shear = getattr(p25_obj, "shear", {})

        # Axial parameters
        k_ax = _safe_param(tens, "stiff", _safe_param(p, "stiff_tens", _safe_param(p, "k_ax", _safe_param(p, "k", 0.0))))
        c_ax = _safe_param(tens, "damp", _safe_param(p, "damp_tens", _safe_param(p, "c_ax", _safe_param(p, "c", 0.0))))
        a_ax = _safe_param(tens, "a", _safe_param(p, "a_tens", 0.0))
        b_ax = _safe_param(tens, "b", _safe_param(p, "b_tens", 0.0))
        d_ax = _safe_param(tens, "d", _safe_param(p, "d_tens", 0.0))
        fun_a_ax = _safe_int_param(tens, "fun_a", _safe_int_param(p, "fun_a_tens", _safe_int_param(p, "fun_a1", 0)))
        hflag_ax = _safe_int_param(tens, "hflag", _safe_int_param(p, "hflag_tens", _safe_int_param(p, "hflag1", 0)))
        min_rup_ax = _safe_param(tens, "min_rup", _safe_param(p, "min_rup_tens", _safe_param(p, "min_rup1", _safe_param(p, "min_rup", 0.0))))
        max_rup_ax = _safe_param(tens, "max_rup", _safe_param(p, "max_rup_tens", _safe_param(p, "max_rup1", _safe_param(p, "max_rup", 0.0))))

        # Shear parameters
        k_sh = _safe_param(shear, "stiff", _safe_param(p, "stiff_shear", _safe_param(p, "k_sh", 0.0)))
        c_sh = _safe_param(shear, "damp", _safe_param(p, "damp_shear", _safe_param(p, "c_sh", 0.0)))
        fun_a_sh = _safe_int_param(shear, "fun_a", _safe_int_param(p, "fun_a_shear", _safe_int_param(p, "fun_a2", 0)))
        hflag_sh = _safe_int_param(shear, "hflag", _safe_int_param(p, "hflag_shear", _safe_int_param(p, "hflag2", 0)))
        min_rup_sh = _safe_param(shear, "min_rup", _safe_param(p, "min_rup_shear", _safe_param(p, "min_rup2", 0.0)))
        max_rup_sh = _safe_param(shear, "max_rup", _safe_param(p, "max_rup_shear", _safe_param(p, "max_rup2", 0.0)))

        # Metadata
        mass = _safe_param(p, "mass", getattr(p25_obj, "mass", 0.0) if p25_obj else 0.0)
        inertia = _safe_param(p, "inertia", getattr(p25_obj, "inertia", 0.0) if p25_obj else 0.0)
        ileng = _safe_int_param(p, "ileng", getattr(p25_obj, "ileng", 0) if p25_obj else 0)
        ifail = _safe_int_param(p, "ifail", getattr(p25_obj, "ifail", 0) if p25_obj else 0)
        ifail2 = _safe_int_param(p, "ifail2", getattr(p25_obj, "ifail2", 0) if p25_obj else 0)
        skew_id = _safe_int_param(p, "skew_id", getattr(p25_obj, "skew_id", 0) if p25_obj else 0)

        st["t25_k_ax"][local] = k_ax
        st["t25_c_ax"][local] = c_ax
        st["t25_fun_a_ax"][local] = fun_a_ax
        st["t25_hflag_ax"][local] = hflag_ax
        st["t25_a_ax"][local] = a_ax
        st["t25_b_ax"][local] = b_ax
        st["t25_d_ax"][local] = d_ax
        st["t25_min_rup_ax"][local] = min_rup_ax
        st["t25_max_rup_ax"][local] = max_rup_ax

        st["t25_k_sh"][local] = k_sh
        st["t25_c_sh"][local] = c_sh
        st["t25_fun_a_sh"][local] = fun_a_sh
        st["t25_hflag_sh"][local] = hflag_sh
        st["t25_min_rup_sh"][local] = min_rup_sh
        st["t25_max_rup_sh"][local] = max_rup_sh

        st["t25_ileng"][local] = ileng
        st["t25_ifail"][local] = ifail
        st["t25_ifail2"][local] = ifail2
        st["t25_skew_id"][local] = skew_id
        st["t25_mass"][local] = mass
        st["t25_inertia"][local] = inertia

        st["k"][local] = max(k_ax, k_sh)
        st["cdamp"][local] = max(c_ax, c_sh)
        st["mass"][local] = mass

        if massn is not None and mass > 0.0:
            for e in local:
                if 2 * e + 1 < len(massn):
                    massn[2 * e] += mass / 2.0
                    massn[2 * e + 1] += mass / 2.0

        if inertn is not None and inertia > 0.0:
            for e in local:
                if 2 * e + 1 < len(inertn):
                    inertn[2 * e] += inertia / 2.0
                    inertn[2 * e + 1] += inertia / 2.0


def forces_axi_type25(group, x, v, dt, fint, idx25):
    """Compute forces for TYPE25 axisymmetric nonlinear springs (r6def3.F, redef3.F90)."""
    if idx25 is None or len(idx25) == 0:
        return np.empty(0)
    st = group.state
    conn = group.conn[idx25]
    n1, n2 = conn[:, 0], conn[:, 1]
    model = st.get("model")
    n_elem = len(idx25)

    dx = x[n2] - x[n1]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    L0 = st["L0"][idx25]
    delta_L = L - L0

    if v is None:
        dv = np.zeros_like(dx)
        v_ax = np.zeros(n_elem)
        v_sh = np.zeros_like(dx)
    else:
        dv = v[n2] - v[n1]
        v_ax = np.einsum("nb,nb->n", dv, a)
        v_sh = dv - v_ax[:, None] * a

    dt_val = max(float(dt), 0.0) if dt is not None else 0.0

    # Accumulated transverse shear displacement
    if dt_val > 0.0:
        st["t25_delta_sh"][idx25] += v_sh * dt_val
    delta_sh_vec = st["t25_delta_sh"][idx25]
    # Remove any axial component from numerical drift to stay exactly in transverse plane
    axial_proj = np.einsum("nb,nb->n", delta_sh_vec, a)
    delta_sh_vec -= axial_proj[:, None] * a
    delta_sh = norm3(delta_sh_vec)
    u_sh = np.where((delta_sh > EM20)[:, None], delta_sh_vec / np.maximum(delta_sh[:, None], EM20), 0.0)

    # Engineering strain vs displacement
    ileng = st["t25_ileng"][idx25]
    use_strain = (ileng == 1) & (L0 > EM20)
    scale_len = np.where(use_strain, L0, 1.0)
    dx_tens = delta_L / scale_len
    dx_sh = delta_sh / scale_len

    k_ax = st["t25_k_ax"][idx25]
    c_ax = st["t25_c_ax"][idx25]
    fun_a_ax = st["t25_fun_a_ax"][idx25]
    hflag_ax = st["t25_hflag_ax"][idx25]
    a_ax = st["t25_a_ax"][idx25]
    b_ax = st["t25_b_ax"][idx25]
    d_ax = st["t25_d_ax"][idx25]
    min_rup_ax = st["t25_min_rup_ax"][idx25]
    max_rup_ax = st["t25_max_rup_ax"][idx25]

    k_sh = st["t25_k_sh"][idx25]
    c_sh = st["t25_c_sh"][idx25]
    fun_a_sh = st["t25_fun_a_sh"][idx25]
    hflag_sh = st["t25_hflag_sh"][idx25]
    min_rup_sh = st["t25_min_rup_sh"][idx25]
    max_rup_sh = st["t25_max_rup_sh"][idx25]

    ifail = st["t25_ifail"][idx25]
    ifail2 = st["t25_ifail2"][idx25]

    # Active status
    if "off" not in st:
        st["off"] = np.ones(group.n)
    alive = (st["off"][idx25] > 0.0)

    # 1. Axial quasi-static force
    F_quasi_ax = k_ax * dx_tens
    has_model_funcs = (model is not None and hasattr(model, "functions"))

    for i in range(n_elem):
        fa = fun_a_ax[i]
        if fa > 0 and has_model_funcs and fa in model.functions:
            func = model.functions[fa]
            f_val = func.eval(dx_tens[i])
            hf = hflag_ax[i]
            if hf == 0:
                # Nonlinear elastic: pure curve evaluation
                F_quasi_ax[i] = f_val
            elif hf == 1:
                # Isotropic hardening
                f_ep = st["t25_fxep"][idx25[i]]
                ddx = dx_tens[i] - st["t25_dx_old"][idx25[i]]
                f_trial = f_ep + k_ax[i] * ddx
                if abs(f_trial) > abs(f_val):
                    f_trial = np.sign(f_trial) * abs(f_val)
                F_quasi_ax[i] = f_trial
                st["t25_fxep"][idx25[i]] = f_trial
            elif hf == 2:
                # Decoupled tension / compression
                dp = st["t25_delta_p_ax"][idx25[i]]
                if dx_tens[i] > dp:
                    f_trial = k_ax[i] * (dx_tens[i] - dp)
                    f_clamp = min(f_trial, f_val)
                    st["t25_delta_p_ax"][idx25[i]] = dx_tens[i] - f_clamp / max(k_ax[i], EM20)
                    F_quasi_ax[i] = f_clamp
                else:
                    F_quasi_ax[i] = 0.0
            else:
                F_quasi_ax[i] = k_ax[i] * dx_tens[i] + f_val

    st["t25_dx_old"][idx25] = dx_tens.copy()

    # Rate effect dfac = ak + b * log(max(1, |v_ax/d|))
    has_rate = (d_ax > 0.0) & (b_ax > 0.0)
    dfac = np.ones(n_elem)
    if np.any(has_rate):
        dvv = np.maximum(1.0, np.abs(v_ax[has_rate]) / d_ax[has_rate])
        a_term = np.where(a_ax[has_rate] > 0.0, a_ax[has_rate], 1.0)
        dfac[has_rate] = a_term + b_ax[has_rate] * np.log(dvv)
    pure_a = (~has_rate) & (a_ax > 0.0)
    if np.any(pure_a):
        dfac[pure_a] = a_ax[pure_a]

    F_quasi_ax *= dfac
    F_damp_ax = c_ax * v_ax
    F_ax = F_quasi_ax + F_damp_ax

    # 2. Shear quasi-static force
    F_quasi_sh = k_sh * dx_sh
    for i in range(n_elem):
        fa_s = fun_a_sh[i]
        if fa_s > 0 and has_model_funcs and fa_s in model.functions:
            func_s = model.functions[fa_s]
            f_val_s = func_s.eval(dx_sh[i])
            hf_s = hflag_sh[i]
            if hf_s == 0:
                F_quasi_sh[i] = f_val_s
            else:
                F_quasi_sh[i] = k_sh[i] * dx_sh[i] + f_val_s

    F_sh_vec = F_quasi_sh[:, None] * u_sh + c_sh[:, None] * v_sh

    # 3. Rupture checks
    for i in range(n_elem):
        if not alive[i]:
            continue
        rup = False
        if ifail2[i] == 2:
            # Force rupture criterion
            f_a = F_ax[i]
            f_s = norm3(F_sh_vec[i:i+1])[0]
            if ifail[i] == 0:
                if f_a > 0.0 and max_rup_ax[i] > 0.0 and f_a >= max_rup_ax[i]:
                    rup = True
                if f_a < 0.0 and min_rup_ax[i] != 0.0 and f_a <= min_rup_ax[i]:
                    rup = True
                if max_rup_sh[i] > 0.0 and f_s >= max_rup_sh[i]:
                    rup = True
            else:
                # Multiaxial force interaction
                crit = 0.0
                if f_a > 0.0 and max_rup_ax[i] > 0.0:
                    crit += (f_a / max_rup_ax[i]) ** 2
                elif f_a < 0.0 and min_rup_ax[i] != 0.0:
                    crit += (f_a / min_rup_ax[i]) ** 2
                if max_rup_sh[i] > 0.0:
                    crit += (f_s / max_rup_sh[i]) ** 2
                if crit >= 1.0:
                    rup = True
        else:
            # Displacement rupture criterion (ifail2 == 0)
            u_a = dx_tens[i]
            u_s = dx_sh[i]
            if ifail[i] == 0:
                # Uniaxial failure
                if u_a > 0.0 and max_rup_ax[i] > 0.0 and u_a >= max_rup_ax[i]:
                    rup = True
                if u_a < 0.0 and min_rup_ax[i] != 0.0 and u_a <= min_rup_ax[i]:
                    rup = True
                if max_rup_sh[i] > 0.0 and u_s >= max_rup_sh[i]:
                    rup = True
            else:
                # Multiaxial failure
                crit = 0.0
                if u_a > 0.0 and max_rup_ax[i] > 0.0:
                    crit += (u_a / max_rup_ax[i]) ** 2
                elif u_a < 0.0 and min_rup_ax[i] != 0.0:
                    crit += (u_a / min_rup_ax[i]) ** 2
                if max_rup_sh[i] > 0.0:
                    crit += (u_s / max_rup_sh[i]) ** 2
                if crit >= 1.0:
                    rup = True

        if rup:
            st["off"][idx25[i]] = 0.0
            alive[i] = False

    # Zero out forces for ruptured elements
    F_ax = np.where(alive, F_ax, 0.0)
    F_sh_vec = np.where(alive[:, None], F_sh_vec, 0.0)

    # 4. Total force and nodal application
    F_tot = F_ax[:, None] * a + F_sh_vec

    if fint is not None:
        np.add.at(fint, n1, F_tot)
        np.add.at(fint, n2, -F_tot)

    # 5. Energy and force bookkeeping
    F_old = st["force"][idx25].copy()
    st["force"][idx25] = F_ax
    if dt_val > 0.0:
        p_ax = 0.5 * (F_old + F_ax) * v_ax
        p_sh = np.einsum("nb,nb->n", F_sh_vec, v_sh)
        st["eint"][idx25] += (p_ax + p_sh) * dt_val

    # 6. Critical time step
    mass = np.maximum(st["t25_mass"][idx25], EM20)
    k_eff = np.maximum(k_ax, k_sh)
    c_eff = np.maximum(c_ax, c_sh)
    pos_k = (k_eff > 0.0) & (st["t25_mass"][idx25] > 0.0)
    pure_c = (k_eff <= 0.0) & (c_eff > 0.0) & (st["t25_mass"][idx25] > 0.0)

    omega = 2.0 * np.sqrt(np.where(pos_k, k_eff / mass, 1.0))
    xi = np.where(pos_k, c_eff / np.sqrt(np.maximum(k_eff * mass, EM20)), 0.0)
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    dt_c = np.where(pure_c, 0.5 * mass / np.maximum(c_eff, EM20), EP30)
    return np.where(pos_k, dt_crit, dt_c)


# ============================================================================
# TYPE26: Tabulated Nonlinear Spring (/PROP/SPR_TAB)
# ============================================================================
# Fortran origin:
#   engine/source/elements/spring/r26def3.F
#   engine/source/elements/spring/r26sig.F
#   engine/source/elements/spring/rforc3.F
#   starter/source/properties/spring/hm_read_prop26.F

def init_tab_type26(group, model, log, idx26, massn, inertn):
    """Initialize state arrays for TYPE26 (/PROP/SPR_TAB) tabulated nonlinear springs.

    Fortran origin: starter/source/properties/spring/hm_read_prop26.F,
    engine/source/elements/spring/r26def3.F.
    """
    st = group.state
    n = group.n
    if len(idx26) == 0:
        return

    if "t26_kmax" not in st:
        st["t26_kmax"] = np.zeros(n)
        st["t26_mass"] = np.zeros(n)
        st["t26_alpha"] = np.ones(n)
        st["t26_scale"] = np.ones(n)
        st["t26_dmin"] = np.full(n, -1e30)
        st["t26_dmax"] = np.full(n, 1e30)
        st["t26_ileng"] = np.zeros(n, dtype=np.int64)
        st["t26_dx_old"] = np.zeros(n)
        st["t26_dv0"] = np.zeros(n)
        st["t26_load_curves"] = [[] for _ in range(n)]
        st["t26_unload_curves"] = [[] for _ in range(n)]

    pos = {int(e): j for j, e in enumerate(idx26)}
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 0) != 26:
            continue
        rng = np.arange(group.n)[sl]
        local = [e for e in rng if int(e) in pos]
        if not len(local):
            continue

        p = getattr(prop, "params", {}) or {}
        ms = float(getattr(prop, "mass", _safe_param(p, "mass", 0.0)))
        km = float(getattr(prop, "kmax", _safe_param(p, "stiff0", _safe_param(p, "kmax", _safe_param(p, "k", 0.0)))))
        al = float(getattr(prop, "alpha", _safe_param(p, "alpha", _safe_param(p, "alpha1", 1.0))))
        if al == 0.0:
            al = 1.0
        sc = float(getattr(prop, "lscale", _safe_param(p, "scale", _safe_param(p, "lscale", 1.0))))
        if sc == 0.0:
            sc = 1.0
        d_min = float(getattr(prop, "dmin", _safe_param(p, "dmin", 0.0)))
        if d_min == 0.0:
            d_min = -1e30
        else:
            d_min = -abs(d_min)
        d_max = float(getattr(prop, "dmax", _safe_param(p, "dmax", 0.0)))
        if d_max == 0.0:
            d_max = 1e30
        else:
            d_max = abs(d_max)
        il = int(getattr(prop, "ileng", _safe_int_param(p, "ileng", 0)))
        if il == 1:
            sc = 1.0

        # Load curves
        ld_curves = []
        if hasattr(prop, "loading_curves") and prop.loading_curves:
            for c in prop.loading_curves:
                fid = getattr(c, "fct_id", 0)
                fsc = getattr(c, "fscale", 1.0)
                sr = getattr(c, "strain_rate", 0.0)
                ld_curves.append((fid, fsc, sr))
        elif "load_curves" in p and p["load_curves"]:
            for c in p["load_curves"]:
                fid = int(c.get("fun_load", 0))
                fsc = float(c.get("scale_load", 1.0))
                sr = float(c.get("strainrate_load", 0.0))
                ld_curves.append((fid, fsc, sr))
        ld_curves.sort(key=lambda x: x[2])

        # Unload curves
        uld_curves = []
        if hasattr(prop, "unloading_curves") and prop.unloading_curves:
            for c in prop.unloading_curves:
                fid = getattr(c, "fct_id", 0)
                fsc = getattr(c, "fscale", 1.0)
                sr = getattr(c, "strain_rate", 0.0)
                uld_curves.append((fid, fsc, sr))
        elif "unload_curves" in p and p["unload_curves"]:
            for c in p["unload_curves"]:
                fid = int(c.get("fun_unload", 0))
                fsc = float(c.get("scale_unload", 1.0))
                sr = float(c.get("strainrate_unload", 0.0))
                uld_curves.append((fid, fsc, sr))
        if not uld_curves:
            uld_curves = list(ld_curves)
        else:
            uld_curves.sort(key=lambda x: x[2])

        st["t26_kmax"][local] = km
        st["t26_mass"][local] = ms
        st["t26_alpha"][local] = al
        st["t26_scale"][local] = sc
        st["t26_dmin"][local] = d_min
        st["t26_dmax"][local] = d_max
        st["t26_ileng"][local] = il
        st["k"][local] = km
        st["mass"][local] = ms

        for e in local:
            st["t26_load_curves"][e] = list(ld_curves)
            st["t26_unload_curves"][e] = list(uld_curves)

        if massn is not None and ms > 0.0:
            for e in local:
                if 2 * e + 1 < len(massn):
                    massn[2 * e] += ms / 2.0
                    massn[2 * e + 1] += ms / 2.0


def forces_tab_type26(group, x, v, dt, fint, idx26):
    """Compute forces for TYPE26 (/PROP/SPR_TAB) tabulated nonlinear springs.

    Fortran origin: engine/source/elements/spring/r26def3.F, r26sig.F, rforc3.F.
    """
    if idx26 is None or len(idx26) == 0:
        return np.empty(0)

    st = group.state
    conn = group.conn[idx26]
    n1, n2 = conn[:, 0], conn[:, 1]
    n_elem = len(idx26)

    dx = x[n2] - x[n1]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    e1 = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    L0 = st["L0"][idx26]
    alive = st.get("off", np.ones(group.n))[idx26] > 0.0

    ileng = st["t26_ileng"][idx26]
    xl0 = np.where(ileng != 0, np.maximum(L0, EM20), 1.0)
    dl_total = L - L0
    dx_val = dl_total / xl0
    dx_old = st["t26_dx_old"][idx26]
    ddx = dx_val - dx_old

    dt_val = dt if (dt is not None and dt > 0.0) else EP30
    dvx = ddx / dt_val
    dv_raw = np.abs(dvx)

    alpha = st["t26_alpha"][idx26]
    dv0 = st["t26_dv0"][idx26]
    dv = (1.0 - alpha) * dv0 + alpha * dv_raw
    st["t26_dv0"][idx26] = dv
    st["t26_dx_old"][idx26] = dx_val.copy()

    kmax = st["t26_kmax"][idx26]
    scale_x = st["t26_scale"][idx26]
    f_old = st["force"][idx26].copy()
    FX = np.zeros(n_elem)

    model = st.get("model")
    has_model_funcs = model is not None and hasattr(model, "functions")

    for i in range(n_elem):
        if not alive[i]:
            continue

        elem_idx = idx26[i]
        d_x = dx_val[i]
        k_val = kmax[i]
        sc_x = scale_x[i]
        v_rel = dv[i]

        if d_x >= 0.0:
            # Linear elastic tension (r26sig.F lines 125-126)
            FX[i] = k_val * d_x
        else:
            # Trial compression force (r26sig.F line 128)
            f_trial = f_old[i] + k_val * ddx[i]
            x_eval = abs(d_x) / max(sc_x, EM20)

            # 1. Calculation of upper bound FMAX (loading curves, r26sig.F lines 131-175)
            load_c = st["t26_load_curves"][elem_idx]
            n_ld = len(load_c)
            if n_ld == 0:
                f_max_bound = -EM20
            elif n_ld == 1:
                fid1, yfac1, rate1 = load_c[0]
                y1 = 0.0
                if has_model_funcs and fid1 in model.functions:
                    y1 = yfac1 * model.functions[fid1].eval(x_eval)
                f_max_bound = -max(y1, EM20)
            else:
                j1 = 0
                for j in range(1, n_ld):
                    if v_rel >= load_c[j][2]:
                        j1 = j
                if j1 >= n_ld - 1:
                    j1 = n_ld - 2
                j2 = j1 + 1

                fid1, yfac1, rate1 = load_c[j1]
                fid2, yfac2, rate2 = load_c[j2]

                y1 = 0.0
                if has_model_funcs and fid1 in model.functions:
                    y1 = yfac1 * model.functions[fid1].eval(x_eval)
                y2 = 0.0
                if has_model_funcs and fid2 in model.functions:
                    y2 = yfac2 * model.functions[fid2].eval(x_eval)

                fac = (v_rel - rate1) / max(rate2 - rate1, EM20)
                fac = min(max(fac, 0.0), 1.0)
                y_interp = y1 + fac * (y2 - y1)
                f_max_bound = -max(y_interp, EM20)

            # 2. Calculation of lower bound FMIN (unloading curves, r26sig.F lines 176-220)
            unload_c = st["t26_unload_curves"][elem_idx]
            n_uld = len(unload_c)
            if n_uld == 0:
                f_min_bound = f_max_bound
            elif n_uld == 1:
                fid1, yfac1, rate1 = unload_c[0]
                y1 = 0.0
                if has_model_funcs and fid1 in model.functions:
                    y1 = yfac1 * model.functions[fid1].eval(x_eval)
                f_min_bound = -max(y1, EM20)
            else:
                j1 = 0
                for j in range(1, n_uld):
                    if v_rel >= unload_c[j][2]:
                        j1 = j
                if j1 >= n_uld - 1:
                    j1 = n_uld - 2
                j2 = j1 + 1

                fid1, yfac1, rate1 = unload_c[j1]
                fid2, yfac2, rate2 = unload_c[j2]

                y1 = 0.0
                if has_model_funcs and fid1 in model.functions:
                    y1 = yfac1 * model.functions[fid1].eval(x_eval)
                y2 = 0.0
                if has_model_funcs and fid2 in model.functions:
                    y2 = yfac2 * model.functions[fid2].eval(x_eval)

                fac = (v_rel - rate1) / max(rate2 - rate1, EM20)
                fac = min(max(fac, 0.0), 1.0)
                y_interp = y1 + fac * (y2 - y1)
                f_min_bound = -max(y_interp, EM20)

            # Clamping: FMAX <= F <= FMIN <= 0 (r26sig.F lines 222-228)
            if f_min_bound < f_max_bound:
                f_min_bound, f_max_bound = f_max_bound, f_min_bound

            if f_trial > f_min_bound:
                FX[i] = f_min_bound
            elif f_trial < f_max_bound:
                FX[i] = f_max_bound
            else:
                FX[i] = f_trial

    # Rupture checks (r26def3.F lines 156-170)
    dmin = st["t26_dmin"][idx26]
    dmax = st["t26_dmax"][idx26]
    for i in range(n_elem):
        if not alive[i]:
            continue
        if dl_total[i] > dmax[i] * xl0[i] or dl_total[i] < dmin[i] * xl0[i]:
            st["off"][idx26[i]] = 0.0
            alive[i] = False
            FX[i] = 0.0

    FX = np.where(alive, FX, 0.0)
    st["force"][idx26] = FX

    # Internal energy update (r26sig.F lines 236-237)
    if dt is not None and dt > 0.0:
        dE = 0.5 * ddx * (FX + f_old) * xl0
        st["eint"][idx26] += np.where(alive, dE, 0.0)

    # Nodal force scatter
    fvec = FX[:, None] * e1
    if fint is not None:
        np.add.at(fint, n1, fvec)
        np.add.at(fint, n2, -fvec)

    # Critical time step (r26def3.F lines 178-181)
    mass = np.maximum(st["t26_mass"][idx26], EM20) * xl0
    k_dt = np.maximum(kmax, EM20) / xl0
    omega = 2.0 * np.sqrt(k_dt / mass)
    dt_crit = 2.0 / omega
    return np.where(alive, dt_crit, EP30)


# ============================================================================
# TYPE27: Spring with Bilinear Damping & Nonlinear Exponent (/PROP/SPR_BDAMP)
# ============================================================================
# Fortran origin:
#   engine/source/elements/spring/r27def3.F
#   engine/source/elements/spring/rforc3.F
#   starter/source/properties/spring/hm_read_prop27.F

def init_bdamp_type27(group, model, log, idx27, massn, inertn):
    """Initialize state arrays for TYPE27 (/PROP/SPR_BDAMP) springs.

    Fortran origin: starter/source/properties/spring/hm_read_prop27.F,
    engine/source/elements/spring/r27def3.F.
    """
    st = group.state
    n = group.n
    if len(idx27) == 0:
        return

    if "t27_stiff" not in st:
        st["t27_mass"] = np.zeros(n)
        st["t27_stiff"] = np.zeros(n)
        st["t27_damp"] = np.zeros(n)
        st["t27_nexp"] = np.ones(n)
        st["t27_dmin"] = np.full(n, -1e30)
        st["t27_dmax"] = np.full(n, 1e30)
        st["t27_gap"] = np.zeros(n)
        st["t27_ifail"] = np.zeros(n, dtype=np.int64)
        st["t27_ileng"] = np.zeros(n, dtype=np.int64)
        st["t27_itens"] = np.zeros(n, dtype=np.int64)
        st["t27_fsmooth"] = np.zeros(n, dtype=np.int64)
        st["t27_fcut"] = np.zeros(n)
        st["t27_fun1"] = np.zeros(n, dtype=np.int64)
        st["t27_ascale1"] = np.ones(n)
        st["t27_fscale1"] = np.ones(n)
        st["t27_fun2"] = np.zeros(n, dtype=np.int64)
        st["t27_ascale2"] = np.ones(n)
        st["t27_fscale2"] = np.ones(n)
        st["t27_dx_old"] = np.zeros(n)

    pos = {int(e): j for j, e in enumerate(idx27)}
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 0) != 27:
            continue
        rng = np.arange(group.n)[sl]
        local = [e for e in rng if int(e) in pos]
        if not len(local):
            continue

        p = getattr(prop, "params", {}) or {}
        ms = float(getattr(prop, "mass", _safe_param(p, "mass", 0.0)))
        km = float(getattr(prop, "stiff", _safe_param(p, "stiff", _safe_param(p, "k", 0.0))))
        dm = float(getattr(prop, "damp", _safe_param(p, "damp", _safe_param(p, "c", 0.0))))
        nx = float(getattr(prop, "nexp", _safe_param(p, "nexp", _safe_param(p, "n", 1.0))))
        if nx <= 0.0:
            nx = 1.0

        d_min = float(getattr(prop, "delta_min", _safe_param(p, "delta_min", _safe_param(p, "min_rup", 0.0))))
        if d_min == 0.0:
            d_min = -1e30
        else:
            d_min = -abs(d_min)

        d_max = float(getattr(prop, "delta_max", _safe_param(p, "delta_max", _safe_param(p, "max_rup", 0.0))))
        if d_max == 0.0:
            d_max = 1e30
        else:
            d_max = abs(d_max)

        gp = -abs(float(getattr(prop, "gap", _safe_param(p, "gap", 0.0))))
        it = int(getattr(prop, "itens", _safe_int_param(p, "itens", 0)))
        if gp < 0.0:
            it = 0
        ifl = int(getattr(prop, "ifail", _safe_int_param(p, "ifail", 0)))
        il = int(getattr(prop, "ileng", _safe_int_param(p, "ileng", 0)))
        fsm = int(getattr(prop, "fsmooth", _safe_int_param(p, "fsmooth", 0)))
        fc = float(getattr(prop, "fcut", _safe_param(p, "fcut", 0.0)))
        if fc > 0.0:
            fsm = 1

        fn1 = int(getattr(prop, "fct_id1", _safe_int_param(p, "fun1", _safe_int_param(p, "fct_id1", 0))))
        as1 = float(getattr(prop, "ascale1", _safe_param(p, "ascale1", 1.0)))
        fs1 = float(getattr(prop, "fscale1", _safe_param(p, "fscale1", 1.0)))
        if as1 == 0.0:
            as1 = 1.0
        if fs1 == 0.0:
            fs1 = 1.0

        fn2 = int(getattr(prop, "fct_id2", _safe_int_param(p, "fun2", _safe_int_param(p, "fct_id2", 0))))
        as2 = float(getattr(prop, "ascale2", _safe_param(p, "ascale2", 1.0)))
        fs2 = float(getattr(prop, "fscale2", _safe_param(p, "fscale2", 1.0)))
        if as2 == 0.0:
            as2 = 1.0
        if fs2 == 0.0:
            fs2 = 1.0

        st["t27_mass"][local] = ms
        st["t27_stiff"][local] = km
        st["t27_damp"][local] = dm
        st["t27_nexp"][local] = nx
        st["t27_dmin"][local] = d_min
        st["t27_dmax"][local] = d_max
        st["t27_gap"][local] = gp
        st["t27_ifail"][local] = ifl
        st["t27_ileng"][local] = il
        st["t27_itens"][local] = it
        st["t27_fsmooth"][local] = fsm
        st["t27_fcut"][local] = fc
        st["t27_fun1"][local] = fn1
        st["t27_ascale1"][local] = as1
        st["t27_fscale1"][local] = fs1
        st["t27_fun2"][local] = fn2
        st["t27_ascale2"][local] = as2
        st["t27_fscale2"][local] = fs2
        st["k"][local] = km
        st["cdamp"][local] = dm
        st["mass"][local] = ms

        if massn is not None and ms > 0.0:
            for e in local:
                if 2 * e + 1 < len(massn):
                    massn[2 * e] += ms / 2.0
                    massn[2 * e + 1] += ms / 2.0


def forces_bdamp_type27(group, x, v, dt, fint, idx27):
    """Compute forces for TYPE27 (/PROP/SPR_BDAMP) springs.

    Fortran origin: engine/source/elements/spring/r27def3.F, rforc3.F.
    """
    if idx27 is None or len(idx27) == 0:
        return np.empty(0)

    st = group.state
    conn = group.conn[idx27]
    n1, n2 = conn[:, 0], conn[:, 1]
    n_elem = len(idx27)

    dx = x[n2] - x[n1]
    norm = norm3(dx)
    degen = (norm < EM20)
    L = np.where(degen, EM20, norm)
    e1 = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])

    L0 = st["L0"][idx27]
    alive = st.get("off", np.ones(group.n))[idx27] > 0.0

    ileng = st["t27_ileng"][idx27]
    xl0 = np.where(ileng != 0, np.maximum(L0, EM20), 1.0)
    dl_total = L - L0
    dx_val = dl_total / xl0
    dx_old = st["t27_dx_old"][idx27]
    ddx = dx_val - dx_old

    dt_val = dt if (dt is not None and dt > 0.0) else EP30
    dvl = ddx / dt_val
    st["t27_dx_old"][idx27] = dx_val.copy()

    f_old = st["force"][idx27].copy()
    gap = st["t27_gap"][idx27]
    itens = st["t27_itens"][idx27]

    active = alive & ((dx_val < gap) | (itens > 0))

    F = np.zeros(n_elem)
    k_eff = np.zeros(n_elem)
    c_eff = np.zeros(n_elem)

    model = st.get("model")
    has_model_funcs = model is not None and hasattr(model, "functions")

    for i in range(n_elem):
        if not active[i]:
            continue

        elem_idx = idx27[i]
        delta = dx_val[i] - gap[i]
        fn1 = st["t27_fun1"][elem_idx]
        fn2 = st["t27_fun2"][elem_idx]
        km = st["t27_stiff"][elem_idx]
        cm = st["t27_damp"][elem_idx]
        nx = st["t27_nexp"][elem_idx]

        # 1. Stiffness force FK
        if fn1 > 0 and has_model_funcs and fn1 in model.functions:
            as1 = st["t27_ascale1"][elem_idx]
            fs1 = st["t27_fscale1"][elem_idx]
            fk = fs1 * model.functions[fn1].eval(delta / as1)
            k_eff[i] = km if km > 0.0 else 1000.0
        else:
            if abs(delta) > 0.0:
                fk = math.copysign(1.0, delta) * km * (abs(delta) ** nx)
            else:
                fk = 0.0
            k_eff[i] = km
            if nx > 1.0:
                delta_old = dx_old[i] - gap[i]
                fk_old = math.copysign(1.0, delta_old) * km * (abs(delta_old) ** nx) if abs(delta_old) > 0.0 else 0.0
                slope = abs(fk - fk_old) / max(abs(ddx[i]), EM20)
                k_eff[i] = max(slope, km)

        # 2. Damping force FD
        if fn2 > 0 and has_model_funcs and fn2 in model.functions:
            as2 = st["t27_ascale2"][elem_idx]
            fs2 = st["t27_fscale2"][elem_idx]
            fd = fs2 * model.functions[fn2].eval(dvl[i] / as2)
            c_eff[i] = cm
        else:
            fd = cm * dvl[i]
            c_eff[i] = cm

        # 3. Assembling forces (r27def3.F lines 259-265)
        if abs(fk) > abs(fd):
            F[i] = fk + fd
        else:
            F[i] = 2.0 * fk
            k_eff[i] = 2.0 * k_eff[i]
            c_eff[i] = 0.0

        # 4. Spring force filtering (r27def3.F lines 272-275)
        fsm = st["t27_fsmooth"][elem_idx]
        fc = st["t27_fcut"][elem_idx]
        if fsm > 0 and fc > 0.0:
            omega_cut = 2.0 * math.pi * dt_val * fc
            alpha = omega_cut / (omega_cut + 1.0)
            F[i] = alpha * F[i] + (1.0 - alpha) * f_old[i]

    # Rupture checks (r27def3.F lines 322-374)
    ifail = st["t27_ifail"][idx27]
    dmin = st["t27_dmin"][idx27]
    dmax = st["t27_dmax"][idx27]
    for i in range(n_elem):
        if not alive[i]:
            continue
        elem_idx = idx27[i]
        ifl = ifail[i]
        it = itens[i]
        d_val = dx_val[i] * xl0[i]
        f_val = F[i]

        if ifl == 1:
            # Displacement rupture
            if it > 0:
                if d_val > dmax[i] * xl0[i] or d_val < dmin[i] * xl0[i]:
                    st["off"][elem_idx] = 0.0
                    alive[i] = False
                    F[i] = 0.0
            else:
                if d_val < dmin[i] * xl0[i]:
                    st["off"][elem_idx] = 0.0
                    alive[i] = False
                    F[i] = 0.0
        elif ifl == 2:
            # Force rupture
            if it > 0:
                if f_val > dmax[i] or f_val < dmin[i]:
                    st["off"][elem_idx] = 0.0
                    alive[i] = False
                    F[i] = 0.0
            else:
                if f_val < dmin[i]:
                    st["off"][elem_idx] = 0.0
                    alive[i] = False
                    F[i] = 0.0

    F = np.where(alive, F, 0.0)
    st["force"][idx27] = F

    # Internal energy update (r27def3.F lines 308-309)
    if dt is not None and dt > 0.0:
        dE = 0.5 * ddx * (F + f_old) * xl0
        st["eint"][idx27] += np.where(alive, dE, 0.0)

    # Nodal force scatter
    fvec = F[:, None] * e1
    if fint is not None:
        np.add.at(fint, n1, fvec)
        np.add.at(fint, n2, -fvec)

    # Critical time step (r27def3.F lines 312-315)
    mass = np.maximum(st["t27_mass"][idx27], EM20) * xl0
    k_dt = np.maximum(k_eff, EM20) / xl0
    c_dt = c_eff / xl0
    pos_k = k_dt > 0.0
    omega = 2.0 * np.sqrt(np.where(pos_k, k_dt / mass, 1.0))
    xi = np.where(pos_k, c_dt / np.sqrt(np.maximum(k_dt * mass, EM20)), 0.0)
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    return np.where(alive, dt_crit, EP30)


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

        # 6-DOF rotational & bending states (Fortran ruser44.F)
        st["t44_k44"] = np.zeros(n)
        st["t44_k55"] = np.zeros(n)
        st["t44_k66"] = np.zeros(n)
        st["t44_k5b"] = np.zeros(n)
        st["t44_k6c"] = np.zeros(n)
        st["t44_c_rot"] = np.zeros(n)
        st["t44_c_by"] = np.zeros(n)
        st["t44_c_bz"] = np.zeros(n)
        st["t44_inertia"] = np.zeros(n)

        st["t44_rot_x"] = np.zeros(n)
        st["t44_rot_y1"] = np.zeros(n)
        st["t44_rot_y2"] = np.zeros(n)
        st["t44_rot_z1"] = np.zeros(n)
        st["t44_rot_z2"] = np.zeros(n)

        st["t44_fun_a1"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_b1"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_a2"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_b2"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_a3"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_b3"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_a4"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_b4"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_a5"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_b5"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_a6"] = np.zeros(n, dtype=np.int64)
        st["t44_fun_b6"] = np.zeros(n, dtype=np.int64)

        st["t44_fscale11"] = np.ones(n)
        st["t44_fscale22"] = np.ones(n)
        st["t44_fscale33"] = np.ones(n)

    pos = {int(e): j for j, e in enumerate(idx44)}
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 0) != 44:
            continue
        rng = np.arange(group.n)[sl]
        local = [e for e in rng if int(e) in pos]
        if not len(local):
            continue
        p = getattr(prop, "params", {}) or {}
        p44_obj = None
        if hasattr(model, "prop_type44s"):
            p44_obj = model.prop_type44s.get(getattr(prop, "id", 0))

        # Stiffness: k_unload, k11, k, or stiff1
        k_un = _safe_param(p, "k_unload", _safe_param(p, "k11", _safe_param(p, "k", _safe_param(p, "stiff1", 0.0))))
        fy = _safe_param(p, "f_yield", _safe_param(p, "f_max", 0.0))
        fy_t = _safe_param(p, "f_yield_tensile", 0.0)
        d_crush = _safe_param(p, "delta_crush_max", _safe_param(p, "delta_crush", 1e20))
        c = _safe_param(p, "c", _safe_param(p, "cdamp", _safe_param(p, "dscale_x", 0.0)))
        ms = _safe_param(p, "mass", getattr(p44_obj, "mass", 0.0) if p44_obj else 0.0)
        fct_y = _safe_int_param(p, "fun_b1", _safe_int_param(p, "fct_yield", 0))

        # 6-DOF bending and torsion parameters
        k44 = _safe_param(p, "k44", _safe_param(p, "k_torsion", getattr(p44_obj, "k44", 0.0) if p44_obj else 0.0))
        k55 = _safe_param(p, "k55", _safe_param(p, "k_bend_y", getattr(p44_obj, "k55", 0.0) if p44_obj else 0.0))
        k66 = _safe_param(p, "k66", _safe_param(p, "k_bend_z", getattr(p44_obj, "k66", 0.0) if p44_obj else 0.0))
        k5b = _safe_param(p, "k5b", getattr(p44_obj, "k5b", 0.0) if p44_obj else 0.0)
        k6c = _safe_param(p, "k6c", getattr(p44_obj, "k6c", 0.0) if p44_obj else 0.0)

        c_rot = _safe_param(p, "dscale_xx", _safe_param(p, "c_rot", 0.0))
        c_by = _safe_param(p, "dscale_yy", _safe_param(p, "c_bend_y", 0.0))
        c_bz = _safe_param(p, "dscale_zz", _safe_param(p, "c_bend_z", 0.0))
        iner = _safe_param(p, "inertia", getattr(p44_obj, "inertia", 0.0) if p44_obj else 0.0)

        fun_a1 = _safe_int_param(p, "fun_a1", getattr(p44_obj, "fun_a1", 0) if p44_obj else 0)
        fun_b1 = _safe_int_param(p, "fun_b1", _safe_int_param(p, "fct_yield", getattr(p44_obj, "fun_b1", 0) if p44_obj else 0))
        fun_a2 = _safe_int_param(p, "fun_a2", getattr(p44_obj, "fun_a2", 0) if p44_obj else 0)
        fun_b2 = _safe_int_param(p, "fun_b2", getattr(p44_obj, "fun_b2", 0) if p44_obj else 0)
        fun_a3 = _safe_int_param(p, "fun_a3", getattr(p44_obj, "fun_a3", 0) if p44_obj else 0)
        fun_b3 = _safe_int_param(p, "fun_b3", getattr(p44_obj, "fun_b3", 0) if p44_obj else 0)
        fun_a4 = _safe_int_param(p, "fun_a4", getattr(p44_obj, "fun_a4", 0) if p44_obj else 0)
        fun_b4 = _safe_int_param(p, "fun_b4", getattr(p44_obj, "fun_b4", 0) if p44_obj else 0)
        fun_a5 = _safe_int_param(p, "fun_a5", getattr(p44_obj, "fun_a5", 0) if p44_obj else 0)
        fun_b5 = _safe_int_param(p, "fun_b5", getattr(p44_obj, "fun_b5", 0) if p44_obj else 0)
        fun_a6 = _safe_int_param(p, "fun_a6", getattr(p44_obj, "fun_a6", 0) if p44_obj else 0)
        fun_b6 = _safe_int_param(p, "fun_b6", getattr(p44_obj, "fun_b6", 0) if p44_obj else 0)

        fscale11 = _safe_param(p, "fscale11", _safe_param(p, "fscal_x", getattr(p44_obj, "fscale11", 1.0) if p44_obj else 1.0))
        fscale22 = _safe_param(p, "fscale22", _safe_param(p, "fscal_rx", getattr(p44_obj, "fscale22", 1.0) if p44_obj else 1.0))
        fscale33 = _safe_param(p, "fscale33", getattr(p44_obj, "fscale33", 1.0) if p44_obj else 1.0)

        st["t44_k_unload"][local] = k_un
        st["t44_f_yield"][local] = fy
        st["t44_f_yield_tensile"][local] = fy_t
        st["t44_delta_crush_max"][local] = d_crush
        st["t44_cdamp"][local] = c
        st["t44_mass"][local] = ms
        st["t44_fct_yield"][local] = fct_y

        st["t44_k44"][local] = k44
        st["t44_k55"][local] = k55
        st["t44_k66"][local] = k66
        st["t44_k5b"][local] = k5b
        st["t44_k6c"][local] = k6c
        st["t44_c_rot"][local] = c_rot
        st["t44_c_by"][local] = c_by
        st["t44_c_bz"][local] = c_bz
        st["t44_inertia"][local] = iner

        st["t44_fun_a1"][local] = fun_a1
        st["t44_fun_b1"][local] = fun_b1
        st["t44_fun_a2"][local] = fun_a2
        st["t44_fun_b2"][local] = fun_b2
        st["t44_fun_a3"][local] = fun_a3
        st["t44_fun_b3"][local] = fun_b3
        st["t44_fun_a4"][local] = fun_a4
        st["t44_fun_b4"][local] = fun_b4
        st["t44_fun_a5"][local] = fun_a5
        st["t44_fun_b5"][local] = fun_b5
        st["t44_fun_a6"][local] = fun_a6
        st["t44_fun_b6"][local] = fun_b6

        st["t44_fscale11"][local] = fscale11
        st["t44_fscale22"][local] = fscale22
        st["t44_fscale33"][local] = fscale33

        st["k"][local] = k_un
        st["cdamp"][local] = c
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


def forces_crushing_type44(group, x, v, dt, fint, idx44, vr=None, mint=None):
    """Compute forces and moments for TYPE44 crushing springs with 6-DOF support (Fortran ruser44.F).

    Features:
      - 1D axial plastic crushing in compression with plateau and elastic unloading.
      - 6-DOF bending & torsion stiffnesses (K44, K55, K66, K5b, K6c).
      - Multi-DOF yield curves (torsion, bending about Y and Z at nodes 1 and 2).
      - Equilibrium transverse shear forces derived from bending moments:
        F_z = (M_y1 + M_y2) / L, F_y = -(M_z1 + M_z2) / L.
      - Exact linear and angular momentum conservation: sum(F) = 0, sum(M) + r x F = 0.
      - Damping and energy dissipation booking into eint.
    """
    if idx44 is None or len(idx44) == 0:
        return np.empty(0)
    st = group.state
    conn = group.conn[idx44]
    n1, n2 = conn[:, 0], conn[:, 1]
    model = st.get("model")
    n_elem = len(idx44)

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
    has_funcs = (model is not None and hasattr(model, "functions"))
    if has_funcs:
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

    # Apply axial forces to nodes: tension pulls together, compression pushes apart
    fvec = F[:, None] * a
    if fint is not None:
        np.add.at(fint, n1, fvec)
        np.add.at(fint, n2, -fvec)

    # Internal energy work booking for axial force
    if dt_val > 0.0:
        st["eint"][idx44] += 0.5 * (F_old + F) * Ldot * dt_val

    # ========================================================================
    # 6-DOF Rotational & Bending Moments (Fortran ruser44.F)
    # ========================================================================
    k44 = st.get("t44_k44", np.zeros(group.n))[idx44]
    k55 = st.get("t44_k55", np.zeros(group.n))[idx44]
    k66 = st.get("t44_k66", np.zeros(group.n))[idx44]
    has_rot = np.any((k44 > 0.0) | (k55 > 0.0) | (k66 > 0.0))

    if has_rot and vr is not None:
        # Construct orthogonal local coordinate triad (e1, e2, e3)
        e1 = a
        ref = np.array([0.0, 1.0, 0.0])
        dot_y = np.abs(np.einsum("nb,b->n", e1, ref))
        par = dot_y > 0.99
        ref_vecs = np.where(par[:, None], np.array([0.0, 0.0, 1.0]), ref)
        e3_raw = np.cross(e1, ref_vecs)
        norm_e3 = norm3(e3_raw)
        e3 = np.where((norm_e3 > EM20)[:, None], e3_raw / np.maximum(norm_e3[:, None], EM20), np.array([0.0, 0.0, 1.0]))
        e2 = np.cross(e3, e1)

        omega1 = vr[n1]
        omega2 = vr[n2]
        d_omega = omega2 - omega1

        # Relative rotational rates in local axes
        rx_rate = np.einsum("nb,nb->n", d_omega, e1)
        ry1_rate = np.einsum("nb,nb->n", omega1, e2)
        ry2_rate = np.einsum("nb,nb->n", omega2, e2)
        rz1_rate = np.einsum("nb,nb->n", omega1, e3)
        rz2_rate = np.einsum("nb,nb->n", omega2, e3)

        if dt_val > 0.0:
            st["t44_rot_x"][idx44] += rx_rate * dt_val
            st["t44_rot_y1"][idx44] += ry1_rate * dt_val
            st["t44_rot_y2"][idx44] += ry2_rate * dt_val
            st["t44_rot_z1"][idx44] += rz1_rate * dt_val
            st["t44_rot_z2"][idx44] += rz2_rate * dt_val

        rot_x = st["t44_rot_x"][idx44]
        rot_y1 = st["t44_rot_y1"][idx44]
        rot_y2 = st["t44_rot_y2"][idx44]
        rot_z1 = st["t44_rot_z1"][idx44]
        rot_z2 = st["t44_rot_z2"][idx44]

        k5b = st["t44_k5b"][idx44]
        k6c = st["t44_k6c"][idx44]
        c_rot = st["t44_c_rot"][idx44]
        c_by = st["t44_c_by"][idx44]
        c_bz = st["t44_c_bz"][idx44]

        # Elastic moments + damping
        xmom = k44 * rot_x + c_rot * rx_rate
        my1 = (k55 * rot_y1 + k5b * rot_y2) + c_by * ry1_rate
        my2 = (k5b * rot_y1 + k55 * rot_y2) + c_by * ry2_rate
        mz1 = (k66 * rot_z1 + k6c * rot_z2) + c_bz * rz1_rate
        mz2 = (k6c * rot_z1 + k66 * rot_z2) + c_bz * rz2_rate

        # Clamping to yield curves if specified
        if has_funcs:
            for i in range(n_elem):
                sc_rx = st["t44_fscale22"][idx44[i]]
                sc_bend = st["t44_fscale33"][idx44[i]]

                # Torsion yield: fun_b2 (+), fun_a2 (-)
                fb2 = st["t44_fun_b2"][idx44[i]]
                fa2 = st["t44_fun_a2"][idx44[i]]
                if fb2 > 0 and fb2 in model.functions:
                    fxxp = sc_rx * abs(model.functions[fb2].eval(rot_x[i]))
                    xmom[i] = min(xmom[i], fxxp)
                if fa2 > 0 and fa2 in model.functions:
                    fxxm = -sc_rx * abs(model.functions[fa2].eval(rot_x[i]))
                    xmom[i] = max(xmom[i], fxxm)

                # Bending Y1: fun_b3 (+), fun_a3 (-)
                fb3 = st["t44_fun_b3"][idx44[i]]
                fa3 = st["t44_fun_a3"][idx44[i]]
                if fb3 > 0 and fb3 in model.functions:
                    fyy1p = sc_bend * abs(model.functions[fb3].eval(rot_y1[i]))
                    my1[i] = min(my1[i], fyy1p)
                if fa3 > 0 and fa3 in model.functions:
                    fyy1m = -sc_bend * abs(model.functions[fa3].eval(rot_y1[i]))
                    my1[i] = max(my1[i], fyy1m)

                # Bending Y2: fun_b5 (+), fun_a5 (-)
                fb5 = st["t44_fun_b5"][idx44[i]]
                fa5 = st["t44_fun_a5"][idx44[i]]
                if fb5 > 0 and fb5 in model.functions:
                    fyy2p = sc_bend * abs(model.functions[fb5].eval(rot_y2[i]))
                    my2[i] = min(my2[i], fyy2p)
                if fa5 > 0 and fa5 in model.functions:
                    fyy2m = -sc_bend * abs(model.functions[fa5].eval(rot_y2[i]))
                    my2[i] = max(my2[i], fyy2m)

                # Bending Z1: fun_b4 (+), fun_a4 (-)
                fb4 = st["t44_fun_b4"][idx44[i]]
                fa4 = st["t44_fun_a4"][idx44[i]]
                if fb4 > 0 and fb4 in model.functions:
                    fzz1p = sc_bend * abs(model.functions[fb4].eval(rot_z1[i]))
                    mz1[i] = min(mz1[i], fzz1p)
                if fa4 > 0 and fa4 in model.functions:
                    fzz1m = -sc_bend * abs(model.functions[fa4].eval(rot_z1[i]))
                    mz1[i] = max(mz1[i], fzz1m)

                # Bending Z2: fun_b6 (+), fun_a6 (-)
                fb6 = st["t44_fun_b6"][idx44[i]]
                fa6 = st["t44_fun_a6"][idx44[i]]
                if fb6 > 0 and fb6 in model.functions:
                    fzz2p = sc_bend * abs(model.functions[fb6].eval(rot_z2[i]))
                    mz2[i] = min(mz2[i], fzz2p)
                if fa6 > 0 and fa6 in model.functions:
                    fzz2m = -sc_bend * abs(model.functions[fa6].eval(rot_z2[i]))
                    mz2[i] = max(mz2[i], fzz2m)

        # Transverse shear forces from bending moment equilibrium (ruser44.F lines 556-557)
        inv_L = 1.0 / np.maximum(L, EM20)
        fz_shear = (my1 + my2) * inv_L
        fy_shear = -(mz1 + mz2) * inv_L

        # Nodal moments in global coordinates
        m1_vec = xmom[:, None] * e1 + my1[:, None] * e2 + mz1[:, None] * e3
        m2_vec = -xmom[:, None] * e1 + my2[:, None] * e2 + mz2[:, None] * e3

        if mint is not None:
            np.add.at(mint, n1, m1_vec)
            np.add.at(mint, n2, m2_vec)

        # Transverse forces applied to nodes:
        # Moment equilibrium about node 1: M1 + M2 + (x2 - x1) x F2 = 0
        # (x2 - x1) x F2 = L e1 x (fy*e2 + fz*e3) = L*(fy*e3 - fz*e2)
        # So for M1_y + M2_y - L*fz = 0, F2 must have +fz along e3.
        # And M1_z + M2_z + L*fy = 0, F2 must have +fy along e2.
        # Thus F2 = fy*e2 + fz*e3, and F1 = -F2 = -(fy*e2 + fz*e3)
        f_shear_vec = fy_shear[:, None] * e2 + fz_shear[:, None] * e3
        if fint is not None:
            np.add.at(fint, n1, -f_shear_vec)
            np.add.at(fint, n2, f_shear_vec)

        # Rotational work booking into eint
        if dt_val > 0.0:
            p_rot = xmom * rx_rate + my1 * ry1_rate + my2 * ry2_rate + mz1 * rz1_rate + mz2 * rz2_rate
            st["eint"][idx44] += p_rot * dt_val

    # Time step calculation
    mass = np.maximum(st["t44_mass"][idx44], EM20)
    k_dt = np.maximum(k_unload, 0.0)
    pos_k = (k_dt > 0.0) & (st["t44_mass"][idx44] > 0.0)
    pure_c = (k_dt <= 0.0) & (c_damp > 0.0) & (st["t44_mass"][idx44] > 0.0)

    omega = 2.0 * np.sqrt(np.where(pos_k, k_dt / mass, 1.0))
    xi = np.where(pos_k, c_damp / np.sqrt(np.maximum(k_dt * mass, EM20)), 0.0)
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    dt_c = np.where(pure_c, 0.5 * mass / np.maximum(c_damp, EM20), EP30)
    dt_final = np.where(pos_k, dt_crit, dt_c)

    # Rotational time step check if rotational stiffness and inertia are specified
    iner = st.get("t44_inertia", np.zeros(group.n))[idx44]
    k_rot_max = np.maximum(k44, np.maximum(k55, k66))
    c_rot_max = np.maximum(st.get("t44_c_rot", np.zeros(group.n))[idx44], st.get("t44_c_by", np.zeros(group.n))[idx44])
    has_iner = (iner > 0.0) & (k_rot_max > 0.0)
    if np.any(has_iner):
        i_p = iner[has_iner]
        k_p = k_rot_max[has_iner]
        c_p = c_rot_max[has_iner]
        denom = np.sqrt(c_p * c_p + i_p * k_p) + c_p
        dt_rot = i_p / np.maximum(denom, EM20)
        dt_final[has_iner] = np.minimum(dt_final[has_iner], dt_rot)

    return dt_final


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

def init_advanced(group, model, log, *pos_args, idx19=None, idx25=None, idx26=None, idx27=None, idx44=None, idx46=None, massn=None, inertn=None, **kwargs):
    """Dispatcher called by spring.py init_group for advanced spring types."""
    if len(pos_args) == 5:
        idx19, idx44, idx46, massn, inertn = pos_args
    elif len(pos_args) == 6:
        idx19, idx25, idx44, idx46, massn, inertn = pos_args
    elif len(pos_args) == 7:
        idx19, idx25, idx26, idx44, idx46, massn, inertn = pos_args
    elif len(pos_args) == 8:
        idx19, idx25, idx26, idx27, idx44, idx46, massn, inertn = pos_args

    if idx19 is not None and len(idx19):
        init_torsion_type19(group, model, log, idx19, massn, inertn)
    if idx25 is not None and len(idx25):
        init_axi_type25(group, model, log, idx25, massn, inertn)
    if idx26 is not None and len(idx26):
        init_tab_type26(group, model, log, idx26, massn, inertn)
    if idx27 is not None and len(idx27):
        init_bdamp_type27(group, model, log, idx27, massn, inertn)
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


# ----------------------------------------------------------------------------
# Implicit element matrices: tangent, kgeo, consistent_mass (M614 Component 1B)
# ----------------------------------------------------------------------------
# Fortran origin:
#   engine/source/elements/spring/rforc3.F (driver)
#   engine/source/elements/spring/r1tors.F (torsion)
#   engine/source/elements/spring/ruser44.F (crushing)
#   engine/source/elements/spring/ruser46.F (muscle)
#   starter/source/elements/spring/rmass3.F (mass)
#   engine/source/implicit/assem_r3.F (implicit assembly for springs)
# ----------------------------------------------------------------------------

def _spring_axis(group, x):
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = norm3(dx)
    degen = (L < EM20)
    L = np.where(degen, EM20, L)
    a = np.where(degen[:, None], np.array([1.0, 0.0, 0.0]), dx / L[:, None])
    return conn, L, a


def _spring_edofs(conn):
    if len(conn) == 0:
        return np.empty((0, 6), dtype=np.int64)
    edofs = np.empty((len(conn), 6), dtype=np.int64)
    for c in range(3):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 3 + c] = conn[:, 1] * 6 + c
    return edofs


def _spring_edofs12(conn):
    if len(conn) == 0:
        return np.empty((0, 12), dtype=np.int64)
    edofs = np.empty((len(conn), 12), dtype=np.int64)
    for c in range(6):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 6 + c] = conn[:, 1] * 6 + c
    return edofs


def local_stiffness(group, x=None):
    """Compute local axial and torsional stiffness coefficients for advanced springs."""
    st = group.state
    n = group.n
    k_ax = np.zeros(n, dtype=np.float64)
    k_tor = np.zeros(n, dtype=np.float64)

    if "k" in st:
        k_ax[:] = st["k"]
    if "t19_k_theta" in st:
        k_tor[:] = st["t19_k_theta"]

    conn = group.conn
    L0 = st.get("L0")
    if L0 is None and x is not None and len(conn):
        dx = x[conn[:, 1]] - x[conn[:, 0]]
        L0 = np.maximum(norm3(dx), EM20)
    elif L0 is None:
        L0 = np.ones(n)

    L = L0
    if x is not None and len(conn):
        dx = x[conn[:, 1]] - x[conn[:, 0]]
        L = np.maximum(norm3(dx), EM20)
    dl = L - L0

    for sl, mat, prop in st.get("slices", []):
        ptype = getattr(prop, "type", 0) or getattr(prop, "prop_type", 0)
        pname = type(prop).__name__.upper()
        if ptype == 0:
            if "26" in pname or "SPR_TAB" in pname:
                ptype = 26
            elif "27" in pname or "BDAMP" in pname:
                ptype = 27
            elif "19" in pname or "TORS" in pname:
                ptype = 19
            elif "44" in pname or "CRUS" in pname:
                ptype = 44
            elif "46" in pname or "MUSCLE" in pname:
                ptype = 46
            elif "MAT" in pname or "23" in pname:
                ptype = 23
        p = getattr(prop, "params", {}) or {}

        if ptype == 26:  # /PROP/TYPE26, /PROP/SPR_TAB
            kmax = float(getattr(prop, "kmax", 0.0) or p.get("kmax", 0.0) or p.get("k", 0.0))
            curves = getattr(prop, "loading_curves", [])
            evaluated_k = None
            if curves and hasattr(curves[0], "eval"):
                u_curr = dl[sl]
                eps_u = 1e-5
                try:
                    f_plus = curves[0].eval(u_curr + eps_u)
                    f_minus = curves[0].eval(u_curr - eps_u)
                    evaluated_k = (f_plus - f_minus) / (2.0 * eps_u)
                except Exception:
                    pass
            k_val = evaluated_k if evaluated_k is not None and np.all(np.isfinite(evaluated_k)) else kmax
            if k_val == 0.0:
                k_val = float(p.get("k", 1.0))
            k_ax[sl] = k_val

        elif ptype == 27:  # /PROP/TYPE27, /PROP/SPR_BDAMP
            stiff = float(getattr(prop, "stiff", 0.0) or p.get("stiff", 0.0) or p.get("k", 0.0))
            if stiff == 0.0:
                stiff = float(p.get("k", 1.0))
            k_ax[sl] = stiff

        elif ptype == 23 or getattr(prop, "prop_name", "") == "SPR_MAT":  # /PROP/SPR_MAT
            E = float(getattr(mat, "E", 0.0) or 0.0)
            area = float(p.get("area", 1.0))
            L_sl = np.maximum(L0[sl], EM20)
            k_ax[sl] = E * area / L_sl if E > 0.0 else float(p.get("k", 1.0))

        elif ptype == 19:  # /PROP/TYPE19 (SPR_TORS)
            kt = float(p.get("k_theta", p.get("k", 0.0)))
            k_tor[sl] = kt

        elif ptype == 44:  # /PROP/TYPE44 (SPR_CRUS)
            k_un = float(p.get("k_unload", p.get("k", 0.0)))
            k_ax[sl] = k_un

        elif ptype == 46:  # /PROP/TYPE46 (SPR_MUSCLE)
            k_pe = float(p.get("k_pe", 0.0))
            f_max = float(p.get("f_max", 0.0))
            k_ax[sl] = max(k_pe, f_max / max(float(L0[sl][0] if len(sl) else 1.0), EM20))

    return k_ax, k_tor


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for advanced springs.

    Fortran origin: engine/source/elements/spring/rforc3.F, r1tors.F,
    ruser44.F, ruser46.F; engine/source/implicit/assem_r3.F.

    Returns (ke, edofs):
      ke: (n, 6, 6) or (n, 12, 12) global stiffness matrix
      edofs: (n, 6) or (n, 12)
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)

    conn, L, a = _spring_axis(group, x)
    k_ax, k_tor = local_stiffness(group, x)

    has_rot = np.any(k_tor > 0.0)

    if has_rot:
        edofs = _spring_edofs12(conn)
        ke = np.zeros((n, 12, 12), dtype=np.float64)
        kb_ax = k_ax[:, None, None] * np.einsum("ni,nj->nij", a, a)
        ke[:, 0:3, 0:3] = kb_ax
        ke[:, 6:9, 6:9] = kb_ax
        ke[:, 0:3, 6:9] = -kb_ax
        ke[:, 6:9, 0:3] = -kb_ax

        kb_tor = k_tor[:, None, None] * np.einsum("ni,nj->nij", a, a)
        ke[:, 3:6, 3:6] = kb_tor
        ke[:, 9:12, 9:12] = kb_tor
        ke[:, 3:6, 9:12] = -kb_tor
        ke[:, 9:12, 3:6] = -kb_tor
    else:
        edofs = _spring_edofs(conn)
        ke = np.zeros((n, 6, 6), dtype=np.float64)
        kb_ax = k_ax[:, None, None] * np.einsum("ni,nj->nij", a, a)
        ke[:, :3, :3] = kb_ax
        ke[:, 3:, 3:] = kb_ax
        ke[:, :3, 3:] = -kb_ax
        ke[:, 3:, :3] = -kb_ax

    dead = (st.get("off", np.ones(n)) <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, edofs


def kgeo(group, x):
    """Geometric (initial-stress) stiffness (F/L)(I - a a^T) from current spring force.

    Fortran origin: engine/source/elements/spring/rforc3.F,
    engine/source/implicit/assem_r3.F.

    Returns (ke, edofs): ke (n, 6, 6) or (n, 12, 12), edofs (n, 6) or (n, 12).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)

    conn, L, a = _spring_axis(group, x)
    force = st.get("force", np.zeros(n))
    F_over_L = force / L
    eye = np.eye(3)
    kb = F_over_L[:, None, None] * (eye[None, :, :] - np.einsum("ni,nj->nij", a, a))

    k_tor = st.get("t19_k_theta", np.zeros(n))
    has_rot = np.any(k_tor > 0.0)

    if has_rot:
        edofs = _spring_edofs12(conn)
        ke = np.zeros((n, 12, 12), dtype=np.float64)
        ke[:, 0:3, 0:3] = kb
        ke[:, 6:9, 6:9] = kb
        ke[:, 0:3, 6:9] = -kb
        ke[:, 6:9, 0:3] = -kb
    else:
        edofs = _spring_edofs(conn)
        ke = np.zeros((n, 6, 6), dtype=np.float64)
        ke[:, :3, :3] = kb
        ke[:, 3:, 3:] = kb
        ke[:, :3, 3:] = -kb
        ke[:, 3:, :3] = -kb

    dead = (st.get("off", np.ones(n)) <= 0.0)
    if np.any(dead):
        ke[dead] = 0.0

    return ke, edofs


def consistent_mass(group, x=None):
    """Lumped / exact element mass of the advanced spring: M/2 on each node's translations.

    Fortran origin: starter/source/elements/spring/rmass3.F,
    engine/source/implicit/assem_r3.F.

    Returns (me, edofs): me (n, 6, 6) or (n, 12, 12), edofs (n, 6) or (n, 12).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 6, 6)), np.empty((0, 6), dtype=np.int64)

    m = st.get("mass", np.zeros(n))
    half_m = m / 2.0

    k_tor = st.get("t19_k_theta", np.zeros(n))
    has_rot = np.any(k_tor > 0.0)

    if has_rot:
        edofs = _spring_edofs12(conn)
        me = np.zeros((n, 12, 12), dtype=np.float64)
        for i in range(3):
            me[:, i, i] = half_m
            me[:, 6 + i, 6 + i] = half_m
        iner = st.get("t19_inertia", np.zeros(n))
        half_iner = iner / 2.0
        for i in range(3):
            me[:, 3 + i, 3 + i] = half_iner
            me[:, 9 + i, 9 + i] = half_iner
    else:
        edofs = _spring_edofs(conn)
        me = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(6):
            me[:, i, i] = half_m

    dead = (st.get("off", np.ones(n)) <= 0.0)
    if np.any(dead):
        me[dead] = 0.0

    return me, edofs

