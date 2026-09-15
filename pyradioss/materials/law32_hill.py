"""
LAW32 — Hill's Orthotropic Anisotropic Plasticity for Shell Elements (/MAT/LAW32, /MAT/HILL).

Fortran origin:
  - starter/source/materials/mat/mat032/hm_read_mat32.F (starter parser & Hill coefficients)
  - engine/source/materials/mat/mat032/sigeps32c.F (shell wrapper & thickness strain)
  - engine/source/materials/mat/mat032/m32elas.F (elastic trial stresses)
  - engine/source/materials/mat/mat032/m32plas.F (plastic return mapping: IPLA=0, 1, 2)
  - engine/source/airbag/uroto.F (coordinate transformation between element & material axes)
  - hm_cfg_files/config/CFG/radioss110/MAT/matl32_hill.cfg (CFG attributes & card format)

Theory
------
Hill 1948 quadratic anisotropic yield criterion for plane-stress shell elements:

    f(S) = sigma_eq - sigma_y <= 0

where S is the in-plane Cauchy stress tensor in the orthotropic material axes:

    sigma_eq = sqrt(A11*S11^2 + A22*S22^2 - A1122*S11*S22 + A12*S12^2)

The Hill coefficients A11, A22, A1122, A12 are computed from Lankford parameters
R00, R45, R90 (measured at 0, 45, and 90 degrees to rolling direction):

    R = 0.25 * (R00 + 2*R45 + R90)
    H = R / (1.0 + R)
    A11   = H * (1.0 + 1.0 / R00)
    A22   = H * (1.0 + 1.0 / R90)
    A1122 = 2.0 * H
    A12   = 2.0 * H * (R45 + 0.5) * (1.0 / R00 + 1.0 / R90)

If I_yield > 0 (ir0 > 0 in hm_read_mat32.F), coefficients are normalized so that
yielding in the orthotropic direction 1 is exactly at sigma_y:

    A22   <- A22 / A11
    A1122 <- A1122 / A11
    A12   <- A12 / A11
    A11   <- 1.0

Hardening and Strain-Rate Sensitivity:
    sigma_y = min(sig_max, A * (B + eps_p)^n * (eps_dot / eps0)^m)

where eps_dot is the strain rate:
    eps_dot = max(|deps_xx|, |deps_yy|, 0.5*|deps_xy|) / dt
clamped >= eps0 (reference strain rate).

Plastic Return Algorithms:
  - IPLA == 0: Radial projection scaling (scale = min(1, sigma_y / sigma_eq)).
  - IPLA == 2: Plane-stress projection with s33 = 0 (solving quadratic in scale).
  - IPLA == 1: 3-iteration Newton-Raphson return in plane stress.

Failure:
  When accumulated plastic strain eps_p >= eps_max, element is deleted (off32 = 0, sig = 0).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_INF = 1.0e30


# ============================================================================
# Parameter Extraction & Constructor
# ============================================================================

def build_law32(rec: Any = None, **kwargs: Any) -> Material:
    """Physics constructor for the cfg-parsed /MAT/LAW32 or /MAT/HILL record.

    Follows starter/source/materials/mat/mat032/hm_read_mat32.F.
    """
    if rec is None:
        p: Dict[str, Any] = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", kwargs.get("density", kwargs.get("MAT_RHO", 0.0)))
        _title = str(kwargs.get("title", "LAW32_HILL"))
    elif isinstance(rec, dict):
        base_params = rec.get("params", rec)
        p = {**base_params, **kwargs}
        _id = int(rec.get("id", kwargs.get("id", 1)))
        rho0_in = rec.get("rho0", rec.get("density", kwargs.get("rho0", kwargs.get("density", 0.0))))
        _title = str(rec.get("title", kwargs.get("title", "LAW32_HILL")))
    elif hasattr(rec, "params"):
        p = {**rec.params, **kwargs}
        _id = int(getattr(rec, "id", kwargs.get("id", 1)))
        rho0_in = getattr(rec, "rho0", getattr(rec, "density", kwargs.get("rho0", 0.0)))
        _title = str(getattr(rec, "title", kwargs.get("title", "LAW32_HILL")))
    else:
        p = dict(kwargs)
        _id = int(kwargs.get("id", 1))
        rho0_in = kwargs.get("rho0", 0.0)
        _title = str(kwargs.get("title", "LAW32_HILL"))

    def _get(keys: Sequence[str], default: float = 0.0) -> float:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return float(p[k])
                except (ValueError, TypeError):
                    pass
        return float(default)

    def _geti(keys: Sequence[str], default: int = 0) -> int:
        for k in keys:
            if k in p and p[k] is not None:
                try:
                    return int(p[k])
                except (ValueError, TypeError):
                    pass
        return int(default)

    rho0 = _get(["MAT_RHO", "Refer_Rho", "rho0", "density", "rho"], float(rho0_in))
    e = _get(["MAT_E", "E", "e", "young", "Young"], 0.0)
    if e <= 0.0:
        raise ValueError(f"/MAT/LAW32/{_id}: Young's modulus E must be > 0.")

    nu = _get(["MAT_NU", "NU", "nu", "anu"], 0.0)
    if nu >= 0.5:
        nu = 0.499  # hm_read_mat32.F: IF(ANU==HALF) ANU=ZEP499
    if nu < 0.0:
        raise ValueError(f"/MAT/LAW32/{_id}: Poisson's ratio nu must be >= 0.")

    # Yield parameter A (MAT_SIGY, default 1e30 if 0)
    a = _get(["MAT_SIGY", "A", "a", "sigy", "sig_y", "ca", "MAT_SIG_Y"], _INF)
    if a == 0.0:
        a = _INF

    # Hardening parameter B (MAT_BETA, default 0.0)
    b = _get(["MAT_BETA", "B", "b", "beta", "ce", "epsilon_0", "EPSILON_0", "EPS0"], 0.0)

    # Hardening exponent n (MAT_HARD, default 1.0 if 0, error if > 1.0)
    n = _get(["MAT_HARD", "N", "n", "hard", "cn"], 1.0)
    if n == 0.0:
        n = 1.0
    if n > 1.0:
        raise ValueError(f"/MAT/LAW32/{_id}: Hardening exponent n={n:g} must be <= 1.0 (upstream error 213).")

    # Failure plastic strain (MAT_EPS, default 1e30 if 0)
    eps_max = _get(["MAT_EPS", "EPS", "eps", "eps_max", "epsm", "eps_p_max"], _INF)
    if eps_max == 0.0:
        eps_max = _INF

    # Maximum stress (MAT_SIG, default 1e30 if 0)
    sig_max = _get(["MAT_SIG", "SIG", "sig", "sig_max", "sigm"], _INF)
    if sig_max == 0.0:
        sig_max = _INF

    # Strain rate exponent (MAT_SRC, default 0.0)
    m = _get(["MAT_SRC", "SRC", "src", "m", "cm"], 0.0)

    # Reference strain rate (MAT_SRP, hm_read_mat32.F: IF(CM==ZERO) EPS0=ONE)
    eps0 = _get(["MAT_SRP", "SRP", "srp", "eps0", "eps_dot_0", "eps_min"], 1.0 if m == 0.0 else 0.0)
    if m == 0.0:
        eps0 = 1.0
    if eps0 <= 0.0:
        raise ValueError(f"/MAT/LAW32/{_id}: Reference strain rate EPS0={eps0:g} must be > 0 (upstream error 207).")

    # Lankford parameters R00, R45, R90 (default 1.0 if 0)
    r00 = _get(["MAT_R00", "R00", "r00", "r_00"], 1.0)
    if r00 <= 0.0:
        r00 = 1.0
    r45 = _get(["MAT_R45", "R45", "r45", "r_45"], 1.0)
    if r45 <= 0.0:
        r45 = 1.0
    r90 = _get(["MAT_R90", "R90", "r90", "r_90"], 1.0)
    if r90 <= 0.0:
        r90 = 1.0

    # Yield stress normalization flag (MAT_Iyield, default 0)
    i_yield = _geti(["MAT_Iyield", "Iyield", "iyield", "i_yield", "ir0"], 0)

    # Return mapping algorithm IPLA (default 0)
    ipla = _geti(["IPLA", "ipla", "Iplas", "iplas"], 0)

    # Compute Hill 1948 anisotropy parameters (hm_read_mat32.F lines 157-168)
    r = 0.25 * (r00 + 2.0 * r45 + r90)
    h = r / (1.0 + r)
    a11 = h * (1.0 + 1.0 / r00)
    a22 = h * (1.0 + 1.0 / r90)
    a1122 = 2.0 * h
    a12 = 2.0 * h * (r45 + 0.5) * (1.0 / r00 + 1.0 / r90)

    if i_yield > 0:
        a22 = a22 / a11
        a1122 = a1122 / a11
        a12 = a12 / a11
        a11 = 1.0

    # Elastic plane stress moduli
    a1 = e / (1.0 - nu ** 2)
    a2 = nu * a1
    g = e / (2.0 * (1.0 + nu))
    k = e / (3.0 * (1.0 - 2.0 * nu))
    c_sound = math.sqrt(e / max(rho0, _EM20)) if rho0 > 0.0 else math.sqrt(e)

    params: Dict[str, Any] = {
        "E": e,
        "nu": nu,
        "G": g,
        "K": k,
        "A1": a1,
        "A2": a2,
        "A": a,
        "B": b,
        "n": n,
        "eps_max": eps_max,
        "sig_max": sig_max,
        "eps0": eps0,
        "m": m,
        "r00": r00,
        "r45": r45,
        "r90": r90,
        "i_yield": i_yield,
        "ipla": ipla,
        "A11": a11,
        "A22": a22,
        "A1122": a1122,
        "A12": a12,
        "c": c_sound,
        "eps_p_max": eps_max,
    }

    return Material(
        id=_id,
        law=32,
        rho0=rho0,
        title=_title,
        law_name="LAW32",
        params=params,
    )


# ============================================================================
# Sound Speed & Kinematics
# ============================================================================

def sound_speed(mat: Material, rho: Optional[float] = None, extra: Any = None) -> float:
    """Sound speed SDSP = sqrt(YOUNG / RHO0) matching hm_read_mat32.F:155."""
    rho_val = float(rho) if rho is not None else float(mat.rho0)
    e = float(mat.params.get("E", mat.E))
    return float(math.sqrt(e / max(rho_val, _EM20)))


def extra_shapes(mat: Material, nip: int = 1) -> Dict[str, Tuple[int, ...]]:
    """Allocations required in element layer state."""
    return {
        "uv32": (nip, 1),   # accumulated plastic strain [:, 0]
        "off32": (nip,),    # element active flag (1=active, 0=failed)
    }


def solid_update(*args: Any, **kwargs: Any) -> None:
    """LAW32 (/MAT/LAW32 / /MAT/HILL) is implemented for shell elements only."""
    raise NotImplementedError("LAW32 (Hill orthotropic plasticity) is implemented for shell elements only.")


# ============================================================================
# Coordinate Transformation (uroto.F)
# ============================================================================

def _get_dir_cosines(extra: Optional[Dict[str, Any]], n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return D11 = cos^2(theta), D22 = sin^2(theta), D12 = cos(theta)*sin(theta)."""
    if extra is not None and "dir" in extra and extra["dir"] is not None:
        d = np.asarray(extra["dir"], dtype=float)
        if d.ndim == 1:
            c = np.full(n, d[0], dtype=float)
            s = np.full(n, d[1], dtype=float)
        else:
            c = d[:, 0]
            s = d[:, 1]
    elif extra is not None and "theta" in extra and extra["theta"] is not None:
        th = np.asarray(extra["theta"], dtype=float)
        if th.ndim == 0:
            c = np.full(n, math.cos(float(th)), dtype=float)
            s = np.full(n, math.sin(float(th)), dtype=float)
        else:
            c = np.cos(th)
            s = np.sin(th)
    else:
        c = np.ones(n, dtype=float)
        s = np.zeros(n, dtype=float)

    d11 = c * c
    d22 = s * s
    d12 = c * s
    return d11, d22, d12


def _rot_elem_to_mat(sxx: np.ndarray, syy: np.ndarray, sxy: np.ndarray,
                     d11: np.ndarray, d22: np.ndarray, d12: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate in-plane stresses from element frame to material orthotropic axes."""
    s11 = d11 * sxx + d22 * syy + 2.0 * d12 * sxy
    s22 = d22 * sxx + d11 * syy - 2.0 * d12 * sxy
    s12 = d12 * (syy - sxx) + (d11 - d22) * sxy
    return s11, s22, s12


def _rot_mat_to_elem(s11: np.ndarray, s22: np.ndarray, s12: np.ndarray,
                     d11: np.ndarray, d22: np.ndarray, d12: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rotate in-plane stresses from material axes back to element frame (uroto.F)."""
    sxx = d11 * s11 + d22 * s22 - 2.0 * d12 * s12
    syy = d22 * s11 + d11 * s22 + 2.0 * d12 * s12
    sxy = d12 * (s11 - s22) + (d11 - d22) * s12
    return sxx, syy, sxy


# ============================================================================
# Shell Constitutive Update (sigeps32c.F / m32elas.F / m32plas.F)
# ============================================================================

def shell_update(mat: Material, sig: np.ndarray, deps: np.ndarray,
                 *args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Constitutive plane-stress update for shell elements under Hill plasticity.

    Flexible signature supporting:
      - shell_update(mat, sig, deps, dt, extra)
      - shell_update(mat, sig, deps, epsp, dt, extra)
      - keyword-based arguments
    """
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float)
    if not isinstance(deps, np.ndarray):
        deps = np.array(deps, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]
        deps = deps[None, :]

    n = sig.shape[0]
    if n == 0:
        return sig, np.zeros(0, dtype=float)

    # Argument disambiguation
    epsp_in: Optional[np.ndarray] = None
    dt = 0.0
    extra: Optional[Dict[str, Any]] = None

    if len(args) >= 3:
        epsp_in = args[0]
        dt = float(args[1])
        extra = args[2]
    elif len(args) == 2:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
            extra = args[1]
        elif isinstance(args[1], dict) or args[1] is None:
            dt = float(args[0]) if isinstance(args[0], (int, float)) else 0.0
            extra = args[1]
        else:
            epsp_in = args[0]
            dt = float(args[1])
    elif len(args) == 1:
        if isinstance(args[0], (int, float)) and not isinstance(args[0], np.ndarray):
            dt = float(args[0])
        else:
            epsp_in = args[0]

    if "dt" in kwargs:
        dt = float(kwargs["dt"])
    if "extra" in kwargs:
        extra = kwargs["extra"]
    if "epsp" in kwargs and kwargs["epsp"] is not None:
        epsp_in = kwargs["epsp"]

    # Parameter extraction
    p = mat.params
    e = float(p.get("E", mat.E))
    nu = float(p.get("nu", mat.nu))
    g = float(p.get("G", mat.G))
    a1 = float(p.get("A1", e / (1.0 - nu ** 2)))
    a2 = float(p.get("A2", nu * a1))

    a_yield = float(p.get("A", _INF))
    b_yield = float(p.get("B", 0.0))
    n_hard = float(p.get("n", 1.0))
    eps_max = float(p.get("eps_max", _INF))
    sig_max = float(p.get("sig_max", _INF))
    eps0 = float(p.get("eps0", 1.0))
    m_rate = float(p.get("m", 0.0))

    a11 = float(p.get("A11", 1.0))
    a22 = float(p.get("A22", 1.0))
    a1122 = float(p.get("A1122", 1.0))
    a12 = float(p.get("A12", 3.0))

    ipla = int(extra.get("ipla", p.get("ipla", 0)) if extra is not None else p.get("ipla", 0))

    # Retrieve plastic strain epsp and element active flag off
    if extra is not None and "uv32" in extra and extra["uv32"] is not None:
        uv32 = extra["uv32"]
        if uv32.ndim == 1:
            epsp = uv32.copy()
        else:
            epsp = uv32[:, 0].copy()
    elif epsp_in is not None:
        epsp = np.asarray(epsp_in, dtype=float).copy()
        if epsp.ndim == 0:
            epsp = np.full(n, float(epsp), dtype=float)
    else:
        epsp = np.zeros(n, dtype=float)

    if extra is not None and "off32" in extra and extra["off32"] is not None:
        off = np.asarray(extra["off32"], dtype=float).flatten()
    elif extra is not None and "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).flatten()
    else:
        off = np.ones(n, dtype=float)

    # 1. Elastic trial stresses (m32elas.F)
    sxx_tr = sig[:, 0] + a1 * deps[:, 0] + a2 * deps[:, 1]
    syy_tr = sig[:, 1] + a2 * deps[:, 0] + a1 * deps[:, 1]
    sxy_tr = sig[:, 2] + g * deps[:, 2]

    # Transverse shear (elastic with 5/6 factor, sigeps32c.F)
    gs = float(extra.get("gs", 5.0 / 6.0 * g) if extra is not None else 5.0 / 6.0 * g)
    if sig.shape[1] >= 5 and deps.shape[1] >= 5:
        syz_tr = sig[:, 3] + gs * deps[:, 3]
        szx_tr = sig[:, 4] + gs * deps[:, 4]
    else:
        syz_tr = None
        szx_tr = None

    # 2. Orientation & rotation to material axes
    d11, d22, d12 = _get_dir_cosines(extra, n)
    s11, s22, s12 = _rot_elem_to_mat(sxx_tr, syy_tr, sxy_tr, d11, d22, d12)

    # 3. Hill equivalent stress
    seq_sq = a11 * s11 * s11 + a22 * s22 * s22 - a1122 * s11 * s22 + a12 * s12 * s12
    seq = np.sqrt(np.maximum(seq_sq, 0.0))

    # 4. Deformation rate and dynamic yield stress
    if dt > 0.0:
        epsp_rate = np.maximum(
            np.maximum(np.abs(deps[:, 0]), np.abs(deps[:, 1])),
            0.5 * np.abs(deps[:, 2])
        ) / dt
    else:
        epsp_rate = np.zeros(n, dtype=float)

    epsp_clamped = np.maximum(epsp_rate, eps0)
    if m_rate != 0.0:
        rate_fac = epsp_clamped ** m_rate
    else:
        rate_fac = np.ones(n, dtype=float)

    eff_ep = np.maximum(b_yield + epsp, _EM20)
    sigy = a_yield * (eff_ep ** n_hard) * rate_fac
    sigy = np.minimum(sigy, sig_max)

    # 5. Plastic return mapping
    plastic = (seq > sigy) & (off > 0.0)

    sxx_new = sxx_tr.copy()
    syy_new = syy_tr.copy()
    sxy_new = sxy_tr.copy()
    dpla = np.zeros(n, dtype=float)

    nu1 = nu / max(1.0 - nu, _EM20)
    nu5 = 1.0 - nu1
    g3 = 1.5 * e / (1.0 + nu)

    if np.any(plastic):
        idx = np.where(plastic)[0]

        if ipla == 0:
            # Radial return (m32plas.F lines 103-143)
            scale = np.minimum(1.0, sigy[idx] / np.maximum(seq[idx], _EM20))
            sxx_new[idx] = sxx_tr[idx] * scale
            syy_new[idx] = syy_tr[idx] * scale
            sxy_new[idx] = sxy_tr[idx] * scale
            dpla[idx] = off[idx] * np.maximum(0.0, (seq[idx] - sigy[idx]) / e)

        elif ipla == 2:
            # Plane-stress projection with s33 = 0 (m32plas.F lines 144-196)
            p_hydro = -(s11[idx] + s22[idx]) / 3.0
            q_val = (1.0 - nu1) * p_hydro
            s11_p = s11[idx] + q_val
            s22_p = s22[idx] + q_val
            s12_p = s12[idx]

            a_quad = (a11 * s11_p * s11_p + a22 * s22_p * s22_p
                      - a1122 * s11_p * s22_p + a12 * s12_p * s12_p)
            b_quad = -q_val * (a11 * s11_p + a22 * s22_p - 0.5 * a1122 * (s11_p + s22_p))
            c_quad = (a11 + a22 - a1122) * q_val * q_val
            seq_p = np.sqrt(np.maximum(0.0, a_quad + 2.0 * b_quad + c_quad))
            c_quad = c_quad - sigy[idx] * sigy[idx]

            disc = np.maximum(0.0, b_quad * b_quad - a_quad * c_quad)
            scale = np.minimum(1.0, (-b_quad + np.sqrt(disc)) / np.maximum(a_quad, _EM20))
            umr = 1.0 - scale
            q_corr = q_val * umr

            sxx_new[idx] = sxx_tr[idx] * scale - q_corr
            syy_new[idx] = syy_tr[idx] * scale - q_corr
            sxy_new[idx] = sxy_tr[idx] * scale
            dpla[idx] = off[idx] * seq_p * umr / max(_EM20, g3)

        elif ipla == 1:
            # Iterative plane-stress Newton-Raphson return (m32plas.F lines 198-376)
            nu2 = 1.0 - nu * nu
            nu3 = nu * 0.5
            nu4 = 0.5 * (1.0 - nu)

            s1 = 2.0 * nu * a11 - a1122
            s2 = 2.0 * nu * a22 - a1122
            s12_val = a1122 - nu * (a11 + a22)
            s3 = math.sqrt(max(0.0, nu2 * (a11 - a22) ** 2 + s12_val * s12_val))

            q12 = 0.0 if abs(s1) < _EM20 else -(a11 - a22 + s3) / s1
            q21 = 0.0 if abs(s2) < _EM20 else (a11 - a22 + s3) / s2
            jq = 1.0 / (1.0 - q12 * q21)
            jq2 = jq * jq

            a_1 = (a11 + a1122 * q21 + a22 * q21 * q21) * jq2
            a_2 = (a22 + a1122 * q12 + a11 * q12 * q12) * jq2
            a_3 = (a11 * q12 + a22 * q21) * jq2 * 2.0 + a1122 * (jq2 * 2.0 - jq)

            b_1 = a22 - a1122 * nu3 - s3 * jq
            b_2 = a11 - a1122 * nu3 + s3 * jq
            b_3 = a12 * nu4

            for ii in idx:
                s_11_i = s11[ii]
                s_22_i = s22[ii]
                s_12_i = s12[ii]
                seqh_i = seq[ii]
                yld_i = sigy[ii]

                if yld_i >= sig_max:
                    h_slope = 0.0
                else:
                    h_slope = a_yield * n_hard * (max(b_yield + epsp[ii], _EM20) ** (n_hard - 1.0)) * rate_fac[ii]

                dpla_val = (seqh_i - yld_i) / (g3 + h_slope)
                st11 = s_11_i + s_22_i * q12
                st22 = q21 * s_11_i + s_22_i
                axx = a_1 * st11 * st11
                ayy = a_2 * st22 * st22
                axy_cross = a_3 * st11 * st22
                axy_shear = a12 * s_12_i * s_12_i

                for _ in range(3):
                    if dpla_val > 0.0:
                        y_iter = min(sig_max, yld_i + h_slope * dpla_val)
                        dr = a1 * dpla_val / max(y_iter, _EM20)
                        p_1 = 1.0 / (1.0 + b_1 * dr)
                        p_2 = 1.0 / (1.0 + b_2 * dr)
                        p_3 = 1.0 / (1.0 + b_3 * dr)
                        pp1 = p_1 * p_1
                        pp2 = p_2 * p_2
                        pp3 = p_3 * p_3

                        f_res = (axx * pp1 + ayy * pp2 - axy_cross * p_1 * p_2
                                 + axy_shear * pp3 - y_iter * y_iter)
                        df_res = -(
                            (axx * p_1 - 0.5 * axy_cross * p_2) * pp1 * b_1
                            + (ayy * p_2 - 0.5 * axy_cross * p_1) * pp2 * b_2
                            + axy_shear * pp3 * p_3 * b_3
                        ) * (a1 - dr * h_slope) / max(y_iter, _EM20) - h_slope * y_iter

                        if abs(df_res) > _EM20:
                            dpla_val = max(0.0, dpla_val - 0.5 * f_res / df_res)
                    else:
                        dpla_val = 0.0

                dpla[ii] = off[ii] * dpla_val
                y_final = min(sig_max, yld_i + h_slope * dpla[ii])
                dr_final = a1 * dpla[ii] / max(y_final, _EM20)
                p_1 = 1.0 / (1.0 + b_1 * dr_final)
                p_2 = 1.0 / (1.0 + b_2 * dr_final)
                p_3 = 1.0 / (1.0 + b_3 * dr_final)

                s1_ret = st11 * p_1
                s2_ret = st22 * p_2
                s11_ret = jq * (s1_ret - s2_ret * q12)
                s22_ret = jq * (s2_ret - s1_ret * q21)
                s12_ret = s_12_i * p_3

                sx_r, sy_r, sxy_r = _rot_mat_to_elem(
                    np.array([s11_ret]), np.array([s22_ret]), np.array([s12_ret]),
                    np.array([d11[ii]]), np.array([d22[ii]]), np.array([d12[ii]])
                )
                sxx_new[ii] = sx_r[0]
                syy_new[ii] = sy_r[0]
                sxy_new[ii] = sxy_r[0]

    # 6. Update accumulated plastic strain
    epsp_new = epsp + dpla

    # 7. Through-thickness strain increment (sigeps32c.F line 185-190, m32plas.F:140, 193, 344)
    ezz_el = -(deps[:, 0] + deps[:, 1]) * (nu / max(1.0 - nu, _EM20))
    if ipla == 1:
        # Material axes formulation (m32plas.F lines 342-344)
        s11_m, s22_m, _ = _rot_elem_to_mat(sxx_new, syy_new, sxy_new, d11, d22, d12)
        s1_ezz = a11 * s11_m + a22 * s22_m - 0.5 * a1122 * (s11_m + s22_m)
        ezz_pl = -nu5 * dpla * s1_ezz / np.maximum(sigy, _EM20)
    else:
        # Radial projection & plane stress projection (m32plas.F lines 140, 193)
        ezz_pl = -nu5 * dpla * 0.5 * (sxx_new + syy_new) / np.maximum(sigy, _EM20)
    ezz_tot = ezz_el + ezz_pl

    # 8. Plastic failure element deletion (m32plas.F lines 354-358)
    failed = (epsp_new >= eps_max) & (off > 0.0)
    if np.any(failed):
        f_idx = np.where(failed)[0]
        off[f_idx] = 0.0
        sxx_new[f_idx] = 0.0
        syy_new[f_idx] = 0.0
        sxy_new[f_idx] = 0.0
        if syz_tr is not None:
            syz_tr[f_idx] = 0.0
            szx_tr[f_idx] = 0.0

    # Write state back to extra and arrays
    if extra is not None:
        if "uv32" in extra and extra["uv32"] is not None:
            uv32 = extra["uv32"]
            if uv32.ndim == 1:
                uv32[:] = epsp_new
            else:
                uv32[:, 0] = epsp_new
                if uv32.shape[-1] > 1:
                    uv32[:, 1] += ezz_tot
        if "off32" in extra and extra["off32"] is not None:
            extra["off32"][:] = off
        if "off" in extra and extra["off"] is not None:
            extra["off"][:] = off
        if "layfail" in extra and extra["layfail"] is not None:
            extra["layfail"][:] = off
        if "ezz" in extra:
            extra["ezz"] = ezz_tot
        if "thk" in extra and "thklyl" in extra:
            thk = np.asarray(extra["thk"], dtype=float)
            thklyl = np.asarray(extra["thklyl"], dtype=float)
            extra["thk"] = thk + ezz_tot * thklyl * off

    if epsp_in is not None and isinstance(epsp_in, np.ndarray):
        epsp_in[:] = epsp_new

    # Store new stresses
    sig[:, 0] = sxx_new
    sig[:, 1] = syy_new
    sig[:, 2] = sxy_new
    if syz_tr is not None:
        sig[:, 3] = syz_tr
        sig[:, 4] = szx_tr

    if is_1d:
        return sig[0], epsp_new[0]
    return sig, epsp_new


# ============================================================================
# Consistent Algorithmic Membrane Tangent
# ============================================================================

def shell_membrane_tangent(mat: Material) -> np.ndarray:
    """Constant (3, 3) elastic plane-stress membrane constitutive matrix."""
    p = mat.params
    e = float(p.get("E", mat.E))
    nu = float(p.get("nu", mat.nu))
    g = float(p.get("G", mat.G))
    a1 = float(p.get("A1", e / (1.0 - nu ** 2)))
    a2 = float(p.get("A2", nu * a1))

    return np.array([
        [a1, a2, 0.0],
        [a2, a1, 0.0],
        [0.0, 0.0, g],
    ], dtype=float)


def consistent_shell_tangent(mat: Material, sig: np.ndarray,
                             epsp: Optional[np.ndarray] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             extra: Optional[Dict[str, Any]] = None,
                             symmetric: bool = False) -> np.ndarray:
    """Consistent (algorithmic) plane-stress elastoplastic tangent (n, 3, 3).

    Differentiates the discrete Hill plasticity return mapping with respect to
    in-plane strain increment.

    Parameters
    ----------
    mat : Material
    sig : (n, 3) or (n, >=3) Cauchy stresses
    epsp : (n,) accumulated plastic strain at end of step
    epsp_incr : (n,) plastic strain increment of current step
    extra : layer state (orientation angle/dir, etc.)
    symmetric : whether to enforce symmetry (default False for exact algorithmic derivative)
    """
    if not isinstance(sig, np.ndarray):
        sig = np.array(sig, dtype=float)

    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig[None, :]

    n = sig.shape[0]
    if n == 0:
        return np.empty((0, 3, 3), dtype=float)

    c_el = shell_membrane_tangent(mat)
    d_tangent = np.tile(c_el, (n, 1, 1))

    if epsp_incr is None:
        return d_tangent[0] if is_1d else d_tangent

    epsp_incr_arr = np.asarray(epsp_incr, dtype=float).flatten()
    if epsp_incr_arr.ndim == 0 or len(epsp_incr_arr) == 1:
        epsp_incr_arr = np.full(n, float(epsp_incr_arr.item() if epsp_incr_arr.ndim == 0 else epsp_incr_arr[0]), dtype=float)

    if epsp is not None:
        epsp_arr = np.asarray(epsp, dtype=float).flatten()
        if epsp_arr.ndim == 0 or len(epsp_arr) == 1:
            epsp_arr = np.full(n, float(epsp_arr.item() if epsp_arr.ndim == 0 else epsp_arr[0]), dtype=float)
    else:
        epsp_arr = None

    if extra is not None and "off32" in extra and extra["off32"] is not None:
        off = np.asarray(extra["off32"], dtype=float).flatten()
    elif extra is not None and "off" in extra and extra["off"] is not None:
        off = np.asarray(extra["off"], dtype=float).flatten()
    elif extra is not None and "layfail" in extra and extra["layfail"] is not None:
        off = np.asarray(extra["layfail"], dtype=float).flatten()
    else:
        off = np.ones(n, dtype=float)
    if len(off) == 1 and n > 1:
        off = np.full(n, float(off[0]), dtype=float)

    plastic = (epsp_incr_arr > 0.0) & (off > 0.0)
    for ii in range(n):
        if off[ii] <= 0.0:
            d_tangent[ii] = 0.0

    if not np.any(plastic):
        return d_tangent[0] if is_1d else d_tangent

    p = mat.params
    e = float(p.get("E", mat.E))
    a_yield = float(p.get("A", _INF))
    b_yield = float(p.get("B", 0.0))
    n_hard = float(p.get("n", 1.0))
    eps_max = float(p.get("eps_max", _INF))
    sig_max = float(p.get("sig_max", _INF))
    a11 = float(p.get("A11", 1.0))
    a22 = float(p.get("A22", 1.0))
    a1122 = float(p.get("A1122", 1.0))
    a12 = float(p.get("A12", 3.0))

    d11, d22, d12 = _get_dir_cosines(extra, n)
    p_hill = np.array([
        [a11, -0.5 * a1122, 0.0],
        [-0.5 * a1122, a22, 0.0],
        [0.0, 0.0, a12],
    ], dtype=float)

    idx = np.where(plastic)[0]
    for ii in idx:
        dl = float(epsp_incr_arr[ii])
        ep_ii = float(epsp_arr[ii]) if epsp_arr is not None else dl

        if ep_ii >= eps_max or off[ii] <= 0.0:
            d_tangent[ii] = 0.0
            continue

        # For IPLA=0 (explicit radial return), yield stress is frozen from start of step,
        # so d(sigy)/d(deps) = 0 (H_eff = 0), matching the discrete algorithm to machine precision.
        # For IPLA=1 (implicit iterative return) or when hardening is explicitly requested,
        # H_eff is the hardening slope.
        ipla = int(extra.get("ipla", p.get("ipla", 0)) if extra is not None else p.get("ipla", 0))
        hardening = extra.get("hardening", None) if extra is not None else None
        if hardening is True or (ipla != 0 and hardening is not False):
            eff_ep = max(b_yield + ep_ii, _EM20)
            cur_sigy = a_yield * (eff_ep ** n_hard)
            if cur_sigy >= sig_max:
                h_eff = 0.0
            else:
                h_eff = a_yield * n_hard * (eff_ep ** (n_hard - 1.0))
        else:
            h_eff = 0.0

        # Direct rotation matrix Q_sigma from element to material axes
        d11_i = float(d11[ii])
        d22_i = float(d22[ii])
        d12_i = float(d12[ii])
        q_sigma = np.array([
            [d11_i, d22_i, 2.0 * d12_i],
            [d22_i, d11_i, -2.0 * d12_i],
            [-d12_i, d12_i, d11_i - d22_i],
        ], dtype=float)

        p_elem = q_sigma.T @ p_hill @ q_sigma

        # Converged stress and trial stress reconstruction
        s_c = sig[ii, :3]
        seq_c = math.sqrt(max(0.0, float(s_c @ p_elem @ s_c)))
        seq_c = max(seq_c, _EM20)

        # Trial equivalent stress q_tr = seq_c + E * dl
        q_tr = seq_c + e * dl
        scale = seq_c / max(q_tr, _EM20)
        s_tr = s_c / max(scale, _EM20)

        # Gradient of trial equivalent stress: dq_tr / d(deps) = C_el @ P_elem @ s_tr / q_tr
        g_vec = c_el @ p_elem @ s_tr
        denom = q_tr * q_tr
        gamma = (h_eff / (e + h_eff) - scale) / max(denom, _EM20)

        rank1 = np.outer(s_tr, g_vec)
        d_tangent[ii] = scale * c_el + gamma * rank1

        if symmetric:
            d_tangent[ii] = 0.5 * (d_tangent[ii] + d_tangent[ii].T)

    return d_tangent[0] if is_1d else d_tangent


# ============================================================================
# Registry Hook
# ============================================================================

def _register() -> None:
    """Register LAW32 in MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (32, "32", "LAW32", "HILL", "MAT_HILL", "LAW32_HILL"):
            MAT_PHYSICS_REGISTRY[key] = build_law32
    except Exception:
        pass


_register()
