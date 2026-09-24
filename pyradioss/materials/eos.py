"""
/EOS — equations of state for solid and fluid elements (M6, M580, M610).

Fortran origin: ``starter/source/materials/eos/hm_read_eos.F`` (readers,
one per EOS type) and ``common_source/eos/eosmain.F`` (the engine solve).
OpenRadioss defines 21 distinct EOS formulations:

  1.  POLYNOMIAL      (eospolyno.F)
  2.  GRUNEISEN       (gruneisen.F)
  3.  TILLOTSON       (tillotson.F)
  4.  PUFF            (puff.F)
  5.  SESAME          (sesame.F, mintp_*.F)
  6.  NOBLE-ABEL      (noble_abel.F)
  7.  IDEAL-GAS       (idealgas.F)
  8.  MURNAGHAN       (murnaghan.F)
  9.  OSBORNE         (osborne.F)
  10. STIFF-GAS       (stiffgas.F)
  11. LSZK            (lszk.F)
  12. POWDER-BURN     (powder_burn.F)
  13. COMPACTION      (compaction.F90)
  14. NASG            (nasg.F)
  15. JWL             (jwl.F)
  16. IDEAL-GAS-VT    (idealgas_vt.F)
  17. TABULATED       (tabulated.F)
  18. LINEAR          (eoslinear.F)
  19. EXPONENTIAL     (eosexponential.F90)
  20. COMPACTION2     (compaction2.F90)
  21. COMPACTION_TAB  (compaction_tab.F90)

Theory & Solvers:
  - Linear in E: p_new = A(mu) + B(mu) E_new -> closed-form Crank-Nicolson solve.
  - Non-linear in E: iterative 2-step solve (OSBORNE) or Newton-Raphson (IDEAL-GAS-VT).
  - Purely mechanical (E-independent): LINEAR, MURNAGHAN, COMPACTION suite, EXPONENTIAL.
  - Two-phase mixture & burn kinetics: POWDER-BURN (Atwood-Friis-Moxnes).
  - Tabulated 1D/2D: TABULATED (/FUNCT), SESAME (rational interpolation in 2D grids).
"""

from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np


# ============================================================================
# Interpolation Utilities
# ============================================================================

def minter1d_rat(x0: float, x1: float, x2: float, x3: float,
                 y0: float, y1: float, y2: float, y3: float,
                 x: float, i: int, n: int) -> tuple[float, float]:
    """1D Rational function interpolation (common_source/eos/minter1d_rat.F).

    Returns (y, yp) where yp is the first derivative dy/dx.
    """
    q = x - x1
    d = x2 - x1
    r = d - q
    d_safe = d if abs(d) > 1e-20 else (1e-20 if d >= 0 else -1e-20)
    s = (y2 - y1) / d_safe
    dx32 = x3 - x2
    dx32_safe = dx32 if abs(dx32) > 1e-20 else (1e-20 if dx32 >= 0 else -1e-20)
    sp = (y3 - y2) / dx32_safe
    dx31 = x3 - x1
    dx31_safe = dx31 if abs(dx31) > 1e-20 else (1e-20 if dx31 >= 0 else -1e-20)
    c2 = (sp - s) / dx31_safe

    dm = x1 - x0
    dm_abs = max(1e-20, abs(dm))
    dm = math.copysign(dm_abs, dm)
    sm = (y1 - y0) / dm
    c1 = (s - sm) / (d + dm)
    c6 = 0.0

    if i == 1:
        if s * (s - d * c2) <= 0.0:
            c2 = s / d_safe
        c4 = c2
    elif i == n - 1:
        c4 = c1
    else:
        if i == 2 and sm * (sm - dm * c1) <= 0.0:
            c1 = (s - sm - sm) / d_safe
        c3 = abs(c2 * r)
        c3d = c3 + abs(c1 * q)
        c5 = 0.0
        if c3d > 0.0:
            c3 = c3 / c3d
            c5 = c3 * (c1 - c2)
        c4 = c2 + c5
        c6 = d * c5 * (1.0 - c3)

    y = y1 + q * (s - r * c4)
    yp = s + (q - r) * c4 + c6
    return float(y), float(yp)


def mindex_1d(arr: np.ndarray | list[float], val: float) -> int:
    """1-based binary search index matching engine/source/materials/mat/mat026/mindex.F.

    Returns index 1 <= i <= len(arr) - 1.
    """
    n = len(arr)
    if n <= 1:
        return 1
    idx = int(np.searchsorted(arr, val, side="right"))
    return max(1, min(n - 1, idx))


def mintp1_rt(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float, float]:
    """2D rational interpolation Z(x,y) and partial derivatives dZ/dx, dZ/dy.

    Upstream Fortran reference:
      common_source/eos/mintp1_rt.F
      common_source/eos/mintp_rt.F
    """
    nx = len(xx)
    ny = len(yy)
    ix = mindex_1d(xx, x)
    iy = mindex_1d(yy, y)

    ix = max(1, min(nx - 1, ix))
    iy = max(1, min(ny - 1, iy))

    ixm1 = max(1, ix - 1)
    ixp1 = ix + 1
    ixp2 = min(nx, ix + 2)

    iym1 = max(1, iy - 1)
    iyp1 = iy + 1
    iyp2 = min(ny, iy + 2)

    i_x = [ixm1 - 1, ix - 1, ixp1 - 1, ixp2 - 1]
    i_y = [iym1 - 1, iy - 1, iyp1 - 1, iyp2 - 1]

    xx0, xx1, xx2, xx3 = float(xx[i_x[0]]), float(xx[i_x[1]]), float(xx[i_x[2]]), float(xx[i_x[3]])

    z_lev = []
    dzdx_lev = []
    for ky in i_y:
        z_k, dzdx_k = minter1d_rat(
            xx0, xx1, xx2, xx3,
            float(zz[i_x[0], ky]), float(zz[i_x[1], ky]),
            float(zz[i_x[2], ky]), float(zz[i_x[3], ky]),
            x, ix, nx,
        )
        z_lev.append(z_k)
        dzdx_lev.append(dzdx_k)

    yy0, yy1, yy2, yy3 = float(yy[i_y[0]]), float(yy[i_y[1]]), float(yy[i_y[2]]), float(yy[i_y[3]])

    z_val, dzdy = minter1d_rat(
        yy0, yy1, yy2, yy3,
        z_lev[0], z_lev[1], z_lev[2], z_lev[3],
        y, iy, ny,
    )
    dzdx, _ = minter1d_rat(
        yy0, yy1, yy2, yy3,
        dzdx_lev[0], dzdx_lev[1], dzdx_lev[2], dzdx_lev[3],
        y, iy, ny,
    )
    return float(z_val), float(dzdx), float(dzdy)


def mintp_re(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    x: float,
    z: float,
) -> tuple[float, float]:
    """Inverse 2D rational interpolation: find y(x, z) such that Z(x, y) = z, and dy/dz.

    Upstream Fortran reference:
      common_source/eos/mintp_re.F
    """
    nx = len(xx)
    ny = len(yy)
    ix = mindex_1d(xx, x)
    ix = max(1, min(nx - 1, ix))
    col_z = zz[ix - 1, :]
    iy = mindex_1d(col_z, z)
    iy = max(1, min(ny - 1, iy))

    ixm1 = max(1, ix - 1)
    ixp1 = ix + 1
    ixp2 = min(nx, ix + 2)

    iym1 = max(1, iy - 1)
    iyp1 = iy + 1
    iyp2 = min(ny, iy + 2)

    i_x = [ixm1 - 1, ix - 1, ixp1 - 1, ixp2 - 1]
    i_y = [iym1 - 1, iy - 1, iyp1 - 1, iyp2 - 1]

    xx0, xx1, xx2, xx3 = float(xx[i_x[0]]), float(xx[i_x[1]]), float(xx[i_x[2]]), float(xx[i_x[3]])

    z_lev = []
    dzdx_lev = []
    for ky in i_y:
        z_k, dzdx_k = minter1d_rat(
            xx0, xx1, xx2, xx3,
            float(zz[i_x[0], ky]), float(zz[i_x[1], ky]),
            float(zz[i_x[2], ky]), float(zz[i_x[3], ky]),
            x, ix, nx,
        )
        z_lev.append(z_k)
        dzdx_lev.append(dzdx_k)

    yy0, yy1, yy2, yy3 = float(yy[i_y[0]]), float(yy[i_y[1]]), float(yy[i_y[2]]), float(yy[i_y[3]])

    y_val, dydz = minter1d_rat(
        z_lev[0], z_lev[1], z_lev[2], z_lev[3],
        yy0, yy1, yy2, yy3,
        z, iy, ny,
    )
    return float(y_val), float(dydz)


def read_sesame_file(filepath: str) -> dict[str, Any]:
    """Parse standard ASCII SESAME table (format 301) and convert to SI units.

    Upstream Fortran reference:
      starter/source/materials/mat/mat026/mrdse2.F
      starter/source/materials/eos/sesame_tools.F
    """
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    header_tokens = lines[1].split()
    nr = int(float(header_tokens[0]))
    nt = int(float(header_tokens[1]))

    tokens: list[float] = []
    for line in lines[2:]:
        for tok in line.split():
            try:
                tokens.append(float(tok))
            except ValueError:
                pass

    idx = 2
    r_tab = np.array(tokens[idx:idx + nr], dtype=float) * 1000.0  # Mg/m^3 -> kg/m^3
    idx += nr
    t_tab = np.array(tokens[idx:idx + nt], dtype=float)  # K
    idx += nt

    p_flat = np.array(tokens[idx:idx + nr * nt], dtype=float) * 1.0e9  # GPa -> Pa
    p_tab = p_flat.reshape((nt, nr)).T  # (nr, nt)
    idx += nr * nt

    e_flat = np.array(tokens[idx:idx + nr * nt], dtype=float) * 1.0e6  # MJ/kg -> J/kg
    e_tab = e_flat.reshape((nt, nr)).T  # (nr, nt)

    return {
        "nr": nr,
        "nt": nt,
        "rho_table": r_tab,
        "theta_table": t_tab,
        "p_table": p_tab,
        "e_table": e_tab,
    }


def _eval_funct_1d(func: Any, x: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate 1D function or curve returning (y, dy/dx)."""
    if callable(func):
        val = func(x)
        # Numerical slope
        eps = 1e-6 * np.maximum(np.abs(x), 1.0)
        val_p = func(x + eps)
        der = (val_p - val) / eps
        return np.asarray(val, dtype=float), np.asarray(der, dtype=float)

    if isinstance(func, (tuple, list)) and len(func) == 2:
        xs, ys = np.asarray(func[0], dtype=float), np.asarray(func[1], dtype=float)
        x_arr = np.asarray(x, dtype=float)
        y = np.interp(x_arr, xs, ys)
        # Segment derivative
        idx = np.searchsorted(xs, x_arr, side="right")
        idx = np.clip(idx, 1, len(xs) - 1)
        dx = xs[idx] - xs[idx - 1]
        dx_safe = np.where(np.abs(dx) < 1e-15, 1e-15, dx)
        der = (ys[idx] - ys[idx - 1]) / dx_safe
        return y, der

    # Default fallback: linear through origin with unit slope
    x_arr = np.asarray(x, dtype=float)
    return x_arr, np.ones_like(x_arr)


# ============================================================================
# EOS Coefficients Formulations (A, B in p = A + B*E)
# ============================================================================

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
    p_star = p.get("p_star", p.get("pstar", 0.0))
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
    t1 = p.get("t1", p.get("sigt1", 0.0))
    if t1 == 0.0:
        t1 = c1
    t2 = p.get("t2", p.get("sigt2", 0.0))
    esubl = p.get("es", p.get("esubl", p.get("es_subl", 0.0)))
    gamma0 = p.get("gamma0", p.get("g0", p.get("gamma", 0.0)))
    h = p.get("h", p.get("hh", p.get("eoh", 0.0)))
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
    denom_cc = gamma0 * esubl
    safe_denom_cc = denom_cc if abs(denom_cc) > 1e-12 else 1.0
    cc = np.where(np.abs(denom_cc) > 1e-12, c1 / safe_denom_cc, 0.0)
    expa = np.exp(np.clip(cc * xx, -50.0, 50.0))
    aa_hot = bb_hot * esubl * (expa - 1.0)

    is_comp = (mu >= 0.0)
    is_cold = (mu < 0.0) & (e_arr < esubl)

    A_unscaled = np.where(is_comp, aa_comp, np.where(is_cold, aa_cold, aa_hot))
    A = A_unscaled - psh
    B = np.where(is_comp | is_cold, gamma0, bb_hot)
    return A, B


def _coefficients_osborne(eos, mu: np.ndarray, e: np.ndarray | float | None = None):
    """A(mu, e), B(mu, e) for Osborne EOS (common_source/eos/osborne.F)."""
    p = eos.params
    a1 = p.get("a1", 0.0)
    a2 = p.get("a2", 0.0)
    b0 = p.get("b0", 0.0)
    b1 = p.get("b1", 0.0)
    b2 = p.get("b2", 0.0)
    c0 = p.get("c0", 0.0)
    c1 = p.get("c1", 0.0)
    d0 = p.get("d0", 1.0)
    psh = p.get("psh", 0.0)

    if e is None:
        e = p.get("e0", 0.0)
    e_arr = np.asarray(e, dtype=float)

    a2_star = np.where(mu >= 0.0, a2, -a2)
    denom = np.maximum(e_arr + d0, 1e-12)
    A = (a1 * mu + a2_star * (mu ** 2)) / denom - psh
    B = ((b0 + b1 * mu + b2 * (mu ** 2)) + (c0 + c1 * mu) * e_arr) / denom
    return A, B


def _coefficients_lszk(eos, mu: np.ndarray):
    """A(mu), B(mu) for LSZK EOS (common_source/eos/lszk.F)."""
    p = eos.params
    gamma = p.get("gamma", 1.4)
    a = p.get("a", 0.0)
    b = p.get("b", 0.0)
    psh = p.get("psh", 0.0)

    eta = np.maximum(1.0 + mu, 1e-12)
    A = a * np.power(eta, b) - psh
    B = (gamma - 1.0) * eta
    return A, B


def _coefficients_linear(eos, mu: np.ndarray):
    """A(mu), B(mu) for Linear Hookean EOS (common_source/eos/eoslinear.F)."""
    p = eos.params
    c0 = p.get("c0", p.get("p0", 0.0))
    bulk = p.get("bulk", p.get("c1", 0.0))
    psh = p.get("psh", 0.0)

    A = c0 + bulk * mu - psh
    B = np.zeros_like(A)
    return A, B


def _coefficients_exponential(eos, time: float = 0.0):
    """A(t), B(t) for Exponential EOS (common_source/eos/eosexponential.F90)."""
    p = eos.params
    p0 = p.get("p0", 0.0)
    alpha = p.get("alpha", 0.0)
    psh = p.get("psh", 0.0)

    p0_net = p0 - psh
    A = p0_net * math.exp(alpha * time)
    B = 0.0
    return A, B


def _coefficients_tabulated(eos, mu: np.ndarray):
    """A(mu), B(mu) for Tabulated EOS (common_source/eos/tabulated.F)."""
    p = eos.params
    f_a = p.get("func_a", p.get("a_func"))
    f_b = p.get("func_b", p.get("b_func"))
    fscale_a = p.get("fscale_a", 1.0)
    fscale_b = p.get("fscale_b", 1.0)
    psh = p.get("psh", 0.0)

    ya, _ = _eval_funct_1d(f_a, mu) if f_a is not None else (np.zeros_like(mu), np.zeros_like(mu))
    yb, _ = _eval_funct_1d(f_b, mu) if f_b is not None else (np.zeros_like(mu), np.zeros_like(mu))

    A = fscale_a * ya - psh
    B = fscale_b * yb
    return A, B


def _coefficients_compaction(eos, mu: np.ndarray, mu_bak: np.ndarray | float | None = None):
    """A(mu), B(mu) for Compaction EOS (common_source/eos/compaction.F90)."""
    p = eos.params
    c0 = p.get("c0", 0.0)
    c1 = p.get("c1", 0.0)
    c2 = p.get("c2", 0.0)
    c3 = p.get("c3", 0.0)
    iform = p.get("iform", 2)
    mumin = p.get("mumin", p.get("mue_min", 0.0))
    mumax = p.get("mumax", p.get("mue_max", 1e20))
    bunl = p.get("bunl", p.get("b", c1))
    psh = p.get("psh", 0.0)
    pmin = p.get("pmin", -1e30)

    mu_pos = np.maximum(mu, 0.0)
    p_virgin = c0 + c1 * mu + (c2 + c3 * mu) * (mu_pos ** 2)

    if mu_bak is None:
        mu_bak = mu
    mu_bak_arr = np.minimum(mumax, np.maximum(np.asarray(mu_bak, dtype=float), 0.0))

    if iform == 1:
        b_slope = np.full_like(mu, bunl, dtype=float)
    else:
        alpha = np.where(mumax > 0.0, mu_bak_arr / mumax, 1.0)
        b_slope = alpha * bunl + (1.0 - alpha) * c1

    p_star = c0 + c1 * mu_bak_arr + (c2 + c3 * mu_bak_arr) * (mu_bak_arr ** 2)
    p_unl = p_star - (mu_bak_arr - mu) * b_slope

    p_val = np.where(mu_bak_arr > mumin, np.minimum(p_unl, p_virgin), p_virgin)
    p_val = np.maximum(p_val, pmin) - psh
    return p_val, np.zeros_like(p_val)


def _coefficients_compaction2(eos, mu: np.ndarray, mu_bak: np.ndarray | float | None = None):
    """A(mu), B(mu) for Compaction2 EOS (common_source/eos/compaction2.F90)."""
    p = eos.params
    p_func = p.get("p_func", p.get("func_p"))
    fscale = p.get("fscale", p.get("fscale_p", 1.0))
    xscale = p.get("xscale", p.get("xscale_p", 1.0))
    iform = p.get("iform", 2)
    mumin = p.get("mumin", p.get("mue_min", 0.0))
    mumax = p.get("mumax", p.get("mue_max", 1e20))
    bmin = p.get("bmin", p.get("bt", 1.0))
    bmax = p.get("bmax", p.get("b", bmin))
    psh = p.get("psh", 0.0)
    pmin = p.get("pmin", -1e30)

    y_virg, _ = _eval_funct_1d(p_func, xscale * mu) if p_func is not None else (mu, np.ones_like(mu))
    p_virgin = fscale * y_virg

    if mu_bak is None:
        mu_bak = mu
    mu_bak_arr = np.minimum(mumax, np.maximum(np.asarray(mu_bak, dtype=float), 0.0))

    if iform == 1:
        b_slope = np.full_like(mu, bmax, dtype=float)
    else:
        alpha = np.where(mumax > 0.0, mu_bak_arr / mumax, 1.0)
        b_slope = alpha * bmax + (1.0 - alpha) * bmin

    y_star, _ = _eval_funct_1d(p_func, xscale * mu_bak_arr) if p_func is not None else (mu_bak_arr, np.ones_like(mu_bak_arr))
    p_star = fscale * y_star
    p_unl = p_star - (mu_bak_arr - mu) * b_slope

    p_val = np.where(mu_bak_arr > mumin, np.minimum(p_unl, p_virgin), p_virgin)
    p_val = np.maximum(p_val, pmin) - psh
    return p_val, np.zeros_like(p_val)


def _coefficients_compaction_tab(eos, mu: np.ndarray, mu_bak: np.ndarray | float | None = None):
    """A(mu), B(mu) for Compaction Tab EOS (common_source/eos/compaction_tab.F90)."""
    p = eos.params
    rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)
    p_func = p.get("p_func", p.get("func_p"))
    psh = p.get("psh", 0.0)
    pmin = p.get("pmin", -1e30)
    rho = rho0 * (1.0 + mu)
    if p_func is not None:
        y_pc, _ = _eval_funct_1d(p_func, rho)
    else:
        y_pc = np.zeros_like(mu)
    p_val = np.maximum(y_pc, pmin) - psh
    return p_val, np.zeros_like(p_val)


def _coefficients_idealgas_vt(eos, mu: np.ndarray, e: np.ndarray | float | None = None):
    """A(mu, e), B(mu) for Ideal Gas with Cp(T) (common_source/eos/idealgas_vt.F)."""
    p = eos.params
    r_gas = p.get("r_gas", p.get("r", 287.0))
    psh = p.get("psh", 0.0)
    a0 = p.get("a0", p.get("c0", 1000.0))
    a1 = p.get("a1", p.get("c1", 0.0))
    a2 = p.get("a2", p.get("c2", 0.0))
    a3 = p.get("a3", p.get("c3", 0.0))
    a4 = p.get("a4", p.get("c4", 0.0))
    rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

    if e is None:
        e = p.get("e0", 0.0)
    e_arr = np.asarray(e, dtype=float)
    e_spec = e_arr / rho0

    # Newton-Raphson solve for temperature T from int_0^T (Cp - r) dT = e_spec
    temp = np.full_like(e_spec, p.get("t0", 300.0))
    for _ in range(30):
        t2 = temp ** 2
        t3 = temp * t2
        t4 = t2 ** 2
        t5 = temp * t4
        f_val = a0 * temp + 0.5 * a1 * t2 + (1.0 / 3.0) * a2 * t3 + 0.25 * a3 * t4 + 0.2 * a4 * t5 - r_gas * temp - e_spec
        df_val = a0 + a1 * temp + a2 * t2 + a3 * t3 + a4 * t4 - r_gas
        df_safe = np.where(np.abs(df_val) < 1e-12, 1e-12, df_val)
        delta = -f_val / df_safe
        temp = np.maximum(temp + delta, 1.0)
        if np.all(np.abs(delta) < 1e-5):
            break

    cv = a0 + a1 * temp + a2 * (temp ** 2) + a3 * (temp ** 3) + a4 * (temp ** 4) - r_gas
    cp = cv + r_gas
    gamma = cp / np.maximum(cv, 1e-12)

    eta = 1.0 + mu
    p_val = rho0 * eta * r_gas * temp - psh
    B = gamma * eta
    A = p_val - B * e_arr
    return A, B


def _coefficients_sesame(eos, mu: np.ndarray, e: np.ndarray | float | None = None):
    """A(mu), B(mu) for SESAME Tabular EOS (common_source/eos/sesame.F)."""
    p = eos.params
    rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)
    rho_tab = p.get("rho_table", p.get("r_table"))
    theta_tab = p.get("theta_table", p.get("t_table"))
    p_tab = p.get("p_table")
    e_tab = p.get("e_table")
    filename = p.get("filename")
    if (rho_tab is None or p_tab is None or e_tab is None) and filename:
        try:
            table_dict = read_sesame_file(filename)
            p.update(table_dict)
            rho_tab = p.get("rho_table")
            theta_tab = p.get("theta_table")
            p_tab = p.get("p_table")
            e_tab = p.get("e_table")
        except Exception:
            pass

    has_table = (rho_tab is not None and theta_tab is not None
                 and p_tab is not None and e_tab is not None)

    mu_arr = np.asarray(mu, dtype=float)
    if e is None:
        e = p.get("e0", 0.0)
    e_arr = np.asarray(e, dtype=float)
    if e_arr.ndim == 0:
        e_arr = np.full_like(mu_arr, float(e_arr))

    if has_table:
        r_arr = np.asarray(rho_tab, dtype=float)
        t_arr = np.asarray(theta_tab, dtype=float)
        p_mat = np.asarray(p_tab, dtype=float)
        e_mat = np.asarray(e_tab, dtype=float)

        A = np.zeros_like(mu_arr)
        B = np.zeros_like(mu_arr)
        for i in range(len(mu_arr)):
            rho_i = rho0 * (1.0 + float(mu_arr[i]))
            espem = float(e_arr[i]) / max(rho0, 1e-12)
            t_val, dtde = mintp_re(r_arr, t_arr, e_mat, rho_i, espem)
            p_val, _, dpdt = mintp1_rt(r_arr, t_arr, p_mat, rho_i, t_val)
            dpde = dpdt * dtde / max(rho0, 1e-12)
            B[i] = dpde
            A[i] = p_val - dpde * float(e_arr[i])
        return A, B
    else:
        k0 = float(p.get("k0", 2.0e9))
        gamma0 = float(p.get("gamma0", 1.4))
        A = k0 * mu_arr
        B = (gamma0 - 1.0) * (1.0 + mu_arr)
        return A, B


def _coefficients_idealgas(eos, mu: np.ndarray):
    """A(mu), B(mu) for Ideal Gas EOS (common_source/eos/idealgas.F)."""
    p = eos.params
    gamma = p.get("gamma", p.get("c4", 0.4) + 1.0)
    psh = p.get("psh", 0.0)
    eta = 1.0 + mu
    A = np.full_like(mu, -psh, dtype=float)
    B = (gamma - 1.0) * eta
    return A, B


def _coefficients_polynomial(eos, mu: np.ndarray):
    """A(mu), B(mu) for Polynomial EOS (common_source/eos/eospolyno.F)."""
    p = eos.params
    mu_pos = np.maximum(mu, 0.0)
    mu2 = mu * mu_pos
    c0 = p.get("c0", 0.0)
    c1 = p.get("c1", 0.0)
    c2 = p.get("c2", 0.0)
    c3 = p.get("c3", 0.0)
    c4 = p.get("c4", 0.0)
    c5 = p.get("c5", 0.0)
    c6 = p.get("c6", 0.0)
    psh = p.get("psh", 0.0)
    A = (c0 - psh) + (c1 + c3 * (mu ** 2)) * mu + c2 * mu2
    B = c4 + c5 * mu + c6 * mu2
    return A, B


def _coefficients_stiffgas(eos, mu: np.ndarray):
    """A(mu), B(mu) for Stiffened Gas EOS (common_source/eos/stiffgas.F)."""
    p = eos.params
    gamma = p.get("gamma", 1.4)
    p_star = p.get("p_star", p.get("pstar", 0.0))
    psh = p.get("psh", 0.0)
    A = np.full_like(mu, -gamma * p_star - psh, dtype=float)
    B = (gamma - 1.0) * (1.0 + mu)
    return A, B


def coefficients(eos, mu: np.ndarray, e: np.ndarray | float | None = None, time: float = 0.0):
    """A(mu), B(mu) of p = A + B E (see module docstring)."""
    kind = eos.kind.upper()
    if kind == "SESAME":
        return _coefficients_sesame(eos, mu, e)
    if kind in ("IDEAL-GAS", "IDEAL_GAS"):
        return _coefficients_idealgas(eos, mu)
    if kind in ("POLYNOMIAL", "POLY"):
        return _coefficients_polynomial(eos, mu)
    if kind == "GRUNEISEN":
        return _coefficients_gruneisen(eos, mu)
    if kind == "TILLOTSON":
        return _coefficients_tillotson(eos, mu, e)
    if kind == "JWL":
        return _coefficients_jwl(eos, mu)
    if kind == "MURNAGHAN":
        return _coefficients_murnaghan(eos, mu)
    if kind in ("NOBLE-ABEL", "NOBLE_ABEL"):
        return _coefficients_noble_abel(eos, mu)
    if kind in ("NASG", "NOBLE-ABEL-STIFFENED-GAS", "NOBLE_ABEL_STIFFENED_GAS"):
        return _coefficients_nasg(eos, mu)
    if kind == "PUFF":
        return _coefficients_puff(eos, mu, e)
    if kind in ("STIFF-GAS", "STIFF_GAS", "STIFFENED_GAS", "STIFFGAS", "SG"):
        return _coefficients_stiffgas(eos, mu)
    if kind in ("OSBORNE", "OSBORN"):
        return _coefficients_osborne(eos, mu, e)
    if kind == "LSZK":
        return _coefficients_lszk(eos, mu)
    if kind == "LINEAR":
        return _coefficients_linear(eos, mu)
    if kind == "EXPONENTIAL":
        return _coefficients_exponential(eos, time)
    if kind in ("COMPACTION", "COMPACT"):
        return _coefficients_compaction(eos, mu)
    if kind == "COMPACTION2":
        return _coefficients_compaction2(eos, mu)
    if kind == "COMPACTION_TAB":
        return _coefficients_compaction_tab(eos, mu)
    if kind in ("IDEAL-GAS-VT", "IDEALGAS_VT"):
        return _coefficients_idealgas_vt(eos, mu, e)
    if kind == "TABULATED":
        return _coefficients_tabulated(eos, mu)

    # Default fallback to polynomial
    return _coefficients_polynomial(eos, mu)


# ============================================================================
# Solver Update & Sound Speed
# ============================================================================

def update(eos, mu: np.ndarray, dv: np.ndarray, e_old: np.ndarray,
           p_old: np.ndarray, de_other: np.ndarray,
           dt: float = 0.0, time: float = 0.0, state: dict | None = None):
    """One implicit E-p update (eosmain). All arrays per element slice.

    Returns (p_new, e_new, c2_bulk).
    """
    kind = eos.kind.upper()
    p = eos.params
    rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)

    if kind in ("STIFF-GAS", "STIFF_GAS", "STIFFENED_GAS", "STIFFGAS", "SG"):
        gamma = p.get("gamma", 1.4)
        p_star = p.get("p_star", p.get("pstar", 0.0))
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)
        A, B = _coefficients_stiffgas(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A + 2.0 * psh)) / np.maximum(denom, 1e-6)
        p_raw = A + B * e_new
        p_new = np.maximum(p_raw, pmin - psh)
        eta = 1.0 + mu
        df = 1.0 / np.maximum(eta, 1e-12)
        dpdm = (gamma - 1.0) * e_new + (gamma - 1.0) * (df ** 2) * eta * (p_new + psh)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "GRUNEISEN":
        c = p.get("c", 0.0)
        s1 = p.get("s1", 0.0)
        s2 = p.get("s2", 0.0)
        s3 = p.get("s3", 0.0)
        gamma0 = p.get("gamma0", 0.0)
        a = p.get("a", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

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

    if kind == "TILLOTSON":
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

        eta = 1.0 + mu
        df = 1.0 / np.maximum(eta, 1e-12)
        xx = np.where(np.abs(eta) > 1e-12, mu / eta, 0.0)
        expa = np.exp(-alpha * (xx ** 2))
        expb = np.exp(beta * xx)

        hot0 = (mu < 0.0) & ((df > vs) | ((df <= vs) & (e_old >= es)))
        facc1_0 = np.where(hot0, expa * expb, 1.0)
        facc2_0 = np.where(mu >= 0.0, 1.0, 0.0)
        facpb_0 = np.where(hot0, expa, 1.0)
        A0 = facc1_0 * c1 * mu + facc2_0 * c2 * (mu ** 2)
        omega0 = 1.0 + e_old / np.maximum(er * (eta ** 2), 1e-15)
        B0 = (a + facpb_0 * b / omega0) * eta

        denom0 = 1.0 + 0.5 * B0 * dv
        e_pred = (e_old + de_other - 0.5 * dv * (p_old + A0)) / np.maximum(denom0, 1e-6)

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

    if kind == "JWL":
        a = p.get("a", 0.0)
        b = p.get("b", 0.0)
        r1 = p.get("r1", 0.0)
        r2 = p.get("r2", 0.0)
        omega = p.get("omega", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -psh)

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

    if kind == "MURNAGHAN":
        k0 = p.get("k0", 0.0)
        k1 = p.get("k1", 1.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        A, _ = _coefficients_murnaghan(eos, mu)
        p_tot = np.maximum(A + psh, pmin)
        p_new = p_tot - psh
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)

        eta = np.maximum(1.0 + mu, 1e-12)
        dpdm = k0 * np.power(eta, k1 - 1.0)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind in ("NOBLE-ABEL", "NOBLE_ABEL"):
        b = p.get("b", 0.0)
        gamma = p.get("gamma", 1.4)
        psh = p.get("psh", 0.0)

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

    if kind in ("NASG", "NOBLE-ABEL-STIFFENED-GAS", "NOBLE_ABEL_STIFFENED_GAS"):
        b = p.get("b", 0.0)
        gamma = p.get("gamma", 1.4)
        p_star = p.get("p_star", p.get("pstar", 0.0))
        q = p.get("q", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

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

    if kind == "PUFF":
        c1 = p.get("c1", 0.0)
        c2 = p.get("c2", 0.0)
        c3 = p.get("c3", 0.0)
        t1 = p.get("t1", p.get("sigt1", 0.0))
        if t1 == 0.0:
            t1 = c1
        t2 = p.get("t2", p.get("sigt2", 0.0))
        esubl = p.get("es", p.get("esubl", p.get("es_subl", 0.0)))
        gamma0 = p.get("gamma0", p.get("g0", p.get("gamma", 0.0)))
        h = p.get("h", p.get("hh", p.get("eoh", 0.0)))
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        A0, B0 = _coefficients_puff(eos, mu, e_old)
        denom0 = 1.0 + 0.5 * B0 * dv
        e_pred = (e_old + de_other - 0.5 * dv * (p_old + A0)) / np.maximum(denom0, 1e-6)

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
        denom_cc = gamma0 * esubl
        safe_denom_cc = denom_cc if abs(denom_cc) > 1e-12 else 1.0
        cc = np.where(np.abs(denom_cc) > 1e-12, c1 / safe_denom_cc, 0.0)
        expa = np.exp(np.clip(cc * xx, -50.0, 50.0))

        dpdm_comp = (c1 + 2.0 * c2 * mu + 3.0 * c3 * (mu ** 2)) * gx + gamma0 * (df ** 2) * (p_tot - 0.5 * aa_raw_comp)
        dpdm_cold = (t1 + 2.0 * t2 * mu) * gx + gamma0 * (df ** 2) * (p_tot - 0.5 * aa_raw_cold)
        dpdm_hot = bb_hot * (df ** 2) * (p_tot + esubl * expa * cc) + (e_new + esubl * (expa - 1.0)) * (h + 1.5 * ee * (gamma0 - h))

        dpdm = np.where(is_comp, dpdm_comp, np.where(is_cold, dpdm_cold, dpdm_hot))
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "LINEAR":
        c0 = p.get("c0", p.get("p0", 0.0))
        bulk = p.get("bulk", p.get("c1", 0.0))
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        p_new = np.maximum(c0 + bulk * mu - psh, pmin)
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)
        c2 = np.full_like(mu, bulk / rho0)
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "LSZK":
        gamma = p.get("gamma", 1.4)
        a = p.get("a", 0.0)
        b = p.get("b", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        A, B = _coefficients_lszk(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A + 2.0 * psh)) / np.maximum(denom, 1e-6)
        p_raw = A + B * e_new
        p_new = np.maximum(p_raw, pmin)

        eta = np.maximum(1.0 + mu, 1e-12)
        df = 1.0 / eta
        dpdm = (gamma - 1.0) * e_new + a * b * np.power(eta, b - 1.0) + (gamma - 1.0) * df * (p_new + psh)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind in ("OSBORNE", "OSBORN"):
        a1 = p.get("a1", 0.0)
        a2 = p.get("a2", 0.0)
        b0 = p.get("b0", 0.0)
        b1 = p.get("b1", 0.0)
        b2 = p.get("b2", 0.0)
        c0 = p.get("c0", 0.0)
        c1 = p.get("c1", 0.0)
        d0 = p.get("d0", 1.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        eta = np.maximum(1.0 + mu, 1e-12)
        df = 1.0 / eta
        dvv = 0.5 * dv * df

        # Step 1: Predictor
        a2_star = np.where(mu >= 0.0, a2, -a2)
        denom1 = np.maximum(e_old + d0, 1e-12)
        p1 = (a1 * mu + a2_star * (mu ** 2) + (b0 + b1 * mu + b2 * (mu ** 2)) * e_old + (c0 + c1 * mu) * (e_old ** 2)) / denom1
        p1 = np.maximum(p1, pmin)
        e1 = e_old + de_other - p1 * dvv

        # Step 2: Corrector
        denom2 = np.maximum(e1 + d0, 1e-12)
        p2 = (a1 * mu + a2_star * (mu ** 2) + (b0 + b1 * mu + b2 * (mu ** 2)) * e1 + (c0 + c1 * mu) * (e1 ** 2)) / denom2
        p2 = np.maximum(p2, pmin)
        e_new = e_old + de_other - 0.5 * dv * (p_old + p2 + 2.0 * psh)
        p_new = p2 - psh

        dpdmu_e = (a1 + 2.0 * a2_star * mu + (2.0 * b2 * mu + b1) * e_new + c1 * (e_new ** 2)) / denom2
        dpde = (((b2 * mu + b1) * mu + b0) + 2.0 * (c1 * mu + c0) * e_new - p2 / denom2) / denom2
        dpdm = dpdmu_e + dpde * (df ** 2) * p2
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "EXPONENTIAL":
        p0_param = p.get("p0", 0.0)
        alpha = p.get("alpha", 0.0)
        psh = p.get("psh", 0.0)

        p_net = (p0_param - psh) * math.exp(alpha * time)
        p_new = np.full_like(mu, p_net)
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)
        c2 = np.full_like(mu, 1e-20)
        return p_new, e_new, c2

    if kind in ("COMPACTION", "COMPACT"):
        c0 = p.get("c0", 0.0)
        c1 = p.get("c1", 0.0)
        c2 = p.get("c2", 0.0)
        c3 = p.get("c3", 0.0)
        iform = p.get("iform", 2)
        mumin = p.get("mumin", p.get("mue_min", 0.0))
        mumax = p.get("mumax", p.get("mue_max", 1e20))
        bunl = p.get("bunl", p.get("b", c1))
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        mu_bak = eos.params.get("_mu_bak")
        if mu_bak is None:
            mu_bak = np.copy(mu)
        else:
            mu_bak = np.maximum(mu_bak, mu)
        mu_bak = np.minimum(mumax, mu_bak)
        eos.params["_mu_bak"] = mu_bak

        A, _ = _coefficients_compaction(eos, mu, mu_bak)
        p_new = A
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)

        if iform == 1:
            b_slope = bunl
        else:
            alpha = np.where(mumax > 0.0, mu_bak / mumax, 1.0)
            b_slope = alpha * bunl + (1.0 - alpha) * c1
        dpdm_load = c1 + np.maximum(0.0, mu) * (2.0 * c2 + 3.0 * c3 * mu)
        dpdm = np.maximum(b_slope, dpdm_load)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "COMPACTION2":
        fscale = p.get("fscale", p.get("fscale_p", 1.0))
        xscale = p.get("xscale", p.get("xscale_p", 1.0))
        iform = p.get("iform", 2)
        mumin = p.get("mumin", p.get("mue_min", 0.0))
        mumax = p.get("mumax", p.get("mue_max", 1e20))
        bmin = p.get("bmin", p.get("bt", 1.0))
        bmax = p.get("bmax", p.get("b", bmin))
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        mu_bak = eos.params.get("_mu_bak")
        if mu_bak is None:
            mu_bak = np.copy(mu)
        else:
            mu_bak = np.maximum(mu_bak, mu)
        mu_bak = np.minimum(mumax, mu_bak)
        eos.params["_mu_bak"] = mu_bak

        A, _ = _coefficients_compaction2(eos, mu, mu_bak)
        p_new = A
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)

        if iform == 1:
            b_slope = bmax
        else:
            alpha = np.where(mumax > 0.0, mu_bak / mumax, 1.0)
            b_slope = alpha * bmax + (1.0 - alpha) * bmin

        p_func = p.get("p_func", p.get("func_p"))
        _, der_bak = _eval_funct_1d(p_func, xscale * mu_bak) if p_func is not None else (mu_bak, np.ones_like(mu_bak))
        dpdm = np.maximum(b_slope, der_bak * fscale * xscale)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "COMPACTION_TAB":
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)
        p_func = p.get("p_func", p.get("func_p"))
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        rho = rho0 * (1.0 + mu)
        rho_bak = eos.params.get("_rho_bak")
        if rho_bak is None:
            rho_bak = np.copy(rho)
        else:
            rho_bak = np.maximum(rho_bak, rho)
        eos.params["_rho_bak"] = rho_bak

        if p_func is not None:
            pc, dpdr = _eval_funct_1d(p_func, rho)
        else:
            pc = np.zeros_like(rho)
            dpdr = np.zeros_like(rho)

        p_new = np.maximum(pc, pmin) - psh
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)
        dpdm = rho0 * dpdr
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind in ("IDEAL-GAS-VT", "IDEALGAS_VT"):
        psh = p.get("psh", 0.0)
        A, B = _coefficients_idealgas_vt(eos, mu, e_old)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A + 2.0 * psh)) / np.maximum(denom, 1e-6)
        A1, B1 = _coefficients_idealgas_vt(eos, mu, e_new)
        p_new = A1 + B1 * e_new

        r_gas = p.get("r_gas", p.get("r", 287.0))
        temp = (p_new + psh) / np.maximum(rho0 * (1.0 + mu) * r_gas, 1e-12)
        a0 = p.get("a0", p.get("c0", 1000.0))
        a1 = p.get("a1", p.get("c1", 0.0))
        a2 = p.get("a2", p.get("c2", 0.0))
        a3 = p.get("a3", p.get("c3", 0.0))
        a4 = p.get("a4", p.get("c4", 0.0))
        cv = a0 + a1 * temp + a2 * (temp ** 2) + a3 * (temp ** 3) + a4 * (temp ** 4) - r_gas
        cp = cv + r_gas
        gamma = cp / np.maximum(cv, 1e-12)
        dpdm = rho0 * gamma * r_gas * temp
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "TABULATED":
        f_a = p.get("func_a", p.get("a_func"))
        f_b = p.get("func_b", p.get("b_func"))
        fscale_a = p.get("fscale_a", 1.0)
        fscale_b = p.get("fscale_b", 1.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        ya, der_a = _eval_funct_1d(f_a, mu) if f_a is not None else (np.zeros_like(mu), np.zeros_like(mu))
        yb, der_b = _eval_funct_1d(f_b, mu) if f_b is not None else (np.zeros_like(mu), np.zeros_like(mu))
        AA = fscale_a * ya
        BB = fscale_b * yb

        dvv = 0.5 * dv / np.maximum(1.0 + mu, 1e-12)
        denom = 1.0 + BB * dvv
        p_raw = (AA + BB * (e_old + de_other - psh * dvv)) / np.maximum(denom, 1e-6)
        p_new = np.maximum(p_raw, pmin) - psh
        e_new = e_old + de_other - 0.5 * dv * (p_old + p_new + 2.0 * psh)

        dpdm = der_a + der_b * e_new + BB * (p_new + psh) / np.maximum((1.0 + mu) ** 2, 1e-12)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind in ("IDEAL-GAS", "IDEAL_GAS"):
        gamma = p.get("gamma", p.get("c4", 0.4) + 1.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -psh)
        eta = 1.0 + mu
        A, B = _coefficients_idealgas(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A + 2.0 * psh)) / np.maximum(denom, 1e-6)
        e_new = np.maximum(e_new, 0.0)
        p_raw = A + B * e_new
        p_new = np.maximum(p_raw, pmin)

        df = 1.0 / np.maximum(eta, 1e-12)
        dpdm = (gamma - 1.0) * (e_new + (p_new + psh) * df)
        c2 = dpdm / rho0
        return p_new, e_new, np.maximum(c2, 0.0)

    if kind == "SESAME":
        # Upstream Fortran reference: common_source/eos/sesame.F
        rho_tab = p.get("rho_table", p.get("r_table"))
        theta_tab = p.get("theta_table", p.get("t_table"))
        p_tab = p.get("p_table")
        e_tab = p.get("e_table")
        filename = p.get("filename")
        if (rho_tab is None or p_tab is None or e_tab is None) and filename:
            try:
                table_dict = read_sesame_file(filename)
                p.update(table_dict)
                rho_tab = p.get("rho_table")
                theta_tab = p.get("theta_table")
                p_tab = p.get("p_table")
                e_tab = p.get("e_table")
            except Exception:
                pass

        pmin = p.get("pmin", -1e30)
        psh = p.get("psh", 0.0)

        n = len(mu)
        p_new = np.zeros(n, dtype=float)
        e_new = np.zeros(n, dtype=float)
        c2 = np.zeros(n, dtype=float)

        has_table = (rho_tab is not None and theta_tab is not None
                     and p_tab is not None and e_tab is not None)

        if has_table:
            r_arr = np.asarray(rho_tab, dtype=float)
            t_arr = np.asarray(theta_tab, dtype=float)
            p_mat = np.asarray(p_tab, dtype=float)
            e_mat = np.asarray(e_tab, dtype=float)

            for i in range(n):
                mu_i = float(mu[i])
                dv_i = float(dv[i])
                e_old_i = float(e_old[i])
                p_old_i = float(p_old[i])
                de_oth_i = float(de_other[i])

                rho_i = rho0 * (1.0 + mu_i)
                espem_old = e_old_i / max(rho0, 1e-12)

                # Pass 1: predictor at e_old
                t_pred, dtde_pred = mintp_re(r_arr, t_arr, e_mat, rho_i, espem_old)
                p_pred, dpdr_pred, dpdt_pred = mintp1_rt(r_arr, t_arr, p_mat, rho_i, t_pred)
                dpde_pred = dpdt_pred * dtde_pred / max(rho0, 1e-12)
                B0 = dpde_pred
                A0 = p_pred - B0 * e_old_i

                # Predictor energy
                denom0 = 1.0 + 0.5 * B0 * dv_i
                e_pred = (e_old_i + de_oth_i - 0.5 * dv_i * (p_old_i + A0)) / max(denom0, 1e-6)

                # Pass 2: corrector at e_pred
                espem_pred = e_pred / max(rho0, 1e-12)
                t_corr, dtde_corr = mintp_re(r_arr, t_arr, e_mat, rho_i, espem_pred)
                p_corr, dpdr_corr, dpdt_corr = mintp1_rt(r_arr, t_arr, p_mat, rho_i, t_corr)
                dpde_corr = dpdt_corr * dtde_corr / max(rho0, 1e-12)
                B1 = dpde_corr
                A1 = p_corr - B1 * e_pred

                denom1 = 1.0 + 0.5 * B1 * dv_i
                e_res = (e_old_i + de_oth_i - 0.5 * dv_i * (p_old_i + A1)) / max(denom1, 1e-6)
                p_raw = A1 + B1 * e_res
                p_tot = max(p_raw, pmin)
                p_res = p_tot - psh

                dpdm_corr = rho0 * dpdr_corr
                eta_i = max(1.0 + mu_i, 1e-12)
                dpdm_tot = dpdm_corr + (p_tot / (eta_i ** 2)) * dpde_corr
                c2_i = dpdm_tot / max(rho0, 1e-12)

                p_new[i] = p_res
                e_new[i] = e_res
                c2[i] = max(c2_i, 0.0)
                if state is not None:
                    state[f"theta_{i}"] = t_corr
            return p_new, e_new, c2
        else:
            k0 = float(p.get("k0", 2.0e9))
            gamma0 = float(p.get("gamma0", 1.4))
            B = (gamma0 - 1.0) * (1.0 + mu)
            A = k0 * mu - psh
            denom = 1.0 + 0.5 * B * dv
            e_new = (e_old + de_other - 0.5 * dv * (p_old + A)) / np.maximum(denom, 1e-6)
            p_raw = A + B * e_new
            p_tot = np.maximum(p_raw, pmin)
            p_new = p_tot - psh
            eta = np.maximum(1.0 + mu, 1e-12)
            c2 = (k0 + (p_tot / (eta ** 2)) * B) / max(rho0, 1e-12)
            return p_new, e_new, np.maximum(c2, 0.0)

    if kind in ("POLYNOMIAL", "POLY"):
        c1 = p.get("c1", 0.0)
        c2 = p.get("c2", 0.0)
        c3 = p.get("c3", 0.0)
        c5 = p.get("c5", 0.0)
        c6 = p.get("c6", 0.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)

        mu_pos = np.maximum(mu, 0.0)
        A, B = _coefficients_polynomial(eos, mu)
        denom = 1.0 + 0.5 * B * dv
        e_new = (e_old + de_other - 0.5 * dv * (p_old + A + 2.0 * psh)) / np.maximum(denom, 1e-6)
        p_raw = A + B * e_new
        p_new = np.maximum(p_raw + psh, pmin) - psh

        eta = 1.0 + mu
        df = 1.0 / np.maximum(eta, 1e-12)
        dpdm = (c1 + 2.0 * c2 * mu_pos + 3.0 * c3 * (mu ** 2)
                + (c5 + c6 * mu_pos) * e_new
                + B * (df ** 2) * (p_new + psh))
        c2_bulk = dpdm / rho0
        return p_new, e_new, np.maximum(c2_bulk, 0.0)

    # General default fallback
    A, B = coefficients(eos, mu)
    denom = 1.0 + 0.5 * B * dv
    e_new = (e_old + de_other - 0.5 * dv * (p_old + A)) / np.maximum(denom, 1e-6)
    p_new = A + B * e_new

    mubar = np.maximum(mu, 0.0)
    dpdmu = p.get("c1", 0.0) + 2.0 * p.get("c2", 0.0) * mubar + 3.0 * p.get("c3", 0.0) * (mu ** 2) + p.get("c5", 0.0) * e_new
    c2 = (dpdmu + B * p_new / (1.0 + mu) ** 2) / rho0
    return p_new, e_new, np.maximum(c2, 0.0)


# ============================================================================
# Pressure & Sound Speed Accessors
# ============================================================================

def pressure(eos, mu: np.ndarray | float, e: np.ndarray | float, time: float = 0.0) -> np.ndarray | float:
    """Evaluate EOS pressure p(mu, E, time) at given compression mu and internal energy E."""
    is_scalar = np.isscalar(mu) and np.isscalar(e)
    mu_arr = np.asarray(mu, dtype=float)
    e_arr = np.asarray(e, dtype=float)
    kind = eos.kind.upper()

    if kind in ("STIFF-GAS", "STIFF_GAS", "STIFFENED_GAS", "STIFFGAS", "SG"):
        A, B = _coefficients_stiffgas(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A + B * e_arr, pmin - psh)
        return float(p_val) if is_scalar else p_val

    if kind == "GRUNEISEN":
        A, B = _coefficients_gruneisen(eos, mu_arr)
        p_val = A + B * e_arr
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(p_val, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if kind == "TILLOTSON":
        A, B = _coefficients_tillotson(eos, mu_arr, e_arr)
        p_val = A + B * e_arr
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(p_val, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if kind == "JWL":
        A, B = _coefficients_jwl(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        p_val = np.maximum(A + B * e_arr, -psh)
        return float(p_val) if is_scalar else p_val

    if kind == "MURNAGHAN":
        A, _ = _coefficients_murnaghan(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A + psh, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if kind in ("NOBLE-ABEL", "NOBLE_ABEL"):
        A, B = _coefficients_noble_abel(eos, mu_arr)
        p_val = A + B * e_arr
        return float(p_val) if is_scalar else p_val

    if kind in ("NASG", "NOBLE-ABEL-STIFFENED-GAS", "NOBLE_ABEL_STIFFENED_GAS"):
        A, B = _coefficients_nasg(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        gamma = eos.params.get("gamma", 1.4)
        p_star = eos.params.get("p_star", eos.params.get("pstar", 0.0))
        p_val = np.maximum(A + B * e_arr + psh, np.maximum(pmin, -gamma * p_star)) - psh
        return float(p_val) if is_scalar else p_val

    if kind == "PUFF":
        A, B = _coefficients_puff(eos, mu_arr, e_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A + B * e_arr + psh, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if kind in ("OSBORNE", "OSBORN"):
        p = eos.params
        a1 = p.get("a1", 0.0)
        a2 = p.get("a2", 0.0)
        b0 = p.get("b0", 0.0)
        b1 = p.get("b1", 0.0)
        b2 = p.get("b2", 0.0)
        c0 = p.get("c0", 0.0)
        c1 = p.get("c1", 0.0)
        d0 = p.get("d0", 1.0)
        psh = p.get("psh", 0.0)
        pmin = p.get("pmin", -1e30)
        a2_star = np.where(mu_arr >= 0.0, a2, -a2)
        denom = np.maximum(e_arr + d0, 1e-12)
        p_raw = (a1 * mu_arr + a2_star * (mu_arr ** 2) + (b0 + b1 * mu_arr + b2 * (mu_arr ** 2)) * e_arr + (c0 + c1 * mu_arr) * (e_arr ** 2)) / denom
        p_val = np.maximum(p_raw, pmin) - psh
        return float(p_val) if is_scalar else p_val

    if kind == "LSZK":
        A, B = _coefficients_lszk(eos, mu_arr)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A + B * e_arr, pmin)
        return float(p_val) if is_scalar else p_val

    if kind == "LINEAR":
        A, _ = _coefficients_linear(eos, mu_arr)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A, pmin)
        return float(p_val) if is_scalar else p_val

    if kind == "EXPONENTIAL":
        p0_param = eos.params.get("p0", 0.0)
        alpha = eos.params.get("alpha", 0.0)
        psh = eos.params.get("psh", 0.0)
        p_val = (p0_param - psh) * math.exp(alpha * time)
        if not is_scalar:
            p_val = np.full_like(mu_arr, p_val)
        return float(p_val) if is_scalar else p_val

    if kind in ("COMPACTION", "COMPACT"):
        A, _ = _coefficients_compaction(eos, mu_arr)
        return float(A) if is_scalar else A

    if kind == "COMPACTION2":
        A, _ = _coefficients_compaction2(eos, mu_arr)
        return float(A) if is_scalar else A

    if kind == "COMPACTION_TAB":
        A, _ = _coefficients_compaction_tab(eos, mu_arr)
        return float(A) if is_scalar else A

    if kind in ("IDEAL-GAS-VT", "IDEALGAS_VT"):
        A, B = _coefficients_idealgas_vt(eos, mu_arr, e_arr)
        p_val = A + B * e_arr
        return float(p_val) if is_scalar else p_val

    if kind == "TABULATED":
        A, B = _coefficients_tabulated(eos, mu_arr)
        pmin = eos.params.get("pmin", -1e30)
        p_val = np.maximum(A + B * e_arr, pmin)
        return float(p_val) if is_scalar else p_val

    if kind in ("IDEAL-GAS", "IDEAL_GAS"):
        A, B = _coefficients_idealgas(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -psh)
        p_val = np.maximum(A + B * e_arr, pmin)
        return float(p_val) if is_scalar else p_val

    if kind == "SESAME":
        # Upstream Fortran reference: common_source/eos/sesame.F
        rho_tab = eos.params.get("rho_table", eos.params.get("r_table"))
        theta_tab = eos.params.get("theta_table", eos.params.get("t_table"))
        p_tab = eos.params.get("p_table")
        e_tab = eos.params.get("e_table")
        filename = eos.params.get("filename")
        if (rho_tab is None or p_tab is None or e_tab is None) and filename:
            try:
                table_dict = read_sesame_file(filename)
                eos.params.update(table_dict)
                rho_tab = eos.params.get("rho_table")
                theta_tab = eos.params.get("theta_table")
                p_tab = eos.params.get("p_table")
                e_tab = eos.params.get("e_table")
            except Exception:
                pass

        pmin = eos.params.get("pmin", -1e30)
        psh = eos.params.get("psh", 0.0)
        rho0 = getattr(eos, "rho0", None) or eos.params.get("rho0_card", 1.0)

        has_table = (rho_tab is not None and theta_tab is not None
                     and p_tab is not None and e_tab is not None)

        if has_table:
            r_arr = np.asarray(rho_tab, dtype=float)
            t_arr = np.asarray(theta_tab, dtype=float)
            p_mat = np.asarray(p_tab, dtype=float)
            e_mat = np.asarray(e_tab, dtype=float)

            if is_scalar:
                rho_val = rho0 * (1.0 + float(mu))
                e_spec = float(e) / max(rho0, 1e-12)
                t_val, _ = mintp_re(r_arr, t_arr, e_mat, rho_val, e_spec)
                p_val, _, _ = mintp1_rt(r_arr, t_arr, p_mat, rho_val, t_val)
                p_tot = max(p_val, pmin)
                return float(p_tot - psh)
            else:
                res = np.zeros_like(mu_arr)
                for i in range(len(mu_arr)):
                    rho_i = rho0 * (1.0 + float(mu_arr[i]))
                    e_spec = float(e_arr[i]) / max(rho0, 1e-12)
                    t_val, _ = mintp_re(r_arr, t_arr, e_mat, rho_i, e_spec)
                    p_val, _, _ = mintp1_rt(r_arr, t_arr, p_mat, rho_i, t_val)
                    res[i] = max(p_val, pmin) - psh
                return res
        else:
            k0 = float(eos.params.get("k0", 2.0e9))
            gamma0 = float(eos.params.get("gamma0", 1.4))
            p_val = k0 * mu_arr + (gamma0 - 1.0) * (1.0 + mu_arr) * e_arr
            p_tot = np.maximum(p_val, pmin) - psh
            return float(p_tot) if is_scalar else p_tot

    if kind in ("POLYNOMIAL", "POLY"):
        A, B = _coefficients_polynomial(eos, mu_arr)
        psh = eos.params.get("psh", 0.0)
        pmin = eos.params.get("pmin", -1e30)
        p_raw = A + B * e_arr
        p_val = np.maximum(p_raw + psh, pmin) - psh
        return float(p_val) if is_scalar else p_val

    A, B = coefficients(eos, mu_arr)
    p_val = A + B * e_arr
    return float(p_val) if is_scalar else p_val


def sound_speed(eos, mu: np.ndarray | float, e: np.ndarray | float, time: float = 0.0) -> np.ndarray | float:
    """Evaluate bulk sound speed c = sqrt(max(c2_bulk, 0)) at given mu and energy E."""
    is_scalar = np.isscalar(mu) and np.isscalar(e)
    mu_arr = np.atleast_1d(np.asarray(mu, dtype=float))
    e_arr = np.atleast_1d(np.asarray(e, dtype=float))
    dv = np.zeros_like(mu_arr)
    p_arr = np.atleast_1d(np.asarray(pressure(eos, mu_arr, e_arr, time), dtype=float))
    de = np.zeros_like(mu_arr)
    _, _, c2 = update(eos, mu_arr, dv, e_arr, p_arr, de, time=time)
    c = np.atleast_1d(np.sqrt(np.maximum(c2, 0.0)))
    return float(c[0]) if is_scalar else c


# ============================================================================
# Initial State
# ============================================================================

def initial_state(eos):
    """(e0, p0) at reference state mu = 0."""
    kind = eos.kind.upper()
    p = eos.params

    if kind in ("STIFF-GAS", "STIFF_GAS", "STIFFENED_GAS", "STIFFGAS", "SG"):
        gamma = p.get("gamma", 1.4)
        p_star = p.get("p_star", p.get("pstar", 0.0))
        p0_param = p.get("p0", 0.0)
        psh = p.get("psh", 0.0)
        e0 = p.get("e0")
        if e0 is None:
            e0 = (p0_param + gamma * p_star) / (gamma - 1.0) if gamma > 1.0 else 0.0
        p0 = p0_param - psh if p0_param > 0.0 else (gamma - 1.0) * e0 - gamma * p_star - psh
        return e0, p0

    if kind == "GRUNEISEN":
        e0 = p.get("e0", 0.0)
        gamma0 = p.get("gamma0", 0.0)
        p0_param = p.get("p0", 0.0)
        if p0_param > 0.0 and e0 == 0.0 and gamma0 > 0.0:
            e0 = p0_param / gamma0
        psh = p.get("psh", 0.0)
        p0 = gamma0 * e0 - psh
        return e0, p0

    if kind == "TILLOTSON":
        e0 = p.get("e0", 0.0)
        a = p.get("a", 0.0)
        b = p.get("b", 0.0)
        er = p.get("er", p.get("ezero", p.get("e0_ref", 1.0)))
        omega = 1.0 + e0 / er if er > 0.0 else 1.0
        psh = p.get("psh", 0.0)
        p0 = (a + b / omega) * e0 - psh
        return e0, p0

    if kind == "JWL":
        e0 = p.get("e0", 0.0)
        p0 = pressure(eos, 0.0, e0)
        return e0, p0

    if kind == "MURNAGHAN":
        e0 = 0.0
        p0 = p.get("p0", 0.0) - p.get("psh", 0.0)
        return e0, p0

    if kind in ("NOBLE-ABEL", "NOBLE_ABEL"):
        e0 = p.get("e0", 0.0)
        p0 = pressure(eos, 0.0, e0)
        return e0, p0

    if kind in ("NASG", "NOBLE-ABEL-STIFFENED-GAS", "NOBLE_ABEL_STIFFENED_GAS"):
        b = p.get("b", 0.0)
        gamma = p.get("gamma", 1.4)
        p_star = p.get("p_star", p.get("pstar", 0.0))
        q = p.get("q", 0.0)
        p0_param = p.get("p0", 0.0)
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)
        e0 = p.get("e0", None)
        if e0 is None or e0 == 0.0:
            e0 = (p0_param + gamma * p_star) * (1.0 - rho0 * b) / (gamma - 1.0) + rho0 * q
        p0 = p0_param - p.get("psh", 0.0)
        return e0, p0

    if kind == "PUFF":
        e0 = p.get("e0", 0.0)
        p0 = pressure(eos, 0.0, e0)
        return e0, p0

    if kind in ("OSBORNE", "OSBORN"):
        p0_param = p.get("p0", 0.0)
        psh = p.get("psh", 0.0)
        e0 = p.get("e0")
        if e0 is None or e0 == 0.0:
            b0 = p.get("b0", 0.0)
            c0 = p.get("c0", 1e-10)
            d0 = p.get("d0", 1.0)
            # Root of C0*E0^2 + (B0 - P0)*E0 - P0*D0 = 0
            delta = (b0 - p0_param) ** 2 + 4.0 * c0 * d0 * p0_param
            if delta >= 0.0 and c0 > 0.0:
                e0 = (- (b0 - p0_param) + math.sqrt(delta)) / (2.0 * c0)
            else:
                e0 = 0.0
        p0 = p0_param - psh if p0_param > 0.0 else float(pressure(eos, 0.0, e0))
        return e0, p0

    if kind == "LSZK":
        p0_param = p.get("p0", 0.0)
        gamma = p.get("gamma", 1.4)
        a = p.get("a", 0.0)
        psh = p.get("psh", 0.0)
        e0 = (p0_param - a) / (gamma - 1.0) if gamma > 1.0 else 0.0
        p0 = p0_param - psh
        return e0, p0

    if kind == "LINEAR":
        c0 = p.get("c0", p.get("p0", 0.0))
        psh = p.get("psh", 0.0)
        return 0.0, c0 - psh

    if kind == "EXPONENTIAL":
        p0_param = p.get("p0", 0.0)
        psh = p.get("psh", 0.0)
        return 0.0, p0_param - psh

    if kind in ("COMPACTION", "COMPACT"):
        c0 = p.get("c0", 0.0)
        psh = p.get("psh", 0.0)
        return 0.0, c0 - psh

    if kind == "COMPACTION2":
        psh = p.get("psh", 0.0)
        return 0.0, -psh

    if kind == "COMPACTION_TAB":
        psh = p.get("psh", 0.0)
        p_func = p.get("p_func")
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)
        p0_val = 0.0
        if p_func is not None:
            y, _ = _eval_funct_1d(p_func, rho0)
            p0_val = float(y)
        return 0.0, p0_val - psh

    if kind in ("IDEAL-GAS-VT", "IDEALGAS_VT"):
        t0 = p.get("t0", 300.0)
        r_gas = p.get("r_gas", p.get("r", 287.0))
        rho0 = getattr(eos, "rho0", None) or p.get("rho0_card", 1.0)
        a0 = p.get("a0", p.get("c0", 1000.0))
        a1 = p.get("a1", p.get("c1", 0.0))
        a2 = p.get("a2", p.get("c2", 0.0))
        a3 = p.get("a3", p.get("c3", 0.0))
        a4 = p.get("a4", p.get("c4", 0.0))
        # e = int_0^T0 (Cp - r) dT
        e_spec = a0 * t0 + 0.5 * a1 * (t0 ** 2) + (1.0 / 3.0) * a2 * (t0 ** 3) + 0.25 * a3 * (t0 ** 4) + 0.2 * a4 * (t0 ** 5) - r_gas * t0
        e0 = rho0 * e_spec
        psh = p.get("psh", 0.0)
        p0 = rho0 * r_gas * t0 - psh
        return e0, p0

    if kind == "TABULATED":
        e0 = p.get("e0", 0.0)
        psh = p.get("psh", 0.0)
        return e0, -psh

    if kind == "POWDER-BURN":
        eg = p.get("eg", 0.0)
        p0 = p.get("p0", 0.0) - p.get("psh", 0.0)
        return eg, p0

    if kind == "SESAME":
        # Upstream Fortran reference: starter/source/materials/eos/hm_read_eos_sesame.F lines 160-171
        e0 = p.get("e0", 0.0)
        psh = p.get("psh", 0.0)
        rho_tab = p.get("rho_table", p.get("r_table"))
        p_tab = p.get("p_table")
        e_tab = p.get("e_table")
        theta_tab = p.get("theta_table", p.get("t_table"))
        filename = p.get("filename")
        if (rho_tab is None or p_tab is None or e_tab is None) and filename:
            try:
                table_dict = read_sesame_file(filename)
                p.update(table_dict)
                rho_tab = p.get("rho_table")
                theta_tab = p.get("theta_table")
                p_tab = p.get("p_table")
                e_tab = p.get("e_table")
            except Exception:
                pass
        has_table = (rho_tab is not None and theta_tab is not None
                     and p_tab is not None and e_tab is not None)
        if has_table:
            p0 = pressure(eos, 0.0, e0)
        else:
            p0 = p.get("p0", 0.0) - psh
        return e0, p0

    if kind in ("IDEAL-GAS", "IDEAL_GAS"):
        gamma = p.get("gamma", p.get("c4", 0.4) + 1.0)
        p0_param = p.get("p0", 0.0)
        psh = p.get("psh", 0.0)
        e0 = p.get("e0")
        if e0 is None:
            e0 = p0_param / (gamma - 1.0) if gamma > 1.0 else 0.0
        p0 = p0_param - psh if p0_param > 0.0 else (gamma - 1.0) * e0 - psh
        return e0, p0

    if kind in ("POLYNOMIAL", "POLY"):
        c0 = p.get("c0", 0.0)
        c4 = p.get("c4", 0.0)
        e0 = p.get("e0", 0.0)
        psh = p.get("psh", 0.0)
        p0 = (c0 - psh) + c4 * e0
        return e0, p0

    e0 = eos.params.get("e0", 0.0)
    p0 = eos.params.get("c0", 0.0) + eos.params.get("c4", 0.0) * e0
    return e0, p0
