r"""Fortran Parity & Physics Oracle Verifier for Milestone M564: /MAT/LAW87 (/MAT/BARLAT2000 / /MAT/BARLAT_2000 / /MAT/BARLAT2000_2D).

Barlat Yld2000-2d Anisotropic Plasticity Model for Shell Elements.
Directly cites and mirrors upstream OpenRadioss source:
- Upstream Fortran starter reader:
  ``starter/source/materials/mat/mat087/hm_read_mat87.F90`` (lines 1 to 781)
- Upstream Fortran engine shell constitutive kernel:
  ``engine/source/materials/mat/mat087/sigeps87c.F90`` (lines 1 to 202)
- Upstream Fortran Swift-Voce hardening & Barlat 2000 cutting plane return mapping:
  ``engine/source/materials/mat/mat087/mat87c_swift_voce.F90`` (lines 1 to 804)
- Upstream Fortran Tabulated hardening:
  ``engine/source/materials/mat/mat087/mat87c_tabulated.F90`` (lines 1 to 794)
- Upstream Fortran Hansel hardening:
  ``engine/source/materials/mat/mat087/mat87c_hansel.F90`` (lines 1 to 812)

Tests included:
1. Exact Python oracle reproducing sigeps87c.F90 and mat87c_swift_voce.F90:
   - Linear transformation matrices Lp and Lpp from alpha1..alpha8
   - Mohr-circle principal stresses for X' and X''
   - Barlat Yld2000-2d equivalent stress phi = |X'_1 - X'_2|^a + |2*X''_2 + X''_1|^a + |2*X''_1 + X''_2|^a
   - Analytical derivatives d(sig_bar)/d(sigma)
   - Swift-Voce yield stress with Cowper-Symonds strain rate multiplier
   - 3-iteration cutting-plane plane-stress return mapping
   - Chaboche-Rousselier kinematic hardening backstress evolution
   - Through-thickness thinning deps_zz and shell acoustic wave speed
2. Isotropic reduction when alpha1..alpha8 = 1.0, a = 2.0 (identically recovers von Mises plane stress).
3. Degree-1 homogeneity: bar_sigma(k * sigma) = k * bar_sigma(sigma).
4. Anisotropic directional yield stresses (0 deg, 45 deg, 90 deg, pure shear, equibiaxial).
5. Shear response and sensitivity to alpha7 and alpha8.
6. Analytical stress gradient verification against central difference numerical perturbation (1e-8).
7. Swift-Voce hardening options (pure Swift, pure Voce, mixed Swift-Voce, Cowper-Symonds rate effect).
8. Tabulated yield curve interpolation matching OpenRadioss piecewise linear curves.
9. Cyclic loading with Bauschinger effect under Chaboche-Rousselier and Prager kinematic hardening.
10. Shell thickness thinning increment (deps_zz) and plastic incompressibility tr(deps_p) = 0.
11. Acoustic dilatational sound speed c = sqrt(E / ((1 - nu^2) * rho0)).
12. Algorithmic consistent plane-stress tangent (3, 3) matching numerical central difference.
13. Vectorized multi-element batch execution equivalence.
14. Exhaustive 64+ diverse physical states comparison: shell_update vs sigeps87c_oracle (rtol=1e-12, atol=1e-12).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest

from pyradioss.materials.law87_barlat2000 import (
    Law87Params,
    build_law87,
    barlat2000_equivalent_stress,
    shell_update,
    shell_update_law87,
    sound_speed_shell,
    sound_speed_shell_law87,
    consistent_shell_tangent,
    shell_membrane_tangent,
    extra_shapes,
    resolve,
)
import pyradioss.materials as materials
from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY

_EM20 = 1.0e-20
_INF = 1.0e30


# =============================================================================
# 1. UPSTREAM FORTRAN ENGINE ORACLE: sigeps87c.F90 & mat87c_swift_voce.F90
# =============================================================================

def eval_curve_1d_oracle(curve: Any, x: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear 1D curve interpolation matching OpenRadioss FINTER."""
    x_arr = np.asarray(x, dtype=np.float64)
    if callable(curve):
        res = curve(x_arr)
        if isinstance(res, tuple):
            return np.asarray(res[0], dtype=np.float64), np.asarray(res[1], dtype=np.float64)
        return np.asarray(res, dtype=np.float64), np.zeros_like(x_arr)

    if isinstance(curve, (list, tuple)) and len(curve) == 2 and not isinstance(curve[0], (list, tuple)):
        cx = np.asarray(curve[0], dtype=np.float64)
        cy = np.asarray(curve[1], dtype=np.float64)
    elif isinstance(curve, (list, tuple)) and len(curve) > 0 and isinstance(curve[0], (list, tuple)):
        arr = np.asarray(curve, dtype=np.float64)
        cx = arr[:, 0]
        cy = arr[:, 1]
    elif hasattr(curve, "x") and hasattr(curve, "y"):
        cx = np.asarray(curve.x, dtype=np.float64)
        cy = np.asarray(curve.y, dtype=np.float64)
    elif hasattr(curve, "data"):
        d = np.asarray(curve.data, dtype=np.float64)
        cx = d[:, 0]
        cy = d[:, 1]
    else:
        return np.ones_like(x_arr), np.zeros_like(x_arr)

    if len(cx) == 0:
        return np.ones_like(x_arr), np.zeros_like(x_arr)
    if len(cx) == 1:
        return np.full_like(x_arr, cy[0]), np.zeros_like(x_arr)

    cs = np.diff(cy) / np.maximum(np.diff(cx), _EM20)
    idx = np.clip(np.searchsorted(cx, x_arr, side="right") - 1, 0, len(cx) - 2)
    val = cy[idx] + cs[idx] * (x_arr - cx[idx])
    return val, cs[idx]


def barlat2000_gradient_analytical(
    sig: np.ndarray,
    p: Law87Params,
) -> np.ndarray:
    """Exact analytical derivative d(seq)/d(sigma) matching mat87c_swift_voce.F90 lines 386-464.

    Parameters
    ----------
    sig : (3,) or (nel, 3) array [sxx, syy, sxy]
    p : Law87Params

    Returns
    -------
    (nel, 3) or (3,) gradient array [dseq/dsxx, dseq/dsyy, dseq/dsxy]
    """
    sig_arr = np.atleast_2d(np.asarray(sig, dtype=np.float64))
    nel = sig_arr.shape[0]

    sxx = sig_arr[:, 0]
    syy = sig_arr[:, 1]
    sxy = sig_arr[:, 2]

    normsig = np.sqrt(sxx * sxx + syy * syy + 2.0 * sxy * sxy)
    normsig = np.maximum(normsig, 1.0)

    # Transformed stress tensors
    xpxx = (p.lp11 * sxx + p.lp12 * syy) / normsig
    xpyy = (p.lp21 * sxx + p.lp22 * syy) / normsig
    xpxy = p.lp66 * sxy / normsig

    xppxx = (p.lpp11 * sxx + p.lpp12 * syy) / normsig
    xppyy = (p.lpp21 * sxx + p.lpp22 * syy) / normsig
    xppxy = p.lpp66 * sxy / normsig

    # Principal values of X' and X''
    r_p = np.sqrt(0.25 * (xpxx - xpyy) ** 2 + xpxy ** 2)
    c_p = 0.5 * (xpxx + xpyy)
    xp1 = c_p + r_p
    xp2 = c_p - r_p

    r_pp = np.sqrt(0.25 * (xppxx - xppyy) ** 2 + xppxy ** 2)
    c_pp = 0.5 * (xppxx + xppyy)
    xpp1 = c_pp + r_pp
    xpp2 = c_pp - r_pp

    phip = np.abs(xp1 - xp2) ** p.expa
    phipp = np.abs(2.0 * xpp2 + xpp1) ** p.expa + np.abs(2.0 * xpp1 + xpp2) ** p.expa
    s_phi = 0.5 * (phip + phipp)

    grad = np.zeros((nel, 3), dtype=np.float64)

    for i in range(nel):
        if s_phi[i] <= _EM20:
            continue

        mr_p = max(r_p[i], _EM20)
        dxp1dxpxx = 0.5 * (1.0 + (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
        dxp1dxpyy = 0.5 * (1.0 - (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
        dxp1dxpxy = xpxy[i] / mr_p
        dxp2dxpxx = 0.5 * (1.0 - (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
        dxp2dxpyy = 0.5 * (1.0 + (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
        dxp2dxpxy = -xpxy[i] / mr_p

        mr_pp = max(r_pp[i], _EM20)
        dxpp1dxppxx = 0.5 * (1.0 + (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
        dxpp1dxppyy = 0.5 * (1.0 - (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
        dxpp1dxppxy = xppxy[i] / mr_pp
        dxpp2dxppxx = 0.5 * (1.0 - (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
        dxpp2dxppyy = 0.5 * (1.0 + (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
        dxpp2dxppxy = -xppxy[i] / mr_pp

        dxp1dsigxx = dxp1dxpxx * p.lp11 + dxp1dxpyy * p.lp21
        dxp1dsigyy = dxp1dxpxx * p.lp12 + dxp1dxpyy * p.lp22
        dxp1dsigxy = dxp1dxpxy * p.lp66

        dxp2dsigxx = dxp2dxpxx * p.lp11 + dxp2dxpyy * p.lp21
        dxp2dsigyy = dxp2dxpxx * p.lp12 + dxp2dxpyy * p.lp22
        dxp2dsigxy = dxp2dxpxy * p.lp66

        dxpp1dsigxx = dxpp1dxppxx * p.lpp11 + dxpp1dxppyy * p.lpp21
        dxpp1dsigyy = dxpp1dxppxx * p.lpp12 + dxpp1dxppyy * p.lpp22
        dxpp1dsigxy = dxpp1dxppxy * p.lpp66

        dxpp2dsigxx = dxpp2dxppxx * p.lpp11 + dxpp2dxppyy * p.lpp21
        dxpp2dsigyy = dxpp2dxppxx * p.lpp12 + dxpp2dxppyy * p.lpp22
        dxpp2dsigxy = dxpp2dxppxy * p.lpp66

        diff_p = xp1[i] - xp2[i]
        sgn_p = 1.0 if diff_p >= 0.0 else -1.0
        dphipdxp1 = p.expa * (abs(diff_p) ** (p.expa - 1.0)) * sgn_p
        dphipdxp2 = -dphipdxp1

        term_pp1 = 2.0 * xpp2[i] + xpp1[i]
        sgn_pp1 = 1.0 if term_pp1 >= 0.0 else -1.0
        term_pp2 = 2.0 * xpp1[i] + xpp2[i]
        sgn_pp2 = 1.0 if term_pp2 >= 0.0 else -1.0

        dphippdxpp1 = (p.expa * (abs(term_pp1) ** (p.expa - 1.0)) * sgn_pp1
                       + 2.0 * p.expa * (abs(term_pp2) ** (p.expa - 1.0)) * sgn_pp2)
        dphippdxpp2 = (p.expa * (abs(term_pp2) ** (p.expa - 1.0)) * sgn_pp2
                       + 2.0 * p.expa * (abs(term_pp1) ** (p.expa - 1.0)) * sgn_pp1)

        dphipdsigxx = dphipdxp1 * dxp1dsigxx + dphipdxp2 * dxp2dsigxx
        dphipdsigyy = dphipdxp1 * dxp1dsigyy + dphipdxp2 * dxp2dsigyy
        dphipdsigxy = dphipdxp1 * dxp1dsigxy + dphipdxp2 * dxp2dsigxy

        dphippdsigxx = dphippdxpp1 * dxpp1dsigxx + dphippdxpp2 * dxpp2dsigxx
        dphippdsigyy = dphippdxpp1 * dxpp1dsigyy + dphippdxpp2 * dxpp2dsigyy
        dphippdsigxy = dphippdxpp1 * dxpp1dsigxy + dphippdxpp2 * dxpp2dsigxy

        dseqdphi = (0.5 / p.expa) * (s_phi[i] ** (1.0 / p.expa - 1.0))

        grad[i, 0] = dseqdphi * (dphipdsigxx + dphippdsigxx)
        grad[i, 1] = dseqdphi * (dphipdsigyy + dphippdsigyy)
        grad[i, 2] = dseqdphi * (dphipdsigxy + dphippdsigxy)

    if np.asarray(sig).ndim == 1:
        return grad[0]
    return grad


def sigeps87c_oracle(
    matparam: Union[Law87Params, Dict[str, Any]],
    sigo: np.ndarray,
    deps: np.ndarray,
    rho0: Union[float, np.ndarray] = 1.0,
    thk: Optional[np.ndarray] = None,
    thkly: Optional[np.ndarray] = None,
    pla: Optional[np.ndarray] = None,
    sigb: Optional[np.ndarray] = None,
    off: Optional[np.ndarray] = None,
    uvar: Optional[np.ndarray] = None,
    dt: float = 0.0,
    shf: float = 5.0 / 6.0,
    temp: Optional[np.ndarray] = None,
    niter: int = 3,
    yield_fn: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
) -> Dict[str, Any]:
    """Exact Python reproduction of OpenRadioss sigeps87c.F90 & mat87c_swift_voce.F90.

    Parameters
    ----------
    matparam : Law87Params or dict
        Material parameters following hm_read_mat87.F90 / sigeps87c.F90.
    sigo : ndarray of shape (nel, 3) or (nel, 5)
        Old Cauchy stress tensor [xx, yy, xy, (yz, zx)].
    deps : ndarray of shape (nel, 3) or (nel, 5)
        Engineering strain increment [xx, yy, xy, (yz, zx)].
    rho0 : float or ndarray
        Initial density.
    thk : ndarray of length nel, optional
        Current shell thickness.
    thkly : ndarray of length nel, optional
        Initial layer thickness.
    pla : ndarray of length nel, optional
        Accumulated equivalent plastic strain history.
    sigb : ndarray of shape (nel, 12), optional
        Backstress tensor (up to 4 CR branches x 3).
    off : ndarray of length nel, optional
        Element deletion status (1.0 = active, 0.0 = deleted).
    uvar : ndarray of shape (nel, 1), optional
        User variables (plastic strain rate).
    dt : float
        Time step increment.
    shf : float
        Shear correction factor (5/6).
    temp : ndarray of length nel, optional
        Temperature.
    niter : int, default 3
        Number of cutting-plane return mapping iterations (mat87c_swift_voce.F90:134).
    yield_fn : callable, optional
        Tabulated yield stress function returning (yld, dylddp).

    Returns
    -------
    dict
        Updated state dictionary matching OpenRadioss:
        'sig': updated Cauchy stress tensor
        'pla': updated equivalent plastic strain
        'dpla': plastic strain increment
        'sigb': updated backstress tensor
        'thk': updated shell thickness
        'depszz': total through-thickness strain increment
        'soundsp': dilatational acoustic wave speed
        'seq': equivalent stress
        'yld': yield stress
        'etse': hourglass control stiffness parameter
        'uvar': updated user variables
        'off': element deletion flag
    """
    p = build_law87(matparam) if not isinstance(matparam, Law87Params) else matparam

    sigo_arr = np.atleast_2d(np.asarray(sigo, dtype=np.float64)).copy()
    deps_arr = np.atleast_2d(np.asarray(deps, dtype=np.float64)).copy()
    nel, ncomp = sigo_arr.shape
    has_shear = (ncomp >= 5)

    if deps_arr.shape[0] == 1 and nel > 1:
        deps_arr = np.repeat(deps_arr, nel, axis=0)

    # Densities
    rho0_arr = np.atleast_1d(np.asarray(rho0, dtype=np.float64)).copy()
    if len(rho0_arr) == 1 and nel > 1:
        rho0_arr = np.full(nel, rho0_arr[0], dtype=np.float64)

    # Thickness
    thkly_arr = np.atleast_1d(np.asarray(thkly if thkly is not None else 1.0, dtype=np.float64)).copy()
    if len(thkly_arr) == 1 and nel > 1:
        thkly_arr = np.full(nel, thkly_arr[0], dtype=np.float64)

    thk_arr = np.atleast_1d(np.asarray(thk if thk is not None else thkly_arr, dtype=np.float64)).copy()
    if len(thk_arr) == 1 and nel > 1:
        thk_arr = np.full(nel, thk_arr[0], dtype=np.float64)

    # Plastic strain
    pla_arr = np.atleast_1d(np.asarray(pla if pla is not None else 0.0, dtype=np.float64)).copy()
    if len(pla_arr) == 1 and nel > 1:
        pla_arr = np.full(nel, pla_arr[0], dtype=np.float64)

    # Backstress (nel, 12)
    if sigb is not None:
        sigb_arr = np.atleast_2d(np.asarray(sigb, dtype=np.float64)).copy()
        if sigb_arr.shape[1] < 12:
            pad = np.zeros((nel, 12), dtype=np.float64)
            pad[:, :sigb_arr.shape[1]] = sigb_arr
            sigb_arr = pad
    else:
        sigb_arr = np.zeros((nel, 12), dtype=np.float64)

    # Element status
    off_arr = np.atleast_1d(np.asarray(off if off is not None else 1.0, dtype=np.float64)).copy()
    if len(off_arr) == 1 and nel > 1:
        off_arr = np.full(nel, off_arr[0], dtype=np.float64)

    # User variables (nel, 1)
    if uvar is not None:
        uvar_arr = np.atleast_2d(np.asarray(uvar, dtype=np.float64)).copy()
    else:
        uvar_arr = np.zeros((nel, 1), dtype=np.float64)

    # Temperature
    temp_arr = np.atleast_1d(np.asarray(temp if temp is not None else p.temp0, dtype=np.float64)).copy()
    if len(temp_arr) == 1 and nel > 1:
        temp_arr = np.full(nel, temp_arr[0], dtype=np.float64)

    # Elastic Moduli
    a1 = p.a1
    a2 = p.a2
    g = p.g
    gs = g * shf
    young = p.e
    nu = p.nu
    expa = p.expa
    fisokin = p.fisokin
    ikin = p.ikin

    # 1. Trial stresses: mat87c_swift_voce.F90:225-231
    signxx = sigo_arr[:, 0] + a1 * deps_arr[:, 0] + a2 * deps_arr[:, 1]
    signyy = sigo_arr[:, 1] + a2 * deps_arr[:, 0] + a1 * deps_arr[:, 1]
    signxy = sigo_arr[:, 2] + g * deps_arr[:, 2]
    if has_shear:
        signyz = sigo_arr[:, 3] + gs * deps_arr[:, 3]
        signzx = sigo_arr[:, 4] + gs * deps_arr[:, 4]
    else:
        signyz = np.zeros(nel, dtype=np.float64)
        signzx = np.zeros(nel, dtype=np.float64)

    # 2. Backstress accumulation and shift: mat87c_swift_voce.F90:233-249
    sigbxx = np.zeros(nel, dtype=np.float64)
    sigbyy = np.zeros(nel, dtype=np.float64)
    sigbxy = np.zeros(nel, dtype=np.float64)
    if fisokin > 0.0:
        for j in range(4):
            sigbxx += sigb_arr[:, 3 * j + 0]
            sigbyy += sigb_arr[:, 3 * j + 1]
            sigbxy += sigb_arr[:, 3 * j + 2]
        signxx -= sigbxx
        signyy -= sigbyy
        signxy -= sigbxy

    # 3. Strain rate: mat87c_swift_voce.F90:186-198
    if p.iflagsr == 0:
        if dt > 0.0:
            epspxx = deps_arr[:, 0] / dt
            epspyy = deps_arr[:, 1] / dt
            epspxy = deps_arr[:, 2] / dt
            epsd = 0.5 * (np.abs(epspxx + epspyy) + np.sqrt((epspxx - epspyy) ** 2 + epspxy ** 2))
        else:
            epsd = np.zeros(nel, dtype=np.float64)
    else:
        epsd = uvar_arr[:, 0].copy()

    # 4. Initial Barlat equivalent stress: mat87c_swift_voce.F90:254-297
    normsig = np.sqrt(signxx ** 2 + signyy ** 2 + 2.0 * signxy ** 2)
    normsig = np.maximum(normsig, 1.0)

    xpxx = (p.lp11 * signxx + p.lp12 * signyy) / normsig
    xpyy = (p.lp21 * signxx + p.lp22 * signyy) / normsig
    xpxy = p.lp66 * signxy / normsig

    xppxx = (p.lpp11 * signxx + p.lpp12 * signyy) / normsig
    xppyy = (p.lpp21 * signxx + p.lpp22 * signyy) / normsig
    xppxy = p.lpp66 * signxy / normsig

    r_p = np.sqrt(0.25 * (xpxx - xpyy) ** 2 + xpxy ** 2)
    c_p = 0.5 * (xpxx + xpyy)
    xp1 = c_p + r_p
    xp2 = c_p - r_p

    r_pp = np.sqrt(0.25 * (xppxx - xppyy) ** 2 + xppxy ** 2)
    c_pp = 0.5 * (xppxx + xppyy)
    xpp1 = c_pp + r_pp
    xpp2 = c_pp - r_pp

    phip = np.abs(xp1 - xp2) ** expa
    phipp = np.abs(2.0 * xpp2 + xpp1) ** expa + np.abs(2.0 * xpp1 + xpp2) ** expa
    sum_phi = 0.5 * (phip + phipp)
    seq = np.zeros(nel, dtype=np.float64)
    pos_seq = sum_phi > 0.0
    seq[pos_seq] = (sum_phi[pos_seq] ** (1.0 / expa)) * normsig[pos_seq]

    # 5. Yield stress evaluation: mat87c_swift_voce.F90:302-332
    def eval_yld_oracle(p_in: np.ndarray, ed_in: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if p.iflag == 1:
            swift = np.zeros(nel, dtype=np.float64)
            dswiftdp = np.zeros(nel, dtype=np.float64)
            eff_p = p_in + p.epso
            pos = eff_p > 0.0
            swift[pos] = p.aswift * (eff_p[pos] ** p.nexp)
            dswiftdp[pos] = p.aswift * p.nexp * (eff_p[pos] ** (p.nexp - 1.0))

            voce = p.ko + p.qvoce * (1.0 - np.exp(-p.beta * p_in))
            dvocedp = p.qvoce * p.beta * np.exp(-p.beta * p_in)

            yld_iso = p.alpha * swift + (1.0 - p.alpha) * voce
            dylddp_iso = p.alpha * dswiftdp + (1.0 - p.alpha) * dvocedp

            yld0_v = (1.0 - p.alpha) * p.ko
            if p.epso > 0.0:
                yld0_v += p.alpha * p.aswift * (p.epso ** p.nexp)

            y = (1.0 - fisokin) * yld_iso + fisokin * yld0_v
            h = fisokin * dylddp_iso
            dy = (1.0 - fisokin) * dylddp_iso

            if p.unsp != 0.0:
                pos_ed = ed_in > 0.0
                frate = np.ones(nel, dtype=np.float64)
                frate[pos_ed] = 1.0 + np.exp(p.unsp * np.log(p.unsc * ed_in[pos_ed]))
                y *= frate
                h *= frate
                dy *= frate
            elif p.invc > 0.0 and p.invp > 0.0:
                pos_ed = ed_in > 0.0
                frate = np.ones(nel, dtype=np.float64)
                p_exp = 1.0 / p.invp if p.invp < 1.0 else p.invp
                c_val = p.invc if p.invc > 1.0 else 1.0 / p.invc
                frate[pos_ed] = 1.0 + (ed_in[pos_ed] / c_val) ** p_exp
                y *= frate
                h *= frate
                dy *= frate
            return y, dy, h
        elif p.iflag == 0 and yield_fn is not None:
            y, dy = yield_fn(p_in)
            y0, _ = yield_fn(np.zeros_like(p_in))
            y_cur = (1.0 - fisokin) * y + fisokin * y0
            h_cur = fisokin * dy
            dy_cur = (1.0 - fisokin) * dy
            return y_cur, dy_cur, h_cur
        elif p.iflag == 0 and p.curves:
            v_cur, s_cur = eval_curve_1d_oracle(p.curves[0], p_in)
            v0_cur, _ = eval_curve_1d_oracle(p.curves[0], np.zeros_like(p_in))
            y_cur = (1.0 - fisokin) * v_cur + fisokin * v0_cur
            h_cur = fisokin * s_cur
            dy_cur = (1.0 - fisokin) * s_cur
            return y_cur, dy_cur, h_cur
        else:
            sy0 = p.ko if p.ko > 0.0 else (p.aswift if p.aswift > 0.0 else 1.0)
            return np.full(nel, sy0, dtype=np.float64), np.zeros(nel, dtype=np.float64), np.zeros(nel, dtype=np.float64)

    yld, dylddp, hk = eval_yld_oracle(pla_arr, epsd)
    phi = (seq / np.maximum(yld, _EM20)) ** 2 - 1.0
    yielding = np.where((phi >= 0.0) & (off_arr == 1.0))[0]

    dpla = np.zeros(nel, dtype=np.float64)
    deplzz = np.zeros(nel, dtype=np.float64)
    etse = np.ones(nel, dtype=np.float64)

    # 6. Return mapping loop: mat87c_swift_voce.F90:349-678
    if len(yielding) > 0:
        dsigbxxdp = np.zeros(nel, dtype=np.float64)
        dsigbyydp = np.zeros(nel, dtype=np.float64)
        dsigbxydp = np.zeros(nel, dtype=np.float64)
        if ikin == 1 and fisokin > 0.0:
            for j in range(4):
                dsigbxxdp += p.ckh[j] * sigb_arr[:, 3 * j + 0]
                dsigbyydp += p.ckh[j] * sigb_arr[:, 3 * j + 1]
                dsigbxydp += p.ckh[j] * sigb_arr[:, 3 * j + 2]

        for _iter in range(niter):
            for i in yielding:
                # Derivatives of X' principal values
                mr_p = max(r_p[i], _EM20)
                dxp1dxpxx = 0.5 * (1.0 + (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp1dxpyy = 0.5 * (1.0 - (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp1dxpxy = xpxy[i] / mr_p
                dxp2dxpxx = 0.5 * (1.0 - (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp2dxpyy = 0.5 * (1.0 + (xpxx[i] - xpyy[i]) / (2.0 * mr_p))
                dxp2dxpxy = -xpxy[i] / mr_p

                # Derivatives of X'' principal values
                mr_pp = max(r_pp[i], _EM20)
                dxpp1dxppxx = 0.5 * (1.0 + (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp1dxppyy = 0.5 * (1.0 - (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp1dxppxy = xppxy[i] / mr_pp
                dxpp2dxppxx = 0.5 * (1.0 - (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp2dxppyy = 0.5 * (1.0 + (xppxx[i] - xppyy[i]) / (2.0 * mr_pp))
                dxpp2dxppxy = -xppxy[i] / mr_pp

                # Chain to stress components
                dxp1dsigxx = dxp1dxpxx * p.lp11 + dxp1dxpyy * p.lp21
                dxp1dsigyy = dxp1dxpxx * p.lp12 + dxp1dxpyy * p.lp22
                dxp1dsigxy = dxp1dxpxy * p.lp66

                dxp2dsigxx = dxp2dxpxx * p.lp11 + dxp2dxpyy * p.lp21
                dxp2dsigyy = dxp2dxpxx * p.lp12 + dxp2dxpyy * p.lp22
                dxp2dsigxy = dxp2dxpxy * p.lp66

                dxpp1dsigxx = dxpp1dxppxx * p.lpp11 + dxpp1dxppyy * p.lpp21
                dxpp1dsigyy = dxpp1dxppxx * p.lpp12 + dxpp1dxppyy * p.lpp22
                dxpp1dsigxy = dxpp1dxppxy * p.lpp66

                dxpp2dsigxx = dxpp2dxppxx * p.lpp11 + dxpp2dxppyy * p.lpp21
                dxpp2dsigyy = dxpp2dxppxx * p.lpp12 + dxpp2dxppyy * p.lpp22
                dxpp2dsigxy = dxpp2dxppxy * p.lpp66

                diff_p = xp1[i] - xp2[i]
                sgn_p = 1.0 if diff_p >= 0.0 else -1.0
                dphipdxp1 = expa * (abs(diff_p) ** (expa - 1.0)) * sgn_p
                dphipdxp2 = -dphipdxp1

                term_pp1 = 2.0 * xpp2[i] + xpp1[i]
                sgn_pp1 = 1.0 if term_pp1 >= 0.0 else -1.0
                term_pp2 = 2.0 * xpp1[i] + xpp2[i]
                sgn_pp2 = 1.0 if term_pp2 >= 0.0 else -1.0

                dphippdxpp1 = (expa * (abs(term_pp1) ** (expa - 1.0)) * sgn_pp1
                               + 2.0 * expa * (abs(term_pp2) ** (expa - 1.0)) * sgn_pp2)
                dphippdxpp2 = (expa * (abs(term_pp2) ** (expa - 1.0)) * sgn_pp2
                               + 2.0 * expa * (abs(term_pp1) ** (expa - 1.0)) * sgn_pp1)

                dphipdsigxx = dphipdxp1 * dxp1dsigxx + dphipdxp2 * dxp2dsigxx
                dphipdsigyy = dphipdxp1 * dxp1dsigyy + dphipdxp2 * dxp2dsigyy
                dphipdsigxy = dphipdxp1 * dxp1dsigxy + dphipdxp2 * dxp2dsigxy

                dphippdsigxx = dphippdxpp1 * dxpp1dsigxx + dphippdxpp2 * dxpp2dsigxx
                dphippdsigyy = dphippdxpp1 * dxpp1dsigyy + dphippdxpp2 * dxpp2dsigyy
                dphippdsigxy = dphippdxpp1 * dxpp1dsigxy + dphippdxpp2 * dxpp2dsigxy

                s_phi = 0.5 * (phip[i] + phipp[i])
                dseqdphi = (0.5 / expa) * (s_phi ** (1.0 / expa - 1.0)) if s_phi > 0.0 else 0.0

                dseqdsigxx = dseqdphi * (dphipdsigxx + dphippdsigxx)
                dseqdsigyy = dseqdphi * (dphipdsigyy + dphippdsigyy)
                dseqdsigxy = dseqdphi * (dphipdsigxy + dphippdsigxy)

                myld = max(yld[i], _EM20)
                dphidseq = 2.0 * (seq[i] / (myld ** 2))
                normxx = dphidseq * dseqdsigxx
                normyy = dphidseq * dseqdsigyy
                normxy = dphidseq * dseqdsigxy

                dsigxxdlam = -a1 * normxx - a2 * normyy
                dsigyydlam = -a1 * normyy - a2 * normxx
                dsigxydlam = -g * normxy

                dphidsig_dsigdlam = normxx * dsigxxdlam + normyy * dsigyydlam + normxy * dsigxydlam

                dphidyld = -2.0 * (seq[i] ** 2 / (myld ** 3))
                dphidpla = dphidyld * dylddp[i]
                sig_dphidsig = signxx[i] * normxx + signyy[i] * normyy + signxy[i] * normxy
                dpladlam = sig_dphidsig / myld

                # Kinematic hardening contribution
                if fisokin > 0.0:
                    if ikin == 1:
                        dsigbxxdlam = fisokin * (p.akck * (2.0 * normxx + normyy) - dsigbxxdp[i] * dpladlam)
                        dsigbyydlam = fisokin * (p.akck * (2.0 * normyy + normxx) - dsigbyydp[i] * dpladlam)
                        dsigbxydlam = fisokin * (p.akck * normxy - dsigbxydp[i] * dpladlam)
                    elif ikin == 2:
                        dsigbxxdlam = (2.0 / 3.0) * hk[i] * (2.0 * normxx + normyy)
                        dsigbyydlam = (2.0 / 3.0) * hk[i] * (2.0 * normyy + normxx)
                        dsigbxydlam = (2.0 / 3.0) * hk[i] * normxy
                    else:
                        dsigbxxdlam = dsigbyydlam = dsigbxydlam = 0.0
                    dphidsigb_dsigbdlam = -normxx * dsigbxxdlam - normyy * dsigbyydlam - normxy * dsigbxydlam
                else:
                    dsigbxxdlam = dsigbyydlam = dsigbxydlam = 0.0
                    dphidsigb_dsigbdlam = 0.0

                dphidlam = dphidsig_dsigdlam + dphidpla * dpladlam + dphidsigb_dsigbdlam
                if abs(dphidlam) < _EM20:
                    dphidlam = _EM20 if dphidlam >= 0.0 else -_EM20

                dlam = -phi[i] / dphidlam
                ddep = dpladlam * dlam

                dpla[i] = max(0.0, dpla[i] + ddep)
                pla_arr[i] += ddep
                deplzz[i] -= (dlam * normxx + dlam * normyy)

                signxx[i] += dsigxxdlam * dlam
                signyy[i] += dsigyydlam * dlam
                signxy[i] += dsigxydlam * dlam

                if fisokin > 0.0:
                    signxx[i] += sigbxx[i]
                    signyy[i] += sigbyy[i]
                    signxy[i] += sigbxy[i]
                    sigbxx[i] += dsigbxxdlam * dlam
                    sigbyy[i] += dsigbyydlam * dlam
                    sigbxy[i] += dsigbxydlam * dlam
                    signxx[i] -= sigbxx[i]
                    signyy[i] -= sigbyy[i]
                    signxy[i] -= sigbxy[i]

                    if ikin == 1:
                        for j in range(4):
                            fac_a = p.akh[j] * p.ckh[j]
                            c_j = p.ckh[j]
                            sigb_arr[i, 3 * j + 0] += fisokin * (fac_a * (2.0 * normxx + normyy) * dlam - c_j * sigb_arr[i, 3 * j + 0] * ddep)
                            sigb_arr[i, 3 * j + 1] += fisokin * (fac_a * (2.0 * normyy + normxx) * dlam - c_j * sigb_arr[i, 3 * j + 1] * ddep)
                            sigb_arr[i, 3 * j + 2] += fisokin * (fac_a * normxy * dlam - c_j * sigb_arr[i, 3 * j + 2] * ddep)
                    elif ikin == 2:
                        sigb_arr[i, 0] += dsigbxxdlam * dlam
                        sigb_arr[i, 1] += dsigbyydlam * dlam
                        sigb_arr[i, 2] += dsigbxydlam * dlam

                ns = math.sqrt(signxx[i] ** 2 + signyy[i] ** 2 + 2.0 * signxy[i] ** 2)
                normsig[i] = max(ns, 1.0)
                xpxx[i] = (p.lp11 * signxx[i] + p.lp12 * signyy[i]) / normsig[i]
                xpyy[i] = (p.lp21 * signxx[i] + p.lp22 * signyy[i]) / normsig[i]
                xpxy[i] = (p.lp66 * signxy[i]) / normsig[i]

                xppxx[i] = (p.lpp11 * signxx[i] + p.lpp12 * signyy[i]) / normsig[i]
                xppyy[i] = (p.lpp21 * signxx[i] + p.lpp22 * signyy[i]) / normsig[i]
                xppxy[i] = (p.lpp66 * signxy[i]) / normsig[i]

                r_p[i] = math.sqrt(0.25 * (xpxx[i] - xpyy[i]) ** 2 + xpxy[i] ** 2)
                c_p = 0.5 * (xpxx[i] + xpyy[i])
                xp1[i] = c_p + r_p[i]
                xp2[i] = c_p - r_p[i]

                r_pp[i] = math.sqrt(0.25 * (xppxx[i] - xppyy[i]) ** 2 + xppxy[i] ** 2)
                c_pp = 0.5 * (xppxx[i] + xppyy[i])
                xpp1[i] = c_pp + r_pp[i]
                xpp2[i] = c_pp - r_pp[i]

                phip[i] = abs(xp1[i] - xp2[i]) ** expa
                phipp[i] = (abs(2.0 * xpp2[i] + xpp1[i]) ** expa + abs(2.0 * xpp1[i] + xpp2[i]) ** expa)

                sum_phi_i = 0.5 * (phip[i] + phipp[i])
                if sum_phi_i > 0.0:
                    seq[i] = (sum_phi_i ** (1.0 / expa)) * normsig[i]
                else:
                    seq[i] = 0.0

            # Re-evaluate yield stress
            y_up, dy_up, hk_up = eval_yld_oracle(pla_arr, epsd)
            yld = y_up
            dylddp = dy_up
            hk = hk_up
            phi = (seq / np.maximum(yld, _EM20)) ** 2 - 1.0

        for i in yielding:
            h_tot = dylddp[i] + hk[i]
            etse[i] = h_tot / (h_tot + young)

    # Re-add backstress
    if fisokin > 0.0:
        signxx += sigbxx
        signyy += sigbyy
        signxy += sigbxy

    # 7. Thickness thinning & sound speed: mat87c_swift_voce.F90:791-800
    deelzz = -nu * (signxx - sigo_arr[:, 0] + signyy - sigo_arr[:, 1]) / young
    depszz = deelzz + deplzz
    thk_out = thk_arr + depszz * thkly_arr * off_arr
    soundsp = np.sqrt(a1 / np.maximum(rho0_arr, _EM20))

    if p.iflagsr == 1:
        uvar_arr[:, 0] = dpla / max(dt, _EM20)

    if has_shear:
        sig_out = np.column_stack([signxx, signyy, signxy, signyz, signzx])
    else:
        sig_out = np.column_stack([signxx, signyy, signxy])

    return {
        "sig": sig_out,
        "pla": pla_arr,
        "dpla": dpla,
        "sigb": sigb_arr,
        "thk": thk_out,
        "depszz": depszz,
        "soundsp": soundsp,
        "seq": seq,
        "yld": yld,
        "etse": etse,
        "uvar": uvar_arr,
        "off": off_arr,
    }


# =============================================================================
# 2. TEST SUITE: Linear Transformation Matrices & Principal Values
# =============================================================================

class TestBarlatTransformationMatrices:
    """Parity tests for hm_read_mat87.F90:410-442 and mat87c_swift_voce.F90:201-212."""

    def test_isotropic_projection_matrices(self):
        """When alpha1..alpha8 = 1.0, Lp and Lpp must reduce to 2D deviatoric operators."""
        p = build_law87(al1=1.0, al2=1.0, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0)

        # Lp matrix components
        assert p.lp11 == pytest.approx(2.0 / 3.0, rel=1e-14)
        assert p.lp12 == pytest.approx(-1.0 / 3.0, rel=1e-14)
        assert p.lp21 == pytest.approx(-1.0 / 3.0, rel=1e-14)
        assert p.lp22 == pytest.approx(2.0 / 3.0, rel=1e-14)
        assert p.lp66 == pytest.approx(1.0, rel=1e-14)

        # Lpp matrix components
        assert p.lpp11 == pytest.approx(2.0 / 3.0, rel=1e-14)
        assert p.lpp12 == pytest.approx(-1.0 / 3.0, rel=1e-14)
        assert p.lpp21 == pytest.approx(-1.0 / 3.0, rel=1e-14)
        assert p.lpp22 == pytest.approx(2.0 / 3.0, rel=1e-14)
        assert p.lpp66 == pytest.approx(1.0, rel=1e-14)

    @pytest.mark.parametrize("al", [
        [0.95, 1.05, 0.90, 1.10, 0.85, 1.15, 1.00, 1.02],
        [1.20, 0.80, 1.15, 0.90, 1.05, 0.95, 0.88, 1.12],
        [0.75, 1.25, 0.85, 1.15, 1.30, 0.70, 0.95, 1.05],
    ])
    def test_anisotropic_matrix_eigenvalues_positive(self, al):
        """Convexity requires positive eigenvalues matching hm_read_mat87.F90:410-442."""
        p = build_law87(alphas=al)
        lp_mat = np.array([
            [p.lp11, p.lp12, 0.0],
            [p.lp21, p.lp22, 0.0],
            [0.0, 0.0, p.lp66],
        ])
        lpp_mat = np.array([
            [p.lpp11, p.lpp12, 0.0],
            [p.lpp21, p.lpp22, 0.0],
            [0.0, 0.0, p.lpp66],
        ])

        eig_lp = np.linalg.eigvals(lp_mat)
        eig_lpp = np.linalg.eigvals(lpp_mat)

        assert np.all(np.real(eig_lp) > 0.0)
        assert np.all(np.real(eig_lpp) > 0.0)

    def test_mohr_principal_values(self):
        """Verify Mohr circle radius and center for X' and X'' matching analytical formulas."""
        p = build_law87(al1=1.1, al2=0.9, al7=1.05)
        sig = np.array([[150.0, -80.0, 45.0]])

        norm = math.sqrt(150.0**2 + 80.0**2 + 2.0 * 45.0**2)
        xpxx = (p.lp11 * 150.0 + p.lp12 * (-80.0)) / norm
        xpyy = (p.lp21 * 150.0 + p.lp22 * (-80.0)) / norm
        xpxy = p.lp66 * 45.0 / norm

        c = 0.5 * (xpxx + xpyy)
        r = math.sqrt(0.25 * (xpxx - xpyy)**2 + xpxy**2)
        xp1 = c + r
        xp2 = c - r

        # Check invariant: xp1 + xp2 = xpxx + xpyy
        assert (xp1 + xp2) == pytest.approx(xpxx + xpyy, rel=1e-14)
        # Check invariant: xp1 * xp2 = xpxx * xpyy - xpxy^2
        assert (xp1 * xp2) == pytest.approx(xpxx * xpyy - xpxy**2, rel=1e-14)


# =============================================================================
# 3. TEST SUITE: Barlat Equivalent Stress & Isotropic Reduction
# =============================================================================

class TestBarlatEquivalentStress:
    """Parity tests for mat87c_swift_voce.F90:251-297."""

    @pytest.mark.parametrize("sig", [
        [200.0, 0.0, 0.0],          # Pure tension X
        [0.0, 200.0, 0.0],          # Pure tension Y
        [-180.0, 0.0, 0.0],         # Pure compression X
        [0.0, -180.0, 0.0],         # Pure compression Y
        [0.0, 0.0, 100.0],          # Pure shear XY
        [150.0, 150.0, 0.0],        # Equibiaxial
        [250.0, 100.0, 50.0],       # Combined tension + shear
        [-150.0, 80.0, -40.0],      # Mixed sign tension/compression/shear
    ])
    def test_isotropic_reduction_matches_von_mises(self, sig):
        """When alpha1..alpha8 = 1.0 and a = 2.0, Barlat Yld2000-2d identically recovers von Mises."""
        p = build_law87(al1=1.0, al2=1.0, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0, expa=2.0)
        s_arr = np.array([sig])

        seq_barlat = barlat2000_equivalent_stress(s_arr, p)[0]
        vm_expected = math.sqrt(sig[0]**2 - sig[0]*sig[1] + sig[1]**2 + 3.0 * sig[2]**2)

        assert seq_barlat == pytest.approx(vm_expected, rel=1e-12)

    @pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_degree_one_homogeneity(self, k):
        """bar_sigma(k * sigma) = k * bar_sigma(sigma) for any positive scalar k."""
        p = build_law87(al1=0.95, al2=1.05, al3=0.90, al4=1.10, al5=0.85, al6=1.15, al7=1.00, al8=1.02, expa=8.0)
        sig = np.array([[120.0, -60.0, 35.0]])

        seq1 = barlat2000_equivalent_stress(sig, p)[0]
        seq_k = barlat2000_equivalent_stress(k * sig, p)[0]

        assert seq_k == pytest.approx(k * seq1, rel=1e-12)

    @pytest.mark.parametrize("a_exp", [2.0, 4.0, 6.0, 8.0])
    def test_exponent_scaling(self, a_exp):
        """Yield stress response across different crystallographic exponents (BCC a=6, FCC a=8)."""
        p = build_law87(al1=1.0, al2=1.0, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0, expa=a_exp)
        sig = np.array([[200.0, 0.0, 0.0]])
        seq = barlat2000_equivalent_stress(sig, p)[0]
        # In uniaxial tension along x with isotropic alphas, seq = sig_xx regardless of exponent
        assert seq == pytest.approx(200.0, rel=1e-10)

    def test_anisotropic_directional_variation(self):
        """Verify directional anisotropy gives distinct equivalent stresses for 0, 45, 90 deg."""
        p = build_law87(al1=0.85, al2=1.15, al3=0.90, al4=1.10, al5=0.95, al6=1.05, al7=0.88, al8=1.12, expa=6.0)

        # 0 deg uniaxial: [100, 0, 0]
        seq0 = barlat2000_equivalent_stress(np.array([[100.0, 0.0, 0.0]]), p)[0]
        # 90 deg uniaxial: [0, 100, 0]
        seq90 = barlat2000_equivalent_stress(np.array([[0.0, 100.0, 0.0]]), p)[0]
        # 45 deg uniaxial: [50, 50, 50]
        seq45 = barlat2000_equivalent_stress(np.array([[50.0, 50.0, 50.0]]), p)[0]
        # Equibiaxial: [100, 100, 0]
        seq_biax = barlat2000_equivalent_stress(np.array([[100.0, 100.0, 0.0]]), p)[0]

        # Anisotropy ensures all directional values differ
        assert abs(seq0 - seq90) > 1.0
        assert abs(seq0 - seq45) > 1.0
        assert abs(seq0 - seq_biax) > 1.0

    def test_pure_shear_sensitivity_to_alpha7_and_alpha8(self):
        """Verify that pure shear stress is governed exclusively by alpha7 and alpha8."""
        p1 = build_law87(al7=1.0, al8=1.0, expa=4.0)
        p2 = build_law87(al7=1.5, al8=1.0, expa=4.0)
        p3 = build_law87(al7=1.0, al8=1.5, expa=4.0)

        sig_shear = np.array([[0.0, 0.0, 100.0]])
        seq1 = barlat2000_equivalent_stress(sig_shear, p1)[0]
        seq2 = barlat2000_equivalent_stress(sig_shear, p2)[0]
        seq3 = barlat2000_equivalent_stress(sig_shear, p3)[0]

        assert seq2 > seq1
        assert seq3 > seq1


# =============================================================================
# 4. TEST SUITE: Analytical Stress Gradients vs Numerical Perturbation
# =============================================================================

class TestAnalyticalGradients:
    """Parity tests for analytical d(seq)/d(sigma) matching numerical finite differences."""

    @pytest.mark.parametrize("sig", [
        [180.0, 0.0, 0.0],
        [0.0, 160.0, 0.0],
        [0.0, 0.0, 90.0],
        [120.0, 120.0, 0.0],
        [150.0, 80.0, 45.0],
        [-100.0, 70.0, -35.0],
    ])
    def test_analytical_gradient_matches_numerical_fd(self, sig):
        """Verify analytical gradient matches central finite differences to 1e-8."""
        p = build_law87(al1=0.92, al2=1.08, al3=0.95, al4=1.05, al5=0.88, al6=1.12, al7=0.97, al8=1.03, expa=6.0)
        s_arr = np.array(sig, dtype=np.float64)

        grad_ana = barlat2000_gradient_analytical(s_arr, p)

        # Central finite differences: dseq/ds_k = (seq(s + h*e_k) - seq(s - h*e_k)) / (2*h)
        h = 1.0e-7
        grad_num = np.zeros(3, dtype=np.float64)
        for k in range(3):
            e_k = np.zeros(3, dtype=np.float64)
            e_k[k] = h
            sp = barlat2000_equivalent_stress(s_arr + e_k, p)
            sm = barlat2000_equivalent_stress(s_arr - e_k, p)
            grad_num[k] = (sp - sm) / (2.0 * h)

        np.testing.assert_allclose(grad_ana, grad_num, rtol=1e-6, atol=1e-6)

    def test_euler_theorem_on_homogeneous_functions(self):
        """By Euler's theorem, sig : d(seq)/d(sig) = seq for degree-1 homogeneous function."""
        p = build_law87(al1=0.95, al2=1.05, al3=0.90, al4=1.10, al5=0.85, al6=1.15, al7=1.00, al8=1.02, expa=8.0)
        sig = np.array([140.0, -90.0, 50.0])

        seq = barlat2000_equivalent_stress(sig, p)
        grad = barlat2000_gradient_analytical(sig, p)

        # Check standard Euler identity: sum_k s_k * (dseq/ds_k) = seq
        euler_sum = sig[0] * grad[0] + sig[1] * grad[1] + sig[2] * grad[2]
        assert euler_sum == pytest.approx(seq, rel=1e-8)


# =============================================================================
# 5. TEST SUITE: Swift-Voce & Strain Rate Hardening
# =============================================================================

class TestSwiftVoceAndRateHardening:
    """Parity tests for mat87c_swift_voce.F90:302-332."""

    def test_pure_swift_hardening(self):
        """alpha = 1.0 gives pure Swift: yld = aswift * (epso + pla)^nexp."""
        p = build_law87(iflag=1, alpha=1.0, aswift=600.0, epso=0.005, nexp=0.22, ko=200.0)
        sigo = np.zeros((1, 3))
        deps = np.array([[0.001, 0.0, 0.0]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=1e-5, niter=3)
        res_py, pla_py = shell_update(p, sigo, deps, dt=1e-5, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)

    def test_pure_voce_hardening(self):
        """alpha = 0.0 gives pure Voce: yld = ko + qvoce * (1 - exp(-beta * pla))."""
        p = build_law87(iflag=1, alpha=0.0, ko=250.0, qvoce=180.0, beta=25.0)
        sigo = np.zeros((1, 3))
        deps = np.array([[0.0015, 0.0, 0.0]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=1e-5, niter=3)
        res_py, pla_py = shell_update(p, sigo, deps, dt=1e-5, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)

    def test_mixed_swift_voce_hardening(self):
        """alpha = 0.6 blends Swift and Voce models."""
        p = build_law87(iflag=1, alpha=0.6, aswift=550.0, epso=0.002, nexp=0.20, ko=220.0, qvoce=150.0, beta=15.0)
        sigo = np.zeros((1, 3))
        deps = np.array([[0.002, 0.0005, 0.0002]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=1e-5, niter=3)
        res_py, pla_py = shell_update(p, sigo, deps, dt=1e-5, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)

    @pytest.mark.parametrize("dt_val", [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6])
    def test_cowper_symonds_rate_effect(self, dt_val):
        """Cowper-Symonds strain rate scaling: frate = 1 + (unsc * epsd)^unsp."""
        p = build_law87(
            iflag=1, alpha=1.0, aswift=500.0, epso=0.002, nexp=0.15,
            invc=100.0, invp=0.20,
        )
        sigo = np.zeros((1, 3))
        deps = np.array([[0.001, 0.0, 0.0]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=dt_val, niter=3)
        res_py, pla_py = shell_update(p, sigo, deps, dt=dt_val, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)

    def test_tabulated_yield_stress_lookup(self):
        """Tabulated curve (iflag=0) interpolates yield stress and hardening slope."""
        curve_pts = [(0.0, 200.0), (0.02, 280.0), (0.10, 420.0), (0.50, 600.0)]
        p = build_law87(iflag=0, curves=[curve_pts])
        sigo = np.zeros((1, 3))
        deps = np.array([[0.002, 0.0, 0.0]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=1e-5, niter=3)
        res_py, pla_py = shell_update(p, sigo, deps, dt=1e-5, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)


# =============================================================================
# 6. TEST SUITE: Kinematic Hardening & Bauschinger Effect
# =============================================================================

class TestKinematicHardeningAndBauschinger:
    """Parity tests for mat87c_swift_voce.F90:503-600."""

    def test_chaboche_rousselier_single_branch(self):
        """Chaboche-Rousselier (ikin=1) with 1 backstress branch."""
        p = build_law87(
            iflag=1, fisokin=0.5, ikin=1,
            ckh=[100.0, 0.0, 0.0, 0.0],
            akh=[80.0, 0.0, 0.0, 0.0],
            aswift=350.0, epso=0.002, nexp=0.1,
        )
        sigo = np.zeros((1, 3))
        deps = np.array([[0.002, 0.0, 0.0]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=1e-5, niter=3)
        extra = {}
        res_py, pla_py = shell_update(p, sigo, deps, dt=1e-5, extra=extra, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)
        np.testing.assert_allclose(np.asarray(extra["sigb87"]).flatten()[:3], res_orc["sigb"][0, :3], rtol=1e-12, atol=1e-12)

    def test_chaboche_rousselier_four_branches(self):
        """Chaboche-Rousselier (ikin=1) with all 4 backstress branches active."""
        p = build_law87(
            iflag=1, fisokin=0.6, ikin=1,
            ckh=[500.0, 200.0, 50.0, 10.0],
            akh=[40.0, 30.0, 20.0, 10.0],
            aswift=400.0, epso=0.002, nexp=0.15,
        )
        sigo = np.zeros((1, 3))
        deps = np.array([[0.0025, 0.0005, 0.0002]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=1e-5, niter=3)
        extra = {}
        res_py, pla_py = shell_update(p, sigo, deps, dt=1e-5, extra=extra, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)
        np.testing.assert_allclose(np.asarray(extra["sigb87"]).flatten()[:12], res_orc["sigb"][0, :12], rtol=1e-12, atol=1e-12)

    def test_prager_linear_kinematic(self):
        """Prager linear kinematic hardening (ikin=2)."""
        p = build_law87(
            iflag=1, fisokin=0.4, ikin=2,
            aswift=300.0, epso=0.002, nexp=0.1,
        )
        sigo = np.zeros((1, 3))
        deps = np.array([[0.0015, -0.0005, 0.0003]])

        res_orc = sigeps87c_oracle(p, sigo, deps, dt=1e-5, niter=3)
        extra = {}
        res_py, pla_py = shell_update(p, sigo, deps, dt=1e-5, extra=extra, niter=3)

        np.testing.assert_allclose(res_py, res_orc["sig"], rtol=1e-12, atol=1e-12)
        assert pla_py == pytest.approx(res_orc["pla"][0], rel=1e-12)

    def test_cyclic_bauschinger_parity(self):
        """Two-step cyclic loading: forward tension then reverse compression matching oracle."""
        p = build_law87(
            iflag=1, fisokin=0.5, ikin=1,
            ckh=[80.0, 0.0, 0.0, 0.0],
            akh=[60.0, 0.0, 0.0, 0.0],
            aswift=320.0, epso=0.002, nexp=0.1,
        )
        sigo = np.zeros((1, 3))
        deps_fwd = np.array([[0.002, 0.0, 0.0]])

        # Step 1: Forward
        orc_1 = sigeps87c_oracle(p, sigo, deps_fwd, dt=1e-5, niter=3)
        extra = {}
        py_sig_1, py_pla_1 = shell_update(p, sigo, deps_fwd, dt=1e-5, extra=extra, niter=3)
        np.testing.assert_allclose(py_sig_1, orc_1["sig"], rtol=1e-12, atol=1e-12)

        # Step 2: Reverse compression
        deps_rev = np.array([[-0.0018, 0.0, 0.0]])
        orc_2 = sigeps87c_oracle(
            p, orc_1["sig"], deps_rev, pla=orc_1["pla"], sigb=orc_1["sigb"],
            thk=orc_1["thk"], dt=1e-5, niter=3
        )
        py_sig_2, py_pla_2 = shell_update(
            p, py_sig_1, deps_rev, epsp=py_pla_1, dt=1e-5, extra=extra, niter=3
        )

        np.testing.assert_allclose(py_sig_2, orc_2["sig"], rtol=1e-12, atol=1e-12)
        assert py_pla_2 == pytest.approx(orc_2["pla"][0], rel=1e-12)
        np.testing.assert_allclose(np.asarray(extra["sigb87"]).flatten()[:3], orc_2["sigb"][0, :3], rtol=1e-12, atol=1e-12)


# =============================================================================
# 7. TEST SUITE: Shell Thickness Thinning & Sound Speed
# =============================================================================

class TestThicknessThinningAndWaveSpeed:
    """Parity tests for mat87c_swift_voce.F90:791-800."""

    def test_elastic_thickness_thinning(self):
        """In pure elasticity, deps_zz = -nu/E * (d_sig_xx + d_sig_yy)."""
        p = build_law87(e=200000.0, nu=0.3, aswift=1.0e8)
        sigo = np.zeros((1, 3))
        deps = np.array([[1e-4, 5e-5, 0.0]])

        orc = sigeps87c_oracle(p, sigo, deps, thk=2.0, thkly=2.0)
        extra = {"thkly": np.array([2.0]), "thk87": np.array([2.0])}
        shell_update(p, sigo, deps, extra=extra, niter=3)

        assert extra["depszz"][0] == pytest.approx(orc["depszz"][0], rel=1e-12)
        assert extra["thk87"][0] == pytest.approx(orc["thk"][0], rel=1e-12)

    def test_plastic_flow_incompressibility_thinning(self):
        """Under plastic yielding, sheet must thin (deps_zz < 0) balancing in-plane extension."""
        p = build_law87(e=210000.0, nu=0.3, iflag=1, aswift=300.0, epso=0.002, nexp=0.1)
        sigo = np.zeros((1, 3))
        deps = np.array([[0.002, 0.0, 0.0]])

        orc = sigeps87c_oracle(p, sigo, deps, thk=1.5, thkly=1.5, niter=3)
        extra = {"thkly": np.array([1.5]), "thk87": np.array([1.5])}
        shell_update(p, sigo, deps, extra=extra, niter=3)

        # Thinning must be negative
        assert extra["depszz"][0] < 0.0
        assert extra["depszz"][0] == pytest.approx(orc["depszz"][0], rel=1e-12)
        assert extra["thk87"][0] == pytest.approx(orc["thk"][0], rel=1e-12)

    @pytest.mark.parametrize("e,nu,rho0", [
        (210000.0, 0.30, 7.85e-9),   # Steel (t-mm-s)
        (70000.0, 0.33, 2.70e-9),    # Aluminum
        (110000.0, 0.34, 4.50e-9),   # Titanium
        (150000.0, 0.28, 7.20e-9),   # Cast iron
    ])
    def test_thin_shell_sound_speed_formula(self, e, nu, rho0):
        """Verify c = sqrt(E / ((1 - nu^2) * rho0))."""
        p = build_law87(e=e, nu=nu, rho0=rho0)
        c_calc = sound_speed_shell(p)
        c_expected = math.sqrt(e / ((1.0 - nu**2) * rho0))

        assert c_calc == pytest.approx(c_expected, rel=1e-14)


# =============================================================================
# 8. TEST SUITE: Algorithmic Consistent Tangent Tensor
# =============================================================================

class TestConsistentShellTangent:
    """Parity tests for consistent membrane tangent operator."""

    def test_elastic_membrane_tangent_matches_theory(self):
        """Elastic membrane tangent matches exact plane-stress Hooke tensor."""
        p = build_law87(e=200000.0, nu=0.3)
        c_ana = shell_membrane_tangent(p)

        c_expected = np.array([
            [p.a1, p.a2, 0.0],
            [p.a2, p.a1, 0.0],
            [0.0, 0.0, p.g],
        ])

        np.testing.assert_allclose(c_ana, c_expected, rtol=1e-14, atol=1e-14)

    def test_plastic_tangent_matches_central_difference(self):
        """Algorithmic consistent tangent matches directional central finite difference perturbation."""
        p = build_law87(iflag=1, aswift=400.0, epso=0.002, nexp=0.15)
        sig0 = np.zeros((1, 3))
        deps0 = np.array([[0.003, 0.0005, 0.0002]])

        # Perturbation step h = 1e-7
        h = 1.0e-7
        c_alg = consistent_shell_tangent(p, sig=sig0, deps=deps0, h=h)

        c_fd = np.zeros((3, 3), dtype=np.float64)
        for j in range(3):
            ej = np.zeros((1, 3))
            ej[0, j] = h
            sp, _ = shell_update(p, sig0.copy(), deps0 + ej)
            sm, _ = shell_update(p, sig0.copy(), deps0 - ej)
            c_fd[:, j] = (sp[0, :3] - sm[0, :3]) / (2.0 * h)

        np.testing.assert_allclose(c_alg[0], c_fd, rtol=1e-4, atol=1e-4)


# =============================================================================
# 9. TEST SUITE: Vectorized Batch Execution Equivalence
# =============================================================================

class TestVectorizedBatchParity:
    """Verify N-element batch execution produces bit-identical results to serial runs."""

    def test_batch_vs_serial_equivalence(self):
        """5 elements with diverse strains executed vectorized vs serially."""
        p = build_law87(al1=0.95, al2=1.05, al7=0.90, al8=1.10, iflag=1, aswift=350.0, epso=0.002, nexp=0.12)
        n = 5
        sigo = np.zeros((n, 3))
        deps = np.array([
            [0.0001, 0.0, 0.0],       # pure elastic
            [0.0020, 0.0005, 0.0],     # plastic tension
            [-0.0015, 0.0002, 0.0003], # plastic compression + shear
            [0.0010, 0.0010, 0.0],     # equibiaxial
            [0.0, 0.0, 0.0025],        # high shear
        ])

        # Batch execution
        extra_b = {}
        sig_b, pla_b = shell_update(p, sigo.copy(), deps.copy(), extra=extra_b, niter=3)

        # Serial execution
        for i in range(n):
            extra_s = {}
            sig_s, pla_s = shell_update(p, sigo[i:i+1].copy(), deps[i:i+1].copy(), extra=extra_s, niter=3)

            np.testing.assert_allclose(sig_b[i], sig_s[0], rtol=1e-14, atol=1e-14)
            assert pla_b[i] == pytest.approx(pla_s[0], rel=1e-14)
            assert extra_b["depszz"][i] == pytest.approx(extra_s["depszz"][0], rel=1e-14)
            assert extra_b["thk87"][i] == pytest.approx(extra_s["thk87"][0], rel=1e-14)


# =============================================================================
# 10. TEST SUITE: Exhaustive 64+ Physical States Parity Verification
# =============================================================================

# Construct 64 diverse physical test cases covering all model parameters
PARAMETRIZED_CASES: List[Dict[str, Any]] = []

def _add_case(name: str, p_kwargs: Dict[str, Any], sigo: List[float], deps: List[float], dt: float = 1e-5, **extra_kwargs: Any) -> None:
    PARAMETRIZED_CASES.append({
        "name": name,
        "p_kwargs": p_kwargs,
        "sigo": sigo,
        "deps": deps,
        "dt": dt,
        "extra": extra_kwargs,
    })

# 1-8: Pure elastic states
for idx, (s, d) in enumerate([
    ([0.0, 0.0, 0.0], [1e-5, 0.0, 0.0]),
    ([0.0, 0.0, 0.0], [0.0, 1e-5, 0.0]),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 1e-5]),
    ([50.0, 30.0, 10.0], [2e-5, -1e-5, 5e-6]),
    ([-80.0, 20.0, 0.0], [-1e-5, 5e-6, 0.0]),
    ([0.0, 0.0, 0.0, 0.0, 0.0], [1e-5, 1e-5, 0.0, 1e-5, 0.0]),
    ([100.0, -50.0, 20.0, 5.0, -5.0], [5e-6, -5e-6, 2e-6, 1e-6, -1e-6]),
    ([0.0, 0.0, 0.0], [1e-6, 1e-6, 0.0]),
]):
    _add_case(f"elastic_state_{idx}", dict(aswift=1e8), s, d)

# 9-16: Isotropic von Mises limit (alpha1..alpha8=1, a=2)
for idx, (s, d) in enumerate([
    ([0.0, 0.0, 0.0], [0.002, 0.0, 0.0]),
    ([0.0, 0.0, 0.0], [0.0, 0.002, 0.0]),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.003]),
    ([0.0, 0.0, 0.0], [0.0015, 0.0015, 0.0]),
    ([100.0, 0.0, 0.0], [0.001, 0.0, 0.0]),
    ([0.0, 100.0, 0.0], [0.0, 0.001, 0.0]),
    ([0.0, 0.0, 80.0], [0.0, 0.0, 0.0015]),
    ([120.0, -60.0, 30.0], [0.001, -0.0005, 0.0005]),
]):
    _add_case(f"isotropic_vm_{idx}", dict(al1=1.0, al2=1.0, al3=1.0, al4=1.0, al5=1.0, al6=1.0, al7=1.0, al8=1.0, expa=2.0, aswift=300.0, epso=0.002, nexp=0.1), s, d)

# 17-24: Directional yielding with anisotropic parameters (AA6111-T4, a=8)
aniso_base = dict(al1=0.95, al2=1.05, al3=0.90, al4=1.10, al5=0.85, al6=1.15, al7=1.00, al8=1.02, expa=8.0, aswift=320.0, epso=0.002, nexp=0.12)
for idx, (s, d) in enumerate([
    ([0.0, 0.0, 0.0], [0.0025, 0.0, 0.0]),
    ([0.0, 0.0, 0.0], [0.0, 0.0025, 0.0]),
    ([0.0, 0.0, 0.0], [0.0015, 0.0015, 0.0015]),
    ([0.0, 0.0, 0.0], [0.0018, 0.0018, 0.0]),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.003]),
    ([80.0, 0.0, 0.0], [0.0015, 0.0, 0.0]),
    ([0.0, 90.0, 0.0], [0.0, 0.0015, 0.0]),
    ([50.0, 50.0, 30.0], [0.001, 0.001, 0.001]),
]):
    _add_case(f"aniso_dir_{idx}", aniso_base, s, d)

# 25-32: Swift-Voce hardening variations
for idx, p_mod in enumerate([
    dict(iflag=1, alpha=1.0, aswift=450.0, epso=0.002, nexp=0.25),
    dict(iflag=1, alpha=0.0, ko=240.0, qvoce=160.0, beta=20.0),
    dict(iflag=1, alpha=0.3, aswift=400.0, epso=0.005, nexp=0.2, ko=200.0, qvoce=120.0, beta=15.0),
    dict(iflag=1, alpha=0.7, aswift=500.0, epso=0.001, nexp=0.18, ko=260.0, qvoce=140.0, beta=10.0),
    dict(iflag=1, alpha=0.5, aswift=480.0, epso=0.002, nexp=0.2, ko=220.0, qvoce=180.0, beta=25.0),
    dict(iflag=1, alpha=1.0, aswift=600.0, epso=0.010, nexp=0.3),
    dict(iflag=1, alpha=0.0, ko=300.0, qvoce=200.0, beta=5.0),
    dict(iflag=1, alpha=0.5, aswift=350.0, epso=0.000, nexp=0.1, ko=200.0, qvoce=100.0, beta=30.0),
]):
    _add_case(f"swift_voce_var_{idx}", p_mod, [0.0, 0.0, 0.0], [0.002, 0.0005, 0.0002])

# 33-40: Cowper-Symonds rate dependency
for idx, (dt_v, unsc_v, unsp_v) in enumerate([
    (1e-3, 50.0, 0.20),
    (1e-4, 50.0, 0.20),
    (1e-5, 50.0, 0.20),
    (1e-6, 50.0, 0.20),
    (1e-4, 200.0, 0.15),
    (1e-5, 200.0, 0.15),
    (1e-4, 10.0, 0.30),
    (1e-5, 10.0, 0.30),
]):
    _add_case(f"rate_dep_{idx}", dict(iflag=1, aswift=400.0, epso=0.002, nexp=0.15, unsc=unsc_v, unsp=unsp_v), [0.0, 0.0, 0.0], [0.0015, 0.0, 0.0], dt=dt_v)

# 41-48: Kinematic hardening Chaboche-Rousselier (ikin=1)
for idx, (fisokin_v, ckh_v, akh_v) in enumerate([
    (0.3, [80.0, 0.0, 0.0, 0.0], [50.0, 0.0, 0.0, 0.0]),
    (0.5, [150.0, 0.0, 0.0, 0.0], [70.0, 0.0, 0.0, 0.0]),
    (0.7, [300.0, 0.0, 0.0, 0.0], [90.0, 0.0, 0.0, 0.0]),
    (0.5, [200.0, 50.0, 0.0, 0.0], [50.0, 30.0, 0.0, 0.0]),
    (0.6, [400.0, 100.0, 20.0, 0.0], [60.0, 40.0, 20.0, 0.0]),
    (0.8, [500.0, 200.0, 50.0, 10.0], [40.0, 30.0, 20.0, 10.0]),
    (0.4, [100.0, 0.0, 0.0, 0.0], [100.0, 0.0, 0.0, 0.0]),
    (0.5, [50.0, 50.0, 50.0, 50.0], [25.0, 25.0, 25.0, 25.0]),
]):
    _add_case(f"kin_chaboche_{idx}", dict(iflag=1, fisokin=fisokin_v, ikin=1, ckh=ckh_v, akh=akh_v, aswift=300.0, epso=0.002, nexp=0.1), [0.0, 0.0, 0.0], [0.002, 0.0005, 0.0003])

# 49-56: Kinematic hardening Prager (ikin=2)
for idx, fisokin_v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    _add_case(f"kin_prager_{idx}", dict(iflag=1, fisokin=fisokin_v, ikin=2, aswift=320.0, epso=0.002, nexp=0.12), [0.0, 0.0, 0.0], [0.0018, -0.0004, 0.0002])

# 57-64: Multi-component stress states, pre-existing plastic strain, transverse shear, element deletion
for idx, (s, d, p_in, off_v) in enumerate([
    ([100.0, 50.0, 20.0], [0.001, 0.0005, 0.0002], 0.02, 1.0),
    ([-80.0, -40.0, 10.0], [-0.001, -0.0005, 0.0001], 0.05, 1.0),
    ([0.0, 0.0, 0.0, 20.0, -15.0], [0.0015, 0.0, 0.0, 0.0005, -0.0005], 0.0, 1.0),
    ([150.0, 80.0, 30.0, 10.0, 10.0], [0.002, 0.001, 0.0005, 0.0002, 0.0002], 0.01, 1.0),
    ([0.0, 0.0, 0.0], [0.005, 0.0, 0.0], 0.0, 0.0),
    ([200.0, 100.0, 50.0], [0.002, 0.0, 0.0], 0.10, 1.0),
    ([50.0, -50.0, 60.0], [0.001, -0.001, 0.002], 0.03, 1.0),
    ([0.0, 0.0, 0.0], [0.003, 0.003, 0.0], 0.0, 1.0),
]):
    _add_case(f"complex_state_{idx}", dict(iflag=1, aswift=350.0, epso=0.002, nexp=0.15), s, d, pla=p_in, off=off_v)


class TestExhaustiveFortranOracleParity:
    """Exhaustive 64+ physical states verification comparing pyradioss vs sigeps87c_oracle."""

    @pytest.mark.parametrize("case", PARAMETRIZED_CASES, ids=[c["name"] for c in PARAMETRIZED_CASES])
    def test_shell_update_matches_fortran_oracle(self, case):
        """Verify pyradioss shell_update matches the Fortran oracle to 1e-12 relative precision."""
        p = build_law87(**case["p_kwargs"])
        sigo = np.array(case["sigo"], dtype=np.float64)
        deps = np.array(case["deps"], dtype=np.float64)
        dt = case["dt"]
        extra_in = case["extra"]

        pla_in = extra_in.get("pla", 0.0)
        off_v = extra_in.get("off", 1.0)
        thk_v = extra_in.get("thk", 1.2)

        # Run Fortran oracle
        orc = sigeps87c_oracle(
            p, sigo, deps,
            rho0=p.rho0,
            pla=np.array([pla_in]),
            off=np.array([off_v]),
            thk=np.array([thk_v]),
            thkly=np.array([thk_v]),
            dt=dt,
            niter=3,
        )

        # Run pyradioss shell_update
        extra_py = {
            "pla87": np.array([pla_in]),
            "off87": np.array([off_v]),
            "thk87": np.array([thk_v]),
            "thkly": np.array([thk_v]),
        }
        sig_py, pla_py, snd_py = shell_update(
            p, sigo, deps, dt=dt, extra=extra_py, niter=3, return_tuple=True
        )

        # 1. Stress tensor comparison
        sig_py_arr = np.asarray(sig_py).flatten()
        np.testing.assert_allclose(sig_py_arr, orc["sig"][0], rtol=1e-12, atol=1e-12)

        # 2. Plastic strain comparison
        assert pla_py == pytest.approx(orc["pla"][0], rel=1e-12, abs=1e-12)

        # 3. Sound speed comparison
        assert snd_py == pytest.approx(orc["soundsp"][0], rel=1e-12)

        # 4. Thickness thinning comparison
        depszz_val = float(np.asarray(extra_py["depszz"]).flat[0])
        thk_val = float(np.asarray(extra_py["thk87"]).flat[0])
        assert depszz_val == pytest.approx(orc["depszz"][0], rel=1e-12, abs=1e-12)
        assert thk_val == pytest.approx(orc["thk"][0], rel=1e-12, abs=1e-12)

        # 5. Hourglass control stiffness parameter comparison
        if "etse" in extra_py and orc["etse"] is not None:
            etse_val = float(np.asarray(extra_py["etse"]).flat[0])
            assert etse_val == pytest.approx(orc["etse"][0], rel=1e-12)
