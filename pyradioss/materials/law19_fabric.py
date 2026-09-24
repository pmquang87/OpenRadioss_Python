# Ported from OpenRadioss Fortran:
# Upstream Fortran origins:
# - engine: engine/source/materials/mat/mat019/sigeps19c.F (SUBROUTINE SIGEPS19C, lines 30-179) [shells]
# - starter: starter/source/materials/mat/mat019/hm_read_mat19.F (SUBROUTINE HM_READ_MAT19, lines 38-271) [starter reader]
# - starter: starter/source/materials/mat/mat019/law19_upd.F90 (SUBROUTINE LAW19_UPD, lines 47-84) [sensor updating]
#
# Context:
# LAW19 (/MAT/LAW19, /MAT/FABRI) models orthotropic fabric and membrane materials used in
# automotive airbags, sports balls, architectural membranes, and woven composites.
# Features:
# 1. Total-strain based orthotropic in-plane elastic formulation in two fiber directions (warp/weft).
# 2. Reduced compression stiffness (RCOMP) representing fiber/yarn buckling under compression.
# 3. Bilinear fiber stress-strain curve support with knee strain and tangent modulus (ET1, ET2).
# 4. Rate-dependent strain-rate enhancement (C_RATE, EPS0).
# 5. Ogden-type hyperelastic membrane stretch extension (MU_OGDEN, ALPHA_OGDEN).
# 6. REF-STATE zerostress option for folded airbag initialization and sensor-gated stress relaxation.
# 7. Transverse shear integration (G23, G31) with shell shear factor SHF = 5/6.
"""
LAW19 — Orthotropic membrane fabric with reduced compression, bilinear/rate hardening,
Ogden-type hyperelasticity, and REF-STATE zerostress relaxation (/MAT/LAW19, /MAT/FABRI).

Upstream OpenRadioss Fortran reference:
- engine: engine/source/materials/mat/mat019/sigeps19c.F (lines 30-179)
- starter: starter/source/materials/mat/mat019/hm_read_mat19.F (lines 38-271)

Theory:
-------
1. Orthotropic total-strain in-plane response:
   sig_xx = A11 * eps_xx + A12 * eps_yy
   sig_yy = A12 * eps_xx + A22 * eps_yy
   sig_xy = G12 * gam_xy   (engineering shear)

   Derived constants:
   NU21 = NU12 * E22 / E11
   DETC = 1 - NU12 * NU21
   A11 = E11 / DETC,  A22 = E22 / DETC,  A12 = A11 * NU21

2. Bilinear fiber response (past knee strain epsy1, epsy2):
   When eps_xx > epsy1 and et1 > 0:
       sig_xx -= (A11 - A11_t) * (eps_xx - epsy1)
   When eps_yy > epsy2 and et2 > 0:
       sig_yy -= (A22 - A22_t) * (eps_yy - epsy2)

3. Rate-dependent enhancement:
   For strain rates eps_dot > eps0 and c_rate > 0:
       F_rate = 1 + c_rate * ln(eps_dot / eps0)
       sig_xx *= F_rate,  sig_yy *= F_rate

4. Ogden-type hyperelastic extension (rubberized / coated fabric):
   lambda_1 = 1 + eps_xx,  lambda_2 = 1 + eps_yy
   sig_xx += mu * (lambda_1^alpha - lambda_1^(-alpha/2))
   sig_yy += mu * (lambda_2^alpha - lambda_2^(-alpha/2))

5. Reduced compression scaling (sigeps19c.F lines 106-127):
   Principal in-plane stresses P1 <= P2:
   S = (sig_xx + sig_yy)/2,  D = (sig_xx - sig_yy)/2,  R = sqrt(sig_xy^2 + D^2)
   P1 = S - R,  P2 = S + R
   - If P1 < 0 and P2 > 0 (mixed tension/compression):
     beta = ((1 - RCOMP)*S/R + 1 + RCOMP)/2
     sig_xx = beta*(sig_xx - P2) + P2
     sig_yy = beta*(sig_yy - P2) + P2
     sig_xy = beta*sig_xy
   - If P1 < 0 and P2 <= 0 (bi-compression):
     sig_xx *= RCOMP,  sig_yy *= RCOMP,  sig_xy *= RCOMP

6. REF-STATE zerostress (sigeps19c.F lines 129-168):
   - For TT <= TSTART: sig = 0, sigi = sig_elastic
   - For TT > TSTART: sigi relaxes towards 0 on unloading, sig = sig_elastic - sigi.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20
_INF = 1e30


@dataclass
class FabricMaterial(Material):
    """LAW19 material carrying the upstream 'effective' elastic constants
    for every kernel-side use (PM(20)/PM(21)/PM(22)/PM(27) exactly):

    * ``E``  = MAX(E11, E22)/DETC   (PM(20) = PM(24) = C1)
    * ``nu`` = sqrt(NU12*NU21)      (PM(21))
    * ``G``  = MAX(G12, G23, G31)   (PM(22) — kernel transverse shear
                                     and hourglass)
    * ``sound_speed_shell`` = sqrt(C1/rho0)  (PM(27) SSP)
    """

    @property
    def G(self) -> float:                       # PM(22)
        p = self.params
        return max(p.get("G12", 0.0), p.get("G23", 0.0), p.get("G31", 0.0))

    def sound_speed_shell(self) -> float:       # PM(27)
        return float(np.sqrt(self.params["E"] / self.rho0))


@dataclass
class Law19Params:
    """Strongly-typed parameter set for LAW19 (/MAT/LAW19, /MAT/FABRI)."""

    rho0: float = 1.0
    refer_rho: float = 1.0
    e11: float = 1000.0
    e22: float = 1000.0
    nu12: float = 0.3
    g12: float = 500.0
    g23: float = 500.0
    g31: float = 500.0
    rcomp: float = 1.0
    zerostress: float = 0.0
    tstart: float = 0.0
    isensor: int = 0
    porosity: float = 0.0

    # Bilinear fiber parameters
    et1: float = 0.0
    et2: float = 0.0
    epsy1: float = _INF
    epsy2: float = _INF

    # Rate dependence
    c_rate: float = 0.0
    eps0: float = 1.0

    # Ogden-type hyperelasticity
    mu_ogden: float = 0.0
    alpha_ogden: float = 0.0

    title: str = ""

    # Derived
    nu21: float = field(init=False)
    detc: float = field(init=False)
    a11: float = field(init=False)
    a22: float = field(init=False)
    a12: float = field(init=False)
    c1: float = field(init=False)

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        self.nu21 = self.nu12 * self.e22 / self.e11 if self.e11 > 0.0 else self.nu12
        self.detc = 1.0 - self.nu12 * self.nu21
        if self.detc <= 0.0:
            self.detc = 1e-6
        self.a11 = self.e11 / self.detc
        self.a22 = self.e22 / self.detc
        self.a12 = self.a11 * self.nu21
        self.c1 = max(self.e11, self.e22) / self.detc
        if self.rcomp <= 0.0:
            self.rcomp = 1.0
        self.rcomp = max(self.rcomp, 1e-3)


def _extract(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def resolve(mat: Any) -> FabricMaterial:
    """Resolve FabricMaterial from Material, dict, or FabricMaterial."""
    if isinstance(mat, FabricMaterial):
        return mat
    if hasattr(mat, "params") and isinstance(mat.params, dict) and "A11" in mat.params:
        return FabricMaterial(
            id=getattr(mat, "id", 1),
            law=19,
            rho0=getattr(mat, "rho0", 1.0),
            title=getattr(mat, "title", "FABRIC"),
            params=dict(mat.params),
        )

    p: Dict[str, Any] = {}
    rho0 = 1.0
    title = "FABRIC"
    mat_id = 1

    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        rho0 = getattr(mat, "rho0", 1.0)
        title = getattr(mat, "title", "FABRIC")
        mat_id = getattr(mat, "id", 1)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        rho0 = mat.get("rho0") or mat.get("rho") or mat.get("density") or 1.0
        title = mat.get("title", "FABRIC")
        mat_id = mat.get("id", 1)
    elif hasattr(mat, "__dict__"):
        p = dict(getattr(mat, "params", mat.__dict__))
        rho0 = getattr(mat, "rho0", getattr(mat, "rho", getattr(mat, "density", 1.0)))
        title = getattr(mat, "title", "FABRIC")
        mat_id = getattr(mat, "id", 1)

    if "G12" in p and p["G12"]:
        p.setdefault("G23", p["G12"])
        p.setdefault("G31", p["G12"])

    rec = SimpleNamespace(id=mat_id, title=title, density=rho0, params=p)
    return build_fabric(rec)


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Union[Tuple[np.ndarray, Optional[np.ndarray]], Tuple[np.ndarray, Optional[np.ndarray], float]]:
    """One layer plane-stress update (vectorized over the part slice) — sigeps19c.F.

    Total-strain law: the incoming (Jaumann-rotated) stress is only used
    as SIGO by the zerostress relaxation; the new stress is rebuilt from
    the accumulated strain in the orthotropy frame.
    """
    n = sig.shape[0]
    if n == 0:
        if return_sound_speed:
            return sig, epsp if epsp is not None else np.empty(0), 0.0
        return sig, epsp if epsp is not None else np.empty(0)

    p = mat.params if hasattr(mat, "params") else mat
    a11, a22, a12 = p["A11"], p["A22"], p["A12"]
    g12 = p["G12"]
    rcomp = p.get("RCOMP", 1.0)
    zerostress = p.get("ZEROSTRESS", 0.0)
    isens = p.get("ISENSOR", 0)
    tstart = 0.0
    if isens > 0 and getattr(mat, "sensors", None) is not None:
        if mat.sensors.active(isens):
            tstart = mat.sensors.fire_time.get(isens, 0.0)
        else:
            tstart = 1e20
    else:
        tstart = p.get("TSTART", 0.0)

    # Bilinear fiber parameters
    epsy1 = float(p.get("EPSY1", p.get("MAT_EPSY1", _INF)))
    epsy2 = float(p.get("EPSY2", p.get("MAT_EPSY2", _INF)))
    et1 = float(p.get("ET1", p.get("MAT_ET1", 0.0)))
    et2 = float(p.get("ET2", p.get("MAT_ET2", 0.0)))

    # Rate dependence
    c_rate = float(p.get("C_RATE", p.get("CRATE", p.get("MAT_CRATE", 0.0))))
    eps0 = float(p.get("EPS0", p.get("MAT_EPS0", 1.0)))

    # Ogden-type hyperelastic extension
    mu_ogden = float(p.get("MU_OGDEN", p.get("MU", p.get("MAT_MU", 0.0))))
    alpha_ogden = float(p.get("ALPHA_OGDEN", p.get("ALPHA", p.get("MAT_ALPHA", 0.0))))

    if extra is None:
        extra = {}
    eps = extra.get("eps19")
    if eps is None or eps.shape[0] != n:
        eps = np.zeros((n, 3))
        extra["eps19"] = eps
    sigi = extra.get("sigi19")
    if sigi is None or sigi.shape[0] != n:
        sigi = np.zeros((n, 3))
        extra["sigi19"] = sigi
    tt = extra.get("t19")
    if tt is None or len(tt) != n:
        tt = np.zeros(n)
        extra["t19"] = tt

    deps_arr = np.atleast_2d(deps)
    if deps_arr.shape[1] >= 3:
        eps += deps_arr[:, :3]
    else:
        eps[:, :deps_arr.shape[1]] += deps_arr

    sigo = sig.copy()  # SIGOXX/SIGOYY/SIGOXY

    # ---- 1. Elastic total-strain stress in the orthotropy frame (sigeps19c.F:96-102) ----
    sxx = a11 * eps[:, 0] + a12 * eps[:, 1]
    syy = a12 * eps[:, 0] + a22 * eps[:, 1]
    sxy = g12 * eps[:, 2]

    # ---- 2. Bilinear fiber response past knee strain ----
    nu12 = p.get("NU12", 0.3)
    nu21 = p.get("NU21", nu12 * p["E22"] / p["E11"] if p.get("E11", 0.0) > 0.0 else nu12)
    detc = 1.0 - nu12 * nu21
    if detc <= 0.0:
        detc = 1e-6

    if et1 > 0.0 and epsy1 < _INF:
        at11 = et1 / detc
        mask1 = eps[:, 0] > epsy1
        if np.any(mask1):
            sxx[mask1] -= (a11 - at11) * (eps[mask1, 0] - epsy1)

    if et2 > 0.0 and epsy2 < _INF:
        at22 = et2 / detc
        mask2 = eps[:, 1] > epsy2
        if np.any(mask2):
            syy[mask2] -= (a22 - at22) * (eps[mask2, 1] - epsy2)

    # ---- 3. Rate-dependent strain-rate enhancement ----
    if c_rate > 0.0 and dt > 0.0:
        epsdot1 = np.abs(deps_arr[:, 0]) / dt
        epsdot2 = np.abs(deps_arr[:, 1]) / dt
        frate1 = 1.0 + c_rate * np.log(np.maximum(1.0, epsdot1 / max(eps0, 1e-12)))
        frate2 = 1.0 + c_rate * np.log(np.maximum(1.0, epsdot2 / max(eps0, 1e-12)))
        sxx *= frate1
        syy *= frate2

    # ---- 4. Ogden-type hyperelastic membrane extension ----
    if mu_ogden > 0.0 and abs(alpha_ogden) > 1e-12:
        lam1 = np.maximum(1.0 + eps[:, 0], 1e-6)
        lam2 = np.maximum(1.0 + eps[:, 1], 1e-6)
        sxx += mu_ogden * (lam1 ** alpha_ogden - lam1 ** (-0.5 * alpha_ogden))
        syy += mu_ogden * (lam2 ** alpha_ogden - lam2 ** (-0.5 * alpha_ogden))

    # ---- 5. Reduced compression (principal-stress scaling, sigeps19c.F:106-127) ----
    s = 0.5 * (sxx + syy)
    d = 0.5 * (sxx - syy)
    r = np.sqrt(sxy ** 2 + d * d)
    p1 = s - r
    p2 = s + r
    mixed = (p1 < 0.0) & (p2 > 0.0)
    if np.any(mixed):
        r_safe = np.maximum(r[mixed], 1e-20)
        beta = 0.5 * ((1.0 - rcomp) * s[mixed] / r_safe + 1.0 + rcomp)
        p2m = p2[mixed]
        sxx[mixed] = beta * (sxx[mixed] - p2m) + p2m
        syy[mixed] = beta * (syy[mixed] - p2m) + p2m
        sxy[mixed] = beta * sxy[mixed]
    bicomp = (p1 < 0.0) & (p2 <= 0.0)
    if np.any(bicomp):
        sxx[bicomp] *= rcomp
        syy[bicomp] *= rcomp
        sxy[bicomp] *= rcomp

    # ---- 6. REF-STATE zerostress option (sigeps19c.F:129-168) ----
    if zerostress != 0.0:
        hold = tt <= tstart
        if np.any(hold):
            sigi[hold, 0] = sxx[hold]
            sigi[hold, 1] = syy[hold]
            sigi[hold, 2] = sxy[hold]
            sxx[hold] = 0.0
            syy[hold] = 0.0
            sxy[hold] = 0.0
        rest = ~hold
        if np.any(rest):
            for k, snew in enumerate((sxx, syy, sxy)):
                dsig = snew[rest] - sigo[rest, k] - sigi[rest, k]
                si = sigi[rest, k]
                si = np.where((si > 0.0) & (dsig < 0.0),
                              np.maximum(0.0, si + zerostress * dsig), si)
                si = np.where((si < 0.0) & (dsig > 0.0),
                              np.minimum(0.0, si + zerostress * dsig), si)
                sigi[rest, k] = si
            sxx[rest] -= sigi[rest, 0]
            syy[rest] -= sigi[rest, 1]
            sxy[rest] -= sigi[rest, 2]
        if dt > 0.0:
            tt += dt

    sig[:, 0] = sxx
    sig[:, 1] = syy
    sig[:, 2] = sxy

    # Transverse shear (G23/G31) if 5-component tensor passed
    if sig.shape[1] >= 5 and deps_arr.shape[1] >= 5:
        g23 = p.get("G23", g12)
        g31 = p.get("G31", g12)
        sig[:, 3] += g23 * deps_arr[:, 3]
        sig[:, 4] += g31 * deps_arr[:, 4]

    if return_sound_speed:
        c_spd = sound_speed(mat)
        return sig, epsp, c_spd
    return sig, epsp


def solid_update(mat: Any, sig: Any = None, deps: Any = None, dt: float = 0.0, extra: Any = None, *args: Any, **kwargs: Any) -> Any:
    """LAW19 is defined strictly for shell/membrane elements.
    OpenRadioss starter rejects it on solids (starter/source/materials/mat/mat019/hm_read_mat19.F).
    """
    if sig is not None and hasattr(sig, "shape") and sig.shape[0] == 0:
        return sig, np.empty(0)
    raise NotImplementedError("LAW19 (fabric) is implemented for shell elements only.")


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """(3, 3) unreduced elastic plane-stress orthotropic membrane tangent."""
    p = mat.params if hasattr(mat, "params") else mat
    a11 = p.get("A11", p.get("E", 1.0))
    a22 = p.get("A22", p.get("E", 1.0))
    a12 = p.get("A12", 0.0)
    g12 = p.get("G12", p.get("G", 0.0))
    return np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, g12],
    ], dtype=float)


shell_tangent = shell_membrane_tangent


def consistent_shell_tangent(mat: Any, extra: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """(n, 3, 3) consistent plane-stress shell tangent tensor for LAW19 fabric.
    Accounts for reduced compression stiffness scaling (RCOMP) and mixed tension/compression
    scaling (beta).
    """
    p = mat.params if hasattr(mat, "params") else mat
    a11 = p.get("A11", p.get("E", 1.0))
    a22 = p.get("A22", p.get("E", 1.0))
    a12 = p.get("A12", 0.0)
    g12 = p.get("G12", p.get("G", 0.0))
    rcomp = p.get("RCOMP", 1.0)
    C_el = np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, g12],
    ], dtype=float)

    if extra is None or "eps19" not in extra:
        return np.empty((0, 3, 3))
    eps = extra["eps19"]
    n = eps.shape[0]
    if n == 0:
        return np.empty((0, 3, 3))

    # Evaluate unreduced orthotropic stress state
    sxx = a11 * eps[:, 0] + a12 * eps[:, 1]
    syy = a12 * eps[:, 0] + a22 * eps[:, 1]
    sxy = g12 * eps[:, 2]

    s = 0.5 * (sxx + syy)
    d = 0.5 * (sxx - syy)
    r = np.sqrt(sxy ** 2 + d * d)
    p1 = s - r
    p2 = s + r

    # Default: pure tension (P1 >= 0) -> full elastic stiffness C_el
    D = np.broadcast_to(C_el, (n, 3, 3)).copy()

    # Bi-compression (P1 < 0 and P2 <= 0) -> scaled by RCOMP
    bicomp = (p1 < 0.0) & (p2 <= 0.0)
    if np.any(bicomp):
        D[bicomp] *= rcomp

    # Mixed tension/compression (P1 < 0 and P2 > 0) -> continuous beta scaling
    mixed = (p1 < 0.0) & (p2 > 0.0)
    if np.any(mixed):
        r_safe = np.maximum(r[mixed], 1e-20)
        beta = 0.5 * ((1.0 - rcomp) * s[mixed] / r_safe + 1.0 + rcomp)
        D[mixed] *= beta[:, None, None]

    return D


def tangent(mat: Any = None, **kwargs: Any) -> np.ndarray:
    """Plane-stress membrane tangent matrix matching material law template."""
    return shell_membrane_tangent(mat)


def sound_speed(mat: Any, eps: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None) -> float:
    """Sound speed for LAW19 fabric."""
    if hasattr(mat, "sound_speed_shell"):
        return mat.sound_speed_shell()
    p = mat.params if hasattr(mat, "params") else mat
    e_val = p.get("E", max(p.get("E11", 1000.0), p.get("E22", 1000.0)))
    rho = getattr(mat, "rho0", p.get("rho0", 1.0))
    return float(np.sqrt(max(e_val / max(rho, 1e-20), 1e-20)))


def extra_shapes(mat: Any = None, nip: Optional[int] = 1) -> Dict[str, Tuple[int, ...]]:
    """Extra history variables: total strain, zerostress reference stress, and time."""
    if nip:
        return {"eps19": (nip, 3), "sigi19": (nip, 3), "t19": (nip,)}
    return {"eps19": (3,), "sigi19": (3,), "t19": ()}


def needs_defgrad(mat: Any = None) -> bool:
    return False


# ----------------------------------------------------------------------------
# cfg-record constructor (mat_reader physics registry)
# ----------------------------------------------------------------------------

def build_fabric(rec: Any) -> FabricMaterial:
    """hm_read_mat19.F: cfg attributes -> derived constants -> Material.
    Raises ValueError on the upstream fatal checks (ANCMSG 306/307).
    """
    p = rec.params if hasattr(rec, "params") else rec
    e11 = float(p.get("MAT_EA") or p.get("E11") or 0.0)
    e22 = float(p.get("MAT_EB") or p.get("E22") or 0.0)
    n12 = float(p.get("MAT_PRAB") or p.get("NU12") or 0.0)
    g12 = float(p.get("MAT_GAB") or p.get("G12") or 0.0)
    g23 = float(p.get("MAT_GBC") or p.get("G23") or 0.0)
    g31 = float(p.get("MAT_GCA") or p.get("G31") or 0.0)

    rcomp = float(p.get("MAT_REDFACT") or p.get("RCOMP") or 0.0)
    zerostress = float(p.get("M58_Zerostress") or p.get("ZEROSTRESS") or 0.0)
    porosity = float(p.get("MAT_POROS") or p.get("POROSITY") or 0.0)
    isens = int(p.get("ISENSOR") or 0)

    if e11 <= 0.0 or e22 <= 0.0 or g12 <= 0.0 or g23 <= 0.0 or g31 <= 0.0:
        raise ValueError("E11, E22, G12, G23 and G31 must all be nonzero and positive "
                         "(hm_read_mat19 error 306)")
    n21 = n12 * e22 / e11
    detc = 1.0 - n12 * n21
    if detc <= 0.0:
        raise ValueError("1 - NU12*NU21 <= 0: non positive-definite "
                         "orthotropic matrix (hm_read_mat19 error 307)")
    if rcomp == 0.0:
        rcomp = 1.0                       # default: NO compression reduction
    rcomp = max(rcomp, 1e-3)              # upstream warning 1572 clamp

    a11 = e11 / detc
    a22 = e22 / detc
    a12 = a11 * n21
    c1 = max(e11, e22) / detc             # PM(20)=PM(24)=PM(32)

    params = {
        "E": c1, "nu": float(np.sqrt(max(n12 * n21, 0.0))),
        "E11": e11, "E22": e22, "NU12": n12, "NU21": n21,
        "G12": g12, "G23": g23, "G31": g31,
        "A11": a11, "A22": a22, "A12": a12,
        "RCOMP": rcomp, "ZEROSTRESS": zerostress,
        "POROSITY": porosity, "ISENSOR": isens,
        "TSTART": float(p.get("TSTART", 0.0) or 0.0),
        # Bilinear & nonlinear extensions
        "ET1": float(p.get("ET1", p.get("MAT_ET1", 0.0))),
        "ET2": float(p.get("ET2", p.get("MAT_ET2", 0.0))),
        "EPSY1": float(p.get("EPSY1", p.get("MAT_EPSY1", _INF))),
        "EPSY2": float(p.get("EPSY2", p.get("MAT_EPSY2", _INF))),
        "C_RATE": float(p.get("C_RATE", p.get("CRATE", p.get("MAT_CRATE", 0.0)))),
        "EPS0": float(p.get("EPS0", p.get("MAT_EPS0", 1.0))),
        "MU_OGDEN": float(p.get("MU_OGDEN", p.get("MU", p.get("MAT_MU", 0.0)))),
        "ALPHA_OGDEN": float(p.get("ALPHA_OGDEN", p.get("ALPHA", p.get("MAT_ALPHA", 0.0)))),
    }
    rec_id = getattr(rec, "id", 1)
    rec_density = getattr(rec, "density", getattr(rec, "rho0", 1.0))
    rec_title = getattr(rec, "title", "FABRIC")
    return FabricMaterial(id=rec_id, law=19, rho0=rec_density,
                          title=rec_title, params=params)


def _register():
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        MAT_PHYSICS_REGISTRY.setdefault("FABRI", build_fabric)
        MAT_PHYSICS_REGISTRY.setdefault("LAW19", build_fabric)
    except (ImportError, AttributeError):
        pass


_register()
