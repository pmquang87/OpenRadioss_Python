"""
LAW19 — orthotropic linear-elastic membrane FABRIC with reduced
compression stiffness and the REF-STATE "zerostress" option
(/MAT/FABRI, /MAT/LAW19).  Shells only (airbag / football / seat fabric).

Fortran origin
--------------
* engine : ``engine/source/materials/mat/mat019/sigeps19c.F`` (the law
  exists only in its shell form — called from mulawc);
* starter: ``starter/source/materials/mat/mat019/hm_read_mat19.F``
  (derived constants: NU21 = NU12*E22/E11, DETC = 1 - NU12*NU21,
  A11 = E11/DETC, A22 = E22/DETC, A12 = A11*NU21,
  C1 = MAX(E11,E22)/DETC, SSP = sqrt(C1/RHO0); defaults RCOMP: 0 -> 1,
  clamped to >= 1e-3).

Theory (what sigeps19c does, line for line)
-------------------------------------------
The law is TOTAL-strain based: each cycle the in-plane stress is rebuilt
from the accumulated strain in the orthotropy frame,

    sig_xx = A11 eps_xx + A12 eps_yy
    sig_yy = A12 eps_xx + A22 eps_yy
    sig_xy = G12 gam_xy                       (engineering shear)

then two corrections:

1. **Reduced compression** (RCOMP = the deck's R_E factor): fabric has no
   real compressive stiffness (yarns buckle).  The in-plane principal
   stresses P1 <= P2 are computed; if P1 < 0 the stress is scaled:

   * P2 > 0 (mixed tension/compression): the stress tensor is shrunk
     AROUND the tensile principal value P2 with the interpolated factor
     BETA = ((1-RCOMP) S/R + 1 + RCOMP)/2   (S = mean, R = radius), which
     is 1 when P1 -> 0 and RCOMP when P2 -> 0 — a continuous blend;
   * P2 <= 0 (bi-compression): the whole tensor is scaled by RCOMP.

2. **Zerostress** (REF-STATE, M58_Zerostress): used with /REFSTA or /XREF
   folded-airbag geometries.  While TT <= TSTART the just-computed stress
   is stored per element in SIGI and the output stress is zeroed — the
   reference (folded) shape carries no stress.  Afterwards each component
   of SIGI decays toward zero at the given rate whenever the increment
   DSIG = SIG_new - SIG_old - SIGI unloads it, and the output is
   SIG - SIGI.  TSTART comes from an optional /SENSOR.

Transverse shear (G23/G31 with the 5/6 factor) is handled by the shell
kernel's elastic q-update using ``mat.G`` — :class:`FabricMaterial`
overrides ``G`` to the upstream PM(22) = max(G12, G23, G31) and the
sound speed to the upstream PM(27) = sqrt(C1/rho0) so the kernel-side
uses (time step, hourglass, transverse shear, contact stiffness) match
hm_read_mat19's PM table.

Documented deviations of the port
---------------------------------
* the QEPH ZCFAC stiffness-reduction feedback (FLAG_ZCFAC) has no port
  equivalent (the port shells are BT4/C3-style);
* if G23 != G31 the kernel's single transverse modulus uses their max
  (upstream integrates each with its own modulus inside the law);
* POROSITY (FVM airbag leakage scale) is parsed and stored, unused here
  (it acts in the FVM monvol, not in the stress update).

Extra state (allocated per layer by the shell kernels via
materials.extra_shapes):
    eps19  (m, 3)  accumulated total strain [xx, yy, xy(eng)] in the
                   element/orthotropy frame
    sigi19 (m, 3)  zerostress reference stress SIGI
    t19    (m,)    accumulated time (the Fortran reads the global TT;
                   the port's shell kernels do not pass time, so the law
                   integrates it — bit-identical for a constant-start
                   run with TSTART = 0)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.entities import Material


@dataclass
class FabricMaterial(Material):
    """LAW19 material carrying the upstream 'effective' elastic constants
    for every kernel-side use (PM(20)/PM(21)/PM(22)/PM(27) exactly):

    * ``E``  = MAX(E11, E22)/DETC   (PM(20) = PM(24) = C1)
    * ``nu`` = sqrt(NU12*NU21)      (PM(21))
    * ``G``  = MAX(G12, G23, G31)   (PM(22) — kernel transverse shear
                                     and hourglass)
    * ``sound_speed_shell`` = sqrt(C1/rho0)  (PM(27) SSP — NOT the base
      class's plane-stress formula, which would divide by (1-nu^2) a
      second time)
    """

    @property
    def G(self) -> float:                       # PM(22)
        p = self.params
        return max(p.get("G12", 0.0), p.get("G23", 0.0), p.get("G31", 0.0))

    def sound_speed_shell(self) -> float:       # PM(27)
        return float(np.sqrt(self.params["E"] / self.rho0))


def shell_update(mat, sig: np.ndarray, deps: np.ndarray,
                 epsp=None, dt: float = 0.0, extra: dict = None):
    """One layer update (vectorized over the part slice) — sigeps19c.F.
    Total-strain law: the incoming (Jaumann-rotated) stress is only used
    as SIGO by the zerostress relaxation; the new stress is rebuilt from
    the accumulated strain."""
    n = sig.shape[0]
    if n == 0:
        return sig, epsp if epsp is not None else np.empty(0)

    p = mat.params
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

    if extra is None:
        extra = {}
    eps = extra.get("eps19")
    if eps is None:
        eps = np.zeros((n, 3))
        extra["eps19"] = eps
    sigi = extra.get("sigi19")
    if sigi is None:
        sigi = np.zeros((n, 3))
        extra["sigi19"] = sigi
    tt = extra.get("t19")
    if tt is None:
        tt = np.zeros(n)
        extra["t19"] = tt

    if deps.ndim == 1:
        deps = deps.reshape(1, -1)
    if deps.shape[1] >= 3:
        eps += deps[:, :3]
    else:
        eps[:, :deps.shape[1]] += deps

    sigo = sig.copy()                       # SIGOXX/SIGOYY/SIGOXY

    # ---- elastic total-strain stress in the orthotropy frame -------------
    sxx = a11 * eps[:, 0] + a12 * eps[:, 1]
    syy = a12 * eps[:, 0] + a22 * eps[:, 1]
    sxy = g12 * eps[:, 2]

    # ---- reduced compression (principal-stress scaling) -------------------
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

    # ---- REF-STATE zerostress option --------------------------------------
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
    if sig.shape[1] >= 5 and deps.shape[1] >= 5:
        g23 = p.get("G23", g12)
        g31 = p.get("G31", g12)
        sig[:, 3] += g23 * deps[:, 3]
        sig[:, 4] += g31 * deps[:, 4]

    return sig, epsp


def solid_update(mat, sig, deps, dt=0.0, extra=None):
    """LAW19 is defined strictly for shell elements.
    OpenRadioss starter rejects it on solids (starter/source/materials/mat/mat019/hm_read_mat19.F).
    """
    raise NotImplementedError("LAW19 (fabric) is implemented for shell elements only.")


def shell_membrane_tangent(mat):
    """(3, 3) unreduced elastic plane-stress orthotropic membrane tangent."""
    p = mat.params
    a11 = p.get("A11", p.get("E", 1.0))
    a22 = p.get("A22", p.get("E", 1.0))
    a12 = p.get("A12", 0.0)
    g12 = p.get("G12", p.get("G", 0.0))
    return np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, g12],
    ], dtype=float)


def consistent_shell_tangent(mat, extra=None):
    """(n, 3, 3) consistent plane-stress shell tangent tensor for LAW19 fabric.
    Accounts for reduced compression stiffness scaling (RCOMP) and mixed tension/compression
    scaling (beta).
    """
    p = mat.params
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


# ----------------------------------------------------------------------------
# cfg-record constructor (mat_reader physics registry)
# ----------------------------------------------------------------------------

def build_fabric(rec) -> FabricMaterial:
    """hm_read_mat19.F: cfg attributes -> derived constants -> Material.
    Raises ValueError on the upstream fatal checks (ANCMSG 306/307)."""
    p = rec.params
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
    }
    return FabricMaterial(id=rec.id, law=19, rho0=rec.density,
                          title=rec.title, params=params)


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY.setdefault("FABRI", build_fabric)
    MAT_PHYSICS_REGISTRY.setdefault("LAW19", build_fabric)


_register()
