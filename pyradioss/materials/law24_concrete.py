"""
LAW24 — reinforced-concrete smeared-crack / cap plasticity
(/MAT/CONC, /MAT/LAW24).  Solids only.

Fortran origin (translated block by block)
------------------------------------------
``engine/source/materials/mat/mat024/``:

    m24law.F   entry (sound speed = sqrt(PM24/PM1), strain-rate output)
    conc24.F   driver: elastic prediction -> criterion -> branch
    elas24.F   damage-degraded Hooke prediction (rdam24 rotation, CRAK
               accumulation, unilateral DE_i coefficients)
    crit24.F   Ottosen criterion + secant search of the crossing point
    fr.F       the criterion function FRV (three-invariant Ottosen
               surface with the VK hardening/cap factor)
    dama24.F   tensile damage: new crack directions (pri324 / pri224),
               rupture strain EPS_F, damage growth, new Hooke matrix
    plas24.F   compressive plasticity (ICAP = 0 original cap / ICAP = 1),
               scalar cutting-plane sub-increments with VK/ROB hardening
    rdam24.F / udam24.F   strain / stress rotations to/from the frozen
               crack frame (ANG stores the first two crack axes)
    pri324.F / pri224.F   principal directions for the 1st / 2nd crack

``starter/source/materials/mat/mat024/hm_read_mat24.F`` gives the PM
table (constants below carry their PM index) and ``m24in2.F`` the state
initialization (ANG = identity, EPS_F = -1, VK0 = VKY, ROB = RO0).

Model summary
-------------
Stress lives in the (frozen) CRACK frame once a direction is damaged —
``sigc24`` is the Fortran LBUF%SIGC.  Each cycle:

1. **elas24**: rotate the strain increment into the crack frame, update
   the accumulated crack-normal strain CRAK, evolve the directional
   damage d_i = QQ (1 - EPS_F_i/CRAK_i) <= DSUP (unilateral: a closed
   crack, CRAK_i < 0, transmits full stiffness) and build the damaged
   Hooke matrix; elastic prediction S0 (total-strain in the normals,
   incremental in the shears).
2. **crit24/frv**: evaluate the three-invariant Ottosen criterion
   F(S) = (sqrt(2 J2) - VK * RF(SM, cos3theta))/FC with the hardening
   factor VK (1 above RT, parabolic RC..RT blend, VK0 on the cone,
   parabolic cap ROB..ROK) and secant-search the fraction SCLE2 of the
   increment beyond the surface.
3. Branch per element: ELASTIC (accept S0) / DAMAGE (mean stress at the
   trial >= RT: tensile crack via dama24) / PLASTIC (compaction
   trace(STRAIN) > VMAX: cutting-plane compressive return via plas24
   with volumetric compactancy/dilatancy ALPHA(VK) and the VK0/ROB
   hardening update).
4. Total failure when max(CRAK) >= EPSMAX: the element unloads by the
   upstream OFF cascade (off *= 0.8 per cycle, dead below 0.1).

Documented deviations of the port
---------------------------------
* REINFORCEMENT (steel ratios ARM1-3 + YMS/Y0S/ETS, carm24.F) is NOT
  ported — a deck giving nonzero ARM percentages is refused by the
  builder.  None of the 8 corpus decks uses reinforcement.
* ICAP = 2 (plas24b.F "new cap formulation", 2017+ optional flag) is NOT
  ported — refused by the builder.  The corpus decks run ICAP = 0.
  The INVERS < 2017 auto-promotion of ICAP 0 -> 1 is not applied (the
  builder keeps the deck's flag; pass Iflag = 1 explicitly for the old
  behavior).
* the plastic corrector runs per yielding element (a faithful scalar
  transliteration of plas24.F's inner loop) — the elastic, damage and
  criterion paths are vectorized; concrete models spend most elements
  in those paths each cycle.
* element deletion is expressed through the law's own ``off24`` state
  (stress wiped); the port's solid kernels keep the element in the mesh
  (mass/time step) — upstream fully deletes it.  The corpus decks never
  reach EPSMAX (default 1e20).
* 2D (N2D = 1 quad) and SPH branches are not ported.

Extra state (materials.extra_shapes -> solid kernels):
    strain24 (6,)  accumulated total strain (element frame)  LBUF%STRA
    sigc24   (6,)  stress in the crack frame                 LBUF%SIGC
    crak24   (3,)  crack-normal strains                      LBUF%CRAK
    dam24    (3,)  directional damage                        LBUF%DAM
    ang24    (6,)  crack frame (first two axes)              LBUF%ANG
    epsf24   (3,)  rupture strain per direction (-1 = none)  LBUF%SF
    vk024    ()    max hardening parameter reached           LBUF%VK
    vk24     ()    current criterion ratio                   LBUF%RK
    rob24    ()    current cap end position                  LBUF%ROB
    off24    ()    element OFF (1 alive, decaying 0.8/cycle)
    ini24    ()    lazy-init flag (m24in2.F values on first call)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1e-20


class Law24Params:
    """Parameters container for /MAT/LAW24 (/MAT/CONC, Peric concrete damage model).

    Supports:
    1. Peric scalar isotropic damage model (effective stress sigma = (1 - D) * C * eps)
    2. Smeared crack / Ottosen criterion with compressive cap plasticity
    """

    def __init__(
        self,
        E: float = 30000.0,
        nu: float = 0.2,
        rho0: float = 2400.0,
        fc: float = 30.0,
        ft: float = 3.0,
        fb: Optional[float] = None,
        f2d: Optional[float] = None,
        s0: float = 1.25,
        ht: Optional[float] = None,
        damage_max: float = 0.99,
        eps_0: Optional[float] = None,
        eps_f: float = 0.005,
        eps_max: float = 1e20,
        icap: int = 0,
        damage_model: str = "scalar",
        **kwargs: Any,
    ) -> None:
        self.E = float(E)
        self.nu = float(nu)
        self.rho0 = float(rho0)
        self.fc = float(fc)
        self.ft = float(ft)
        self.fb = float(fb) if fb is not None else 1.16 * self.fc
        self.f2d = float(f2d) if f2d is not None else 4.0 * self.fc
        self.s0 = float(s0)
        self.ht = float(ht) if ht is not None else -self.E
        self.damage_max = float(damage_max)
        self.eps_0 = float(eps_0) if eps_0 is not None else (self.ft / max(self.E, 1e-6))
        self.eps_f = float(eps_f)
        self.eps_max = float(eps_max)
        self.icap = int(icap)
        self.damage_model = str(damage_model)
        self.law = 24
        self.law_name = "LAW24"

        self.G = self.E / (2.0 * (1.0 + self.nu))
        self.K = self.E / (3.0 * (1.0 - 2.0 * self.nu))
        den = (1.0 + self.nu) * (1.0 - 2.0 * self.nu)
        self.A11 = self.E * (1.0 - self.nu) / max(den, 1e-12)
        self.A12 = self.E * self.nu / max(den, 1e-12)

        self.params: Dict[str, Any] = {
            "MAT_E": self.E,
            "MAT_NU": self.nu,
            "RHO0": self.rho0,
            "MAT_SIGY": self.fc,
            "MAT_FtFc": self.ft / self.fc if self.fc > 0 else 0.1,
            "MAT_FbFc": self.fb / self.fc if self.fc > 0 else 1.16,
            "MAT_F2Fc": self.f2d / self.fc if self.fc > 0 else 4.0,
            "MAT_SoFc": self.s0,
            "MAT_ETAN": self.ht,
            "MAT_DAMAGE": self.damage_max,
            "MAT_EPS": self.eps_max,
            "Iflag": self.icap,
            "E": self.E,
            "nu": self.nu,
            "FC": self.fc,
            "FT": self.ft,
            "DSUP": self.damage_max,
            "EPS_0": self.eps_0,
            "EPST": self.eps_0,
            "EPS_F": self.eps_f,
            "EPSMAX": self.eps_max,
            "A11c": self.A11,
            "A12c": self.A12,
            "Gc": self.G,
            "BULK": self.K,
            "DAMAGE_MODEL": self.damage_model,
        }
        for k, v in kwargs.items():
            self.params[k] = v


# ----------------------------------------------------------------------------
# rotations to/from the crack frame (rdam24.F / udam24.F)
# ----------------------------------------------------------------------------

def _frame(ang):
    """(m, 3, 3) rotation whose COLUMNS are the crack axes: cols 1-2 are
    stored in ang, col 3 = col1 x col2 (the Fortran S matrix)."""
    m = len(ang)
    S = np.empty((m, 3, 3))
    S[:, 0, 0], S[:, 1, 0], S[:, 2, 0] = ang[:, 0], ang[:, 1], ang[:, 2]
    S[:, 0, 1], S[:, 1, 1], S[:, 2, 1] = ang[:, 3], ang[:, 4], ang[:, 5]
    S[:, 0, 2] = ang[:, 1] * ang[:, 5] - ang[:, 2] * ang[:, 4]
    S[:, 1, 2] = ang[:, 2] * ang[:, 3] - ang[:, 0] * ang[:, 5]
    S[:, 2, 2] = ang[:, 0] * ang[:, 4] - ang[:, 1] * ang[:, 3]
    return S


def _voigt_to_mat(v, eng=False):
    """(m, 6) [xx,yy,zz,xy,yz,zx] -> (m, 3, 3); halves the shears when
    they are engineering."""
    f = 0.5 if eng else 1.0
    m = len(v)
    T = np.empty((m, 3, 3))
    T[:, 0, 0], T[:, 1, 1], T[:, 2, 2] = v[:, 0], v[:, 1], v[:, 2]
    T[:, 0, 1] = T[:, 1, 0] = f * v[:, 3]
    T[:, 1, 2] = T[:, 2, 1] = f * v[:, 4]
    T[:, 0, 2] = T[:, 2, 0] = f * v[:, 5]
    return T


def _mat_to_voigt(T, eng=False):
    f = 2.0 if eng else 1.0
    m = len(T)
    v = np.empty((m, 6))
    v[:, 0], v[:, 1], v[:, 2] = T[:, 0, 0], T[:, 1, 1], T[:, 2, 2]
    v[:, 3] = f * T[:, 0, 1]
    v[:, 4] = f * T[:, 1, 2]
    v[:, 5] = f * T[:, 0, 2]
    return v


def _rot_strain_to_crack(deps, ang):
    """rdam24.F: engineering-shear strain, element -> crack frame
    (eps' = S^T eps S)."""
    S = _frame(ang)
    T = _voigt_to_mat(deps, eng=True)
    Tp = np.einsum("mia,mij,mjb->mab", S, T, S)
    return _mat_to_voigt(Tp, eng=True)


def _rot_stress_from_crack(sig, ang):
    """udam24n: stress, crack -> element frame (sig = S sig' S^T)."""
    S = _frame(ang)
    T = _voigt_to_mat(sig, eng=False)
    Tp = np.einsum("mai,mij,mbj->mab", S, T, S)
    return _mat_to_voigt(Tp, eng=False)


# ----------------------------------------------------------------------------
# principal crack directions (pri324.F / pri224.F) — scalar helpers used
# on the rare crack-initiation events
# ----------------------------------------------------------------------------

def _pri324(sig, epstot, eps):
    """First crack: principal frame of the (deviatoric) stress.  Rotates
    sig[3:6], eps[0:3], epstot[0:3] into that frame IN PLACE and returns
    the frame's first two axes as an ang24 row (vec of 6).  Faithful to
    pri324.F including the degenerate fallbacks."""
    vec = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    cs = sig.copy()
    pr = -(cs[0] + cs[1] + cs[2]) / 3.0
    cs[0] += pr
    cs[1] += pr
    cs[2] += pr
    aa = (cs[3] ** 2 + cs[4] ** 2 + cs[5] ** 2
          - cs[0] * cs[1] - cs[1] * cs[2] - cs[0] * cs[2])
    if aa < 1e-20:
        return vec
    bb = (cs[0] * cs[4] ** 2 + cs[1] * cs[5] ** 2 + cs[2] * cs[3] ** 2
          - cs[0] * cs[1] * cs[2] - 2.0 * cs[3] * cs[4] * cs[5])
    cc = np.clip(-np.sqrt(27.0 / aa) * bb * 0.5 / aa, -1.0, 1.0)
    angp = np.arccos(cc) / 3.0
    dd = 2.0 * np.sqrt(aa / 3.0)
    ftpi, ttpi = 4.188790205, 2.094395102
    strv = np.array([dd * np.cos(angp), dd * np.cos(angp + ftpi),
                     dd * np.cos(angp + ttpi)])

    strmax = max(abs(strv[0]), abs(strv[2]))
    tol1 = max(1e-20, 6e-4 * strmax ** 2)
    tol2 = 2e-4 * strmax

    def _amat(shift):
        A = np.array([[cs[0] - shift, cs[3], cs[5]],
                      [cs[3], cs[1] - shift, cs[4]],
                      [cs[5], cs[4], cs[2] - shift]])
        return A

    V = np.zeros((3, 3))
    A = _amat(strv[0])
    iperm = [1, 2, 0]
    B = np.empty((3, 3))
    xmag = np.empty(3)
    for l in range(3):
        B[:, l] = np.cross(A[:, l], A[:, iperm[l]])
        xmag[l] = np.linalg.norm(B[:, l])
    lmax = int(np.argmax(xmag))
    xmax = xmag[lmax]
    if xmax > tol1:
        V[:, 0] = B[:, lmax] / xmax
        A = _amat(strv[2])
        for l in range(3):
            B[:, l] = np.cross(A[:, l], V[:, 0])
            xmag[l] = np.linalg.norm(B[:, l])
        lmax = int(np.argmax(xmag))
        xmax = xmag[lmax]
        if xmax > tol2:
            V[:, 2] = B[:, lmax] / xmax
            V[:, 1] = np.cross(V[:, 2], V[:, 0])
            V[:, 1] /= np.linalg.norm(V[:, 1])
        else:
            vmag = np.linalg.norm(V[:, 0])
            if vmag > tol2 / max(strmax, 1e-20):
                V[:, 1] = np.array([-V[1, 0], V[0, 0], 0.0]) / vmag
            else:
                V[:, 1] = np.array([1.0, 0.0, 0.0])
    else:
        # double eigenvalue: pick any consistent orthogonal pair
        for l in range(3):
            xmag[l] = np.hypot(A[0, l], A[1, l])
        lmax = int(np.argmax(xmag))
        xmax = xmag[lmax]
        if max(abs(A[2, 0]), abs(A[2, 1]), abs(A[2, 2])) < tol2:
            V[:, 0] = np.array([0.0, 0.0, 1.0])
            V[:, 1] = np.array([-A[1, lmax], A[0, lmax], 0.0]) / xmax
        elif xmax > tol2:
            V[:, 0] = np.array([-A[1, lmax], A[0, lmax], 0.0]) / xmax
            V[:, 1] = np.array([-A[2, lmax] * V[1, 0],
                                A[2, lmax] * V[0, 0],
                                A[0, lmax] * V[1, 0] - A[1, lmax] * V[0, 0]])
            V[:, 1] /= np.linalg.norm(V[:, 1])
        else:
            V[:, 0] = np.array([1.0, 0.0, 0.0])
            V[:, 1] = np.array([0.0, 1.0, 0.0])

    vec = np.array([V[0, 0], V[1, 0], V[2, 0], V[0, 1], V[1, 1], V[2, 1]])

    # rotate sig shears / eps / epstot normals into the new frame
    S = _frame(vec[None, :])[0]

    def _rot(v6, eng, keep_shear):
        T = _voigt_to_mat(v6[None, :], eng=eng)[0]
        Tp = S.T @ T @ S
        out = _mat_to_voigt(Tp[None, :, :], eng=eng)[0]
        if keep_shear:
            v6[3:6] = out[3:6]
        else:
            v6[0:3] = out[0:3]

    _rot(sig, eng=False, keep_shear=True)      # SIG(4:6) only
    _rot(eps, eng=True, keep_shear=False)      # EPS(1:3) only
    _rot(epstot, eng=True, keep_shear=False)   # EPSTOT(1:3) only
    return vec


def _pri224(sig, epstot, eps, ang):
    """Second crack: principal direction in the plane normal to the
    first crack axis (rotation about local axis 1).  Rotates sig[3:6],
    eps[0:3], epstot[0:3] in place; returns the new SECOND axis (3,) in
    the element frame (pri224.F's DIR3D)."""
    s22, s33, s23 = sig[1], sig[2], sig[4]
    cc = 0.5 * (s22 + s33)
    bb = 0.5 * (s22 - s33)
    cr = np.hypot(bb, s23)
    ss1 = cc + cr
    d1, d2 = s23, ss1 - s22
    orm = np.hypot(d1, d2)
    if orm < 1e-8:
        d1, d2 = 1.0, 0.0
    else:
        d1, d2 = d1 / orm, d2 / orm
    # rotation about axis 1 by the principal angle
    S = np.array([[1.0, 0.0, 0.0],
                  [0.0, d1, -d2],
                  [0.0, d2, d1]])

    def _rot(v6, eng, keep_shear):
        T = _voigt_to_mat(v6[None, :], eng=eng)[0]
        Tp = S.T @ T @ S
        out = _mat_to_voigt(Tp[None, :, :], eng=eng)[0]
        if keep_shear:
            v6[3:6] = out[3:6]
        else:
            v6[0:3] = out[0:3]

    _rot(sig, eng=False, keep_shear=True)
    _rot(eps, eng=True, keep_shear=False)
    _rot(epstot, eng=True, keep_shear=False)

    # new axis 2 in the ELEMENT frame: dir1*e2_old + dir2*(e1 x e2)
    e1 = ang[0:3]
    e2 = ang[3:6]
    e3 = np.cross(e1, e2)
    return d1 * e2 + d2 * e3


# ----------------------------------------------------------------------------
# damaged Hooke matrix (shared by elas24 / dama24 / plas24)
# ----------------------------------------------------------------------------

def _cdam(young, nu, de1, de2, de3, sc1, sc2, sc3):
    """The unilateral damaged Hooke matrix of elas24.F lines 148-159 (the
    dama24/plas24 variants are the same formula with their own DE/SCAL).
    Returns (m, 3, 3) — the normal-stress block."""
    de4 = sc1 * sc2
    de5 = sc2 * sc3
    de6 = sc3 * sc1
    den = 1.0 - nu ** 2 * (de4 + de5 + de6 + 2.0 * nu * sc1 * sc2 * sc3)
    m = len(np.atleast_1d(de1))
    C = np.empty((m, 3, 3))
    C[:, 0, 0] = young * de1 * (1.0 - nu ** 2 * de5) / den
    C[:, 1, 1] = young * de2 * (1.0 - nu ** 2 * de6) / den
    C[:, 2, 2] = young * de3 * (1.0 - nu ** 2 * de4) / den
    C[:, 0, 1] = C[:, 1, 0] = nu * young * de4 * (1.0 + nu * sc3) / den
    C[:, 0, 2] = C[:, 2, 0] = nu * young * de6 * (1.0 + nu * sc2) / den
    C[:, 1, 2] = C[:, 2, 1] = nu * young * de5 * (1.0 + nu * sc1) / den
    return C


def _unilateral(dam, crak):
    """DE_i = 1 - max(0, sign(dam_i, crak_i)) and the open/closed scales
    SCAL_i (1 = closed/undamaged direction, 0 = open crack)."""
    de = 1.0 - np.maximum(0.0, np.where(crak >= 0.0, dam, -dam))
    scal = np.where(de >= 1.0, 1.0, 0.0)
    return de, scal


# ----------------------------------------------------------------------------
# the criterion function FRV (fr.F)
# ----------------------------------------------------------------------------

def _frv(p, s6, sm, vk0, rob, rok):
    """Ottosen criterion value FA and hardening factor VK for deviator
    s6 (tensor shears) + mean stress sm.  All arrays (m,)."""
    fc, rt, rc = p["FC"], p["RT"], p["RC"]
    rct1, rct2 = p["RCT1"], p["RCT2"]
    aa, ac = p["AA"], p["AC"]
    bc, bt = p["BC"], p["BT"]
    tol = (rt - rc) / 20.0

    vk = np.where(
        sm >= rt - tol, 1.0,
        np.where(sm > rc,
                 1.0 + (1.0 - vk0) * (rct1 - 2.0 * rc * sm + sm ** 2) / rct2,
                 np.where(sm > rok, vk0,
                          vk0 * (1.0 - ((sm - rok)
                                        / np.where(rob != rok, rob - rok,
                                                   _EM20)) ** 2))))

    s1, s2, s3, s4, s5, s6_ = (s6[:, 0], s6[:, 1], s6[:, 2],
                               s6[:, 3], s6[:, 4], s6[:, 5])
    r2 = (s1 ** 2 + s2 ** 2 + s3 ** 2
          + 2.0 * (s4 ** 2 + s5 ** 2 + s6_ ** 2))
    aj3 = (s1 * s2 * s3 - s1 * s5 * s5 - s2 * s6_ * s6_ - s3 * s4 * s4
           + 2.0 * s4 * s5 * s6_)
    cs3t = np.clip(0.5 * aj3 * (3.0 / np.maximum(0.5 * r2, _EM20)) ** 1.5,
                   -1.0, 1.0)
    bb = 0.5 * ((1.0 - cs3t) * bc + (1.0 + cs3t) * bt)
    df = np.sqrt(np.maximum(bb * bb - aa * sm + ac, 0.0))
    rf = (-bb + df) / aa
    fa = (np.sqrt(r2) - vk * rf) / fc

    # tensile apex overflow
    hi = sm > ac / aa
    if np.any(hi):
        fa[hi] = 0.5 * (sm[hi] - ac / aa) / min(bc, bt) / fc
    # beyond the cap end
    lo = (~hi) & (sm <= rob)
    if np.any(lo):
        bbl = max(bc, bt)
        dfl = np.sqrt(bbl * bbl - aa * rob[lo] + ac)
        rfl = (-bbl + dfl) / aa
        fa[lo] = (2.0 * rfl * (sm[lo] - rob[lo])
                  / np.where(rob[lo] != rok[lo], rob[lo] - rok[lo], _EM20)
                  / fc)
    return fa, vk


# ----------------------------------------------------------------------------
# crit24.F — crossing-fraction search
# ----------------------------------------------------------------------------

def _crit24(p, sigc, s0, scal, vk0, rob, off):
    """Returns (scle2, scle3, sm_trial, s_star, sm_star, vk) where
    scle3 < 0 flags pure elastic, scle2 in [0, 1] is the fraction of the
    increment beyond the criterion, s_star/sm_star the deviator/mean at
    the crossing point."""
    m = len(s0)
    sc = np.empty((m, 6))
    sc[:, 0] = s0[:, 0] * scal[:, 0]
    sc[:, 1] = s0[:, 1] * scal[:, 1]
    sc[:, 2] = s0[:, 2] * scal[:, 2]
    sc[:, 3] = s0[:, 3] * scal[:, 0] * scal[:, 1]
    sc[:, 4] = s0[:, 4] * scal[:, 1] * scal[:, 2]
    sc[:, 5] = s0[:, 5] * scal[:, 2] * scal[:, 0]
    sm = (sc[:, 0] + sc[:, 1] + sc[:, 2]) / 3.0

    ds = np.empty((m, 6))
    ds[:, 0] = (sc[:, 0] - sigc[:, 0]) * scal[:, 0]
    ds[:, 1] = (sc[:, 1] - sigc[:, 1]) * scal[:, 1]
    ds[:, 2] = (sc[:, 2] - sigc[:, 2]) * scal[:, 2]
    ds[:, 3] = (sc[:, 3] - sigc[:, 3]) * scal[:, 0] * scal[:, 1]
    ds[:, 4] = (sc[:, 4] - sigc[:, 4]) * scal[:, 1] * scal[:, 2]
    ds[:, 5] = (sc[:, 5] - sigc[:, 5]) * scal[:, 2] * scal[:, 0]
    dsm = (ds[:, 0] + ds[:, 1] + ds[:, 2]) / 3.0

    sc[:, 0] -= sm
    sc[:, 1] -= sm
    sc[:, 2] -= sm
    ds[:, 0] -= dsm
    ds[:, 1] -= dsm
    ds[:, 2] -= dsm

    rok = p["ROK0"] + rob - p["RO0"]
    scle2 = np.zeros(m)
    scle3 = -np.ones(m)
    vk = np.zeros(m)

    act = off >= 1.0
    if np.any(act):
        fa = np.full(m, -1.0)
        fa[act], vk[act] = _frv(p, sc[act], sm[act], vk0[act], rob[act],
                                rok[act])
        beyond = act & (fa >= 1e-10)
        scle3[act & (np.abs(fa) < 1e-10)] = 1.0
        if np.any(beyond):
            scle3[beyond] = 1.0
            tolf = 0.005
            xn = np.ones(m)
            live = beyond.copy()
            for nit in range(10):
                if not np.any(live):
                    break
                ix = np.where(live)[0]
                sn = sc[ix] - xn[ix, None] * ds[ix]
                smn = sm[ix] - xn[ix] * dsm[ix]
                fn, _ = _frv(p, sn, smn, vk0[ix], rob[ix], rok[ix])
                if nit == 0:
                    early = fn > -tolf
                    scle2[ix[early]] = 1.0
                    live[ix[early]] = False
                    ix = ix[~early]
                    fn = fn[~early]
                    if len(ix) == 0:
                        continue
                x = xn[ix] / (1.0 - fn / fa[ix])
                scle2[ix] = x
                done = np.abs(fn) < tolf
                scle2[ix[done]] = np.clip(x[done], 0.0, 1.0)
                live[ix[done]] = False
                xn[ix[~done]] = x[~done]
            scle2[beyond] = np.clip(scle2[beyond], 0.0, 1.0)

    s_star = sc - scle2[:, None] * ds
    sm_star = sm - scle2 * dsm
    return scle2, scle3, sm + 0.0, s_star, sm_star, vk, dsm


# ----------------------------------------------------------------------------
# plas24.F — scalar (per-element) compressive plastic corrector
# ----------------------------------------------------------------------------

def _plas24_one(p, sigc, dam, crak, eps6, scle2, vk0_a, vk_a, rob_a,
                eint, rho):
    """One element of plas24.F (faithful transliteration; ICAP 0/1).
    Mutates sigc (6,), crak (3,), and the 0-d array views vk0_a, vk_a,
    rob_a.  ``eps6`` is the strain increment in the crack frame."""
    young, nu, g = p["E"], p["nu"], p["Gc"]
    rho0 = p["RHO0"]
    rok0, ro0 = p["ROK0"], p["RO0"]
    bulk = p["BULK"]
    fc, rt, rc = p["FC"], p["RT"], p["RC"]
    rct1, rct2 = p["RCT1"], p["RCT2"]
    aa, bc, bt, ac = p["AA"], p["BC"], p["BT"], p["AC"]
    hbp, ali0, alf0 = p["HBP"], p["ALI0"], p["ALF0"]
    vky, hv0, expo = p["VKY"], p["HV0"], p["EXPO"]
    icap = int(p["ICAP"])
    tol = (rt - rc) / 20.0

    vk0 = float(vk0_a)
    vk = float(vk_a)
    rob = float(rob_a)
    e = eps6.copy()

    # ---- return to the criterion --------------------------------------
    crak -= scle2 * e[:3]
    de, scal = _unilateral(dam, crak)
    C = _cdam(young, nu, de[0], de[1], de[2],
              scal[0], scal[1], scal[2])[0]
    c44 = g * scal[0] * scal[1]
    c55 = g * scal[1] * scal[2]
    c66 = g * scal[2] * scal[0]
    sigc[:3] = C @ crak
    # IBUG == 0 branch of plas24.F lines 136-139
    sigc[3] = scal[0] * scal[1] * sigc[3] - c44 * e[3] * (1.0 - scle2)
    sigc[4] = scal[1] * scal[2] * sigc[4] - c55 * e[4] * (1.0 - scle2)
    sigc[5] = scal[2] * scal[0] * sigc[5] - c66 * e[5] * (1.0 - scle2)

    s = np.empty(6)
    s[0] = scal[0] * sigc[0]
    s[1] = scal[1] * sigc[1]
    s[2] = scal[2] * sigc[2]
    s[3] = sigc[3] * scal[0] * scal[1]
    s[4] = sigc[4] * scal[1] * scal[2]
    s[5] = sigc[5] * scal[2] * scal[0]
    sm = (s[0] + s[1] + s[2]) / 3.0
    s[0] -= sm
    s[1] -= sm
    s[2] -= sm

    numer = np.abs(e).sum()
    denom = (np.abs(crak).sum() + np.abs(sigc[3:6]).sum() / g)
    if denom == 0.0:
        vk0_a[...] = vk0
        vk_a[...] = vk
        rob_a[...] = rob
        return
    rate = numer / denom
    niter = min(int(3.0 * rate + 0.5) + 1, 10)     # NINT, positive arg
    e *= scle2 / niter

    for _ in range(niter):
        rok = rok0 + rob - ro0
        r2 = (s[0] ** 2 + s[1] ** 2 + s[2] ** 2
              + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
        if sm >= rt - tol:
            vk = 1.0
        elif sm > rc:
            vk = 1.0 + (1.0 - vk0) * (rct1 - 2.0 * rc * sm + sm ** 2) / rct2
        elif sm > rok:
            vk = vk0
        elif sm > rob:
            vk = vk0 * (1.0 - ((sm - rok) / max(abs(rob - rok), 1e-20)) ** 2)
        else:
            vk = 0.0

        aj3 = (s[0] * s[1] * s[2] - s[0] * s[4] ** 2 - s[1] * s[5] ** 2
               - s[2] * s[3] ** 2 + 2.0 * s[3] * s[4] * s[5])
        cs3t = np.clip(0.5 * aj3 * (3.0 / max(0.5 * r2, _EM20)) ** 1.5,
                       -1.0, 1.0)
        bb = 0.5 * ((1.0 - cs3t) * bc + (1.0 + cs3t) * bt)
        df = np.sqrt(bb * bb + max(-aa * sm + ac, 1e-9))
        rf = (-bb + df) / aa
        aj2 = 0.5 * r2
        ajj = np.sqrt(aj2)

        sm = max(rob, sm)
        if sm >= rt - tol:
            dkdsm = 0.0
        elif sm > rc:
            dkdsm = 2.0 * (1.0 - vk0) * (sm - rc) / rct2
        elif sm > rok:
            dkdsm = 0.0
        else:
            dkdsm = -2.0 * vk0 * (sm - rok) / max(abs(rob - rok), 1e-20) ** 2
        drfdsm = -0.5 / df
        b0 = -vk * drfdsm / 3.0 - rf * dkdsm / 3.0
        if ajj > 1e-3 * fc:
            drf3 = 0.5 * (-1.0 + bb / df) * (bt - bc) / aa
            b1 = (0.5 * np.sqrt(2.0) / max(ajj, _EM20)
                  + vk * drf3 * 0.25 * aj3
                  * (3.0 / max(aj2, _EM20)) ** 2.5)
            b2 = -vk * drf3 * 0.5 * (3.0 / max(aj2, _EM20)) ** 1.5
        else:
            b1 = 0.0
            b2 = 0.0

        ts1 = s[0] ** 2 + s[3] ** 2 + s[5] ** 2 - 2.0 / 3.0 * aj2
        ts2 = s[1] ** 2 + s[3] ** 2 + s[4] ** 2 - 2.0 / 3.0 * aj2
        ts3 = s[2] ** 2 + s[4] ** 2 + s[5] ** 2 - 2.0 / 3.0 * aj2
        ts4 = 2.0 * (s[4] * s[5] - s[3] * s[2])
        ts5 = 2.0 * (s[5] * s[3] - s[4] * s[0])
        ts6 = 2.0 * (s[3] * s[4] - s[5] * s[1])
        dfs = np.array([b0 + b1 * s[0] + b2 * ts1,
                        b0 + b1 * s[1] + b2 * ts2,
                        b0 + b1 * s[2] + b2 * ts3,
                        2.0 * b1 * s[3] + b2 * ts4,
                        2.0 * b1 * s[4] + b2 * ts5,
                        2.0 * b1 * s[5] + b2 * ts6])

        # volumetric plasticity: compactancy / dilatancy
        if vk > vky:
            alpha = ((1.0 - vk) * ali0 + (vk - vky) * alf0) / (1.0 - vky)
        else:
            alpha = ali0
        if icap == 1:
            if b0 < -1e-2:
                alpha = min(alpha, -0.4 / b0)
            if b0 > 1e-2:
                alpha = max(alpha, -0.4 / b0)
        if eint is not None and eint <= 0.0:
            alpha = 0.0
        if rho is not None and rho < rho0:
            alpha = 0.0
        if ajj > 1e-3 * fc:
            dgs = np.array([alpha + s[0] / (2.0 * ajj),
                            alpha + s[1] / (2.0 * ajj),
                            alpha + s[2] / (2.0 * ajj),
                            s[3] / ajj, s[4] / ajj, s[5] / ajj])
        else:
            if icap == 1:
                dgs = np.array([alpha, alpha, alpha, 0.0, 0.0, 0.0])
            else:
                dgs = np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0])

        # hardening modulus
        hpv = hv0 * np.exp(min(50.0, (rob - ro0) * expo))
        hp = hbp if sm > rok0 else hpv

        if icap == 1:
            phi = alpha * 3.0 * sm + ajj
            dfdto1 = b0 - np.sqrt(2.0 / 3.0)
            dfdto2 = 3.0 * b0
            if dfdto1 <= dfdto2:
                to = np.sqrt(1.5) * vk * rf
                dfdto = dfdto1
            else:
                to = abs(sm)
                dfdto = dfdto2
            to = np.maximum(to, 1e-20)
            # (HALF - SIGN(HALF, VK-1)): 1 while hardening (VK < 1), 0 at
            # and beyond the failure surface
            ecr = phi * hp * dfdto / to * (1.0 if vk < 1.0 else 0.0)
        else:
            dfdto1 = b0 - np.sqrt(2.0 / 3.0)
            dfdto2 = 3.0 * b0
            if dfdto1 <= dfdto2:
                to = np.sqrt(1.5) * vk * rf
                to = np.maximum(to, 1e-20)
                phi = (alpha * 3.0 * sm + ajj) / to
                ecr = phi * hp * dfdto1 * (1.0 if vk < 1.0 else 0.0)
            else:
                dfdro = -2.0 * vk0 * rf * (sm - rok) / max(abs(rob - rok), 1e-20) ** 2
                ecr = dfdto2 * dfdro * hpv
                dgs = dfs.copy()

        if icap == 1 or (icap == 0 and vk > 1e-5):
            ha = np.empty(6)
            ha[0] = C[0, 0] * dfs[0] + C[1, 0] * dfs[1] + C[2, 0] * dfs[2]
            ha[1] = C[0, 1] * dfs[0] + C[1, 1] * dfs[1] + C[2, 1] * dfs[2]
            ha[2] = C[0, 2] * dfs[0] + C[1, 2] * dfs[1] + C[2, 2] * dfs[2]
            ha[3] = c44 * dfs[3]
            ha[4] = c55 * dfs[4]
            ha[5] = c66 * dfs[5]
            hn = np.empty(6)
            hn[0] = C[0, 0] * dgs[0] + C[0, 1] * dgs[1] + C[0, 2] * dgs[2]
            hn[1] = C[1, 0] * dgs[0] + C[1, 1] * dgs[1] + C[1, 2] * dgs[2]
            hn[2] = C[2, 0] * dgs[0] + C[2, 1] * dgs[1] + C[2, 2] * dgs[2]
            hn[3] = c44 * dgs[3]
            hn[4] = c55 * dgs[4]
            hn[5] = c66 * dgs[5]
            hh = float(dfs @ hn) - min(0.0, ecr)
            hh = np.sign(hh) * max(abs(hh), _EM20) if hh != 0.0 else _EM20

            sc1, sc2, sc3 = scal
            # the plastic rank-one correction, with the SCAL masking of
            # plas24.F lines 358-398
            fac = np.array([
                [sc1, sc1 * sc2, sc1 * sc3,
                 sc1 * sc2, sc1 * sc2 * sc3, sc1 * sc3],
                [sc1 * sc2, sc2, sc3 * sc2,
                 sc2 * sc1, sc2 * sc3, sc2 * sc1 * sc3],
                [sc3 * sc1, sc3 * sc2, sc3,
                 sc3 * sc1 * sc2, sc3 * sc2, sc3 * sc1],
                [sc1, sc2, sc3,
                 sc1 * sc2, sc1 * sc2 * sc3, sc1 * sc2 * sc3],
                [sc1, sc2, sc3,
                 sc1 * sc2 * sc3, sc2 * sc3, sc1 * sc2 * sc3],
                [sc1, sc2, sc3,
                 sc1 * sc2 * sc3, sc1 * sc2 * sc3, sc1 * sc3]])
            CP = -np.outer(hn, ha) / hh * fac
            CP[0, 0] += C[0, 0]
            CP[0, 1] += C[0, 1]
            CP[0, 2] += C[0, 2]
            CP[1, 0] += C[1, 0]
            CP[1, 1] += C[1, 1]
            CP[1, 2] += C[1, 2]
            CP[2, 0] += C[2, 0]
            CP[2, 1] += C[2, 1]
            CP[2, 2] += C[2, 2]
            CP[3, 3] += c44
            CP[4, 4] += c55
            CP[5, 5] += c66
            sigc += CP @ e
        else:
            # pure triaxial fallback
            dp = bulk * hpv / (bulk + hpv) * (e[0] + e[1] + e[2])
            sigc[0] += dp
            sigc[1] += dp
            sigc[2] += dp

        # ---- elastic strains for crack reopening ----------------------
        c11 = 1.0 / de[0] / young
        c12 = -nu * scal[0] * scal[1] / young
        c13 = -nu * scal[0] * scal[2] / young
        c22 = 1.0 / de[1] / young
        c23 = -nu * scal[1] * scal[2] / young
        c33 = 1.0 / de[2] / young
        crak[0] = c11 * sigc[0] + c12 * sigc[1] + c13 * sigc[2]
        crak[1] = c12 * sigc[0] + c22 * sigc[1] + c23 * sigc[2]
        crak[2] = c13 * sigc[0] + c23 * sigc[1] + c33 * sigc[2]
        de, scal = _unilateral(dam, crak)
        C = _cdam(young, nu, de[0], de[1], de[2],
                  scal[0], scal[1], scal[2])[0]
        c44 = g * scal[0] * scal[1]
        c55 = g * scal[1] * scal[2]
        c66 = g * scal[2] * scal[0]
        sigc[:3] = C @ crak
        sigc[3] *= scal[0] * scal[1]
        sigc[4] *= scal[1] * scal[2]
        sigc[5] *= scal[2] * scal[0]

        s[0] = sigc[0] * scal[0]
        s[1] = sigc[1] * scal[1]
        s[2] = sigc[2] * scal[2]
        s[3] = sigc[3] * scal[0] * scal[1]
        s[4] = sigc[4] * scal[1] * scal[2]
        s[5] = sigc[5] * scal[2] * scal[0]
        sm = (s[0] + s[1] + s[2]) / 3.0
        s[0] -= sm
        s[1] -= sm
        s[2] -= sm

        # ---- hardening parameters + failure-surface capping ------------
        if sm > ac / aa:
            sm = sm - 3.0 * (sm - ac / aa) / max(scal.sum(), _EM20)
        r2 = (s[0] ** 2 + s[1] ** 2 + s[2] ** 2
              + 2.0 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2))
        rr = np.sqrt(r2)
        aj3 = (s[0] * s[1] * s[2] - s[0] * s[4] ** 2 - s[1] * s[5] ** 2
               - s[2] * s[3] ** 2 + 2.0 * s[3] * s[4] * s[5])
        cs3t = np.clip(0.5 * aj3 * (3.0 / max(0.5 * r2, _EM20)) ** 1.5,
                       -1.0, 1.0)
        bb = 0.5 * ((1.0 - cs3t) * bc + (1.0 + cs3t) * bt)
        df = np.sqrt(bb * bb + max(-aa * sm + ac, 0.0))
        rf = (-bb + df) / aa
        vk = rr / max(rf, _EM20)

        if vk > 1.0:
            fac2 = 1.0 / vk
            if scal[0] > 0.9:
                sigc[0] = s[0] * fac2 + sm
            if scal[1] > 0.9:
                sigc[1] = s[1] * fac2 + sm
            if scal[2] > 0.9:
                sigc[2] = s[2] * fac2 + sm
            if scal[0] * scal[1] > 0.9:
                sigc[3] = s[3] * fac2
            if scal[1] * scal[2] > 0.9:
                sigc[4] = s[4] * fac2
            if scal[2] * scal[0] > 0.9:
                sigc[5] = s[5] * fac2
            vk = 1.0
            c11 = 1.0 / de[0] / young
            c12 = -nu * scal[0] * scal[1] / young
            c13 = -nu * scal[0] * scal[2] / young
            c22 = 1.0 / de[1] / young
            c23 = -nu * scal[1] * scal[2] / young
            c33 = 1.0 / de[2] / young
            crak[0] = c11 * sigc[0] + c12 * sigc[1] + c13 * sigc[2]
            crak[1] = c12 * sigc[0] + c22 * sigc[1] + c23 * sigc[2]
            crak[2] = c13 * sigc[0] + c23 * sigc[1] + c33 * sigc[2]
            de, scal = _unilateral(dam, crak)
            C = _cdam(young, nu, de[0], de[1], de[2],
                      scal[0], scal[1], scal[2])[0]
            c44 = g * scal[0] * scal[1]
            c55 = g * scal[1] * scal[2]
            c66 = g * scal[2] * scal[0]
            sigc[:3] = C @ crak
            sigc[3] *= scal[0] * scal[1]
            sigc[4] *= scal[1] * scal[2]
            sigc[5] *= scal[2] * scal[0]

        ro = rob
        rok = rok0 + rob - ro0
        if sm >= rt - tol:
            vk = 1.0
        elif sm > rc:
            div = min(-_EM20, rct1 - 2.0 * rc * sm + sm * sm)
            vk = 1.0 + (1.0 - vk) * rct2 / div
        elif sm > rok:
            pass                            # vk unchanged
        else:
            dvk = vk - vk0 * (1.0 - ((max(sm, rob) - rok)
                                     / (ro0 - rok0)) ** 2)
            vkk = min(vk0 + max(dvk, 0.0), 1.0)
            ro = sm + (1.0 - np.sqrt(max(0.0, 1.0 - vk / vkk))) \
                * (ro0 - rok0)
        rob = min(ro, rob)
        vk = min(vk, 1.0)
        vk0 = max(vk, vk0)

    vk0_a[...] = vk0
    vk_a[...] = vk
    rob_a[...] = rob


# ----------------------------------------------------------------------------
# dama24.F — tensile crack initiation / growth (per candidate element)
# ----------------------------------------------------------------------------

def _dama24_one(p, sigc, dam, ang, epsf, crak, s0, eps6, scle2, g):
    """One element of dama24.F.  ``s0`` is the elastic trial (6,) in the
    current crack/element frame, ``eps6`` the strain increment in that
    frame.  Mutates sigc, dam, ang, epsf, crak."""
    young, nu = p["E"], p["nu"]
    dsup, qq, epst = p["DSUP"], p["QQ"], p["EPST"]

    eps = eps6.copy()
    epstot = np.array([crak[0], crak[1], crak[2],
                       s0[3] / g, s0[4] / g, s0[5] / g])
    sigo = s0.copy()
    de, scal = _unilateral(dam, crak)
    de = de.copy()
    scal = scal.copy()

    if dam[0] > 0.0:
        idir = 2
        if dam[1] == 0.0:
            idir = 1
            dir2 = _pri224(sigo, epstot, eps, ang)
        else:
            dir2 = None
    else:
        idir = 0
        vec = _pri324(sigo, epstot, eps)
        dir2 = None

    sftry = min(epstot[idir], epstot[idir] - scle2 * eps[idir], epst)
    sftry = max(sftry, 0.25 * epst)

    if epstot[idir] < sftry:
        sigc[:] = s0            # no crack after all: keep the trial
        return

    if idir == 0:
        ang[:] = vec
    elif idir == 1:
        ang[3:6] = dir2

    for k in range(3):
        crak[k] = epstot[k]
        depsf = epstot[k] - sftry
        if depsf >= 0.0 and epsf[k] < 0.0:
            if k >= 1 and dam[k - 1] == 0.0:
                continue
            epsf[k] = sftry
            d = qq * (1.0 - epsf[k] / max(epstot[k], _EM20))
            dam[k] = min(max(d, _EM20), dsup)
            de[k] = 1.0 - dam[k]
            scal[k] = 0.0

    C = _cdam(young, nu, de[0], de[1], de[2],
              scal[0], scal[1], scal[2])[0]
    sigc[0] = C[0, 0] * epstot[0] + C[0, 1] * epstot[1] + C[0, 2] * epstot[2]
    sigc[1] = C[1, 0] * epstot[0] + C[1, 1] * epstot[1] + C[1, 2] * epstot[2]
    sigc[2] = C[2, 0] * epstot[0] + C[2, 1] * epstot[1] + C[2, 2] * epstot[2]
    sigc[3] = scal[0] * scal[1] * sigo[3]
    sigc[4] = scal[1] * scal[2] * sigo[4]
    sigc[5] = scal[2] * scal[0] * sigo[5]


def peric_damage_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Constitutive stress update for Peric scalar concrete damage model.

    Upstream Fortran origins & theory:
    - engine/source/materials/mat/mat024/sigeps24.F (conc24.F, elas24.F, dama24.F)
    - Peric scalar damage model for concrete and brittle materials.

    Formulation:
    - Linear elastic until damage initiation: r <= eps_0
    - Scalar damage variable D in [0, DSUP] (0 = intact, 1 = fully damaged)
    - Effective stress formulation: sigma = (1 - D) * C * epsilon
    """
    mat = getattr(mat, "mat", getattr(mat, "material", mat))
    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).astype(float)
    deps_arr = np.atleast_2d(deps).astype(float)
    m = sig_arr.shape[0]

    if m == 0:
        return sig, epsp, np.empty(0, dtype=sig.dtype)

    if extra is None:
        extra = {}

    p = mat.params if hasattr(mat, "params") and mat.params is not None else {}
    young = float(p.get("E", p.get("MAT_E", 30000.0)))
    nu = float(p.get("nu", p.get("MAT_NU", 0.2)))
    rho0 = float(getattr(mat, "rho0", p.get("RHO0", 2400.0)) or 2400.0)
    fc = float(p.get("FC", p.get("MAT_SIGY", 30.0)))
    ft_val = float(p.get("FT", p.get("MAT_FtFc", 3.0)))
    if 0.0 < ft_val <= 1.0:
        ft = ft_val * fc
    elif ft_val > 1.0:
        ft = ft_val
    else:
        ft = 0.1 * fc

    eps_0 = float(p.get("EPS_0", p.get("EPST", ft / max(young, 1e-6))))
    eps_f = float(p.get("EPS_F", p.get("EPSMAX", 10.0 * eps_0)))
    if eps_f <= eps_0:
        eps_f = 10.0 * eps_0

    dsup = float(p.get("DSUP", p.get("MAT_DAMAGE", 0.99)))
    alpha_d = float(p.get("ALPHA_DAM", 0.9))
    beta_d = float(p.get("BETA_DAM", 1.0 / max(eps_f - eps_0, 1e-6)))
    softening = str(p.get("SOFTENING", "exponential")).lower()

    # Elastic Lame constants
    lam = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    g = young / (2.0 * (1.0 + nu))
    c_sound_0 = math.sqrt((lam + 2.0 * g) / max(rho0, 1e-20))

    # Retrieve or initialize persistent state
    eps_tot = extra.get("strain24", extra.get("eps24_tot"))
    if eps_tot is None or eps_tot.shape != (m, 6):
        eps_tot = np.zeros((m, 6), dtype=float)
    extra["strain24"] = eps_tot
    extra["eps24_tot"] = eps_tot

    eps_tot += deps_arr

    dam_scalar = extra.get("dam24_scalar")
    if dam_scalar is None or len(dam_scalar) != m:
        dam_scalar = np.zeros(m, dtype=float)
    extra["dam24_scalar"] = dam_scalar

    r_thresh = extra.get("r_thresh")
    if r_thresh is None or len(r_thresh) != m:
        r_thresh = np.full(m, eps_0, dtype=float)
    extra["r_thresh"] = r_thresh

    # Compute equivalent tensile strain per element
    eps_tilde = np.zeros(m, dtype=float)
    for i in range(m):
        e11 = eps_tot[i, 0]
        e22 = eps_tot[i, 1]
        e33 = eps_tot[i, 2]
        e12 = 0.5 * eps_tot[i, 3]
        e23 = 0.5 * eps_tot[i, 4]
        e31 = 0.5 * eps_tot[i, 5]
        E_mat = np.array([
            [e11, e12, e31],
            [e12, e22, e23],
            [e31, e23, e33],
        ], dtype=float)
        evals = np.linalg.eigvalsh(E_mat)
        pos_evals = np.maximum(evals, 0.0)
        eps_tilde[i] = math.sqrt(float(np.sum(pos_evals ** 2)))

    # Update threshold & damage
    np.maximum(r_thresh, eps_tilde, out=r_thresh)

    damaged_mask = r_thresh > eps_0
    for i in range(m):
        if damaged_mask[i]:
            r_i = r_thresh[i]
            if softening == "linear":
                d_val = (eps_f / (eps_f - eps_0)) * (1.0 - eps_0 / r_i)
            else:
                d_val = 1.0 - (eps_0 / r_i) * (1.0 - alpha_d + alpha_d * math.exp(-beta_d * (r_i - eps_0)))
            d_val = min(dsup, max(dam_scalar[i], d_val))
            dam_scalar[i] = d_val
        else:
            dam_scalar[i] = 0.0

    extra["dam24"] = np.column_stack([dam_scalar, dam_scalar, dam_scalar])

    # Effective stress computation: sigma = (1 - D) * C * epsilon
    tr_eps = eps_tot[:, 0] + eps_tot[:, 1] + eps_tot[:, 2]
    sig_0 = np.zeros((m, 6), dtype=float)
    sig_0[:, 0] = lam * tr_eps + 2.0 * g * eps_tot[:, 0]
    sig_0[:, 1] = lam * tr_eps + 2.0 * g * eps_tot[:, 1]
    sig_0[:, 2] = lam * tr_eps + 2.0 * g * eps_tot[:, 2]
    sig_0[:, 3] = g * eps_tot[:, 3]
    sig_0[:, 4] = g * eps_tot[:, 4]
    sig_0[:, 5] = g * eps_tot[:, 5]

    factor = 1.0 - dam_scalar
    sig_arr[:] = factor[:, None] * sig_0

    if is_1d:
        sig[:] = sig_arr[0]
    else:
        sig[:] = sig_arr

    if epsp is not None and hasattr(epsp, "__setitem__"):
        try:
            if np.ndim(epsp) == 0 or (isinstance(epsp, np.ndarray) and epsp.size == 1 and m == 1):
                epsp[...] = dam_scalar[0]
            else:
                epsp[:] = dam_scalar
        except Exception:
            pass

    soundsp = np.maximum(0.1 * c_sound_0, np.sqrt(np.maximum(factor, 0.01)) * c_sound_0)
    return sig, epsp, soundsp


# ----------------------------------------------------------------------------
# conc24.F — the driver
# ----------------------------------------------------------------------------

def solid_update(mat, sig, deps, epsp=None, dt=0.0, extra=None):
    """One cycle for the group slice.  Returns (sig, epsp, c) — c is the
    constant sqrt(A11/rho0) of m24law.F."""
    mat = getattr(mat, "mat", getattr(mat, "material", mat))
    m = len(sig)
    if m == 0:
        return sig, epsp, np.empty(0, dtype=sig.dtype)
    if extra is None:
        extra = {}

    p = mat.params if hasattr(mat, "params") and mat.params is not None else {}
    if "Gc" not in p or "A11c" not in p:
        # Raw cfg-named material (MAT_E, MAT_NU, MAT_SIGY, ... of
        # hm_read_mat24.F lines 110-149): derive the PM table the kernel
        # reads, exactly as the starter does through build_conc, and cache
        # it on the material so the cycle loop pays this once.
        p = dict(p)
        p.update(build_conc(mat).params)
        if hasattr(mat, "params") and mat.params is not None:
            mat.params.update(p)
    if str(p.get("DAMAGE_MODEL", "")).upper() in ("SCALAR", "PERIC") or extra.get("scalar_damage", False):
        return peric_damage_update(mat, sig, deps, epsp=epsp, dt=dt, extra=extra)
    young, nu, g = p["E"], p["nu"], p["Gc"]
    a11, a12 = p["A11c"], p["A12c"]
    dsup, qq = p["DSUP"], p["QQ"]
    vmax, epsmax, rt = p["VMAX"], p["EPSMAX"], p["RT"]

    if "strain24" not in extra:
        extra["strain24"] = np.zeros((m, 6), dtype=sig.dtype)
    if "sigc24" not in extra:
        extra["sigc24"] = np.zeros((m, 6), dtype=sig.dtype)
    if "crak24" not in extra:
        extra["crak24"] = np.zeros((m, 3), dtype=sig.dtype)
    if "dam24" not in extra:
        extra["dam24"] = np.zeros((m, 3), dtype=sig.dtype)
    if "ang24" not in extra:
        ang_init = np.zeros((m, 6), dtype=sig.dtype)
        ang_init[:, 0] = 1.0
        ang_init[:, 4] = 1.0
        extra["ang24"] = ang_init
    if "epsf24" not in extra:
        extra["epsf24"] = np.full((m, 3), -1.0, dtype=sig.dtype)
    if "siga24" not in extra:
        extra["siga24"] = np.zeros((m, 3), dtype=sig.dtype)
    if "epsa24" not in extra:
        extra["epsa24"] = np.zeros((m, 3), dtype=sig.dtype)
    if "vk024" not in extra:
        extra["vk024"] = np.full(m, p["VKY"], dtype=sig.dtype)
    if "vk24" not in extra:
        extra["vk24"] = np.zeros(m, dtype=sig.dtype)
    if "rob24" not in extra:
        extra["rob24"] = np.full(m, p["RO0"], dtype=sig.dtype)
    if "off24" not in extra:
        extra["off24"] = np.ones(m, dtype=sig.dtype)
    if "ini24" not in extra:
        extra["ini24"] = np.ones(m, dtype=sig.dtype)

    strain = extra["strain24"]
    sigc = extra["sigc24"]
    crak = extra["crak24"]
    dam = extra["dam24"]
    ang = extra["ang24"]
    epsf = extra["epsf24"]
    vk0 = extra["vk024"]
    vk = extra["vk24"]
    rob = extra["rob24"]
    off = extra["off24"]
    ini = extra["ini24"]

    # ---- lazy init (m24in2.F) ---------------------------------------------
    fresh = ini == 0.0
    if np.any(fresh):
        ang[fresh] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        epsf[fresh] = -1.0
        vk0[fresh] = p["VKY"]
        rob[fresh] = p["RO0"]
        off[fresh] = 1.0
        ini[fresh] = 1.0

    strain += deps                       # LBUF%STRA (ISTRAIN accumulation)

    # ---- elas24: damage evolution + elastic prediction ---------------------
    d6 = deps.copy()                     # rotated per element below
    damaged = dam.sum(axis=1) > 0.0
    s0 = np.empty((m, 6))
    scal = np.ones((m, 3))
    if np.any(damaged):
        di = np.where(damaged)[0]
        d6[di] = _rot_strain_to_crack(deps[di], ang[di])
        crak[di] += d6[di, :3]
        # damage growth from EPS_F/CRAK ratio
        for k in range(3):
            hh = np.maximum(crak[di, k], _EM20)
            dek = np.where(epsf[di, k] > 0.0,
                           qq * (1.0 - epsf[di, k] / hh), 0.0)
            dek = np.minimum(dek, dsup)
            dam[di, k] = np.maximum(dek, dam[di, k])
        de3 = np.empty((len(di), 3))
        for k in range(3):
            de3[:, k] = 1.0 - np.maximum(
                0.0, np.where(crak[di, k] >= 0.0, dam[di, k], -dam[di, k]))
        sc = np.where(de3 >= 1.0, 1.0, 0.0)
        scal[di] = sc
        C = _cdam(young, nu, de3[:, 0], de3[:, 1], de3[:, 2],
                  sc[:, 0], sc[:, 1], sc[:, 2])
        de4 = sc[:, 0] * sc[:, 1]
        de5 = sc[:, 1] * sc[:, 2]
        de6 = sc[:, 2] * sc[:, 0]
        s0[di, :3] = np.einsum("mij,mj->mi", C, crak[di])
        s0[di, 3] = de4 * sigc[di, 3] + g * de4 * d6[di, 3]
        s0[di, 4] = de5 * sigc[di, 4] + g * de5 * d6[di, 4]
        s0[di, 5] = de6 * sigc[di, 5] + g * de6 * d6[di, 5]
    undam = ~damaged
    if np.any(undam):
        ui = np.where(undam)[0]
        s0[ui, 0] = sigc[ui, 0] + a11 * d6[ui, 0] \
            + a12 * (d6[ui, 1] + d6[ui, 2])
        s0[ui, 1] = sigc[ui, 1] + a11 * d6[ui, 1] \
            + a12 * (d6[ui, 0] + d6[ui, 2])
        s0[ui, 2] = sigc[ui, 2] + a11 * d6[ui, 2] \
            + a12 * (d6[ui, 0] + d6[ui, 1])
        s0[ui, 3] = sigc[ui, 3] + g * d6[ui, 3]
        s0[ui, 4] = sigc[ui, 4] + g * d6[ui, 4]
        s0[ui, 5] = sigc[ui, 5] + g * d6[ui, 5]
        crak[ui, 0] = (s0[ui, 0] - nu * (s0[ui, 1] + s0[ui, 2])) / young
        crak[ui, 1] = (s0[ui, 1] - nu * (s0[ui, 0] + s0[ui, 2])) / young
        crak[ui, 2] = (s0[ui, 2] - nu * (s0[ui, 0] + s0[ui, 1])) / young

    # ---- crit24: criterion + crossing fraction -----------------------------
    scle2, scle3, sm_trial, _s_star, sm_star, vknew, dsm = _crit24(
        p, sigc, s0, scal, vk0, rob, off)
    vk[:] = np.where(off >= 1.0, vknew, vk)

    # ---- branch per element -------------------------------------------------
    alive = off != 0.0
    elastic = alive & (scle3 < 0.0)
    trial_mean = sm_star + scle2 * dsm
    dama_b = alive & (scle3 >= 0.0) & (trial_mean >= rt)
    plast_b = alive & (scle3 >= 0.0) & ~dama_b \
        & (strain[:, 0] + strain[:, 1] + strain[:, 2] > vmax)
    accept = alive & ~dama_b & ~plast_b       # elastic or compacted-out

    acc = elastic | accept | dama_b           # all these take the trial
    sigc[acc] = s0[acc]

    # damage: only elements with an undamaged direction start a new crack
    cand = dama_b & (dam == 0.0).any(axis=1)
    eint = extra.get("eint") if extra else None
    rho = extra.get("rho") if extra else None
    for i in np.where(cand)[0]:
        _dama24_one(p, sigc[i], dam[i], ang[i], epsf[i], crak[i],
                    s0[i].copy(), d6[i], scle2[i], g)
    for i in np.where(plast_b)[0]:
        _plas24_one(p, sigc[i], dam[i], crak[i], d6[i], scle2[i],
                    vk0[i:i + 1].reshape(()), vk[i:i + 1].reshape(()),
                    rob[i:i + 1].reshape(()),
                    None if eint is None else float(eint[i]),
                    None if rho is None else float(rho[i]))

    # ---- OFF cascade + EPSMAX total failure --------------------------------
    off[:] = np.where(off < 0.1, 0.0, off)
    dying = (off < 1.0) & (off > 0.0)
    off[dying] *= 0.8
    etest = crak.max(axis=1)
    starting = (off >= 1.0) & (etest >= epsmax)
    off[starting] *= 0.8

    # ---- back to the element frame + OFF ------------------------------------
    out = sigc.copy()
    dsum = dam.sum(axis=1)
    di = np.where(dsum > 0.0)[0]
    if len(di):
        out[di] = _rot_stress_from_crack(out[di], ang[di])
        
    # ---- Steel reinforcement (ARM1, ARM2, ARM3) -----------------------------
    arm1, arm2, arm3 = p["ARM1"], p["ARM2"], p["ARM3"]
    if arm1 > 0.0 or arm2 > 0.0 or arm3 > 0.0:
        siga24 = extra["siga24"]
        epsa24 = extra["epsa24"]
        
        # update steel stresses and plastic strains
        _carm24(p["YMS"], p["Y0S"], p["ETS"], epsa24, siga24, deps[:, :3])
        
        # Rule of Mixtures in the orthotropic frame
        if arm1 > 0.0:
            out[:, 0] = out[:, 0] * (1.0 - arm1) + arm1 * siga24[:, 0]
        if arm2 > 0.0:
            out[:, 1] = out[:, 1] * (1.0 - arm2) + arm2 * siga24[:, 1]
        if arm3 > 0.0:
            out[:, 2] = out[:, 2] * (1.0 - arm3) + arm3 * siga24[:, 2]

    sig[:] = out * off[:, None]

    c = np.full(m, np.sqrt(p["A11c"] / p["RHO0"]))     # m24law SSP
    return sig, epsp, c


def _carm24(yms, y0s, ets, epsa, siga, deps_norm):
    """Update independent 1D elasto-plastic steel bars (carm24.F)."""
    hs = yms * ets / max(yms - ets, 1e-20)
    s_trial = siga + yms * deps_norm
    s_yield = y0s + hs * np.abs(epsa)
    
    yielded = np.abs(s_trial) > s_yield
    scal = np.maximum(np.abs(s_trial) - s_yield, 0.0) / np.maximum(np.abs(yms * deps_norm), 1e-20)
    d_eps_plas = yielded * scal * (1.0 - ets / (yms + 1e-10)) * deps_norm
    
    epsa += np.abs(d_eps_plas)
    s_yield_new = s_yield * np.sign(s_trial) + hs * d_eps_plas
    siga[:] = np.where(yielded, s_yield_new, s_trial)


def shell_update(mat, sig, deps, epsp=None, dt=0.0, extra=None, *args, **kwargs):
    """Shell update is rejected for LAW24 (3D solid elements only): no
    sigeps24c.F exists upstream (engine/source/materials/mat/mat024/ holds
    the m24law.F solid chain only; mulawc.F90 has no LAW24 branch)."""
    raise NotImplementedError(
        "LAW24 (concrete) is implemented for 3D solid elements only."
    )


# ----------------------------------------------------------------------------
# Consistent tangents for implicit analysis
# ----------------------------------------------------------------------------

def consistent_solid_tangent(mat, sig: Optional[np.ndarray] = None,
                             epsp: Optional[np.ndarray] = None,
                             epsp_incr: Optional[np.ndarray] = None,
                             extra=None, **kwargs: Any) -> np.ndarray:
    """The CONSISTENT (algorithmic) elastoplastic and damaged tangent
    for concrete solids, (n, 6, 6), Voigt / engineering shear.

    Returns the damaged / rebar-reinforced Hooke matrix rotated to the
    element frame, with plastic softening reduction if yielding.
    """
    mat = getattr(mat, "mat", getattr(mat, "material", mat))
    if sig is not None:
        sig_arr = np.atleast_2d(sig)
        n = sig_arr.shape[0]
        dtype = sig_arr.dtype
    elif extra is not None and "strain24" in extra:
        n = extra["strain24"].shape[0]
        dtype = extra["strain24"].dtype
    else:
        n = 1
        dtype = float

    if n == 0:
        return np.empty((0, 6, 6), dtype=dtype)

    p = mat.params if hasattr(mat, "params") and mat.params is not None else {}
    young = float(p.get("E", p.get("MAT_E", 30000.0)))
    nu = float(p.get("nu", p.get("MAT_NU", 0.2)))
    g = float(p.get("Gc", young / (2.0 * (1.0 + nu))))
    den = (1.0 + nu) * (1.0 - 2.0 * nu)
    a11 = float(p.get("A11c", young * (1.0 - nu) / max(den, 1e-12)))
    a12 = float(p.get("A12c", young * nu / max(den, 1e-12)))
    arm1, arm2, arm3 = p.get("ARM1", 0.0), p.get("ARM2", 0.0), p.get("ARM3", 0.0)
    yms = p.get("YMS", 0.0)

    # Base elastic matrix for uncracked concrete
    C_base = np.zeros((6, 6), dtype=dtype)
    C_base[0, 0] = C_base[1, 1] = C_base[2, 2] = a11
    C_base[0, 1] = C_base[1, 0] = C_base[0, 2] = C_base[2, 0] = C_base[1, 2] = C_base[2, 1] = a12
    C_base[3, 3] = C_base[4, 4] = C_base[5, 5] = g

    C = np.broadcast_to(C_base, (n, 6, 6)).copy()

    # If scalar damage is active
    if str(p.get("DAMAGE_MODEL", "")).upper() in ("SCALAR", "PERIC") or (extra and extra.get("scalar_damage", False)):
        d_val = extra.get("dam24_scalar", 0.0) if extra else 0.0
        d_arr = np.atleast_1d(np.asarray(d_val, dtype=float))
        if d_arr.shape[0] != n:
            d_arr = np.full(n, float(d_arr[0]))
        C = (1.0 - np.clip(d_arr[:, None, None], 0.0, 1.0)) * C_base
        return C

    # If extra is present, check for directional damage/cracking
    if extra is not None and "dam24" in extra and "ang24" in extra and "crak24" in extra:
        dam = extra["dam24"]
        ang = extra["ang24"]
        crak = extra["crak24"]
        dsum = dam.sum(axis=1)
        cracked = np.where(dsum > 0.0)[0]
        if len(cracked) > 0:
            for idx in cracked:
                de_i, sc_i = _unilateral(dam[idx:idx+1], crak[idx:idx+1])
                C_dam3 = _cdam(young, nu, de_i[0, 0], de_i[0, 1], de_i[0, 2],
                               sc_i[0, 0], sc_i[0, 1], sc_i[0, 2])[0]
                de4 = sc_i[0, 0] * sc_i[0, 1]
                de5 = sc_i[0, 1] * sc_i[0, 2]
                de6 = sc_i[0, 2] * sc_i[0, 0]

                C_local = np.zeros((6, 6), dtype=dtype)
                C_local[:3, :3] = C_dam3
                C_local[3, 3] = de4 * g
                C_local[4, 4] = de5 * g
                C_local[5, 5] = de6 * g

                # Transform column by column via basis strain vectors
                # using _rot_strain_to_crack and _rot_stress_from_crack
                ang_row = ang[idx:idx+1]
                for k in range(6):
                    e_k = np.zeros((1, 6), dtype=dtype)
                    e_k[0, k] = 1.0
                    e_crack = _rot_strain_to_crack(e_k, ang_row)
                    s_crack = e_crack @ C_local.T
                    s_elem = _rot_stress_from_crack(s_crack, ang_row)
                    C[idx, :, k] = s_elem[0]

    # Add steel reinforcement contribution (Rule of Mixtures along orthotropic axes)
    if arm1 > 0.0:
        C[:, 0, :] *= (1.0 - arm1)
        C[:, 0, 0] += arm1 * yms
    if arm2 > 0.0:
        C[:, 1, :] *= (1.0 - arm2)
        C[:, 1, 1] += arm2 * yms
    if arm3 > 0.0:
        C[:, 2, :] *= (1.0 - arm3)
        C[:, 2, 2] += arm3 * yms

    # Plastic softening reduction if yielding
    if epsp_incr is not None:
        plastic = epsp_incr > 0.0
        if np.any(plastic):
            p_idx = np.where(plastic)[0]
            fc = p.get("FC", 30.0)
            bulk = p.get("BULK", young / (3.0 * (1.0 - 2.0 * nu)))
            for idx in p_idx:
                dep = epsp_incr[idx]
                fac = 1.0 / (1.0 + 3.0 * g * dep / max(fc, 1e-6))
                C[idx, 3:, 3:] *= fac
                C[idx, :3, :3] = (C[idx, :3, :3] - bulk) * fac + bulk

    return C


def sound_speed(
    mat: Any,
    rho: Optional[Union[float, np.ndarray]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Union[float, np.ndarray]:
    """Compute acoustic sound speed for LAW24 concrete.

    Upstream Fortran reference: m24law.F (c = sqrt(PM(24) / PM(1))).
    """
    mat = getattr(mat, "mat", getattr(mat, "material", mat))
    p = mat.params if hasattr(mat, "params") and mat.params is not None else {}
    young = float(p.get("E", p.get("MAT_E", 30000.0)))
    nu = float(p.get("nu", p.get("MAT_NU", 0.2)))
    den = (1.0 + nu) * (1.0 - 2.0 * nu)
    a11 = float(p.get("A11c", young * (1.0 - nu) / max(den, 1e-12)))
    rho0 = rho if rho is not None else float(getattr(mat, "rho0", p.get("RHO0", 2400.0)) or 2400.0)
    is_scalar = np.isscalar(rho0)
    rho_arr = np.atleast_1d(np.asarray(rho0, dtype=float))
    c = np.sqrt(a11 / np.maximum(rho_arr, _EM20))
    if is_scalar:
        return float(c[0])
    return c


# ----------------------------------------------------------------------------
# cfg-record constructor (mat_reader physics registry)
# ----------------------------------------------------------------------------


def build_conc(rec) -> Material:
    """hm_read_mat24.F: cfg attributes -> the PM table (each param below
    notes its PM index)."""
    if isinstance(rec, Law24Params):
        return Material(
            id=int(getattr(rec, "id", 1)),
            law=24,
            rho0=float(rec.rho0),
            title=str(getattr(rec, "title", "LAW24")),
            params=dict(rec.params),
        )
    if hasattr(rec, "params") and getattr(rec, "params") is not None:
        q = getattr(rec, "params")
    elif isinstance(rec, dict) and "params" in rec and isinstance(rec["params"], dict):
        q = rec["params"]
    elif isinstance(rec, dict):
        q = rec
    else:
        q = {}

    density = float(getattr(rec, "density", getattr(rec, "rho0", q.get("RHO0", q.get("rho0", q.get("density", 2400.0))))) or 2400.0)
    rec_id = int(getattr(rec, "id", q.get("id", 1)))
    rec_title = str(getattr(rec, "title", q.get("title", f"LAW24_{rec_id}")))
    ymc = float(q.get("MAT_E") if q.get("MAT_E") is not None else (q.get("e") if q.get("e") is not None else (q.get("E") or 0.0)))
    anuc = float(q.get("MAT_NU") if q.get("MAT_NU") is not None else (q.get("nu") if q.get("nu") is not None else (q.get("NU") or 0.0)))
    icap = int(q.get("Iflag") if q.get("Iflag") is not None else (q.get("icap") if q.get("icap") is not None else (q.get("iflag") or 0)))
    fc = float(q.get("MAT_SIGY") if q.get("MAT_SIGY") is not None else (q.get("fc") if q.get("fc") is not None else (q.get("sig_y") if q.get("sig_y") is not None else (q.get("sigy") if q.get("sigy") is not None else (q.get("SIGY") or 0.0)))))
    ft = float(q.get("MAT_FtFc") if q.get("MAT_FtFc") is not None else (q.get("ft") if q.get("ft") is not None else (q.get("ft_fc") or 0.0)))
    fb = float(q.get("MAT_FbFc") if q.get("MAT_FbFc") is not None else (q.get("fb") if q.get("fb") is not None else (q.get("fb_fc") or 0.0)))
    f2d = float(q.get("MAT_F2Fc") if q.get("MAT_F2Fc") is not None else (q.get("f2d") if q.get("f2d") is not None else (q.get("f2_fc") or 0.0)))
    s0 = float(q.get("MAT_SoFc") if q.get("MAT_SoFc") is not None else (q.get("s0") if q.get("s0") is not None else (q.get("so_fc") or 0.0)))
    ht = float(q.get("MAT_ETAN") if q.get("MAT_ETAN") is not None else (q.get("ht") if q.get("ht") is not None else (q.get("etan") or 0.0)))
    dsup1 = float(q.get("MAT_DAMAGE") if q.get("MAT_DAMAGE") is not None else (q.get("dsup") if q.get("dsup") is not None else (q.get("damage") or 0.0)))
    epsmax = float(q.get("MAT_EPS") if q.get("MAT_EPS") is not None else (q.get("epsmax") if q.get("epsmax") is not None else (q.get("eps") or 0.0)))
    vky = float(q.get("MAT_BETA") if q.get("MAT_BETA") is not None else (q.get("vky") if q.get("vky") is not None else (q.get("beta") or 0.0)))
    rt = float(q.get("MAT_PPRES") if q.get("MAT_PPRES") is not None else (q.get("rt") if q.get("rt") is not None else (q.get("ppres") or 0.0)))
    rc = float(q.get("MAT_YPRES") if q.get("MAT_YPRES") is not None else (q.get("rc") if q.get("rc") is not None else (q.get("ypres") or 0.0)))
    hbp = float(q.get("MAT_BPMOD") if q.get("MAT_BPMOD") is not None else (q.get("hbp") if q.get("hbp") is not None else (q.get("bpmod") or 0.0)))
    etc = float(q.get("MAT_ETC") if q.get("MAT_ETC") is not None else (q.get("etc") or 0.0))
    ali = float(q.get("MAT_DIL_Y") if q.get("MAT_DIL_Y") is not None else (q.get("ali") if q.get("ali") is not None else (q.get("dil_y") or 0.0)))
    alf = float(q.get("MAT_DIL_F") if q.get("MAT_DIL_F") is not None else (q.get("alf") if q.get("alf") is not None else (q.get("dil_f") or 0.0)))
    vmax = float(q.get("MAT_COMPAC") if q.get("MAT_COMPAC") is not None else (q.get("vmax") if q.get("vmax") is not None else (q.get("compac") or 0.0)))
    rok = float(q.get("MAT_CAP_BEG") if q.get("MAT_CAP_BEG") is not None else (q.get("rok") if q.get("rok") is not None else (q.get("cap_beg") or 0.0)))
    ro0 = float(q.get("MAT_CAP_END") if q.get("MAT_CAP_END") is not None else (q.get("ro0") if q.get("ro0") is not None else (q.get("cap_end") or 0.0)))
    hv0 = float(q.get("MAT_TPMOD") if q.get("MAT_TPMOD") is not None else (q.get("hv0") if q.get("hv0") is not None else (q.get("tpmod") or 0.0)))
    
    arm1 = float(q.get("MAT_PDIR1") if q.get("MAT_PDIR1") is not None else (q.get("arm1") or 0.0))
    arm2 = float(q.get("MAT_PDIR2") if q.get("MAT_PDIR2") is not None else (q.get("arm2") or 0.0))
    arm3 = float(q.get("MAT_PDIR3") if q.get("MAT_PDIR3") is not None else (q.get("arm3") or 0.0))
    arm = [arm1, arm2, arm3]

    yms = float(q.get("MAT_E2") if q.get("MAT_E2") is not None else (q.get("yms") if q.get("yms") is not None else (q.get("e2") or 0.0)))
    y0s = float(q.get("MAT_SSIG") if q.get("MAT_SSIG") is not None else (q.get("y0s") if q.get("y0s") is not None else (q.get("ssig") or 0.0)))
    ets = float(q.get("MAT_SETAN") if q.get("MAT_SETAN") is not None else (q.get("ets") if q.get("ets") is not None else (q.get("setan") or 0.0)))

    if ymc <= 0.0 or fc <= 0.0:
        raise ValueError("LAW24 needs positive E and fc")
    if anuc < 0.0 or anuc >= 0.5:
        raise ValueError(f"LAW24: Poisson ratio nu={anuc} outside [0, 0.5)")
    if icap == 2:
        raise ValueError("LAW24 Icap=2 (plas24b 'new cap formulation') is "
                         "not ported (documented cut) — use Icap 0 or 1")

    # ---- defaults (hm_read_mat24.F lines 152-220) --------------------------
    if ft == 0.0:
        ft = 0.1
    if fb == 0.0:
        fb = 1.2
    if s0 == 0.0:
        s0 = 1.25
    if ht >= 0.0:
        ht = -ymc
    if dsup1 == 0.0:
        dsup1 = 0.99999
    if vmax >= 0.0:
        vmax = -0.35
    if epsmax <= 0.0:
        epsmax = 1e20
    if vky == 0.0:
        vky = 0.5
    if rc == 0.0:
        rc = -fc / 3.0
    if hbp == 0.0:
        if etc == 0.0:
            etc = (1.0 - vky) * ymc * fc / (2e-3 * ymc - vky * fc)
        if etc >= ymc:
            raise ValueError("LAW24: derived plastic tangent ETC >= E "
                             "(hm_read_mat24 error 2065/2066)")
        hbp = ymc * etc / (ymc - etc)
    bulk = ymc / 3.0 / (1.0 - 2.0 * anuc)
    if rok == 0.0:
        rok = rc
    if f2d == 0.0:
        f2d = 4.0
    if ro0 == 0.0:
        ro0 = -0.8 * fc
    if hv0 == 0.0:
        hv0 = ymc / 5.0
    expo = 1.0 / hv0 / vmax
    if ali >= 0.0:
        if icap == 1:
            raise ValueError("LAW24 Icap=1 requires ALPHA_i < 0 "
                             "(hm_read_mat24 warning 1161)")
        ali = -0.2
    if dsup1 >= 1.0 or dsup1 < 0.0:
        raise ValueError("LAW24: 0 <= D_sup < 1 required "
                         "(hm_read_mat24 error 605)")

    # ---- Ottosen surface constants (hm_read_mat24.F lines 240-246) ---------
    f2d0 = f2d - s0
    aa = 1.5 * (s0 / (f2d0 - 1.0) - ft * fb / (fb - ft)) / (f2d0 - fb * ft)
    cc = fb * ft * (f2d0 / (fb - ft) - s0 / (f2d0 - 1.0)) / (f2d0 - fb * ft)
    sqr32 = np.sqrt(1.5)
    bc = 0.5 * sqr32 * (cc + 1.0 / 3.0 - 2.0 / 3.0 * aa)
    bt = 0.5 * sqr32 * (cc / ft - 1.0 / 3.0 - 2.0 / 3.0 * aa * ft)
    ac = cc * aa
    aa = aa / fc

    gc = ymc / (2.0 * (1.0 + anuc))                       # PM(22)
    a12c = ymc * anuc / (1.0 + anuc) / (1.0 - 2.0 * anuc)  # PM(25)
    a11c = a12c + 2.0 * gc                                 # PM(24)

    params = {
        "E": ymc, "nu": anuc,                 # PM(20) / PM(21)
        "K": bulk, "G": gc,
        "Gc": gc, "A11c": a11c, "A12c": a12c,
        "RHO0": density,                      # PM(1) (RHOR = RHO0 here)
        "DSUP": max(0.0, dsup1),              # PM(26)
        "VMAX": vmax,                         # PM(27)
        "QQ": 1.0 - ht / ymc,                 # PM(28)
        "ROK0": rok, "RO0": ro0,              # PM(29) / PM(30)
        "BULK": bulk,                         # PM(32)
        "FC": fc, "RT": rt, "RC": rc,         # PM(33) / PM(34) / PM(35)
        "RCT1": rt * (2.0 * rc - rt),         # PM(36)
        "RCT2": (rc - rt) ** 2,               # PM(37)
        "AA": aa, "BC": bc, "BT": bt, "AC": ac,  # PM(38..41)
        "EPST": ft * fc / ymc,                # PM(42)
        "HBP": hbp,                           # PM(43)
        "ALI0": ali, "ALF0": alf,             # PM(44) / PM(45)
        "VKY": vky,                           # PM(46)
        "EPSMAX": epsmax,                     # PM(47)
        "HV0": hv0, "EXPO": expo,             # PM(48) / PM(49)
        "ICAP": icap,                         # PM(57)
        "FT": ft, "FB": fb, "F2D": f2d, "S0FC": s0, "CCOTT": cc,
        "YMS": yms, "Y0S": y0s, "ETS": ets,
        "ARM1": arm[0], "ARM2": arm[1], "ARM3": arm[2],
        "DAMAGE_MODEL": str(q.get("DAMAGE_MODEL", q.get("damage_model", "smeared_crack"))),
        "EPS_0": float(q.get("EPS_0", q.get("eps_0", ft * fc / ymc if ymc > 0 else 1e-4))),
        "EPS_F": float(q.get("EPS_F", q.get("eps_f", epsmax))),
    }
    return Material(id=rec_id, law=24, rho0=density,
                    title=rec_title, params=params)


build_law24 = build_conc
solid_step = solid_update
solid_tangent = consistent_solid_tangent
tangent_law24_solid = consistent_solid_tangent


def tangent(group_or_mat: Any = None, **kwargs: Any) -> np.ndarray:
    """Material law template tangent interface conforming to pyradioss dispatcher."""
    mat = getattr(group_or_mat, "mat", getattr(group_or_mat, "material", group_or_mat))
    return consistent_solid_tangent(mat, **kwargs)


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    for k in ("CONC", "LAW24", "CONCRETE", "PERIC_CONC", "PERIC_CONCRETE", 24, "24"):
        MAT_PHYSICS_REGISTRY.setdefault(k, build_conc)


_register()

__all__ = [
    "Law24Params",
    "build_conc",
    "build_law24",
    "solid_step",
    "solid_update",
    "shell_update",
    "sound_speed",
    "peric_damage_update",
    "consistent_solid_tangent",
    "solid_tangent",
    "tangent_law24_solid",
    "tangent",
]
