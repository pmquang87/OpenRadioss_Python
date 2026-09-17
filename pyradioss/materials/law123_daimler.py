"""
LAW123 & LAW132 — Daimler-Pinho and Daimler-Camanho Composite Failure Models.
(/MAT/LAW123, /MAT/DAIMLER_PINHO, /MAT/LAMINATED_FRACTURE_DAIMLER_PINHO,
 /MAT/LAW132, /MAT/DAIMLER_CAMANHO, /MAT/LAMINATED_FRACTURE_DAIMLER_CAMANHO).

Fortran origin:
  - starter/source/materials/mat/mat123/hm_read_mat123.F90
  - starter/source/materials/mat/mat123/law123_upd.F90
  - engine/source/materials/mat/mat123/sigeps123c.F90 (shells LAW123)
  - engine/source/materials/mat/mat123/sigeps123.F90  (solids LAW123)
  - engine/source/materials/mat/mat123/analyze_failure.F90
  - engine/source/materials/mat/mat123/failure_tools_mod.F90
  - starter/source/materials/mat/mat132/hm_read_mat132.F90
  - starter/source/materials/mat/mat132/m132init.F90
  - engine/source/materials/mat/mat132/sigeps132c.F90 (shells LAW132)
  - hm_cfg_files/config/CFG/radioss2026/MAT/matl123_daimler_pinho.cfg
  - hm_cfg_files/config/CFG/radioss2026/MAT/matl132_daimler_camanho.cfg

Constitutive features:
  1. 3D & 2D plane-stress orthotropic elasticity:
     (E1, E2, E3, nu12, nu23, nu31, G12, G23, G31).
  2. Daimler-Pinho / Daimler-Camanho physically-based failure criteria:
     - Fiber tensile failure (maximum stress / strain criterion).
     - Fiber compressive failure via fiber kinking in the kink band plane with
       misalignment angle phi_0 and friction coefficients.
     - Matrix tensile cracking (transverse tension + in-plane/out-of-plane shear)
       with fracture plane angle search (default psi_0 = 53 deg).
     - Matrix compressive cracking under transverse compression.
  3. Damage evolution:
     - LAW123: scalar degradation d = max(d_fiber, d_kink, d_mat).
     - LAW132: directional damage variables d_1+, d_1-, d_2+, d_2-, d_6 with
       crack closure effects and stiffness recovery.
  4. Regularized energy dissipation:
     - Softening calibrated by fracture energies (G_1t, G_1c, G_2t, G_2c, G_sl)
       and characteristic element length l_car.
  5. Acoustic sound speed and consistent algorithmic tangent stiffness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..model.entities import Material

_EM20 = 1.0e-20
_EM10 = 1.0e-10
_INF = 1.0e30
_RAD53 = 53.0 * math.pi / 180.0


# ============================================================================
# Dataclasses and Parameter Containers
# ============================================================================

@dataclass
class Law123Params:
    """Material parameters for /MAT/LAW123 (Daimler-Pinho)."""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    enkink: float = _INF
    ena: float = _INF
    enb: float = _INF
    ent: float = _INF
    enl: float = _INF
    xc: float = _INF
    xt: float = _INF
    yc: float = _INF
    yt: float = _INF
    sl: float = _INF
    fio: float = 53.0
    sigy: float = 0.0
    beta: float = 0.0
    lcss: int = 0
    efs: float = _INF
    ratio: float = 1.0
    fcut: float = 0.0

    # Precomputed elastic and failure coefficients
    nu12: float = field(init=False, default=0.0)
    nu21: float = field(init=False, default=0.0)
    nu13: float = field(init=False, default=0.0)
    nu31: float = field(init=False, default=0.0)
    nu23: float = field(init=False, default=0.0)
    nu32: float = field(init=False, default=0.0)
    st: float = field(init=False, default=0.0)
    mut: float = field(init=False, default=0.0)
    mul: float = field(init=False, default=0.0)
    thetac: float = field(init=False, default=0.0)
    thetai: float = field(init=False, default=0.0)
    c_plane: np.ndarray = field(init=False)
    d_solid: np.ndarray = field(init=False)
    sound_spd_shell: float = field(init=False, default=0.0)
    sound_spd_solid: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        # Defaults if secondary moduli are unset
        if self.eb <= 0.0:
            self.eb = self.ea
        if self.ec <= 0.0:
            self.ec = self.eb
        if self.gca <= 0.0:
            self.gca = self.gab
        if self.gbc <= 0.0:
            self.gbc = self.gca
        if self.prca <= 0.0:
            self.prca = self.prba
        if self.prcb <= 0.0:
            self.prcb = self.prba

        self.nu21 = self.prba
        self.nu31 = self.prca
        self.nu32 = self.prcb

        self.nu12 = self.nu21 * self.ea / max(self.eb, _EM20)
        self.nu23 = self.nu32 * self.eb / max(self.ec, _EM20)
        self.nu13 = self.nu31 * self.ea / max(self.ec, _EM20)

        # 2D Plane stress elasticity matrix C (3x3: xx, yy, xy)
        det_c = max(1.0 - self.nu12 * self.nu21, _EM20)
        fac = 1.0 / det_c
        a11 = self.ea * fac
        a12 = self.nu21 * a11
        a22 = self.eb * fac
        self.c_plane = np.array([
            [a11, a12, 0.0],
            [a12, a22, 0.0],
            [0.0, 0.0, self.gab]
        ], dtype=float)

        # 3D Compliance and Stiffness matrix D (6x6)
        s11 = 1.0 / max(self.ea, _EM20)
        s22 = 1.0 / max(self.eb, _EM20)
        s33 = 1.0 / max(self.ec, _EM20)
        s12 = -self.nu12 / max(self.ea, _EM20)
        s13 = -self.nu31 / max(self.ec, _EM20)
        s23 = -self.nu23 / max(self.eb, _EM20)

        det_s = (s11 * s22 * s33 - s11 * (s23**2) - s22 * (s13**2)
                 - s33 * (s12**2) + 2.0 * s12 * s13 * s23)
        det_s = max(det_s, _EM20)

        d11 = (s22 * s33 - s23 * s23) / det_s
        d12 = -(s12 * s33 - s13 * s23) / det_s
        d13 = (s12 * s23 - s13 * s22) / det_s
        d22 = (s11 * s33 - s13 * s13) / det_s
        d23 = -(s11 * s23 - s13 * s12) / det_s
        d33 = (s11 * s22 - s12 * s12) / det_s

        self.d_solid = np.zeros((6, 6), dtype=float)
        self.d_solid[0, 0] = d11
        self.d_solid[0, 1] = self.d_solid[1, 0] = d12
        self.d_solid[0, 2] = self.d_solid[2, 0] = d13
        self.d_solid[1, 1] = d22
        self.d_solid[1, 2] = self.d_solid[2, 1] = d23
        self.d_solid[2, 2] = d33
        self.d_solid[3, 3] = self.gab
        self.d_solid[4, 4] = self.gbc
        self.d_solid[5, 5] = self.gca

        # Fracture failure parameters (Fortran hm_read_mat123.F90 lines 324-337)
        ang0 = (self.fio if self.fio > 0.0 else 53.0) * math.pi / 180.0
        scale = 1.0 / math.tan(ang0)
        self.st = 0.5 * scale * self.yc
        self.mut = -1.0 / math.tan(2.0 * ang0)
        self.mul = self.sl * self.mut / max(self.st, _EM20)

        aa = 2.0 * (self.sl / max(self.xc, _EM20) + self.mul)
        bb = 1.0 - 2.0 * aa * self.sl / max(self.xc, _EM20)
        if bb < 0.0 or abs(aa) < _EM20:
            self.thetac = self.sl / max(self.gab, _EM20)
        else:
            self.thetac = math.atan((1.0 - math.sqrt(max(_EM20, bb))) / aa)

        # Initial misalignment angle solution thetai (Fortran law123_upd.F90 lines 88-98)
        theta = 0.0
        if self.xc > 0.0:
            r = self.thetac
            for _ in range(50):
                if abs(r) <= 1.0e-5:
                    break
                dydx = 1.0 + (self.xc * math.cos(2.0 * theta) / max(self.gab, _EM20))
                theta += r / dydx
                xvec = 0.5 * math.sin(2.0 * theta) * self.xc
                yy = xvec / max(self.gab, _EM20)
                r = self.thetac - theta - yy
        self.thetai = theta

        # Sound speeds
        rho_eff = max(self.rho0, _EM20)
        self.sound_spd_shell = math.sqrt(max(self.ea, self.eb) / (rho_eff * det_c))
        self.sound_spd_solid = math.sqrt(max(d11, d22, d33) / rho_eff)


@dataclass
class Law132Params:
    """Material parameters for /MAT/LAW132 (Daimler-Camanho)."""
    rho0: float = 0.0
    ea: float = 0.0
    eb: float = 0.0
    ec: float = 0.0
    gab: float = 0.0
    gca: float = 0.0
    gbc: float = 0.0
    prba: float = 0.0
    prca: float = 0.0
    prcb: float = 0.0
    gxc: float = _INF
    gxt: float = _INF
    gyc: float = _INF
    gyt: float = _INF
    gsl: float = _INF
    xc: float = _INF
    xt: float = _INF
    yc: float = _INF
    yt: float = _INF
    sl: float = _INF
    gxc0: float = 0.0
    gxt0: float = 0.0
    xc0: float = 0.0
    xt0: float = 0.0
    fio: float = 53.0
    sigy: float = 0.0
    etan: float = 0.0
    beta: float = 0.0
    lcss: int = 0
    epsf23: float = _INF
    epsr23: float = _INF
    tsmd23: float = 0.99
    epsf31: float = _INF
    epsr31: float = _INF
    tsmd31: float = 0.99
    ef11t: float = _INF
    ef11c: float = _INF
    ef22t: float = _INF
    ef22c: float = _INF
    ef12: float = _INF
    ef23: float = _INF
    ef31: float = _INF
    cf12: float = 0.0
    cf23: float = 0.0
    cf31: float = 0.0
    ratio: float = 1.0
    fcut: float = 0.0

    # Derived
    nu12: float = field(init=False, default=0.0)
    nu21: float = field(init=False, default=0.0)
    nu13: float = field(init=False, default=0.0)
    nu31: float = field(init=False, default=0.0)
    nu23: float = field(init=False, default=0.0)
    nu32: float = field(init=False, default=0.0)
    eta_l: float = field(init=False, default=0.0)
    eta_t: float = field(init=False, default=0.0)
    st: float = field(init=False, default=0.0)
    theta_c: float = field(init=False, default=0.0)
    g_ratio: float = field(init=False, default=0.0)
    c_plane: np.ndarray = field(init=False)
    d_solid: np.ndarray = field(init=False)
    sound_spd_shell: float = field(init=False, default=0.0)
    sound_spd_solid: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.eb <= 0.0:
            self.eb = self.ea
        if self.ec <= 0.0:
            self.ec = self.eb
        if self.gca <= 0.0:
            self.gca = self.gab
        if self.gbc <= 0.0:
            self.gbc = self.gca
        if self.prca <= 0.0:
            self.prca = self.prba
        if self.prcb <= 0.0:
            self.prcb = self.prba

        self.nu21 = self.prba
        self.nu31 = self.prca
        self.nu32 = self.prcb

        self.nu12 = self.nu21 * self.ea / max(self.eb, _EM20)
        self.nu23 = self.nu32 * self.eb / max(self.ec, _EM20)
        self.nu13 = self.nu31 * self.ea / max(self.ec, _EM20)

        det_c = max(1.0 - self.nu12 * self.nu21, _EM20)
        fac = 1.0 / det_c
        a11 = self.ea * fac
        a12 = self.nu21 * a11
        a22 = self.eb * fac
        self.c_plane = np.array([
            [a11, a12, 0.0],
            [a12, a22, 0.0],
            [0.0, 0.0, self.gab]
        ], dtype=float)

        s11 = 1.0 / max(self.ea, _EM20)
        s22 = 1.0 / max(self.eb, _EM20)
        s33 = 1.0 / max(self.ec, _EM20)
        s12 = -self.nu12 / max(self.ea, _EM20)
        s13 = -self.nu31 / max(self.ec, _EM20)
        s23 = -self.nu23 / max(self.eb, _EM20)
        det_s = max(s11 * s22 * s33 - s11 * (s23**2) - s22 * (s13**2)
                    - s33 * (s12**2) + 2.0 * s12 * s13 * s23, _EM20)

        d11 = (s22 * s33 - s23 * s23) / det_s
        d12 = -(s12 * s33 - s13 * s23) / det_s
        d13 = (s12 * s23 - s13 * s22) / det_s
        d22 = (s11 * s33 - s13 * s13) / det_s
        d23 = -(s11 * s23 - s13 * s12) / det_s
        d33 = (s11 * s22 - s12 * s12) / det_s

        self.d_solid = np.zeros((6, 6), dtype=float)
        self.d_solid[0, 0] = d11
        self.d_solid[0, 1] = self.d_solid[1, 0] = d12
        self.d_solid[0, 2] = self.d_solid[2, 0] = d13
        self.d_solid[1, 1] = d22
        self.d_solid[1, 2] = self.d_solid[2, 1] = d23
        self.d_solid[2, 2] = d33
        self.d_solid[3, 3] = self.gab
        self.d_solid[4, 4] = self.gbc
        self.d_solid[5, 5] = self.gca

        # Fortran hm_read_mat132.F90 lines 464-474
        ang0 = (self.fio if self.fio > 0.0 else 53.0) * math.pi / 180.0
        self.eta_l = -self.sl * math.cos(2.0 * ang0) / max(self.yc * (math.cos(ang0)**2), _EM20)
        self.eta_t = -1.0 / math.tan(2.0 * ang0)
        self.st = self.yc * math.cos(ang0) * (math.sin(ang0) + math.cos(ang0) / math.tan(2.0 * ang0))
        tmp = self.sl / max(self.xc, _EM20)
        denom = 2.0 * (tmp + self.eta_l)
        disc = 1.0 - 4.0 * (tmp + self.eta_l) * tmp
        if disc < 0.0 or abs(denom) < _EM20:
            self.theta_c = self.sl / max(self.gab, _EM20)
        else:
            self.theta_c = math.atan((1.0 - math.sqrt(max(_EM20, disc))) / denom)

        self.g_ratio = (self.gyc / max(self.gxc, _EM20)) if self.gxc < _INF else 1.0

        rho_eff = max(self.rho0, _EM20)
        self.sound_spd_shell = math.sqrt(max(self.ea, self.eb) / (rho_eff * det_c))
        self.sound_spd_solid = math.sqrt(max(d11, d22, d33) / rho_eff)


# ============================================================================
# Helpers: Parameter extraction & Constructors
# ============================================================================

def _fval(d: Any, *keys: str, default: float = 0.0) -> float:
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return float(d[k])
                except (ValueError, TypeError):
                    pass
    else:
        for k in keys:
            val = getattr(d, k, None)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
    return default


def _ival(d: Any, *keys: str, default: int = 0) -> int:
    if isinstance(d, dict):
        for k in keys:
            if k in d and d[k] is not None:
                try:
                    return int(d[k])
                except (ValueError, TypeError):
                    pass
    else:
        for k in keys:
            val = getattr(d, k, None)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
    return default


def _get_params_123(mat_or_params: Any) -> Law123Params:
    if isinstance(mat_or_params, Law123Params):
        return mat_or_params
    p = getattr(mat_or_params, "params", mat_or_params) or {}
    rho = getattr(mat_or_params, "rho0", None)
    if rho is None:
        rho = _fval(p, "Rho", "rho", "rho0", "RHO", default=0.0)

    ea = _fval(p, "LSDYNA_EA", "ea", "EA", "E1", "e1", "E", default=0.0)
    eb = _fval(p, "LSDYNA_EB", "eb", "EB", "E2", "e2", default=ea)
    ec = _fval(p, "LSDYNA_EC", "ec", "EC", "E3", "e3", default=eb)
    gab = _fval(p, "LSDYNA_GAB", "gab", "GAB", "G12", "g12", default=0.0)
    gca = _fval(p, "LSDYNA_GCA", "gca", "GCA", "G31", "g31", "G13", "g13", default=gab)
    gbc = _fval(p, "LSDYNA_GBC", "gbc", "GBC", "G23", "g23", default=gca)
    prba = _fval(p, "LSDYNA_PRBA", "prba", "PRBA", "nu21", "Nu12", "nu12", default=0.3)
    prca = _fval(p, "LSDYNA_PRCA", "prca", "PRCA", "nu31", "Nu31", default=prba)
    prcb = _fval(p, "LSDYNA_PRCB", "prcb", "PRCB", "nu32", "Nu32", default=prba)

    return Law123Params(
        rho0=float(rho),
        ea=ea, eb=eb, ec=ec,
        gab=gab, gca=gca, gbc=gbc,
        prba=prba, prca=prca, prcb=prcb,
        enkink=_fval(p, "LSD_ENKINK", "enkink", "ENKINK", "G1c", default=_INF),
        ena=_fval(p, "LSD_ENA", "ena", "ENA", "G1t", default=_INF),
        enb=_fval(p, "LSD_ENB", "enb", "ENB", "G2t", default=_INF),
        ent=_fval(p, "LSD_ENT", "ent", "ENT", "G2s", default=_INF),
        enl=_fval(p, "LSD_ENL", "enl", "ENL", "G12_fail", default=_INF),
        xc=_fval(p, "LSD_XC", "xc", "XC", default=_INF),
        xt=_fval(p, "LSD_MAT_XT", "xt", "XT", default=_INF),
        yc=_fval(p, "LSD_MAT_YC", "yc", "YC", default=_INF),
        yt=_fval(p, "LSD_MAT_YT", "yt", "YT", default=_INF),
        sl=_fval(p, "LSD_SL", "sl", "SL", default=_INF),
        fio=_fval(p, "LSD_FIO", "fio", "FIO", default=53.0),
        sigy=_fval(p, "LSDYNA_SIGY", "sigy", "SIGY", default=0.0),
        beta=_fval(p, "LSD_MAT_BETA", "beta", "BETA", default=0.0),
        lcss=_ival(p, "LSD_LCSS", "lcss", "LCSS", default=0),
        efs=_fval(p, "EFS", "efs", "LSD_EFS", default=_INF),
        ratio=_fval(p, "LRD_RATIO", "ratio", "RATIO", default=1.0),
        fcut=_fval(p, "FCUT", "fcut", default=0.0),
    )


def _get_params_132(mat_or_params: Any) -> Law132Params:
    if isinstance(mat_or_params, Law132Params):
        return mat_or_params
    p = getattr(mat_or_params, "params", mat_or_params) or {}
    rho = getattr(mat_or_params, "rho0", None)
    if rho is None:
        rho = _fval(p, "Rho", "rho", "rho0", "RHO", default=0.0)

    ea = _fval(p, "LSDYNA_EA", "ea", "EA", "E1", "e1", "E", default=0.0)
    eb = _fval(p, "LSDYNA_EB", "eb", "EB", "E2", "e2", default=ea)
    ec = _fval(p, "LSDYNA_EC", "ec", "EC", "E3", "e3", default=eb)
    gab = _fval(p, "LSDYNA_GAB", "gab", "GAB", "G12", "g12", default=0.0)
    gca = _fval(p, "LSDYNA_GCA", "gca", "GCA", "G31", "g31", "G13", "g13", default=gab)
    gbc = _fval(p, "LSDYNA_GBC", "gbc", "GBC", "G23", "g23", default=gca)
    prba = _fval(p, "LSDYNA_PRBA", "prba", "PRBA", "nu21", "Nu12", "nu12", default=0.3)
    prca = _fval(p, "LSDYNA_PRCA", "prca", "PRCA", "nu31", "Nu31", default=prba)
    prcb = _fval(p, "LSDYNA_PRCB", "prcb", "PRCB", "nu32", "Nu32", default=prba)

    return Law132Params(
        rho0=float(rho),
        ea=ea, eb=eb, ec=ec,
        gab=gab, gca=gca, gbc=gbc,
        prba=prba, prca=prca, prcb=prcb,
        gxc=_fval(p, "LSD_GXC", "gxc", "GXC", default=_INF),
        gxt=_fval(p, "LSD_GXT", "gxt", "GXT", default=_INF),
        gyc=_fval(p, "LSD_GYC", "gyc", "GYC", default=_INF),
        gyt=_fval(p, "LSD_GYT", "gyt", "GYT", default=_INF),
        gsl=_fval(p, "LSD_GSL", "gsl", "GSL", default=_INF),
        xc=_fval(p, "LSD_MAT_XC", "xc", "XC", default=_INF),
        xt=_fval(p, "LSD_MAT_XT", "xt", "XT", default=_INF),
        yc=_fval(p, "LSD_MAT_YC", "yc", "YC", default=_INF),
        yt=_fval(p, "LSD_MAT_YT", "yt", "YT", default=_INF),
        sl=_fval(p, "LSD_MAT_SL", "sl", "SL", default=_INF),
        gxc0=_fval(p, "LSD_GXC0", "gxc0", "GXC0", default=0.0),
        gxt0=_fval(p, "LSD_GXT0", "gxt0", "GXT0", default=0.0),
        xc0=_fval(p, "LSD_MAT_XC0", "xc0", "XC0", default=0.0),
        xt0=_fval(p, "LSD_MAT_XT0", "xt0", "XT0", default=0.0),
        fio=_fval(p, "LSD_FIO", "fio", "FIO", default=53.0),
        sigy=_fval(p, "LSDYNA_SIGY", "sigy", "SIGY", default=0.0),
        etan=_fval(p, "LSDYNA_ETAN", "etan", "ETAN", default=0.0),
        beta=_fval(p, "LSD_MAT_BETA", "beta", "BETA", default=0.0),
        lcss=_ival(p, "LSD_LCSS", "lcss", "LCSS", default=0),
        epsf23=_fval(p, "LSD_MAT_EPSF23", "epsf23", default=_INF),
        epsr23=_fval(p, "LSD_MAT_EPSR23", "epsr23", default=_INF),
        tsmd23=_fval(p, "LSD_MAT_TSMD23", "tsmd23", default=0.99),
        epsf31=_fval(p, "LSD_MAT_EPSF31", "epsf31", default=_INF),
        epsr31=_fval(p, "LSD_MAT_EPSR31", "epsr31", default=_INF),
        tsmd31=_fval(p, "LSD_MAT_TSMD31", "tsmd31", default=0.99),
        ef11t=_fval(p, "LSD_MAT_EF11T", "ef11t", default=_INF),
        ef11c=_fval(p, "LSD_MAT_EF11C", "ef11c", default=_INF),
        ef22t=_fval(p, "LSD_MAT_EF22T", "ef22t", default=_INF),
        ef22c=_fval(p, "LSD_MAT_EF22C", "ef22c", default=_INF),
        ef12=_fval(p, "LSD_MAT_EF12", "ef12", default=_INF),
        ef23=_fval(p, "LSD_MAT_EF23", "ef23", default=_INF),
        ef31=_fval(p, "LSD_MAT_EF31", "ef31", default=_INF),
        cf12=_fval(p, "LSD_MAT_CF12", "cf12", default=0.0),
        cf23=_fval(p, "LSD_MAT_CF23", "cf23", default=0.0),
        cf31=_fval(p, "LSD_MAT_CF31", "cf31", default=0.0),
        ratio=_fval(p, "LRD_RATIO", "ratio", "RATIO", default=1.0),
        fcut=_fval(p, "FCUT", "fcut", default=0.0),
    )


def build_law123(rec: Any) -> Material:
    """Factory creating Material for LAW123 (Daimler-Pinho)."""
    p = getattr(rec, "params", None)
    if p is None:
        if isinstance(rec, dict):
            p = rec
        elif hasattr(rec, "__dict__"):
            p = rec.__dict__
        else:
            p = {}
    rho = getattr(rec, "density", None)
    if rho is None:
        rho = getattr(rec, "rho0", _fval(p, "Rho", "rho", "rho0", default=0.0))
    mat_id = getattr(rec, "id", 1)
    title = getattr(rec, "title", "LAW123 Daimler-Pinho")
    params = dict(p)
    return Material(id=mat_id, law=123, rho0=float(rho), title=title, params=params)


def build_law132(rec: Any) -> Material:
    """Factory creating Material for LAW132 (Daimler-Camanho)."""
    p = getattr(rec, "params", None)
    if p is None:
        if isinstance(rec, dict):
            p = rec
        elif hasattr(rec, "__dict__"):
            p = rec.__dict__
        else:
            p = {}
    rho = getattr(rec, "density", None)
    if rho is None:
        rho = getattr(rec, "rho0", _fval(p, "Rho", "rho", "rho0", default=0.0))
    mat_id = getattr(rec, "id", 1)
    title = getattr(rec, "title", "LAW132 Daimler-Camanho")
    params = dict(p)
    return Material(id=mat_id, law=132, rho0=float(rho), title=title, params=params)


# ============================================================================
# Physically-Based Daimler-Pinho / Daimler-Camanho Failure Criteria
# ============================================================================

def analyze_failure(
    s_t: float,
    s_l: float,
    y_t: float,
    mu_t: float,
    mu_l: float,
    sigma_b: float,
    sigma_c: float,
    tau_bc: float,
    tau_ab: float,
    tau_ca: float,
) -> Tuple[float, float]:
    """Find the fracture plane angle phi in [0, pi] maximizing the failure index f(phi).

    Fortran origin:
      engine/source/materials/mat/mat123/failure_tools_mod.F90 lines 48-150.
      engine/source/materials/mat/mat123/analyze_failure.F90 lines 51-97.

    Formulas on fracture plane at angle phi:
      sigma_n = 0.5*(sigma_b + sigma_c) + 0.5*(sigma_b - sigma_c)*cos(2*phi) + tau_bc*sin(2*phi)
      tau_T   = -0.5*(sigma_b - sigma_c)*sin(2*phi) + tau_bc*cos(2*phi)
      tau_L   = tau_ab*cos(phi) + tau_ca*sin(phi)

      If sigma_n <= 0 (compression/shear):
        denom_T = S_T - mu_T * sigma_n
        denom_L = S_L - mu_L * sigma_n
        f(phi)  = (tau_T / denom_T)^2 + (tau_L / denom_L)^2
      Else (tension/shear):
        f(phi)  = (sigma_n / Y_t)^2 + (tau_T / S_T)^2 + (tau_L / S_L)^2
    """
    s_t_safe = max(s_t, _EM20)
    s_l_safe = max(s_l, _EM20)
    y_t_safe = max(y_t, _EM20)

    # 1. Vectorized coarse sweep across [0, pi] (73 points = every 2.5 deg)
    phi_grid = np.linspace(0.0, math.pi, 73)
    cos2p = np.cos(2.0 * phi_grid)
    sin2p = np.sin(2.0 * phi_grid)
    cosp = np.cos(phi_grid)
    sinp = np.sin(phi_grid)

    aa = 0.5 * (sigma_b + sigma_c)
    bb = 0.5 * (sigma_b - sigma_c)

    sig_n = aa + bb * cos2p + tau_bc * sin2p
    tau_t = -bb * sin2p + tau_bc * cos2p
    tau_l = tau_ab * cosp + tau_ca * sinp

    f_vals = np.zeros_like(phi_grid)
    mask_neg = (sig_n <= 0.0)

    # Compression regime
    denom_t = s_t_safe - mu_t * sig_n[mask_neg]
    denom_l = s_l_safe - mu_l * sig_n[mask_neg]
    invalid = (denom_t <= 0.0) | (denom_l <= 0.0)
    denom_t = np.where(invalid, 1.0, denom_t)
    denom_l = np.where(invalid, 1.0, denom_l)
    f_comp = (tau_t[mask_neg] / denom_t)**2 + (tau_l[mask_neg] / denom_l)**2
    f_comp[invalid] = _INF
    f_vals[mask_neg] = f_comp

    # Tension regime
    mask_pos = ~mask_neg
    f_tens = (sig_n[mask_pos] / y_t_safe)**2 + (tau_t[mask_pos] / s_t_safe)**2 + (tau_l[mask_pos] / s_l_safe)**2
    f_vals[mask_pos] = f_tens

    best_idx = int(np.argmax(f_vals))
    phi_best = float(phi_grid[best_idx])
    f_best = float(f_vals[best_idx])

    # 2. Golden section refinement around the best candidate
    delta = math.pi / 72.0
    a = max(0.0, phi_best - delta)
    b = min(math.pi, phi_best + delta)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0

    def _eval(p_val: float) -> float:
        c2 = math.cos(2.0 * p_val)
        s2 = math.sin(2.0 * p_val)
        cp = math.cos(p_val)
        sp = math.sin(p_val)
        sn = aa + bb * c2 + tau_bc * s2
        tt = -bb * s2 + tau_bc * c2
        tl = tau_ab * cp + tau_ca * sp
        if sn <= 0.0:
            dt = s_t_safe - mu_t * sn
            dl = s_l_safe - mu_l * sn
            if dt <= 0.0 or dl <= 0.0:
                return _INF
            return (tt / dt)**2 + (tl / dl)**2
        return (sn / y_t_safe)**2 + (tt / s_t_safe)**2 + (tl / s_l_safe)**2

    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = _eval(c)
    fd = _eval(d)

    for _ in range(25):
        if abs(b - a) < 1.0e-6:
            break
        if fc > fd:
            b = d
            d = c
            fd = fc
            c = b - inv_phi * (b - a)
            fc = _eval(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + inv_phi * (b - a)
            fd = _eval(d)

    phi_opt = 0.5 * (a + b)
    f_opt = _eval(phi_opt)
    if f_opt > f_best:
        return phi_opt, f_opt
    return phi_best, f_best


# ============================================================================
# Single-Integration-Point Update Logic
# ============================================================================

def _update_point_law123(
    p: Law123Params,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float,
    dt: float,
    is_solid: bool,
    uvar: np.ndarray,
    dmg: np.ndarray,
    off: float,
    offl: float,
    char_len: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float, float]:
    """Constitutive integration for /MAT/LAW123 (Daimler-Pinho)."""
    # 0. Check element deletion
    if off <= 0.0 or offl <= 0.0:
        return np.zeros_like(sig), epsp, uvar, dmg, 0.0, 0.0, p.sound_spd_solid if is_solid else p.sound_spd_shell

    l_car = max(char_len, _EM10) if char_len > 0.0 else (uvar[15] if uvar[15] > 0.0 else 1.0)
    uvar[15] = l_car

    # 1. Total strain integration & effective elastic stress (undamaged)
    # Total strain stored in uvar[16:22]
    eps_tot = uvar[16:22] + deps[:6] if len(deps) >= 6 else np.pad(deps, (0, 6 - len(deps)))
    uvar[16:22] = eps_tot

    if is_solid:
        # 3D solid kinematics: deps = [xx, yy, zz, xy, yz, zx]
        sigma_a = p.d_solid[0, 0] * eps_tot[0] + p.d_solid[0, 1] * eps_tot[1] + p.d_solid[0, 2] * eps_tot[2]
        sigma_b = p.d_solid[1, 0] * eps_tot[0] + p.d_solid[1, 1] * eps_tot[1] + p.d_solid[1, 2] * eps_tot[2]
        sigma_c = p.d_solid[2, 0] * eps_tot[0] + p.d_solid[2, 1] * eps_tot[1] + p.d_solid[2, 2] * eps_tot[2]
        tau_ab = p.gab * eps_tot[3]
        tau_bc = p.gbc * eps_tot[4]
        tau_ca = p.gca * eps_tot[5]
    else:
        # 2D plane stress kinematics: deps = [xx, yy, xy]
        sigma_a = p.c_plane[0, 0] * eps_tot[0] + p.c_plane[0, 1] * eps_tot[1]
        sigma_b = p.c_plane[1, 0] * eps_tot[0] + p.c_plane[1, 1] * eps_tot[1]
        sigma_c = 0.0
        tau_ab = p.gab * eps_tot[2]
        tau_bc = 0.0
        tau_ca = 0.0

    # 2. Check effective strain failure limit (EFS)
    eps_eq = math.sqrt(2.0 / 3.0 * float(np.sum(eps_tot**2)))
    if eps_eq >= p.efs:
        dmg[0] = 1.0
        off = 0.0
        offl = 0.0
        return np.zeros_like(sig), epsp, uvar, dmg, 0.0, 0.0, p.sound_spd_solid if is_solid else p.sound_spd_shell

    dfiber = dmg[1]
    dkink = dmg[2]
    dmat = dmg[3]

    # =========================================================================
    # A. Fiber Tensile Failure Mode (dfiber)
    # =========================================================================
    if dmg[4] == 1.0 and dfiber < 1.0:
        sig0 = uvar[1]
        eps0 = uvar[2]
        epsf = uvar[3]
        if eps_tot[0] > eps0 and (epsf - eps0) > _EM20:
            dfiber = max(dfiber, epsf * (eps_tot[0] - eps0) / max(eps_tot[0] * (epsf - eps0), _EM20))
            dfiber = min(1.0, max(0.0, dfiber))
            dmg[1] = dfiber
    elif sigma_a >= p.xt and dfiber == 0.0:
        sig0 = sigma_a
        eps0 = eps_tot[0]
        epsf = 2.0 * p.ena / max(sig0 * l_car, _EM20)
        if epsf < eps0:
            epsf = 1.1 * eps0
        uvar[1] = sig0
        uvar[2] = eps0
        uvar[3] = epsf
        dmg[4] = 1.0

    # =========================================================================
    # B. Fiber Compressive Failure Mode / Kinking (dkink)
    # =========================================================================
    if dmg[5] == 1.0 and dkink < 1.0:
        sig0_kink = uvar[4]
        eps0_kink = uvar[5]
        epsf_kink = uvar[6]
        if sig0_kink > 0.0:
            psi = uvar[12]
            c_p, s_p = math.cos(psi), math.sin(psi)
            sigma_b_psi = (c_p**2) * sigma_b + (s_p**2) * sigma_c + 2.0 * c_p * s_p * tau_bc
            tau_ab_psi = c_p * tau_ab + s_p * tau_ca
            theta = uvar[13]
            c_t, s_t = math.cos(theta), math.sin(theta)
            tau_ab_m = abs((sigma_b_psi - sigma_a) * c_t * s_t + (c_t**2 - s_t**2) * tau_ab_psi)
            eps_kink = tau_ab_m / max(p.gab, _EM20)
            if eps_kink > eps0_kink and (epsf_kink - eps0_kink) > _EM20:
                dkink = max(dkink, epsf_kink * (eps_kink - eps0_kink) / max(eps_kink * (epsf_kink - eps0_kink), _EM20))
                dkink = min(1.0, max(0.0, dkink))
                dmg[2] = dkink
        else:
            if abs(eps_tot[0]) > eps0_kink and (epsf_kink - eps0_kink) > _EM20:
                dkink = max(dkink, epsf_kink * (abs(eps_tot[0]) - eps0_kink) / max(abs(eps_tot[0]) * (epsf_kink - eps0_kink), _EM20))
                dkink = min(1.0, max(0.0, dkink))
                dmg[2] = dkink
    elif sigma_a < 0.0 and dmg[5] == 0.0:
        # Step 1: Kink band orientation angle psi in (b, c) plane
        if abs(sigma_b - sigma_c) > _EM10:
            psi = 0.5 * math.atan2(2.0 * tau_bc, sigma_b - sigma_c)
        elif abs(tau_bc) > _EM10:
            psi = 0.5 * math.pi
        else:
            psi = 0.0

        c_p, s_p = math.cos(psi), math.sin(psi)
        sigma_b_psi = (c_p**2) * sigma_b + (s_p**2) * sigma_c + 2.0 * c_p * s_p * tau_bc
        sigma_c_psi = (s_p**2) * sigma_b + (c_p**2) * sigma_c - 2.0 * c_p * s_p * tau_bc
        tau_bc_psi = (sigma_c - sigma_b) * c_p * s_p + (c_p**2 - s_p**2) * tau_bc
        tau_ab_psi = c_p * tau_ab + s_p * tau_ca
        tau_ca_psi = -s_p * tau_ab + c_p * tau_ca

        # Step 2: Inelastic misalignment shear
        tau_eff = abs(0.5 * (sigma_a - sigma_b_psi) * math.sin(2.0 * p.thetai)
                      + abs(tau_ab_psi) * math.cos(2.0 * p.thetai))
        gamai = tau_eff / max(p.gab, _EM20)
        sgn_tau = 1.0 if tau_ab_psi >= 0.0 else -1.0
        theta = sgn_tau * (p.thetai + gamai)

        c_t, s_t = math.cos(theta), math.sin(theta)
        sigma_a_m = (c_t**2) * sigma_a + (s_t**2) * sigma_b_psi + 2.0 * c_t * s_t * tau_ab_psi
        sigma_b_m = (s_t**2) * sigma_a + (c_t**2) * sigma_b_psi - 2.0 * c_t * s_t * tau_ab_psi
        tau_ab_m = (sigma_b_psi - sigma_a) * c_t * s_t + (c_t**2 - s_t**2) * tau_ab_psi
        tau_cpsi_am = c_t * tau_ca_psi + s_t * tau_bc_psi
        tau_bm_cpsi = -s_t * tau_ca_psi + c_t * tau_bc_psi

        # Step 3: Critical kinking fracture plane search
        crit_phi, max_f = analyze_failure(
            p.st, p.sl, p.yt, p.mut, p.mul,
            sigma_b_m, sigma_c_psi, tau_bm_cpsi, tau_ab_m, tau_cpsi_am
        )
        if max_f >= 1.0:
            tau_ab_m = abs(tau_ab_m)
            uvar[12] = psi
            uvar[13] = theta
            sig0_kink = tau_ab_m
            eps0_kink = tau_ab_m / max(p.gab, _EM20)
            epsf_kink = 2.0 * p.enkink / max(tau_ab_m * l_car, _EM20)
            if epsf_kink < eps0_kink:
                epsf_kink = 1.1 * eps0_kink
            uvar[4] = sig0_kink
            uvar[5] = eps0_kink
            uvar[6] = epsf_kink
            dmg[5] = 1.0
        elif abs(sigma_a) >= p.xc:
            uvar[4] = -p.xc
            uvar[5] = p.xc / max(p.ea, _EM20)
            uvar[6] = 2.0 * p.enl / max(p.xc * l_car, _EM20)
            dmg[5] = 1.0

    # =========================================================================
    # C. Matrix Cracking Failure Mode (dmat)
    # =========================================================================
    if dmg[6] == 1.0 and dmat < 1.0:
        sig0_mat = uvar[7]
        eps0_mat = uvar[8]
        epsf_mat = uvar[9]
        phi = uvar[14]

        cos2p = math.cos(2.0 * phi)
        sin2p = math.sin(2.0 * phi)
        cosp = math.cos(phi)
        sinp = math.sin(phi)
        aa = 0.5 * (sigma_b + sigma_c)
        bb = 0.5 * (sigma_b - sigma_c)

        sig_n = aa + bb * cos2p + tau_bc * sin2p
        tau_t = -bb * sin2p + tau_bc * cos2p
        tau_l = tau_ab * cosp + tau_ca * sinp
        tau_mat = math.sqrt(tau_t**2 + tau_l**2)
        sig_n_p = max(0.0, sig_n)

        # Strain components
        epsb = eps_tot[1]
        epsc = eps_tot[2] if is_solid else (-p.nu13 * eps_tot[0] - p.nu23 * eps_tot[1])
        gambc = eps_tot[4] if is_solid else 0.0
        gamca = eps_tot[5] if is_solid else 0.0
        gamab = eps_tot[3] if is_solid else eps_tot[2]

        epsn = 0.5 * (epsb + epsc + (epsb - epsc) * cos2p + gambc * sin2p)
        gamat = -(epsb - epsc) * sin2p + gambc * sin2p
        gamal = gamab * cosp + gamca * sinp

        omega = math.atan2(sig_n_p, tau_mat) if tau_mat > _EM10 else (0.5 * math.pi if sig_n_p > _EM10 else 0.0)
        lamda = math.atan2(tau_l, tau_t) if abs(tau_t) > _EM10 else (0.5 * math.pi if abs(tau_l) > _EM10 else 0.0)
        gam_mat = abs(gamat * math.cos(lamda) + gamal * math.sin(lamda))
        eps_mat = (sig_n_p * epsn * math.sin(omega) / max(abs(sig_n), _EM20)) + gam_mat * math.cos(omega)

        if eps_mat > eps0_mat and (epsf_mat - eps0_mat) > _EM20:
            dmat = max(dmat, epsf_mat * (eps_mat - eps0_mat) / max(eps_mat * (epsf_mat - eps0_mat), _EM20))
            dmat = min(1.0, max(0.0, dmat))
            dmg[3] = dmat
    elif dmg[6] == 0.0:
        crit_phi, max_f = analyze_failure(
            p.st, p.sl, p.yt, p.mut, p.mul,
            sigma_b, sigma_c, tau_bc, tau_ab, tau_ca
        )
        if max_f >= 1.0:
            phi = crit_phi
            cos2p = math.cos(2.0 * phi)
            sin2p = math.sin(2.0 * phi)
            cosp = math.cos(phi)
            sinp = math.sin(phi)
            aa = 0.5 * (sigma_b + sigma_c)
            bb = 0.5 * (sigma_b - sigma_c)
            sig_n = aa + bb * cos2p + tau_bc * sin2p
            tau_t = -bb * sin2p + tau_bc * cos2p
            tau_l = tau_ab * cosp + tau_ca * sinp
            tau_mat = math.sqrt(tau_t**2 + tau_l**2)
            sig_n_p = max(0.0, sig_n)
            sig_mat = math.sqrt(sig_n_p**2 + tau_mat**2)

            epsb = eps_tot[1]
            epsc = eps_tot[2] if is_solid else (-p.nu13 * eps_tot[0] - p.nu23 * eps_tot[1])
            gambc = eps_tot[4] if is_solid else 0.0
            gamca = eps_tot[5] if is_solid else 0.0
            gamab = eps_tot[3] if is_solid else eps_tot[2]

            epsn = 0.5 * (epsb + epsc + (epsb - epsc) * cos2p + gambc * sin2p)
            gamat = -(epsb - epsc) * sin2p + gambc * sin2p
            gamal = gamab * cosp + gamca * sinp

            omega = math.atan2(sig_n_p, tau_mat) if tau_mat > _EM10 else (0.5 * math.pi if sig_n_p > _EM10 else 0.0)
            lamda = math.atan2(tau_l, tau_t) if abs(tau_t) > _EM10 else (0.5 * math.pi if abs(tau_l) > _EM10 else 0.0)
            gam_mat = abs(gamat * math.cos(lamda) + gamal * math.sin(lamda))
            eps_mat = (sig_n_p * epsn * math.sin(omega) / max(abs(sig_n), _EM20)) + gam_mat * math.cos(omega)

            # Mixed-mode toughness
            s_mat_sq = max(sig_mat**2, _EM20)
            en_mat = (p.enb * ((sig_n_p**2) / s_mat_sq)
                      + p.ent * ((tau_t**2) / s_mat_sq)
                      + p.enl * ((tau_l**2) / s_mat_sq))
            epsf_mat = 2.0 * max(en_mat, p.enb) / max(sig_mat * l_car, _EM20)
            if epsf_mat < eps_mat:
                epsf_mat = 2.0 * eps_mat

            uvar[7] = sig_mat
            uvar[8] = eps_mat
            uvar[9] = epsf_mat
            uvar[14] = crit_phi
            dmg[6] = 1.0

    # Total scalar damage d
    d = min(1.0, max(dfiber, dkink, dmat))
    dmg[0] = d

    # Element deletion
    if d >= 0.999:
        off = 0.0
        offl = 0.0
        return np.zeros_like(sig), epsp, uvar, dmg, 0.0, 0.0, p.sound_spd_solid if is_solid else p.sound_spd_shell

    # Degraded stress
    factor = max(0.0, 1.0 - d)
    if is_solid:
        sig_out = np.array([
            factor * sigma_a,
            factor * sigma_b,
            factor * sigma_c,
            factor * tau_ab,
            factor * tau_bc,
            factor * tau_ca,
        ], dtype=float)
    else:
        sig_out = np.array([
            factor * sigma_a,
            factor * sigma_b,
            factor * tau_ab,
        ], dtype=float)

    return sig_out, epsp, uvar, dmg, off, offl, p.sound_spd_solid if is_solid else p.sound_spd_shell


def _update_point_law132(
    p: Law132Params,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float,
    dt: float,
    is_solid: bool,
    uvar: np.ndarray,
    dmg: np.ndarray,
    off: float,
    offl: float,
    char_len: float,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float, float]:
    """Constitutive integration for /MAT/LAW132 (Daimler-Camanho)."""
    if off <= 0.0 or offl <= 0.0:
        return np.zeros_like(sig), epsp, uvar, dmg, 0.0, 0.0, p.sound_spd_solid if is_solid else p.sound_spd_shell

    l_char = max(char_len, _EM10) if char_len > 0.0 else (uvar[4] if uvar[4] > 0.0 else 1.0)
    uvar[4] = l_char

    # Total strain
    eps_tot = uvar[16:22] + deps[:6] if len(deps) >= 6 else np.pad(deps, (0, 6 - len(deps)))
    uvar[16:22] = eps_tot

    # Check failure strain limits
    if (eps_tot[0] >= p.ef11t or eps_tot[0] <= -p.ef11c
            or eps_tot[1] >= p.ef22t or eps_tot[1] <= -p.ef22c
            or abs(eps_tot[2]) >= p.ef12):
        dmg[0] = 1.0
        off = 0.0
        offl = 0.0
        return np.zeros_like(sig), epsp, uvar, dmg, 0.0, 0.0, p.sound_spd_solid if is_solid else p.sound_spd_shell

    # State variables from previous cycle: thresholds r(1..4)
    r1p, r1n, r2p, r2n = uvar[0], uvar[1], uvar[2], uvar[3]
    d1p, d1n, d2p, d2n, d6 = dmg[1], dmg[2], dmg[3], dmg[4], dmg[5]

    # Undamaged effective stress in plane
    inv_det = 1.0 / max(1.0 - p.nu12 * p.nu21, _EM20)
    sigma_a = inv_det * (p.ea * eps_tot[0] + p.nu21 * p.ea * eps_tot[1])
    sigma_b = inv_det * (p.nu12 * p.eb * eps_tot[0] + p.eb * eps_tot[1])
    tau_ab = p.gab * eps_tot[2]
    tau_bc = p.gbc * eps_tot[4] if is_solid else 0.0
    tau_ca = p.gca * eps_tot[5] if is_solid else 0.0

    # In-plane shear plasticity
    if p.etan > 0.0 or p.sigy > 0.0:
        sig_yld = p.sigy + p.etan * uvar[6]
        if abs(tau_ab) >= sig_yld:
            tau_ab = math.copysign(sig_yld, tau_ab)
            uvar[6] = abs(eps_tot[2]) - sig_yld / max(p.gab, _EM20)

    # 1. Loading functions phi
    phi_1p, phi_1n, phi_2p, phi_2n = 0.0, 0.0, 0.0, 0.0

    # Fiber tension / compression
    if sigma_a >= 0.0:
        phi_1p = (sigma_a - p.nu12 * sigma_b) / max(p.xt, _EM20)
    else:
        cos_phi = math.cos(p.theta_c)
        sin_phi = math.sin(p.theta_c)
        cos2_phi = cos_phi * cos_phi
        sin2_phi = sin_phi * sin_phi
        sigma_bm = sigma_a * sin2_phi + sigma_b * cos2_phi - 2.0 * abs(tau_ab) * sin_phi * cos_phi
        sigma_ab_m = (sigma_b - sigma_a) * sin_phi * cos_phi + abs(tau_ab) * (cos2_phi - sin2_phi)
        phi_1n = max(0.0, (abs(sigma_ab_m) + p.eta_l * sigma_bm) / max(p.sl, _EM20))

    # Matrix tension / compression
    if sigma_b >= 0.0:
        t1 = sigma_b / max(p.yt, _EM20)
        t2 = tau_ab / max(p.sl, _EM20)
        phi_2p = math.sqrt(max(0.0, (1.0 - p.g_ratio) * t1 + p.g_ratio * (t1**2) + (t2**2)))
    else:
        # Sliding angle theta (Eq. 16)
        sin_a = math.sin(_RAD53)
        cos_a = math.cos(_RAD53)
        if abs(sigma_b) * sin_a > _EM10 and abs(tau_ab) > _EM10:
            theta = math.atan2(-abs(tau_ab), sigma_b * sin_a)
        else:
            theta = 0.5 * math.pi if abs(tau_ab) > _EM10 else 0.0

        cos_th = math.cos(theta)
        sin_th = math.sin(theta)
        tau_t = (-sigma_b * cos_a) * (sin_a - p.eta_t * cos_a * cos_th)
        tau_l = cos_a * (abs(tau_ab) + p.eta_l * sigma_b * cos_a * sin_th)
        phi_2n = math.sqrt((tau_t / max(p.st, _EM20))**2 + (tau_l / max(p.sl, _EM20))**2)

    # 2. Update damage thresholds
    r1p = max(r1p, phi_1p)
    r1n = max(r1n, phi_1n)
    r2p = max(r2p, phi_2p)
    r2n = max(r2n, phi_2n)
    uvar[0], uvar[1], uvar[2], uvar[3] = r1p, r1n, r2p, r2n

    # Volumetric fracture energy densities
    g1p_vol = p.gxt / l_char
    g1n_vol = p.gxc / l_char
    g2p_vol = p.gyt / l_char
    g2n_vol = p.gyc / l_char
    g6_vol = p.gsl / l_char

    # 3. Damage variable evolutions
    # Matrix tension (d2+)
    if d2p == 0.0 and r2p >= 1.0:
        uvar[12] = eps_tot[1]
        uvar[13] = abs(sigma_b)
        d2p = 1.0e-10
    elif 0.0 < d2p < 1.0:
        eps0 = uvar[12]
        sig0 = max(uvar[13], _EM20)
        epsf = 2.0 * g2p_vol / sig0
        if epsf < eps0:
            epsf = 1.2 * eps0
        if eps_tot[1] < eps0:
            d2p = 0.0
        elif eps_tot[1] < epsf:
            d2p = min(1.0, epsf * (abs(eps_tot[1]) - eps0) / max(abs(eps_tot[1]) * (epsf - eps0), _EM20))
        else:
            d2p = 1.0

    # Matrix compression (d2-)
    if d2n == 0.0 and r2n >= 1.0:
        uvar[12] = abs(eps_tot[1])
        uvar[13] = abs(sigma_b)
        d2n = 1.0e-10
    elif 0.0 < d2n < 1.0:
        eps0 = uvar[12]
        sig0 = max(uvar[13], _EM20)
        epsf = 2.0 * g2n_vol / sig0
        if epsf < eps0:
            epsf = 1.2 * eps0
        d2n = min(1.0, epsf * (abs(eps_tot[1]) - eps0) / max(abs(eps_tot[1]) * (epsf - eps0), _EM20))

    # Fiber tension (d1+)
    if d1p == 0.0 and r1p >= 1.0:
        uvar[8] = abs(eps_tot[0])
        uvar[9] = abs(sigma_a)
        d1p = 1.0e-10
    elif 0.0 < d1p < 1.0:
        eps0 = p.xt / max(p.ea, _EM20)
        epsf = 2.0 * g1p_vol / max(p.xt, _EM20)
        if epsf < eps0:
            epsf = 1.2 * eps0
        k1 = p.xt / max(epsf - eps0, _EM20) / max(p.ea, _EM20)
        d1p = max(d1p, 1.0 + k1 - (k1 + 1.0) / max(r1p, _EM20))
        d1p = min(1.0, max(0.0, d1p))

    # Fiber compression (d1-)
    if d1n == 0.0 and r1n >= 1.0:
        uvar[8] = abs(eps_tot[0])
        uvar[9] = abs(sigma_a)
        d1n = 1.0e-10
    elif 0.0 < d1n < 1.0:
        eps0 = p.xc / max(p.ea, _EM20)
        epsf = 2.0 * g1n_vol / max(p.xc, _EM20)
        if epsf < eps0:
            epsf = 1.2 * eps0
        d1n = max(d1n, epsf * (abs(eps_tot[0]) - eps0) / max(abs(eps_tot[0]) * (epsf - eps0), _EM20))
        d1n = min(1.0, max(0.0, d1n))

    # In-plane shear damage (d6)
    if d6 == 0.0 and r2p > 1.0:
        uvar[14] = abs(eps_tot[2])
        uvar[15] = abs(tau_ab)
        d6 = 1.0e-10
    elif 0.0 < d6 < 1.0:
        gam0 = uvar[14]
        tau0 = max(uvar[15], _EM20)
        gamf = 2.0 * g6_vol / tau0
        if gamf < gam0:
            gamf = 1.2 * gam0
        if abs(eps_tot[2]) < gam0:
            dam_shear = 0.0
        elif abs(eps_tot[2]) < gamf:
            dam_shear = min(1.0, gamf * (abs(eps_tot[2]) - gam0) / max(abs(eps_tot[2]) * (gamf - gam0), _EM20))
        else:
            dam_shear = 1.0
        d6 = min(1.0, 1.0 - (1.0 - dam_shear) * (1.0 - d1p))

    # 4. Active damage variables with crack closure
    d1_act = d1p if sigma_a > 0.0 else (d1n if sigma_a < 0.0 else 0.0)
    d2_act = d2p if sigma_b > 0.0 else (d2n if sigma_b < 0.0 else 0.0)
    d6_act = min(1.0, 1.0 - (1.0 - d6) * (1.0 - d1p))

    dmg[1], dmg[2], dmg[3], dmg[4], dmg[5] = d1p, d1n, d2p, d2n, d6
    dmg[0] = max(d1_act, d2_act, d6_act)

    if dmg[0] >= 0.99:
        off = 0.0
        offl = 0.0
        return np.zeros_like(sig), epsp, uvar, dmg, 0.0, 0.0, p.sound_spd_solid if is_solid else p.sound_spd_shell

    # Damaged stiffness matrix
    e1_dam = max(_EM20, p.ea * (1.0 - d1_act))
    e2_dam = max(_EM20, p.eb * (1.0 - d2_act))
    g12_dam = max(_EM20, p.gab * (1.0 - d6_act))
    nu12_dam = p.nu12 * (1.0 - d1_act)
    nu21_dam = nu12_dam * e2_dam / e1_dam
    det_dam = max(1.0 - nu12_dam * nu21_dam, _EM20)

    d11 = e1_dam / det_dam
    d22 = e2_dam / det_dam
    d12 = nu12_dam * e2_dam / det_dam

    sign_a = d11 * eps_tot[0] + d12 * eps_tot[1]
    sign_b = d12 * eps_tot[0] + d22 * eps_tot[1]
    sign_ab = g12_dam * eps_tot[2]

    if is_solid:
        sig_out = np.array([sign_a, sign_b, 0.0, sign_ab, tau_bc, tau_ca], dtype=float)
    else:
        sig_out = np.array([sign_a, sign_b, sign_ab], dtype=float)

    return sig_out, epsp, uvar, dmg, off, offl, p.sound_spd_solid if is_solid else p.sound_spd_shell


# ============================================================================
# Step & Update APIs
# ============================================================================

def solid_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float = 0.0,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, float]:
    """Solid single-point step for LAW123 or LAW132."""
    ex = extra or {}
    uvar = ex.get("uvar")
    if uvar is None or len(uvar) < 22:
        uvar = np.zeros(22, dtype=float)
    else:
        uvar = np.asarray(uvar, dtype=float).copy()

    dmg = ex.get("dmg")
    if dmg is None or len(dmg) < 8:
        dmg = np.zeros(8, dtype=float)
    else:
        dmg = np.asarray(dmg, dtype=float).copy()

    off = float(ex.get("off", 1.0))
    offl = float(ex.get("offl", 1.0))
    char_len = float(ex.get("char_len", ex.get("l_car", 1.0)))

    law_num = getattr(mat, "law", 123)
    if law_num == 132:
        p132 = _get_params_132(mat)
        s_out, ep_out, uv_out, d_out, o_out, ol_out, c_out = _update_point_law132(
            p132, sig, deps, epsp, dt, True, uvar, dmg, off, offl, char_len
        )
    else:
        p123 = _get_params_123(mat)
        s_out, ep_out, uv_out, d_out, o_out, ol_out, c_out = _update_point_law123(
            p123, sig, deps, epsp, dt, True, uvar, dmg, off, offl, char_len
        )

    if extra is not None:
        extra["uvar"] = uv_out
        extra["dmg"] = d_out
        extra["off"] = o_out
        extra["offl"] = ol_out

    return s_out, ep_out, c_out


def shell_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float = 0.0,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, float, float]:
    """Shell single-point plane-stress step for LAW123 or LAW132."""
    ex = extra or {}
    uvar = ex.get("uvar")
    if uvar is None or len(uvar) < 22:
        uvar = np.zeros(22, dtype=float)
    else:
        uvar = np.asarray(uvar, dtype=float).copy()

    dmg = ex.get("dmg")
    if dmg is None or len(dmg) < 8:
        dmg = np.zeros(8, dtype=float)
    else:
        dmg = np.asarray(dmg, dtype=float).copy()

    off = float(ex.get("off", 1.0))
    offl = float(ex.get("offl", 1.0))
    char_len = float(ex.get("char_len", ex.get("l_car", 1.0)))

    law_num = getattr(mat, "law", 123)
    if law_num == 132:
        p132 = _get_params_132(mat)
        s_out, ep_out, uv_out, d_out, o_out, ol_out, c_out = _update_point_law132(
            p132, sig, deps, epsp, dt, False, uvar, dmg, off, offl, char_len
        )
    else:
        p123 = _get_params_123(mat)
        s_out, ep_out, uv_out, d_out, o_out, ol_out, c_out = _update_point_law123(
            p123, sig, deps, epsp, dt, False, uvar, dmg, off, offl, char_len
        )

    if extra is not None:
        extra["uvar"] = uv_out
        extra["dmg"] = d_out
        extra["off"] = o_out
        extra["offl"] = ol_out

    return s_out, ep_out, c_out


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], Union[float, np.ndarray]]:
    """Solid vectorized constitutive update for LAW123 and LAW132."""
    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)

    s = np.atleast_2d(sig_arr).copy()
    d = np.atleast_2d(deps_arr).copy()
    n = s.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=float)
    elif np.isscalar(epsp):
        ep = np.full(n, float(epsp), dtype=float)
    else:
        ep = np.asarray(epsp, dtype=float).copy()

    ex = extra or {}
    uvar_all = ex.get("uvar")
    if uvar_all is None or len(uvar_all) != n:
        uvar_all = np.zeros((n, 22), dtype=float)
    else:
        uvar_all = np.asarray(uvar_all, dtype=float).copy()

    dmg_all = ex.get("dmg")
    if dmg_all is None or len(dmg_all) != n:
        dmg_all = np.zeros((n, 8), dtype=float)
    else:
        dmg_all = np.asarray(dmg_all, dtype=float).copy()

    off_all = ex.get("off")
    if off_all is None or len(off_all) != n:
        off_all = np.ones(n, dtype=float)
    else:
        off_all = np.asarray(off_all, dtype=float).copy()

    offl_all = ex.get("offl")
    if offl_all is None or len(offl_all) != n:
        offl_all = np.ones(n, dtype=float)
    else:
        offl_all = np.asarray(offl_all, dtype=float).copy()

    l_car_all = ex.get("char_len", ex.get("l_car"))
    if l_car_all is None or len(np.atleast_1d(l_car_all)) != n:
        l_car_all = np.ones(n, dtype=float)
    else:
        l_car_all = np.asarray(l_car_all, dtype=float)

    s_out = np.zeros_like(s)
    c_arr = np.zeros(n, dtype=float)
    law_num = getattr(mat, "law", 123)

    for i in range(n):
        if law_num == 132:
            p132 = _get_params_132(mat)
            so, ep[i], uvar_all[i], dmg_all[i], off_all[i], offl_all[i], c_arr[i] = _update_point_law132(
                p132, s[i], d[i], ep[i], dt, True, uvar_all[i], dmg_all[i], off_all[i], offl_all[i], float(l_car_all[i])
            )
        else:
            p123 = _get_params_123(mat)
            so, ep[i], uvar_all[i], dmg_all[i], off_all[i], offl_all[i], c_arr[i] = _update_point_law123(
                p123, s[i], d[i], ep[i], dt, True, uvar_all[i], dmg_all[i], off_all[i], offl_all[i], float(l_car_all[i])
            )
        s_out[i] = so

    if extra is not None:
        extra["uvar"] = uvar_all
        extra["dmg"] = dmg_all
        extra["off"] = off_all
        extra["offl"] = offl_all

    if is_1d:
        return s_out[0], ep[0], float(c_arr[0])
    return s_out, ep, c_arr


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], Union[float, np.ndarray]]:
    """Shell vectorized constitutive update for LAW123 and LAW132."""
    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)
    is_1d = (sig_arr.ndim == 1)

    s = np.atleast_2d(sig_arr).copy()
    d = np.atleast_2d(deps_arr).copy()
    n = s.shape[0]

    if epsp is None:
        ep = np.zeros(n, dtype=float)
    elif np.isscalar(epsp):
        ep = np.full(n, float(epsp), dtype=float)
    else:
        ep = np.asarray(epsp, dtype=float).copy()

    ex = extra or {}
    uvar_all = ex.get("uvar")
    if uvar_all is None or len(uvar_all) != n:
        uvar_all = np.zeros((n, 22), dtype=float)
    else:
        uvar_all = np.asarray(uvar_all, dtype=float).copy()

    dmg_all = ex.get("dmg")
    if dmg_all is None or len(dmg_all) != n:
        dmg_all = np.zeros((n, 8), dtype=float)
    else:
        dmg_all = np.asarray(dmg_all, dtype=float).copy()

    off_all = ex.get("off")
    if off_all is None or len(off_all) != n:
        off_all = np.ones(n, dtype=float)
    else:
        off_all = np.asarray(off_all, dtype=float).copy()

    offl_all = ex.get("offl")
    if offl_all is None or len(offl_all) != n:
        offl_all = np.ones(n, dtype=float)
    else:
        offl_all = np.asarray(offl_all, dtype=float).copy()

    l_car_all = ex.get("char_len", ex.get("l_car"))
    if l_car_all is None or len(np.atleast_1d(l_car_all)) != n:
        l_car_all = np.ones(n, dtype=float)
    else:
        l_car_all = np.asarray(l_car_all, dtype=float)

    s_out = np.zeros_like(s)
    c_arr = np.zeros(n, dtype=float)
    law_num = getattr(mat, "law", 123)

    for i in range(n):
        if law_num == 132:
            p132 = _get_params_132(mat)
            so, ep[i], uvar_all[i], dmg_all[i], off_all[i], offl_all[i], c_arr[i] = _update_point_law132(
                p132, s[i], d[i], ep[i], dt, False, uvar_all[i], dmg_all[i], off_all[i], offl_all[i], float(l_car_all[i])
            )
        else:
            p123 = _get_params_123(mat)
            so, ep[i], uvar_all[i], dmg_all[i], off_all[i], offl_all[i], c_arr[i] = _update_point_law123(
                p123, s[i], d[i], ep[i], dt, False, uvar_all[i], dmg_all[i], off_all[i], offl_all[i], float(l_car_all[i])
            )
        s_out[i] = so

    if extra is not None:
        extra["uvar"] = uvar_all
        extra["dmg"] = dmg_all
        extra["off"] = off_all
        extra["offl"] = offl_all

    if is_1d:
        return s_out[0], ep[0], float(c_arr[0])
    return s_out, ep, c_arr


# ============================================================================
# Sound Speed & Algorithmic Tangent Stiffness
# ============================================================================

def sound_speed(mat: Any) -> float:
    """Acoustic sound speed calculation for LAW123 and LAW132."""
    law_num = getattr(mat, "law", 123)
    if law_num == 132:
        p132 = _get_params_132(mat)
        return p132.sound_spd_solid
    p123 = _get_params_123(mat)
    return p123.sound_spd_solid


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent stiffness tensor C (6, 6) for 3D solids."""
    law_num = getattr(mat, "law", 123)
    p = _get_params_132(mat) if law_num == 132 else _get_params_123(mat)
    c_elastic = p.d_solid.copy()

    if sig is None or deps is None:
        return c_elastic

    sig0 = np.asarray(sig, dtype=float).flatten()
    deps0 = np.asarray(deps, dtype=float).flatten()
    if len(sig0) < 6:
        sig0 = np.pad(sig0, (0, 6 - len(sig0)))
    if len(deps0) < 6:
        deps0 = np.pad(deps0, (0, 6 - len(deps0)))

    h = 1.0e-7
    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    s_base, _, _ = solid_step(mat, sig0, deps0, dt=dt, extra=ex0)

    c_algo = np.zeros((6, 6), dtype=float)
    for j in range(6):
        d_p = deps0.copy()
        d_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        s_p, _, _ = solid_step(mat, sig0, d_p, dt=dt, extra=ex_p)
        c_algo[:, j] = (s_p - s_base) / h

    return c_algo


def consistent_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    return solid_tangent(mat, sig=sig, deps=deps, dt=dt, extra=extra, **kwargs)


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent stiffness tensor C (3, 3) for plane-stress shells."""
    law_num = getattr(mat, "law", 123)
    p = _get_params_132(mat) if law_num == 132 else _get_params_123(mat)
    c_elastic = p.c_plane.copy()

    if sig is None or deps is None:
        return c_elastic

    sig0 = np.asarray(sig, dtype=float).flatten()
    deps0 = np.asarray(deps, dtype=float).flatten()
    if len(sig0) < 3:
        sig0 = np.pad(sig0, (0, 3 - len(sig0)))
    if len(deps0) < 3:
        deps0 = np.pad(deps0, (0, 3 - len(deps0)))

    h = 1.0e-7
    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    s_base, _, _ = shell_step(mat, sig0, deps0, dt=dt, extra=ex0)

    c_algo = np.zeros((3, 3), dtype=float)
    for j in range(3):
        d_p = deps0.copy()
        d_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        s_p, _, _ = shell_step(mat, sig0, d_p, dt=dt, extra=ex_p)
        c_algo[:, j] = (s_p - s_base) / h

    return c_algo


def consistent_shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    return shell_tangent(mat, sig=sig, deps=deps, dt=dt, extra=extra, **kwargs)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Per-element/per-layer persistent state array shapes required by LAW123/LAW132."""
    if nip:
        return {
            "uvar": (nip, 22),
            "dmg": (nip, 8),
            "off": (nip,),
            "offl": (nip,),
            "epst": (nip, 6),
        }
    return {
        "uvar": (22,),
        "dmg": (8,),
        "off": (),
        "offl": (),
        "epst": (6,),
    }


def _register() -> None:
    """Register LAW123 and LAW132 in pyradioss MAT_PHYSICS_REGISTRY."""
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (
            123, "123", "LAW123", "DAIMLER_PINHO", "DAIMLER-PINHO",
            "LAMINATED_FRACTURE_DAIMLER_PINHO", "LAW123_DAIMLER_PINHO", "MAT_LAW123"
        ):
            MAT_PHYSICS_REGISTRY[key] = build_law123
        for key in (
            132, "132", "LAW132", "DAIMLER_CAMANHO", "DAIMLER-CAMANHO",
            "LAMINATED_FRACTURE_DAIMLER_CAMANHO", "LAW132_DAIMLER_CAMANHO", "MAT_LAW132"
        ):
            MAT_PHYSICS_REGISTRY[key] = build_law132
    except Exception:
        pass


_register()
