"""
LAW95 — Bergstrom-Boyce visco-hyperelastic polymer model (/MAT/LAW95, /MAT/BERGSTROM_BOYCE).

Implements the finite strain Bergstrom-Boyce constitutive law for 3D solid continuum
elements in pure Python / NumPy, matching OpenRadioss upstream.

Fortran reference sources:
- engine/source/materials/mat/mat095/sigeps95.F      (3D continuum solid kernel)
- starter/source/materials/mat/mat095/hm_read_mat95.F (starter reader & parameters)
- starter/source/materials/mat/mat095/m95init.F       (history variable initialization)
- engine/source/materials/mat/mat100/viscbb.F        (nonlinear creep evolution)
- engine/source/materials/mat/mat100/calcmatb.F      (multiplicative kinematic split)
- engine/source/materials/mat/mat100/sigpoly.F       (polynomial potential & directional stiffness)
- hm_cfg_files/config/CFG/radioss2024/MAT/LAW95.cfg   (card layout definitions)

Theory
------
The Bergstrom-Boyce model decomposes the mechanical response into two parallel networks:
    sigma = sigma_A + sigma_B

1. Network A (Hyperelastic equilibrium network):
    W_A = sum_{i+j=1,2,3} C_ij * (I1_bar - 3)^i * (I2_bar - 3)^j + W_vol(J)
    b_A = F * F^T

2. Network B (Viscoelastic time-dependent network):
    F = F_e * F_p  =>  F_e = F * F_p^(-1)
    b_e = F_e * F_e^T
    sigma_B_trial = Sb * sigma(b_e)
    s_B = dev(sigma_B_trial)
    tau_B = ||s_B||_F = sqrt(s_B : s_B)

    Inelastic creep rate:
    lambda_p = sqrt((F_p_11^2 + F_p_22^2 + F_p_33^2) / 3)
    dgamma = A * dt * (lambda_p - 1 + xi)^C * (tau_B / tau_ref)^M
    L_B = (dgamma / tau_B) * s_B
    D_Fp = F_e^(-1) * L_B * F_e
    F_p^(n+1) = (I + D_Fp) * F_p^n

Linear ground-state elastic properties:
    G_0 = 2 * (C10 + C01) * (1 + Sb)
    If D1_raw > 0:
        d1 = 1 / D1_raw
        K = 2 * d1 * (1 + Sb)
        nu = (3*K - 2*G_0) / (2 * (3*K + G_0))
        E = 9*K*G_0 / (3*K + G_0)
    Else (or if nu > 0):
        nu = nu_input if nu_input > 0 else 0.495
        K = (2/3) * G_0 * (1 + nu) / (1 - 2*nu)
        d1 = K / 2
        E = 2 * G_0 * (1 + nu)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple
import numpy as np

_EM10 = 1e-10
_EM20 = 1e-20
_EM30 = 1e-30


@dataclass
class BergstromBoyceParams:
    """Parameters for /MAT/LAW95 (/MAT/BERGSTROM_BOYCE).

    Attributes
    ----------
    id : int
        Material ID.
    title : str
        Material title.
    rho0 : float
        Initial density.
    ref_rho : float
        Reference density.
    c10, c01, c20, c11, c02, c30, c21, c12, c03 : float
        Coefficients of the polynomial strain energy potential.
    sb : float
        Stress scaling factor for secondary network B.
    d1_raw, d2_raw, d3_raw : float
        Input compressibility parameters.
    d1, d2, d3 : float
        Stored inverted compressibility parameters (1/D1, 1/D2, 1/D3).
    nu : float
        Poisson's ratio.
    iform : int
        Flag for strain energy density formulation (1: standard, 2: modified).
    a : float
        Effective creep strain rate coefficient A.
    expc : float
        Creep strain exponent C (-1 < C < 0, default -0.7).
    expm : float
        Effective stress exponent M (M >= 1.0, default 1.0).
    ksi : float
        Regularization constant xi near undeformed state (default 0.01).
    tauref : float
        Reference stress for Network B (default 1.0).
    g0 : float
        Initial ground-state shear modulus G_0.
    rbulk : float
        Initial bulk modulus K.
    e : float
        Equivalent Young's modulus E.
    """

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    ref_rho: float = 0.0
    c10: float = 0.0
    c01: float = 0.0
    c20: float = 0.0
    c11: float = 0.0
    c02: float = 0.0
    c30: float = 0.0
    c21: float = 0.0
    c12: float = 0.0
    c03: float = 0.0
    sb: float = 0.0
    d1_raw: float = 0.0
    d2_raw: float = 0.0
    d3_raw: float = 0.0
    d1: float = 0.0
    d2: float = 0.0
    d3: float = 0.0
    nu: float = 0.495
    iform: int = 1
    a: float = 0.0
    expc: float = -0.7
    expm: float = 1.0
    ksi: float = 0.01
    tauref: float = 1.0
    g0: float = 0.0
    rbulk: float = 0.0
    e: float = 0.0

    @property
    def K(self) -> float:
        """Bulk modulus alias."""
        return self.rbulk

    @property
    def bulk(self) -> float:
        """Bulk modulus alias."""
        return self.rbulk

    @property
    def G(self) -> float:
        """Initial shear modulus alias."""
        return self.g0

    @property
    def G0(self) -> float:
        """Initial shear modulus alias."""
        return self.g0

    @property
    def E(self) -> float:
        """Equivalent Young's modulus."""
        return self.e

    @property
    def young(self) -> float:
        """Equivalent Young's modulus alias."""
        return self.e

    @property
    def rho(self) -> float:
        """Density alias."""
        return self.ref_rho if self.ref_rho > 0.0 else self.rho0

    def as_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "rho0": self.rho0,
            "ref_rho": self.ref_rho,
            "c10": self.c10,
            "c01": self.c01,
            "c20": self.c20,
            "c11": self.c11,
            "c02": self.c02,
            "c30": self.c30,
            "c21": self.c21,
            "c12": self.c12,
            "c03": self.c03,
            "sb": self.sb,
            "d1": self.d1,
            "d2": self.d2,
            "d3": self.d3,
            "nu": self.nu,
            "iform": self.iform,
            "a": self.a,
            "expc": self.expc,
            "expm": self.expm,
            "ksi": self.ksi,
            "tauref": self.tauref,
            "g0": self.g0,
            "rbulk": self.rbulk,
            "e": self.e,
        }


def build_law95(
    id: Any = 1,
    title: str = "",
    rho0: float = 1.0,
    ref_rho: float = 0.0,
    c10: float = 0.0,
    c01: float = 0.0,
    c20: float = 0.0,
    c11: float = 0.0,
    c02: float = 0.0,
    c30: float = 0.0,
    c21: float = 0.0,
    c12: float = 0.0,
    c03: float = 0.0,
    sb: float = 0.0,
    d1: float = 0.0,
    d2: float = 0.0,
    d3: float = 0.0,
    nu: float = 0.0,
    iform: int = 1,
    a: float = 0.0,
    expc: float = -0.7,
    expm: float = 1.0,
    ksi: float = 0.01,
    tauref: float = 1.0,
    **kwargs: Any,
) -> BergstromBoyceParams:
    """Build and initialize BergstromBoyceParams according to hm_read_mat95.F."""
    if isinstance(id, dict):
        rec = id
        mid = int(rec.get("id", 1))
        rho0 = float(rec.get("density", rec.get("rho0", rec.get("rho", 1.0))) or 1.0)
        title = str(rec.get("title", ""))
        p = rec.get("params", rec)
        c10 = float(p.get("MAT_C_10", p.get("c10", 0.0)))
        c01 = float(p.get("MAT_C_01", p.get("c01", 0.0)))
        c20 = float(p.get("MAT_C_20", p.get("c20", 0.0)))
        c11 = float(p.get("MAT_C_11", p.get("c11", 0.0)))
        c02 = float(p.get("MAT_C_02", p.get("c02", 0.0)))
        c30 = float(p.get("MAT_C_30", p.get("c30", 0.0)))
        c21 = float(p.get("MAT_C_21", p.get("c21", 0.0)))
        c12 = float(p.get("MAT_C_12", p.get("c12", 0.0)))
        c03 = float(p.get("MAT_C_03", p.get("c03", 0.0)))
        sb = float(p.get("MAT_Sb", p.get("sb", 0.0)))
        d1 = float(p.get("MAT_D_1", p.get("d1", 0.0)))
        d2 = float(p.get("MAT_D_2", p.get("d2", 0.0)))
        d3 = float(p.get("MAT_D_3", p.get("d3", 0.0)))
        nu = float(p.get("MAT_NU", p.get("nu", p.get("nu_val", 0.0))))
        iform = int(p.get("IFORM", p.get("iform", 1)))
        a = float(p.get("MLAW95_A", p.get("a", 0.0)))
        expc = float(p.get("MLAW95_C", p.get("expc", p.get("c", -0.7))))
        expm = float(p.get("MLAW95_M", p.get("expm", p.get("m", 1.0))))
        ksi = float(p.get("MLAW95_KSI", p.get("ksi", 0.01)))
        tauref = float(p.get("MAT_TAU_REF", p.get("tauref", p.get("tau_ref", 1.0))))
        ref_rho = float(p.get("refer_rho", p.get("ref_rho", 0.0)))
        id = mid
    elif not isinstance(id, (int, np.integer, float)):
        rec = id
        mid = int(getattr(rec, "id", 1))
        rho0 = float(getattr(rec, "density", getattr(rec, "rho0", getattr(rec, "rho", 1.0))) or 1.0)
        title = str(getattr(rec, "title", ""))
        p = getattr(rec, "params", {})
        if p and isinstance(p, dict):
            c10 = float(p.get("MAT_C_10", p.get("c10", getattr(rec, "c10", 0.0))))
            c01 = float(p.get("MAT_C_01", p.get("c01", getattr(rec, "c01", 0.0))))
            c20 = float(p.get("MAT_C_20", p.get("c20", getattr(rec, "c20", 0.0))))
            c11 = float(p.get("MAT_C_11", p.get("c11", getattr(rec, "c11", 0.0))))
            c02 = float(p.get("MAT_C_02", p.get("c02", getattr(rec, "c02", 0.0))))
            c30 = float(p.get("MAT_C_30", p.get("c30", getattr(rec, "c30", 0.0))))
            c21 = float(p.get("MAT_C_21", p.get("c21", getattr(rec, "c21", 0.0))))
            c12 = float(p.get("MAT_C_12", p.get("c12", getattr(rec, "c12", 0.0))))
            c03 = float(p.get("MAT_C_03", p.get("c03", getattr(rec, "c03", 0.0))))
            sb = float(p.get("MAT_Sb", p.get("sb", getattr(rec, "sb", 0.0))))
            d1 = float(p.get("MAT_D_1", p.get("d1", getattr(rec, "d1", 0.0))))
            d2 = float(p.get("MAT_D_2", p.get("d2", getattr(rec, "d2", 0.0))))
            d3 = float(p.get("MAT_D_3", p.get("d3", getattr(rec, "d3", 0.0))))
            nu = float(p.get("MAT_NU", p.get("nu", getattr(rec, "nu", 0.0))))
            iform = int(p.get("IFORM", p.get("iform", getattr(rec, "iform", 1))))
            a = float(p.get("MLAW95_A", p.get("a", getattr(rec, "a", 0.0))))
            expc = float(p.get("MLAW95_C", p.get("expc", getattr(rec, "expc", -0.7))))
            expm = float(p.get("MLAW95_M", p.get("expm", getattr(rec, "expm", 1.0))))
            ksi = float(p.get("MLAW95_KSI", p.get("ksi", getattr(rec, "ksi", 0.01))))
            tauref = float(p.get("MAT_TAU_REF", p.get("tauref", getattr(rec, "tauref", 1.0))))
            ref_rho = float(p.get("refer_rho", getattr(rec, "ref_rho", 0.0)))
        else:
            c10 = float(getattr(rec, "c10", 0.0))
            c01 = float(getattr(rec, "c01", 0.0))
            c20 = float(getattr(rec, "c20", 0.0))
            c11 = float(getattr(rec, "c11", 0.0))
            c02 = float(getattr(rec, "c02", 0.0))
            c30 = float(getattr(rec, "c30", 0.0))
            c21 = float(getattr(rec, "c21", 0.0))
            c12 = float(getattr(rec, "c12", 0.0))
            c03 = float(getattr(rec, "c03", 0.0))
            sb = float(getattr(rec, "sb", 0.0))
            d1 = float(getattr(rec, "d1", 0.0))
            d2 = float(getattr(rec, "d2", 0.0))
            d3 = float(getattr(rec, "d3", 0.0))
            nu = float(getattr(rec, "nu", 0.0))
            iform = int(getattr(rec, "iform", 1))
            a = float(getattr(rec, "a", 0.0))
            expc = float(getattr(rec, "expc", -0.7))
            expm = float(getattr(rec, "expm", 1.0))
            ksi = float(getattr(rec, "ksi", 0.01))
            tauref = float(getattr(rec, "tauref", getattr(rec, "tau_ref", 1.0)))
            ref_rho = float(getattr(rec, "ref_rho", 0.0))
        id = mid

    for key, val in kwargs.items():
        k = key.lower()
        if k in ("c10", "mat_c_10"):
            c10 = float(val)
        elif k in ("c01", "mat_c_01"):
            c01 = float(val)
        elif k in ("c20", "mat_c_20"):
            c20 = float(val)
        elif k in ("c11", "mat_c_11"):
            c11 = float(val)
        elif k in ("c02", "mat_c_02"):
            c02 = float(val)
        elif k in ("c30", "mat_c_30"):
            c30 = float(val)
        elif k in ("c21", "mat_c_21"):
            c21 = float(val)
        elif k in ("c12", "mat_c_12"):
            c12 = float(val)
        elif k in ("c03", "mat_c_03"):
            c03 = float(val)
        elif k in ("sb", "mat_sb"):
            sb = float(val)
        elif k in ("d1", "mat_d_1"):
            d1 = float(val)
        elif k in ("d2", "mat_d_2"):
            d2 = float(val)
        elif k in ("d3", "mat_d_3"):
            d3 = float(val)
        elif k in ("nu", "mat_nu", "nu_val"):
            nu = float(val)
        elif k in ("iform",):
            iform = int(val)
        elif k in ("a", "mlaw95_a"):
            a = float(val)
        elif k in ("expc", "c", "mlaw95_c"):
            expc = float(val)
        elif k in ("expm", "m", "mlaw95_m"):
            expm = float(val)
        elif k in ("ksi", "mlaw95_ksi"):
            ksi = float(val)
        elif k in ("tauref", "tau_ref", "mat_tau_ref"):
            tauref = float(val)
        elif k in ("rho", "rho0", "mat_rho"):
            rho0 = float(val)
        elif k in ("ref_rho", "refer_rho"):
            ref_rho = float(val)

    # Defaults matching hm_read_mat95.F:167-173, 200-202
    d1_raw = float(d1)
    d2_raw = float(d2)
    d3_raw = float(d3)

    d2_inv = (1.0 / d2_raw) if d2_raw != 0.0 else 0.0
    d3_inv = (1.0 / d3_raw) if d3_raw != 0.0 else 0.0

    if iform == 0:
        iform = 1
    if expm == 0.0:
        expm = 1.0
    if expc == 0.0:
        expc = -0.7
    if ksi == 0.0:
        ksi = 0.01
    if tauref == 0.0:
        tauref = 1.0

    # Linear elasticity & compressibility derivation (hm_read_mat95.F:176-195)
    g0 = 2.0 * (c10 + c01) * (sb + 1.0)
    if d1_raw != 0.0:
        d1_inv = 1.0 / d1_raw
        rbulk = 2.0 * d1_inv * (1.0 + sb)
        nu_calc = (3.0 * rbulk - 2.0 * g0) / max(_EM30, (2.0 * (3.0 * rbulk + g0)))
        e_calc = 9.0 * rbulk * g0 / max(_EM30, (3.0 * rbulk + g0))
    elif nu != 0.0:
        d2_inv = 0.0
        d3_inv = 0.0
        nu_calc = float(nu)
        e_calc = 2.0 * g0 * (1.0 + nu_calc)
        rbulk = (2.0 / 3.0) * g0 * (1.0 + nu_calc) / max(_EM30, (1.0 - 2.0 * nu_calc))
        d1_inv = rbulk / 2.0
    else:
        d2_inv = 0.0
        d3_inv = 0.0
        nu_calc = 0.495
        rbulk = (2.0 / 3.0) * g0 * (1.0 + nu_calc) / max(_EM30, (1.0 - 2.0 * nu_calc))
        d1_inv = rbulk / 2.0
        e_calc = 2.0 * g0 * (1.0 + nu_calc)

    return BergstromBoyceParams(
        id=int(id),
        title=str(title),
        rho0=float(rho0),
        ref_rho=float(ref_rho),
        c10=float(c10),
        c01=float(c01),
        c20=float(c20),
        c11=float(c11),
        c02=float(c02),
        c30=float(c30),
        c21=float(c21),
        c12=float(c12),
        c03=float(c03),
        sb=float(sb),
        d1_raw=d1_raw,
        d2_raw=d2_raw,
        d3_raw=d3_raw,
        d1=d1_inv,
        d2=d2_inv,
        d3=d3_inv,
        nu=float(nu_calc),
        iform=int(iform),
        a=float(a),
        expc=float(expc),
        expm=float(expm),
        ksi=float(ksi),
        tauref=float(tauref),
        g0=float(g0),
        rbulk=float(rbulk),
        e=float(e_calc),
    )


# ============================================================================
# Core Physics Subroutines
# ============================================================================

def poly_stress(
    b: np.ndarray,
    c_coeffs: tuple[float, ...],
    d_params: tuple[float, float, float],
    rbulk: float,
    iform: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Cauchy stress and directional stiffness for polynomial hyperelastic potential.

    Mirrors POLYSTREST2 and POLYSTRESS2 in engine/source/materials/mat/mat100/sigpoly.F.

    Parameters
    ----------
    b : ndarray of shape (3, 3)
        Left Cauchy-Green tensor b = F * F^T or b_e = F_e * F_e^T.
    c_coeffs : tuple of 9 floats
        (C10, C01, C20, C11, C02, C30, C21, C12, C03).
    d_params : tuple of 3 floats
        Stored inverse parameters (D1, D2, D3).
    rbulk : float
        Bulk modulus K.
    iform : int
        Volumetric formulation flag (1: polynomial, 2: modified logarithmic).

    Returns
    -------
    sig : ndarray of shape (3, 3)
        Cauchy stress tensor.
    cii : ndarray of shape (3,)
        Principal directional tangent stiffness terms.
    """
    c10, c01, c20, c11, c02, c30, c21, c12, c03 = c_coeffs
    d1, d2, d3 = d_params

    # b^2 (sigpoly.F:80)
    b2 = b @ b

    # Relative volume J = det(F) = sqrt(det(b)) (sigpoly.F:83-86)
    det_b = (
        b[0, 0] * (b[1, 1] * b[2, 2] - b[1, 2] * b[2, 1])
        - b[0, 1] * (b[1, 0] * b[2, 2] - b[1, 2] * b[2, 0])
        + b[0, 2] * (b[1, 0] * b[2, 1] - b[1, 1] * b[2, 0])
    )
    jdet = np.sqrt(max(_EM20, det_b))

    # Invariants (sigpoly.F:88-94)
    i1 = b[0, 0] + b[1, 1] + b[2, 2]
    tr_b2 = b2[0, 0] + b2[1, 1] + b2[2, 2]
    i2 = 0.5 * (i1 * i1 - tr_b2)

    if jdet > 0.0:
        jthird = np.exp((-1.0 / 3.0) * np.log(jdet))
        j2third = jthird * jthird
        j4third = j2third * j2third
    else:
        j2third = 0.0
        j4third = 0.0

    # Isochoric invariants (sigpoly.F:109-111)
    bi1 = i1 * j2third
    bi2 = i2 * j4third

    # Derivatives of isochoric potential (sigpoly.F:137-144)
    di1 = bi1 - 3.0
    di2 = bi2 - 3.0

    dphi_di1 = (
        c10
        + 2.0 * c20 * di1
        + 3.0 * c30 * (di1 ** 2)
        + c11 * di2
        + c12 * (di2 ** 2)
        + 2.0 * c21 * di1 * di2
    )
    dphi_di2 = (
        c01
        + 2.0 * c02 * di2
        + 3.0 * c03 * (di2 ** 2)
        + c11 * di1
        + c21 * (di1 ** 2)
        + 2.0 * c12 * di1 * di2
    )

    inv2j = 2.0 / max(_EM20, jdet)

    # Volumetric derivative dPhi/dJ (sigpoly.F:148-155)
    j_minus_1 = jdet - 1.0
    if iform == 1:
        dphi_dj = 2.0 * d1 * j_minus_1 + 4.0 * d2 * (j_minus_1 ** 3) + 6.0 * d3 * (j_minus_1 ** 5)
        dphi2_dj = 2.0 * d1 + 12.0 * d2 * (j_minus_1 ** 2) + 30.0 * d3 * (j_minus_1 ** 4)
    else:
        dphi_dj = rbulk * (1.0 - 1.0 / jdet)
        dphi2_dj = rbulk / (jdet ** 2)

    # Cauchy stress tensor (sigpoly.F:160-185)
    aa = (dphi_di1 + dphi_di2 * bi1) * inv2j * j2third
    bb = dphi_di2 * inv2j * j4third
    cc = (1.0 / 3.0) * inv2j * (bi1 * dphi_di1 + 2.0 * bi2 * dphi_di2)

    sig = aa * b - bb * b2
    sig[0, 0] += -cc + dphi_dj
    sig[1, 1] += -cc + dphi_dj
    sig[2, 2] += -cc + dphi_dj

    # Directional stiffness CII for Courant sound speed (sigpoly.F:360-373)
    dphi2_di1 = 2.0 * (c20 + 3.0 * c30 * di1 + c21 * di2)
    dphi2_di2 = 2.0 * (c02 + 3.0 * c03 * di2 + c12 * di1)

    lam_b = np.array([b[0, 0], b[1, 1], b[2, 2]], dtype=np.float64) * j2third
    lam_b_safe = np.maximum(lam_b, 1e-12)
    lam_b_inv = 1.0 / lam_b_safe

    bi1_3 = bi1 / 3.0
    bi2_3 = bi2 / 3.0

    term1 = (2.0 / 3.0) * dphi_di1 * (lam_b + bi1_3) + dphi2_di1 * (lam_b - bi1_3)
    term2 = (2.0 / 3.0) * dphi_di2 * (lam_b_inv + bi2_3) + dphi2_di2 * (lam_b_inv - bi2_3)
    cii = 2.0 * (term1 + term2) + dphi2_dj

    return sig, cii


def visc_bb(
    fp: np.ndarray,
    tbnorm: float,
    a1: float,
    expc: float,
    expm: float,
    ksi: float,
    tauref: float,
) -> float:
    """Compute effective creep strain increment dgamma.

    Mirrors engine/source/materials/mat/mat100/viscbb.F.

    Parameters
    ----------
    fp : ndarray of shape (3, 3)
        Inelastic deformation gradient F_p.
    tbnorm : float
        Effective Frobenius stress norm of Network B deviatoric stress.
    a1 : float
        Effective rate factor A * dt.
    expc : float
        Exponent C.
    expm : float
        Exponent M.
    ksi : float
        Regularization parameter xi.
    tauref : float
        Reference stress tau_ref.

    Returns
    -------
    dgamma : float
        Inelastic creep strain increment.
    """
    ip1 = fp[0, 0] ** 2 + fp[1, 1] ** 2 + fp[2, 2] ** 2
    lpchain = np.sqrt(max(0.0, ip1 / 3.0))
    temp = max(_EM20, lpchain - 1.0 + ksi)
    stress_ratio = (tbnorm ** expm) / max(_EM20, tauref ** expm)
    dgamma = a1 * np.exp(expc * np.log(temp)) * stress_ratio
    return float(dgamma)


def calc_mat_b(f: np.ndarray, fp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute elastic left Cauchy-Green tensor b_e and F_e = F * F_p^(-1).

    Mirrors engine/source/materials/mat/mat100/calcmatb.F.
    """
    inv_fp = np.linalg.inv(fp)
    fe = f @ inv_fp
    matb = fe @ fe.T
    return matb, fe


# ============================================================================
# 3D Continuum Solid Kernel: solid_update
# ============================================================================

def solid_update(
    mat: Any = None,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    ismstr: int = 0,
    *,
    epsp: np.ndarray | None = None,
    return_tuple: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, dict, float]:
    """3D solid continuum Bergstrom-Boyce visco-hyperelastic stress update.

    Fortran reference: engine/source/materials/mat/mat095/sigeps95.F.

    Parameters
    ----------
    mat : BergstromBoyceParams or Material or dict
        Material definition.
    sig : ndarray of shape (6,) or (n, 6)
        Current Cauchy stress (xx, yy, zz, xy, yz, zx).
    deps : ndarray of shape (6,) or (n, 6)
        Strain increment tensor (engineering shear: 2*eps_ij).
    eps : ndarray of shape (6,) or (n, 6), optional
        Total strain tensor.
    dt : float
        Time step increment.
    extra : dict, optional
        Extra state storage containing:
        - "uvar": array of shape (n, 10) or (10,) storing F_p (components 0..8) and dgamma (component 9).
        - "F": optional deformation gradient array of shape (n, 3, 3) or (3, 3).
        - "rho": optional current density array.
    ismstr : int
        Strain formulation flag (0: log, 1: eng).
    epsp : ndarray, optional
        Accumulated plastic / creep strain.

    Returns
    -------
    sig_out, epsp_out, sound_speed (or dict / tuple depending on call mode).
    """
    return_constitutive_dict = False
    if isinstance(mat, np.ndarray) and (sig is None or not isinstance(sig, np.ndarray)):
        # Direct constitutive call: solid_update(strain_tensor, c10=..., ...)
        eps_in = mat
        params_kw = kwargs.copy()
        if isinstance(sig, dict):
            params_kw.update(sig)
        mat = build_law95(**params_kw)
        sig_arr = np.zeros_like(eps_in) if eps_in.ndim > 1 else np.zeros((1, 6))
        deps_arr = np.zeros_like(sig_arr)
        eps_arr = eps_in[np.newaxis, :] if eps_in.ndim == 1 else eps_in
        epsp_arr = np.zeros(len(eps_arr))
        is_1d = (eps_in.ndim == 1)
        return_constitutive_dict = True
        if "rho" in kwargs and extra is None:
            extra = {"rho": kwargs["rho"]}
    else:
        if not isinstance(mat, BergstromBoyceParams):
            mat = build_law95(mat, **kwargs)
        if sig is None:
            sig = np.zeros(6, dtype=np.float64)
        is_1d = (sig.ndim == 1)
        if is_1d:
            sig_arr = sig[np.newaxis, :]
            deps_arr = deps[np.newaxis, :] if deps is not None else np.zeros((1, 6))
            eps_arr = eps[np.newaxis, :] if eps is not None else np.zeros((1, 6))
            epsp_arr = epsp[np.newaxis] if epsp is not None else np.zeros(1)
        else:
            sig_arr = sig
            deps_arr = deps if deps is not None else np.zeros_like(sig)
            eps_arr = eps if eps is not None else np.zeros_like(sig)
            epsp_arr = epsp if epsp is not None else np.zeros(len(sig))

    nel = len(sig_arr)
    sig_out = np.zeros_like(sig_arr)
    sound_sp = np.zeros(nel, dtype=np.float64)
    epsp_out = epsp_arr.copy()

    # Total strain tensor
    tot_eps = eps_arr + deps_arr if deps is not None else eps_arr.copy()

    # Material coefficients
    c_coeffs = (
        mat.c10, mat.c01, mat.c20, mat.c11, mat.c02,
        mat.c30, mat.c21, mat.c12, mat.c03
    )
    d_params = (mat.d1, mat.d2, mat.d3)
    sb = mat.sb
    a10 = mat.a
    a1 = a10 * dt
    expc = mat.expc
    expm = mat.expm
    ksi = mat.ksi
    tauref = mat.tauref
    rbulk = mat.rbulk
    g0 = mat.g0
    iform = mat.iform
    stiff0 = (4.0 / 3.0) * g0 + rbulk

    iavis = 1 if (a10 * sb > 0.0) else 0

    # Density handling
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        rho_val = np.asarray(extra["rho"])
        rho_arr = np.full(nel, float(rho_val)) if rho_val.ndim == 0 else rho_val
    else:
        rho_arr = np.full(nel, mat.rho0)

    # Persistent history retrieval / allocation
    # History UVAR (sigeps95.F:171-180):
    # uvar[0]=Fp11, uvar[1]=Fp22, uvar[2]=Fp33, uvar[3]=Fp12, uvar[4]=Fp23,
    # uvar[5]=Fp31, uvar[6]=Fp21, uvar[7]=Fp32, uvar[8]=Fp13, uvar[9]=dgamma
    uvar_hist = None
    if extra is not None:
        for uvar_key in ("uvar", "uvar95", "mat_uvar"):
            if uvar_key in extra and extra[uvar_key] is not None:
                uvar_hist = extra[uvar_key]
                break

    if uvar_hist is None:
        uvar_hist = np.zeros((nel, 10), dtype=np.float64)
        uvar_hist[:, 0] = 1.0
        uvar_hist[:, 1] = 1.0
        uvar_hist[:, 2] = 1.0
    else:
        if uvar_hist.ndim == 1:
            uvar_hist = uvar_hist[np.newaxis, :]
        # Check if uninitialized (all zeros on diagonal)
        if np.all(uvar_hist[:, :3] == 0.0):
            uvar_hist[:, 0] = 1.0
            uvar_hist[:, 1] = 1.0
            uvar_hist[:, 2] = 1.0

    # Deformation gradient F
    f_input = None
    if extra is not None and "F" in extra and extra["F"] is not None:
        f_val = np.asarray(extra["F"])
        f_input = f_val[np.newaxis, :, :] if f_val.ndim == 2 else f_val

    for i in range(nel):
        cur_rho = max(float(rho_arr[i]), 1e-12)

        # 1. Total deformation gradient F
        if f_input is not None:
            f = f_input[i].copy()
        else:
            # Reconstruct F from total strain tensor via exponential / matrix form
            exx = tot_eps[i, 0]
            eyy = tot_eps[i, 1]
            ezz = tot_eps[i, 2]
            exy = 0.5 * tot_eps[i, 3]
            eyz = 0.5 * tot_eps[i, 4]
            ezx = 0.5 * tot_eps[i, 5]

            mat_eps = np.array([
                [exx, exy, ezx],
                [exy, eyy, eyz],
                [ezx, eyz, ezz]
            ], dtype=np.float64)

            evals, evecs = np.linalg.eigh(mat_eps)
            if ismstr in (0, 2, 4):
                stretches = np.exp(evals)
            else:
                stretches = evals + 1.0
            stretches = np.maximum(stretches, 1e-12)

            # Left stretch tensor V = sum lambda_k (n_k (x) n_k)
            # Assuming rotation R = I in local material frame: F = V
            diag_v = np.diag(stretches)
            f = evecs @ diag_v @ evecs.T

        # 2. Network A stress and stiffness (sigeps95.F:220-234)
        b_a = f @ f.T
        sig_a, c_a_ii = poly_stress(b_a, c_coeffs, d_params, rbulk, iform)

        # 3. Network B stress and stiffness (sigeps95.F:236-379)
        if iavis > 0 and dt > 0.0:
            # Retrieve previous F_p (sigeps95.F:171-180)
            fpo = np.array([
                [uvar_hist[i, 0], uvar_hist[i, 3], uvar_hist[i, 8]],
                [uvar_hist[i, 6], uvar_hist[i, 1], uvar_hist[i, 4]],
                [uvar_hist[i, 5], uvar_hist[i, 7], uvar_hist[i, 2]],
            ], dtype=np.float64)

            # Trial elastic kinematics: Fe = F * F_p^(-1), B_e = Fe * Fe^T (sigeps95.F:251-253)
            matb, fe = calc_mat_b(f, fpo)

            # Trial Cauchy stress in Network B (sigeps95.F:259-275)
            sig_b_raw, _ = poly_stress(matb, c_coeffs, d_params, rbulk, iform)
            sig_b = sb * sig_b_raw

            # Deviator of stress B (sigeps95.F:278-286)
            tr_b = (1.0 / 3.0) * (sig_b[0, 0] + sig_b[1, 1] + sig_b[2, 2])
            sb_dev = sig_b - tr_b * np.eye(3)

            # Frobenius stress norm (sigeps95.F:288-289)
            norm_sq = (
                sb_dev[0, 0] ** 2 + sb_dev[1, 1] ** 2 + sb_dev[2, 2] ** 2
                + 2.0 * (sb_dev[0, 1] ** 2 + sb_dev[1, 2] ** 2 + sb_dev[2, 0] ** 2)
            )
            tbnorm = np.sqrt(max(_EM20, norm_sq))

            # Effective creep strain rate (sigeps95.F:295-296)
            dgamma = visc_bb(fpo, tbnorm, a1, expc, expm, ksi, tauref)
            uvar_hist[i, 9] = dgamma
            epsp_out[i] += dgamma

            # Velocity gradient L_B = (dgamma / tbnorm) * s_B (sigeps95.F:302-312)
            factor = dgamma / max(_EM20, tbnorm)
            lb = factor * sb_dev

            # Inelastic gradient update: D_Fp = Fe^(-1) * L_B * Fe (sigeps95.F:318-320)
            inv_fe = np.linalg.inv(fe)
            fedp = lb @ fe
            dfp = inv_fe @ fedp

            # F_p^(n+1) = (I + D_Fp) * F_p^n (sigeps95.F:325-335)
            sn = np.eye(3) + dfp
            fp_new = sn @ fpo

            # Updated elastic left Cauchy-Green tensor and stress (sigeps95.F:336-348)
            matb_new, _ = calc_mat_b(f, fp_new)
            sig_b_new, c_b_ii = poly_stress(matb_new, c_coeffs, d_params, rbulk, iform)
            sig_b = sb * sig_b_new

            # Store updated F_p (sigeps95.F:363-371)
            uvar_hist[i, 0] = fp_new[0, 0]
            uvar_hist[i, 1] = fp_new[1, 1]
            uvar_hist[i, 2] = fp_new[2, 2]
            uvar_hist[i, 3] = fp_new[0, 1]
            uvar_hist[i, 4] = fp_new[1, 2]
            uvar_hist[i, 5] = fp_new[2, 0]
            uvar_hist[i, 6] = fp_new[1, 0]
            uvar_hist[i, 7] = fp_new[2, 1]
            uvar_hist[i, 8] = fp_new[0, 2]
        else:
            # A10 * SB == 0: Network B is elastic proportional to A (sigeps95.F:375-378)
            sig_b = sb * sig_a
            c_b_ii = c_a_ii.copy()

        # 4. Total Cauchy stress = sigma_A + sigma_B (sigeps95.F:384-391)
        sig_tot = sig_a + sig_b
        sig_out[i, 0] = sig_tot[0, 0]
        sig_out[i, 1] = sig_tot[1, 1]
        sig_out[i, 2] = sig_tot[2, 2]
        sig_out[i, 3] = sig_tot[0, 1]
        sig_out[i, 4] = sig_tot[1, 2]
        sig_out[i, 5] = sig_tot[2, 0]

        # 5. Directional stiffness and sound speed (sigeps95.F:395-414)
        c_ii_tot = c_a_ii + sb * c_b_ii
        c_max = max(stiff0, float(np.max(c_ii_tot)))
        sound_sp[i] = np.sqrt(c_max / cur_rho)

    if extra is not None:
        extra["uvar"] = uvar_hist[0] if is_1d else uvar_hist
        extra["uvar95"] = uvar_hist[0] if is_1d else uvar_hist

    sig_res = sig_out[0] if is_1d else sig_out
    epsp_res = epsp_out[0] if is_1d else epsp_out
    sound_res = sound_sp[0] if is_1d else sound_sp

    if return_constitutive_dict:
        hist_dict = {"dgamma": float(uvar_hist[0, 9]), "uvar": uvar_hist[0]}
        return sig_res, hist_dict, float(sound_res)

    if return_tuple:
        return sig_res, epsp_res, sound_res
    return sig_res, epsp_res


# ============================================================================
# Dynamic Wave Speeds
# ============================================================================

def sound_speed(mat: Any, rho: float | np.ndarray | None = None) -> float | np.ndarray:
    """Dilatational sound speed for LAW95 3D continuum solids.

    Mirrors engine/source/materials/mat/mat095/sigeps95.F:414:
        c = sqrt(C_max / rho) where C_max >= 4/3*G_0 + K
    """
    if not isinstance(mat, BergstromBoyceParams):
        mat = build_law95(mat)

    stiff0 = (4.0 / 3.0) * mat.g0 + mat.rbulk
    if rho is None or np.all(rho == 0.0):
        rho = mat.rho0

    if isinstance(rho, np.ndarray):
        safe_rho = np.maximum(rho, 1e-12)
        return np.sqrt(stiff0 / safe_rho)
    safe_rho = max(float(rho), 1e-12)
    return float(np.sqrt(stiff0 / safe_rho))


sound_speed_solid = sound_speed


# ============================================================================
# Algorithmic Consistent Tangent Tensor (6 x 6)
# ============================================================================

def consistent_tangent(
    mat: Any,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Compute (6, 6) solid algorithmic consistent tangent stiffness matrix.

    Evaluates d(sigma) / d(eps) using central difference perturbation matching
    the exact numerical response of solid_update.
    """
    if not isinstance(mat, BergstromBoyceParams):
        mat = build_law95(mat, **kwargs)

    if eps is None:
        eps = np.zeros(6, dtype=np.float64)
    else:
        eps = np.asarray(eps, dtype=np.float64).flatten()

    c_tangent = np.zeros((6, 6), dtype=np.float64)
    delta = 1e-7

    for j in range(6):
        eps_plus = eps.copy()
        eps_minus = eps.copy()
        eps_plus[j] += delta
        eps_minus[j] -= delta

        extra_p = extra.copy() if extra is not None else {}
        extra_m = extra.copy() if extra is not None else {}

        sig_plus, _, _ = solid_update(mat, None, None, eps=eps_plus, dt=dt, extra=extra_p)
        sig_minus, _, _ = solid_update(mat, None, None, eps=eps_minus, dt=dt, extra=extra_m)

        c_tangent[:, j] = (sig_plus - sig_minus) / (2.0 * delta)

    # Enforce minor symmetry
    return 0.5 * (c_tangent + c_tangent.T)


# ============================================================================
# Persistent State Shapes
# ============================================================================

def extra_shapes(mat: Any = None, nip: int | None = None) -> Dict[str, Tuple[int, ...]]:
    """Persistent state arrays required by LAW95 in element kernels.

    Mirrors starter/source/materials/mat/mat095/m95init.F:
        UVAR(NEL, 10):
        1..9: Components of F_p
        10: dgamma (effective creep strain increment)
    """
    return {
        "uvar": (10,),
        "uvar95": (10,),
    }


def resolve(mat: Any, model: Any = None, log: Any = None) -> BergstromBoyceParams:
    """Starter resolve hook: build and return BergstromBoyceParams."""
    return build_law95(mat)

