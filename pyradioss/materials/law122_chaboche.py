"""OpenRadioss /MAT/LAW122 — Modified Ladevèze Composite & Chaboche Hardening Model.

Elastoplastic-damage constitutive model integrating:
1. OpenRadioss MAT122 (Modified Ladevèze Delamination & Composite Damage Model):
   - Transverse isotropic plasticity in matrix and shear directions (A * (sig22^2 + sig33^2) + sig12^2 + sig23^2 + sig31^2)
   - Power-law hardening (sig_y = sigy0*(1+fr0) + beta * pla^m)
   - Cutting plane Newton return mapping (IRES=2) and NICE explicit scheme (IRES=1)
   - Fiber tensile/compressive damage (D_ft, D_fc) with optional buckling (IBUCK)
   - In-plane shear damage (D) and transverse matrix damage (D')
   - Directional strain-rate dependency laws
2. Classical Chaboche / Armstrong-Frederick Nonlinear Kinematic Hardening:
   - J2 elastoplasticity with backstress tensor alpha and Voce isotropic hardening
   - Algorithmic tangent operator for implicit/explicit solvers

Upstream Fortran references:
  - `engine/source/materials/mat/mat122/sigeps122.F` (lines 33-113: solid dispatcher)
  - `engine/source/materials/mat/mat122/sigeps122c.F` (lines 33-129: shell dispatcher)
  - `engine/source/materials/mat/mat122/mat122_newton.F` (lines 30-676: 3D Newton cutting plane)
  - `engine/source/materials/mat/mat122/mat122_nice.F` (lines 30-697: 3D NICE explicit scheme)
  - `engine/source/materials/mat/mat122/mat122c_newton.F` (lines 30-668: shell Newton cutting plane)
  - `engine/source/materials/mat/mat122/mat122c_nice.F` (lines 30-687: shell NICE explicit scheme)
  - `starter/source/materials/mat/mat122/hm_read_mat122.F` (lines 39-601: 15 cards reader)
  - `hm_cfg_files/config/CFG/radioss2023/MAT/matl122_modified_ladeveze.cfg`
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_EM20 = 1.0e-20
_EM10 = 1.0e-10


@dataclass
class Law122Params:
    """Parameters for OpenRadioss /MAT/LAW122 (Modified Ladevèze & Chaboche)."""
    id: int = 1
    title: str = ""
    rho0: float = 0.0
    refer_rho: float = 0.0
    rho: float = 0.0

    # Orthotropic elastic moduli (Card 2, 3 in hm_read_mat122.F)
    young1: float = 1.0      # E1: Longitudinal fiber Young modulus
    young2: float = 1.0      # E2: Transverse matrix Young modulus
    young3: float = 1.0      # E3: Out-of-plane Young modulus
    nu12: float = 0.3        # In-plane Poisson's ratio
    nu21: float = 0.3
    nu13: float = 0.3
    nu31: float = 0.3
    nu23: float = 0.3
    nu32: float = 0.3
    g12: float = 0.0         # In-plane shear modulus
    g23: float = 0.0         # Transverse shear modulus
    g31: float = 0.0         # Transverse shear modulus

    # Card 4: Compression elasticity & flags (mat122_newton.F lines 113-117)
    e1c: float = 0.0         # Longitudinal compression modulus
    gamma: float = 0.0       # Compressive non-linear parameter
    ish: int = 0             # Shear damage law type (1=linear, 2=exponential, 3=tabulated)
    itr: int = 0             # Transverse damage law type (1=linear, 2=exponential, 3=tabulated)
    ires: int = 2            # Return mapping method (1=NICE explicit, 2=Newton cutting-plane)

    # Card 5: Plasticity parameters (mat122_newton.F lines 118-121)
    sigy0: float = 1.0       # Initial yield stress
    beta: float = 0.0        # Hardening parameter beta
    hard_m: float = 0.0      # Hardening exponent m
    hard_a: float = 1.0      # Plastic eccentricity parameter A

    # Card 6 & 7: Fiber damage parameters (mat122_newton.F lines 122-128)
    eps_fti: float = 0.0     # Initial fiber tensile failure strain
    eps_ftu: float = 0.0     # Ultimate fiber tensile failure strain
    dftu: float = 0.0        # Fiber tensile ultimate damage
    eps_fci: float = 0.0     # Initial fiber compressive failure strain
    eps_fcu: float = 0.0     # Ultimate fiber compressive failure strain
    dfcu: float = 0.0        # Fiber compressive ultimate damage
    ibuck: int = 0           # Fiber buckling flag

    # Card 8, 9: Matrix shear damage (mat122_newton.F lines 129-135)
    ifuncd1: int = 0
    dsat1: float = 0.0       # Shear damage saturation
    y0: float = 0.0          # Initial shear damage threshold
    yc: float = 0.0          # Critical shear damage parameter
    b: float = 0.0           # Shear-transverse coupling parameter B
    dmax: float = 0.99       # Maximum allowable damage
    yr: float = 0.0          # Rupture energy density threshold
    ysp: float = 0.0         # Critical energy density

    # Card 10, 11: Matrix transverse damage (mat122_newton.F lines 136-141)
    ifuncd2: int = 0
    dsat2: float = 0.0       # Transverse tensile damage saturation
    y0p: float = 0.0         # Initial transverse tensile damage threshold
    ycp: float = 0.0         # Critical transverse tensile damage parameter
    ifuncd2c: int = 0
    dsat2c: float = 0.0      # Transverse compressive damage saturation
    y0pc: float = 0.0        # Initial transverse compressive damage threshold
    ycpc: float = 0.0        # Critical transverse compressive damage parameter

    # Card 12-14: Strain rate dependency (mat122_newton.F lines 142-157)
    epsd11: float = 0.0
    d11: float = 0.0
    n11: float = 0.0
    d11u: float = 0.0
    n11u: float = 0.0
    epsd12: float = 0.0
    d22: float = 0.0
    n22: float = 0.0
    d12: float = 0.0
    n12: float = 0.0
    epsdr0: float = 0.0
    dr0: float = 0.0
    nr0: float = 0.0
    ltype11: int = 0
    ltype12: int = 0
    ltyper0: int = 0
    fcut: float = 0.0

    # Chaboche kinematic hardening extensions (for cyclic plasticity)
    r_inf: float = 0.0       # Voce isotropic saturation increment
    b_iso: float = 0.0       # Voce isotropic hardening rate
    c_kin: float = 0.0       # Chaboche kinematic modulus C
    gamma_kin: float = 0.0   # Chaboche recall parameter gamma

    # Derived moduli
    bulk: float = field(init=False)
    lame: float = field(init=False)
    a11: float = field(init=False)
    a12: float = field(init=False)
    s11: float = field(init=False)
    s12: float = field(init=False)
    s13: float = field(init=False)
    s22: float = field(init=False)
    s23: float = field(init=False)
    s33: float = field(init=False)

    def __post_init__(self) -> None:
        if self.rho > 0.0 and self.rho0 <= 0.0:
            self.rho0 = self.rho
        if self.rho0 > 0.0 and self.rho <= 0.0:
            self.rho = self.rho0
        if self.refer_rho <= 0.0:
            self.refer_rho = self.rho0

        if self.young1 <= 0.0:
            self.young1 = 1.0
        if self.young2 <= 0.0:
            self.young2 = self.young1
        if self.young3 <= 0.0:
            self.young3 = self.young1

        if self.g12 <= 0.0:
            self.g12 = self.young1 / (2.0 * (1.0 + self.nu12))
        if self.g23 <= 0.0:
            self.g23 = self.g12
        if self.g31 <= 0.0:
            self.g31 = self.g12

        mean_young = (self.young1 + self.young2 + self.young3) / 3.0
        mean_nu = max(0.0, min(0.499, (self.nu12 + self.nu23 + self.nu31) / 3.0))

        self.bulk = mean_young / max(_EM20, 3.0 * (1.0 - 2.0 * mean_nu))
        self.lame = (mean_young * mean_nu) / max(_EM20, (1.0 + mean_nu) * (1.0 - 2.0 * mean_nu))

        denom = 1.0 - self.nu12 * self.nu21
        self.a11 = self.young1 / max(_EM20, denom)
        self.a12 = (self.young2 * self.nu12) / max(_EM20, denom)

        # 3D Orthotropic compliance and stiffness components (mat122_newton.F lines 261-276)
        c11 = 1.0 / max(self.young1, _EM20)
        c22 = 1.0 / max(self.young2, _EM20)
        c33 = 1.0 / max(self.young3, _EM20)
        c12 = -self.nu12 / max(self.young1, _EM20)
        c13 = -self.nu31 / max(self.young3, _EM20)
        c23 = -self.nu23 / max(self.young2, _EM20)

        detc = (
            c11 * c22 * c33
            - c11 * c23 * c23
            - c12 * c12 * c33
            + c12 * c13 * c23
            + c13 * c12 * c23
            - c13 * c22 * c13
        )
        if abs(detc) < _EM20:
            detc = _EM20

        self.s11 = (c22 * c33 - c23 * c23) / detc
        self.s12 = -(c12 * c33 - c13 * c23) / detc
        self.s13 = (c12 * c23 - c13 * c22) / detc
        self.s22 = (c11 * c33 - c13 * c13) / detc
        self.s23 = -(c11 * c23 - c13 * c12) / detc
        self.s33 = (c11 * c22 - c12 * c12) / detc

    @property
    def young(self) -> float:
        return self.young1

    @property
    def nu(self) -> float:
        return self.nu12

    @property
    def g(self) -> float:
        return self.g12

    @classmethod
    def from_material(cls, mat: Any) -> Law122Params:
        """Construct Law122Params from generic Material, MaterialLaw122, or dictionary."""
        if isinstance(mat, Law122Params):
            return mat

        def _get(keys: Sequence[str], default: Any) -> Any:
            for k in keys:
                if hasattr(mat, k):
                    v = getattr(mat, k)
                    if v is not None:
                        return v
                if hasattr(mat, "params") and isinstance(mat.params, dict) and k in mat.params:
                    v = mat.params[k]
                    if v is not None:
                        return v
                if isinstance(mat, dict) and k in mat:
                    v = mat[k]
                    if v is not None:
                        return v
            return default

        mid = int(_get(["id", "mid", "mat_id"], 1))
        title = str(_get(["title", "name"], ""))
        rho0 = float(_get(["rho0", "rho", "MAT_RHO"], 0.0))
        young1 = float(_get(["young1", "e1", "E10", "MAT_E1", "young", "e", "MAT_E"], 1.0))
        young2 = float(_get(["young2", "e2", "E20", "MAT_E2"], young1))
        young3 = float(_get(["young3", "e3", "E30", "MAT_E3"], young1))
        nu12 = float(_get(["nu12", "NU12", "MAT_NU12", "nu", "MAT_NU"], 0.3))
        nu21 = float(_get(["nu21", "NU21", "MAT_NU21"], nu12 * (young2 / max(_EM20, young1))))
        nu13 = float(_get(["nu13", "NU13", "MAT_NU13"], nu12))
        nu31 = float(_get(["nu31", "NU31", "MAT_NU31"], nu12))
        nu23 = float(_get(["nu23", "NU23", "MAT_NU23"], nu12))
        nu32 = float(_get(["nu32", "NU32", "MAT_NU32"], nu12))
        g12 = float(_get(["g12", "G120", "MAT_G12"], young1 / (2.0 * (1.0 + nu12))))
        g23 = float(_get(["g23", "G230", "MAT_G23"], g12))
        g31 = float(_get(["g31", "G310", "MAT_G31"], g12))

        # Compression elasticity & flags
        e1c = float(_get(["e1c", "E1C", "MAT_E1C"], 0.0))
        gamma = float(_get(["gamma", "GAMMA", "MAT_GAMMA"], 0.0))
        ish = int(_get(["ish", "ISH", "MAT_ISH"], 0))
        itr = int(_get(["itr", "ITR", "MAT_ITR"], 0))
        ires = int(_get(["ires", "IRES", "MAT_IRES"], 2))

        # Plasticity parameters
        sigy0 = float(_get(["sigy0", "SIGY0", "MAT_SIGY0", "sigy"], 1.0))
        beta = float(_get(["beta", "BETA", "MAT_BETA"], 0.0))
        hard_m = float(_get(["hard_m", "m", "M", "MAT_M"], 0.0))
        hard_a = float(_get(["hard_a", "a", "A", "MAT_A"], 1.0))

        # Fiber damage
        eps_fti = float(_get(["eps_fti", "EFTI", "MAT_EFTI"], 0.0))
        eps_ftu = float(_get(["eps_ftu", "EFTU", "MAT_EFTU"], 0.0))
        dftu = float(_get(["dftu", "DFTU", "MAT_DFTU"], 0.0))
        eps_fci = float(_get(["eps_fci", "EFCI", "MAT_EFCI"], 0.0))
        eps_fcu = float(_get(["eps_fcu", "EFCU", "MAT_EFCU"], 0.0))
        dfcu = float(_get(["dfcu", "DFCU", "MAT_DFCU"], 0.0))
        ibuck = int(_get(["ibuck", "IBUCK", "MAT_IBUCK"], 0))

        # Matrix shear damage
        ifuncd1 = int(_get(["ifuncd1", "IFUNCD1"], 0))
        dsat1 = float(_get(["dsat1", "DSAT1", "MAT_DSAT1"], 0.0))
        y0 = float(_get(["y0", "Y0", "MAT_Y0"], 0.0))
        yc = float(_get(["yc", "YC", "MAT_YC"], 0.0))
        b = float(_get(["b", "B", "MAT_B"], 0.0))
        dmax = float(_get(["dmax", "DMAX", "MAT_DMAX"], 0.99))
        yr = float(_get(["yr", "YR", "MAT_YR"], 0.0))
        ysp = float(_get(["ysp", "YSP", "MAT_YSP"], 0.0))

        # Matrix transverse damage
        ifuncd2 = int(_get(["ifuncd2", "IFUNCD2"], 0))
        dsat2 = float(_get(["dsat2", "DSAT2", "MAT_DSAT2"], 0.0))
        y0p = float(_get(["y0p", "Y0P", "MAT_Y0P"], 0.0))
        ycp = float(_get(["ycp", "YCP", "MAT_YCP"], 0.0))
        ifuncd2c = int(_get(["ifuncd2c", "IFUNCD2C"], 0))
        dsat2c = float(_get(["dsat2c", "DSAT2C", "MAT_DSAT2C"], 0.0))
        y0pc = float(_get(["y0pc", "Y0PC", "MAT_Y0PC"], 0.0))
        ycpc = float(_get(["ycpc", "YCPC", "MAT_YCPC"], 0.0))

        # Rate dependency
        epsd11 = float(_get(["epsd11", "EPSD11", "MAT_EPSD11"], 0.0))
        d11 = float(_get(["d11", "D11", "MAT_D11"], 0.0))
        n11 = float(_get(["n11", "N11", "MAT_N11"], 0.0))
        d11u = float(_get(["d11u", "D11U", "MAT_D11U"], 0.0))
        n11u = float(_get(["n11u", "N11U", "MAT_N11U"], 0.0))
        epsd12 = float(_get(["epsd12", "EPSD12", "MAT_EPSD12"], 0.0))
        d22 = float(_get(["d22", "D22", "MAT_D22"], 0.0))
        n22 = float(_get(["n22", "N22", "MAT_N22"], 0.0))
        d12 = float(_get(["d12", "D12", "MAT_D12"], 0.0))
        n12 = float(_get(["n12", "N12", "MAT_N12"], 0.0))
        epsdr0 = float(_get(["epsdr0", "EPSDR0", "MAT_EPSDR0"], 0.0))
        dr0 = float(_get(["dr0", "DR0", "MAT_DR0"], 0.0))
        nr0 = float(_get(["nr0", "NR0", "MAT_NR0"], 0.0))
        ltype11 = int(_get(["ltype11", "LTYPE11"], 0))
        ltype12 = int(_get(["ltype12", "LTYPE12"], 0))
        ltyper0 = int(_get(["ltyper0", "LTYPER0"], 0))
        fcut = float(_get(["fcut", "FCUT"], 0.0))

        # Chaboche cyclic plasticity aliases
        r_inf = float(_get(["r_inf", "R_INF", "MAT_R_INF"], dsat1 if dsat1 > 0.0 else 0.0))
        b_iso = float(_get(["b_iso", "B_ISO"], 0.0))
        c_kin = float(_get(["c_kin", "C_KIN", "MAT_C_KIN"], 0.0))
        gamma_kin = float(_get(["gamma_kin", "GAMMA_KIN", "MAT_GAMMA_KIN"], 0.0))

        return cls(
            id=mid,
            title=title,
            rho0=rho0,
            young1=young1,
            young2=young2,
            young3=young3,
            nu12=nu12,
            nu21=nu21,
            nu13=nu13,
            nu31=nu31,
            nu23=nu23,
            nu32=nu32,
            g12=g12,
            g23=g23,
            g31=g31,
            e1c=e1c,
            gamma=gamma,
            ish=ish,
            itr=itr,
            ires=ires,
            sigy0=sigy0,
            beta=beta,
            hard_m=hard_m,
            hard_a=hard_a,
            eps_fti=eps_fti,
            eps_ftu=eps_ftu,
            dftu=dftu,
            eps_fci=eps_fci,
            eps_fcu=eps_fcu,
            dfcu=dfcu,
            ibuck=ibuck,
            ifuncd1=ifuncd1,
            dsat1=dsat1,
            y0=y0,
            yc=yc,
            b=b,
            dmax=dmax,
            yr=yr,
            ysp=ysp,
            ifuncd2=ifuncd2,
            dsat2=dsat2,
            y0p=y0p,
            ycp=ycp,
            ifuncd2c=ifuncd2c,
            dsat2c=dsat2c,
            y0pc=y0pc,
            ycpc=ycpc,
            epsd11=epsd11,
            d11=d11,
            n11=n11,
            d11u=d11u,
            n11u=n11u,
            epsd12=epsd12,
            d22=d22,
            n22=n22,
            d12=d12,
            n12=n12,
            epsdr0=epsdr0,
            dr0=dr0,
            nr0=nr0,
            ltype11=ltype11,
            ltype12=ltype12,
            ltyper0=ltyper0,
            fcut=fcut,
            r_inf=r_inf,
            b_iso=b_iso,
            c_kin=c_kin,
            gamma_kin=gamma_kin,
        )


def build_law122(mat: Any = None, **kwargs: Any) -> Law122Params:
    """Construct Law122Params from material or keyword arguments."""
    if mat is not None:
        p = Law122Params.from_material(mat)
        for k, v in kwargs.items():
            if hasattr(p, k):
                setattr(p, k, v)
        p.__post_init__()
        return p
    valid_keys = {f.name for f in Law122Params.__dataclass_fields__.values() if f.init}
    init_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    extra_kwargs = {k: v for k, v in kwargs.items() if k not in valid_keys}
    p = Law122Params(**init_kwargs)
    for k, v in extra_kwargs.items():
        if hasattr(p, k):
            setattr(p, k, v)
    p.__post_init__()
    return p


def resolve(mat: Any, model: Any = None, log: Any = None) -> Law122Params:
    """Resolve references for /MAT/LAW122 and return Law122Params."""
    return build_law122(mat)


def extra_shapes(mat: Any = None, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Return extra history variable shapes for LAW122."""
    if nip is not None:
        return {
            "uvar122": (nip, 18),
            "backstress": (nip, 6),
            "damage": (nip, 6),
            "epsp": (nip,),
        }
    return {
        "uvar122": (18,),
        "backstress": (6,),
        "damage": (6,),
        "epsp": (),
    }


def needs_defgrad(mat: Any = None) -> bool:
    """LAW122 uses small strain rate plasticity formulation; defgrad is False."""
    return False


def _eval_chaboche_yield_stress(p: Law122Params, eps_p: float) -> Tuple[float, float]:
    """Evaluate isotropic hardening yield stress and slope R'(p)."""
    p_eff = max(0.0, eps_p)
    if p.b_iso > 0.0 and p.r_inf > 0.0:
        exp_term = math.exp(-p.b_iso * p_eff)
        r = p.r_inf * (1.0 - exp_term)
        dr = p.r_inf * p.b_iso * exp_term
    elif p.beta > 0.0 and p.hard_m > 0.0:
        r = p.beta * ((p_eff + _EM20) ** p.hard_m)
        dr = p.beta * p.hard_m * ((p_eff + _EM20) ** (p.hard_m - 1.0))
    else:
        r = 0.0
        dr = 0.0
    return p.sigy0 + r, dr


def mat122_newton_solid_update(
    p: Law122Params,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float = 0.0,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """OpenRadioss MAT122 Newton cutting-plane 3D solid return mapping.

    Cites:
      - `engine/source/materials/mat/mat122/mat122_newton.F` lines 306-445, 603-663.
    """
    # Recover internal variables
    # UVAR: 1:Y, 2:YP, 3:EFTI, 4:EFTU, 5:EFCI, 6:EFCU, 7:Y0, 8:YC, 9:Y0P, 10:YCP,
    #       11:Y0PC, 12:YCPC, 14:DPY, 15:DPZ, 16:EPSPYY, 17:EPSPZZ
    uvar = np.zeros(18, dtype=np.float64)
    dmg = np.zeros(6, dtype=np.float64)
    if extra is not None:
        if "uvar122" in extra:
            uvar = np.asarray(extra["uvar122"], dtype=np.float64).copy()
        if "damage" in extra:
            dmg = np.asarray(extra["damage"], dtype=np.float64).copy()

    df = dmg[1]
    d = dmg[2]
    dp = dmg[3]
    dft = dmg[4]
    dfc = dmg[5]
    y_dmg = uvar[1]
    yp_dmg = uvar[2]
    dpy = uvar[14]
    dpz = uvar[15]
    epspyy = uvar[16]
    epspzz = uvar[17]

    pla = max(0.0, epsp)
    dpla = 0.0

    # Total strain estimation
    epsxx = eps[0] if eps is not None else deps[0]
    epsyy = eps[1] if eps is not None else deps[1]
    epszz = eps[2] if eps is not None else deps[2]

    # Moduli
    e2 = p.young2
    e3 = p.young3
    g12 = p.g12
    g23 = p.g23
    g31 = p.g31
    s12 = p.s12
    s13 = p.s13
    s22 = p.s22
    s23 = p.s23
    s33 = p.s33

    # Trial stress tensor (mat122_newton.F lines 309-315)
    signyy = sig[1] / max(1.0 - dpy, _EM20) + s12 * deps[0] + s22 * deps[1] + s23 * deps[2]
    signzz = sig[2] / max(1.0 - dpz, _EM20) + s13 * deps[0] + s23 * deps[1] + s33 * deps[2]
    signxy = sig[3] / max(1.0 - d, _EM20) + g12 * deps[3]
    signyz = sig[4] / max(1.0 - d, _EM20) + g23 * deps[4]
    signzx = sig[5] / max(1.0 - d, _EM20) + g31 * deps[5]

    # Equivalent stress (mat122_newton.F lines 318-319)
    hard_a = p.hard_a if p.hard_a > 0.0 else 1.0
    seq = math.sqrt(signxy**2 + signyz**2 + signzx**2 + hard_a * (signyy**2 + signzz**2))

    # Yield stress (mat122_newton.F line 278)
    if p.hard_m > 0.0 and p.beta > 0.0:
        sig_y = p.sigy0 + p.beta * ((pla + _EM20) ** p.hard_m)
    else:
        sig_y = p.sigy0

    phi = seq - sig_y

    # Plastic correction with cutting plane Newton iterations (mat122_newton.F lines 345-441)
    if phi > 0.0:
        niter = 3
        for _ in range(niter):
            normyy = hard_a * signyy / max(seq, _EM20)
            normzz = hard_a * signzz / max(seq, _EM20)
            normxy = signxy / max(seq, _EM20)
            normyz = signyz / max(seq, _EM20)
            normzx = signzx / max(seq, _EM20)

            dfdsig2 = (
                normyy * (s22 * normyy + s23 * normzz)
                + normzz * (s23 * normyy + s33 * normzz)
                + normxy * normxy * g12
                + normyz * normyz * g23
                + normzx * normzx * g31
            )

            if p.hard_m > 0.0 and p.beta > 0.0:
                h = p.beta * p.hard_m * ((pla + _EM20) ** (p.hard_m - 1.0))
            else:
                h = 0.0
            h = min(h, max(2.0 * g12, e2))

            sig_dfdsig = (
                signyy * normyy
                + signzz * normzz
                + signxy * normxy
                + signyz * normyz
                + signzx * normzx
            )
            dpla_dlam = sig_dfdsig / max(sig_y, _EM20)

            dphi_dlam = -dfdsig2 - h * dpla_dlam
            if abs(dphi_dlam) < _EM20:
                dphi_dlam = math.copysign(_EM20, dphi_dlam)

            dlam = -phi / dphi_dlam

            dpyy = dlam * normyy
            dpzz = dlam * normzz
            dpxy = dlam * normxy
            dpyz = dlam * normyz
            dpzx = dlam * normzx

            epspyy += dpyy
            epspzz += dpzz

            signyy -= (s22 * dpyy + s23 * dpzz)
            signzz -= (s23 * dpyy + s33 * dpzz)
            signxy -= dpxy * g12
            signyz -= dpyz * g23
            signzx -= dpzx * g31

            ddep = dlam * dpla_dlam
            dpla = max(0.0, dpla + ddep)
            pla += ddep

            seq = math.sqrt(signxy**2 + signyz**2 + signzx**2 + hard_a * (signyy**2 + signzz**2))
            sig_y += h * dlam * dpla_dlam
            phi = seq - sig_y

    # Damage variables computation (mat122_newton.F lines 455-598)
    epsf_eq = (
        (1.0 - p.nu23 * p.nu32) * epsxx
        + (p.nu23 * p.nu31 + p.nu21) * (epsyy - epspyy)
        + (p.nu21 * p.nu32 + p.nu31) * (epszz - epspzz)
    )

    if epsf_eq >= 0.0:
        if p.eps_ftu > p.eps_fti and p.eps_fti > 0.0:
            if epsf_eq >= p.eps_fti and epsf_eq < p.eps_ftu:
                dft = max(p.dftu * ((epsf_eq - p.eps_fti) / (p.eps_ftu - p.eps_fti)), dft)
            elif epsf_eq >= p.eps_ftu:
                dft = max(1.0 - (1.0 - p.dftu) * (p.eps_ftu / epsf_eq), dft)
        dft = min(max(dft, 0.0), 1.0)
        df = dft
    elif p.ibuck > 1:
        abs_eps = abs(epsf_eq)
        if p.eps_fcu > p.eps_fci and p.eps_fci > 0.0:
            if abs_eps >= p.eps_fci and abs_eps < p.eps_fcu:
                dfc = max(p.dfcu * ((abs_eps - p.eps_fci) / (p.eps_fcu - p.eps_fci)), dfc)
            elif abs_eps >= p.eps_fcu:
                dfc = max(1.0 - (1.0 - p.dfcu) * (p.eps_fcu / abs_eps), dfc)
        dfc = min(max(dfc, 0.0), 1.0)
        df = dfc

    # Matrix damage energy
    zd = 0.5 * (signxy**2 / max(g12, _EM20) + signyz**2 / max(g23, _EM20) + signzx**2 / max(g31, _EM20))
    zdp = 0.5 * (max(signyy, 0.0)**2 / max(e2, _EM20) + max(signzz, 0.0)**2 / max(e3, _EM20))
    y_dmg = max(y_dmg, math.sqrt(max(0.0, zd + p.b * zdp)))
    yp_dmg = max(yp_dmg, math.sqrt(max(0.0, zdp)))

    # Shear damage evolution
    if p.ish == 1:  # Linear
        if y_dmg >= p.y0:
            d = min(p.dmax, max(0.0, y_dmg - p.y0) / max(p.yc, _EM20))
        d = min(max(d, 0.0), 1.0)
    elif p.ish == 2:  # Exponential
        if y_dmg > p.y0:
            d = p.dsat1 * (1.0 - math.exp((p.y0 - y_dmg) / max(p.yc, _EM20)))
        d = min(max(d, 0.0), 1.0)

    # Transverse damage evolution
    if p.itr == 1:  # Linear
        if yp_dmg >= p.y0p:
            dp = min(p.dmax, max(0.0, yp_dmg - p.y0p) / max(p.ycp, _EM20))
        dp = min(max(dp, 0.0), 1.0)
    elif p.itr == 2:  # Exponential
        if yp_dmg > p.y0p:
            dp = p.dsat2 * (1.0 - math.exp((p.y0p - yp_dmg) / max(p.ycp, _EM20)))
        dp = min(max(dp, 0.0), 1.0)

    dpy = dp if epsyy >= 0.0 else 0.0
    dpz = dp if epszz >= 0.0 else 0.0

    # Damaged stiffness matrix (mat122_newton.F lines 619-624)
    s11_d = p.s11 * (1.0 - df)
    s12_d = p.s12 * (1.0 - df) * (1.0 - dpy)
    s13_d = p.s13 * (1.0 - df) * (1.0 - dpz)
    s22_d = p.s22 * (1.0 - dpy)
    s23_d = p.s23 * (1.0 - dpy) * (1.0 - dpz)
    s33_d = p.s33 * (1.0 - dpz)

    # Stresses update with damage softening (mat122_newton.F lines 629-646)
    sign = np.zeros(6, dtype=np.float64)
    if p.gamma > 0.0 and epsxx < 0.0 and p.e1c > 0.0:
        sign[0] = -(1.0 / p.gamma) * math.log(1.0 + p.gamma * p.e1c * abs(epsxx)) * (1.0 - df)
    else:
        sign[0] = s11_d * epsxx

    sign[0] += s12_d * (epsyy - epspyy) + s13_d * (epszz - epspzz)
    sign[1] = s12_d * epsxx + s22_d * (epsyy - epspyy) + s23_d * (epszz - epspzz)
    sign[2] = s13_d * epsxx + s23_d * (epsyy - epspyy) + s33_d * (epszz - epspzz)
    sign[3] = signxy * (1.0 - d)
    sign[4] = signyz * (1.0 - d)
    sign[5] = signzx * (1.0 - d)

    # State update
    dmg[0] = max(df, d, dp)
    dmg[1] = df
    dmg[2] = d
    dmg[3] = dp
    dmg[4] = dft
    dmg[5] = dfc
    uvar[1] = y_dmg
    uvar[2] = yp_dmg
    uvar[14] = dpy
    uvar[15] = dpz
    uvar[16] = epspyy
    uvar[17] = epspzz

    extra_out = {
        "uvar122": uvar,
        "damage": dmg,
    }

    # Sound speed (mat122_newton.F lines 603-605)
    c_sound = math.sqrt(max(p.s11, p.s22, p.s33, 2.0 * g12, 2.0 * g23, 2.0 * g31) / max(p.rho0, _EM20))

    return sign, pla, c_sound, extra_out


def mat122_nice_solid_update(
    p: Law122Params,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float = 0.0,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """OpenRadioss MAT122 NICE explicit algorithm 3D solid return mapping.

    Cites:
      - `engine/source/materials/mat/mat122/mat122_nice.F` lines 311-455, 600-660.
    """
    uvar = np.zeros(18, dtype=np.float64)
    dmg = np.zeros(6, dtype=np.float64)
    if extra is not None:
        if "uvar122" in extra:
            uvar = np.asarray(extra["uvar122"], dtype=np.float64).copy()
        if "damage" in extra:
            dmg = np.asarray(extra["damage"], dtype=np.float64).copy()

    df = dmg[1]
    d = dmg[2]
    dp = dmg[3]
    dft = dmg[4]
    dfc = dmg[5]
    y_dmg = uvar[1]
    yp_dmg = uvar[2]
    dpy = uvar[14]
    dpz = uvar[15]
    epspyy = uvar[16]
    epspzz = uvar[17]

    pla = max(0.0, epsp)
    dpla = 0.0

    # Total strain estimation
    epsxx = eps[0] if eps is not None else deps[0]
    epsyy = eps[1] if eps is not None else deps[1]
    epszz = eps[2] if eps is not None else deps[2]

    # Moduli
    e2 = p.young2
    e3 = p.young3
    g12 = p.g12
    g23 = p.g23
    g31 = p.g31
    s12 = p.s12
    s13 = p.s13
    s22 = p.s22
    s23 = p.s23
    s33 = p.s33
    hard_a = p.hard_a if p.hard_a > 0.0 else 1.0

    # Trial stress increment (mat122_nice.F lines 352-356)
    dsigyy = s12 * deps[0] + s22 * deps[1] + s23 * deps[2]
    dsigzz = s13 * deps[0] + s23 * deps[1] + s33 * deps[2]
    dsigxy = g12 * deps[3]
    dsigyz = g23 * deps[4]
    dsigzx = g31 * deps[5]

    # Trial stress tensor (mat122_nice.F lines 314-320)
    signyy = sig[1] / max(1.0 - dpy, _EM20) + dsigyy
    signzz = sig[2] / max(1.0 - dpz, _EM20) + dsigzz
    signxy = sig[3] / max(1.0 - d, _EM20) + dsigxy
    signyz = sig[4] / max(1.0 - d, _EM20) + dsigyz
    signzx = sig[5] / max(1.0 - d, _EM20) + dsigzx

    # Old equivalent stress SEQ0 (mat122_nice.F lines 370-374)
    sigo_yy_eff = sig[1] / max(1.0 - dpy, _EM20)
    sigo_zz_eff = sig[2] / max(1.0 - dpz, _EM20)
    sigo_xy_eff = sig[3] / max(1.0 - d, _EM20)
    sigo_yz_eff = sig[4] / max(1.0 - d, _EM20)
    sigo_zx_eff = sig[5] / max(1.0 - d, _EM20)
    seq0 = math.sqrt(
        sigo_xy_eff**2 + sigo_yz_eff**2 + sigo_zx_eff**2
        + hard_a * (sigo_yy_eff**2 + sigo_zz_eff**2)
    )

    # Current equivalent stress
    seq = math.sqrt(signxy**2 + signyz**2 + signzx**2 + hard_a * (signyy**2 + signzz**2))

    # Yield stress (mat122_nice.F line 278)
    if p.hard_m > 0.0 and p.beta > 0.0:
        sig_y = p.sigy0 + p.beta * ((pla + _EM20) ** p.hard_m)
    else:
        sig_y = p.sigy0

    phi = seq - sig_y

    if phi > 0.0:
        if seq0 < _EM10:
            seq_norm = max(seq, _EM20)
            normyy = hard_a * signyy / seq_norm
            normzz = hard_a * signzz / seq_norm
            normxy = signxy / seq_norm
            normyz = signyz / seq_norm
            normzx = signzx / seq_norm
            sig_eff_yy = signyy
            sig_eff_zz = signzz
            sig_eff_xy = signxy
            sig_eff_yz = signyz
            sig_eff_zx = signzx
            phi0 = 0.0
            dphi = phi
        else:
            seq_norm = max(seq0, _EM20)
            normyy = hard_a * sigo_yy_eff / seq_norm
            normzz = hard_a * sigo_zz_eff / seq_norm
            normxy = sigo_xy_eff / seq_norm
            normyz = sigo_yz_eff / seq_norm
            normzx = sigo_zx_eff / seq_norm
            sig_eff_yy = sigo_yy_eff
            sig_eff_zz = sigo_zz_eff
            sig_eff_xy = sigo_xy_eff
            sig_eff_yz = sigo_yz_eff
            sig_eff_zx = sigo_zx_eff
            phi0 = seq0 - sig_y
            dphi = normyy * dsigyy + normzz * dsigzz + normxy * dsigxy + normyz * dsigyz + normzx * dsigzx

        dfdsig2 = (
            normyy * (s22 * normyy + s23 * normzz)
            + normzz * (s23 * normyy + s33 * normzz)
            + normxy * normxy * g12
            + normyz * normyz * g23
            + normzx * normzx * g31
        )

        if p.hard_m > 0.0 and p.beta > 0.0:
            h = p.beta * p.hard_m * ((pla + _EM20) ** (p.hard_m - 1.0))
        else:
            h = 0.0
        h = min(h, max(2.0 * g12, e2))

        sig_dfdsig = (
            sig_eff_yy * normyy
            + sig_eff_zz * normzz
            + sig_eff_xy * normxy
            + sig_eff_yz * normyz
            + sig_eff_zx * normzx
        )
        dpla_dlam = sig_dfdsig / max(sig_y, _EM20)

        dphi_dlam = -dfdsig2 - h * dpla_dlam
        if abs(dphi_dlam) < _EM20:
            dphi_dlam = math.copysign(_EM20, dphi_dlam)

        # Explicit NICE plastic multiplier (mat122_nice.F line 419)
        dlam = -(phi0 + dphi) / dphi_dlam
        dlam = max(0.0, dlam)

        dpyy = dlam * normyy
        dpzz = dlam * normzz
        dpxy = dlam * normxy
        dpyz = dlam * normyz
        dpzx = dlam * normzx

        epspyy += dpyy
        epspzz += dpzz

        signyy -= (s22 * dpyy + s23 * dpzz)
        signzz -= (s23 * dpyy + s33 * dpzz)
        signxy -= dpxy * g12
        signyz -= dpyz * g23
        signzx -= dpzx * g31

        ddep = dlam * dpla_dlam
        dpla = max(0.0, dpla + ddep)
        pla += ddep

    # Damage variables computation (mat122_nice.F lines 455-598)
    epsf_eq = (
        (1.0 - p.nu23 * p.nu32) * epsxx
        + (p.nu23 * p.nu31 + p.nu21) * (epsyy - epspyy)
        + (p.nu21 * p.nu32 + p.nu31) * (epszz - epspzz)
    )

    if epsf_eq >= 0.0:
        if p.eps_ftu > p.eps_fti and p.eps_fti > 0.0:
            if epsf_eq >= p.eps_fti and epsf_eq < p.eps_ftu:
                dft = max(p.dftu * ((epsf_eq - p.eps_fti) / (p.eps_ftu - p.eps_fti)), dft)
            elif epsf_eq >= p.eps_ftu:
                dft = max(1.0 - (1.0 - p.dftu) * (p.eps_ftu / epsf_eq), dft)
        dft = min(max(dft, 0.0), 1.0)
        df = dft
    elif p.ibuck > 1:
        abs_eps = abs(epsf_eq)
        if p.eps_fcu > p.eps_fci and p.eps_fci > 0.0:
            if abs_eps >= p.eps_fci and abs_eps < p.eps_fcu:
                dfc = max(p.dfcu * ((abs_eps - p.eps_fci) / (p.eps_fcu - p.eps_fci)), dfc)
            elif abs_eps >= p.eps_fcu:
                dfc = max(1.0 - (1.0 - p.dfcu) * (p.eps_fcu / abs_eps), dfc)
        dfc = min(max(dfc, 0.0), 1.0)
        df = dfc

    # Matrix damage energy
    zd = 0.5 * (signxy**2 / max(g12, _EM20) + signyz**2 / max(g23, _EM20) + signzx**2 / max(g31, _EM20))
    zdp = 0.5 * (max(signyy, 0.0)**2 / max(e2, _EM20) + max(signzz, 0.0)**2 / max(e3, _EM20))
    y_dmg = max(y_dmg, math.sqrt(max(0.0, zd + p.b * zdp)))
    yp_dmg = max(yp_dmg, math.sqrt(max(0.0, zdp)))

    # Shear damage evolution
    if p.ish == 1:
        if y_dmg >= p.y0:
            d = min(p.dmax, max(0.0, y_dmg - p.y0) / max(p.yc, _EM20))
        d = min(max(d, 0.0), 1.0)
    elif p.ish == 2:
        if y_dmg > p.y0:
            d = p.dsat1 * (1.0 - math.exp((p.y0 - y_dmg) / max(p.yc, _EM20)))
        d = min(max(d, 0.0), 1.0)

    # Transverse damage evolution
    if p.itr == 1:
        if yp_dmg >= p.y0p:
            dp = min(p.dmax, max(0.0, yp_dmg - p.y0p) / max(p.ycp, _EM20))
        dp = min(max(dp, 0.0), 1.0)
    elif p.itr == 2:
        if yp_dmg > p.y0p:
            dp = p.dsat2 * (1.0 - math.exp((p.y0p - yp_dmg) / max(p.ycp, _EM20)))
        dp = min(max(dp, 0.0), 1.0)

    dpy = dp if epsyy >= 0.0 else 0.0
    dpz = dp if epszz >= 0.0 else 0.0

    # Damaged stiffness matrix (mat122_nice.F lines 634-639)
    s11_d = p.s11 * (1.0 - df)
    s12_d = p.s12 * (1.0 - df) * (1.0 - dpy)
    s13_d = p.s13 * (1.0 - df) * (1.0 - dpz)
    s22_d = p.s22 * (1.0 - dpy)
    s23_d = p.s23 * (1.0 - dpy) * (1.0 - dpz)
    s33_d = p.s33 * (1.0 - dpz)

    # Stresses update
    sign = np.zeros(6, dtype=np.float64)
    if p.gamma > 0.0 and epsxx < 0.0 and p.e1c > 0.0:
        sign[0] = -(1.0 / p.gamma) * math.log(1.0 + p.gamma * p.e1c * abs(epsxx)) * (1.0 - df)
    else:
        sign[0] = s11_d * epsxx

    sign[0] += s12_d * (epsyy - epspyy) + s13_d * (epszz - epspzz)
    sign[1] = s12_d * epsxx + s22_d * (epsyy - epspyy) + s23_d * (epszz - epspzz)
    sign[2] = s13_d * epsxx + s23_d * (epsyy - epspyy) + s33_d * (epszz - epspzz)
    sign[3] = signxy * (1.0 - d)
    sign[4] = signyz * (1.0 - d)
    sign[5] = signzx * (1.0 - d)

    dmg[0] = max(df, d, dp)
    dmg[1] = df
    dmg[2] = d
    dmg[3] = dp
    dmg[4] = dft
    dmg[5] = dfc
    uvar[1] = y_dmg
    uvar[2] = yp_dmg
    uvar[14] = dpy
    uvar[15] = dpz
    uvar[16] = epspyy
    uvar[17] = epspzz

    extra_out = {
        "uvar122": uvar,
        "damage": dmg,
    }
    c_sound = math.sqrt(max(p.s11, p.s22, p.s33, 2.0 * g12, 2.0 * g23, 2.0 * g31) / max(p.rho0, _EM20))
    return sign, pla, c_sound, extra_out


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """3D continuum solid stress update for LAW122.

    Dispatches between:
      1. OpenRadioss MAT122 Ladevèze cutting-plane model (when Ladevèze parameters are provided)
      2. Chaboche Armstrong-Frederick nonlinear kinematic hardening J2 model (when c_kin > 0)
    """
    p = build_law122(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, 6) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, 6) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    # Determine mode: Ladevèze vs Chaboche
    # If explicit Chaboche parameters c_kin/gamma_kin > 0, run Chaboche J2
    use_chaboche = (p.c_kin > 0.0 or p.gamma_kin > 0.0 or kwargs.get("chaboche", False))

    if not use_chaboche and (p.beta > 0.0 or p.hard_a != 1.0 or p.ish > 0 or p.itr > 0 or p.dmax > 0.0):
        # OpenRadioss MAT122 Ladevèze formulation (IRES=1: NICE, IRES=2: Newton cutting-plane)
        sig_new = np.zeros_like(sig_2d)
        epsp_new = np.zeros_like(epsp_arr)
        c_sound = sound_speed(p)
        update_fn = mat122_nice_solid_update if p.ires == 1 else mat122_newton_solid_update
        for i in range(n):
            s_i, p_i, c_i, ex_i = update_fn(
                p, sig_2d[i], deps_2d[i], epsp=epsp_arr[i], extra=extra
            )
            sig_new[i] = s_i
            epsp_new[i] = p_i
            c_sound = c_i
            if extra is not None:
                extra.update(ex_i)

        c_out = c_sound if is_1d else np.full(n, c_sound, dtype=np.float64)
        if is_1d:
            return sig_new[0], float(epsp_new[0]), float(c_out)
        return sig_new, epsp_new, c_out

    # Chaboche J2 return mapping with Armstrong-Frederick kinematic hardening
    if extra is not None and "backstress" in extra:
        alpha_arr = np.asarray(extra["backstress"], dtype=np.float64).reshape(-1, 6)
    else:
        alpha_arr = np.zeros((n, 6), dtype=np.float64)

    sig_new = np.zeros_like(sig_2d)
    epsp_new = np.zeros_like(epsp_arr)
    alpha_new = np.zeros_like(alpha_arr)

    lame = p.lame
    g = p.g12
    g2 = 2.0 * g

    for i in range(n):
        deps_i = deps_2d[i]
        tr_deps = deps_i[0] + deps_i[1] + deps_i[2]

        # Elastic trial stress
        sig_tr = np.zeros(6, dtype=np.float64)
        sig_tr[0] = sig_2d[i, 0] + lame * tr_deps + g2 * deps_i[0]
        sig_tr[1] = sig_2d[i, 1] + lame * tr_deps + g2 * deps_i[1]
        sig_tr[2] = sig_2d[i, 2] + lame * tr_deps + g2 * deps_i[2]
        sig_tr[3] = sig_2d[i, 3] + g * deps_i[3]
        sig_tr[4] = sig_2d[i, 4] + g * deps_i[4]
        sig_tr[5] = sig_2d[i, 5] + g * deps_i[5]

        p_m = (sig_tr[0] + sig_tr[1] + sig_tr[2]) / 3.0
        s_tr = np.array([
            sig_tr[0] - p_m,
            sig_tr[1] - p_m,
            sig_tr[2] - p_m,
            sig_tr[3],
            sig_tr[4],
            sig_tr[5],
        ], dtype=np.float64)

        # Relative stress eta = s - alpha
        alpha_i = alpha_arr[i] if i < len(alpha_arr) else np.zeros(6, dtype=np.float64)
        eta_tr = s_tr - alpha_i

        j2 = 0.5 * (eta_tr[0]**2 + eta_tr[1]**2 + eta_tr[2]**2) + eta_tr[3]**2 + eta_tr[4]**2 + eta_tr[5]**2
        seq_tr = math.sqrt(max(0.0, 3.0 * j2))

        sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp_arr[i])

        if seq_tr > sig_y and seq_tr > _EM10:
            n_flow = np.zeros(6, dtype=np.float64)
            n_flow[:3] = 1.5 * eta_tr[:3] / seq_tr
            n_flow[3:] = 3.0 * eta_tr[3:] / seq_tr

            # Chaboche kinematic hardening modulus: H_kin = C - gamma * (alpha : n)
            alpha_dot_n = (
                alpha_i[0] * n_flow[0]
                + alpha_i[1] * n_flow[1]
                + alpha_i[2] * n_flow[2]
                + 2.0 * (alpha_i[3] * n_flow[3] + alpha_i[4] * n_flow[4] + alpha_i[5] * n_flow[5])
            )
            h_kin = p.c_kin - p.gamma_kin * alpha_dot_n

            denom = 3.0 * g + h_iso + h_kin
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)

            # Update backstress: d_alpha = 2/3 C d_eps_p - gamma * alpha * dgamma
            d_alpha = (2.0 / 3.0) * p.c_kin * dgamma * n_flow - p.gamma_kin * alpha_i * dgamma
            alpha_new[i] = alpha_i + d_alpha

            factor = max(0.0, 1.0 - (3.0 * g * dgamma) / seq_tr)
            sig_new[i, 0] = p_m + alpha_new[i, 0] + eta_tr[0] * factor
            sig_new[i, 1] = p_m + alpha_new[i, 1] + eta_tr[1] * factor
            sig_new[i, 2] = p_m + alpha_new[i, 2] + eta_tr[2] * factor
            sig_new[i, 3] = alpha_new[i, 3] + eta_tr[3] * factor
            sig_new[i, 4] = alpha_new[i, 4] + eta_tr[4] * factor
            sig_new[i, 5] = alpha_new[i, 5] + eta_tr[5] * factor

            epsp_new[i] = epsp_arr[i] + dgamma
        else:
            sig_new[i] = sig_tr
            alpha_new[i] = alpha_i
            epsp_new[i] = epsp_arr[i]

    if extra is not None:
        extra["backstress"] = alpha_new

    c = sound_speed(p)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def mat122c_newton_shell_update(
    p: Law122Params,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: float = 0.0,
    eps: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """OpenRadioss MAT122C Newton cutting-plane 2D plane-stress shell return mapping.

    Cites:
      - `engine/source/materials/mat/mat122/mat122c_newton.F` lines 289-405, 413-500, 617-654.
    """
    uvar = np.zeros(18, dtype=np.float64)
    dmg = np.zeros(6, dtype=np.float64)
    if extra is not None:
        if "uvar122" in extra:
            uvar = np.asarray(extra["uvar122"], dtype=np.float64).copy()
        if "damage" in extra:
            dmg = np.asarray(extra["damage"], dtype=np.float64).copy()

    df = dmg[1]
    d = dmg[2]
    dp = dmg[3]
    dft = dmg[4]
    dfc = dmg[5]
    y_dmg = uvar[1]
    yp_dmg = uvar[2]
    epspyy = uvar[16]

    pla = max(0.0, epsp)
    dpla = 0.0

    epsxx = eps[0] if eps is not None else deps[0]
    epsyy = eps[1] if eps is not None else deps[1]

    # Moduli (mat122c_newton.F lines 243-259)
    e1 = p.young1
    e2 = p.young2
    g12 = p.g12
    g23 = p.g23
    g31 = p.g31
    hard_a = p.hard_a if p.hard_a > 0.0 else 1.0

    denom = max(1.0 - p.nu12 * p.nu21, _EM20)
    a11 = e1 / denom
    a12 = p.nu21 * a11
    a22 = e2 / denom

    # Trial stress components (mat122c_newton.F lines 292-295)
    sig0_yy = sig[1] if len(sig) > 1 else 0.0
    sig0_xy = sig[2] if len(sig) > 2 else 0.0
    sig0_yz = sig[3] if len(sig) > 3 else 0.0
    sig0_zx = sig[4] if len(sig) > 4 else 0.0

    deps_xx = deps[0]
    deps_yy = deps[1] if len(deps) > 1 else 0.0
    deps_xy = deps[2] if len(deps) > 2 else 0.0
    deps_yz = deps[3] if len(deps) > 3 else 0.0
    deps_zx = deps[4] if len(deps) > 4 else 0.0

    signyy = sig0_yy / max(1.0 - dp, _EM20) + a12 * deps_xx + a22 * deps_yy
    signxy = sig0_xy / max(1.0 - d, _EM20) + g12 * deps_xy
    d_eff = max(min(1.0 - d, 1.0 - dp), _EM20)
    signyz = sig0_yz / d_eff + g23 * deps_yz
    signzx = sig0_zx / d_eff + g31 * deps_zx

    seq = math.sqrt(signxy**2 + hard_a * (signyy**2))

    if p.hard_m > 0.0 and p.beta > 0.0:
        sig_y = p.sigy0 + p.beta * ((pla + _EM20) ** p.hard_m)
    else:
        sig_y = p.sigy0

    phi = seq - sig_y

    if phi > 0.0:
        niter = 3
        for _ in range(niter):
            normyy = hard_a * signyy / max(seq, _EM20)
            normxy = signxy / max(seq, _EM20)

            dfdsig2 = normyy * normyy * a22 + normxy * normxy * g12

            if p.hard_m > 0.0 and p.beta > 0.0:
                h = p.beta * p.hard_m * ((pla + _EM20) ** (p.hard_m - 1.0))
            else:
                h = 0.0
            h = min(h, max(2.0 * g12, e2))

            sig_dfdsig = signyy * normyy + signxy * normxy
            dpla_dlam = sig_dfdsig / max(sig_y, _EM20)

            dphi_dlam = -dfdsig2 - h * dpla_dlam
            if abs(dphi_dlam) < _EM20:
                dphi_dlam = math.copysign(_EM20, dphi_dlam)

            dlam = -phi / dphi_dlam
            dpyy = dlam * normyy
            dpxy = dlam * normxy

            epspyy += dpyy
            signyy -= dpyy * a22
            signxy -= dpxy * g12

            ddep = dlam * dpla_dlam
            dpla = max(0.0, dpla + ddep)
            pla += ddep

            seq = math.sqrt(signxy**2 + hard_a * (signyy**2))
            sig_y += h * dlam * dpla_dlam
            phi = seq - sig_y

    # Damage computation (mat122c_newton.F lines 417-449)
    epsf_eq = epsxx + p.nu21 * (epsyy - epspyy)
    if epsf_eq >= 0.0:
        if p.eps_ftu > p.eps_fti and p.eps_fti > 0.0:
            if epsf_eq >= p.eps_fti and epsf_eq < p.eps_ftu:
                dft = max(p.dftu * ((epsf_eq - p.eps_fti) / (p.eps_ftu - p.eps_fti)), dft)
            elif epsf_eq >= p.eps_ftu:
                dft = max(1.0 - (1.0 - p.dftu) * (p.eps_ftu / epsf_eq), dft)
        dft = min(max(dft, 0.0), 1.0)
        df = dft
    elif p.ibuck > 1:
        abs_eps = abs(epsf_eq)
        if p.eps_fcu > p.eps_fci and p.eps_fci > 0.0:
            if abs_eps >= p.eps_fci and abs_eps < p.eps_fcu:
                dfc = max(p.dfcu * ((abs_eps - p.eps_fci) / (p.eps_fcu - p.eps_fci)), dfc)
            elif abs_eps >= p.eps_fcu:
                dfc = max(1.0 - (1.0 - p.dfcu) * (p.eps_fcu / abs_eps), dfc)
        dfc = min(max(dfc, 0.0), 1.0)
        df = dfc

    # Matrix damage energy
    zd = 0.5 * (signxy**2 / max(g12, _EM20) + signzx**2 / max(g31, _EM20))
    zdp = 0.5 * (max(signyy, 0.0)**2 / max(e2, _EM20))
    y_dmg = max(y_dmg, math.sqrt(max(0.0, zd + p.b * zdp)))
    yp_dmg = max(yp_dmg, math.sqrt(max(0.0, zdp)))

    # Shear damage
    if p.ish == 1:
        if y_dmg >= p.y0:
            d = min(p.dmax, max(0.0, y_dmg - p.y0) / max(p.yc, _EM20))
        d = min(max(d, 0.0), 1.0)
    elif p.ish == 2:
        if y_dmg > p.y0:
            d = p.dsat1 * (1.0 - math.exp((p.y0 - y_dmg) / max(p.yc, _EM20)))
        d = min(max(d, 0.0), 1.0)

    # Transverse damage
    if p.itr == 1:
        if yp_dmg >= p.y0p:
            dp = min(p.dmax, max(0.0, yp_dmg - p.y0p) / max(p.ycp, _EM20))
        dp = min(max(dp, 0.0), 1.0)
    elif p.itr == 2:
        if yp_dmg > p.y0p:
            dp = p.dsat2 * (1.0 - math.exp((p.y0p - yp_dmg) / max(p.ycp, _EM20)))
        dp = min(max(dp, 0.0), 1.0)

    # Damaged stiffness matrix (mat122c_newton.F lines 623-625)
    a11_d = a11 * (1.0 - df)
    a12_d = p.nu21 * a11_d * (1.0 - dp)
    a22_d = a22 * (1.0 - dp)

    # Stress update (mat122c_newton.F lines 630-640)
    ncomp = max(3, len(sig))
    sign = np.zeros(ncomp, dtype=np.float64)

    if p.gamma > 0.0 and epsxx < 0.0 and p.e1c > 0.0:
        sign[0] = -(1.0 / p.gamma) * math.log(1.0 + p.gamma * p.e1c * abs(epsxx)) * (1.0 - df)
    else:
        sign[0] = a11_d * epsxx

    sign[0] += a12_d * (epsyy - epspyy)
    sign[1] = a12_d * epsxx + a22_d * (epsyy - epspyy)
    sign[2] = signxy * (1.0 - d)

    if ncomp > 3:
        sign[3] = signyz * min(1.0 - d, 1.0 - dp)
    if ncomp > 4:
        sign[4] = signzx * min(1.0 - d, 1.0 - dp)

    dmg[0] = max(df, d, dp)
    dmg[1] = df
    dmg[2] = d
    dmg[3] = dp
    dmg[4] = dft
    dmg[5] = dfc
    uvar[1] = y_dmg
    uvar[2] = yp_dmg
    uvar[16] = epspyy

    extra_out = {
        "uvar122": uvar,
        "damage": dmg,
    }
    c_sound = math.sqrt(max(a11, a22) / max(p.rho0, _EM20))
    return sign, pla, c_sound, extra_out


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[Union[float, np.ndarray]] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """2D plane-stress shell stress update with Chaboche kinematic hardening."""
    p = build_law122(mat)
    sig_arr = np.asarray(sig, dtype=np.float64)
    deps_arr = np.asarray(deps, dtype=np.float64)
    is_1d = (deps_arr.ndim == 1)

    sig_2d = sig_arr.reshape(1, -1) if is_1d else sig_arr.copy()
    deps_2d = deps_arr.reshape(1, -1) if is_1d else deps_arr.copy()
    n = sig_2d.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(n, dtype=np.float64)
    else:
        epsp_in = np.asarray(epsp, dtype=np.float64)
        epsp_arr = np.full(n, float(epsp_in)) if epsp_in.ndim == 0 else epsp_in.copy()

    # Determine mode: Ladevèze vs Chaboche
    use_chaboche = (p.c_kin > 0.0 or p.gamma_kin > 0.0 or kwargs.get("chaboche", False))

    if not use_chaboche and (p.beta > 0.0 or p.hard_a != 1.0 or p.ish > 0 or p.itr > 0 or p.dmax > 0.0):
        sig_new = np.zeros_like(sig_2d)
        epsp_new = np.zeros_like(epsp_arr)
        c_sound = sound_speed(p, is_shell=True)
        for i in range(n):
            s_i, p_i, c_i, ex_i = mat122c_newton_shell_update(
                p, sig_2d[i], deps_2d[i], epsp=epsp_arr[i], extra=extra
            )
            sig_new[i] = s_i
            epsp_new[i] = p_i
            c_sound = c_i
            if extra is not None:
                extra.update(ex_i)

        c_out = c_sound if is_1d else np.full(n, c_sound, dtype=np.float64)
        if is_1d:
            return sig_new[0], float(epsp_new[0]), float(c_out)
        return sig_new, epsp_new, c_out

    ncomp = sig_2d.shape[1]
    sig_new = np.zeros_like(sig_2d)
    epsp_new = np.zeros_like(epsp_arr)

    a11 = p.a11
    a12 = p.a12
    g = p.g12

    for i in range(n):
        deps_i = deps_2d[i]
        s_xx = sig_2d[i, 0] + a11 * deps_i[0] + a12 * deps_i[1]
        s_yy = sig_2d[i, 1] + a12 * deps_i[0] + a11 * deps_i[1]
        s_xy = sig_2d[i, 2] + g * deps_i[2]

        seq_tr = math.sqrt(max(0.0, s_xx**2 + s_yy**2 - s_xx * s_yy + 3.0 * s_xy**2))
        sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp_arr[i])

        if seq_tr > sig_y and seq_tr > _EM10:
            denom = a11 + h_iso + p.c_kin
            dgamma = (seq_tr - sig_y) / max(_EM20, denom)
            factor = max(0.0, 1.0 - (a11 * dgamma) / seq_tr)

            sig_new[i, 0] = s_xx * factor
            sig_new[i, 1] = s_yy * factor
            sig_new[i, 2] = s_xy * factor
            epsp_new[i] = epsp_arr[i] + dgamma
        else:
            sig_new[i, 0] = s_xx
            sig_new[i, 1] = s_yy
            sig_new[i, 2] = s_xy
            epsp_new[i] = epsp_arr[i]

        if ncomp > 3:
            sig_new[i, 3:] = sig_2d[i, 3:]

    c = sound_speed(p, is_shell=True)
    c_out = c if is_1d else np.full(n, c, dtype=np.float64)

    if is_1d:
        return sig_new[0], float(epsp_new[0]), float(c_out)
    return sig_new, epsp_new, c_out


def sound_speed(
    mat: Any,
    eps: Optional[Any] = None,
    extra: Optional[Any] = None,
    is_shell: bool = False,
    **kwargs: Any,
) -> float:
    """Compute acoustic wave speed for LAW122."""
    p = build_law122(mat)
    rho = p.rho0 if p.rho0 > 0.0 else 1.0
    if is_shell:
        mod = p.a11
    else:
        mod = p.bulk + (4.0 / 3.0) * p.g12
    return float(math.sqrt(max(0.0, mod / rho)))


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 6x6 continuum solid tangent stiffness for LAW122."""
    p = build_law122(mat)
    c_el = np.zeros((6, 6), dtype=np.float64)
    lame = p.lame
    g = p.g12
    g2 = 2.0 * g

    c_el[0, 0] = lame + g2
    c_el[1, 1] = lame + g2
    c_el[2, 2] = lame + g2
    c_el[0, 1] = c_el[1, 0] = lame
    c_el[0, 2] = c_el[2, 0] = lame
    c_el[1, 2] = c_el[2, 1] = lame
    c_el[3, 3] = g
    c_el[4, 4] = g
    c_el[5, 5] = g

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 6, 6), dtype=np.float64)
        for i in range(n):
            t[i] = solid_tangent(p, sig=sig_arr[i], epsp=epsp)
        return t

    p_m = (sig_arr[0] + sig_arr[1] + sig_arr[2]) / 3.0
    s = sig_arr - np.array([p_m, p_m, p_m, 0.0, 0.0, 0.0])
    seq = math.sqrt(max(0.0, 1.5 * (s[0]**2 + s[1]**2 + s[2]**2 + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2))))
    sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp if epsp is not None else 0.0)

    if seq < sig_y or seq <= _EM10:
        return c_el

    n_vec = np.zeros(6, dtype=np.float64)
    n_vec[:3] = 1.5 * s[:3] / seq
    n_vec[3:] = 3.0 * s[3:] / seq

    denom = 3.0 * g + h_iso + p.c_kin
    if denom > _EM20:
        gamma = (2.0 * g) / denom
        c_tan = c_el - (2.0 * g * gamma) * np.outer(n_vec, n_vec)
        return c_tan
    return c_el


def shell_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    epsp: Optional[float] = None,
    epsp_incr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent 3x3 plane stress algorithmic tangent matrix for LAW122."""
    p = build_law122(mat)
    c_el = np.array([
        [p.a11, p.a12, 0.0],
        [p.a12, p.a11, 0.0],
        [0.0, 0.0, p.g12],
    ], dtype=np.float64)

    if sig is None:
        return c_el

    sig_arr = np.asarray(sig, dtype=np.float64)
    if sig_arr.ndim == 2:
        n = sig_arr.shape[0]
        t = np.zeros((n, 3, 3), dtype=np.float64)
        for i in range(n):
            t[i] = shell_tangent(p, sig=sig_arr[i], epsp=epsp)
        return t

    sxx, syy, sxy = sig_arr[0], sig_arr[1], sig_arr[2]
    seq = math.sqrt(max(0.0, sxx**2 + syy**2 - sxx * syy + 3.0 * sxy**2))
    sig_y, h_iso = _eval_chaboche_yield_stress(p, epsp if epsp is not None else 0.0)

    if seq < sig_y or seq <= _EM10:
        return c_el

    n_vec = np.array([
        (2.0 * sxx - syy) / (2.0 * seq),
        (2.0 * syy - sxx) / (2.0 * seq),
        (3.0 * sxy) / seq,
    ], dtype=np.float64)

    cn = c_el @ n_vec
    denom = float(n_vec @ cn) + h_iso + p.c_kin
    if denom > _EM20:
        return c_el - np.outer(cn, cn) / denom
    return c_el


def tangent(group: Any = None, **kwargs: Any) -> np.ndarray:
    """General tangent interface conforming to pyradioss material conventions."""
    return solid_tangent(group, **kwargs)


consistent_solid_tangent = solid_tangent
consistent_shell_tangent = shell_tangent
