r"""LAW126 — Johnson-Holmquist Concrete (HJC) damage model (/MAT/LAW126, /MAT/JOHNSON_HOLMQUIST_CONCRETE).

Fortran origins:
- ``engine/source/materials/mat/mat126/sigeps126.F90`` (solid constitutive update)
- ``starter/source/materials/mat/mat126/hm_read_mat126.F90`` (starter card reader, defaults & parameter derivation)
- ``hm_cfg_files/config/CFG/radioss2025/MAT/matl126_johnson_holmquist_concrete.cfg`` (CFG attributes & format)

Theory
------
LAW126 models concrete subjected to large strains, high strain rates, and high pressures
based on the Johnson-Holmquist Concrete (HJC) formulation (Johnson & Holmquist, 1993):

1. Normalized Strengths:
   Normalized equivalent stress:
   \(\sigma^* = [A (1 - D) + B (P^*)^N] (1 + C \ln(\dot{\varepsilon} / \dot{\varepsilon}_0))\) in compression (\(P^* > 0\)),
   \(\sigma^* = A \max(0, 1 + P / T) (1 - D) (1 + C \ln(\dot{\varepsilon} / \dot{\varepsilon}_0))\) in tension (\(P^* \le 0\)),
   where \(P^* = P / f'_c\), \(T^* = T / f'_c\), and \(\sigma_y = \sigma^* \cdot f'_c \le S_{f,\max} \cdot f'_c\).
   Optional Cowper-Symonds rate dependency:
   Compression: \(1 + ((\dot{\varepsilon} + 10^{-20}) / C_c)^{1 / P_c}\)
   Tension: \(1 + ((\dot{\varepsilon} + 10^{-20}) / C_t)^{1 / P_t}\).

2. Deviatoric Radial Return:
   \(s_{ij}^{trial} = \sigma_{ij}^{old} + P_{old} \delta_{ij} + 2 G (\Delta\varepsilon_{ij} - D_{av}\delta_{ij})\) (normal)
   \(s_{ij}^{trial} = \sigma_{ij}^{old} + G \Delta\varepsilon_{ij}\) (shear, engineering shear)
   \(J_2 = \frac{1}{2} \mathbf{s}^{trial} : \mathbf{s}^{trial}\), \(\sigma_{vm} = \sqrt{3 J_2}\), \(\sigma^* = \sigma_{vm} / f'_c\).
   If \(\sigma^* < \sigma_y^*\): \(\text{scale} = 1.0\), \(\Delta\varepsilon_p = 0.0\).
   Else: \(\text{scale} = \sigma_y^* / \sigma^*\), \(\Delta\varepsilon_p = \frac{(1 - \text{scale}) \sigma_{vm}}{3 G}\).
   \(s_{ij} = \text{scale} \cdot s_{ij}^{trial}\), \(\varepsilon_p \leftarrow \varepsilon_p + \Delta\varepsilon_p\).

3. Three-Phase Compacting Equation of State:
   - Region 1 (elastic): \(P = K_0 \mu\) up to crushing \((\mu_c, P_c)\), where \(K_0 = P_c / \mu_c\).
   - Region 2 (crushing / void collapse): \(\mu_p < \mu_l\),
     \(K_{av} = K_0 + (K_1 - K_0) \frac{\mu_p}{\mu_l}\),
     \(P = K_{av} (\mu - \mu_p)\).
     When \(P > P_{hard}\):
     \(\Delta\mu_p = \frac{P - P_{hard}}{K_{av} + H}\) where \(H = (P_l - P_c) / \mu_l\).
     \(\mu_p \leftarrow \min(\mu_p + \Delta\mu_p, \mu_l)\), \(P \leftarrow P - K_{av} \Delta\mu_p\), \(P_{hard} \leftarrow P\).
   - Region 3 (fully compacted solid concrete): \(\mu_p \ge \mu_l\),
     \(\bar{\mu} = \frac{\mu - \mu_l}{1 + \mu_l}\),
     \(P = K_1 \bar{\mu} + K_2 \bar{\mu}^2 + K_3 \bar{\mu}^3\).
   Tension limit: \(P \ge -T\).

4. Damage Accumulation:
   Fracture plastic strain:
   \(\varepsilon_p^f = \max\left(D_1 (P^* + T^*)^{D_2}, EF_{\min}\right)\) (if \(P^* + T^* \ge 0\), else 0).
   \(D \leftarrow \min\left(1.0, D + \frac{\max(\Delta\varepsilon_p + \Delta\mu_p, 0)}{\varepsilon_p^f}\right)\).

5. Element Deletion (IDEL):
   - IDEL = 0: No element failure
   - IDEL = 1: Tensile failure if \(P^* + T^* \le 0\)
   - IDEL = 2: Maximum plastic strain failure if \(\varepsilon_p > \varepsilon_{\max}\)
   - IDEL = 3: Failure if \(\sigma_y \le 0\)
   - IDEL = 4: Failure if \(D \ge 1.0\).

6. Post-Failure Behavior (IFAILSO):
   - IFAILSO = 1: Classic deletion (off *= 0.8 per cycle, dead below 0.1)
   - IFAILSO = 2: Deviatoric stress vanishes (\(\mathbf{s} = 0\))
   - IFAILSO = 3: Deviatoric stress vanishes in compression; all stress vanishes in tension
   - IFAILSO = 4: Complete stress tensor vanishes
   - IFAILSO = 5: Stress tensor vanishes in tension only.

7. Longitudinal Acoustic Wave Speed:
   \(c = \sqrt{\frac{\frac{\partial P}{\partial \mu} + \frac{4}{3} G}{\rho_0}}\).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pyradioss.model.entities import Material

_EP20 = 1.0e20
_EP30 = 1.0e30
_EM20 = 1.0e-20
_EM30 = 1.0e-30


@dataclass
class Law126Params:
    """Parameters for /MAT/LAW126 (Johnson-Holmquist Concrete / HJC).

    Cites:
    - ``starter/source/materials/mat/mat126/hm_read_mat126.F90``
    - ``engine/source/materials/mat/mat126/sigeps126.F90``
    """

    rho0: float = 0.0
    refer_rho: float = 0.0
    shear: float = 0.0
    aa: float = 0.0
    bb: float = 0.0
    nn: float = 1.0
    fc: float = 0.0
    t0: float = 0.0
    cc: float = 0.0
    eps0: float = 1.0
    fcut: float = 0.0
    sfmax: float = _EP30
    efmin: float = _EM20
    pc: float = 0.0
    muc: float = 0.0
    pl: float = 0.0
    mul: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    d1: float = 0.0
    d2: float = 1.0
    idel: int = 0
    eps_max: float = _EP30
    ifailso: int = 1
    cst: float = 0.0
    powt: float = 1.0
    csc: float = 0.0
    powc: float = 1.0
    title: str = ""

    def __post_init__(self) -> None:
        if self.refer_rho == 0.0:
            self.refer_rho = self.rho0
        if self.eps0 == 0.0:
            self.eps0 = 1.0
        if self.sfmax == 0.0:
            self.sfmax = _EP30
        if self.eps_max == 0.0:
            self.eps_max = _EP30
        if self.efmin == 0.0:
            self.efmin = _EM20
        self.idel = max(0, min(int(self.idel), 4))
        self.ifailso = max(1, min(int(self.ifailso), 5))
        if self.cst != 0.0:
            if self.powt == 0.0:
                self.powt = 1.0
            if self.csc == 0.0:
                self.csc = self.cst
            if self.powc == 0.0:
                self.powc = self.powt

    @property
    def k0(self) -> float:
        """Initial elastic bulk modulus K0 = Pc / muc (hm_read_mat126.F90 line 140)."""
        if self.muc != 0.0:
            return self.pc / self.muc
        if self.k1 != 0.0:
            return self.k1
        return 0.0

    @property
    def h(self) -> float:
        """Crushing tangent bulk modulus H = (Pl - Pc) / mul (line 146)."""
        return ((self.pl - self.pc) / self.mul) if self.mul != 0.0 else 0.0

    @property
    def young(self) -> float:
        """Derived initial Young's modulus E = 9 * K0 * G / (3 * K0 + G)."""
        denom = 3.0 * self.k0 + self.shear
        return (9.0 * self.k0 * self.shear / denom) if denom != 0.0 else 0.0

    @property
    def nu(self) -> float:
        """Derived initial Poisson's ratio nu = (3 * K0 - 2 * G) / (6 * K0 + 2 * G)."""
        denom = 6.0 * self.k0 + 2.0 * self.shear
        return ((3.0 * self.k0 - 2.0 * self.shear) / denom) if denom != 0.0 else 0.2

    @property
    def E(self) -> float:
        return self.young

    @property
    def G(self) -> float:
        return self.shear

    @property
    def K(self) -> float:
        return self.k0

    @property
    def bulk(self) -> float:
        return self.k0

    @property
    def rho(self) -> float:
        return self.rho0

    @property
    def tstar(self) -> float:
        return (self.t0 / self.fc) if self.fc != 0.0 else 0.0


def _extract_param(d: Dict[str, Any], keys: Tuple[str, ...], default: Any = 0.0) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get_params(mat: Any) -> Law126Params:
    """Extract Law126Params from Material, Law126Params, MaterialLaw126, or dict."""
    if isinstance(mat, Law126Params):
        return mat

    if hasattr(mat, "law126_params") and isinstance(mat.law126_params, Law126Params):
        return mat.law126_params

    p: Dict[str, Any] = {}
    title = ""
    rho0_val = None

    if isinstance(mat, Material):
        p = dict(mat.params) if mat.params is not None else {}
        title = mat.title
        rho0_val = getattr(mat, "rho0", None)
    elif isinstance(mat, dict):
        p = dict(mat.get("params", mat))
        title = mat.get("title", "")
        rho0_val = mat.get("rho0") or mat.get("rho") or mat.get("density")
    elif hasattr(mat, "params") and isinstance(mat.params, dict):
        p = dict(mat.params)
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho", getattr(mat, "rho0", None))
    else:
        title = getattr(mat, "title", "")
        rho0_val = getattr(mat, "rho", getattr(mat, "rho0", None))
        for attr in (
            "rho", "refer_rho", "shear", "g", "a", "b", "nn", "n", "fc", "t0",
            "c", "eps0", "fcut", "sfmax", "efmin", "pc", "muc", "pl", "mul",
            "k1", "k2", "k3", "d1", "d2", "idel", "eps_max", "ifailso",
            "ct", "powt", "cc", "powc",
        ):
            if hasattr(mat, attr):
                p[attr] = getattr(mat, attr)

    if rho0_val is None or float(rho0_val) == 0.0:
        rho0_val = _extract_param(p, ("rho0", "rho", "density", "MAT_RHO", "Refer_Rho"), 0.0)
    rho0 = float(rho0_val)

    refer_rho_val = _extract_param(p, ("refer_rho", "Refer_Rho", "rho_ref", "refer_density"), rho0)
    refer_rho = float(refer_rho_val) if refer_rho_val is not None and float(refer_rho_val) != 0.0 else rho0

    shear = float(_extract_param(p, ("shear", "G", "g", "MAT_G"), 0.0))
    aa = float(_extract_param(p, ("aa", "a", "A", "MAT_A"), 0.0))
    bb = float(_extract_param(p, ("bb", "b", "B", "MAT_B"), 0.0))
    nn = float(_extract_param(p, ("nn", "n", "N", "MAT_N"), 1.0))
    fc = float(_extract_param(p, ("fc", "FC", "MAT_FC"), 0.0))
    t0 = float(_extract_param(p, ("t0", "t", "T0", "T", "MAT_T0"), 0.0))

    cc = float(_extract_param(p, ("cc", "c", "C", "MAT_C"), 0.0))
    eps0_val = _extract_param(p, ("eps0", "EPS0", "MAT_EPS0"), 1.0)
    eps0 = float(eps0_val) if eps0_val is not None and float(eps0_val) != 0.0 else 1.0
    fcut = float(_extract_param(p, ("fcut", "FCUT", "asrate", "MAT_FCUT"), 0.0))
    sfmax_val = _extract_param(p, ("sfmax", "SFMAX", "MAT_SFMAX"), _EP30)
    sfmax = float(sfmax_val) if sfmax_val is not None and float(sfmax_val) not in (0.0, _EP30) else _EP30
    efmin_val = _extract_param(p, ("efmin", "EFMIN", "MAT_EFMIN"), _EM20)
    efmin = float(efmin_val) if efmin_val is not None and float(efmin_val) != 0.0 else _EM20

    pc = float(_extract_param(p, ("pc", "PC", "MAT_PC"), 0.0))
    muc = float(_extract_param(p, ("muc", "MUC", "MAT_MUC"), 0.0))
    pl = float(_extract_param(p, ("pl", "PL", "MAT_PL"), 0.0))
    mul = float(_extract_param(p, ("mul", "MUL", "MAT_MUL"), 0.0))

    k1 = float(_extract_param(p, ("k1", "K1", "MAT_K1"), 0.0))
    k2 = float(_extract_param(p, ("k2", "K2", "MAT_K2"), 0.0))
    k3 = float(_extract_param(p, ("k3", "K3", "MAT_K3"), 0.0))

    d1 = float(_extract_param(p, ("d1", "D1", "MAT_D1"), 0.0))
    d2 = float(_extract_param(p, ("d2", "D2", "MAT_D2"), 1.0))
    idel = int(_extract_param(p, ("idel", "IDEL"), 0))
    epsmax_val = _extract_param(p, ("eps_max", "epsmax", "EPS_MAX", "EPSMAX", "MAT_EPSMAX"), _EP30)
    eps_max = float(epsmax_val) if epsmax_val is not None and float(epsmax_val) not in (0.0, _EP30) else _EP30
    ifailso = int(_extract_param(p, ("ifailso", "IFAILSO"), 1))

    cst = float(_extract_param(p, ("cst", "ct", "CT", "MAT_CT"), 0.0))
    powt = float(_extract_param(p, ("powt", "POWT", "MAT_POWT"), 1.0))
    csc = float(_extract_param(p, ("csc", "cc_cs", "CC", "MAT_CC"), 0.0))
    powc = float(_extract_param(p, ("powc", "POWC", "MAT_POWC"), 1.0))

    return Law126Params(
        rho0=rho0,
        refer_rho=refer_rho,
        shear=shear,
        aa=aa,
        bb=bb,
        nn=nn,
        fc=fc,
        t0=t0,
        cc=cc,
        eps0=eps0,
        fcut=fcut,
        sfmax=sfmax,
        efmin=efmin,
        pc=pc,
        muc=muc,
        pl=pl,
        mul=mul,
        k1=k1,
        k2=k2,
        k3=k3,
        d1=d1,
        d2=d2,
        idel=idel,
        eps_max=eps_max,
        ifailso=ifailso,
        cst=cst,
        powt=powt,
        csc=csc,
        powc=powc,
        title=title,
    )


def build_law126(rec: Any) -> Material:
    """Build a Material instance for /MAT/LAW126 (Johnson-Holmquist Concrete).

    Receives GenericMaterialRecord or dictionary and constructs Material
    with consistent elastic and HJC plastic parameters.
    """
    mat_id = getattr(rec, "id", 1)
    title = getattr(rec, "title", "")
    params_dict = dict(getattr(rec, "params", {})) if hasattr(rec, "params") and rec.params else {}
    if hasattr(rec, "density") and rec.density:
        params_dict["rho0"] = rec.density

    p = _get_params(params_dict)
    p.title = title

    params: Dict[str, Any] = {
        "E": p.young if p.young > 0.0 else 1.0,
        "nu": p.nu,
        "G": p.shear,
        "K": p.k0,
        "bulk": p.k0,
        "rho0": p.rho0,
        "MAT_RHO": p.rho0,
        "MAT_G": p.shear,
        "MAT_A": p.aa,
        "MAT_B": p.bb,
        "MAT_N": p.nn,
        "MAT_FC": p.fc,
        "MAT_T0": p.t0,
        "MAT_C": p.cc,
        "MAT_EPS0": p.eps0,
        "MAT_FCUT": p.fcut,
        "MAT_SFMAX": p.sfmax,
        "MAT_EFMIN": p.efmin,
        "MAT_PC": p.pc,
        "MAT_MUC": p.muc,
        "MAT_PL": p.pl,
        "MAT_MUL": p.mul,
        "MAT_K1": p.k1,
        "MAT_K2": p.k2,
        "MAT_K3": p.k3,
        "MAT_D1": p.d1,
        "MAT_D2": p.d2,
        "IDEL": p.idel,
        "MAT_EPSMAX": p.eps_max,
        "IFAILSO": p.ifailso,
        "MAT_CT": p.cst,
        "MAT_POWT": p.powt,
        "MAT_CC": p.csc,
        "MAT_POWC": p.powc,
        "law126_params": p,
    }

    mat = Material(
        id=mat_id,
        law=126,
        rho0=p.rho0,
        title=title,
        params=params,
    )
    mat.law126_params = p
    return mat


def extra_shapes(mat: Any, nip: Optional[int] = None) -> Dict[str, Tuple[int, ...]]:
    """Define persistent internal state variables for LAW126.

    Corresponds to:
    - uvar: (4,) [mup, phard, sigy, noff] (sigeps126.F90 lines 68 & 385-388)
    - dmg: () scalar damage parameter D
    - off: () scalar element deletion status (1.0 alive, decays to 0)
    - defp: () accumulated equivalent plastic strain
    """
    if nip is not None:
        return {
            "uvar126": (nip, 4),
            "dmg126": (nip,),
            "off126": (nip,),
            "defp126": (nip,),
        }
    return {
        "uvar126": (4,),
        "dmg126": (),
        "off126": (),
        "defp126": (),
    }


def sound_speed(
    mat: Any,
    rho: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Any:
    """Longitudinal acoustic sound speed for LAW126 solids (sigeps126.F90 line 394):

    c = sqrt((dpdmu + 4/3 * G) / rho0)
    """
    p = _get_params(mat)
    rho_val = float(np.mean(rho)) if rho is not None and np.size(rho) > 0 else p.rho0
    if rho_val <= 0.0:
        rho_val = p.rho0 if p.rho0 > 0.0 else 1.0

    dpdmu = p.k0
    if extra is not None and "uvar" in extra:
        uvar = extra["uvar"]
        mup = uvar[..., 0] if hasattr(uvar, "ndim") and uvar.ndim >= 2 else (uvar[0] if isinstance(uvar, (list, np.ndarray)) and len(uvar) > 0 else 0.0)
        mup_val = float(np.mean(mup))
        if mup_val < p.mul and p.mul > 0.0:
            dpdmu = p.k0 + (p.k1 - p.k0) * (mup_val / p.mul)
        elif mup_val >= p.mul and p.mul > 0.0:
            amu_val = float(np.mean(extra.get("amu", 0.0)))
            mubar = (amu_val - p.mul) / (1.0 + p.mul)
            dpdmu = (p.k1 + 2.0 * p.k2 * mubar + 3.0 * p.k3 * (mubar**2)) / (1.0 + p.mul)

    dpdmu = max(0.0, dpdmu)
    return math.sqrt(max(0.0, (dpdmu + 4.0 / 3.0 * p.shear) / rho_val))


def solid_step(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Fortran-faithful stress update for LAW126 3D solids (sigeps126.F90).

    Parameters
    ----------
    mat : Material or Law126Params
        Material parameters instance.
    sig : ndarray, shape (6,) or (n, 6)
        Initial Cauchy stress [xx, yy, zz, xy, yz, zx].
    deps : ndarray, shape (6,) or (n, 6)
        Strain increment [xx, yy, zz, xy, yz, zx] (engineering shears).
    epsp : ndarray, optional
        Accumulated equivalent plastic strain.
    dt : float
        Current time step size.
    extra : dict, optional
        State variables dictionary ('uvar', 'dmg', 'off', 'amu', 'epsd').

    Returns
    -------
    (sign, epsp, ssp) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        sign : Updated Cauchy stress.
        epsp : Updated accumulated equivalent plastic strain.
        ssp  : Longitudinal sound speed.
    """
    p = _get_params(mat)

    is_1d = (sig.ndim == 1)
    sig_arr = np.atleast_2d(sig).copy()
    deps_arr = np.atleast_2d(deps).copy()
    nel = sig_arr.shape[0]

    if epsp is None:
        epsp_arr = np.zeros(nel, dtype=float)
    else:
        epsp_arr = np.atleast_1d(epsp).astype(float).copy()

    if extra is None:
        extra = {}

    # State extraction / initialization (lines 156-165)
    uvar = extra.get("uvar126", extra.get("uvar"))
    if uvar is None:
        uvar = np.zeros((nel, 4), dtype=float)
        uvar[:, 1] = p.pc
        uvar[:, 2] = p.fc * p.aa
        extra["uvar"] = uvar
        extra["uvar126"] = uvar
    else:
        uvar = np.atleast_2d(uvar)
        if uvar.shape[0] != nel or uvar.shape[1] < 4:
            new_uvar = np.zeros((nel, 4), dtype=float)
            new_uvar[:, 1] = p.pc
            new_uvar[:, 2] = p.fc * p.aa
            uvar = new_uvar
            extra["uvar"] = uvar
            extra["uvar126"] = uvar

    dmg = extra.get("dmg126", extra.get("dmg"))
    if dmg is None:
        dmg = np.zeros(nel, dtype=float)
        extra["dmg"] = dmg
        extra["dmg126"] = dmg
    else:
        dmg = np.atleast_1d(dmg).astype(float).copy()

    off = extra.get("off126", extra.get("off"))
    if off is None:
        off = np.ones(nel, dtype=float)
        extra["off"] = off
        extra["off126"] = off
    else:
        off = np.atleast_1d(off).astype(float).copy()

    # Volumetric strain amu = trace(deps) accumulation or from extra
    amu = extra.get("amu")
    if amu is None:
        rho_curr = extra.get("rho")
        if rho_curr is not None and p.rho0 > 0.0:
            amu = np.asarray(rho_curr) / p.rho0 - 1.0
        else:
            amu_prev = extra.get("amu_prev", np.zeros(nel, dtype=float))
            dav_trace = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2])
            amu = amu_prev + dav_trace
            extra["amu_prev"] = amu
    amu = np.atleast_1d(amu).astype(float)

    # Strain rate epsd
    epsd = extra.get("epsd")
    if epsd is None:
        if dt > 0.0:
            dav_all = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0
            dexx = deps_arr[:, 0] - dav_all
            deyy = deps_arr[:, 1] - dav_all
            dezz = deps_arr[:, 2] - dav_all
            de_dev2 = (2.0 / 3.0) * (dexx**2 + deyy**2 + dezz**2 + 0.5 * (deps_arr[:, 3]**2 + deps_arr[:, 4]**2 + deps_arr[:, 5]**2))
            epsd = np.sqrt(np.maximum(0.0, de_dev2)) / dt
        else:
            epsd = np.zeros(nel, dtype=float)
    epsd = np.atleast_1d(epsd).astype(float)

    # Local parameters
    g = p.shear
    g2 = 2.0 * g
    k0 = p.k0
    aa = p.aa
    bb = p.bb
    nn = p.nn
    fc = p.fc
    t0 = p.t0
    cc = p.cc
    eps0 = p.eps0
    sfmax = p.sfmax
    efmin = p.efmin
    pc = p.pc
    muc = p.muc
    pl = p.pl
    mul = p.mul
    k1 = p.k1
    k2 = p.k2
    k3 = p.k3
    d1 = p.d1
    d2 = p.d2
    emax = p.eps_max
    h = p.h
    cst = p.cst
    powt = p.powt
    csc = p.csc
    powc = p.powc
    icowpsym = 1 if cst > 0.0 else 0
    idel = p.idel
    ifail = p.ifailso

    mup = uvar[:, 0].copy()
    phard = uvar[:, 1].copy()
    sigy_prev = uvar[:, 2].copy()
    noff = uvar[:, 3].astype(int).copy()

    dpla = np.zeros(nel, dtype=float)
    dmup = np.zeros(nel, dtype=float)
    dmg_on = (d1 > 0.0)

    # =========================================================================
    # Computation of elastic deviatoric stresses and equivalent stress (lines 170-182)
    # =========================================================================
    dav = (deps_arr[:, 0] + deps_arr[:, 1] + deps_arr[:, 2]) / 3.0
    pold = - (sig_arr[:, 0] + sig_arr[:, 1] + sig_arr[:, 2]) / 3.0

    s_trial = np.zeros_like(sig_arr)
    s_trial[:, 0] = sig_arr[:, 0] + pold + g2 * (deps_arr[:, 0] - dav)
    s_trial[:, 1] = sig_arr[:, 1] + pold + g2 * (deps_arr[:, 1] - dav)
    s_trial[:, 2] = sig_arr[:, 2] + pold + g2 * (deps_arr[:, 2] - dav)
    s_trial[:, 3] = sig_arr[:, 3] + g * deps_arr[:, 3]
    s_trial[:, 4] = sig_arr[:, 4] + g * deps_arr[:, 4]
    s_trial[:, 5] = sig_arr[:, 5] + g * deps_arr[:, 5]

    j2 = 0.5 * (s_trial[:, 0]**2 + s_trial[:, 1]**2 + s_trial[:, 2]**2) + (
        s_trial[:, 3]**2 + s_trial[:, 4]**2 + s_trial[:, 5]**2
    )
    vm = np.sqrt(np.maximum(0.0, 3.0 * j2))

    # =========================================================================
    # Computation of pressure (3 regions EOS) (lines 238-265)
    # =========================================================================
    pmin = -t0
    pnew = np.zeros(nel, dtype=float)
    dpdmu = np.zeros(nel, dtype=float)

    for i in range(nel):
        if mup[i] < mul:
            kav = k0 + (k1 - k0) * (mup[i] / mul) if mul > 0.0 else k0
            pnew[i] = kav * (amu[i] - mup[i])
            if pnew[i] > phard[i]:
                denom = kav + h
                dmup[i] = (pnew[i] - phard[i]) / denom if denom > 0.0 else 0.0
                mup[i] = min(mup[i] + dmup[i], mul)
                pnew[i] = pnew[i] - kav * dmup[i]
                phard[i] = pnew[i]
            dpdmu[i] = kav
        else:
            mubar = (amu[i] - mul) / (1.0 + mul)
            pnew[i] = k1 * mubar + k2 * (mubar**2) + k3 * (mubar**3)
            dpdmu[i] = (k1 + 2.0 * k2 * mubar + 3.0 * k3 * (mubar**2)) / (1.0 + mul)

        # Tensile pressure clamp
        pnew[i] = max(pnew[i], pmin)

    pstar = pnew / fc if fc > 0.0 else np.zeros(nel, dtype=float)

    # =========================================================================
    # Computation of deviatoric yield stress (lines 270-297)
    # =========================================================================
    sigy = np.zeros(nel, dtype=float)
    for i in range(nel):
        if pstar[i] > 0.0:
            sigy[i] = aa * (1.0 - dmg[i]) + bb * (pstar[i]**nn)
            if icowpsym == 1:
                sigy[i] *= (1.0 + ((epsd[i] + _EM20) / csc)**(1.0 / powc))
            else:
                if epsd[i] > eps0:
                    sigy[i] *= (1.0 + cc * math.log(epsd[i] / eps0))
            sigy[i] = min(sfmax, sigy[i])
        else:
            tens_factor = max(0.0, 1.0 + pnew[i] / t0) if t0 > 0.0 else 0.0
            sigy[i] = aa * tens_factor * (1.0 - dmg[i])
            if icowpsym == 1:
                sigy[i] *= (1.0 + ((epsd[i] + _EM20) / cst)**(1.0 / powt))
            else:
                if epsd[i] > eps0:
                    sigy[i] *= (1.0 + cc * math.log(epsd[i] / eps0))

    # =========================================================================
    # Radial return mapping for deviatoric stress tensor (lines 302-325)
    # =========================================================================
    scale = np.ones(nel, dtype=float)
    sign_dev = np.zeros_like(s_trial)

    for i in range(nel):
        if off[i] == 1.0 and noff[i] == 0:
            sigstar = vm[i] / fc if fc > 0.0 else 0.0
            if sigstar < sigy[i]:
                scale[i] = 1.0
            elif vm[i] > 0.0:
                scale[i] = (sigy[i] * fc) / vm[i]
            else:
                scale[i] = 0.0

            sign_dev[i, :] = scale[i] * s_trial[i, :]
            dpla[i] = (1.0 - scale[i]) * vm[i] / (3.0 * g) if g > 0.0 else 0.0
            epsp_arr[i] += dpla[i]
        else:
            sign_dev[i, :] = s_trial[i, :]

    # =========================================================================
    # Update plastic strain and damage (lines 330-376)
    # =========================================================================
    for i in range(nel):
        if off[i] == 1.0 and noff[i] == 0 and dmg_on:
            tstar_val = t0 / fc if fc > 0.0 else 0.0
            p_sum = pstar[i] + tstar_val
            if p_sum >= 0.0:
                epfail = d1 * (p_sum**d2)
            else:
                epfail = 0.0
            epfail = max(epfail, efmin)
            dmg_incr = max(dpla[i] + dmup[i], 0.0) / epfail
            dmg[i] = min(1.0, dmg[i] + dmg_incr)

        # Check element deletion triggers (lines 348-375)
        if off[i] == 1.0 and noff[i] == 0:
            if idel == 1:
                tstar_val = t0 / fc if fc > 0.0 else 0.0
                if (pstar[i] + tstar_val) <= 0.0:
                    noff[i] = 1
            elif idel == 2:
                if epsp_arr[i] > emax:
                    noff[i] = 1
            elif idel == 3:
                if fc * sigy[i] <= 0.0:
                    noff[i] = 1
            elif idel == 4:
                if dmg[i] >= 1.0:
                    noff[i] = 1

    # =========================================================================
    # Update total stress tensor and sound speed (lines 381-409)
    # =========================================================================
    sign = np.zeros_like(sig_arr)
    ssp = np.zeros(nel, dtype=float)

    for i in range(nel):
        uvar[i, 0] = mup[i]
        uvar[i, 1] = phard[i]
        uvar[i, 2] = fc * sigy[i]
        uvar[i, 3] = float(noff[i])

        # Total stress = deviator - pressure * I
        sign[i, 0] = sign_dev[i, 0] - pnew[i]
        sign[i, 1] = sign_dev[i, 1] - pnew[i]
        sign[i, 2] = sign_dev[i, 2] - pnew[i]
        sign[i, 3] = sign_dev[i, 3]
        sign[i, 4] = sign_dev[i, 4]
        sign[i, 5] = sign_dev[i, 5]

        # Wave speed
        rho_val = p.rho0 if p.rho0 > 0.0 else 1.0
        ssp[i] = math.sqrt(max(0.0, (dpdmu[i] + (4.0 / 3.0) * g) / rho_val))

    # =========================================================================
    # Post-failure behavior (lines 413-469)
    # =========================================================================
    for i in range(nel):
        if ifail == 1:
            if off[i] < 0.1:
                off[i] = 0.0
            if off[i] < 1.0:
                off[i] = off[i] * 0.8
            if noff[i] == 1 and off[i] == 1.0:
                off[i] = 0.8
            if off[i] < 1.0:
                sign[i, :] *= off[i]
        elif ifail == 2:
            if noff[i] == 1:
                sign[i, 0] = -pnew[i]
                sign[i, 1] = -pnew[i]
                sign[i, 2] = -pnew[i]
                sign[i, 3:] = 0.0
        elif ifail == 3:
            if noff[i] == 1:
                p_comp = max(pnew[i], 0.0)
                sign[i, 0] = -p_comp
                sign[i, 1] = -p_comp
                sign[i, 2] = -p_comp
                sign[i, 3:] = 0.0
        elif ifail == 4:
            if noff[i] == 1:
                sign[i, :] = 0.0
        elif ifail == 5:
            if noff[i] == 1 and pnew[i] < 0.0:
                sign[i, :] = 0.0

    # Write back state into extra
    extra["uvar"] = uvar
    extra["uvar126"] = uvar
    extra["dmg"] = dmg
    extra["dmg126"] = dmg
    extra["off"] = off
    extra["off126"] = off
    extra["dpla"] = dpla

    out_sig = sign[0] if is_1d else sign
    out_epsp = epsp_arr[0] if is_1d else epsp_arr
    out_ssp = ssp[0] if is_1d else ssp

    return out_sig, out_epsp, out_ssp


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Solid stress update entry point for LAW126."""
    return solid_step(mat, sig, deps, epsp=epsp, dt=dt, extra=extra, **kwargs)


def solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent stiffness tensor C (6, 6) for LAW126 solids."""
    p = _get_params(mat)
    k0 = p.k0
    g = p.shear
    c_elastic = np.zeros((6, 6), dtype=float)
    lam = k0 - (2.0 / 3.0) * g
    c_elastic[0, 0] = c_elastic[1, 1] = c_elastic[2, 2] = lam + 2.0 * g
    c_elastic[0, 1] = c_elastic[0, 2] = c_elastic[1, 0] = lam
    c_elastic[1, 2] = c_elastic[2, 0] = c_elastic[2, 1] = lam
    c_elastic[3, 3] = c_elastic[4, 4] = c_elastic[5, 5] = g

    if sig is None or deps is None:
        return c_elastic

    sig_arr = np.asarray(sig, dtype=float)
    deps_arr = np.asarray(deps, dtype=float)

    if sig_arr.ndim == 2 and sig_arr.shape[0] > 1:
        n = sig_arr.shape[0]
        return np.broadcast_to(c_elastic, (n, 6, 6)).copy()

    sig0 = sig_arr.flatten()
    deps0 = deps_arr.flatten()
    if len(sig0) < 6:
        s_pad = np.zeros(6, dtype=float)
        s_pad[:len(sig0)] = sig0
        sig0 = s_pad
    if len(deps0) < 6:
        d_pad = np.zeros(6, dtype=float)
        d_pad[:len(deps0)] = deps0
        deps0 = d_pad

    ex0 = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
    sig_base, _, _ = solid_step(mat, sig0, deps0, dt=dt, extra=ex0)

    h = 1.0e-7
    c_algo = np.zeros((6, 6), dtype=float)
    for j in range(6):
        deps_p = deps0.copy()
        deps_p[j] += h
        ex_p = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in (extra or {}).items()}
        sig_p, _, _ = solid_step(mat, sig0, deps_p, dt=dt, extra=ex_p)
        c_algo[:, j] = (sig_p - sig_base) / h

    return c_algo


def consistent_solid_tangent(
    mat: Any,
    sig: Optional[np.ndarray] = None,
    deps: Optional[np.ndarray] = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic solid tangent alias."""
    return solid_tangent(mat, sig=sig, deps=deps, dt=dt, extra=extra, **kwargs)


def _register():
    try:
        from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
        for k in (126, "126", "LAW126", "JOHNSON_HOLMQUIST_CONCRETE", "JH_CONC", "JHC", "MAT_LAW126"):
            MAT_PHYSICS_REGISTRY[k] = build_law126
    except Exception:
        pass


_register()
