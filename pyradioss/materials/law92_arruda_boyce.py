"""
LAW92 — Arruda-Boyce 8-chain hyperelastic polymer material (/MAT/LAW92, /MAT/ARRUDA_BOYCE).

Implements the Arruda-Boyce hyperelastic constitutive law for 3D solid continuum
elements and 2D shell / membrane elements in pure Python / NumPy.

Fortran reference sources:
- engine/source/materials/mat/mat092/sigeps92.F   (3D continuum solid kernel)
- starter/source/materials/mat/mat092/hm_read_mat92.F (starter reader & parameters)
- starter/source/materials/mat/mat092/law92_upd.F     (property update & curve processing)
- starter/source/materials/mat/mat092/law92_nlsqf.F90 (non-linear least-squares fitting)

Theory
------
The Arruda-Boyce (1993) 8-chain model represents network stretch through an
expansion of the inverse Langevin function with 5 terms:
    W = mu * sum_{j=1}^5 C_j * beta^(j-1) * (I_1^j - 3^j) + (K / 2) * (J - 1)^2 (or (K/2)*(J - 1/J))

Inverse Langevin coefficients:
    C_1 = 1/2
    C_2 = 1/20
    C_3 = 11/1050
    C_4 = 19/7000
    C_5 = 519/673750

where:
    beta = 1 / lambda_m^2  (LAM is the limiting chain stretch)
    I_1 = lambda_bar_1^2 + lambda_bar_2^2 + lambda_bar_3^2 (first isochoric invariant)
    lambda_bar_i = lambda_i * J^(-1/3) (isochoric principal stretches)
    J = lambda_1 * lambda_2 * lambda_3 = rho0 / rho (relative volume)

Ground-state initial shear modulus G_0 (hm_read_mat92.F:159-160):
    G_0 = mu * (1 + 3/5*beta + 99/175*beta^2 + 513/875*beta^3 + 42039/67375*beta^4)

Bulk modulus (hm_read_mat92.F:172):
    K = RBULK = 2 / D
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

_EM10 = 1e-10
_EM20 = 1e-20
_EM30 = 1e-30

# Inverse Langevin expansion coefficients (sigeps92.F:114-118, hm_read_mat92.F:148-152)
C_LANGEVIN = (
    0.5,                    # C(1) = 1/2
    0.05,                   # C(2) = 1/20
    11.0 / 1050.0,          # C(3) = 11/1050
    19.0 / 7000.0,          # C(4) = 19/7000
    519.0 / 673750.0,       # C(5) = 519/673750
)


@dataclass
class ArrudaBoyceParams:
    """Parameters for /MAT/LAW92 (/MAT/ARRUDA_BOYCE).

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
    mu : float
        Initial shear modulus factor.
    d : float
        Volumetric compressibility parameter D (K = 2 / D).
    lam : float
        Limiting network chain stretch lambda_m (default 7.0).
    itype : int
        Test type for curve fitting (1=uniaxial, 2=equibiaxial, 3=planar).
    fct_id : int
        Function ID for experimental test curve.
    nu : float
        Poisson's ratio (default 0.495).
    fscale : float
        Stress scale factor for test curve.
    g0 : float
        Initial ground-state shear modulus G_0.
    rbulk : float
        Initial bulk modulus K = 2 / D.
    """

    id: int = 1
    title: str = ""
    rho0: float = 1.0
    ref_rho: float = 0.0
    mu: float = 1.0
    d: float = 0.0
    lam: float = 7.0
    itype: int = 1
    fct_id: int = 0
    nu: float = 0.495
    fscale: float = 1.0
    g0: float = 1.0
    rbulk: float = 200.0
    curve_data: Any = None

    @property
    def K(self) -> float:
        """Bulk modulus alias."""
        return self.rbulk

    @property
    def G(self) -> float:
        """Ground-state shear modulus alias."""
        return self.g0

    @property
    def E(self) -> float:
        """Equivalent Young's modulus."""
        return 9.0 * self.rbulk * self.g0 / max(_EM20, 3.0 * self.rbulk + self.g0)

    @property
    def params(self) -> dict[str, Any]:
        """Dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "rho0": self.rho0,
            "ref_rho": self.ref_rho,
            "mu": self.mu,
            "d": self.d,
            "lam": self.lam,
            "itype": self.itype,
            "fct_id": self.fct_id,
            "nu": self.nu,
            "fscale": self.fscale,
            "g0": self.g0,
            "rbulk": self.rbulk,
            "G": self.g0,
            "K": self.rbulk,
            "E": self.E,
        }


def compute_ground_state_shear_modulus(mu: float, lam: float) -> float:
    """Compute initial ground-state shear modulus G_0.

    Fortran reference: hm_read_mat92.F:154-160
        A1 = ONE / LAM**2
        A2 = A1**2
        A3 = A1 * A2
        A4 = A2**2
        G = MU * (ONE + THREE*A1/FIVE + 99.0*A2/175.0 + 513.0*A3/875.0 + 42039.0*A4/67375.0)
    """
    if lam <= 0.0:
        lam = 7.0
    beta = 1.0 / (lam * lam)
    a1 = beta
    a2 = a1 * a1
    a3 = a1 * a2
    a4 = a2 * a2
    return float(mu * (1.0 + 0.6 * a1 + (99.0 / 175.0) * a2 + (513.0 / 875.0) * a3 + (42039.0 / 67375.0) * a4))


initial_shear_modulus = compute_ground_state_shear_modulus


def bulk_modulus(d: float = 0.0, nu: float = 0.495, g0: float = 0.0) -> float:
    """Compute bulk modulus K = 2 / D or from Poisson's ratio and shear modulus."""
    if d > 0.0:
        return 2.0 / d
    if nu < 0.5:
        return (2.0 / 3.0) * (1.0 + nu) * g0 / max(_EM30, (1.0 - 2.0 * nu))
    return 0.0


def build_law92(
    id: Any = 1,
    title: str = "",
    rho0: float = 1.0,
    ref_rho: float = 0.0,
    mu: float = 1.0,
    d: float = 0.0,
    lam: float = 7.0,
    itype: int = 1,
    fct_id: int = 0,
    nu: float = 0.495,
    fscale: float = 1.0,
    **kwargs: Any,
) -> ArrudaBoyceParams:
    """Build and initialize ArrudaBoyceParams according to starter hm_read_mat92.F.

    Handles record/dict/entity unpacking or direct keyword construction.
    """
    if not isinstance(id, (int, np.integer, float)):
        rec = id
        mid = int(getattr(rec, "id", 1))
        rho0 = float(getattr(rec, "density", getattr(rec, "rho0", 1.0)) or 1.0)
        title = str(getattr(rec, "title", ""))
        p = getattr(rec, "params", {})
        if p and isinstance(p, dict):
            mu = float(p.get("MAT_MUE1", p.get("mu", getattr(rec, "mu", 1.0))))
            d = float(p.get("MAT_D", p.get("d", getattr(rec, "d", 0.0))))
            lam = float(p.get("MAT_Lamda", p.get("lam", getattr(rec, "lam", 7.0))))
            itype = int(p.get("Itype", p.get("itype", getattr(rec, "itype", 1))))
            fct_id = int(p.get("MAT_FCT_IDI", p.get("fct_id", getattr(rec, "fct_id", 0))))
            nu = float(p.get("MAT_NU", p.get("nu", getattr(rec, "nu", 0.495))))
            fscale = float(p.get("MAT_FScale", p.get("fscale", getattr(rec, "fscale", 1.0))))
            ref_rho = float(p.get("refer_rho", getattr(rec, "ref_rho", 0.0)))
        else:
            mu = float(getattr(rec, "mu", 1.0))
            d = float(getattr(rec, "d", 0.0))
            lam = float(getattr(rec, "lam", 7.0))
            itype = int(getattr(rec, "itype", 1))
            fct_id = int(getattr(rec, "fct_id", 0))
            nu = float(getattr(rec, "nu", 0.495))
            fscale = float(getattr(rec, "fscale", 1.0))
            ref_rho = float(getattr(rec, "ref_rho", 0.0))
        id = mid

    if "mu" in kwargs:
        mu = float(kwargs["mu"])
    if "d" in kwargs:
        d = float(kwargs["d"])
    if "lam" in kwargs:
        lam = float(kwargs["lam"])
    if "nu" in kwargs:
        nu = float(kwargs["nu"])
    if "rho" in kwargs:
        rho0 = float(kwargs["rho"])
    if "rho0" in kwargs:
        rho0 = float(kwargs["rho0"])

    if lam <= 0.0:
        lam = 7.0
    if itype == 0:
        itype = 1
    if fscale == 0.0:
        fscale = 1.0

    g0 = compute_ground_state_shear_modulus(mu, lam)

    # Bulk modulus & Poisson's ratio logic (hm_read_mat92.F:163-173)
    nu_final = float(nu)
    if fct_id != 0:
        if nu_final <= 0.0:
            nu_final = 0.495
        if g0 > 0.0 and d == 0.0:
            rbulk = (2.0 / 3.0) * (1.0 + nu_final) * g0 / max(_EM30, (1.0 - 2.0 * nu_final))
            d = 2.0 / rbulk if rbulk > 0.0 else 0.0
        elif d > 0.0:
            rbulk = 2.0 / d
        else:
            rbulk = 0.0
    elif d == 0.0:
        if nu_final <= 0.0:
            nu_final = 0.495
        rbulk = (2.0 / 3.0) * (1.0 + nu_final) * g0 / max(_EM30, (1.0 - 2.0 * nu_final))
        d = 2.0 / rbulk if rbulk > 0.0 else 0.0
    else:
        rbulk = 2.0 / d
        if nu_final == 0.0 and (3.0 * rbulk + g0) > 0.0:
            nu_final = (3.0 * rbulk - 2.0 * g0) / (2.0 * (3.0 * rbulk + g0))

    return ArrudaBoyceParams(
        id=int(id),
        title=str(title),
        rho0=float(rho0),
        ref_rho=float(ref_rho),
        mu=float(mu),
        d=float(d),
        lam=float(lam),
        itype=int(itype),
        fct_id=int(fct_id),
        nu=float(nu_final),
        fscale=float(fscale),
        g0=float(g0),
        rbulk=float(rbulk),
    )


def arruda_boyce_analytical_stress(stretch: float, mu: float, lam: float, itype: int = 1) -> float:
    """Calculate nominal/engineering tensile stress for test curves.

    Fortran reference: starter/source/materials/mat/mat092/law92_nlsqf.F90:450-498
    """
    if stretch <= 0.0:
        return 0.0
    c = C_LANGEVIN
    sig = 0.0
    if itype == 1:
        # Uniaxial test: lambda_1 = stretch, lambda_2 = lambda_3 = 1/sqrt(stretch)
        ev2 = 1.0 / stretch
        trace = stretch * stretch + 2.0 * ev2
        fac = 2.0 * mu * (stretch - ev2 * ev2)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c[j]
            sig += aa * bb * (trace ** j)
        sig *= fac
    elif itype == 2:
        # Equibiaxial test: lambda_1 = lambda_2 = stretch, lambda_3 = 1/stretch^2
        ev = 1.0 / (stretch * stretch)
        trace = 2.0 * stretch * stretch + ev * ev
        fac = 4.0 * mu * (stretch - (ev * ev) / stretch)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c[j]
            sig += aa * bb * (trace ** j)
        sig *= fac
    elif itype == 3:
        # Planar shear test: lambda_1 = stretch, lambda_2 = 1, lambda_3 = 1/stretch
        ev = 1.0 / stretch
        trace = stretch * stretch + 1.0 + ev * ev
        fac = 2.0 * mu * (stretch - (ev * ev) / stretch)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c[j]
            sig += aa * bb * (trace ** j)
        sig *= fac
    return float(sig)


def arruda_boyce_dyda(stretch: float, mu: float, lam: float, itype: int = 1) -> tuple[float, float]:
    """Calculate derivatives of nominal tensile stress w.r.t. mu and lam.

    Fortran reference: starter/source/materials/mat/mat092/law92_nlsqf.F90:535-582
    """
    if stretch <= 0.0 or lam <= 0.0:
        return 0.0, 0.0
    c = C_LANGEVIN
    dyda1 = 0.0
    dyda2 = 0.0
    if itype == 1:
        ev2 = 1.0 / stretch
        trace = stretch * stretch + 2.0 * ev2
        fac = 2.0 * (stretch - ev2 * ev2)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c[j]
            cc = aa * bb * (trace ** j)
            dd = -2.0 * j * cc / lam
            dyda1 += cc
            dyda2 += dd
        dyda1 *= fac
        dyda2 *= fac * mu
    elif itype == 2:
        ev = 1.0 / (stretch * stretch)
        trace = 2.0 * stretch * stretch + ev * ev
        fac = 4.0 * (stretch - (ev * ev) / stretch)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c[j]
            cc = aa * bb * (trace ** j)
            dd = -2.0 * j * cc / lam
            dyda1 += cc
            dyda2 += dd
        dyda1 *= fac
        dyda2 *= fac * mu
    elif itype == 3:
        ev = 1.0 / stretch
        trace = stretch * stretch + 1.0 + ev * ev
        fac = 2.0 * (stretch - (ev * ev) / stretch)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c[j]
            cc = aa * bb * (trace ** j)
            dd = -2.0 * j * cc / lam
            dyda1 += cc
            dyda2 += dd
        dyda1 *= fac
        dyda2 *= fac * mu
    return float(dyda1), float(dyda2)


def fit_curve_arruda_boyce(
    stretches: np.ndarray,
    stresses: np.ndarray,
    itype: int = 1,
    nu: float = 0.495,
    return_all: bool = False,
    **kwargs: Any,
) -> tuple[float, float] | tuple[float, float, float, float]:
    """Fit Arruda-Boyce parameters (mu, lam) from experimental test curves.

    Fortran reference: starter/source/materials/mat/mat092/law92_nlsqf.F90
    Returns (mu, lam) by default, or (mu, d, lam, g0) if return_all=True.
    """
    if "itest" in kwargs:
        itype = int(kwargs["itest"])

    stretches = np.asarray(stretches, dtype=np.float64).flatten()
    stresses = np.asarray(stresses, dtype=np.float64).flatten()

    ndata = len(stretches)
    if ndata < 2:
        mu_def = 1.0
        lam_def = 7.0
        g0_def = compute_ground_state_shear_modulus(mu_def, lam_def)
        rbulk_def = (2.0 / 3.0) * (1.0 + nu) * g0_def / max(_EM30, (1.0 - 2.0 * nu))
        if return_all:
            return mu_def, 2.0 / rbulk_def, lam_def, g0_def
        return mu_def, lam_def

    # Initial guess estimation (law92_upd.F:108-128)
    lam_max = float(np.max(np.abs(stretches)))
    ave_slope = 0.0
    mu_max = 0.0
    for k in range(ndata):
        dx = abs(stretches[k] - 1.0)
        dx = max(dx, 1e-6)
        slope = abs(stresses[k]) / dx
        mu_max = max(mu_max, slope)
        ave_slope += slope
    ave_slope /= max(1, ndata)
    mu_init = max(ave_slope, mu_max * 0.1)
    lam_init = max(7.0, 3.0 * lam_max)

    # Levenberg-Marquardt fitting matching mrqmin_law92
    a = np.array([mu_init, lam_init], dtype=np.float64)
    gamma = 0.01

    for _ in range(50):
        # Evaluate current model and residuals
        y_mod = np.array([arruda_boyce_analytical_stress(s, a[0], a[1], itype) for s in stretches])
        dy = stresses - y_mod
        err = np.sum(dy * dy)

        # Build normal equations: alpha and beta (mrqcof_law92)
        alpha = np.zeros((2, 2), dtype=np.float64)
        beta = np.zeros(2, dtype=np.float64)
        for i in range(ndata):
            d1, d2 = arruda_boyce_dyda(stretches[i], a[0], a[1], itype)
            dyda = np.array([d1, d2], dtype=np.float64)
            alpha += np.outer(dyda, dyda)
            beta += dy[i] * dyda

        # Marquardt damping: COVAR(J,J) = ALPHA(J,J) * (1 + GAMMA) + eps
        covar = alpha.copy()
        for j in range(2):
            covar[j, j] += gamma * (abs(alpha[j, j]) + 1.0)

        try:
            da = np.linalg.solve(covar, beta)
        except np.linalg.LinAlgError:
            gamma *= 10.0
            continue

        a_try = a + da
        if a_try[0] <= 0.0 or a_try[1] <= 1.0:
            gamma *= 10.0
            continue

        y_try = np.array([arruda_boyce_analytical_stress(s, a_try[0], a_try[1], itype) for s in stretches])
        dy_try = stresses - y_try
        err_try = np.sum(dy_try * dy_try)

        if err_try < err:
            gamma = max(1e-7, gamma / 10.0)
            a = a_try
            if abs(err - err_try) / max(1.0, err) < 1e-6:
                break
        else:
            gamma = min(1e10, gamma * 10.0)

    mu_fit = float(max(1e-6, a[0]))
    lam_fit = float(max(1.01, a[1]))
    g0_fit = compute_ground_state_shear_modulus(mu_fit, lam_fit)
    rbulk_fit = (2.0 / 3.0) * (1.0 + nu) * g0_fit / max(_EM30, (1.0 - 2.0 * nu))
    d_fit = 2.0 / rbulk_fit

    if return_all:
        return mu_fit, d_fit, lam_fit, g0_fit
    return mu_fit, lam_fit


def resolve(mat: Any, model: Any, log: Any = None) -> Any:
    """Starter hook resolving curve references for /MAT/LAW92."""
    if not isinstance(mat, ArrudaBoyceParams):
        mat_params = build_law92(mat)
    else:
        mat_params = mat

    if mat_params.fct_id > 0:
        curve = None
        if hasattr(model, "curves") and mat_params.fct_id in model.curves:
            curve = model.curves[mat_params.fct_id]
        elif hasattr(model, "functions") and mat_params.fct_id in model.functions:
            curve = model.functions[mat_params.fct_id]

        if curve is not None and hasattr(curve, "x") and hasattr(curve, "y") and len(curve.x) > 1:
            x_arr = np.asarray(curve.x, dtype=np.float64)
            if np.min(x_arr) < 0.5:
                stretches = x_arr + 1.0
            else:
                stretches = x_arr
            stresses = np.asarray(curve.y, dtype=np.float64) * mat_params.fscale
            mu_fit, d_fit, lam_fit, g0_fit = fit_curve_arruda_boyce(
                stretches, stresses, itype=mat_params.itype, nu=mat_params.nu, return_all=True
            )
            mat_params.mu = mu_fit
            mat_params.d = d_fit
            mat_params.lam = lam_fit
            mat_params.g0 = g0_fit
            mat_params.rbulk = 2.0 / d_fit if d_fit > 0.0 else (2.0 / 3.0) * (1.0 + mat_params.nu) * g0_fit / (1.0 - 2.0 * mat_params.nu)
            mat_params.curve_data = (stretches, stresses)
            if hasattr(mat, "mu"):
                mat.mu = mu_fit
            if hasattr(mat, "d"):
                mat.d = d_fit
            if hasattr(mat, "lam"):
                mat.lam = lam_fit
            if hasattr(mat, "params") and isinstance(mat.params, dict):
                mat.params["mu"] = mu_fit
                mat.params["d"] = d_fit
                mat.params["lam"] = lam_fit
                mat.params["g0"] = g0_fit
                mat.params["G"] = g0_fit
                mat.params["K"] = mat_params.rbulk
                mat.params["rbulk"] = mat_params.rbulk
                mat.params["MAT_MUE1"] = mu_fit
                mat.params["MAT_D"] = d_fit
                mat.params["MAT_Lamda"] = lam_fit

    return mat_params


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
    """3D solid continuum Arruda-Boyce hyperelastic stress update.

    Fortran reference: engine/source/materials/mat/mat092/sigeps92.F
    Returns (sig_out, epsp, sound_speed) or (sig_out, hist_dict, sound_speed).
    """
    return_constitutive_dict = False
    if isinstance(mat, np.ndarray) and (sig is None or not isinstance(sig, np.ndarray)):
        # Constitutive / unit test call: solid_update(eps, mu=..., d=..., ...)
        eps_in = mat
        params_kw = kwargs.copy()
        if isinstance(sig, dict):
            params_kw.update(sig)
        mat = build_law92(**params_kw)
        sig_arr = np.zeros_like(eps_in) if eps_in.ndim > 1 else np.zeros((1, 6))
        deps_arr = np.zeros_like(sig_arr)
        eps_arr = eps_in[np.newaxis, :] if eps_in.ndim == 1 else eps_in
        epsp_arr = np.zeros(len(eps_arr))
        is_1d = (eps_in.ndim == 1)
        return_constitutive_dict = True
        if "rho" in kwargs and extra is None:
            extra = {"rho": kwargs["rho"]}
    else:
        if not isinstance(mat, ArrudaBoyceParams):
            mat = build_law92(mat, **kwargs)
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

    # Current strain tensor: preserve total strain via extra['eps92']
    if extra is not None and "eps92" in extra and extra["eps92"] is not None:
        if np.asarray(extra["eps92"]).ndim == 1 and deps_arr.shape[0] == 1:
            extra["eps92"] = np.asarray(extra["eps92"]) + deps_arr[0]
            tot_eps = np.asarray(extra["eps92"])[np.newaxis, :]
        else:
            extra["eps92"] = np.asarray(extra["eps92"]) + deps_arr
            tot_eps = np.asarray(extra["eps92"])
    elif eps is not None:
        tot_eps = eps_arr + deps_arr if deps is not None else eps_arr.copy()
        if extra is not None and isinstance(extra, dict):
            extra["eps92"] = tot_eps[0].copy() if is_1d else tot_eps.copy()
    else:
        tot_eps = deps_arr.copy()
        if extra is not None and isinstance(extra, dict):
            extra["eps92"] = tot_eps[0].copy() if is_1d else tot_eps.copy()

    mu = mat.mu
    lam = mat.lam
    g0 = mat.g0
    rbulk = mat.rbulk
    c_coeffs = C_LANGEVIN

    rho = mat.rho0
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        rho_val = np.asarray(extra["rho"])
        if rho_val.ndim == 0:
            rho_arr = np.full(nel, float(rho_val))
        else:
            rho_arr = rho_val
    else:
        rho_arr = np.full(nel, mat.rho0)

    # Mullins energy storage array if requested
    mullins_w = np.zeros(nel, dtype=np.float64)

    for i in range(nel):
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

        # Principal stretches (sigeps92.F:148-174)
        if ismstr in (0, 2, 4):
            # Logarithmic strain
            stretches = np.exp(evals)
        else:
            # Engineering strain
            stretches = evals + 1.0
        stretches = np.maximum(stretches, 1e-12)

        # Relative volume J = rho0 / rho = lambda_1 * lambda_2 * lambda_3
        rv = stretches[0] * stretches[1] * stretches[2]
        rv = max(rv, 1e-12)
        rvd = rv ** (-1.0 / 3.0)

        # Isochoric (deviatoric) principal stretches
        evd = stretches * rvd
        trace = np.sum(evd * evd)

        di1lam = 2.0 * (evd * evd) - (2.0 / 3.0) * trace

        # Isochoric terms T_k (sigeps92.F:231-241)
        t_terms = np.zeros(3, dtype=np.float64)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c_coeffs[j]
            cc = aa * bb * (trace ** j)
            t_terms += cc * di1lam

        # Volumetric pressure P (sigeps92.F:244-245)
        rv_1 = 1.0 / rv
        p = rv_1 * rbulk * (rv * rv - 1.0) * 0.5

        # Principal Cauchy stresses (sigeps92.F:246-248)
        t_cauchy = (mu * t_terms + p) * rv_1

        # Mullins strain energy (sigeps92.F:251-260)
        beta = 1.0 / (lam * lam)
        mullins_w[i] = mu * (
            c_coeffs[0] * (trace - 3.0)
            + c_coeffs[1] * beta * (trace ** 2 - 9.0)
            + c_coeffs[2] * (beta ** 2) * (trace ** 3 - 27.0)
            + c_coeffs[3] * (beta ** 3) * (trace ** 4 - 81.0)
            + c_coeffs[4] * (beta ** 4) * (trace ** 5 - 243.0)
        )

        # Moduli for sound speed (sigeps92.F:261-287)
        cii = np.zeros(3, dtype=np.float64)
        for ii in range(1, 6):
            clam_ii = c_coeffs[ii - 1] / (lam ** (2 * ii - 2))
            clp = 4.0 * ii * clam_ii
            lam_2 = evd ** 2
            lam_4 = lam_2 ** 2
            aa_mod = (1.0 / 9.0) * ii * (trace ** ii)
            bb_mod = (1.0 / 3.0) * (3.0 - ii) * (trace ** (ii - 1)) if ii > 1 else 0.0
            cc_mod = (ii - 1.0) * (trace ** (ii - 2)) if ii > 2 else 0.0
            cii += clp * (aa_mod + bb_mod * lam_2 + cc_mod * lam_4)

        amax = float(np.max(cii))
        eti = max(1.0, amax * 0.81)
        gtmax = g0 * eti
        rkmax = 0.5 * rbulk * (1.0 + rv_1 * rv_1)
        rkmax = max(rbulk, rkmax)

        # Reconstruct Cauchy stress tensor in global coordinates (sigeps92.F:290-313)
        diag_t = np.diag(t_cauchy)
        sig_tensor = evecs @ diag_t @ evecs.T

        sig_out[i, 0] = sig_tensor[0, 0]
        sig_out[i, 1] = sig_tensor[1, 1]
        sig_out[i, 2] = sig_tensor[2, 2]
        sig_out[i, 3] = sig_tensor[0, 1]
        sig_out[i, 4] = sig_tensor[1, 2]
        sig_out[i, 5] = sig_tensor[2, 0]

        cur_rho = max(rho_arr[i], 1e-12)
        sound_sp[i] = np.sqrt(((4.0 / 3.0) * gtmax + rkmax) / cur_rho)

    if extra is not None:
        extra["mullins_w"] = mullins_w

    sig_res = sig_out[0] if is_1d else sig_out
    epsp_res = epsp_arr[0] if is_1d else epsp_arr
    sound_res = sound_sp[0] if is_1d else sound_sp

    if return_constitutive_dict:
        hist_dict = {"w_mullins": float(mullins_w[0])}
        return sig_res, hist_dict, float(sound_res)

    if return_tuple:
        return sig_res, epsp_res, sound_res
    return sig_res, epsp_res


def shell_update(
    mat: Any = None,
    sig: np.ndarray | None = None,
    deps: np.ndarray | None = None,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    *,
    eps: np.ndarray | None = None,
    **kwargs: Any,
) -> Any:
    """2D shell plane-stress Arruda-Boyce hyperelastic stress update.

    Enforces plane-stress condition sigma_zz = 0 iteratively via Newton-Raphson
    on out-of-plane stretch lambda_3, analogous to LAW82/LAW69/LAW88.
    """
    return_constitutive_shell = False
    if isinstance(mat, np.ndarray) and (sig is None or not isinstance(sig, np.ndarray)):
        # Direct constitutive call: shell_update(eps_inplane, mu=..., d=..., ...)
        eps_in = mat
        params_kw = kwargs.copy()
        if isinstance(sig, dict):
            params_kw.update(sig)
        mat = build_law92(**params_kw)
        is_1d = (eps_in.ndim == 1)
        sig_arr = np.zeros((1, 3) if is_1d else (len(eps_in), 3), dtype=np.float64)
        deps_arr = eps_in[np.newaxis, :3] if is_1d else eps_in[:, :3]
        epsp_arr = np.zeros(1 if is_1d else len(eps_in))
        return_constitutive_shell = True
        if "rho" in kwargs and extra is None:
            extra = {"rho": kwargs["rho"]}
    else:
        if not isinstance(mat, ArrudaBoyceParams):
            mat = build_law92(mat, **kwargs)
        if sig is None:
            sig = np.zeros(3, dtype=np.float64)
        is_1d = (sig.ndim == 1)
        if is_1d:
            sig_arr = sig[np.newaxis, :]
            deps_arr = deps[np.newaxis, :] if deps is not None else np.zeros((1, len(sig)))
            epsp_arr = epsp[np.newaxis] if epsp is not None else np.zeros(1)
        else:
            sig_arr = sig
            deps_arr = deps if deps is not None else np.zeros_like(sig)
            epsp_arr = epsp if epsp is not None else np.zeros(len(sig))

    nel = len(sig_arr)
    ncomp = sig_arr.shape[1]
    sig_out = np.zeros_like(sig_arr)

    # Accumulated in-plane total strain
    if return_constitutive_shell:
        eps_tot = deps_arr[:, :3]
    elif eps is not None:
        eps_in = np.asarray(eps, dtype=np.float64)
        eps_tot = eps_in[np.newaxis, :3] if eps_in.ndim == 1 else eps_in[:, :3]
    elif extra is not None and "eps92" in extra:
        extra["eps92"] += deps_arr[:, :3]
        eps_tot = extra["eps92"]
    elif extra is not None and "eps" in extra:
        eps_in = np.asarray(extra["eps"], dtype=np.float64)
        eps_tot = eps_in[np.newaxis, :3] if eps_in.ndim == 1 else eps_in[:, :3]
    elif "eps" in kwargs and kwargs["eps"] is not None:
        eps_in = np.asarray(kwargs["eps"], dtype=np.float64)
        eps_tot = eps_in[np.newaxis, :3] if eps_in.ndim == 1 else eps_in[:, :3]
    else:
        if extra is not None and isinstance(extra, dict):
            extra["eps92"] = deps_arr[:, :3].copy()
            eps_tot = extra["eps92"]
        else:
            eps_tot = deps_arr[:, :3]

    # History variable for out-of-plane stretch lambda_3
    if extra is not None and "uvar_lam3" in extra:
        lam3_old = np.asarray(extra["uvar_lam3"]).copy()
    else:
        lam3_old = np.ones(nel, dtype=np.float64)

    lam3_new = np.zeros(nel, dtype=np.float64)

    mu = mat.mu
    lam = mat.lam
    g0 = mat.g0
    rbulk = mat.rbulk
    c_coeffs = C_LANGEVIN

    for i in range(nel):
        exx = float(eps_tot[i, 0])
        eyy = float(eps_tot[i, 1])
        exy = float(eps_tot[i, 2]) if ncomp > 2 else 0.0

        mat_2d = np.array([
            [exx, 0.5 * exy],
            [0.5 * exy, eyy]
        ], dtype=np.float64)

        evals_2d, evecs_2d = np.linalg.eigh(mat_2d)
        evals_2d = np.clip(evals_2d, -10.0, 10.0)
        lam1 = max(1e-4, min(100.0, float(np.exp(evals_2d[0]))))
        lam2 = max(1e-4, min(100.0, float(np.exp(evals_2d[1]))))

        # Newton-Raphson solver for lambda_3 such that sigma_3(lambda_3) = 0
        l3 = max(0.1, min(10.0, float(lam3_old[i])))
        for _ in range(15):
            rv = lam1 * lam2 * l3
            rvd = rv ** (-1.0 / 3.0)
            evd = np.array([lam1, lam2, l3]) * rvd
            trace = np.sum(evd * evd)

            di1lam3 = 2.0 * (evd[2] ** 2) - (2.0 / 3.0) * trace
            t3 = 0.0
            for j in range(5):
                bb = 1.0 / (lam ** (2 * j))
                aa = (j + 1) * c_coeffs[j]
                t3 += aa * bb * (trace ** j) * di1lam3

            rv_1 = 1.0 / max(rv, 1e-12)
            p = rv_1 * rbulk * (rv * rv - 1.0) * 0.5
            sig3 = (mu * t3 + p) * rv_1

            if abs(sig3) < 1e-6 * max(1.0, g0):
                break

            # Numerical derivative d(sig3)/d(l3)
            dl3 = 1e-6 * l3
            l3_p = l3 + dl3
            rv_p = lam1 * lam2 * l3_p
            rvd_p = rv_p ** (-1.0 / 3.0)
            evd_p = np.array([lam1, lam2, l3_p]) * rvd_p
            trace_p = np.sum(evd_p * evd_p)
            di1lam3_p = 2.0 * (evd_p[2] ** 2) - (2.0 / 3.0) * trace_p
            t3_p = 0.0
            for j in range(5):
                bb = 1.0 / (lam ** (2 * j))
                aa = (j + 1) * c_coeffs[j]
                t3_p += aa * bb * (trace_p ** j) * di1lam3_p
            p_p = (1.0 / max(rv_p, 1e-12)) * rbulk * (rv_p * rv_p - 1.0) * 0.5
            sig3_p = (mu * t3_p + p_p) / max(rv_p, 1e-12)

            dsig3 = (sig3_p - sig3) / dl3
            if abs(dsig3) < 1e-12:
                break
            step = sig3 / dsig3
            step = np.clip(step, -0.5 * l3, 0.5 * l3)
            l3 = max(0.01, min(100.0, l3 - step))

        lam3_new[i] = l3

        # Compute in-plane principal Cauchy stresses
        rv = lam1 * lam2 * l3
        rvd = rv ** (-1.0 / 3.0)
        evd = np.array([lam1, lam2, l3]) * rvd
        trace = np.sum(evd * evd)

        di1lam = 2.0 * (evd ** 2) - (2.0 / 3.0) * trace
        t_terms = np.zeros(2, dtype=np.float64)
        for j in range(5):
            bb = 1.0 / (lam ** (2 * j))
            aa = (j + 1) * c_coeffs[j]
            cc = aa * bb * (trace ** j)
            t_terms += cc * di1lam[:2]

        p = (1.0 / max(rv, 1e-12)) * rbulk * (rv * rv - 1.0) * 0.5
        sig_princ_2d = (mu * t_terms + p) / max(rv, 1e-12)

        # Rotate back to global/local shell axes
        sig_tensor_2d = evecs_2d @ np.diag(sig_princ_2d) @ evecs_2d.T
        sig_out[i, 0] = sig_tensor_2d[0, 0]
        sig_out[i, 1] = sig_tensor_2d[1, 1]
        if ncomp > 2:
            sig_out[i, 2] = sig_tensor_2d[0, 1]

        # Transverse shear stresses if present
        if ncomp >= 5:
            sig_out[i, 3] = sig_arr[i, 3] + 2.0 * g0 * deps_arr[i, 3]
            sig_out[i, 4] = sig_arr[i, 4] + 2.0 * g0 * deps_arr[i, 4]

    if extra is not None:
        extra["uvar_lam3"] = lam3_new
        if "thk" in extra:
            # Thickness thinning: h = h0 * l3
            extra["thk"] = extra["thk"] * (lam3_new / np.maximum(1e-12, lam3_old))

    r = mat.rho0
    if extra is not None and "rho" in extra and extra["rho"] is not None:
        r = float(np.asarray(extra["rho"]).flat[0])
    c_shell = float(np.sqrt(mat.E / (max(1e-4, 1.0 - mat.nu ** 2) * max(r, _EM20))))

    if return_constitutive_shell:
        sig_ret = sig_out[0, :3] if is_1d else sig_out[:, :3]
        eps_zz = float(np.log(lam3_new[0])) if is_1d else np.log(lam3_new)
        hist = {"lam3": float(lam3_new[0]), "w_mullins": 0.0}
        return sig_ret, eps_zz, hist, c_shell

    if kwargs.get("return_sound_speed", False) or kwargs.get("return_tuple", False):
        if is_1d:
            return sig_out[0], epsp_arr[0], c_shell
        return sig_out, epsp_arr, c_shell

    if is_1d:
        return sig_out[0], epsp_arr[0]
    return sig_out, epsp_arr


def sound_speed(
    mat: Any = None,
    eps: np.ndarray | None = None,
    *,
    rho: float | None = None,
    is_shell: bool = False,
    extra: dict | None = None,
    **kwargs: Any,
) -> float:
    """Solid or shell acoustic wave sound speed for /MAT/LAW92."""
    if isinstance(mat, np.ndarray):
        eps = mat
        mat = build_law92(**kwargs)
    elif not isinstance(mat, ArrudaBoyceParams):
        mat = build_law92(mat, **kwargs)

    r = rho if rho is not None else kwargs.get("rho", None)
    if is_shell:
        return sound_speed_shell(mat, rho=r, extra=extra)

    if eps is not None:
        eps_arr = np.asarray(eps, dtype=np.float64)
        if eps_arr.size >= 6:
            r_val = r if r is not None else mat.rho0
            _, _, c_arr = solid_update(mat, np.zeros(6), eps=eps_arr, extra={"rho": r_val})
            return float(c_arr if np.isscalar(c_arr) else c_arr[0])

    r_val = r if r is not None else mat.rho0
    r_val = max(float(r_val), _EM20)
    return float(np.sqrt(((4.0 / 3.0) * mat.g0 + mat.rbulk) / r_val))


def sound_speed_shell(mat: Any, rho: float | None = None, extra: dict | None = None) -> float:
    """Shell acoustic wave sound speed for /MAT/LAW92."""
    if not isinstance(mat, ArrudaBoyceParams):
        mat = build_law92(mat)
    r = rho if rho is not None else mat.rho0
    r = max(float(r), _EM20)
    nu = mat.nu
    e = mat.E
    return float(np.sqrt(e / (max(1e-4, 1.0 - nu * nu) * r)))


def consistent_tangent(
    mat: Any = None,
    eps: np.ndarray | None = None,
    sig: np.ndarray | None = None,
    dt: float = 0.0,
    ismstr: int = 0,
    is_shell: bool = False,
    perturb: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Compute algorithmic consistent tangent stiffness tensor via central finite differences.

    Returns (6, 6) for solids or (3, 3) for shells.
    """
    if isinstance(mat, np.ndarray):
        eps = mat
        mat = build_law92(**kwargs)
    elif not isinstance(mat, ArrudaBoyceParams):
        mat = build_law92(mat, **kwargs)

    if eps is None:
        eps = np.zeros(3 if is_shell else 6, dtype=np.float64)

    if is_shell:
        ncomp = 3
        c_mat = np.zeros((ncomp, ncomp), dtype=np.float64)
        base_eps = eps[:3].copy()
        for j in range(ncomp):
            eps_pos = base_eps.copy()
            eps_neg = base_eps.copy()
            eps_pos[j] += perturb
            eps_neg[j] -= perturb
            s_pos, _ = shell_update(mat, np.zeros(ncomp), deps=eps_pos)
            s_neg, _ = shell_update(mat, np.zeros(ncomp), deps=eps_neg)
            c_mat[:, j] = (s_pos[:ncomp] - s_neg[:ncomp]) / (2.0 * perturb)
        return c_mat
    else:
        ncomp = 6
        c_mat = np.zeros((ncomp, ncomp), dtype=np.float64)
        base_eps = eps[:6].copy()
        for j in range(ncomp):
            eps_pos = base_eps.copy()
            eps_neg = base_eps.copy()
            eps_pos[j] += perturb
            eps_neg[j] -= perturb
            s_pos, _, _ = solid_update(mat, np.zeros(ncomp), deps=eps_pos, ismstr=ismstr)
            s_neg, _, _ = solid_update(mat, np.zeros(ncomp), deps=eps_neg, ismstr=ismstr)
            c_mat[:, j] = (s_pos[:ncomp] - s_neg[:ncomp]) / (2.0 * perturb)
        return c_mat


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Extra history shapes needed for LAW92 Arruda-Boyce."""
    if nip:
        return {"eps92": (nip, 3), "uvar_lam3": (nip,)}
    return {"eps92": (6,)}


