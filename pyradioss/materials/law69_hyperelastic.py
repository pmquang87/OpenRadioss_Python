"""
LAW69 — Tabulated hyperelastic material (/MAT/LAW69, /MAT/HYP_ELAS, /MAT/HYPERELASTIC).

Pure Python / NumPy implementation of the OpenRadioss LAW69 hyperelastic law
for both 3D solid continuum elements and 2D shell / membrane elements.

Fortran reference sources:
- starter/source/materials/mat/mat069/hm_read_mat69.F   (deck reader & defaults)
- starter/source/materials/mat/mat069/law69_upd.F       (material update & setup)
- starter/source/materials/tools/nlsqf.F                (curve fitting & OGDEN0/1)
- starter/source/materials/tools/law69_nlsqf_auto.F     (automatic curve fitting)
- engine/source/materials/mat/mat069/sigeps69.F         (3D continuum solid kernel)
- engine/source/materials/mat/mat069/sigeps69c.F        (2D shell plane-stress kernel)

Theory
------
Hyperelastic isotropic material formulated in deviatoric principal stretches
with nearly-incompressible volumetric penalty or tabulated bulk function:

Deviatoric strain energy:
    W_dev = sum_{k=1}^N (mu_k / alpha_k) * (lambda_bar_1^alpha_k + lambda_bar_2^alpha_k + lambda_bar_3^alpha_k - 3)

where:
    lambda_i: principal stretches
    J = RV = lambda_1 * lambda_2 * lambda_3 (relative volume ratio)
    lambda_bar_i = EVM_i = lambda_i * J^(-1/3) (deviatoric stretches)
    G_s = sum_{k=1}^N mu_k * alpha_k (initial shear modulus parameter)
    G_0 = G_s / 2 (initial ground-state shear modulus)
    K = G_s * (1 + nu) / (3 * (1 - 2*nu)) (initial bulk modulus)
    E = G_s * (1 + nu) (initial Young's modulus)

Uniaxial nominal / engineering stress (nlsqf.F: OGDEN0):
    sigma_eng(lambda) = sum_{k=1}^N mu_k * (lambda^(alpha_k - 1) - lambda^(-0.5 * alpha_k - 1))

Mooney-Rivlin special case (law_id = 2, N = 2, alpha_1 = 2, alpha_2 = -2):
    sigma_eng(lambda) = mu_1 * (lambda - lambda^(-2)) + mu_2 * (lambda^(-3) - 1)
    with C_10 = mu_1 / 2, C_01 = -mu_2 / 2.

Principal Cauchy stresses (sigeps69.F: 267-300):
    DWDL_i = sum_{k=1}^N mu_k * lambda_bar_i^alpha_k
    SUMDWDL = (DWDL_1 + DWDL_2 + DWDL_3) / 3
    T_i = (DWDL_i - SUMDWDL) / J + P * (J - 1)

For 2D shells (sigeps69c.F):
    Plane stress T_3(lambda_3) = 0 is solved iteratively via a 4-step
    Newton-Raphson iteration for out-of-plane stretch lambda_3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
from scipy.optimize import least_squares

_EM10 = 1e-10
_EM20 = 1e-20
_VOIGT = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


@dataclass
class Law69Params:
    """Parameters for /MAT/LAW69 (/MAT/HYP_ELAS, /MAT/HYPERELASTIC).

    Attributes
    ----------
    id : int
        Material identification number.
    rho0 : float
        Initial mass density.
    rhor : float
        Reference density (defaults to rho0).
    law_id : int
        Type of hyperelastic law: 1=Ogden, 2=Mooney-Rivlin, -1=Auto fitting.
    fct_id : int
        Curve ID for non-linear bulk modulus function (FUN_A1).
    nu : float
        Poisson's ratio (clamped to 0.495 if 0 or 0.5).
    fscale : float
        Scale factor for bulk function (FUN_A1).
    nip : int
        Number of mu/alpha term pairs (N <= 5).
    fct_id1 : int
        Curve ID for test data curve (FUN_B1).
    mu : np.ndarray
        Ogden/Mooney-Rivlin shear coefficients mu_k (shape (nordre,)).
    alpha : np.ndarray
        Ogden/Mooney-Rivlin exponents alpha_k (shape (nordre,)).
    rbulk : float
        Initial bulk modulus K.
    tenscut : float
        Tensile stress cut-off.
    icheck : int
        Validity checking option for mu and alpha.
    gmax : float
        GMAX = sum(mu_k * alpha_k).
    g0 : float
        G0 = gmax / 2 (ground-state shear modulus).
    E : float
        Young's modulus = gmax * (1 + nu).
    title : str
        Material title.
    """

    id: int = 1
    law: int = 69
    law_name: str = "LAW69"
    rho0: float = 1.0
    rhor: float = 0.0
    law_id: int = 1
    fct_id: int = 0
    nu: float = 0.495
    fscale: float = 1.0
    nip: int = 2
    fct_id1: int = 0
    mu: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0], dtype=np.float64))
    alpha: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0], dtype=np.float64))
    rbulk: float = 200.0
    tenscut: float = 1e20
    icheck: int = -3
    gmax: float = 0.0
    g0: float = 0.0
    E: float = 0.0
    title: str = ""
    fail: Any = None
    eos: Any = None

    def __post_init__(self) -> None:
        if self.rhor == 0.0 or self.rhor is None:
            self.rhor = self.rho0

        # Ensure nu is valid
        if self.nu == 0.0 or self.nu >= 0.5:
            self.nu = 0.495

        # Ensure mu and alpha are 1D float64 numpy arrays
        self.mu = np.asarray(self.mu, dtype=np.float64).flatten()
        self.alpha = np.asarray(self.alpha, dtype=np.float64).flatten()

        n_terms = min(len(self.mu), len(self.alpha))
        if n_terms > 5:
            n_terms = 5
        self.mu = self.mu[:n_terms]
        self.alpha = self.alpha[:n_terms]
        self.nip = n_terms

        # Compute GMAX, G0, E, RBULK if not explicitly set
        if self.gmax == 0.0 and n_terms > 0:
            self.gmax = float(np.sum(self.mu * self.alpha))

        if self.g0 == 0.0:
            self.g0 = self.gmax / 2.0

        if self.E == 0.0:
            self.E = self.gmax * (1.0 + self.nu)

        if (self.rbulk == 200.0 or self.rbulk == 0.0) and self.gmax != 0.0:
            denom = max(_EM20, 3.0 * (1.0 - 2.0 * self.nu))
            self.rbulk = self.gmax * (1.0 + self.nu) / denom

    @property
    def nordre(self) -> int:
        """Number of terms."""
        return len(self.mu)

    @property
    def K(self) -> float:
        """Alias for rbulk."""
        return self.rbulk

    @property
    def G(self) -> float:
        """Alias for g0."""
        return self.g0

    @property
    def params(self) -> dict:
        """Dictionary representation of parameters."""
        return {
            "id": self.id,
            "rho0": self.rho0,
            "rhor": self.rhor,
            "law_id": self.law_id,
            "fct_id": self.fct_id,
            "nu": self.nu,
            "fscale": self.fscale,
            "nip": self.nip,
            "fct_id1": self.fct_id1,
            "mu": self.mu,
            "alpha": self.alpha,
            "rbulk": self.rbulk,
            "tenscut": self.tenscut,
            "icheck": self.icheck,
            "gmax": self.gmax,
            "g0": self.g0,
            "E": self.E,
            "K": self.rbulk,
            "G": self.g0,
            "title": self.title,
            "law": self.law,
            "law_name": self.law_name,
            "MAT_NU": self.nu,
            "MAT_RHO": self.rho0,
        }

    def sound_speed_solid(self, rho: Any = None, eps: Any = None) -> float:
        """Ground-state / nonlinear solid sound speed."""
        return float(solid_sound_speed(self, rho=rho, eps=eps))

    def sound_speed_shell(self, rho: Any = None, eps: Any = None) -> float:
        """Ground-state / nonlinear shell sound speed."""
        return float(shell_sound_speed(self, rho=rho, eps=eps))


def fit_law69_curve(
    strain_eng: np.ndarray | list,
    stress_eng: np.ndarray | list,
    law_id: int = 1,
    n_pair: int = 2,
    nu: float = 0.495,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Fit hyperelastic parameters from engineering strain vs engineering stress.

    Fortran reference: starter/source/materials/tools/nlsqf.F and law69_nlsqf_auto.F.

    Parameters
    ----------
    strain_eng : array_like
        Engineering strain values (eps = Delta L / L_0).
    stress_eng : array_like
        Engineering stress values (nominal stress = Force / A_0).
    law_id : int
        1 = Ogden formulation,
        2 = Mooney-Rivlin formulation (linear least-squares),
        -1 = Automatic selection between Mooney-Rivlin and Ogden.
    n_pair : int
        Number of term pairs (N <= 5).
    nu : float
        Poisson's ratio for bulk modulus calculation.

    Returns
    -------
    mu : np.ndarray
        Shear coefficients mu_k (shape (n_pair,)).
    alpha : np.ndarray
        Exponents alpha_k (shape (n_pair,)).
    gmax : float
        Sum(mu_k * alpha_k).
    g0 : float
        Initial shear modulus G_0 = gmax / 2.
    rbulk : float
        Initial bulk modulus K.
    E : float
        Initial Young's modulus E = gmax * (1 + nu).
    """
    eps = np.asarray(strain_eng, dtype=np.float64).flatten()
    sig = np.asarray(stress_eng, dtype=np.float64).flatten()

    if len(eps) != len(sig) or len(eps) == 0:
        raise ValueError("strain_eng and stress_eng must be non-empty and of equal length")

    lam = 1.0 + eps
    if np.any(lam <= 0.0):
        raise ValueError("Stretches lambda = 1 + strain must be strictly positive")

    n_pair = min(max(int(n_pair), 1), 5)
    if nu == 0.0 or nu >= 0.5:
        nu = 0.495

    def _eval_ogden_eng(lam_val: np.ndarray, mu_arr: np.ndarray, al_arr: np.ndarray) -> np.ndarray:
        pred = np.zeros_like(lam_val)
        for m, a in zip(mu_arr, al_arr):
            pred += m * (lam_val ** (a - 1.0) - lam_val ** (-0.5 * a - 1.0))
        return pred

    def _fit_mooney_rivlin() -> tuple[np.ndarray, np.ndarray, float]:
        # N = 2, alpha_1 = 2.0, alpha_2 = -2.0
        # sigma_eng(lambda) = mu_1 * (lambda - lambda^(-2)) + mu_2 * (lambda^(-3) - 1)
        A = np.column_stack([lam - lam ** (-2.0), lam ** (-3.0) - 1.0])
        res, residuals, rank, s = np.linalg.lstsq(A, sig, rcond=None)
        mu_fit = res
        al_fit = np.array([2.0, -2.0], dtype=np.float64)
        pred = _eval_ogden_eng(lam, mu_fit, al_fit)
        err = float(np.mean((pred - sig) ** 2))
        return mu_fit, al_fit, err

    def _fit_ogden(n_terms: int) -> tuple[np.ndarray, np.ndarray, float]:
        dx = np.where(np.abs(eps) >= 1e-6, np.abs(eps), 1e-6)
        ave_slope = float(np.mean(np.abs(sig) / dx))
        mu_max = max(ave_slope * 10.0, 20.0, float(np.max(np.abs(sig))) * 5.0)

        lb = np.zeros(2 * n_terms, dtype=np.float64)
        ub = np.zeros(2 * n_terms, dtype=np.float64)
        for i in range(n_terms):
            lb[2 * i] = -mu_max
            ub[2 * i] = mu_max
            lb[2 * i + 1] = -10.0
            ub[2 * i + 1] = 10.0

        def _residual(p: np.ndarray) -> np.ndarray:
            mu_k = p[0::2]
            al_k = p[1::2]
            return _eval_ogden_eng(lam, mu_k, al_k) - sig

        guesses = []
        # Guess 1: Mooney-Rivlin based starting guess
        if n_terms >= 2:
            try:
                mu_mr, _, _ = _fit_mooney_rivlin()
                p_mr = np.zeros(2 * n_terms, dtype=np.float64)
                p_mr[0] = mu_mr[0]
                p_mr[1] = 2.0
                p_mr[2] = mu_mr[1]
                p_mr[3] = -2.0
                for k in range(2, n_terms):
                    p_mr[2 * k] = 1.0
                    p_mr[2 * k + 1] = 1.0
                guesses.append(p_mr)
            except Exception:
                pass

        # Guess 2: Default Fortran start [5.0, 2.0, 5.0, 2.0, ...]
        p_def = np.zeros(2 * n_terms, dtype=np.float64)
        for i in range(n_terms):
            p_def[2 * i] = 5.0
            p_def[2 * i + 1] = 2.0
        guesses.append(p_def)

        # Guess 3: Alternating exponents with slope-based mu
        p_alt = np.zeros(2 * n_terms, dtype=np.float64)
        for i in range(n_terms):
            p_alt[2 * i] = ave_slope / (2.0 * n_terms)
            p_alt[2 * i + 1] = 2.0 if (i % 2 == 0) else -2.0
        guesses.append(p_alt)

        # Guess 4: Other exponents
        p_exp = np.zeros(2 * n_terms, dtype=np.float64)
        for i in range(n_terms):
            p_exp[2 * i] = (ave_slope / n_terms) * (1.0 if i % 2 == 0 else -0.5)
            p_exp[2 * i + 1] = 1.5 if (i % 2 == 0) else -1.5
        guesses.append(p_exp)

        # Guess 5: Scaled data max
        p_sc = np.zeros(2 * n_terms, dtype=np.float64)
        for i in range(n_terms):
            p_sc[2 * i] = float(np.max(sig)) / n_terms
            p_sc[2 * i + 1] = 1.0 if (i % 2 == 0) else -1.0
        guesses.append(p_sc)

        best_res = None
        best_err = float("inf")

        for p0 in guesses:
            p0_clamped = np.clip(p0, lb + 1e-4, ub - 1e-4)
            try:
                opt = least_squares(_residual, p0_clamped, bounds=(lb, ub), ftol=1e-8, xtol=1e-8, max_nfev=2000)
                err = float(np.mean(opt.fun ** 2))
                if err < best_err:
                    best_err = err
                    best_res = opt
            except Exception:
                continue

        if best_res is None:
            opt = least_squares(_residual, p_def, bounds=(lb, ub), ftol=1e-8, xtol=1e-8, max_nfev=2000)
            best_res = opt
            best_err = float(np.mean(opt.fun ** 2))

        mu_fit = best_res.x[0::2]
        al_fit = best_res.x[1::2]
        return mu_fit, al_fit, best_err

    if law_id == 2:
        mu, alpha, _ = _fit_mooney_rivlin()
    elif law_id == 1:
        mu, alpha, _ = _fit_ogden(n_pair)
    elif law_id == -1:
        # Automatic fitting: try Mooney-Rivlin first, then Ogden
        mu_mr, al_mr, err_mr = _fit_mooney_rivlin()
        mu_og, al_og, err_og = _fit_ogden(n_pair)
        sig_scale = max(float(np.mean(sig ** 2)), 1e-12)
        if err_mr / sig_scale < 1e-3 or err_mr <= err_og:
            mu, alpha = mu_mr, al_mr
        else:
            mu, alpha = mu_og, al_og
    else:
        raise ValueError(f"Unknown law_id {law_id}; supported: 1 (Ogden), 2 (Mooney-Rivlin), -1 (Auto)")

    gmax = float(np.sum(mu * alpha))
    g0 = gmax / 2.0
    denom = max(_EM20, 3.0 * (1.0 - 2.0 * nu))
    rbulk = gmax * (1.0 + nu) / denom
    E = gmax * (1.0 + nu)

    return mu, alpha, gmax, g0, rbulk, E


def build_law69(
    id: Any = 1,
    rho0: float = 1.0,
    rhor: float = 0.0,
    law_id: int = 1,
    fct_id: int = 0,
    nu: float = 0.495,
    fscale: float = 1.0,
    nip: int = 2,
    fct_id1: int = 0,
    mu: np.ndarray | list | None = None,
    alpha: np.ndarray | list | None = None,
    rbulk: float = 200.0,
    tenscut: float = 1e20,
    icheck: int = -3,
    curve_strain: np.ndarray | list | None = None,
    curve_stress: np.ndarray | list | None = None,
    curves: Optional[Dict[Any, Any]] = None,
    title: str = "",
    **kwargs: Any,
) -> Law69Params:
    """Factory function to build and configure Law69Params.

    Can be constructed from direct keyword arguments or from a
    GenericMaterialRecord / dictionary.
    """
    # Unpack from record if passed as first argument
    if not isinstance(id, (int, np.integer, float)):
        rec = id
        mid = int(getattr(rec, "id", 1))
        rho0 = float(getattr(rec, "density", getattr(rec, "rho0", 1.0)) or 1.0)
        title = str(getattr(rec, "title", ""))
        p = getattr(rec, "params", {})
        if isinstance(p, dict):
            law_id = int(p.get("MAT_Iflag", p.get("law_id", law_id)))
            fct_id = int(p.get("FUN_A1", p.get("fct_id", fct_id)))
            nu = float(p.get("MAT_NU", p.get("nu", getattr(rec, "nu", nu))))
            fscale = float(p.get("MAT_FScale", p.get("fscale", fscale)))
            nip = int(p.get("NIP", p.get("nip", nip)))
            fct_id1 = int(p.get("FUN_B1", p.get("fct_id1", fct_id1)))
            icheck = int(p.get("Gflag", p.get("icheck", icheck)))
            rhor = float(p.get("Refer_Rho", p.get("rhor", rho0)))
            tenscut = float(p.get("tenscut", tenscut))
            rbulk = float(p.get("rbulk", rbulk))
            if "mu" in p:
                mu = p["mu"]
            if "alpha" in p:
                alpha = p["alpha"]
        id = mid

    if nu == 0.0 or nu >= 0.5:
        nu = 0.495

    # Check if curve fitting is requested via curve arrays or curve dictionary
    if curve_strain is not None and curve_stress is not None:
        mu_fit, al_fit, gmax, g0, bulk_fit, E_fit = fit_law69_curve(
            curve_strain, curve_stress, law_id=law_id, n_pair=nip, nu=nu
        )
        return Law69Params(
            id=int(id),
            rho0=float(rho0),
            rhor=float(rhor if rhor != 0.0 else rho0),
            law_id=int(law_id),
            fct_id=int(fct_id),
            nu=float(nu),
            fscale=float(fscale),
            nip=int(len(mu_fit)),
            fct_id1=int(fct_id1),
            mu=mu_fit,
            alpha=al_fit,
            rbulk=float(bulk_fit),
            tenscut=float(tenscut),
            icheck=int(icheck),
            gmax=float(gmax),
            g0=float(g0),
            E=float(E_fit),
            title=str(title),
        )

    if fct_id1 > 0 and curves is not None and fct_id1 in curves:
        c_data = curves[fct_id1]
        x_pts = getattr(c_data, "x", getattr(c_data, "strain", None))
        y_pts = getattr(c_data, "y", getattr(c_data, "stress", None))
        if x_pts is not None and y_pts is not None:
            mu_fit, al_fit, gmax, g0, bulk_fit, E_fit = fit_law69_curve(
                x_pts, y_pts, law_id=law_id, n_pair=nip, nu=nu
            )
            return Law69Params(
                id=int(id),
                rho0=float(rho0),
                rhor=float(rhor if rhor != 0.0 else rho0),
                law_id=int(law_id),
                fct_id=int(fct_id),
                nu=float(nu),
                fscale=float(fscale),
                nip=int(len(mu_fit)),
                fct_id1=int(fct_id1),
                mu=mu_fit,
                alpha=al_fit,
                rbulk=float(bulk_fit),
                tenscut=float(tenscut),
                icheck=int(icheck),
                gmax=float(gmax),
                g0=float(g0),
                E=float(E_fit),
                title=str(title),
            )

    # If mu and alpha are supplied directly
    if mu is None:
        if law_id == 2:
            mu = np.array([10.0, 10.0], dtype=np.float64)
            alpha = np.array([2.0, -2.0], dtype=np.float64)
        else:
            mu = np.array([10.0] * max(nip, 1), dtype=np.float64)
            alpha = np.array([2.0] * max(nip, 1), dtype=np.float64)
    else:
        mu = np.asarray(mu, dtype=np.float64).flatten()

    if alpha is None:
        if law_id == 2:
            alpha = np.array([2.0, -2.0], dtype=np.float64)
        else:
            alpha = np.array([2.0] * len(mu), dtype=np.float64)
    else:
        alpha = np.asarray(alpha, dtype=np.float64).flatten()

    nordre = min(len(mu), len(alpha))
    mu = mu[:nordre]
    alpha = alpha[:nordre]

    gmax = float(np.sum(mu * alpha))
    g0 = gmax / 2.0
    denom = max(_EM20, 3.0 * (1.0 - 2.0 * nu))
    calculated_bulk = gmax * (1.0 + nu) / denom
    if rbulk == 200.0 or rbulk == 0.0:
        rbulk = calculated_bulk
    E = gmax * (1.0 + nu)

    return Law69Params(
        id=int(id),
        rho0=float(rho0),
        rhor=float(rhor if rhor != 0.0 else rho0),
        law_id=int(law_id),
        fct_id=int(fct_id),
        nu=float(nu),
        fscale=float(fscale),
        nip=int(nordre),
        fct_id1=int(fct_id1),
        mu=mu,
        alpha=alpha,
        rbulk=float(rbulk),
        tenscut=float(tenscut),
        icheck=int(icheck),
        gmax=float(gmax),
        g0=float(g0),
        E=float(E),
        title=str(title),
    )


def sigeps69_solid(
    params: Any,
    sig: np.ndarray,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    rho: np.ndarray | float | None = None,
    off: np.ndarray | float | None = None,
    *,
    epsp: np.ndarray | None = None,
    return_sound_speed: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """3D continuum solid kernel for LAW69 hyperelasticity.

    Fortran reference: engine/source/materials/mat/mat069/sigeps69.F.

    Parameters
    ----------
    params : Law69Params or Material
        Material parameters.
    sig : np.ndarray
        Cauchy stress array of shape (NEL, 6) or (6,).
    deps : np.ndarray, optional
        Strain increment array of shape (NEL, 6).
    eps : np.ndarray, optional
        Total strain array of shape (NEL, 6) or deformation gradient.
    dt : float
        Time step.
    extra : dict, optional
        Element scratch/extra views (e.g. "F", "curves", "off").
    ismstr : int
        Strain formulation: 0 = logarithmic/true, 1/3 = engineering, 10/12 = Green-Lagrange.
    rho : array_like, optional
        Current element densities.
    off : array_like, optional
        Element activity flag (1.0 = active, 0.0 = deleted).

    Returns
    -------
    sig_new : np.ndarray
        Updated Cauchy stress array.
    epsp_new : np.ndarray
        Plastic strain (zeros for hyperelastic).
    soundsp : np.ndarray
        Per-element sound speed array.
    """
    if not isinstance(params, Law69Params):
        params = build_law69(params)

    is_1d = (sig.ndim == 1)
    sig_arr = sig.reshape(1, -1) if is_1d else sig
    nel = sig_arr.shape[0]

    if nel == 0:
        empty = np.empty((0, 6), dtype=np.float64)
        return (empty, np.empty(0), np.empty(0)) if return_sound_speed else (empty, np.empty(0))

    # Determine eps array
    if eps is not None:
        eps_in = np.asarray(eps, dtype=np.float64)
        if eps_in.ndim == 1:
            if eps_in.shape[0] == 6:
                eps_arr = eps_in.reshape(1, 6)
            elif eps_in.shape[0] == nel and nel != 6:
                epsp = eps_in
                eps_arr = None
            else:
                eps_arr = eps_in.reshape(1, -1)
        else:
            eps_arr = eps_in
    else:
        eps_arr = None

    if eps_arr is None:
        if extra is not None and "F" in extra and extra["F"] is not None:
            F = extra["F"]
            if is_1d and F.ndim == 2:
                F = F[None, :, :]
            m = F.shape[0]
            B = np.einsum("nab,ncb->nac", F, F)
            w, v = np.linalg.eigh(B)
            ln_w = 0.5 * np.log(np.maximum(w, _EM20))
            E_mat = np.einsum("nik,nk,njk->nij", v, ln_w, v)
            eps_arr = np.zeros((m, 6), dtype=np.float64)
            eps_arr[:, 0] = E_mat[:, 0, 0]
            eps_arr[:, 1] = E_mat[:, 1, 1]
            eps_arr[:, 2] = E_mat[:, 2, 2]
            eps_arr[:, 3] = 2.0 * E_mat[:, 0, 1]
            eps_arr[:, 4] = 2.0 * E_mat[:, 1, 2]
            eps_arr[:, 5] = 2.0 * E_mat[:, 0, 2]
        elif deps is not None:
            deps_in = np.asarray(deps, dtype=np.float64)
            eps_arr = deps_in.reshape(1, -1) if deps_in.ndim == 1 else deps_in
        else:
            eps_arr = np.zeros_like(sig_arr)

    if eps_arr.ndim == 1:
        eps_arr = eps_arr.reshape(1, -1)

    # 1. Build symmetric 3x3 strain tensor AV: sigeps69.F: 139-146
    A = np.zeros((nel, 3, 3), dtype=np.float64)
    A[:, 0, 0] = eps_arr[:, 0]
    A[:, 1, 1] = eps_arr[:, 1]
    A[:, 2, 2] = eps_arr[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = 0.5 * eps_arr[:, 3]
    A[:, 1, 2] = A[:, 2, 1] = 0.5 * eps_arr[:, 4]
    A[:, 0, 2] = A[:, 2, 0] = 0.5 * eps_arr[:, 5]

    # 2. Eigendecomposition (VALPVEC_V): sigeps69.F: 147-153
    evv, dirprv = np.linalg.eigh(A)  # evv: eigenvalues, dirprv: eigenvectors as columns
    # Sort descending to match Fortran VALPVEC_V (VAL(1) >= VAL(2) >= VAL(3))
    idx = np.argsort(evv, axis=-1)[:, ::-1]
    evv = np.take_along_axis(evv, idx, axis=-1)
    dirprv = np.take_along_axis(dirprv, idx[:, None, :], axis=-1)

    # 3. Principal stretches EV: sigeps69.F: 154-179
    if ismstr in (1, 3):
        ev = evv + 1.0
    elif ismstr in (10, 12):
        ev = np.sqrt(np.maximum(evv + 1.0, _EM20))
    else:
        ev = np.exp(evv)

    ev = np.maximum(ev, _EM20)

    # 4. Relative volume J = RV = lambda_1 * lambda_2 * lambda_3: sigeps69.F: 213-215
    rv = np.maximum(ev[:, 0] * ev[:, 1] * ev[:, 2], _EM20)

    # 5. Deviatoric stretches EVM_i = lambda_i * RV^(-1/3): sigeps69.F: 222-233
    rvt = rv ** (-1.0 / 3.0)
    evm = ev * rvt[:, None]

    # 6. Anti-buckling check and Bulk modulus pressure P: sigeps69.F: 203-211, 234-241
    p_fac = np.ones(nel, dtype=np.float64)
    if params.rbulk > 24.0 * params.gmax and params.gmax > 0.0:
        nu_1 = 40.0 * (0.5 - (3.0 * params.rbulk - params.gmax) / (6.0 * params.rbulk + params.gmax))
        amin = np.min(ev, axis=1)
        mask = amin < 0.2
        if np.any(mask):
            p_fac[mask] = np.maximum(1.0, nu_1 / np.maximum(_EM20, amin[mask]))

    # Pressure calculation
    p_arr = np.zeros(nel, dtype=np.float64)
    curve_interp = False
    if params.fct_id > 0 and extra is not None and "curves" in extra:
        curves_dict = extra["curves"]
        if params.fct_id in curves_dict:
            curve = curves_dict[params.fct_id]
            if callable(curve):
                p_arr = params.rbulk * params.fscale * curve(rv)
                curve_interp = True
            elif hasattr(curve, "evaluate"):
                p_arr = params.rbulk * params.fscale * np.array([curve.evaluate(v) for v in rv])
                curve_interp = True
            elif hasattr(curve, "x") and hasattr(curve, "y"):
                p_arr = params.rbulk * params.fscale * np.interp(rv, curve.x, curve.y)
                curve_interp = True

    if not curve_interp:
        p_arr = p_fac * params.rbulk

    # Hydrostatic Cauchy contribution DWDRV: sigeps69.F: 293
    dhdrv = p_arr * (rv - 1.0)

    # 7. Deviatoric stress derivatives DWDL: sigeps69.F: 267-292
    dwdl = np.zeros((nel, 3), dtype=np.float64)
    for k in range(params.nordre):
        mu_k = params.mu[k]
        al_k = params.alpha[k]
        if al_k != 0.0:
            dwdl += mu_k * (evm ** al_k)
        else:
            dwdl += mu_k

    sumdwdl = np.sum(dwdl, axis=1, keepdims=True) / 3.0

    # Principal Cauchy stresses: sigeps69.F: 297-300
    T = (dwdl - sumdwdl) / rv[:, None] + dhdrv[:, None]

    # 8. Tensile cut-off check: sigeps69.F: 304-312
    if off is None:
        if extra is not None and "off" in extra and extra["off"] is not None:
            off_arr = np.asarray(extra["off"], dtype=np.float64)
        else:
            off_arr = np.ones(nel, dtype=np.float64)
    else:
        off_arr = np.asarray(off, dtype=np.float64).flatten()
        if len(off_arr) == 1 and nel > 1:
            off_arr = np.full(nel, off_arr[0], dtype=np.float64)

    tenscut_val = abs(float(params.tenscut))
    cut_mask = (off_arr != 0.0) & np.any(T > tenscut_val, axis=1)
    if np.any(cut_mask):
        T[cut_mask] = 0.0
        off_arr[cut_mask] = 0.0
        if off is not None and isinstance(off, np.ndarray):
            off.flat[:] = off_arr.flat[:]
        if extra is not None and "off" in extra and extra["off"] is not None:
            extra["off"][:] = off_arr

    zero_mask = (off_arr == 0.0)
    if np.any(zero_mask):
        T[zero_mask] = 0.0

    # 9. Rotate principal Cauchy stresses to global directions: sigeps69.F: 314-333
    # SIGN = sum_k T_k * (v_k @ v_k^T)
    T_mat = np.einsum("nik,nk,njk->nij", dirprv, T, dirprv)

    sig_new = np.zeros((nel, 6), dtype=np.float64)
    sig_new[:, 0] = T_mat[:, 0, 0]
    sig_new[:, 1] = T_mat[:, 1, 1]
    sig_new[:, 2] = T_mat[:, 2, 2]
    sig_new[:, 3] = T_mat[:, 0, 1]
    sig_new[:, 4] = T_mat[:, 1, 2]
    sig_new[:, 5] = T_mat[:, 0, 2]

    # 10. Sound speed and GTMAX: sigeps69.F: 247-265, 336-339
    cii = np.zeros((nel, 3), dtype=np.float64)
    for k in range(params.nordre):
        mu_k = params.mu[k]
        al_k = params.alpha[k]
        if mu_k * al_k != 0.0:
            lam_al = evm ** al_k
            amax = np.sum(lam_al, axis=1, keepdims=True) / 3.0
            cii += (mu_k * al_k) * (lam_al + amax)

    amax1 = 0.81 * 0.5 / max(params.gmax, 1e-20)
    amax_cii = amax1 * np.max(cii, axis=1)
    eti = np.maximum(1.0, amax_cii)
    gtmax = params.gmax * eti

    if rho is None:
        if extra is not None and "rho" in extra and extra["rho"] is not None:
            rho_arr = np.asarray(extra["rho"], dtype=np.float64).flatten()
        else:
            rho_arr = params.rho0 / rv
    else:
        rho_arr = np.asarray(rho, dtype=np.float64).flatten()
        if len(rho_arr) == 1 and nel > 1:
            rho_arr = np.full(nel, rho_arr[0], dtype=np.float64)

    soundsp = np.sqrt(np.maximum(((2.0 / 3.0) * gtmax + p_arr) / np.maximum(rho_arr, _EM20), _EM20))

    if epsp is not None:
        epsp_out = np.asarray(epsp, dtype=np.float64).copy()
    else:
        epsp_out = np.zeros(nel, dtype=np.float64)

    if is_1d:
        res_sig = sig_new[0]
        res_epsp = epsp_out[0] if epsp_out.ndim > 0 else float(epsp_out)
        res_c = float(soundsp[0])
        return (res_sig, res_epsp, res_c) if return_sound_speed else (res_sig, res_epsp)

    return (sig_new, epsp_out, soundsp) if return_sound_speed else (sig_new, epsp_out)


def sigeps69c_shell(
    params: Any,
    sig: np.ndarray,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    uvar: np.ndarray | None = None,
    off: np.ndarray | float | None = None,
    thkn: np.ndarray | None = None,
    thklyl: np.ndarray | float | None = None,
    rho0: np.ndarray | float | None = None,
    *,
    epsp: np.ndarray | None = None,
    return_sound_speed: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray]:
    """2D shell / membrane plane-stress kernel for LAW69 hyperelasticity.

    Fortran reference: engine/source/materials/mat/mat069/sigeps69c.F.

    Parameters
    ----------
    params : Law69Params or Material
        Material parameters.
    sig : np.ndarray
        Cauchy stress array of shape (NEL, 3) or (NEL, 5).
    deps : np.ndarray, optional
        Strain increment array of shape (NEL, 3) or (NEL, 5).
    eps : np.ndarray, optional
        Total in-plane strain array [xx, yy, xy].
    dt : float
        Time step.
    extra : dict, optional
        Element scratch/extra views (e.g. "uvar", "off", "thkn", "thklyl").
    ismstr : int
        Strain formulation: 0 = logarithmic, 1/3/11 = engineering, 10 = Green-Lagrange.
    uvar : np.ndarray, optional
        Internal variable array of shape (NEL, >=3) storing lambda_3 at index 2.
    off : array_like, optional
        Element activity flag (1.0 = active, 0.8 = cut, 0.0 = inactive).
    thkn : np.ndarray, optional
        Current thickness array.
    thklyl : array_like, optional
        Layer thickness array.
    rho0 : array_like, optional
        Initial density.

    Returns
    -------
    sig_new : np.ndarray
        Updated shell stress array.
    epsp_new : np.ndarray
        Plastic strain array (zeros).
    soundsp : np.ndarray
        Shell acoustic wave speed.
    """
    if not isinstance(params, Law69Params):
        params = build_law69(params)

    is_1d = (sig.ndim == 1)
    sig_arr = sig.reshape(1, -1) if is_1d else sig
    nel = sig_arr.shape[0]

    if nel == 0:
        empty = np.empty((0, sig_arr.shape[1]), dtype=np.float64)
        return (empty, np.empty(0), np.empty(0)) if return_sound_speed else (empty, np.empty(0))

    if eps is not None:
        if eps.ndim == 1 and not is_1d and eps.shape[0] == nel:
            epsp = eps
            eps_arr = None
        else:
            eps_arr = eps.reshape(1, -1) if is_1d else eps
    else:
        eps_arr = None

    if eps_arr is None:
        if deps is not None:
            eps_arr = deps.reshape(1, -1) if is_1d else deps
        else:
            eps_arr = np.zeros((nel, 3), dtype=np.float64)

    if eps_arr.ndim == 1:
        eps_arr = eps_arr.reshape(1, -1)

    deps_arr = deps.reshape(1, -1) if (deps is not None and is_1d) else deps

    # 1. In-plane eigenvalues & eigenvectors: sigeps69c.F: 107-142
    trav = eps_arr[:, 0] + eps_arr[:, 1]
    diff = eps_arr[:, 0] - eps_arr[:, 1]
    rootv = np.sqrt(diff * diff + eps_arr[:, 2] * eps_arr[:, 2])

    evv1 = 0.5 * (trav + rootv)
    evv2 = 0.5 * (trav - rootv)

    if off is None:
        if extra is not None and "off" in extra and extra["off"] is not None:
            off_arr = np.asarray(extra["off"], dtype=np.float64).flatten()
        else:
            off_arr = np.ones(nel, dtype=np.float64)
    else:
        off_arr = np.asarray(off, dtype=np.float64).flatten()
        if len(off_arr) == 1 and nel > 1:
            off_arr = np.full(nel, off_arr[0], dtype=np.float64)

    # Avoid NaN for Green-Lagrange strain: sigeps69c.F: 116-124
    if ismstr == 10:
        nan_mask = np.minimum(evv1, evv2) <= -1.0
        if np.any(nan_mask):
            evv1[nan_mask] = 0.0
            evv2[nan_mask] = 0.0
            off_arr[nan_mask] = 0.8

    # Eigenvector matrix EIGV (nel, 3, 2): sigeps69c.F: 126-142
    iso = np.abs(evv2 - evv1) < _EM10
    inv_root = np.where(iso, 0.0, 1.0 / np.maximum(rootv, _EM20))

    eigv = np.zeros((nel, 3, 2), dtype=np.float64)
    eigv[:, 0, 0] = np.where(iso, 1.0, (eps_arr[:, 0] - evv2) * inv_root)
    eigv[:, 1, 0] = np.where(iso, 1.0, (eps_arr[:, 1] - evv2) * inv_root)
    eigv[:, 2, 0] = np.where(iso, 0.0, (0.5 * eps_arr[:, 2]) * inv_root)

    eigv[:, 0, 1] = np.where(iso, 0.0, (evv1 - eps_arr[:, 0]) * inv_root)
    eigv[:, 1, 1] = np.where(iso, 0.0, (evv1 - eps_arr[:, 1]) * inv_root)
    eigv[:, 2, 1] = np.where(iso, 0.0, -(0.5 * eps_arr[:, 2]) * inv_root)

    # 2. In-plane principal stretches: sigeps69c.F: 143-162
    if ismstr in (1, 3, 11):
        ev1 = evv1 + 1.0
        ev2 = evv2 + 1.0
    elif ismstr == 10:
        ev1 = np.sqrt(np.maximum(evv1 + 1.0, _EM20))
        ev2 = np.sqrt(np.maximum(evv2 + 1.0, _EM20))
    else:
        ev1 = np.exp(evv1)
        ev2 = np.exp(evv2)

    ev1 = np.maximum(ev1, _EM20)
    ev2 = np.maximum(ev2, _EM20)

    # 3. Retrieve or initialize uvar array for lambda_3 at index 2 (1-based index 3): sigeps69c.F: 148, 169
    if uvar is None:
        if extra is not None and "uvar" in extra and extra["uvar"] is not None:
            uvar_arr = extra["uvar"]
        elif extra is not None and "uv69" in extra and extra["uv69"] is not None:
            uvar_arr = extra["uv69"]
        elif extra is not None and "uvar69" in extra and extra["uvar69"] is not None:
            uvar_arr = extra["uvar69"]
        else:
            uvar_arr = np.zeros((nel, 9), dtype=np.float64)
            uvar_arr[:, 2] = 1.0
    else:
        uvar_arr = uvar

    if uvar_arr.ndim == 1:
        uvar_arr = uvar_arr.reshape(nel, -1)

    lam3_0 = uvar_arr[:, 2].copy()
    invalid_mask = lam3_0 <= 1e-12
    if np.any(invalid_mask):
        lam3_0[invalid_mask] = 1.0

    # 4. 4-step Newton iteration solving Kirchoff J*T_3(lambda_3) = 0: sigeps69c.F: 163-196
    dlam3 = np.zeros(nel, dtype=np.float64)
    for _ in range(4):
        ev3 = np.maximum(lam3_0 + dlam3, _EM20)
        rv = ev1 * ev2 * ev3
        rvt = np.exp((-1.0 / 3.0) * np.log(np.maximum(rv, _EM20)))
        evm1 = ev1 * rvt
        evm2 = ev2 * rvt
        evm3 = ev3 * rvt

        kir3 = np.zeros(nel, dtype=np.float64)
        kt3 = np.zeros(nel, dtype=np.float64)

        for k in range(params.nordre):
            mu_k = params.mu[k]
            al_k = params.alpha[k]
            if mu_k * al_k != 0.0:
                lam_al1 = np.exp(al_k * np.log(np.maximum(evm1, _EM20)))
                lam_al2 = np.exp(al_k * np.log(np.maximum(evm2, _EM20)))
                lam_al3 = np.exp(al_k * np.log(np.maximum(evm3, _EM20)))
                sumdwdl = (1.0 / 3.0) * (lam_al1 + lam_al2 + lam_al3)
                s_term = mu_k * (lam_al3 - sumdwdl)
                kir3 += s_term
                kt3 += al_k * s_term

        partp = params.rbulk * (rv - 1.0)
        t3 = kir3 + partp * rv
        kt3_full = (2.0 / 3.0) * kt3 / ev3 + params.rbulk * (2.0 * rv - 1.0) * ev1 * ev2

        active = (off_arr != 0.0) & (off_arr != 0.8) & (kt3_full > _EM20)
        if np.any(active):
            dlam3[active] -= t3[active] / kt3_full[active]

    ev3 = np.maximum(lam3_0 + dlam3, _EM20)
    dezz = np.log(np.maximum(1.0 + dlam3 / lam3_0, _EM20))
    uvar_arr[:, 2] = ev3
    if uvar is not None and isinstance(uvar, np.ndarray):
        if uvar.ndim == 1 and uvar.shape[0] >= 3:
            uvar[2] = uvar_arr[0, 2]
        elif uvar.ndim == 2 and uvar.shape[1] >= 3:
            uvar[:, 2] = uvar_arr[:, 2]

    # Update thickness: sigeps69c.F: 256
    if thkn is None and extra is not None:
        if "thkn" in extra and extra["thkn"] is not None:
            thkn = extra["thkn"]
        elif "thk" in extra and extra["thk"] is not None:
            thkn = extra["thk"]
    if thklyl is None and extra is not None:
        if "thklyl" in extra and extra["thklyl"] is not None:
            thklyl = extra["thklyl"]
        elif "thk0" in extra and extra["thk0"] is not None:
            thklyl = extra["thk0"]

    if thkn is not None and thklyl is not None:
        thk_val = np.asarray(thklyl, dtype=np.float64).flatten()
        thkn_new = np.asarray(thkn, dtype=np.float64).flatten() + dezz * thk_val * off_arr
        if isinstance(thkn, np.ndarray):
            thkn.flat[:] = thkn_new

    # 5. In-plane Cauchy principal stresses T1, T2: sigeps69c.F: 201-221
    rv = np.maximum(ev1 * ev2 * ev3, _EM20)
    rvt = np.exp((-1.0 / 3.0) * np.log(rv))
    evm1 = ev1 * rvt
    evm2 = ev2 * rvt
    evm3 = ev3 * rvt
    invrv = 1.0 / rv

    s_ldwdl = np.zeros((nel, 3), dtype=np.float64)
    for k in range(params.nordre):
        mu_k = params.mu[k]
        al_k = params.alpha[k]
        if mu_k * al_k != 0.0:
            s_ldwdl[:, 0] += mu_k * np.exp(al_k * np.log(np.maximum(evm1, _EM20)))
            s_ldwdl[:, 1] += mu_k * np.exp(al_k * np.log(np.maximum(evm2, _EM20)))
            s_ldwdl[:, 2] += mu_k * np.exp(al_k * np.log(np.maximum(evm3, _EM20)))

    sumdwdl = (s_ldwdl[:, 0] + s_ldwdl[:, 1] + s_ldwdl[:, 2]) * (1.0 / 3.0)
    partp = params.rbulk * (rv - 1.0)
    t1 = (s_ldwdl[:, 0] - sumdwdl) * invrv + partp
    t2 = (s_ldwdl[:, 1] - sumdwdl) * invrv + partp

    # 6. Tension cut: sigeps69c.F: 224-232
    tenscut_val = abs(float(params.tenscut))
    cut_mask = (off_arr != 0.0) & ((t1 > tenscut_val) | (t2 > tenscut_val))
    if np.any(cut_mask):
        t1[cut_mask] = 0.0
        t2[cut_mask] = 0.0
        off_arr[cut_mask] = 0.8
        if off is not None and isinstance(off, np.ndarray):
            off.flat[:] = off_arr.flat[:]
        if extra is not None and "off" in extra and extra["off"] is not None:
            extra["off"][:] = off_arr
        if extra is not None and "layfail" in extra and extra["layfail"] is not None:
            extra["layfail"][cut_mask] = 0.0

    zero_mask = (off_arr == 0.0)
    if np.any(zero_mask):
        t1[zero_mask] = 0.0
        t2[zero_mask] = 0.0

    # 7. Shell sound speed: sigeps69c.F: 234-262
    cii = np.zeros((nel, 3), dtype=np.float64)
    for k in range(params.nordre):
        mu_k = params.mu[k]
        al_k = params.alpha[k]
        if mu_k * al_k != 0.0:
            lam_al1 = np.exp(al_k * np.log(np.maximum(evm1, _EM20)))
            lam_al2 = np.exp(al_k * np.log(np.maximum(evm2, _EM20)))
            lam_al3 = np.exp(al_k * np.log(np.maximum(evm3, _EM20)))
            amax = (1.0 / 3.0) * (lam_al1 + lam_al2 + lam_al3)
            cii[:, 0] += mu_k * al_k * (lam_al1 + amax)
            cii[:, 1] += mu_k * al_k * (lam_al2 + amax)
            cii[:, 2] += mu_k * al_k * (lam_al3 + amax)

    gtmax = 0.5 * np.max(cii, axis=1)
    emax = np.maximum(params.gmax, gtmax) * (1.0 + params.nu)
    a11 = emax / max(1.0 - params.nu ** 2, _EM20)
    r0 = params.rho0 if rho0 is None else rho0
    r0_arr = np.asarray(r0, dtype=np.float64).flatten()
    if len(r0_arr) == 1 and nel > 1:
        r0_arr = np.full(nel, r0_arr[0], dtype=np.float64)
    soundsp = np.sqrt(np.maximum(a11 / np.maximum(r0_arr, _EM20), _EM20))

    # 8. Reassemble stresses in shell frame: sigeps69c.F: 249-254
    sig_new = np.zeros_like(sig_arr)
    sig_new[:, 0] = eigv[:, 0, 0] * t1 + eigv[:, 0, 1] * t2
    sig_new[:, 1] = eigv[:, 1, 0] * t1 + eigv[:, 1, 1] * t2
    sig_new[:, 2] = eigv[:, 2, 0] * t1 + eigv[:, 2, 1] * t2

    # Transverse shear stresses: sigeps69c.F: 253-254
    if sig_arr.shape[1] >= 5 and deps_arr is not None and deps_arr.shape[1] >= 5:
        sig_new[:, 3] = sig_arr[:, 3] + params.g0 * deps_arr[:, 3]
        sig_new[:, 4] = sig_arr[:, 4] + params.g0 * deps_arr[:, 4]

    if epsp is not None:
        epsp_out = np.asarray(epsp, dtype=np.float64).copy()
    else:
        epsp_out = np.zeros(nel, dtype=np.float64)

    if is_1d:
        res_sig = sig_new[0]
        res_epsp = epsp_out[0] if epsp_out.ndim > 0 else float(epsp_out)
        res_c = float(soundsp[0])
        return (res_sig, res_epsp, res_c) if return_sound_speed else (res_sig, res_epsp)

    return (sig_new, epsp_out, soundsp) if return_sound_speed else (sig_new, epsp_out)


def solid_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    *,
    epsp: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solid update adapter compatible with pyradioss materials interface."""
    return sigeps69_solid(
        params=mat,
        sig=sig,
        deps=deps,
        eps=eps,
        dt=dt,
        extra=extra,
        ismstr=ismstr,
        epsp=epsp,
        return_sound_speed=True,
        **kwargs,
    )


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray | None = None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    *,
    epsp: np.ndarray | None = None,
    return_sound_speed: bool = False,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shell update adapter compatible with pyradioss materials interface."""
    return sigeps69c_shell(
        params=mat,
        sig=sig,
        deps=deps,
        eps=eps,
        dt=dt,
        extra=extra,
        ismstr=ismstr,
        epsp=epsp,
        return_sound_speed=return_sound_speed,
        **kwargs,
    )


def solid_sound_speed(
    mat: Any,
    rho: np.ndarray | float | None = None,
    eps: np.ndarray | None = None,
    ismstr: int = 0,
    **kwargs: Any,
) -> np.ndarray | float:
    """Nonlinear solid sound speed calculation."""
    if not isinstance(mat, Law69Params):
        mat = build_law69(mat)

    if eps is None:
        r0 = mat.rho0 if rho is None else rho
        denom = max(float(r0) if np.isscalar(r0) else 1.0, _EM20)
        c0 = np.sqrt(((2.0 / 3.0) * mat.gmax + mat.rbulk) / denom)
        return float(c0) if np.isscalar(r0) else np.sqrt(((2.0 / 3.0) * mat.gmax + mat.rbulk) / np.maximum(rho, _EM20))

    dummy_sig = np.zeros((1, 6) if np.asarray(eps).ndim == 1 else (len(eps), 6), dtype=np.float64)
    _, _, c = sigeps69_solid(mat, dummy_sig, eps=eps, ismstr=ismstr, rho=rho, return_sound_speed=True, **kwargs)
    if np.asarray(eps).ndim == 1:
        return float(c[0]) if hasattr(c, "__len__") else float(c)
    return c


def shell_sound_speed(
    mat: Any,
    rho: np.ndarray | float | None = None,
    eps: np.ndarray | None = None,
    **kwargs: Any,
) -> np.ndarray | float:
    """Nonlinear shell sound speed calculation."""
    if not isinstance(mat, Law69Params):
        mat = build_law69(mat)

    if eps is None:
        r0 = mat.rho0 if rho is None else rho
        emax = mat.gmax * (1.0 + mat.nu)
        a11 = emax / max(1.0 - mat.nu ** 2, _EM20)
        c0 = np.sqrt(a11 / max(float(r0) if np.isscalar(r0) else 1.0, _EM20))
        return float(c0) if np.isscalar(r0) else np.sqrt(a11 / np.maximum(rho, _EM20))

    dummy_sig = np.zeros((1, 3) if np.asarray(eps).ndim == 1 else (len(eps), 3), dtype=np.float64)
    _, _, c = sigeps69c_shell(mat, dummy_sig, eps=eps, rho0=rho, return_sound_speed=True, **kwargs)
    if np.asarray(eps).ndim == 1:
        return float(c[0]) if hasattr(c, "__len__") else float(c)
    return c


def tangent_law69_solid(
    params: Any,
    eps: np.ndarray,
    deps: np.ndarray | None = None,
    dt: float = 0.0,
    h: float = 1e-7,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent for 3D solid continuum elements.

    Parameters
    ----------
    params : Law69Params or Material
        Material parameters.
    eps : np.ndarray
        Strain state (NEL, 6) or (6,).
    h : float
        Perturbation size for central finite differences (default 1e-7).

    Returns
    -------
    C : np.ndarray
        Spatial algorithmic tangent tensor of shape (NEL, 6, 6).
    """
    if not isinstance(params, Law69Params):
        params = build_law69(params)

    eps_arr = np.asarray(eps, dtype=np.float64)
    is_1d = (eps_arr.ndim == 1)
    if is_1d:
        eps_arr = eps_arr.reshape(1, -1)

    nel = eps_arr.shape[0]
    C = np.zeros((nel, 6, 6), dtype=np.float64)
    dummy_sig = np.zeros((nel, 6), dtype=np.float64)

    for j in range(6):
        eps_p = eps_arr.copy()
        eps_p[:, j] += h
        eps_m = eps_arr.copy()
        eps_m[:, j] -= h

        sp, _, _ = sigeps69_solid(params, dummy_sig, eps=eps_p, dt=dt, extra=extra, ismstr=ismstr, return_sound_speed=True)
        sm, _, _ = sigeps69_solid(params, dummy_sig, eps=eps_m, dt=dt, extra=extra, ismstr=ismstr, return_sound_speed=True)

        if is_1d:
            sp = sp.reshape(1, -1)
            sm = sm.reshape(1, -1)

        C[:, :, j] = (sp - sm) / (2.0 * h)

    return C[0] if is_1d else C


def tangent_law69_shell(
    params: Any,
    eps: np.ndarray,
    deps: np.ndarray | None = None,
    dt: float = 0.0,
    h: float = 1e-7,
    extra: Optional[Dict[str, Any]] = None,
    ismstr: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent for 2D shells/membranes.

    Parameters
    ----------
    params : Law69Params or Material
        Material parameters.
    eps : np.ndarray
        In-plane strain state (NEL, 3) or (3,).
    h : float
        Perturbation size for central finite differences (default 1e-7).

    Returns
    -------
    C : np.ndarray
        In-plane condensed algorithmic tangent tensor of shape (NEL, 3, 3).
    """
    if not isinstance(params, Law69Params):
        params = build_law69(params)

    eps_arr = np.asarray(eps, dtype=np.float64)
    is_1d = (eps_arr.ndim == 1)
    if is_1d:
        eps_arr = eps_arr.reshape(1, -1)

    nel = eps_arr.shape[0]
    C = np.zeros((nel, 3, 3), dtype=np.float64)
    dummy_sig = np.zeros((nel, 3), dtype=np.float64)

    for j in range(3):
        eps_p = eps_arr.copy()
        eps_p[:, j] += h
        eps_m = eps_arr.copy()
        eps_m[:, j] -= h

        sp, _ = sigeps69c_shell(params, dummy_sig, eps=eps_p, dt=dt, extra=extra, ismstr=ismstr, return_sound_speed=False)
        sm, _ = sigeps69c_shell(params, dummy_sig, eps=eps_m, dt=dt, extra=extra, ismstr=ismstr, return_sound_speed=False)

        if is_1d:
            sp = sp.reshape(1, -1)
            sm = sm.reshape(1, -1)

        C[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)

    return C[0] if is_1d else C


# Aliases for package dispatch compatibility
consistent_solid_tangent = tangent_law69_solid
consistent_shell_tangent = tangent_law69_shell
sound_speed = solid_sound_speed
sound_speed_shell = shell_sound_speed
law69_solid_update = solid_update
law69_shell_update = shell_update
law69_solid_sound_speed = solid_sound_speed
law69_shell_sound_speed = shell_sound_speed
law69_consistent_solid_tangent = consistent_solid_tangent
law69_consistent_shell_tangent = consistent_shell_tangent


def extra_shapes(mat: Any = None, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Extra state variable allocations for LAW69 elements.

    NUVAR = 9 according to hm_read_mat69.F.
    """
    if nip is not None and nip > 0:
        return {"uvar": (nip, 9)}
    return {"uvar": (9,)}


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (
            69,
            "69",
            "LAW69",
            "HYP_ELAS",
            "HYPERELASTIC",
            "MAT_LAW69",
            "MAT_HYP_ELAS",
            "MAT_HYPERELASTIC",
            "LAW69_HYPERELASTIC",
        ):
            MAT_PHYSICS_REGISTRY[key] = build_law69
    except Exception:
        pass


_register()
