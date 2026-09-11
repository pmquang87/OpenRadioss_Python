"""
LAW82 — Ogden hyperelastic material (/MAT/LAW82, /MAT/OGDEN).

Implements the Ogden hyperelastic constitutive law for both 3D solid continuum
elements and 2D shell / membrane elements in pure Python / NumPy.

Fortran reference sources:
- engine/source/materials/mat/mat082/sigeps82.F   (3D continuum solid kernel)
- engine/source/materials/mat/mat082/sigeps82c.F  (2D shell / membrane kernel)
- starter/source/materials/mat/mat082/hm_read_mat82.F (starter reader & parameter setup)

Theory
------
Hyperelastic isotropic law formulated in principal stretches:
    W = sum_{k=1}^N (2 mu_k / alpha_k^2) * (lambda_bar_1^alpha_k + lambda_bar_2^alpha_k + lambda_bar_3^alpha_k - 3)
        + sum_{k=1}^N (1 / D_k) * (J - 1)^(2k)

where:
    lambda_i: principal stretches (from eigendecomposition of strain tensor)
    J = lambda_1 * lambda_2 * lambda_3: relative volume ratio (rho0 / rho)
    lambda_bar_i = lambda_i * J^(-1/3): deviatoric stretches
    G0 = sum_{k=1}^N mu_k: initial ground-state shear modulus

Principal Cauchy stresses (sigeps82.F:255-265):
    S_i = sum_{k=1}^N (2 mu_k / (alpha_k * J)) * [ (2/3) * lambda_bar_i^alpha_k - (1/3) * (lambda_bar_j^alpha_k + lambda_bar_m^alpha_k) ] + P
where:
    P = sum_{k=1}^N (2k / D_k) * (J - 1)^(2k - 1)

For 2D shells/membranes (plane stress sigma_zz = 0, sigeps82c.F):
    Out-of-plane stretch lambda_3 is determined iteratively via Newton-Raphson
    enforcing T_3(lambda_3) = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

_EM10 = 1e-10
_EM20 = 1e-20


@dataclass
class OgdenParams:
    """Parameters for /MAT/LAW82 (/MAT/OGDEN).

    Attributes
    ----------
    id : int
        Material identification number.
    rho0 : float
        Initial reference density.
    rhor : float
        Reference density (defaults to rho0 if 0).
    nu : float
        Poisson's ratio (clamped to 0.495 if 0.5).
    nordre : int
        Order of Ogden expansion (number of terms, 1..10).
    mu : np.ndarray
        Ogden shear coefficients mu_k (shape: (nordre,)).
    alpha : np.ndarray
        Ogden exponents alpha_k (shape: (nordre,)).
    d : np.ndarray
        Volumetric compressibility parameters D_k (shape: (nordre,)).
    g0 : float
        Initial shear modulus G_0 = sum(mu_k).
    rbulk : float
        Initial bulk modulus K = 2 / D_1.
    title : str
        Material title / description.
    """

    id: int = 1
    rho0: float = 1.0
    rhor: float = 0.0
    nu: float = 0.475
    nordre: int = 1
    mu: np.ndarray = field(default_factory=lambda: np.array([1.0], dtype=np.float64))
    alpha: np.ndarray = field(default_factory=lambda: np.array([2.0], dtype=np.float64))
    d: np.ndarray = field(default_factory=lambda: np.array([0.0], dtype=np.float64))
    g0: float = 1.0
    rbulk: float = 200.0
    title: str = ""

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
            "nu": self.nu,
            "nordre": self.nordre,
            "mu": self.mu,
            "alpha": self.alpha,
            "d": self.d,
            "g0": self.g0,
            "rbulk": self.rbulk,
            "G": self.g0,
            "K": self.rbulk,
        }


def build_law82(
    id: Any = 1,
    rho0: float = 1.0,
    nu: float = 0.475,
    nordre: int = 1,
    mu: np.ndarray | list | None = None,
    alpha: np.ndarray | list | None = None,
    d: np.ndarray | list | None = None,
    title: str = "",
    rhor: float = 0.0,
) -> OgdenParams:
    """Build and initialize OgdenParams according to starter hm_read_mat82.F.

    Supports both keyword-argument construction and card/CFG record unpacking.
    """
    if not isinstance(id, (int, np.integer, float)) and hasattr(id, "params"):
        rec = id
        mid = int(getattr(rec, "id", 1))
        rho0 = float(getattr(rec, "density", getattr(rec, "rho0", 1.0)) or 1.0)
        title = str(getattr(rec, "title", ""))
        p = getattr(rec, "params", {})
        nu = float(p.get("MAT_NU", p.get("nu", 0.475)))
        nordre = int(p.get("ORDER", p.get("order", 1)))
        mu = p.get("Mu_arr", p.get("mu", None))
        alpha = p.get("Alpha_arr", p.get("alpha", None))
        d = p.get("Gamma_arr", p.get("d", None))
        id = mid

    if mu is None:
        mu_arr = np.array([1.0] * max(nordre, 1), dtype=np.float64)
    else:
        mu_arr = np.asarray(mu, dtype=np.float64).flatten()
        nordre = len(mu_arr)

    if alpha is None:
        alpha_arr = np.array([2.0] * max(nordre, 1), dtype=np.float64)
    else:
        alpha_arr = np.asarray(alpha, dtype=np.float64).flatten()

    if d is None:
        d_arr = np.zeros(max(nordre, 1), dtype=np.float64)
    else:
        d_arr = np.asarray(d, dtype=np.float64).flatten()
        if len(d_arr) < nordre:
            d_full = np.zeros(nordre, dtype=np.float64)
            d_full[: len(d_arr)] = d_arr
            d_arr = d_full

    # Fortran hm_read_mat82.F:113-121:
    # Count first non-zero Ogden parameters from the input lines
    count = 0
    for i in range(min(nordre, len(mu_arr), len(alpha_arr))):
        if mu_arr[i] != 0.0 and alpha_arr[i] != 0.0:
            count += 1
        else:
            break
    if count > 0:
        nordre = count
        mu_arr = mu_arr[:nordre]
        alpha_arr = alpha_arr[:nordre]
        d_arr = d_arr[:nordre]

    # Initial shear modulus: hm_read_mat82.F:128-131
    g0 = float(np.sum(mu_arr))

    nu0 = float(nu)
    zep495 = 0.495
    zep499 = 0.499

    if nu0 == 0.5:
        nu0 = zep495

    if nu0 == 0.0:
        if d_arr[0] > 0.0:
            P = 2.0 / d_arr[0]
            nu_calc = (3.0 * P - 2.0 * g0) / (6.0 * P + 2.0 * g0)
            if nu_calc == 0.5:
                nu_calc = zep499
            nu_final = nu_calc
            d_arr[0] = 3.0 * (1.0 - 2.0 * nu_final) / (g0 * (1.0 + nu_final))
            rbulk = 2.0 / d_arr[0]
        else:
            nu_final = zep495
            d_arr[0] = 3.0 * (1.0 - 2.0 * nu_final) / (g0 * (1.0 + nu_final))
            rbulk = 2.0 / d_arr[0]
    else:
        nu_final = nu0
        d_arr[0] = 3.0 * (1.0 - 2.0 * nu_final) / (g0 * (1.0 + nu_final))
        rbulk = 2.0 / d_arr[0]

    if rhor == 0.0:
        rhor = rho0

    return OgdenParams(
        id=int(id),
        rho0=float(rho0),
        rhor=float(rhor),
        nu=float(nu_final),
        nordre=int(nordre),
        mu=mu_arr,
        alpha=alpha_arr,
        d=d_arr,
        g0=float(g0),
        rbulk=float(rbulk),
        title=str(title),
    )


def solid_update(
    mat: OgdenParams,
    sig: np.ndarray,
    deps: np.ndarray | None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    ismstr: int = 0,
    *,
    epsp: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """3D solid continuum Ogden hyperelastic stress update.

    Fortran reference: engine/source/materials/mat/mat082/sigeps82.F
    """
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1) if deps is not None else None

    # Handle case where 4th argument is epsp instead of eps
    if eps is not None:
        if eps.ndim == 1 and not is_1d and eps.shape[0] == sig.shape[0]:
            # Passed epsp as 4th positional argument
            epsp = eps
            eps = None
        else:
            eps_arr = eps.reshape(1, -1) if is_1d else eps
    else:
        eps_arr = None

    if eps_arr is None:
        if extra is not None and "F" in extra and extra["F"] is not None:
            # Reconstruct eps from F
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
            eps_arr = deps.copy()
        else:
            eps_arr = np.zeros_like(sig)

    if deps is None:
        deps = np.zeros_like(eps_arr)

    n = sig.shape[0]
    if n == 0:
        return sig, np.zeros(0, dtype=np.float64)

    # 1. Build symmetric 3x3 strain tensor for each element: sigeps82.F:137-144
    A = np.zeros((n, 3, 3), dtype=np.float64)
    A[:, 0, 0] = eps_arr[:, 0]
    A[:, 1, 1] = eps_arr[:, 1]
    A[:, 2, 2] = eps_arr[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = 0.5 * eps_arr[:, 3]
    A[:, 1, 2] = A[:, 2, 1] = 0.5 * eps_arr[:, 4]
    A[:, 0, 2] = A[:, 2, 0] = 0.5 * eps_arr[:, 5]

    # 2. Symmetric tensor eigendecomposition (VALPVEC_V): sigeps82.F:147-151
    w, v = np.linalg.eigh(A)

    # 3. Principal stretches: sigeps82.F:156-182
    if ismstr in (0, 2, 4):
        lam = np.exp(w)
    elif ismstr in (10, 12):
        lam = np.sqrt(np.maximum(w + 1.0, _EM20))
    else:
        lam = w + 1.0

    # 4. Relative volume J = RV = lambda_1 * lambda_2 * lambda_3: sigeps82.F:205-208
    rv = np.maximum(lam[:, 0] * lam[:, 1] * lam[:, 2], _EM20)

    # 5. Deviatoric stretches: sigeps82.F:216-225
    rvt = rv ** (-1.0 / 3.0)
    evd = lam * rvt[:, None]

    # 6. Principal Cauchy stresses S: sigeps82.F:226-270
    S = np.zeros((n, 3), dtype=np.float64)
    P = np.zeros(n, dtype=np.float64)

    for j in range(mat.nordre):
        mu_k = mat.mu[j]
        al_k = mat.alpha[j]
        d_k = mat.d[j]

        dd = (2.0 * mu_k) / (al_k * rv)
        pui_tab = np.maximum(evd, _EM20) ** al_k

        S[:, 0] += dd * ((2.0 / 3.0) * pui_tab[:, 0] - (1.0 / 3.0) * (pui_tab[:, 1] + pui_tab[:, 2]))
        S[:, 1] += dd * ((2.0 / 3.0) * pui_tab[:, 1] - (1.0 / 3.0) * (pui_tab[:, 0] + pui_tab[:, 2]))
        S[:, 2] += dd * ((2.0 / 3.0) * pui_tab[:, 2] - (1.0 / 3.0) * (pui_tab[:, 0] + pui_tab[:, 1]))

        if d_k != 0.0:
            k = j + 1
            P += 2.0 * k * (1.0 / d_k) * (rv - 1.0) ** (2 * k - 1)

    S += P[:, None]

    # 7. Rotate principal stresses to Cartesian Cauchy stresses: sigeps82.F:306-330
    T = np.einsum("nik,nk,njk->nij", v, S, v)

    sig_new = np.zeros_like(sig)
    sig_new[:, 0] = T[:, 0, 0]
    sig_new[:, 1] = T[:, 1, 1]
    sig_new[:, 2] = T[:, 2, 2]
    sig_new[:, 3] = T[:, 0, 1]
    sig_new[:, 4] = T[:, 1, 2]
    sig_new[:, 5] = T[:, 0, 2]

    if epsp is not None:
        epsp_new = np.asarray(epsp).copy()
    elif extra is not None and "epsp" in extra:
        epsp_new = np.asarray(extra["epsp"]).copy()
    else:
        epsp_new = np.zeros(n, dtype=np.float64)

    if is_1d:
        return sig_new[0], (epsp_new[0] if epsp_new.ndim > 0 else 0.0)
    return sig_new, epsp_new


def shell_update(
    mat: OgdenParams,
    sig: np.ndarray,
    deps: np.ndarray | None,
    eps: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
    ismstr: int = 0,
    *,
    epsp: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """2D shell / membrane plane-stress Ogden hyperelastic update.

    Fortran reference: engine/source/materials/mat/mat082/sigeps82c.F
    """
    is_1d = (sig.ndim == 1)
    if is_1d:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1) if deps is not None else None

    # Handle case where 4th argument is epsp instead of eps
    if eps is not None:
        if eps.ndim == 1 and not is_1d and eps.shape[0] == sig.shape[0]:
            epsp = eps
            eps = None
        else:
            eps_arr = eps.reshape(1, -1) if is_1d else eps
    else:
        eps_arr = None

    if eps_arr is None:
        if deps is not None:
            eps_arr = deps.copy()
        else:
            eps_arr = np.zeros_like(sig)

    if deps is None:
        deps = np.zeros_like(eps_arr)

    n = sig.shape[0]
    if n == 0:
        return sig, np.zeros(0, dtype=np.float64)

    # 1. Analytical in-plane eigenvalues/eigenvectors: sigeps82c.F:108-132
    trav = eps_arr[:, 0] + eps_arr[:, 1]
    diff = eps_arr[:, 0] - eps_arr[:, 1]
    rootv = np.sqrt(diff * diff + eps_arr[:, 2] * eps_arr[:, 2])

    evv1 = 0.5 * (trav + rootv)
    evv2 = 0.5 * (trav - rootv)

    # 2. In-plane principal stretches: sigeps82c.F:134-146
    if ismstr in (1, 3):
        lam1 = evv1 + 1.0
        lam2 = evv2 + 1.0
    elif ismstr in (10, 12):
        lam1 = np.sqrt(np.maximum(evv1 + 1.0, _EM20))
        lam2 = np.sqrt(np.maximum(evv2 + 1.0, _EM20))
    else:
        lam1 = np.exp(evv1)
        lam2 = np.exp(evv2)

    # Initial out-of-plane stretch lambda_3 from extra["uvar"][:, 0] or 1.0
    if extra is not None and "uvar" in extra and extra["uvar"] is not None:
        lam3 = np.asarray(extra["uvar"][:, 0], dtype=np.float64).copy()
    else:
        lam3 = np.ones(n, dtype=np.float64)

    ev = np.column_stack([lam1, lam2, lam3])

    # 3. 3-step Newton-Raphson enforcing T_3(lambda_3) = 0: sigeps82c.F:150-225
    for _ in range(3):
        rv = np.maximum(ev[:, 0] * ev[:, 1] * ev[:, 2], _EM20)
        rvt = rv ** (-1.0 / 3.0)
        evm = ev * rvt[:, None]

        partt = np.zeros(n, dtype=np.float64)
        partp = np.zeros(n, dtype=np.float64)
        partt2 = np.zeros(n, dtype=np.float64)
        partp2 = np.zeros(n, dtype=np.float64)

        for k_idx in range(mat.nordre):
            k = k_idx + 1
            mu_k = mat.mu[k_idx]
            al_k = mat.alpha[k_idx]
            d_k = mat.d[k_idx]

            evma1 = np.maximum(evm[:, 0], _EM20) ** al_k
            evma2 = np.maximum(evm[:, 1], _EM20) ** al_k
            evma3 = np.maximum(evm[:, 2], _EM20) ** al_k

            sum_m = (evma1 + evma2 + evma3) / 3.0
            dd = 2.0 * mu_k / al_k
            partt += dd * (evma3 - sum_m)
            partt2 += 2.0 * mu_k * (evma1 + evma2 + 4.0 * evma3) / 9.0

            if d_k != 0.0:
                k2 = 2 * k
                dd_p = k2 / d_k
                partp += dd_p * (rv - 1.0) ** (k2 - 1)
                dd_p2 = k2 * (k2 - 1.0) / d_k
                if k == 1:
                    partp2 += dd_p2
                else:
                    partp2 += dd_p2 * (rv - 1.0) ** (k2 - 2)

        t3 = partt / rv + partp
        dpartt = partt2 / rv + partp2 - t3
        denom = np.where(np.abs(dpartt) > _EM20, dpartt, 1.0)
        ev[:, 2] = np.maximum(ev[:, 2] * (1.0 - t3 / denom), 1e-12)

    # Store lambda_3 in extra["uvar"][:, 0]: sigeps82c.F:285-287
    if extra is not None:
        if "uvar" not in extra or extra["uvar"] is None or extra["uvar"].shape[0] != n:
            extra["uvar"] = np.ones((n, 1), dtype=np.float64)
        extra["uvar"][:, 0] = ev[:, 2]

    # 4. Recalculate principal stresses T1, T2: sigeps82c.F:228-283
    rv = np.maximum(ev[:, 0] * ev[:, 1] * ev[:, 2], _EM20)
    rvt = rv ** (-1.0 / 3.0)
    evm = ev * rvt[:, None]

    dwdl = np.zeros((n, 3), dtype=np.float64)
    partp = np.zeros(n, dtype=np.float64)

    for k_idx in range(mat.nordre):
        k = k_idx + 1
        mu_k = mat.mu[k_idx]
        al_k = mat.alpha[k_idx]
        d_k = mat.d[k_idx]

        evma1 = np.maximum(evm[:, 0], _EM20) ** al_k
        evma2 = np.maximum(evm[:, 1], _EM20) ** al_k
        evma3 = np.maximum(evm[:, 2], _EM20) ** al_k

        sum_m = (evma1 + evma2 + evma3) / 3.0
        dd = mu_k / al_k
        dwdl[:, 0] += dd * (evma1 - sum_m)
        dwdl[:, 1] += dd * (evma2 - sum_m)
        dwdl[:, 2] += dd * (evma3 - sum_m)

        if d_k != 0.0:
            k2 = 2 * k
            partp += (k2 / d_k) * (rv - 1.0) ** (k2 - 1)

    t1 = 2.0 * dwdl[:, 0] / rv + partp
    t2 = 2.0 * dwdl[:, 1] / rv + partp

    # 5. Transform principal Cauchy stresses to global directions: sigeps82c.F:290-296
    iso_mask = (rootv < _EM10)
    inv_root = np.where(iso_mask, 0.0, 1.0 / np.maximum(rootv, _EM20))

    eigv = np.zeros((n, 3, 2), dtype=np.float64)
    eigv[:, 0, 0] = np.where(iso_mask, 1.0, inv_root * (eps_arr[:, 0] - evv2))
    eigv[:, 1, 0] = np.where(iso_mask, 1.0, inv_root * (eps_arr[:, 1] - evv2))
    eigv[:, 2, 0] = np.where(iso_mask, 0.0, inv_root * (0.5 * eps_arr[:, 2]))

    eigv[:, 0, 1] = np.where(iso_mask, 0.0, inv_root * (evv1 - eps_arr[:, 0]))
    eigv[:, 1, 1] = np.where(iso_mask, 0.0, inv_root * (evv1 - eps_arr[:, 1]))
    eigv[:, 2, 1] = np.where(iso_mask, 0.0, -inv_root * (0.5 * eps_arr[:, 2]))

    sig_new = np.zeros_like(sig)
    sig_new[:, 0] = eigv[:, 0, 0] * t1 + eigv[:, 0, 1] * t2
    sig_new[:, 1] = eigv[:, 1, 0] * t1 + eigv[:, 1, 1] * t2
    sig_new[:, 2] = eigv[:, 2, 0] * t1 + eigv[:, 2, 1] * t2

    # 6. Transverse shear stresses (sigeps82c.F:294-295)
    if sig.shape[1] >= 5:
        deps_yz = deps[:, 3] if deps.shape[1] > 3 else 0.0
        deps_zx = deps[:, 4] if deps.shape[1] > 4 else 0.0
        sig_new[:, 3] = sig[:, 3] + mat.g0 * deps_yz
        sig_new[:, 4] = sig[:, 4] + mat.g0 * deps_zx

    # 7. Thickness update: sigeps82c.F:300-301
    dezz = -mat.nu / (1.0 - mat.nu) * (deps[:, 0] + deps[:, 1])
    if extra is not None:
        thk0 = extra.get("thklyl", extra.get("thk0", 1.0))
        if "thkn" in extra and extra["thkn"] is not None:
            extra["thkn"] += dezz * thk0
        elif "thk" in extra and extra["thk"] is not None:
            extra["thk"] += dezz * thk0

    if epsp is not None:
        epsp_new = np.asarray(epsp).copy()
    elif extra is not None and "epsp" in extra:
        epsp_new = np.asarray(extra["epsp"]).copy()
    else:
        epsp_new = np.zeros(n, dtype=np.float64)

    if is_1d:
        return sig_new[0], (epsp_new[0] if epsp_new.ndim > 0 else 0.0)
    return sig_new, epsp_new


def solid_sound_speed(
    mat: OgdenParams,
    rho: np.ndarray | float,
    eps: np.ndarray | None = None,
    ismstr: int = 0,
    **kwargs: Any,
) -> np.ndarray | float:
    """Nonlinear sound speed for 3D solids.

    Fortran reference: engine/source/materials/mat/mat082/sigeps82.F:273-335
    """
    if rho is None:
        rho = getattr(mat, "rho0", 1.0)

    extra = kwargs.get("extra")
    if extra is not None:
        if eps is None:
            eps = extra.get("eps", extra.get("F", None))
    gmax = mat.g0
    rbulk = mat.rbulk

    if eps is None:
        return np.sqrt(((4.0 / 3.0) * gmax + rbulk) / rho)

    is_scalar_rho = np.isscalar(rho)
    eps_arr = np.asarray(eps, dtype=np.float64)
    is_1d = (eps_arr.ndim == 1)
    if is_1d:
        eps_arr = eps_arr.reshape(1, -1)
    n = eps_arr.shape[0]

    A = np.zeros((n, 3, 3), dtype=np.float64)
    A[:, 0, 0] = eps_arr[:, 0]
    A[:, 1, 1] = eps_arr[:, 1]
    A[:, 2, 2] = eps_arr[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = 0.5 * eps_arr[:, 3]
    A[:, 1, 2] = A[:, 2, 1] = 0.5 * eps_arr[:, 4]
    A[:, 0, 2] = A[:, 2, 0] = 0.5 * eps_arr[:, 5]

    w, _ = np.linalg.eigh(A)
    if ismstr in (0, 2, 4):
        lam = np.exp(w)
    elif ismstr in (10, 12):
        lam = np.sqrt(np.maximum(w + 1.0, _EM20))
    else:
        lam = w + 1.0

    rv = np.maximum(lam[:, 0] * lam[:, 1] * lam[:, 2], _EM20)
    rvt = rv ** (-1.0 / 3.0)
    evd = lam * rvt[:, None]

    gtmax = np.full(n, gmax, dtype=np.float64)
    rkmax = np.full(n, rbulk, dtype=np.float64)

    cii = np.zeros((n, 3), dtype=np.float64)
    for j in range(mat.nordre):
        mu_k = mat.mu[j]
        al_k = mat.alpha[j]
        if mu_k != 0.0:
            lam_al = np.maximum(evd, _EM20) ** al_k
            amax_val = np.sum(lam_al, axis=1) / 3.0
            cii += mu_k * (lam_al + amax_val[:, None])

    for j in range(1, mat.nordre):
        d_k = mat.d[j]
        if d_k == 0.0:
            continue
        k = j + 1
        pp = 2.0 * k * (2.0 * k - 1.0) / d_k
        jj = 2 * k - 2
        mask = np.abs(rv - 1.0) >= _EM20
        rkmax[mask] += pp * ((rv[mask] - 1.0) ** jj)

    amax = 0.5 * np.max(cii, axis=1) / max(gmax, 1e-12)
    eti = np.maximum(1.0, amax * 0.81)
    gtmax = gmax * eti
    rkmax = np.maximum(rbulk, rkmax)

    rho_arr = np.asarray(rho, dtype=np.float64)
    c = np.sqrt(((4.0 / 3.0) * gtmax + rkmax) / rho_arr)
    if is_scalar_rho and is_1d:
        return float(c[0])
    return c


def shell_sound_speed(
    mat: OgdenParams,
    rho: np.ndarray | float | None = None,
    **kwargs: Any,
) -> np.ndarray | float:
    """Sound speed for 2D shells/membranes.

    Fortran reference: engine/source/materials/mat/mat082/sigeps82c.F:303
    """
    if rho is None:
        rho = getattr(mat, "rho0", 1.0)
    modulus = (2.0 / 3.0) * mat.g0 + mat.rbulk
    return np.sqrt(modulus / rho)


def consistent_solid_tangent(
    mat: OgdenParams,
    eps: np.ndarray,
    deps: np.ndarray | None = None,
    dt: float = 1e-6,
    h: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent for 3D solid continuum elements.

    Returns algorithmic tangent tensor with shape (n, 6, 6).
    """
    if eps.ndim == 3 and eps.shape[1:] == (3, 3):
        # Passed deformation gradient F (m, 3, 3)
        F = eps
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
    else:
        eps_arr = np.asarray(eps, dtype=np.float64)
        if eps_arr.ndim == 1:
            eps_arr = eps_arr.reshape(1, -1)

    n = eps_arr.shape[0]
    C = np.zeros((n, 6, 6), dtype=np.float64)
    dummy_sig = np.zeros((n, 6), dtype=np.float64)
    dummy_deps = np.zeros((n, 6), dtype=np.float64)

    for j in range(6):
        eps_p = eps_arr.copy()
        eps_p[:, j] += h
        eps_m = eps_arr.copy()
        eps_m[:, j] -= h

        sp, _ = solid_update(mat, dummy_sig, dummy_deps, eps_p, dt)
        sm, _ = solid_update(mat, dummy_sig, dummy_deps, eps_m, dt)

        C[:, :, j] = (sp - sm) / (2.0 * h)

    return C


def consistent_shell_tangent(
    mat: OgdenParams,
    eps: np.ndarray,
    deps: np.ndarray | None = None,
    dt: float = 1e-6,
    h: float = 1e-7,
    **kwargs: Any,
) -> np.ndarray:
    """Consistent algorithmic tangent for 2D shells/membranes.

    Returns in-plane condensed tangent tensor with shape (n, 3, 3).
    """
    eps_arr = np.asarray(eps, dtype=np.float64)
    if eps_arr.ndim == 1:
        eps_arr = eps_arr.reshape(1, -1)
    n = eps_arr.shape[0]

    C = np.zeros((n, 3, 3), dtype=np.float64)
    dummy_sig = np.zeros((n, 3), dtype=np.float64)
    dummy_deps = np.zeros((n, 3), dtype=np.float64)

    for j in range(3):
        eps_p = eps_arr.copy()
        eps_p[:, j] += h
        eps_m = eps_arr.copy()
        eps_m[:, j] -= h

        sp, _ = shell_update(mat, dummy_sig, dummy_deps, eps_p, dt)
        sm, _ = shell_update(mat, dummy_sig, dummy_deps, eps_m, dt)

        C[:, :, j] = (sp[:, :3] - sm[:, :3]) / (2.0 * h)

    return C


# Aliases for package dispatch compatibility
sound_speed = solid_sound_speed
sound_speed_shell = shell_sound_speed
solid_tangent = consistent_solid_tangent
shell_tangent = consistent_shell_tangent
law82_solid_update = solid_update
law82_shell_update = shell_update
law82_solid_sound_speed = solid_sound_speed
law82_shell_sound_speed = shell_sound_speed
law82_consistent_solid_tangent = consistent_solid_tangent
law82_consistent_shell_tangent = consistent_shell_tangent
extra_shapes = lambda mat, nip=None: {"uvar": (1,)}


def _register() -> None:
    try:
        from ..input.mat_reader import MAT_PHYSICS_REGISTRY
        for key in (82, "82", "LAW82", "OGDEN", "MAT_LAW82", "MAT_OGDEN", "LAW82_OGDEN", "OGDEN_82", "MAT_OGDEN_82"):
            MAT_PHYSICS_REGISTRY[key] = build_law82
    except Exception:
        pass


_register()
