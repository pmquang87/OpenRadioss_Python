"""
LAW119 — 2D shell seatbelt material model (/MAT/LAW119, /MAT/SH_SEATBELT).

Fortran origin:
  - ``starter/source/materials/mat/mat119/hm_read_mat119.F`` (starter card reader)
  - ``starter/source/materials/mat/mat119/law119_upd.F`` (derived parameters & intersection)
  - ``engine/source/materials/mat/mat119/sigeps119c.F`` (shell material caller & coating)
  - ``engine/source/materials/mat/mat119/law119_membrane.F`` (orthotropic membrane & hysteresis)

Theory:
  - Total in-plane orthotropic membrane behavior:
      nu21 = nu12 * FSCALET
      DET = 1 / (1 - nu12 * nu21)
      A11 = E11 * DET
      A22 = A11 * FSCALET
      A12 = A11 * nu21
  - Tension vs compression tagging via in-plane principal strains:
      S = 0.5 * (eps_xx + eps_yy), D = 0.5 * (eps_xx - eps_yy), R = sqrt(eps_xy^2 + D^2)
      P1 = S + R, P2 = S - R
      If P1 > 0 and P1 >= -P2:
          tension branch (beta = 1.0)
      Else:
          compression branch (beta = RCOMP, clamped to >= 1e-3, modelling yarn wrinkling/buckling)
  - Equivalent strain:
      eps_q = sqrt((eps_xx^2 + eps_yy^2) / (1 + nu21^2))
  - Nonlinear loading / unloading with hysteresis (FUN_L, FUN_UL, Ireload):
      Tracks peak equivalent strain (Emax) and equivalent von Mises stress (Smax)
      with unloading / reloading hysteresis.
  - Optional coating layer (ECOAT, NUCOAT, TCOAT).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..model.entities import Material


def _eval_funct(func: Any, x: float) -> float:
    """Evaluate curve function at scalar abscissa x."""
    if func is None or func == 0:
        return 0.0
    if hasattr(func, "eval"):
        return float(func.eval(x))
    if callable(func):
        return float(func(x))
    if isinstance(func, (int, float)):
        return float(func)
    return 0.0


def _eval_funct_deriv(func: Any, x: float) -> float:
    """Evaluate derivative dy/dx of curve function at scalar abscissa x."""
    if func is None or func == 0:
        return 0.0
    if hasattr(func, "slope") and hasattr(func, "x") and len(func.x) > 1:
        x_arr = func.x
        slopes = func.slope
        if x <= x_arr[0]:
            return float(slopes[0])
        if x >= x_arr[-1]:
            return float(slopes[-1])
        idx = int(np.searchsorted(x_arr, x, side="right") - 1)
        idx = max(0, min(len(slopes) - 1, idx))
        return float(slopes[idx])
    if hasattr(func, "eval"):
        h = max(1e-6, abs(x) * 1e-5)
        return float((func.eval(x + h) - func.eval(x - h)) / (2.0 * h))
    if callable(func):
        h = max(1e-6, abs(x) * 1e-5)
        return float((func(x + h) - func(x - h)) / (2.0 * h))
    return 0.0


class Law119Seatbelt(Material):
    """LAW119 (/MAT/SH_SEATBELT) 2D shell seatbelt material."""

    def __init__(self, id: int, rho0: float = 0.0, title: str = "", params: dict[str, Any] | None = None):
        p = params or {}
        super().__init__(id=id, law=119, rho0=rho0, title=title, params=p)
        self.e11 = float(p.get("STIFF1", p.get("stiff1", p.get("E11", p.get("e11", p.get("E", 1.0))))))
        self.e22 = float(p.get("E22", p.get("e22", 0.0)))
        self.nu12 = float(p.get("NU12", p.get("nu12", p.get("nu", 0.19))))
        self.g12 = float(p.get("G12", p.get("g12", 0.0)))
        self.re = float(p.get("RE", p.get("re", p.get("rcomp", p.get("RCOMP", 1.0)))))
        self.rcomp = max(1e-3, self.re if self.re > 0.0 else 1.0)
        self.lmin = float(p.get("LMIN", p.get("lmin", 0.0)))
        self.fun_l = p.get("FUN_L", p.get("fun_l", 0))
        self.fun_ul = p.get("FUN_UL", p.get("fun_ul", 0))
        self.fscale1 = float(p.get("Fcoeft1", p.get("fscale1", p.get("fcoeft1", 1.0))))
        self.fscale2 = float(p.get("Fcoeft2", p.get("fscale2", p.get("fcoeft2", 1.0))))
        self.fscale22 = float(p.get("Fcoeft22", p.get("fscale22", p.get("fcoeft22", 1.0))))
        self.ireload = int(p.get("Ireload", p.get("ireload", 0)))
        self.ecoat = float(p.get("ECOAT", p.get("ecoat", 0.0)))
        self.nucoat = float(p.get("NUCOAT", p.get("nucoat", self.nu12)))
        self.tcoat = float(p.get("TCOAT", p.get("tcoat", 0.0)))
        self.c_rate = float(p.get("C_RATE", p.get("c_rate", 0.0)))
        self.eps0_rate = float(p.get("EPS0", p.get("eps0", p.get("eps0_rate", 1.0))))

        # Derived orthotropic constants
        fscalet = self.fscale22 if self.fscale22 > 0.0 else (self.e22 / self.e11 if self.e11 > 0.0 else 0.1)
        self.fscalet = fscalet
        self.nu21 = self.nu12 * fscalet
        self.det = 1.0 / max(1e-12, 1.0 - self.nu12 * self.nu21)
        self.a11 = self.e11 * self.det
        self.a22 = self.a11 * fscalet
        self.a12 = self.a11 * self.nu21

        # Fallback shear modulus if zero
        if self.g12 <= 0.0:
            self.g12 = self.e11 / (2.0 * (1.0 + self.nu12))

    @property
    def G(self) -> float:
        return self.g12

    def sound_speed_shell(self) -> float:
        """Characteristic acoustic sound speed for Courant time step."""
        c1 = max(self.e11, self.e22) * self.det
        rho = max(self.rho0, 1e-12)
        return float(math.sqrt(c1 / rho))


def extra_shapes(mat: Any, nip: int | None = None) -> dict[str, tuple[int, ...]]:
    """Persistent state per shell integration point for LAW119 seatbelt."""
    return {
        "eps119": (nip, 3) if nip else (3,),
        "uv119": (nip, 10) if nip else (10,),
    }


def shell_membrane_tangent(mat: Any) -> np.ndarray:
    """(3, 3) plane-stress orthotropic membrane tangent matrix."""
    a11 = getattr(mat, "a11", mat.params.get("A11", 1.0))
    a22 = getattr(mat, "a22", mat.params.get("A22", 1.0))
    a12 = getattr(mat, "a12", mat.params.get("A12", 0.0))
    g12 = getattr(mat, "g12", mat.params.get("G12", 0.0))
    return np.array([
        [a11, a12, 0.0],
        [a12, a22, 0.0],
        [0.0, 0.0, g12],
    ], dtype=float)


def consistent_shell_tangent(mat: Any, sig: np.ndarray | None = None, deps: np.ndarray | None = None, extra: dict | None = None) -> np.ndarray:
    """(3, 3) or (n, 3, 3) consistent shell tangent tensor for LAW119."""
    c_el = shell_membrane_tangent(mat)
    if extra is not None and "eps119" in extra:
        eps = extra["eps119"]
        if eps.ndim == 2:
            n = eps.shape[0]
            s = 0.5 * (eps[:, 0] + eps[:, 1])
            d = 0.5 * (eps[:, 0] - eps[:, 1])
            r = np.sqrt(eps[:, 2]**2 + d**2)
            p1 = s + r
            p2 = s - r
            is_tension = (p1 > 0.0) & (p1 >= -p2)
            rcomp = getattr(mat, "rcomp", 1.0)
            beta = np.where(is_tension, 1.0, rcomp)[:, None, None]
            return np.broadcast_to(c_el, (n, 3, 3)) * beta
    return c_el


def shell_update(
    mat: Any,
    sig: np.ndarray,
    deps: np.ndarray,
    epsp: np.ndarray | None = None,
    dt: float = 0.0,
    extra: dict | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Vectorized layer update for LAW119 shell elements (sigeps119c.F & law119_membrane.F).

    Parameters
    ----------
    mat : Law119Seatbelt or Material
        Material definition.
    sig : np.ndarray
        (n, 3) or (n, 5) stress tensor [xx, yy, xy, (yz, zx)].
    deps : np.ndarray
        (n, 3) or (n, 5) strain increment [xx, yy, xy, (yz, zx)].
    epsp : np.ndarray, optional
        Effective plastic strain.
    dt : float
        Time step increment.
    extra : dict, optional
        Persistent internal state arrays (eps119, uv119).

    Returns
    -------
    sig : np.ndarray
        Updated stress array.
    epsp : np.ndarray
        Effective plastic strain array.
    """
    sig_shape = sig.shape
    if sig.ndim == 1:
        sig = sig.reshape(1, -1)
        deps = deps.reshape(1, -1)

    n = sig.shape[0]
    if n == 0:
        return sig.reshape(sig_shape), epsp

    p = getattr(mat, "params", {})
    a11 = getattr(mat, "a11", float(p.get("A11", p.get("E", 1.0))))
    a22 = getattr(mat, "a22", float(p.get("A22", p.get("E", 1.0))))
    a12 = getattr(mat, "a12", float(p.get("A12", 0.0)))
    g12 = getattr(mat, "g12", float(p.get("G12", p.get("G", 0.0))))
    nu12 = getattr(mat, "nu12", float(p.get("NU12", 0.19)))
    nu21 = getattr(mat, "nu21", nu12 * 0.1)
    det = getattr(mat, "det", 1.0 / max(1e-12, 1.0 - nu12 * nu21))
    fscalet = getattr(mat, "fscalet", 0.1)
    rcomp = getattr(mat, "rcomp", max(1e-3, float(p.get("RE", p.get("re", 1.0)))))
    fscale1 = getattr(mat, "fscale1", float(p.get("Fcoeft1", 1.0)))
    fscale2 = getattr(mat, "fscale2", float(p.get("Fcoeft2", 1.0)))
    ireload = getattr(mat, "ireload", int(p.get("Ireload", 0)))
    c_rate = getattr(mat, "c_rate", float(p.get("C_RATE", 0.0)))
    eps0_rate = getattr(mat, "eps0_rate", float(p.get("EPS0", 1.0)))

    # Functions retrieval
    func = getattr(mat, "fun_l", p.get("FUN_L", p.get("fun_l", 0)))
    fund = getattr(mat, "fun_ul", p.get("FUN_UL", p.get("fun_ul", 0)))
    functions = getattr(mat, "functions", None)
    if functions is None and extra is not None:
        functions = extra.get("functions")

    f_l = None
    f_ul = None
    if functions is not None:
        if func in functions:
            f_l = functions[func]
        if fund in functions:
            f_ul = functions[fund]
    if f_l is None and (hasattr(func, "eval") or callable(func)):
        f_l = func
    if f_ul is None and (hasattr(fund, "eval") or callable(fund)):
        f_ul = fund

    if extra is None:
        extra = {}
    eps = extra.get("eps119")
    if eps is None or eps.shape[0] != n:
        eps = np.zeros((n, 3), dtype=float)
        extra["eps119"] = eps

    uv = extra.get("uv119")
    if uv is None or uv.shape[0] != n:
        uv = np.zeros((n, 10), dtype=float)
        extra["uv119"] = uv

    # Accumulate in-plane strain
    eps[:, :min(3, deps.shape[1])] += deps[:, :min(3, deps.shape[1])]
    sigo = sig.copy()

    # Principal strain calculation (law119_membrane.F:126-146)
    s = 0.5 * (eps[:, 0] + eps[:, 1])
    d = 0.5 * (eps[:, 0] - eps[:, 1])
    r = np.sqrt(eps[:, 2]**2 + d**2)
    p1 = s + r
    p2 = s - r
    is_tension = (p1 > 0.0) & (p1 >= -p2)
    beta = np.where(is_tension, 1.0, rcomp)

    # In-plane shear stresses
    sig[:, 2] = g12 * eps[:, 2] * beta
    if sig.shape[1] >= 5 and deps.shape[1] >= 5:
        sig[:, 3] += g12 * deps[:, 3] * beta
        sig[:, 4] += g12 * deps[:, 4] * beta

    # Direct in-plane stresses
    if f_l is None:
        # Linear elastic orthotropic fabric
        sig[:, 0] = (a11 * eps[:, 0] + a12 * eps[:, 1]) * beta
        sig[:, 1] = (a12 * eps[:, 0] + a22 * eps[:, 1]) * beta
    else:
        # Nonlinear fabric behavior
        epsq = np.sqrt((eps[:, 0]**2 + eps[:, 1]**2) / (1.0 + nu21**2))

        # Strain-rate scaling factor
        rate_fac = 1.0
        if dt > 0.0 and c_rate > 0.0 and eps0_rate > 0.0:
            depsq = np.abs(epsq - uv[:, 7])
            eps_dot = depsq / dt
            rate_fac = 1.0 + c_rate * np.log(np.maximum(1.0, eps_dot / eps0_rate))

        if f_ul is None:
            # Nonlinear loading curve without separate unloading curve
            etl = np.array([_eval_funct_deriv(f_l, float(eq)) for eq in epsq]) * fscale1 * rate_fac
            a11_eff = etl * det
            a22_eff = a11_eff * fscalet
            a12_eff = a11_eff * nu21

            sig_tens_xx = sigo[:, 0] + a11_eff * deps[:, 0] + a12_eff * deps[:, 1]
            sig_tens_yy = sigo[:, 1] + a12_eff * deps[:, 0] + a22_eff * deps[:, 1]
            sig_comp_xx = (a11 * eps[:, 0] + a12 * eps[:, 1]) * rcomp
            sig_comp_yy = (a12 * eps[:, 0] + a22 * eps[:, 1]) * rcomp

            sig[:, 0] = np.where(is_tension, sig_tens_xx, sig_comp_xx)
            sig[:, 1] = np.where(is_tension, sig_tens_yy, sig_comp_yy)
        else:
            # Full hysteresis (loading, unloading, reloading) - law119_membrane.F:254-379
            xint = float(p.get("XINT", 1.0))
            yint = float(p.get("YINT", 1.0))

            for i in range(n):
                eq_i = epsq[i]
                if not is_tension[i]:
                    # Compression branch
                    sig[i, 0] = (a11 * eps[i, 0] + a12 * eps[i, 1]) * rcomp
                    sig[i, 1] = (a12 * eps[i, 0] + a22 * eps[i, 1]) * rcomp
                    uv[i, 0] = 1e-20  # EMAX
                    uv[i, 1] = 0.0    # SMAX
                    uv[i, 2] = 0.0    # EMINRL
                    uv[i, 3] = 1e-20  # EMAXRL
                    uv[i, 4] = 0.0    # SMINRL
                    uv[i, 5] = 0.0    # SMAXRL
                    uv[i, 9] = -1.0   # compression flag
                else:
                    dw = eq_i - uv[i, 7]
                    svm = math.sqrt(max(0.0, sigo[i, 0]**2 + sigo[i, 1]**2 - sigo[i, 0] * sigo[i, 1]))

                    if dw < 0.0 and uv[i, 9] >= 0.0:
                        # Unloading branch
                        emax_rl = max(1e-20, uv[i, 3])
                        smax_rl = max(1e-20, uv[i, 5])
                        x_val = eq_i * xint / emax_rl
                        etu = _eval_funct_deriv(f_ul, x_val)
                        etx = etu * (smax_rl / emax_rl) * (xint / max(1e-20, yint))
                        if eq_i > 0.0:
                            etx = max(etx, svm / eq_i)
                        a11_eff = etx * det * fscale2 * rate_fac
                        a22_eff = a11_eff * fscalet
                        a12_eff = a11_eff * nu21

                        sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                        sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                        new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))
                        uv[i, 2] = eq_i      # EMINRL
                        uv[i, 4] = new_svm   # SMINRL

                    elif svm >= uv[i, 1] or uv[i, 9] == -1.0:
                        # Loading branch
                        etl = _eval_funct_deriv(f_l, eq_i) * fscale1 * rate_fac
                        a11_eff = etl * det
                        a22_eff = a11_eff * fscalet
                        a12_eff = a11_eff * nu21

                        sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                        sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                        new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))

                        uv[i, 0] = max(1e-20, eq_i)    # EMAX
                        uv[i, 1] = new_svm             # SMAX
                        uv[i, 2] = eq_i                # EMINRL
                        uv[i, 3] = max(1e-20, eq_i)    # EMAXRL
                        uv[i, 4] = new_svm             # SMINRL
                        uv[i, 5] = new_svm             # SMAXRL
                        uv[i, 9] = 0.0                 # clear compression flag

                    else:
                        # Reloading branch
                        emax_val = max(1e-20, uv[i, 0])
                        smax_val = max(1e-20, uv[i, 1])
                        emin_rl = uv[i, 2]
                        smin_rl = uv[i, 4]

                        if ireload == 1:
                            # Reloading follows loading curve
                            etl = _eval_funct_deriv(f_l, eq_i) * fscale1 * rate_fac
                            ht = (smax_val - smin_rl) / max(1e-20, emax_val - emin_rl)
                            hf = smax_val / emax_val
                            etx = etl * ht / max(1e-20, hf)
                            a11_eff = etx * det
                            a22_eff = a11_eff * fscalet
                            a12_eff = a11_eff * nu21

                            sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                            sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                            new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))
                            uv[i, 3] = max(1e-20, eq_i)
                            uv[i, 5] = new_svm
                        else:
                            # Reloading follows unloading curve
                            emax_rl = max(1e-20, uv[i, 3])
                            x_val = eq_i * xint / emax_rl
                            etu = _eval_funct_deriv(f_ul, x_val)
                            ht = smax_val / emax_val
                            hf = yint / max(1e-20, xint)
                            etx = fscale2 * etu * ht / max(1e-20, hf) * rate_fac
                            a11_eff = etx * det
                            a22_eff = a11_eff * fscalet
                            a12_eff = a11_eff * nu21

                            sig[i, 0] = sigo[i, 0] + a11_eff * deps[i, 0] + a12_eff * deps[i, 1]
                            sig[i, 1] = sigo[i, 1] + a12_eff * deps[i, 0] + a22_eff * deps[i, 1]
                            new_svm = math.sqrt(max(0.0, sig[i, 0]**2 + sig[i, 1]**2 - sig[i, 0] * sig[i, 1]))
                            uv[i, 3] = max(1e-20, emax_val)
                            uv[i, 5] = smax_val

        uv[:, 7] = epsq

    # Coating layer contribution (sigeps119c.F:116-132)
    ecoat = getattr(mat, "ecoat", float(p.get("ECOAT", 0.0)))
    tcoat = getattr(mat, "tcoat", float(p.get("TCOAT", 0.0)))
    nucoat = getattr(mat, "nucoat", float(p.get("NUCOAT", nu12)))
    if ecoat > 0.0 and tcoat > 0.0:
        a1c = ecoat / max(1e-12, 1.0 - nucoat**2)
        a2c = a1c * nucoat
        gc = ecoat / (2.0 * (1.0 + nucoat))
        sig_coat_xx = a1c * eps[:, 0] + a2c * eps[:, 1]
        sig_coat_yy = a2c * eps[:, 0] + a1c * eps[:, 1]
        sig_coat_xy = gc * eps[:, 2]

        thk = extra.get("thk") if extra is not None else None
        if thk is not None:
            w_c = np.clip(2.0 * tcoat / np.maximum(thk, 1e-12), 0.0, 1.0)
            sig[:, 0] = (1.0 - w_c) * sig[:, 0] + w_c * sig_coat_xx
            sig[:, 1] = (1.0 - w_c) * sig[:, 1] + w_c * sig_coat_yy
            sig[:, 2] = (1.0 - w_c) * sig[:, 2] + w_c * sig_coat_xy
        else:
            # If thickness not passed, combine membrane + coating
            sig[:, 0] += sig_coat_xx * (tcoat / max(1e-3, getattr(mat, "lmin", 1.0)))
            sig[:, 1] += sig_coat_yy * (tcoat / max(1e-3, getattr(mat, "lmin", 1.0)))
            sig[:, 2] += sig_coat_xy * (tcoat / max(1e-3, getattr(mat, "lmin", 1.0)))

    return sig.reshape(sig_shape), epsp


def solid_update(mat: Any, sig: np.ndarray, deps: np.ndarray, dt: float = 0.0, extra: dict | None = None) -> Any:
    """LAW119 is defined strictly for shell elements (sigeps119c.F)."""
    raise NotImplementedError("LAW119 (/MAT/SH_SEATBELT) is implemented for shell elements only.")


def sound_speed(mat: Any, rho: float | None = None, extra: dict | None = None, is_shell: bool = True) -> float:
    """Characteristic sound speed for LAW119."""
    if hasattr(mat, "sound_speed_shell"):
        return mat.sound_speed_shell()
    e11 = getattr(mat, "e11", mat.params.get("E11", mat.params.get("E", 1.0)))
    e22 = getattr(mat, "e22", mat.params.get("E22", 0.0))
    det = getattr(mat, "det", 1.0)
    c1 = max(e11, e22) * det
    r0 = rho if rho is not None else getattr(mat, "rho0", 1.0)
    return float(math.sqrt(max(c1, 1e-12) / max(r0, 1e-12)))


def build_law119(rec: Any) -> Law119Seatbelt:
    """Constructor for /MAT/LAW119 (/MAT/SH_SEATBELT) 2D shell seatbelt material."""
    p = rec.params
    stiff1 = float(p.get("STIFF1", p.get("stiff1", p.get("E11", 0.0))))
    damp1 = float(p.get("DAMP1", p.get("damp1", 0.0)))
    re = float(p.get("RE", p.get("re", 1.0)))
    lmin = float(p.get("LMIN", p.get("lmin", 0.0)))
    fun_l = p.get("FUN_L", p.get("fun_l", 0))
    fun_ul = p.get("FUN_UL", p.get("fun_ul", 0))
    fcoeft1 = float(p.get("Fcoeft1", p.get("fcoeft1", p.get("fscale1", 1.0))))
    fcoeft2 = float(p.get("Fcoeft2", p.get("fcoeft2", p.get("fscale2", 1.0))))
    ireload = int(p.get("Ireload", p.get("ireload", 0)))
    e22 = float(p.get("E22", p.get("e22", 0.0)))
    nu12 = float(p.get("NU12", p.get("nu12", 0.19)))
    g12 = float(p.get("G12", p.get("g12", 0.0)))
    e = e22 if e22 > 0.0 else (stiff1 if stiff1 > 0.0 else 1.0)
    nu = nu12 if (0.0 <= nu12 < 0.5) else 0.19

    params = {
        "E": e,
        "nu": nu,
        "STIFF1": stiff1,
        "stiff1": stiff1,
        "E11": stiff1,
        "e11": stiff1,
        "damp1": damp1,
        "re": re,
        "RE": re,
        "RCOMP": re,
        "lmin": lmin,
        "fun_l": fun_l,
        "fun_ul": fun_ul,
        "fcoeft1": fcoeft1,
        "fscale1": fcoeft1,
        "fcoeft2": fcoeft2,
        "fscale2": fcoeft2,
        "ireload": ireload,
        "e22": e22,
        "E22": e22,
        "nu12": nu12,
        "NU12": nu12,
        "g12": g12,
        "G12": g12,
        "fcoeft22": float(p.get("Fcoeft22", p.get("fcoeft22", p.get("fscale22", 1.0)))),
        "fscale22": float(p.get("Fcoeft22", p.get("fcoeft22", p.get("fscale22", 1.0)))),
        "ecoat": float(p.get("ECOAT", p.get("ecoat", 0.0))),
        "nucoat": float(p.get("NUCOAT", p.get("nucoat", 0.0))),
        "tcoat": float(p.get("TCOAT", p.get("tcoat", 0.0))),
        "c_rate": float(p.get("C_RATE", p.get("c_rate", 0.0))),
        "eps0_rate": float(p.get("EPS0", p.get("eps0", p.get("eps0_rate", 1.0)))),
    }
    return Law119Seatbelt(id=rec.id, rho0=rec.density, title=rec.title, params=params)


def _register() -> None:
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW119"] = build_law119
    MAT_PHYSICS_REGISTRY["SH_SEATBELT"] = build_law119
    MAT_PHYSICS_REGISTRY["MAT_LAW119"] = build_law119
    MAT_PHYSICS_REGISTRY["MAT_SH_SEATBELT"] = build_law119


_register()
