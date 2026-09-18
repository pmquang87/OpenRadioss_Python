"""
LAW114 — Seatbelt spring material model (/MAT/LAW114, /MAT/SPR_SEATBELT).

Fortran origin:
  - ``starter/source/materials/mat/mat114/hm_read_mat114.F`` (1D spring seatbelt reader)
  - ``engine/source/elements/spring/r23law114.F`` (spring element dispatcher)
  - ``engine/source/elements/spring/r23l114def3.F`` (kinematics & deformation)
  - ``engine/source/tools/seatbelts/redef_seatbelt.F90`` (constitutive law & hysteresis)

Theory:
  - Engineering strain: eps = delta / L_ref, where delta = L - L0, L_ref = max(L0, LMIN).
  - Tension vs compression:
      * Tension: evaluated from loading curve FUN_L (with force scale Fcoeft1, displacement
        scale Xcoeft1) and optional rate sensitivity factor (1 + c_rate * ln(max(1, |eps_dot|/eps0))).
      * Unloading: uses unloading curve FUN_UL with hysteresis tracking (yield_f, eps_max, dpx).
        Force clamped to non-negative (F >= 0).
      * Compression / slack: if YOUNG > 0, compressive stiffness K_comp = YOUNG * Area (or YOUNG);
        if YOUNG == 0, tension-only with zero stiffness and zero force (slack accommodates).
  - Damping: F_damp = DAMP1 * v_rel (tension only).
  - Energy: dE = 0.5 * (F_old + F) * d_delta integrated into internal energy.
"""

from __future__ import annotations

import math
from typing import Any, Callable

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


class Law114Seatbelt(Material):
    """LAW114 (/MAT/SPR_SEATBELT) 1D seatbelt spring material."""

    def __init__(self, id: int, rho0: float = 0.0, title: str = "", params: dict[str, Any] | None = None):
        p = params or {}
        super().__init__(id=id, law=114, rho0=rho0, title=title, params=p)
        self.k = float(p.get("STIFF1", p.get("stiff1", p.get("k", 0.0))))
        self.c = float(p.get("DAMP1", p.get("damp1", p.get("c", 0.0))))
        self.lmin = float(p.get("LMIN", p.get("lmin", 0.0)))
        self.fun_l = p.get("FUN_L", p.get("fun_l", 0))
        self.fun_ul = p.get("FUN_UL", p.get("fun_ul", 0))
        self.xscale = float(p.get("Xcoeft1", p.get("xscale", p.get("xcoeft1", 1.0))))
        self.fscale = float(p.get("Fcoeft1", p.get("fscale", p.get("fcoeft1", 1.0))))
        self.young = float(p.get("YOUNG", p.get("young", p.get("E", p.get("e", 0.0)))))
        self.fmax = float(p.get("FMAX", p.get("fmax", 0.0)))
        self.mmax = float(p.get("MMAX", p.get("mmax", 0.0)))
        self.c_rate = float(p.get("C_RATE", p.get("c_rate", 0.0)))
        self.eps0_rate = float(p.get("EPS0", p.get("eps0", p.get("eps0_rate", 1.0))))
        self.shear_area = float(p.get("SHEAR_AREA", p.get("shear_area", p.get("as", p.get("as_", 0.0)))))
        self.rfac = float(p.get("Rfac", p.get("rfac", p.get("r", 1.0))))
        self.ibend = float(p.get("Ibend", p.get("ibend", p.get("i", 0.0))))
        self.itors = float(p.get("Itors", p.get("itors", p.get("j", 0.0))))

    def compute_force(
        self,
        delta_L: float,
        v_rel: float = 0.0,
        L0: float = 1.0,
        state: dict[str, Any] | None = None,
        dt: float = 0.0,
        functions: dict[int, Any] | None = None,
    ) -> tuple[float, dict[str, Any]]:
        """Compute axial spring force F and update state."""
        F, _, updated_state = self.spring_update(
            L=L0 + delta_L,
            L0=L0,
            v_rel=v_rel,
            state=state,
            dt=dt,
            functions=functions,
        )
        return F, updated_state

    def spring_update(
        self,
        L: float,
        L0: float,
        v_rel: float = 0.0,
        state: dict[str, Any] | None = None,
        dt: float = 0.0,
        functions: dict[int, Any] | None = None,
    ) -> tuple[float, float, dict[str, Any]]:
        """
        Compute spring force, tangent stiffness, and update internal state.

        Parameters
        ----------
        L : float
            Current length of spring element.
        L0 : float
            Initial / unstrained length.
        v_rel : float
            Relative velocity (dL/dt).
        state : dict, optional
            State dictionary tracking yield_f, eps_max, dpx, force_old, eint, etc.
        dt : float
            Time increment.
        functions : dict, optional
            Model functions mapping func_id -> FunctTable / callable.

        Returns
        -------
        force : float
            Axial force (positive = tension).
        k_tan : float
            Tangent stiffness dF/dL.
        state : dict
            Updated internal state.
        """
        if state is None:
            state = {
                "yield_f": 0.0,
                "eps_max": 0.0,
                "dpx": 0.0,
                "force_old": 0.0,
                "eint": 0.0,
                "eps_old": 0.0,
                "delta_old": 0.0,
            }

        l_ref = max(L0, self.lmin) if self.lmin > 0.0 else L0
        if l_ref <= 0.0:
            l_ref = 1.0

        delta = L - L0
        eps = delta / l_ref

        # Retrieve curves
        f_load = None
        f_unload = None
        if functions is not None:
            if self.fun_l in functions:
                f_load = functions[self.fun_l]
            if self.fun_ul in functions:
                f_unload = functions[self.fun_ul]
        if f_load is None and (hasattr(self.fun_l, "eval") or callable(self.fun_l)):
            f_load = self.fun_l
        if f_unload is None and (hasattr(self.fun_ul, "eval") or callable(self.fun_ul)):
            f_unload = self.fun_ul
        if f_unload is None and f_load is not None:
            f_unload = f_load

        # Strain rate sensitivity factor
        rate_fac = 1.0
        if abs(dt) > 0.0 and self.c_rate > 0.0 and self.eps0_rate > 0.0:
            eps_dot = abs(v_rel) / l_ref
            rate_fac = 1.0 + self.c_rate * math.log(max(1.0, eps_dot / self.eps0_rate))

        dpx = state.get("dpx", 0.0)
        yield_f = state.get("yield_f", 0.0)
        f_old = state.get("force_old", 0.0)
        eps_ela = eps - dpx

        # Check tension vs compression
        # Elastic strain beyond permanent/slack offset
        is_tension = (delta > 0.0 and eps_ela > 0.0)

        if not is_tension:
            # Compression / Slack branch
            if self.young > 0.0:
                area = self.shear_area if self.shear_area > 0.0 else 1.0
                k_comp = self.young * area / l_ref
                F = k_comp * delta
                if self.fmax > 0.0:
                    F = max(-self.fmax, F)
                k_tan = k_comp
            else:
                # Tension-only: zero compressive stiffness and zero force
                F = 0.0
                k_tan = 0.0
                if dpx <= 0.0:
                    dpx = 0.0
        else:
            # Tension branch
            if f_load is not None:
                # Piecewise / tabulated loading curve
                x_eval = self.xscale * max(0.0, eps)
                f_curve = self.fscale * _eval_funct(f_load, x_eval) * rate_fac
                k_load = (self.fscale * self.xscale * _eval_funct_deriv(f_load, x_eval) * rate_fac) / l_ref

                # Unloading tangent from FUN_UL (or fallback to FUN_L)
                x_unl = self.xscale * max(0.0, eps)
                k_unl_deriv = _eval_funct_deriv(f_unload, x_unl)
                if k_unl_deriv <= 0.0:
                    k_unl_deriv = _eval_funct_deriv(f_load, x_unl)
                if k_unl_deriv <= 0.0:
                    k_unl_deriv = self.k if self.k > 0.0 else 1.0
                k_unl = (self.fscale * self.xscale * k_unl_deriv * rate_fac) / l_ref
                if k_unl <= 0.0:
                    k_unl = max(self.k, 1.0) / l_ref

                eps_max = state.get("eps_max", 0.0)
                if f_curve >= yield_f and eps >= eps_max:
                    # Primary loading branch
                    F = f_curve
                    yield_f = F
                    state["eps_max"] = eps
                    k_tan = k_load
                    # Establish permanent plastic offset upon unload
                    dpx = max(0.0, eps - F / (k_unl * l_ref))
                else:
                    # Unloading / reloading branch
                    deps = eps - state.get("eps_old", eps)
                    F = f_old + k_unl * (deps * l_ref)
                    F = max(0.0, min(yield_f, F))
                    k_tan = k_unl if F > 0.0 else 0.0

                if self.fmax > 0.0:
                    F = min(self.fmax, F)
            else:
                # Linear elastic in tension
                k_eff = self.k if self.k > 0.0 else 1.0
                F = k_eff * max(0.0, delta) * rate_fac
                k_tan = k_eff * rate_fac
                if self.fmax > 0.0:
                    F = min(self.fmax, F)

            # Viscous damping in tension
            if self.c > 0.0 and F > 0.0:
                f_damp = self.c * v_rel
                F = max(0.0, F + f_damp)

        # Internal energy accounting: dE = 0.5 * (F_old + F) * d_delta
        delta_old = state.get("delta_old", delta)
        d_delta = delta - delta_old
        dE = 0.5 * (f_old + F) * d_delta
        state["eint"] = state.get("eint", 0.0) + dE

        # Save committed state
        state["yield_f"] = yield_f
        state["dpx"] = dpx
        state["force_old"] = F
        state["eps_old"] = eps
        state["delta_old"] = delta

        return F, k_tan, state


def build_law114(rec) -> Law114Seatbelt:
    """Constructor for /MAT/LAW114 (/MAT/SPR_SEATBELT) 1D seatbelt spring material."""
    p = rec.params
    stiff1 = float(p.get("STIFF1", p.get("stiff1", p.get("k", 0.0))))
    damp1 = float(p.get("DAMP1", p.get("damp1", p.get("c", 0.0))))
    lmin = float(p.get("LMIN", p.get("lmin", 0.0)))
    fun_l = p.get("FUN_L", p.get("fun_l", 0))
    fun_ul = p.get("FUN_UL", p.get("fun_ul", 0))
    xscale = float(p.get("Xcoeft1", p.get("xscale", p.get("xcoeft1", 1.0))))
    fscale = float(p.get("Fcoeft1", p.get("fscale", p.get("fcoeft1", 1.0))))
    young = float(p.get("YOUNG", p.get("young", p.get("E", p.get("e", 0.0)))))
    e = young if young > 0.0 else (stiff1 if stiff1 > 0.0 else 1.0)
    nu = 0.3

    params = {
        "E": e,
        "nu": nu,
        "stiff1": stiff1,
        "k": stiff1,
        "damp1": damp1,
        "c": damp1,
        "lmin": lmin,
        "fun_l": fun_l,
        "fun_ul": fun_ul,
        "xscale": xscale,
        "fscale": fscale,
        "young": young,
        "ibend": float(p.get("Ibend", p.get("ibend", p.get("i", 0.0)))),
        "itors": float(p.get("Itors", p.get("itors", p.get("j", 0.0)))),
        "shear_area": float(p.get("SHEAR_AREA", p.get("shear_area", p.get("as", p.get("as_", 0.0))))),
        "fmax": float(p.get("FMAX", p.get("fmax", 0.0))),
        "mmax": float(p.get("MMAX", p.get("mmax", 0.0))),
        "rfac": float(p.get("Rfac", p.get("rfac", p.get("r", 1.0)))),
        "c_rate": float(p.get("C_RATE", p.get("c_rate", 0.0))),
        "eps0_rate": float(p.get("EPS0", p.get("eps0", p.get("eps0_rate", 1.0)))),
    }
    return Law114Seatbelt(id=rec.id, rho0=rec.density, title=rec.title, params=params)


def build_law119(rec):
    """Delegate to law119_seatbelt."""
    from . import law119_seatbelt
    return law119_seatbelt.build_law119(rec)


def _register() -> None:
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["LAW114"] = build_law114
    MAT_PHYSICS_REGISTRY["SPR_SEATBELT"] = build_law114
    MAT_PHYSICS_REGISTRY["MAT_LAW114"] = build_law114
    MAT_PHYSICS_REGISTRY["MAT_SPR_SEATBELT"] = build_law114


_register()
