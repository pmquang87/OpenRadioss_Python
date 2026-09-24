"""
OpenRadioss /PROP/TYPE23 (/PROP/SPR_MAT: Spring with Material Laws).

Element driver and material dispatch for non-linear springs coupled to material laws:
  - /MAT/LAW108 (/MAT/SPR_GENE): 6-DOF general spring with tabulated yield curves,
    unloading hysteresis, damping, rate effects, and skew alignment.
  - /MAT/LAW113 (/MAT/SPR_BEAM): 6-DOF coupled beam/spring connector with co-rotational
    beam axis kinematics and transverse shear moment arm equilibrium.
  - /MAT/LAW114 (/MAT/SPR_SEATBELT): 1D seatbelt tension spring with loading/unloading
    curves, slack/compression handling, damping, and rate sensitivity.

Upstream Fortran reference:
  - engine/source/elements/spring/r23forc3.F (element driver)
  - engine/source/elements/spring/r23law108.F & r23l108def3.F (LAW108 kinematics & dispatch)
  - engine/source/elements/spring/r23law113.F & r23l113def3.F (LAW113 beam axes & dispatch)
  - engine/source/elements/spring/r23law114.F & r23l114def3.F (LAW114 seatbelt driver)
  - engine/source/elements/spring/redef3.F90 (6-DOF general nonlinear constitutive law)
  - engine/source/tools/seatbelts/redef_seatbelt.F90 (seatbelt constitutive law & hysteresis)
  - engine/source/elements/spring/r4cum3.F (transverse shear moment arm equilibrium)
  - engine/source/elements/spring/r2cum3.F (rotational equilibrium)
  - engine/source/elements/spring/r2len3.F (critical time step calculation)
  - starter/source/properties/spring/hm_read_prop23.F (property reader)
  - starter/source/elements/spring/rinit3.F (spring initialization)
  - starter/source/elements/spring/rmass.F (nodal mass & inertia distribution)
  - hm_cfg_files/config/CFG/radioss2020/PROP/prop_p23_SPR_MAT.cfg
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3
from ..materials.law108_yield_fit import Law108Params, build_law108
from ..materials.law113_yield_curve_fit import Law113Params, build_law113
from ..materials.law114_seatbelt import Law114Seatbelt


def _eval_func(func: Any, x: float) -> float:
    """Evaluate curve function at scalar abscissa x."""
    if func is None:
        return 0.0
    if hasattr(func, "eval"):
        return float(func.eval(x))
    if hasattr(func, "evaluate"):
        return float(func.evaluate(x))
    if callable(func):
        return float(func(x))
    if isinstance(func, (int, float)):
        return float(func)
    return 0.0


def _eval_deriv(func: Any, x: float) -> float:
    """Evaluate derivative dy/dx of curve function at scalar abscissa x."""
    if func is None:
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


def _get_skew_frame(prop: Any, skews: Any, xe: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retrieve 3 local orthogonal axes (e1, e2, e3) for LAW108.

    Fortran origin: r23l108def3.F lines 198-207.
    Uses SKEW(1..9, ISK) if skew_id > 0, otherwise global axes [1,0,0], [0,1,0], [0,0,1].
    """
    p = getattr(prop, "params", {}) or {}
    skew_id = int(getattr(prop, "skew_id", p.get("skew_id", 0)) or 0)
    skew_row = int(p.get("skew_row", 0) or 0)

    if skew_row and skews is not None and hasattr(skews, "axes"):
        if 0 <= skew_row < len(skews.axes):
            a = skews.axes[skew_row]
            return a[0].copy(), a[1].copy(), a[2].copy()

    if skew_id > 0 and skews is not None:
        if hasattr(skews, "get_frame"):
            frame = skews.get_frame(skew_id)
            if frame is not None:
                return frame[0].copy(), frame[1].copy(), frame[2].copy()
        if hasattr(skews, "axes") and 0 <= skew_id < len(skews.axes):
            a = skews.axes[skew_id]
            return a[0].copy(), a[1].copy(), a[2].copy()

    # Global Cartesian frame fallback (Fortran skew 0)
    return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])


def _compute_beam_corotational_frame(d: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute co-rotational beam local frame (e1, e2, e3) for LAW113.

    Fortran origin: r4evec3.F & r23law113.F lines 202-213.
    e1 is aligned with the element axis (x2 - x1) / L.
    e2 and e3 are perpendicular axes constructed via Gram-Schmidt.
    """
    if L > EM20:
        e1 = d / L
    else:
        e1 = np.array([1.0, 0.0, 0.0])

    # Reference vector least aligned with e1
    ax = np.array([0.0, 0.0, 1.0])
    if abs(e1[2]) > 0.9:
        ax = np.array([1.0, 0.0, 0.0])

    e2 = np.cross(ax, e1)
    norm_e2 = float(np.linalg.norm(e2))
    if norm_e2 > EM20:
        e2 /= norm_e2
    else:
        e2 = np.array([0.0, 1.0, 0.0])

    e3 = np.cross(e1, e2)
    norm_e3 = float(np.linalg.norm(e3))
    if norm_e3 > EM20:
        e3 /= norm_e3
    else:
        e3 = np.array([0.0, 0.0, 1.0])

    return e1, e2, e3


def _update_dof_redef3(
    dof_idx: int,
    delta: float,
    delta_old: float,
    v_rel: float,
    dof_params: Any,
    state: Dict[str, Any],
    dt: float,
    functions: Optional[Dict[int, Any]],
) -> Tuple[float, float, float]:
    """Single-DOF constitutive update matching OpenRadioss redef3.F90.

    Fortran origin: engine/source/elements/spring/redef3.F90 lines 405-442, 736-840, 1140-1151.

    Parameters:
        dof_idx: 0..5 (0..2 translations, 3..5 rotations)
        delta: current relative displacement or rotation
        delta_old: committed relative displacement or rotation from previous step
        v_rel: relative translational or rotational velocity
        dof_params: SpringDOFParams or BeamSpringDOF
        state: element history dictionary
        dt: current time increment
        functions: model function dictionary

    Returns:
        F: computed force or moment
        k_eff: effective stiffness for explicit time step
        c_eff: effective damping for explicit time step
    """
    stiff = float(getattr(dof_params, "stiff", 1.0))
    damp = float(getattr(dof_params, "damp", 0.0))
    acoeft = float(getattr(dof_params, "acoeft", 1.0))
    bcoeft = float(getattr(dof_params, "bcoeft", 0.0))
    dcoeft = float(getattr(dof_params, "dcoeft", 1.0))
    hflag = int(getattr(dof_params, "hflag", 1))
    fun_a = int(getattr(dof_params, "fun_a", 0))
    fun_b = int(getattr(dof_params, "fun_b", 0))
    scale = float(getattr(dof_params, "scale", 1.0))
    prop_f = float(getattr(dof_params, "prop_f", 1.0))
    prop_e = float(getattr(dof_params, "prop_e", 1.0))
    f_x0 = float(getattr(dof_params, "f_x0", 0.0))

    curve_a = None
    if functions and fun_a > 0 and fun_a in functions:
        curve_a = functions[fun_a]
    elif hasattr(dof_params, "curve_a") and dof_params.curve_a is not None:
        curve_a = dof_params.curve_a

    curve_b = None
    if functions and fun_b > 0 and fun_b in functions:
        curve_b = functions[fun_b]
    elif hasattr(dof_params, "curve_b") and dof_params.curve_b is not None:
        curve_b = dof_params.curve_b

    f_old = float(state.get(f"force_old_{dof_idx}", 0.0))
    dpx = float(state.get(f"dpx_{dof_idx}", 0.0))
    dpx2 = float(state.get(f"dpx2_{dof_idx}", 0.0))

    d_delta = delta - delta_old
    k_eff = max(stiff, 0.0)
    c_eff = max(damp, 0.0)

    # 1. Trial static force evaluation
    if curve_a is None or fun_a == 0:
        # Linear elastic (redef3.F90:736-742)
        f_trial = stiff * delta
    else:
        x_scale = scale * prop_e
        f_scale = scale * prop_f
        x_eval = delta * x_scale
        f_curve = f_scale * _eval_func(curve_a, x_eval)

        if hflag in (0, 8):
            # Nonlinear elastic (redef3.F90:746-758)
            f_trial = f_curve
            k_tan = abs(f_scale * x_scale * _eval_deriv(curve_a, x_eval))
            if k_tan > 0.0:
                k_eff = max(k_eff, k_tan)

        elif hflag in (1, 3):
            # Elasto-plastic with isotropic hardening (redef3.F90:762-775)
            f_trial_step = f_old + stiff * d_delta
            denom_k = max(stiff, EM20)
            u_eq = dpx + abs(f_trial_step) / denom_k
            f_yield = abs(f_scale * _eval_func(curve_a, u_eq * x_scale))

            if f_trial_step >= 0.0 and f_trial_step > f_yield:
                dpx += (f_trial_step - f_yield) / denom_k
                f_trial = f_yield
            elif f_trial_step < 0.0 and f_trial_step < -f_yield:
                dpx += (-f_yield - f_trial_step) / denom_k
                f_trial = -f_yield
            else:
                f_trial = f_trial_step
            state[f"dpx_{dof_idx}"] = dpx

        elif hflag == 2:
            # Decoupled tension/compression (redef3.F90:779-797)
            denom_k = max(stiff, EM20)
            if delta > dpx:
                f_trial_step = stiff * (delta - dpx)
                f_max = f_curve
                f_trial = min(f_trial_step, f_max)
                dpx = delta - f_trial / denom_k
            elif delta < dpx2:
                f_trial_step = stiff * (delta - dpx2)
                f_min = f_curve
                f_trial = max(f_trial_step, f_min)
                dpx2 = delta - f_trial / denom_k
            else:
                f_trial = 0.0
            state[f"dpx_{dof_idx}"] = dpx
            state[f"dpx2_{dof_idx}"] = dpx2

        elif hflag == 9:
            # Tabulated yield curve fit with F(0) offset
            f_trial = f_curve + f_x0
        else:
            f_trial = f_curve

    # 2. Rate sensitivity (redef3.F90:1141-1143)
    d_rate = dcoeft if dcoeft > 0.0 else 1.0
    dvv = max(1.0, abs(v_rel) / d_rate)
    dfac = acoeft + bcoeft * math.log(dvv)
    if curve_b is not None:
        gx = _eval_func(curve_b, abs(v_rel))
        dfac += prop_e * gx

    # 3. Total force with damping
    total_f = dfac * f_trial + damp * v_rel

    # 4. Check failure / rupture limits (r23l108def3.F:519-544)
    min_rup = float(getattr(dof_params, "min_rup", -1.0e30))
    max_rup = float(getattr(dof_params, "max_rup", 1.0e30))
    if delta < min_rup or delta > max_rup:
        state["failed"] = True
        total_f = 0.0

    state[f"force_old_{dof_idx}"] = total_f
    return total_f, k_eff, c_eff


def init_spring_mat_type23(
    group: Any,
    model: Any,
    log: Any,
    idx23: Optional[np.ndarray] = None,
    massn: Optional[np.ndarray] = None,
    inertn: Optional[np.ndarray] = None,
) -> None:
    """Initialize /PROP/TYPE23 (/PROP/SPR_MAT) element buffer, masses, inertias, and local frames.

    Fortran origin:
      - starter/source/properties/spring/hm_read_prop23.F
      - starter/source/elements/spring/rinit3.F lines 440-507
      - starter/source/elements/spring/rmass.F lines 156-185

    Mass calculation:
      - Imass == 1: mass = Area * L0 * rho (L0 = max(L0, Lmin) for LAW114)
      - Imass == 2 (default): mass = Volume * rho
    Inertia calculation:
      - LAW114: inertia = max(EM20, rfac * max((rho * Area * L_ref^3)/12 + rho*Iyy*L_ref, rho*Ixx*L_ref))
      - LAW108 / LAW113: inertia from /PROP/TYPE23 card INERTIA (GEO(2))
    Distributed half/half to node 1 and node 2.
    """
    st = group.state
    n = group.n
    if n == 0 or len(group.conn) == 0:
        return

    if idx23 is None:
        idx23 = st.get("idx23", np.empty(0, dtype=np.int64))
    if len(idx23) == 0:
        return

    if "mass" not in st:
        st["mass"] = np.zeros(n)
    if "inertia" not in st:
        st["inertia"] = np.zeros(n)
    if "L0" not in st:
        st["L0"] = np.zeros(n)
    if "force" not in st:
        st["force"] = np.zeros(n)
    if "eint" not in st:
        st["eint"] = np.zeros(n)
    if "off" not in st:
        st["off"] = np.ones(n, dtype=float)

    mat_data = st.get("spring_mat_data")
    if mat_data is None:
        mat_data = {}
        st["spring_mat_data"] = mat_data

    slices = st.get("slices", [])
    stride = group.conn.shape[1] if group.conn.ndim == 2 else 2
    skews = getattr(model, "skews", None)

    for e in idx23:
        n1 = group.conn[e, 0]
        n2 = group.conn[e, 1]
        x1_0 = model.x0[n1]
        x2_0 = model.x0[n2]
        d0 = x2_0 - x1_0
        L0 = float(norm3(d0.reshape(1, 3))[0])

        # Identify property and material from slice
        mat = None
        prop = None
        for sl, s_mat, s_prop in slices:
            if isinstance(sl, slice):
                start = sl.start or 0
                stop = sl.stop or n
                if start <= e < stop:
                    mat = s_mat
                    prop = s_prop
                    break
            elif isinstance(sl, np.ndarray) and len(sl) > 0:
                if e in sl:
                    mat = s_mat
                    prop = s_prop
                    break

        if prop is None:
            # Fallback to model property
            pid = group.part[e] if hasattr(group, "part") and len(group.part) > e else 1
            prop = getattr(model, "properties", {}).get(pid)
        if mat is None:
            mid = getattr(prop, "mat_id", 1) if prop is not None else 1
            mat = getattr(model, "materials", {}).get(mid)

        p = getattr(prop, "params", {}) or {}
        imass = int(getattr(prop, "imass", p.get("imass", 2)))
        area_or_vol = float(getattr(prop, "area_or_volume", p.get("area_or_volume", p.get("mass", 0.0))))
        prop_inertia = float(getattr(prop, "inertia", p.get("inertia", 0.0)))
        skew_id = int(getattr(prop, "skew_id", p.get("skew_id", 0)))
        sens_id = int(getattr(prop, "isens", p.get("isens", p.get("sensor_id", p.get("sens_id", 0)))))
        isflag = int(getattr(prop, "isflag", p.get("isflag", p.get("iflag", 0))))

        # Determine material law type
        law_type = 108
        if mat is not None:
            law_val = getattr(mat, "law", getattr(mat, "type", 0))
            if law_val in (108, 113, 114):
                law_type = law_val
            else:
                mname = type(mat).__name__
                if "114" in mname or "Seatbelt" in mname:
                    law_type = 114
                elif "113" in mname or "Beam" in mname or "SprBeam" in mname:
                    law_type = 113
                elif "108" in mname or "Gene" in mname or "SprGene" in mname:
                    law_type = 108
                elif "law" in getattr(mat, "params", {}):
                    law_type = int(mat.params["law"])

        rho = float(getattr(mat, "rho0", getattr(mat, "rho", 0.0)))
        if rho == 0.0 and hasattr(mat, "params"):
            rho = float(mat.params.get("rho0", mat.params.get("rho", 0.0)))

        # -------------------------------------------------------------
        # Mass and Inertia per rinit3.F:473-502 & rmass.F:156-185
        # -------------------------------------------------------------
        if law_type == 114:
            # LAW114 seatbelt
            lmin = float(getattr(mat, "lmin", mat.params.get("LMIN", mat.params.get("lmin", 0.0)))) if mat else 0.0
            lref = max(L0, lmin) if lmin > 0.0 else (L0 if L0 > 0.0 else 1.0)
            area = float(p.get("area", area_or_vol)) if imass == 1 else area_or_vol
            mass = area * lref * rho if imass == 1 else area_or_vol * rho

            young = float(getattr(mat, "young", mat.params.get("YOUNG", mat.params.get("young", mat.params.get("E", 0.0))))) if mat else 0.0
            if young > 0.0 and mat is not None:
                rfac = float(getattr(mat, "rfac", mat.params.get("Rfac", mat.params.get("rfac", 1.0))))
                ixx = float(mat.params.get("Ixx", mat.params.get("ixx", mat.params.get("IXX", 0.0))))
                iyy = float(mat.params.get("Iyy", mat.params.get("iyy", mat.params.get("IYY", 0.0))))
                i_calc = rfac * max((rho * area * (lref ** 3)) / 12.0 + rho * iyy * lref, rho * ixx * lref)
                inertia = max(EM20, i_calc)
            else:
                inertia = prop_inertia
        else:
            # LAW108 or LAW113
            lref = L0 if L0 > 0.0 else 1.0
            if imass == 1:
                area = float(p.get("area", area_or_vol))
                mass = area * L0 * rho
            else:
                vol = float(p.get("volume", area_or_vol))
                mass = vol * rho
            inertia = prop_inertia

        st["mass"][e] = mass
        st["inertia"][e] = inertia
        st["L0"][e] = L0

        if massn is not None and e * stride + 1 < len(massn):
            massn[stride * e] += mass / 2.0
            massn[stride * e + 1] += mass / 2.0
        if inertn is not None and e * stride + 1 < len(inertn):
            inertn[stride * e] += inertia / 2.0
            inertn[stride * e + 1] += inertia / 2.0

        # Construct parameter wrappers and reference geometry
        if law_type == 108:
            law_params = build_law108(mat)
            e1, e2, e3 = _get_skew_frame(prop, skews, np.array([x1_0, x2_0]))
            # Initial projection (r23l108def3.F:245-253)
            X0 = float(np.dot(d0, e1))
            Y0 = float(np.dot(d0, e2))
            Z0 = float(np.dot(d0, e3))
            elem_state = {
                "law_type": 108,
                "params": law_params,
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "X0": X0,
                "Y0": Y0,
                "Z0": Z0,
                "delta_old": np.zeros(6, dtype=np.float64),
                "theta": np.zeros(3, dtype=np.float64),
                "failed": False,
            }
        elif law_type == 113:
            law_params = build_law113(mat)
            e1, e2, e3 = _compute_beam_corotational_frame(d0, L0)
            elem_state = {
                "law_type": 113,
                "params": law_params,
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "L0": L0,
                "delta_old": np.zeros(6, dtype=np.float64),
                "theta": np.zeros(3, dtype=np.float64),
                "failed": False,
            }
        else:
            # law_type == 114
            if isinstance(mat, Law114Seatbelt):
                sb_mat = mat
            else:
                sb_mat = Law114Seatbelt(
                    id=getattr(mat, "id", 1),
                    rho0=rho,
                    params=getattr(mat, "params", {}) if mat else {},
                )
            elem_state = {
                "law_type": 114,
                "mat": sb_mat,
                "L0": L0,
                "Lref": lref,
                "L_old": L0,
                "delta_old": 0.0,
                "eps_old": 0.0,
                "yield_f": 0.0,
                "eps_max": 0.0,
                "dpx": 0.0,
                "force_old": 0.0,
                "eint": 0.0,
                "failed": False,
            }

        elem_state["sens_id"] = sens_id
        elem_state["isflag"] = isflag
        elem_state["iequil"] = int(getattr(prop, "iequil", p.get("iequil", 0)))
        mat_data[e] = elem_state


def forces_spring_mat_type23(
    group: Any,
    x: np.ndarray,
    v: Optional[np.ndarray],
    vr: Optional[np.ndarray],
    dt: float,
    fint: Optional[np.ndarray],
    mint: Optional[np.ndarray],
    idx23: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute /PROP/TYPE23 internal forces, moments, energy accounting, and critical time step.

    Fortran origin:
      - engine/source/elements/spring/r23forc3.F (element driver)
      - engine/source/elements/spring/r23law108.F & r23l108def3.F (LAW108)
      - engine/source/elements/spring/r23law113.F & r23l113def3.F (LAW113)
      - engine/source/elements/spring/r23law114.F & r23l114def3.F (LAW114)
      - engine/source/elements/spring/r4cum3.F (transverse shear moment arm equilibrium)
      - engine/source/elements/spring/r2cum3.F (rotational equilibrium)
      - engine/source/elements/spring/r2len3.F (explicit critical time step)

    Parameters:
        group: ElementGroup containing the springs
        x: current nodal coordinates (n_nodes, 3)
        v: current nodal translational velocities (n_nodes, 3)
        vr: current nodal rotational velocities (n_nodes, 3)
        dt: current time step increment
        fint: global internal translational force accumulator (n_nodes, 3)
        mint: global internal rotational moment accumulator (n_nodes, 3)
        idx23: indices of elements in group to process

    Returns:
        dtc: 1D array of critical explicit time steps for the elements in idx23
    """
    st = group.state
    if idx23 is None:
        idx23 = st.get("idx23", np.empty(0, dtype=np.int64))
    if len(idx23) == 0:
        return np.empty(0, dtype=np.float64)

    dt_val = float(dt) if (dt is not None and dt > 0.0) else 0.0
    model = st.get("model")
    sensors = getattr(model, "sensors_state", None) if model is not None else None
    funcs = getattr(model, "functions", {}) if model is not None else {}
    mat_data = st.get("spring_mat_data", {})
    off = st.get("off", np.ones(group.n, dtype=float))

    dtc = np.full(len(idx23), EP30, dtype=np.float64)

    for i_local, e in enumerate(idx23):
        elem_state = mat_data.get(e)
        if elem_state is None:
            continue

        # Sensor activation / deactivation check (r23sens3.F)
        sens_id = elem_state.get("sens_id", 0)
        isflag = elem_state.get("isflag", 0)
        is_active = True
        if sensors is not None and sens_id != 0:
            s_act = sensors.active(abs(sens_id))
            if isflag == 0:
                is_active = s_act
            elif isflag == 1:
                is_active = not s_act
            elif isflag == 2:
                is_active = s_act

        if not is_active or off[e] <= 0.0 or elem_state.get("failed", False):
            off[e] = 0.0
            st["force"][e] = 0.0
            dtc[i_local] = EP30
            continue

        n1 = group.conn[e, 0]
        n2 = group.conn[e, 1]
        x1 = x[n1]
        x2 = x[n2]
        dx = x2 - x1
        L = float(norm3(dx.reshape(1, 3))[0])

        v1 = v[n1] if v is not None else np.zeros(3)
        v2 = v[n2] if v is not None else np.zeros(3)
        dv = v2 - v1

        w1 = vr[n1] if vr is not None else np.zeros(3)
        w2 = vr[n2] if vr is not None else np.zeros(3)
        dw = w2 - w1

        law_type = elem_state["law_type"]

        # =====================================================================
        # LAW 114: Seatbelt 1D Spring
        # =====================================================================
        if law_type == 114:
            sb_mat = elem_state["mat"]
            L0 = elem_state["L0"]
            e1 = dx / max(L, EM20) if L > EM20 else np.array([1.0, 0.0, 0.0])
            v_rel = float(np.dot(dv, e1))

            F, k_tan, updated_state = sb_mat.spring_update(
                L=L,
                L0=L0,
                v_rel=v_rel,
                state=elem_state,
                dt=dt_val,
                functions=funcs,
            )
            elem_state.update(updated_state)
            st["eint"][e] = float(elem_state.get("eint", 0.0))
            st["force"][e] = F

            # Scatter axial force to nodes
            fvec = F * e1
            if fint is not None:
                np.add.at(fint, n1, fvec)
                np.add.at(fint, n2, -fvec)

            # Explicit critical time step (r2len3.F:179-187)
            elem_mass = max(st["mass"][e], 0.0)
            k_eff = max(k_tan, sb_mat.k, EM20)
            c_eff = max(sb_mat.c, 0.0)
            denom = math.sqrt(c_eff * c_eff + elem_mass * k_eff) + c_eff
            dt_tr = elem_mass / max(denom, EM20) if elem_mass > 0.0 else EP30
            dtc[i_local] = dt_tr

        # =====================================================================
        # LAW 108: General 6-DOF Spring / Skew Alignment
        # =====================================================================
        elif law_type == 108:
            p108 = elem_state["params"]
            e1 = elem_state["e1"]
            e2 = elem_state["e2"]
            e3 = elem_state["e3"]

            # Skew-aligned translational deformations (r23l108def3.F:338-341)
            u1 = float(np.dot(dx, e1)) - elem_state["X0"]
            u2 = float(np.dot(dx, e2)) - elem_state["Y0"]
            u3 = float(np.dot(dx, e3)) - elem_state["Z0"]

            v_rel_t = np.array([np.dot(dv, e1), np.dot(dv, e2), np.dot(dv, e3)], dtype=np.float64)

            # Skew-aligned rotational deformations (r23l108def3.F:582-585)
            dw_proj = np.array([np.dot(dw, e1), np.dot(dw, e2), np.dot(dw, e3)], dtype=np.float64)
            dtheta = dw_proj * dt_val
            elem_state["theta"] += dtheta
            theta = elem_state["theta"].copy()

            cur_delta = np.array([u1, u2, u3, theta[0], theta[1], theta[2]], dtype=np.float64)
            old_delta = elem_state["delta_old"]
            v_all = np.array([v_rel_t[0], v_rel_t[1], v_rel_t[2], dw_proj[0], dw_proj[1], dw_proj[2]], dtype=np.float64)

            forces_6 = np.zeros(6, dtype=np.float64)
            k_all = np.zeros(6, dtype=np.float64)
            c_all = np.zeros(6, dtype=np.float64)

            # Solve constitutive relations for 6 DOFs
            for d_idx in range(6):
                dof_p = p108.dofs[d_idx]
                f_i, k_i, c_i = _update_dof_redef3(
                    dof_idx=d_idx,
                    delta=cur_delta[d_idx],
                    delta_old=old_delta[d_idx],
                    v_rel=v_all[d_idx],
                    dof_params=dof_p,
                    state=elem_state,
                    dt=dt_val,
                    functions=funcs,
                )
                forces_6[d_idx] = f_i
                k_all[d_idx] = k_i
                c_all[d_idx] = c_i

            if elem_state.get("failed", False):
                off[e] = 0.0
                st["force"][e] = 0.0
                dtc[i_local] = EP30
                continue

            # Internal energy update: trapezoidal work sum (redef3.F90:1144)
            d_work = 0.0
            for d_idx in range(6):
                f_old = float(elem_state.get(f"force_old_{d_idx}", forces_6[d_idx]))
                d_work += 0.5 * (f_old + forces_6[d_idx]) * (cur_delta[d_idx] - old_delta[d_idx])
            st["eint"][e] += d_work
            elem_state["delta_old"] = cur_delta.copy()

            # Global force & moment vectors
            F_local = forces_6[:3]
            M_local = forces_6[3:]
            fvec = F_local[0] * e1 + F_local[1] * e2 + F_local[2] * e3
            mvec = M_local[0] * e1 + M_local[1] * e2 + M_local[2] * e3
            st["force"][e] = float(norm3(fvec.reshape(1, 3))[0])

            # Assemble to nodes (r2cum3.F:83-111, 125-140)
            if fint is not None:
                np.add.at(fint, n1, fvec)
                np.add.at(fint, n2, -fvec)

            if mint is not None:
                iequil = elem_state.get("iequil", 0)
                if iequil == 1:
                    # Moment arm correction for angular momentum conservation
                    arm = 0.5 * np.cross(dx, fvec)
                    mvec1 = mvec + arm
                    mvec2 = -mvec + arm
                    np.add.at(mint, n1, mvec1)
                    np.add.at(mint, n2, mvec2)
                else:
                    np.add.at(mint, n1, mvec)
                    np.add.at(mint, n2, -mvec)

            # Explicit critical time step (r2len3.F:179-187)
            elem_mass = max(st["mass"][e], 0.0)
            elem_inertia = max(st["inertia"][e], 0.0)
            kt_max = max(float(np.max(k_all[:3])), EM20)
            ct_max = max(float(np.max(c_all[:3])), 0.0)
            kr_max = max(float(np.max(k_all[3:])), EM20)
            cr_max = max(float(np.max(c_all[3:])), 0.0)

            denom_t = math.sqrt(ct_max * ct_max + elem_mass * kt_max) + ct_max
            dt_tr = elem_mass / max(denom_t, EM20) if elem_mass > 0.0 else EP30

            if elem_inertia > 0.0:
                denom_r = math.sqrt(cr_max * cr_max + elem_inertia * kr_max) + cr_max
                dt_rot = elem_inertia / max(denom_r, EM20)
            else:
                dt_rot = EP30

            dtc[i_local] = min(dt_tr, dt_rot)

        # =====================================================================
        # LAW 113: Co-rotational Beam/Spring with Shear Moment Arm
        # =====================================================================
        elif law_type == 113:
            p113 = elem_state["params"]
            e1, e2, e3 = _compute_beam_corotational_frame(dx, L)
            elem_state["e1"] = e1
            elem_state["e2"] = e2
            elem_state["e3"] = e3

            L0 = elem_state["L0"]
            u1 = L - L0  # Axial extension along co-rotational axis

            # Beam co-rotational transverse kinematics (r23l113def3.F:293-320)
            half_L = max(0.5 * L, EM20)
            dt05 = 0.5 * dt_val
            v21_e2 = float(np.dot(dv, e2))
            v21_e3 = float(np.dot(dv, e3))
            epxy = v21_e2 * dt05
            epxz = v21_e3 * dt05

            w_sum = w1 + w2
            ryav1 = float(np.dot(w_sum, e2))
            rzav1 = float(np.dot(w_sum, e3))

            at_z = math.atan(epxy / half_L)
            at_y = math.atan(epxz / half_L)

            ryav = dt05 * ryav1 + 2.0 * at_y
            rzav = dt05 * rzav1 - 2.0 * at_z

            dy_old = float(elem_state.get("DY", 0.0))
            dz_old = float(elem_state.get("DZ", 0.0))
            u2 = dy_old - rzav * half_L
            u3 = dz_old + ryav * half_L
            elem_state["DY"] = u2
            elem_state["DZ"] = u3

            v_rel_t = np.array([np.dot(dv, e1), np.dot(dv, e2), np.dot(dv, e3)], dtype=np.float64)
            dw_proj = np.array([np.dot(dw, e1), np.dot(dw, e2), np.dot(dw, e3)], dtype=np.float64)
            dtheta = dw_proj * dt_val
            elem_state["theta"] += dtheta
            theta = elem_state["theta"].copy()

            cur_delta = np.array([u1, u2, u3, theta[0], theta[1], theta[2]], dtype=np.float64)
            old_delta = elem_state["delta_old"]
            v_all = np.array([v_rel_t[0], v_rel_t[1], v_rel_t[2], dw_proj[0], dw_proj[1], dw_proj[2]], dtype=np.float64)

            forces_6 = np.zeros(6, dtype=np.float64)
            k_all = np.zeros(6, dtype=np.float64)
            c_all = np.zeros(6, dtype=np.float64)

            for d_idx in range(6):
                dof_p = p113.dofs[d_idx]
                f_i, k_i, c_i = _update_dof_redef3(
                    dof_idx=d_idx,
                    delta=cur_delta[d_idx],
                    delta_old=old_delta[d_idx],
                    v_rel=v_all[d_idx],
                    dof_params=dof_p,
                    state=elem_state,
                    dt=dt_val,
                    functions=funcs,
                )
                forces_6[d_idx] = f_i
                k_all[d_idx] = k_i
                c_all[d_idx] = c_i

            if elem_state.get("failed", False):
                off[e] = 0.0
                st["force"][e] = 0.0
                dtc[i_local] = EP30
                continue

            # Internal energy update
            d_work = 0.0
            for d_idx in range(6):
                f_old = float(elem_state.get(f"force_old_{d_idx}", forces_6[d_idx]))
                d_work += 0.5 * (f_old + forces_6[d_idx]) * (cur_delta[d_idx] - old_delta[d_idx])
            st["eint"][e] += d_work
            elem_state["delta_old"] = cur_delta.copy()

            # Local forces and moments
            FX = forces_6[0]
            FY = forces_6[1]
            FZ = forces_6[2]
            MX = forces_6[3]
            MY = forces_6[4]
            MZ = forces_6[5]

            # Transverse shear moment arm equilibrium (r4cum3.F:107-138)
            # YMOM1 = MY - 0.5 * L * FZ,  ZMOM1 = MZ + 0.5 * L * FY
            # YMOM2 = MY + 0.5 * L * FZ,  ZMOM2 = MZ - 0.5 * L * FY
            YMOM1 = MY - 0.5 * L * FZ
            ZMOM1 = MZ + 0.5 * L * FY
            YMOM2 = MY + 0.5 * L * FZ
            ZMOM2 = MZ - 0.5 * L * FY

            fvec = FX * e1 + FY * e2 + FZ * e3
            mvec1 = MX * e1 + YMOM1 * e2 + ZMOM1 * e3
            mvec2 = MX * e1 + YMOM2 * e2 + ZMOM2 * e3
            st["force"][e] = float(norm3(fvec.reshape(1, 3))[0])

            # Assembly: node 1 gets +fvec and +mvec1, node 2 gets -fvec and -mvec2
            if fint is not None:
                np.add.at(fint, n1, fvec)
                np.add.at(fint, n2, -fvec)
            if mint is not None:
                np.add.at(mint, n1, mvec1)
                np.add.at(mint, n2, -mvec2)

            # Explicit critical time step (r2len3.F:179-187)
            elem_mass = max(st["mass"][e], 0.0)
            elem_inertia = max(st["inertia"][e], 0.0)
            kt_max = max(float(np.max(k_all[:3])), EM20)
            ct_max = max(float(np.max(c_all[:3])), 0.0)
            kr_max = max(float(np.max(k_all[3:])), EM20)
            cr_max = max(float(np.max(c_all[3:])), 0.0)

            denom_t = math.sqrt(ct_max * ct_max + elem_mass * kt_max) + ct_max
            dt_tr = elem_mass / max(denom_t, EM20) if elem_mass > 0.0 else EP30

            if elem_inertia > 0.0:
                denom_r = math.sqrt(cr_max * cr_max + elem_inertia * kr_max) + cr_max
                dt_rot = elem_inertia / max(denom_r, EM20)
            else:
                dt_rot = EP30

            dtc[i_local] = min(dt_tr, dt_rot)

    return dtc
