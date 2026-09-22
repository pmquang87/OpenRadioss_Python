"""
/PROP/TYPE13 (/PROP/SPR_BEAM: 6-DOF nonlinear spring-beam element).

Upstream Fortran reference
--------------------------
* starter/source/properties/spring/hm_read_prop13.F:
  Parameter reader, default values, scale factors, and rupture limit initialization.
* starter/source/elements/spring/r4buf3.F (lines 146-244):
  Initial local triad (e1, e2, e3) definition from node 1, node 2, optional node 3 or skew frame.
* engine/source/elements/spring/r4evec3.F (lines 119-255):
  Local triad computation and dynamic torsion reorientation.
* engine/source/elements/spring/r4def3.F (lines 218-289, 336-377, 799-826):
  Generalized displacements and rates, transverse rotation-coupling, per-DOF rupture and deactivation.
* engine/source/elements/spring/redef3.F90 (lines 736-1158):
  Constitutive models: linear elastic, nonlinear elastic (HFLAG 0), isotropic hardening (HFLAG 1),
  decoupled tension/compression (HFLAG 2), kinematic hardening (HFLAG 4), hysteresis (HFLAG 7);
  strain-rate dependency (dfac = A + B ln(dvv) + E gx);
  nonlinear damping (C v + H gx2); trapezoidal internal energy accounting.
* engine/source/elements/spring/r4cum3.F (lines 80-138):
  Force and moment scattering to nodes 1 and 2 with moment-arm correction for shear equilibrium:
  f1 = +f, f2 = -f, m1 + m2 + r x f2 = 0.
* engine/source/elements/spring/r2len3.F (lines 180-186):
  Translational and rotational critical time step calculation.
* engine/source/elements/spring/r13ke3.F & r13sumg3.F (lines 56-150):
  12x12 tangent stiffness matrix in global frame with shear-moment coupling blocks.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..common.constants import EM15, EM20, EP30
from ..common.fastmath import norm3


def _eval_func(model, fid: int, x: float) -> float:
    """Evaluate curve fid from model.functions with linear interpolation and extrapolation."""
    if fid <= 0 or model is None:
        return 0.0
    funcs = getattr(model, "functions", None)
    if funcs is None or fid not in funcs:
        return 0.0
    f = funcs[fid]
    if hasattr(f, "eval"):
        return float(f.eval(x))
    if hasattr(f, "x") and hasattr(f, "y"):
        return float(np.interp(x, f.x, f.y))
    return 0.0


def init_spring_beam_type13(group, model, log, idx13, massn=None, inertn=None):
    """Initialize state for TYPE13 (/PROP/SPR_BEAM) elements.

    Fortran origin:
      starter/source/properties/spring/hm_read_prop13.F
      starter/source/elements/spring/r4buf3.F lines 146-244
    """
    if idx13 is None or len(idx13) == 0:
        return

    st = group.state
    m13 = len(idx13)
    conn = group.conn[idx13]

    # Model initial nodal coordinates
    n1 = conn[:, 0]
    n2 = conn[:, 1]
    n3 = conn[:, 2] if conn.shape[1] >= 3 else np.full(m13, -1, dtype=np.int64)

    x1 = model.x0[n1]
    x2 = model.x0[n2]
    d = x2 - x1
    L0 = norm3(d)

    e1 = np.zeros((m13, 3))
    e2 = np.zeros((m13, 3))
    e3 = np.zeros((m13, 3))

    mass = np.zeros(m13)
    inertia = np.zeros(m13)
    skew_id = np.zeros(m13, dtype=np.int64)
    sens_id = np.zeros(m13, dtype=np.int64)
    isflag = np.zeros(m13, dtype=np.int64)
    ifail = np.zeros(m13, dtype=np.int64)
    ileng = np.zeros(m13, dtype=np.int64)
    ifail2 = np.zeros(m13, dtype=np.int64)

    k6 = np.zeros((m13, 6))
    c6 = np.zeros((m13, 6))
    a6 = np.ones((m13, 6))
    b6 = np.zeros((m13, 6))
    d6 = np.ones((m13, 6))

    fun_a = np.zeros((m13, 6), dtype=np.int64)
    hflag = np.zeros((m13, 6), dtype=np.int64)
    fun_b = np.zeros((m13, 6), dtype=np.int64)
    fun_c = np.zeros((m13, 6), dtype=np.int64)
    fun_d = np.zeros((m13, 6), dtype=np.int64)

    min_rup = np.full((m13, 6), -1e30)
    max_rup = np.full((m13, 6), 1e30)

    f_scale = np.ones((m13, 6))
    e_scale = np.zeros((m13, 6))
    scale = np.ones((m13, 6))
    h_scale = np.ones((m13, 6))

    pos = {int(e): j for j, e in enumerate(idx13)}
    skews = getattr(model, "skews", None)

    for sl, _mat, prop in st.get("slices", []):
        pt = getattr(prop, "type", 4)
        if pt != 13:
            continue
        rng = np.arange(group.n)[sl]
        local = np.array([pos[int(e)] for e in rng if int(e) in pos], dtype=np.int64)
        if not len(local):
            continue
        p = getattr(prop, "params", {}) or {}

        mass[local] = float(p.get("mass", 0.0))
        inertia[local] = float(p.get("inertia", 0.0))
        skew_id[local] = int(p.get("skew_id", 0) or 0)
        sens_id[local] = int(p.get("sens_id", p.get("isensor", 0)) or 0)
        isflag[local] = int(p.get("isflag", 0) or 0)
        ifail[local] = int(p.get("ifail", 0) or 0)
        ileng[local] = int(p.get("ileng", 0) or 0)
        ifail2[local] = int(p.get("ifail2", 0) or 0)

        for i in range(6):
            dof = i + 1
            k6[local, i] = float(p.get(f"k{dof}", 0.0))
            c6[local, i] = float(p.get(f"c{dof}", 0.0))
            a_val = float(p.get(f"a{dof}", 1.0))
            a6[local, i] = 1.0 if a_val == 0.0 else a_val
            b6[local, i] = float(p.get(f"b{dof}", 0.0))
            d_val = float(p.get(f"d{dof}", 1.0))
            d6[local, i] = 1.0 if d_val == 0.0 else d_val

            fa = int(p.get(f"fun_a{dof}", 0) or 0)
            fun_a[local, i] = fa
            hflag[local, i] = int(p.get(f"hflag{dof}", 0) or 0)
            fun_b[local, i] = int(p.get(f"fun_b{dof}", 0) or 0)
            fun_c[local, i] = int(p.get(f"fun_c{dof}", 0) or 0)
            fun_d[local, i] = int(p.get(f"fun_d{dof}", 0) or 0)

            # hm_read_prop13.F lines 271-275: if IFUNC==0, A=1, B=0, E=0
            if fa == 0:
                a6[local, i] = 1.0
                b6[local, i] = 0.0

            min_rup[local, i] = float(p.get(f"min_rup{dof}", -1e30))
            max_rup[local, i] = float(p.get(f"max_rup{dof}", 1e30))

            f_scale[local, i] = float(p.get(f"f{dof}", 1.0) or 1.0)
            e_scale[local, i] = float(p.get(f"e{dof}", 0.0) or 0.0)
            scale[local, i] = float(p.get(f"scale{dof}", 1.0) or 1.0)
            h_scale[local, i] = float(p.get(f"h{dof}", 1.0) or 1.0)

    # Local triad construction (r4buf3.F lines 146-244)
    for j in range(m13):
        Lj = L0[j]
        if Lj > 1e-15:
            e1_j = d[j] / Lj
            # Node 3 vector or skew/global vector
            node3_idx = n3[j]
            if node3_idx >= 0 and node3_idx < len(model.x0):
                p_vec = model.x0[node3_idx] - x1[j]
            elif skew_id[j] != 0 and skews is not None and hasattr(skews, "axes"):
                row = int(skew_id[j])
                p_vec = skews.axes[row][1] if row < len(skews.axes) else np.array([0.0, 1.0, 0.0])
            else:
                p_vec = np.array([0.0, 1.0, 0.0])
                if abs(np.dot(e1_j, p_vec)) > 0.9:
                    p_vec = np.array([1.0, 0.0, 0.0])

            prvc = np.cross(e1_j, p_vec)
            if norm3(prvc) < 1e-5:
                p_vec = np.array([1.0, 0.0, 0.0])
                prvc = np.cross(e1_j, p_vec)

            # Ylocal = PRVC x e1 (r4buf3.F lines 237-243)
            p_perp = np.cross(prvc, e1_j)
            norm_perp = norm3(p_perp)
            if norm_perp > 1e-15:
                e2_j = p_perp / norm_perp
            else:
                e2_j = np.array([0.0, 1.0, 0.0])
            e3_j = np.cross(e1_j, e2_j)
            e3_j /= max(norm3(e3_j), 1e-15)
        else:
            # Degenerate zero-length spring: skew or global axes
            if skew_id[j] != 0 and skews is not None and hasattr(skews, "axes"):
                row = int(skew_id[j])
                ax = skews.axes[row] if row < len(skews.axes) else np.eye(3)
                e1_j, e2_j, e3_j = ax[0], ax[1], ax[2]
            else:
                e1_j = np.array([1.0, 0.0, 0.0])
                e2_j = np.array([0.0, 1.0, 0.0])
                e3_j = np.array([0.0, 0.0, 1.0])

        e1[j] = e1_j
        e2[j] = e2_j
        e3[j] = e3_j

    # Nodal lumped mass and inertia accumulation (half to each node)
    if massn is not None and inertn is not None:
        stride = group.conn.shape[1] if hasattr(group, "conn") and group.conn.ndim == 2 else 2
        for j, e in enumerate(idx13):
            m_half = mass[j] / 2.0
            in_half = inertia[j] / 2.0
            if stride * e + 1 < len(massn):
                massn[stride * e] += m_half
                massn[stride * e + 1] += m_half
            if stride * e + 1 < len(inertn):
                inertn[stride * e] += in_half
                inertn[stride * e + 1] += in_half

    # Element state storage
    beam_state = dict(
        idx=np.asarray(idx13, dtype=np.int64),
        conn=conn[:, :2],
        L0=L0,
        d0=d.copy(),
        e1=e1,
        e2=e2,
        e3=e3,
        e1_0=e1.copy(),
        e2_0=e2.copy(),
        e3_0=e3.copy(),
        mass=mass,
        inertia=inertia,
        skew_id=skew_id,
        sens_id=sens_id,
        isflag=isflag,
        ifail=ifail,
        ileng=ileng,
        ifail2=ifail2,
        k6=k6,
        c6=c6,
        a6=a6,
        b6=b6,
        d6=d6,
        fun_a=fun_a,
        hflag=hflag,
        fun_b=fun_b,
        fun_c=fun_c,
        fun_d=fun_d,
        min_rup=min_rup,
        max_rup=max_rup,
        f_scale=f_scale,
        e_scale=e_scale,
        scale=scale,
        h_scale=h_scale,
        dx=np.zeros((m13, 3)),
        dx_old=np.zeros((m13, 3)),
        theta=np.zeros((m13, 3)),
        theta_old=np.zeros((m13, 3)),
        force=np.zeros((m13, 3)),
        moment=np.zeros((m13, 3)),
        f_old=np.zeros((m13, 6)),
        dpx=np.zeros((m13, 6)),
        dpx2=np.zeros((m13, 6)),
        f_ep=np.zeros((m13, 6)),
        eint=np.zeros(m13),
        e6=np.zeros((m13, 6)),
        crit=np.zeros(m13),
        off=np.ones(m13, dtype=float),
    )
    st["spr_beam13"] = beam_state

    # Mirror into group.state["gen6"] for compatibility with general spring inspections
    # only when group has no TYPE8 elements (idx6)
    has_idx6 = len(st.get("idx6", [])) > 0
    if not has_idx6:
        if "gen6" not in st:
            st["gen6"] = {}
        st["gen6"].update({
            "e1": e1, "e2": e2, "e3": e3,
            "k6": k6, "c6": c6, "mass": mass, "inertia": inertia,
            "force": np.zeros((m13, 3)), "moment": np.zeros((m13, 3)),
            "eint": np.zeros(m13),
        })


def forces_spring_beam_type13(group, x, v, vr, dt, fint, mint, idx13=None):
    """Compute forces and moments for TYPE13 spring-beam elements.

    Fortran origin:
      engine/source/elements/spring/r4evec3.F (local triad update)
      engine/source/elements/spring/r4def3.F (kinematics, rupture)
      engine/source/elements/spring/redef3.F90 (constitutive law, damping, work)
      engine/source/elements/spring/r4cum3.F (force and moment equilibrium assembly)
      engine/source/elements/spring/r2len3.F (critical time step)
    """
    st = group.state
    b = st.get("spr_beam13")
    if b is None:
        return np.empty(0)

    m13 = len(b["idx"])
    if m13 == 0:
        return np.empty(0)

    model = st.get("model", None)
    conn = b["conn"]
    n1 = conn[:, 0]
    n2 = conn[:, 1]

    L0 = b["L0"]
    e1 = b["e1"]
    e2 = b["e2"]
    e3 = b["e3"]

    dt_safe = max(float(dt), 1e-30) if dt is not None and dt > 0.0 else 1e-30
    dt05 = 0.5 * dt_safe

    # Current geometry
    x1 = x[n1]
    x2 = x[n2]
    d = x2 - x1
    L = norm3(d)

    # 1. Update element orientation e1, e2, e3 (r4evec3.F lines 119-255)
    for j in range(m13):
        Lj = L[j]
        if Lj > 1e-15:
            e1_j = d[j] / Lj
        else:
            e1_j = e1[j]

        # Torsion rotation around e1 (r4evec3.F lines 186-228)
        if vr is not None and dt is not None and dt > 0.0:
            rx1 = float(np.dot(vr[n1[j]], e1_j))
            rx2 = float(np.dot(vr[n2[j]], e1_j))
            theta_tors = 0.5 * (rx1 + rx2) * dt05
            cc = math.cos(theta_tors)
            ss = math.sin(theta_tors)
            e2_rot = e2[j] * cc + e3[j] * ss
        else:
            e2_rot = e2[j]

        # Re-orthonormalize (r4evec3.F lines 230-255)
        e3_j = np.cross(e1_j, e2_rot)
        norm_e3 = norm3(e3_j)
        if norm_e3 > 1e-15:
            e3_j /= norm_e3
        else:
            e3_j = e3[j]

        e2_j = np.cross(e3_j, e1_j)
        norm_e2 = norm3(e2_j)
        if norm_e2 > 1e-15:
            e2_j /= norm_e2
        else:
            e2_j = e2[j]

        e1[j] = e1_j
        e2[j] = e2_j
        e3[j] = e3_j

    d0 = b.get("d0")
    if d0 is None:
        if model is not None and hasattr(model, "x0"):
            d0 = model.x0[n2] - model.x0[n1]
        else:
            d0 = np.zeros_like(d)

    # 2. Kinematics (r4def3.F lines 218-289)
    dx = np.zeros((m13, 3))
    theta = np.zeros((m13, 3))

    dx_old = b["dx"].copy()
    theta_old = b["theta"].copy()

    e1_ref = b.get("e1_0", e1)
    e2_ref = b.get("e2_0", e2)
    e3_ref = b.get("e3_0", e3)

    d_rel = d - d0
    for j in range(m13):
        # Local translational displacements resolved on element triad
        dx[j, 0] = float(np.dot(d_rel[j], e1_ref[j]))
        dx[j, 1] = float(np.dot(d_rel[j], e2_ref[j]))
        dx[j, 2] = float(np.dot(d_rel[j], e3_ref[j]))

        # Relative rotations (rate integration)
        if vr is not None and dt is not None and dt > 0.0:
            w_diff = vr[n2[j]] - vr[n1[j]]
            theta[j, 0] = theta_old[j, 0] + float(np.dot(w_diff, e1_ref[j])) * dt_safe
            theta[j, 1] = theta_old[j, 1] + float(np.dot(w_diff, e2_ref[j])) * dt_safe
            theta[j, 2] = theta_old[j, 2] + float(np.dot(w_diff, e3_ref[j])) * dt_safe
        else:
            theta[j] = theta_old[j]

    # Generalized displacement and velocity vectors for all 6 DOFs
    gen_disp = np.hstack([dx, theta])          # (m13, 6)
    gen_disp_old = np.hstack([dx_old, theta_old])

    v_loc = np.zeros((m13, 3))
    w_loc = np.zeros((m13, 3))
    if v is not None:
        v_diff = v[n2] - v[n1]
        for j in range(m13):
            v_loc[j, 0] = float(np.dot(v_diff[j], e1_ref[j]))
            v_loc[j, 1] = float(np.dot(v_diff[j], e2_ref[j]))
            v_loc[j, 2] = float(np.dot(v_diff[j], e3_ref[j]))
    else:
        v_loc = (dx - dx_old) / dt_safe

    if vr is not None:
        w_diff = vr[n2] - vr[n1]
        for j in range(m13):
            w_loc[j, 0] = float(np.dot(w_diff[j], e1_ref[j]))
            w_loc[j, 1] = float(np.dot(w_diff[j], e2_ref[j]))
            w_loc[j, 2] = float(np.dot(w_diff[j], e3_ref[j]))
    else:
        w_loc = (theta - theta_old) / dt_safe

    gen_vel = np.hstack([v_loc, w_loc])

    # 3. Constitutive model evaluation (redef3.F90 lines 736-1158)
    forces_local = np.zeros((m13, 6))
    dt_elem = np.full(m13, EP30)

    for j in range(m13):
        if b["off"][j] <= 0.0:
            continue

        ileng_flag = b["ileng"][j]
        L_ref = L0[j] if ileng_flag != 0 and L0[j] > 1e-15 else 1.0

        crit_elem = 0.0
        ifail_mode = b["ifail"][j]
        ifail2_mode = b["ifail2"][j]

        for k in range(6):
            d_k = gen_disp[j, k]
            d_old_k = gen_disp_old[j, k]
            v_k = gen_vel[j, k]

            d_bar = d_k / L_ref
            d_bar_old = d_old_k / L_ref
            delta_d = d_bar - d_bar_old
            v_bar = v_k / L_ref

            stiff = b["k6"][j, k]
            damp = b["c6"][j, k]
            acoef = b["a6"][j, k]
            bcoef = b["b6"][j, k]
            dcoef = b["d6"][j, k]

            fa = b["fun_a"][j, k]
            hf = b["hflag"][j, k]
            fb = b["fun_b"][j, k]
            fc = b["fun_c"][j, k]
            fd = b["fun_d"][j, k]

            f_sc = b["f_scale"][j, k]
            e_sc = b["e_scale"][j, k]
            ascale = b["scale"][j, k]
            h_sc = b["h_scale"][j, k]

            # Base force/moment calculation
            if fa == 0:
                # Linear elastic (redef3.F90 line 739)
                f_base = stiff * d_bar
            else:
                x_eval = d_bar * ascale
                if hf == 0:
                    # Nonlinear elastic (redef3.F90 line 756)
                    f_base = _eval_func(model, fa, x_eval)
                elif hf == 1:
                    # Isotropic hardening (redef3.F90 lines 764-774)
                    f_trial = b["f_ep"][j, k] + stiff * delta_d
                    if f_trial >= 0.0:
                        f_cap = _eval_func(model, fa, (b["dpx"][j, k] + f_trial / max(stiff, 1e-20)) * ascale)
                        if f_trial > f_cap:
                            b["dpx"][j, k] += (f_trial - f_cap) / max(stiff, 1e-20)
                            f_base = f_cap
                        else:
                            f_base = f_trial
                    else:
                        f_cap = _eval_func(model, fa, (-b["dpx"][j, k] + f_trial / max(stiff, 1e-20)) * ascale)
                        if f_trial < f_cap:
                            b["dpx"][j, k] += (f_cap - f_trial) / max(stiff, 1e-20)
                            f_base = f_cap
                        else:
                            f_base = f_trial
                    b["f_ep"][j, k] = f_base
                elif hf == 2:
                    # Decoupled tension/compression hardening (redef3.F90 lines 780-796)
                    if d_bar > b["dpx"][j, k]:
                        f_trial = stiff * (d_bar - b["dpx"][j, k])
                        f_cap = _eval_func(model, fa, x_eval)
                        f_base = min(f_trial, f_cap)
                        b["dpx"][j, k] = d_bar - f_base / max(stiff, 1e-20)
                    elif d_bar < b["dpx2"][j, k]:
                        f_trial = stiff * (d_bar - b["dpx2"][j, k])
                        f_cap = _eval_func(model, fa, x_eval)
                        f_base = max(f_trial, f_cap)
                        b["dpx2"][j, k] = d_bar - f_base / max(stiff, 1e-20)
                    else:
                        f_base = 0.0
                elif hf == 4:
                    # Kinematic hardening with loading/unloading curves (redef3.F90 lines 819-838)
                    f_trial = b["f_ep"][j, k] + stiff * delta_d
                    f_max = _eval_func(model, fa, x_eval)
                    f_min = _eval_func(model, fc, x_eval) if fc > 0 else -f_max
                    f_base = min(max(f_trial, f_min), f_max)
                    b["f_ep"][j, k] = f_base
                    b["dpx"][j, k] = d_bar - f_base / max(stiff, 1e-20)
                elif hf == 7:
                    # Hysteresis (redef3.F90 lines 932-947)
                    f_trial = b["f_ep"][j, k] + stiff * delta_d
                    y1 = _eval_func(model, fa, x_eval)
                    y2 = _eval_func(model, fc, x_eval) if fc > 0 else y1
                    if d_bar >= d_bar_old and d_bar >= 0.0:
                        f_base = min(f_trial, y1)
                    elif d_bar < d_bar_old and d_bar >= 0.0:
                        f_base = max(f_trial, y2)
                    elif d_bar >= d_bar_old and d_bar < 0.0:
                        f_base = min(f_trial, y2)
                    elif d_bar < d_bar_old and d_bar < 0.0:
                        f_base = max(f_trial, y1)
                    else:
                        f_base = f_trial
                    b["f_ep"][j, k] = f_base
                    b["dpx"][j, k] = d_bar - f_base / max(stiff, 1e-20)
                else:
                    f_base = stiff * d_bar

            # Damping & Strain Rate (redef3.F90 lines 1140-1145)
            v_scaled = v_bar * f_sc
            gx = _eval_func(model, fb, v_scaled) if fb > 0 else 0.0
            gx2 = _eval_func(model, fd, v_scaled) if fd > 0 else 0.0

            dvv = max(1.0, abs(v_bar / max(dcoef, 1e-20)))
            dfac = acoef + bcoef * math.log(dvv) + e_sc * gx
            f_tot = dfac * f_base + damp * v_bar + h_sc * gx2

            forces_local[j, k] = f_tot

            # Trapezoidal work integration into internal energy
            de_k = (d_bar - d_bar_old) * 0.5 * (f_tot + b["f_old"][j, k]) * L_ref
            b["e6"][j, k] += de_k
            b["eint"][j] += de_k
            b["f_old"][j, k] = f_tot

            # Rupture / Failure criterion check (r4def3.F lines 336-377, 751-791)
            dmin = b["min_rup"][j, k]
            dmax = b["max_rup"][j, k]
            if dmin != 0.0 or dmax != 0.0:
                dlim = 0.0
                if ifail2_mode == 0:
                    # Displacement limit
                    if d_bar > 0.0 and dmax < 1e29:
                        dlim = d_bar / max(dmax, 1e-20)
                    elif d_bar < 0.0 and dmin > -1e29:
                        dlim = d_bar / min(dmin, -1e-20)
                elif ifail2_mode == 2:
                    # Force limit
                    if f_tot > 0.0 and dmax < 1e29:
                        dlim = f_tot / max(dmax, 1e-20)
                    elif f_tot < 0.0 and dmin > -1e29:
                        dlim = f_tot / min(dmin, -1e-20)
                elif ifail2_mode == 3:
                    # Energy limit
                    if dmax < 1e29:
                        dlim = max(0.0, b["e6"][j, k]) / max(dmax, 1e-20)

                if ifail_mode == 0:
                    crit_elem = max(crit_elem, dlim)
                else:
                    crit_elem += dlim ** 2

        b["crit"][j] = crit_elem
        if crit_elem >= 1.0:
            # Rupture / deactivation (r4def3.F lines 819-826)
            b["off"][j] = 0.0
            forces_local[j, :] = 0.0

        # Critical time step (r2len3.F lines 180-186)
        if b["off"][j] > 0.0:
            k_tr = max(float(np.max(b["k6"][j, :3])), 1e-15)
            c_tr = float(np.max(b["c6"][j, :3]))
            m_tr = max(b["mass"][j], 1e-15)
            dt_tr = m_tr / (math.sqrt(c_tr ** 2 + m_tr * k_tr) + c_tr)

            L_elem = max(L[j], 1e-15)
            k_rot = max(float(np.max(b["k6"][j, 3:])), 1e-15) * (L_elem ** 2)
            c_rot = float(np.max(b["c6"][j, 3:])) * (L_elem ** 2)
            i_rot = max(b["inertia"][j], 1e-15)
            dt_rot = i_rot / (math.sqrt(c_rot ** 2 + i_rot * k_rot) + c_rot)

            dt_elem[j] = min(dt_tr, dt_rot)

    # Store state
    b["dx"] = dx
    b["theta"] = theta
    b["force"] = forces_local[:, :3]
    b["moment"] = forces_local[:, 3:]

    # 4. Assembly of forces and moments with moment arm correction (r4cum3.F lines 80-138)
    for j in range(m13):
        if b["off"][j] <= 0.0:
            continue

        fx_loc = forces_local[j, 0]
        fy_loc = forces_local[j, 1]
        fz_loc = forces_local[j, 2]

        mx_loc = forces_local[j, 3]
        my_loc = forces_local[j, 4]
        mz_loc = forces_local[j, 5]

        e1_j = e1[j]
        e2_j = e2[j]
        e3_j = e3[j]
        Lj = L[j]

        # Global force: f = F1 e1 + F2 e2 + F3 e3 (r4cum3.F line 80-82)
        f_glob = fx_loc * e1_j + fy_loc * e2_j + fz_loc * e3_j

        # Transverse force moment arms (r4cum3.F lines 108-117)
        # Node 1: My1 = My - 0.5 L Fz, Mz1 = Mz + 0.5 L Fy
        # Node 2: My2 = My + 0.5 L Fz, Mz2 = Mz - 0.5 L Fy
        half_l = 0.5 * Lj
        my1 = my_loc - half_l * fz_loc
        mz1 = mz_loc + half_l * fy_loc

        my2 = my_loc + half_l * fz_loc
        mz2 = mz_loc - half_l * fy_loc

        m1_glob = mx_loc * e1_j + my1 * e2_j + mz1 * e3_j
        m2_glob = -(mx_loc * e1_j + my2 * e2_j + mz2 * e3_j)

        node1 = n1[j]
        node2 = n2[j]

        if fint is not None:
            fint[node1] += f_glob
            fint[node2] -= f_glob

        if mint is not None:
            mint[node1] += m1_glob
            mint[node2] += m2_glob

    # Mirror force/moment into gen6 for inspection if no TYPE8 elements
    if "gen6" in st and len(st.get("idx6", [])) == 0:
        st["gen6"]["force"] = b["force"]
        st["gen6"]["moment"] = b["moment"]
        st["gen6"]["eint"] = b["eint"]

    if "eint" in st:
        st["eint"][idx13] = b["eint"]
    if "force" in st:
        st["force"][idx13] = b["force"][:, 0]

    return dt_elem


def ke_spring_beam_type13(group, x, idx13=None):
    """Compute the 12x12 tangent stiffness matrix in global frame for TYPE13 elements.

    Fortran origin:
      engine/source/elements/spring/r13sumg3.F lines 56-150
      engine/source/elements/spring/r13ke3.F
    """
    st = group.state
    b = st.get("spr_beam13")
    if b is None:
        return np.empty((0, 12, 12))

    if idx13 is None:
        idx13 = np.arange(len(b["idx"]))
    else:
        pos = {int(e): j for j, e in enumerate(b["idx"])}
        idx13 = np.array([pos[int(e)] for e in idx13 if int(e) in pos], dtype=np.int64)

    m = len(idx13)
    if m == 0:
        return np.empty((0, 12, 12))

    conn = b["conn"][idx13]
    n1 = conn[:, 0]
    n2 = conn[:, 1]

    x1 = x[n1]
    x2 = x[n2]
    d = x2 - x1
    L = norm3(d)

    ke = np.zeros((m, 12, 12))

    for j_idx, j in enumerate(idx13):
        if b["off"][j] <= 0.0:
            continue

        Lj = max(L[j_idx], 1e-15)
        e1_j = b["e1"][j]
        e2_j = b["e2"][j]
        e3_j = b["e3"][j]

        # Transformation matrix Q = [e1, e2, e3] (columns)
        Q = np.column_stack([e1_j, e2_j, e3_j])

        kx = b["k6"][j, 0]
        ky = b["k6"][j, 1]
        kz = b["k6"][j, 2]

        mx = b["k6"][j, 3]
        my = b["k6"][j, 4]
        mz = b["k6"][j, 5]

        # r13sumg3.F lines 56-70
        aldemi = 0.5 * Lj
        mf23 = aldemi * ky
        mf32 = aldemi * kz

        m11_loc = np.array([mx, my + aldemi * mf23, mz + aldemi * mf32])
        m12_loc = np.array([-mx, Lj * mf32 - m11_loc[1], Lj * mf23 - m11_loc[2]])

        # Local to global transformation
        K11_trans = Q @ np.diag([kx, ky, kz]) @ Q.T
        M11_rot = Q @ np.diag(m11_loc) @ Q.T
        M12_rot = Q @ np.diag(m12_loc) @ Q.T

        # Cross coupling blocks (r13sumg3.F lines 106-113)
        # MF11 = Q(:,2)*MF23*Q(:,3)^T - Q(:,3)*MF32*Q(:,2)^T
        MF11 = mf23 * np.outer(Q[:, 1], Q[:, 2]) - mf32 * np.outer(Q[:, 2], Q[:, 1])

        # 6x6 blocks
        KE11 = np.zeros((6, 6))
        KE11[:3, :3] = K11_trans
        KE11[:3, 3:] = MF11
        KE11[3:, :3] = MF11.T
        KE11[3:, 3:] = M11_rot

        KE22 = np.zeros((6, 6))
        KE22[:3, :3] = K11_trans
        KE22[:3, 3:] = -MF11
        KE22[3:, :3] = -MF11.T
        KE22[3:, 3:] = M11_rot

        KE12 = np.zeros((6, 6))
        KE12[:3, :3] = -K11_trans
        KE12[:3, 3:] = MF11
        KE12[3:, :3] = -MF11.T
        KE12[3:, 3:] = M12_rot

        ke[j_idx, :6, :6] = KE11
        ke[j_idx, :6, 6:] = KE12
        ke[j_idx, 6:, :6] = KE12.T
        ke[j_idx, 6:, 6:] = KE22

    edofs = np.empty((m, 12), dtype=np.int64)
    for c in range(6):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 6 + c] = conn[:, 1] * 6 + c

    return ke, edofs
