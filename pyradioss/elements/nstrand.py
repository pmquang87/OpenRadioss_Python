"""
pyradioss.elements.nstrand — /PROP/TYPE28 (/PROP/NSTRAND) Multi-strand cable / wire rope.

Fortran origin:
  * Engine:
    - ``engine/source/elements/xelem/xforc28.F``: constitutive law, capstan pulley friction,
      damping, rupture criteria, internal energy, critical time step.
    - ``engine/source/elements/xelem/xforc3.F``: multi-purpose element driver.
    - ``engine/source/user_interface/ufunc.F``: GET_U_FUNC function and derivative evaluation.
  * Starter:
    - ``starter/source/properties/xelem/hm_read_prop28.F``: property reader, parameters, scale factors.
    - ``starter/source/elements/xelem/xini28.F``: initial geometry, nodal masses, initial dt.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from ..common.constants import EM20, EP30
EM15 = 1.0e-15
EP20 = 1.0e20
from ..common.messages import MessageLog


def _safe_param(params: dict, key: str, default: float = 0.0) -> float:
    """Safely extract float parameter from prop.params with fallback."""
    val = params.get(key)
    if val is None:
        return default
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (ValueError, TypeError):
        return default


def _safe_int_param(params: dict, key: str, default: int = 0) -> int:
    """Safely extract int parameter from prop.params with fallback."""
    val = params.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def eval_func_and_slope(func: Any, x_val: float) -> Tuple[float, float]:
    """Evaluate curve value and slope dF/dx at x_val with exact Fortran GET_U_FUNC fidelity.

    Fortran origin: ``engine/source/user_interface/ufunc.F`` (lines 174-219):
    Piecewise linear interpolation with slope from adjacent points, and linear
    extrapolation using the first or last segment slope outside bounds.
    """
    if func is None:
        return 0.0, 0.0

    if hasattr(func, "eval_and_slope"):
        return func.eval_and_slope(x_val)

    if hasattr(func, "x") and hasattr(func, "y"):
        x = np.asarray(func.x, dtype=np.float64)
        y = np.asarray(func.y, dtype=np.float64)
        npts = len(x)
        if npts == 0:
            return 0.0, 0.0
        if npts == 1:
            return float(y[0]), 0.0

        if hasattr(func, "slope") and len(func.slope) == npts - 1:
            slope = np.asarray(func.slope, dtype=np.float64)
        else:
            dx = np.diff(x)
            dx = np.where(np.abs(dx) < 1.0e-30, 1.0e-30, dx)
            slope = np.diff(y) / dx

        # Fortran ufunc.F lines 193-216: find segment where x_val <= x[i] or last segment
        if x_val <= x[0]:
            val = float(y[0] + slope[0] * (x_val - x[0]))
            return val, float(slope[0])
        elif x_val >= x[-1]:
            val = float(y[-1] + slope[-1] * (x_val - x[-1]))
            return val, float(slope[-1])
        else:
            # Segment index: searchsorted
            idx = int(np.clip(np.searchsorted(x, x_val, side="right") - 1, 0, npts - 2))
            val = float(y[idx] + slope[idx] * (x_val - x[idx]))
            return val, float(slope[idx])

    # Fallback if only eval is provided
    if hasattr(func, "eval"):
        val = float(func.eval(x_val))
        eps = max(1.0e-7, abs(x_val) * 1.0e-5)
        val_plus = float(func.eval(x_val + eps))
        deri = (val_plus - val) / eps
        return val, float(deri)

    return 0.0, 0.0


def init_nstrand_type28(
    group: Any,
    model: Any,
    log: Any,
    idx28: Optional[Sequence[int]] = None,
    massn: Optional[np.ndarray] = None,
    inertn: Optional[np.ndarray] = None,
) -> None:
    """Initialize state arrays, segment lengths, and lumped nodal masses for /PROP/TYPE28 (NSTRAND).

    Fortran origin:
      - Starter reader: ``starter/source/properties/xelem/hm_read_prop28.F``
      - Initialization: ``starter/source/elements/xelem/xini28.F`` (lines 166-382)
    """
    st = group.state
    n = group.n
    if idx28 is None:
        elem_indices = list(range(n))
    elif isinstance(idx28, slice):
        elem_indices = list(range(n)[idx28])
    else:
        elem_indices = list(idx28)

    if len(elem_indices) == 0:
        return

    # Base arrays in group.state
    if "off" not in st:
        st["off"] = np.ones(n, dtype=np.float64)
    if "force" not in st:
        st["force"] = np.zeros(n, dtype=np.float64)
    if "eint" not in st:
        st["eint"] = np.zeros(n, dtype=np.float64)
    if "L0" not in st:
        st["L0"] = np.zeros(n, dtype=np.float64)
    if "mass" not in st:
        st["mass"] = np.zeros(n, dtype=np.float64)
    if "k" not in st:
        st["k"] = np.zeros(n, dtype=np.float64)
    if "cdamp" not in st:
        st["cdamp"] = np.zeros(n, dtype=np.float64)

    # Per-element persistent lists / containers for multi-strand data
    if "t28_L0_k" not in st:
        st["t28_L0_k"] = [np.empty(0, dtype=np.float64) for _ in range(n)]
        st["t28_dx_old"] = np.zeros(n, dtype=np.float64)
        st["t28_fx_old"] = np.zeros(n, dtype=np.float64)
        st["t28_dfs"] = [np.empty(0, dtype=np.float64) for _ in range(n)]
        st["t28_dfs_old"] = [np.empty(0, dtype=np.float64) for _ in range(n)]
        st["t28_dl_old"] = [np.empty(0, dtype=np.float64) for _ in range(n)]
        st["t28_eint_k"] = [np.empty(0, dtype=np.float64) for _ in range(n)]
        st["t28_xk"] = np.zeros(n, dtype=np.float64)
        st["t28_xc"] = np.zeros(n, dtype=np.float64)
        st["t28_fun_a1"] = np.zeros(n, dtype=np.int64)
        st["t28_fun_b1"] = np.zeros(n, dtype=np.int64)
        st["t28_ffac"] = np.ones(n, dtype=np.float64)
        st["t28_xfac"] = np.ones(n, dtype=np.float64)
        st["t28_epsmin"] = np.full(n, -1.0e30, dtype=np.float64)
        st["t28_epsmax"] = np.full(n, 1.0e30, dtype=np.float64)
        st["t28_mu1"] = np.zeros(n, dtype=np.float64)
        st["t28_mu2"] = np.zeros(n, dtype=np.float64)
        st["t28_rho"] = np.zeros(n, dtype=np.float64)
        st["t28_pulley_fric"] = [{} for _ in range(n)]
        st["t28_strand_fric"] = [{} for _ in range(n)]

    # Slices mapping to properties
    slices = st.get("slices", [])

    for e in elem_indices:
        # Determine property for element e
        prop = None
        for sl, mat, pr in slices:
            if isinstance(sl, slice):
                start = sl.start or 0
                stop = sl.stop if sl.stop is not None else n
                step = sl.step or 1
                if e in range(start, stop, step):
                    prop = pr
                    break
            elif isinstance(sl, (list, tuple, np.ndarray)):
                if e in sl:
                    prop = pr
                    break

        # Connectivity for element e (list or 1D array of node indices)
        if hasattr(group.conn, "ndim") and group.conn.ndim == 2:
            conn_e = group.conn[e]
        else:
            conn_e = group.conn[e]
        conn_e = np.asarray(conn_e, dtype=np.int64)
        nx = len(conn_e)

        # Fortran xini28.F line 192: check NX >= 3
        if nx < 3:
            msg = f"** ERROR NSTRAND : LESS THAN 3 NODES, ELEMENT={getattr(group, 'ids', [e])[e]}"
            if log is not None:
                log.error(msg)
            raise ValueError(msg)

        # Extract parameters from prop
        params: dict = {}
        if prop is not None:
            if hasattr(prop, "params") and isinstance(prop.params, dict):
                params = prop.params
            elif hasattr(prop, "__dict__"):
                params = prop.__dict__

        rho = getattr(prop, "mass", _safe_param(params, "mass", 0.0))
        xk = getattr(prop, "k", _safe_param(params, "stiff2", _safe_param(params, "k", 0.0)))
        xc = getattr(prop, "c", _safe_param(params, "damp2", _safe_param(params, "c", 0.0)))
        fun_a1 = getattr(prop, "fun_a1", _safe_int_param(params, "fun_a1", _safe_int_param(params, "fun_k", 0)))
        fun_b1 = getattr(prop, "fun_b1", _safe_int_param(params, "fun_b1", _safe_int_param(params, "fun_c", 0)))
        strain1 = getattr(prop, "strain1", _safe_param(params, "strain1", _safe_param(params, "dmin", -1.0e30)))
        strain2 = getattr(prop, "strain2", _safe_param(params, "strain2", _safe_param(params, "dmax", 1.0e30)))
        if strain1 == 0.0:
            strain1 = -1.0e30
        if strain2 == 0.0:
            strain2 = 1.0e30

        # Scale factors (hm_read_prop28.F lines 166-167, 269-285)
        # Y_SCAL = FFAC, X_SCAL = FScale22 -> XFAC = 1.0 / X_SCAL
        fscale11 = getattr(prop, "fscale11", _safe_param(params, "fscale11", _safe_param(params, "ffac", 1.0)))
        fscale22 = getattr(prop, "fscale22", _safe_param(params, "fscale22", 1.0))
        ffac = fscale11 if fscale11 != 0.0 else 1.0
        xfac = (1.0 / fscale22) if fscale22 != 0.0 else 1.0
        if "xfac" in params and "fscale22" not in params:
            xfac = _safe_param(params, "xfac", 1.0)

        mu1 = getattr(prop, "mu1", _safe_param(params, "mu1", _safe_param(params, "mat_mue1", 0.0)))
        mu2 = getattr(prop, "mu2", _safe_param(params, "mu2", _safe_param(params, "mat_mue2", 0.0)))

        # Layer overrides (pulley / strand specific friction)
        pulley_fric: Dict[int, float] = {}
        strand_fric: Dict[int, float] = {}
        layers = getattr(prop, "layers", params.get("layers", []))
        if layers:
            for lay in layers:
                tname = getattr(lay, "type_name", lay.get("type_name", "") if isinstance(lay, dict) else "")
                tname = str(tname).strip().upper()
                kid = getattr(lay, "k_id", lay.get("k_id", 0) if isinstance(lay, dict) else 0)
                mu_val = getattr(lay, "mu", lay.get("mu", 0.0) if isinstance(lay, dict) else 0.0)
                if tname == "PULLEY":
                    pulley_fric[int(kid)] = float(mu_val)
                elif tname == "STRAND":
                    strand_fric[int(kid)] = float(mu_val)

        # Segment lengths at initial configuration (xini28.F lines 208-270)
        x0_elem = model.x0[conn_e]  # shape (nx, 3)
        n_strands = nx - 1
        L0_k = np.zeros(n_strands, dtype=np.float64)
        for k in range(n_strands):
            diff = x0_elem[k + 1] - x0_elem[k]
            L_seg = float(np.linalg.norm(diff))
            if L_seg <= EM15:
                msg = f"** ERROR NSTRAND : NULL STRAND LENGTH, ELEMENT={getattr(group, 'ids', [e])[e]}"
                if log is not None:
                    log.error(msg)
                raise ValueError(msg)
            L0_k[k] = L_seg

        L0_tot = float(np.sum(L0_k))

        # Nodal lumped masses (xini28.F lines 215, 254, 270)
        # MASS(1) = 0.5 * RHO * L1
        # MASS(K) = 0.5 * RHO * (L_{K-1} + L_K) for K=2..NX-1
        # MASS(NX) = 0.5 * RHO * L_{NX-1}
        node_mass = np.zeros(nx, dtype=np.float64)
        node_mass[0] = 0.5 * rho * L0_k[0]
        for k in range(1, nx - 1):
            node_mass[k] = 0.5 * rho * (L0_k[k - 1] + L0_k[k])
        node_mass[nx - 1] = 0.5 * rho * L0_k[-1]

        # Accumulate lumped mass into massn if requested
        if massn is not None:
            stride = group.conn.shape[1] if hasattr(group, "conn") and hasattr(group.conn, "shape") and group.conn.ndim == 2 else nx
            if len(massn) == n * stride:
                for k in range(nx):
                    massn[e * stride + k] += node_mass[k]
            elif len(massn) == getattr(model, "numnod", 0):
                for k in range(nx):
                    massn[conn_e[k]] += node_mass[k]

        # Store parameters into state
        st["L0"][e] = L0_tot
        st["mass"][e] = rho * L0_tot
        st["k"][e] = xk
        st["cdamp"][e] = xc

        st["t28_L0_k"][e] = L0_k
        st["t28_dx_old"][e] = 0.0
        st["t28_fx_old"][e] = 0.0
        st["t28_dfs"][e] = np.zeros(n_strands, dtype=np.float64)
        st["t28_dfs_old"][e] = np.zeros(n_strands, dtype=np.float64)
        st["t28_dl_old"][e] = np.zeros(n_strands, dtype=np.float64)
        st["t28_eint_k"][e] = np.zeros(n_strands, dtype=np.float64)

        st["t28_xk"][e] = xk
        st["t28_xc"][e] = xc
        st["t28_fun_a1"][e] = fun_a1
        st["t28_fun_b1"][e] = fun_b1
        st["t28_ffac"][e] = ffac
        st["t28_xfac"][e] = xfac
        st["t28_epsmin"][e] = strain1
        st["t28_epsmax"][e] = strain2
        st["t28_mu1"][e] = mu1
        st["t28_mu2"][e] = mu2
        st["t28_rho"][e] = rho
        st["t28_pulley_fric"][e] = pulley_fric
        st["t28_strand_fric"][e] = strand_fric


def forces_nstrand_type28(
    group: Any,
    x: np.ndarray,
    v: Optional[np.ndarray],
    vr: Optional[np.ndarray],
    dt: Optional[float],
    fint: Optional[np.ndarray],
    mint: Optional[np.ndarray],
    idx28: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Compute internal forces, capstan pulley friction, damping, energy, and critical dt for /PROP/TYPE28.

    Fortran origin:
      - ``engine/source/elements/xelem/xforc28.F`` (lines 208-540)
      - ``engine/source/elements/xelem/xforc3.F`` (lines 151-161)
    """
    st = group.state
    n = group.n
    if idx28 is None:
        elem_indices = list(range(n))
    elif isinstance(idx28, slice):
        elem_indices = list(range(n)[idx28])
    else:
        elem_indices = list(idx28)

    n_sel = len(elem_indices)
    if n_sel == 0:
        return np.empty(0, dtype=np.float64)

    dt_val = float(dt) if (dt is not None and dt > 0.0) else 0.0
    dt11 = dt_val if dt_val > 0.0 else EP30

    model = st.get("model")
    has_funcs = model is not None and hasattr(model, "functions")

    dt_crit = np.full(n_sel, EP30, dtype=np.float64)

    for i_local, e in enumerate(elem_indices):
        # Check if already broken (xforc28.F line 208)
        if st["off"][e] == 0.0:
            dt_crit[i_local] = EP30
            continue

        # Node coordinates
        if hasattr(group.conn, "ndim") and group.conn.ndim == 2:
            conn_e = group.conn[e]
        else:
            conn_e = group.conn[e]
        conn_e = np.asarray(conn_e, dtype=np.int64)
        nx = len(conn_e)
        n_strands = nx - 1

        xel = x[conn_e]  # shape (nx, 3)

        # Segment current lengths L_k and total length L (xforc28.F lines 213-221)
        L_k = np.zeros(n_strands, dtype=np.float64)
        torq = np.zeros((n_strands, 3), dtype=np.float64)  # unit vectors u_k
        L_tot = 0.0
        for k in range(n_strands):
            diff = xel[k + 1] - xel[k]
            Lk = float(np.linalg.norm(diff))
            L_k[k] = Lk
            L_tot += Lk
            vv = 1.0 / max(EM15, Lk)
            torq[k, :] = diff * vv  # xforc28.F line 298-300

        L0_tot = st["L0"][e]
        L0_k = st["t28_L0_k"][e]
        if L0_tot <= EM15:
            L0_tot = 1.0e-15

        dx = L_tot - L0_tot
        epstot = dx / L0_tot  # engineering strain (xforc28.F line 238, 247, 253)

        # Rupture criteria (xforc28.F lines 311-320)
        epsmin = st["t28_epsmin"][e]
        epsmax = st["t28_epsmax"][e]
        if epstot < epsmin or epstot > epsmax:
            st["off"][e] = 0.0
            st["force"][e] = 0.0
            # Energy finalization on rupture (xforc28.F lines 437-458)
            fx_old = st["t28_fx_old"][e]
            dl_old = st["t28_dl_old"][e]
            dfs = st["t28_dfs"][e]
            dfs_old = st["t28_dfs_old"][e]
            for k in range(n_strands):
                dl_k = L_k[k] - L0_k[k]
                w_k = 0.5 * (dl_k - dl_old[k]) * (0.0 + fx_old + dfs[k] + dfs_old[k])
                st["t28_eint_k"][e][k] += w_k
                dl_old[k] = dl_k
                dfs[k] = 0.0  # vanish forces into strands (line 455)
            st["t28_fx_old"][e] = 0.0
            st["eint"][e] = float(np.sum(st["t28_eint_k"][e]))
            dt_crit[i_local] = EP30
            continue

        # Constitutive law (xforc28.F lines 230-258)
        xk = st["t28_xk"][e]
        fun_a1 = st["t28_fun_a1"][e]
        fun_b1 = st["t28_fun_b1"][e]
        ffac = st["t28_ffac"][e]
        xfac = st["t28_xfac"][e]
        xc = st["t28_xc"][e]

        if fun_a1 == 0 and fun_b1 == 0:
            # Linear elastic (xforc28.F lines 232-238)
            stif = xk
            f_elast = xk * dx / L0_tot
        elif fun_a1 == 0 and fun_b1 != 0:
            # Viscous only (xforc28.F lines 240-248)
            stif = 0.0
            f_elast = ffac
        else:
            # Nonlinear elastic (xforc28.F lines 250-258)
            func_a = model.functions.get(fun_a1) if has_funcs else None
            f_raw, dfdx = eval_func_and_slope(func_a, epstot)
            f_elast = ffac * f_raw
            stif = ffac * dfdx

        # Linear damping & strain rate effects (xforc28.F lines 260-285)
        dx_old = st["t28_dx_old"][e]
        deps = (epstot - dx_old / L0_tot) / dt11
        st["t28_dx_old"][e] = dx

        dgdx = 0.0
        if fun_b1 > 0:
            func_b = model.functions.get(fun_b1) if has_funcs else None
            g, dgdx = eval_func_and_slope(func_b, deps * xfac)
            dgdx = dgdx * xfac
            fx = f_elast * g + xc * deps
            stif = stif * g
        else:
            fx = f_elast + xc * deps
            g = 1.0

        stif = stif * L_tot / L0_tot

        # Pulley friction setup (xforc28.F lines 322-338)
        mu1 = st["t28_mu1"][e]
        mu2 = st["t28_mu2"][e]
        pulley_fric = st["t28_pulley_fric"][e]
        strand_fric = st["t28_strand_fric"][e]

        epsmoy = (dx - dx_old) / max(EM15, L_tot)
        xn = float(nx)

        vel = v[conn_e] if v is not None else np.zeros_like(xel)

        # In explicit time integration, dt * DDX is the segment length increment.
        # If velocity v is not provided or uncoupled from x (e.g. static/prescribed displacement),
        # the physical segment increment is Delta L_k = L_k - (dl_old[k] + L0_k[k]).
        use_pos_inc = (v is None or (dt_val > 0.0 and np.all(vel == 0.0) and abs(dx - dx_old) > 1.0e-15))

        dfs = st["t28_dfs"][e]
        dl_old = st["t28_dl_old"][e]
        for k in range(n_strands):
            if use_pos_inc:
                dl_inc = L_k[k] - (dl_old[k] + L0_k[k])
                dfs[k] += stif * (dl_inc * (xn - 1.0) / max(EM15, L_tot) - epsmoy)
            else:
                vprev = torq[k]
                v_rel = vel[k + 1] - vel[k]
                ddx = float(np.dot(vprev, v_rel))
                dfs[k] += stif * (dt_val * ddx * (xn - 1.0) / max(EM15, L_tot) - epsmoy)

        # Euler-Eytelwein capstan pulley friction relaxation loop (xforc28.F lines 341-390)
        echange = 1
        nit = 0
        while echange == 1 and nit < 10:
            echange = 0
            nit += 1
            stifr = dfs.copy()
            # Loop over interior nodes K (in Fortran K=2..NX-1, corresponding to pulley K)
            # In 0-based indexing: interior node k from 1 to nx - 2.
            # Segment before is k - 1, segment after is k.
            for k in range(1, nx - 1):
                # Pulley index in 1-based convention is k + 1
                pulley_num = k + 1
                strand_prev_num = k      # segment k-1 (1-based index k)
                strand_next_num = k + 1  # segment k   (1-based index k+1)

                fric = mu1
                if pulley_num in pulley_fric and pulley_fric[pulley_num] >= 0.0:
                    fric = pulley_fric[pulley_num]

                if strand_prev_num in strand_fric and strand_fric[strand_prev_num] >= 0.0:
                    fric += 0.5 * strand_fric[strand_prev_num]
                else:
                    fric += 0.5 * mu2

                if strand_next_num in strand_fric and strand_fric[strand_next_num] >= 0.0:
                    fric += 0.5 * strand_fric[strand_next_num]
                else:
                    fric += 0.5 * mu2

                # Direction vectors incoming to interior node k (xforc28.F lines 368-377)
                vprev = torq[k - 1]  # from node k-1 to node k
                vnext = -torq[k]     # from node k+1 to node k
                alpha = float(np.dot(vprev, vnext))
                alpha = min(1.0, max(-1.0, alpha))
                beta = math.pi - math.acos(alpha)

                # Capstan tension limit (xforc28.F lines 379-388)
                ff = 2.0 * fx + stifr[k - 1] + stifr[k]
                fmax = max(0.0, ff * math.tanh(0.5 * fric * beta))
                dfs_diff = stifr[k - 1] - stifr[k]
                if abs(dfs_diff) > fmax:
                    df = math.copysign(abs(dfs_diff) - fmax, dfs_diff)
                    dfs[k - 1] -= 0.5 * df
                    dfs[k] += 0.5 * df
                    echange = 1

        # Return forces on nodes (xforc28.F lines 394-426)
        forc = np.zeros((nx, 3), dtype=np.float64)
        # Base mean tension force
        vprev = torq[0]
        forc[0] = fx * vprev
        for k in range(1, nx - 1):
            vnext = torq[k]
            forc[k] = fx * (vnext - vprev)
            vprev = vnext
        forc[nx - 1] = -fx * vprev

        # Add strand force differences (xforc28.F lines 415-426)
        for k in range(n_strands):
            vstrand = dfs[k] * torq[k]
            forc[k] += vstrand
            forc[k + 1] -= vstrand

        # Scatter internal forces satisfying sum F = 0
        if fint is not None:
            np.add.at(fint, conn_e, forc)

        # Energy accounting: trapezoidal work (xforc28.F lines 437-450)
        fx_old = st["t28_fx_old"][e]
        dl_old = st["t28_dl_old"][e]
        dfs_old = st["t28_dfs_old"][e]

        for k in range(n_strands):
            dl_k = L_k[k] - L0_k[k]
            delta_dl = dl_k - dl_old[k]
            w_k = 0.5 * delta_dl * (fx + fx_old + dfs[k] + dfs_old[k])
            st["t28_eint_k"][e][k] += w_k
            dl_old[k] = dl_k
            dfs_old[k] = dfs[k]

        st["t28_fx_old"][e] = fx
        st["force"][e] = fx
        st["eint"][e] = float(np.sum(st["t28_eint_k"][e]))

        # Critical time step: min(DTK, DTC) per strand (xforc28.F lines 462-502)
        rho = st["t28_rho"][e]
        dte = EP20
        for k in range(n_strands):
            if L_k[k] <= EM15:
                continue
            xm = rho * L0_k[k]
            xkm = stif * float(nx - 1) / max(EM15, L_tot)
            xcm = (f_elast * dgdx + xc) * L_tot / (max(EM15, L_k[k]) * L0_tot)

            xkm = max(EM15, xkm)
            dtk = (math.sqrt(xcm * xcm + xm * xkm) - xcm) / xkm
            dtc = xm / max(EM15, xcm)
            if dtk == 0.0:
                dtk = dtc
            else:
                dtk = min(dtk, dtc)
            dte = min(dte, dtk)

        dt_crit[i_local] = dte

    return dt_crit


# Standard element kernel entry points for direct element group invocation
def init_group(group: Any, model: Any, log: Any) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Initialize a dedicated /PROP/TYPE28 multi-strand cable element group."""
    st = group.state
    n = group.n
    if "off" not in st:
        st["off"] = np.ones(n, dtype=np.float64)

    # Determine stride
    stride = group.conn.shape[1] if hasattr(group, "conn") and hasattr(group.conn, "shape") and group.conn.ndim == 2 else 3
    massn = np.zeros(n * stride, dtype=np.float64)
    inertn = np.zeros(n * stride, dtype=np.float64)

    init_nstrand_type28(group, model, log, idx28=None, massn=massn, inertn=inertn)

    if hasattr(group.conn, "reshape"):
        node_idx = group.conn.reshape(-1)
    else:
        node_idx = np.array([node for elem in group.conn for node in elem], dtype=np.int64)

    valid = (node_idx >= 0)
    return node_idx[valid], massn[valid], (inertn[valid] if inertn.any() else None)


def forces(
    group: Any,
    x: np.ndarray,
    v: Optional[np.ndarray],
    vr: Optional[np.ndarray],
    dt: Optional[float],
    fint: Optional[np.ndarray],
    mint: Optional[np.ndarray],
) -> np.ndarray:
    """Compute forces for a dedicated multi-strand cable element group."""
    return forces_nstrand_type28(group, x, v, vr, dt, fint, mint, idx28=None)
