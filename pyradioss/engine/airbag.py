"""pyradioss.engine.airbag — Monitored volume and airbag thermodynamics.

Fortran source references:
- MONVOL/AIRBAG1 thermodynamics: engine/source/airbag/airbag1.F (lines 35-140)
- MONVOL/AIRBAG1 volume calculation: engine/source/airbag/get_volume_area.F90 (lines 20-95)
- MONVOL/AIRBAG1 nodal pressure forces: engine/source/airbag/volpres.F (lines 88-165)
- MONVOL/GAS ideal gas thermodynamics: engine/source/airbag/volpvg.F (volpvga, lines 29-191)
- MONVOL/PRES prescribed pressure: engine/source/airbag/volpfv.F (volpfv, lines 31-112)
- /LEAK orifice flow & fabric leakage: engine/source/airbag/volpvg.F (volpvgb, lines 277-457),
  engine/source/airbag/fvvent0.F (lines 189-216), starter/source/airbag/hm_read_leak.F (lines 123-174)
"""

from __future__ import annotations

import math
import numpy as np
from pyradioss.model.model import Model


def update_airbag_thermodynamics(
    mv, model: Model, dt: float, current_time: float
) -> None:
    """Update thermodynamic state of MONVOL/AIRBAG1.

    Upstream Fortran: engine/source/airbag/airbag1.F (lines 35-140) and
    engine/source/airbag/monvol0.F (lines 188-196).
    """
    mat = model.materials.get(getattr(mv, "matid", 0))
    if mat is not None:
        r_spec = getattr(mat, "r_spec", 287.05)
        cpa = mat.params.get("CPA", 0.0)
        cpb = mat.params.get("CPB", 0.0)
        cpc = mat.params.get("CPC", 0.0)
    else:
        r_spec = getattr(mv, "r_spec", 287.05)
        cpa = getattr(mv, "cpa", 1004.0)
        cpb = getattr(mv, "cpb", 0.0)
        cpc = getattr(mv, "cpc", 0.0)

    if getattr(mv, "iequil", 0) in (0, 1) or getattr(mv, "mass", None) is None:
        vol_floor = max(getattr(mv, "volume", 0.0), 1e-9)
        t_init = max(getattr(mv, "t_initial", 293.15), 1e-30)
        pext = getattr(mv, "pext", getattr(mv, "p_ext", 0.0))
        if pext > 0.0 and r_spec > 0.0:
            gmi = pext * vol_floor / (r_spec * t_init)
        else:
            rho = getattr(mat, "rho0", 1.0) if mat is not None else 1.0
            gmi = rho * vol_floor
        mv.mass = gmi
        mv.temperature = t_init
        mv.pressure = pext
        mv.volume_old = mv.volume
        mv.work = getattr(mv, "work", 0.0)
        cv_init = cpa + cpb * t_init + cpc * t_init**2 - r_spec
        if cv_init <= 0.0:
            cv_init = max(r_spec, 1.0)
        mv.energy = gmi * cv_init * t_init
        mv.energy_initial = mv.energy
        mv.iequil = -1

    t_bag_old = getattr(mv, "temperature", getattr(mv, "t_initial", 293.15))
    vol_old = getattr(mv, "volume_old", getattr(mv, "volume", 1e-9))
    if vol_old <= 0.0:
        vol_old = max(getattr(mv, "volume", 1e-9), 1e-9)

    vol = max(getattr(mv, "volume", 1e-9), 1e-9)
    dv = vol - vol_old
    p_old = getattr(mv, "pressure", getattr(mv, "pext", 0.0))

    gmi = getattr(mv, "mass", 0.0)

    cvi = cpa + cpb * t_bag_old + cpc * t_bag_old**2 - r_spec
    if cvi <= 0.0:
        cvi = max(r_spec, 1.0)

    left = gmi * cvi
    right = gmi * cvi * t_bag_old

    rnm_old = gmi * r_spec
    rnm = gmi * r_spec

    left += 0.5 * rnm * dv / vol
    right -= 0.5 * rnm_old * t_bag_old * dv / vol_old

    if left <= 1e-20 * gmi * cvi:
        # fallback to adiabatic step
        gamma = (cvi + r_spec) / cvi if cvi > 0 else 1.4
        t_bag = t_bag_old * (max(vol_old, 1e-9) / max(vol, 1e-9)) ** (gamma - 1.0)
    elif abs(left) > 1e-30:
        t_bag = right / left
    else:
        t_bag = t_bag_old
    t_bag = max(t_bag, 0.0)

    pext_val = getattr(mv, "pext", getattr(mv, "p_ext", 0.0))
    if getattr(mv, "volume", 0.0) <= 1e-9:
        p = pext_val
        t_bag = getattr(mv, "t_initial", 293.15)
    else:
        p = rnm * t_bag / vol if vol > 0.0 else pext_val

    # Book work in external/volume work accumulator: dW = 0.5*(P + P_old)*dV
    dw = 0.5 * (p + p_old) * dv
    mv.work = getattr(mv, "work", 0.0) + dw

    # Update state and internal energy (1st law: dE = -dW)
    cv_new = cpa + cpb * t_bag + cpc * t_bag**2 - r_spec
    if cv_new <= 0.0:
        cv_new = max(r_spec, 1.0)
    mv.energy = gmi * cv_new * t_bag

    mv.temperature = t_bag
    mv.pressure = p
    mv.volume_old = mv.volume


def update_airbag_volume(mv, model: Model, x: np.ndarray) -> None:
    """Recompute volume based on current nodal coordinates.

    Upstream Fortran: engine/source/airbag/get_volume_area.F90 (lines 20-95)
    and engine/source/airbag/monvol0.F (lines 125-133).
    """
    surf = model.surfaces.get(mv.surf_id) if hasattr(model, "surfaces") else None
    if surf is None or surf.segments is None or len(surf.segments) == 0:
        if getattr(mv, "volume", None) is None:
            mv.volume = 0.0
        return

    total_vol = 0.0
    for segment_nodes in surf.segments:
        n1 = segment_nodes[0]
        n2 = segment_nodes[1]
        n3 = segment_nodes[2]
        is_tri = (
            (len(segment_nodes) < 4)
            or (segment_nodes[3] == n3)
            or (segment_nodes[3] < 0)
        )

        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]

        v31 = x3 - x1
        if not is_tri:
            n4 = segment_nodes[3]
            x4 = x[n4]
            v42 = x4 - x2
            xn = 0.5 * np.cross(v31, v42)
            vol_cont = np.dot(x1 + x2 + x3 + x4, xn) / 12.0
        else:
            v21 = x2 - x1
            xn = 0.5 * np.cross(v21, v31)
            vol_cont = np.dot(x1 + x2 + x3, xn) / 9.0

        total_vol += vol_cont

    mv.volume = total_vol


def apply_airbag_forces(
    mv, model: Model, x: np.ndarray, f: np.ndarray
) -> None:
    """Apply pressure forces to FE nodal force array.

    Upstream Fortran: engine/source/airbag/volpres.F (volpre, lines 88-165).
    """
    surf = model.surfaces.get(mv.surf_id) if hasattr(model, "surfaces") else None
    if surf is None or surf.segments is None or len(surf.segments) == 0:
        return

    pext_val = getattr(mv, "pext", getattr(mv, "p_ext", 0.0))
    p_eff = getattr(mv, "pressure", 0.0) - pext_val

    for segment_nodes in surf.segments:
        n1 = segment_nodes[0]
        n2 = segment_nodes[1]
        n3 = segment_nodes[2]
        is_tri = (
            (len(segment_nodes) < 4)
            or (segment_nodes[3] == n3)
            or (segment_nodes[3] < 0)
        )

        x1 = x[n1]
        x2 = x[n2]
        x3 = x[n3]

        v31 = x3 - x1
        if not is_tri:
            n4 = segment_nodes[3]
            x4 = x[n4]
            v42 = x4 - x2
            xn = 0.5 * np.cross(v31, v42)

            fx = p_eff * xn[0] / 4.0
            fy = p_eff * xn[1] / 4.0
            fz = p_eff * xn[2] / 4.0

            f[n1, 0] += fx; f[n1, 1] += fy; f[n1, 2] += fz
            f[n2, 0] += fx; f[n2, 1] += fy; f[n2, 2] += fz
            f[n3, 0] += fx; f[n3, 1] += fy; f[n3, 2] += fz
            f[n4, 0] += fx; f[n4, 1] += fy; f[n4, 2] += fz
        else:
            v21 = x2 - x1
            xn = 0.5 * np.cross(v21, v31)

            fx = p_eff * xn[0] / 3.0
            fy = p_eff * xn[1] / 3.0
            fz = p_eff * xn[2] / 3.0

            f[n1, 0] += fx; f[n1, 1] += fy; f[n1, 2] += fz
            f[n2, 0] += fx; f[n2, 1] += fy; f[n2, 2] += fz
            f[n3, 0] += fx; f[n3, 1] += fy; f[n3, 2] += fz


def update_monvol_gas(
    mv, model: Model, dt: float, current_time: float
) -> None:
    """Update thermodynamic state of MONVOL/GAS using ideal gas law (p*V = m*R*T).

    Upstream Fortran: engine/source/airbag/volpvg.F (volpvga, lines 29-191).
    Supports:
    - Isothermal expansion/compression: T = const, p*V = m*R*T = const
      (volpvga.F lines 122-130).
    - Isentropic closed-form: T*V^(gamma-1) = const, p*V^gamma = const
      (volpvga.F lines 131-139).
    - Crank-Nicolson adiabatic energy integration: dE = -p*dV
      (volpvga.F lines 111-121).
    """
    vinc = getattr(mv, "vinc", 0.0)
    gamma = getattr(mv, "gamma", 1.4)
    if gamma <= 1.0:
        gamma = 1.4

    # 1. Initialization on first step
    if not getattr(mv, "_initialized", False) or getattr(mv, "mass", None) is None:
        v_curr = getattr(mv, "volume", 0.0)
        if v_curr <= 0.0:
            update_airbag_volume(mv, model, getattr(model, "x", getattr(model, "x0", np.zeros((1, 3)))))
            v_curr = getattr(mv, "volume", 0.0)
        v0 = max(v_curr, 1e-9)
        v_eff0 = max(v0 - vinc, 1e-9)

        t0 = getattr(mv, "tini", getattr(mv, "t_initial", 293.15))

        # Gas constant R and specific heat cv
        r_spec = getattr(mv, "r_spec", None)
        if r_spec is None:
            mat = (
                model.materials.get(getattr(mv, "matid", 0))
                if hasattr(model, "materials")
                else None
            )
            if mat is not None and getattr(mat, "r_spec", 0.0) > 0.0:
                r_spec = mat.r_spec
            elif getattr(mv, "pini", 0.0) > 0.0 and getattr(mv, "rho_gas", 0.0) > 0.0 and t0 > 0.0:
                r_spec = mv.pini / (mv.rho_gas * t0)
            elif getattr(mv, "pext", 0.0) > 0.0 and getattr(mv, "rho_gas", 0.0) > 0.0 and t0 > 0.0:
                r_spec = mv.pext / (mv.rho_gas * t0)
            else:
                r_spec = 287.05  # J/(kg*K) standard for air

        cv = r_spec / (gamma - 1.0)

        # Initial mass
        m0 = getattr(mv, "mini", 0.0)
        if m0 <= 0.0:
            rho0 = getattr(mv, "rho_gas", 0.0)
            if rho0 > 0.0:
                m0 = rho0 * v_eff0
            elif getattr(mv, "pini", 0.0) > 0.0:
                m0 = mv.pini * v_eff0 / (r_spec * t0)
            else:
                m0 = 1.2 * v_eff0

        mv.mass = m0
        mv.m0 = m0
        mv.r_spec = r_spec
        mv.cv = cv
        mv.temperature = t0
        mv.t0 = t0
        mv.volume0 = v0
        mv.volume_old = v0
        mv.energy = m0 * cv * t0
        mv.energy_initial = mv.energy
        mv.work = getattr(mv, "work", 0.0)
        mv.pmax = getattr(mv, "pmax", 1e30)

        if getattr(mv, "pini", 0.0) > 0.0:
            p_init = mv.pini
        else:
            p_init = (m0 * r_spec * t0) / v_eff0
        mv.pressure = p_init
        mv.p_old = p_init
        mv._initialized = True

    # 2. Geometry & Increments
    v_curr = getattr(mv, "volume", 1e-9)
    v_eff = max(v_curr - vinc, 1e-9)

    vol_old = getattr(mv, "volume_old", v_curr)
    v_eff_old = max(vol_old - vinc, 1e-9)
    dv = v_curr - vol_old

    r_spec = mv.r_spec
    cv = mv.cv
    m = getattr(mv, "mass", mv.m0)
    t_old = getattr(mv, "temperature", mv.t0)
    p_old = getattr(mv, "pressure", mv.p_old)
    e_old = getattr(mv, "energy", m * cv * t_old)

    iequil = getattr(mv, "iequil", 0)
    isothermal = getattr(mv, "isothermal", False) or (iequil == 1)
    isentropic = getattr(mv, "isentropic", False) or (iequil == 2)

    # 3. Thermodynamic State Integration
    if isothermal:
        # Isothermal: T = const -> p*V = m*R*T (volpvga.F lines 122-130)
        t = t_old
        p = (m * r_spec * t) / v_eff
        energy = m * cv * t
        dw = 0.5 * (p + p_old) * dv
        mv.work = getattr(mv, "work", 0.0) + dw
    elif isentropic:
        # Isentropic: T*V^(gamma-1) = const, p*V^gamma = const (volpvga.F lines 131-139)
        ratio = v_eff_old / v_eff
        t = t_old * (ratio ** (gamma - 1.0))
        p = (m * r_spec * t) / v_eff
        energy = m * cv * t
        dw = 0.5 * (p + p_old) * dv
        mv.work = getattr(mv, "work", 0.0) + dw
    else:
        # Crank-Nicolson adiabatic 1st law integration (volpvga.F lines 111-121):
        # E = ((1 - FAC / V_old)*E_old + (dE_in - dE_out)*dt) / (1 + FAC / V_new)
        fac = 0.5 * (gamma - 1.0) * dv
        de_in = getattr(mv, "de_in", 0.0)
        de_out = getattr(mv, "de_out", 0.0)

        num = (1.0 - fac / v_eff_old) * e_old + (de_in - de_out) * dt
        denom = 1.0 + fac / v_eff
        energy = max(num / denom, 0.0)

        p = (gamma - 1.0) * energy / v_eff
        t = energy / (cv * max(m, 1e-12))
        dw = 0.5 * (p + p_old) * dv
        mv.work = getattr(mv, "work", 0.0) + dw

    # Burst pressure check (volpvga.F line 154)
    pmax = getattr(mv, "pmax", 1e30)
    pext = getattr(mv, "pext", getattr(mv, "p_ext", 0.0))
    if pmax > 0.0 and p > pmax:
        p = pext

    mv.pressure = p
    mv.temperature = t
    mv.energy = energy
    mv.p_old = p
    mv.volume_old = v_curr


def update_monvol_pres(
    mv, model: Model, dt: float, current_time: float
) -> None:
    """Update pressure for /MONVOL/PRES from prescribed curve.

    Upstream Fortran: engine/source/airbag/volpfv.F (volpfv, lines 31-112).
    Evaluates P(t) from function fct_id with scale factor fscale.
    Accumulates volume change work: dW = 0.5*(P + P_old)*dV (volpfv.F line 92).
    """
    fscale = getattr(mv, "fscale", 1.0)
    p_ext = getattr(mv, "p_ext", getattr(mv, "pext", 0.0))
    mv.pext = p_ext

    v_curr = getattr(mv, "volume", 0.0)
    vol_old = getattr(mv, "volume_old", v_curr)
    dv = v_curr - vol_old
    p_old = getattr(mv, "pressure", p_ext)

    fct_id = getattr(mv, "fct_id", 0)
    itypfun = getattr(mv, "itypfun", 1)  # Default: function of time (volpfv.F line 78)

    v0 = getattr(mv, "v0", getattr(mv, "volume0", vol_old if vol_old > 0 else v_curr))
    vinc = getattr(mv, "vinc", 0.0)
    v_eff = max(v_curr - vinc, 1e-12)
    v0_eff = max(v0 - vinc, 1e-12)

    if itypfun == 0:
        xfun = v0_eff / v_eff
    elif itypfun == 2:
        xfun = v_eff / v0_eff
    else:
        xfun = current_time

    p_val = 0.0
    if fct_id > 0 and hasattr(model, "functions") and fct_id in model.functions:
        func = model.functions[fct_id]
        p_val = float(func.eval(xfun))
    elif hasattr(mv, "pressure_curve") and callable(mv.pressure_curve):
        p_val = float(mv.pressure_curve(xfun))
    elif getattr(mv, "pres_val", None) is not None:
        p_val = float(mv.pres_val)
    else:
        p_val = p_old

    pres = fscale * p_val
    if itypfun == 3:
        pres = pres * (v0_eff / v_eff)

    # Accumulate p*dV work (volpfv.F line 92: WFEXT = WFEXT + HALF*(PRES+POLD)*DV)
    dw = 0.5 * (pres + p_old) * dv
    mv.work = getattr(mv, "work", 0.0) + dw

    mv.pressure = pres
    mv.volume_old = v_curr


def apply_leak_flow(
    mv, model: Model, dt: float, current_time: float
) -> float:
    """Calculate and apply /LEAK mass outflow rate from monitored volume.

    Upstream Fortran:
    - engine/source/airbag/volpvg.F (volpvgb, lines 277-457): Isentropic orifice flow.
    - engine/source/airbag/fvvent0.F (lines 189-216): Fabric leakage formulation.
    - starter/source/airbag/hm_read_leak.F (lines 123-174): /LEAK parameters.

    Returns:
        dm_out (float): Mass decrease during time step dt.
    """
    dt = max(float(dt), 1e-12)
    m = getattr(mv, "mass", 0.0)
    if m <= 0.0:
        mv.dm_leak = 0.0
        mv.mass_flow_out = 0.0
        return 0.0

    p = getattr(mv, "pressure", 0.0)
    pext = getattr(mv, "pext", getattr(mv, "p_ext", 0.0))
    if p <= pext:
        mv.dm_leak = 0.0
        mv.mass_flow_out = 0.0
        return 0.0

    # 1. Determine effective leakage area A_leak
    a_leak = getattr(mv, "leak_area", 0.0)
    leak_rate_const = getattr(mv, "leak_rate", None)

    if a_leak <= 0.0 and leak_rate_const is None:
        # Check /LEAK models in model
        leak_mat = None
        leak_id = getattr(mv, "leak_id", getattr(mv, "matid", 0))
        if hasattr(model, "leak_mats") and leak_id in model.leak_mats:
            leak_mat = model.leak_mats[leak_id]
        elif hasattr(model, "leak_parts") and getattr(mv, "part_id", 0) in model.leak_parts:
            leak_mat = model.leak_parts[mv.part_id]
        elif hasattr(model, "leak_areas") and getattr(mv, "id", 0) in model.leak_areas:
            leak_mat = model.leak_areas[mv.id]

        if leak_mat is not None:
            # fvvent0.F lines 189-216: SVTFAC factor
            ileak = leak_mat.ileakage
            if ileak == 1:
                flc = leak_mat.bcoeft1 if leak_mat.bcoeft1 > 0.0 else 1.0
                fac = leak_mat.acoeft2 if leak_mat.acoeft2 > 0.0 else (
                    leak_mat.acoeft1 if leak_mat.acoeft1 > 0.0 else 1.0
                )
                svtfac = flc * fac
            elif ileak in (2, 3):
                flc = 1.0
                if leak_mat.fct_id_lc > 0 and hasattr(model, "functions") and leak_mat.fct_id_lc in model.functions:
                    flc = leak_mat.fscale_lc * float(
                        model.functions[leak_mat.fct_id_lc].eval(current_time * leak_mat.scale_t)
                    )
                fac = 1.0
                if leak_mat.fct_id_ac > 0 and hasattr(model, "functions") and leak_mat.fct_id_ac in model.functions:
                    dp = p if ileak == 2 else (p - pext)
                    fac = leak_mat.fscale_ac * float(
                        model.functions[leak_mat.fct_id_ac].eval(dp * leak_mat.scale_p)
                    )
                svtfac = flc * fac
            else:
                svtfac = getattr(leak_mat, "acoeft1", 0.0)

            surf_area = getattr(mv, "area", 1.0)
            a_leak = surf_area * svtfac

    # 2. Compute mass outflow rate
    if leak_rate_const is not None:
        m_dot = float(leak_rate_const)
    elif a_leak > 0.0:
        # Isentropic orifice nozzle flow (volpvgb in volpvg.F lines 441-452)
        v_eff = max(getattr(mv, "volume", 1.0) - getattr(mv, "vinc", 0.0), 1e-9)
        rho = m / v_eff
        gamma = getattr(mv, "gamma", 1.4)
        if gamma <= 1.0:
            gamma = 1.4

        # Critical sonic choking pressure: pcrit = p * (2 / (gamma + 1))**(gamma / (gamma - 1))
        pcrit = p * (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        pe_eff = max(pext, pcrit)

        # Isentropic exit velocity: u = sqrt(2*gamma/(gamma - 1) * p / rho * (1 - (pe/p)**((gamma - 1)/gamma)))
        press_ratio = pe_eff / p
        u_sq = 2.0 * gamma / (gamma - 1.0) * (p / rho) * (1.0 - press_ratio ** ((gamma - 1.0) / gamma))
        u = math.sqrt(max(0.0, u_sq))

        # Exit density: rho_e = rho * (pe/p)**(1 / gamma)
        rho_e = rho * (press_ratio ** (1.0 / gamma))

        m_dot = a_leak * rho_e * u
    else:
        m_dot = 0.0

    # 3. Mass balance decrement
    dm = min(m_dot * dt, m)
    mv.mass = m - dm
    mv.dm_leak = dm
    mv.mass_flow_out = m_dot

    # Deduct internal energy carried out by mass flow
    if dm > 0.0 and getattr(mv, "energy", None) is not None:
        cv = getattr(mv, "cv", 717.6)
        temp = getattr(mv, "temperature", 293.15)
        gamma = getattr(mv, "gamma", 1.4)
        cp = gamma * cv
        de_out = dm * cp * temp
        mv.energy = max(0.0, mv.energy - de_out)
        mv.de_out = getattr(mv, "de_out", 0.0) + de_out / dt

    return dm


# =======================================================================
# /MONVOL/COMMU1 — Communicating airbag chambers (re-exported)
# =======================================================================
from pyradioss.engine.airbag_commu import (  # noqa: E402
    MonvolCommu1,
    CommuStepResult,
    critical_pressure_ratio,
    choked_flow_function,
    subsonic_flow_function,
    flow_function,
    compute_orifice_flow,
    step_airbag_commu,
    apply_airbag_communications,
)
