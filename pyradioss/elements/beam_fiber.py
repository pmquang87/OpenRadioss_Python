"""
Integrated Fiber Beam Element (/BEAM + /PROP/TYPE18, /PROP/INT_BEAM).

Fortran origin: ``engine/source/elements/beam/main_beam18.F``,
``engine/source/elements/beam/mulaw_ib.F``,
``engine/source/materials/mat/mat002/m2lawpi.F``,
``engine/source/elements/beam/fail_beam18.F``,
``starter/source/properties/beam/hm_read_prop18.F``,
``starter/source/properties/beam/defbeam_sect_new.F90``.

Unlike the global-resultant TYPE3 beam (/PROP/BEAM), the TYPE18 integrated
fiber beam discretizes the 2D cross-section into NIP fiber integration points
(y_k, z_k) with tributary areas A_k. At each fiber point:

1. Kinematics (corotational Timoshenko beam):
   - Axial strain: eps_xx(y, z) = eps_0 - y * kappa_z + z * kappa_y
   - Transverse shear: gamma_xy, gamma_xz with shear factor ks = 5/6
   - Torsional twist: kappa_x

2. Constitutive evaluation:
   - 1D/2D elastoplastic stress update (LAW1 elastic, LAW2 plastic)
   - von Mises equivalent stress: sigma_vm = sqrt(sigma_xx^2 + 3*(sigma_xy^2 + sigma_xz^2))
   - Radial return consistency solve with Johnson-Cook hardening
   - Fiber-level plastic strain accumulation

3. Section resultant force and moment integration (main_beam18.F):
   - N  = sum_k (sigma_xx,k * A_k)
   - Vy = sum_k (tau_xy,k   * A_k)
   - Vz = sum_k (tau_xz,k   * A_k)
   - Mx = sum_k (tau_xy,k * z_k - tau_xz,k * y_k) * A_k
   - My = sum_k (sigma_xx,k * z_k * A_k)
   - Mz = -sum_k (sigma_xx,k * y_k * A_k)

4. Internal nodal force and moment assembly (pfint3.F):
   - f1 = (-N, -Vy, -Vz), m1 = (-Mx, -My + Vz*L/2, -Mz - Vy*L/2)
   - f2 = ( N,  Vy,  Vz), m2 = ( Mx,  My + Vz*L/2,  Mz - Vy*L/2)
   - Transformed to global frame via corotational triad E = [e1, e2, e3]

5. Critical time step:
   - dt = L / c with c = sqrt(E / rho0)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, norm3


# ----------------------------------------------------------------------------
# Cross-section geometry & fiber discretization (hm_read_prop18.F, defbeam_sect_new.F90)
# ----------------------------------------------------------------------------

def generate_fiber_section(
    isflag: int,
    nitrs: int = 3,
    l_params: Optional[List[float]] = None,
    nip: int = 0,
    user_fibers: Optional[List[Tuple[float, float, float]]] = None,
    iref: int = 0,
    y0: float = 0.0,
    z0: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate integration points (y_k, z_k) and weights A_k for cross-section.

    Parameters:
        isflag: section type flag:
            0: user-defined integration points
            1: rectangular section (Gauss integration rule)
            2: circular section (Gauss / concentric rings)
            3: rectangular section (Lobatto integration rule)
            10: I-shape section
            17: circular section (concentric rings, defbeam_sect_new Case 17)
            18: tubular section (defbeam_sect_new Case 18)
        nitrs: integration order / subdivision parameter
        l_params: section dimensions [L1, L2, L3, L4, L5, L6]
        nip: number of user integration points (for isflag == 0)
        user_fibers: list of (y, z, area) for isflag == 0
        iref: 0 = barycenter calculation, 1 = user-specified (y0, z0)
        y0, z0: section center offsets

    Returns:
        (y_pts, z_pts, area_pts, props)
        where props contains: area, iyy, izz, ixx, zy, zz, y0, z0.
    """
    if l_params is None:
        l_params = [0.0] * 6
    else:
        l_params = list(l_params) + [0.0] * max(0, 6 - len(l_params))

    y_list: List[float] = []
    z_list: List[float] = []
    a_list: List[float] = []

    if isflag == 0:
        # User-defined integration points
        if user_fibers:
            for y_i, z_i, a_i in user_fibers:
                y_list.append(float(y_i))
                z_list.append(float(z_i))
                a_list.append(float(a_i))
        tot_a = sum(a_list)
        if iref == 0 and tot_a > 0.0:
            calc_y0 = sum(y * a for y, a in zip(y_list, a_list)) / tot_a
            calc_z0 = sum(z * a for z, a in zip(z_list, a_list)) / tot_a
        else:
            calc_y0 = float(y0)
            calc_z0 = float(z0)

        y_arr = np.array(y_list, dtype=float) - calc_y0
        z_arr = np.array(z_list, dtype=float) - calc_z0
        a_arr = np.array(a_list, dtype=float)

    elif isflag in (1, 3):
        # Rectangular section: L1 = width (y), L2 = height (z)
        b = float(l_params[0]) if l_params[0] > 0 else 1.0
        h = float(l_params[1]) if l_params[1] > 0 else b
        order = max(1, min(9, int(nitrs) if nitrs > 0 else 3))

        if isflag == 1:
            # Gauss-Legendre quadrature
            pts, wts = np.polynomial.legendre.leggauss(order)
        else:
            # Gauss-Lobatto quadrature: endpoints included
            if order == 1:
                pts, wts = np.array([0.0]), np.array([2.0])
            else:
                pts_internal = np.polynomial.legendre.Legendre.basis(order - 1).deriv().roots()
                pts = np.sort(np.concatenate(([-1.0], pts_internal, [1.0])))
                p_n1 = np.polynomial.legendre.legval(pts, np.eye(order)[order - 1])
                wts = 2.0 / ((order * (order - 1)) * p_n1**2)

        area_total = b * h
        for i in range(order):
            y_i = pts[i] * (b * 0.5)
            w_i = wts[i]
            for j in range(order):
                z_j = pts[j] * (h * 0.5)
                w_j = wts[j]
                a_ij = w_i * w_j * (area_total * 0.25)
                y_list.append(float(y_i))
                z_list.append(float(z_j))
                a_list.append(float(a_ij))

        y_arr = np.array(y_list, dtype=float)
        z_arr = np.array(z_list, dtype=float)
        a_arr = np.array(a_list, dtype=float)
        calc_y0, calc_z0 = 0.0, 0.0

    elif isflag in (2, 17):
        # Circular section: L1 = radius R
        R = float(l_params[0]) if l_params[0] > 0 else 1.0
        order = max(1, min(10, int(nitrs) if nitrs > 0 else 2))
        nr = order + 3
        nphi = 4 * order + 12
        dr = R / nr
        dphi = 2.0 * math.pi / nphi
        for j in range(1, nr + 1):
            r_sup = j * dr
            r_inf = (j - 1) * dr
            r_mid = 0.5 * (r_sup + r_inf)
            area_ring_cell = math.pi * (r_sup**2 - r_inf**2) / nphi
            phi_0 = 0.5 * dphi
            for i in range(nphi):
                phi = phi_0 + i * dphi
                y_list.append(float(r_mid * math.cos(phi)))
                z_list.append(float(r_mid * math.sin(phi)))
                a_list.append(float(area_ring_cell))

        y_arr = np.array(y_list, dtype=float)
        z_arr = np.array(z_list, dtype=float)
        a_arr = np.array(a_list, dtype=float)
        calc_y0, calc_z0 = 0.0, 0.0

    elif isflag == 18:
        # Tubular section: L1 = R_outer, L2 = R_inner
        Ro = float(l_params[0]) if l_params[0] > 0 else 1.0
        Ri = float(l_params[1]) if l_params[1] > 0 else 0.5 * Ro
        order = max(1, min(10, int(nitrs) if nitrs > 0 else 2))
        nr = order + 3
        nphi = 4 * order + 12
        dr = (Ro - Ri) / nr
        dphi = 2.0 * math.pi / nphi
        for j in range(1, nr + 1):
            r_sup = Ri + j * dr
            r_inf = Ri + (j - 1) * dr
            r_mid = 0.5 * (r_sup + r_inf)
            area_ring_cell = math.pi * (r_sup**2 - r_inf**2) / nphi
            phi_0 = 0.5 * dphi
            for i in range(nphi):
                phi = phi_0 + i * dphi
                y_list.append(float(r_mid * math.cos(phi)))
                z_list.append(float(r_mid * math.sin(phi)))
                a_list.append(float(area_ring_cell))

        y_arr = np.array(y_list, dtype=float)
        z_arr = np.array(z_list, dtype=float)
        a_arr = np.array(a_list, dtype=float)
        calc_y0, calc_z0 = 0.0, 0.0

    elif isflag == 10:
        # I-beam section (defbeam_sect_new.F90 Case 10)
        bf = float(l_params[0]) if l_params[0] > 0 else 10.0
        tf = float(l_params[1]) if l_params[1] > 0 else 1.0
        h = float(l_params[2]) if l_params[2] > 0 else 20.0
        tw = float(l_params[3]) if l_params[3] > 0 else 1.0
        intr = max(1, min(15, int(nitrs) if nitrs > 0 else 2))
        n_seg = 2 * intr + 3
        fac = 1.0 / n_seg

        area_flange_pt = bf * tf * fac
        area_web_pt = tw * (h - 2.0 * tf) * fac
        dy1 = bf * fac
        y1_0 = -0.5 * bf + 0.5 * dy1
        dz2 = -(h - 2.0 * tf) * fac
        z2_0 = 0.5 * h - tf + 0.5 * dz2

        # Top flange
        for i in range(n_seg):
            y_list.append(y1_0 + i * dy1)
            z_list.append(0.5 * (h - tf))
            a_list.append(area_flange_pt)
        # Bottom flange
        for i in range(n_seg):
            y_list.append(y1_0 + i * dy1)
            z_list.append(-0.5 * (h - tf))
            a_list.append(area_flange_pt)
        # Web
        for i in range(n_seg):
            y_list.append(0.0)
            z_list.append(z2_0 + i * dz2)
            a_list.append(area_web_pt)

        y_arr = np.array(y_list, dtype=float)
        z_arr = np.array(z_list, dtype=float)
        a_arr = np.array(a_list, dtype=float)
        calc_y0, calc_z0 = 0.0, 0.0

    else:
        # Fallback: single point at origin
        y_arr = np.array([0.0], dtype=float)
        z_arr = np.array([0.0], dtype=float)
        a_arr = np.array([1.0], dtype=float)
        calc_y0, calc_z0 = 0.0, 0.0

    # Section properties
    tot_area = float(np.sum(a_arr))
    iyy = float(np.sum(z_arr**2 * a_arr))
    izz = float(np.sum(y_arr**2 * a_arr))
    ixx = iyy + izz

    zy = float(np.sum(np.abs(z_arr) * a_arr))
    zz = float(np.sum(np.abs(y_arr) * a_arr))

    props = {
        "area": tot_area,
        "iyy": iyy,
        "izz": izz,
        "ixx": ixx,
        "zy": zy,
        "zz": zz,
        "y0": calc_y0,
        "z0": calc_z0,
        "nip": len(y_arr),
    }
    return y_arr, z_arr, a_arr, props


# ----------------------------------------------------------------------------
# Geometry: corotational frame (pcoor3.F)
# ----------------------------------------------------------------------------

def _frame(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Local triad from the two end nodes and the orientation node.

    Returns (E (n,3,3) with columns e1|e2|e3, L (n,))."""
    d = x2 - x1
    n = len(d)
    if n == 0:
        return np.zeros((0, 3, 3)), np.zeros(0)
    nd = norm3(d)
    deg_d = nd <= EM20
    L = np.maximum(nd, EM20)
    e1 = d / L[:, None]
    yref = x3 - x1                                # local y lies in (e1, yref)
    dot = np.einsum("nb,nb->n", yref, e1)
    e2 = yref - dot[:, None] * e1
    ne2 = norm3(e2)
    deg = (ne2 <= 1e-12) | deg_d
    if np.any(deg):
        # Fallback reference vector for collinear/degenerate orientation nodes
        for i in np.where(deg)[0]:
            e1_i = e1[i]
            ref = np.array([1.0, 0.0, 0.0]) if abs(e1_i[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e2_cand = ref - np.dot(ref, e1_i) * e1_i
            ne2_cand = np.linalg.norm(e2_cand)
            if ne2_cand > 1e-12:
                e2[i] = e2_cand / ne2_cand
                ne2[i] = 1.0
            else:
                e2[i] = np.array([0.0, 1.0, 0.0]) if abs(e1_i[1]) < 0.9 else np.array([0.0, 0.0, 1.0])
                ne2[i] = 1.0
    e2 /= np.maximum(ne2, EM20)[:, None]
    e3 = cross3(e1, e2)
    if np.any(deg_d):
        for i in np.where(deg_d)[0]:
            e1[i] = np.array([1.0, 0.0, 0.0])
            e2[i] = np.array([0.0, 1.0, 0.0])
            e3[i] = np.array([0.0, 0.0, 1.0])
    return np.stack([e1, e2, e3], axis=2), L


def _b_operator(L: float) -> np.ndarray:
    """The 6x12 generalized-strain-rate operator, local dof order
    (v1x v1y v1z th1x th1y th1z v2x ... th2z)."""
    B = np.zeros((6, 12))
    L_safe = max(float(L), EM20)
    B[0, 0], B[0, 6] = -1 / L_safe, 1 / L_safe                       # eps
    B[1, 1], B[1, 7] = -1 / L_safe, 1 / L_safe                       # gy
    B[1, 5], B[1, 11] = -0.5, -0.5
    B[2, 2], B[2, 8] = -1 / L_safe, 1 / L_safe                       # gz
    B[2, 4], B[2, 10] = 0.5, 0.5
    B[3, 3], B[3, 9] = -1 / L_safe, 1 / L_safe                       # kx (twist)
    B[4, 4], B[4, 10] = -1 / L_safe, 1 / L_safe                      # ky
    B[5, 5], B[5, 11] = -1 / L_safe, 1 / L_safe                      # kz
    return B


# ----------------------------------------------------------------------------
# Starter-side initialization (init_group)
# ----------------------------------------------------------------------------

def init_group(group: Any, model: Any, log: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize element buffer, fiber coordinates, lumped mass and inertia.

    Fortran origin: starter hm_read_prop18.F + engine pinit3.F / pmass3.F.
    Only nodes N1 and N2 receive mass; N3 is orientation only.
    """
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)

    conn = group.conn
    x1, x2, x3 = model.x0[conn[:, 0]], model.x0[conn[:, 1]], model.x0[conn[:, 2]]
    E, L0 = _frame(x1, x2, x3)

    dx = np.linalg.norm(x2 - x1, axis=1)
    if np.any(dx <= 0):
        for eid in group.ids[dx <= 0]:
            log.error(f"/BEAM {eid}: zero length", "BEAM INIT")

    n = group.n
    area = np.zeros(n)
    rho0 = np.zeros(n)
    igyr = np.zeros(n)
    dt0 = np.zeros(n)

    # Per-element fiber structures
    elem_fibers_y: List[np.ndarray] = []
    elem_fibers_z: List[np.ndarray] = []
    elem_fibers_a: List[np.ndarray] = []

    slices = group.state.get("slices", [])
    for sl, mat, prop in slices:
        # Extract property parameters
        p_params = getattr(prop, "params", {})
        isflag = p_params.get("isflag", getattr(prop, "isflag", 0))
        nitrs = p_params.get("nitrs", getattr(prop, "nitrs", 3))
        l_params = p_params.get("l_params", [
            getattr(prop, "l1", 0.0), getattr(prop, "l2", 0.0),
            getattr(prop, "l3", 0.0), getattr(prop, "l4", 0.0),
            getattr(prop, "l5", 0.0), getattr(prop, "l6", 0.0)
        ])
        user_fib = p_params.get("fibers", None)
        if user_fib is None and hasattr(prop, "ips") and prop.ips:
            user_fib = [(ip.y, ip.z, ip.area) for ip in prop.ips]

        iref = p_params.get("iref", getattr(prop, "iref", 0))
        y0 = p_params.get("y0", getattr(prop, "y0", 0.0))
        z0 = p_params.get("z0", getattr(prop, "z0", 0.0))

        # Generate fibers
        y_pts, z_pts, a_pts, sec_props = generate_fiber_section(
            isflag=isflag,
            nitrs=nitrs,
            l_params=l_params,
            nip=len(user_fib) if user_fib else 0,
            user_fibers=user_fib,
            iref=iref,
            y0=y0,
            z0=z0,
        )

        n_sl = sl.stop - sl.start
        for _ in range(n_sl):
            elem_fibers_y.append(y_pts.copy())
            elem_fibers_z.append(z_pts.copy())
            elem_fibers_a.append(a_pts.copy())

        a_val = sec_props["area"]
        iyy_val = sec_props["iyy"]
        izz_val = sec_props["izz"]

        area[sl] = a_val
        rho_val = getattr(mat, "rho0", 0.0)
        rho0[sl] = rho_val
        igyr[sl] = (iyy_val + izz_val) / max(a_val, EM20)

        # Critical wave speed c = sqrt(E / rho0)
        E_mod = getattr(mat, "E", 0.0)
        c_sound = math.sqrt(max(E_mod / max(rho_val, EM20), EM20)) if rho_val > 0 and E_mod > 0 else 1.0
        dt0[sl] = L0[sl] / c_sound

    mass = rho0 * area * L0
    # Lumped rotary inertia with Key's boost: m_i * (L^2/12 + (Iyy+Izz)/A)
    inertia_c = mass / 2.0 * (L0**2 / 12.0 + igyr)

    # Initialize fiber stress state buffers:
    # sig_xx: (N_elem, N_ip)
    # sig_xy: (N_elem, N_ip)
    # sig_xz: (N_elem, N_ip)
    # eps_p:  (N_elem, N_ip)
    max_nip = max((len(y) for y in elem_fibers_y), default=1)
    sig_xx = np.zeros((n, max_nip), dtype=float)
    sig_xy = np.zeros((n, max_nip), dtype=float)
    sig_xz = np.zeros((n, max_nip), dtype=float)
    eps_p = np.zeros((n, max_nip), dtype=float)

    group.state.update(
        fres=np.zeros((n, 3)),        # N, Vy, Vz
        mres=np.zeros((n, 3)),        # Mx, My, Mz
        L0=L0,
        mass=mass,
        off=np.ones(n, dtype=float),
        eint=np.zeros(n),
        ehour=np.zeros(n),
        dt0=dt0,
        mass_conn=conn[:, :2].copy(),
        dt_iner=inertia_c.copy(),
        # Fiber section data
        fibers_y=elem_fibers_y,
        fibers_z=elem_fibers_z,
        fibers_a=elem_fibers_a,
        sig_xx=sig_xx,
        sig_xy=sig_xy,
        sig_xz=sig_xz,
        eps_p=eps_p,
    )

    node_idx = conn[:, :2].reshape(-1)
    mass_c = np.repeat(mass / 2.0, 2)
    inertia_r = np.repeat(inertia_c, 2)
    return node_idx, mass_c, inertia_r


# ----------------------------------------------------------------------------
# Engine-side forces (main_beam18.F, mulaw_ib.F, m2lawpi.F)
# ----------------------------------------------------------------------------

def forces(
    group: Any,
    x: np.ndarray,
    v: Optional[np.ndarray],
    vr: Optional[np.ndarray],
    dt: float,
    fint: np.ndarray,
    mint: np.ndarray,
) -> np.ndarray:
    """Explicit cycle path: evaluates fiber strain increments, constitutive
    update, cross-section resultant integration, and nodal force assembly.

    Cites OpenRadioss:
        engine/source/elements/beam/main_beam18.F
        engine/source/elements/beam/mulaw_ib.F
        engine/source/materials/mat/mat002/m2lawpi.F
    """
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return np.zeros(0)
    if dt is None or dt <= 0.0:
        alive = group.state.get("off", np.ones(group.n, dtype=float)) > 0.0
        return np.where(alive, group.state.get("dt0", np.zeros(group.n)), EP30)
    if v is None:
        v = np.zeros_like(x)
    if vr is None:
        vr = np.zeros_like(x)

    st = group.state
    conn = group.conn
    n1, n2, n3 = conn[:, 0], conn[:, 1], conn[:, 2]
    E, L = _frame(x[n1], x[n2], x[n3])

    # Local translational velocities and rotation rates in corotational frame
    v1 = np.einsum("nb,nba->na", v[n1], E)
    v2 = np.einsum("nb,nba->na", v[n2], E)
    t1 = np.einsum("nb,nba->na", vr[n1], E)
    t2 = np.einsum("nb,nba->na", vr[n2], E)

    invL = 1.0 / np.maximum(L, EM20)
    eps_dot = (v2[:, 0] - v1[:, 0]) * invL
    gy_dot = (v2[:, 1] - v1[:, 1]) * invL - 0.5 * (t1[:, 2] + t2[:, 2])
    gz_dot = (v2[:, 2] - v1[:, 2]) * invL + 0.5 * (t1[:, 1] + t2[:, 1])
    kx_dot = (t2[:, 0] - t1[:, 0]) * invL
    ky_dot = (t2[:, 1] - t1[:, 1]) * invL
    kz_dot = (t2[:, 2] - t1[:, 2]) * invL

    # Generalized incremental deformations
    deps0 = eps_dot * dt
    dgy = gy_dot * dt
    dgz = gz_dot * dt
    dkx = kx_dot * dt
    dky = ky_dot * dt
    dkz = kz_dot * dt

    fres = st["fres"]
    mres = st["mres"]
    f_old = fres.copy()
    m_old = mres.copy()

    # Reset current resultants
    fres[:] = 0.0
    mres[:] = 0.0

    fibers_y = st["fibers_y"]
    fibers_z = st["fibers_z"]
    fibers_a = st["fibers_a"]
    sig_xx = st["sig_xx"]
    sig_xy = st["sig_xy"]
    sig_xz = st["sig_xz"]
    eps_p = st["eps_p"]

    # Timoshenko shear factor: OpenRadioss mulaw_ib.F line 94: SHFACT = 5/6
    shfact = 5.0 / 6.0

    for sl, mat, prop in st.get("slices", []):
        law = getattr(mat, "law", 1)
        if law == 0:
            # Void element
            continue

        E_mod = float(getattr(mat, "E", 0.0))
        nu = float(getattr(mat, "nu", 0.3))
        G_mod = float(getattr(mat, "G", E_mod / (2.0 * (1.0 + nu))))
        Gs = shfact * G_mod

        mp = getattr(mat, "params", {})
        is_plastic = (law == 2)
        sig_y0 = float(mp.get("A", mp.get("sig_y", 1e30)))
        B_hard = float(mp.get("B", 0.0))
        n_hard = float(mp.get("n", 1.0))
        sig_max = float(mp.get("sig_max", 1e30))

        for ie in range(sl.start, sl.stop):
            y_pts = fibers_y[ie]
            z_pts = fibers_z[ie]
            a_pts = fibers_a[ie]
            nip_ie = len(y_pts)

            de0_i = deps0[ie]
            dgy_i = dgy[ie]
            dgz_i = dgz[ie]
            dkx_i = dkx[ie]
            dky_i = dky[ie]
            dkz_i = dkz[ie]

            # Section resultant accumulators
            N_tot = 0.0
            Vy_tot = 0.0
            Vz_tot = 0.0
            Mx_tot = 0.0
            My_tot = 0.0
            Mz_tot = 0.0

            for k in range(nip_ie):
                yk = y_pts[k]
                zk = z_pts[k]
                Ak = a_pts[k]

                # mulaw_ib.F lines 140-144:
                # DEPSXX = EXX - YPT*KZZ + ZPT*KYY
                # DEPSXY = (EXY + ZPT*KXX) / SHFACT
                # DEPSXZ = (EXZ - YPT*KXX) / SHFACT
                deps_xx = de0_i - yk * dkz_i + zk * dky_i
                deps_xy = (dgy_i + zk * dkx_i) / shfact
                deps_xz = (dgz_i - yk * dkx_i) / shfact

                # Elastic trial stress
                sig_xx_tr = sig_xx[ie, k] + E_mod * deps_xx
                sig_xy_tr = sig_xy[ie, k] + Gs * deps_xy
                sig_xz_tr = sig_xz[ie, k] + Gs * deps_xz

                if is_plastic:
                    # von Mises equivalent stress (m2lawpi.F line 244):
                    # SVM1 = SIGNXX**2 + 3*(SIGNXY**2 + SIGNXZ**2)
                    svm2 = sig_xx_tr**2 + 3.0 * (sig_xy_tr**2 + sig_xz_tr**2)
                    svm = math.sqrt(max(svm2, 1e-30))

                    # Current yield stress
                    ep = eps_p[ie, k]
                    if B_hard > 0.0 and ep > 0.0:
                        sy = sig_y0 + B_hard * (ep**n_hard)
                    else:
                        sy = sig_y0
                    sy = min(sy, sig_max)

                    if svm > sy:
                        # Radial return consistency solve
                        dl = 0.0
                        for _ in range(5):
                            ep_curr = ep + dl
                            if B_hard > 0.0 and ep_curr > 0.0:
                                sy_curr = sig_y0 + B_hard * (ep_curr**n_hard)
                                H_curr = B_hard * n_hard * (ep_curr**(n_hard - 1.0))
                            else:
                                sy_curr = sig_y0
                                H_curr = 0.0
                            sy_curr = min(sy_curr, sig_max)
                            res = svm - E_mod * dl - sy_curr
                            denom = E_mod + max(H_curr, 0.0)
                            step = res / max(denom, EM20)
                            dl += step
                            if abs(step) < 1e-12:
                                break
                        dl = max(dl, 0.0)
                        ep_new = ep + dl
                        if B_hard > 0.0 and ep_new > 0.0:
                            sy_final = sig_y0 + B_hard * (ep_new**n_hard)
                        else:
                            sy_final = sig_y0
                        sy_final = min(sy_final, sig_max)

                        scale = min(1.0, sy_final / svm)
                        sig_xx_tr *= scale
                        sig_xy_tr *= scale
                        sig_xz_tr *= scale
                        eps_p[ie, k] = ep_new

                # Store updated stresses
                sig_xx[ie, k] = sig_xx_tr
                sig_xy[ie, k] = sig_xy_tr
                sig_xz[ie, k] = sig_xz_tr

                # Resultant integration (main_beam18.F lines 266-274)
                dfxx = Ak * sig_xx_tr
                dfxy = Ak * sig_xy_tr
                dfxz = Ak * sig_xz_tr

                N_tot += dfxx
                Vy_tot += dfxy
                Vz_tot += dfxz
                Mx_tot += dfxy * zk - dfxz * yk
                My_tot += dfxx * zk
                Mz_tot -= dfxx * yk

            fres[ie, 0] = N_tot
            fres[ie, 1] = Vy_tot
            fres[ie, 2] = Vz_tot
            mres[ie, 0] = Mx_tot
            mres[ie, 1] = My_tot
            mres[ie, 2] = Mz_tot

    # Internal energy update: midpoint trapezoidal work
    # main_beam18.F lines 300-308
    off = st.get("off", np.ones(group.n, dtype=float))
    N_avg = 0.5 * (f_old[:, 0] + fres[:, 0])
    Vy_avg = 0.5 * (f_old[:, 1] + fres[:, 1])
    Vz_avg = 0.5 * (f_old[:, 2] + fres[:, 2])
    Mx_avg = 0.5 * (m_old[:, 0] + mres[:, 0])
    My_avg = 0.5 * (m_old[:, 1] + mres[:, 1])
    Mz_avg = 0.5 * (m_old[:, 2] + mres[:, 2])

    dE_mb = N_avg * deps0
    dE_sh = Vy_avg * dgy + Vz_avg * dgz
    dE_fx = Mx_avg * dkx + My_avg * dky + Mz_avg * dkz
    st["eint"] += off * L * (dE_mb + dE_sh + dE_fx)

    # Local nodal internal forces and moments (pfint3.F)
    # f1 = (-N, -Vy, -Vz)
    # m1 = (-Mx, -My + Vz*L/2, -Mz - Vy*L/2)
    # f2 = ( N,  Vy,  Vz)
    # m2 = ( Mx,  My + Vz*L/2,  Mz - Vy*L/2)
    N = fres[:, 0] * off
    Vy = fres[:, 1] * off
    Vz = fres[:, 2] * off
    Mx = mres[:, 0] * off
    My = mres[:, 1] * off
    Mz = mres[:, 2] * off

    L_half = 0.5 * L
    f1_loc = np.stack([-N, -Vy, -Vz], axis=-1)
    m1_loc = np.stack([-Mx, -My + Vz * L_half, -Mz - Vy * L_half], axis=-1)
    f2_loc = np.stack([N, Vy, Vz], axis=-1)
    m2_loc = np.stack([Mx, My + Vz * L_half, Mz - Vy * L_half], axis=-1)

    # Global rotation: F = E @ f_loc, M = E @ m_loc
    f1_glob = np.einsum("nab,nb->na", E, f1_loc)
    m1_glob = np.einsum("nab,nb->na", E, m1_loc)
    f2_glob = np.einsum("nab,nb->na", E, f2_loc)
    m2_glob = np.einsum("nab,nb->na", E, m2_loc)

    # Global accumulation with minus sign (Radioss convention: a = (Fext + Fint)/m)
    np.add.at(fint, n1, -f1_glob)
    np.add.at(fint, n2, -f2_glob)
    np.add.at(mint, n1, -m1_glob)
    np.add.at(mint, n2, -m2_glob)

    # Critical time step: dt = dt0 * min(L / L0, sqrt(L0 / L))
    L0 = st["L0"]
    ratio = L / np.maximum(L0, EM20)
    dt_scale = np.minimum(ratio, np.sqrt(1.0 / np.maximum(ratio, EM20)))
    dt_e = st["dt0"] * dt_scale
    return np.where(off > 0.0, dt_e, EP30)


# ----------------------------------------------------------------------------
# Static and implicit routines (compatibility with implicit solver)
# ----------------------------------------------------------------------------

def static_internal_forces(group: Any, x: np.ndarray, fint: np.ndarray, mint: np.ndarray) -> None:
    """Evaluate static internal forces (zero velocity / rate limit)."""
    forces(group, x, None, None, 1.0, fint, mint)


def implicit_internal_forces(group: Any, x: np.ndarray, fint: np.ndarray, mint: np.ndarray) -> None:
    """Evaluate implicit internal forces."""
    forces(group, x, None, None, 1.0, fint, mint)


# ----------------------------------------------------------------------------
# Implicit element matrices: tangent, kgeo, consistent_mass (M614 Component 1B)
# ----------------------------------------------------------------------------
# Fortran origin:
#   engine/source/elements/beam/main_beam18.F (driver)
#   engine/source/elements/beam/mulaw_ib.F (constitutive)
#   engine/source/elements/beam/pmass3.F (mass)
#   engine/source/implicit/assem_p.F (implicit assembly for beams)
# ----------------------------------------------------------------------------

def _beam_edofs(conn: np.ndarray) -> np.ndarray:
    """(n, 12) global scalar DOF slot ids: nodes 0 and 1, 6 DOFs each."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 12), dtype=np.int64)
    edofs = np.empty((n, 12), dtype=np.int64)
    for c in range(6):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 6 + c] = conn[:, 1] * 6 + c
    return edofs


class TangentTuple(tuple):
    """2-tuple (ke, edofs) for implicit assembly, with backward compatibility
    for legacy tests that accessed tangent() as returning ke directly."""
    def __new__(cls, ke: np.ndarray, edofs: np.ndarray):
        return super().__new__(cls, (ke, edofs))

    @property
    def shape(self):
        return super().__getitem__(0).shape

    def __getitem__(self, idx):
        ke = super().__getitem__(0)
        edofs = super().__getitem__(1)
        if idx == 0 and len(ke) > 0:
            return ke[0]
        elif idx == 1 and len(ke) <= 1:
            return edofs
        elif isinstance(idx, (slice, tuple)):
            return ke[idx]
        return super().__getitem__(idx)

    def __iter__(self):
        yield super().__getitem__(0)
        yield super().__getitem__(1)


def tangent(group: Any, x: np.ndarray, epsp_incr: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Build global tangent stiffness matrix for implicit solver.

    Fortran origin: engine/source/elements/beam/main_beam18.F,
    engine/source/implicit/assem_p.F.

    Returns (ke, edofs): ke (n, 12, 12), edofs (n, 12).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return TangentTuple(np.zeros((0, 12, 12)), np.zeros((0, 12), dtype=np.int64))

    n1, n2, n3 = conn[:, 0], conn[:, 1], conn[:, 2]
    E, L = _frame(x[n1], x[n2], x[n3])

    K_glob = np.zeros((n, 12, 12), dtype=np.float64)
    for sl, mat, prop in st.get("slices", []):
        E_mod = float(getattr(mat, "E", 0.0) or 0.0)
        nu = float(getattr(mat, "nu", 0.3) or 0.3)
        G_mod = float(getattr(mat, "G", E_mod / (2.0 * (1.0 + nu))) or E_mod / (2.0 * (1.0 + nu)))
        p = getattr(prop, "params", {})
        A = float(p.get("area", getattr(prop, "area", 1.0)))
        Iyy = float(p.get("iyy", getattr(prop, "iyy", 1.0)))
        Izz = float(p.get("izz", getattr(prop, "izz", 1.0)))
        Ixx = float(p.get("ixx", getattr(prop, "ixx", Iyy + Izz)))

        C = np.diag([E_mod * A, G_mod * A, G_mod * A, G_mod * Ixx, E_mod * Iyy, E_mod * Izz])

        for ie in range(sl.start, sl.stop):
            B = _b_operator(L[ie])
            K_loc = L[ie] * (B.T @ C @ B)

            # Transform from local to global (4 blocks of 3x3)
            T = np.zeros((12, 12))
            for b in range(4):
                T[b * 3:(b + 1) * 3, b * 3:(b + 1) * 3] = E[ie]
            K_glob[ie] = T @ K_loc @ T.T

    return TangentTuple(K_glob, _beam_edofs(conn))


def kgeo(group: Any, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Geometric (initial-stress) stiffness for integrated fiber beam (/PROP/TYPE18).

    Fortran origin: engine/source/elements/beam/main_beam18.F,
    engine/source/implicit/assem_p.F.

    Accounts for axial force prestress N and bending moments My, Mz from fres and mres.
    Returns (ke, edofs): ke (n, 12, 12), edofs (n, 12).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12)), np.zeros((0, 12), dtype=np.int64)

    n1, n2, n3 = conn[:, 0], conn[:, 1], conn[:, 2]
    E, L = _frame(x[n1], x[n2], x[n3])

    fres = st.get("fres")
    mres = st.get("mres")
    if fres is None:
        fres = np.zeros((n, 3))
    if mres is None:
        mres = np.zeros((n, 3))

    N = fres[:, 0]
    My = mres[:, 1]
    Mz = mres[:, 2]

    ke_glob = np.zeros((n, 12, 12), dtype=np.float64)

    for ie in range(n):
        Lie = max(float(L[ie]), EM20)
        Nie = float(N[ie])
        Myie = float(My[ie])
        Mzie = float(Mz[ie])

        Kg_loc = np.zeros((12, 12), dtype=np.float64)

        # 1. Axial force contribution (Hermite cubic transverse + rotary terms)
        if abs(Nie) > 1e-20:
            c_y = Nie / (30.0 * Lie)
            Kg_loc[1, 1] += 36.0 * c_y
            Kg_loc[1, 5] += 3.0 * Lie * c_y
            Kg_loc[1, 7] -= 36.0 * c_y
            Kg_loc[1, 11] += 3.0 * Lie * c_y

            Kg_loc[5, 5] += 4.0 * (Lie**2) * c_y
            Kg_loc[5, 7] -= 3.0 * Lie * c_y
            Kg_loc[5, 11] -= (Lie**2) * c_y

            Kg_loc[7, 7] += 36.0 * c_y
            Kg_loc[7, 11] -= 3.0 * Lie * c_y

            Kg_loc[11, 11] += 4.0 * (Lie**2) * c_y

            c_z = Nie / (30.0 * Lie)
            Kg_loc[2, 2] += 36.0 * c_z
            Kg_loc[2, 4] -= 3.0 * Lie * c_z
            Kg_loc[2, 8] -= 36.0 * c_z
            Kg_loc[2, 10] -= 3.0 * Lie * c_z

            Kg_loc[4, 4] += 4.0 * (Lie**2) * c_z
            Kg_loc[4, 8] += 3.0 * Lie * c_z
            Kg_loc[4, 10] -= (Lie**2) * c_z

            Kg_loc[8, 8] += 36.0 * c_z
            Kg_loc[8, 10] += 3.0 * Lie * c_z

            Kg_loc[10, 10] += 4.0 * (Lie**2) * c_z

        # 2. Moment prestress contributions (lateral-torsional buckling coupling)
        if abs(Myie) > 1e-20:
            my_fac = Myie / Lie
            Kg_loc[1, 9] += my_fac
            Kg_loc[9, 1] += my_fac
            Kg_loc[7, 3] -= my_fac
            Kg_loc[3, 7] -= my_fac

        if abs(Mzie) > 1e-20:
            mz_fac = Mzie / Lie
            Kg_loc[2, 9] += mz_fac
            Kg_loc[9, 2] += mz_fac
            Kg_loc[8, 3] -= mz_fac
            Kg_loc[3, 8] -= mz_fac

        Kg_loc = 0.5 * (Kg_loc + Kg_loc.T)

        T = np.zeros((12, 12), dtype=np.float64)
        for b in range(4):
            T[b * 3:(b + 1) * 3, b * 3:(b + 1) * 3] = E[ie]
        ke_glob[ie] = T @ Kg_loc @ T.T

    return ke_glob, _beam_edofs(conn)


def _bending_mass_blocks_fiber(L, rhoA, rhoI):
    """The 4x4 Hermite cubic consistent mass for beam bending."""
    n = len(L)
    L2 = L * L
    Mt = np.empty((n, 4, 4), dtype=np.float64)
    c = rhoA * L / 420.0
    Mt[:, 0, 0] = 156 * c;      Mt[:, 0, 1] = 22 * L * c
    Mt[:, 0, 2] = 54 * c;       Mt[:, 0, 3] = -13 * L * c
    Mt[:, 1, 1] = 4 * L2 * c;   Mt[:, 1, 2] = 13 * L * c
    Mt[:, 1, 3] = -3 * L2 * c
    Mt[:, 2, 2] = 156 * c;      Mt[:, 2, 3] = -22 * L * c
    Mt[:, 3, 3] = 4 * L2 * c

    d = rhoI / (30.0 * np.maximum(L, EM20))
    Mt[:, 0, 0] += 36 * d;      Mt[:, 0, 1] += 3 * L * d
    Mt[:, 0, 2] += -36 * d;     Mt[:, 0, 3] += 3 * L * d
    Mt[:, 1, 1] += 4 * L2 * d;  Mt[:, 1, 2] += -3 * L * d
    Mt[:, 1, 3] += -1 * L2 * d
    Mt[:, 2, 2] += 36 * d;      Mt[:, 2, 3] += -3 * L * d
    Mt[:, 3, 3] += 4 * L2 * d

    i_lo = np.tril_indices(4, -1)
    Mt[:, i_lo[0], i_lo[1]] = Mt[:, i_lo[1], i_lo[0]]
    return Mt


def consistent_mass(group: Any, x: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Consistent element mass matrix for integrated fiber beam (/PROP/TYPE18) (12x12).

    Fortran origin: starter/source/properties/beam/hm_read_prop18.F,
    engine/source/elements/beam/pmass3.F, engine/source/implicit/assem_p.F.

    Returns (me, edofs): me (n, 12, 12), edofs (n, 12).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12)), np.zeros((0, 12), dtype=np.int64)

    n1, n2, n3 = conn[:, 0], conn[:, 1], conn[:, 2]
    x_coords = x if x is not None else st.get("x0")
    if x_coords is None:
        L0 = st.get("L0", np.ones(n))
        E = np.tile(np.eye(3), (n, 1, 1))
    else:
        E, L0 = _frame(x_coords[n1], x_coords[n2], x_coords[n3])

    rhoA = np.zeros(n, dtype=np.float64)
    rhoIyy = np.zeros(n, dtype=np.float64)
    rhoIzz = np.zeros(n, dtype=np.float64)
    rhoIp = np.zeros(n, dtype=np.float64)

    for sl, mat, prop in st.get("slices", []):
        rho0 = float(getattr(mat, "rho0", 0.0) or 0.0)
        p = getattr(prop, "params", {})
        area = float(p.get("area", getattr(prop, "area", 1.0)))
        iyy = float(p.get("iyy", getattr(prop, "iyy", 1.0)))
        izz = float(p.get("izz", getattr(prop, "izz", 1.0)))
        ixx = float(p.get("ixx", getattr(prop, "ixx", iyy + izz)))

        rhoA[sl] = rho0 * area
        rhoIyy[sl] = rho0 * iyy
        rhoIzz[sl] = rho0 * izz
        rhoIp[sl] = rho0 * ixx

    Ml = np.zeros((n, 12, 12), dtype=np.float64)
    ax = rhoA * L0 / 6.0
    Ml[:, 0, 0] = 2.0 * ax;   Ml[:, 0, 6] = ax
    Ml[:, 6, 0] = ax;         Ml[:, 6, 6] = 2.0 * ax

    tor = rhoIp * L0 / 6.0
    Ml[:, 3, 3] = 2.0 * tor;  Ml[:, 3, 9] = tor
    Ml[:, 9, 3] = tor;        Ml[:, 9, 9] = 2.0 * tor

    Mxy = _bending_mass_blocks_fiber(L0, rhoA, rhoIzz)
    ib = [1, 5, 7, 11]
    for a in range(4):
        for b in range(4):
            Ml[:, ib[a], ib[b]] = Mxy[:, a, b]

    Mxz = _bending_mass_blocks_fiber(L0, rhoA, rhoIyy)
    P = np.array([1.0, -1.0, 1.0, -1.0])
    Mxz = P[None, :, None] * Mxz * P[None, None, :]
    iz = [2, 4, 8, 10]
    for a in range(4):
        for b in range(4):
            Ml[:, iz[a], iz[b]] = Mxz[:, a, b]

    me = np.zeros((n, 12, 12), dtype=np.float64)
    for ie in range(n):
        T = np.zeros((12, 12), dtype=np.float64)
        for b in range(4):
            T[b * 3:(b + 1) * 3, b * 3:(b + 1) * 3] = E[ie]
        me[ie] = T @ Ml[ie] @ T.T

    return me, _beam_edofs(conn)

