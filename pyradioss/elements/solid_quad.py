"""
4-node quadrilateral 2D solid element (plane strain or axisymmetric).
(/QUAD + /PROP/SOLID, /PROP/TYPE15, /PROP/QUAD)

Fortran origin: ``engine/source/elements/solid_2d/quad/``
    qforc2.F  driver: gather coords/velocities, call the chain below
    qcoor2.F  geometry (area, volume)
    qdefo2.F  velocity gradient (Flanagan-Belytschko 1-point integration)
    qfint2.F  internal nodal forces
    qhvis2.F  Flanagan-Belytschko hourglass control
    qrota2.F  Jaumann stress rate rotation
    qdlen2.F  characteristic length and critical time step
    qmass2.F  lumped mass
    qvolu2.F  volume calculation (axisymmetric & plane strain)
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3


# ----------------------------------------------------------------------------
# Flanagan-Belytschko Hourglass Vector h = [1, -1, 1, -1]
# ----------------------------------------------------------------------------
_H = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float64)

# Reference 2D Bilinear Quad Consistent Mass Matrix M0 / m:
# M0 = (m / 36) * [[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]]
_M_QUAD4 = np.array([
    [4.0, 2.0, 1.0, 2.0],
    [2.0, 4.0, 2.0, 1.0],
    [1.0, 2.0, 4.0, 2.0],
    [2.0, 1.0, 2.0, 4.0],
], dtype=np.float64) / 36.0


def _geometry(xe: np.ndarray, n2d: int = 2):
    """
    Area, Volume, and gradients for the 2D quad element.
    xe : (n, 4, 3) nodal coordinates. In 2D, Y is index 1, Z is index 2.
    Returns:
        PY1, PY2, PZ1, PZ2: (n,) gradient weight components
        dndx: (n, 4, 2) shape function gradients in (Y, Z)
        area: (n,) in-plane cross-sectional area
        vol: (n,) element volume (Area for plane strain, 1-radian revolution for axisymmetric)
    """
    n = len(xe)
    if n == 0:
        return (np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0),
                np.zeros((0, 4, 2)), np.zeros(0), np.zeros(0))

    Y1, Y2, Y3, Y4 = xe[:, 0, 1], xe[:, 1, 1], xe[:, 2, 1], xe[:, 3, 1]
    Z1, Z2, Z3, Z4 = xe[:, 0, 2], xe[:, 1, 2], xe[:, 2, 2], xe[:, 3, 2]

    # Half-diagonal differences (qdefo2.F / qfint2.F)
    PY1 = 0.5 * (Z2 - Z4)
    PY2 = 0.5 * (Z3 - Z1)
    PZ1 = 0.5 * (Y4 - Y2)
    PZ2 = 0.5 * (Y1 - Y3)

    # Determinant components (twice triangle areas, qvolu2.F)
    A1 = Y2 * (Z3 - Z4) + Y3 * (Z4 - Z2) + Y4 * (Z2 - Z3)
    A2 = Y2 * (Z4 - Z1) + Y4 * (Z1 - Z2) + Y1 * (Z2 - Z4)
    area = 0.5 * (A1 + A2)

    deg = area <= 1e-12
    safe_area = np.where(deg, 1.0, area)

    # Shape function gradients [dN/dy, dN/dz] at element center (n, 4, 2)
    dndx = np.zeros((n, 4, 2), dtype=np.float64)
    dndx[:, 0, 0] = PY1 / safe_area
    dndx[:, 0, 1] = PZ1 / safe_area
    dndx[:, 1, 0] = PY2 / safe_area
    dndx[:, 1, 1] = PZ2 / safe_area
    dndx[:, 2, 0] = -PY1 / safe_area
    dndx[:, 2, 1] = -PZ1 / safe_area
    dndx[:, 3, 0] = -PY2 / safe_area
    dndx[:, 3, 1] = -PZ2 / safe_area
    if deg.any():
        dndx[deg] = 0.0

    if n2d == 1:
        # Axisymmetric (N2D=1): Exact 1-radian volume (qvolu2.F)
        # vol = ((Y2 + Y3 + Y4)*A1 + (Y1 + Y2 + Y4)*A2) / 6.0
        vol = ((Y2 + Y3 + Y4) * A1 + (Y1 + Y2 + Y4) * A2) * (1.0 / 6.0)
        vol = np.maximum(vol, 0.0)
    else:
        # Plane strain (N2D=2): volume per unit depth is area
        vol = np.maximum(area, 0.0)

    return PY1, PY2, PZ1, PZ2, dndx, np.maximum(area, 0.0), vol


def _char_length(xe: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Characteristic minimum altitude length (qdlen2.F)."""
    n = len(xe)
    if n == 0:
        return np.zeros(0)
    Y = xe[:, :, 1]
    Z = xe[:, :, 2]

    # Edge squared lengths
    al1 = (Y[:, 1] - Y[:, 0])**2 + (Z[:, 1] - Z[:, 0])**2
    al2 = (Y[:, 2] - Y[:, 1])**2 + (Z[:, 2] - Z[:, 1])**2
    al3 = (Y[:, 3] - Y[:, 2])**2 + (Z[:, 3] - Z[:, 2])**2
    al4 = (Y[:, 0] - Y[:, 3])**2 + (Z[:, 0] - Z[:, 3])**2
    # Diagonal squared lengths
    al5 = (Y[:, 2] - Y[:, 0])**2 + (Z[:, 2] - Z[:, 0])**2
    al6 = (Y[:, 3] - Y[:, 1])**2 + (Z[:, 3] - Z[:, 1])**2

    al_max = np.maximum.reduce([al1, al2, al3, al4, al5, al6])
    diag_len = np.sqrt(np.maximum(al_max, EM20))
    return np.where(diag_len > 0.0, area / diag_len, 0.0)


def init_group(group, model, log):
    """Element buffer + lumped mass."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 6)),
            epsp=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            vol0=np.zeros(0),
            mass=np.zeros(0),
            n2d=np.full(0, 2, dtype=int),
            off=np.zeros(0),
            qvw_pend=np.zeros(0),
            dtfac=np.zeros(0),
            chk_fail=False,
            dama=np.zeros(0),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe = model.x0[conn]
    n2d = getattr(model, "n2d", 2)

    PY1, PY2, PZ1, PZ2, dndx, area, vol = _geometry(xe, n2d)

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = getattr(mat, "rho0", 0.0)

    mass = rho0 * vol

    chk_fail = any(
        getattr(mat, "fail", None) is not None
        or getattr(mat, "params", {}).get("eps_p_max", EP30) < 1e30
        for _, mat, _ in group.state.get("slices", [])
    )

    group.state.update(
        sig=np.zeros((n, 6)),        # Cauchy stress, Voigt [xx, yy, zz, xy, yz, zx]
        epsp=np.zeros(n),            # equivalent plastic strain
        eint=np.zeros(n),
        ehour=np.zeros(n),           # hourglass energy
        vol0=vol.copy(),
        mass=mass,
        n2d=np.full(n, n2d, dtype=int), # Save n2d in state
        off=np.ones(n),              # 1 alive / 0 deleted
        qvw_pend=np.zeros(n),
        dtfac=np.ones(n),
        chk_fail=chk_fail,
        dama=np.zeros(n),
    )

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the 2D quad group (qforc2.F chain)."""
    n = group.n
    conn = group.conn
    if n == 0 or len(conn) == 0:
        return np.zeros(0)

    st = group.state
    alive = st["off"] > 0.0

    n2d_arr = st.get("n2d", np.array([2]))
    n2d = int(n2d_arr[0]) if len(n2d_arr) > 0 else 2

    xe = x[conn]                                   # (n, 4, 3) gather
    PY1, PY2, PZ1, PZ2, dndx, area, vol = _geometry(xe, n2d)
    safe_vol = np.maximum(vol, EM20)
    safe_area = np.maximum(area, EM20)

    rho = st["mass"] / safe_vol
    lc = _char_length(xe, safe_area)

    # Initial cycle dt <= 0 check
    if dt is None or dt <= 0.0 or v is None:
        c_sound = np.zeros(n)
        for sl, mat, prop in st.get("slices", []):
            law = getattr(mat, "law", 1)
            rho0_sl = getattr(mat, "rho0", 0.0)
            E_sl = getattr(mat, "E", 0.0)
            if law == 0 or rho0_sl <= 0.0 or E_sl <= 0.0:
                c_sound[sl] = 0.0
            else:
                K_sl = getattr(mat, "K", E_sl / 3.0)
                G_sl = getattr(mat, "G", E_sl / 2.0)
                c_sound[sl] = np.sqrt(max(K_sl + 4.0 * G_sl / 3.0, 0.0) / max(rho0_sl, EM20))
        dt_crit = np.where(alive & (c_sound > 0.0), st["dtfac"] * lc / np.maximum(c_sound, EM20), EP30)
        return dt_crit

    ve = v[conn]

    VY13 = ve[:, 0, 1] - ve[:, 2, 1]
    VY24 = ve[:, 1, 1] - ve[:, 3, 1]
    VZ13 = ve[:, 0, 2] - ve[:, 2, 2]
    VZ24 = ve[:, 1, 2] - ve[:, 3, 2]

    # Strains rates (qdefo2.F)
    DYY = (PY1 * VY13 + PY2 * VY24) / safe_area
    DZZ = (PZ1 * VZ13 + PZ2 * VZ24) / safe_area
    DZY = (PY1 * VZ13 + PY2 * VZ24) / safe_area
    DYZ = (PZ1 * VY13 + PZ2 * VY24) / safe_area
    DYZ_eng = DZY + DYZ

    if n2d == 1:
        YAVG = (xe[:, 0, 1] + xe[:, 1, 1] + xe[:, 2, 1] + xe[:, 3, 1]) * 0.25
        safe_yavg = np.where(np.abs(YAVG) <= EM20, EM20, YAVG)
        DTT = (ve[:, 0, 1] + ve[:, 1, 1] + ve[:, 2, 1] + ve[:, 3, 1]) / (4.0 * safe_yavg)
    else:
        DTT = np.zeros(n)

    deps = np.zeros((n, 6))
    deps[:, 0] = DTT * dt     # XX (hoop strain in axisymmetric, 0 in plane strain)
    deps[:, 1] = DYY * dt     # YY
    deps[:, 2] = DZZ * dt     # ZZ
    deps[:, 3] = 0.0          # XY
    deps[:, 4] = DYZ_eng * dt # YZ (engineering shear)
    deps[:, 5] = 0.0          # ZX

    trD = DYY + DZZ + DTT

    # ---- Jaumann stress rate rotation (qrota2.F) ---------------------------
    sig = st["sig"]
    sig_old = sig.copy()
    WYZ = 0.5 * dt * (DZY - DYZ)
    Q1 = 2.0 * sig[:, 4] * WYZ
    sig[:, 1] += Q1
    sig[:, 2] -= Q1
    sig[:, 4] += WYZ * (sig[:, 2] - sig[:, 1])

    # ---- Material evaluation -----------------------------------------------
    epsp_old = st["epsp"].copy() if st.get("chk_fail") else None
    c = np.zeros(n)
    c_from_law = np.zeros(n, dtype=bool)

    for sl, mat, prop in st.get("slices", []):
        mask_sl = alive[sl]
        if not np.any(mask_sl):
            continue
        law = getattr(mat, "law", 1)
        rho0_sl = getattr(mat, "rho0", 0.0)
        E_sl = getattr(mat, "E", 0.0)
        if law == 0 or rho0_sl <= 0.0 or E_sl <= 0.0:
            sig[sl] = 0.0
            c[sl] = 0.0
            c_from_law[sl] = True
            continue

        extra = st.get("mat_extra", {})
        if law == 1:
            materials.law01_elastic.solid_update(mat, sig[sl], deps[sl])
        elif law == 2:
            materials.law02_johnson_cook.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt, extra)
        elif law == 36:
            materials.law36_tabulated.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt, extra)
        else:
            _, _, c_new = materials.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt, extra)
            if c_new is not None:
                c[sl] = c_new
                c_from_law[sl] = True

    # ---- Failure handling (AUD-010 Fix) ------------------------------------
    if st.get("chk_fail"):
        off = st["off"]
        for sl, mat, prop in st.get("slices", []):
            eps_max = getattr(mat, "params", {}).get("eps_p_max", EP30)
            if getattr(mat, "fail", None) is None and eps_max >= 1e30:
                continue
            broken = np.zeros(sl.stop - sl.start, dtype=bool)
            if getattr(mat, "fail", None) is not None:
                broken |= failure.solid_step(
                    mat.fail, sig[sl], st["epsp"][sl] - epsp_old[sl],
                    deps[sl], dt, st["dama"][sl])
            if eps_max < 1e30:
                broken |= st["epsp"][sl] > eps_max
            off[sl][broken] = 0.0
        alive = off > 0.0
        sig[~alive] = 0.0

    # ---- Sound speed, parameters & bulk viscosity (qdlen2.F / qfint2.F) ----
    qa = np.zeros(n)
    qb = np.zeros(n)
    hcoef = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        if not c_from_law[sl.start]:
            rho0_sl = getattr(mat, "rho0", 0.0)
            if getattr(mat, "law", 1) == 0 or rho0_sl <= 0.0:
                c[sl] = 0.0
            else:
                K_sl = getattr(mat, "K", 0.0)
                G_sl = getattr(mat, "G", 0.0)
                c[sl] = np.sqrt(max(K_sl + 4.0 * G_sl / 3.0, 0.0) / max(rho0_sl, EM20))
        params = getattr(prop, "params", {}) if hasattr(prop, "params") else {}
        qa[sl] = params.get("qa", getattr(prop, "qa", 1.1))
        qb[sl] = params.get("qb", getattr(prop, "qb", 0.05))
        hcoef[sl] = params.get("h", getattr(prop, "h", 0.1))

    compressing = (trD < 0.0) & alive
    qvisc = np.where(
        compressing,
        rho * lc * (qa**2 * lc * trD**2 - qb * c * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, 0] -= qvisc
    sig_tot[:, 1] -= qvisc
    sig_tot[:, 2] -= qvisc

    # ---- Flanagan-Belytschko Hourglass Forces (qhvis2.F) -------------------
    # h = [1, -1, 1, -1]
    HY_node = xe[:, 0, 1] - xe[:, 1, 1] + xe[:, 2, 1] - xe[:, 3, 1]
    HZ_node = xe[:, 0, 2] - xe[:, 1, 2] + xe[:, 2, 2] - xe[:, 3, 2]
    fac_h = 1.0 / safe_area
    PX1H1 = fac_h * (PY1 * HY_node + PZ1 * HZ_node)
    PX2H1 = fac_h * (PY2 * HY_node + PZ2 * HZ_node)
    gamma = np.empty((n, 4), dtype=np.float64)
    gamma[:, 0] = 1.0 - PX1H1
    gamma[:, 1] = -1.0 - PX2H1
    gamma[:, 2] = 1.0 + PX1H1
    gamma[:, 3] = -1.0 + PX2H1

    HGY = 0.5 * (gamma[:, 0] * ve[:, 0, 1] + gamma[:, 1] * ve[:, 1, 1] +
                 gamma[:, 2] * ve[:, 2, 1] + gamma[:, 3] * ve[:, 3, 1])
    HGZ = 0.5 * (gamma[:, 0] * ve[:, 0, 2] + gamma[:, 1] * ve[:, 1, 2] +
                 gamma[:, 2] * ve[:, 2, 2] + gamma[:, 3] * ve[:, 3, 2])

    FCQ = rho * np.sqrt(safe_area)
    FCL = hcoef * FCQ * c * alive
    HY_force = HGY * (FCL + np.abs(HGY) * FCQ * hcoef * 100.0)
    HZ_force = HGZ * (FCL + np.abs(HGZ) * FCQ * hcoef * 100.0)

    T1 = gamma * HY_force[:, None]   # (n, 4) in Y
    T2 = gamma * HZ_force[:, None]   # (n, 4) in Z
    dehour = 2.0 * dt * (HY_force * HGY + HZ_force * HGZ) * alive
    st["ehour"] += dehour

    # ---- Internal Nodal Forces (qfint2.F & qcumu2.F) -----------------------
    S1 = sig_tot[:, 1]  # YY
    S2 = sig_tot[:, 2]  # ZZ
    S4 = sig_tot[:, 4]  # YZ

    F11 = S1 * PY1 + S4 * PZ1
    F21 = S4 * PY1 + S2 * PZ1
    F12 = S1 * PY2 + S4 * PZ2
    F22 = S4 * PY2 + S2 * PZ2

    AX1 = np.zeros(n)
    AX2 = np.zeros(n)
    if n2d == 1:
        S3 = sig_tot[:, 0]  # XX (hoop)
        fac_ax = safe_area * safe_area / (4.0 * safe_vol)
        AX1 = (S3 - S1) * fac_ax
        AX2 = S4 * fac_ax

    # QCUMU2 assembly of nodal internal forces (before negation)
    fe_elem = np.zeros((n, 4, 3))
    # Y-forces
    fe_elem[:, 0, 1] = F11 + AX1 + T1[:, 0]
    fe_elem[:, 1, 1] = F12 + AX1 + T1[:, 1]
    fe_elem[:, 2, 1] = -F11 + AX1 + T1[:, 2]
    fe_elem[:, 3, 1] = -F12 + AX1 + T1[:, 3]

    # Z-forces
    fe_elem[:, 0, 2] = F21 - AX2 + T2[:, 0]
    fe_elem[:, 1, 2] = F22 - AX2 + T2[:, 1]
    fe_elem[:, 2, 2] = -F21 - AX2 + T2[:, 2]
    fe_elem[:, 3, 2] = -F22 - AX2 + T2[:, 3]

    # Negate: fint accumulates resisting force -integral(B^T sigma)
    fe = -fe_elem
    fe *= alive[:, None, None]

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3),
                     st.get("color_indices"), st.get("color_offsets"))

    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = 0.5 * vol * qvisc * (-trD * dt) + st["qvw_pend"] * (-trD)
    st["eint"] += vol * np.einsum("nk,nk->n", sig_mid, deps) * alive + w_visc * alive
    st["qvw_pend"] = 0.5 * vol * qvisc * dt * alive

    # Time step
    Q = np.where(compressing, qb * c + qa * lc * np.abs(trD), 0.0)
    denom = Q + np.sqrt(Q * Q + c * c)
    dt_crit = np.where(alive & (denom > 0.0) & (c > 0.0), st["dtfac"] * lc / np.maximum(denom, EM20), EP30)
    return dt_crit


def _edofs(conn: np.ndarray) -> np.ndarray:
    """(n, 12) global scalar DOF slot ids, node-major [ux, uy, uz] * 4."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 12), dtype=np.int64)
    ix = np.arange(4)
    edofs = np.empty((n, 12), dtype=np.int64)
    for c in range(3):
        edofs[:, 3 * ix + c] = conn * 6 + c
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the 2D quad group.
    Returns (ke, edofs): ke (n, 12, 12), edofs (n, 12)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12), dtype=float), np.zeros((0, 12), dtype=np.int64)

    xe = x[conn]
    n2d_arr = st.get("n2d", np.array([2]))
    n2d = int(n2d_arr[0]) if len(n2d_arr) > 0 else 2

    PY1, PY2, PZ1, PZ2, dndx, area, vol = _geometry(xe, n2d)
    safe_vol = np.maximum(vol, EM20)
    safe_area = np.maximum(area, EM20)

    # Strain-displacement matrix B (n, 6, 12)
    # Strains in Voigt: [xx, yy, zz, xy, yz, zx]
    # In 2D quad: Y is DOF 3*ix + 1, Z is DOF 3*ix + 2
    B = np.zeros((n, 6, 12), dtype=np.float64)
    for ix in range(4):
        gy = dndx[:, ix, 0]
        gz = dndx[:, ix, 1]
        # YY strain
        B[:, 1, 3 * ix + 1] = gy
        # ZZ strain
        B[:, 2, 3 * ix + 2] = gz
        # YZ engineering shear strain
        B[:, 4, 3 * ix + 1] = gz
        B[:, 4, 3 * ix + 2] = gy
        if n2d == 1:
            # Axisymmetric hoop strain: DTT = (sum v_y) / (4 * Yavg)
            YAVG = (xe[:, 0, 1] + xe[:, 1, 1] + xe[:, 2, 1] + xe[:, 3, 1]) * 0.25
            safe_yavg = np.where(np.abs(YAVG) <= EM20, EM20, YAVG)
            B[:, 0, 3 * ix + 1] = 0.25 / safe_yavg

    ke = np.zeros((n, 12, 12), dtype=np.float64)
    epi = np.zeros(n) if epsp_incr is None else epsp_incr

    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        D = materials.solid_tangent(mat, st["sig"][sl], st["epsp"][sl], epi[sl], None)
        Bs = B[sl]
        DB = np.einsum("mij,mjk->mik", D, Bs)
        ke[sl] = safe_vol[sl, None, None] * np.einsum("mji,mjk->mik", Bs, DB)

        # Flanagan-Belytschko hourglass stiffness stabilization (q4vis2.F)
        # Adds k_s * gamma_i * gamma_j to in-plane Y and Z DOFs
        G = getattr(mat, "G", getattr(mat, "E", 0.0) / 2.6)
        trace_grad = np.sum(dndx[sl]**2, axis=(1, 2))  # sum |gradN|^2
        ks = 0.05 * G * safe_area[sl] * trace_grad

        HY_node = xe[sl, 0, 1] - xe[sl, 1, 1] + xe[sl, 2, 1] - xe[sl, 3, 1]
        HZ_node = xe[sl, 0, 2] - xe[sl, 1, 2] + xe[sl, 2, 2] - xe[sl, 3, 2]
        fac_h = 1.0 / safe_area[sl]
        PX1H1 = fac_h * (PY1[sl] * HY_node + PZ1[sl] * HZ_node)
        PX2H1 = fac_h * (PY2[sl] * HY_node + PZ2[sl] * HZ_node)
        m_sl = len(ke[sl])
        gamma = np.empty((m_sl, 4), dtype=np.float64)
        gamma[:, 0] = 1.0 - PX1H1
        gamma[:, 1] = -1.0 - PX2H1
        gamma[:, 2] = 1.0 + PX1H1
        gamma[:, 3] = -1.0 + PX2H1

        gamma_outer = gamma[:, :, None] * gamma[:, None, :]  # (m, 4, 4)
        for a in range(4):
            for b in range(4):
                gab = ks * gamma_outer[:, a, b]
                # Y-Y block
                ke[sl, 3 * a + 1, 3 * b + 1] += gab
                # Z-Z block
                ke[sl, 3 * a + 2, 3 * b + 2] += gab

    return ke, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) stiffness matrix for 2D quad group.
    Returns (k_geo, edofs): k_geo (n, 12, 12), edofs (n, 12)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12), dtype=float), np.zeros((0, 12), dtype=np.int64)

    xe = x[conn]
    n2d_arr = st.get("n2d", np.array([2]))
    n2d = int(n2d_arr[0]) if len(n2d_arr) > 0 else 2

    _, _, _, _, dndx, _, vol = _geometry(xe, n2d)
    safe_vol = np.maximum(vol, EM20)

    # 2D in-plane stress tensor S = [[syy, syz], [syz, szz]]
    syy = st["sig"][:, 1]
    szz = st["sig"][:, 2]
    syz = st["sig"][:, 4]

    S = np.empty((n, 2, 2), dtype=np.float64)
    S[:, 0, 0] = syy
    S[:, 1, 1] = szz
    S[:, 0, 1] = S[:, 1, 0] = syz

    # g_ab = vol * sum_{c, d} dN_{a, c} S_{cd} dN_{b, d}
    g = safe_vol[:, None, None] * np.einsum("nac,ncd,nbd->nab", dndx, S, dndx)

    k_geo = np.zeros((n, 12, 12), dtype=np.float64)
    ix = np.arange(4)
    for c in range(3):
        rows = (3 * ix + c)[:, None]
        cols = (3 * ix + c)[None, :]
        k_geo[:, rows, cols] += g

    return k_geo, _edofs(conn)


def consistent_mass(group, x=None):
    """Analytical 2D quad consistent mass matrix (12x12).
    Returns (me, edofs): me (n, 12, 12), edofs (n, 12)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12), dtype=float), np.zeros((0, 12), dtype=np.int64)

    m = st["mass"]  # rho * vol
    me = np.zeros((n, 12, 12), dtype=np.float64)
    for a in range(4):
        for b in range(4):
            val = m * _M_QUAD4[a, b]
            for c in range(3):
                me[:, 3 * a + c, 3 * b + c] = val

    return me, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal force at configuration x from CURRENT stress state."""
    if group.n == 0 or len(group.conn) == 0 or fint is None:
        return
    st = group.state
    conn = group.conn
    n = group.n
    xe = x[conn]

    n2d_arr = st.get("n2d", np.array([2]))
    n2d = int(n2d_arr[0]) if len(n2d_arr) > 0 else 2

    PY1, PY2, PZ1, PZ2, dndx, area, vol = _geometry(xe, n2d)
    safe_vol = np.maximum(vol, EM20)
    safe_area = np.maximum(area, EM20)

    sig = st["sig"]
    S1 = sig[:, 1]
    S2 = sig[:, 2]
    S4 = sig[:, 4]

    F11 = S1 * PY1 + S4 * PZ1
    F21 = S4 * PY1 + S2 * PZ1
    F12 = S1 * PY2 + S4 * PZ2
    F22 = S4 * PY2 + S2 * PZ2

    AX1 = np.zeros(n)
    AX2 = np.zeros(n)
    if n2d == 1:
        S3 = sig[:, 0]
        fac_ax = safe_area * safe_area / (4.0 * safe_vol)
        AX1 = (S3 - S1) * fac_ax
        AX2 = S4 * fac_ax

    fe_elem = np.zeros((n, 4, 3))
    fe_elem[:, 0, 1] = F11 + AX1
    fe_elem[:, 1, 1] = F12 + AX1
    fe_elem[:, 2, 1] = -F11 + AX1
    fe_elem[:, 3, 1] = -F12 + AX1

    fe_elem[:, 0, 2] = F21 - AX2
    fe_elem[:, 1, 2] = F22 - AX2
    fe_elem[:, 2, 2] = -F21 - AX2
    fe_elem[:, 3, 2] = -F22 - AX2

    fe = -fe_elem
    alive = st["off"] > 0.0
    fe *= alive[:, None, None]

    scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3),
                 st.get("color_indices"), st.get("color_offsets"))


def implicit_internal_forces(group, x, u, ur, fint, mint, nlgeom=False):
    """Implicit solver internal force vector dispatcher.
    nlgeom=False: linear Ku from initial configuration.
    nlgeom=True: nonlinear updated Lagrangian static assembly."""
    if group.n == 0 or len(group.conn) == 0 or fint is None:
        return
    if not nlgeom:
        ke, _ = tangent(group, x)
        ue = np.zeros((group.n, 12))
        for ix in range(4):
            nid = group.conn[:, ix]
            valid = nid >= 0
            for c in range(3):
                ue[valid, 3 * ix + c] = u[nid[valid], c]
        fe = -np.einsum("nij,nj->ni", ke, ue).reshape(-1, 4, 3)
        scatter_add3(fint, group.conn.reshape(-1), fe.reshape(-1, 3),
                     group.state.get("color_indices"), group.state.get("color_offsets"))
    else:
        static_internal_forces(group, x, u, ur, fint, mint)
