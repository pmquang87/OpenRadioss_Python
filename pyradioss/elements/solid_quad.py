"""
4-node quadrilateral 2D solid element (plane strain or axisymmetric).
(/QUAD + /PROP/SOLID)

Fortran origin: ``engine/source/elements/solid_2d/quad/``
    qforc2.F  driver: gather coords/velocities, call the chain below
    qcoor2.F  geometry (area, volume)
    qdefo2.F  velocity gradient (Flanagan-Belytschko 1-point integration)
    qfint2.F  internal nodal forces
    qmass2.F  lumped mass
"""

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import scatter_add3

def _geometry(xe: np.ndarray, n2d: int):
    """
    Area, Volume, and gradients for the 2D quad element.
    xe : (n, 4, 3) nodal coordinates. In 2D, Y is index 1, Z is index 2.
    """
    Y1, Y2, Y3, Y4 = xe[:, 0, 1], xe[:, 1, 1], xe[:, 2, 1], xe[:, 3, 1]
    Z1, Z2, Z3, Z4 = xe[:, 0, 2], xe[:, 1, 2], xe[:, 2, 2], xe[:, 3, 2]

    PY1 = 0.5 * (Z2 - Z4)
    PY2 = 0.5 * (Z3 - Z1)
    PZ1 = 0.5 * (Y4 - Y2)
    PZ2 = 0.5 * (Y1 - Y3)

    area = 2.0 * (PZ2 * PY1 - PZ1 * PY2)
    
    if n2d == 1:
        # Axisymmetric (N2D=1): Exact 1-radian volume using two triangles
        A1 = 0.5 * ((Y2 - Y1) * (Z4 - Z1) - (Y4 - Y1) * (Z2 - Z1))
        A2 = 0.5 * ((Y3 - Y2) * (Z4 - Z2) - (Y4 - Y2) * (Z3 - Z2))
        vol = ((Y2 + Y3 + Y4) * A2 + (Y1 + Y2 + Y4) * A1) / 3.0 # subagent said ONE_OVER_6 but triangle areas are already 0.5
    else:
        # Plane strain (N2D=2): volume per unit depth is area
        vol = area
        
    return PY1, PY2, PZ1, PZ2, area, vol

def init_group(group, model, log):
    """Element buffer + lumped mass."""
    conn = group.conn
    xe = model.x0[conn]
    n2d = model.n2d
    
    PY1, PY2, PZ1, PZ2, area, vol = _geometry(xe, n2d)
    
    n = group.n
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = mat.rho0

    mass = rho0 * vol
    
    group.state.update(
        sig=np.zeros((n, 6)),        # Cauchy stress, Voigt
        epsp=np.zeros(n),            # equivalent plastic strain
        eint=np.zeros(n),
        vol0=vol.copy(),
        mass=mass,
        n2d=np.full(n, n2d, dtype=int), # Save n2d in state
        off=np.ones(n),              # 1 alive / 0 deleted
        qvw_pend=np.zeros(n),
        # dt_exact factor could be added here, setting 1.0 for now
        dtfac=np.ones(n),
        chk_fail=False,
    )
    
    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    return node_idx, mass_c, None

def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the 2D quad group (qforc2.F chain)."""
    st = group.state
    conn = group.conn
    xe = x[conn]                                   # (n, 4, 3) gather
    ve = v[conn]
    n2d = st.get("n2d", np.array([2]))[0]
    
    PY1, PY2, PZ1, PZ2, area, vol = _geometry(xe, n2d)
    vol = np.maximum(vol, EM20)
    area = np.maximum(area, EM20)
    
    # Engine uses area-weighted mass for axisymmetric elements!
    rho = st["mass"] / vol
    
    # Characteristic length (from qdlen2.F approximation)
    lc = area / np.sqrt(PY1**2 + PY2**2 + PZ1**2 + PZ2**2 + EM20)
    
    VY13 = ve[:, 0, 1] - ve[:, 2, 1]
    VY24 = ve[:, 1, 1] - ve[:, 3, 1]
    VZ13 = ve[:, 0, 2] - ve[:, 2, 2]
    VZ24 = ve[:, 1, 2] - ve[:, 3, 2]

    # Strains
    DYY = (PY1 * VY13 + PY2 * VY24) / area
    DZZ = (PZ1 * VZ13 + PZ2 * VZ24) / area
    DZY = (PY1 * VZ13 + PY2 * VZ24) / area
    DYZ = (PZ1 * VY13 + PZ2 * VY24) / area
    DYZ_eng = DZY + DYZ
    
    if n2d == 1:
        Y1, Y2, Y3, Y4 = xe[:, 0, 1], xe[:, 1, 1], xe[:, 2, 1], xe[:, 3, 1]
        YAVG = (Y1 + Y2 + Y3 + Y4) / 4.0
        # Prevent div by zero
        YAVG = np.where(YAVG == 0, EM20, YAVG)
        DTT = (ve[:, 0, 1] + ve[:, 1, 1] + ve[:, 2, 1] + ve[:, 3, 1]) / (4.0 * YAVG)
    else:
        DTT = np.zeros(group.n)
        
    deps = np.empty((group.n, 6))
    deps[:, 0] = 0.0          # XX (out of plane, hoop)
    deps[:, 1] = DYY * dt     # YY
    deps[:, 2] = DZZ * dt     # ZZ
    deps[:, 3] = 0.0          # XY
    deps[:, 4] = DYZ_eng * dt # YZ (engineering)
    deps[:, 5] = 0.0          # ZX
    
    # Hoop strain is stored in XX for axisymmetric
    if n2d == 1:
        deps[:, 0] = DTT * dt
        
    trD = DYY + DZZ + DTT
    
    sig = st["sig"]
    sig_old = sig.copy()
    
    # No Jaumann rate for 1-point quad formulation unless requested (standard radioss drops it for purely 2D).
    # Material law evaluation
    c = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        _, _, c_new = materials.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt, None)
        if c_new is not None:
            c[sl] = c_new
        else:
            c[sl] = np.sqrt((mat.K + 4.0 * mat.G / 3.0) / rho[sl])

    # Bulk Viscosity
    compressing = (trD < 0.0)
    qvisc = np.where(compressing, rho * lc * (1.5**2 * lc * trD ** 2 - 0.06 * c * trD), 0.0)
    
    sig_tot = sig.copy()
    sig_tot[:, 0] -= qvisc
    sig_tot[:, 1] -= qvisc
    sig_tot[:, 2] -= qvisc

    # Internal nodal forces (qfint2.F)
    # The gradient weights for each node:
    # N1: PY1, PZ1
    # N2: PY2, PZ2
    # N3: -PY1, -PZ1
    # N4: -PY2, -PZ2
    
    # S1 (YY) -> index 1
    # S2 (ZZ) -> index 2
    # S4 (YZ) -> index 4
    # S3 (XX) -> index 0 (Hoop stress)
    S1 = sig_tot[:, 1]
    S2 = sig_tot[:, 2]
    S4 = sig_tot[:, 4]
    
    # Force Y: S1*PY + S4*PZ
    # Force Z: S4*PY + S2*PZ
    FY1 = S1 * PY1 + S4 * PZ1
    FZ1 = S4 * PY1 + S2 * PZ1
    
    FY2 = S1 * PY2 + S4 * PZ2
    FZ2 = S4 * PY2 + S2 * PZ2
    
    AX1 = np.zeros(group.n)
    if n2d == 1:
        # Hoop stress contribution (AX1)
        S3 = sig_tot[:, 0]
        # (S3 - S1) * Area / (4 * Yavg)
        AX1 = (S3 - S1) * area / (4.0 * YAVG)
    
    fe = np.zeros((group.n, 4, 3))
    # Y-forces
    fe[:, 0, 1] = FY1 + AX1
    fe[:, 1, 1] = FY2 + AX1
    fe[:, 2, 1] = -FY1 + AX1
    fe[:, 3, 1] = -FY2 + AX1
    
    # Z-forces
    fe[:, 0, 2] = FZ1
    fe[:, 1, 2] = FZ2
    fe[:, 2, 2] = -FZ1
    fe[:, 3, 2] = -FZ2
    
    # Scatter to global array (negated because fint is defined as internal force resisting)
    # Radioss accumulates with MINUS sign into fint
    fe = -fe
    scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
    
    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = 0.5 * vol * qvisc * (-trD * dt) + st["qvw_pend"] * (-trD)
    st["eint"] += vol * np.einsum("nk,nk->n", sig_mid, deps) + w_visc
    st["qvw_pend"] = 0.5 * vol * qvisc * dt
    
    # Time step
    Q = np.where(compressing, 0.06 * c + 1.5 * lc * np.abs(trD), 0.0)
    dt_crit = st["dtfac"] * lc / (Q + np.sqrt(Q * Q + c * c))
    
    alive = st["off"] > 0.0
    return np.where(alive, dt_crit, EP30)

def tangent(group, x, epsp_incr=None):
    pass

def kgeo(group, x):
    pass

def consistent_mass(group, x=None):
    pass

def static_internal_forces(group, x, u, ur, fint, mint):
    pass
