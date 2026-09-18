"""
8-node Cohesive Zone Solid Element (solidez, /PROP/TYPE21).

Fortran origin: ``engine/source/elements/solid/solidez/``
    szforc3.F   driver: calculate opening jumps, traction-separation law, nodal forces
    szderi3.F   mid-surface geometry, normal/tangent frames, and surface areas
    sz_dt1.F90  eigenvalue-based critical time step calculation for zero/small thickness

Theory notes:
* Models interfaces, adhesive layers, delamination, and fracture between continuum parts.
* Connectivity: nodes 1-4 on bottom interface, nodes 5-8 on top interface.
* Kinematics:
    Displacement jump across the interface:
        delta = x_top - x_bottom - (x0_top - x0_bottom)
    Decomposed into normal opening delta_n = delta . n and tangential slips delta_t.
* Constitutive:
    Bilinear traction-separation law:
        T_n = K_n * delta_n * (1 - D)   (delta_n > 0)
        T_n = K_n * delta_n             (delta_n <= 0, contact penalty)
        T_t = K_t * delta_t * (1 - D)
    where D is the interface damage parameter progressing from 0 (pristine) to 1 (failed).
* Critical time step (sz_dt1.F90):
    Because cohesive elements often have zero or near-zero physical thickness (h -> 0),
    the standard Courant condition dt = h / c would vanish!
    Instead, Radioss sz_dt1 bounds the maximum eigenvalue of the stiffness matrix:
        omega_max^2 = 4 * K_max / (rho * h_eff)
        dt_crit = 2 / omega_max = sqrt(rho * h_eff / K_max)
    where h_eff is derived from the in-plane element dimensions.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3, scatter_add3


def _midsurface_geometry(xe: np.ndarray):
    """Compute mid-surface coordinates, normal vector, and interface area.

    xe: (n, 8, 3) nodal coordinates.
    Nodes 0..3: bottom face, 4..7: top face.
    Returns:
        x_bot: (n, 4, 3)
        x_top: (n, 4, 3)
        n_mid: (n, 3) unit normal pointing from bottom to top
        area: (n,) interface area
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 4, 3)), np.zeros((0, 4, 3)), np.zeros((0, 3)), np.zeros(0)

    x_bot = xe[:, 0:4]
    x_top = xe[:, 4:8]
    x_mid = 0.5 * (x_bot + x_top)  # (n, 4, 3)

    # Mid-surface diagonal vectors
    d13 = x_mid[:, 2] - x_mid[:, 0]
    d24 = x_mid[:, 3] - x_mid[:, 1]
    n_cross = np.cross(d13, d24)  # (n, 3)
    area2 = np.linalg.norm(n_cross, axis=1)  # 2 * area
    area = 0.5 * area2

    safe_area2 = np.where(area2 > EM20, area2, 1.0)
    n_mid = n_cross / safe_area2[:, None]

    return x_bot, x_top, n_mid, area


def init_group(group, model, log):
    """Starter initialization for 8-node cohesive element group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 3)),       # tractions [Tn, Tt1, Tt2]
            jump=np.zeros((0, 3)),      # displacement jump
            dama=np.zeros(0),           # damage D in [0, 1]
            vol0=np.zeros(0),
            area0=np.zeros(0),
            mass=np.zeros(0),
            eint=np.zeros(0),
            off=np.zeros(0),
            dtfac=np.zeros(0),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe0 = model.x0[conn]
    x_bot0, x_top0, n_mid0, area0 = _midsurface_geometry(xe0)

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 1.0) if mat else 1.0

    # For cohesive elements, assign a nominal lumped mass based on area * effective density
    # In OpenRadioss, cohesive mass is typically assigned from adjacent parts or rho * area * h_eff
    h_eff = np.sqrt(np.maximum(area0, EM20))  # in-plane characteristic size
    mass = rho0 * area0 * h_eff

    group.state.update(
        sig=np.zeros((n, 3)),
        jump=np.zeros((n, 3)),
        dama=np.zeros(n),
        vol0=area0 * h_eff,
        area0=area0.copy(),
        xe0=xe0.copy(),
        mass=mass,
        eint=np.zeros(n),
        off=np.ones(n),
        dtfac=np.ones(n),
    )

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 8-node cohesive elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]
    xe0 = st["xe0"]
    x_bot, x_top, n_mid, area = _midsurface_geometry(xe)
    alive = st["off"] > 0.0

    # Average top vs bottom relative displacement: delta = mean(x_top - x_bot) - mean(x0_top - x0_bot)
    delta_curr = np.mean(x_top - x_bot, axis=1)      # (n, 3)
    delta_init = np.mean(xe0[:, 4:8] - xe0[:, 0:4], axis=1)
    delta = delta_curr - delta_init                  # net displacement jump (n, 3)

    # Normal opening
    delta_n = np.sum(delta * n_mid, axis=1)          # (n,)
    delta_t_vec = delta - delta_n[:, None] * n_mid   # tangential vector (n, 3)
    delta_t = np.linalg.norm(delta_t_vec, axis=1)

    # Cohesive stiffness parameters from property
    Kn = np.zeros(n)
    Kt = np.zeros(n)
    sigma_max = np.zeros(n)
    delta_max = np.zeros(n)

    for sl, mat, prop in st.get("slices", []):
        Kn[sl] = getattr(prop, "kn", 1.0e6) if prop else 1.0e6
        Kt[sl] = getattr(prop, "kt", 1.0e6) if prop else 1.0e6
        sigma_max[sl] = getattr(prop, "sigma_max", 1.0e8) if prop else 1.0e8
        delta_max[sl] = getattr(prop, "delta_max", 1.0e-3) if prop else 1.0e-3

    # Damage evolution (bilinear traction-separation)
    # Onset opening: delta_0 = sigma_max / Kn
    delta_0 = sigma_max / np.maximum(Kn, EM20)
    delta_f = np.maximum(delta_max, delta_0 * 1.01)

    # Equivalent mixed-mode opening: delta_eq = sqrt(max(0, delta_n)^2 + delta_t^2)
    delta_eq = np.sqrt(np.maximum(delta_n, 0.0)**2 + delta_t**2)

    # Update damage variable D
    dama = st["dama"]
    dama_new = np.where(
        delta_eq > delta_0,
        (delta_f / (delta_f - delta_0)) * (1.0 - delta_0 / np.maximum(delta_eq, EM20)),
        0.0
    )
    dama_new = np.clip(dama_new, 0.0, 1.0)
    dama = np.maximum(dama, dama_new)
    st["dama"] = dama

    # Tractions: Tn and Tt
    Tn = np.where(
        delta_n >= 0.0,
        Kn * delta_n * (1.0 - dama),
        Kn * delta_n  # compressive penetration penalty (no damage)
    )

    Tt_mag = np.where(
        delta_t > EM20,
        Kt * delta_t * (1.0 - dama),
        0.0
    )
    safe_dt = np.where(delta_t > EM20, delta_t, 1.0)
    t_tangent = delta_t_vec / safe_dt[:, None]

    # Total interface force vector per unit area: T = Tn * n + Tt * t
    T_vec = Tn[:, None] * n_mid + Tt_mag[:, None] * t_tangent  # (n, 3)

    # Negated internal force convention:
    # Under positive opening, top nodes receive downward resisting force (-f_node)
    # and bottom nodes receive upward restoring force (+f_node)
    f_node = 0.25 * area[:, None] * T_vec  # (n, 3)
    fe = np.zeros((n, 8, 3), dtype=np.float64)
    for i in range(4):
        fe[:, i, :] = f_node       # bottom nodes
        fe[:, i + 4, :] = -f_node  # top nodes


    # Critical time step (sz_dt1.F90): dt = 2 / omega_max = sqrt(rho * h_eff / K_max)
    h_eff = np.sqrt(np.maximum(area, EM20))
    rho = st["mass"] / np.maximum(area * h_eff, EM20)
    K_max = np.maximum(Kn, Kt)
    dt_crit = 2.0 * np.sqrt(np.maximum(rho * h_eff / np.maximum(K_max, EM20), EM20))
    dt_crit = np.where(alive, dt_crit, EP30)

    # Scatter forces into global fint
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit
