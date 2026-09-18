"""
6-node Prismatic Wedge Solid Element with HEPH Physical Stabilization (/PENTA6 + HEPH).

Fortran origin: ``engine/source/elements/solid/solide6z/``
    s6zforc3.F90   driver: kinematics, physical stabilization, internal forces
    s6zhourg3.F90  HEPH physical hourglass stabilization for wedge geometry
    s6zderi3.F90   shape function derivatives for 6-node prism
    s6zdefo3.F90   rate of deformation and spin tensors

Theory notes:
* 6-node triangular prism / wedge element with 2 triangular faces and 3 quadrilateral sides.
* While the triangular cross-section is naturally free of in-plane hourglass modes,
  the axial coupling between bottom and top faces generates zero-energy modes in bending
  and axial-shear distortion.
* HEPH physical stabilization (Belytschko-Bindeman applied to wedges) computes the
  orthogonal hourglass vectors gamma from the element geometry and applies physical
  stiffness proportional to the material shear modulus:
      k_hg = hcoef * G * V * sum |gradN|^2
      f_hg = - k_hg * (gamma . u) * gamma
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3
from .solid_penta6 import _DN_DXI, _FACES, _char_length, _geometry


def init_group(group, model, log):
    """Starter initialization for 6-node HEPH wedge solid group."""
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        group.state.update(
            sig=np.zeros((0, 2, 6)),
            epsp=np.zeros((0, 2)),
            vol0=np.zeros(0),
            vol_g0=np.zeros((0, 2)),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            off=np.zeros(0),
            qvw_pend=np.zeros(0),
            hgqex=np.zeros((0, 2, 3)),
            dtfac=np.zeros(0),
            chk_fail=False,
            dama=np.zeros(0),
        )
        return np.zeros(0, dtype=np.int64), np.zeros(0), None

    xe = model.x0[conn]
    dndx0, vol_g0, vol0 = _geometry(xe)

    bad = vol0 <= EM20
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/PENTA6(HEPH) {eid}: zero or negative volume", "SOLID INIT")

    rho0 = np.zeros(n)
    for sl, mat, prop in group.state.get("slices", []):
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol0

    lc0 = _char_length(xe, vol0)
    dtfac = np.full(n, 0.9)

    group.state.update(
        sig=np.zeros((n, 2, 6)),
        epsp=np.zeros((n, 2)),
        vol0=vol0.copy(),
        vol_g0=vol_g0.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),
        qvw_pend=np.zeros(n),
        hgqex=np.zeros((n, 2, 3)),     # modal displacements for 2 wedge hourglass modes
        dtfac=dtfac,
        chk_fail=False,
        dama=np.zeros(n),
    )

    for sl, mat, prop in group.state.get("slices", []):
        if getattr(mat, "fail_models", None) or getattr(mat, "eps_p_max", 0.0) > 0.0:
            group.state["chk_fail"] = True

    node_idx = conn.reshape(-1)
    mass_c = np.repeat(mass / 6.0, 6)
    return node_idx, mass_c, None


def _wedge_hg_modes(xe: np.ndarray, dndx: np.ndarray):
    """Compute HEPH physical hourglass vectors for the 6-node wedge.

    The 2 hourglass modes of the wedge represent out-of-phase triangular bending:
        h1 = [1, -1, 0, -1, 1, 0]
        h2 = [1, 1, -2, -1, -1, 2]
    Gamma is obtained by projecting h orthogonal to the linear velocity field:
        gamma = h - sum_i h_i x_i . gradN
    """
    n = len(xe)
    H = np.array([
        [1.0, -1.0, 0.0, -1.0, 1.0, 0.0],
        [1.0, 1.0, -2.0, -1.0, -1.0, 2.0],
    ], dtype=np.float64)  # (2, 6)

    # Average Cartesian gradients across the 2 Gauss points: (n, 6, 3)
    dndx_avg = np.mean(dndx, axis=1)

    # hx[n, m, b] = sum_i H[m, i] * xe[n, i, b] -> (n, 2, 3)
    hx = np.einsum("mi,nib->nmb", H, xe)

    # gamma[n, m, i] = H[m, i] - sum_b hx[n, m, b] * dndx_avg[n, i, b]
    hx_dndx = np.einsum("nmb,nib->nmi", hx, dndx_avg)
    gamma = H[None, :, :] - hx_dndx  # (n, 2, 6)

    return gamma


def forces(group, x, v, vr, dt, fint, mint):
    """Engine explicit cycle force kernel for 6-node HEPH wedge elements."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is None or dt < 0.0:
        return np.full(n, EP30)

    xe = x[conn]
    ve = v[conn] if v is not None else np.zeros_like(xe)

    dndx, vol_g, vol_tot = _geometry(xe)
    vol_tot_safe = np.maximum(vol_tot, EM20)
    lc = _char_length(xe, vol_tot_safe)

    # Probe cycle 0
    if dt == 0.0 or v is None:
        rho = st["mass"] / vol_tot_safe
        c = np.zeros(n)
        for sl, mat, _ in st.get("slices", []):
            K_sl = getattr(mat, "K", 0.0)
            G_sl = getattr(mat, "G", 0.0)
            c[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))
        return np.where(c > 0.0, st["dtfac"] * lc / c, EP30)

    sig = st["sig"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    # Velocity gradient at 2 GPs
    L = np.einsum("nib,ngic->ngbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 1, 3, 2)))
    trD = D[:, :, 0, 0] + D[:, :, 1, 1] + D[:, :, 2, 2]

    W = 0.5 * (L - np.transpose(L, (0, 1, 3, 2)))

    deps = np.empty((n, 2, 6), dtype=np.float64)
    deps[:, :, 0] = D[:, :, 0, 0] * dt
    deps[:, :, 1] = D[:, :, 1, 1] * dt
    deps[:, :, 2] = D[:, :, 2, 2] * dt
    deps[:, :, 3] = 2.0 * D[:, :, 0, 1] * dt
    deps[:, :, 4] = 2.0 * D[:, :, 1, 2] * dt
    deps[:, :, 5] = 2.0 * D[:, :, 0, 2] * dt

    if not np.all(alive):
        deps[~alive] = 0.0
        trD[~alive] = 0.0

    # Jaumann stress rotation at 2 GPs
    wxy = W[:, :, 0, 1] * dt
    wyz = W[:, :, 1, 2] * dt
    wxz = W[:, :, 0, 2] * dt

    sxx = sig[:, :, 0].copy()
    syy = sig[:, :, 1].copy()
    szz = sig[:, :, 2].copy()
    sxy = sig[:, :, 3].copy()
    syz = sig[:, :, 4].copy()
    szx = sig[:, :, 5].copy()

    sig[:, :, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, :, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, :, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, :, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, :, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, :, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    # Constitutive update at 2 GPs
    c_sound = np.zeros(n)
    G_mod = np.zeros(n)
    qa = np.zeros(n)
    qb = np.zeros(n)
    rho = st["mass"] / vol_tot_safe

    for sl, mat, prop in st.get("slices", []):
        qa[sl] = getattr(prop, "qa", 1.1) if prop else 1.1
        qb[sl] = getattr(prop, "qb", 0.05) if prop else 0.05

        law = getattr(mat, "law", 1)
        if law == 0:
            sig[sl] = 0.0
            continue

        K_sl = getattr(mat, "K", 0.0)
        G_sl = getattr(mat, "G", 0.0)
        G_mod[sl] = G_sl
        c_sound[sl] = np.sqrt(np.maximum(K_sl + 4.0 * G_sl / 3.0, 0.0) / np.maximum(rho[sl], EM20))

        for g in range(2):
            materials.solid_update(mat, sig[sl, g], deps[sl, g], st["epsp"][sl, g], dt, None)

    # Bulk viscosity
    compressing = (trD < 0.0) & alive[:, None]
    qvisc = np.where(
        compressing,
        rho[:, None] * lc[:, None] * (qa[:, None]**2 * lc[:, None] * trD**2 - qb[:, None] * c_sound[:, None] * trD),
        0.0)

    sig_tot = sig.copy()
    sig_tot[:, :, 0] -= qvisc
    sig_tot[:, :, 1] -= qvisc
    sig_tot[:, :, 2] -= qvisc

    # Internal force assembly from 2 GPs
    S = np.empty((n, 2, 3, 3), dtype=np.float64)
    S[:, :, 0, 0], S[:, :, 1, 1], S[:, :, 2, 2] = sig_tot[:, :, 0], sig_tot[:, :, 1], sig_tot[:, :, 2]
    S[:, :, 0, 1] = S[:, :, 1, 0] = sig_tot[:, :, 3]
    S[:, :, 1, 2] = S[:, :, 2, 1] = sig_tot[:, :, 4]
    S[:, :, 0, 2] = S[:, :, 2, 0] = sig_tot[:, :, 5]

    fe = -np.einsum("ng,ngbc,ngic->nib", vol_g, S, dndx)

    # HEPH physical hourglass stabilization (s6zhourg3.F90)
    gamma = _wedge_hg_modes(xe, dndx)  # (n, 2, 6)
    # trace of B^T B
    dndx_avg = np.mean(dndx, axis=1)
    trace_b = np.einsum("nia,nia->n", dndx_avg, dndx_avg)
    kstiff = 0.05 * G_mod * vol_tot_safe * trace_b * alive  # (n,)

    # Update modal displacements: hgqex += (gamma @ ve) * dt
    # gamma @ ve: (n, 2, 6) @ (n, 6, 3) -> (n, 2, 3)
    hg_rate = np.einsum("nmi,nib->nmb", gamma, ve)
    st["hgqex"] += hg_rate * dt * alive[:, None, None]

    # Hourglass force: f_hg = - kstiff * gamma^T @ hgqex
    # (n, 6, 2) @ (n, 2, 3) -> (n, 6, 3)
    fhg = -kstiff[:, None, None] * np.einsum("nmi,nmb->nib", gamma, st["hgqex"])
    fe += fhg

    # Energy bookkeeping
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = np.sum(vol_g * 0.5 * qvisc * (-trD * dt), axis=1) + st["qvw_pend"] * np.mean(-trD, axis=1)
    deint0 = np.sum(vol_g * np.einsum("nga,nga->ng", sig_mid, deps), axis=1) + w_visc
    dehour = -np.einsum("nib,nib->n", fhg, ve) * dt
    st["qvw_pend"] = np.sum(vol_g * 0.5 * qvisc * dt, axis=1)
    st["eint"] += deint0
    st["ehour"] += dehour

    # Courant time step
    trD_min = np.min(trD, axis=1)
    Q = np.where(trD_min < 0.0, qb * c_sound + qa * lc * np.abs(trD_min), 0.0)
    denom = Q + np.sqrt(Q * Q + c_sound * c_sound)
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, st["dtfac"] * lc / safe_denom, EP30)
    dt_crit = np.where(alive, dt_crit, EP30)

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit
