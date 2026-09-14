"""DKT18 (Discrete Kirchhoff Triangle) 3-node shell element.

Fortran origin:
  Starter: starter/source/elements/sh3n/coquedk/
    - cdkinit3.F: group initialization
    - cmaini3.F: lumped mass and rotary inertia
  Engine: engine/source/elements/sh3n/coquedk/
    - cdkforc3.F: driver for internal forces and time step
    - cdkcoor3.F / clskew3.F: local frame and coordinates
    - cdkderic3.F: geometric derivatives and constants
    - cdkdefo3.F: membrane strain rates
    - cdkderi3.F: DKT bending shape function derivatives at 3 Hammer points
    - cdkcurv3.F: curvature rates
    - cdkfint3.F: internal forces/moments at Gauss points
    - cdkfcum3.F: accumulation and transformation to global coordinates
    - cndt3.F: critical Courant time step

Kinematics & Implicit Extensions (M47, M511):
  - 3 Hammer integration points in-plane for DKT bending
  - 18 DOFs per element: [ux, uy, uz, thx, thy, thz] x 3 nodes
  - Full rank 12 tangent stiffness matrix (3 membrane + 6 bending + 3 drilling)
  - 6 exact rigid-body null modes (3 translations + 3 rotations)
  - Analytical consistent mass matrix (translational + rotary bending inertia)
  - Geometric (initial-stress) stiffness matrix K_geo
  - Static & implicit internal force evaluation for updated-Lagrangian solver
"""

import logging
import numpy as np
from ..common.constants import EM20, EP30
from .shell_bt4 import _init_material_state, _layer_extra, _DRILL_COEF
from .shell_tri3 import _local_geometry, _char_length, _exact_dt_factor
from ..accel.jit_kernels.shells_dkt18 import (
    cdkcoor3, cdkderic3, cdkdefo3, cdkderi3, cdkcurv3, cdkfint3, cdkfcum3
)
from .shell_bt4 import _layer_extra, _layer_failure
from .. import materials

log = logging.getLogger(__name__)

#: 3 Hammer integration points in area coordinates (eta, ksi)
_A_HAMMER = [
    (0.166666666666667, 0.666666666666667),
    (0.666666666666667, 0.166666666666667),
    (0.166666666666667, 0.166666666666667),
]

#: CST shape function product integral S_ij = int N_i N_j dA / A
_S_TRI = (np.ones((3, 3)) + np.eye(3)) / 12.0


def _edofs(conn):
    """(n, 18) global scalar DOF slot ids, node-major [ux..rz] * 3 nodes."""
    n = len(conn)
    edofs = np.empty((n, 18), dtype=np.int64)
    for i in range(3):
        for c in range(6):
            edofs[:, i * 6 + c] = conn[:, i] * 6 + c
    return edofs


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter c3init3/c3mass3)."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float), np.empty(0, dtype=float)

    xe = model.x0[group.conn]
    E, xl, area, B1, B2 = _local_geometry(xe)
    bad = area <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SH3N_DKT18 {eid}: zero area (coincident nodes?)",
                      "DKT18 INIT")

    thick = np.zeros(n)
    rho0 = np.zeros(n)
    ssp0 = np.zeros(n)
    amu = np.zeros(n)
    nip_max = 1
    _DN_DEFAULT = 1e-3
    slices = group.state.get("slices", [])
    for sl, mat, prop in slices:
        t_val = prop.params.get("thick", getattr(prop, "thick", 1.0)) if hasattr(prop, "params") else getattr(prop, "thick", 1.0)
        nip_val = int(prop.params.get("nip", getattr(prop, "nip", 1))) if hasattr(prop, "params") else int(getattr(prop, "nip", 1))
        thick[sl] = t_val
        rho0[sl] = getattr(mat, "rho0", 0.0)
        ssp0[sl] = mat.sound_speed_shell() if (hasattr(mat, "sound_speed_shell") and getattr(mat, "law", 1) != 0) else 0.0
        amu_val = float(prop.params.get("dn", 0.0) or 0.0) if hasattr(prop, "params") else _DN_DEFAULT
        amu[sl] = amu_val if amu_val != 0.0 else _DN_DEFAULT
        nip_max = max(nip_max, nip_val)
    mass = rho0 * thick * area

    # Through-thickness Gauss stations per part slice (same as shell_bt4)
    zw = []
    for sl, mat, prop in slices:
        nip_val = int(prop.params.get("nip", getattr(prop, "nip", 1))) if hasattr(prop, "params") else int(getattr(prop, "nip", 1))
        gp, gw = np.polynomial.legendre.leggauss(nip_val)
        zw.append((gp * 0.5, gw * 0.5))
    nk = 3 * nip_max
    group.state.update(
        sig=np.zeros((n, nk, 3)),        # in-plane stress per layer x 3 GP
        qshear=np.zeros((n, 2)),         # transverse shear stress (elastic)
        epsp=np.zeros((n, nk)),
        thick=thick,
        area0=area.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        off=np.ones(n),                  # 1 alive / 0 deleted
        zw=zw,
        dtfac=_exact_dt_factor(B1, B2, area, _char_length(xl, area),
                               thick, slices),
        ssp0=ssp0,
        amu=amu,
        nip_max=nip_max,
    )
    _init_material_state(group, nk)
    # orthotropy fiber frame (/PROP/TYPE9 SH_ORTH, TYPE16) — see shell_bt4
    from . import shell_ortho
    group.state["ortho"] = shell_ortho.build_group_ortho(
        slices, E, n, log, group.ids)
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 3.0, 3)
    # generous lumped rotational inertia (Key's trick, see module docstring)
    group.state["dt_iner"] = mass / 3.0 * (area / 4.5 + thick ** 2 / 12.0)
    inertia_c = np.repeat(group.state["dt_iner"], 3)
    return node_idx, mass_c, inertia_c


def forces(group, x, v, vr, dt, fint, mint):
    """Compute DKT18 internal forces and critical time step."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)

    xe = x[conn]
    thick = st["thick"]
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    # Cycle 0 or velocity-free evaluation
    if dt is None or dt <= 0.0 or v is None:
        ve0 = np.zeros((n, 3, 3), dtype=float)
        re0 = np.zeros((n, 3, 3), dtype=float)
        area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly, _e_frame = cdkcoor3(xe, ve0, re0, 0.0)
        area2 = np.maximum(area2, EM20)
        vol0 = 0.5 * area2 * thick
        alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
            xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)
        viscdt = np.sqrt(1.0 + st["amu"]**2) - st["amu"]
        dt_e = st["dtfac"] * aldt * viscdt / np.maximum(st["ssp0"], EM20)
        alive = st["off"] > 0.0
        return np.where(alive & (st["ssp0"] > 0.0), dt_e, EP30)

    ve = v[conn]
    re = vr[conn] if vr is not None else np.zeros((n, 3, 3), dtype=float)
    nip_max = st.get("nip_max", 1)

    # 1. Geometry and local frame
    (area2, xl2, yl2, xl3, yl3, vlx, vly, vlz, rlx, rly,
     e_frame) = cdkcoor3(xe, ve, re, dt)
    e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z = e_frame

    # 2. Derivatives and interpolation constants
    area = 0.5 * np.maximum(area2, EM20)
    vol0 = area * thick
    vol00 = vol0.copy()
    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol00, nu, thick**2)

    # 3. Membrane rates
    exx = np.zeros(n)
    eyy = np.zeros(n)
    exy = np.zeros(n)
    exz = np.zeros(n)
    eyz = np.zeros(n)
    epsdot = np.zeros((3, n))
    gstr = np.zeros((n, 3))
    vdef = np.zeros((n, 3))

    cdkdefo3(vlx, vly, px2, py2, px3, py3, exx, eyy, exy, exz, eyz, dt, epsdot, 0, False, gstr, vdef, False)

    f11 = np.zeros(n)
    f12 = np.zeros(n)
    f13 = np.zeros(n)
    f21 = np.zeros(n)
    f22 = np.zeros(n)
    f23 = np.zeros(n)
    f31 = np.zeros(n)
    f32 = np.zeros(n)
    f33 = np.zeros(n)

    m11 = np.zeros(n)
    m12 = np.zeros(n)
    m13 = np.zeros(n)
    m21 = np.zeros(n)
    m22 = np.zeros(n)
    m23 = np.zeros(n)
    m31 = np.zeros(n)
    m32 = np.zeros(n)
    m33 = np.zeros(n)

    epsp_old = st["epsp"].copy() if st.get("chk_fail", False) else None
    nip_of = []

    for NG in range(3):
        eta, ksi = _A_HAMMER[NG]

        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)

        kxx = np.zeros(n)
        kyy = np.zeros(n)
        kxy = np.zeros(n)
        cdkcurv3(bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3, vlz, rlx, rly, kxx, kyy, kxy)

        dexx_bend = kxx * dt
        deyy_bend = kyy * dt
        dexy_bend = kxy * dt

        vol_gp = vol0 / 3.0
        force_pg = np.zeros((n, 3))
        mom_pg = np.zeros((n, 3))

        for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
            if NG == 0:
                if getattr(mat, "law", 1) == 0 or isl >= len(st.get("zw", [])):
                    nip_of.append(0)
                else:
                    nip_of.append(len(st["zw"][isl][0]))
            mask = sl
            if not np.any(mask):
                continue
            if getattr(mat, "law", 1) == 0:
                continue

            zrel, wrel = st["zw"][isl]
            nip = len(zrel)

            for il in range(nip):
                k = NG * nip_max + il
                z = zrel[il] * thick[mask]
                gw = wrel[il]

                # Strain increment at layer k
                deps = np.stack([
                    exx[mask] + z * dexx_bend[mask],
                    eyy[mask] + z * deyy_bend[mask],
                    exy[mask] + z * dexy_bend[mask]
                ], axis=1)

                st_sig_k = st["sig"][mask, k, :].copy()
                st_epsp_k = st["epsp"][mask, k].copy()

                sig_new, epsp_new = materials.shell_update(
                    mat, st_sig_k, deps, st_epsp_k, dt, _layer_extra(st, mask, k))

                st["epsp"][mask, k] = epsp_new
                if st.get("chk_fail", False):
                    _layer_failure(st, mask, mat, k, sig_new, epsp_old, deps, dt)

                st["sig"][mask, k, :] = sig_new

                force_pg[mask] += sig_new * gw
                mom_pg[mask] += sig_new * zrel[il] * gw

                dz_vol = gw * vol_gp[mask]
                st["eint"][mask] += np.sum(0.5 * (st_sig_k + sig_new) * deps, axis=1) * dz_vol

        cdkfint3(vol_gp, thick, force_pg, mom_pg, px2, py2, px3, py3,
                 bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3,
                 f11, f12, f13, f21, f22, f23, f32, f33,
                 m11, m12, m13, m21, m22, m23)

    if st.get("chk_fail", False):
        from .shell_bt4 import _element_deletion
        alive = _element_deletion(st, nip_of)
        if not alive.all():
            dead = ~alive
            f11[dead] = 0.0
            f12[dead] = 0.0
            f13[dead] = 0.0
            f21[dead] = 0.0
            f22[dead] = 0.0
            f23[dead] = 0.0
            f31[dead] = 0.0
            f32[dead] = 0.0
            f33[dead] = 0.0
            m11[dead] = 0.0
            m12[dead] = 0.0
            m13[dead] = 0.0
            m21[dead] = 0.0
            m22[dead] = 0.0
            m23[dead] = 0.0
            m31[dead] = 0.0
            m32[dead] = 0.0
            m33[dead] = 0.0

    cdkfcum3(px2, py2, px3, py3, e1x, e2x, e3x, e1y, e2y, e3y, e1z, e2z, e3z,
             f11, f12, f13, f21, f22, f23, f31, f32, f33,
             m11, m12, m13, m21, m22, m23, m31, m32, m33)

    if fint is not None:
        np.subtract.at(fint[:, 0], conn[:, 0], f11)
        np.subtract.at(fint[:, 1], conn[:, 0], f21)
        np.subtract.at(fint[:, 2], conn[:, 0], f31)
        np.subtract.at(fint[:, 0], conn[:, 1], f12)
        np.subtract.at(fint[:, 1], conn[:, 1], f22)
        np.subtract.at(fint[:, 2], conn[:, 1], f32)
        np.subtract.at(fint[:, 0], conn[:, 2], f13)
        np.subtract.at(fint[:, 1], conn[:, 2], f23)
        np.subtract.at(fint[:, 2], conn[:, 2], f33)

    if mint is not None:
        np.subtract.at(mint[:, 0], conn[:, 0], m11)
        np.subtract.at(mint[:, 1], conn[:, 0], m21)
        np.subtract.at(mint[:, 2], conn[:, 0], m31)
        np.subtract.at(mint[:, 0], conn[:, 1], m12)
        np.subtract.at(mint[:, 1], conn[:, 1], m22)
        np.subtract.at(mint[:, 2], conn[:, 1], m32)
        np.subtract.at(mint[:, 0], conn[:, 2], m13)
        np.subtract.at(mint[:, 1], conn[:, 2], m23)
        np.subtract.at(mint[:, 2], conn[:, 2], m33)

    viscdt = np.sqrt(1.0 + st["amu"]**2) - st["amu"]
    dt_e = st["dtfac"] * aldt * viscdt / np.maximum(st["ssp0"], EM20)
    alive = st["off"] > 0.0
    return np.where(alive & (st["ssp0"] > 0.0), dt_e, EP30)


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole DKT18 group.

    Combines CST membrane, DKT bending at 3 Hammer points, and drilling penalty.
    Returns (ke, edofs): ke (n, 18, 18) in global coordinates, edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    ve0 = np.zeros((n, 3, 3), dtype=float)
    re0 = np.zeros((n, 3, 3), dtype=float)
    (area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly,
     e_frame) = cdkcoor3(xe, ve0, re0, 0.0)
    area = 0.5 * np.maximum(area2, EM20)
    thick = st["thick"]
    vol0 = area * thick
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)

    # Membrane operator Bm (n, 3, 6) on [ux0, uy0, ux1, uy1, ux2, uy2]
    px1 = -(px2 + px3)
    py1 = -(py2 + py3)

    Bm = np.zeros((n, 3, 6))
    Bm[:, 0, 0] = px1
    Bm[:, 0, 2] = px2
    Bm[:, 0, 4] = px3
    Bm[:, 1, 1] = py1
    Bm[:, 1, 3] = py2
    Bm[:, 1, 5] = py3
    Bm[:, 2, 0] = py1
    Bm[:, 2, 1] = px1
    Bm[:, 2, 2] = py2
    Bm[:, 2, 3] = px2
    Bm[:, 2, 4] = py3
    Bm[:, 2, 5] = px3

    # DKT Bending operators Bb (n, 3, 9) at 3 Hammer points
    # on [uz0, thx0, thy0, uz1, thx1, thy1, uz2, thx2, thy2]
    Bb_gp = []
    for NG in range(3):
        eta, ksi = _A_HAMMER[NG]
        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)
        B = np.zeros((n, 3, 9))
        B[:, 0, 0] = -(bz1[:, 0] + bz1[:, 1])
        B[:, 0, 1] = brx1[:, 0]
        B[:, 0, 2] = bry1[:, 0]
        B[:, 0, 3] = bz1[:, 0]
        B[:, 0, 4] = brx1[:, 1]
        B[:, 0, 5] = bry1[:, 1]
        B[:, 0, 6] = bz1[:, 1]
        B[:, 0, 7] = brx1[:, 2]
        B[:, 0, 8] = bry1[:, 2]

        B[:, 1, 0] = -(bz2[:, 0] + bz2[:, 1])
        B[:, 1, 1] = brx2[:, 0]
        B[:, 1, 2] = bry2[:, 0]
        B[:, 1, 3] = bz2[:, 0]
        B[:, 1, 4] = brx2[:, 1]
        B[:, 1, 5] = bry2[:, 1]
        B[:, 1, 6] = bz2[:, 1]
        B[:, 1, 7] = brx2[:, 2]
        B[:, 1, 8] = bry2[:, 2]

        B[:, 2, 0] = -(bz3[:, 0] + bz3[:, 1])
        B[:, 2, 1] = brx3[:, 0]
        B[:, 2, 2] = bry3[:, 0]
        B[:, 2, 3] = bz3[:, 0]
        B[:, 2, 4] = brx3[:, 1]
        B[:, 2, 5] = bry3[:, 1]
        B[:, 2, 6] = bz3[:, 1]
        B[:, 2, 7] = brx3[:, 2]
        B[:, 2, 8] = bry3[:, 2]

        Bb_gp.append(B)

    # Local stiffness accumulation (n, 18, 18)
    Kl = np.zeros((n, 18, 18))
    kdrill = np.zeros(n)
    mem_dofs = [0, 1, 6, 7, 12, 13]
    bend_dofs = [2, 3, 4, 8, 9, 10, 14, 15, 16]
    nip_max = st.get("nip_max", 1)

    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        if getattr(mat, "law", 1) == 0:
            continue
        t_sl = thick[sl]
        A_sl = area[sl]
        m_count = np.count_nonzero(sl) if isinstance(sl, np.ndarray) and sl.dtype == bool else (sl.stop - sl.start if isinstance(sl, slice) else len(sl))
        if m_count == 0:
            continue

        Bms = Bm[sl]
        if getattr(mat, "law", 1) == 1:
            C = materials.shell_membrane_tangent(mat)
            Km = (A_sl * t_sl)[:, None, None] * np.einsum("nai,ab,nbj->nij", Bms, C, Bms)
            Kl[sl, :, :][np.ix_(range(m_count), mem_dofs, mem_dofs)] += Km

            D_bend = (t_sl ** 3 / 12.0)[:, None, None] * C[None, :, :]
            for NG in range(3):
                Bbs = Bb_gp[NG][sl]
                Kb_ng = (A_sl / 3.0)[:, None, None] * np.einsum("nai,nab,nbj->nij", Bbs, D_bend, Bbs)
                Kl[sl, :, :][np.ix_(range(m_count), bend_dofs, bend_dofs)] += Kb_ng
        else:
            # Multi-layer elastoplastic integration
            zrel, wrel = st["zw"][isl]
            nip = len(zrel)
            for NG in range(3):
                Am_ng = np.zeros((m_count, 3, 3))
                Bm_ng = np.zeros((m_count, 3, 3))
                Dm_ng = np.zeros((m_count, 3, 3))
                for il in range(nip):
                    k = NG * nip_max + il
                    zk = zrel[il] * t_sl
                    wk = (wrel[il] * t_sl) / 3.0
                    dep_k = None if epsp_incr is None else epsp_incr[sl, k]
                    Dk = materials.shell_layer_tangent(
                        mat, st["sig"][sl, k, :], st["epsp"][sl, k], dep_k,
                        extra=_layer_extra(st, sl, k))
                    Am_ng += wk[:, None, None] * Dk
                    Bm_ng += (wk * zk)[:, None, None] * Dk
                    Dm_ng += (wk * zk * zk)[:, None, None] * Dk

                Bbs = Bb_gp[NG][sl]
                A3 = A_sl[:, None, None]
                # Membrane-membrane
                Km_ng = A3 * np.einsum("nai,nab,nbj->nij", Bms, Am_ng, Bms)
                Kl[sl, :, :][np.ix_(range(m_count), mem_dofs, mem_dofs)] += Km_ng
                # Bending-bending
                Kb_ng = A3 * np.einsum("nai,nab,nbj->nij", Bbs, Dm_ng, Bbs)
                Kl[sl, :, :][np.ix_(range(m_count), bend_dofs, bend_dofs)] += Kb_ng
                # Membrane-bending coupling
                Kmb_ng = A3 * np.einsum("nai,nab,nbj->nij", Bms, Bm_ng, Bbs)
                Kl[sl, :, :][np.ix_(range(m_count), mem_dofs, bend_dofs)] += Kmb_ng
                Kl[sl, :, :][np.ix_(range(m_count), bend_dofs, mem_dofs)] += np.swapaxes(Kmb_ng, 1, 2)

        kdrill[sl] = _DRILL_COEF * getattr(mat, "E", 0.0) * t_sl ** 3 * A_sl / 12.0

    # Drilling penalty coupled to continuum spin omega = 0.5 * (B1.vy - B2.vx)
    px_all = [px1, px2, px3]
    py_all = [py1, py2, py3]
    wrow = np.zeros((n, 18))
    for j in range(3):
        wrow[:, j * 6 + 0] = -0.5 * py_all[j]
        wrow[:, j * 6 + 1] = 0.5 * px_all[j]

    for i in range(3):
        gi = -wrow.copy()
        gi[:, i * 6 + 5] += 1.0
        Kl += kdrill[:, None, None] * np.einsum("ni,nj->nij", gi, gi)

    # Transform from local to global frame: R = [e1, e2, e3]
    e1 = np.stack([e_frame[0], e_frame[1], e_frame[2]], axis=1)  # (n, 3)
    e2 = np.stack([e_frame[3], e_frame[4], e_frame[5]], axis=1)  # (n, 3)
    e3 = np.stack([e_frame[6], e_frame[7], e_frame[8]], axis=1)  # (n, 3)
    R = np.stack([e1, e2, e3], axis=2)                            # (n, 3, 3)

    Kl_blocks = Kl.reshape(n, 6, 3, 6, 3)
    ke = np.einsum("nap,nIpJq,nbq->nIaJb", R, Kl_blocks, R).reshape(n, 18, 18)

    # Deactivated elements
    alive = st["off"] > 0.0
    ke[~alive] = 0.0

    return ke, _edofs(conn)


def consistent_mass(group, x=None):
    """Consistent element mass matrix for the DKT18 shell.

    Translational block: rho t S (x) I3
    Rotational block: rho t (t^2 / 12) S (x) I3
    Returns (me, edofs): me (n, 18, 18) strictly positive definite, edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    mass = st["mass"]
    thick = st["thick"]
    m_trans = mass
    m_rot = mass * thick ** 2 / 12.0

    me = np.zeros((n, 18, 18))
    for a in range(3):
        for b in range(3):
            s = _S_TRI[a, b]
            ft = m_trans * s
            fr = m_rot * s
            for c in range(3):
                me[:, a * 6 + c, b * 6 + c] = ft
                me[:, a * 6 + 3 + c, b * 6 + 3 + c] = fr

    alive = st.get("off")
    if alive is not None:
        me[alive <= 0.0] = 0.0

    return me, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the DKT18 group.

    Membrane-resultant von-Karman operator over translations.
    Returns (ke, edofs): ke (n, 18, 18), edofs (n, 18).
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 18, 18), dtype=float), np.zeros((0, 18), dtype=np.int64)

    xe = x[conn]
    ve0 = np.zeros((n, 3, 3), dtype=float)
    re0 = np.zeros((n, 3, 3), dtype=float)
    (area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly,
     _e_frame) = cdkcoor3(xe, ve0, re0, 0.0)
    area = 0.5 * np.maximum(area2, EM20)
    thick = st["thick"]
    vol0 = area * thick
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)

    # Membrane force resultants N = sum_g 1/3 sum_k w_k sigma_k (force/length)
    sig = st["sig"]
    Nres = np.zeros((n, 3))
    nip_max = st.get("nip_max", 1)
    for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        nip = len(zrel)
        for NG in range(3):
            for il in range(nip):
                k = NG * nip_max + il
                wk = (wrel[il] * t_sl) / 3.0
                Nres[sl] += wk[:, None] * sig[sl, k, :]

    px1 = -(px2 + px3)
    py1 = -(py2 + py3)
    B1 = np.stack([px1, px2, px3], axis=1)  # (n, 3)
    B2 = np.stack([py1, py2, py3], axis=1)  # (n, 3)

    # g_ab = A (B1a B1b Nxx + B2a B2b Nyy + (B1a B2b + B2a B1b) Nxy)
    g = area[:, None, None] * (
        Nres[:, 0, None, None] * B1[:, :, None] * B1[:, None, :]
        + Nres[:, 1, None, None] * B2[:, :, None] * B2[:, None, :]
        + Nres[:, 2, None, None] * (B1[:, :, None] * B2[:, None, :]
                                    + B2[:, :, None] * B1[:, None, :]))

    ke = np.zeros((n, 18, 18))
    ni = 6 * np.arange(3)
    for c in range(3):
        rows = (ni + c)[:, None]
        cols = (ni + c)[None, :]
        ke[:, rows, cols] += g

    alive = st["off"] > 0.0
    ke[~alive] = 0.0
    return ke, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal forces/moments at configuration x from CURRENT layer stresses."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return
    if fint is None and mint is None:
        return

    xe = x[conn]
    ve0 = np.zeros((n, 3, 3), dtype=float)
    re0 = np.zeros((n, 3, 3), dtype=float)
    (area2, xl2, yl2, xl3, yl3, _vlx, _vly, _vlz, _rlx, _rly,
     e_frame) = cdkcoor3(xe, ve0, re0, 0.0)
    e1x, e1y, e1z, e2x, e2y, e2z, e3x, e3y, e3z = e_frame

    thick = st["thick"]
    nu = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        nu[sl] = getattr(mat, "nu", 0.3)

    area = 0.5 * np.maximum(area2, EM20)
    vol0 = area * thick
    alpe, aldt, px2, py2, px3, py3, px, py, pxy, pyy, _vol00 = cdkderic3(
        xl2, yl2, xl3, yl3, area2, vol0.copy(), nu, thick**2)

    f11 = np.zeros(n)
    f12 = np.zeros(n)
    f13 = np.zeros(n)
    f21 = np.zeros(n)
    f22 = np.zeros(n)
    f23 = np.zeros(n)
    f31 = np.zeros(n)
    f32 = np.zeros(n)
    f33 = np.zeros(n)

    m11 = np.zeros(n)
    m12 = np.zeros(n)
    m13 = np.zeros(n)
    m21 = np.zeros(n)
    m22 = np.zeros(n)
    m23 = np.zeros(n)
    m31 = np.zeros(n)
    m32 = np.zeros(n)
    m33 = np.zeros(n)

    nip_max = st.get("nip_max", 1)
    vol_gp = vol0 / 3.0

    for NG in range(3):
        eta, ksi = _A_HAMMER[NG]
        bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3 = cdkderi3(
            px2, py2, px3, py3, px, py, pxy, pyy, ksi, eta)

        force_pg = np.zeros((n, 3))
        mom_pg = np.zeros((n, 3))

        for isl, (sl, mat, prop) in enumerate(st.get("slices", [])):
            mask = sl
            if not np.any(mask):
                continue
            if getattr(mat, "law", 1) == 0:
                continue
            zrel, wrel = st["zw"][isl]
            nip = len(zrel)
            for il in range(nip):
                k = NG * nip_max + il
                gw = wrel[il]
                sig_k = st["sig"][mask, k, :]
                force_pg[mask] += sig_k * gw
                mom_pg[mask] += sig_k * zrel[il] * gw

        cdkfint3(vol_gp, thick, force_pg, mom_pg, px2, py2, px3, py3,
                 bz1, bz2, bz3, brx1, brx2, brx3, bry1, bry2, bry3,
                 f11, f12, f13, f21, f22, f23, f32, f33,
                 m11, m12, m13, m21, m22, m23)

    alive = st["off"] > 0.0
    if not alive.all():
        dead = ~alive
        f11[dead] = 0.0
        f12[dead] = 0.0
        f13[dead] = 0.0
        f21[dead] = 0.0
        f22[dead] = 0.0
        f23[dead] = 0.0
        f31[dead] = 0.0
        f32[dead] = 0.0
        f33[dead] = 0.0
        m11[dead] = 0.0
        m12[dead] = 0.0
        m13[dead] = 0.0
        m21[dead] = 0.0
        m22[dead] = 0.0
        m23[dead] = 0.0
        m31[dead] = 0.0
        m32[dead] = 0.0
        m33[dead] = 0.0

    cdkfcum3(px2, py2, px3, py3, e1x, e2x, e3x, e1y, e2y, e3y, e1z, e2z, e3z,
             f11, f12, f13, f21, f22, f23, f31, f32, f33,
             m11, m12, m13, m21, m22, m23, m31, m32, m33)

    if fint is not None:
        np.subtract.at(fint[:, 0], conn[:, 0], f11)
        np.subtract.at(fint[:, 1], conn[:, 0], f21)
        np.subtract.at(fint[:, 2], conn[:, 0], f31)
        np.subtract.at(fint[:, 0], conn[:, 1], f12)
        np.subtract.at(fint[:, 1], conn[:, 1], f22)
        np.subtract.at(fint[:, 2], conn[:, 1], f32)
        np.subtract.at(fint[:, 0], conn[:, 2], f13)
        np.subtract.at(fint[:, 1], conn[:, 2], f23)
        np.subtract.at(fint[:, 2], conn[:, 2], f33)

    if mint is not None:
        np.subtract.at(mint[:, 0], conn[:, 0], m11)
        np.subtract.at(mint[:, 1], conn[:, 0], m21)
        np.subtract.at(mint[:, 2], conn[:, 0], m31)
        np.subtract.at(mint[:, 0], conn[:, 1], m12)
        np.subtract.at(mint[:, 1], conn[:, 1], m22)
        np.subtract.at(mint[:, 2], conn[:, 1], m32)
        np.subtract.at(mint[:, 0], conn[:, 2], m13)
        np.subtract.at(mint[:, 1], conn[:, 2], m23)
        np.subtract.at(mint[:, 2], conn[:, 2], m33)


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom=False):
    """Implicit residual internal forces and moments dispatch for DKT18."""
    if group.n == 0 or len(group.conn) == 0:
        return
    if not nlgeom:
        forces(group, x_ref, u, ur, 1.0, fint, mint)
    else:
        x_mid = x_ref + 0.5 * u
        x_end = x_ref + u
        junk_f = np.zeros_like(fint) if fint is not None else None
        junk_m = np.zeros_like(mint) if mint is not None else None
        forces(group, x_mid, u, ur, 1.0, junk_f, junk_m)
        static_internal_forces(group, x_end, u, ur, fint, mint)
