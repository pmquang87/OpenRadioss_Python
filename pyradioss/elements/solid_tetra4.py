"""
4-node tetrahedral solid element, linear (constant-strain) formulation
(/TETRA4 + /PROP/SOLID).

Fortran origin: ``engine/source/elements/solid/solide4/`` — the cycle path
mirrors the 8-node brick one (see solid_hexa8.py), with the "4" variants:

    s4forc3.F   driver: gather coords/velocities, call the chain below
    s4coor3.F   geometry (constant Jacobian, volume)
    s4defo3.F   velocity gradient  ->  rate of deformation D, spin W
    srota3.F    Jaumann rotation of the old stress (shared with the brick)
    mmain.F     material law (SIGEPS..)
    s4fint3.F   internal nodal forces  f_i = V * sigma . gradN_i
    s4dlen (in s4coor3/sdlen)  characteristic length -> time step

Theory notes:

* The linear tetrahedron interpolates displacement linearly, so the
  strain field is **exactly constant** over the element — one integration
  point is *full* integration here, and there are **no hourglass modes**
  (12 dofs = 6 rigid-body + 6 constant-strain modes, nothing left over).
  That is why this file has no hourglass block at all, unlike the brick.

* The price is the well-known stiffness of the constant-strain tet: it
  locks volumetrically in incompressible plasticity and needs several
  elements through a bending span. Radioss mitigates this with nodal-
  pressure tetra formulations (Itetra4 options); this port implements the
  plain element and documents the limitation (roadmap M3+).

* Kinematics, Jaumann rotation, bulk viscosity and the material interface
  are IDENTICAL to the brick (deliberately re-derived here line by line
  rather than shared, mirroring the Fortran which keeps a solide4/ copy —
  each element file stays readable top-to-bottom on its own).

* **Characteristic length**: the true minimum altitude of a tetrahedron
  is  h_min = 3 V / A_max  (volume over largest face area, times 3 —
  compare the brick's V/A_max which IS the height of a box). The Courant
  step is then lc/c, corrected by the exact eigenvalue factor below.

* **Exact time step** (same port refinement as the brick, see
  solid_hexa8._exact_dt_factor): with constant B the element stiffness is
  K = V B^T C B, its nonzero eigenvalues are those of the 6x6 C.(B B^T),
  and with the lumped nodal mass m = rho V/4

      omega_max^2 = (4 / rho) * max eig( C . (B B^T) )

  The Starter stores dt_exact/(lc/c) as 'dtfac' per element; forces()
  multiplies the running lc/c estimate by it. This matters even more for
  tets than for bricks: for near-regular tets the naive lc/c estimate can
  sit ~30-40% above the true one-element stability limit.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3

# dN_i/dxi_a of the linear tetrahedron with natural coordinates
#   N1 = 1 - xi1 - xi2 - xi3,  N2 = xi1,  N3 = xi2,  N4 = xi3
#
# Node-ordering / volume-sign convention (M38, matched to upstream):
# _geometry() below returns the standard isoparametric determinant
#   V_std = det[x2-x1, x3-x1, x4-x1] / 6,
# which is the NEGATIVE of Radioss's signed tetra volume
#   VOLDP = (x1-x4).((x2-x4) x (x3-x4)) / 6
# (starter/source/elements/solid/solide4/s4deri3.F, and its read-time twin
#  CHECKVOLUME_4N in .../solide/checksvolume.F).  A /TETRA4 card is accepted
# in EITHER winding: the starter canonicalises a wrong-signed element by
# swapping its 2nd and 4th local nodes (hm_read_solid.F / s4coor3.F swap
# IXS(4)<->IXS(6)).  init_group() below mirrors that swap so an official
# mesh (written in the VOLDP>0 winding = V_std<0 here) initialises with a
# positive volume, arriving at an equivalent positively-signed labelling;
# a genuinely degenerate (near-coplanar) tetra survives the swap with
# |V|~0 and is flagged, exactly as s4deri3.F's DET<=0 guard does.
_DN_DXI = np.array([
    [-1.0, -1.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])

# The 4 triangular faces (node indices), for the characteristic length
# and the free-surface extraction (starter surfaces from tetra parts).
_FACES = np.array([
    [0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2],
])


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _geometry(xe: np.ndarray):
    """Constant Jacobian, volume and cartesian shape gradients.

    xe : (n, 4, 3) nodal coordinates.
    Returns (dndx (n,4,3), vol (n,)). Fortran: s4coor3.F/s4deri3.F.
    """
    n = len(xe)
    if n == 0:
        return np.zeros((0, 4, 3)), np.zeros(0)
    # J[a,b] = d x_b / d xi_a  (edge vectors from node 1); explicit 3x3
    # cofactor det/inverse — the M7 cheap win, see fastmath.det_inv33
    J = np.einsum("ia,nib->nab", _DN_DXI, xe)
    a, b, c = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
    d, e, f = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
    g, h, i = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    detJ = a * A + b * B + c * C
    vol = detJ / 6.0                      # tet volume = det(edges)/6

    safe_det = np.where(np.abs(detJ) < 1e-12, np.where(detJ >= 0, 1e-12, -1e-12), detJ)
    idet = 1.0 / safe_det
    Jinv = np.empty_like(J)
    Jinv[:, 0, 0] = A * idet
    Jinv[:, 0, 1] = (c * h - b * i) * idet
    Jinv[:, 0, 2] = (b * f - c * e) * idet
    Jinv[:, 1, 0] = B * idet
    Jinv[:, 1, 1] = (a * i - c * g) * idet
    Jinv[:, 1, 2] = (c * d - a * f) * idet
    Jinv[:, 2, 0] = C * idet
    Jinv[:, 2, 1] = (b * g - a * h) * idet
    Jinv[:, 2, 2] = (a * e - b * d) * idet

    # dN_i/dx_b = dN_i/dxi_a * dxi_a/dx_b ; dxi_a/dx_b = inv(J)[b,a]
    dndx = np.einsum("ia,nba->nib", _DN_DXI, Jinv)
    deg = np.abs(detJ) < 1e-12
    if np.any(deg):
        dndx[deg] = 0.0
    return dndx, vol


def _char_length(xe: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Characteristic length = minimum altitude = 3 V / max face area."""
    if len(vol) == 0:
        return np.zeros(0)
    # all 4 faces at once (fastmath cross/norm — the M7 cheap win, same
    # rewrite as solid_hexa8._char_length)
    e1 = xe[:, _FACES[:, 1]] - xe[:, _FACES[:, 0]]      # (n, 4, 3)
    e2 = xe[:, _FACES[:, 2]] - xe[:, _FACES[:, 0]]
    a = 0.5 * norm3(cross3(e1, e2))                     # (n, 4) face areas
    return 3.0 * vol / np.maximum(a.max(axis=1), EM20)


def _exact_dt_factor(dndx: np.ndarray, vol: np.ndarray, lc: np.ndarray,
                     slices) -> np.ndarray:
    """Per-element ratio dt_exact / (lc/c) — see the module docstring and
    solid_hexa8._exact_dt_factor (same construction, nodal mass rho*V/4)."""
    n = len(vol)
    if n == 0:
        return np.ones(0)
    b = dndx                                        # (n, 4, 3)
    S = np.einsum("nia,nib->nab", b, b)             # gradient moment (n,3,3)
    BBt = np.zeros((n, 6, 6))
    Sxx, Syy, Szz = S[:, 0, 0], S[:, 1, 1], S[:, 2, 2]
    Sxy, Syz, Sxz = S[:, 0, 1], S[:, 1, 2], S[:, 0, 2]
    # rows/cols: [xx, yy, zz, xy, yz, zx] (engineering shear)
    BBt[:, 0, 0], BBt[:, 1, 1], BBt[:, 2, 2] = Sxx, Syy, Szz
    BBt[:, 3, 3] = Sxx + Syy
    BBt[:, 4, 4] = Syy + Szz
    BBt[:, 5, 5] = Sxx + Szz
    BBt[:, 0, 3] = BBt[:, 3, 0] = Sxy
    BBt[:, 1, 3] = BBt[:, 3, 1] = Sxy
    BBt[:, 0, 5] = BBt[:, 5, 0] = Sxz
    BBt[:, 2, 5] = BBt[:, 5, 2] = Sxz
    BBt[:, 1, 4] = BBt[:, 4, 1] = Syz
    BBt[:, 2, 4] = BBt[:, 4, 2] = Syz
    BBt[:, 3, 4] = BBt[:, 4, 3] = Sxz
    BBt[:, 3, 5] = BBt[:, 5, 3] = Syz
    BBt[:, 4, 5] = BBt[:, 5, 4] = Sxy

    fac = np.ones(n)
    for sl, mat, prop in slices:
        rho0_val = getattr(mat, "rho0", 0.0)
        E_val = getattr(mat, "E", 0.0)
        if not (rho0_val > 0.0 and E_val > 0.0) or np.any(vol[sl] <= 1e-12):
            fac[sl] = 1.0
            continue
        nu_val = getattr(mat, "nu", 0.3)
        K_val = getattr(mat, "K", E_val / (3.0 * (1.0 - 2.0 * nu_val)) if abs(1.0 - 2.0 * nu_val) > 1e-6 else E_val)
        G_val = getattr(mat, "G", E_val / (2.0 * (1.0 + nu_val)) if abs(1.0 + nu_val) > 1e-6 else E_val / 2.6)
        lam = K_val - 2.0 * G_val / 3.0
        C = np.array([
            [lam + 2 * G_val, lam, lam, 0, 0, 0],
            [lam, lam + 2 * G_val, lam, 0, 0, 0],
            [lam, lam, lam + 2 * G_val, 0, 0, 0],
            [0, 0, 0, G_val, 0, 0],
            [0, 0, 0, 0, G_val, 0],
            [0, 0, 0, 0, 0, G_val],
        ])
        c = mat.sound_speed_solid() if (hasattr(mat, "sound_speed_solid") and rho0_val > 0.0 and E_val > 0.0) else (
            np.sqrt((K_val + 4.0 * G_val / 3.0) / rho0_val) if rho0_val > 0.0 else 0.0
        )
        if c <= 0.0:
            fac[sl] = 1.0
            continue
        eig = np.linalg.eigvals(C[None, :, :] @ BBt[sl])
        w2max = (4.0 / rho0_val) * eig.real.max(axis=1)   # m = rho*V/4
        dt_exact = 2.0 / np.sqrt(np.maximum(w2max, EM20))
        fac[sl] = np.minimum(dt_exact / np.maximum(lc[sl] / c, EM20), 1.0)
    return fac


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Element buffer + lumped mass (starter s4init3/s4mass3): volume from
    the initial geometry, element mass rho0*V spread equally to the 4
    nodes.

    Node-ordering canonicalisation mirrors the Radioss starter (s4coor3.F /
    hm_read_solid.F): a /TETRA4 whose signed volume is negative in the port's
    isoparametric convention (= Radioss VOLDP>0 winding, how official meshes
    are written) is fixed IN PLACE by the same 2<->4 local-node swap, so the
    stored connectivity always yields a positive volume for the tested
    forces()/tangent() math.  See the _DN_DXI convention note above."""
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            sig=np.zeros((0, 6)),
            epsp=np.zeros(0),
            vol0=np.zeros(0),
            mass=np.zeros(0),
            eint=np.zeros(0),
            ehour=np.zeros(0),
            off=np.ones(0),
            qvw_pend=np.zeros(0),
            dtfac=np.ones(0),
        )
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float), None

    xe = model.x0[group.conn]                      # (n, 4, 3)
    dndx0, vol = _geometry(xe)
    # --- canonicalise winding: swap local nodes 2 and 4 where V_std<0 -------
    # (Radioss hm_read_solid.F: IC2=IXS(6); IC4=IXS(4); IXS(4)=IC2; IXS(6)=IC4)
    flip = vol < 0.0
    if np.any(flip):
        group.conn[flip] = group.conn[flip][:, [0, 3, 2, 1]]
        xe = model.x0[group.conn]
        dndx0, vol = _geometry(xe)
    # --- genuinely degenerate (coplanar) tetra: |V|~0 survives the swap -----
    # size-relative floor so a collapsed element is caught in either winding;
    # mirror s4deri3.F's error path (MSGID 245 for a solid property, VOL=EM20
    # "to prevent crash") so the flag never turns into a downstream NaN.
    ev = xe[:, 1:, :] - xe[:, :1, :]               # edges from node 1 (n,3,3)
    vtol = 1.0e-9 * np.sqrt((ev * ev).sum(-1)).mean(axis=1) ** 3
    bad = vol <= vtol
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/TETRA4 {eid}: zero or negative volume "
                      f"(check node ordering)", "TETRA INIT")
        # s4deri3.F sets VOL=EM20 "to prevent crash"; the port additionally
        # neutralises the singular (NaN/huge) shape gradients of the collapsed
        # element so the flagged run cannot trip the dt eigensolver below.
        vol = np.where(bad, EM20, vol)
        dndx0[bad] = 0.0

    slices = group.state.get("slices", [])
    rho0 = np.zeros(n)
    for sl, mat, prop in slices:
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol

    lc0 = _char_length(xe, vol)
    group.state.update(
        sig=np.zeros((n, 6)),        # Cauchy stress, Voigt (GBUF%SIG)
        epsp=np.zeros(n),            # equivalent plastic strain (GBUF%PLA)
        vol0=vol.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),           # always zero: no hourglass modes (doc)
        off=np.ones(n),              # 1 alive / 0 deleted (GBUF%OFF)
        qvw_pend=np.zeros(n),        # deferred half of the viscous work
        # (midstep booking, see solid_hexa8)
        dtfac=_exact_dt_factor(dndx0, vol, lc0, slices),
    )
    # dndx0 / damage / failure-flag plumbing shared with the brick kernel
    from .solid_hexa8 import _init_material_state
    _init_material_state(group, dndx0)
    group._model = model
    
    # ---- M36: Smoothing FEM (Itetra4 = 3) initialization -------------------
    # Determine which slices have itetra4 == 3
    isrot3 = np.zeros(n, dtype=bool)
    for sl, mat, prop in slices:
        if hasattr(prop, "params") and prop.params.get("itetra4", 0) == 3:
            isrot3[sl] = True
            
    if isrot3.any():
        if not hasattr(model, "nodal_vol_0"):
            model.nodal_vol_0 = np.zeros(len(model.x0))
            model.nodal_vol_t = np.zeros(len(model.x0))
        
        # S4VOLNOD_SM: scatter initial element volumes to nodes
        v_sfem = vol * isrot3
        nodes_sfem = group.conn[isrot3]      # (m, 4)
        v_sfem_m = v_sfem[isrot3]            # (m,)
        
        np.add.at(model.nodal_vol_0, nodes_sfem.reshape(-1), np.repeat(v_sfem_m, 4))
        
        # Store for the cycle loop
        group.state["sfem_isrot3"] = isrot3
        group.state["sfem_amu0"] = np.zeros(n)  # for volumetric strain rate
        group.state["sfem_v0_nodes"] = model.nodal_vol_0
        group.state["sfem_v_nodes"] = model.nodal_vol_t

    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    return node_idx, mass_c, None


def pre_forces(group, model, x, dt):
    """Pre-forces pass for global nodal volume scattering (Itetra=3)."""
    if group.n == 0 or len(group.conn) == 0:
        return
    if not group.state.get("sfem_isrot3", np.array(False)).any():
        return
    st = group.state
    conn = group.conn
    isrot3 = st["sfem_isrot3"]
    xe = x[conn[isrot3]]
    _, vol = _geometry(xe)
    nodes_sfem = conn[isrot3]
    if hasattr(model, "nodal_vol_t"):
        np.add.at(model.nodal_vol_t, nodes_sfem.reshape(-1), np.repeat(vol, 4))


def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the whole tetra group (s4forc3.F chain).
    Returns the per-element critical time step."""
    n = group.n
    conn = group.conn
    if n == 0 or len(conn) == 0:
        return np.empty(0, dtype=float)
    if dt is not None and dt < 0.0:
        return np.full(n, EP30)
    st = group.state
    xe = x[conn]                                   # (n, 4, 3) gather
    if dt is None or dt == 0.0 or v is None:
        dndx, vol = _geometry(xe)
        vol = np.maximum(vol, EM20)
        lc = _char_length(xe, vol)
        rho = st["mass"] / vol
        c = np.zeros(n)
        is_void = np.zeros(n, dtype=bool)
        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) == 0:
                is_void[sl] = True
            elif hasattr(mat, "sound_speed_solid"):
                c[sl] = mat.sound_speed_solid()
            else:
                c_val = None
                try:
                    c_val = materials.sound_speed(mat, rho=rho[sl])
                except Exception:
                    pass
                if c_val is not None:
                    c[sl] = c_val
                else:
                    K = getattr(mat, "K", 0.0)
                    G = getattr(mat, "G", 0.0)
                    c[sl] = np.sqrt(np.maximum(K + 4.0 * G / 3.0, 0.0) / np.maximum(rho[sl], EM20))
        alive = st.get("off", np.ones(n)) > 0.0
        dt_e = np.where(alive & (c > 0.0), st.get("dtfac", np.ones(n)) * lc / np.maximum(c, EM20), EP30)
        return np.where(is_void, EP30, dt_e)
    ve = np.zeros_like(xe) if v is None else v[conn]

    # ---- geometry (s4coor3) ----------------------------------------------
    dndx, vol = _geometry(xe)
    vol = np.maximum(vol, EM20)
    rho = st["mass"] / vol
    lc = _char_length(xe, vol)

    # ---- velocity gradient, D and W (s4defo3) ------------------------------
    L = np.einsum("nib,nic->nbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    trD = D[:, 0, 0] + D[:, 1, 1] + D[:, 2, 2]
    # flush round-off traces to exact zero — rigid-interior elements must
    # not flicker into the compression branch; see solid_hexa8._pre (M7)
    vgm = np.abs(ve).max(axis=(1, 2)) * np.abs(dndx).max(axis=(1, 2))
    trD = np.where(np.abs(trD) <= 1e-14 * vgm, 0.0, trD)
    deps = np.empty((group.n, 6))                  # Voigt, engineering shear
    deps[:, 0] = D[:, 0, 0] * dt
    deps[:, 1] = D[:, 1, 1] * dt
    deps[:, 2] = D[:, 2, 2] * dt
    deps[:, 3] = 2.0 * D[:, 0, 1] * dt
    deps[:, 4] = 2.0 * D[:, 1, 2] * dt
    deps[:, 5] = 2.0 * D[:, 0, 2] * dt

    # deleted elements (GBUF%OFF = 0): freeze their state (see the brick)
    alive = st["off"] > 0.0
    if not alive.all():
        deps[~alive] = 0.0
        trD = np.where(alive, trD, 0.0)

    # ---- M36: Smoothing FEM (Itetra4 = 3) ---------------------------------
    if st.get("sfem_isrot3") is not None and st["sfem_isrot3"].any():
        isrot3 = st["sfem_isrot3"]
        v0_nodes = st["sfem_v0_nodes"]
        v_nodes = st["sfem_v_nodes"]
        
        # J_a = V_a(t) / V_{0,a}
        # (v_nodes was already scattered across ALL groups in pre_forces!)
        nodes_sfem = conn[isrot3]
        Ja = v_nodes / np.maximum(v0_nodes, EM20)
        
        # \bar{J}_e = 1/4 * sum(J_a)
        Je_bar = 0.25 * Ja[nodes_sfem].sum(axis=1)
        
        # S4VOLN_M volumetric strain rate correction
        amu = 1.0 / np.maximum(Je_bar, EM20) - 1.0
        amu0 = st["sfem_amu0"][isrot3]
        divde = amu0 - amu
        
        tr_deps = deps[isrot3, 0] + deps[isrot3, 1] + deps[isrot3, 2]
        corr = (divde - tr_deps) / 3.0
        deps[isrot3, 0] += corr
        deps[isrot3, 1] += corr
        deps[isrot3, 2] += corr
        st["sfem_amu0"][isrot3] = amu
        
        # Update trace for bulk viscosity
        trD = np.where(isrot3, divde / np.maximum(dt, 1e-30), trD)
        
        # Modified element volume V_{e,eff} = \bar{J}_e * V_{0,e}
        vol_eff = vol.copy()
        vol_eff[isrot3] = Je_bar * st["vol0"][isrot3]
        
        # Override physical volume with the effective volume for material/eos pressure
        # Note: 'rho' computed above remains physical density, as required by OpenRadioss
        vol = vol_eff

    # ---- Jaumann rotation of the old stress (srota3) -----------------------
    sig = st["sig"]
    sig_old = sig.copy()
    wxy = 0.5 * (L[:, 0, 1] - L[:, 1, 0]) * dt
    wyz = 0.5 * (L[:, 1, 2] - L[:, 2, 1]) * dt
    wxz = 0.5 * (L[:, 0, 2] - L[:, 2, 0]) * dt
    sxx, syy, szz = sig[:, 0].copy(), sig[:, 1].copy(), sig[:, 2].copy()
    sxy, syz, szx = sig[:, 3].copy(), sig[:, 4].copy(), sig[:, 5].copy()
    sig[:, 0] += 2.0 * (wxy * sxy + wxz * szx)
    sig[:, 1] += 2.0 * (-wxy * sxy + wyz * syz)
    sig[:, 2] += 2.0 * (-wxz * szx - wyz * syz)
    sig[:, 3] += wxy * (syy - sxx) + wxz * syz + wyz * szx
    sig[:, 4] += wyz * (szz - syy) - wxy * szx - wxz * sxy
    sig[:, 5] += wxz * (szz - sxx) + wxy * syz - wyz * sxy

    # ---- material law per part slice (mmain -> sigeps) ---------------------
    # laws may return their own sound speed (LAW42 stiffens with stretch —
    # its c MUST feed the time step; same for an /EOS since M6; see the
    # brick kernel for the full commentary — the two blocks below are its
    # line-by-line siblings)
    epsp_old = st["epsp"].copy() if st.get("chk_fail") else None
    c = np.zeros(group.n)
    c_from_law = np.zeros(group.n, dtype=bool)
    F = None
    if "dndx0" in st:
        F = np.einsum("nia,nib->nab", xe, st["dndx0"])
    for sl, mat, prop in st.get("slices", []):
        law = getattr(mat, "law", 1)
        if law == 0 or getattr(mat, "rho0", 0.0) <= 0.0:
            sig[sl] = 0.0
            c[sl] = 0.0
            c_from_law[sl] = True
            continue
        extra = {}
        if F is not None:
            extra["F"] = F[sl]
        extra["off"] = st["off"][sl]
        for name, arr in st.get("mat_extra", {}).items():
            extra[name] = arr[sl]
        if materials.needs_env(mat) and not st.get("_impl_static_hg"):
            # M37 pack 2: LAW24/LAW81 gate their dilatancy on the current
            # density / internal energy (see materials.needs_env).
            # NOT under the implicit pseudo-velocity drive (M40): frozen-
            # frame rho == rho0 would degenerate LAW36's total pressure
            # P = K*(rho/rho0 - 1) to P == 0 while the M13 consistent
            # tangent carries K — see solid_hexa8.forces for the full note.
            extra["rho"] = rho[sl]
            extra["eint"] = st["eint"][sl]
            extra["vol"] = vol[sl]
            extra["vol0"] = st["vol0"][sl]
            extra["deltax"] = lc[sl]
            extra["le"] = lc[sl]
            extra["aldt"] = lc[sl]
            if hasattr(group, "_model") and hasattr(group._model, "t"):
                extra["time"] = group._model.t
        _, _, c_new = materials.solid_update(
            mat, sig[sl], deps[sl], st["epsp"][sl], dt, extra or None)
        if c_new is not None:
            c[sl] = c_new
            c_from_law[sl] = True
        for name in st.get("mat_extra", {}):
            if name in extra and name != "eint":
                st["mat_extra"][name][sl] = extra[name]
        if "uvar88" in extra:
            if "uvar88" not in st:
                st["uvar88"] = np.zeros((group.n, 30))
            st["uvar88"][sl] = extra["uvar88"]
        law_id = getattr(mat, "law", None)
        law_str = str(law_id).lower().replace("law", "") if law_id is not None else ""
        off_key = f"off{law_str}"
        if off_key in extra:
            st["off"][sl] = np.minimum(st["off"][sl], extra[off_key])
        elif "off" in extra:
            st["off"][sl] = np.minimum(st["off"][sl], extra["off"])

        # ---- /EOS pressure (M6, eosmain) — see solid_hexa8 -----------------
        if mat.eos is not None:
            from ..materials import eos as eos_mod
            live = alive[sl]
            J = vol[sl] / st["vol0"][sl]
            mu = 1.0 / J - 1.0
            dv = np.where(live, J - st["j_prev"][sl], 0.0)
            st["j_prev"][sl] = np.where(live, J, st["j_prev"][sl])
            sgsl = sig[sl]
            pm = (sgsl[:, 0] + sgsl[:, 1] + sgsl[:, 2]) / 3.0
            s_new = sgsl.copy()
            s_new[:, 0] -= pm
            s_new[:, 1] -= pm
            s_new[:, 2] -= pm
            so = sig_old[sl]
            pm_o = (so[:, 0] + so[:, 1] + so[:, 2]) / 3.0
            s_mid = 0.5 * (s_new + so)
            s_mid[:, 0] -= 0.5 * pm_o
            s_mid[:, 1] -= 0.5 * pm_o
            s_mid[:, 2] -= 0.5 * pm_o
            de_dev = J * np.einsum("nk,nk->n", s_mid, deps[sl])
            p_new, e_new, c2 = eos_mod.update(
                mat.eos, mu, dv, st["e_eos"][sl], st["p_eos"][sl], de_dev)
            p_new = np.where(live, p_new, st["p_eos"][sl])
            e_new = np.where(live, e_new, st["e_eos"][sl])
            st["p_eos"][sl] = p_new
            st["e_eos"][sl] = e_new
            sgsl[:] = s_new
            sgsl[:, 0] -= p_new
            sgsl[:, 1] -= p_new
            sgsl[:, 2] -= p_new
            c[sl] = np.sqrt(c2 + (4.0 * mat.G / 3.0) / rho[sl])
            c_from_law[sl] = True

    # ---- failure models + eps_p_max deletion (pyradioss/failure/) ----------
    if st.get("chk_fail"):
        off = st["off"]
        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) in (49, "49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", 50, "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "LAW50_VISC_HONEY", "LAW50_HYP_FOAM", 79, "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", 163, "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM") or getattr(mat, "law_name", None) in ("49", "LAW49", "STEINB", "STEINBERG", "STEINBERG_GUINAN", "MAT_LAW49", "MAT_STEINB", "MAT_STEINBERG", "LAW49_STEINB", "79", "LAW79", "JOHN_HOLM", "JOHNSON_HOLMQUIST", "JH2", "MAT_LAW79", "MAT_JOHN_HOLM", "LAW79_JOHN_HOLM", "50", "LAW50", "VISC_HONEY", "HYP_FOAM", "MAT_LAW50", "MAT_VISC_HONEY", "MAT_HYP_FOAM", "LAW50_VISC_HONEY", "LAW50_HYP_FOAM", "163", "LAW163", "CRUSHABLE_FOAM", "CRUSH_FOAM", "MAT_LAW163", "MAT_CRUSHABLE_FOAM", "MAT_CRUSH_FOAM"):
                eps_max = mat.params.get("eps_p_max", EP30)
            else:
                eps_max = mat.params.get("eps_p_max", mat.params.get("eps_max", EP30))
            if mat.fail is None and eps_max >= 1e30:
                continue
            broken = np.zeros(sl.stop - sl.start, dtype=bool)
            if mat.fail is not None:
                tstar = None                 # /FAIL/JOHNSON D5 (M6)
                if "temp" in st.get("mat_extra", {}) and "mT" in mat.params:
                    tstar = np.clip(
                        st["mat_extra"]["temp"][sl]
                        / (mat.params["T_melt"] - mat.params["T_i"]),
                        0.0, 1.0)
                broken |= failure.solid_step(
                    mat.fail, sig[sl], st["epsp"][sl] - epsp_old[sl],
                    deps[sl], dt, st["dama"][sl], tstar)
            if eps_max < 1e30:
                broken |= st["epsp"][sl] > eps_max
            off[sl][broken] = 0.0
    alive = st["off"] > 0.0
    sig[~alive] = 0.0            # a deleted element carries no stress

    # ---- sound speed & bulk viscosity (sbulk3) ------------------------------
    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    for sl, mat, prop in st.get("slices", []):
        if sl.stop > sl.start and not c_from_law[sl.start]:
            rho0_sl = getattr(mat, "rho0", 0.0)
            if getattr(mat, "law", 1) == 0 or rho0_sl <= 0.0:
                c[sl] = 0.0
            elif hasattr(mat, "sound_speed_solid"):
                c[sl] = mat.sound_speed_solid()
            else:
                K_sl = getattr(mat, "K", 0.0)
                G_sl = getattr(mat, "G", 0.0)
                c[sl] = np.sqrt((K_sl + 4.0 * G_sl / 3.0) / rho0_sl)
        qa[sl] = getattr(prop, "params", {}).get("qa", 1.1) if hasattr(prop, "params") else getattr(prop, "qa", 1.1)
        qb[sl] = getattr(prop, "params", {}).get("qb", 0.05) if hasattr(prop, "params") else getattr(prop, "qb", 0.05)
    compressing = (trD < 0.0) & alive
    qvisc = np.where(
        compressing,
        rho * lc * (qa ** 2 * lc * trD ** 2 - qb * c * trD),
        0.0)
    sig_tot = sig.copy()
    sig_tot[:, 0] -= qvisc
    sig_tot[:, 1] -= qvisc
    sig_tot[:, 2] -= qvisc

    # ---- internal nodal forces (s4fint3):  f_i = V * sigma . gradN_i -------
    S = np.empty((group.n, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig_tot[:, 0], sig_tot[:, 1], sig_tot[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig_tot[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig_tot[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig_tot[:, 5]
    fe = -np.einsum("n,nbc,nic->nib", vol, S, dndx)   # minus sign: fint
    # accumulates -integral(B^T sigma), see the elements package docstring

    # ---- energy bookkeeping -------------------------------------------------
    # bulk-viscosity work booked trapezoidally: half at this cycle's trD,
    # half deferred to the next cycle's (= the post-update velocities) —
    # the leapfrog-consistent midstep booking. See the long comment in
    # solid_hexa8.forces() for the barely-resolved-ringing failure mode of
    # the one-sided M1 booking that this fixes (M6).
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = 0.5 * vol * qvisc * (-trD * dt) + st["qvw_pend"] * (-trD)
    if "eos_mask" in st:
        # /EOS elements: energy from the EOS state (+ viscous shock
        # heating) — see the matching block in solid_hexa8
        em = st["eos_mask"]
        st["e_eos"][em] += w_visc[em] / st["vol0"][em]
        deint = vol * np.einsum("nk,nk->n", sig_mid, deps) + w_visc
        st["eint"] += np.where(em, 0.0, deint)
        st["eint"][em] = st["e_eos"][em] * st["vol0"][em]
    else:
        st["eint"] += vol * np.einsum("nk,nk->n", sig_mid, deps) + w_visc
    st["qvw_pend"] = 0.5 * vol * qvisc * dt          # booked next cycle

    # ---- scatter to global arrays (asspar) ----------------------------------
    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))

    # ---- critical time step --------------------------------------------------
    Q = np.where(compressing, qb * c + qa * lc * np.abs(trD), 0.0)
    denom = Q + np.sqrt(Q * Q + c * c)
    dt_crit = np.where((denom > 0.0) & (c > 0.0), st["dtfac"] * lc / np.maximum(denom, EM20), EP30)
    # deleted elements no longer constrain the global step
    return np.where(alive, dt_crit, EP30)


# ----------------------------------------------------------------------------
# Implicit tangent stiffness (M11) — a NEW entry point alongside forces()
# ----------------------------------------------------------------------------
# Fortran origin: the element-KE branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F`` dispatching the solide4 stiffness)
# and, for the geometric part, its ``imp_kgeo`` path (/IMPL/NONLIN).
#
# The constant-strain tetra is the SIMPLEST implicit solid: the linear
# displacement field makes B constant over the element, one point is FULL
# integration and there are NO hourglass modes (module docstring) — so,
# unlike the one-point hexa, the tangent needs no stabilization block at
# all: K_e = V B^T D B is already rank 6 (12 dofs - 6 rigid modes), exactly
# what a full-rank element tangent must be. D is the material consistent
# tangent (LAW1: C; LAW2: the algorithmic radial-return tangent —
# materials.solid_tangent, shared with the hexa). Same for the residual:
# forces() driven with the pseudo-velocity at dt = 1 IS the exact
# incremental force, nothing to add (no static_stabilization here).
#
# K_geo is the same delta_ij initial-stress operator as the hexa
# (V * gradN_a . sigma . gradN_b — see solid_hexa8.kgeo for the derivation),
# with the tet's own constant gradients; static_internal_forces re-states
# the s4fint3 force expression f_i = -V sigma gradN_i standalone for the M9
# updated-Lagrangian end-configuration assembly.

def _edofs(conn):
    """(n, 12) global scalar DOF slot ids, node-major [ux, uy, uz] * 4."""
    n = len(conn)
    if n == 0:
        return np.zeros((0, 12), dtype=np.int64)
    ix = np.arange(4)
    edofs = np.empty((n, 12), dtype=np.int64)
    edofs[:, 3 * ix + 0] = conn * 6 + 0
    edofs[:, 3 * ix + 1] = conn * 6 + 1
    edofs[:, 3 * ix + 2] = conn * 6 + 2
    return edofs


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole tetra group.

    Returns ``(ke, edofs)``: ``ke`` (n, 12, 12) dense element tangents
    (translations only), ``edofs`` (n, 12) global scalar DOF slot ids in the
    ``implicit.dofmap`` numbering. ``epsp_incr`` (n,) is the increment's
    plastic-strain step for the LAW2 consistent tangent (None = elastic)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12), dtype=float), np.zeros((0, 12), dtype=np.int64)
    dndx, vol = _geometry(x[conn])
    vol = np.maximum(vol, EM20)

    # strain-displacement operator B (n, 6, 12), engineering shear, rows
    # [xx, yy, zz, xy, yz, zx] — the same layout as the hexa tangent
    B = np.zeros((n, 6, 12))
    gx, gy, gz = dndx[:, :, 0], dndx[:, :, 1], dndx[:, :, 2]   # (n, 4)
    ix = np.arange(4)
    B[:, 0, 3 * ix + 0] = gx
    B[:, 1, 3 * ix + 1] = gy
    B[:, 2, 3 * ix + 2] = gz
    B[:, 3, 3 * ix + 0] = gy
    B[:, 3, 3 * ix + 1] = gx
    B[:, 4, 3 * ix + 1] = gz
    B[:, 4, 3 * ix + 2] = gy
    B[:, 5, 3 * ix + 0] = gz
    B[:, 5, 3 * ix + 2] = gx

    from .. import materials as _materials
    ke = np.zeros((n, 12, 12))
    epi = np.zeros(n) if epsp_incr is None else epsp_incr
    # M14: total-form laws (LAW42) get the deformation gradient of the
    # linearization geometry (see solid_hexa8.tangent)
    F = (np.einsum("nia,nib->nab", x[conn], st["dndx0"])
         if "dndx0" in st else None)
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        extra = ({"F": F[sl]} if F is not None
                 and _materials.needs_defgrad(mat) else None)
        D = _materials.solid_tangent(mat, st["sig"][sl], st["epsp"][sl],
                                     epi[sl], extra)       # (m, 6, 6)
        Bs = B[sl]
        DB = np.einsum("mij,mjk->mik", D, Bs)              # (m, 6, 12)
        ke[sl] = vol[sl][:, None, None] * np.einsum("mji,mjk->mik", Bs, DB)
    return ke, _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the tetra group at
    geometry ``x`` — V * gradN_a . sigma . gradN_b replicated over the three
    translation directions (see solid_hexa8.kgeo). Identically zero at zero
    stress."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12), dtype=float), np.zeros((0, 12), dtype=np.int64)
    dndx, vol = _geometry(x[conn])
    vol = np.maximum(vol, EM20)
    s = st["sig"]
    S = np.empty((n, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = s[:, 5]
    g = vol[:, None, None] * np.einsum("nac,ncd,nbd->nab", dndx, S, dndx)
    ke = np.zeros((n, 12, 12))
    ix = np.arange(4)
    for b in range(3):
        rows = (3 * ix + b)[:, None]
        cols = (3 * ix + b)[None, :]
        ke[:, rows, cols] += g
    return ke, _edofs(conn)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the lumped mass is ``starter/source/elements/solid/solide4/
# s4mass3.F`` (the /4 quarter-mass-to-each-node lumping ``init_group`` returns,
# = the smass3.F family this milestone builds on). The CONSISTENT mass is the
# shape-function integral M = ∫_V ρ Nᵀ N dV; ported here for the M16 modal
# eigensolver alongside — never mutating — the lumped path.
#
# Theory (Cook, Malkus & Plesha ch. 11; Hughes "The FEM" ch. 7). The linear
# (constant-strain) tetrahedron interpolates displacement with the barycentric
# coordinates N_i, whose products integrate EXACTLY over the element:
#
#     ∫_V N_i N_j dV = V/20 (1 + δ_ij)   ⇒   ∫ Nᵀ N dV = V/20 [[2,1,1,1],
#                                                               [1,2,1,1],
#                                                               [1,1,2,1],
#                                                               [1,1,1,2]]
#
# (the standard simplex moment formula ∫ N_i^a N_j^b ... dV =
# a! b! ... 3! / (a+b+...+3)! · 6V). The Jacobian is CONSTANT over the tet, so
# this analytic form is EXACT for any tet shape (no quadrature error, unlike
# the 8-node brick which needs 2×2×2 Gauss). The mass is isotropic in the
# three translation directions (∝ I3), hence frame-invariant, and built on the
# reference volume (mass conservation). Row-sum = V/4 · ρ = m/4 per node —
# the lumped nodal mass — so ½ vᵀ M v = ½ m |v|² is exact for rigid v.

#: ∫ Nᵀ N over the reference tet, in units of V (the simplex moment matrix).
_M_TET = (np.ones((4, 4)) + np.eye(4)) / 20.0


def consistent_mass(group, x=None):
    """Consistent element mass ∫ρ Nᵀ N dV of the 4-node tet (see the note
    above): ρ V/20 [[2,1,1,1],…] ⊗ I3, built on the stored element mass.

    Returns ``(me (n,12,12), edofs (n,12))`` — translations only, the same
    node-major addressing as ``tangent()``. ``x`` unused (the mass uses the
    reference volume and is frame-invariant)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.zeros((0, 12, 12), dtype=float), np.zeros((0, 12), dtype=np.int64)
    m = st["mass"]                                    # ρ V0, per element
    me = np.zeros((n, 12, 12))
    for a in range(4):
        for b in range(4):
            f = m * _M_TET[a, b]                       # (n,) = ρ V · factor
            for c in range(3):
                me[:, a * 3 + c, b * 3 + c] = f
    return me, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal force at configuration ``x`` from the CURRENT stress
    state — the updated-Lagrangian end-configuration force assembly of the
    M9/M11 implicit residual (s4fint3 standalone): f_i = -V sigma gradN_i.
    No hourglass term exists for the tet (full integration). ``u``/``ur``/
    ``mint`` unused (no rotational DOFs)."""
    if group.n == 0 or len(group.conn) == 0 or fint is None:
        return
    st = group.state
    conn = group.conn
    dndx, vol = _geometry(x[conn])
    vol = np.maximum(vol, EM20)
    s = st["sig"]
    # M14: total-form laws re-evaluate at the END configuration (see
    # solid_hexa8.static_internal_forces for the rationale)
    if "dndx0" in st:
        from .. import materials as _materials
        F = np.einsum("nia,nib->nab", x[conn], st["dndx0"])
        for sl, mat, prop in st.get("slices", []):
            if _materials.needs_defgrad(mat):
                _materials.solid_update(
                    mat, s[sl], np.zeros((sl.stop - sl.start, 6)),
                    st["epsp"][sl], 1.0, {"F": F[sl]})
    S = np.empty((group.n, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = s[:, 5]
    fe = -vol[:, None, None] * np.einsum("nid,ncd->nic", dndx, S)
    scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom):
    """Implicit residual internal forces dispatch for tetra4.

    Linear geometry (nlgeom=False): evaluates forces at x_ref with displacement u.
    Nonlinear geometry (nlgeom=True): advances state at midpoint configuration
    x_ref + 0.5*u, then assembles internal forces on end configuration x_ref + u."""
    if group.n == 0 or len(group.conn) == 0:
        return
    if not nlgeom:
        forces(group, x_ref, u, ur, 1.0, fint, mint)
    else:
        x_mid = x_ref + 0.5 * u
        x_end = x_ref + u
        junk_f = np.zeros_like(fint) if fint is not None else None
        forces(group, x_mid, u, ur, 1.0, junk_f, mint)
        static_internal_forces(group, x_end, u, ur, fint, mint)

