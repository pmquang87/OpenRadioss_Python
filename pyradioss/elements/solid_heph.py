"""
8-node hexahedral solid element, one-point integration with
Flanagan–Belytschko hourglass control (/BRICK + /PROP/SOLID, Isolid=1).

Fortran origin: ``engine/source/elements/solid/solide/`` — the cycle path is

    sforc3.F   driver: gather coords/velocities, call the chain below
    srcoor3.F  geometry (Jacobian at the centroid, volume)
    sdefo3.F   velocity gradient  ->  rate of deformation D, spin W
    srota3.F   Jaumann rotation of the old stress by the spin increment
    smalla3.F / mmain.F   call the material law (SIGEPS..)
    sbulk3.F   bulk viscosity (shock damping) pressure
    shour3.F   hourglass (zero-energy mode) control forces
    sfint3.F   internal nodal forces  f_i = V * sigma . gradN_i
    sdlen3.F   characteristic length -> critical time step

Theory notes (kept close to Belytschko, Liu & Moran, "Nonlinear Finite
Elements for Continua and Structures", ch. 8, and Flanagan & Belytschko
IJNME 1981):

* One integration point at the element centroid: the strain field is
  evaluated with the "uniform gradient" B-matrix. Cheap and robust for
  crash/impact, but admits 12 zero-energy ("hourglass") deformation modes
  which must be stabilized — that is the role of the physical (assumed strain,
  Q-scheme) hourglass control in this file.

* The **strain rate** is the symmetric part of the spatial velocity
  gradient L = dv/dx; the skew part W (spin) drives the **Jaumann
  objective rate**: rigid rotation must rotate the stress without changing
  it, so the stress update is
      sigma <- sigma + (W.sigma - sigma.W) dt   (rotation, srota3)
      sigma <- sigma + C : D dt                 (material law, sigeps)

* **Bulk viscosity** (sbulk3): explicit codes smear shocks over a few
  elements by adding a viscous pressure in compression
      q = rho * lc * (qa^2 * lc * trD^2 - qb * c * trD),   trD < 0
  (qa quadratic, qb linear coefficient — /PROP/SOLID defaults 1.1, 0.05).

* **Critical time step** (sdlen3): Courant condition on the P-wave,
      dt = lc / (Q + sqrt(Q^2 + c^2)),   Q = qb*c + qa*lc*|trD^-|
  with lc = V / A_max the volume over the largest face area (a safe
  generalization of "smallest height" to distorted hexas).

  IMPORTANT refinement over the textbook estimate: lc/c is NOT a strict
  bound for the one-point hexa — the exact maximum eigenfrequency of the
  element (a free single cube, nu = 0.3) is ~1.36x higher than 2c/lc, so
  a run at 0.9 * lc/c can be genuinely unstable (we verified this both by
  eigenanalysis and by watching round-off grow in a rigid-body-motion
  test). The element stiffness is K = V B^T C B with a CONSTANT B, so
  its nonzero eigenvalues are those of the 6x6 matrix C.(B B^T) — cheap
  to get exactly. init_group computes, per element,

      dt_exact / (lc/c) = ( 2 / omega_max ) / (lc0 / c)

  and stores it as 'dtfac'; forces() multiplies the running lc/c by it.
  (Frequencies rise as an element distorts, but lc tracks that; the /DT
  scale factor 0.9 covers the drift.)

M7 performance structure
------------------------
forces() is split around the Python material/failure loop into two array
blocks, ``_pre`` (geometry + kinematics + Jaumann rotation, the
srcoor3/sdefo3/srota3 chain) and ``_post`` (bulk viscosity, internal +
hourglass forces, energies, critical dt — sbulk3/sfint3/shour3/sdlen3).
Both have an optional numba mirror in ``pyradioss.accel.jit_kernels``
selected via ``accel.get`` (see the accel package docstring for the
backend architecture and the parity contract). The NumPy code HERE is
the reference implementation. Within the NumPy code the M7 profiling
pass replaced np.cross / np.linalg.det / np.linalg.inv / np.add.at with
the formula-identical small-array primitives of ``common.fastmath``, and
fused the einsum chains into stacked matmuls — see fastmath's docstring
for which replacements are bitwise-identical and which reassociate at
machine precision.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..accel import get as accel_get
from ..common.constants import EM20, EP30
from ..common.fastmath import cross3, det_inv33, norm3, scatter_add3

# Node sign pattern of the trilinear hexa (Radioss /BRICK node ordering:
# nodes 1-4 = bottom face counter-clockwise, 5-8 = top face).
_XI = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)
# dN_i/dxi_a at the centroid = xi_sign/8 (uniform gradient operator)
_DN_DXI = _XI / 8.0
_DN_DXI_T = np.ascontiguousarray(_DN_DXI.T)     # (3, 8) for the matmuls

# The 4 hourglass base vectors of Flanagan-Belytschko (their table 2):
# each is a deformation pattern with zero uniform strain at the centroid.
_H = np.array([
    [1, 1, -1, -1, -1, -1, 1, 1],    # hourglass mode 1
    [1, -1, -1, 1, -1, 1, 1, -1],    # hourglass mode 2
    [1, -1, 1, -1, 1, -1, 1, -1],    # hourglass mode 3
    [-1, 1, -1, 1, 1, -1, 1, -1],    # hourglass mode 4
], dtype=float)

# Faces of the hexa (node indices), used for the characteristic length.
_FACES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
])

# Physical hourglass control coefficient (Belytschko-Bindeman)
HG_PHYS = 0.05
DT_HG_SF = 0.9

# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _edofs(conn: np.ndarray) -> np.ndarray:
    """Global translation DOF indices for 8-node hexas: 3 per node in node-major order."""
    n = len(conn)
    if n == 0:
        return np.empty((0, 24), dtype=np.int64)
    ix = np.arange(8)
    edofs = np.empty((n, 24), dtype=np.int64)
    edofs[:, 3 * ix + 0] = conn * 6 + 0
    edofs[:, 3 * ix + 1] = conn * 6 + 1
    edofs[:, 3 * ix + 2] = conn * 6 + 2
    return edofs


def _geometry(xe: np.ndarray):
    """Centroid Jacobian, volume and cartesian shape gradients.

    xe : (n, 8, 3) nodal coordinates.
    Returns (dndx (n,8,3), vol (n,)). Fortran: srcoor3.F + sderi3.F.
    """
    n = len(xe)
    if n == 0:
        return np.empty((0, 8, 3)), np.empty(0)
    # J[a,b] = d x_b / d xi_a  summed over nodes: (3,8) @ (n,8,3) matmul
    J = _DN_DXI_T @ xe
    a, b, c = J[:, 0, 0], J[:, 0, 1], J[:, 0, 2]
    d, e, f = J[:, 1, 0], J[:, 1, 1], J[:, 1, 2]
    g, h, i = J[:, 2, 0], J[:, 2, 1], J[:, 2, 2]
    A = e * i - f * h
    B = f * g - d * i
    C = d * h - e * g
    detJ = a * A + b * B + c * C
    bad = np.abs(detJ) <= EM20
    if np.any(bad):
        # Safe fallback for degenerate / zero-volume / collapsed elements
        J_safe = np.where(bad[:, None, None], np.eye(3)[None, :, :], J)
        _, Jinv = det_inv33(J_safe)
        Jinv[bad] = 0.0
        vol = 8.0 * np.where(bad, EM20, detJ)
        dndx = _DN_DXI @ Jinv.transpose(0, 2, 1)
        dndx[bad] = 0.0
        return dndx, vol
    _, Jinv = det_inv33(J)
    vol = 8.0 * detJ
    # dN_i/dx_b = dN_i/dxi_a * dxi_a/dx_b ; dxi_a/dx_b = inv(J)[b,a]
    dndx = _DN_DXI @ Jinv.transpose(0, 2, 1)
    return dndx, vol


def _char_length(xe: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Characteristic length lc = V / max face area (sdlen3.F).

    Face area from the cross product of its diagonals: for a (possibly
    warped) quad face with corners a,b,c,d the vector area is
    0.5 * (c-a) x (d-b). All 6 faces at once: gather the diagonal
    endpoints per face (two (n,6,3) arrays), one cross, one norm, max.
    """
    d1 = xe[:, _FACES[:, 2]] - xe[:, _FACES[:, 0]]      # (n, 6, 3)
    d2 = xe[:, _FACES[:, 3]] - xe[:, _FACES[:, 1]]
    a = 0.5 * norm3(cross3(d1, d2))                     # (n, 6) face areas
    return vol / np.maximum(a.max(axis=1), EM20)


def _exact_dt_factor(dndx: np.ndarray, vol: np.ndarray, lc: np.ndarray,
                     slices) -> np.ndarray:
    """Per-element ratio  dt_exact / (lc/c)  from the exact eigenvalue of
    the one-point element (see module docstring).

    With lumped nodal mass m = rho*V/8 and K = V * B^T C B (B constant),
    omega^2 = (8/rho) * eig(B^T C B) and the nonzero eigenvalues of
    B^T C B equal those of C.(B B^T). B B^T is assembled from the 3x3
    gradient moment  S = sum_i gradN_i gradN_i^T.
    """
    n = len(vol)
    b = dndx                                        # (n, 8, 3)
    S = np.einsum("nia,nib->nab", b, b)             # (n, 3, 3)
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
        lam = mat.K - 2.0 * mat.G / 3.0
        G = mat.G
        C = np.array([
            [lam + 2 * G, lam, lam, 0, 0, 0],
            [lam, lam + 2 * G, lam, 0, 0, 0],
            [lam, lam, lam + 2 * G, 0, 0, 0],
            [0, 0, 0, G, 0, 0],
            [0, 0, 0, 0, G, 0],
            [0, 0, 0, 0, 0, G],
        ])
        if hasattr(mat, "sound_speed_solid"):
            c = mat.sound_speed_solid() if (getattr(mat, "rho0", 0.0) > 0.0 and getattr(mat, "E", 0.0) > 0.0) else 0.0
        else:
            rho0 = getattr(mat, "rho0", 0.0)
            E = getattr(mat, "E", 0.0)
            c = np.sqrt(max(mat.K + 4.0 * mat.G / 3.0, 0.0) / max(rho0, EM20)) if (rho0 > 0.0 and E > 0.0) else 0.0
        if c <= 0.0:
            # stiffness-free material (a /MAT/VOID with E = 0, a bare
            # /MAT/GAS): the element claims no time step at all
            # (upstream lc/SSP with SSP = 0), so the correction ratio is
            # moot — keep 1 instead of the 0/inf division artifact that
            # would zero dt_crit (the Starter checks already error on
            # such materials when they cannot run)
            fac[sl] = 1.0
            continue
        eig = np.linalg.eigvals(C[None, :, :] @ BBt[sl])
        w2max = (8.0 / mat.rho0) * eig.real.max(axis=1)
        dt_exact = 2.0 / np.sqrt(np.maximum(w2max, EM20))
        fac[sl] = np.minimum(dt_exact / (lc[sl] / c), 1.0)
    return fac


# ----------------------------------------------------------------------------
# Starter-side initialization (element buffer creation)
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Create the element buffer and return lumped-mass contributions.

    Fortran: starter/source/elements/solid/... (sinit3.F, smass3.F):
    element volume from the initial geometry, element mass = rho0 * V,
    spread equally to the 8 nodes (consistent with the original's lumping).
    """
    n = group.n
    if n == 0 or len(group.conn) == 0:
        group.state.update(
            sig=np.empty((0, 6)),
            epsp=np.empty(0),
            vol0=np.empty(0),
            mass=np.empty(0),
            eint=np.empty(0),
            ehour=np.empty(0),
            off=np.empty(0),
            qvw_pend=np.empty(0),
            hgq=np.empty((0, 4, 3)),
            hgqex=np.empty((0, 4, 3)),
            dtfac=np.empty(0),
            lc_scale=np.empty(0),
            chk_fail=False,
            mat_extra={},
        )
        return np.empty(0, dtype=np.int64), np.empty(0), None

    xe = model.x0[group.conn]                      # (n, 8, 3)
    dndx0, vol = _geometry(xe)
    bad = vol <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/BRICK {eid}: zero or negative volume "
                      f"(check node ordering)", "SOLID INIT")
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * vol

    # Degenerate element (wedge/pyramid/tetra) length scale
    # IDEGE count: duplicate nodes in the connectivity (sdlen_dege.F)
    eq = group.conn[:, :, None] == group.conn[:, None, :]
    eq[:, np.arange(8), np.arange(8)] = False
    idege = eq.any(axis=2).sum(axis=1) // 2

    # Scale characteristic length for degenerate elements
    # Hex (idege=0, 1): scale = 1.0
    # Wedge/Pyramid (idege=2): scale = 2.0
    # Tetra (idege>2): scale = 3.0
    lc_scale = np.ones(n)
    lc_scale[idege > 2] = 3.0
    lc_scale[(idege > 1) & (idege <= 2)] = 2.0
    group.state["lc_scale"] = lc_scale

    lc0 = _char_length(xe, vol) * lc_scale
    group.state.update(
        sig=np.zeros((n, 6)),        # Cauchy stress, Voigt (GBUF%SIG)
        epsp=np.zeros(n),            # equivalent plastic strain (GBUF%PLA)
        vol0=vol.copy(),             # initial volume
        mass=mass,                   # element mass (constant)
        eint=np.zeros(n),            # internal energy (GBUF%EINT)
        ehour=np.zeros(n),           # hourglass energy
        off=np.ones(n),              # 1 alive / 0 deleted (GBUF%OFF)
        # pending half of the bulk-viscosity work, booked at the NEXT
        # cycle's trD (the leapfrog-consistent midstep booking — see the
        # energy block in forces())
        qvw_pend=np.zeros(n),
        # accumulated hourglass MODAL displacement
        hgq=np.zeros((n, 4, 3)),
        hgqex=np.zeros((n, 4, 3)),
        # exact stability correction to the lc/c estimate (module docstring)
        dtfac=_exact_dt_factor(dndx0, vol, lc0, group.state["slices"]),
    )
    _init_material_state(group, dndx0)
    # nodal mass: 1/8 of the element mass to each node
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


def _init_material_state(group, dndx0):
    """M3/M6 material/failure plumbing shared by both solid kernels:

    * ``dndx0`` — the INITIAL shape-function gradients are stored when a
      slice's law is total-strain (LAW42): the cycle then computes the
      deformation gradient exactly as F = sum_i x_i (x) gradN0_i, with no
      rate integration and hence no drift;
    * ``dama`` — /FAIL damage per element (single integration point);
    * ``chk_fail`` — precomputed flag: True when any slice can delete
      elements (a /FAIL card or a material eps_p_max threshold), so the
      cycle skips the whole failure block for plain models;
    * ``mat_extra`` (M6) — law-specific persistent per-element state
      (the LAW2 adiabatic temperature rise), allocated from
      materials.extra_shapes — the solid analogue of the shell kernels'
      per-layer allocation;
    * /EOS state (M6) — elements whose material carries an equation of
      state track their relative volume ``j_prev`` = V/V0, energy per
      reference volume ``e_eos`` and pressure ``p_eos`` (see
      materials/eos.py); ``eos_mask`` flags them for the cycle's energy
      bookkeeping, and their eint starts at the EOS initial energy.
    """
    st = group.state
    n = group.n
    if any(materials.needs_defgrad(mat) for _, mat, _ in st["slices"]):
        st["dndx0"] = dndx0.copy()
    if any(mat.fail is not None for _, mat, _ in st["slices"]):
        st["dama"] = np.zeros(n)
    st["chk_fail"] = any(
        mat.fail is not None or mat.params.get("eps_p_max", EP30) < 1e30
        for _, mat, _ in st["slices"])
    st["mat_extra"] = {}
    for sl, mat, prop in st["slices"]:
        for name, shape in materials.extra_shapes(mat).items():
            if name not in st["mat_extra"]:
                if name.startswith("off"):
                    st["mat_extra"][name] = np.ones((n,) + shape)
                else:
                    st["mat_extra"][name] = np.zeros((n,) + shape)
    if any(mat.eos is not None for _, mat, _ in st["slices"]):
        from ..materials import eos as eos_mod
        st["eos_mask"] = np.zeros(n, dtype=bool)
        st["e_eos"] = np.zeros(n)
        st["p_eos"] = np.zeros(n)
        st["j_prev"] = np.ones(n)
        for sl, mat, prop in st["slices"]:
            if mat.eos is None:
                continue
            e0, p0 = eos_mod.initial_state(mat.eos)
            st["eos_mask"][sl] = True
            st["e_eos"][sl] = e0
            st["p_eos"][sl] = p0
            # the EOS initial energy IS internal energy from cycle 0
            # (the Engine's balance reference includes initial IE)
            st["eint"][sl] = e0 * st["vol0"][sl]


# ----------------------------------------------------------------------------
# Engine-side force computation (one cycle)
# ----------------------------------------------------------------------------

def _pre(xe, ve, sig, dt, off, lc_scale):
    """Geometry + kinematics + Jaumann rotation: the srcoor3 / sdefo3 /
    srota3 / sdlen3-geometry part of the cycle, everything BEFORE the
    material law. Rotates ``sig`` in place; returns
    (dndx, vol, lc, deps, trD). Mirrored by accel.jit_kernels.hexa_pre
    (same formulas, element-serial — the M7 parity contract)."""
    n = len(xe)

    # ---- geometry at t_{n+1/2} (srcoor3) --------------------------------
    dndx, vol = _geometry(xe)
    vol = np.maximum(vol, EM20)
    lc = _char_length(xe, vol) * lc_scale

    # ---- velocity gradient, D and W (sdefo3) -----------------------------
    # L = sum_i v_i (x) gradN_i as a stacked matmul: (n,3,8) @ (n,8,3)
    L = ve.transpose(0, 2, 1) @ dndx
    trD = L[:, 0, 0] + L[:, 1, 1] + L[:, 2, 2]
    # flush ROUND-OFF traces to exact zero (M7). On a rigid velocity
    # field (an element interior to a /RBODY, or uniform translation)
    # the trace cancels only in exact arithmetic — floating point leaves
    # trD at the eps level of the v_i*gradN_i products, with a SIGN that
    # is luck of the summation order. A stray negative flips the
    # 'compressing' branch below and puts the bulk-viscosity qb*c term
    # into the time step: a ~5% dt penalty for zero physical
    # compression, different between backends/op orderings (caught by
    # the M6 /RBODY chain test when M7 reordered these reductions —
    # einsum happened to cancel exactly, matmul leaves ~1e-18).
    # The noise floor of every L entry is eps * max|v| * max|gradN|, so
    # a |trD| below 1e-14 of that product scale is numerical zero by
    # construction — treat it as the exact zero it represents. (The
    # scale must be the v*g PRODUCT, not |L| itself: on a rigid field
    # ALL of L is round-off.) Mirrored in accel.jit_kernels.hexa_pre.
    vgm = np.abs(ve).max(axis=(1, 2)) * np.abs(dndx).max(axis=(1, 2))
    trD = np.where(np.abs(trD) <= 1e-14 * vgm, 0.0, trD)
    # strain increment in Voigt form, ENGINEERING shear (gamma = 2 eps);
    # the off-diagonal D entries are (L + L^T)/2, engineering doubles them
    deps = np.empty((n, 6))
    deps[:, 0] = L[:, 0, 0] * dt
    deps[:, 1] = L[:, 1, 1] * dt
    deps[:, 2] = L[:, 2, 2] * dt
    deps[:, 3] = (L[:, 0, 1] + L[:, 1, 0]) * dt
    deps[:, 4] = (L[:, 1, 2] + L[:, 2, 1]) * dt
    deps[:, 5] = (L[:, 0, 2] + L[:, 2, 0]) * dt

    # deleted elements (GBUF%OFF = 0): freeze their state — no straining,
    # and downstream no stress, viscosity, hourglass force or dt claim
    alive = off > 0.0
    if not alive.all():
        deps[~alive] = 0.0
        trD = np.where(alive, trD, 0.0)

    # ---- Jaumann rotation of the old stress (srota3) ---------------------
    wxy = 0.5 * (L[:, 0, 1] - L[:, 1, 0]) * dt      # spin increments W*dt
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
    return dndx, vol, lc, deps, trD


def _post(xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
          qa, qb, c, hcoef, alive, qvw_pend, dt, dtfac, mass, vol0, hgqex):
    """Bulk viscosity + internal & hourglass forces + energy increments +
    critical dt: the sbulk3 / sfint3 / shour3 / sdlen3 part of the cycle,
    everything AFTER the material law. Returns
    (fe, dt_crit, w_visc, qvw_new, deint0, dehour) with ``fe`` the
    (n, 8, 3) nodal forces (internal + hourglass, already negated for the
    fint accumulation) — the caller scatters. Mirrored by
    accel.jit_kernels.hexa_post."""
    n = len(xe)

    # ---- bulk viscosity (sbulk3) — see module docstring -------------------
    compressing = (trD < 0.0) & alive
    qvisc = np.where(
        compressing,
        rho * lc * (qa ** 2 * lc * trD ** 2 - qb * c * trD),
        0.0)

    # ---- internal nodal forces (sfint3) -----------------------------------
    # f_i = V * sigma . gradN_i   (3x3 stress from Voigt; the viscous
    # pressure adds to the three normal stresses, compression +)
    S = np.empty((n, 3, 3))
    S[:, 0, 0] = sig[:, 0] - qvisc
    S[:, 1, 1] = sig[:, 1] - qvisc
    S[:, 2, 2] = sig[:, 2] - qvisc
    S[:, 0, 1] = S[:, 1, 0] = sig[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig[:, 5]
    # fe[i,b] = -vol * sum_c S[b,c] dndx[i,c]: S is symmetric, so this is
    # the stacked matmul dndx @ S. Minus sign: fint holds -integral(B^T s)
    fe = (dndx @ S) * (-vol)[:, None, None]

    # ---- physical hourglass control (Belytschko-Bindeman) -----------------
    aa1 = (mass / np.maximum(vol0, EM20)) * c * c
    hx = _H @ xe                                       # (n, 4, 3)
    gamma = _H[None, :, :] - hx @ dndx.transpose(0, 2, 1)   # (n, 4, 8)
    traceS = np.einsum("nia,nia->n", dndx, dndx)            # sum|gradN|^2
    kstiff = hcoef * aa1 * vol * traceS * alive
    
    # accumulate the hourglass modal displacement (rate form)
    hgqex += (gamma @ ve) * dt * alive[:, None, None]
    fhg = (gamma.transpose(0, 2, 1) @ hgqex) * (-kstiff)[:, None, None]
    fe += fhg

    # ---- energy bookkeeping (units: work) ---------------------------------
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = 0.5 * vol * qvisc * (-trD * dt) + qvw_pend * (-trD)
    qvw_new = 0.5 * vol * qvisc * dt                 # booked next cycle
    deint0 = vol * np.einsum("nk,nk->n", sig_mid, deps)
    # hourglass dissipation
    dehour = -np.einsum("nib,nib->n", fhg, ve) * dt

    # ---- critical time step (sdlen3 + material) ----------------------------
    Q = np.where(compressing, qb * c + qa * lc * np.abs(trD), 0.0)
    denom = Q + np.sqrt(Q * Q + c * c)
    dt_crit = np.where(denom > 0.0, dtfac * lc / denom, EP30)
    
    # dt cap for physical hourglass
    gnorm = np.einsum("nai,nai->n", gamma, gamma)
    with np.errstate(divide="ignore", invalid="ignore"):
        dt_hg = np.where(
            kstiff > 0.0,
            DT_HG_SF * np.sqrt(np.maximum(mass, EM20)
                               / np.maximum(2.0 * kstiff * gnorm, EM20)),
            EP30)
    dt_crit = np.minimum(dt_crit, dt_hg)
    dt_crit = np.where(alive, dt_crit, EP30)
    return fe, dt_crit, w_visc, qvw_new, deint0, dehour


def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the whole brick group."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty(0)

    xe = x[conn]

    # Cycle 0 Courant step probe or evaluation without velocity
    if dt <= 0.0 or v is None:
        dndx, vol = _geometry(xe)
        lc = _char_length(xe, vol) * st.get("lc_scale", np.ones(n))
        rho = st["mass"] / np.maximum(vol, EM20)
        c = np.zeros(n)
        is_void = np.zeros(n, dtype=bool)
        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) == 0:
                is_void[sl] = True
            else:
                K = getattr(mat, "K", 0.0)
                G = getattr(mat, "G", 0.0)
                c[sl] = np.sqrt(np.maximum(K + 4.0 * G / 3.0, 0.0) / np.maximum(rho[sl], EM20))
        alive = st["off"] > 0.0
        dt_e = np.where(alive, st.get("dtfac", np.ones(n)) * lc / np.maximum(c, EM20), EP30)
        return np.where(is_void, EP30, dt_e)

    ve = v[conn]

    sig = st["sig"]
    sig_old = sig.copy()
    alive = st["off"] > 0.0

    jit = accel_get("hexa_pre")
    if jit is not None:
        dndx, vol, lc, deps, trD = jit(xe, ve, sig, dt, st["off"], st["lc_scale"])
    else:
        dndx, vol, lc, deps, trD = _pre(xe, ve, sig, dt, st["off"], st["lc_scale"])
    rho = st["mass"] / vol

    epsp_old = st["epsp"].copy() if st["chk_fail"] else None
    c = np.zeros(group.n)
    c_from_law = np.zeros(group.n, dtype=bool)
    F = None
    if "dndx0" in st:
        F = np.einsum("nia,nib->nab", xe, st["dndx0"])
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            continue
        extra = {}
        if F is not None:
            extra["F"] = F[sl]
        extra["off"] = st["off"][sl]
        for name, arr in st["mat_extra"].items():
            extra[name] = arr[sl]
        if materials.needs_env(mat) and not st.get("_impl_static_hg"):
            extra["rho"] = rho[sl]
            extra["eint"] = st["eint"][sl]
        _, _, c_new = materials.solid_update(
            mat, sig[sl], deps[sl], st["epsp"][sl], dt, extra or None)
        if c_new is not None:
            c[sl] = c_new
            c_from_law[sl] = True
        for name in st["mat_extra"]:
            if name in extra and name != "eint":
                st["mat_extra"][name][sl] = extra[name]
        if "off28" in extra:
            st["off"][sl] = np.minimum(st["off"][sl], extra["off28"])
        elif "off38" in extra:
            st["off"][sl] = np.minimum(st["off"][sl], extra["off38"])
        elif "off" in extra:
            st["off"][sl] = extra["off"]

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

    if st["chk_fail"]:
        off = st["off"]
        for sl, mat, prop in st["slices"]:
            if getattr(mat, "law", 1) == 0:
                continue
            eps_max = mat.params.get("eps_p_max", EP30)
            if mat.fail is None and eps_max >= 1e30:
                continue
            broken = np.zeros(sl.stop - sl.start, dtype=bool)
            if mat.fail is not None:
                tstar = None
                if "temp" in st["mat_extra"] and "mT" in mat.params:
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
    sig[~alive] = 0.0

    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    hcoef = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            continue
        if not c_from_law[sl.start]:
            c[sl] = np.sqrt((getattr(mat, "K", 0.0) + 4.0 * getattr(mat, "G", 0.0) / 3.0) / rho[sl])
        qa[sl] = getattr(prop, "params", {}).get("qa", 1.1)
        qb[sl] = getattr(prop, "params", {}).get("qb", 0.05)
        hcoef[sl] = getattr(prop, "params", {}).get("h", HG_PHYS)

    fe, dt_crit, w_visc, qvw_new, deint0, dehour = _post(
        xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
        qa, qb, c, hcoef, alive, st["qvw_pend"], dt, st["dtfac"],
        st["mass"], st["vol0"], st["hgqex"])

    if "eos_mask" in st:
        em = st["eos_mask"]
        st["e_eos"][em] += w_visc[em] / st["vol0"][em]
        deint = deint0 + w_visc
        st["eint"] += np.where(em, 0.0, deint)
        st["eint"][em] = st["e_eos"][em] * st["vol0"][em]
    else:
        st["eint"] += deint0 + w_visc
    st["qvw_pend"] = qvw_new
    st["ehour"] += dehour

    is_void = np.zeros(group.n, dtype=bool)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    if np.any(is_void):
        fe[is_void] = 0.0
        dt_crit[is_void] = EP30

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))

    return dt_crit


# ----------------------------------------------------------------------------
# Implicit tangent stiffness (M8)
# ----------------------------------------------------------------------------

HG_STIFF = 0.1

def _hg_operators(group, x):
    """Shared hourglass geometry for the implicit tangent + residual:
    returns (conn, gamma (n,4,8), GG (n,8,8), k_hg (n,), k_stiff (n,)) where
    ``k_hg`` = a_h (viscous, as forces() emits it at dt=1) + k_stiff, and
    ``k_stiff`` is the added FB stiffness-hourglass coefficient."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return conn, np.empty((0, 4, 8)), np.empty((0, 8, 8)), np.empty(0), np.empty(0)
    xe = x[conn]
    dndx, vol = _geometry(xe)
    vol = np.maximum(vol, EM20)
    hx = _H @ xe
    gamma = _H[None, :, :] - hx @ dndx.transpose(0, 2, 1)      # (n, 4, 8)
    GG = np.einsum("nai,naj->nij", gamma, gamma)               # (n, 8, 8)
    traceS = np.einsum("nia,nia->n", dndx, dndx)               # sum|gradN|^2
    rho = st["mass"] / vol
    n = group.n
    k_hg = np.zeros(n)
    k_stiff = np.zeros(n)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            continue
        c = np.sqrt((getattr(mat, "K", 0.0) + 4.0 * getattr(mat, "G", 0.0) / 3.0) / rho[sl])
        ah = getattr(prop, "params", {}).get("h", 0.1) * rho[sl] * c * vol[sl] ** (2.0 / 3.0) / 4.0
        ks = HG_STIFF * getattr(mat, "G", 0.0) * vol[sl] * traceS[sl]
        k_stiff[sl] = ks
        k_hg[sl] = ah + ks
    return conn, gamma, GG, k_hg, k_stiff


def static_stabilization(group, x, u, ur, fint, mint):
    """Add the static stiffness-hourglass NODAL FORCE to ``fint`` (the part of
    the implicit residual that forces() does not supply, because its hourglass
    is viscous — see the note above); ``ur``/``mint`` are unused (solids
    carry no rotational DOF). Consistent with the k_hg term of tangent().

    M13 made the hourglass deformation PERSISTENT across committed
    increments (state ``hgq``, the accumulated modal displacement,
    committed/restored with the stress by the driver's snapshot
    machinery): the original incremental form  -k_stiff (gamma . u)
    forgot the accumulated hourglass deformation at every commit — each
    increment's converged hourglass content turned into a permanent
    out-of-balance jump at the next increment's start and the modes
    RATCHETED increment by increment (exposed by a moment-loaded block
    whose corner forces excite the modes hard: every M8-M12 validation
    loads solids symmetrically enough that the term stayed invisible).
    The total hourglass force of the implicit residual is now

        f = -k_hg * (q_committed + gamma.u) . gamma,   k_hg = a_h + k_s

    of which forces() already emits the viscous  a_h * (gamma.u)  part
    (the pseudo-velocity drive), so this adds  k_s*(gamma.u) + k_hg*q  —
    the tangent's k_hg block is exactly its derivative. The stored modal
    state is updated in place (pure: the driver restores the committed
    base before every evaluation). The NLGEOM branch keeps its own
    treatment (static_internal_forces — updated-Lagrangian hourglass,
    where the committed deformation lives in the advanced frame itself);
    the stored hourglass strain energy of this term is not booked into
    ``ehour`` (a statics device; documented, like the incremental form
    before it)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return
    conn, gamma, GG, k_hg, k_stiff = _hg_operators(group, x)
    ue = u[conn]                                               # (n, 8, 3)
    modal = np.einsum("nai,nid->nad", gamma, ue)              # (n, 4, 3)
    if "hgq" not in st:
        st["hgq"] = np.zeros((n, 4, 3))
    q0 = st["hgq"]                        # committed base (just restored)
    fe = -np.einsum("nad,nai->nid",
                    k_stiff[:, None, None] * modal
                    + k_hg[:, None, None] * q0, gamma)
    q0[...] = q0 + modal                  # trial state, in place (array
    #                                       identity kept — the snapshot/
    #                                       restore contract); committed
    #                                       on convergence
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        fe[dead] = 0.0

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole brick group.

    Returns ``(ke, edofs)``:

    * ``ke``    (n, 24, 24) dense element tangents (translations only);
    * ``edofs`` (n, 24) global scalar DOF slot ids (node*6 + component) in
      the ``implicit.dofmap`` numbering — component 0,1,2 = ux,uy,uz.

    ``epsp_incr`` (n,) is the plastic-strain increment of the current load
    step, used by the LAW2 consistent tangent; None / zeros = elastic.
    """
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)
    xe = x[conn]                                   # (n, 8, 3)

    # geometry: uniform-gradient shape derivatives + volume (srcoor3/sderi3)
    dndx, vol = _geometry(xe)
    vol = np.maximum(vol, EM20)

    # ---- strain-displacement operator B (n, 6, 24), engineering shear -----
    # rows [xx, yy, zz, xy, yz, zx]; columns node-major [ux0,uy0,uz0, ...].
    B = np.zeros((n, 6, 24))
    gx, gy, gz = dndx[:, :, 0], dndx[:, :, 1], dndx[:, :, 2]   # (n, 8)
    ix = np.arange(8)
    B[:, 0, 3 * ix + 0] = gx            # eps_xx = dN_i/dx * ux_i
    B[:, 1, 3 * ix + 1] = gy            # eps_yy = dN_i/dy * uy_i
    B[:, 2, 3 * ix + 2] = gz            # eps_zz = dN_i/dz * uz_i
    B[:, 3, 3 * ix + 0] = gy            # gamma_xy = dN/dy ux + dN/dx uy
    B[:, 3, 3 * ix + 1] = gx
    B[:, 4, 3 * ix + 1] = gz            # gamma_yz = dN/dz uy + dN/dy uz
    B[:, 4, 3 * ix + 2] = gy
    B[:, 5, 3 * ix + 0] = gz            # gamma_zx = dN/dz ux + dN/dx uz
    B[:, 5, 3 * ix + 2] = gx

    # ---- constitutive stiffness  K_c = V B^T D B --------------------------
    ke = np.zeros((n, 24, 24))
    epi = np.zeros(n) if epsp_incr is None else epsp_incr
    # M14: total-form laws (LAW42) build their SPATIAL tangent from the
    # deformation gradient of the linearization geometry ``x`` — exactly
    # the F the residual's stress was evaluated at (static_internal_forces
    # re-evaluates total-form slices at the end configuration)
    F = (np.einsum("nia,nib->nab", xe, st["dndx0"])
         if "dndx0" in st else None)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            continue
        extra = ({"F": F[sl]} if F is not None
                 and materials.needs_defgrad(mat) else None)
        D = materials.solid_tangent(mat, st["sig"][sl], st["epsp"][sl],
                                    epi[sl], extra)       # (m, 6, 6)
        Bs = B[sl]
        # V * B^T D B, per element (einsum keeps it a stacked matmul)
        DB = np.einsum("mij,mjk->mik", D, Bs)             # (m, 6, 24)
        ke[sl] = vol[sl][:, None, None] * np.einsum("mji,mjk->mik", Bs, DB)

    # ---- hourglass stabilization  K_h (see the function comment) ----------
    # k_hg = a_h (viscous, matching what forces() emits at dt=1) + k_stiff
    # (the FB stiffness hourglass, also added to the residual by
    # static_stabilization) — so the tangent matches the residual exactly.
    _, _, GG, k_hg, _ = _hg_operators(group, x)
    kh = k_hg[:, None, None] * GG                          # (n, 8, 8)
    for b in range(3):
        rows = (3 * ix + b)[:, None]
        cols = (3 * ix + b)[None, :]
        ke[:, rows, cols] += kh

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the lumped mass is ``starter/source/elements/solid/solide/
# smass3.F`` (MASS = RHO*VOLU/8 spread to the 8 nodes — the value ``init_group``
# returns and the explicit leapfrog / M10 implicit dynamics divide by). The
# CONSISTENT mass is the shape-function integral M = ∫_V ρ Nᵀ N dV; the
# open-source element ships only the lumped form, so this is ported as a clean
# M16 library capability for the modal eigensolver — NEVER touching the lumped
# path (the mass analogue of tangent() sitting beside forces()).
#
# Theory (Cook, Malkus & Plesha ch. 11; Hughes "The FEM" ch. 7). Unlike the
# constant-B one-point STIFFNESS (which the hourglass block stabilizes), the
# consistent mass MUST be integrated with FULL 2×2×2 Gauss quadrature: a
# one-point evaluation would put every trilinear N_i = 1/8 at the centroid and
# give a rank-1 (physically wrong, singular) mass. With the 8 Gauss points
# ξ_g = ±1/√3 and the trilinear shape functions
#
#     N_i(ξ,η,ζ) = 1/8 (1+ξξ_i)(1+ηη_i)(1+ζζ_i)          (ξ_i,η_i,ζ_i = _XI)
#
# the block is  M[a i, b j] = δ_ij Σ_g w_g detJ_g N_a(ξ_g) N_b(ξ_g)  (w_g = 1),
# isotropic in the three translation directions (δ_ij), hence frame-invariant.
# Density ρ = m/V0 comes from the stored element mass and reference volume, so
# the mass is conserved (built on the undeformed geometry). For a rectangular /
# parallelepiped brick detJ is constant and the quadrature is EXACT: each row
# then sums to ρV/8 = m/8 (the lumped nodal mass) and ½ vᵀMv = ½ m|v|² is exact
# for rigid v. For a DISTORTED hexa N_a N_b detJ exceeds the degree the 2×2×2
# rule integrates exactly, so the mass carries the standard O(distortion²)
# quadrature error of the consistent brick mass (documented; well-shaped
# meshes — the modal validations — are unaffected).

# 2×2×2 Gauss points (rows) in (ξ,η,ζ); the 8-point rule has unit weights.
_GAUSS3 = _XI / np.sqrt(3.0)                            # reuse the node signs


def _shape8(xi):
    """Trilinear shape values N (8,) at one natural point xi = (ξ,η,ζ)."""
    return 0.125 * np.prod(1.0 + _XI * xi[None, :], axis=1)


def consistent_mass(group, x=None):
    """Consistent element mass ∫ρ Nᵀ N dV of the 8-node brick by 2×2×2 Gauss
    integration (see the note above): M[a,b] = ρ Σ_g detJ_g N_a N_b, isotropic
    over the three translations.

    Returns ``(me (n,24,24), edofs (n,24))`` — translations only, the same
    node-major addressing as ``tangent()``. ``x`` unused (the mass is built on
    the reference geometry and is frame-invariant)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)
    # density ρ = m / V0 from the stored element mass and reference volume:
    # the mass is conserved, so it is integrated on the UNDEFORMED element.
    rho = st["mass"] / np.maximum(st["vol0"], EM20)     # (n,)
    # detJ at each Gauss point needs the reference nodal coordinates; the
    # element buffer stored only the centroid dndx0, so the driver passes
    # model.x0 through ``x`` for the modal path (mass conservation).
    if x is None:
        raise ValueError("solid_hexa8.consistent_mass needs the reference "
                         "coordinates (pass model.x0)")
    xe = x[conn]                                        # (n, 8, 3)

    S = np.zeros((n, 8, 8))                             # Σ_g detJ_g Nᵀ N
    for g in range(8):
        xi = _GAUSS3[g]
        N = _shape8(xi)                                # (8,)
        # dN_i/dξ_a at this Gauss point = (xi_sign_a / 8) * prod_{b≠a}(1+..)
        dN = np.empty((8, 3))
        for a in range(3):
            other = [c for c in range(3) if c != a]
            dN[:, a] = 0.125 * _XI[:, a] * np.prod(
                1.0 + _XI[:, other] * xi[None, other], axis=1)
        J = np.einsum("ia,nib->nab", dN, xe)           # (n,3,3)
        detJ, _ = det_inv33(J)
        S += (detJ[:, None, None]) * np.einsum("i,j->ij", N, N)[None]

    me = np.zeros((n, 24, 24))
    MS = rho[:, None, None] * S                         # (n,8,8) mass factor
    ix = np.arange(8)
    for c in range(3):
        rows = (3 * ix + c)[:, None]
        cols = (3 * ix + c)[None, :]
        me[:, rows, cols] = MS
    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        me[dead] = 0.0

    return me, _edofs(conn)


# ----------------------------------------------------------------------------
# Geometric (initial-stress) stiffness K_geo (M9) — see PORTING_GUIDE M9
# ----------------------------------------------------------------------------
# Fortran origin: the geometric-stiffness branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F`` dispatching the element KGEO
# routines — the ``imp_kgeo`` path that OpenRadioss activates for its
# large-displacement implicit nonlinear analysis, /IMPL/NONLIN).
#
# Theory (BLM ch. 6.4; Bathe ch. 6.3 — the updated-Lagrangian linearization).
# Linearizing the internal virtual work at a CURRENT (stressed) configuration
# splits the tangent into the material part K_c = ∫ B^T D B dV (tangent())
# and the INITIAL-STRESS part carrying the current Cauchy stress:
#
#     K_geo[a i, b j] = δ_ij ∫ (∇N_a · σ · ∇N_b) dV
#
# — identical in every translation direction (the δ_ij), which makes it the
# term through which a membrane/axial stress resists (tension) or drives
# (compression) a TRANSVERSE perturbation: exactly the physics of stress
# stiffening and of buckling (K_c + λ K_geo singular at the critical load).
# For the one-point hexa the integrand is constant, so the integral is
# V · ∇N_a σ ∇N_b with the same centroid gradients the forces use. At zero
# stress K_geo vanishes identically — the M8 small-strain path is untouched.
# (The hourglass modes get no geometric term: they are orthogonal to the
# linear field, and their stabilization stiffness dominates any σ-scale
# correction — standard one-point-element practice.)

def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the brick group,
    from the CURRENT stress state ``st['sig']`` at geometry ``x``.

    Returns ``(ke, edofs)`` shaped exactly like ``tangent()`` (n, 24, 24) so
    the assembler can simply add it. Zero wherever the stress is zero
    (deleted elements carry zero stress, so they drop out automatically)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return np.empty((0, 24, 24)), np.empty((0, 24), dtype=np.int64)
    dndx, vol = _geometry(x[conn])
    vol = np.maximum(vol, EM20)

    # Cauchy stress as a 3x3 per element (Voigt [xx, yy, zz, xy, yz, zx])
    s = st["sig"]
    S = np.empty((n, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = s[:, 5]

    # g_ab = V * gradN_a . sigma . gradN_b   (n, 8, 8), replicated over the
    # three translation directions (the delta_ij of the derivation above)
    g = vol[:, None, None] * np.einsum("nac,ncd,nbd->nab", dndx, S, dndx)
    ke = np.zeros((n, 24, 24))
    ix = np.arange(8)
    for b in range(3):
        rows = (3 * ix + b)[:, None]
        cols = (3 * ix + b)[None, :]
        ke[:, rows, cols] += g

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        ke[dead] = 0.0

    return ke, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal force at configuration ``x`` from the CURRENT stress
    state — the updated-Lagrangian force assembly of the M9 implicit residual
    (see ``implicit.statics._internal_forces``: the stress was just updated
    by a ``forces()`` call at the MIDPOINT geometry — the Hughes–Winget
    objective increment — and this routine re-assembles the nodal force at
    the END geometry, where equilibrium is stated). It is the sfint3.F
    force expression evaluated standalone:

        f_i = - V * sigma . gradN_i        (negated-internal convention)

    plus the hourglass stabilization force at the same configuration with
    the SAME total modal stiffness k_hg = a_h + k_stiff the tangent carries
    (a_h is what forces() emits at dt=1, k_stiff the static FB hourglass of
    ``static_stabilization`` — here both are applied in one term so residual
    and tangent stay consistent in the nonlinear-geometry path). ``ur`` and
    ``mint`` are unused (solids carry no rotational DOFs)."""
    st = group.state
    conn = group.conn
    n = group.n
    if n == 0 or len(conn) == 0:
        return
    dndx, vol = _geometry(x[conn])
    vol = np.maximum(vol, EM20)
    s = st["sig"]
    # M14: TOTAL-form laws (LAW42) re-evaluate their stress at THIS (end)
    # configuration — the midpoint forces() call left sigma(F_mid) in the
    # state, which is the right objective INCREMENT for the hypoelastic
    # laws but simply the wrong configuration for a law whose stress is a
    # pure function of F. The re-evaluation is exact and free of drift
    # (F comes from the stored initial gradients), overwrites the state
    # in place (a pure function — nothing is lost), and is what makes the
    # M14 LAW42 tangent CONSISTENT with the residual assembled here.
    if "dndx0" in st:
        F = np.einsum("nia,nib->nab", x[conn], st["dndx0"])
        for sl, mat, prop in st["slices"]:
            if getattr(mat, "law", 1) == 0:
                continue
            if materials.needs_defgrad(mat):
                materials.solid_update(mat, s[sl], np.zeros((sl.stop - sl.start, 6)),
                                       st["epsp"][sl], 1.0, {"F": F[sl]})
    S = np.empty((group.n, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = s[:, 0], s[:, 1], s[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = s[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = s[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = s[:, 5]
    # f_i = -V sigma gradN_i  (n, 8, 3)
    fe = -vol[:, None, None] * np.einsum("nid,ncd->nic", dndx, S)
    # hourglass stabilization at this configuration: -k_hg (gamma.u) gamma
    _, gamma, _, k_hg, _ = _hg_operators(group, x)
    modal = np.einsum("nai,nid->nad", gamma, u[conn])          # (n, 4, 3)
    fe -= k_hg[:, None, None] * np.einsum("nad,nai->nid", modal, gamma)

    is_void = np.zeros(n, dtype=bool)
    for sl, mat, prop in st["slices"]:
        if getattr(mat, "law", 1) == 0:
            is_void[sl] = True
    dead = (st["off"] <= 0.0) | is_void
    if np.any(dead):
        fe[dead] = 0.0

    if fint is not None:
        scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3), st.get('color_indices'), st.get('color_offsets'))
