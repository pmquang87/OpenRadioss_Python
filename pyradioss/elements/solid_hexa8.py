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
  which must be stabilized — that is the role of shour3/this file's
  hourglass block.

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


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------

def _geometry(xe: np.ndarray):
    """Centroid Jacobian, volume and cartesian shape gradients.

    xe : (n, 8, 3) nodal coordinates.
    Returns (dndx (n,8,3), vol (n,)). Fortran: srcoor3.F + sderi3.F.
    """
    # J[a,b] = d x_b / d xi_a  summed over nodes: (3,8) @ (n,8,3) matmul
    J = _DN_DXI_T @ xe
    # explicit 3x3 cofactor det/inverse (fastmath — LAPACK is ~5x slower
    # at group sizes and not reproducible by the numba mirror)
    detJ, Jinv = det_inv33(J)
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
        eig = np.linalg.eigvals(C[None, :, :] @ BBt[sl])
        w2max = (8.0 / mat.rho0) * eig.real.max(axis=1)
        c = mat.sound_speed_solid()
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
    xe = model.x0[group.conn]                      # (n, 8, 3)
    dndx0, vol = _geometry(xe)
    bad = vol <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/BRICK {eid}: zero or negative volume "
                      f"(check node ordering)", "SOLID INIT")
    n = group.n
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = mat.rho0
    mass = rho0 * vol

    lc0 = _char_length(xe, vol)
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

def _pre(xe, ve, sig, dt, off):
    """Geometry + kinematics + Jaumann rotation: the srcoor3 / sdefo3 /
    srota3 / sdlen3-geometry part of the cycle, everything BEFORE the
    material law. Rotates ``sig`` in place; returns
    (dndx, vol, lc, deps, trD). Mirrored by accel.jit_kernels.hexa_pre
    (same formulas, element-serial — the M7 parity contract)."""
    n = len(xe)

    # ---- geometry at t_{n+1/2} (srcoor3) --------------------------------
    dndx, vol = _geometry(xe)
    vol = np.maximum(vol, EM20)
    lc = _char_length(xe, vol)

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
          qa, qb, c, hcoef, alive, qvw_pend, dt, dtfac):
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

    # ---- hourglass control (shour3, viscous Flanagan-Belytschko) ----------
    # gamma_ai = h_ai - (sum_j h_aj x_j.) gradN_i  : hourglass shape vectors
    # orthogonalized against the linear field so pure deformation produces
    # no hourglass force (essential for coarse-mesh bending accuracy).
    hx = _H @ xe                                       # (n, 4, 3)
    gamma = _H[None, :, :] - hx @ dndx.transpose(0, 2, 1)   # (n, 4, 8)
    qdot = gamma @ ve                                  # modal velocities
    # viscous coefficient (FB 1981 eq. 79 flavour): a = h*rho*c*V^(2/3)/4
    # (deleted elements exert no hourglass force either)
    ah = hcoef * rho * c * vol ** (2.0 / 3.0) / 4.0 * alive
    # fhg[i,b] = -ah * sum_a qdot[a,b] gamma[a,i]
    fhg = (gamma.transpose(0, 2, 1) @ qdot) * (-ah)[:, None, None]
    fe += fhg

    # ---- energy bookkeeping (units: work) ---------------------------------
    # internal energy: midpoint rule  dE = V * sigma_mid : deps
    #
    # Bulk-viscosity work — the M6 fix of an M1-era misbooking. The
    # viscous nodal force built from q^n acts through the COMING velocity
    # update: its kinetic-energy extraction over the cycle is exactly
    #     W = V q^n * ( -(trD(v^{n-1/2}) + trD(v^{n+1/2})) / 2 ) * dt
    # (a force f changes the leapfrog KE by f . (v_old + v_new)/2 dt — the
    # same midstep identity as the M4 contact-work lesson). Booking the
    # whole of W at trD(v^{n-1/2}) — the M1 form — is fine while trD barely
    # changes per cycle, but under BARELY-RESOLVED RINGING (strain-rate
    # sign flipping every cycle on a single element through the thickness)
    # trD(v^{n+1/2}) ~ -trD(v^{n-1/2}): the ledger then books a full
    # dissipation the damper never extracted, and the balance drifts
    # SECULARLY (the 2x2x2-cube reproducer read -35% over 2 ms of free
    # flight at /DT 0.9). The elastic sigma_mid : deps term has no such
    # secular mode — stress is a state function, its booking error cannot
    # accumulate — which is why the qb linear damper was the isolated
    # culprit. Fix: trapezoidal booking — half the viscous work now (at
    # trD of v^{n-1/2}), half DEFERRED one cycle, when the next forces()
    # call holds v^{n+1/2} in its trD (the O(dt) geometry difference
    # between the two evaluations is oscillatory, not secular).
    sig_mid = 0.5 * (sig_old + sig)
    w_visc = 0.5 * vol * qvisc * (-trD * dt) + qvw_pend * (-trD)
    qvw_new = 0.5 * vol * qvisc * dt                 # booked next cycle
    deint0 = vol * np.einsum("nk,nk->n", sig_mid, deps)
    # hourglass dissipation: - f_hg . v * dt  (>= 0 for viscous control)
    dehour = -np.einsum("nib,nib->n", fhg, ve) * dt

    # ---- critical time step (sdlen3 + material) ----------------------------
    # the bulk-viscosity pressure stiffens the response, eroding the
    # Courant limit — but only where it acts, i.e. in compression:
    Q = np.where(compressing, qb * c + qa * lc * np.abs(trD), 0.0)
    dt_crit = dtfac * lc / (Q + np.sqrt(Q * Q + c * c))
    # deleted elements no longer constrain the global step
    dt_crit = np.where(alive, dt_crit, EP30)
    return fe, dt_crit, w_visc, qvw_new, deint0, dehour


def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the whole brick group. See module docstring
    for the sforc3.F call chain this reproduces (and for the M7 pre/post
    split around the material loop). Returns the per-element critical
    time step."""
    st = group.state
    conn = group.conn
    xe = x[conn]                                   # (n, 8, 3) gather
    ve = v[conn]

    sig = st["sig"]
    sig_old = sig.copy()                           # kept for the energy
    alive = st["off"] > 0.0

    # ---- pre block: geometry, D & W, Jaumann rotation ---------------------
    # (dispatched to the numba mirror when that backend is active)
    jit = accel_get("hexa_pre")
    if jit is not None:
        dndx, vol, lc, deps, trD = jit(xe, ve, sig, dt, st["off"])
    else:
        dndx, vol, lc, deps, trD = _pre(xe, ve, sig, dt, st["off"])
    rho = st["mass"] / vol                          # current density

    # ---- material law per part slice (mmain -> sigeps) -------------------
    # laws may return their own sound speed (Fortran SOUNDSP): LAW42's
    # tangent stiffness grows with stretch, so its c MUST feed the dt —
    # and so does an /EOS, whose bulk stiffness is state-dependent (M6).
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None
    c = np.zeros(group.n)
    c_from_law = np.zeros(group.n, dtype=bool)
    F = None
    if "dndx0" in st:
        # exact deformation gradient at the point: F = sum_i x_i (x) gradN0_i
        F = np.einsum("nia,nib->nab", xe, st["dndx0"])
    for sl, mat, prop in st["slices"]:
        extra = {}
        if F is not None:
            extra["F"] = F[sl]
        for name, arr in st["mat_extra"].items():
            extra[name] = arr[sl]
        _, _, c_new = materials.solid_update(
            mat, sig[sl], deps[sl], st["epsp"][sl], dt, extra or None)
        if c_new is not None:
            c[sl] = c_new
            c_from_law[sl] = True

        # ---- /EOS pressure (M6, eosmain): replace the law's pressure by
        # the implicit E-p update — deviator from the law, pressure from
        # the equation of state; see materials/eos.py for the theory
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
            # deviator work per unit REFERENCE volume (vol/vol0 = J)
            de_dev = J * np.einsum("nk,nk->n", s_mid, deps[sl])
            p_new, e_new, c2 = eos_mod.update(
                mat.eos, mu, dv, st["e_eos"][sl], st["p_eos"][sl], de_dev)
            p_new = np.where(live, p_new, st["p_eos"][sl])   # frozen dead
            e_new = np.where(live, e_new, st["e_eos"][sl])
            st["p_eos"][sl] = p_new
            st["e_eos"][sl] = e_new
            sgsl[:] = s_new
            sgsl[:, 0] -= p_new
            sgsl[:, 1] -= p_new
            sgsl[:, 2] -= p_new
            # EOS bulk stiffness + the law's shear feeds the time step
            c[sl] = np.sqrt(c2 + (4.0 * mat.G / 3.0) / rho[sl])
            c_from_law[sl] = True

    # ---- failure models + eps_p_max deletion (engine/source/materials/
    # fail/, see pyradioss/failure/) — after the law so the damage sees
    # the updated stress state and the plastic-strain increment ----------
    if st["chk_fail"]:
        off = st["off"]
        for sl, mat, prop in st["slices"]:
            eps_max = mat.params.get("eps_p_max", EP30)
            if mat.fail is None and eps_max >= 1e30:
                continue
            broken = np.zeros(sl.stop - sl.start, dtype=bool)
            if mat.fail is not None:
                # homologous temperature for /FAIL/JOHNSON D5 (M6)
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
        alive = off > 0.0
        sig[~alive] = 0.0            # a deleted element carries no stress

    # ---- sound speed & bulk-viscosity coefficients per slice --------------
    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    hcoef = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        # current sound speed uses current density (stiffness constant);
        # laws that returned their own (nonlinear) c keep it
        if not c_from_law[sl.start]:
            c[sl] = np.sqrt((mat.K + 4.0 * mat.G / 3.0) / rho[sl])
        qa[sl] = prop.params["qa"]
        qb[sl] = prop.params["qb"]
        hcoef[sl] = prop.params["h"]

    # ---- post block: viscosity, forces, hourglass, energies, dt -----------
    # (dispatched to the numba mirror when that backend is active)
    jit = accel_get("hexa_post")
    if jit is not None:
        fe, dt_crit, w_visc, qvw_new, deint0, dehour = jit(
            xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
            qa, qb, c, hcoef, alive, st["qvw_pend"], dt, st["dtfac"])
    else:
        fe, dt_crit, w_visc, qvw_new, deint0, dehour = _post(
            xe, ve, dndx, vol, lc, rho, trD, deps, sig, sig_old,
            qa, qb, c, hcoef, alive, st["qvw_pend"], dt, st["dtfac"])

    if "eos_mask" in st:
        # /EOS elements (M6): their energy equation already integrated
        # the deviator + pdV work implicitly (the law-loop EOS block);
        # the viscous SHOCK HEATING is added to that energy state (it is
        # what puts computed shocks on the Hugoniot instead of the
        # isentrope — see materials/eos.py) and eint mirrors it.
        em = st["eos_mask"]
        st["e_eos"][em] += w_visc[em] / st["vol0"][em]
        deint = deint0 + w_visc
        st["eint"] += np.where(em, 0.0, deint)
        st["eint"][em] = st["e_eos"][em] * st["vol0"][em]
    else:
        st["eint"] += deint0 + w_visc
    st["qvw_pend"] = qvw_new
    st["ehour"] += dehour

    # ---- scatter to global arrays (asspar) ---------------------------------
    scatter_add3(fint, conn.reshape(-1), fe.reshape(-1, 3))

    return dt_crit
