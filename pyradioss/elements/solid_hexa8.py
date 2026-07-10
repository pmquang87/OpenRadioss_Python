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
"""

from __future__ import annotations

import numpy as np

from .. import materials
from ..common.constants import EM20

# Node sign pattern of the trilinear hexa (Radioss /BRICK node ordering:
# nodes 1-4 = bottom face counter-clockwise, 5-8 = top face).
_XI = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
], dtype=float)
# dN_i/dxi_a at the centroid = xi_sign/8 (uniform gradient operator)
_DN_DXI = _XI / 8.0

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
    # J[a,b] = d x_b / d xi_a  summed over nodes
    J = np.einsum("ia,nib->nab", _DN_DXI, xe)
    detJ = np.linalg.det(J)
    vol = 8.0 * detJ
    # dN_i/dx_b = dN_i/dxi_a * dxi_a/dx_b ; dxi_a/dx_b = inv(J)[b,a]
    Jinv = np.linalg.inv(J)
    dndx = np.einsum("ia,nba->nib", _DN_DXI, Jinv)
    return dndx, vol


def _char_length(xe: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Characteristic length lc = V / max face area (sdlen3.F).

    Face area from the cross product of its diagonals: for a (possibly
    warped) quad face with corners a,b,c,d the vector area is
    0.5 * (c-a) x (d-b).
    """
    amax = np.zeros(len(xe))
    for f in _FACES:
        d1 = xe[:, f[2], :] - xe[:, f[0], :]
        d2 = xe[:, f[3], :] - xe[:, f[1], :]
        a = 0.5 * np.linalg.norm(np.cross(d1, d2), axis=1)
        amax = np.maximum(amax, a)
    return vol / np.maximum(amax, EM20)


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
        # exact stability correction to the lc/c estimate (module docstring)
        dtfac=_exact_dt_factor(dndx0, vol, lc0, group.state["slices"]),
    )
    # nodal mass: 1/8 of the element mass to each node
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 8.0, 8)
    return node_idx, mass_c, None


# ----------------------------------------------------------------------------
# Engine-side force computation (one cycle)
# ----------------------------------------------------------------------------

def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the whole brick group. See module docstring
    for the sforc3.F call chain this reproduces. Returns the per-element
    critical time step."""
    st = group.state
    conn = group.conn
    xe = x[conn]                                   # (n, 8, 3) gather
    ve = v[conn]

    # ---- geometry at t_{n+1/2} (srcoor3) --------------------------------
    dndx, vol = _geometry(xe)
    vol = np.maximum(vol, EM20)
    rho = st["mass"] / vol                          # current density
    lc = _char_length(xe, vol)

    # ---- velocity gradient, D and W (sdefo3) -----------------------------
    L = np.einsum("nib,nic->nbc", ve, dndx)         # L = sum v_i (x) gradN_i
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    trD = D[:, 0, 0] + D[:, 1, 1] + D[:, 2, 2]
    # strain increment in Voigt form, ENGINEERING shear (gamma = 2 eps)
    deps = np.empty((group.n, 6))
    deps[:, 0] = D[:, 0, 0] * dt
    deps[:, 1] = D[:, 1, 1] * dt
    deps[:, 2] = D[:, 2, 2] * dt
    deps[:, 3] = 2.0 * D[:, 0, 1] * dt
    deps[:, 4] = 2.0 * D[:, 1, 2] * dt
    deps[:, 5] = 2.0 * D[:, 0, 2] * dt

    # ---- Jaumann rotation of the old stress (srota3) ---------------------
    sig = st["sig"]
    sig_old = sig.copy()                            # kept for the energy
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

    # ---- material law per part slice (mmain -> sigeps) -------------------
    for sl, mat, prop in st["slices"]:
        materials.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt)

    # ---- sound speed & bulk viscosity (sbulk3) ----------------------------
    c = np.zeros(group.n)
    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        # current sound speed uses current density (stiffness constant)
        c[sl] = np.sqrt((mat.K + 4.0 * mat.G / 3.0) / rho[sl])
        qa[sl] = prop.params["qa"]
        qb[sl] = prop.params["qb"]
    compressing = trD < 0.0
    qvisc = np.where(
        compressing,
        rho * lc * (qa ** 2 * lc * trD ** 2 - qb * c * trD),
        0.0)
    # viscous pressure adds to the three normal stresses (compression +)
    sig_tot = sig.copy()
    sig_tot[:, 0] -= qvisc
    sig_tot[:, 1] -= qvisc
    sig_tot[:, 2] -= qvisc

    # ---- internal nodal forces (sfint3) -----------------------------------
    # f_i = V * sigma . gradN_i   (3x3 stress from Voigt)
    S = np.empty((group.n, 3, 3))
    S[:, 0, 0], S[:, 1, 1], S[:, 2, 2] = sig_tot[:, 0], sig_tot[:, 1], sig_tot[:, 2]
    S[:, 0, 1] = S[:, 1, 0] = sig_tot[:, 3]
    S[:, 1, 2] = S[:, 2, 1] = sig_tot[:, 4]
    S[:, 0, 2] = S[:, 2, 0] = sig_tot[:, 5]
    fe = -np.einsum("n,nbc,nic->nib", vol, S, dndx)   # (n, 8, 3), minus sign:
    # accumulated so that fint holds  -integral(B^T sigma)  (see package doc)

    # ---- hourglass control (shour3, viscous Flanagan-Belytschko) ----------
    # gamma_ai = h_ai - (sum_j h_aj x_j.) gradN_i  : hourglass shape vectors
    # orthogonalized against the linear field so pure deformation produces
    # no hourglass force (essential for coarse-mesh bending accuracy).
    hx = np.einsum("ai,nib->nab", _H, xe)             # (n, 4, 3)
    gamma = _H[None, :, :] - np.einsum("nab,nib->nai", hx, dndx)  # (n,4,8)
    qdot = np.einsum("nai,nib->nab", gamma, ve)       # modal velocities
    hcoef = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        hcoef[sl] = prop.params["h"]
    # viscous coefficient (FB 1981 eq. 79 flavour): a = h*rho*c*V^(2/3)/4
    ah = hcoef * rho * c * vol ** (2.0 / 3.0) / 4.0
    fhg = -np.einsum("n,nab,nai->nib", ah, qdot, gamma)
    fe += fhg

    # ---- energy bookkeeping (units: work) ---------------------------------
    # internal energy: midpoint rule  dE = V * sigma_mid : deps
    sig_mid = 0.5 * (sig_old + sig)
    st["eint"] += vol * np.einsum("nk,nk->n", sig_mid, deps) \
        + vol * qvisc * (-trD * dt)                  # bulk viscosity work
    # hourglass dissipation: - f_hg . v * dt  (>= 0 for viscous control)
    st["ehour"] += -np.einsum("nib,nib->n", fhg, ve) * dt

    # ---- scatter to global arrays (asspar) ---------------------------------
    np.add.at(fint, conn.reshape(-1), fe.reshape(-1, 3))

    # ---- critical time step (sdlen3 + material) ----------------------------
    # the bulk-viscosity pressure stiffens the response, eroding the
    # Courant limit — but only where it acts, i.e. in compression:
    Q = np.where(compressing, qb * c + qa * lc * np.abs(trD), 0.0)
    dt_crit = st["dtfac"] * lc / (Q + np.sqrt(Q * Q + c * c))
    return dt_crit
