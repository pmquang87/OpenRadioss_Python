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

from .. import materials
from ..common.constants import EM20

# dN_i/dxi_a of the linear tetrahedron with natural coordinates
#   N1 = 1 - xi1 - xi2 - xi3,  N2 = xi1,  N3 = xi2,  N4 = xi3
# (Radioss /TETRA4 node ordering: base triangle 1-2-3 counter-clockwise
#  seen from node 4, i.e. node 4 on the positive-normal side).
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
    # J[a,b] = d x_b / d xi_a  (edge vectors from node 1)
    J = np.einsum("ia,nib->nab", _DN_DXI, xe)
    detJ = np.linalg.det(J)
    vol = detJ / 6.0                      # tet volume = det(edges)/6
    Jinv = np.linalg.inv(J)
    # dN_i/dx_b = dN_i/dxi_a * dxi_a/dx_b ; dxi_a/dx_b = inv(J)[b,a]
    dndx = np.einsum("ia,nba->nib", _DN_DXI, Jinv)
    return dndx, vol


def _char_length(xe: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Characteristic length = minimum altitude = 3 V / max face area."""
    amax = np.zeros(len(xe))
    for f in _FACES:
        e1 = xe[:, f[1], :] - xe[:, f[0], :]
        e2 = xe[:, f[2], :] - xe[:, f[0], :]
        a = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        amax = np.maximum(amax, a)
    return 3.0 * vol / np.maximum(amax, EM20)


def _exact_dt_factor(dndx: np.ndarray, vol: np.ndarray, lc: np.ndarray,
                     slices) -> np.ndarray:
    """Per-element ratio dt_exact / (lc/c) — see the module docstring and
    solid_hexa8._exact_dt_factor (same construction, nodal mass rho*V/4)."""
    n = len(vol)
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
        w2max = (4.0 / mat.rho0) * eig.real.max(axis=1)   # m = rho*V/4
        c = mat.sound_speed_solid()
        dt_exact = 2.0 / np.sqrt(np.maximum(w2max, EM20))
        fac[sl] = np.minimum(dt_exact / (lc[sl] / c), 1.0)
    return fac


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def init_group(group, model, log):
    """Element buffer + lumped mass (starter s4init3/s4mass3): volume from
    the initial geometry, element mass rho0*V spread equally to the 4
    nodes."""
    xe = model.x0[group.conn]                      # (n, 4, 3)
    dndx0, vol = _geometry(xe)
    bad = vol <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/TETRA4 {eid}: zero or negative volume "
                      f"(check node ordering)", "TETRA INIT")
    n = group.n
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        rho0[sl] = mat.rho0
    mass = rho0 * vol

    lc0 = _char_length(xe, vol)
    group.state.update(
        sig=np.zeros((n, 6)),        # Cauchy stress, Voigt (GBUF%SIG)
        epsp=np.zeros(n),            # equivalent plastic strain (GBUF%PLA)
        vol0=vol.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),           # always zero: no hourglass modes (doc)
        dtfac=_exact_dt_factor(dndx0, vol, lc0, group.state["slices"]),
    )
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    return node_idx, mass_c, None


# ----------------------------------------------------------------------------
# Engine-side force computation (one cycle)
# ----------------------------------------------------------------------------

def forces(group, x, v, vr, dt, fint, mint):
    """One explicit cycle for the whole tetra group (s4forc3.F chain).
    Returns the per-element critical time step."""
    st = group.state
    conn = group.conn
    xe = x[conn]                                   # (n, 4, 3) gather
    ve = v[conn]

    # ---- geometry (s4coor3) ----------------------------------------------
    dndx, vol = _geometry(xe)
    vol = np.maximum(vol, EM20)
    rho = st["mass"] / vol
    lc = _char_length(xe, vol)

    # ---- velocity gradient, D and W (s4defo3) ------------------------------
    L = np.einsum("nib,nic->nbc", ve, dndx)
    D = 0.5 * (L + np.transpose(L, (0, 2, 1)))
    trD = D[:, 0, 0] + D[:, 1, 1] + D[:, 2, 2]
    deps = np.empty((group.n, 6))                  # Voigt, engineering shear
    deps[:, 0] = D[:, 0, 0] * dt
    deps[:, 1] = D[:, 1, 1] * dt
    deps[:, 2] = D[:, 2, 2] * dt
    deps[:, 3] = 2.0 * D[:, 0, 1] * dt
    deps[:, 4] = 2.0 * D[:, 1, 2] * dt
    deps[:, 5] = 2.0 * D[:, 0, 2] * dt

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
    for sl, mat, prop in st["slices"]:
        materials.solid_update(mat, sig[sl], deps[sl], st["epsp"][sl], dt)

    # ---- sound speed & bulk viscosity (sbulk3) ------------------------------
    c = np.zeros(group.n)
    qa = np.zeros(group.n)
    qb = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        c[sl] = np.sqrt((mat.K + 4.0 * mat.G / 3.0) / rho[sl])
        qa[sl] = prop.params["qa"]
        qb[sl] = prop.params["qb"]
    compressing = trD < 0.0
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
    sig_mid = 0.5 * (sig_old + sig)
    st["eint"] += vol * np.einsum("nk,nk->n", sig_mid, deps) \
        + vol * qvisc * (-trD * dt)

    # ---- scatter to global arrays (asspar) ----------------------------------
    np.add.at(fint, conn.reshape(-1), fe.reshape(-1, 3))

    # ---- critical time step --------------------------------------------------
    Q = np.where(compressing, qb * c + qa * lc * np.abs(trD), 0.0)
    return st["dtfac"] * lc / (Q + np.sqrt(Q * Q + c * c))
