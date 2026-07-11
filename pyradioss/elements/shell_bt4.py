"""
4-node Belytschko–Tsay shell element (/SHELL + /PROP/SHELL, Ishell=1).

Fortran origin: ``engine/source/elements/shell/coque/`` — the cycle path is

    cforc3.F   driver (gather, frame, call chain, scatter)
    ccoor3.F   corotational frame + projection to local coordinates
    cdefo3.F   membrane velocity strains
    cdlen3.F   characteristic length / time step
    czforc3.F / cbilan.F  bending + shear rates, resultants, forces
    chour3.F   hourglass control
    + the plane-stress material calls sigeps..c.F per integration layer

Theory (Belytschko, Lin & Tsay, CMAME 42 (1984) 225-251; also BLM ch. 9):

* **Corotational frame**: a local orthonormal triad (e1, e2, e3) is built
  from the current geometry each cycle: e3 is normal to the element
  (cross product of the diagonals — insensitive to in-plane node
  numbering), e1 is side 1-2 projected onto the plane, e2 = e3 x e1.
  All rates are measured in this frame, which removes the large rigid
  rotation from the formulation — objective stress rates are then
  unnecessary for the (small) in-frame rotations.  This is the classic
  explicit-shell trick: accuracy O(element rotation per step), perfectly
  adequate at explicit time steps.

* **One-point quadrature** in the plane (element center), NIP-point
  Gauss quadrature through the thickness. The mid-plane gradient operator
  for a quad with local corner coords (x_i, y_i):

      B1 = [y2-y4, y3-y1, y4-y2, y1-y3] / (2A)
      B2 = [x4-x2, x1-x3, x2-x4, x3-x1] / (2A)

* **Mindlin-Reissner kinematics** (first-order shear deformable):
  velocity of a point at distance z from the mid-plane is
  v(z) = v_m + z * (theta_dot x e3), giving

      membrane rates   d_xx = B1.vx, d_yy = B2.vy,
                       d_xy = B1.vy + B2.vx           (engineering)
      curvature rates  k_xx = B1.thy, k_yy = -B2.thx,
                       k_xy = B2.thy - B1.thx
      shear rates      g_xz = B1.vz + mean(thy),
                       g_yz = B2.vz - mean(thx)

  strain rate at layer z:  d(z) = d_m + z * k. Each layer is updated by
  the plane-stress material law; transverse shear is elastic with the 5/6
  correction factor (BT assumption, matches the original for elastic
  shear).

* **Resultants** (force/length and moment/length):
      N = sum_k w_k sigma_k,  M = sum_k w_k z_k sigma_k,
      q = kappa * G * t * gamma
  and the internal nodal forces/moments follow from the virtual power
  identity  P = A (N:d_m + M:k + q.g)  — the exact transpose of the rate
  operators above (each B-term in a rate produces the matching force
  term; see the code, it is written line by line against the rates).

* **Hourglass control** (chour3): one-point quadrature leaves 5 zero-
  energy modes (2 membrane, 1 transverse 'w', 2 bending) with the pattern
  h = (1,-1,1,-1). The stabilizing shape vector is orthogonalized against
  the linear field (Flanagan-Belytschko), gamma_i = h_i - (h.x) B1i -
  (h.y) B2i, so pure deformation and rigid motion produce no hourglass
  force. **Stiffness type** (the BLT84 paper's own control, and the
  Radioss default): each mode carries a persistent generalized force Q
  integrated in rate form,

      Q += k_mode * (gamma . velocity) * dt,   f_i = -Q * gamma_i

  with the stiffness scaled from the matching physical stiffness of the
  element (the classic BLT84 / LS-DYNA calibration constants):

      membrane   k_m = hm * E  t   A (B1.B1 + B2.B2) / 8
      transverse k_w = hf * kGA/t->  kappa G t A (B1.B1 + B2.B2) / 8
      bending    k_r = hr * E t^3 A (B1.B1 + B2.B2) / 192

  Unlike the M1 viscous form, stiffness control stores (rather than
  dissipates) the hourglass energy — coarse dynamic bending is no longer
  artificially damped. Q is a scalar modal amplitude, so it transports
  exactly under the corotational frame (no rotation bookkeeping needed).
  The added frequency is O(sqrt(hm)) of the membrane one (~10% at the
  default hm = 0.01), comfortably inside the /DT scale factor 0.9 —
  verified by the rigid-body and vibration validations in tests/.

* **Lumped inertia**: m_i = rho t A / 4; rotational inertia
  I_i = m_i (t^2 + A) / 12 — deliberately generous (Key's trick) to push
  the rotational stability limit up toward the membrane one. It does NOT
  always clear it: for thick or large elements the transverse-shear /
  rotation branch (stiffness ~ kappa G t A) still governs, which is why
  the Starter computes the exact eigenvalue of BOTH branches (see
  _exact_dt_factor — an M2 fix after a nu=0 strip diverged at /DT 0.9).

M7 performance structure
------------------------
forces() is split around the Python layer/material loop into ``_pre``
(corotational frame, local geometry, rate kinematics — ccoor3/cdefo3)
and ``_post`` (resultant nodal forces, BLT84 hourglass, back-transform —
czforc3/chour3), each with an optional numba mirror in
``pyradioss.accel.jit_kernels`` (see the accel package docstring for the
architecture and the parity contract). The NumPy code here is the
reference. The M7 profiling pass fused the per-rate einsum calls into
ONE stacked matmul (all ten B·v dot products at once: Bt (n,2,4) @
V (n,4,5), with V the local [vx,vy,vz,thx,thy]), replaced np.cross /
np.linalg.norm with the bitwise-identical fastmath forms, and replaced
np.add.at with the bincount scatter — see common/fastmath.py.
"""

from __future__ import annotations

import numpy as np

from .. import failure, materials
from ..accel import get as accel_get
from ..common.constants import EM20, EP30, SHEAR_FACTOR
from ..common.fastmath import cross3, norm3, scatter_add3

# side-index helper for the characteristic length: side i = (i, i+1)
_NEXT = np.array([1, 2, 3, 0])


# ----------------------------------------------------------------------------
# geometry: corotational frame and local coordinates (ccoor3.F)
# ----------------------------------------------------------------------------

def _frame(xe: np.ndarray):
    """Build the corotational triad E = [e1|e2|e3] per element.

    xe: (n, 4, 3). Returns E (n, 3, 3) with COLUMNS e1, e2, e3.
    """
    r31 = xe[:, 2] - xe[:, 0]
    r42 = xe[:, 3] - xe[:, 1]
    e3 = cross3(r31, r42)
    e3 /= np.maximum(norm3(e3), EM20)[:, None]
    s1 = xe[:, 1] - xe[:, 0]
    e1 = s1 - (np.einsum("nb,nb->n", s1, e3))[:, None] * e3
    e1 /= np.maximum(norm3(e1), EM20)[:, None]
    e2 = cross3(e3, e1)
    return np.stack([e1, e2, e3], axis=2)


def _local_geometry(xe: np.ndarray):
    """Frame, local corner coordinates, area and gradient operators."""
    E = _frame(xe)
    center = xe.mean(axis=1)
    # local coords: xl[n,i,a] = (x_i - c) . e_a  — one stacked matmul
    xl = (xe - center[:, None, :]) @ E
    x, y = xl[:, :, 0], xl[:, :, 1]
    area = 0.5 * ((x[:, 2] - x[:, 0]) * (y[:, 3] - y[:, 1])
                  + (x[:, 1] - x[:, 3]) * (y[:, 2] - y[:, 0]))
    inv2A = 1.0 / np.maximum(2.0 * area, EM20)
    B1 = np.empty((len(xe), 4))
    B1[:, 0] = y[:, 1] - y[:, 3]
    B1[:, 1] = y[:, 2] - y[:, 0]
    B1[:, 2] = y[:, 3] - y[:, 1]
    B1[:, 3] = y[:, 0] - y[:, 2]
    B1 *= inv2A[:, None]
    B2 = np.empty((len(xe), 4))
    B2[:, 0] = x[:, 3] - x[:, 1]
    B2[:, 1] = x[:, 0] - x[:, 2]
    B2[:, 2] = x[:, 1] - x[:, 3]
    B2[:, 3] = x[:, 2] - x[:, 0]
    B2 *= inv2A[:, None]
    return E, xl, area, B1, B2


def _char_length(xl: np.ndarray, area: np.ndarray) -> np.ndarray:
    """lc = A / longest side (cdlen3.F flavour) — all 4 sides at once."""
    d = xl[:, _NEXT, :2] - xl[:, :, :2]              # (n, 4, 2) side vectors
    lmax = (d[:, :, 0] ** 2 + d[:, :, 1] ** 2).max(axis=1)
    return area / np.maximum(np.sqrt(lmax), EM20)


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def _bend_shear_omega2(B1, B2, area, sl, mat, t, nnode, rho) -> np.ndarray:
    """Exact max eigenfrequency^2 of the bending/shear branch of a FLAT
    one-point shell element (membrane decouples on a flat element).

    Local dofs (w_i, thx_i, thy_i); generalized rates (kxx, kyy, kxy,
    gx, gy) from the SAME operators as forces() — see the kinematics
    there; K = A * B^T C_b B with C_b = diag(t^3/12 * C_planestress,
    kappa*G*t * I2); lumped mass m = rho t A/nnode and the (deliberately
    generous) rotational inertia I = m (t^2 + A)/12 actually used by
    init_group. This branch GOVERNS the time step for thick/large
    elements where kappa*G*t*A outruns the membrane stiffness — assuming
    'membrane always binds' was an M1 bug fixed in M2 (a nu=0 cantilever
    strip at /DT 0.9 diverged on the shear-rotation mode)."""
    b1, b2, A = B1[sl], B2[sl], area[sl]
    n = len(A)
    B = np.zeros((n, 5, 3 * nnode))
    B[:, 0, 2 * nnode:] = b1                    # kxx =  B1 . thy
    B[:, 1, nnode:2 * nnode] = -b2              # kyy = -B2 . thx
    B[:, 2, 2 * nnode:] = b2                    # kxy = B2.thy - B1.thx
    B[:, 2, nnode:2 * nnode] = -b1
    B[:, 3, :nnode] = b1                        # gx = B1.w + mean(thy)
    B[:, 3, 2 * nnode:] = 1.0 / nnode
    B[:, 4, :nnode] = b2                        # gy = B2.w - mean(thx)
    B[:, 4, nnode:2 * nnode] = -1.0 / nnode
    Ep = mat.E / (1.0 - mat.nu ** 2)
    Cb = np.zeros((5, 5))
    Cb[:3, :3] = t ** 3 / 12.0 * np.array(
        [[Ep, mat.nu * Ep, 0.0], [mat.nu * Ep, Ep, 0.0], [0.0, 0.0, mat.G]])
    Cb[3, 3] = Cb[4, 4] = SHEAR_FACTOR * mat.G * t
    K = A[:, None, None] * np.einsum("nai,ab,nbj->nij", B, Cb, B)
    m = rho * t * A / nnode                     # nodal mass
    inertia = m * (t ** 2 + A) / 12.0           # nodal inertia (init_group)
    minv = np.empty((n, 3 * nnode))
    minv[:, :nnode] = 1.0 / np.sqrt(m)[:, None]
    minv[:, nnode:] = np.repeat(1.0 / np.sqrt(inertia), 2 * nnode
                                ).reshape(n, 2 * nnode)
    Ksym = minv[:, :, None] * K * minv[:, None, :]
    return np.linalg.eigvalsh(Ksym)[:, -1]


def _exact_dt_factor(B1, B2, area, lc, thick, slices) -> np.ndarray:
    """Per-element ratio dt_exact/(lc/c) over BOTH stiffness branches of
    the flat one-point element:

    * membrane — the 2-D analogue of solid_hexa8._exact_dt_factor: the
      membrane stiffness is K = t*A * B^T C B with constant B, so the
      exact max frequency is the 3x3 eigenproblem
      omega^2 = (4/rho) eig(C_planestress . B B^T)  (lumped m = rho t A/4);
    * bending/transverse-shear — the 12-dof (w, thx, thy) eigenproblem of
      _bend_shear_omega2, which takes over for thick or large elements.

    dt_exact = 2 / max(omega) is a strict bound for the linearized
    element; forces() rescales it by the running lc/c."""
    n = len(area)
    Sxx = np.einsum("ni,ni->n", B1, B1)
    Syy = np.einsum("ni,ni->n", B2, B2)
    Sxy = np.einsum("ni,ni->n", B1, B2)
    BBt = np.zeros((n, 3, 3))
    BBt[:, 0, 0], BBt[:, 1, 1] = Sxx, Syy
    BBt[:, 2, 2] = Sxx + Syy
    BBt[:, 0, 2] = BBt[:, 2, 0] = Sxy
    BBt[:, 1, 2] = BBt[:, 2, 1] = Sxy
    fac = np.ones(n)
    for sl, mat, prop in slices:
        Ep = mat.E / (1.0 - mat.nu ** 2)
        C = np.array([[Ep, mat.nu * Ep, 0.0],
                      [mat.nu * Ep, Ep, 0.0],
                      [0.0, 0.0, mat.G]])
        eig = np.linalg.eigvals(C[None, :, :] @ BBt[sl])
        w2max = (4.0 / mat.rho0) * eig.real.max(axis=1)
        w2bend = _bend_shear_omega2(B1, B2, area, sl, mat,
                                    prop.params["thick"], 4, mat.rho0)
        w2max = np.maximum(w2max, w2bend)
        c = mat.sound_speed_shell()
        dt_exact = 2.0 / np.sqrt(np.maximum(w2max, EM20))
        fac[sl] = np.minimum(dt_exact / (lc[sl] / c), 1.0)
    return fac


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter cinit3/cmass3)."""
    xe = model.x0[group.conn]
    E, xl, area, B1, B2 = _local_geometry(xe)
    bad = area <= 0.0
    if np.any(bad):
        for eid in group.ids[bad]:
            log.error(f"/SHELL {eid}: zero or negative area", "SHELL INIT")

    n = group.n
    thick = np.zeros(n)
    rho0 = np.zeros(n)
    nip_max = 1
    for sl, mat, prop in group.state["slices"]:
        thick[sl] = prop.params["thick"]
        rho0[sl] = mat.rho0
        nip_max = max(nip_max, int(prop.params["nip"]))
    mass = rho0 * thick * area

    # Through-thickness Gauss stations per part slice: z_k in [-t/2, t/2],
    # weights scaled so sum(w_k) = t. Stored per slice (nip may differ).
    zw = []
    for sl, mat, prop in group.state["slices"]:
        nip = int(prop.params["nip"])
        gp, gw = np.polynomial.legendre.leggauss(nip)
        zw.append((gp * 0.5, gw * 0.5))  # relative to thickness
    group.state.update(
        sig=np.zeros((n, nip_max, 3)),   # in-plane stress per layer
        qshear=np.zeros((n, 2)),         # transverse shear stress (elastic)
        epsp=np.zeros((n, nip_max)),     # plastic strain per layer
        thick=thick,
        area0=area.copy(),
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),
        # persistent hourglass generalized forces (stiffness control):
        # columns = [membrane-x, membrane-y, transverse-w, theta-x, theta-y]
        hgq=np.zeros((n, 5)),
        zw=zw,
        # exact stability correction to the lc/c estimate (see helper)
        dtfac=_exact_dt_factor(B1, B2, area, _char_length(xl, area),
                               thick, group.state["slices"]),
    )
    _init_material_state(group, nip_max)
    node_idx = group.conn.reshape(-1)
    mass_c = np.repeat(mass / 4.0, 4)
    # generous lumped rotational inertia (see module docstring)
    inertia_c = np.repeat(mass / 4.0 * (thick ** 2 + area) / 12.0, 4)
    return node_idx, mass_c, inertia_c


# ----------------------------------------------------------------------------
# M3 material/failure plumbing shared by both shell kernels
# ----------------------------------------------------------------------------

def _init_material_state(group, nip_max):
    """Allocate the per-layer material/failure state (see the materials
    and failure package docstrings):

    * ``off``      (n,)          1 alive / 0 deleted (GBUF%OFF)
    * ``layfail``  (n, nip)      1 intact / 0 broken, per layer — written
                                 by /FAIL criteria AND by layer-breaking
                                 laws (LAW27 rupture strain)
    * ``dama``     (n, nip)      /FAIL damage per layer (when needed)
    * ``mat_extra``{name: array} law-specific state (LAW27 crack memory)
    * ``chk_fail``               precomputed 'anything can delete here'
    """
    st = group.state
    n = group.n
    st["off"] = np.ones(n)
    st["layfail"] = np.ones((n, nip_max))
    st["mat_extra"] = {}
    for sl, mat, prop in st["slices"]:
        for name, shape in materials.extra_shapes(mat, nip_max).items():
            if name not in st["mat_extra"]:
                st["mat_extra"][name] = np.zeros((n,) + shape)
    if any(mat.fail is not None for _, mat, _ in st["slices"]):
        st["dama"] = np.zeros((n, nip_max))
    st["chk_fail"] = any(
        mat.fail is not None or mat.law == 27
        or mat.params.get("eps_p_max", EP30) < 1e30
        for _, mat, _ in st["slices"])


def _layer_extra(st, sl, k):
    """The ``extra`` dict for one layer of one part slice: views into the
    law-specific arrays plus the shared layer-failure flags."""
    extra = {name: arr[sl, k] for name, arr in st["mat_extra"].items()}
    extra["layfail"] = st["layfail"][sl, k]
    return extra


def _layer_failure(st, sl, mat, k, sig_k, epsp_old, deps_k, dt):
    """/FAIL damage + eps_p_max for one layer; breaks layers in place and
    zeroes their stress so the resultant integration never sees them."""
    layf = st["layfail"][sl, k]
    if mat.fail is not None:
        d_ep = st["epsp"][sl, k] - epsp_old[sl, k]
        tstar = None                 # /FAIL/JOHNSON D5 (M6): homologous
        if "temp" in st["mat_extra"] and "mT" in mat.params:
            tstar = np.clip(
                st["mat_extra"]["temp"][sl, k]
                / (mat.params["T_melt"] - mat.params["T_i"]), 0.0, 1.0)
        broken = failure.shell_step(mat.fail, sig_k, d_ep, deps_k, dt,
                                    st["dama"][sl, k], tstar)
        layf[broken] = 0.0
    eps_max = mat.params.get("eps_p_max", EP30)
    if eps_max < 1e30:
        layf[st["epsp"][sl, k] > eps_max] = 0.0
    sig_k[layf == 0.0] = 0.0


def _element_deletion(st, nip_of):
    """Element OFF from the layer flags, per part slice.

    Deletion rule: /FAIL's Ifail_sh (1 = one broken layer kills the
    element — the Radioss default, also used for the material eps_p_max
    thresholds; 2 = all layers), while LAW27 uses the all-layers rule of
    the original brittle law. Returns the updated alive mask."""
    off = st["off"]
    layfail = st["layfail"]
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        if not (mat.fail is not None or mat.law == 27
                or mat.params.get("eps_p_max", EP30) < 1e30):
            continue
        nip = nip_of[isl]
        nbroken = (layfail[sl, :nip] == 0.0).sum(axis=1)
        if mat.law == 27 or (mat.fail is not None and mat.fail.ifail_sh == 2):
            dead = nbroken == nip
        else:
            dead = nbroken >= 1
        off[sl][dead] = 0.0
    return off > 0.0


# ----------------------------------------------------------------------------
# Engine-side forces (cforc3.F)
# ----------------------------------------------------------------------------

def _pre(xe, ve, vre, off):
    """Corotational frame + local geometry + rate kinematics — the
    ccoor3/cdefo3(+czforc3 kinematics) part of the cycle, everything
    BEFORE the layer/material loop. Returns
    (E, area, lc, B1, B2, bb, gam, V, dm, kap, gs) where

    * ``V`` (n, 4, 5) are the LOCAL nodal rates [vx, vy, vz, thx, thy] —
      reused verbatim by the hourglass block of _post;
    * ``gam`` (n, 4) is the FB-orthogonalized hourglass shape vector;
    * ``bb`` = B1.B1 + B2.B2 feeds the per-slice hourglass stiffness.

    Mirrored by accel.jit_kernels.shell_pre (the M7 parity contract)."""
    n = len(xe)
    E, xl, area, B1, B2 = _local_geometry(xe)
    area = np.maximum(area, EM20)
    lc = _char_length(xl, area)

    # velocities in the corotational frame: one matmul for the three
    # translations, one for the rotations (only thx/thy are used)
    V = np.empty((n, 4, 5))
    V[:, :, :3] = ve @ E
    V[:, :, 3:] = (vre @ E)[:, :, :2]

    # ---- rate of deformation (cdefo3 + czforc3 kinematics) ----------------
    # ALL ten B.v dot products in one stacked matmul: Bt (n,2,4) @ V
    # (n,4,5) -> M[n, which-B, which-field]; the rate lines below then
    # mirror the kinematics table of the module docstring entry by entry.
    Bt = np.empty((n, 2, 4))
    Bt[:, 0, :] = B1
    Bt[:, 1, :] = B2
    M = Bt @ V
    dm = np.empty((n, 3))            # membrane rates [xx, yy, xy(eng)]
    dm[:, 0] = M[:, 0, 0]                              # B1 . vx
    dm[:, 1] = M[:, 1, 1]                              # B2 . vy
    dm[:, 2] = M[:, 0, 1] + M[:, 1, 0]                 # B1.vy + B2.vx
    kap = np.empty((n, 3))           # curvature rates
    kap[:, 0] = M[:, 0, 4]                             # B1 . thy
    kap[:, 1] = -M[:, 1, 3]                            # -B2 . thx
    kap[:, 2] = M[:, 1, 4] - M[:, 0, 3]                # B2.thy - B1.thx
    gs = np.empty((n, 2))            # transverse shear rates
    gs[:, 0] = M[:, 0, 2] + V[:, :, 4].mean(axis=1)    # B1.vz + mean(thy)
    gs[:, 1] = M[:, 1, 2] - V[:, :, 3].mean(axis=1)    # B2.vz - mean(thx)

    # hourglass geometry (chour3): the FB shape vector gamma and the
    # B1.B1+B2.B2 stiffness factor — pure geometry, computed here so the
    # post block never re-touches xl
    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = np.empty((n, 4))
    gam[:, 0] = 1.0
    gam[:, 1] = -1.0
    gam[:, 2] = 1.0
    gam[:, 3] = -1.0
    gam -= hx[:, None] * B1
    gam -= hy[:, None] * B2
    bb = (np.einsum("ni,ni->n", B1, B1)
          + np.einsum("ni,ni->n", B2, B2))             # B1.B1 + B2.B2

    # deleted elements (GBUF%OFF = 0): freeze their state — no straining,
    # and downstream no resultants, hourglass force or time-step claim
    alive = off > 0.0
    if not alive.all():
        dm[~alive] = 0.0
        kap[~alive] = 0.0
        gs[~alive] = 0.0
    return E, area, lc, B1, B2, bb, gam, V, dm, kap, gs


def _post(E, area, B1, B2, gam, V, Nres, Mres, qres, Q,
          k_m, k_w, k_r, dt):
    """Resultants -> nodal forces/moments, BLT84 stiffness hourglass and
    the back-transform to global axes — the czforc3/chour3 part of the
    cycle, everything AFTER the layer loop. ``Q`` is the persistent
    hourglass state st["hgq"], updated IN PLACE; k_m/k_w/k_r arrive
    pre-masked by ``alive``. Returns (fg, mg, dehg): global nodal
    forces/moments (n,4,3) ready to scatter, and the stored hourglass
    energy increment. Mirrored by accel.jit_kernels.shell_post."""
    n = len(area)

    # ---- internal nodal forces & moments (transpose of the rates) ---------
    # each line mirrors one line of the rate kinematics in _pre.
    f = np.empty((n, 4, 3))
    m = np.zeros((n, 4, 3))
    A_ = area[:, None]
    f[:, :, 0] = A_ * (B1 * Nres[:, 0:1] + B2 * Nres[:, 2:3])
    f[:, :, 1] = A_ * (B2 * Nres[:, 1:2] + B1 * Nres[:, 2:3])
    f[:, :, 2] = A_ * (B1 * qres[:, 0:1] + B2 * qres[:, 1:2])
    m[:, :, 0] = A_ * (-B2 * Mres[:, 1:2] - B1 * Mres[:, 2:3]
                       - 0.25 * qres[:, 1:2])
    m[:, :, 1] = A_ * (B1 * Mres[:, 0:1] + B2 * Mres[:, 2:3]
                       + 0.25 * qres[:, 0:1])

    # ---- hourglass control (chour3, BLT84 stiffness type — module doc) ----
    # all five modes at once: modal velocities qd = gamma . (local rates),
    # stiffness per mode [k_m, k_m, k_w, k_r, k_r]; each mode integrates
    # Q += k*qd*dt and pushes back f = -Q*gamma (translations x/y/w from
    # columns 0-2, rotations thx/thy from columns 3-4).
    qd = np.einsum("ni,nik->nk", gam, V)               # (n, 5)
    kvec = np.empty((n, 5))
    kvec[:, 0] = k_m
    kvec[:, 1] = k_m
    kvec[:, 2] = k_w
    kvec[:, 3] = k_r
    kvec[:, 4] = k_r
    q_old = Q.copy()
    Q += kvec * qd * dt
    # stored hourglass energy increment: midpoint force x modal rate
    dehg = 0.5 * ((q_old + Q) * qd).sum(axis=1) * dt

    # total local force = -(internal) + hourglass, back to global frame
    fl = -f
    fl -= gam[:, :, None] * Q[:, None, :3]
    ml = -m
    ml[:, :, 0] -= gam * Q[:, 3:4]
    ml[:, :, 1] -= gam * Q[:, 4:5]
    # back to global axes: fg[n,i,b] = sum_a fl[n,i,a] E[n,b,a]
    Et = E.transpose(0, 2, 1)
    fg = fl @ Et
    mg = ml @ Et
    return fg, mg, dehg


def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    n = group.n
    xe = x[conn]
    thick = st["thick"]

    # ---- pre block: frame, geometry, rates (numba mirror when active) -----
    jit = accel_get("shell_pre")
    if jit is not None:
        E, area, lc, B1, B2, bb, gam, V, dm, kap, gs = jit(
            xe, v[conn], vr[conn], st["off"])
    else:
        E, area, lc, B1, B2, bb, gam, V, dm, kap, gs = _pre(
            xe, v[conn], vr[conn], st["off"])
    alive = st["off"] > 0.0

    # ---- layer stress updates + resultants ---------------------------------
    sig = st["sig"]
    epsp_old = st["epsp"].copy() if st["chk_fail"] else None
    Nres = np.zeros((n, 3))     # membrane force / length
    Mres = np.zeros((n, 3))     # moment / length
    de_layers = np.zeros(n)     # internal energy density accumulation
    c = np.zeros(n)
    nip_of = []
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        zrel, wrel = st["zw"][isl]
        nip_of.append(len(zrel))
        t_sl = thick[sl]
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl                     # layer position
            wk = wrel[k] * t_sl                     # layer weight (sums to t)
            deps = (dm[sl] + zk[:, None] * kap[sl]) * dt
            s_old = sig[sl, k, :].copy()
            s_new, _ = materials.shell_update(
                mat, sig[sl, k, :], deps, st["epsp"][sl, k], dt,
                _layer_extra(st, sl, k))
            if st["chk_fail"]:
                # /FAIL damage + eps_p_max: break layers, zero their stress
                # BEFORE they enter the resultants
                _layer_failure(st, sl, mat, k, s_new, epsp_old, deps, dt)
            sig[sl, k, :] = s_new
            s_mid = 0.5 * (s_old + s_new)
            Nres[sl] += wk[:, None] * s_new
            Mres[sl] += (wk * zk)[:, None] * s_new
            de_layers[sl] += wk * np.einsum("nk,nk->n", s_mid, deps)
        c[sl] = mat.sound_speed_shell()
        # elastic transverse shear resultant stress (with 5/6 factor)
        qold = st["qshear"][sl].copy()
        st["qshear"][sl] += SHEAR_FACTOR * mat.G * gs[sl] * dt
        de_layers[sl] += t_sl * np.einsum(
            "nk,nk->n", 0.5 * (qold + st["qshear"][sl]), gs[sl] * dt)

    # ---- element deletion from the layer flags -----------------------------
    if st["chk_fail"]:
        alive = _element_deletion(st, nip_of)
        if not alive.all():
            dead = ~alive
            # a deleted element carries nothing: wipe this cycle's
            # resultants and every bit of persistent stress state
            Nres[dead] = 0.0
            Mres[dead] = 0.0
            sig[dead] = 0.0
            st["qshear"][dead] = 0.0
            st["hgq"][dead] = 0.0
    qres = st["qshear"] * thick[:, None]            # shear force / length

    # per-mode hourglass stiffness, scaled from the element's physical
    # membrane / transverse-shear / bending stiffness (BLT84 constants);
    # deleted elements exert no hourglass force (their Q was wiped above)
    k_m = np.zeros(n)
    k_w = np.zeros(n)
    k_r = np.zeros(n)
    for sl, mat, prop in st["slices"]:
        p = prop.params
        t_sl = thick[sl]
        k_m[sl] = p["hm"] * mat.E * t_sl * area[sl] * bb[sl] / 8.0
        k_w[sl] = p["hf"] * SHEAR_FACTOR * mat.G * t_sl * area[sl] * bb[sl] / 8.0
        k_r[sl] = p["hr"] * mat.E * t_sl ** 3 * area[sl] * bb[sl] / 192.0
    k_m *= alive
    k_w *= alive
    k_r *= alive

    # ---- post block: forces, hourglass, back-transform ---------------------
    jit = accel_get("shell_post")
    if jit is not None:
        fg, mg, dehg = jit(E, area, B1, B2, gam, V, Nres, Mres, qres,
                           st["hgq"], k_m, k_w, k_r, dt)
    else:
        fg, mg, dehg = _post(E, area, B1, B2, gam, V, Nres, Mres, qres,
                             st["hgq"], k_m, k_w, k_r, dt)

    st["ehour"] += dehg
    st["eint"] += area * de_layers

    # ---- scatter to global arrays (asspar) ---------------------------------
    flat = conn.reshape(-1)
    scatter_add3(fint, flat, fg.reshape(-1, 3))
    scatter_add3(mint, flat, mg.reshape(-1, 3))

    # ---- critical time step ------------------------------------------------
    # deleted elements no longer constrain the global step
    return np.where(alive, st["dtfac"] * lc / c, EP30)


# ----------------------------------------------------------------------------
# Implicit tangent stiffness (M8) — a NEW entry point alongside forces()
# ----------------------------------------------------------------------------
# The Belytschko-Tsay tangent is assembled in the corotational LOCAL frame and
# rotated to global, exactly like the force path. Per node the element carries
# five LOCAL dofs — three translations (vx, vy, vz) and two bending rotations
# (thx, thy); there is no local drilling (thz) stiffness (the classic BT
# feature). The local element stiffness is the sum of four consistent blocks,
# each the exact linearization of the matching rate operator in _pre/_post:
#
#   membrane   K_m = A t     B_m^T C B_m         (N   = t   C  dm)
#   bending    K_b = A t^3/12 B_b^T C B_b        (M   = t^3/12 C kappa)
#   shear      K_s = A kG t   B_s^T B_s          (q   = kG t gamma_s)
#   hourglass  K_h = sum_modes k_mode g (x) g    (BLT84 stiffness control)
#
# with C the plane-stress membrane tangent (materials.shell_membrane_tangent,
# LAW1 in M8). The local dofs are then mapped to the six global dofs per node
# (three translations + three rotations) by the corotational frame E, and a
# small DRILLING stiffness about the local normal e3 is added so the global
# rotation block is non-singular (a flat shell gives no stiffness to rotation
# about its normal — the standard shell drilling-DOF penalty; the residual
# never loads it, so it stays zero and does not affect the solution).
#
# LAW2 shell (through-thickness elastoplastic layers) tangent is DEFERRED —
# M8 ports the LAW1 elastic shell tangent (see PORTING_GUIDE). The geometric /
# initial-stress stiffness is the M9 addition — ``kgeo()`` below, added to
# this tangent by the assembler when /IMPL/NONLIN is active.

#: drilling-stiffness fraction of the bending stiffness (conditioning only —
#: the drilling DOF carries no load on the M8 validations, so the exact value
#: does not change results; small enough not to pollute a curved-shell answer).
_DRILL_COEF = 1.0e-3


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole shell group (LAW1 elastic).

    Returns ``(ke, edofs)``:

    * ``ke``    (n, 24, 24) dense element tangents over the 4 nodes x 6 global
      dofs (translations + rotations);
    * ``edofs`` (n, 24) global scalar DOF slot ids (node*6 + component) in the
      ``implicit.dofmap`` numbering — 0,1,2 = ux,uy,uz ; 3,4,5 = rx,ry,rz.

    ``epsp_incr`` is accepted for signature parity with the solid tangent and
    ignored (elastic shell)."""
    st = group.state
    conn = group.conn
    n = group.n
    xe = x[conn]

    E, xl, area, B1, B2 = _local_geometry(xe)
    area = np.maximum(area, EM20)
    thick = st["thick"]

    # ---- generalized strain-displacement operators in the LOCAL frame -----
    # local dof layout is COMPONENT-grouped: d = [vx(4), vy(4), vz(4),
    # thx(4), thy(4)] (20 dofs). Each B maps d -> the generalized strains,
    # written line-for-line against the rate kinematics of _pre.
    z = np.zeros((n, 4))
    Bm = np.zeros((n, 3, 20))            # membrane [dm_xx, dm_yy, dm_xy]
    Bm[:, 0, 0:4] = B1                                   # dm_xx = B1.vx
    Bm[:, 1, 4:8] = B2                                   # dm_yy = B2.vy
    Bm[:, 2, 0:4] = B2                                   # dm_xy = B2.vx +
    Bm[:, 2, 4:8] = B1                                   #         B1.vy
    Bb = np.zeros((n, 3, 20))            # curvature [k_xx, k_yy, k_xy]
    Bb[:, 0, 16:20] = B1                                 # k_xx = B1.thy
    Bb[:, 1, 12:16] = -B2                                # k_yy = -B2.thx
    Bb[:, 2, 12:16] = -B1                                # k_xy = B2.thy -
    Bb[:, 2, 16:20] = B2                                 #        B1.thx
    Bs = np.zeros((n, 2, 20))            # shear [g_x, g_y]
    Bs[:, 0, 8:12] = B1                                  # g_x = B1.vz +
    Bs[:, 0, 16:20] = 0.25                               #       mean(thy)
    Bs[:, 1, 8:12] = B2                                  # g_y = B2.vz -
    Bs[:, 1, 12:16] = -0.25                              #       mean(thx)

    # ---- constitutive blocks (membrane + bending + shear) -----------------
    Kl = np.zeros((n, 20, 20))
    kdrill = np.zeros(n)
    for sl, mat, prop in st["slices"]:
        C = materials.shell_membrane_tangent(mat)        # (3, 3) plane stress
        t_sl = thick[sl]
        A_sl = area[sl]
        kGt = SHEAR_FACTOR * mat.G * t_sl                # transverse shear
        Bms, Bbs, Bss = Bm[sl], Bb[sl], Bs[sl]
        # membrane: A t B_m^T C B_m
        Kl[sl] += (A_sl * t_sl)[:, None, None] * np.einsum(
            "nai,ab,nbj->nij", Bms, C, Bms)
        # bending: A t^3/12 B_b^T C B_b
        Kl[sl] += (A_sl * t_sl ** 3 / 12.0)[:, None, None] * np.einsum(
            "nai,ab,nbj->nij", Bbs, C, Bbs)
        # shear: A kGt B_s^T B_s
        Kl[sl] += (A_sl * kGt)[:, None, None] * np.einsum(
            "nai,naj->nij", Bss, Bss)
        # drilling penalty scale (bending stiffness order, see module note)
        kdrill[sl] = _DRILL_COEF * mat.E * t_sl ** 3 * A_sl / 12.0

    # ---- hourglass stiffness (BLT84 stiffness control, same as _post) -----
    # gamma: FB-orthogonalized shape vector (identical to _pre's construction)
    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = np.empty((n, 4))
    gam[:, 0] = 1.0
    gam[:, 1] = -1.0
    gam[:, 2] = 1.0
    gam[:, 3] = -1.0
    gam -= hx[:, None] * B1
    gam -= hy[:, None] * B2
    GG = np.einsum("ni,nj->nij", gam, gam)               # (n, 4, 4) Gram
    bb = (np.einsum("ni,ni->n", B1, B1)
          + np.einsum("ni,ni->n", B2, B2))
    # per-mode stiffness (identical formulas to forces()), fields ordered
    # [vx, vy, vz, thx, thy] -> [k_m, k_m, k_w, k_r, k_r]
    kfield = np.zeros((n, 5))
    for sl, mat, prop in st["slices"]:
        p = prop.params
        t_sl = thick[sl]
        A_sl = area[sl]
        k_m = p["hm"] * mat.E * t_sl * A_sl * bb[sl] / 8.0
        k_w = p["hf"] * SHEAR_FACTOR * mat.G * t_sl * A_sl * bb[sl] / 8.0
        k_r = p["hr"] * mat.E * t_sl ** 3 * A_sl * bb[sl] / 192.0
        kfield[sl, 0] = k_m
        kfield[sl, 1] = k_m
        kfield[sl, 2] = k_w
        kfield[sl, 3] = k_r
        kfield[sl, 4] = k_r
    # scatter k_field * GG onto each field's 4x4 block of the local matrix
    fi = np.arange(4)
    for f in range(5):
        rows = (f * 4 + fi)[:, None]
        cols = (f * 4 + fi)[None, :]
        Kl[:, rows, cols] += kfield[:, f, None, None] * GG

    # ---- local (20) -> global (24) transformation via the frame E ----------
    # local layout [vx(4),vy(4),vz(4),thx(4),thy(4)]; global node-major
    # [ux,uy,uz,rx,ry,rz] per node. local translation = E^T global_trans,
    # local (thx,thy) = (e1,e2) . global_rot.
    e1, e2, e3 = E[:, :, 0], E[:, :, 1], E[:, :, 2]      # (n, 3) each
    Tg = np.zeros((n, 20, 24))
    for i in range(4):
        for c in range(3):
            Tg[:, 0 * 4 + i, i * 6 + c] = e1[:, c]       # vx = e1.trans
            Tg[:, 1 * 4 + i, i * 6 + c] = e2[:, c]       # vy = e2.trans
            Tg[:, 2 * 4 + i, i * 6 + c] = e3[:, c]       # vz = e3.trans
            Tg[:, 3 * 4 + i, i * 6 + 3 + c] = e1[:, c]   # thx = e1.rot
            Tg[:, 4 * 4 + i, i * 6 + 3 + c] = e2[:, c]   # thy = e2.rot
    ke = np.einsum("nki,nkl,nlj->nij", Tg, Kl, Tg)       # (n, 24, 24)

    # ---- drilling penalty about the local normal e3 (global rot block) -----
    # add k_d (e3 (x) e3) to each node's rotation block so the drilling DOF
    # (rotation about the shell normal, unstiffened by BT) has a small,
    # non-singular stiffness. e3 (x) e3 restricts exactly that rotation.
    e3e3 = np.einsum("ni,nj->nij", e3, e3)               # (n, 3, 3)
    for i in range(4):
        r = i * 6 + 3
        ke[:, r:r + 3, r:r + 3] += kdrill[:, None, None] * e3e3

    # ---- global DOF addressing --------------------------------------------
    edofs = np.empty((n, 24), dtype=np.int64)
    for i in range(4):
        for c in range(6):
            edofs[:, i * 6 + c] = conn[:, i] * 6 + c
    return ke, edofs


# ----------------------------------------------------------------------------
# Geometric (initial-stress) stiffness K_geo (M9)
# ----------------------------------------------------------------------------
# Fortran origin: the geometric-stiffness branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F`` + the shell KGEO routines — the
# ``imp_kgeo`` path of OpenRadioss's /IMPL/NONLIN large-displacement branch).
#
# Theory. For a shell the dominant initial-stress term carries the MEMBRANE
# force resultants N (force/length): linearizing the internal virtual work of
# the current membrane state against a transverse (or in-plane) perturbation
# of the midsurface gives, per element and in the local frame,
#
#     K_geo[a i, b j] = δ_ij A ( B1_a B1_b N_xx + B2_a B2_b N_yy
#                                + (B1_a B2_b + B2_a B1_b) N_xy )
#
# on the TRANSLATIONS — the classic von-Kármán initial-stress matrix built
# from the same one-point gradient operators B1, B2 the force path uses.
# It is δ_ij (isotropic over the translation components), so rotating the
# local block to global axes leaves it unchanged: it can be added directly to
# the global translation blocks (frame invariance of c_ab * I3). Compressive
# N makes it negative — the plate/column buckling driver; tensile N stiffens
# (stress stiffening / membranes). At zero stress it vanishes: the M8 path is
# untouched. The higher-order geometric couplings through the ROTATIONAL dofs
# (moment-resultant terms) are O(t/L) of the membrane term and are omitted —
# the standard BT shell buckling practice (they matter only for problems
# dominated by pre-stress moments, out of M9 scope, see PORTING_GUIDE).

def _membrane_resultants(st, thick):
    """Membrane force resultants N = sum_k w_k sigma_k (force/length, local
    [xx, yy, xy]) from the stored layer stresses — the same quadrature the
    force path applies (deleted elements have wiped stress => N = 0)."""
    sig = st["sig"]
    Nres = np.zeros((len(thick), 3))
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        for k in range(len(zrel)):
            wk = wrel[k] * t_sl                 # layer weight (sums to t)
            Nres[sl] += wk[:, None] * sig[sl, k, :]
    return Nres


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness for the shell group,
    from the current layer stresses at geometry ``x``. Returns
    ``(ke, edofs)`` shaped exactly like ``tangent()`` (n, 24, 24)."""
    st = group.state
    conn = group.conn
    n = group.n
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)
    Nres = _membrane_resultants(st, st["thick"])

    # g_ab = A (B1a B1b Nxx + B2a B2b Nyy + (B1a B2b + B2a B1b) Nxy)  (n,4,4)
    g = area[:, None, None] * (
        Nres[:, 0, None, None] * B1[:, :, None] * B1[:, None, :]
        + Nres[:, 1, None, None] * B2[:, :, None] * B2[:, None, :]
        + Nres[:, 2, None, None] * (B1[:, :, None] * B2[:, None, :]
                                    + B2[:, :, None] * B1[:, None, :]))
    ke = np.zeros((n, 24, 24))
    ni = 6 * np.arange(4)
    for c in range(3):                     # delta_ij over the translations
        rows = (ni + c)[:, None]
        cols = (ni + c)[None, :]
        ke[:, rows, cols] += g

    edofs = np.empty((n, 24), dtype=np.int64)
    for i in range(4):
        for c in range(6):
            edofs[:, i * 6 + c] = conn[:, i] * 6 + c
    return ke, edofs


def static_internal_forces(group, x, u, ur, fint, mint):
    """Internal nodal forces/moments at configuration ``x`` from the CURRENT
    resultant state — the updated-Lagrangian force assembly of the M9
    implicit residual (the stress/hourglass state was just updated by a
    ``forces()`` call at the MIDPOINT geometry; this re-states the czforc3
    force expressions on the END geometry, where equilibrium holds).

    Implementation: rebuild N/M from the stored layer stresses (the same
    quadrature the force path uses), q from the stored shear state, then
    call the existing ``_post`` with dt = 0 — at dt = 0 the hourglass state
    Q is NOT advanced (it already holds the trial values from the midpoint
    call) and _post reduces to exactly the resultant->nodal-force transpose
    plus the -gamma*Q hourglass push-back, all on this geometry."""
    st = group.state
    conn = group.conn
    n = group.n
    thick = st["thick"]
    E, xl, area, B1, B2 = _local_geometry(x[conn])
    area = np.maximum(area, EM20)

    sig = st["sig"]
    Nres = np.zeros((n, 3))
    Mres = np.zeros((n, 3))
    for isl, (sl, mat, prop) in enumerate(st["slices"]):
        zrel, wrel = st["zw"][isl]
        t_sl = thick[sl]
        for k in range(len(zrel)):
            zk = zrel[k] * t_sl
            wk = wrel[k] * t_sl
            Nres[sl] += wk[:, None] * sig[sl, k, :]
            Mres[sl] += (wk * zk)[:, None] * sig[sl, k, :]
    qres = st["qshear"] * thick[:, None]

    # hourglass shape vector at THIS geometry (same construction as _pre)
    hx = xl[:, 0, 0] - xl[:, 1, 0] + xl[:, 2, 0] - xl[:, 3, 0]
    hy = xl[:, 0, 1] - xl[:, 1, 1] + xl[:, 2, 1] - xl[:, 3, 1]
    gam = np.empty((n, 4))
    gam[:, 0], gam[:, 1], gam[:, 2], gam[:, 3] = 1.0, -1.0, 1.0, -1.0
    gam -= hx[:, None] * B1
    gam -= hy[:, None] * B2

    zeros_n = np.zeros(n)
    fg, mg, _ = _post(E, area, B1, B2, gam, np.zeros((n, 4, 5)),
                      Nres, Mres, qres, st["hgq"],
                      zeros_n, zeros_n, zeros_n, 0.0)
    flat = conn.reshape(-1)
    scatter_add3(fint, flat, fg.reshape(-1, 3))
    scatter_add3(mint, flat, mg.reshape(-1, 3))
