"""
2-node Timoshenko beam element (/BEAM + /PROP/TYPE3), corotational.

Fortran origin: ``engine/source/elements/beam/`` — the cycle path is

    pforc3.F   driver (gather, frame, call chain, scatter)
    pcoor3.F   corotational frame from N1, N2 and the orientation node N3
    pdefo3.F   generalized strain rates (axial, 2 x shear, torsion,
               2 x curvature) in the local frame
    pmat3.F / pmain.F   resultant constitutive update
    pfint3.F   internal nodal forces & moments (transpose of the rates)
    pdlen3.F   critical time step

Connectivity: /BEAM gives THREE nodes — N1 and N2 carry the element,
N3 only orients the local frame (classic Radioss/NASTRAN convention):
the local y axis lies in the plane (N1, N2, N3). N3 receives no mass and
no force. If N3 is a standalone node it never moves; the frame then stays
tied to the initial (N1,N2,N3) plane, which is exact as long as the beam
does not spin bodily about its own axis (the torsion strain uses relative
rotation velocities and is frame-independent either way).

Theory (Timoshenko beam, one integration point at the element mid-span —
Przemieniecki ch. 5 / BLM ch. 9 for the corotational rate treatment):

* **Corotational frame**: e1 = (x2-x1)/L (axis), e2 = in-plane-of-N3
  normal to e1, e3 = e1 x e2. All rates are measured in this frame; the
  six resultants (N, Qy, Qz, Mx, My, Mz) live in it, so a rigid rotation
  of the element transports them exactly (no objective-rate error).

* **Generalized strain rates** — with local translational velocities
  v_i = E^T v(node i) and rotation rates th_i = E^T vr(node i):

      axial       eps_dot = (v2x - v1x) / L
      shear y     gy_dot  = (v2y - v1y)/L - (th1z + th2z)/2
      shear z     gz_dot  = (v2z - v1z)/L + (th1y + th2y)/2
      torsion     kx_dot  = (th2x - th1x) / L
      bending     ky_dot  = (th2y - th1y) / L
      bending     kz_dot  = (th2z - th1z) / L

  Under a rigid rotation with rate w about e3, (v2y-v1y)/L = w and
  th1z = th2z = w, so gy_dot = 0 — the shear (and every other rate) is
  exactly zero for rigid motion, checked in the unit tests.

* **Resultants** (rate form; elastic prediction for every law):

      N  += E A  eps_dot dt        Qy += G A gy_dot dt
      Mx += G Ixx kx_dot dt        My += E Iyy ky_dot dt   (etc.)

  Shear uses the full section area (the Radioss TYPE3 default; a shear
  factor is an optional refinement). NOTE the one-point (reduced)
  integration is precisely what avoids shear locking in this element.

* **Global plasticity** (LAW2, ported in M3 — Fortran
  ``engine/source/elements/beam/pmat3.F``, the TYPE3 beam's *global*
  model: no fiber integration over the section, a single yield criterion
  on the RESULTANTS): an equivalent section stress is built from the
  extreme-fiber estimates

      sigma_n  = |N|/A + |My|/Wy + |Mz|/Wz          (normal, worst fiber)
      tau      = |Mx|/Wx + sqrt(Qy^2 + Qz^2)/A      (shear, worst fiber)
      sigma_eq = sqrt(sigma_n^2 + 3 tau^2)          (von Mises flavour)

  with the ELASTIC section moduli approximated by the rectangle-exact map
  W = sqrt(I*A/3) (for a b x h rectangle sqrt(I*A/3) = b h^2/6 = W
  exactly; only A and I are known from /PROP/TYPE3, so the section shape
  must be estimated — the original's global model makes the same kind of
  approximation). When sigma_eq exceeds the Johnson-Cook yield stress the
  return solves the 1-D consistency  sigma_eq - E*dl = sigma_y(eps_p+dl)
  and scales ALL SIX resultants by sigma_y/sigma_eq (radial return in
  resultant space). Consequences, documented and unit-tested:
  a beam in pure bending yields at exactly M = W*sigma_y — the global
  model cannot represent the elastic-core spread to the plastic-hinge
  moment 1.5*W*sigma_y (that needs the fiber-integrated TYPE18 beam, a
  roadmap item); the Johnson-Cook strain-RATE term is ignored for beams
  (a global rate measure is ambiguous — the Starter warns when c > 0).

* **Internal nodal forces** from the virtual-power transpose of the rate
  operators, P = L (N eps_dot + Qy gy_dot + Qz gz_dot + Mx kx_dot +
  My ky_dot + Mz kz_dot):

      f1 = -(N, Qy, Qz)                    f2 = +(N, Qy, Qz)
      m1 = (-Mx, -My + Qz L/2, -Mz - Qy L/2)
      m2 = (+Mx, +My + Qz L/2, +Mz - Qy L/2)

  (the Qy/Qz halves on BOTH nodes' moments mirror the (th1+th2)/2 in the
  shear rates; total force and moment balance exactly — unit tested).

* **Lumped inertia** (pmass3-flavoured): m_i = rho A L / 2 and

      I_i = m_i * ( L^2/12 + (Iyy + Izz)/A )

  The L^2/12 term is Key's deliberate boost (as in the shells): with the
  *physical* rotary inertia rho Ip L/2 alone, the shear/rotation mode
  frequency would scale as c_s/r_gyration — for a slender beam that is
  orders of magnitude above the axial 2c/L and would annihilate the time
  step. The boost drags it back to O(c/L). Cost: an artificial rotary
  inertia ~ m L^2/12 per node, i.e. a Timoshenko rotary-inertia error
  O((l_elem/wavelength)^2) that vanishes with mesh refinement — the
  standard explicit-code trade (Radioss and LS-DYNA beams do the same).

* **Critical time step**: the local 12x12 stiffness K = L B^T C B (B the
  6x12 rate operator above, C = diag(EA, GA, GA, GIxx, EIyy, EIzz)) and
  the lumped mass matrix M are both exactly known, so the Starter gets
  the EXACT element limit dt = 2/omega_max from the generalized
  eigenproblem (via the symmetric M^-1/2 K M^-1/2) — the same "the naive
  L/c estimate is not a bound" lesson as the solids/shells, applied by
  construction. In the cycle the stored dt0 is rescaled with the current
  length: stiffness terms scale as 1/L (axial EA/L, shear GA/L) or L
  (the GA L/4 shear-rotation coupling), so

      dt(L) = dt0 * min( L/L0, sqrt(L0/L) )

  is a strict bound on both sides (shrinking: frequencies grow at most
  as 1/sqrt(L/L0), and L/L0 < sqrt(L/L0) is safer still; growing: the
  coupling term grows as L so dt shrinks as 1/sqrt(L/L0)).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20


# ----------------------------------------------------------------------------
# geometry: corotational frame (pcoor3.F)
# ----------------------------------------------------------------------------

def _frame(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
    """Local triad from the two end nodes and the orientation node.

    Returns (E (n,3,3) with columns e1|e2|e3, L (n,))."""
    d = x2 - x1
    L = np.maximum(np.linalg.norm(d, axis=1), EM20)
    e1 = d / L[:, None]
    yref = x3 - x1                                # local y lies in (e1, yref)
    e2 = yref - np.einsum("nb,nb->n", yref, e1)[:, None] * e1
    e2 /= np.maximum(np.linalg.norm(e2, axis=1), EM20)[:, None]
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=2), L


# ----------------------------------------------------------------------------
# Starter-side initialization
# ----------------------------------------------------------------------------

def _b_operator(L: float) -> np.ndarray:
    """The 6x12 generalized-strain-rate operator of the module docstring,
    local dof order (v1x v1y v1z th1x th1y th1z v2x ... th2z)."""
    B = np.zeros((6, 12))
    B[0, 0], B[0, 6] = -1 / L, 1 / L                       # eps
    B[1, 1], B[1, 7] = -1 / L, 1 / L                       # gy
    B[1, 5], B[1, 11] = -0.5, -0.5
    B[2, 2], B[2, 8] = -1 / L, 1 / L                       # gz
    B[2, 4], B[2, 10] = 0.5, 0.5
    B[3, 3], B[3, 9] = -1 / L, 1 / L                       # kx (twist)
    B[4, 4], B[4, 10] = -1 / L, 1 / L                      # ky
    B[5, 5], B[5, 11] = -1 / L, 1 / L                      # kz
    return B


def _exact_dt(L0, mass, inertia_c, slices) -> np.ndarray:
    """Exact per-element stability limit dt = 2/omega_max from the local
    12x12 eigenproblem (see module docstring). Vectorized per part slice."""
    n = len(L0)
    dt0 = np.zeros(n)
    for sl, mat, prop in slices:
        p = prop.params
        C = np.diag([mat.E * p["area"], mat.G * p["area"],
                     mat.G * p["area"], mat.G * p["ixx"],
                     mat.E * p["iyy"], mat.E * p["izz"]])
        Ls = L0[sl]
        ms = mass[sl] / 2.0                       # nodal mass
        Is = inertia_c[sl]                        # nodal inertia
        K = np.empty((len(Ls), 12, 12))
        for k, L in enumerate(Ls):                # small setup loop: init only
            B = _b_operator(L)
            K[k] = L * B.T @ C @ B
        # symmetric similarity: eig(M^-1 K) = eig(M^-1/2 K M^-1/2)
        minv_sqrt = np.zeros((len(Ls), 12))
        for d in range(3):
            minv_sqrt[:, d] = minv_sqrt[:, 6 + d] = 1.0 / np.sqrt(ms)
            minv_sqrt[:, 3 + d] = minv_sqrt[:, 9 + d] = 1.0 / np.sqrt(Is)
        Ksym = minv_sqrt[:, :, None] * K * minv_sqrt[:, None, :]
        w2 = np.linalg.eigvalsh(Ksym)[:, -1]      # largest eigenvalue
        dt0[sl] = 2.0 / np.sqrt(np.maximum(w2, EM20))
    return dt0


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter pinit3/pmass3).
    Only nodes N1, N2 receive mass — N3 is orientation only."""
    conn = group.conn
    x1, x2, x3 = model.x0[conn[:, 0]], model.x0[conn[:, 1]], model.x0[conn[:, 2]]
    E, L0 = _frame(x1, x2, x3)
    dx = np.linalg.norm(x2 - x1, axis=1)
    if np.any(dx <= 0):
        for eid in group.ids[dx <= 0]:
            log.error(f"/BEAM {eid}: zero length", "BEAM INIT")
    # a degenerate frame means N3 is on the beam axis (e2 undefined)
    ondeg = np.linalg.norm(np.cross(x2 - x1, x3 - x1), axis=1) <= \
        EM20 * np.maximum(dx, EM20)
    if np.any(ondeg):
        for eid in group.ids[ondeg]:
            log.error(f"/BEAM {eid}: orientation node N3 lies on the beam "
                      f"axis", "BEAM INIT")

    n = group.n
    area = np.zeros(n)
    rho0 = np.zeros(n)
    igyr = np.zeros(n)                            # (Iyy+Izz)/A gyration^2
    wy = np.zeros(n)                              # elastic section moduli
    wz = np.zeros(n)                              # (rectangle-exact map
    wx = np.zeros(n)                              #  W = sqrt(I*A/3), doc)
    plastic = False
    for sl, mat, prop in group.state["slices"]:
        if mat.law not in (1, 2):
            log.error(f"/BEAM: material LAW{mat.law} not ported for beams "
                      f"(LAW1 elastic, LAW2 global plasticity)", "BEAM INIT")
        if mat.law == 2:
            plastic = True
            if mat.params.get("c", 0.0) > 0.0:
                log.warning("/BEAM: the Johnson-Cook strain-rate term is "
                            "ignored by the global beam plasticity model",
                            "BEAM INIT")
        p = prop.params
        area[sl] = p["area"]
        rho0[sl] = mat.rho0
        igyr[sl] = (p["iyy"] + p["izz"]) / max(p["area"], EM20)
        wy[sl] = np.sqrt(p["iyy"] * p["area"] / 3.0)
        wz[sl] = np.sqrt(p["izz"] * p["area"] / 3.0)
        wx[sl] = np.sqrt(p["ixx"] * p["area"] / 3.0)
    mass = rho0 * area * L0
    # nodal inertia: Key's boosted lumping (module docstring)
    inertia_c = mass / 2.0 * (L0 ** 2 / 12.0 + igyr)

    group.state.update(
        # the six force/moment resultants, local frame (GBUF%FOR/%MOM)
        fres=np.zeros((n, 3)),        # N, Qy, Qz
        mres=np.zeros((n, 3)),        # Mx, My, Mz
        L0=L0,
        mass=mass,
        eint=np.zeros(n),
        ehour=np.zeros(n),            # always zero: no hourglass modes
        dt0=_exact_dt(L0, mass, inertia_c, group.state["slices"]),
        # mass-carrying connectivity (N1, N2 only) for output/energy code
        mass_conn=conn[:, :2].copy(),
    )
    if plastic:
        group.state.update(
            epsp=np.zeros(n),         # global equivalent plastic strain
            wy=wy, wz=wz, wx=wx,      # section moduli (module docstring)
        )
    node_idx = conn[:, :2].reshape(-1)
    mass_c = np.repeat(mass / 2.0, 2)
    inertia_r = np.repeat(inertia_c, 2)
    return node_idx, mass_c, inertia_r


# ----------------------------------------------------------------------------
# Global plasticity (LAW2, pmat3.F flavour — see module docstring)
# ----------------------------------------------------------------------------

_NEWTON_ITERS = 5


def _global_plastic_return(st, sl, mat, p):
    """Radial return of the six resultants onto the Johnson-Cook yield
    stress (rate term ignored — Starter warns). In-place on fres/mres and
    the global plastic strain epsp."""
    mp = mat.params
    A = p["area"]
    fres = st["fres"]
    mres = st["mres"]
    N, Qy, Qz = fres[sl, 0], fres[sl, 1], fres[sl, 2]
    Mx, My, Mz = mres[sl, 0], mres[sl, 1], mres[sl, 2]

    # equivalent extreme-fiber stress (normal + shear, von Mises flavour)
    sn = np.abs(N) / A + np.abs(My) / st["wy"][sl] + np.abs(Mz) / st["wz"][sl]
    tau = np.abs(Mx) / st["wx"][sl] + np.sqrt(Qy ** 2 + Qz ** 2) / A
    seq = np.sqrt(sn ** 2 + 3.0 * tau ** 2) + 1e-30

    def sy_h(ep):
        """Johnson-Cook static curve and slope (LAW2 without rate term)."""
        e = np.maximum(ep, 1e-20)
        sy = mp["A"] + mp["B"] * e ** mp["n"]
        H = mp["B"] * mp["n"] * e ** (mp["n"] - 1.0)
        capped = sy > mp["sig_max"]
        return np.where(capped, mp["sig_max"], sy), np.where(capped, 0.0, H)

    epsp = st["epsp"]
    sy, _ = sy_h(epsp[sl])
    plastic = seq > sy
    if not np.any(plastic):
        return
    idx = np.where(plastic)[0]
    gidx = np.arange(sl.start, sl.stop)[idx]
    dl = np.zeros(len(idx))
    seq_p = seq[idx]
    ep0 = epsp[gidx]
    # 1-D consistency: seq - E*dl = sigma_y(ep0 + dl) — Newton, exactly
    # the truss return with E as the effective section modulus
    for _ in range(_NEWTON_ITERS):
        sy_i, H_i = sy_h(ep0 + dl)
        res = seq_p - mat.E * dl - sy_i
        dl += res / (mat.E + np.maximum(H_i, 0.0))
        dl = np.maximum(dl, 0.0)
    sy_new, _ = sy_h(ep0 + dl)
    scale = sy_new / seq_p
    fres[gidx] *= scale[:, None]
    mres[gidx] *= scale[:, None]
    epsp[gidx] = ep0 + dl


# ----------------------------------------------------------------------------
# Engine-side forces (pforc3.F)
# ----------------------------------------------------------------------------

def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    n1, n2, n3 = conn[:, 0], conn[:, 1], conn[:, 2]
    E, L = _frame(x[n1], x[n2], x[n3])

    # local velocities / rotation rates (pdefo3)
    v1 = np.einsum("nb,nba->na", v[n1], E)
    v2 = np.einsum("nb,nba->na", v[n2], E)
    t1 = np.einsum("nb,nba->na", vr[n1], E)
    t2 = np.einsum("nb,nba->na", vr[n2], E)

    invL = 1.0 / L
    eps_dot = (v2[:, 0] - v1[:, 0]) * invL
    gy_dot = (v2[:, 1] - v1[:, 1]) * invL - 0.5 * (t1[:, 2] + t2[:, 2])
    gz_dot = (v2[:, 2] - v1[:, 2]) * invL + 0.5 * (t1[:, 1] + t2[:, 1])
    kx_dot = (t2[:, 0] - t1[:, 0]) * invL
    ky_dot = (t2[:, 1] - t1[:, 1]) * invL
    kz_dot = (t2[:, 2] - t1[:, 2]) * invL

    # ---- resultant update (pmat3): elastic prediction ... ------------------
    fres, mres = st["fres"], st["mres"]
    f_old, m_old = fres.copy(), mres.copy()
    for sl, mat, prop in st["slices"]:
        p = prop.params
        fres[sl, 0] += mat.E * p["area"] * eps_dot[sl] * dt
        fres[sl, 1] += mat.G * p["area"] * gy_dot[sl] * dt
        fres[sl, 2] += mat.G * p["area"] * gz_dot[sl] * dt
        mres[sl, 0] += mat.G * p["ixx"] * kx_dot[sl] * dt
        mres[sl, 1] += mat.E * p["iyy"] * ky_dot[sl] * dt
        mres[sl, 2] += mat.E * p["izz"] * kz_dot[sl] * dt

        # ... then the LAW2 global-plasticity return (module docstring):
        # equivalent extreme-fiber stress from the resultants, 1-D
        # consistency solve on the Johnson-Cook curve, radial scaling of
        # all six resultants back to the yield surface.
        if mat.law == 2:
            _global_plastic_return(st, sl, mat, p)

    # ---- internal nodal forces & moments (pfint3, see docstring) -----------
    N, Qy, Qz = fres[:, 0], fres[:, 1], fres[:, 2]
    Mx, My, Mz = mres[:, 0], mres[:, 1], mres[:, 2]
    f2 = np.stack([N, Qy, Qz], axis=1)             # local internal force
    hL = 0.5 * L
    m1 = np.stack([-Mx, -My + Qz * hL, -Mz - Qy * hL], axis=1)
    m2 = np.stack([Mx, My + Qz * hL, Mz - Qy * hL], axis=1)

    # energy: midpoint resultants x rates x L (before scatter)
    fmid = 0.5 * (f_old + fres)
    mmid = 0.5 * (m_old + mres)
    st["eint"] += L * dt * (
        fmid[:, 0] * eps_dot + fmid[:, 1] * gy_dot + fmid[:, 2] * gz_dot
        + mmid[:, 0] * kx_dot + mmid[:, 1] * ky_dot + mmid[:, 2] * kz_dot)

    # back to global; fint/mint accumulate MINUS the internal terms
    # (elements package sign convention): -f1 = +f2 on node 1.
    fg = np.einsum("na,nba->nb", f2, E)
    np.add.at(fint, n1, fg)
    np.add.at(fint, n2, -fg)
    np.add.at(mint, n1, -np.einsum("na,nba->nb", m1, E))
    np.add.at(mint, n2, -np.einsum("na,nba->nb", m2, E))

    # ---- critical time step (exact init value, length-rescaled) ------------
    ratio = L / st["L0"]
    return st["dt0"] * np.where(ratio < 1.0, ratio, ratio ** -0.5)
