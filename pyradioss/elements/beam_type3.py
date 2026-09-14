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
from ..common.fastmath import cross3, norm3


# ----------------------------------------------------------------------------
# geometry: corotational frame (pcoor3.F)
# ----------------------------------------------------------------------------

def _frame(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray):
    """Local triad from the two end nodes and the orientation node.

    Returns (E (n,3,3) with columns e1|e2|e3, L (n,))."""
    d = x2 - x1
    n = len(d)
    if n == 0:
        return np.zeros((0, 3, 3)), np.zeros(0)
    L = np.maximum(norm3(d), EM20)
    e1 = d / L[:, None]
    yref = x3 - x1                                # local y lies in (e1, yref)
    dot = np.einsum("nb,nb->n", yref, e1)
    e2 = yref - dot[:, None] * e1
    ne2 = norm3(e2)
    deg = ne2 <= 1e-12
    if np.any(deg):
        # Fallback reference vector for collinear/degenerate orientation nodes:
        # choose global X if e1 is not aligned with X, else global Y
        for i in np.where(deg)[0]:
            e1_i = e1[i]
            ref = np.array([1.0, 0.0, 0.0]) if abs(e1_i[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            e2_cand = ref - np.dot(ref, e1_i) * e1_i
            ne2_cand = np.linalg.norm(e2_cand)
            if ne2_cand > 1e-12:
                e2[i] = e2_cand / ne2_cand
                ne2[i] = 1.0
            else:
                e2[i] = np.array([0.0, 0.0, 1.0])
                ne2[i] = 1.0
    e2 /= np.maximum(ne2, EM20)[:, None]
    e3 = cross3(e1, e2)
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
    if n == 0 or not slices:
        return dt0
    for sl, mat, prop in slices:
        p = getattr(prop, "params", {})
        area = p.get("area", getattr(prop, "area", 0.0))
        iyy = p.get("iyy", getattr(prop, "iyy", 0.0))
        izz = p.get("izz", getattr(prop, "izz", 0.0))
        ixx = p.get("ixx", getattr(prop, "ixx", iyy + izz))
        E = getattr(mat, "E", 0.0)
        G = getattr(mat, "G", 0.0)
        C = np.diag([E * area, G * area, G * area, G * ixx, E * iyy, E * izz])
        Ls = L0[sl]
        if len(Ls) == 0:
            continue
        ms = mass[sl] / 2.0                       # nodal mass
        Is = inertia_c[sl]                        # nodal inertia
        K = np.empty((len(Ls), 12, 12))
        for k, L in enumerate(Ls):                # small setup loop: init only
            B = _b_operator(L)
            K[k] = L * B.T @ C @ B
        minv_sqrt = np.zeros((len(Ls), 12))
        for d in range(3):
            minv_sqrt[:, d] = minv_sqrt[:, 6 + d] = \
                1.0 / np.sqrt(np.maximum(ms, EM20))
            minv_sqrt[:, 3 + d] = minv_sqrt[:, 9 + d] = \
                1.0 / np.sqrt(np.maximum(Is, EM20))
        Ksym = minv_sqrt[:, :, None] * K * minv_sqrt[:, None, :]
        w2 = np.linalg.eigvalsh(Ksym)[:, -1]      # largest eigenvalue
        dt0[sl] = 2.0 / np.sqrt(np.maximum(w2, EM20))
    return dt0


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia (starter pinit3/pmass3).
    Only nodes N1, N2 receive mass — N3 is orientation only."""
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0)
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
    slices = group.state.get("slices", [])
    for sl, mat, prop in slices:
        law = getattr(mat, "law", 1)
        if law not in (0, 1, 2):
            log.error(f"/BEAM: material LAW{law} not ported for beams "
                      f"(LAW0 void, LAW1 elastic, LAW2 global plasticity)",
                      "BEAM INIT")
        if law == 2:
            plastic = True
            if getattr(mat, "params", {}).get("c", 0.0) > 0.0:
                log.warning("/BEAM: the Johnson-Cook strain-rate term is "
                            "ignored by the global beam plasticity model",
                            "BEAM INIT")
        p = getattr(prop, "params", {})
        a_val = p.get("area", getattr(prop, "area", 0.0))
        iyy_val = p.get("iyy", getattr(prop, "iyy", 0.0))
        izz_val = p.get("izz", getattr(prop, "izz", 0.0))
        ixx_val = p.get("ixx", getattr(prop, "ixx", iyy_val + izz_val))
        area[sl] = a_val
        rho0[sl] = getattr(mat, "rho0", 0.0)
        igyr[sl] = (iyy_val + izz_val) / max(a_val, EM20)
        wy[sl] = np.sqrt(iyy_val * a_val / 3.0)
        wz[sl] = np.sqrt(izz_val * a_val / 3.0)
        wx[sl] = np.sqrt(ixx_val * a_val / 3.0)
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
        dt0=_exact_dt(L0, mass, inertia_c, slices),
        # mass-carrying connectivity (N1, N2 only) for output/energy code
        mass_conn=conn[:, :2].copy(),
        # dt_iner: per-NODE inertia share for the ROTATIONAL /DT/NODA
        # claim (M40, engine/mass_scaling.py): kr = 2 I/dt_e^2 with the
        # SAME lumped inertia _exact_dt used, so sqrt(2 I/kr) = dt_e —
        # the beam analogue of the shell STIR claim (upstream pdlen3.F
        # sets STIR = MAX(G*Ixx, KPHI*E*max(Iyy,Izz))/L and pmcum3.F adds
        # it per node; the port's equivalent-spring reading reproduces
        # its own exact-eigenvalue dt instead, the established /DT/NODA
        # contract of engine/mass_scaling.py).
        dt_iner=inertia_c.copy(),
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


def _global_plastic_return(st, sl, mat, p, iters=_NEWTON_ITERS):
    """Radial return of the six resultants onto the Johnson-Cook yield
    stress (rate term ignored — Starter warns). In-place on fres/mres and
    the global plastic strain epsp.

    ``iters`` is the Newton budget of the 1-D consistency solve — the
    explicit cycle keeps its historical 5 (bit-identical M3 contract; per
    tiny explicit step the residual re-enters every cycle so 5 is plenty),
    while the IMPLICIT residual passes ``_IMPL_NEWTON_ITERS`` because a
    large increment from a near-virgin state converges slowly at first
    (the JC slope B*n*e^(n-1) diverges as e -> 0 — the M11 truss lesson,
    MEASURED again here for the resultant return by the M15 tests: 5
    iterations leave an O(1) consistency residual on a first-yield
    implicit-size increment; 60 converge it to round-off)."""
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
    for _ in range(iters):
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
    """The explicit cycle path — the M3 code verbatim (a thin wrapper
    since M15: the body moved to ``_forces_core`` so the implicit
    residual can thread a converged plastic-return budget through the
    SAME kinematics; the explicit call is bit-identical)."""
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return np.zeros(0)
    if dt is None or dt <= 0.0:
        return group.state.get("dt0", np.zeros(group.n))
    if v is None:
        v = np.zeros_like(x)
    if vr is None:
        vr = np.zeros_like(x)
    return _forces_core(group, x, v, vr, dt, fint, mint, _NEWTON_ITERS)


def _forces_core(group, x, v, vr, dt, fint, mint, plast_iters):
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
    for sl, mat, prop in st.get("slices", []):
        if getattr(mat, "law", 1) == 0:
            continue
        p = getattr(prop, "params", {})
        area = p.get("area", getattr(prop, "area", 0.0))
        iyy = p.get("iyy", getattr(prop, "iyy", 0.0))
        izz = p.get("izz", getattr(prop, "izz", 0.0))
        ixx = p.get("ixx", getattr(prop, "ixx", iyy + izz))
        E_mod = getattr(mat, "E", 0.0)
        G_mod = getattr(mat, "G", 0.0)
        fres[sl, 0] += E_mod * area * eps_dot[sl] * dt
        fres[sl, 1] += G_mod * area * gy_dot[sl] * dt
        fres[sl, 2] += G_mod * area * gz_dot[sl] * dt
        mres[sl, 0] += G_mod * ixx * kx_dot[sl] * dt
        mres[sl, 1] += E_mod * iyy * ky_dot[sl] * dt
        mres[sl, 2] += E_mod * izz * kz_dot[sl] * dt

        # ... then the LAW2 global-plasticity return (module docstring):
        # equivalent extreme-fiber stress from the resultants, 1-D
        # consistency solve on the Johnson-Cook curve, radial scaling of
        # all six resultants back to the yield surface.
        if getattr(mat, "law", 1) == 2:
            _global_plastic_return(st, sl, mat, p, plast_iters)

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
    if fint is not None:
        fg = np.einsum("na,nba->nb", f2, E)
        np.add.at(fint, n1, fg)
        np.add.at(fint, n2, -fg)
    if mint is not None:
        np.add.at(mint, n1, -np.einsum("na,nba->nb", m1, E))
        np.add.at(mint, n2, -np.einsum("na,nba->nb", m2, E))

    # ---- critical time step (exact init value, length-rescaled) ------------
    ratio = L / st["L0"]
    return st["dt0"] * np.where(ratio < 1.0, ratio, ratio ** -0.5)


# ----------------------------------------------------------------------------
# Implicit tangent stiffness (M11) — a NEW entry point alongside forces()
# ----------------------------------------------------------------------------
# Fortran origin: the element-KE branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F`` dispatching the beam stiffness)
# + its ``imp_kgeo`` geometric path (/IMPL/NONLIN).
#
# The corotational Timoshenko beam's material tangent is the local 12x12
#
#     K_l = L * B^T C B,   C = diag(EA, GA, GA, GIxx, EIyy, EIzz)
#
# with B the SAME 6x12 generalized-strain-rate operator the force path
# integrates (``_b_operator`` — one point at mid-span, the reduced
# integration that avoids shear locking) — the exact stiffness the exact-dt
# eigenproblem of ``_exact_dt`` already builds. It is rotated to global by
# the corotational frame E per 3-dof block. Note the one-point linear
# element is nodally EXACT for end loads on a Timoshenko cantilever
# (delta = FL^3/3EI + FL/GA) — the M11 closed-form check.
#
# The orientation node N3 carries no DOFs (as it carries no force): the
# frame's dependence on N3 (and the frame-rotation derivative terms in
# general) multiplies the current RESULTANTS, so at zero prestress the
# material tangent above is the EXACT linearization (asserted by finite
# differences); with prestress those frame terms are exactly what kgeo()
# carries for the dominant axial resultant.
#
# K_geo: the transverse ("taut string") stiffness of the AXIAL force,
# (N/L)(I - a a^T) on the translations — for the LINEAR (one-point)
# interpolation this IS the consistent initial-stress operator
# (int N w'^2 dx with linear w gives exactly N/L; the L/12-type rotational
# couplings belong to CUBIC beam shape functions, which this element does
# not have). The shear-force and moment frame-coupling terms are omitted —
# they are O(Q/N, M/NL) of the axial term at a buckling state and the
# standard beam-column practice drops them (documented deferral,
# PORTING_GUIDE M11). A beam column therefore buckles at Euler's load like
# the truss-braced systems the M9 validations cover.
#
# LAW2 beams — the CONSISTENT tangent of the GLOBAL resultant-plasticity
# return (M15; removes the M11 deferral). NOTE what the checked Fortran
# offers to mirror: ``pmat3.F`` (this module's namesake, fetched) is the
# implicit beam's ELASTIC shear-stiffness setup called from pke3.F — the
# original's implicit beam KE has no resultant-plasticity linearization
# at all. The tangent below is therefore the exact derivative of the
# PORT'S OWN return map (the M13 IMP_KPRES principle: Newton needs
# consistency with the residual actually iterated).
#
# Derivation (resultant space R = [N, Qy, Qz, Mx, My, Mz], the order of
# the B rows and of C = diag(EA, GA, GA, GIxx, EIyy, EIzz)): the return
# scales the TRIAL resultants radially, R_new = s * R_tr with
# s = sy(ep0 + dl)/seq_tr and dl from the 1-D consistency
# seq_tr - E dl = sy(ep0 + dl). Because the equivalent stress seq(R) is
# positively HOMOGENEOUS of degree 1 (it is built from absolute values
# and a Euclidean norm of resultants over constant section moduli), its
# gradient q = d seq/d R is homogeneous of degree 0 — q(R_new) = q(R_tr)
# — and seq(R_new) = s seq_tr = sy_new. Both let the tangent be built
# from the POST-return state the implicit driver holds plus the
# increment's dl (the epsp_incr plumbing):
#
#     seq_tr = sy_new + E dl,       R_tr = R_new * seq_tr / sy_new,
#     d R_new / d R_tr = s I + R_tr (ds/dseq) q^T,
#     ds/dseq = (H/(E+H) - s)/seq_tr        (H = dsy/dep at ep0 + dl,
#                                            0 where sig_max caps)
#     =>  C_alg = s C + [(H/(E+H) - s)/seq_tr] R_tr (q^T C)
#
# a mildly NONSYMMETRIC rank-one update of the scaled elastic C — the
# same structure as the LAW2 shell tangent (M11), living in resultant
# space. The gradient q has the extreme-fiber pattern
#     q_N = (sn/seq) sgn(N)/A,  q_My = (sn/seq) sgn(My)/Wy,  (etc.)
#     q_Qy = (3 tau/seq) Qy/(A |Q|),  q_Mx = (3 tau/seq) sgn(Mx)/Wx
# — non-smooth at resultant sign changes exactly like the |.| terms of
# the yield function itself (the usual vertex of a piecewise-smooth
# surface; the M13 line search is the backstop). The element tangent is
# then K_l = L B^T C_alg B rotated by the frame, replacing the elastic
# diag(C) ONLY on the elements the increment actually yielded
# (epsp_incr > 0) — elastic beams keep the M11 code path bit-identical.
#
# The IMPLICIT residual runs its own ITERATED consistency solve
# (``implicit_internal_forces`` below — the M11 truss lesson applied to
# the resultant return, MEASURED by the M15 tests: the explicit path's
# historical 5 Newton iterations leave a first-yield implicit-size
# increment visibly off the hardening curve, because the JC slope
# B*n*e^(n-1) diverges at e -> 0; the shared explicit kernel is
# untouched — the M7 parity contract). The JC strain-RATE term was never
# in the beam model (the Starter warns), so nothing else needs
# disabling.

def _beam_edofs(conn):
    """(n, 12) global scalar DOF slot ids over N1, N2 (N3 carries none)."""
    n = len(conn)
    edofs = np.empty((n, 12), dtype=np.int64)
    for c in range(6):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 6 + c] = conn[:, 1] * 6 + c
    return edofs


def _frame_transform(E):
    """(n, 12, 12) local<-global transformation: each 3-dof block (v1, th1,
    v2, th2) transforms by E^T (local component a = sum_b E[b,a] global_b)."""
    n = len(E)
    T = np.zeros((n, 12, 12))
    for q in range(4):
        # T[3q+a, 3q+b] = E[b, a]
        T[:, 3 * q:3 * q + 3, 3 * q:3 * q + 3] = np.transpose(E, (0, 2, 1))
    return T


def tangent(group, x, epsp_incr=None):
    """Element tangent stiffness for the whole beam group — LAW1 elastic;
    LAW2 with the CONSISTENT resultant-plasticity tangent on the elements
    the increment yielded (M15, see the derivation note above).

    Returns ``(ke, edofs)``: ``ke`` (n, 12, 12) over the two force-carrying
    nodes x 6 global dofs, ``edofs`` (n, 12). ``epsp_incr`` (n,) is the
    increment's global plastic-strain step (None / zeros = all elastic —
    that path is the M11 code verbatim)."""
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return np.zeros((0, 12, 12)), np.zeros((0, 12), dtype=np.int64)
    st = group.state
    conn = group.conn
    n = group.n
    E, L = _frame(x[conn[:, 0]], x[conn[:, 1]], x[conn[:, 2]])

    # local K_l = L * B^T C B, vectorized over the group: B(L) has entries
    # +-1/L and +-1/2 only (see _b_operator) — build it stacked
    B = np.zeros((n, 6, 12))
    invL = 1.0 / L
    B[:, 0, 0], B[:, 0, 6] = -invL, invL                   # eps
    B[:, 1, 1], B[:, 1, 7] = -invL, invL                   # gy
    B[:, 1, 5] = B[:, 1, 11] = -0.5
    B[:, 2, 2], B[:, 2, 8] = -invL, invL                   # gz
    B[:, 2, 4] = B[:, 2, 10] = 0.5
    B[:, 3, 3], B[:, 3, 9] = -invL, invL                   # kx (twist)
    B[:, 4, 4], B[:, 4, 10] = -invL, invL                  # ky
    B[:, 5, 5], B[:, 5, 11] = -invL, invL                  # kz

    Cd = np.zeros((n, 6))                                  # diag of C
    for sl, mat, prop in st.get("slices", []):
        law = getattr(mat, "law", 1)
        if law not in (0, 1, 2):
            raise NotImplementedError(
                f"the implicit beam tangent supports LAW0, LAW1 and LAW2 (the "
                f"global resultant-plasticity model, M15); got "
                f"LAW{law} — see PORTING_GUIDE")
        if law == 0:
            Cd[sl] = 0.0
            continue
        p = getattr(prop, "params", {})
        area = p.get("area", getattr(prop, "area", 0.0))
        iyy = p.get("iyy", getattr(prop, "iyy", 0.0))
        izz = p.get("izz", getattr(prop, "izz", 0.0))
        ixx = p.get("ixx", getattr(prop, "ixx", iyy + izz))
        E_mod = getattr(mat, "E", 0.0)
        G_mod = getattr(mat, "G", 0.0)
        Cd[sl, 0] = E_mod * area
        Cd[sl, 1] = Cd[sl, 2] = G_mod * area
        Cd[sl, 3] = G_mod * ixx
        Cd[sl, 4] = E_mod * iyy
        Cd[sl, 5] = E_mod * izz
    # K_l = L * B^T diag(C) B  (stacked)
    CB = Cd[:, :, None] * B                                # (n, 6, 12)
    Kl = L[:, None, None] * np.einsum("nai,naj->nij", B, CB)

    # ---- LAW2 consistent resultant-plasticity blocks (M15) -----------------
    # Elements the increment yielded (epsp_incr > 0) get their local block
    # RECOMPUTED with the algorithmic C_alg (the derivation note above);
    # elastic elements keep the diag(C) block bit-for-bit.
    if epsp_incr is not None:
        for sl, mat, prop in st.get("slices", []):
            if getattr(mat, "law", 1) != 2:
                continue
            dl_sl = epsp_incr[sl]
            plas = np.where(dl_sl > 0.0)[0]
            if len(plas) == 0:
                continue
            gidx = np.arange(sl.start, sl.stop)[plas]
            mp = getattr(mat, "params", {})
            p = getattr(prop, "params", {})
            A = p.get("area", getattr(prop, "area", 0.0))
            # POST-return state (the driver's trial buffers) + increment
            R = np.concatenate([st["fres"][gidx], st["mres"][gidx]],
                               axis=1)                     # (m, 6)
            ep = np.maximum(st["epsp"][gidx], 1e-20)
            sy = mp["A"] + mp["B"] * ep ** mp["n"]
            H = mp["B"] * mp["n"] * ep ** (mp["n"] - 1.0)
            capped = sy > mp.get("sig_max", 1e30)
            sy = np.where(capped, mp.get("sig_max", 1e30), sy)
            H = np.where(capped, 0.0, np.maximum(H, 0.0))
            dl = dl_sl[plas]
            seq_tr = sy + getattr(mat, "E", 0.0) * dl        # the consistency identity
            scale = sy / seq_tr             # s = sy_new / seq_tr
            R_tr = R / scale[:, None]       # radial: homogeneity deg 1
            # q = grad seq at the post state (degree-0 homogeneous —
            # identical at the trial state; same sign pattern)
            N, Qy, Qz = R[:, 0], R[:, 1], R[:, 2]
            Mx, My, Mz = R[:, 3], R[:, 4], R[:, 5]
            wy, wz, wx = st["wy"][gidx], st["wz"][gidx], st["wx"][gidx]
            sn = np.abs(N) / A + np.abs(My) / wy + np.abs(Mz) / wz
            tau = np.abs(Mx) / wx + np.sqrt(Qy ** 2 + Qz ** 2) / A
            seq = np.sqrt(sn ** 2 + 3.0 * tau ** 2) + 1e-30
            Qn = np.maximum(np.sqrt(Qy ** 2 + Qz ** 2), 1e-30)
            q = np.empty((len(plas), 6))
            q[:, 0] = (sn / seq) * np.sign(N) / A
            q[:, 1] = (3.0 * tau / seq) * Qy / (A * Qn)
            q[:, 2] = (3.0 * tau / seq) * Qz / (A * Qn)
            q[:, 3] = (3.0 * tau / seq) * np.sign(Mx) / wx
            q[:, 4] = (sn / seq) * np.sign(My) / wy
            q[:, 5] = (sn / seq) * np.sign(Mz) / wz
            # C_alg = s C + [(H/(E+H) - s)/seq_tr] R_tr (q^T C)
            Csub = Cd[gidx]                              # (m, 6) diag
            coef = (H / (getattr(mat, "E", 0.0) + H) - scale) / seq_tr
            Calg = scale[:, None, None] * \
                np.einsum("ma,ab->mab", Csub, np.eye(6))
            Calg += coef[:, None, None] * np.einsum(
                "ma,mb->mab", R_tr, q * Csub)
            Bp = B[gidx]
            Kl[gidx] = L[gidx, None, None] * np.einsum(
                "mai,mab,mbj->mij", Bp, Calg, Bp)

    T = _frame_transform(E)
    ke = np.einsum("nki,nkl,nlj->nij", T, Kl, T)           # (n, 12, 12)
    return ke, _beam_edofs(conn)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the lumped mass/inertia is ``starter/source/elements/beam/
# pmass3.F`` (the m/2 nodal mass and Key's boosted rotary inertia this file's
# ``init_group`` returns). The CONSISTENT mass is the Timoshenko/Rayleigh-beam
# shape-function integral M = ∫ ρ (A Nᵀ_t N_t + I Nᵀ_r N_r) dx; ported here for
# the M16 modal eigensolver alongside — never mutating — the lumped path
# (whose artificial rotary-inertia boost would corrupt the natural frequencies
# it was never meant to feed).
#
# Theory (Przemieniecki "Theory of Matrix Structural Analysis" ch. 11 — the
# same reference this module's stiffness cites for the beam, ch. 5). The local
# 12×12 consistent mass decouples into the four classical blocks, each built on
# the reference length L0 (mass conservation) and then rotated to global axes
# by the SAME corotational frame E as the stiffness (me_g = Tᵀ me_l T):
#
# * AXIAL (v1x, v2x): the 2-node bar mass  ρA L/6 [[2,1],[1,2]].
# * TORSION (θ1x, θ2x): the polar rotary-inertia bar mass  ρ Ip L/6 [[2,1],
#   [1,2]]  with the mass polar moment Ip = Iyy + Izz (NOT the St-Venant torsion
#   constant Ixx, which is a STIFFNESS quantity — the twisting kinetic energy
#   is governed by the true second moments).
# * BENDING (transverse translation + its slope) in each principal plane: the
#   Hermite-cubic translational mass PLUS the rotary-inertia mass
#       M_t = ρA L/420 [[156, 22L, 54,-13L],[22L,4L²,13L,-3L²],
#                       [54, 13L, 156,-22L],[-13L,-3L²,-22L, 4L²]]
#       M_r = ρ I /(30L) [[36, 3L,-36, 3L],[3L, 4L²,-3L,-L²],
#                        [-36,-3L, 36,-3L],[3L, -L²,-3L, 4L²]]
#   coupling translation and rotation (the off-diagonal 22L / 3L terms) — the
#   term that makes the cantilever bending frequencies match the Euler–Bernoulli
#   βₙL roots to <1% at a handful of elements (validated in M16). The x–z plane
#   uses the slope convention θy = -w', so its block is P M P with
#   P = diag(1,-1,1,-1) (sign-flipping the translation↔rotation couplings);
#   I = Izz for the x–y plane (bending about z), Iyy for the x–z plane.
#
# A NOTE on consistency: this is the RAYLEIGH beam consistent mass (Hermite
# cubic translation field), one order richer than the beam's LINEAR Timoshenko
# stiffness interpolation. That mild inconsistency is the standard engineering
# choice (Przemieniecki, Cook et al.) and is what buys the bending-frequency
# accuracy; a fully interpolation-consistent linear mass would need far finer
# meshes to reach the same %. Partition of unity: the four translational rows
# each sum to m/2 (rigid translation → ½ vᵀMv = ½ m|v|² exact); the Hermite
# rotation rows sum to zero (a rigid translation excites no rotation), exactly
# as they must.

def _bending_mass_blocks(L, rhoA, rhoI):
    """The 4×4 bending consistent mass (translational + rotary) for DOFs
    [w1, θ1, w2, θ2] with θ = +w', vectorized over the group. ``L``, ``rhoA``,
    ``rhoI`` are (n,)."""
    n = len(L)
    L2 = L * L
    Mt = np.empty((n, 4, 4))
    # translational Hermite mass ρA L / 420 * [...]
    c = rhoA * L / 420.0
    Mt[:, 0, 0] = 156 * c;      Mt[:, 0, 1] = 22 * L * c
    Mt[:, 0, 2] = 54 * c;       Mt[:, 0, 3] = -13 * L * c
    Mt[:, 1, 1] = 4 * L2 * c;   Mt[:, 1, 2] = 13 * L * c
    Mt[:, 1, 3] = -3 * L2 * c
    Mt[:, 2, 2] = 156 * c;      Mt[:, 2, 3] = -22 * L * c
    Mt[:, 3, 3] = 4 * L2 * c
    # rotary-inertia mass ρI /(30 L) * [...]
    d = rhoI / (30.0 * L)
    Mt[:, 0, 0] += 36 * d;      Mt[:, 0, 1] += 3 * L * d
    Mt[:, 0, 2] += -36 * d;     Mt[:, 0, 3] += 3 * L * d
    Mt[:, 1, 1] += 4 * L2 * d;  Mt[:, 1, 2] += -3 * L * d
    Mt[:, 1, 3] += -1 * L2 * d
    Mt[:, 2, 2] += 36 * d;      Mt[:, 2, 3] += -3 * L * d
    Mt[:, 3, 3] += 4 * L2 * d
    # symmetrize (only the upper triangle was filled)
    i_lo = np.tril_indices(4, -1)
    Mt[:, i_lo[0], i_lo[1]] = Mt[:, i_lo[1], i_lo[0]]
    return Mt


def consistent_mass(group, x):
    """Consistent element mass of the corotational Timoshenko beam (see the
    note above): the local 12×12 axial+torsion+two-plane-bending mass built on
    the reference length, rotated to global axes by the frame at geometry
    ``x``.

    Returns ``(me (n,12,12), edofs (n,12))`` over the two force-carrying nodes
    × 6 global DOFs — the same addressing and frame transform as
    ``tangent()``."""
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return np.zeros((0, 12, 12)), np.zeros((0, 12), dtype=np.int64)
    st = group.state
    conn = group.conn
    n = group.n
    L0 = st.get("L0")
    if L0 is None:
        _, L0 = _frame(x[conn[:, 0]], x[conn[:, 1]], x[conn[:, 2]])
    # per-element ρA, ρ Iyy, ρ Izz, ρ Ip from the part slices
    rhoA = np.zeros(n)
    rhoIyy = np.zeros(n)
    rhoIzz = np.zeros(n)
    rhoIp = np.zeros(n)
    for sl, mat, prop in st.get("slices", []):
        p = getattr(prop, "params", {})
        area = p.get("area", getattr(prop, "area", 0.0))
        iyy = p.get("iyy", getattr(prop, "iyy", 0.0))
        izz = p.get("izz", getattr(prop, "izz", 0.0))
        rho0 = getattr(mat, "rho0", 0.0)
        rhoA[sl] = rho0 * area
        rhoIyy[sl] = rho0 * iyy
        rhoIzz[sl] = rho0 * izz
        rhoIp[sl] = rho0 * (iyy + izz)   # mass polar moment

    Ml = np.zeros((n, 12, 12))
    # axial (v1x=0, v2x=6): ρA L/6 [[2,1],[1,2]]
    ax = rhoA * L0 / 6.0
    Ml[:, 0, 0] = 2 * ax;  Ml[:, 0, 6] = ax
    Ml[:, 6, 0] = ax;      Ml[:, 6, 6] = 2 * ax
    # torsion (th1x=3, th2x=9): ρ Ip L/6 [[2,1],[1,2]]
    tor = rhoIp * L0 / 6.0
    Ml[:, 3, 3] = 2 * tor;  Ml[:, 3, 9] = tor
    Ml[:, 9, 3] = tor;      Ml[:, 9, 9] = 2 * tor
    # bending in the x-y plane, DOFs [v1y=1, th1z=5, v2y=7, th2z=11] (θz=+w')
    Mxy = _bending_mass_blocks(L0, rhoA, rhoIzz)
    ib = [1, 5, 7, 11]
    for a in range(4):
        for b in range(4):
            Ml[:, ib[a], ib[b]] = Mxy[:, a, b]
    # bending in the x-z plane, DOFs [v1z=2, th1y=4, v2z=8, th2y=10] (θy=-w'):
    # apply P = diag(1,-1,1,-1) to sign-flip the translation<->rotation terms
    Mxz = _bending_mass_blocks(L0, rhoA, rhoIyy)
    P = np.array([1.0, -1.0, 1.0, -1.0])
    Mxz = P[None, :, None] * Mxz * P[None, None, :]
    iz = [2, 4, 8, 10]
    for a in range(4):
        for b in range(4):
            Ml[:, iz[a], iz[b]] = Mxz[:, a, b]

    # rotate the local mass to global axes with the corotational frame,
    # exactly as tangent() rotates the local stiffness
    E, _ = _frame(x[conn[:, 0]], x[conn[:, 1]], x[conn[:, 2]])
    T = _frame_transform(E)
    me = np.einsum("nki,nkl,nlj->nij", T, Ml, T)
    return me, _beam_edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness of the AXIAL resultant,
    (N/L)(I - a a^T) on the two nodes' translations (see the note above for
    why this is the consistent operator of the linear element, and which
    coupling terms are deferred). Identically zero at zero axial force."""
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return np.zeros((0, 12, 12)), np.zeros((0, 12), dtype=np.int64)
    st = group.state
    conn = group.conn
    n = group.n
    fres = st.get("fres")
    if fres is None or np.all(fres[:, 0] == 0.0):
        return np.zeros((n, 12, 12)), _beam_edofs(conn)
    d = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(d), EM20)
    a = d / L[:, None]
    N_over_L = fres[:, 0] / L
    eye = np.eye(3)
    kb = N_over_L[:, None, None] * (eye[None, :, :]
                                    - np.einsum("ni,nj->nij", a, a))
    ke = np.zeros((n, 12, 12))
    ke[:, 0:3, 0:3] = kb
    ke[:, 6:9, 6:9] = kb
    ke[:, 0:3, 6:9] = -kb
    ke[:, 6:9, 0:3] = -kb
    return ke, _beam_edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Nodal forces/moments at configuration ``x`` from the CURRENT local
    resultants — the updated-Lagrangian end-configuration force assembly of
    the M9/M11 implicit residual (the resultants were just advanced by a
    midpoint-geometry ``forces()`` call; this re-states the pfint3
    expressions with the frame and length OF THIS geometry).
    ``u``/``ur`` unused."""
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return
    st = group.state
    conn = group.conn
    n1, n2, n3 = conn[:, 0], conn[:, 1], conn[:, 2]
    E, L = _frame(x[n1], x[n2], x[n3])

    fres, mres = st.get("fres"), st.get("mres")
    if fres is None:
        fres = np.zeros((group.n, 3))
    if mres is None:
        mres = np.zeros((group.n, 3))
    N, Qy, Qz = fres[:, 0], fres[:, 1], fres[:, 2]
    Mx, My, Mz = mres[:, 0], mres[:, 1], mres[:, 2]
    f2 = np.stack([N, Qy, Qz], axis=1)
    hL = 0.5 * L
    m1 = np.stack([-Mx, -My + Qz * hL, -Mz - Qy * hL], axis=1)
    m2 = np.stack([Mx, My + Qz * hL, Mz - Qy * hL], axis=1)

    if fint is not None:
        fg = np.einsum("na,nba->nb", f2, E)
        np.add.at(fint, n1, fg)
        np.add.at(fint, n2, -fg)
    if mint is not None:
        np.add.at(mint, n1, -np.einsum("na,nba->nb", m1, E))
        np.add.at(mint, n2, -np.einsum("na,nba->nb", m2, E))


#: Newton budget of the IMPLICIT 1-D consistency solve — sized so a
#: first-yield implicit-size increment converges to round-off (the slow
#: near-virgin start of the JC slope needs ~10 iterations before the
#: quadratic tail kicks in; 60 is cheap at implicit model sizes and the
#: loop is vectorized over the group).
_IMPL_NEWTON_ITERS = 60


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom):
    """The beam's own implicit residual (M15 — dispatched by the drivers
    instead of forces(), exactly like the M11 truss/spring hooks; see the
    LAW2 tangent note above for WHY: the explicit kernel's historical
    5-iteration consistency solve veers off the hardening curve at
    implicit increment sizes, and the shared explicit path must stay
    bit-identical under the M7 parity contract).

    Kinematics mirror the drivers' historical use of the rate-form
    kernels VERBATIM (LAW1 groups reproduce the old route bit for bit —
    asserted by the M15 tests): under linear geometry, one
    ``_forces_core`` call at the committed frame with the increment as a
    pseudo-velocity at dt = 1 (state update + force assembly in one);
    under /IMPL/NONLIN, the midpoint-geometry state update with the
    force discarded, then the END-configuration ``static_internal_forces``
    re-statement. The ONLY difference is the plastic-return budget:
    ``_IMPL_NEWTON_ITERS`` instead of 5, i.e. a CONVERGED resultant
    return (the M11 truss iterated-return lesson in resultant space)."""
    if group is None or getattr(group, "n", 0) == 0 or len(getattr(group, "conn", [])) == 0:
        return
    if u is None:
        u = np.zeros_like(x_ref)
    if ur is None:
        ur = np.zeros_like(x_ref)
    if not nlgeom:
        _forces_core(group, x_ref, u, ur, 1.0, fint, mint,
                     _IMPL_NEWTON_ITERS)
        return
    nn = len(x_ref)
    junk_f = np.zeros((nn, 3))
    junk_m = np.zeros((nn, 3))
    _forces_core(group, x_ref + 0.5 * u, u, ur, 1.0, junk_f, junk_m,
                 _IMPL_NEWTON_ITERS)
    static_internal_forces(group, x_ref + u, u, ur, fint, mint)
