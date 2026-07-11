"""
2-node truss element (/TRUSS + /PROP/TRUSS).

Fortran origin: ``engine/source/elements/truss/`` (tforc3.F, tdlen3.F).

A truss carries only axial force. The kinematics are exact for large
displacement / small strain: the axial strain rate is the relative axial
velocity over the current length,

    eps_dot = (v2 - v1) . a / L,      a = (x2 - x1)/L

and the axial stress follows the 1-D form of the part's material law:

    LAW1:  d sigma = E * deps
    LAW2:  elastic trial then 1-D radial return on the Johnson-Cook yield
           stress (uniaxial plasticity: dlambda = (|s| - sy)/(E + H))

The internal force is F = A0 * sigma along the current axis (engineering
stress on the initial section — the small-strain assumption of the
original element). Critical time step: dt = L / c with c = sqrt(E/rho).
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..common.fastmath import norm3


def init_group(group, model, log):
    xe = model.x0[group.conn]                     # (n, 2, 3)
    dx = xe[:, 1] - xe[:, 0]
    L0 = norm3(dx)
    if np.any(L0 <= 0):
        for eid in group.ids[L0 <= 0]:
            log.error(f"/TRUSS {eid}: zero length", "TRUSS INIT")
    n = group.n
    area = np.zeros(n)
    rho0 = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        area[sl] = prop.params["area"]
        rho0[sl] = mat.rho0
    mass = rho0 * area * L0
    group.state.update(
        sig=np.zeros(n), epsp=np.zeros(n), area=area, L0=L0,
        mass=mass, eint=np.zeros(n), ehour=np.zeros(n),
    )
    node_idx = group.conn.reshape(-1)
    return node_idx, np.repeat(mass / 2.0, 2), None


def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    a = dx / L[:, None]
    dv = v[conn[:, 1]] - v[conn[:, 0]]
    eps_dot = np.einsum("nb,nb->n", dv, a) / L
    deps = eps_dot * dt

    sig = st["sig"]
    sig_old = sig.copy()
    c = np.zeros(group.n)
    for sl, mat, prop in st["slices"]:
        E = mat.E
        c[sl] = np.sqrt(E / mat.rho0)
        sig[sl] += E * deps[sl]                     # elastic trial
        if mat.law == 2:
            # 1-D radial return on the Johnson-Cook curve
            p = mat.params
            epsp = st["epsp"][sl]
            e = np.maximum(epsp, 1e-20)
            rate_fac = 1.0
            if p.get("c", 0.0) > 0.0:
                r = np.maximum(np.abs(eps_dot[sl]) / p["eps_dot_0"], 1.0)
                rate_fac = 1.0 + p["c"] * np.log(r)
            sy = np.minimum((p["A"] + p["B"] * e ** p["n"]) * rate_fac,
                            p["sig_max"])
            H = p["B"] * p["n"] * e ** (p["n"] - 1.0) * rate_fac
            over = np.abs(sig[sl]) - sy
            plastic = over > 0.0
            dl = np.where(plastic, over / (E + np.maximum(H, 0.0)), 0.0)
            st["epsp"][sl] = epsp + dl
            sig[sl] = np.where(plastic, np.sign(sig[sl]) * (sy + H * dl),
                               sig[sl])

    F = st["area"] * sig
    # tension (sig>0) pulls node 1 toward node 2: this force is already
    # the "-internal" contribution (see elements package docstring).
    fvec = F[:, None] * a
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)

    st["eint"] += st["area"] * L * 0.5 * (sig_old + sig) * deps
    return L / c


# ----------------------------------------------------------------------------
# Implicit tangent stiffness (M9) — alongside forces(), like the hexa/BT4 M8
# tangents. Fortran origin: the element-KE branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F``); geometric part = the ``imp_kgeo``
# path of /IMPL/NONLIN.
# ----------------------------------------------------------------------------
# The corotational truss has the textbook exact tangent (Crisfield vol. 1,
# ch. 3 — his introductory example for geometric nonlinearity). With
# a = (x2-x1)/L the current axis and F = A*sigma the axial force,
#
#     f_on_node1 = +F a ,  f_on_node2 = -F a      (the forces() convention)
#
# and the exact linearization w.r.t. the END-configuration coordinates is
#
#     K = d(-f)/du = [ Kb  -Kb ]      Kb = (E A / L) a a^T          (material)
#                    [-Kb   Kb ]         + (F / L) (I - a a^T)      (geometric)
#
# The material part is the axial stiffness along the current axis; the
# geometric part is the "taut string" transverse stiffness — the tension
# (or, negated, the compression softening) resisting a perpendicular motion
# by rotating the force direction: exactly d(a)/dx = (I - aa^T)/L acting on
# the carried force. This is the term that lets a shallow two-bar (von
# Mises) truss soften, reach its limit point and snap through — the M9
# arc-length validation traces it against the closed form.
#
# Materials (M11 removes the M9 LAW2 deferral). The implicit residual does
# NOT reuse forces() for the truss anymore: the explicit kernel's 1-D
# return is a SINGLE linearized step (dl = (|sig_tr| - sy0)/(E + H0) with
# H0 frozen at the committed plastic strain) — a per-cycle approximation
# that is exact in the explicit limit of tiny steps but VEERS OFF the
# hardening curve at implicit load-increment sizes (measured: 20 x 0.05
# increments to sigma = 0.5 leave eps_p at 0.006 instead of the JC 0.04 —
# the near-virgin JC slope B*n*e^(n-1) diverges as e -> 0, so the one-step
# return barely flows). The implicit path therefore supplies its own
# residual, ``implicit_internal_forces`` (the total-form-spring mechanism),
# identical to forces() for LAW1 bit for bit, but running the ITERATED
# Newton consistency solve for LAW2 —
#
#     |sig_tr| - E dl = sy(ep0 + dl),   sy capped at sig_max (H = 0 there)
#
# — exactly the solid/shell radial-return discipline (forces() itself is
# untouched: the M7 parity contract). The JC strain-RATE term is OFF here
# like every implicit rate device (M10 convention). The CONSISTENT tangent
# of that return is the textbook elastoplastic modulus
#
#     d sig / d eps = E * H / (E + H)         (H = dsy/dep at ep0 + dl, the
#                                             END point of the converged
#                                             consistency; 0 where sig_max
#                                             caps the curve)
#
# — in 1-D the consistent and continuum tangents coincide (no frozen radial
# direction), and this is what carries the quadratic Newton tail asserted
# by the M11 elastoplastic-truss validation. Yielding is detected from the
# increment's epsp_incr > 0, exactly like the solid/shell tangents.

def _axis(group, x):
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    return conn, L, dx / L[:, None]


def _blocks_to_element(kb):
    """Expand the (n, 3, 3) relative block Kb into the (n, 6, 6) element
    tangent [[Kb, -Kb], [-Kb, Kb]] (dof order: node1 xyz, node2 xyz)."""
    n = len(kb)
    ke = np.empty((n, 6, 6))
    ke[:, :3, :3] = kb
    ke[:, 3:, 3:] = kb
    ke[:, :3, 3:] = -kb
    ke[:, 3:, :3] = -kb
    return ke


def _edofs(conn):
    edofs = np.empty((len(conn), 6), dtype=np.int64)
    for c in range(3):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 3 + c] = conn[:, 1] * 6 + c
    return edofs


def tangent(group, x, epsp_incr=None):
    """Material element tangent (E_t A / L) a a^T for the whole truss group
    at geometry ``x``, with E_t the CONSISTENT axial modulus: E elastic,
    E*H/(E+H) on elements that yielded this increment (see the note above).
    ``epsp_incr`` (n,) is the increment's plastic-strain step (None / zeros
    = all elastic). Returns (ke (n,6,6), edofs (n,6)) in the implicit
    assembler's convention."""
    st = group.state
    conn, L, a = _axis(group, x)
    n = group.n
    k_ax = np.zeros(n)
    for sl, mat, prop in st["slices"]:
        if mat.law not in (1, 2):
            raise NotImplementedError(
                f"the implicit truss tangent supports LAW1 and LAW2; got "
                f"LAW{mat.law} (see PORTING_GUIDE)")
        Emod = np.full(sl.stop - sl.start, mat.E)
        if mat.law == 2 and epsp_incr is not None:
            dl = epsp_incr[sl]
            plastic = dl > 0.0
            if np.any(plastic):
                # hardening slope at the END of the converged consistency
                # solve (see the note above); H = 0 where the sig_max cap
                # rules — matching the iterated implicit return exactly
                p = mat.params
                e = np.maximum(st["epsp"][sl], 1e-20)
                sy = p["A"] + p["B"] * e ** p["n"]
                H = np.maximum(p["B"] * p["n"] * e ** (p["n"] - 1.0), 0.0)
                H = np.where(sy > p["sig_max"], 0.0, H)
                Emod = np.where(plastic, mat.E * H / (mat.E + H), Emod)
        k_ax[sl] = Emod * st["area"][sl] / L[sl]
    kb = k_ax[:, None, None] * np.einsum("ni,nj->nij", a, a)
    return _blocks_to_element(kb), _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness (F/L)(I - a a^T) from
    the current axial stress state at geometry ``x`` (see the note above).
    Same shapes as ``tangent()``; identically zero at zero stress."""
    st = group.state
    conn, L, a = _axis(group, x)
    F_over_L = st["area"] * st["sig"] / L
    eye = np.eye(3)
    kb = F_over_L[:, None, None] * (eye[None, :, :]
                                    - np.einsum("ni,nj->nij", a, a))
    return _blocks_to_element(kb), _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Nodal force at configuration ``x`` from the current stress state —
    the updated-Lagrangian force assembly of the M9 implicit residual
    (stress already updated by a midpoint-geometry forces() call):
    F = A*sigma along the CURRENT axis. ``u``/``ur``/``mint`` unused.
    (Kept for callers like the buckling prestress path; the implicit
    drivers reach the truss through ``implicit_internal_forces`` below
    since M11.)"""
    st = group.state
    conn, L, a = _axis(group, x)
    fvec = (st["area"] * st["sig"])[:, None] * a
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)


_IMPL_NEWTON_ITERS = 12   # the iterated 1-D consistency solve (see note)


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom):
    """The truss's own implicit residual (M11 — called by the drivers
    instead of forces(); see the LAW2 note above for WHY the explicit
    kernel's one-step return cannot serve the implicit increment sizes).

    Kinematics mirror the driver's use of forces() exactly: the strain
    increment is measured on the COMMITTED frame under linear geometry and
    on the MIDPOINT geometry under /IMPL/NONLIN (Hughes–Winget — identical
    to the old forces(x_mid) call, so the M9 corotational log-strain
    behaviour and the arc-length validations are reproduced bit for bit
    for LAW1), and the nodal force acts along the committed (linear) or
    END (nonlinear) axis. LAW2 runs the ITERATED radial return with the
    rate term off. State (sig, epsp, eint) updates to the trial values in
    place — the drivers' snapshot/commit machinery rolls back exactly as
    for the rate-form kernels."""
    st = group.state
    conn = group.conn
    x_eval = (x_ref + 0.5 * u) if nlgeom else x_ref
    dxm = x_eval[conn[:, 1]] - x_eval[conn[:, 0]]
    Lm = np.maximum(norm3(dxm), EM20)
    am = dxm / Lm[:, None]
    du = u[conn[:, 1]] - u[conn[:, 0]]
    deps = np.einsum("nb,nb->n", du, am) / Lm

    sig = st["sig"]
    sig_old = sig.copy()
    for sl, mat, prop in st["slices"]:
        E = mat.E
        sig[sl] += E * deps[sl]                      # elastic trial
        if mat.law == 2:
            # iterated 1-D consistency solve on the JC static curve
            # (rate term OFF under implicit — the M10 convention)
            p = mat.params
            ep0 = st["epsp"][sl]
            over = np.abs(sig[sl])
            sy0 = np.minimum(p["A"] + p["B"] *
                             np.maximum(ep0, 1e-20) ** p["n"], p["sig_max"])
            plastic = over > sy0
            if np.any(plastic):
                dl = np.zeros(sl.stop - sl.start)
                for _ in range(_IMPL_NEWTON_ITERS):
                    e = np.maximum(ep0 + dl, 1e-20)
                    sy = p["A"] + p["B"] * e ** p["n"]
                    H = p["B"] * p["n"] * e ** (p["n"] - 1.0)
                    capped = sy > p["sig_max"]
                    sy = np.where(capped, p["sig_max"], sy)
                    H = np.where(capped, 0.0, np.maximum(H, 0.0))
                    res = over - E * dl - sy
                    dl += np.where(plastic, res / (E + H), 0.0)
                    dl = np.maximum(dl, 0.0)
                e = np.maximum(ep0 + dl, 1e-20)
                sy_new = np.minimum(p["A"] + p["B"] * e ** p["n"],
                                    p["sig_max"])
                sig[sl] = np.where(plastic, np.sign(sig[sl]) * sy_new,
                                   sig[sl])
                st["epsp"][sl] = ep0 + dl

    # force along the committed axis (linear) / the END axis (nonlinear) —
    # matching the old forces()/static_internal_forces pairing
    if nlgeom:
        conn2, Lf, af = _axis(group, x_ref + u)
    else:
        af = am
    fvec = (st["area"] * sig)[:, None] * af
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)
    # trapezoidal internal-energy booking, the forces() formula
    st["eint"] += st["area"] * Lm * 0.5 * (sig_old + sig) * deps
