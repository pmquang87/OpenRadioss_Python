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

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3


def init_group(group, model, log):
    if group is None or group.n == 0 or len(group.conn) == 0:
        if group is not None:
            group.state.update(
                sig=np.zeros(0), epsp=np.zeros(0), area=np.zeros(0), L0=np.zeros(0),
                mass=np.zeros(0), eint=np.zeros(0), ehour=np.zeros(0),
                off=np.ones(0, dtype=float),
            )
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=float), None

    xe = model.x0[group.conn]                     # (n, 2, 3)
    dx = xe[:, 1] - xe[:, 0]
    L0 = norm3(dx)
    if np.any(L0 <= 0):
        for eid in group.ids[L0 <= 0]:
            log.error(f"/TRUSS {eid}: zero length", "TRUSS INIT")
    n = group.n
    area = np.zeros(n)
    rho0 = np.zeros(n)
    slices = group.state.get("slices", [])
    for sl, mat, prop in slices:
        area_val = 0.0
        if hasattr(prop, "params") and isinstance(prop.params, dict) and "area" in prop.params:
            area_val = prop.params["area"]
        elif hasattr(prop, "area"):
            area_val = prop.area
        area[sl] = area_val
        rho0[sl] = getattr(mat, "rho0", 0.0)
    mass = rho0 * area * L0
    group.state.update(
        sig=np.zeros(n), epsp=np.zeros(n), area=area, L0=L0,
        mass=mass, eint=np.zeros(n), ehour=np.zeros(n),
        off=np.ones(n, dtype=float),
    )
    node_idx = group.conn.reshape(-1)
    return node_idx, np.repeat(mass / 2.0, 2), None


def forces(group, x, v, vr, dt, fint, mint):
    if group is None or group.n == 0 or len(group.conn) == 0:
        return np.zeros(0, dtype=float)

    st = group.state
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    a = dx / L[:, None]

    if v is None:
        eps_dot = np.zeros(len(conn))
    else:
        dv = v[conn[:, 1]] - v[conn[:, 0]]
        eps_dot = np.einsum("nb,nb->n", dv, a) / L

    deps = eps_dot * dt if (dt is not None and dt > 0.0) else np.zeros(len(conn))

    sig = st["sig"]
    sig_old = sig.copy()
    c = np.zeros(group.n)
    slices = st.get("slices", [])
    for sl, mat, prop in slices:
        E = getattr(mat, "E", 0.0)
        rho0 = getattr(mat, "rho0", 0.0)
        # sound speed sqrt(E/rho) with the density guarded exactly as the
        # reference guards its own: hm_read_mat00.F computes
        # SDSP = SQRT(YOUNG/MAX(RHOR,EM20)).  A /MAT/VOID truss (LAW0) is
        # legally massless (RHO0 = 0 — see starter/checks._NULL_RHO0_OK_LAWS)
        # and would otherwise turn the 0/0 into a NaN time step; with E = 0
        # the guarded form gives c = 0, i.e. the element claims no time-step
        # limit of its own, which is the void semantics (M39 / M38-NEW-2).
        if E > 0.0 and rho0 > 0.0:
            c[sl] = np.sqrt(E / max(rho0, EM20))
        else:
            c[sl] = 0.0

        sig[sl] += E * deps[sl]                     # elastic trial (E = 0
        #                                             for VOID: no stress)
        if getattr(mat, "law", 1) == 2:
            # 1-D radial return on the Johnson-Cook curve
            p = getattr(mat, "params", {})
            epsp = st["epsp"][sl]
            e = np.maximum(epsp, 1e-20)
            rate_fac = 1.0
            if p.get("c", 0.0) > 0.0 and p.get("eps_dot_0", 0.0) > 0.0:
                r = np.maximum(np.abs(eps_dot[sl]) / p["eps_dot_0"], 1.0)
                rate_fac = 1.0 + p["c"] * np.log(r)
            A = p.get("A", 0.0)
            B = p.get("B", 0.0)
            n_exp = p.get("n", 0.0)
            sig_max = p.get("sig_max", 1e30)
            sy = np.minimum((A + B * e ** n_exp) * rate_fac, sig_max)
            H = B * n_exp * e ** (n_exp - 1.0) * rate_fac
            over = np.abs(sig[sl]) - sy
            plastic = over > 0.0
            dl = np.where(plastic, over / (E + np.maximum(H, 0.0)), 0.0)
            st["epsp"][sl] = epsp + dl
            sig[sl] = np.where(plastic, np.sign(sig[sl]) * (sy + H * dl),
                               sig[sl])

    alive = st.get("off", np.ones(len(conn), dtype=float)) > 0.0
    F = st["area"] * sig * alive
    # tension (sig>0) pulls node 1 toward node 2: this force is already
    # the "-internal" contribution (see elements package docstring).
    fvec = F[:, None] * a
    if fint is not None:
        np.add.at(fint, conn[:, 0], fvec)
        np.add.at(fint, conn[:, 1], -fvec)

    st["eint"] += np.where(alive, st["area"] * L * 0.5 * (sig_old + sig) * deps, 0.0)
    # dt = L/c.  A stiffness-free material (a /MAT/VOID truss, E = 0) has
    # c = 0 and claims NO time-step limit of its own — the same convention
    # the solid kernel documents for SSP = 0 (solid_hexa8._exact_dt_factor)
    # and the spring uses for k = 0.  Returned as EP30 rather than letting
    # the division produce a warned inf (M39 / M38-NEW-2).
    dt_crit = np.where(c > 0.0, L / np.maximum(c, EM20), EP30)
    return np.where(alive, dt_crit, EP30)


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
    if len(conn) == 0:
        return conn, np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)
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
    if group is None or group.n == 0 or len(group.conn) == 0:
        return np.zeros((0, 6, 6)), np.zeros((0, 6), dtype=np.int64)
    st = group.state
    conn, L, a = _axis(group, x)
    n = group.n
    k_ax = np.zeros(n)
    slices = st.get("slices", [])
    for sl, mat, prop in slices:
        law = getattr(mat, "law", 1)
        if law not in (0, 1, 2):
            raise NotImplementedError(
                f"the implicit truss tangent supports LAW0, LAW1 and LAW2; got "
                f"LAW{law} (see PORTING_GUIDE)")
        if law == 0:
            Emod = np.zeros(sl.stop - sl.start)
        else:
            Emod = np.full(sl.stop - sl.start, getattr(mat, "E", 0.0))
        if law == 2 and epsp_incr is not None:
            dl = epsp_incr[sl]
            plastic = dl > 0.0
            if np.any(plastic):
                # hardening slope at the END of the converged consistency
                # solve (see the note above); H = 0 where the sig_max cap
                # rules — matching the iterated implicit return exactly
                p = getattr(mat, "params", {})
                e = np.maximum(st["epsp"][sl], 1e-20)
                A = p.get("A", 0.0)
                B = p.get("B", 0.0)
                n_exp = p.get("n", 0.0)
                sig_max = p.get("sig_max", 1e30)
                sy = A + B * e ** n_exp
                H = np.maximum(B * n_exp * e ** (n_exp - 1.0), 0.0)
                H = np.where(sy > sig_max, 0.0, H)
                E = getattr(mat, "E", 0.0)
                Emod = np.where(plastic, E * H / (E + H), Emod)
        k_ax[sl] = Emod * st["area"][sl] / L[sl]
    kb = k_ax[:, None, None] * np.einsum("ni,nj->nij", a, a)
    return _blocks_to_element(kb), _edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) element stiffness (F/L)(I - a a^T) from
    the current axial stress state at geometry ``x`` (see the note above).
    Same shapes as ``tangent()``; identically zero at zero stress."""
    if group is None or group.n == 0 or len(group.conn) == 0:
        return np.zeros((0, 6, 6)), np.zeros((0, 6), dtype=np.int64)
    st = group.state
    conn, L, a = _axis(group, x)
    F_over_L = st["area"] * st["sig"] / L
    eye = np.eye(3)
    kb = F_over_L[:, None, None] * (eye[None, :, :]
                                    - np.einsum("ni,nj->nij", a, a))
    return _blocks_to_element(kb), _edofs(conn)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the lumped mass is ``starter/source/elements/truss/tmass3.F``
# (the /2 half-mass-to-each-node lumping this file's ``init_group`` returns and
# the explicit leapfrog / M10 implicit dynamics divide by). The CONSISTENT mass
# is the standard shape-function integral M = ∫_V ρ Nᵀ N dV; the open-source
# element ships only the lumped form (as the buckling eigensolver's card was
# thin — M9), so the consistent operator is ported here as a clean library
# capability for the M16 modal eigensolver, NEVER touching the lumped path.
#
# Theory (Cook, Malkus & Plesha "Concepts and Applications of FE Analysis",
# ch. 11; Przemieniecki "Theory of Matrix Structural Analysis" ch. 11). The
# 2-node bar interpolates displacement linearly, N1 = 1-ξ, N2 = ξ (ξ in
# [0,1]); the same linear field carries motion in every one of the three
# global directions, so the translational consistent mass is isotropic:
#
#     M = ρ A L0 / 6 * [[2 I3,  I3 ],      (∫₀¹ Ni Nj L dξ = L/6 [[2,1],[1,2]])
#                       [ I3, 2 I3 ]]
#
# with m = ρ A L0 the (constant) element mass this file already stores. Being
# ∝ I3 in each 2×2 nodal pair, it is FRAME-INVARIANT — no corotational
# rotation is needed (unlike the beam), and the mass is evaluated on the
# REFERENCE length L0 because mass is conserved (it does not scale with the
# deformed length). Partition of unity: each row sums to m/2 (the lumped
# nodal mass), so ½ vᵀ M v = ½ m |v|² is exact for a rigid translation v.

_M_BAR = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0     # ∫ Ni Nj dξ, unit length


def consistent_mass(group, x=None):
    """Consistent element mass ∫ρ Nᵀ N dV of the 2-node bar (see the note
    above): (m/6)[[2 I3, I3],[I3, 2 I3]] with m the stored element mass.

    Returns ``(me (n,6,6), edofs (n,6))`` in the implicit assembler's
    convention — translations only, same 6-DOF addressing as ``tangent()``.
    ``x`` is accepted for a uniform kernel signature but unused: the mass is
    built on the reference length (mass conservation) and is frame-invariant
    (isotropic per nodal block)."""
    if group is None or group.n == 0 or len(group.conn) == 0:
        return np.zeros((0, 6, 6)), np.zeros((0, 6), dtype=np.int64)
    st = group.state
    conn = group.conn
    n = group.n
    m = st["mass"]                                    # ρ A L0, per element
    me = np.zeros((n, 6, 6))
    for a in range(2):
        for b in range(2):
            f = m * _M_BAR[a, b]                       # (n,)
            for c in range(3):
                me[:, a * 3 + c, b * 3 + c] = f
    return me, _edofs(conn)


def static_internal_forces(group, x, u, ur, fint, mint):
    """Nodal force at configuration ``x`` from the current stress state —
    the updated-Lagrangian force assembly of the M9 implicit residual
    (stress already updated by a midpoint-geometry forces() call):
    F = A*sigma along the CURRENT axis. ``u``/``ur``/``mint`` unused.
    (Kept for callers like the buckling prestress path; the implicit
    drivers reach the truss through ``implicit_internal_forces`` below
    since M11.)"""
    if group is None or group.n == 0 or len(group.conn) == 0:
        return
    st = group.state
    conn, L, a = _axis(group, x)
    fvec = (st["area"] * st["sig"])[:, None] * a
    if fint is not None:
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
    if group is None or group.n == 0 or len(group.conn) == 0:
        return
    if u is None:
        u = np.zeros_like(x_ref)
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
    slices = st.get("slices", [])
    for sl, mat, prop in slices:
        E = getattr(mat, "E", 0.0)
        sig[sl] += E * deps[sl]                      # elastic trial
        if getattr(mat, "law", 1) == 2:
            # iterated 1-D consistency solve on the JC static curve
            # (rate term OFF under implicit — the M10 convention)
            p = getattr(mat, "params", {})
            ep0 = st["epsp"][sl]
            over = np.abs(sig[sl])
            A = p.get("A", 0.0)
            B = p.get("B", 0.0)
            n_exp = p.get("n", 0.0)
            sig_max = p.get("sig_max", 1e30)
            sy0 = np.minimum(A + B * np.maximum(ep0, 1e-20) ** n_exp, sig_max)
            plastic = over > sy0
            if np.any(plastic):
                dl = np.zeros(sl.stop - sl.start)
                for _ in range(_IMPL_NEWTON_ITERS):
                    e = np.maximum(ep0 + dl, 1e-20)
                    sy = A + B * e ** n_exp
                    H = B * n_exp * e ** (n_exp - 1.0)
                    capped = sy > sig_max
                    sy = np.where(capped, sig_max, sy)
                    H = np.where(capped, 0.0, np.maximum(H, 0.0))
                    res = over - E * dl - sy
                    dl += np.where(plastic, res / (E + H), 0.0)
                    dl = np.maximum(dl, 0.0)
                e = np.maximum(ep0 + dl, 1e-20)
                sy_new = np.minimum(A + B * e ** n_exp, sig_max)
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
    if fint is not None:
        np.add.at(fint, conn[:, 0], fvec)
        np.add.at(fint, conn[:, 1], -fvec)
    # trapezoidal internal-energy booking, the forces() formula
    st["eint"] += st["area"] * Lm * 0.5 * (sig_old + sig) * deps
