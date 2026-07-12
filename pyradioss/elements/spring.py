"""
2-node spring element (/SPRING + /PROP/SPRING, TYPE4).

Fortran origin: ``engine/source/elements/spring/`` (rforc3.F, TYPE4 branch).

The TYPE4 spring acts along the line joining its two nodes:

    F = K * (L - L0) + C * L_dot

with the property's mass M lumped half to each node (a spring with M = 0
would have no stable time step of its own — the Starter enforces M > 0).

Critical time step of the two-mass oscillator (masses M/2, stiffness K),
including the damping reduction used by the original:

    omega = 2 sqrt(K / M),  xi = C / sqrt(K M)
    dt    = (2/omega) * (sqrt(1 + xi^2) - xi)
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3


def init_group(group, model, log):
    xe = model.x0[group.conn]
    L0 = norm3(xe[:, 1] - xe[:, 0])
    n = group.n
    mass = np.zeros(n)
    k = np.zeros(n)
    cdamp = np.zeros(n)
    for sl, mat, prop in group.state["slices"]:
        mass[sl] = prop.params["mass"]
        k[sl] = prop.params["k"]
        cdamp[sl] = prop.params["c"]
    if np.any(mass <= 0):
        for eid in group.ids[mass <= 0]:
            log.error(f"/SPRING {eid}: /PROP/SPRING mass must be > 0 "
                      f"(needed for the explicit time step)", "SPRING INIT")
    group.state.update(
        L0=L0, mass=mass, k=k, cdamp=cdamp,
        force=np.zeros(n), eint=np.zeros(n), ehour=np.zeros(n),
    )
    node_idx = group.conn.reshape(-1)
    return node_idx, np.repeat(mass / 2.0, 2), None


def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    a = dx / L[:, None]
    Ldot = np.einsum("nb,nb->n",
                     v[conn[:, 1]] - v[conn[:, 0]], a)

    F_old = st["force"].copy()
    F = st["k"] * (L - st["L0"]) + st["cdamp"] * Ldot
    st["force"] = F

    fvec = F[:, None] * a          # tension pulls the nodes together
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)

    # elastic part of the work goes to internal energy; damping work too
    # (the original books spring damping into internal energy as well).
    st["eint"] += 0.5 * (F_old + F) * Ldot * dt

    k = np.maximum(st["k"], EM20)
    omega = 2.0 * np.sqrt(k / st["mass"])
    xi = st["cdamp"] / np.sqrt(k * st["mass"])
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    return np.where(st["k"] > 0, dt_crit, EP30)


# ----------------------------------------------------------------------------
# Implicit tangent stiffness + residual (M11) — alongside forces()
# ----------------------------------------------------------------------------
# Fortran origin: the element-KE branch of the implicit assembly
# (``engine/source/implicit/imp_glob_k.F``, spring stiffness) + its
# ``imp_kgeo`` geometric path.
#
# The TYPE4 spring is a TOTAL-form element: forces() rebuilds F = k(L - L0)
# + c*Ldot from the geometry every cycle instead of integrating a rate on
# committed state. That breaks the implicit drivers' pseudo-velocity trick
# (which relies on rate-form kernels accumulating C*grad(u) on the FROZEN
# committed frame): at the frozen geometry the elastic term k(L - L0) never
# feels the trial displacement at all, and the DAMPING term c*(a . du)
# would answer instead — a rate device masquerading as a stiffness. So the
# spring supplies its own implicit residual, ``implicit_internal_forces``,
# which the drivers call INSTEAD of forces() (both geometry modes):
#
# * linear geometry (M8 frozen frame): F = F_committed + k (a_ref . du_rel)
#   along the committed axis a_ref — the total form linearized at the
#   committed geometry, ACCUMULATED on the committed force state exactly
#   like the rate-form kernels accumulate stress (the driver feeds only
#   the CURRENT increment's displacement; the previous increments live in
#   the state, which the driver restores to the committed values before
#   every residual evaluation). Its derivative is EXACTLY the material
#   tangent k a a^T, so a linear step converges in one Newton iteration
#   (the M8 contract);
# * nonlinear geometry (M9 updated-Lagrangian): F = k (L(x_end) - L0) along
#   the CURRENT axis — the total form is exact at any configuration (no
#   midpoint objectivity step is needed: the elastic force is a state
#   function of the end geometry), and its exact derivative is the material
#   tangent PLUS the (F/L)(I - a a^T) geometric term of ``kgeo``.
#
# The DASHPOT (c > 0) is a rate device: like the LAW2 strain-rate term and
# the bulk viscosity (M10 convention), it is DISABLED under implicit with
# an explicit warning — never silently fed the pseudo-velocity du/1
# (deferred, PORTING_GUIDE M11). The stored force/energy state stays the
# state function F^2/2k of the elastic spring.

def _spring_axis(group, x):
    conn = group.conn
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    return conn, L, dx / L[:, None]


def _spring_edofs(conn):
    edofs = np.empty((len(conn), 6), dtype=np.int64)
    for c in range(3):
        edofs[:, c] = conn[:, 0] * 6 + c
        edofs[:, 3 + c] = conn[:, 1] * 6 + c
    return edofs


def _blocks(kb):
    """(n,3,3) relative block -> (n,6,6) element [[kb,-kb],[-kb,kb]]."""
    n = len(kb)
    ke = np.empty((n, 6, 6))
    ke[:, :3, :3] = kb
    ke[:, 3:, 3:] = kb
    ke[:, :3, 3:] = -kb
    ke[:, 3:, :3] = -kb
    return ke


def tangent(group, x, epsp_incr=None):
    """Material element tangent k a a^T along the current axis at geometry
    ``x``. Returns (ke (n,6,6), edofs (n,6)). ``epsp_incr`` unused (the
    spring is elastic; the dashpot is disabled under implicit)."""
    st = group.state
    conn, L, a = _spring_axis(group, x)
    kb = st["k"][:, None, None] * np.einsum("ni,nj->nij", a, a)
    return _blocks(kb), _spring_edofs(conn)


def kgeo(group, x):
    """Geometric (initial-stress) stiffness (F/L)(I - a a^T) from the
    current spring force at geometry ``x`` — the same taut-string operator
    as the truss. Identically zero at zero force."""
    st = group.state
    conn, L, a = _spring_axis(group, x)
    F_over_L = st["force"] / L
    eye = np.eye(3)
    kb = F_over_L[:, None, None] * (eye[None, :, :]
                                    - np.einsum("ni,nj->nij", a, a))
    return _blocks(kb), _spring_edofs(conn)


# ----------------------------------------------------------------------------
# Consistent (element) mass — M16, alongside the lumped mass of init_group.
# ----------------------------------------------------------------------------
# Fortran origin: the TYPE4 spring's mass is the /PROP/SPRING scalar mass M
# lumped half to each node (``rmass3.F`` / this file's ``init_group`` /2
# return). Unlike the truss/beam/solid, the spring carries NO distributed
# density field — its mass is a DISCRETE point property, not ∫ρ dV — so the
# "consistent" mass and the lumped mass COINCIDE exactly: M/2 as a point mass
# on each node's three translations, a diagonal 6×6. There is no shape-
# function integral to do (the spring has no interior), which is why this is
# the EXACT mass, not an approximation. Reported here so the modal assembler
# has a uniform ``consistent_mass()`` on every family; the value is identical
# to what the lumped path already puts on these DOFs, so a spring never shifts
# the consistent-vs-lumped spectrum.

def consistent_mass(group, x=None):
    """Exact element mass of the TYPE4 spring: the point mass M/2 on each
    node's translations (diagonal 6×6) — identical to the lumped mass (see
    the note above; the spring has no distributed density to integrate).

    Returns ``(me (n,6,6), edofs (n,6))``. ``x`` unused (a point mass is
    frame-invariant and configuration-independent)."""
    st = group.state
    conn = group.conn
    n = group.n
    half = st["mass"] / 2.0                            # M/2 per node
    me = np.zeros((n, 6, 6))
    for i in range(6):                                 # 3 trans on each node
        me[:, i, i] = half
    return me, _spring_edofs(conn)


# ----------------------------------------------------------------------------
# Viscous DAMPING matrix — M18, alongside the tangent / consistent mass.
# ----------------------------------------------------------------------------
# Fortran origin: the /PROP/SPRING dashpot ``c`` term of the TYPE4 force law
# F = K(L - L0) + C*L_dot (rforc3.F). Under the DIRECT explicit / implicit
# time march the dashpot is a rate device (it reads the nodal velocity, so it
# is disabled in the implicit Newton residual — see the note above and
# PORTING_GUIDE M11). M18 revives it as a genuine assembled DAMPING operator:
# the viscous force F_c = c*L_dot along the axis is EXACTLY the linear map
# f = C_e u_dot with the element damping matrix
#
#     C_e = c * [[ a a^T, -a a^T], [-a a^T, a a^T]]
#
# (the same relative-block structure as the elastic tangent k a a^T, with c in
# place of k — the dashpot resists the RATE of axial stretch just as the
# spring resists the axial stretch itself). A dashpot on one spring among many
# is the canonical source of NON-CLASSICAL damping: C is then NOT proportional
# to M or K, so the damped modes go complex (implicit/complex_modal.py). This
# operator never enters the M8-M17 residual/tangent paths — it feeds only the
# opt-in complex-eigenvalue / complex-mode-superposition solve.

def damping_matrix(group, x):
    """Viscous element damping matrix c a a^T along the current axis at
    geometry ``x`` — the /PROP/SPRING dashpot ``c`` term as a linear operator
    on the nodal velocities (see the note above). Returns ``(ce (n,6,6),
    edofs (n,6))``, the same shape/addressing as ``tangent()`` so the global
    C assembles through exactly the same COO->CSR scatter as K and M. Springs
    with c = 0 contribute a zero block (harmless)."""
    st = group.state
    conn, L, a = _spring_axis(group, x)
    cb = st["cdamp"][:, None, None] * np.einsum("ni,nj->nij", a, a)
    return _blocks(cb), _spring_edofs(conn)


def implicit_internal_forces(group, x_ref, u, ur, fint, mint, nlgeom):
    """The spring's own implicit residual (called by the drivers INSTEAD of
    ``forces()`` — see the note above): elastic total-form force at the
    trial configuration, linearized at the committed frame under linear
    geometry. Updates the force/energy state in place (trial values; the
    drivers' snapshot/commit machinery handles rollback exactly as for the
    rate-form kernels). ``ur``/``mint`` unused."""
    st = group.state
    if nlgeom:
        # end configuration: the total form is exact there (x_ref advances
        # per increment under the updated-Lagrangian outer step, so
        # x_ref + u IS the trial configuration)
        conn, L, a = _spring_axis(group, x_ref + u)
        F = st["k"] * (L - st["L0"])
    else:
        # frozen committed frame (x_ref stays at x0 for the whole run and
        # u is THIS increment only): accumulate the linearized force on
        # the committed state — see the note above
        conn, L, a = _spring_axis(group, x_ref)
        du_rel = u[conn[:, 1]] - u[conn[:, 0]]
        F = st["force"] + st["k"] * np.einsum("nb,nb->n", du_rel, a)
    st["force"][...] = F
    # elastic state function (the dashpot is off): eint = F^2 / 2k
    st["eint"][...] = np.where(st["k"] > 0.0,
                               F * F / (2.0 * np.maximum(st["k"], EM20)),
                               0.0)
    fvec = F[:, None] * a
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)
