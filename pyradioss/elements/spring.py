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
from . import spring_general

#: Spring property TYPE numbers whose /PROP card carries a mass that THIS
#: PORT actually reads AND whose Starter reader genuinely REQUIRES mass > 0.
#: The check below is a real requirement for an ACTIVE spring — a TYPE4
#: /SPRING with no mass has no stable time step of its own — but it must
#: mirror what the Fortran Starter actually enforces, card for card:
#:
#: * TYPE4 (/PROP/SPRING) — hand reader in starter_keywords.read_prop; the
#:   Fortran /PROP/TYPE4 reader/RINI4 needs a positive mass for the spring's
#:   own explicit step, so the port checks it.
#:
#: TYPE32 (/PROP/SPR_PRE) is NOT in this set (M40): the Fortran Starter does
#: NOT enforce MASS > 0 for the pretensioner.  ``hm_read_prop32.F`` reads the
#: card mass with ``HM_GET_FLOATV('MASS',AMAS,...)`` (a BLANK field gives
#: AMAS = 0), stores it with ``SET_U_GEO(1,AMAS)``, and its only checks are
#: MSGID 408 (F1/D1/E1/STIF1 force-curve over-specification) and MSGID 406
#: (zero spring LENGTH XL) — there is no mass check.  The ``MASS > 0`` in
#: ``prop_p32_spr_pre.cfg``'s CHECK block is a HyperMesh-GUI validation, not
#: a Starter one.  Verified by running the real ``starter_win64.exe`` on
#: RD-HWX-T-1010 cantilever_completed (whose SPR_PRE/2 card has a BLANK mass,
#: Stif0 = 13744.468): 0 ERRORS, the listing printing
#: ``MASS. . . = 0.000000000000`` — accepted.  So a blank/zero pretensioner
#: mass is legal Starter data; the port mirrors that and does not error.
#: (parse_spr_pre still READS the field so a real mass — e.g. RD-V-0031's
#: 1E-5 — feeds the nodal mass; the pretensioner physics is unported, so the
#: Engine refuses the group by :func:`prop_reader.refuse_inactive_properties`
#: — the honest ERROR->SKIPS the check must not pre-empt.  M39-BUG-SPRPRE.)
#:
#: TYPE8/TYPE13 are excluded because :mod:`spring_general` owns their mass.
#: Every OTHER spring spelling (SPR_PUL 12, SPR_MAT 23, SPR_AXI 25,
#: SPR_TAB 26, NSTRAND 28, KJOINT 33/45, SPR_CRUS 44, SPR_MUSCLE 46 ...)
#: parses to an InactiveProperty whose ``mass`` is the PLACEHOLDER 0.0 of
#: ``prop_reader._universal_geo_params`` — not a value off the card.  Mass-
#: checking that placeholder reports a deck error that does not exist, and
#: names the wrong card while doing it; the Engine already refuses those
#: groups (``prop_reader.refuse_inactive_properties``) and the Starter
#: already warns (checks.check_model's PROP CHECK), which is the honest
#: pair of messages.  Add a type here only together with a reader that fills
#: its mass AND a Fortran Starter that actually requires it.
_MASS_REQUIRED_SPRING_TYPES = frozenset({4})

#: /PROP spelling per TYPE for the mass message (the card the user wrote)
_SPRING_PROP_SPELLING = {4: "SPRING", 32: "SPR_PRE"}


def init_group(group, model, log):
    """Element buffer + lumped mass/inertia.  A /SPRING group may mix the
    axial TYPE4 spring (this module) with the 6-DOF TYPE8/TYPE13 general
    springs (:mod:`spring_general`), one property per part slice — each
    element is classified by its slice's property type and the two paths
    write into the SAME per-(element,node) mass/inertia return."""
    st = group.state
    n = group.n
    xe = model.x0[group.conn]
    L0 = norm3(xe[:, 1] - xe[:, 0])
    mass = np.zeros(n)
    k = np.zeros(n)
    cdamp = np.zeros(n)
    kind = np.full(n, 4, dtype=np.int64)
    for sl, mat, prop in st["slices"]:
        pt = getattr(prop, "type", 4)
        kind[sl] = pt
        if pt in spring_general.SPRING_PROP_TYPES:
            continue                       # 6-DOF: built by spring_general
        mass[sl] = prop.params["mass"]
        k[sl] = prop.params["k"]
        cdamp[sl] = prop.params["c"]
    is6 = np.isin(kind, list(spring_general.SPRING_PROP_TYPES))
    idx4 = np.where(~is6)[0]
    idx6 = np.where(is6)[0]

    # A spring with no mass has no stable time step of its own — but only
    # the property types whose mass this port actually READS may be checked
    # for it (see _MASS_REQUIRED_SPRING_TYPES; M39 / M38-NEW-1).  The
    # message names the property the user actually wrote, not TYPE4's card.
    for sl, mat, prop in st["slices"]:
        pt = getattr(prop, "type", 4)
        if pt not in _MASS_REQUIRED_SPRING_TYPES:
            continue
        bad = np.zeros(n, dtype=bool)
        bad[sl] = mass[sl] <= 0.0
        if not bad.any():
            continue
        pn = getattr(prop, "prop_name", None) \
            or _SPRING_PROP_SPELLING.get(pt, f"TYPE{pt}")
        for eid in group.ids[bad]:
            log.error(f"/SPRING {eid}: /PROP/{pn}/{prop.id} mass must be "
                      f"> 0 (needed for the explicit time step)",
                      "SPRING INIT")

    st.update(L0=L0, mass=mass, k=k, cdamp=cdamp,
              force=np.zeros(n), eint=np.zeros(n), ehour=np.zeros(n),
              idx4=idx4, idx6=idx6)

    massn = np.repeat(mass / 2.0, 2)       # per (elem, localnode)
    inertn = np.zeros(2 * n)
    if len(idx6):
        spring_general.init6(group, model, log, idx6, massn, inertn)
    node_idx = group.conn.reshape(-1)
    return node_idx, massn, (inertn if inertn.any() else None)


def _forces_axial(group, x, v, dt, fint, idx):
    """Axial TYPE4 spring forces for the elements ``idx`` (``slice(None)``
    for the whole group — the original vectorized path)."""
    st = group.state
    conn = group.conn[idx]
    dx = x[conn[:, 1]] - x[conn[:, 0]]
    L = np.maximum(norm3(dx), EM20)
    a = dx / L[:, None]
    Ldot = np.einsum("nb,nb->n",
                     v[conn[:, 1]] - v[conn[:, 0]], a)

    F_old = st["force"][idx].copy()
    F = st["k"][idx] * (L - st["L0"][idx]) + st["cdamp"][idx] * Ldot
    st["force"][idx] = F

    fvec = F[:, None] * a          # tension pulls the nodes together
    np.add.at(fint, conn[:, 0], fvec)
    np.add.at(fint, conn[:, 1], -fvec)

    # elastic part of the work goes to internal energy; damping work too
    # (the original books spring damping into internal energy as well).
    st["eint"][idx] += 0.5 * (F_old + F) * Ldot * dt

    k = np.maximum(st["k"][idx], EM20)
    omega = 2.0 * np.sqrt(k / st["mass"][idx])
    xi = st["cdamp"][idx] / np.sqrt(k * st["mass"][idx])
    dt_crit = (2.0 / omega) * (np.sqrt(1.0 + xi ** 2) - xi)
    return np.where(st["k"][idx] > 0, dt_crit, EP30)


def forces(group, x, v, vr, dt, fint, mint):
    st = group.state
    idx6 = st.get("idx6")
    if idx6 is None or len(idx6) == 0:
        # pure axial group (every existing spring deck): the whole-group
        # vectorized path, bit-identical to the pre-M38 kernel
        return _forces_axial(group, x, v, dt, fint, slice(None))
    dtc = np.full(group.n, EP30)
    idx4 = st["idx4"]
    if len(idx4):
        dtc[idx4] = _forces_axial(group, x, v, dt, fint, idx4)
    dtc[idx6] = spring_general.forces6(group, x, v, vr, dt, fint, mint, idx6)
    return dtc


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
