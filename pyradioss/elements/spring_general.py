"""
General 6-DOF springs — /PROP/TYPE8 SPR_GENE and /PROP/TYPE13 SPR_BEAM.

Fortran origin
--------------
* engine : ``engine/source/elements/spring/rforc3.F`` dispatches IGTYP==8
  through ``r2coor3`` (frame) + ``r2def3.F`` (the 6 uncoupled DOF force
  laws) and IGTYP==13 through the r13* kernels; both accumulate a force
  per DOF ``F_i = K_i d_i (+ C_i d_i_dot)`` from the relative motion of
  the two nodes resolved in a local skew frame (``EXX = SKEW(1,ISK)`` …).
* starter: ``starter/source/properties/spring/hm_read_prop08.F`` /
  ``hm_read_prop13.F`` (the K_i/C_i and the per-DOF function slots).

Ported physics (the LINEAR stiffness/damping core)
--------------------------------------------------
Six uncoupled degrees of freedom in a local orthonormal frame
(e1, e2, e3) — three translations, three rotations::

    delta_i = ((x2 - x1) - (X2 - X1)) . e_i          i = 1,2,3  (stretch)
    theta_i = theta_i + ((wr2 - wr1) . e_i) * dt     i = 4,5,6  (twist, accum)
    F_i = K_i delta_i + C_i (v2 - v1) . e_i
    M_i = K_i theta_i + C_i (wr2 - wr1) . e_i

assembled to the nodes with no moment arm (a zero-/short-length discrete
spring): ``f2 = -sum F_i e_i``, ``f1 = +...`` and likewise the moments —
the same relative-block structure as the axial TYPE4 spring, one per DOF.
Translations use the TOTAL form (from the node positions); rotations are
RATE-integrated (the port carries no accumulated nodal-rotation vector),
exactly as r2def3 accumulates ``RX = RXOLD + …``.

Local frame
-----------
* TYPE8 (SPR_GENE): the fixed skew frame ``SKEW(:,skew_ID)``.  The port
  has no /SKEW reader, so ``skew_ID = 0`` uses the GLOBAL frame (the
  reference's skew 0) and a nonzero ``skew_ID`` warns and also falls back
  to global — a documented cut.
* TYPE13 (SPR_BEAM): the element frame — e1 along N1->N2 when the nodes
  are distinct (a default in-plane e2/e3 completes it), else the global
  frame.  It is FROZEN at init: the co-rotational beam frame update
  (r13's per-cycle reorientation) and the moment-arm coupling of a long
  spring-beam are NOT ported (documented cut) — correct for the small
  relative rotations the corpus decks exercise.

Documented cuts (per DOF, beyond the linear K/C core): the force
FUNCTIONS (fct_ID1i…), the hardening flags (A/B/D coeffs, Hi/IECROU), the
DeltaMin/DeltaMax rupture limits, the strain-rate smoothing (ISRATE/Fcut),
sensor activation, TYPE13's Ileng length normalisation and its extra
viscous cards.  Parsed and stored on the property, not acted on here; a
spring whose only nonzero data is K_i/C_i (the common corpus case) is
reproduced exactly.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20, EP30
from ..common.fastmath import norm3

#: property TYPE numbers handled by this module (6-DOF springs)
SPRING_PROP_TYPES = frozenset({8, 13})


def _slice_frame(prop, xe, log):
    """Local orthonormal frame (e1, e2, e3) for the elements of one slice,
    each returned as an (m, 3) array.  See the module docstring."""
    m = len(xe)
    ptype = getattr(prop, "type", 8)
    skew_id = int(prop.params.get("skew_id", 0) or 0)
    if ptype == 13 and skew_id == 0:
        # element frame: e1 along the element, default perpendicular e2/e3
        d = xe[:, 1] - xe[:, 0]
        L = norm3(d)
        e1 = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
        good = L > EM20
        e1[good] = d[good] / L[good, None]
        # e2 = e1 x (least-aligned global axis), e3 = e1 x e2
        ax = np.tile(np.array([0.0, 0.0, 1.0]), (m, 1))
        near_z = np.abs(e1[:, 2]) > 0.9
        ax[near_z] = np.array([1.0, 0.0, 0.0])
        e2 = np.cross(ax, e1)
        e2 /= np.maximum(norm3(e2), EM20)[:, None]
        e3 = np.cross(e1, e2)
        return e1, e2, e3
    if skew_id != 0 and log is not None:
        log.warning(f"/PROP/TYPE{ptype}/{getattr(prop, 'id', '?')}: "
                    f"skew_ID={skew_id} not ported (no /SKEW reader) — "
                    f"the local frame falls back to global", "SPRING INIT")
    # global frame (TYPE8 skew 0, or the fallback)
    e1 = np.tile(np.array([1.0, 0.0, 0.0]), (m, 1))
    e2 = np.tile(np.array([0.0, 1.0, 0.0]), (m, 1))
    e3 = np.tile(np.array([0.0, 0.0, 1.0]), (m, 1))
    return e1, e2, e3


def init6(group, model, log, idx6, massn, inertn):
    """Build the 6-DOF spring state for the ``idx6`` elements of a spring
    group and add their half/half nodal mass + rotational inertia into the
    caller's ``massn`` / ``inertn`` arrays (laid out like
    ``group.conn.reshape(-1)``)."""
    st = group.state
    m6 = len(idx6)
    conn6 = group.conn[idx6]
    xe = model.x0[conn6]                       # (m6, 2, 3)
    e1 = np.zeros((m6, 3))
    e2 = np.zeros((m6, 3))
    e3 = np.zeros((m6, 3))
    k6 = np.zeros((m6, 6))
    c6 = np.zeros((m6, 6))
    mass = np.zeros(m6)
    inertia = np.zeros(m6)

    # position of each idx6 element within group order -> its slice params
    pos = {int(e): j for j, e in enumerate(idx6)}
    for sl, mat, prop in st["slices"]:
        if getattr(prop, "type", 4) not in SPRING_PROP_TYPES:
            continue
        rng = np.arange(group.n)[sl]
        local = np.array([pos[int(e)] for e in rng if int(e) in pos],
                         dtype=np.int64)
        if not len(local):
            continue
        p = prop.params
        se1, se2, se3 = _slice_frame(prop, xe[local], log)
        e1[local], e2[local], e3[local] = se1, se2, se3
        for i in range(6):
            k6[local, i] = float(p.get(f"k{i + 1}", 0.0))
            c6[local, i] = float(p.get(f"c{i + 1}", 0.0))
        mass[local] = float(p.get("mass", 0.0))
        inertia[local] = float(p.get("inertia", 0.0))

    L0 = np.stack([
        np.einsum("mb,mb->m", xe[:, 1] - xe[:, 0], e1),
        np.einsum("mb,mb->m", xe[:, 1] - xe[:, 0], e2),
        np.einsum("mb,mb->m", xe[:, 1] - xe[:, 0], e3),
    ], axis=1)                                 # initial (x2-x1) . e_i

    st["gen6"] = dict(
        idx=np.asarray(idx6, dtype=np.int64), conn=conn6,
        e1=e1, e2=e2, e3=e3, k6=k6, c6=c6, mass=mass, inertia=inertia,
        L0=L0, theta=np.zeros((m6, 3)),
        force=np.zeros((m6, 3)), moment=np.zeros((m6, 3)),
        eint=np.zeros(m6),
    )
    # half/half lumped mass + inertia into the caller's per-(elem,node)
    # arrays: node_idx = conn.reshape(-1) => slot 2*e (node0), 2*e+1 (node1)
    for j, e in enumerate(idx6):
        massn[2 * e] += mass[j] / 2.0
        massn[2 * e + 1] += mass[j] / 2.0
        inertn[2 * e] += inertia[j] / 2.0
        inertn[2 * e + 1] += inertia[j] / 2.0


def forces6(group, x, v, vr, dt, fint, mint, idx6):
    """6-DOF spring forces for the ``idx6`` elements; scatters translation
    forces into ``fint`` and moments into ``mint`` and returns their
    per-element critical time step (aligned to ``idx6``)."""
    g = group.state["gen6"]
    conn = g["conn"]
    n1, n2 = conn[:, 0], conn[:, 1]
    e1, e2, e3 = g["e1"], g["e2"], g["e3"]

    # relative translation (total form) and its rate, per local axis
    d = (x[n2] - x[n1])
    dv = v[n2] - v[n1]
    delta = np.stack([np.einsum("mb,mb->m", d, e1) - g["L0"][:, 0],
                      np.einsum("mb,mb->m", d, e2) - g["L0"][:, 1],
                      np.einsum("mb,mb->m", d, e3) - g["L0"][:, 2]], axis=1)
    dvl = np.stack([np.einsum("mb,mb->m", dv, e1),
                    np.einsum("mb,mb->m", dv, e2),
                    np.einsum("mb,mb->m", dv, e3)], axis=1)
    F = g["k6"][:, :3] * delta + g["c6"][:, :3] * dvl
    g["force"] = F

    # relative rotation (rate-integrated) and its rate, per local axis
    dw = vr[n2] - vr[n1]
    dwl = np.stack([np.einsum("mb,mb->m", dw, e1),
                    np.einsum("mb,mb->m", dw, e2),
                    np.einsum("mb,mb->m", dw, e3)], axis=1)
    g["theta"] += dwl * dt
    M = g["k6"][:, 3:] * g["theta"] + g["c6"][:, 3:] * dwl
    g["moment"] = M

    # assemble to the nodes (no moment arm): global force / moment vectors
    fvec = F[:, 0:1] * e1 + F[:, 1:2] * e2 + F[:, 2:3] * e3
    mvec = M[:, 0:1] * e1 + M[:, 1:2] * e2 + M[:, 2:3] * e3
    np.add.at(fint, n1, fvec)
    np.add.at(fint, n2, -fvec)
    np.add.at(mint, n1, mvec)
    np.add.at(mint, n2, -mvec)

    # internal energy (elastic + damping work, like the axial spring)
    g["eint"] += np.einsum("mi,mi->m", F, dvl) * dt \
        + np.einsum("mi,mi->m", M, dwl) * dt

    # critical time step: min over the active DOF of the two-mass
    # oscillator, the same omega = 2 sqrt(k/m) => dt = 2/omega bound the
    # axial TYPE4 spring uses (masses m/2 on each node).  A DOF with no
    # stiffness (or a rotation with no inertia) imposes no limit.
    mass = np.maximum(g["mass"], EM20)
    inertia = np.maximum(g["inertia"], EM20)
    kt = g["k6"][:, :3]
    kr = g["k6"][:, 3:]
    dt_tr = np.where(kt > 0.0,
                     1.0 / np.sqrt(kt / mass[:, None] + EM20), EP30)
    has_I = g["inertia"] > 0.0
    dt_rot = np.where((kr > 0.0) & has_I[:, None],
                      1.0 / np.sqrt(kr / inertia[:, None] + EM20), EP30)
    return np.minimum(dt_tr.min(axis=1), dt_rot.min(axis=1))
