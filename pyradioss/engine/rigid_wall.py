"""
Rigid walls (/RWALL) — kinematic treatment. Since M5: plane, sphere and
cylinder geometries, and MOVING walls (imposed motion or free with a
mass).

Fortran origin: ``engine/source/constraints/general/rwall/`` —
``rgwal0.F`` (fixed plane), ``rgwals.F`` (sphere), ``rgwalc.F``
(cylinder), and the moving-wall treatment of ``rgwalt.F`` (wall tied to a
node, impulses reacting on it).

Kinematic wall mechanics
------------------------
Each geometry provides, for every candidate node, a signed distance s
(positive on the allowed side) and an outward unit normal n (constant for
the plane, radial for sphere/cylinder). Each cycle, AFTER the velocity
update but BEFORE the position update, a node whose end-of-step position
would be behind the wall — RELATIVE to the wall's own motion,

    s + (v - v_w).n * dt < 0

has its normal velocity replaced so that it lands exactly ON the (moved)
surface:

    v.n  <-  v_w.n - s/dt

* slide = 0 : tangential velocity kept (frictionless sliding)
* slide = 1 : tied — the node takes the wall's full velocity
* slide = 2 : Coulomb friction — the *relative* tangential velocity is
              reduced by the friction impulse |dv_t| <= mu |dv_n|

Moving walls
------------
``node_id`` ties the wall to a node: the geometry translates with the
node's displacement and pushes with its velocity v_w. An /IMPVEL-driven
carrier makes an imposed-motion wall: an external agent, like /IMPVEL
itself — the drive absorbs the reaction. A carrier with a mass makes a
FREE wall, and there the wall velocity and the node corrections must be
solved TOGETHER (the rgwalt treatment): the end-of-cycle wall velocity
v_w' and the landing condition of every hit node,

    v_i'.n_i = v_w'.n_i - s_i/dt          (land ON the wall that MOVED
                                           at v_w', not at the stale v_w)
    m_w v_w' = m_w v_w - sum m_i (v_i' - v_i).n_i n_i

form a 3x3 linear system for v_w' (the matrix is m_w E + sum m_i n_i
n_i^T). Solving it implicitly matters: correcting the nodes against the
PRE-recoil wall velocity and recoiling afterwards makes every riding node
track a one-cycle-stale, systematically faster wall — a feedback that
injects energy at a rate that does NOT vanish with dt (the M5 tests
measured +33% energy creation at /DT 0.2 with the explicit-lag variant —
found the hard way). With the joint solve, a single-node impact
reproduces the exact perfectly-inelastic collision, (m_w v_w - m u) /
(m_w + m), in one cycle. Tangential (tied/friction) impulses still react
explicitly on the carrier — their lag is one order smaller. The wall does
not rotate.

Energy accounting (the M4 lessons, wall edition)
------------------------------------------------
The booking is built on one per-node identity instead of a catalogue of
cases. Over a cycle, a wall-corrected node's kinetic energy changes by
(sampling always happens after the corrections):

    dKE = 1/2 m |v_new|^2 - 1/2 m |v_old|^2

Of this, the force phase (elements, loads) already booked its share into
the other ledgers at the node's carried-in velocity, w = f . v_old dt =
m (v_trial - v_old) . v_old.  Whatever remains was injected (or removed)
by the WALL:

    U = dKE - w = m [ |v_new|^2/2 + |v_old|^2/2 - v_trial . v_old ]

with v_trial the velocity the node reached before the correction. U is
computed with FULL vectors, so the normal landing, tied dragging and the
friction impulse are all captured by the same expression. The three wall
types then close the balance by construction:

* **fixed wall**:   CE += -U.  For an arrest (v_trial = v_old = -u,
  v_new ~ 0) this is the classic +1/2 m u^2 inelastic dissipation; for a
  RESTING node (v_old ~ 0) it vanishes — the re-acquired trial velocity
  m a dt is a phantom (the 2 v_trial.v_old cross term kills it), the /BCS
  argument of kinematics.py. Crucially it also catches the *chatter*
  cycles of a marginal-dt oscillator hammering the wall, where the node
  acquires a large velocity WITHIN the cycle and the wall re-launches it
  — booking against v_old alone (the M1 form) leaks exactly that energy,
  which the M5 moving-wall test found the hard way.
* **driven wall** (/IMPVEL on the carrier node): EW += U. The injection
  is real external work by the drive; an arrest by a driven wall books
  NEGATIVE external work — the external agent absorbs the dissipation.
* **free wall** (carrier node with a mass): the carrier's post-solve
  velocity is known exactly (see the joint solve above), so its
  in-ledger kinetic-energy change is booked exactly too:
  CE += -sum U - [ m_w |v_w'|^2/2 - m_w |v_w|^2/2 ].   The M5 impact
  test checks momentum conservation to round-off and the closed balance.
"""

from __future__ import annotations

import numpy as np

from ..common.constants import EM20
from ..model.model import Model


class RigidWalls:
    def __init__(self, model: Model, log):
        self.model = model
        self.walls = []
        for rw in model.rwalls:
            if rw.grnod_id in (None, 0):
                idx = np.arange(model.numnod)
            else:
                idx = model.node_groups[rw.grnod_id].node_idx
            if getattr(rw, "grnod_id2", None):
                g2 = model.node_groups.get(rw.grnod_id2)
                if g2 is not None and g2.node_idx is not None:
                    idx = np.setdiff1d(idx, g2.node_idx)
            # frozen (massless) nodes and the wall's own carrier node are
            # never wall candidates (a 1e30 mass would wreck the ledger)
            idx = idx[model.mass[idx] < 1e29]
            wnode = model.node_index(rw.node_id) if rw.node_id else -1
            if wnode >= 0:
                idx = idx[idx != wnode]
            # an /IMPVEL anywhere on the carrier node = driven wall: the
            # drive absorbs the reaction (see module docstring)
            driven = False
            if wnode >= 0:
                for imp in model.impvel:
                    g = model.node_groups.get(imp.grnod_id)
                    if g is not None and g.node_idx is not None and \
                            wnode in g.node_idx:
                        driven = True
                if not driven and model.mass[wnode] >= 1e29:
                    log.warning(f"/RWALL/{rw.id}: wall node has no mass "
                                f"and no /IMPVEL drive — give it /ADMAS "
                                f"(free wall) or /IMPVEL (driven wall); "
                                f"treated as driven at its (zero) "
                                f"velocity", "RWALL INIT")
                    driven = True
            if wnode >= 0:
                if np.isnan(rw.point).any():
                    rw.point = model.x0[wnode].copy()
                    if rw.geom in ("PLANE", "CYL", "PARAL"):
                        # For PLANE/CYL, normal holds M1. For PARAL, normal holds M1 and axis2 holds M2 (we will fix parser to store M1 in normal and M2 in axis2).
                        if rw.geom == "PARAL":
                            m1 = rw.normal
                            m2 = rw.axis2
                            a1 = m1 - rw.point
                            a2 = m2 - rw.point
                            nn1 = np.linalg.norm(a1)
                            nn2 = np.linalg.norm(a2)
                            if nn1 < 1e-20 or nn2 < 1e-20:
                                log.error(f"/RWALL/{rw.id}: moving wall node coincides with M1 or M2", "RWALL INIT")
                            else:
                                rw.axis1 = a1 / nn1
                                rw.axis2 = a2 / nn2
                                n = np.cross(rw.axis1, rw.axis2)
                                nn = np.linalg.norm(n)
                                if nn < 1e-20:
                                    log.error(f"/RWALL/{rw.id}: M, M1 and M2 are collinear", "RWALL INIT")
                                else:
                                    rw.normal = n / nn
                        else:
                            n = rw.normal - rw.point
                            nn = np.linalg.norm(n)
                            if nn < 1e-20:
                                log.error(f"/RWALL/{rw.id}: moving wall node "
                                          f"and M1 coincide (zero normal)",
                                          "RWALL INIT")
                            else:
                                rw.normal = n / nn
            self.walls.append((rw, idx, wnode, driven,
                               model.x0[wnode].copy() if wnode >= 0
                               else None))
            log.info(f"     /RWALL/{rw.geom}/{rw.id}: "
                     f"{len(idx)} CANDIDATE NODE(S)"
                     + (f", MOVING (NODE {rw.node_id}"
                        + (", DRIVEN)" if driven else ")")
                        if wnode >= 0 else ""))

    # ------------------------------------------------------------------
    @staticmethod
    def _geometry(rw, point: np.ndarray, xc: np.ndarray):
        """Signed distance + outward normal of every candidate position
        ``xc`` for wall ``rw`` whose reference point (plane point, sphere
        center, axis point) currently sits at ``point``.

        Returns (s, n) with n of shape (n, 3) — the plane's constant
        normal is broadcast, sphere/cylinder normals are radial (the
        'land exactly on the wall' correction is then first-order in the
        curvature, like the original)."""
        if rw.geom == "SPHER":
            r = xc - point
            d = np.linalg.norm(r, axis=1)
            n = r / np.maximum(d, EM20)[:, None]
            if rw.radius < 0.0:
                return -rw.radius - d, -n
            return d - rw.radius, n
        if rw.geom == "CYL":
            a = rw.normal                     # unit axis direction
            r = xc - point
            r = r - (r @ a)[:, None] * a[None, :]   # radial component
            d = np.linalg.norm(r, axis=1)
            n = r / np.maximum(d, EM20)[:, None]
            if rw.radius < 0.0:
                return -rw.radius - d, -n
            return d - rw.radius, n
        if rw.geom == "PARAL":
            r = xc - point
            s = r @ rw.normal
            p_proj = r - s[:, None] * rw.normal[None, :]
            e1 = rw.axis1
            e2 = rw.axis2
            dot11 = e1 @ e1
            dot22 = e2 @ e2
            dot12 = e1 @ e2
            det = dot11 * dot22 - dot12 * dot12
            v_dot_1 = p_proj @ e1
            v_dot_2 = p_proj @ e2
            a = (v_dot_1 * dot22 - v_dot_2 * dot12) / det
            b = (v_dot_2 * dot11 - v_dot_1 * dot12) / det
            s = np.where((a >= 0.0) & (a <= 1.0) & (b >= 0.0) & (b <= 1.0), s, np.inf)
            n = np.broadcast_to(rw.normal, (len(xc), 3))
            return s, n
        # PLANE
        s = (xc - point) @ rw.normal
        n = np.broadcast_to(rw.normal, (len(xc), 3))
        return s, n

    # ------------------------------------------------------------------
    def apply(self, x: np.ndarray, v: np.ndarray, v_old: np.ndarray,
              mass: np.ndarray, dt: float):
        """Correct velocities against every wall.

        v      : velocities after this cycle's acceleration + kinematic
                 updates (modified in place)
        v_old  : velocities the nodes ENTERED the cycle with (for the
                 dissipation bookkeeping, see module docstring)

        Returns (dissipated_energy, external_work): the dissipation goes
        to the contact-energy counter, the work (imposed-motion walls
        only) to the external-work counter."""
        if not self.walls or dt <= 0.0:
            return 0.0, 0.0
        removed = 0.0
        wext = 0.0
        for rw, idx, wnode, driven, x0w in self.walls:
            # current wall position/velocity (moving walls translate with
            # their carrier node; fixed walls have v_w = 0)
            if wnode >= 0:
                point = rw.point + (x[wnode] - x0w)
                v_w = v[wnode].copy()     # copy: v[wnode] is mutated below
            else:
                point = rw.point
                v_w = np.zeros(3)

            s, n = self._geometry(rw, point, x[idx])
            vn = np.einsum("nb,nb->n", v[idx], n)
            vwn = n @ v_w
            hit = s + (vn - vwn) * dt < 0.0          # ends behind the wall
            if rw.dist > 0.0:
                hit &= (s <= rw.dist)                # ends behind the wall, AND started within the search band
            if not np.any(hit):
                continue
            i = idx[hit]
            nh = n[hit] if n.ndim == 2 else n
            m = mass[i]
            v_trial = v[i].copy()                    # pre-correction
            free = wnode >= 0 and not driven

            # ---- free wall: implicit joint momentum solve (module doc):
            # the end-of-cycle wall velocity and the landing corrections
            # are one linear system — correcting against the stale v_w
            # and recoiling afterwards pumps energy (found the hard way)
            if free:
                m_w = mass[wnode]
                A = m_w * np.eye(3) + np.einsum("n,nb,nc->bc", m, nh, nh)
                b = m_w * v_w + (m[:, None] * nh
                                 * (vn[hit] + s[hit] / dt)[:, None]).sum(0)
                v_w_new = np.linalg.solve(A, b)
            else:
                v_w_new = v_w

            # normal correction: land exactly ON the surface that moves
            # at the wall's END-OF-CYCLE velocity
            vn_new_rel = -s[hit] / dt                # relative normal v
            dvn = (nh @ v_w_new + vn_new_rel) - vn[hit]
            v[i] += dvn[:, None] * nh
            if rw.slide == 1:                        # tied: ride the wall
                v[i] = v_w_new + vn_new_rel[:, None] * nh
            elif rw.slide == 2 and rw.fric > 0.0:    # Coulomb friction
                vrel = v[i] - v_w_new
                vt = vrel - np.einsum("nb,nb->n", vrel, nh)[:, None] * nh
                vt_mag = np.linalg.norm(vt, axis=1)
                dv_fric = np.minimum(vt_mag, rw.fric * np.abs(dvn))
                scale = 1.0 - dv_fric / np.maximum(vt_mag, EM20)
                v[i] = v_w_new \
                    + np.einsum("nb,nb->n", vrel, nh)[:, None] * nh \
                    + vt * scale[:, None]

            # ---- energy injected by the wall this cycle (module doc):
            # U = dKE - f.v_old dt, per node, full vectors — covers the
            # normal landing, the tied drag and the friction impulse
            U = float((m * (
                0.5 * np.einsum("nb,nb->n", v[i], v[i])
                + 0.5 * np.einsum("nb,nb->n", v_old[i], v_old[i])
                - np.einsum("nb,nb->n", v_trial, v_old[i]))).sum())

            if wnode < 0:
                wext += U                            # fixed wall (does negative work, goes to EW)
            elif driven:
                wext += U                            # imposed-motion wall
            else:
                # free wall: momentum-exact reaction (the FULL impulse,
                # incl. tied/friction tangential parts) on the carrier,
                # whose in-ledger KE change is then booked exactly
                J = (m[:, None] * (v[i] - v_trial)).sum(axis=0)
                v[wnode] = v_w - J / mass[wnode]
                removed += -U - float(0.5 * mass[wnode]
                                      * (v[wnode] @ v[wnode]
                                         - v_w @ v_w))
        return removed, wext
