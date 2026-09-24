"""
Rigid walls (/RWALL) — kinematic treatment. Since M5: plane, sphere and
cylinder geometries, and MOVING walls (imposed motion or free with a
mass).

Fortran origin: ``engine/source/constraints/general/rwall/`` —
``rgwal0.F`` (driver & ITYP dispatch), ``rgwall.F`` (fixed/moving plane),
``rgwals.F`` (sphere), ``rgwalc.F`` (cylinder), ``rgwalp.F`` (parallelogram),
and the moving-wall treatment of ``rgwalt.F`` (wall tied to a node, impulses
reacting on it). Starter counterpart: ``starter/source/constraints/general/rwall/``
(``hm_read_rwall_plane.F``, ``hm_read_rwall_cyl.F``, ``hm_read_rwall_spher.F``,
``hm_read_rwall_paral.F``, ``read_rwall.F``).

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

from typing import Any, Dict, Optional, Tuple
import numpy as np

from ..common.constants import EM20
from ..model.model import Model
from .rwall_thermal import RwallThermal, RwallThermalParams


class RigidWalls:
    def __init__(self, model: Model, log):
        self.model = model
        self.walls = []
        self.thermal_walls: Dict[int, RwallThermal] = {}
        for rw in model.rwalls:
            if getattr(rw, "lagmul", False):
                continue
            thermal = getattr(rw, "thermal", None)
            if thermal is None and (getattr(rw, "is_therm", False) or hasattr(rw, "thermal_params")):
                p = getattr(rw, "thermal_params", rw)
                thermal = RwallThermal(p)
                rw.thermal = thermal
            if thermal is not None:
                self.thermal_walls[rw.id] = thermal
            if rw.grnod_id in (None, 0):
                idx = np.arange(model.numnod)
            else:
                grp = model.node_groups.get(rw.grnod_id)
                idx = grp.node_idx.copy() if grp is not None and grp.node_idx is not None else np.array([], dtype=int)
            if getattr(rw, "grnod_id2", None):
                g2 = model.node_groups.get(rw.grnod_id2)
                if g2 is not None and g2.node_idx is not None:
                    idx = np.setdiff1d(idx, g2.node_idx)
            # frozen (massless) nodes and the wall's own carrier node are
            # never wall candidates (a 1e30 mass would wreck the ledger)
            idx = idx[model.mass[idx] < 1e29]
            wnode = model._id2idx.get(rw.node_id, -1) if rw.node_id else -1
            if rw.node_id and wnode < 0:
                log.error(f"/RWALL/{rw.id}: moving wall carrier node {rw.node_id} not found in model", "RWALL INIT")
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
                        # For PLANE/CYL, normal holds M1. For PARAL, normal holds M1 and axis2 holds M2.
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
                                rw.axis1 = a1
                                rw.axis2 = a2
                                n = np.cross(a1, a2)
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
              mass: np.ndarray, dt: float,
              weight: Optional[np.ndarray] = None):
        """Correct velocities against every wall.

        weight : SPMD ``WEIGHT`` array (``None`` = serial).  Every domain
                 holding a candidate node applies the same correction
                 (fixed walls act on the local candidate subset, moving
                 walls are replicated with ALL their candidates and the
                 carrier), so only the energy bookings are weighted: the
                 per-node U terms by the node's weight and the carrier's
                 kinetic-energy term by the carrier's weight — each node
                 counted once over the domains (the ``WEIGHT(N)`` factor
                 of ``rgwal0.F``'s force/work accumulation).

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
            if weight is None:
                U = float((m * (
                    0.5 * np.einsum("nb,nb->n", v[i], v[i])
                    + 0.5 * np.einsum("nb,nb->n", v_old[i], v_old[i])
                    - np.einsum("nb,nb->n", v_trial, v_old[i]))).sum())
            else:
                U = float((weight[i] * m * (
                    0.5 * np.einsum("nb,nb->n", v[i], v[i])
                    + 0.5 * np.einsum("nb,nb->n", v_old[i], v_old[i])
                    - np.einsum("nb,nb->n", v_trial, v_old[i]))).sum())

            if wnode < 0:
                removed += -U                        # fixed wall (impact work absorbed into contact energy)
            elif driven:
                wext += U                            # imposed-motion wall
            else:
                # free wall: momentum-exact reaction (the FULL impulse,
                # incl. tied/friction tangential parts) on the carrier,
                # whose in-ledger KE change is then booked exactly
                J = (m[:, None] * (v[i] - v_trial)).sum(axis=0)
                v[wnode] = v_w - J / mass[wnode]
                if weight is None:
                    removed += -U - float(0.5 * mass[wnode]
                                          * (v[wnode] @ v[wnode]
                                             - v_w @ v_w))
                else:
                    removed += -U - float(0.5 * mass[wnode]
                                          * (v[wnode] @ v[wnode]
                                             - v_w @ v_w)) \
                        * float(weight[wnode])
        return removed, wext


class LagmulRWall:
    """Rigid wall enforced via global Lagrange multiplier solver (/RWALL/LAGMUL).

    Fortran origin: ``engine/source/tools/lagmul/lag_rwall.F`` (LAG_RWALL),
    called by ``engine/source/tools/lagmul/lag_mult.F``.
    Starter counterpart: ``starter/source/constraints/general/rwall/hm_read_rwall_lagmul.F``.

    Enforces kinematic constraint between secondary nodes and rigid wall
    (PLANE, CYL, SPHER) within the global sparse Lagrange multiplier system
    (L M^-1 L^T lambda = b).

    * Fixed wall (wnode < 0): rows on secondary nodes only.
    * Moving wall (wnode >= 0): rows on secondary nodes and carrier node with
      opposite signs (+n on secondary, -n on carrier), guaranteeing exact
      linear momentum conservation (sum L_row = 0.0).
    * Sliding wall (slide == 0): 1 constraint row per active node in the normal
      direction (n . v_rel = 0).
    * Tied wall (slide == 1): 3 orthogonal constraint rows per active node (v_rel = 0).
    """

    def __init__(self, rw: Any, model: Model, log: Any = None):
        self.rw = rw
        self.model = model
        self.log = log if log is not None else getattr(model, "log", None)
        self.geom = getattr(rw, "geom", "PLANE").upper()

        # Resolve carrier node (wnode >= 0 for moving wall, -1 for fixed wall)
        node_map = getattr(model, "_id2idx", {})
        if not isinstance(node_map, dict):
            node_map = getattr(model, "node_id_to_idx", {})
        if not isinstance(node_map, dict):
            node_map = {}
        node_id = getattr(rw, "node_id", 0)
        self.wnode = node_map.get(node_id, -1) if node_id else -1
        if node_id and self.wnode < 0 and self.log is not None:
            self.log.error(
                f"/RWALL/{getattr(rw, 'id', 0)}: moving wall carrier node {node_id} not found in model",
                "LAGMUL RWALL INIT"
            )

        # Base point and normal initialization
        x0 = getattr(model, "x0", None)
        if self.wnode >= 0 and x0 is not None and len(x0) > self.wnode:
            if hasattr(rw, "point") and (rw.point is None or np.isnan(rw.point).any()):
                rw.point = x0[self.wnode].copy()
                if self.geom in ("PLANE", "CYL"):
                    n = rw.normal - rw.point
                    nn = np.linalg.norm(n)
                    if nn > 1e-20:
                        rw.normal = n / nn

        # Normalize wall normal / axis
        if hasattr(rw, "normal") and rw.normal is not None:
            nn = np.linalg.norm(rw.normal)
            if nn > 1e-20:
                self.rw.normal = rw.normal / nn

        # Candidate secondary nodes
        cand = None
        for attr in ("secondary_nodes", "snode", "nodes", "candidate_nodes"):
            if hasattr(rw, attr) and getattr(rw, attr) is not None:
                cand = np.asarray(getattr(rw, attr), dtype=np.int64)
                break

        n_coords = len(x0) if x0 is not None else 0
        n_mass = len(model.mass) if hasattr(model, "mass") and model.mass is not None else 0
        n_nodes = len(getattr(model, "node_ids", []))
        n_tot = max(n_coords, n_mass, n_nodes, getattr(model, "numnod", 0))

        if cand is None:
            grnod_id = getattr(rw, "grnod_id", None)
            if grnod_id in (None, 0):
                cand = np.arange(n_tot, dtype=np.int64)
            else:
                grp = None
                if hasattr(model, "node_groups") and model.node_groups:
                    grp = model.node_groups.get(grnod_id)
                if grp is not None:
                    if getattr(grp, "node_idx", None) is not None:
                        cand = np.asarray(grp.node_idx, dtype=np.int64)
                    elif getattr(grp, "node_ids", None) and hasattr(model, "node_id_to_idx"):
                        cand = np.array([
                            model.node_id_to_idx[nid]
                            for nid in grp.node_ids
                            if nid in model.node_id_to_idx
                        ], dtype=np.int64)
                    elif getattr(grp, "nodes", None) is not None:
                        cand = np.asarray(grp.nodes, dtype=np.int64)
                if cand is None:
                    cand = np.zeros(0, dtype=np.int64)

        # Exclude nodes in grnod_id2
        grnod_id2 = getattr(rw, "grnod_id2", None)
        if grnod_id2:
            grp2 = model.node_groups.get(grnod_id2) if hasattr(model, "node_groups") else None
            if grp2 is not None and getattr(grp2, "node_idx", None) is not None:
                cand = np.setdiff1d(cand, grp2.node_idx)

        # Filter out massless/frozen nodes and carrier node
        if hasattr(model, "mass") and model.mass is not None and len(cand) > 0:
            valid_mass = cand < len(model.mass)
            cand = cand[valid_mass]
            cand = cand[model.mass[cand] < 1e29]

        if self.wnode >= 0 and len(cand) > 0:
            cand = cand[cand != self.wnode]

        # Search distance filtering (dist > 0)
        dist = getattr(rw, "dist", 0.0)
        if dist > 0.0 and len(cand) > 0 and x0 is not None:
            xw0 = x0[self.wnode] if self.wnode >= 0 else getattr(rw, "point", np.zeros(3))
            s0, _ = self._geometry_at(x0[cand], xw0)
            within = (s0 >= -EM20) & (s0 <= dist)
            cand = cand[within]

        self.cand = cand if cand is not None else np.zeros(0, dtype=np.int64)
        n_tot = getattr(model, "numnod", len(x0) if x0 is not None else 0)
        max_idx = max(n_tot, (int(np.max(self.cand)) + 1) if len(self.cand) > 0 else 0)
        self.is_tied = np.zeros(max_idx, dtype=bool)

        # If tied wall, initialize tied state
        if getattr(rw, "slide", 0) == 1:
            self.is_tied[self.cand] = True

        if self.log is not None:
            self.log.info(
                f"     /RWALL/{self.geom}/{getattr(rw, 'id', 0)} (LAGMUL): "
                f"{len(self.cand)} SECONDARY CANDIDATE NODE(S)"
            )

    def _geometry_at(self, x_cand: np.ndarray, x_wall: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute signed distance and unit normal for candidate nodes.

        x_cand: (N, 3)
        x_wall: (3,)
        Returns:
            s: (N,) signed distance (<= 0 when penetrating)
            norm: (N, 3) outward unit normal
        """
        x_cand = np.atleast_2d(x_cand)
        n_pts = len(x_cand)
        if n_pts == 0:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 3), dtype=np.float64)

        if self.geom == "PLANE":
            n_raw = getattr(self.rw, "normal", np.array([0.0, 0.0, 1.0]))
            nn = np.linalg.norm(n_raw)
            n_unit = n_raw / (nn if nn > 1e-20 else 1.0)
            disp = x_cand - x_wall
            s = np.einsum("ij,j->i", disp, n_unit)
            norm = np.tile(n_unit, (n_pts, 1))
            return s, norm

        elif self.geom == "SPHER":
            disp = x_cand - x_wall
            r = np.linalg.norm(disp, axis=1)
            r_safe = np.where(r > 1e-20, r, 1.0)
            norm = disp / r_safe[:, None]
            norm[r <= 1e-20] = np.array([0.0, 0.0, 1.0])
            s = r - getattr(self.rw, "radius", 0.0)
            return s, norm

        elif self.geom == "CYL":
            axis_raw = getattr(self.rw, "normal", np.array([0.0, 0.0, 1.0]))
            an = np.linalg.norm(axis_raw)
            a = axis_raw / (an if an > 1e-20 else 1.0)
            disp = x_cand - x_wall
            axial = np.einsum("ij,j->i", disp, a)
            disp_rad = disp - axial[:, None] * a[None, :]
            r = np.linalg.norm(disp_rad, axis=1)
            r_safe = np.where(r > 1e-20, r, 1.0)
            norm = disp_rad / r_safe[:, None]
            degen = r <= 1e-20
            if np.any(degen):
                perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                perp = perp - np.dot(perp, a) * a
                pn = np.linalg.norm(perp)
                norm[degen] = perp / (pn if pn > 1e-20 else 1.0)
            s = r - getattr(self.rw, "radius", 0.0)
            return s, norm

        else:
            # Fallback to PLANE
            n_raw = getattr(self.rw, "normal", np.array([0.0, 0.0, 1.0]))
            nn = np.linalg.norm(n_raw)
            n_unit = n_raw / (nn if nn > 1e-20 else 1.0)
            disp = x_cand - x_wall
            s = np.einsum("ij,j->i", disp, n_unit)
            norm = np.tile(n_unit, (n_pts, 1))
            return s, norm

    def generate_l_matrix(self, dt: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Generate constraint rows for the global sparse Lagrange multiplier solver.

        Returns (data, nodes, dofs, eq_ids, n_rows).
        For sliding walls (slide=0), 1 row per active node (n . v_rel = 0).
        For tied walls (slide=1), 3 rows per active node (v_rel = 0).
        For moving walls, carrier node entries have opposite signs, guaranteeing
        exact linear momentum conservation (sum L_row = 0.0).
        """
        empty_ret = (
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            0,
        )
        if len(self.cand) == 0:
            return empty_ret

        x = getattr(self.model, "x", getattr(self.model, "x0", None))
        if x is None:
            return empty_ret
        v = getattr(self.model, "v", None)
        if v is None or len(v) != len(x):
            v = np.zeros_like(x)

        if dt == 0.0 and hasattr(self.model, "dt"):
            dt = float(getattr(self.model, "dt", 0.0))
        dt2 = 0.5 * dt

        # Wall position and velocity
        if self.wnode >= 0 and self.wnode < len(v) and self.wnode < len(x):
            vw = v[self.wnode]
            xw = x[self.wnode] + vw * dt2
            xw0 = x[self.wnode]
        else:
            vw = np.zeros(3, dtype=np.float64)
            xw = getattr(self.rw, "point", np.zeros(3, dtype=np.float64))
            xw0 = xw

        cand = self.cand
        # Predicted and current node positions
        u = x[cand] + v[cand] * dt2
        dp, norm_pred = self._geometry_at(u, xw)
        dp0, norm_curr = self._geometry_at(x[cand], xw0)

        v_rel = v[cand] - vw
        vn_rel = np.einsum("ij,ij->i", v_rel, norm_curr)

        slide = getattr(self.rw, "slide", 0)
        if slide == 1:
            # Tied mode: nodes that were already tied stay tied;
            # candidate nodes penetrating become tied
            newly_hit = (dp <= 0.0) & ~((vn_rel > 0.0) & (dp0 > 0.0))
            self.is_tied[cand[newly_hit]] = True
            active_mask = self.is_tied[cand]
        else:
            # Sliding mode: active only when in contact (dp <= 0) and not separating from outside
            active_mask = (dp <= 0.0) & ~((vn_rel > 0.0) & (dp0 > 0.0))

        active_nodes = cand[active_mask]
        active_normals = norm_curr[active_mask]
        n_active = len(active_nodes)

        if n_active == 0:
            return empty_ret

        data = []
        nodes = []
        dofs = []
        eq_ids = []
        n_rows = 0

        if slide == 1:
            # Tied wall: 3 orthogonal constraint equations per active node (Vx, Vy, Vz)
            for k in range(n_active):
                sn = int(active_nodes[k])
                for dof in range(3):
                    eq_id = n_rows
                    n_rows += 1
                    data.append(1.0)
                    nodes.append(sn)
                    dofs.append(dof)
                    eq_ids.append(eq_id)
                    if self.wnode >= 0:
                        data.append(-1.0)
                        nodes.append(int(self.wnode))
                        dofs.append(dof)
                        eq_ids.append(eq_id)
        else:
            # Sliding wall: 1 normal constraint equation per active node
            for k in range(n_active):
                sn = int(active_nodes[k])
                nx, ny, nz = active_normals[k]
                eq_id = n_rows
                n_rows += 1
                # Secondary node (+n)
                data.extend([float(nx), float(ny), float(nz)])
                nodes.extend([sn, sn, sn])
                dofs.extend([0, 1, 2])
                eq_ids.extend([eq_id, eq_id, eq_id])
                # Carrier node (-n) for moving wall
                if self.wnode >= 0:
                    data.extend([-float(nx), -float(ny), -float(nz)])
                    nodes.extend([int(self.wnode), int(self.wnode), int(self.wnode)])
                    dofs.extend([0, 1, 2])
                    eq_ids.extend([eq_id, eq_id, eq_id])

        return (
            np.asarray(data, dtype=np.float64),
            np.asarray(nodes, dtype=np.int64),
            np.asarray(dofs, dtype=np.int64),
            np.asarray(eq_ids, dtype=np.int64),
            n_rows,
        )
