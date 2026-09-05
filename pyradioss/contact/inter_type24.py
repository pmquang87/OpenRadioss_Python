"""
/INTER/TYPE24 — penalty node-to-surface contact, full options (M4).

Fortran origin: ``engine/source/interfaces/`` — the cycle path is

    intsort/i7main_tri.F   candidate search bookkeeping (re-sort when needed)
    intsort/i24buce.F       bucket (voxel) search: nodes near segments
    int07/i24dst3.F         exact node-segment distance/projection
    int07/i24for3.F         penalty force + friction, scatter to nodes
    inter3d1/i24sti3.F      (Starter) per-segment/node stiffness and gap
                           — ported in pyradioss/contact/stiffness.py

Algorithm
---------
1. **Broad phase** (every ``refresh`` cycles) — a voxel (bucket) sort, the
   port of i24buce replacing M1's all-pairs bounding-box test:

   * every segment's bounding box, inflated by the gap plus a travel
     margin, is binned into a uniform grid whose cell size equals the
     LARGEST inflated box extent — so a box overlaps at most 2 cells per
     axis (8 cells total) and the binning stays fully vectorized;
   * every tracked secondary node is binned into its single cell;
   * candidate pairs = (node, segment) sharing a cell, deduplicated,
     minus pairs where the node is a corner of the segment (the classic
     self-exclusion, which is also what makes *self-impact* work: a
     surface's own nodes are tested against all its segments except the
     ones they belong to).

   The travel margin is sized from the current maximum nodal speed so no
   pair can penetrate undetected between refreshes — the same safety
   criterion as the original's bucket re-sort test.

2. **Narrow phase** (every cycle): each candidate quad segment is split
   into two triangles; the closest point on each triangle to the node is
   found exactly (barycentric clamping). Penetration is

       p = gap_pair - d,     d = |x_node - x_closest|

   where ``gap_pair`` is constant (Igap=0) or the sum of the two sides'
   physical gaps (Igap=1: half shell thicknesses — see stiffness.py).

3. **Penalty force** (i24for3): if p > 0, a spring pushes the node back
   along the closest-point direction, plus a small viscous damper on the
   normal relative velocity (stabilizes the bounce):

       Fn = K_pair * p + C * vn_rel,    C = 2 * visc * sqrt(K_pair * m_node)

   ``K_pair`` combines the main-segment and secondary-node element
   stiffness according to Istf (see stiffness.combine_stiffness). The
   reaction is distributed to the segment's corner nodes with the
   barycentric weights of the closest point (equal and opposite total
   force — momentum is conserved exactly).

4. **Coulomb friction**: the tangential relative velocity direction gets
   a force min(mu * Fn, ...) opposing it (kinetic friction, smoothly
   regularized near zero slip velocity). Since M15, ``Ifric > 0``
   replaces the constant mu with the MFROT friction MODELS mu(p, v) of
   i24for3.F — pressure p = Fn over the CURRENT main-segment area, slip
   speed v — and ``Ifiltr`` low-pass filters the tangential force with
   the per-pair IFQ exponential moving average (both live in
   ``contact/friction.py``, where the laws, the reader mapping and the
   one documented deviation are derived against the fetched source).
   The Ifric = 0 code path is untouched — bit-identical to M4 (asserted
   by the M15 tests). The mu(p, v) evaluation and the filter sit
   DOWNSTREAM of the numba-mirrored narrow phase (t24_narrow), in the
   NumPy force path both backends share — the M7 kernel parity contract
   holds with no mirror change (stated, not silent: the mirror boundary
   is the narrow phase, see the accel package docstring).
   The IFQ anchor store is engine-side state rebuilt EMPTY on a restart
   chain (like the broad-phase candidates); a chained IFQ run
   re-converges its filter within ~1/alpha cycles — the M6 restart
   bit-match contract is guaranteed for Ifiltr = 0 decks.

5. **Element deletion (M3<->M4)**: segments whose parent element has
   GBUF%OFF = 0 are masked out every cycle (crack faces stop pushing),
   and secondary nodes all of whose elements died stop being tracked at
   the next broad phase (see contact/tracking.py for the physics).

Time step: every penalty spring adds a frequency to the nodes it loads;
the stability bound of the node-on-spring system dt = sqrt(2 m / K) is
evaluated at init for the worst (mass, pair-stiffness) combination on
both sides and fed to the Engine's global minimum every cycle, exactly
like the original's interface time step.

Port simplifications vs the full TYPE24 (documented, roadmap M5+): no
Inacti initial-penetration treatments (small initial penetrations just
push back out), constant linear spring instead of the original's
stiffening K*p/(gap-p) near-crossing guard, no Tstart/Tstop or sensors,
Igap 2/3 (mesh-size-scaled gaps) not ported.
"""

from __future__ import annotations

import numpy as np

from ..accel import get as accel_get
from ..common.constants import EM20
from ..common.fastmath import cross3, norm3, scatter_add3
from ..model.model import Model
from . import friction, tracking
from .stiffness import (combine_stiffness, node_stiffness_gap,
                        segment_stiffness_gap, _segment_areas)

_VISC = 0.05  # normal damping ratio (Radioss VIS_S default 5%)


def _closest_point_on_triangle(p, a, b, c):
    """Vectorized exact closest point on triangles (Ericson, 'Real-Time
    Collision Detection' §5.1.5). All args (n,3). Returns (point, wa, wb, wc)
    barycentric weights."""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.einsum("nk,nk->n", ab, ap)
    d2 = np.einsum("nk,nk->n", ac, ap)
    bp = p - b
    d3 = np.einsum("nk,nk->n", ab, bp)
    d4 = np.einsum("nk,nk->n", ac, bp)
    cp = p - c
    d5 = np.einsum("nk,nk->n", ab, cp)
    d6 = np.einsum("nk,nk->n", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    # start from the inside-face solution and overwrite region by region
    denom = np.maximum(va + vb + vc, EM20)
    v = vb / denom
    w = vc / denom

    # edge AB region
    onAB = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.maximum(d1 - d3, EM20)
    v = np.where(onAB, t, v)
    w = np.where(onAB, 0.0, w)
    # edge AC region
    onAC = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.maximum(d2 - d6, EM20)
    v = np.where(onAC, 0.0, v)
    w = np.where(onAC, t, w)
    # edge BC region
    onBC = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    t = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), EM20)
    v = np.where(onBC, 1.0 - t, v)
    w = np.where(onBC, t, w)
    # vertex regions
    atA = (d1 <= 0) & (d2 <= 0)
    v = np.where(atA, 0.0, v)
    w = np.where(atA, 0.0, w)
    atB = (d3 >= 0) & (d4 <= d3)
    v = np.where(atB, 1.0, v)
    w = np.where(atB, 0.0, w)
    atC = (d6 >= 0) & (d5 <= d6)
    v = np.where(atC, 0.0, v)
    w = np.where(atC, 1.0, w)

    u = 1.0 - v - w
    point = u[:, None] * a + v[:, None] * b + w[:, None] * c
    return point, u, v, w


def _narrow(x, ni, seg):
    """Narrow phase (i24dst3): exact closest point of each candidate node
    on its quad segment, the quad split into triangles (0,1,2) and
    (0,2,3). Returns (best_d, best_pt, best_w) with ``best_w`` the corner
    weights of the closest point. Mirrored by
    accel.jit_kernels.t24_narrow (M7 — this was the hottest single block
    of the self-impact examples; see the accel package docstring)."""
    best_d = np.full(len(ni), np.inf)
    best_pt = np.zeros((len(ni), 3))
    best_w = np.zeros((len(ni), 4))
    p = x[ni]
    for cols in ((0, 1, 2), (0, 2, 3)):
        a, b, c = (x[seg[:, cols[0]]], x[seg[:, cols[1]]],
                   x[seg[:, cols[2]]])
        pt, u, vv, w = _closest_point_on_triangle(p, a, b, c)
        d = norm3(p - pt)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_pt[better] = pt[better]
        wq = np.zeros((len(ni), 4))
        wq[:, cols[0]], wq[:, cols[1]], wq[:, cols[2]] = u, vv, w
        best_w[better] = wq[better]
    return best_d, best_pt, best_w


def _expand_matches(keys_a: np.ndarray, keys_b: np.ndarray):
    """All (i, j) index pairs with keys_a[i] == keys_b[j], fully
    vectorized (the join step of the voxel sort). Returns (ii, jj)."""
    order = np.argsort(keys_b, kind="stable")
    sorted_b = keys_b[order]
    left = np.searchsorted(sorted_b, keys_a, "left")
    right = np.searchsorted(sorted_b, keys_a, "right")
    counts = right - left
    total = int(counts.sum())
    if total == 0:
        return (np.zeros(0, dtype=np.int64),) * 2
    ii = np.repeat(np.arange(len(keys_a)), counts)
    # position of each match inside its own [left, right) run
    run_starts = np.cumsum(counts) - counts
    offs = np.arange(total) - np.repeat(run_starts, counts)
    jj = order[np.repeat(left, counts) + offs]
    return ii, jj


class ContactType24:
    """One /INTER/TYPE24 interface, engine-side."""

    def __init__(self, itf, model: Model, log):
        self.itf = itf
        self.model = model

        surf = model.surfaces[itf.surf_id]
        self.segs = surf.segments
        if self.segs is None or len(self.segs) == 0:
            log.warning(f"/INTER/TYPE24/{itf.id}: main surface is empty — "
                        f"interface inactive", "CONTACT INIT")
            self.segs = np.zeros((0, 4), dtype=np.int64)
            self.seg_gtype = np.zeros(0, dtype="<U8")
            self.seg_elem = np.zeros(0, dtype=np.int64)
        else:
            self.seg_gtype = surf.seg_gtype
            self.seg_elem = surf.seg_elem

        # --- secondary nodes: group, or the surface's own nodes ----------
        # (grnod_ID = 0 -> self-impact, the Radioss single-surface input)
        if itf.grnod_id == 0:
            self.nodes = np.unique(self.segs)
            log.info(f"     /INTER/TYPE24/{itf.id}: SELF-IMPACT — "
                     f"{len(self.nodes)} SECONDARY NODES FROM THE MAIN "
                     f"SURFACE")
        else:
            # kept SORTED: the cycle maps global node index -> position in
            # this array with searchsorted (np.unique output is sorted)
            self.nodes = np.sort(model.node_groups[itf.grnod_id].node_idx)

        # --- penalty stiffness (i24sti3) -----------------------------------
        # Element-based stiffness on both sides; Istf picks the combination
        # (for Istf=1, stfac IS the stiffness and the element values are
        # only kept for the gap computation -> scale 1).
        scale = itf.stfac if itf.istf != 1 else 1.0
        Km, gm = segment_stiffness_gap(model, self.segs, self.seg_gtype,
                                       self.seg_elem, scale)
        Ks_all, gs_all = node_stiffness_gap(model, scale)
        self.Km = Km
        self.Ks = Ks_all[self.nodes] if len(self.nodes) else np.zeros(0)

        # --- contact gap ---------------------------------------------------
        # lc = mean segment size, the reference length for the defaults
        area = _segment_areas(model.x0, self.segs)
        lc = float(np.sqrt(area.mean())) if len(area) else 1.0
        # the default MINIMUM gap: the physical half-thickness where the
        # main side is shells, a small fraction of the segment size
        # otherwise (a zero gap would make contact undetectable)
        gap_floor = itf.gap if itf.gap > 0 else (
            float(gm.mean()) if len(gm) and gm.max() > 0 else 0.02 * lc)
        if itf.igap == 1:
            # variable gap: g = g_s(node) + g_m(segment), clipped
            self.gap_m = gm
            self.gap_s = gs_all[self.nodes] if len(self.nodes) else \
                np.zeros(0)
            self.gap_min = gap_floor
            self.gap_max = itf.gap_max if itf.gap_max > 0 else np.inf
            gap_hi = (self.gap_s.max() if len(self.gap_s) else 0.0) + \
                (gm.max() if len(gm) else 0.0)
            self.gap_bound = float(np.clip(gap_hi, self.gap_min,
                                           self.gap_max))
        else:
            self.gap_const = gap_floor
            self.gap_bound = gap_floor
        self.fric = itf.fric

        # --- friction MODELS + IFQ filter state (M15) -----------------------
        # mfrot/ifq = 0 leaves every M4 path untouched (bit-identical).
        # The filter anchors are the CAND_F store of i24for3.F, keyed
        # node*nseg + segrow, sorted; rebuilt from the active pairs each
        # cycle (a separated pair restarts its history — the IFPEN scope).
        self.mfrot = int(getattr(itf, "mfrot", 0))
        self.ifq = int(getattr(itf, "ifq", 0))
        self.xfiltr = float(getattr(itf, "xfiltr", 0.0))
        self.fric_c = np.asarray(getattr(itf, "fric_c",
                                         (0.0,) * 6), dtype=float)
        self._filt_keys = np.zeros(0, dtype=np.int64)
        self._filt_vals = np.zeros((0, 3))
        if self.mfrot > 0:
            log.info(f"     /INTER/TYPE24/{itf.id}: FRICTION MODEL "
                     f"MFROT={self.mfrot} (i24for3.F mu(p, v)), "
                     f"IFQ={self.ifq}")

        # --- interface time step bound (see module docstring) --------------
        # Worst node-on-spring combination on each side, evaluated once
        # (masses and stiffness are constant; deletion only REMOVES
        # springs). The PHYSICAL pre-mass-scaling masses keep the bound
        # conservative and restart-invariant (M6).
        self.dt_bound = self._compute_dt_bound(
            getattr(model, "mass0", model.mass))

        # --- deletion bookkeeping (M3<->M4) --------------------------------
        # Fortran chkstfn3.F:1352 gates deletion on IDEL >= 1
        self.idel = int(getattr(itf, 'idel', 0))
        self.deletable = (self.idel >= 1 and
                          tracking.any_deletable(model, self.seg_gtype))
        if self.deletable:
            self.ref_total = tracking.node_reference_counts(
                model, alive_only=False)
        self.seg_alive = np.ones(len(self.segs), dtype=bool)
        self.nodes_tracked = self.nodes

        # broad-phase bookkeeping
        self.pairs_node = np.zeros(0, dtype=np.int64)  # global node index
        self.pairs_seg = np.zeros(0, dtype=np.int64)   # segment row
        self._last_refresh = -10**9
        self.refresh = 20                              # cycles

    # ------------------------------------------------------------------
    def _compute_dt_bound(self, mass) -> float:
        """dt <= sqrt(2 m / K) for every node a contact spring can load:
        each secondary node with the stiffest segment it could meet, each
        main corner node with its own segments against the stiffest
        secondary node."""
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return np.inf
        itf = self.itf
        Km_max = np.full(len(self.nodes), self.Km.max())
        K_sec = combine_stiffness(itf.istf, itf.stfac, Km_max, self.Ks)
        dt_sec = np.sqrt(2.0 * mass[self.nodes]
                         / np.maximum(K_sec, EM20)).min()
        Ks_max = np.full(len(self.segs),
                         self.Ks.max() if len(self.Ks) else 0.0)
        K_main = combine_stiffness(itf.istf, itf.stfac, self.Km, Ks_max)
        m_corner = mass[self.segs].min(axis=1)
        dt_main = np.sqrt(2.0 * m_corner / np.maximum(K_main, EM20)).min()
        return float(min(dt_sec, dt_main))

    # ------------------------------------------------------------------
    def _broad_phase(self, x: np.ndarray, v: np.ndarray, dt: float):
        """Voxel candidate search (i24buce port — see module docstring)."""
        segs = self.segs[self.seg_alive]
        seg_rows = np.where(self.seg_alive)[0]
        nodes = self.nodes_tracked
        if len(segs) == 0 or len(nodes) == 0:
            self.pairs_node = np.zeros(0, dtype=np.int64)
            self.pairs_seg = np.zeros(0, dtype=np.int64)
            return
        margin = self.gap_bound + 2.0 * self.refresh * dt * \
            (np.abs(v).max() if len(v) else 0.0)

        xs = x[segs]                                    # (nseg, 4, 3)
        lo = xs.min(axis=1) - margin
        hi = xs.max(axis=1) + margin
        # cell size = the largest inflated box extent -> every box spans
        # at most 2 cells per axis, so the box binning below (8 offsets)
        # is exhaustive
        h = max(float((hi - lo).max()), EM20)
        origin = np.minimum(lo.min(axis=0), x[nodes].min(axis=0))

        ilo = np.floor((lo - origin) / h).astype(np.int64)
        ihi = np.floor((hi - origin) / h).astype(np.int64)
        inode = np.floor((x[nodes] - origin) / h).astype(np.int64)

        # single-integer cell keys (grid dims from the node+box extents)
        dims = np.maximum(np.maximum(ihi.max(axis=0), inode.max(axis=0)),
                          0) + 2

        def key(ijk):
            return (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]

        # bin every segment box into its <= 8 covered cells
        seg_ids, seg_keys = [], []
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ijk = ilo + np.array([dx, dy, dz])
                    inside = np.all(ijk <= ihi, axis=1)
                    seg_ids.append(np.where(inside)[0])
                    seg_keys.append(key(ijk[inside]))
        seg_ids = np.concatenate(seg_ids)
        seg_keys = np.concatenate(seg_keys)

        ii, jj = _expand_matches(key(inode), seg_keys)
        ni = nodes[ii]
        sj = seg_rows[seg_ids[jj]]
        if len(ni):
            # deduplicate (a box can meet a node in several cells)
            pair_key = ni * len(self.segs) + sj
            _, first = np.unique(pair_key, return_index=True)
            ni, sj = ni[first], sj[first]
            # a secondary node that is a corner of the segment is never a
            # candidate against it (self-exclusion; enables self-impact)
            keep = np.ones(len(ni), dtype=bool)
            for k in range(4):
                keep &= self.segs[sj, k] != ni
            ni, sj = ni[keep], sj[keep]
        self.pairs_node = ni
        self.pairs_seg = sj

    # ------------------------------------------------------------------
    def forces(self, x, v, mass, dt, fcont, cycle, stifn=None):
        """Penalty forces for one cycle, scattered into ``fcont``.

        Returns (contact_work_increment, dt_interface). The work increment
        is the interface's own estimate at the pre-update velocities
        (-F.vrel dt); the Engine books the exact leapfrog-consistent value
        from the assembled ``fcont`` at the midstep velocity — see the
        'contact energy booking' block of engine.py for why.

        ``stifn`` (M6, /DT/NODA): when given, the NEAR-candidate spring
        stiffness accumulated per node (the same sums that feed this
        interface's own dt) is also added into this global nodal-stiffness
        array, so the nodal time step / mass scaling sees the contact
        springs exactly like the element stiffness.
        """
        if len(self.segs) == 0 or len(self.nodes) == 0:
            return 0.0, np.inf

        # ---- deletion bookkeeping (cheap gathers; see tracking.py) -------
        if self.deletable:
            self.seg_alive = tracking.alive_segment_mask(
                self.model, self.seg_gtype, self.seg_elem)

        if cycle - self._last_refresh >= self.refresh:
            if self.deletable:
                mask = tracking.tracked_node_mask(self.model, self.ref_total)
                self.nodes_tracked = self.nodes[mask[self.nodes]]
            self._broad_phase(x, v, dt)
            self._last_refresh = cycle
        if len(self.pairs_node) == 0:
            return 0.0, self.dt_bound

        # drop pairs whose segment died since the last refresh
        live = self.seg_alive[self.pairs_seg]
        ni = self.pairs_node[live]                       # global node idx
        srow = self.pairs_seg[live]
        if len(ni) == 0:
            return 0.0, self.dt_bound
        seg = self.segs[srow]                            # (np, 4)

        # ---- narrow phase: exact closest point (i24dst3) -------------------
        # (dispatched to the numba mirror when that backend is active)
        jit = accel_get("t24_narrow")
        if jit is not None:
            best_d, best_pt, best_w = jit(x, ni, seg)
        else:
            best_d, best_pt, best_w = _narrow(x, ni, seg)

        # ---- per-pair gap (Igap) ------------------------------------------
        loc = np.searchsorted(self.nodes, ni)    # nodes is sorted (init)
        if self.itf.igap == 1:
            gap = self.gap_s[loc] + self.gap_m[srow]
            if self.gap_min > 0.0:
                gap = np.maximum(gap, self.gap_min)
            if self.gap_max < np.inf:
                gap = np.minimum(gap, self.gap_max)
        else:
            gap = np.full(len(ni), self.gap_const)

        vmax = float(np.abs(v).max()) if len(v) else 0.0
        near = best_d < gap + np.maximum(gap, 2.0 * vmax * dt)
        if not np.any(near):
            return 0.0, self.dt_bound
        ni = ni[near]
        seg = seg[near]
        srow = srow[near]
        loc = loc[near]
        gap = gap[near]
        best_d = best_d[near]
        best_pt = best_pt[near]
        best_w = best_w[near]

        K = combine_stiffness(self.itf.istf, self.itf.stfac,
                              self.Km[srow], self.Ks[loc])
                              
        # per-node spring-stiffness sums:
        # full K on the secondary node, full K on each corner (weight <= 1)
        n_nod = len(fcont)
        Knode = np.bincount(ni, weights=K, minlength=n_nod)
        Knode += np.bincount(seg.reshape(-1), weights=np.repeat(K, 4),
                             minlength=n_nod)
        loaded = Knode > 0.0
        dt_int = min(self.dt_bound, float(
            np.sqrt(2.0 * mass[loaded] / Knode[loaded]).min())) if np.any(loaded) else self.dt_bound
        if stifn is not None:                    # /DT/NODA accumulation
            stifn[loaded] += Knode[loaded]
        
        pen = gap - best_d
        active = pen > 0.0
        if not np.any(active):
            return 0.0, dt_int

        ni = ni[active]
        seg = seg[active]
        K = K[active]                            # per-pair stiffness (Istf)
        gap = gap[active]
        pen = pen[active]
        d = np.maximum(best_d[active], EM20)
        nvec = (x[ni] - best_pt[active]) / d[:, None]    # push-out direction
        wseg = best_w[active]

        # relative velocity node vs interpolated segment point
        vseg = np.einsum("nk,nkb->nb", wseg, v[seg])
        vrel = v[ni] - vseg
        vn = np.einsum("nb,nb->n", vrel, nvec)

        # normal force: spring + damper (only damp approaching motion)
        C = 2.0 * _VISC * np.sqrt(K * mass[ni])
        Fn = K * pen - C * np.minimum(vn, 0.0)
        Fvec = Fn[:, None] * nvec

        # Coulomb friction, regularized around zero slip. Ifric > 0 (M15)
        # swaps the constant mu for the MFROT mu(p, v) laws of i24for3.F
        # (contact/friction.py) and Ifiltr low-pass filters the tangential
        # force (the CAND_F exponential moving average). The mu = const,
        # no-filter path below is the M4 code verbatim (x + (-a) == x - a
        # exactly in IEEE — bit-identical, asserted by the M15 tests).
        if self.fric > 0.0 or self.mfrot > 0:
            gap_ref = float(np.mean(gap))
            vt = vrel - vn[:, None] * nvec
            vt_mag = norm3(vt)
            if self.mfrot > 0:
                # contact pressure p = Fn / (CURRENT main-segment area),
                # the i24for3 AREA = 1/2 |(x3-x1) x (x4-x2)| — its FNI by
                # this point includes the damper term, exactly like Fn
                d13 = x[seg[:, 2]] - x[seg[:, 0]]
                d24 = x[seg[:, 3]] - x[seg[:, 1]]
                area = 0.5 * norm3(cross3(d13, d24))
                pres = Fn / np.maximum(area, EM20)
                mu = friction.mu_kinetic(self.mfrot, self.fric,
                                         self.fric_c, pres, vt_mag)
            else:
                mu = self.fric
            Ft = mu * Fn * vt_mag / (
                vt_mag + 1e-3 * gap_ref / max(dt, EM20))
            ftvec = -(Ft / np.maximum(vt_mag, EM20))[:, None] * vt
            if self.ifq > 0:
                alpha = friction.filter_alpha(self.ifq, self.xfiltr, dt)
                keys = ni * max(len(self.segs), 1) + srow[active]
                ftvec, self._filt_keys, self._filt_vals = friction.\
                    apply_filter(keys, ftvec, alpha, self._filt_keys,
                                 self._filt_vals)
            Fvec += ftvec

        # scatter: action on the node, exact opposite reaction on the
        # segment corners (momentum conservation)
        scatter_add3(fcont, ni, Fvec)
        scatter_add3(fcont, seg.reshape(-1),
                     (-wseg[:, :, None] * Fvec[:, None, :]).reshape(-1, 3))

        wrk = float(np.einsum("nb,nb->", Fvec, vrel)) * dt
        return -wrk, dt_int
