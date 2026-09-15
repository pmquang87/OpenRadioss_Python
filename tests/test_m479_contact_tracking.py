"""
M479 — Unit tests for contact/tracking.py (element deletion ↔ contact).

The tracking module provides four functions that maintain contact-segment
and contact-node masks when elements are deleted by /FAIL criteria:

* ``alive_segment_mask``   — True for segments whose parent element is alive
* ``any_deletable``        — True when any referenced group can delete
* ``node_reference_counts``— count how many elements reference each node
* ``tracked_node_mask``    — True for nodes contact should keep tracking

Fortran origin: the ``IDEL`` option of the interfaces —
``engine/source/interfaces/interf/intfop2.F`` bookkeeping and
``i7for3.F``'s dead-segment skip (element ``GBUF%OFF`` flags).

These functions are called every cycle (segment mask) or every broad phase
(node mask) from TYPE2, TYPE7, TYPE10, TYPE11, TYPE24, and follower loads.
"""

import numpy as np
import pytest

from pyradioss.contact.tracking import (
    alive_segment_mask,
    any_deletable,
    node_reference_counts,
    tracked_node_mask,
)


# ======================================================================
# Minimal mock objects — just enough to drive the tracking functions
# ======================================================================
class _Group:
    """Minimal element group mock."""
    def __init__(self, conn, state=None):
        self.conn = np.asarray(conn)
        self.state = state if state is not None else {}

    @property
    def n(self):
        return len(self.conn)


class _Model:
    """Minimal model mock with named element groups and node count."""
    def __init__(self, numnod, groups=None):
        self.numnod = numnod
        self._groups = groups or {}
        # Set named attributes so getattr(model, gname) works
        for name, group in self._groups.items():
            setattr(self, name, group)

    def element_groups(self):
        for name, group in self._groups.items():
            if group.n:
                yield name, group


# ======================================================================
# alive_segment_mask
# ======================================================================
class TestAliveSegmentMask:
    """alive_segment_mask(model, seg_gtype, seg_elem) → bool mask."""

    def test_all_alive(self):
        """All elements alive → all segments True."""
        off = np.array([1.0, 1.0, 1.0])
        model = _Model(10, {"shells": _Group(
            conn=np.zeros((3, 4), dtype=int),
            state={"off": off}
        )})
        seg_gtype = np.array(["shells", "shells", "shells"])
        seg_elem = np.array([0, 1, 2])

        mask = alive_segment_mask(model, seg_gtype, seg_elem)

        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_some_dead(self):
        """Element 1 dead → its segment is False."""
        off = np.array([1.0, 0.0, 1.0])
        model = _Model(10, {"shells": _Group(
            conn=np.zeros((3, 4), dtype=int),
            state={"off": off}
        )})
        seg_gtype = np.array(["shells", "shells", "shells"])
        seg_elem = np.array([0, 1, 2])

        mask = alive_segment_mask(model, seg_gtype, seg_elem)

        np.testing.assert_array_equal(mask, [True, False, True])

    def test_no_off_array(self):
        """Group without 'off' in state → all segments stay True."""
        model = _Model(10, {"bricks": _Group(
            conn=np.zeros((3, 8), dtype=int),
            state={}  # no "off" key
        )})
        seg_gtype = np.array(["bricks", "bricks"])
        seg_elem = np.array([0, 1])

        mask = alive_segment_mask(model, seg_gtype, seg_elem)

        np.testing.assert_array_equal(mask, [True, True])

    def test_empty_provenance(self):
        """Segments with '' gtype (no provenance) stay True."""
        off = np.array([0.0])  # dead element
        model = _Model(10, {"shells": _Group(
            conn=np.zeros((1, 4), dtype=int),
            state={"off": off}
        )})
        seg_gtype = np.array(["", "shells"])
        seg_elem = np.array([0, 0])

        mask = alive_segment_mask(model, seg_gtype, seg_elem)

        # '' segment stays True, shells segment sees dead element
        np.testing.assert_array_equal(mask, [True, False])

    def test_mixed_groups(self):
        """Segments from different groups tracked independently."""
        model = _Model(10, {
            "shells": _Group(
                conn=np.zeros((2, 4), dtype=int),
                state={"off": np.array([1.0, 0.0])}  # elem 1 dead
            ),
            "bricks": _Group(
                conn=np.zeros((2, 8), dtype=int),
                state={"off": np.array([0.0, 1.0])}  # elem 0 dead
            ),
        })
        seg_gtype = np.array(["shells", "shells", "bricks", "bricks"])
        seg_elem = np.array([0, 1, 0, 1])

        mask = alive_segment_mask(model, seg_gtype, seg_elem)

        np.testing.assert_array_equal(mask, [True, False, False, True])

    def test_empty_input(self):
        """Zero segments → zero-length True mask."""
        model = _Model(10, {})
        seg_gtype = np.array([], dtype='<U10')
        seg_elem = np.array([], dtype=int)

        mask = alive_segment_mask(model, seg_gtype, seg_elem)

        assert len(mask) == 0
        assert mask.dtype == bool


# ======================================================================
# any_deletable
# ======================================================================
class TestAnyDeletable:
    """any_deletable(model, seg_gtype) → bool."""

    def test_no_off_array(self):
        """Group without 'off' → not deletable."""
        model = _Model(10, {"shells": _Group(
            conn=np.zeros((1, 4), dtype=int),
            state={}
        )})
        seg_gtype = np.array(["shells"])

        assert any_deletable(model, seg_gtype) is False

    def test_off_but_no_chk_fail(self):
        """Group with 'off' but no 'chk_fail' → not deletable."""
        model = _Model(10, {"shells": _Group(
            conn=np.zeros((1, 4), dtype=int),
            state={"off": np.array([1.0])}
        )})
        seg_gtype = np.array(["shells"])

        assert any_deletable(model, seg_gtype) is False

    def test_off_and_chk_fail(self):
        """Group with 'off' AND 'chk_fail'=True → deletable."""
        model = _Model(10, {"shells": _Group(
            conn=np.zeros((1, 4), dtype=int),
            state={"off": np.array([1.0]), "chk_fail": True}
        )})
        seg_gtype = np.array(["shells"])

        assert any_deletable(model, seg_gtype) is True

    def test_empty_provenance_skipped(self):
        """'' gtype is ignored even if something else is deletable."""
        model = _Model(10, {"shells": _Group(
            conn=np.zeros((1, 4), dtype=int),
            state={"off": np.array([1.0]), "chk_fail": True}
        )})
        seg_gtype = np.array(["", "shells"])

        assert any_deletable(model, seg_gtype) is True

    def test_only_empty_provenance(self):
        """Only '' gtype segments → not deletable."""
        model = _Model(10, {})
        seg_gtype = np.array(["", ""])

        assert any_deletable(model, seg_gtype) is False

    def test_secondary_failure_enables_deletable(self):
        """BUG-06: Secondary nodes belonging to failure-capable element enable deletion tracking."""
        model = _Model(10, {
            "shells": _Group(
                conn=np.array([[0, 1, 2, 3]]),
                state={"off": np.array([1.0]), "chk_fail": True}
            )
        })
        seg_gtype = np.array([""])  # Main surface has empty provenance
        sec_nodes = np.array([2, 5])

        assert any_deletable(model, seg_gtype, sec_nodes=sec_nodes) is True

    def test_secondary_without_failure_remains_false(self):
        """Secondary nodes in non-failing elements keep any_deletable False."""
        model = _Model(10, {
            "shells": _Group(
                conn=np.array([[0, 1, 2, 3]]),
                state={"off": np.array([1.0])}  # no chk_fail
            )
        })
        seg_gtype = np.array([""])
        sec_nodes = np.array([2, 5])

        assert any_deletable(model, seg_gtype, sec_nodes=sec_nodes) is False


# ======================================================================
# node_reference_counts
# ======================================================================
class TestNodeReferenceCounts:
    """node_reference_counts(model, alive_only) → int array of length numnod."""

    def test_simple_triangle(self):
        """One triangle element with nodes 0,1,2 — each referenced once."""
        model = _Model(5, {"sh3n": _Group(
            conn=np.array([[0, 1, 2]]),
            state={}
        )})

        cnt = node_reference_counts(model, alive_only=False)

        assert len(cnt) == 5
        np.testing.assert_array_equal(cnt, [1, 1, 1, 0, 0])

    def test_shared_node(self):
        """Two quads sharing an edge (nodes 1,2)."""
        model = _Model(6, {"shells": _Group(
            conn=np.array([[0, 1, 2, 3],
                           [1, 4, 5, 2]]),
            state={}
        )})

        cnt = node_reference_counts(model, alive_only=False)

        assert cnt[0] == 1   # only elem 0
        assert cnt[1] == 2   # shared
        assert cnt[2] == 2   # shared
        assert cnt[3] == 1   # only elem 0
        assert cnt[4] == 1   # only elem 1
        assert cnt[5] == 1   # only elem 1

    def test_alive_only_excludes_dead(self):
        """alive_only=True skips elements with off <= 0."""
        model = _Model(4, {"shells": _Group(
            conn=np.array([[0, 1, 2, 3],
                           [0, 1, 2, 3]]),
            state={"off": np.array([1.0, 0.0])}  # elem 1 dead
        )})

        cnt_all = node_reference_counts(model, alive_only=False)
        cnt_alive = node_reference_counts(model, alive_only=True)

        np.testing.assert_array_equal(cnt_all, [2, 2, 2, 2])
        np.testing.assert_array_equal(cnt_alive, [1, 1, 1, 1])

    def test_multiple_groups(self):
        """Counts accumulate across groups."""
        model = _Model(4, {
            "shells": _Group(conn=np.array([[0, 1, 2, 3]]), state={}),
            "bricks": _Group(conn=np.array([[0, 1, 2, 3, 0, 1, 2, 3]]), state={}),
        })

        cnt = node_reference_counts(model, alive_only=False)

        # shells: each node +1; bricks: each node +2 (appears twice in 8-node conn)
        np.testing.assert_array_equal(cnt, [3, 3, 3, 3])

    def test_no_elements(self):
        """Empty model → zero counts."""
        model = _Model(5, {})

        cnt = node_reference_counts(model, alive_only=False)

        np.testing.assert_array_equal(cnt, [0, 0, 0, 0, 0])

    def test_placeholder_connectivity_ignored(self):
        """BUG-07: -1 placeholders (e.g. slaved TETRA10) must not increment the last node."""
        model = _Model(5, {"tetra10s": _Group(
            conn=np.array([[0, 1, 2, 3, -1, -1, -1, -1, -1, -1]]),
            state={}
        )})
        cnt = node_reference_counts(model, alive_only=False)
        np.testing.assert_array_equal(cnt, [1, 1, 1, 1, 0])


# ======================================================================
# tracked_node_mask
# ======================================================================
class TestTrackedNodeMask:
    """tracked_node_mask(model, ref_total) → bool mask."""

    def test_all_alive(self):
        """All elements alive → all nodes tracked."""
        model = _Model(4, {"shells": _Group(
            conn=np.array([[0, 1, 2, 3]]),
            state={"off": np.array([1.0])}
        )})
        ref_total = node_reference_counts(model, alive_only=False)

        mask = tracked_node_mask(model, ref_total)

        np.testing.assert_array_equal(mask, [True, True, True, True])

    def test_orphaned_node(self):
        """Node 2 loses all elements → stops being tracked."""
        model = _Model(3, {"shells": _Group(
            conn=np.array([[0, 1, 0, 1],    # elem 0: nodes 0,1
                           [2, 2, 2, 2]]),  # elem 1: node 2 only
            state={"off": np.array([1.0, 0.0])}  # elem 1 dead
        )})
        ref_total = np.array([2, 2, 4])  # from initial (all-alive) counts

        mask = tracked_node_mask(model, ref_total)

        assert mask[0] is np.True_   # node 0: alive elem 0
        assert mask[1] is np.True_   # node 1: alive elem 0
        assert mask[2] is np.False_  # node 2: no alive elements

    def test_extra_mass_node(self):
        """Node never in any element (ref_total=0) → stays tracked forever."""
        model = _Model(3, {"shells": _Group(
            conn=np.array([[0, 1, 0, 1]]),
            state={"off": np.array([1.0])}
        )})
        ref_total = np.array([2, 2, 0])  # node 2: added mass, not in any element

        mask = tracked_node_mask(model, ref_total)

        assert mask[0] is np.True_   # structural node
        assert mask[1] is np.True_   # structural node
        assert mask[2] is np.True_   # extra mass node — always tracked

    def test_partial_deletion(self):
        """Node shared between live and dead elements stays tracked."""
        model = _Model(4, {"shells": _Group(
            conn=np.array([[0, 1, 2, 3],
                           [0, 1, 2, 3]]),
            state={"off": np.array([1.0, 0.0])}  # elem 1 dead
        )})
        ref_total = np.array([2, 2, 2, 2])

        mask = tracked_node_mask(model, ref_total)

        # All nodes still have 1 alive element → all tracked
        np.testing.assert_array_equal(mask, [True, True, True, True])
