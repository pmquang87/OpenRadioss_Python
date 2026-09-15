"""
M480 — Unit tests for /SECT section-force output (engine/sections.py).

The ``SectionForces.compute()`` method calculates section resultants using
the side-sum identity on assembled internal forces:

    F_sect = sum fint[side]
    M_sect = sum (x[side] - x_ref) x fint[side] + mint[side]

Fortran origin: ``engine/source/tools/sect/section.F``, ``section_io.F``,
``forint.F`` — element internal-force accumulation at section nodes.

Tests bypass ``SectionForces.__init__`` and directly set ``.sections``
to test the physics of ``compute()`` in isolation.
"""

import numpy as np
import pytest

from pyradioss.engine.sections import SectionForces


class _Sect:
    """Minimal section entity mock."""
    def __init__(self, id):
        self.id = id


def _make_sections(items):
    """Create a SectionForces with pre-set sections list.

    Each item is (sect_id, side_idx, ref_node_or_neg1, x_ref0).
    """
    sf = object.__new__(SectionForces)
    sf.model = None
    sf.sections = [(_Sect(sid), np.asarray(side), ref, x0)
                   for sid, side, ref, x0 in items]
    return sf


# ======================================================================
# Force summation:  F_sect = sum fint[side]
# ======================================================================
class TestForceSummation:
    """Section force is the sum of internal forces on the side nodes."""

    def test_single_node_section(self):
        """One node in section → force equals that node's fint."""
        sf = _make_sections([(1, [0], -1, np.array([0.0, 0.0, 0.0]))])

        fint = np.array([[100.0, 200.0, 300.0]])
        mint = np.zeros((1, 3))
        x = np.array([[1.0, 2.0, 3.0]])

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        np.testing.assert_allclose(F, [100.0, 200.0, 300.0])

    def test_multi_node_sum(self):
        """Multiple nodes → force is their sum."""
        sf = _make_sections([(1, [0, 1, 2], -1, np.array([0.0, 0.0, 0.0]))])

        fint = np.array([[1.0, 0.0, 0.0],
                         [0.0, 2.0, 0.0],
                         [0.0, 0.0, 3.0]])
        mint = np.zeros((3, 3))
        x = np.zeros((3, 3))

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        np.testing.assert_allclose(F, [1.0, 2.0, 3.0])

    def test_equilibrated_element(self):
        """Self-equilibrated internal forces sum to zero."""
        sf = _make_sections([(1, [0, 1, 2, 3], -1, np.array([0.0, 0.0, 0.0]))])

        # Uniform tension on a bar: node forces cancel
        fint = np.array([[10.0, 0.0, 0.0],
                         [-10.0, 0.0, 0.0],
                         [10.0, 0.0, 0.0],
                         [-10.0, 0.0, 0.0]])
        mint = np.zeros((4, 3))
        x = np.zeros((4, 3))

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        np.testing.assert_allclose(F, [0.0, 0.0, 0.0], atol=1e-15)


# ======================================================================
# Moment computation:  M = sum (x - x_ref) x fint + mint
# ======================================================================
class TestMomentComputation:
    """Section moment about x_ref with cross product + internal moments."""

    def test_moment_from_force(self):
        """Force at offset position → moment = r × F."""
        # Node at (1, 0, 0), force (0, 1, 0), ref at origin → M = (0, 0, 1)
        sf = _make_sections([(1, [0], -1, np.array([0.0, 0.0, 0.0]))])

        fint = np.array([[0.0, 1.0, 0.0]])
        mint = np.zeros((1, 3))
        x = np.array([[1.0, 0.0, 0.0]])

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        np.testing.assert_allclose(M, [0.0, 0.0, 1.0])

    def test_moment_with_mint(self):
        """Internal moment (mint) adds directly to the section moment."""
        sf = _make_sections([(1, [0], -1, np.array([0.0, 0.0, 0.0]))])

        fint = np.zeros((1, 3))  # no force contribution
        mint = np.array([[5.0, 10.0, 15.0]])
        x = np.array([[0.0, 0.0, 0.0]])

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        np.testing.assert_allclose(F, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(M, [5.0, 10.0, 15.0])

    def test_combined_force_and_mint(self):
        """Cross-product moment + mint combined."""
        sf = _make_sections([(1, [0], -1, np.array([0.0, 0.0, 0.0]))])

        fint = np.array([[0.0, 0.0, 1.0]])  # force in z
        mint = np.array([[0.0, 0.0, 3.0]])   # extra moment in z
        x = np.array([[1.0, 0.0, 0.0]])      # offset in x

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        # r × F = (1,0,0) × (0,0,1) = (0,-1,0)
        np.testing.assert_allclose(M, [0.0, -1.0, 3.0])

    def test_moment_multiple_nodes(self):
        """Moment summed over multiple nodes."""
        sf = _make_sections([(1, [0, 1], -1, np.array([0.0, 0.0, 0.0]))])

        fint = np.array([[0.0, 1.0, 0.0],   # node 0: force in y
                         [0.0, 1.0, 0.0]])   # node 1: force in y
        mint = np.zeros((2, 3))
        x = np.array([[1.0, 0.0, 0.0],       # offset in x
                       [-1.0, 0.0, 0.0]])     # offset in -x

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        # r0 × F0 = (1,0,0) × (0,1,0) = (0,0,1)
        # r1 × F1 = (-1,0,0) × (0,1,0) = (0,0,-1)
        # Sum = (0,0,0) — couple cancels
        np.testing.assert_allclose(F, [0.0, 2.0, 0.0])
        np.testing.assert_allclose(M, [0.0, 0.0, 0.0], atol=1e-15)


# ======================================================================
# Reference point:  fixed x_ref0 vs moving ref node
# ======================================================================
class TestReferencePoint:
    """Section moment reference: fixed centroid vs moving node."""

    def test_fixed_reference(self):
        """x_ref0 is used when ref < 0 (no reference node)."""
        x_ref = np.array([2.0, 0.0, 0.0])
        sf = _make_sections([(1, [0], -1, x_ref)])

        fint = np.array([[0.0, 1.0, 0.0]])
        mint = np.zeros((1, 3))
        x = np.array([[5.0, 0.0, 0.0]])

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        # r = (5-2, 0, 0) = (3, 0, 0); r × (0,1,0) = (0,0,3)
        np.testing.assert_allclose(M, [0.0, 0.0, 3.0])

    def test_moving_reference_node(self):
        """When ref >= 0, x_ref = x[ref] (tracks deformation)."""
        # ref node is node 1 at position (2, 0, 0)
        sf = _make_sections([(1, [0], 1, None)])

        fint = np.array([[0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0]])
        mint = np.zeros((2, 3))
        x = np.array([[5.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0]])  # ref node position

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        # r = (5-2, 0, 0) = (3, 0, 0); r × (0,1,0) = (0,0,3)
        np.testing.assert_allclose(M, [0.0, 0.0, 3.0])


# ======================================================================
# Multiple sections
# ======================================================================
class TestMultipleSections:
    """Multiple /SECT definitions computed independently."""

    def test_two_sections(self):
        """Two sections on different node sets yield independent results."""
        sf = _make_sections([
            (10, [0, 1], -1, np.array([0.0, 0.0, 0.0])),
            (20, [2, 3], -1, np.array([0.0, 0.0, 0.0])),
        ])

        fint = np.array([[1.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0],
                         [0.0, 3.0, 0.0],
                         [0.0, 4.0, 0.0]])
        mint = np.zeros((4, 3))
        x = np.zeros((4, 3))

        result = sf.compute(x, fint, mint)

        F10, _ = result[10]
        F20, _ = result[20]
        np.testing.assert_allclose(F10, [3.0, 0.0, 0.0])
        np.testing.assert_allclose(F20, [0.0, 7.0, 0.0])


# ======================================================================
# Properties
# ======================================================================
class TestProperties:
    """Test __len__ and ids property."""

    def test_len(self):
        sf = _make_sections([
            (1, [0], -1, np.zeros(3)),
            (2, [1], -1, np.zeros(3)),
        ])
        assert len(sf) == 2

    def test_ids(self):
        sf = _make_sections([
            (10, [0], -1, np.zeros(3)),
            (20, [1], -1, np.zeros(3)),
            (30, [2], -1, np.zeros(3)),
        ])
        assert sf.ids == [10, 20, 30]

    def test_empty(self):
        sf = _make_sections([])
        assert len(sf) == 0
        assert sf.ids == []


# ======================================================================
# Physics validation: pure tension bar
# ======================================================================
class TestPhysicsValidation:
    """Validate the side-sum identity on a simple physical case."""

    def test_tension_bar_section_force(self):
        """A bar in pure tension: section through the middle should
        report the tension force.

        Bar: nodes 0-1-2-3-4 along x, two elements on each side of
        the cut at node 2. Tension T = 1000 N.

        fint at cut node 2 gets contributions from BOTH elements.
        The side-sum over nodes {0, 1, 2} captures the resultant
        from the excluded side (nodes {3, 4}).
        """
        # Simplified: nodes along x-axis
        x = np.array([[0.0, 0.0, 0.0],
                       [1.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0],
                       [3.0, 0.0, 0.0],
                       [4.0, 0.0, 0.0]])

        # Assembled internal forces for uniform tension T = 1000
        # End nodes: reaction; interior: cancel
        # Element A (0-1): f0 = (-T, 0, 0), f1 = (T, 0, 0)
        # Element B (1-2): f1 += (-T, 0, 0), f2 = (T, 0, 0)
        # Element C (2-3): f2 += (-T, 0, 0), f3 = (T, 0, 0)
        # Element D (3-4): f3 += (-T, 0, 0), f4 = (T, 0, 0)
        T = 1000.0
        fint = np.array([[-T, 0.0, 0.0],    # node 0: reaction
                          [0.0, 0.0, 0.0],   # node 1: T - T = 0
                          [0.0, 0.0, 0.0],   # node 2: T - T = 0
                          [0.0, 0.0, 0.0],   # node 3: T - T = 0
                          [T, 0.0, 0.0]])    # node 4: reaction

        mint = np.zeros((5, 3))

        # Section: left side = {0, 1, 2}
        x_ref = np.array([2.0, 0.0, 0.0])  # at the cut
        sf = _make_sections([(1, [0, 1, 2], -1, x_ref)])

        result = sf.compute(x, fint, mint)

        F, M = result[1]
        # The right side pulls the left side with -T in x
        # Sum of fint on {0,1,2} = (-T + 0 + 0) = -T
        np.testing.assert_allclose(F, [-T, 0.0, 0.0])
        # Moment about x_ref = (2,0,0):
        # r0 = (-2,0,0), f0 = (-T,0,0) → r×f = 0
        # r1 = (-1,0,0), f1 = (0,0,0) → 0
        # r2 = (0,0,0), f2 = (0,0,0) → 0
        np.testing.assert_allclose(M, [0.0, 0.0, 0.0], atol=1e-15)
