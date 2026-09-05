"""
M482 — Unit tests for engine/mass_scaling.py (NodalTimeStep).

The NodalTimeStep class implements /DT/NODA and /DT/NODA/CST — the nodal
time step and mass scaling machinery that bounds the global dt by the
WORST node-on-spring system rather than the worst element, and optionally
adds mass to nodes whose nodal dt falls below a target.

Fortran origin: ``engine/source/time_step/dtnoda.F`` (nodal time step),
``rgbodfp.F`` (rigid-body stiffness transport to master), ``cupdt3.F``
/ ``pmcum3.F`` (per-node stiffness accumulation).

Functions under test:
* ``NodalTimeStep.__init__()`` — assembles per-group mass/inertia shares
* ``NodalTimeStep.assemble()`` — nodal stiffness from element dt claims
* ``NodalTimeStep.apply()``    — nodal time step + CST mass scaling
* ``NodalTimeStep.set_prescribed()`` — exclude nodes from dt/mass-add
* ``NodalTimeStep.add_rigid_body()`` / ``_rigid_body_dt()`` — Huygens-
  Steiner stiffness transport (tested in test_m39_rbody_dt.py in full
  integration; here tested at the unit level)
* ``NodalTimeStep.summary()``  — listing report

All tests use lightweight synthetic stubs — no starter or engine run
needed — following the M477–M481 pattern.
"""

import io

import numpy as np
import pytest

from pyradioss.common.constants import EM20, EP30
from pyradioss.common.messages import MessageLog
from pyradioss.engine.mass_scaling import NodalTimeStep


# ======================================================================
# Synthetic stubs — minimal stand-ins for Model, Controls and groups
# ======================================================================

class _Group:
    """Minimal element group stub."""
    def __init__(self, conn, mass_per_elem, dt_iner=None, mass_conn=None):
        self.conn = np.asarray(conn)
        self.state = {"mass": np.asarray(mass_per_elem, dtype=float)}
        if dt_iner is not None:
            self.state["dt_iner"] = np.asarray(dt_iner, dtype=float)
        if mass_conn is not None:
            self.state["mass_conn"] = np.asarray(mass_conn)

    @property
    def n(self):
        return self.conn.shape[0]


class _Model:
    """Minimal model stub with one or more element groups."""
    def __init__(self, numnod, mass, groups=None, x0=None):
        self.numnod = numnod
        self.mass = np.asarray(mass, dtype=float)
        self._groups = groups or {}
        self.x0 = x0 if x0 is not None else np.zeros((numnod, 3))
        self.inertia = np.zeros(numnod)

    def element_groups(self):
        for name, g in self._groups.items():
            if g is not None and g.n:
                yield name, g


class _Controls:
    """Minimal engine controls stub."""
    def __init__(self, dt_noda="NODA", dt_scale=0.9, dt_min=0.0):
        self.dt_noda = dt_noda
        self.dt_scale = dt_scale
        self.dt_min = dt_min


def _make_log():
    """Return a MessageLog that does not print to stdout."""
    log = MessageLog()
    log._listing = io.StringIO()
    # Suppress stdout printing
    log._emit = lambda text: log._listing.write(text + "\n")
    return log


# ======================================================================
# Helper: a simple 2-element quad strip (4 nodes shared at edge)
# ======================================================================
#  Nodes: 0---1---2
#         |   |   |
#         3---4---5
#
#  Elem 0: (0,1,4,3),  Elem 1: (1,2,5,4)  — nodes 1,4 shared
#

def _quad_strip_model(elem_mass=4.0, noda="NODA", dt_min=0.0, dt_scale=0.9):
    """Two 4-node shell elements sharing an edge (6 nodes total)."""
    conn = np.array([[0, 1, 4, 3],
                     [1, 2, 5, 4]])
    mass = np.full(2, elem_mass)
    group = _Group(conn, mass)

    # Each node gets 1/4 of the element mass -> 1.0 per node per element
    # Nodes 1,4 shared -> total mass 2.0; corner nodes total mass 1.0
    node_mass = np.ones(6)  # base
    node_mass[1] += 1.0     # shared -> 2.0
    node_mass[4] += 1.0     # shared -> 2.0

    model = _Model(numnod=6, mass=node_mass,
                   groups={"shells": group})
    controls = _Controls(dt_noda=noda, dt_scale=dt_scale, dt_min=dt_min)
    return model, controls


# ======================================================================
# assemble() — nodal stiffness from element dt claims
# ======================================================================

class TestAssemble:
    """NodalTimeStep.assemble(): k_i^e = 2 * mass_share / dt_e^2."""

    def test_single_element_stiffness(self):
        """One 4-node element: each node gets k = 2*(m/4)/dt^2."""
        conn = np.array([[0, 1, 2, 3]])
        elem_mass = np.array([8.0])      # 2.0 per node
        group = _Group(conn, elem_mass)
        model = _Model(numnod=4, mass=np.full(4, 2.0),
                       groups={"shells": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        # k = 2 * 2.0 / 0.01^2 = 40000.0 per node
        expected = 2.0 * 2.0 / 0.01**2
        np.testing.assert_allclose(noda.stifn[:4], expected, rtol=1e-12)

    def test_shared_node_accumulates(self):
        """Two elements sharing nodes: stiffness adds at the shared node."""
        model, controls = _quad_strip_model()
        noda = NodalTimeStep(model, controls, _make_log())

        dt_e = np.array([0.02, 0.02])    # both elements same dt
        noda.assemble([dt_e])

        mass_share = 4.0 / 4.0           # 1.0 per node per element
        k_single = 2.0 * mass_share / 0.02**2

        # Corner nodes (0,2,3,5): one element contribution
        for idx in [0, 2, 3, 5]:
            assert noda.stifn[idx] == pytest.approx(k_single, rel=1e-12)

        # Shared nodes (1,4): two element contributions
        for idx in [1, 4]:
            assert noda.stifn[idx] == pytest.approx(2.0 * k_single, rel=1e-12)

    def test_deleted_element_zero_stiffness(self):
        """A deleted element (dt=1e30) contributes zero stiffness."""
        conn = np.array([[0, 1, 2, 3]])
        group = _Group(conn, np.array([4.0]))
        model = _Model(numnod=4, mass=np.ones(4),
                       groups={"shells": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())

        noda.assemble([np.array([1e30])])

        # k = 2 * 1.0 / 1e30^2 = 2e-60 ≈ 0
        np.testing.assert_allclose(noda.stifn[:4], 0.0, atol=1e-30)

    def test_rotational_stifr_assembly(self):
        """Groups with dt_iner set assemble rotational stiffness."""
        conn = np.array([[0, 1, 2, 3]])
        elem_mass = np.array([4.0])       # 1.0 per node
        dt_iner = np.array([0.5])         # rotational inertia per node
        group = _Group(conn, elem_mass, dt_iner=dt_iner)
        model = _Model(numnod=4, mass=np.ones(4),
                       groups={"shells": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())
        assert noda._rot is True

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        # kr = 2 * 0.5 / 0.01^2 = 10000.0
        expected_kr = 2.0 * 0.5 / 0.01**2
        np.testing.assert_allclose(noda.stifr[:4], expected_kr, rtol=1e-12)

    def test_no_rotational_groups_stifr_stays_zero(self):
        """Solids-only model: no rotational stiffness assembled."""
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        group = _Group(conn, np.array([8.0]))   # no dt_iner
        model = _Model(numnod=8, mass=np.ones(8),
                       groups={"bricks": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())
        assert noda._rot is False

        noda.assemble([np.array([0.01])])
        np.testing.assert_allclose(noda.stifr, 0.0)


# ======================================================================
# apply() — nodal time step (non-CST)
# ======================================================================

class TestNodalDt:
    """NodalTimeStep.apply() without mass scaling (NODA, not CST)."""

    def test_uniform_mesh_dt_equals_element_dt(self):
        """On a uniform mesh the nodal dt equals the element dt.

        Theory (module docstring): for a uniform mesh each node's mass
        and stiffness scale together, so dt_i = dt_e for all nodes.
        """
        model, controls = _quad_strip_model()
        noda = NodalTimeStep(model, controls, _make_log())

        dt_e = np.array([0.02, 0.02])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((6, 3))

        dt = noda.apply(mass_eff, inv_mass, v, t=0.0)
        assert dt == pytest.approx(0.02, rel=1e-12)

    def test_nonuniform_mesh_binds_at_worst_node(self):
        """Non-uniform dt: the smaller element's nodes bind the step."""
        conn = np.array([[0, 1, 2, 3],
                         [1, 4, 5, 2]])    # shares nodes 1,2
        mass = np.array([4.0, 4.0])
        group = _Group(conn, mass)
        node_mass = np.ones(6)
        node_mass[1] += 1.0
        node_mass[2] += 1.0
        model = _Model(numnod=6, mass=node_mass,
                       groups={"shells": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())

        # Element 0: dt=0.02, Element 1: dt=0.005 (much stiffer)
        dt_e = np.array([0.02, 0.005])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((6, 3))

        dt = noda.apply(mass_eff, inv_mass, v, t=0.0)

        # The nodal dt must be ≤ 0.005 (the stiff element's dt)
        # On the shared nodes it may be even lower
        assert dt <= 0.005 + 1e-15

    def test_prescribed_nodes_excluded(self):
        """Prescribed (rigid-body member) nodes do not bind the step."""
        conn = np.array([[0, 1, 2, 3]])
        group = _Group(conn, np.array([4.0]))
        model = _Model(numnod=4, mass=np.ones(4),
                       groups={"shells": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())

        # Prescribe nodes 0,1 — only nodes 2,3 remain free
        noda.set_prescribed(np.array([0, 1]))
        assert not noda.free[0]
        assert not noda.free[1]
        assert noda.free[2]
        assert noda.free[3]

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        dt = noda.apply(mass_eff, inv_mass, v, t=0.0)
        # Only nodes 2,3 (free and loaded) participate.
        # Their k = 2*1.0/0.01^2 = 20000, m = 1.0
        # dt = sqrt(2*1.0/20000) = 0.01
        assert dt == pytest.approx(0.01, rel=1e-12)

    def test_ams_nodes_excluded(self):
        """AMS-active nodes do not bind the explicit step."""
        conn = np.array([[0, 1, 2, 3]])
        group = _Group(conn, np.array([4.0]))
        model = _Model(numnod=4, mass=np.ones(4),
                       groups={"shells": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        # All nodes marked as AMS → no free loaded node → rigid-body path
        ams = np.ones(4, dtype=bool)
        dt = noda.apply(mass_eff, inv_mass, v, t=0.0, ams_nodes=ams)
        assert dt == EP30   # no body registered, returns EP30

    def test_no_loaded_nodes_returns_ep30(self):
        """No free loaded nodes and no rigid body → returns EP30."""
        model = _Model(numnod=4, mass=np.ones(4), groups={})
        noda = NodalTimeStep(model, _Controls(), _make_log())

        # No groups → no assembly → stifn stays 0 → no loaded nodes
        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        dt = noda.apply(mass_eff, inv_mass, v, t=0.0)
        assert dt == EP30

    def test_stiffness_reset_after_apply(self):
        """Stiffness accumulators are zeroed after apply() — dtnoda.F
        contract: each cycle starts from zero stiffness."""
        model, controls = _quad_strip_model()
        noda = NodalTimeStep(model, controls, _make_log())

        noda.assemble([np.array([0.01, 0.01])])
        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((6, 3))
        noda.apply(mass_eff, inv_mass, v, t=0.0)

        np.testing.assert_allclose(noda.stifn, 0.0)
        np.testing.assert_allclose(noda.stifr, 0.0)


# ======================================================================
# apply() — CST mass scaling
# ======================================================================

class TestCstMassScaling:
    """/DT/NODA/CST: mass is added so dt_sca * sqrt(2M/K) >= dt_min."""

    @staticmethod
    def _make_cst(dt_min=0.02, dt_scale=0.9):
        """One element, 4 nodes, CST mode with a target dt."""
        conn = np.array([[0, 1, 2, 3]])
        group = _Group(conn, np.array([4.0]))  # 1.0 per node
        model = _Model(numnod=4, mass=np.ones(4),
                       groups={"shells": group})
        controls = _Controls(dt_noda="CST", dt_scale=dt_scale,
                             dt_min=dt_min)
        log = _make_log()
        noda = NodalTimeStep(model, controls, log)
        assert noda.cst is True
        return model, noda, log

    def test_mass_added_when_below_target(self):
        """When nodal dt < dt_min, mass is added to raise it."""
        model, noda, log = self._make_cst(dt_min=0.05, dt_scale=1.0)

        dt_e = np.array([0.01])  # element dt much smaller than target
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        old_mass = model.mass.copy()
        dt = noda.apply(mass_eff, inv_mass, v, t=0.0)

        # Mass must have been added to every node
        for i in range(4):
            assert model.mass[i] > old_mass[i]

        # After scaling: dt_sca * sqrt(2 M_new / K) >= dt_min
        # With dt_scale=1.0: sqrt(2 M_new / K) >= 0.05
        # Verify the returned dt is now at least dt_min
        assert dt >= 0.05 - 1e-12

    def test_mass_never_removed(self):
        """If the node is already above dt_min, no mass is removed."""
        model, noda, log = self._make_cst(dt_min=0.001, dt_scale=1.0)

        dt_e = np.array([0.01])  # dt_e >> dt_min
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))
        old_mass = model.mass.copy()

        noda.apply(mass_eff, inv_mass, v, t=0.0)

        # Mass unchanged — no addition, no removal
        np.testing.assert_array_equal(model.mass, old_mass)
        assert noda.mass_added == 0.0

    def test_mass_eff_and_inv_mass_updated(self):
        """mass_eff and inv_mass are updated consistently with model.mass."""
        model, noda, log = self._make_cst(dt_min=0.05, dt_scale=1.0)

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        noda.apply(mass_eff, inv_mass, v, t=0.0)

        # mass_eff should match model.mass after scaling
        np.testing.assert_allclose(mass_eff, model.mass, rtol=1e-12)
        # inv_mass should be 1/mass_eff
        np.testing.assert_allclose(inv_mass, 1.0 / mass_eff, rtol=1e-12)

    def test_ke_booking(self):
        """The KE of added mass is booked: e_madd += 0.5 * dm * |v|^2."""
        model, noda, log = self._make_cst(dt_min=0.05, dt_scale=1.0)

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        # Give nodes non-zero velocity
        v = np.array([[1.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0],
                       [0.0, 0.0, 3.0],
                       [1.0, 1.0, 1.0]])
        old_mass = model.mass.copy()

        noda.apply(mass_eff, inv_mass, v, t=0.0)

        dm = model.mass - old_mass
        expected_ke = float(0.5 * (dm[:, None] * v**2).sum())
        assert noda.e_madd == pytest.approx(expected_ke, rel=1e-12)

    def test_momentum_booking(self):
        """Momentum from added mass is booked: mom_added += dm * v."""
        model, noda, log = self._make_cst(dt_min=0.05, dt_scale=1.0)

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.array([[1.0, 0.0, 0.0],
                       [0.0, 2.0, 0.0],
                       [0.0, 0.0, 3.0],
                       [1.0, 1.0, 1.0]])
        old_mass = model.mass.copy()

        noda.apply(mass_eff, inv_mass, v, t=0.0)

        dm = model.mass - old_mass
        expected_mom = (dm[:, None] * v).sum(axis=0)
        np.testing.assert_allclose(noda.mom_added, expected_mom, rtol=1e-12)

    def test_cumulative_mass_added(self):
        """mass_added accumulates over multiple cycles."""
        model, noda, log = self._make_cst(dt_min=0.05, dt_scale=1.0)

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        # Cycle 1
        noda.assemble([np.array([0.01])])
        noda.apply(mass_eff, inv_mass, v, t=0.0)
        madd_1 = noda.mass_added

        # Cycle 2: assemble again (stiffness was reset)
        noda.assemble([np.array([0.005])])   # stiffer → more mass needed
        noda.apply(mass_eff, inv_mass, v, t=0.01)
        madd_2 = noda.mass_added

        assert madd_1 > 0.0
        assert madd_2 >= madd_1   # cumulative, never decreases

    def test_dt_min_zero_no_mass_added(self):
        """dt_min=0 → CST mode but no mass is ever added (warning path)."""
        model, noda, log = self._make_cst(dt_min=0.0, dt_scale=1.0)

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))
        old_mass = model.mass.copy()

        noda.apply(mass_eff, inv_mass, v, t=0.0)

        np.testing.assert_array_equal(model.mass, old_mass)
        assert noda.mass_added == 0.0

    def test_cst_formula_exact(self):
        """Verify the exact CST formula: M_needed = K * (dt_min/dt_sca)^2 / 2.

        After mass addition, the nodal dt at every scaled node should be
        exactly dt_min/dt_sca (from which the caller multiplies by dt_sca
        to get dt_min).
        """
        dt_min, dt_sca = 0.04, 0.8
        model, noda, log = self._make_cst(dt_min=dt_min, dt_scale=dt_sca)

        dt_e = np.array([0.005])  # much smaller than target
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        noda.apply(mass_eff, inv_mass, v, t=0.0)

        # After scaling, each node's k and M should give:
        # sqrt(2 M / K) = dt_min / dt_sca
        # We can verify via the returned mass:
        k_per_node = 2.0 * 1.0 / 0.005**2   # mass_share=1.0
        m_needed = k_per_node * (dt_min / dt_sca)**2 / 2.0
        for i in range(4):
            assert model.mass[i] == pytest.approx(m_needed, rel=1e-12)


# ======================================================================
# Rotational nodal dt
# ======================================================================

class TestRotationalDt:
    """Rotational stiffness and dt join the translational minimum."""

    def test_rotational_dt_free_node_invariant(self):
        """On element-lumped free nodes, dt_rot == dt_tra (module docstring).

        Theory: k_i^e = 2 m_i / dt_e^2, kr_i^e = 2 I_i / dt_e^2, so
        dt_tra = sqrt(2 m / k) = dt_e and dt_rot = sqrt(2 I / kr) = dt_e.
        They are exactly equal on element-lumped free nodes.
        """
        conn = np.array([[0, 1, 2, 3]])
        elem_mass = np.array([4.0])       # 1.0 per node
        dt_iner = np.array([0.25])        # arbitrary inertia per node
        group = _Group(conn, elem_mass, dt_iner=dt_iner)
        model = _Model(numnod=4, mass=np.ones(4),
                       groups={"shells": group})
        model.inertia = np.full(4, 0.25)

        noda = NodalTimeStep(model, _Controls(), _make_log())

        dt_e = np.array([0.015])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))

        dt = noda.apply(mass_eff, inv_mass, v, t=0.0,
                        inertia=model.inertia)

        # By construction: dt_tra and dt_rot should be equal
        assert dt == pytest.approx(0.015, rel=1e-12)

    def test_rotational_cst_adds_inertia(self):
        """CST mode adds inertia when rotational dt < target."""
        conn = np.array([[0, 1, 2, 3]])
        elem_mass = np.array([4.0])
        dt_iner = np.array([0.01])     # small inertia → will need scaling
        group = _Group(conn, elem_mass, dt_iner=dt_iner)
        model = _Model(numnod=4, mass=np.full(4, 100.0),   # big mass
                       groups={"shells": group})
        model.inertia = np.full(4, 0.01)

        controls = _Controls(dt_noda="CST", dt_scale=1.0, dt_min=0.05)
        noda = NodalTimeStep(model, controls, _make_log())

        dt_e = np.array([0.001])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))
        inv_inertia = 1.0 / model.inertia.copy()
        old_iner = model.inertia.copy()

        noda.apply(mass_eff, inv_mass, v, t=0.0,
                   inertia=model.inertia, inv_inertia=inv_inertia)

        # Inertia must have been added
        assert noda.iner_added > 0.0
        for i in range(4):
            assert model.inertia[i] >= old_iner[i]

    def test_no_rot_groups_inert(self):
        """Solids-only model: rotational branch is completely inert."""
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
        group = _Group(conn, np.array([8.0]))   # no dt_iner
        model = _Model(numnod=8, mass=np.ones(8),
                       groups={"bricks": group})
        noda = NodalTimeStep(model, _Controls(), _make_log())
        assert noda._rot is False

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((8, 3))

        dt = noda.apply(mass_eff, inv_mass, v, t=0.0)

        # Only translational dt, equals element dt (uniform mesh)
        assert dt == pytest.approx(0.01, rel=1e-12)
        np.testing.assert_allclose(noda.stifr, 0.0)


# ======================================================================
# Rigid body transport
# ======================================================================

class TestRigidBodyTransport:
    """add_rigid_body / _rigid_body_dt at the unit level."""

    def test_no_rbody_returns_ep30(self):
        """No registered body → _rigid_body_dt() returns EP30."""
        model = _Model(numnod=4, mass=np.ones(4), groups={})
        noda = NodalTimeStep(model, _Controls(), _make_log())
        assert noda._rigid_body_dt() == EP30

    def test_transport_formula_closed_form(self):
        """Verify K_tra = sum(stifn), K_rot = sum(stifr + DD*stifn),
        dt = min(sqrt(2M/K_tra), sqrt(2 IN_min/K_rot))."""
        # 4 nodes at (0,0,0), (1,0,0), (1,1,0), (0,1,0)
        x0 = np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [1., 1., 0.],
                        [0., 1., 0.]])
        model = _Model(numnod=4, mass=np.ones(4), groups={}, x0=x0)
        noda = NodalTimeStep(model, _Controls(), _make_log())

        # Set up known stiffness
        noda.stifn[:4] = [100., 200., 300., 400.]
        noda.stifr[:4] = [10., 20., 30., 40.]

        # Register body: master = node 0 at (0,0,0)
        mass = 5.0
        inertia = np.diag([2.0, 3.0, 4.0])  # min principal = 2.0
        nodes = np.arange(4)
        noda.add_rigid_body(nodes, 0, mass, inertia, x0)

        # DD = |x_node - x_master|^2
        dd = np.array([0., 1., 2., 1.])  # (0, 1^2, 1^2+1^2, 1^2)

        k_tra = 100. + 200. + 300. + 400.
        k_rot = ((10. + 0.*100.) + (20. + 1.*200.) +
                 (30. + 2.*300.) + (40. + 1.*400.))

        dt_tra = np.sqrt(2.0 * mass / k_tra)
        dt_rot = np.sqrt(2.0 * 2.0 / k_rot)   # 2.0 = min eigenvalue
        expected = min(dt_tra, dt_rot)

        assert noda._rigid_body_dt() == pytest.approx(expected, rel=1e-12)


# ======================================================================
# summary()
# ======================================================================

class TestSummary:
    """summary() reports the honesty contract numbers."""

    def test_cst_summary_reports_mass(self):
        """CST mode summary prints added mass, energy, momentum."""
        model, noda, log = TestCstMassScaling._make_cst(
            dt_min=0.05, dt_scale=1.0)

        dt_e = np.array([0.01])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 1., 1.]])
        noda.apply(mass_eff, inv_mass, v, t=0.0)

        out_log = _make_log()
        noda.summary(out_log)

        text = out_log._listing.getvalue()
        assert "ADDED MASS" in text
        assert "ENERGY FROM ADDED MASS" in text
        assert "MOMENTUM FROM ADDED MASS" in text

    def test_non_cst_summary_silent(self):
        """Non-CST mode summary prints nothing."""
        model = _Model(numnod=4, mass=np.ones(4), groups={})
        noda = NodalTimeStep(model, _Controls(dt_noda="NODA"), _make_log())

        out_log = _make_log()
        noda.summary(out_log)

        text = out_log._listing.getvalue()
        assert text.strip() == ""

    def test_cst_with_inertia_reports_dinert(self):
        """CST with rotational scaling reports DINERT."""
        conn = np.array([[0, 1, 2, 3]])
        elem_mass = np.array([4.0])
        dt_iner = np.array([0.01])
        group = _Group(conn, elem_mass, dt_iner=dt_iner)
        model = _Model(numnod=4, mass=np.full(4, 100.0),
                       groups={"shells": group})
        model.inertia = np.full(4, 0.01)

        controls = _Controls(dt_noda="CST", dt_scale=1.0, dt_min=0.05)
        noda = NodalTimeStep(model, controls, _make_log())

        dt_e = np.array([0.001])
        noda.assemble([dt_e])

        mass_eff = model.mass.copy()
        inv_mass = 1.0 / mass_eff
        v = np.zeros((4, 3))
        inv_inertia = 1.0 / model.inertia.copy()

        noda.apply(mass_eff, inv_mass, v, t=0.0,
                   inertia=model.inertia, inv_inertia=inv_inertia)

        out_log = _make_log()
        noda.summary(out_log)

        text = out_log._listing.getvalue()
        assert "ADDED INERTIA" in text


# ======================================================================
# Constructor edge cases
# ======================================================================

class TestConstructorEdgeCases:
    """NodalTimeStep.__init__ edge cases."""

    def test_cst_zero_dt_min_warns(self):
        """CST with dt_min=0 issues a warning."""
        model = _Model(numnod=4, mass=np.ones(4), groups={})
        log = _make_log()
        NodalTimeStep(model, _Controls(dt_noda="CST", dt_min=0.0), log)
        assert len(log.warnings) == 1
        assert "dT_min is zero" in log.warnings[0]

    def test_mass0_excludes_huge_mass_nodes(self):
        """mass0 only counts physical nodes (mass < 1e29)."""
        mass = np.array([1.0, 2.0, 3.0, 1e30])  # node 3 is a placeholder
        model = _Model(numnod=4, mass=mass, groups={})
        noda = NodalTimeStep(model, _Controls(), _make_log())
        assert noda.mass0 == pytest.approx(6.0, rel=1e-12)

    def test_empty_model_no_groups(self):
        """A model with no element groups creates a valid NodalTimeStep."""
        model = _Model(numnod=2, mass=np.ones(2), groups={})
        noda = NodalTimeStep(model, _Controls(), _make_log())
        assert len(noda._shares) == 0
        assert noda._rot is False
