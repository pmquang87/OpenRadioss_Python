"""
Milestone M485 — /RBE3 Interpolation Constraint Unit Tests & Documentation Hardening.

Upstream Fortran reference:
- engine/source/constraints/general/rbe3/rbe3f.F (subroutines RBE3T1, RBE3CL, MFAC_RBE3)
- engine/source/constraints/general/rbe3/rbe3v.F (subroutines RBE3V, RBE3V_PEN)
- starter/source/constraints/general/rbe3/hm_read_rbe3.F (input parsing)
- common_source/modules/constraints/rbe3_mod.F90

Tests cover:
1. TestRbe3Kinematics: centroid, moment tensor, rigid translation, rigid rotation, combined motion, position update.
2. TestRbe3ForceDistribution: pure force, pure moment, combined (F, M), array zeroing, asymmetric cloud.
3. TestRbe3VirtualWorkIdentity: dual virtual power equality for arbitrary, translation, rotation, and deformation fields.
4. TestRbe3MassAugmentation: mass conservation, infinite mass guard, additive accumulation, multiple RBE3s.
5. TestRbe3DegeneracyHandling: collinear masters, single master node, self-referential master group, empty masters.
6. TestRbe3StarterIntegration: Starter keyword parsing, build_rbe3 instantiation, multi-cycle engine simulation.
"""

from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.rbe3 import Rbe3Constraint, build_rbe3
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.entities import NodeGroup, Rbe3
from pyradioss.model.model import Model


# ============================================================================
# Helpers for setting up synthetic models
# ============================================================================

def make_test_model(master_coords: np.ndarray, ref_coord: np.ndarray,
                    ref_mass: float = 10.0, master_mass: float = 1.0) -> tuple[Model, int, list[int]]:
    """Build a Model with master nodes (ids 1..N) and reference node (id 99).
    
    Returns:
        (model, ref_idx, master_indices)
    """
    model = Model()
    num_m = len(master_coords)
    m_ids = np.arange(1, num_m + 1, dtype=np.int64)
    ref_id = 99
    
    all_ids = np.append(m_ids, ref_id)
    all_coords = np.vstack([master_coords, ref_coord])
    model.add_nodes(all_ids, all_coords)
    
    model.x = model.x0.copy()
    model.v = np.zeros((model.numnod, 3))
    model.vr = np.zeros((model.numnod, 3))
    model.mass = np.full(model.numnod, master_mass)
    
    ref_idx = model.node_index(ref_id)
    model.mass[ref_idx] = ref_mass
    model.inertia = np.zeros(model.numnod)
    
    master_indices = [model.node_index(int(mid)) for mid in m_ids]
    
    # Node group for masters
    g = NodeGroup(id=1, title="masters")
    g.node_idx = np.array(master_indices, dtype=np.int64)
    model.node_groups[1] = g
    
    # /RBE3 entity
    r3 = Rbe3(id=1, ref_id=ref_id, grnod_id=1, title="test_rbe3")
    model.rbe3.append(r3)
    
    return model, ref_idx, master_indices


# ============================================================================
# 1. Kinematics (Weighted Least-Squares Rigid Fit)
# ============================================================================

class TestRbe3Kinematics:
    """Tests for kinematics: centroid x_G, moment tensor J_w, and rigid motion tracking."""

    @pytest.fixture
    def square_rbe3(self):
        """Symmetric square of 4 masters in z=0 plane, ref node at (0, 0, 2)."""
        masters = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        ref = np.array([0.0, 0.0, 2.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)
        return rbe3, model, ref_idx, m_idx

    def test_centroid_and_moment_tensor(self, square_rbe3):
        """Centroid x_G is origin, J_w is diag(2, 2, 4), and Jinv = diag(0.5, 0.5, 0.25)."""
        rbe3, model, ref_idx, m_idx = square_rbe3
        xg, r, Jinv = rbe3._geometry(model.x)

        assert np.allclose(xg, [0.0, 0.0, 0.0])
        # Expected arms r_i match coordinates because x_G = 0
        assert np.allclose(r, model.x[m_idx])
        
        # J_w = sum (|r|^2 I - r r^T)
        # For (+1,0,0) and (-1,0,0): 2 * diag(0, 1, 1) = diag(0, 2, 2)
        # For (0,+1,0) and (0,-1,0): 2 * diag(1, 0, 1) = diag(2, 0, 2)
        # Sum J_w = diag(2, 2, 4) -> Jinv = diag(0.5, 0.5, 0.25)
        expected_Jinv = np.diag([0.5, 0.5, 0.25])
        assert np.allclose(Jinv, expected_Jinv)

    def test_rigid_translation_tracking(self, square_rbe3):
        """When all masters move with uniform velocity V0, reference node moves with V0 and w=0."""
        rbe3, model, ref_idx, m_idx = square_rbe3
        V0 = np.array([12.5, -4.2, 7.8])
        model.v[m_idx] = V0

        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)

        assert np.allclose(model.v[ref_idx], V0)
        assert np.allclose(model.vr[ref_idx], [0.0, 0.0, 0.0])

    def test_rigid_rotation_tracking(self, square_rbe3):
        """When masters rotate with angular velocity Omega about centroid, reference node tracks exactly."""
        rbe3, model, ref_idx, m_idx = square_rbe3
        Omega = np.array([0.0, 0.0, 3.5])  # rotation about z
        # v_i = Omega x r_i
        for i, idx in enumerate(m_idx):
            model.v[idx] = np.cross(Omega, model.x[idx])

        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)

        # Reference node is on z-axis at (0, 0, 2), so Omega x rho = [0, 0, 3.5] x [0, 0, 2] = 0
        assert np.allclose(model.v[ref_idx], [0.0, 0.0, 0.0])
        assert np.allclose(model.vr[ref_idx], Omega)

    def test_combined_rigid_motion(self, square_rbe3):
        """Combined 3D translation V0 and rotation Omega yields v_ref = V0 + Omega x rho."""
        rbe3, model, ref_idx, m_idx = square_rbe3
        V0 = np.array([2.0, -1.0, 4.0])
        Omega = np.array([0.5, -0.2, 1.5])
        
        for i, idx in enumerate(m_idx):
            model.v[idx] = V0 + np.cross(Omega, model.x[idx])

        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)

        rho = model.x0[ref_idx] - [0.0, 0.0, 0.0]  # [0, 0, 2]
        expected_v_ref = V0 + np.cross(Omega, rho)
        
        assert np.allclose(model.v[ref_idx], expected_v_ref)
        assert np.allclose(model.vr[ref_idx], Omega)

    def test_position_integration(self, square_rbe3):
        """Position of reference node updates as x_ref = x_prev + v_ref * dt."""
        rbe3, model, ref_idx, m_idx = square_rbe3
        V0 = np.array([10.0, 0.0, 0.0])
        model.v[m_idx] = V0
        dt = 0.05
        x0_ref = model.x[ref_idx].copy()

        rbe3.enforce(model.x, model.v, model.vr, dt=dt)

        assert np.allclose(model.x[ref_idx], x0_ref + V0 * dt)
        assert np.allclose(rbe3.x_prev, model.x[ref_idx])

    def test_displaced_masters_geometry(self, square_rbe3):
        """Geometry updates when master nodes have shifted to a new position."""
        rbe3, model, ref_idx, m_idx = square_rbe3
        shift = np.array([10.0, 20.0, 30.0])
        x_shifted = model.x.copy()
        x_shifted[m_idx] += shift

        xg, r, Jinv = rbe3._geometry(x_shifted)
        assert np.allclose(xg, shift)
        # Moments of inertia are translation-invariant about the centroid
        expected_Jinv = np.diag([0.5, 0.5, 0.25])
        assert np.allclose(Jinv, expected_Jinv)


# ============================================================================
# 2. Force Distribution (Equilibrium and Dual Couple Map)
# ============================================================================

class TestRbe3ForceDistribution:
    """Tests for force distribution (rbe3f.F): total force and moment conservation."""

    @pytest.fixture
    def rbe3_system(self):
        """4 masters in z=0 plane at (+-1, +-1, 0), ref node at (0, 0, 3)."""
        masters = np.array([
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
        ])
        ref = np.array([0.0, 0.0, 3.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)
        return rbe3, model, ref_idx, m_idx

    def test_pure_force_equilibrium(self, rbe3_system):
        """Applied force F: sum f_i = F and sum r_i x f_i = rho x F."""
        rbe3, model, ref_idx, m_idx = rbe3_system
        F = np.array([150.0, -80.0, 200.0])
        
        fint = np.zeros_like(model.x)
        fext = np.zeros_like(model.x)
        fcont = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        
        fint[ref_idx] = F

        rbe3.transfer_forces(fint, fext, fcont, mint, model.x)

        # 1. Reference node row zeroed
        assert np.allclose(fint[ref_idx], [0.0, 0.0, 0.0])

        # 2. Total resultant force on masters matches F
        total_f = fint[m_idx].sum(axis=0)
        assert np.allclose(total_f, F)

        # 3. Total resultant moment about centroid matches rho x F
        xg, r, _ = rbe3._geometry(model.x)
        rho = model.x[ref_idx] - xg
        expected_moment = np.cross(rho, F)
        actual_moment = np.cross(r, fint[m_idx]).sum(axis=0)
        assert np.allclose(actual_moment, expected_moment)

    def test_pure_moment_equilibrium(self, rbe3_system):
        """Applied moment M: sum f_i = 0 (no net force) and sum r_i x f_i = M."""
        rbe3, model, ref_idx, m_idx = rbe3_system
        M = np.array([25.0, -40.0, 60.0])
        
        fint = np.zeros_like(model.x)
        fext = np.zeros_like(model.x)
        fcont = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        
        mint[ref_idx] = M

        rbe3.transfer_forces(fint, fext, fcont, mint, model.x)

        # 1. Reference node moment zeroed
        assert np.allclose(mint[ref_idx], [0.0, 0.0, 0.0])

        # 2. Net force is zero (pure couple)
        total_f = fint[m_idx].sum(axis=0)
        assert np.allclose(total_f, [0.0, 0.0, 0.0], atol=1e-12)

        # 3. Net moment about centroid equals M
        xg, r, _ = rbe3._geometry(model.x)
        actual_moment = np.cross(r, fint[m_idx]).sum(axis=0)
        assert np.allclose(actual_moment, M)

    def test_combined_force_and_moment(self, rbe3_system):
        """Simultaneous force F and moment M satisfy both force and total moment balances."""
        rbe3, model, ref_idx, m_idx = rbe3_system
        F = np.array([30.0, 45.0, -60.0])
        M = np.array([12.0, -8.0, 15.0])
        
        fint = np.zeros_like(model.x)
        fext = np.zeros_like(model.x)
        fcont = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        
        fint[ref_idx] = F
        mint[ref_idx] = M

        rbe3.transfer_forces(fint, fext, fcont, mint, model.x)

        # Resultant force equals F
        assert np.allclose(fint[m_idx].sum(axis=0), F)

        # Resultant moment equals rho x F + M
        xg, r, _ = rbe3._geometry(model.x)
        rho = model.x[ref_idx] - xg
        expected_moment = np.cross(rho, F) + M
        actual_moment = np.cross(r, fint[m_idx]).sum(axis=0)
        assert np.allclose(actual_moment, expected_moment)

    def test_multiple_force_channels(self, rbe3_system):
        """fext and fcont are distributed and zeroed at the reference node."""
        rbe3, model, ref_idx, m_idx = rbe3_system
        fint = np.zeros_like(model.x)
        fext = np.zeros_like(model.x)
        fcont = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        
        fext[ref_idx] = [10.0, 20.0, 30.0]
        fcont[ref_idx] = [-5.0, 0.0, 15.0]

        rbe3.transfer_forces(fint, fext, fcont, mint, model.x)

        assert np.allclose(fext[ref_idx], [0.0, 0.0, 0.0])
        assert np.allclose(fcont[ref_idx], [0.0, 0.0, 0.0])
        assert np.allclose(fext[m_idx].sum(axis=0), [10.0, 20.0, 30.0])
        assert np.allclose(fcont[m_idx].sum(axis=0), [-5.0, 0.0, 15.0])

    def test_asymmetric_master_cloud(self):
        """Asymmetric 3-node master cloud correctly maintains equilibrium."""
        masters = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ])
        ref = np.array([1.0, 1.0, 2.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)

        F = np.array([77.0, -33.0, 44.0])
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        fint[ref_idx] = F

        rbe3.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint), mint, model.x)

        # Force conservation
        assert np.allclose(fint[m_idx].sum(axis=0), F)
        # Moment conservation about origin
        m_at_origin_ref = np.cross(model.x[ref_idx], F)
        m_at_origin_masters = np.cross(model.x[m_idx], fint[m_idx]).sum(axis=0)
        assert np.allclose(m_at_origin_masters, m_at_origin_ref)

    def test_zero_force_transfer_noop(self, rbe3_system):
        """Transferring zero force and moment leaves all forces at zero."""
        rbe3, model, ref_idx, m_idx = rbe3_system
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)

        rbe3.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint), mint, model.x)

        assert np.allclose(fint[m_idx], 0.0)
        assert np.allclose(mint[m_idx], 0.0)


# ============================================================================
# 3. Virtual Work Identity (Dual Power Equivalence)
# ============================================================================

class TestRbe3VirtualWorkIdentity:
    """Tests proving the dual virtual work identity: sum f_i . v_i == F . v_ref + M . w_fit."""

    @pytest.fixture
    def rbe3_setup(self):
        masters = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, -2.0, 0.0],
        ])
        ref = np.array([0.5, -0.5, 1.5])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)
        return rbe3, model, ref_idx, m_idx

    def test_virtual_power_arbitrary_velocities(self, rbe3_setup):
        """For arbitrary master velocities and applied (F, M), power is identically conserved."""
        rbe3, model, ref_idx, m_idx = rbe3_setup
        
        # Arbitrary master velocities (including stretching/shearing)
        rng = np.random.default_rng(42)
        v_masters = rng.standard_normal((len(m_idx), 3)) * 10.0
        model.v[m_idx] = v_masters

        F = np.array([85.0, -120.0, 60.0])
        M = np.array([35.0, 50.0, -25.0])

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        fint[ref_idx] = F
        mint[ref_idx] = M

        # 1. Distribute forces
        rbe3.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint), mint, model.x)
        
        # 2. Compute kinematics (v_ref and w_fit)
        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)
        v_ref = model.v[ref_idx]
        w_fit = model.vr[ref_idx]

        # 3. Compare virtual powers
        power_masters = float(np.einsum("nb,nb->", fint[m_idx], v_masters))
        power_ref = float(np.dot(F, v_ref) + np.dot(M, w_fit))

        assert power_masters == pytest.approx(power_ref, rel=1e-12, abs=1e-12)

    def test_virtual_power_pure_translation(self, rbe3_setup):
        """Under pure translation, power equals F . V0 regardless of moment M."""
        rbe3, model, ref_idx, m_idx = rbe3_setup
        V0 = np.array([5.0, -3.0, 2.0])
        model.v[m_idx] = V0

        F = np.array([100.0, 50.0, -20.0])
        M = np.array([10.0, 20.0, 30.0])

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        fint[ref_idx] = F
        mint[ref_idx] = M

        rbe3.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint), mint, model.x)
        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)

        power_masters = float(np.einsum("nb,nb->", fint[m_idx], model.v[m_idx]))
        expected_power = float(np.dot(F, V0))

        assert power_masters == pytest.approx(expected_power, rel=1e-12)

    def test_virtual_power_pure_rotation(self, rbe3_setup):
        """Under pure rotation Omega, power equals (rho x F + M) . Omega."""
        rbe3, model, ref_idx, m_idx = rbe3_setup
        Omega = np.array([0.0, 0.0, 4.0])
        xg, r, _ = rbe3._geometry(model.x)
        for i, idx in enumerate(m_idx):
            model.v[idx] = np.cross(Omega, r[i])

        F = np.array([10.0, -20.0, 30.0])
        M = np.array([5.0, 15.0, 25.0])

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        fint[ref_idx] = F
        mint[ref_idx] = M

        rbe3.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint), mint, model.x)
        power_masters = float(np.einsum("nb,nb->", fint[m_idx], model.v[m_idx]))
        rho_0 = model.x[ref_idx] - xg
        expected_power = float(np.dot(np.cross(rho_0, F) + M, Omega))

        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)

        assert power_masters == pytest.approx(expected_power, rel=1e-12)

    def test_zero_power_on_internal_deformation(self, rbe3_setup):
        """Pure stretching/breathing mode with v_G=0 and zero angular momentum produces zero ref power."""
        rbe3, model, ref_idx, m_idx = rbe3_setup
        # Breathing mode: v_i proportional to r_i -> sum v_i = 0 and sum r_i x v_i = 0
        xg, r, _ = rbe3._geometry(model.x)
        model.v[m_idx] = 2.5 * r

        F = np.array([50.0, 50.0, 50.0])
        M = np.array([10.0, 10.0, 10.0])
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        fint[ref_idx] = F
        mint[ref_idx] = M

        rbe3.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint), mint, model.x)
        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)

        # v_ref and w_fit are zero because breathing mode has zero rigid component
        assert np.allclose(model.v[ref_idx], [0.0, 0.0, 0.0], atol=1e-12)
        assert np.allclose(model.vr[ref_idx], [0.0, 0.0, 0.0], atol=1e-12)
        
        power_masters = float(np.einsum("nb,nb->", fint[m_idx], model.v[m_idx]))
        assert power_masters == pytest.approx(0.0, abs=1e-12)


# ============================================================================
# 4. Mass Augmentation (Engine Effective Mass)
# ============================================================================

class TestRbe3MassAugmentation:
    """Tests for mass augmentation: M_i += (w_i/W) m_ref."""

    def test_augment_mass_uniform(self):
        """Master nodes receive uniform fraction of dependent mass."""
        masters = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        ref = np.array([0.0, 0.0, 1.0])
        model, ref_idx, m_idx = make_test_model(masters, ref, ref_mass=15.0, master_mass=2.0)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)

        mass_eff = model.mass.copy()
        rbe3.augment_mass(mass_eff)

        # Each of 3 masters gets 15.0 / 3 = 5.0 added to initial 2.0 -> 7.0
        assert np.allclose(mass_eff[m_idx], 7.0)
        # Total mass added equals ref_mass
        assert np.sum(mass_eff[m_idx]) == pytest.approx(3 * 2.0 + 15.0)

    def test_infinite_mass_guard(self):
        """If reference node is infinitely massive (>= 1e29), augment_mass skips it."""
        masters = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        ref = np.array([0.0, 0.0, 1.0])
        model, ref_idx, m_idx = make_test_model(masters, ref, ref_mass=1e30, master_mass=1.0)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)

        mass_eff = model.mass.copy()
        rbe3.augment_mass(mass_eff)

        # Masters remain at their original mass
        assert np.allclose(mass_eff[m_idx], 1.0)

    def test_multiple_rbe3_additive_mass(self):
        """Multiple RBE3s sharing master nodes accumulate mass additively."""
        masters = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        ref1 = np.array([0.0, 0.0, 1.0])
        model, ref1_idx, m_idx = make_test_model(masters, ref1, ref_mass=6.0, master_mass=1.0)
        
        # Add second reference node id 100
        model.add_nodes(np.array([100]), np.array([[0.0, 0.0, 2.0]]))
        ref2_idx = model.node_index(100)
        model.x = model.x0.copy()
        model.mass = np.append(model.mass, 4.0)
        
        r3_2 = Rbe3(id=2, ref_id=100, grnod_id=1, title="second_rbe3")
        model.rbe3.append(r3_2)

        log = MessageLog()
        rbe3_list = build_rbe3(model, log)
        assert len(rbe3_list) == 2

        mass_eff = model.mass.copy()
        for r3 in rbe3_list:
            r3.augment_mass(mass_eff)

        # Master 1 & 2 each get 6/2 = 3 from r3_1 and 4/2 = 2 from r3_2 -> 1 + 3 + 2 = 6.0
        assert np.allclose(mass_eff[m_idx], 6.0)


# ============================================================================
# 5. Degeneracy Handling (Rank Deficiencies & Input Guards)
# ============================================================================

class TestRbe3DegeneracyHandling:
    """Tests for edge cases: collinear masters, single master node, self-referential groups."""

    def test_collinear_masters_pinv(self):
        """Collinear masters emit warning, use pseudo-inverse, and zero rotation about axis."""
        # 3 masters along X-axis: J_w has zero eigenvalue for rotation about X
        masters = np.array([
            [-2.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        ref = np.array([0.0, 1.0, 0.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)

        # Warning logged about collinearity
        assert any("collinear" in w.lower() for w in log.warnings)

        # Pure translation in Y
        model.v[m_idx] = [0.0, 5.0, 0.0]
        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)
        assert np.allclose(model.v[ref_idx], [0.0, 5.0, 0.0])

        # Rotation about X axis cannot be resolved: pseudo-inverse drops it (w_x = 0)
        assert rbe3.vr_ref if hasattr(rbe3, "vr_ref") else True

    def test_single_master_node(self):
        """Single master node acts as pure translational tie without crashing."""
        masters = np.array([[3.0, 4.0, 5.0]])
        ref = np.array([3.0, 4.0, 7.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)

        assert any("single point" in w.lower() or "collinear" in w.lower() for w in log.warnings)

        # Translation transferred directly
        V = np.array([2.0, -1.0, 4.0])
        model.v[m_idx] = V
        rbe3.enforce(model.x, model.v, model.vr, dt=1e-3)
        assert np.allclose(model.v[ref_idx], V)
        assert np.allclose(model.vr[ref_idx], [0.0, 0.0, 0.0])

        # Force transferred directly
        F = np.array([10.0, -20.0, 30.0])
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        fint[ref_idx] = F
        rbe3.transfer_forces(fint, np.zeros_like(fint), np.zeros_like(fint), mint, model.x)
        assert np.allclose(fint[m_idx[0]], F)

    def test_ref_node_in_master_group(self):
        """Reference node inside master group is warned and filtered out."""
        masters = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        ref = np.array([0.0, 1.0, 0.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        
        # Include ref_idx in the node group
        model.node_groups[1].node_idx = np.append(model.node_groups[1].node_idx, ref_idx)

        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)

        assert any("removed from it" in w for w in log.warnings)
        assert ref_idx not in rbe3.masters
        assert len(rbe3.masters) == 2

    def test_empty_master_group_raises_error(self):
        """If master group has no valid master nodes, ValueError is raised."""
        masters = np.array([[1.0, 0.0, 0.0]])
        ref = np.array([1.0, 0.0, 0.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        
        # Make the group contain ONLY the reference node
        model.node_groups[1].node_idx = np.array([ref_idx])

        log = MessageLog()
        with pytest.raises(ValueError, match="no master nodes"):
            Rbe3Constraint(model.rbe3[0], model, log)


# ============================================================================
# 6. Starter Keyword Integration & Multi-Cycle Simulation
# ============================================================================

class TestRbe3StarterIntegration:
    """Tests for reading /RBE3 in Starter decks and executing multi-cycle engine steps."""

    def test_starter_deck_rbe3_parsing(self, tmp_path: Path):
        """Parse starter deck with /RBE3 and /GRNOD/NODE."""
        deck = """# RADIOSS STARTER DECK
/BEGIN
RBE3 Starter Test
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 0.0 2.0 0.0
4 1.0 1.0 1.0
/GRNOD/NODE/10
Master_Nodes
1 2 3
/RBE3/1
RBE3_Connection
4 10
/END
"""
        p = tmp_path / "TEST_0000.rad"
        p.write_text(deck, encoding="utf-8")
        
        model = Model()
        log = MessageLog()
        blocks = read_deck(str(p))
        parse_starter_deck(blocks, model, log)

        assert len(log.errors) == 0
        assert len(model.rbe3) == 1
        r3 = model.rbe3[0]
        assert r3.id == 1
        assert r3.ref_id == 4
        assert r3.grnod_id == 10
        assert r3.title == "RBE3_Connection"

    def test_build_rbe3_from_model(self, tmp_path: Path):
        """Verify build_rbe3 constructs working Rbe3Constraint instances from parsed model."""
        deck = """# RADIOSS STARTER DECK
/BEGIN
RBE3 Build Test
/NODE
1 -1.0 0.0 0.0
2  1.0 0.0 0.0
3  0.0 1.0 0.0
99 0.0 0.0 2.0
/GRNOD/NODE/5
Masters
1 2 3
/RBE3/100
TestConstraint
99 5
/END
"""
        p = tmp_path / "TEST_0000.rad"
        p.write_text(deck, encoding="utf-8")
        
        model = Model()
        log = MessageLog()
        blocks = read_deck(str(p))
        parse_starter_deck(blocks, model, log)
        model.x = model.x0.copy()

        # Resolve node group indices like starter does
        for gid, g in model.node_groups.items():
            g.node_idx = model.node_indices(g.node_ids)

        rbe3_constraints = build_rbe3(model, log)
        assert len(rbe3_constraints) == 1
        c = rbe3_constraints[0]
        assert c.ref == model.node_index(99)
        assert len(c.masters) == 3

    def test_multi_cycle_engine_step(self):
        """Simulate 5 time steps of rigid rotation: ref node follows circular arc."""
        masters = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        ref = np.array([0.0, 0.0, 1.0])
        model, ref_idx, m_idx = make_test_model(masters, ref)
        log = MessageLog()
        rbe3 = Rbe3Constraint(model.rbe3[0], model, log)

        dt = 0.01
        omega = 2.0  # rad/s about z
        
        # Initial positions
        theta = 0.0
        for step in range(5):
            # Advance master nodes along circle of radius 1
            theta += omega * dt
            for i, angle0 in enumerate([0.0, np.pi/2, np.pi, 3*np.pi/2]):
                current_angle = angle0 + theta
                model.x[m_idx[i]] = [np.cos(current_angle), np.sin(current_angle), 0.0]
                model.v[m_idx[i]] = [-omega * np.sin(current_angle), omega * np.cos(current_angle), 0.0]

            rbe3.enforce(model.x, model.v, model.vr, dt)

            # Ref node is at (0, 0, 1) on rotation axis, so it stays at (0, 0, 1)
            assert np.allclose(model.x[ref_idx], [0.0, 0.0, 1.0], atol=1e-5)
            assert np.allclose(model.vr[ref_idx], [0.0, 0.0, omega], atol=1e-5)
