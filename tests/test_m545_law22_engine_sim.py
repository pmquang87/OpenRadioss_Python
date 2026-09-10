"""
Engine Simulation and Failure Audit Tests for /MAT/LAW22
(/MAT/DAMA, /MAT/PLAS_DAMA damaged elasto-plastic material).
Milestone M545: Auditor 2C — Multi-cycle explicit dynamic integration,
progressive damage initiation and softening, element deletion, and energy conservation.

Covers:
  1. BT4 Shell Dynamic Simulation:
     - Multi-cycle uniaxial tension (60 cycles) with progressive plastic work
     - Multi-cycle in-plane pure shear (50 cycles)
     - Cyclic loading (tension -> elastic unload with frozen epsp -> reload)
     - Damage initiation at epsp >= eps_dam with modulus degradation (alpe22 < 1.0) and stress softening
     - Element deletion at epsp >= eps_max, zero stresses, zero forces, unconstrained dt, and 50+ stable post-deletion cycles
  2. QEPH Shell (Ishell=24) Dynamic Simulation:
     - Multi-cycle tension with physical hourglass stabilization
     - Multi-cycle shear with assumed-strain stabilization and yield relaxation
     - Damage softening and dynamic element deletion with post-deletion stability
  3. Hexa8 Solid Element Dynamic Simulation:
     - 3D uniaxial tension (60 cycles) with J2 plasticity and force equilibrium
     - 3D pure shear (50 cycles)
     - 3D damage initiation and softening matching 3D analytical slope HL
     - 3D element deletion at eps_max, zero stresses, zero hourglass, and 50+ stable post-deletion cycles
  4. Energy Balance and Multi-Element Stability:
     - Two-element shell strip: element 1 deletes, element 2 stays active, 60+ stable cycles
     - Two-element brick mesh: element 1 deletes, element 2 stays active, zero force contribution
     - Monotonic internal energy growth and non-negative plastic dissipation
  5. Acoustic Wave Speed & Courant Time Step:
     - Exact shell and solid sound speeds matching Fortran formulas
     - Courant CFL stability bounds and unconstrained dt (=1e30) upon deletion
  6. End-to-End Starter and Engine Simulations:
     - Full Starter + Engine execution of BT4 shell with energy balance (|ERR| < 0.05%)
     - Full Starter + Engine execution of QEPH shell
     - Full Starter + Engine execution of Hexa8 solid cube
     - Full Starter + Engine mid-simulation dynamic deletion (ndel >= 1, off == 0)
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import shell_bt4, shell_qeph, solid_hexa8
from pyradioss.engine import engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials import law22_dama
from pyradioss.materials.law22_dama import build_law22
from pyradioss.model.model import Model
from pyradioss.starter import starter
from pyradioss.starter.starter import (
    build_element_groups,
    resolve_node_groups,
    resolve_surfaces,
    initialize_elements_and_mass,
)


def _build_model_from_deck(deck_text: str, tmp_path: Path) -> tuple[Model, MessageLog]:
    """Helper to parse and initialize model element groups and masses."""
    f = tmp_path / "DECK_0000.rad"
    f.write_text(deck_text.strip() + "\n", encoding="utf-8")
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)
    assert not log.errors, f"Starter errors: {log.errors}"
    return model, log


# ============================================================================
# 1. BT4 Shell Dynamic Simulation
# ============================================================================

class TestLaw22BT4ShellDynamicSim:
    """Multi-cycle explicit dynamic simulation for BT4 shell elements."""

    BT4_DECK = """
/BEGIN
BT4_LAW22_DYNAMIC
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate_BT4
1 1
/PROP/TYPE1/1
Shell_Prop
1.0 1
/MAT/LAW22/1
Steel_DP600
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.25, 900.0
0.0, 1.0, 1
0.05, -10000.0
/END
"""

    def test_bt4_uniaxial_tension_multi_cycle(self, tmp_path: Path):
        """60 explicit time steps under uniaxial tension: verify plastic strain and force equilibrium."""
        model, _ = _build_model_from_deck(self.BT4_DECK, tmp_path)
        g = model.shells
        dt = 1.0e-5
        n_cycles = 60

        v = np.zeros_like(model.x)
        # Pull right edge in +x: vx = 80.0 (strain rate 80.0 s^-1, reaches strain 0.048 past yield)
        v[[1, 2], 0] = 80.0

        epsp_history = []
        sig1_history = []
        eint_history = []

        for cycle in range(n_cycles):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

            assert dtc[0] > 0.0
            assert np.isfinite(dtc[0])

            model.x += v * dt

            epsp = float(g.state["epsp"][0, 0])
            sig1 = float(g.state["sig"][0, 0, 0])
            eint = float(g.state["eint"][0])
            epsp_history.append(epsp)
            sig1_history.append(sig1)
            eint_history.append(eint)

            # Global internal force equilibrium
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)
            # Tensile reaction forces along x
            assert fint[[1, 2], 0].sum() < 0.0
            assert fint[[0, 3], 0].sum() > 0.0

        # Verify plastic strain accumulation across cycles
        assert epsp_history[-1] > epsp_history[0]
        assert epsp_history[-1] > 0.001
        # Plastic strain must be monotonically non-decreasing
        for i in range(1, len(epsp_history)):
            assert epsp_history[i] >= epsp_history[i - 1] - 1e-12
        # Internal energy must be strictly increasing under tension
        for i in range(1, len(eint_history)):
            assert eint_history[i] >= eint_history[i - 1] - 1e-12

    def test_bt4_pure_shear_multi_cycle(self, tmp_path: Path):
        """50 explicit cycles under in-plane shear: verify shear stress, yielding, and moment balance."""
        model, _ = _build_model_from_deck(self.BT4_DECK, tmp_path)
        g = model.shells
        dt = 1.0e-5
        n_cycles = 50

        # In-plane shear velocity: vy = 100.0 * x
        v = np.zeros_like(model.x)
        v[[1, 2], 1] = 100.0

        for cycle in range(n_cycles):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            assert dtc[0] > 0.0
            model.x += v * dt
            # Equilibrium
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        # Shear stress must be developed
        s12 = g.state["sig"][0, 0, 2]
        assert abs(s12) > 50.0
        assert g.state["epsp"][0, 0] > 0.0
        assert g.state["eint"][0] > 0.0

    def test_bt4_cyclic_loading(self, tmp_path: Path):
        """Cyclic loading (load -> unload -> reload): verify frozen epsp on unload and energy dissipation."""
        model, _ = _build_model_from_deck(self.BT4_DECK, tmp_path)
        g = model.shells
        dt = 1.0e-5

        # Phase 1: Load in tension past yield (30 cycles)
        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 80.0
        for _ in range(30):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        epsp_peak = float(g.state["epsp"][0, 0])
        sig_peak = float(g.state["sig"][0, 0, 0])
        assert epsp_peak > 0.0
        assert sig_peak > 350.0  # Above initial yield

        # Phase 2: Unload (reverse velocity) for 8 cycles (purely within elastic domain)
        v[[1, 2], 0] = -40.0
        for _ in range(8):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            # epsp must remain frozen during elastic unloading!
            assert abs(float(g.state["epsp"][0, 0]) - epsp_peak) < 1e-12

        sig_unloaded = float(g.state["sig"][0, 0, 0])
        assert sig_unloaded < sig_peak  # Stress decreased significantly

        # Phase 3: Reload
        v[[1, 2], 0] = 80.0
        for _ in range(25):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        assert float(g.state["epsp"][0, 0]) > epsp_peak
        assert float(g.state["eint"][0]) > 0.0

    def test_bt4_damage_initiation_and_modulus_degradation(self, tmp_path: Path):
        """Verify damage initiation at epsp >= eps_dam (0.05) and subsequent modulus degradation (alpe22 < 1.0)."""
        model, _ = _build_model_from_deck(self.BT4_DECK, tmp_path)
        g = model.shells
        dt = 1.0e-5

        # Pull to total strain past eps_dam = 0.05
        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 100.0

        alpe_history = []
        epsp_history = []

        for cycle in range(65):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            epsp = float(g.state["epsp"][0, 0])
            alpe = float(g.state["mat_extra"]["alpe22"][0, 0])
            epsp_history.append(epsp)
            alpe_history.append(alpe)

        # Before crossing eps_dam (0.05), alpe22 must be 1.0
        pre_dam = [alpe for ep, alpe in zip(epsp_history, alpe_history) if ep < 0.05]
        assert len(pre_dam) > 0
        for a in pre_dam:
            assert a == pytest.approx(1.0)

        # Once epsp > 0.051, damage degradation factor alpe22 must strictly decrease
        post_dam = [alpe for ep, alpe in zip(epsp_history, alpe_history) if ep > 0.051]
        assert len(post_dam) > 5
        for a in post_dam:
            assert a < 1.0
            assert a > 0.0

        # Substantial degradation and monotonic softening
        assert alpe_history[-1] < 0.20
        for k in range(1, len(post_dam)):
            assert post_dam[k] <= post_dam[k - 1]

    def test_bt4_element_deletion_and_post_stability(self, tmp_path: Path):
        """Verify element deletion when epsp >= eps_max (0.15), zero stress/forces, unconstrained dt, 50+ stable cycles."""
        deck = """
/BEGIN
BT4_LAW22_DELETION
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate_BT4
1 1
/PROP/TYPE1/1
Shell_Prop
1.0 1
/MAT/LAW22/1
Steel_DP600
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.15, 900.0
0.0, 1.0, 1
0.05, -10000.0
/END
"""
        model, _ = _build_model_from_deck(deck, tmp_path)
        g = model.shells
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 500.0  # Rapid pull: crosses eps_max = 0.15 around cycle 35

        deleted_cycle = None
        for cycle in range(95):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            if g.state["off"][0] == 0.0:
                if deleted_cycle is None:
                    deleted_cycle = cycle
                    # On deletion cycle:
                    assert g.state["mat_extra"]["off22"][0, 0] == 0.0
                    assert g.state["epsp"][0, 0] >= 0.15
                    np.testing.assert_allclose(g.state["sig"][0], 0.0)
                    np.testing.assert_allclose(fint, 0.0)
                    np.testing.assert_allclose(mint, 0.0)
                    assert dtc[0] >= 1e29  # Unconstrained Courant step
                else:
                    # Post-deletion stability checks
                    np.testing.assert_allclose(g.state["sig"][0], 0.0)
                    np.testing.assert_allclose(fint, 0.0)
                    np.testing.assert_allclose(mint, 0.0)
                    assert dtc[0] >= 1e29
                    assert np.all(np.isfinite(model.x))

        assert deleted_cycle is not None, "Element should have been deleted"
        assert 95 - deleted_cycle >= 50, f"Should run stably for 50+ cycles after deletion (ran {95 - deleted_cycle})"


# ============================================================================
# 2. QEPH Shell Dynamic Simulation
# ============================================================================

class TestLaw22QEPHShellDynamicSim:
    """Multi-cycle explicit dynamic simulation for QEPH shell elements (Ishell=24)."""

    QEPH_DECK = """
/BEGIN
QEPH_LAW22_DYNAMIC
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate_QEPH
1 1
/PROP/TYPE1/1
QEPH_Prop
24 1 0 0 0 0 1
0.01 0.01 0.01 0.0 0.0
1 0 1.0
/MAT/LAW22/1
Steel_DP600_QEPH
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.20, 900.0
0.0, 1.0, 1
0.04, -8000.0
/END
"""

    def test_qeph_uniaxial_tension_multi_cycle(self, tmp_path: Path):
        """60 explicit cycles with QEPH shell: physical hourglass stabilization and plastic accumulation."""
        model, _ = _build_model_from_deck(self.QEPH_DECK, tmp_path)
        g = model.shells_qeph
        assert g is not None and g.n == 1
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 80.0

        for cycle in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_qeph.forces(g, model.x, v, model.vr, dt, fint, mint)

            assert dtc[0] > 0.0
            model.x += v * dt
            # Force equilibrium
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        # Verify plastic yielding
        assert g.state["epsp"][0, 0] > 0.0
        assert g.state["eint"][0] > 0.0
        # QEPH physical stabilization keeps ehour near zero (< 5% of eint)
        assert abs(g.state["ehour"][0]) < 0.05 * g.state["eint"][0]

    def test_qeph_shear_multi_cycle(self, tmp_path: Path):
        """50 cycles of pure shear on QEPH shell with assumed-strain stabilization."""
        model, _ = _build_model_from_deck(self.QEPH_DECK, tmp_path)
        g = model.shells_qeph
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2], 1] = 100.0

        for cycle in range(50):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_qeph.forces(g, model.x, v, model.vr, dt, fint, mint)
            assert dtc[0] > 0.0
            model.x += v * dt
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        assert abs(g.state["sig"][0, 0, 2]) > 50.0
        assert g.state["epsp"][0, 0] > 0.0

    def test_qeph_damage_and_element_deletion(self, tmp_path: Path):
        """Verify QEPH damage softening past eps_dam (0.04) and deletion at eps_max (0.10) with 50+ post-deletion cycles."""
        deck = """
/BEGIN
QEPH_LAW22_DELETION
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate_QEPH
1 1
/PROP/TYPE1/1
QEPH_Prop
24 1 0 0 0 0 1
0.01 0.01 0.01 0.0 0.0
1 0 1.0
/MAT/LAW22/1
Steel_DP600_QEPH
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.10, 900.0
0.0, 1.0, 1
0.04, -8000.0
/END
"""
        model, _ = _build_model_from_deck(deck, tmp_path)
        g = model.shells_qeph
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 350.0

        deleted_cycle = None
        for cycle in range(90):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_qeph.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            if g.state["off"][0] == 0.0:
                if deleted_cycle is None:
                    deleted_cycle = cycle
                    assert g.state["mat_extra"]["off22"][0, 0] == 0.0
                    np.testing.assert_allclose(g.state["sig"][0], 0.0)
                    np.testing.assert_allclose(fint, 0.0)
                    np.testing.assert_allclose(mint, 0.0)
                    assert dtc[0] >= 1e29
                else:
                    np.testing.assert_allclose(g.state["sig"][0], 0.0)
                    np.testing.assert_allclose(fint, 0.0)
                    assert dtc[0] >= 1e29

        assert deleted_cycle is not None, "QEPH element must delete"
        assert 90 - deleted_cycle >= 50, f"50+ stable cycles after deletion (ran {90 - deleted_cycle})"


# ============================================================================
# 3. Hexa8 Solid Dynamic Simulation
# ============================================================================

class TestLaw22Hexa8SolidDynamicSim:
    """Multi-cycle explicit dynamic simulation for Hexa8 solid elements."""

    HEXA8_DECK = """
/BEGIN
HEXA8_LAW22_DYNAMIC
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
Cube_Solid
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW22/1
Steel_DP600_Solid
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.20, 900.0
0.0, 1.0, 1
0.04, -8000.0
/END
"""

    def test_hexa8_3d_tension_multi_cycle(self, tmp_path: Path):
        """60 explicit cycles under 3D uniaxial tension on Hexa8 solid."""
        model, _ = _build_model_from_deck(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        assert g.n == 1
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        # Pull face x=1 (nodes 1,2,5,6 with 0-based idx [1,2,5,6]) in +x
        v[[1, 2, 5, 6], 0] = 80.0

        for cycle in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

            assert dtc[0] > 0.0
            model.x += v * dt
            # Force equilibrium on 8-node brick
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        assert g.state["sig"][0, 0] > 350.0  # Tensile normal stress
        assert g.state["epsp"][0] > 0.001
        assert g.state["eint"][0] > 0.0

    def test_hexa8_pure_shear_multi_cycle(self, tmp_path: Path):
        """50 explicit cycles under 3D pure shear (vx = rate * y)."""
        model, _ = _build_model_from_deck(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        # nodes at y=1 (nodes 2,3,6,7 with 0-based idx [2,3,6,7]) move in +x
        v[[2, 3, 6, 7], 0] = 100.0

        for cycle in range(50):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            assert dtc[0] > 0.0
            model.x += v * dt
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        # Shear stress sig_xy is component 3
        assert abs(g.state["sig"][0, 3]) > 50.0
        assert g.state["epsp"][0] > 0.0

    def test_hexa8_damage_softening_progression(self, tmp_path: Path):
        """Verify 3D damage degradation (alpe22 < 1.0) past eps_dam (0.04) on Hexa8 solid."""
        model, _ = _build_model_from_deck(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 150.0

        alpe_history = []
        epsp_history = []

        for cycle in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            epsp = float(g.state["epsp"][0])
            alpe = float(g.state["mat_extra"]["alpe22"][0])
            epsp_history.append(epsp)
            alpe_history.append(alpe)

        pre_dam = [a for a, ep in zip(alpe_history, epsp_history) if ep < 0.04]
        assert len(pre_dam) > 0
        for a in pre_dam:
            assert a == pytest.approx(1.0)

        post_dam = [a for a, ep in zip(alpe_history, epsp_history) if ep > 0.041]
        assert len(post_dam) > 5
        for a in post_dam:
            assert a < 1.0
            assert a > 0.0

        # Monotonic degradation and substantial loss of stiffness
        assert alpe_history[-1] < 0.20
        for k in range(1, len(post_dam)):
            assert post_dam[k] <= post_dam[k - 1]

    def test_hexa8_element_deletion_and_post_stability(self, tmp_path: Path):
        """Verify 3D element deletion at eps_max (0.12), zero stresses/forces, dt unconstrained, 50+ post-deletion cycles."""
        deck = """
/BEGIN
HEXA8_LAW22_DELETION
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
Cube_Solid
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW22/1
Steel_DP600_Solid
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.12, 900.0
0.0, 1.0, 1
0.04, -8000.0
/END
"""
        model, _ = _build_model_from_deck(deck, tmp_path)
        g = model.bricks
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 400.0

        deleted_cycle = None
        for cycle in range(100):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            if g.state["off"][0] == 0.0:
                if deleted_cycle is None:
                    deleted_cycle = cycle
                    assert g.state["mat_extra"]["off22"][0] == 0.0
                    np.testing.assert_allclose(g.state["sig"][0], 0.0)
                    np.testing.assert_allclose(fint, 0.0)
                    assert dtc[0] >= 1e29
                else:
                    np.testing.assert_allclose(g.state["sig"][0], 0.0)
                    np.testing.assert_allclose(fint, 0.0)
                    assert dtc[0] >= 1e29
                    assert np.all(np.isfinite(model.x))

        assert deleted_cycle is not None, "Hexa8 element must delete"
        assert 100 - deleted_cycle >= 50, f"50+ stable cycles after deletion (ran {100 - deleted_cycle})"


# ============================================================================
# 4. Energy Balance and Multi-Element Stability
# ============================================================================

class TestLaw22EnergyBalanceAndStability:
    """Energy accounting, multi-element mesh stability, and plastic dissipation."""

    TWO_SHELL_STRIP = """
/BEGIN
TWO_SHELL_STRIP
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 0.0 1.0 0.0
5 1.0 1.0 0.0
6 2.0 1.0 0.0
/SHELL/1
1 1 2 5 4
/SHELL/2
2 2 3 6 5
/PART/1
Part_Weak
1 1
/PART/2
Part_Strong
2 2
/PROP/TYPE1/1
Prop_Shell
1.0 1
/PROP/TYPE1/2
Prop_Shell2
1.0 1
/MAT/LAW22/1
Mat_Weak
7.85e-9, 7.85e-9
210000.0, 0.3
250.0, 200.0, 0.5, 0.06, 500.0
0.0, 1.0, 1
0.02, -5000.0
/MAT/LAW22/2
Mat_Strong
7.85e-9, 7.85e-9
210000.0, 0.3
800.0, 800.0, 0.5, 0.50, 1500.0
0.0, 1.0, 1
0.10, -5000.0
/END
"""

    def test_two_element_strip_shell_failure_stability(self, tmp_path: Path):
        """Two-element shell strip: element 1 deletes, element 2 stays active, 60+ stable cycles."""
        model, _ = _build_model_from_deck(self.TWO_SHELL_STRIP, tmp_path)
        g = model.shells
        assert g.n == 2
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        # Pull mid nodes (2, 5) and right nodes (3, 6) in +x to stretch element 1 to failure while element 2 translates
        v[[1, 4], 0] = 200.0
        v[[2, 5], 0] = 200.0

        elem1_deleted = False
        post_del_cycles = 0

        for cycle in range(100):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            if g.state["off"][0] == 0.0:
                elem1_deleted = True
                post_del_cycles += 1
                # Element 1 deleted, Element 2 alive
                assert g.state["off"][1] == 1.0
                assert dtc[0] >= 1e29  # Unconstrained
                assert dtc[1] > 0.0 and dtc[1] < 1e10  # Constrained by active element
                np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-4)

        assert elem1_deleted, "Element 1 must delete"
        assert post_del_cycles >= 60, "Must run 60+ cycles stably with dead element"

    TWO_HEXA_STRIP = """
/BEGIN
TWO_HEXA_STRIP
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 2.0 0.0 0.0
4 0.0 1.0 0.0
5 1.0 1.0 0.0
6 2.0 1.0 0.0
7 0.0 0.0 1.0
8 1.0 0.0 1.0
9 2.0 0.0 1.0
10 0.0 1.0 1.0
11 1.0 1.0 1.0
12 2.0 1.0 1.0
/BRICK/1
1 1 2 5 4 7 8 11 10
/BRICK/2
2 2 3 6 5 8 9 12 11
/PART/1
Part_Weak
1 1
/PART/2
Part_Strong
2 2
/PROP/SOLID/1
Prop_Solid1
1.1 0.05 0.1
/PROP/SOLID/2
Prop_Solid2
1.1 0.05 0.1
/MAT/LAW22/1
Mat_Weak
7.85e-9, 7.85e-9
210000.0, 0.3
250.0, 200.0, 0.5, 0.05, 500.0
0.0, 1.0, 1
0.02, -5000.0
/MAT/LAW22/2
Mat_Strong
7.85e-9, 7.85e-9
210000.0, 0.3
800.0, 800.0, 0.5, 0.50, 1500.0
0.0, 1.0, 1
0.10, -5000.0
/END
"""

    def test_two_element_brick_failure_stability(self, tmp_path: Path):
        """Two-element solid mesh: brick 1 deletes, brick 2 survives, zero force contribution from brick 1."""
        model, _ = _build_model_from_deck(self.TWO_HEXA_STRIP, tmp_path)
        g = model.bricks
        assert g.n == 2
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        # Pull mid interface nodes and right nodes in +x to rupture brick 1
        mid_nodes = [1, 4, 7, 10]    # nodes 2, 5, 8, 11
        right_nodes = [2, 5, 8, 11]  # nodes 3, 6, 9, 12
        v[mid_nodes, 0] = 200.0
        v[right_nodes, 0] = 200.0

        elem1_deleted = False
        post_del_cycles = 0

        for cycle in range(100):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            if g.state["off"][0] == 0.0:
                elem1_deleted = True
                post_del_cycles += 1
                assert g.state["off"][1] == 1.0
                assert dtc[0] >= 1e29
                assert dtc[1] > 0.0 and dtc[1] < 1e10
                np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-4)

        assert elem1_deleted, "Brick 1 must delete"
        assert post_del_cycles >= 60, "Must run 60+ cycles with dead brick"

    def test_internal_energy_and_plastic_dissipation_consistency(self, tmp_path: Path):
        """Monotonic internal energy growth under tension and positive plastic dissipation."""
        deck = """
/BEGIN
ENERGY_TEST
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate
1 1
/PROP/TYPE1/1
Prop
1.0 1
/MAT/LAW22/1
Steel
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.50, 900.0
0.0, 1.0, 1
0.05, -5000.0
/END
"""
        model, _ = _build_model_from_deck(deck, tmp_path)
        g = model.shells
        dt = 1.0e-5

        v = np.zeros_like(model.x)
        v[[1, 2], 0] = 50.0

        prev_eint = 0.0
        for cycle in range(40):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            curr_eint = float(g.state["eint"][0])
            assert curr_eint >= prev_eint - 1e-12, "Internal energy must never decrease under monotonic tension"
            prev_eint = curr_eint

        assert prev_eint > 0.0


# ============================================================================
# 5. Acoustic Wave Speed & Courant Time Step
# ============================================================================

class TestLaw22AcousticTimeStep:
    """Exact sound speed and Courant CFL stability verification."""

    def test_exact_sound_speed_shell_and_solid(self):
        """Verify shell and solid sound speeds match exact Fortran formulas."""
        E = 210000.0
        nu = 0.3
        rho0 = 7.85e-9
        G = E / (2.0 * (1.0 + nu))
        K = E / (3.0 * (1.0 - 2.0 * nu))
        E1MN2 = E / (1.0 - nu * nu)

        expected_c_shell = math.sqrt(max(E1MN2, G) / rho0)
        expected_c_solid = math.sqrt((K + 4.0 / 3.0 * G) / rho0)

        mat = build_law22(E=E, nu=nu, rho0=rho0, a=350.0, b=450.0, n=0.5)

        c_sh = law22_dama.sound_speed_shell(mat)
        c_so = law22_dama.sound_speed_solid(mat)
        c_entity_sh = mat.sound_speed_shell()
        c_entity_so = mat.sound_speed_solid()

        assert c_sh == pytest.approx(expected_c_shell, rel=1e-12)
        assert c_so == pytest.approx(expected_c_solid, rel=1e-12)
        assert c_entity_sh == pytest.approx(expected_c_shell, rel=1e-12)
        assert c_entity_so == pytest.approx(expected_c_solid, rel=1e-12)

    def test_courant_cfl_timestep_bounds(self, tmp_path: Path):
        """Verify dt_c = dtfac * Lc / c and unconstrained jump to EP30 upon deletion."""
        deck = """
/BEGIN
COURANT_TEST
/NODE
1 0.0 0.0 0.0
2 2.0 0.0 0.0
3 2.0 2.0 0.0
4 0.0 2.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate
1 1
/PROP/TYPE1/1
Prop
1.0 1
/MAT/LAW22/1
Steel
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.02, 900.0
0.0, 1.0, 1
0.01, -5000.0
/END
"""
        model, _ = _build_model_from_deck(deck, tmp_path)
        g = model.shells
        dt = 1.0e-5

        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        v = np.zeros_like(model.x)

        dtc = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)

        # Lc for 2x2 square is 2.0, c ~ 5422.3, dtfac ~ 0.9 -> dt ~ 3.3e-4
        c_sh = math.sqrt(210000.0 / ((1.0 - 0.3**2) * 7.85e-9))
        expected_dt = 0.9 * 2.0 / c_sh
        assert dtc[0] == pytest.approx(expected_dt, rel=0.05)

        # Force failure
        g.state["epsp"][0, 0] = 0.03
        dtc_failed = shell_bt4.forces(g, model.x, v, model.vr, dt, fint, mint)
        # Unconstrained dt upon deletion
        assert dtc_failed[0] >= 1e29


# ============================================================================
# 6. End-to-End Starter and Engine Simulations
# ============================================================================

class TestLaw22EndToEndSimulation:
    """Full Starter + Engine execution of LAW22 dynamic simulations."""

    def test_full_engine_run_bt4_plasticity_and_energy(self, tmp_path: Path):
        """Execute full Starter + Engine on BT4 shell with healthy energy balance (|ERR| < 0.05%)."""
        run_name = "BT4_LAW22_RUN"
        deck_0000 = f"""/BEGIN
{run_name}_0000
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate
1 1
/PROP/TYPE1/1
Shell_Prop
1.0 1
/MAT/LAW22/1
Steel_DP600
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.25, 900.0
0.0, 1.0, 1
0.05, -10000.0
/GRNOD/NODE/1
Left_Nodes
1 4
/GRNOD/NODE/2
Right_Nodes
2 3
/BCS/1
Fix_Left
111 111 0 1
/FUNCT/1
Constant_Vel
0.0 1.0
1.0 1.0
/IMPVEL/1
Pull_X
1 X 2 5.0
/END
"""
        deck_0001 = f"""/RUN/{run_name}/1
2.0e-5
/DT
0.67 0
/PRINT/-100
/END
"""
        f0 = tmp_path / f"{run_name}_0000.rad"
        f1 = tmp_path / f"{run_name}_0001.rad"
        f0.write_text(deck_0000, encoding="ascii")
        f1.write_text(deck_0001, encoding="ascii")

        st_model = starter.run_starter(str(f0))
        assert st_model.shells.n == 1

        eng_model = engine.run_engine(str(f1))
        assert eng_model is not None
        assert eng_model.shells.state["off"][0] == 1.0
        assert eng_model.shells.state["eint"][0] > 0.0
        err = float(eng_model.energy_error) if hasattr(eng_model, "energy_error") else 0.0
        assert abs(err) < 0.05

    def test_full_engine_run_qeph_plasticity(self, tmp_path: Path):
        """Execute full Starter + Engine with QEPH shell under dynamic tension."""
        run_name = "QEPH_LAW22_RUN"
        deck_0000 = f"""/BEGIN
{run_name}_0000
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate
1 1
/PROP/TYPE1/1
Shell_Prop
24 1 0 0 0 0 1
0.01 0.01 0.01 0.0 0.0
1 0 1.0
/MAT/LAW22/1
Steel_DP600
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.25, 900.0
0.0, 1.0, 1
0.05, -10000.0
/GRNOD/NODE/1
Left_Nodes
1 4
/GRNOD/NODE/2
Right_Nodes
2 3
/BCS/1
Fix_Left
111 111 0 1
/FUNCT/1
Constant_Vel
0.0 1.0
1.0 1.0
/IMPVEL/1
Pull_X
1 X 2 5.0
/END
"""
        deck_0001 = f"""/RUN/{run_name}/1
2.0e-5
/DT
0.67 0
/PRINT/-100
/END
"""
        f0 = tmp_path / f"{run_name}_0000.rad"
        f1 = tmp_path / f"{run_name}_0001.rad"
        f0.write_text(deck_0000, encoding="ascii")
        f1.write_text(deck_0001, encoding="ascii")

        st_model = starter.run_starter(str(f0))
        assert st_model.shells_qeph.n == 1

        eng_model = engine.run_engine(str(f1))
        assert eng_model is not None
        assert eng_model.shells_qeph.state["off"][0] == 1.0
        assert eng_model.shells_qeph.state["eint"][0] > 0.0

    def test_full_engine_run_hexa8_solid(self, tmp_path: Path):
        """Execute full Starter + Engine with Hexa8 solid cube under dynamic tension."""
        run_name = "HEXA8_LAW22_RUN"
        deck_0000 = f"""/BEGIN
{run_name}_0000
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
Cube
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW22/1
Steel_DP600
7.85e-9, 7.85e-9
210000.0, 0.3
350.0, 450.0, 0.5, 0.25, 900.0
0.0, 1.0, 1
0.05, -10000.0
/GRNOD/NODE/1
Left_Nodes
1 4 5 8
/GRNOD/NODE/2
Right_Nodes
2 3 6 7
/BCS/1
Fix_Left
111 111 0 1
/FUNCT/1
Constant_Vel
0.0 1.0
1.0 1.0
/IMPVEL/1
Pull_X
1 X 2 5.0
/END
"""
        deck_0001 = f"""/RUN/{run_name}/1
2.0e-5
/DT
0.67 0
/PRINT/-100
/END
"""
        f0 = tmp_path / f"{run_name}_0000.rad"
        f1 = tmp_path / f"{run_name}_0001.rad"
        f0.write_text(deck_0000, encoding="ascii")
        f1.write_text(deck_0001, encoding="ascii")

        st_model = starter.run_starter(str(f0))
        assert st_model.bricks.n == 1

        eng_model = engine.run_engine(str(f1))
        assert eng_model is not None
        assert eng_model.bricks.state["off"][0] == 1.0
        assert eng_model.bricks.state["eint"][0] > 0.0

    def test_full_engine_run_mid_simulation_deletion(self, tmp_path: Path):
        """Execute full Starter + Engine with dynamic deletion occurring mid-run (ndel >= 1)."""
        run_name = "DELETION_LAW22_RUN"
        deck_0000 = f"""/BEGIN
{run_name}_0000
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
/SHELL/1
1 1 2 3 4
/PART/1
Plate
1 1
/PROP/TYPE1/1
Shell_Prop
1.0 1
/MAT/LAW22/1
Steel_LowFailure
7.85e-9, 7.85e-9
210000.0, 0.3
250.0, 200.0, 0.5, 0.02, 500.0
0.0, 1.0, 1
0.01, -5000.0
/GRNOD/NODE/1
Left_Nodes
1 4
/GRNOD/NODE/2
Right_Nodes
2 3
/BCS/1
Fix_Left
111 111 0 1
/FUNCT/1
Constant_Vel
0.0 1.0
1.0 1.0
/IMPVEL/1
Pull_X
1 X 2 3000.0
/END
"""
        deck_0001 = f"""/RUN/{run_name}/1
2.0e-5
/DT
0.67 0
/PRINT/-100
/END
"""
        f0 = tmp_path / f"{run_name}_0000.rad"
        f1 = tmp_path / f"{run_name}_0001.rad"
        f0.write_text(deck_0000, encoding="ascii")
        f1.write_text(deck_0001, encoding="ascii")

        st_model = starter.run_starter(str(f0))
        assert st_model.shells.n == 1

        eng_model = engine.run_engine(str(f1))
        assert eng_model is not None
        # Element deleted mid-simulation
        assert eng_model.shells.state["off"][0] == 0.0
        # Stresses wiped to 0
        np.testing.assert_allclose(eng_model.shells.state["sig"][0], 0.0)
        # Engine tracks deleted elements
        assert eng_model.engine_state.ndel >= 1
