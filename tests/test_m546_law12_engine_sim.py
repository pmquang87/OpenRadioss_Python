"""
Engine Simulation and Dynamic Failure Auditor Tests for /MAT/LAW12
(/MAT/3D_COMP, /MAT/COMP_3D 3D orthotropic composite material with damage).
Milestone M546: Auditor 2C — Multi-element explicit dynamic simulations,
directional tensile cracking, unilateral cyclic crack closure, 3D Tsai-Wu
plasticity, dynamic strain rate effects, progressive degradation, and element deletion.

Covers:
  1. Hexa8 Single-Element Dynamic Explicit Simulations:
     - Multi-cycle uniaxial tension along material direction 1 with force equilibrium
     - Multi-cycle uniaxial compression with intact tensile state and negative stress
     - Cyclic unilateral loading (tension cracks element -> compression closes crack -> reload)
     - Midpoint internal energy integration exactness: dEint = 0.5 * (sig_old + sig_new) : deps * V
  2. Directional Cracking and Progressive Failure:
     - Directional cracking in dir 1 while transverse dirs 2 and 3 remain undamaged
     - Transverse tensile cracking in dir 2 with uncoupled thresholds
     - Progressive element degradation to deletion when fmax is reached (off: 1.0 -> 0.792 -> ... -> 0.0)
     - Post-deletion numerical stability for 60+ cycles with unconstrained dt and zero forces
  3. 3D Tsai-Wu Plasticity and Dynamic Rate Effects:
     - 3D pure shear plasticity with monotonic plastic work wpla accumulation
     - Multi-axial loading with consistent Tsai-Wu yield return
     - Dynamic strain rate elevation: c * ln(eps_dot / eps_dot_0)
     - Fiber reinforcement dynamic response (alpha > 0, efib > 0)
  4. Tetra4 Solids and Combined Hexa8 + Tetra4 System:
     - Tetra4 solid element multi-axial orthotropic dynamic simulation
     - Tetra4 cyclic unilateral cracking and crack closure
     - Combined Hexa8 + Tetra4 hybrid system with shared interface nodes and global equilibrium
  5. Multi-Element Wave Propagation and Global Energy Balance:
     - 4x1x1 Hexa8 bar tensile stress wave propagation with progressive front cracking
     - Multi-element explicit global work-energy balance (error < 1e-4)
  6. End-to-End Starter and Engine Simulations:
     - Full Starter + Engine execution of Hexa8 solid cube with LAW12
     - Full Starter + Engine execution with mid-run dynamic deletion (ndel >= 1, off == 0.0)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.engine import engine
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials import law12_comp3d
from pyradioss.model.model import Model
from pyradioss.starter import starter
from pyradioss.starter.starter import (
    build_element_groups,
    resolve_node_groups,
    resolve_surfaces,
    initialize_elements_and_mass,
)


def _build_model(deck_text: str, tmp_path: Path) -> tuple[Model, MessageLog]:
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
# 1. Hexa8 Single-Element Dynamic Explicit Simulations
# ============================================================================

class TestLaw12Hexa8DynamicSim:
    """Hexa8 solid single-element explicit dynamic simulations with LAW12."""

    HEXA8_DECK = """
/BEGIN
HEXA8_LAW12_DYN
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
Cube_LAW12
1 1
/PROP/SOLID/1
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Composite_3D_Hex
1.55e-9, 1.55e-9
140000.0, 10000.0, 10000.0
0.30, 0.45, 0.02
5000.0, 3500.0, 5000.0
1800.0, 40.0, 40.0, 0.08
150.0, 0.5, 1200.0, 1.0
2000.0, 50.0, 1500.0, 200.0
80.0, 80.0, 50.0, 50.0
50.0, 200.0, 80.0, 80.0
0.25, 180000.0, 0.04, 1.0, 1
/END
"""

    def test_hexa8_uniaxial_tension_multi_cycle(self, tmp_path: Path):
        """60 explicit time steps under uniaxial tension along dir 1: verify force balance and energy."""
        model, _ = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        assert g.n == 1
        dt = 1.0e-6
        n_cycles = 60

        v = np.zeros_like(model.x)
        # Pull face at x=1 (nodes 2, 3, 6, 7 with indices 1, 2, 5, 6) in +x
        pulled_nodes = [1, 2, 5, 6]
        fixed_nodes = [0, 3, 4, 7]
        v[pulled_nodes, 0] = 50.0

        for _ in range(n_cycles):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)

            assert dtc[0] > 0.0
            assert np.isfinite(dtc[0])
            model.x += v * dt

            # Global force equilibrium on internal forces
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)
            # Reaction forces: pulled face negative in fint (resisting stretch), fixed face positive
            assert fint[pulled_nodes, 0].sum() < 0.0
            assert fint[fixed_nodes, 0].sum() > 0.0

        # Normal stress in direction 1 is positive (tensile)
        sig1 = float(g.state["sig"][0, 0])
        assert sig1 > 100.0
        # Internal energy must be strictly positive and accumulated
        assert float(g.state["eint"][0]) > 0.0
        assert g.state["off"][0] == 1.0

    def test_hexa8_uniaxial_compression_multi_cycle(self, tmp_path: Path):
        """60 explicit cycles under uniaxial compression: negative normal stress, undamaged tensile states."""
        model, _ = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6
        n_cycles = 60

        v = np.zeros_like(model.x)
        pulled_nodes = [1, 2, 5, 6]
        v[pulled_nodes, 0] = -50.0

        for _ in range(n_cycles):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            assert dtc[0] > 0.0
            model.x += v * dt
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        sig1 = float(g.state["sig"][0, 0])
        assert sig1 < -100.0
        # No tensile cracking in compression
        dam = g.state["mat_extra"]["dam12"][0]
        assert dam[0] == pytest.approx(0.0)
        assert dam[1] == pytest.approx(0.0)
        assert dam[2] == pytest.approx(0.0)
        assert g.state["off"][0] == 1.0

    def test_hexa8_cyclic_unilateral_cracking_and_closure(self, tmp_path: Path):
        """Cyclic loading: tension cracks element -> compression closes crack (bearing load) -> reload."""
        deck = """
/BEGIN
HEXA8_UNILATERAL
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
Hex_Part
1 1
/PROP/SOLID/1
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Law12
1.5e-9 1.5e-9
100000.0 10000.0 10000.0
0.3 0.3 0.03
5000.0 5000.0 5000.0
200.0 1000.0 1000.0 0.10
/END
"""
        model, _ = _build_model(deck, tmp_path)
        g = model.bricks
        dt = 1.0e-6
        pulled = [1, 2, 5, 6]

        # Phase 1: Tension pulls element past sigt1=200 MPa
        v = np.zeros_like(model.x)
        v[pulled, 0] = 50.0
        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        dam1 = float(g.state["mat_extra"]["dam12"][0, 0])
        epc1 = float(g.state["mat_extra"]["epc12"][0, 0])
        assert dam1 > 0.0, "Direction 1 crack must initiate in tension"
        assert epc1 > 0.0, "Crack opening strain epc1 must be positive"

        # Phase 2: Reverse into compression (v_x = -50.0)
        # Crack closing phase: while epc1 > 0, normal stress is clamped to 0.0 (open crack carries no compression)
        # Once epc1 reaches 0.0 (crack closed), compressive stress builds up (bears compression)
        v[pulled, 0] = -50.0
        zero_stress_seen = False
        comp_stress_seen = False

        for _ in range(120):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            epc_cur = float(g.state["mat_extra"]["epc12"][0, 0])
            s1 = float(g.state["sig"][0, 0])
            if epc_cur > 0.0001:
                assert s1 == pytest.approx(0.0, abs=1e-5), "Open crack must not support compressive stress"
                zero_stress_seen = True
            elif epc_cur == 0.0 and s1 < -50.0:
                comp_stress_seen = True

        assert zero_stress_seen, "Must observe zero-stress crack closing phase"
        assert comp_stress_seen, "Must observe compressive stress after crack closure"
        assert float(g.state["mat_extra"]["epc12"][0, 0]) == pytest.approx(0.0)

        # Phase 3: Reload back into tension (v_x = +50.0)
        v[pulled, 0] = 50.0
        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            assert np.all(np.isfinite(g.state["sig"][0]))

        # Solver completed all 240 cycles stably
        assert g.state["off"][0] == 1.0

    def test_hexa8_midpoint_internal_energy_conservation(self, tmp_path: Path):
        """Verify exact midpoint internal energy integration: dEint = 0.5 * (sig_old + sig_new) : deps * V."""
        model, _ = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6
        n_cycles = 50

        pulled_nodes = [1, 2, 5, 6]
        v = np.zeros_like(model.x)
        v[pulled_nodes, 0] = 10.0

        w_ext = 0.0
        fint_old = np.zeros_like(model.x)

        for _ in range(n_cycles):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            # Midpoint boundary work
            f_mid = 0.5 * (fint_old + fint)
            dw = -np.sum(f_mid[pulled_nodes] * v[pulled_nodes]) * dt
            w_ext += dw
            fint_old = fint.copy()

        eint = float(g.state["eint"][0])
        assert eint > 0.0
        rel_err = abs(w_ext - eint) / max(w_ext, eint)
        assert rel_err < 1.0e-6, f"Midpoint energy conservation error {rel_err:.2e} must be < 1e-6"


# ============================================================================
# 2. Directional Cracking and Progressive Failure
# ============================================================================

class TestLaw12DirectionalCrackingAndDegradation:
    """Directional cracking, orthotropy, and progressive element degradation/deletion."""

    CRACKING_DECK = """
/BEGIN
CRACKING_ORTHOTROPY
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
Cube_Part
1 1
/PROP/SOLID/1
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Ortho
1.5e-9 1.5e-9
100000.0 20000.0 20000.0
0.25 0.20 0.15
8000.0 6000.0 8000.0
200.0 40.0 150.0 0.10
/END
"""

    def test_directional_tensile_cracking_orthotropy(self, tmp_path: Path):
        """Tensile loading in dir 1 generates crack in dir 1 while dir 2 and 3 remain undamaged."""
        model, _ = _build_model(self.CRACKING_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Pull along x (dir 1): vx = 50.0 on face x=1
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        dam = g.state["mat_extra"]["dam12"][0]
        # Direction 1 crack must be active
        assert dam[0] > 0.0, "Direction 1 must crack"
        # Directions 2 and 3 must remain completely undamaged
        assert dam[1] == pytest.approx(0.0), "Direction 2 must remain undamaged"
        assert dam[2] == pytest.approx(0.0), "Direction 3 must remain undamaged"

        # Check decoded status flag: kd1 >= 1, kd2 == 0, kd3 == 0
        dam5 = int(dam[4])
        idam = dam5 - 10000
        kd1 = idam // 1000
        kd2 = (idam - kd1 * 1000) // 100
        kd3 = (idam - kd1 * 1000 - kd2 * 100) // 10
        assert kd1 >= 1
        assert kd2 == 0
        assert kd3 == 0

    def test_directional_tensile_cracking_transverse(self, tmp_path: Path):
        """Tensile loading in dir 2 generates crack in dir 2 while dir 1 remains undamaged."""
        model, _ = _build_model(self.CRACKING_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Pull along y (dir 2): vy = 50.0 on face y=1 (nodes 3, 4, 7, 8 -> indices 2, 3, 6, 7)
        v = np.zeros_like(model.x)
        v[[2, 3, 6, 7], 1] = 50.0

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        dam = g.state["mat_extra"]["dam12"][0]
        assert dam[1] > 0.0, "Direction 2 must crack"
        assert dam[0] == pytest.approx(0.0), "Direction 1 must remain undamaged"
        assert dam[2] == pytest.approx(0.0), "Direction 3 must remain undamaged"

    def test_element_degradation_chain_fmax(self, tmp_path: Path):
        """Stress reaches fmax: off degrades from 1.0 -> 0.792 -> ... -> 0.0, stresses vanish."""
        deck = """
/BEGIN
DEGRADATION_TEST
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
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Fmax
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
10000.0 10000.0 10000.0 0.05
50.0 1.0 1.2 1.0
100.0 100.0 100.0 100.0
50.0 50.0 50.0 50.0
100.0 100.0 50.0 50.0
/END
"""
        model, _ = _build_model(deck, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Shear loading (vx = 500.0 on top face y=1) to enter Tsai-Wu plasticity and accumulate wpla
        v = np.zeros_like(model.x)
        v[[2, 3, 6, 7], 0] = 500.0

        off_history = []
        for _ in range(80):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            off_val = float(g.state["mat_extra"]["off12"][0])
            off_history.append(off_val)

        # Verify the degradation progression
        assert 1.0 in off_history, "Starts with off == 1.0"
        degraded = [o for o in off_history if 0.0 < o < 1.0]
        assert len(degraded) > 0, "Must show intermediate degradation values"
        if len(degraded) >= 2:
            # First degraded step: off was 0.99, then scaled by 0.8 -> 0.792
            assert degraded[0] == pytest.approx(0.792, rel=1e-3)
            assert degraded[1] == pytest.approx(0.6336, rel=1e-3)

        # Final state must be deleted: off == 0.0
        assert off_history[-1] == 0.0
        assert g.state["off"][0] == 0.0
        # When deleted, stresses must be zero
        np.testing.assert_allclose(g.state["sig"][0], 0.0, atol=1e-8)

    def test_post_deletion_numerical_stability(self, tmp_path: Path):
        """60+ cycles after element deletion: zero stresses, zero forces, unconstrained dt, solver stable."""
        deck = """
/BEGIN
POST_DEL_TEST
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
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Fmax
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
10000.0 10000.0 10000.0 0.05
50.0 1.0 1.1 1.0
100.0 100.0 100.0 100.0
50.0 50.0 50.0 50.0
100.0 100.0 50.0 50.0
/END
"""
        model, _ = _build_model(deck, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        v = np.zeros_like(model.x)
        v[[2, 3, 6, 7], 0] = 600.0

        deleted_cycle = None
        for cycle in range(100):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            if g.state["off"][0] == 0.0:
                if deleted_cycle is None:
                    deleted_cycle = cycle
                # Every post-deletion step: zero forces, zero stresses, unconstrained dt
                np.testing.assert_allclose(fint, 0.0, atol=1e-8)
                np.testing.assert_allclose(g.state["sig"][0], 0.0, atol=1e-8)
                assert dtc[0] >= 1.0e29, "Deleted element must not constrain Courant dt"
                assert np.all(np.isfinite(model.x)), "Coordinates must remain finite"

        assert deleted_cycle is not None, "Element must delete"
        assert 100 - deleted_cycle >= 60, f"Must run 60+ post-deletion cycles stably (ran {100 - deleted_cycle})"


# ============================================================================
# 3. 3D Tsai-Wu Plasticity and Dynamic Strain Rate Effects
# ============================================================================

class TestLaw12TsaiWuPlasticityAndRate:
    """3D Tsai-Wu yield return, plastic work, dynamic strain rate, and fibers."""

    PLASTIC_DECK = """
/BEGIN
TSAI_WU_PLASTIC
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
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_TW
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
10000.0 10000.0 10000.0 0.05
20.0 0.8 100.0 1.0
500.0 500.0 500.0 500.0
50.0 50.0 50.0 50.0
500.0 500.0 50.0 50.0
/END
"""

    def test_tsai_wu_pure_shear_plasticity(self, tmp_path: Path):
        """3D pure shear enters plastic regime: wpla accumulates, stresses remain bounded."""
        model, _ = _build_model(self.PLASTIC_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Pure shear in xy: vy = rate * x (nodes at x=1 have vy = 100.0)
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 1] = 100.0

        wpla_history = []
        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            assert dtc[0] > 0.0
            model.x += v * dt
            wpla = float(g.state["mat_extra"]["wpla12"][0])
            wpla_history.append(wpla)
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        # Plastic work must accumulate monotonically
        assert wpla_history[-1] > 0.001
        for i in range(1, len(wpla_history)):
            assert wpla_history[i] >= wpla_history[i - 1] - 1e-12

        # Shear stress sig_xy (component 3) must be non-zero and plastic
        sig_xy = abs(float(g.state["sig"][0, 3]))
        assert sig_xy > 10.0

    def test_tsai_wu_multiaxial_loading(self, tmp_path: Path):
        """Combined tension and shear multi-axial loading: consistent plastic flow."""
        model, _ = _build_model(self.PLASTIC_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Combined tension in x and shear in xy
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0   # tension
        v[[1, 2, 5, 6], 1] = 100.0  # shear

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        wpla = float(g.state["mat_extra"]["wpla12"][0])
        assert wpla > 0.001
        # Both normal stress and shear stress are active
        assert float(g.state["sig"][0, 0]) > 0.0
        assert abs(float(g.state["sig"][0, 3])) > 0.0

    def test_dynamic_strain_rate_elevation(self, tmp_path: Path):
        """High velocity impact shows elevated yield stress due to c * ln(eps_dot / eps_dot_0)."""
        rate_deck = """
/BEGIN
RATE_TEST
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
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Rate
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
10000.0 10000.0 10000.0 0.05
0.0 1.0 1000.0 1.0
500.0 500.0 500.0 500.0
20.0 20.0 20.0 20.0
500.0 500.0 20.0 20.0
0.0 0.0 0.10 1.0 1
/END
"""
        # Run A: low strain rate (velocity 0.5, dt = 1e-4 -> eps_dot = 0.5 <= eps0 = 1.0, rate_fac = 1.0)
        td_a = tmp_path / "rate_a"
        td_a.mkdir()
        model_a, _ = _build_model(rate_deck, td_a)
        v_a = np.zeros_like(model_a.x)
        v_a[[1, 2, 5, 6], 1] = 0.5
        dt_a = 1.0e-4
        for _ in range(100):
            fint_a = np.zeros_like(model_a.x)
            solid_hexa8.forces(model_a.bricks, model_a.x, v_a, model_a.vr, dt_a, fint_a, fint_a)
            model_a.x += v_a * dt_a

        sig_shear_a = abs(float(model_a.bricks.state["sig"][0, 3]))

        # Run B: high strain rate (velocity 500.0, dt = 1e-7 -> eps_dot = 500 >> eps0 = 1.0, rate_fac > 1.6)
        # Both achieve identical total plastic shear strain = 0.005
        td_b = tmp_path / "rate_b"
        td_b.mkdir()
        model_b, _ = _build_model(rate_deck, td_b)
        v_b = np.zeros_like(model_b.x)
        v_b[[1, 2, 5, 6], 1] = 500.0
        dt_b = 1.0e-7
        for _ in range(100):
            fint_b = np.zeros_like(model_b.x)
            solid_hexa8.forces(model_b.bricks, model_b.x, v_b, model_b.vr, dt_b, fint_b, fint_b)
            model_b.x += v_b * dt_b

        sig_shear_b = abs(float(model_b.bricks.state["sig"][0, 3]))

        # High strain rate must produce higher shear resistance: ~20% elevation
        assert sig_shear_b > sig_shear_a
        assert (sig_shear_b / sig_shear_a) == pytest.approx(1.20, rel=0.05)

    def test_fiber_reinforcement_dynamic_response(self, tmp_path: Path):
        """Fiber reinforcement (alpha > 0, efib > 0) tracks epsf and develops fiber stress."""
        fiber_deck = """
/BEGIN
FIBER_TEST
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
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Fiber
1.5e-9 1.5e-9
100000.0 20000.0 20000.0
0.3 0.3 0.3
10000.0 10000.0 10000.0
500.0 500.0 500.0 0.05
0.0 1.0 1000.0 1.0
1000.0 1000.0 1000.0 1000.0
500.0 500.0 500.0 500.0
1000.0 1000.0 500.0 500.0
0.30 200000.0 0.0 1.0 1
/END
"""
        model, _ = _build_model(fiber_deck, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Pull in fiber direction 1 (x)
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0

        for _ in range(50):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        epsf = float(g.state["mat_extra"]["epsf12"][0])
        sigf = float(g.state["mat_extra"]["sigf12"][0])

        assert epsf > 0.0, "Fiber strain epsf must be positive"
        assert sigf > 0.0, "Fiber stress sigf must be positive"
        # Verify analytical formula: sigf == efib * epsf
        assert sigf == pytest.approx(200000.0 * epsf, rel=1e-5)


# ============================================================================
# 4. Tetra4 Solids and Combined Hexa8 + Tetra4 System
# ============================================================================

class TestLaw12Tetra4AndCombinedMesh:
    """Tetra4 solid element simulations and coupled Hexa8 + Tetra4 hybrid meshes."""

    TETRA4_DECK = """
/BEGIN
TETRA4_LAW12_DYN
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 0.0 1.0 0.0
4 0.0 0.0 1.0
/TETRA4/1
1 1 2 3 4
/PART/1
Tet_Part
1 1
/PROP/SOLID/1
Tet_Prop
1.1 0.05 0.1
/MAT/LAW12/1
Mat_Tet
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
50.0 100.0 100.0 0.05
/END
"""

    def test_tetra4_multiaxial_orthotropic_dynamic_sim(self, tmp_path: Path):
        """Tetra4 solid element under multi-axial orthotropic dynamic loading: 60 cycles."""
        model, _ = _build_model(self.TETRA4_DECK, tmp_path)
        g = model.tetras
        assert g.n == 1
        dt = 1.0e-7

        # Pull node 2 in +x and node 3 in +y
        v = np.zeros_like(model.x)
        v[1, 0] = 50.0
        v[2, 1] = 25.0

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_tetra4.forces(g, model.x, v, model.vr, dt, fint, mint)
            assert dtc[0] > 0.0
            model.x += v * dt
            # Constant-strain tetra force equilibrium
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        assert g.state["sig"][0, 0] > 0.0
        assert g.state["eint"][0] > 0.0
        assert g.state["off"][0] == 1.0

    def test_tetra4_cyclic_unilateral_cracking(self, tmp_path: Path):
        """Tetra4 element under cyclic tension-compression: directional cracking and crack closure."""
        model, _ = _build_model(self.TETRA4_DECK, tmp_path)
        g = model.tetras
        dt = 1.0e-7

        # Phase 1: Tension on node 2 pulls past sigt1=50 MPa
        v = np.zeros_like(model.x)
        v[1, 0] = 100.0
        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_tetra4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        dam1 = float(g.state["mat_extra"]["dam12"][0, 0])
        assert dam1 > 0.0, "Tetra4 must crack in tension"

        # Phase 2: Compression (v_x = -100.0) closes crack and bears load
        v[1, 0] = -100.0
        for _ in range(100):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_tetra4.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        assert float(g.state["sig"][0, 0]) < 0.0, "Tetra4 bears compression once crack is closed"

    def test_combined_hexa8_tetra4_dynamic_system(self, tmp_path: Path):
        """Combined mesh of Hexa8 and Tetra4 sharing an interface: simultaneous explicit simulation."""
        hybrid_deck = """
/BEGIN
HYBRID_SYSTEM
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
9 2.0 0.5 0.5
/BRICK/1
1 1 2 3 4 5 6 7 8
/TETRA4/1
1 2 9 3 6
/PART/1
Hex_Part
1 1
/PART/2
Tet_Part
2 2
/PROP/SOLID/1
Hex_Prop
1.1 0.05 0.1
/PROP/SOLID/2
Tet_Prop
1.1 0.05 0.1
/MAT/LAW12/1
Mat_Hex
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
1000.0 1000.0 1000.0 0.05
/MAT/LAW12/2
Mat_Tet
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
1000.0 1000.0 1000.0 0.05
/END
"""
        model, _ = _build_model(hybrid_deck, tmp_path)
        assert model.bricks.n == 1
        assert model.tetras.n == 1
        dt = 1.0e-7

        # Pull outer tip of tetra (node 9, index 8) in +x, and interface nodes in +x
        v = np.zeros_like(model.x)
        v[8, 0] = 50.0
        v[[1, 2, 5], 0] = 25.0

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc_h = solid_hexa8.forces(model.bricks, model.x, v, model.vr, dt, fint, mint)
            dtc_t = solid_tetra4.forces(model.tetras, model.x, v, model.vr, dt, fint, mint)

            assert dtc_h[0] > 0.0
            assert dtc_t[0] > 0.0
            model.x += v * dt

            # Coupled assembly must be in global equilibrium
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        # Both elements active and store internal energy
        assert float(model.bricks.state["eint"][0]) > 0.0
        assert float(model.tetras.state["eint"][0]) > 0.0
        assert model.bricks.state["off"][0] == 1.0
        assert model.tetras.state["off"][0] == 1.0


# ============================================================================
# 5. Multi-Element Wave Propagation and Global Energy Balance
# ============================================================================

class TestLaw12MultiElementWaveAndEnergyBalance:
    """Multi-element wave propagation and global work-energy balance."""

    def test_hexa8_mesh_tensile_wave_propagation(self, tmp_path: Path):
        """4x1x1 Hexa8 bar tensile wave propagation: element 4 cracks while upstream elements wait for arrival."""
        nodes = []
        nid = 1
        for ix in range(5):
            for iy in range(2):
                for iz in range(2):
                    nodes.append(f"{nid} {float(ix)} {float(iy)} {float(iz)}")
                    nid += 1
        node_block = "\n".join(nodes)

        bricks = []
        for e in range(4):
            base0 = 4 * e
            base1 = 4 * (e + 1)
            n1 = base0 + 1
            n2 = base1 + 1
            n3 = base1 + 3
            n4 = base0 + 3
            n5 = base0 + 2
            n6 = base1 + 2
            n7 = base1 + 4
            n8 = base0 + 4
            bricks.append(f"{e+1} {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8}")
        brick_block = "\n".join(bricks)

        deck = f"""/BEGIN
BAR4_WAVE_TEST
/NODE
{node_block}
/BRICK/1
{brick_block}
/PART/1
Bar_Part
1 1
/PROP/SOLID/1
Bar_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Law12
1.5e-9 1.5e-9
100000.0 10000.0 10000.0
0.3 0.3 0.03
5000.0 5000.0 5000.0
10.0 1000.0 1000.0 0.05
/END
"""
        model, _ = _build_model(deck, tmp_path)
        g = model.bricks
        assert g.n == 4

        dt = 1.0e-7

        # Apply tensile velocity pulse at right boundary x=4 (nodes 17, 18, 19, 20 -> indices 16, 17, 18, 19)
        right_end_nodes = [16, 17, 18, 19]
        v = np.zeros_like(model.x)
        v[right_end_nodes, 0] = 1000.0

        # Run 5 steps: wave directly affects element 4 (idx 3)
        for _ in range(5):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        dam1_elem4 = float(g.state["mat_extra"]["dam12"][3, 0])
        dam1_elem1 = float(g.state["mat_extra"]["dam12"][0, 0])
        dam2_elem4 = float(g.state["mat_extra"]["dam12"][3, 1])

        # Element 4 at the pulled end cracks in dir 1
        assert dam1_elem4 > 0.0, "Element 4 must experience directional cracking"
        # Element 1 at the opposite far end is untouched by the crack
        assert dam1_elem1 == pytest.approx(0.0), "Element 1 must remain uncracked"
        # Transverse direction 2 is undamaged
        assert dam2_elem4 == pytest.approx(0.0), "Transverse direction must remain undamaged"

    def test_multi_element_mesh_energy_balance(self, tmp_path: Path):
        """2-element Hexa8 mesh under explicit dynamic loading: verify work-energy balance (|ERR| < 1e-4)."""
        deck = """
/BEGIN
TWO_HEXA_ENERGY
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
2 2 3 6 5 8 9 12 11
/PART/1
Bar
1 1
/PROP/SOLID/1
Hex_Prop
1.1 0.0 0.0
/MAT/LAW12/1
Mat_Law12
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
10000.0 10000.0 10000.0 0.05
/END
"""
        model, _ = _build_model(deck, tmp_path)
        g = model.bricks
        assert g.n == 2
        dt = 1.0e-6

        pulled_nodes = [2, 5, 8, 11]  # nodes 3, 6, 9, 12 at x=2
        v = np.zeros_like(model.x)
        v[pulled_nodes, 0] = 20.0

        w_ext = 0.0
        fint_old = np.zeros_like(model.x)

        for _ in range(80):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            f_mid = 0.5 * (fint_old + fint)
            dw = -np.sum(f_mid[pulled_nodes] * v[pulled_nodes]) * dt
            w_ext += dw
            fint_old = fint.copy()

        eint_total = float(g.state["eint"].sum())
        assert eint_total > 0.0
        rel_err = abs(w_ext - eint_total) / max(w_ext, eint_total)
        assert rel_err < 1.0e-4, f"Multi-element work-energy balance error {rel_err:.2e} must be < 1e-4"


# ============================================================================
# 6. End-to-End Starter and Engine Simulations
# ============================================================================

class TestLaw12EndToEndStarterEngine:
    """Full end-to-end Starter + Engine execution of LAW12 solid models."""

    def test_full_engine_run_hexa8_cube(self, tmp_path: Path):
        """Execute full Starter + Engine on Hexa8 solid cube with LAW12: verify stable integration."""
        run_name = "HEXA8_LAW12_RUN"
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
/MAT/LAW12/1
Mat_Law12
1.55e-9 1.55e-9
140000.0 10000.0 10000.0
0.30 0.45 0.02
5000.0 3500.0 5000.0
1800.0 40.0 40.0 0.08
150.0 0.5 1200.0 1.0
2000.0 50.0 1500.0 200.0
80.0 80.0 50.0 50.0
50.0 200.0 80.0 80.0
0.25 180000.0 0.04 1.0 1
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
1 X 2 10.0
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
        assert eng_model.engine_state.cycle > 5

    def test_full_engine_run_mid_simulation_deletion(self, tmp_path: Path):
        """Execute full Starter + Engine with dynamic deletion occurring mid-run (ndel >= 1)."""
        run_name = "DELETION_LAW12_RUN"
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
/MAT/LAW12/1
Mat_Fmax_Del
1.55e-9 1.55e-9
140000.0 10000.0 10000.0
0.30 0.45 0.02
5000.0 3500.0 5000.0
1800.0 40.0 40.0 0.08
10.0 1.0 1.01 1.0
200.0 200.0 200.0 200.0
50.0 50.0 50.0 50.0
200.0 200.0 50.0 50.0
0.0 0.0 0.0 1.0 1
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
1 X 2 5000.0
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
        # Verify element deleted during simulation
        assert eng_model.bricks.state["off"][0] == 0.0
        np.testing.assert_allclose(eng_model.bricks.state["sig"][0], 0.0)
        assert eng_model.engine_state.ndel >= 1
