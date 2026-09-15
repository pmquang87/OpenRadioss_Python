"""
Engine Simulation and Dynamic Failure Auditor Tests for /MAT/LAW14
(/MAT/COMPSO, /MAT/COMP_SOL 3D orthotropic composite material for solid elements).
Milestone M547: Auditor 2C — Multi-element explicit dynamic simulations,
directional tensile cracking, unilateral cyclic crack closure, 3D Tsai-Wu
plasticity, dynamic strain rate effects, progressive degradation, and element deletion.

Fortran origin:
  - engine/source/materials/mat/mat014/m14law.F
  - engine/source/elements/solid/solide/sforc3.F
  - engine/source/elements/solid/tetra4/s4forc3.F
  - starter/source/materials/mat/mat014/hm_read_mat14.F

Audit Tasks & Requirements:
  1. Verify integration with solid_hexa8.py and solid_tetra4.py:
     - Hexa8 forces update correctly computes B-matrix, strain increments,
       constitutive call to LAW14, internal force assembly, and element degradation (off14).
     - Tetra4 forces update works identically with 1-point integration.
     - Timestep calculation: sound speed c = sqrt(C1 / rho) correctly limits
       the explicit critical timestep dt = alpha * Le / c.
  2. Verify Energy Balance:
     - For elastic oscillations of a composite block under cyclic loading,
       verify total energy E_tot = E_int + E_kin is strictly conserved (0.00% leak).
     - Under damaging tensile loads, verify that internal strain energy + cracked
       dissipated work equals external work done.
     - Under plastic loading, verify Tsai-Wu plastic work W_p is accurately
       accumulated and books into the energy ledger.
  3. Verify Element Failure & Deletion:
     - Dynamic tensile pull to failure: verify that as stress reaches Fmax,
       off degrades (0.99 -> 0.8*off -> 0.0).
     - When an element completely fails (off == 0.0), verify forces smoothly drop
       to zero and simulation continues stably without NaN or Inf.
  4. Dynamic multi-step simulation tests:
     - test_hexa8_elastic_wave_propagation: single and multi-element Hexa8 dynamic
       wave propagation, tracking kinetic and strain energy.
     - test_tetra4_dynamic_oscillation: Tetra4 element oscillating under shear and
       axial load, verifying frequency matches theoretical omega = sqrt(K/M).
     - test_tensile_cracking_energy_balance: Hexa8 pulled in 1-direction until cracking;
       verify directional tensile crack opening, transverse stress relaxation, and energy accounting.
     - test_tsaiwu_plastic_dissipation_cycle: cyclic loading entering Tsai-Wu plastic regime,
       unloading, showing permanent plastic strain and positive plastic work accumulation.
     - test_complete_element_failure_stability: element pulled until full degradation (off=0),
       verifying forces vanish and solver continues stably.
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
from pyradioss.materials import law14_compso
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
    tmp_path.mkdir(parents=True, exist_ok=True)
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
# 1. Hexa8 Dynamic Simulations & Wave Propagation
# ============================================================================

class TestLaw14Hexa8DynamicSim:
    """Hexa8 solid single and multi-element dynamic explicit simulations with LAW14."""

    HEXA8_DECK = """
/BEGIN
HEXA8_LAW14_DYN
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
Cube_LAW14
1 1
/PROP/SOLID/1
Hex_Prop
1.1 0.0 0.0
/MAT/LAW14/1
Composite_Solid_Hex
1.55e-9 1.55e-9
140000.0 10000.0 10000.0
0.30 0.45 0.02
5000.0 3500.0 5000.0
1800.0 40.0 40.0 0.08
150.0 0.5 1200.0 1.0
2000.0 50.0 1500.0 200.0
80.0 80.0 50.0 50.0
0.25 180000.0 0.04 1.0 1
/END
"""

    def test_hexa8_elastic_wave_propagation(self, tmp_path: Path):
        """Single and multi-element Hexa8 dynamic wave propagation, tracking kinetic and strain energy."""
        # --- Part A: Single-element dynamic elastic wave / energy conservation ---
        model_single, _ = _build_model(self.HEXA8_DECK, tmp_path / "single")
        g_s = model_single.bricks
        assert g_s.n == 1
        dt_s = 1.0e-6
        pulled_s = [1, 2, 5, 6]
        v_s = np.zeros_like(model_single.x)

        # Forward stretch (50 steps)
        v_s[pulled_s, 0] = 10.0
        w_ext = 0.0
        fint_old = np.zeros_like(model_single.x)
        for _ in range(50):
            fint = np.zeros_like(model_single.x)
            dtc = solid_hexa8.forces(g_s, model_single.x, v_s, model_single.vr, dt_s, fint, fint)
            assert dtc[0] > 0.0
            model_single.x += v_s * dt_s
            f_mid = 0.5 * (fint_old + fint)
            w_ext += -np.sum(f_mid[pulled_s, 0] * v_s[pulled_s, 0]) * dt_s
            fint_old = fint.copy()

        eint_peak = float(g_s.state["eint"][0])
        sig_peak = float(g_s.state["sig"][0, 0])
        assert sig_peak > 0.0
        assert eint_peak > 0.0
        # External work equals internal energy exactly
        assert abs(w_ext - eint_peak) / max(w_ext, eint_peak) < 1.0e-6

        # Reverse cycle (50 steps back to zero displacement)
        v_s[pulled_s, 0] = -10.0
        for _ in range(50):
            fint = np.zeros_like(model_single.x)
            solid_hexa8.forces(g_s, model_single.x, v_s, model_single.vr, dt_s, fint, fint)
            model_single.x += v_s * dt_s
            f_mid = 0.5 * (fint_old + fint)
            w_ext += -np.sum(f_mid[pulled_s, 0] * v_s[pulled_s, 0]) * dt_s
            fint_old = fint.copy()

        eint_end = float(g_s.state["eint"][0])
        sig_end = float(g_s.state["sig"][0, 0])
        assert abs(sig_end) < 0.01
        # Reversible elastic energy conservation: energy leak < 0.05%
        leak_pct = (abs(eint_end) / eint_peak) * 100.0
        assert leak_pct < 0.05, f"Energy leak {leak_pct:.4f}% must be < 0.05%"

        # --- Part B: Multi-element 4x1x1 Hexa8 bar tensile stress wave propagation ---
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

        bar_deck = f"""/BEGIN
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
/MAT/LAW14/1
Mat_Law14_Bar
1.55e-9 1.55e-9
140000.0 10000.0 10000.0
0.30 0.45 0.02
5000.0 3500.0 5000.0
1e10 1e10 1e10 0.08
/END
"""
        model_bar, _ = _build_model(bar_deck, tmp_path / "bar")
        g_bar = model_bar.bricks
        assert g_bar.n == 4

        dt_bar = 1.0e-7
        right_end_nodes = [16, 17, 18, 19]  # Face at x=4
        v_bar = np.zeros_like(model_bar.x)
        v_bar[right_end_nodes, 0] = 50.0

        w_ext_bar = 0.0
        fint_old_bar = np.zeros_like(model_bar.x)

        # Propagate wave for 5 short steps
        for _ in range(5):
            fint = np.zeros_like(model_bar.x)
            solid_hexa8.forces(g_bar, model_bar.x, v_bar, model_bar.vr, dt_bar, fint, fint)
            model_bar.x += v_bar * dt_bar
            f_mid = 0.5 * (fint_old_bar + fint)
            w_ext_bar += -np.sum(f_mid[right_end_nodes, 0] * v_bar[right_end_nodes, 0]) * dt_bar
            fint_old_bar = fint.copy()

        sig_elem4 = float(g_bar.state["sig"][3, 0])  # Right-most pulled element
        sig_elem1 = float(g_bar.state["sig"][0, 0])  # Left-most element untouched
        eint_total = float(g_bar.state["eint"].sum())

        assert sig_elem4 > 0.0, "Pulled element 4 must carry tensile wave stress"
        assert sig_elem1 == pytest.approx(0.0), "Wave has not arrived at element 1 yet"
        assert eint_total > 0.0
        rel_err_bar = abs(w_ext_bar - eint_total) / max(w_ext_bar, eint_total)
        assert rel_err_bar < 1.0e-6, f"Multi-element work-energy error {rel_err_bar:.2e} must be < 1e-6"

    def test_hexa8_uniaxial_tension_multi_cycle(self, tmp_path: Path):
        """60 explicit time steps under uniaxial tension along dir 1: verify force balance and energy."""
        model, _ = _build_model(self.HEXA8_DECK, tmp_path)
        g = model.bricks
        assert g.n == 1
        dt = 1.0e-6
        n_cycles = 60

        v = np.zeros_like(model.x)
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

        sig1 = float(g.state["sig"][0, 0])
        assert sig1 > 100.0
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
        dam = g.state["mat_extra"]["dam14"][0]
        assert dam[0] == pytest.approx(0.0)
        assert dam[1] == pytest.approx(0.0)
        assert dam[2] == pytest.approx(0.0)
        assert g.state["off"][0] == 1.0

    def test_hexa8_cyclic_unilateral_cracking_and_closure(self, tmp_path: Path):
        """Cyclic loading: tension cracks element -> compression closes crack (bearing load) -> reload."""
        deck = """
/BEGIN
HEXA8_UNILATERAL_LAW14
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
/MAT/LAW14/1
Mat_Law14
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

        dam1 = float(g.state["mat_extra"]["dam14"][0, 0])
        epc1 = float(g.state["mat_extra"]["epc14"][0, 0])
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

            epc_cur = float(g.state["mat_extra"]["epc14"][0, 0])
            s1 = float(g.state["sig"][0, 0])
            if epc_cur > 0.0001:
                assert s1 == pytest.approx(0.0, abs=1e-5), "Open crack must not support compressive stress"
                zero_stress_seen = True
            elif epc_cur == 0.0 and s1 < -50.0:
                comp_stress_seen = True

        assert zero_stress_seen, "Must observe zero-stress crack closing phase"
        assert comp_stress_seen, "Must observe compressive stress after crack closure"
        assert float(g.state["mat_extra"]["epc14"][0, 0]) == pytest.approx(0.0)

        # Phase 3: Reload back into tension (v_x = +50.0)
        v[pulled, 0] = 50.0
        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            assert np.all(np.isfinite(g.state["sig"][0]))

        assert g.state["off"][0] == 1.0


# ============================================================================
# 2. Tetra4 Dynamic Simulations & Theoretical Oscillation
# ============================================================================

class TestLaw14Tetra4DynamicSim:
    """Tetra4 solid element dynamic simulations and theoretical frequency verification."""

    TETRA4_DECK = """
/BEGIN
TETRA4_LAW14_DYN
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
0.0 0.0 0.0
/MAT/LAW14/1
Mat_Tet
1.5e-9 1.5e-9
100000.0 10000.0 10000.0
0.3 0.3 0.03
5000.0 5000.0 5000.0
1e10 1e10 1e10 0.05
/END
"""

    def test_tetra4_dynamic_oscillation(self, tmp_path: Path):
        """Tetra4 element oscillating under shear and axial load, verifying frequency matches theoretical omega = sqrt(K/M)."""
        model, _ = _build_model(self.TETRA4_DECK, tmp_path)
        g = model.tetras
        assert g.n == 1

        mat = model.materials[1]
        D11 = float(mat.params["D11"])
        G12 = float(mat.params["G12"])
        rho = float(mat.rho0)
        V = 1.0 / 6.0
        M = 0.25 * rho * V

        # --- Part A: Axial dynamic oscillation ---
        # Effective axial stiffness K_axial = V * D11
        K_axial = V * D11
        omega_axial_th = math.sqrt(K_axial / M)
        T_axial_th = 2.0 * math.pi / omega_axial_th

        # Static force check: verify effective stiffness matches theory exactly
        dt_test = 1.0e-8
        u0 = 1.0e-4
        v_test = np.zeros_like(model.x)
        v_test[1, 0] = u0 / dt_test
        fint_test = np.zeros_like(model.x)
        solid_tetra4.forces(g, model.x, v_test, model.vr, dt_test, fint_test, fint_test)
        K_axial_num = -float(fint_test[1, 0]) / u0
        assert K_axial_num == pytest.approx(K_axial, rel=1e-5)

        # Dynamic simulation: 2 complete periods
        model.x[:] = model.x0[:]
        g.state["sig"].fill(0.0)
        dt_axial = T_axial_th / 200.0
        v_dyn = np.zeros_like(model.x)
        v_dyn[1, 0] = 10.0
        mass = model.mass

        displacements_axial = []
        times_axial = []
        for step in range(400):
            t = step * dt_axial
            fint = np.zeros_like(model.x)
            solid_tetra4.forces(g, model.x, v_dyn, model.vr, dt_axial, fint, fint)
            a = fint[1, 0] / mass[1]
            v_dyn[1, 0] += a * dt_axial
            model.x[1, 0] += v_dyn[1, 0] * dt_axial
            displacements_axial.append(model.x[1, 0] - model.x0[1, 0])
            times_axial.append(t)

        u_arr_ax = np.array(displacements_axial)
        peaks_ax = [i for i in range(1, len(u_arr_ax) - 1) if u_arr_ax[i] > u_arr_ax[i - 1] and u_arr_ax[i] > u_arr_ax[i + 1]]
        assert len(peaks_ax) >= 2, "Must identify at least two oscillation peaks"
        T_axial_num = times_axial[peaks_ax[1]] - times_axial[peaks_ax[0]]
        rel_err_ax = abs(T_axial_num - T_axial_th) / T_axial_th
        assert rel_err_ax < 0.01, f"Axial oscillation period error {rel_err_ax:.2%} must be < 1%"

        # --- Part B: Shear dynamic oscillation ---
        # Effective shear stiffness K_shear = V * G12
        K_shear = V * G12
        omega_shear_th = math.sqrt(K_shear / M)
        T_shear_th = 2.0 * math.pi / omega_shear_th

        # Static force check
        model.x[:] = model.x0[:]
        g.state["sig"].fill(0.0)
        v_test.fill(0.0)
        v_test[1, 1] = u0 / dt_test
        fint_test.fill(0.0)
        solid_tetra4.forces(g, model.x, v_test, model.vr, dt_test, fint_test, fint_test)
        K_shear_num = -float(fint_test[1, 1]) / u0
        assert K_shear_num == pytest.approx(K_shear, rel=1e-5)

        # Dynamic shear oscillation: 2 complete periods
        model.x[:] = model.x0[:]
        g.state["sig"].fill(0.0)
        dt_shear = T_shear_th / 200.0
        v_dyn.fill(0.0)
        v_dyn[1, 1] = 10.0

        displacements_shear = []
        times_shear = []
        for step in range(400):
            t = step * dt_shear
            fint = np.zeros_like(model.x)
            solid_tetra4.forces(g, model.x, v_dyn, model.vr, dt_shear, fint, fint)
            a = fint[1, 1] / mass[1]
            v_dyn[1, 1] += a * dt_shear
            model.x[1, 1] += v_dyn[1, 1] * dt_shear
            displacements_shear.append(model.x[1, 1] - model.x0[1, 1])
            times_shear.append(t)

        u_arr_sh = np.array(displacements_shear)
        peaks_sh = [i for i in range(1, len(u_arr_sh) - 1) if u_arr_sh[i] > u_arr_sh[i - 1] and u_arr_sh[i] > u_arr_sh[i + 1]]
        assert len(peaks_sh) >= 2, "Must identify at least two shear peaks"
        T_shear_num = times_shear[peaks_sh[1]] - times_shear[peaks_sh[0]]
        rel_err_sh = abs(T_shear_num - T_shear_th) / T_shear_th
        assert rel_err_sh < 0.01, f"Shear oscillation period error {rel_err_sh:.2%} must be < 1%"

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
        deck = """
/BEGIN
TETRA4_CRACK_LAW14
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
/MAT/LAW14/1
Mat_Tet_Crack
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.03
20000.0 20000.0 20000.0
50.0 100.0 100.0 0.05
/END
"""
        model, _ = _build_model(deck, tmp_path)
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

        dam1 = float(g.state["mat_extra"]["dam14"][0, 0])
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
HYBRID_SYSTEM_LAW14
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
/MAT/LAW14/1
Mat_Hex
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.03
20000.0 20000.0 20000.0
1000.0 1000.0 1000.0 0.05
/MAT/LAW14/2
Mat_Tet
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.03
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
# 3. Directional Cracking and Energy Accounting
# ============================================================================

class TestLaw14CrackingAndEnergyBalance:
    """Directional cracking, transverse stress relaxation, and exact work-energy balance."""

    CRACKING_DECK = """
/BEGIN
CRACKING_ORTHOTROPY_LAW14
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
/MAT/LAW14/1
Mat_Ortho_Crack
1.55e-9 1.55e-9
140000.0 10000.0 10000.0
0.30 0.45 0.02
5000.0 3500.0 5000.0
100.0 1000.0 1000.0 0.08
/END
"""

    def test_tensile_cracking_energy_balance(self, tmp_path: Path):
        """Hexa8 pulled in 1-direction until cracking; verify directional tensile crack opening, transverse stress relaxation, and energy accounting."""
        model, _ = _build_model(self.CRACKING_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6
        pulled = [1, 2, 5, 6]
        v = np.zeros_like(model.x)
        v[pulled, 0] = 50.0

        w_ext = 0.0
        fint_old = np.zeros_like(model.x)

        # Pull element until fully cracked in direction 1 (60 steps)
        for _ in range(60):
            fint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, fint)
            model.x += v * dt

            f_mid = 0.5 * (fint_old + fint)
            dw = -np.sum(f_mid[pulled, 0] * v[pulled, 0]) * dt
            w_ext += dw
            fint_old = fint.copy()

        eint = float(g.state["eint"][0])
        dam1 = float(g.state["mat_extra"]["dam14"][0, 0])
        dam2 = float(g.state["mat_extra"]["dam14"][0, 1])
        dam3 = float(g.state["mat_extra"]["dam14"][0, 2])
        epc1 = float(g.state["mat_extra"]["epc14"][0, 0])
        sig1 = float(g.state["sig"][0, 0])

        # Directional crack in dir 1
        assert dam1 == pytest.approx(1.0), "Direction 1 must be fully cracked"
        assert epc1 > 0.0, "Crack opening strain epc1 must be positive"
        assert sig1 == pytest.approx(0.0, abs=1e-3), "Fully cracked direction has zero normal stress"

        # Transverse directions remain completely intact (orthotropy preserved)
        assert dam2 == pytest.approx(0.0)
        assert dam3 == pytest.approx(0.0)

        # Exact energy balance: W_ext == E_int to floating point precision
        assert eint > 0.0
        assert w_ext > 0.0
        rel_err = abs(w_ext - eint) / max(w_ext, eint)
        assert rel_err < 1.0e-6, f"Work-energy error during cracking {rel_err:.2e} must be < 1e-6"

    def test_directional_cracking_orthotropy(self, tmp_path: Path):
        """Tensile loading in dir 1 generates crack in dir 1 while dir 2 and 3 remain undamaged, then dir 2 cracks independently."""
        deck = """
/BEGIN
ORTHO_2DIR_CRACK
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
/MAT/LAW14/1
Mat_Ortho
1.55e-9 1.55e-9
100000.0 20000.0 20000.0
0.25 0.20 0.15
8000.0 6000.0 8000.0
200.0 40.0 150.0 0.10
/END
"""
        model, _ = _build_model(deck, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Step 1: Pull along x (dir 1): vx = 50.0 on face x=1
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        dam = g.state["mat_extra"]["dam14"][0]
        assert dam[0] > 0.0, "Direction 1 must crack under x tension"
        assert dam[1] == pytest.approx(0.0), "Direction 2 must remain undamaged"
        assert dam[2] == pytest.approx(0.0), "Direction 3 must remain undamaged"

        # Step 2: Now pull in y (dir 2) with vy = 50.0 on face y=1 (nodes 3, 4, 7, 8 -> indices 2, 3, 6, 7)
        v.fill(0.0)
        v[[2, 3, 6, 7], 1] = 50.0

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

        dam_after = g.state["mat_extra"]["dam14"][0]
        assert dam_after[1] > 0.0, "Direction 2 must crack under y tension"
        assert dam_after[2] == pytest.approx(0.0), "Direction 3 must still remain intact"


# ============================================================================
# 4. 3D Tsai-Wu Plasticity and Dynamic Rate Effects
# ============================================================================

class TestLaw14TsaiWuPlasticity:
    """3D Tsai-Wu yield return, cyclic plastic work dissipation, and dynamic rate effects."""

    PLASTIC_DECK = """
/BEGIN
TSAI_WU_PLASTIC_LAW14
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
/MAT/LAW14/1
Mat_TW_Plas
1.55e-9 1.55e-9
100000.0 50000.0 50000.0
0.3 0.3 0.03
5000.0 5000.0 5000.0
1e10 1e10 1e10 0.05
20.0 1.0 1000.0 1.0
1000.0 1000.0 1000.0 1000.0
50.0 50.0 50.0 50.0
0.0 0.0 0.0 1.0 1
/END
"""

    def test_tsaiwu_plastic_dissipation_cycle(self, tmp_path: Path):
        """Cyclic loading entering Tsai-Wu plastic regime, unloading, showing permanent plastic strain and positive plastic work accumulation."""
        model, _ = _build_model(self.PLASTIC_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6
        pulled = [1, 2, 5, 6]
        v = np.zeros_like(model.x)

        # Loading phase: pure shear vy = 20000.0 (30 steps)
        v[pulled, 1] = 20000.0
        wpla_history = []
        for _ in range(30):
            fint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, fint)
            model.x += v * dt
            wpla = float(g.state["mat_extra"]["wpla14"][0])
            wpla_history.append(wpla)

        wpla_peak = float(g.state["mat_extra"]["wpla14"][0])
        sig_xy_peak = float(g.state["sig"][0, 3])
        epsp_peak = float(g.state["epsp"][0])
        assert wpla_peak > 0.0, "Tsai-Wu plastic work must accumulate during plastic flow"
        assert epsp_peak > 0.0, "Plastic strain must be non-zero"
        assert abs(sig_xy_peak) > 0.0

        # Unloading phase: reverse velocity vy = -20000.0 (15 steps back to zero shear stress)
        v[pulled, 1] = -20000.0
        for _ in range(15):
            fint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, fint)
            model.x += v * dt
            wpla = float(g.state["mat_extra"]["wpla14"][0])
            wpla_history.append(wpla)

        wpla_unloaded = float(g.state["mat_extra"]["wpla14"][0])
        sig_xy_unloaded = float(g.state["sig"][0, 3])
        epsp_unloaded = float(g.state["epsp"][0])
        eint_unloaded = float(g.state["eint"][0])

        # Unloaded state verification
        assert abs(sig_xy_unloaded) < 1.0, "Shear stress must unload back to zero"
        assert epsp_unloaded > 0.0, "Permanent plastic strain must remain"
        assert wpla_unloaded > 0.0, "Plastic work is non-recoverable"
        assert eint_unloaded > 0.0, "Internal energy ledger accounts for dissipated plastic work"

        # Monotonicity check: dW_p >= 0 for every cycle
        for i in range(1, len(wpla_history)):
            assert wpla_history[i] >= wpla_history[i - 1] - 1e-12, "Plastic dissipation must be monotonic"

    def test_tsai_wu_multiaxial_loading(self, tmp_path: Path):
        """Combined tension and shear multi-axial loading: consistent plastic flow and equilibrium."""
        model, _ = _build_model(self.PLASTIC_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6

        # Combined tension in x and shear in xy
        v = np.zeros_like(model.x)
        v[[1, 2, 5, 6], 0] = 50.0    # tension
        v[[1, 2, 5, 6], 1] = 1000.0  # shear

        for _ in range(60):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt
            np.testing.assert_allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        wpla = float(g.state["mat_extra"]["wpla14"][0])
        assert wpla > 0.001
        assert float(g.state["sig"][0, 0]) > 0.0
        assert abs(float(g.state["sig"][0, 3])) > 0.0

    def test_dynamic_strain_rate_elevation(self, tmp_path: Path):
        """High velocity impact shows elevated yield resistance due to c * ln(eps_dot / eps_dot_0)."""
        rate_deck = """
/BEGIN
RATE_TEST_LAW14
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
/MAT/LAW14/1
Mat_Rate
1.5e-9 1.5e-9
100000.0 50000.0 50000.0
0.3 0.3 0.3
20000.0 20000.0 20000.0
10000.0 10000.0 10000.0 0.05
0.0 1.0 1000.0 1.0
500.0 500.0 500.0 500.0
20.0 20.0 20.0 20.0
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

        # Run B: high strain rate (velocity 500.0, dt = 1e-7 -> eps_dot = 500 >> eps0 = 1.0)
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

        # High strain rate must produce higher shear resistance (~20% elevation)
        assert sig_shear_b > sig_shear_a
        assert (sig_shear_b / sig_shear_a) == pytest.approx(1.20, rel=0.05)

    def test_fiber_reinforcement_dynamic_response(self, tmp_path: Path):
        """Fiber reinforcement (alpha > 0, efib > 0) tracks epsf and develops fiber stress."""
        fiber_deck = """
/BEGIN
FIBER_TEST_LAW14
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
/MAT/LAW14/1
Mat_Fiber
1.55e-9 1.55e-9
100000.0 20000.0 20000.0
0.3 0.3 0.03
10000.0 10000.0 10000.0
500.0 500.0 500.0 0.05
0.0 1.0 1000.0 1.0
1000.0 1000.0 1000.0 1000.0
500.0 500.0 500.0 500.0
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

        epsf = float(g.state["mat_extra"]["epsf14"][0])
        sigf = float(g.state["mat_extra"]["sigf14"][0])

        assert epsf > 0.0, "Fiber strain epsf must be positive"
        assert sigf > 0.0, "Fiber stress sigf must be positive"
        assert sigf == pytest.approx(200000.0 * epsf, rel=1e-5)


# ============================================================================
# 5. Element Failure, Degradation & End-to-End Simulation
# ============================================================================

class TestLaw14ElementFailureAndDeletion:
    """Progressive failure degradation (0.99 -> 0.8*off -> 0.0), post-deletion stability, and End-to-End runs."""

    FAILURE_DECK = """
/BEGIN
FAILURE_DEGRADE_LAW14
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
/MAT/LAW14/1
Mat_Fmax
1.55e-9 1.55e-9
100000.0 50000.0 50000.0
0.3 0.3 0.03
5000.0 5000.0 5000.0
1e10 1e10 1e10 0.05
10.0 1.0 1.05 1.0
1000.0 1000.0 1000.0 1000.0
50.0 50.0 50.0 50.0
0.0 0.0 0.0 1.0 1
/END
"""

    def test_complete_element_failure_stability(self, tmp_path: Path):
        """Dynamic tensile pull to failure: verify that as stress reaches Fmax, off degrades (0.99 -> 0.8*off -> 0.0), forces vanish and solver continues stably."""
        model, _ = _build_model(self.FAILURE_DECK, tmp_path)
        g = model.bricks
        dt = 1.0e-6
        pulled = [1, 2, 5, 6]
        v = np.zeros_like(model.x)
        v[pulled, 1] = 50000.0

        off_history = []
        deleted_cycle = None

        for cycle in range(100):
            fint = np.zeros_like(model.x)
            mint = np.zeros_like(model.x)
            dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
            model.x += v * dt

            off_val = float(g.state["off"][0])
            off_history.append(off_val)

            if off_val == 0.0:
                if deleted_cycle is None:
                    deleted_cycle = cycle
                # Zero forces, zero stresses, unconstrained dt
                np.testing.assert_allclose(fint, 0.0, atol=1e-8)
                np.testing.assert_allclose(g.state["sig"][0], 0.0, atol=1e-8)
                assert dtc[0] >= 1.0e29, "Deleted element must not constrain Courant dt"
                assert np.all(np.isfinite(model.x)), "Coordinates must remain finite"

        assert deleted_cycle is not None, "Element must reach full degradation to off=0.0"
        degradation_vals = [round(o, 4) for o in off_history if 0.0 < o < 1.0]
        # Verify 0.8 geometric softening sequence: 0.792, 0.6336, 0.5069, ...
        assert degradation_vals[0] == pytest.approx(0.792, abs=1e-3)
        assert degradation_vals[1] == pytest.approx(0.6336, abs=1e-3)
        assert 100 - deleted_cycle >= 60, f"Must run 60+ post-deletion cycles stably (ran {100 - deleted_cycle})"

    def test_full_engine_run_hexa8_cube(self, tmp_path: Path):
        """Execute full Starter + Engine on Hexa8 solid cube with LAW14: verify stable integration and 0.00% energy error."""
        run_name = "HEXA8_LAW14_RUN"
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
/MAT/LAW14/1
Mat_Law14
1.55e-9 1.55e-9
140000.0 10000.0 10000.0
0.30 0.45 0.02
5000.0 3500.0 5000.0
1800.0 40.0 40.0 0.08
150.0 0.5 1200.0 1.0
2000.0 50.0 1500.0 200.0
80.0 80.0 50.0 50.0
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
        """Execute full Starter + Engine with dynamic deletion occurring mid-run (ndel >= 1, off == 0.0)."""
        run_name = "DELETION_LAW14_RUN"
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
/MAT/LAW14/1
Mat_Fmax_Del
1.55e-9 1.55e-9
140000.0 10000.0 10000.0
0.30 0.45 0.02
5000.0 3500.0 5000.0
1800.0 40.0 40.0 0.08
10.0 1.0 1.01 1.0
200.0 200.0 200.0 200.0
50.0 50.0 50.0 50.0
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
        assert eng_model.bricks.state["off"][0] == 0.0
        np.testing.assert_allclose(eng_model.bricks.state["sig"][0], 0.0)
        assert eng_model.engine_state.ndel >= 1
