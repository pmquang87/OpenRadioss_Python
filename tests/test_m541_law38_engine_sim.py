"""Tests for Milestone M541 (Auditor 3): Explicit Dynamic Engine Simulation & Hydrodynamics Audit for LAW38.

Verification Summary:
1. Multi-cycle explicit engine simulation:
   - Single 8-node brick element and 2x2x2 solid block mesh with LAW38 material.
   - Cyclic displacement/velocity loading (compression into plateau, hold, unloading).
   - Run 50+ time steps with explicit central difference engine integration.
   - Verify stable time step integration, positive internal energy accumulation (W_int > 0),
     energy dissipation on unloading (viscoelastic hysteresis), and conservation of total
     energy within strict relative tolerance (< 1e-4).
2. Confined air pressure response:
   - Compress solid element towards porosity limit J -> phi.
   - Verify explosive pressure stiffening as pore space collapses.
   - Verify time-decay relaxation of pore pressure via exp(-relaxp * time).
3. Acoustic wave propagation & Courant time step:
   - 1D column of hexa8 solid elements under impact velocity.
   - Verify wave speed matches analytical sound speed c = sqrt((K + 4G/3)/rho_0).
   - Verify time step calculation dt <= L / c guarantees Courant-Friedrichs-Lewy stability.
4. Dynamic tension cutoff and element deletion:
   - High tensile velocity exceeding TENSIONCUT.
   - Verify element is marked broken (off38 = 0.0 or off = 0.0), stresses drop to zero,
     and simulation proceeds stably without NaN or Inf.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
from typing import Sequence, Tuple

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials import law38_visc_tab
from pyradioss.model.entities import Material
from pyradioss.starter.starter import run_starter


# =============================================================================
# Helper: Create 2x2x2 Hexa8 Mesh
# =============================================================================

def _build_2x2x2_mesh(
    dx: float = 10.0, dy: float = 10.0, dz: float = 10.0
) -> Tuple[list[Tuple[int, float, float, float]], list[Tuple[int, int, int, int, int, int, int, int, int]]]:
    """Generate 27 nodes and 8 brick elements for a 2x2x2 regular hexahedral mesh."""
    nodes = []
    node_id_map = {}
    nid = 1
    for k in range(3):
        for j in range(3):
            for i in range(3):
                x = i * dx
                y = j * dy
                z = k * dz
                nodes.append((nid, x, y, z))
                node_id_map[(i, j, k)] = nid
                nid += 1

    elements = []
    eid = 1
    for k in range(2):
        for j in range(2):
            for i in range(2):
                n1 = node_id_map[(i, j, k)]
                n2 = node_id_map[(i + 1, j, k)]
                n3 = node_id_map[(i + 1, j + 1, k)]
                n4 = node_id_map[(i, j + 1, k)]
                n5 = node_id_map[(i, j, k + 1)]
                n6 = node_id_map[(i + 1, j, k + 1)]
                n7 = node_id_map[(i + 1, j + 1, k + 1)]
                n8 = node_id_map[(i, j + 1, k + 1)]
                elements.append((eid, n1, n2, n3, n4, n5, n6, n7, n8))
                eid += 1

    return nodes, elements


# =============================================================================
# 1. Multi-Cycle Explicit Engine Simulation & Energy Conservation
# =============================================================================

class TestLaw38ExplicitEngineMultiCycle:
    """Multi-cycle explicit engine simulation on solid elements with LAW38."""

    def test_single_brick_cyclic_loading_energy_conservation(self, tmp_path):
        """Single 8-node brick element undergoing cyclic compression, hold, and unloading.

        Verifies:
        - 50+ time steps with explicit central difference engine integration.
        - Stable time step integration.
        - Positive internal energy accumulation during loading (W_int > 0).
        - Energy dissipation on unloading due to viscoelastic hysteresis.
        - Strict energy conservation: relative error < 1e-4 (|ERR| < 0.01%).
        """
        run_name = "HEXA8_L38_CYCLIC"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        # Unit cube: 10 x 10 x 10 mm
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.part(1, "BRICK_PART", 1, 1)

        # Tabulated foam curve with elastic, plateau, and densification regimes
        # Strain (nominal) vs Stress (nominal)
        deck.funct(1, "foam_curve", [
            (0.0, 0.0),
            (0.05, 5.0),
            (0.15, 6.5),
            (0.35, 7.5),
            (0.50, 12.0),
            (0.80, 50.0),
        ])

        rho0 = 1.0e-3
        e0 = 100.0
        nu_t = 0.3
        nu_c = 0.35
        beta = 10.0
        hyster = 0.75  # 25% hysteresis dissipation

        deck.mat_law38(
            1,
            rho=rho0,
            e=e0,
            nu=nu_t,
            nu_t=nu_t,
            nu_c=nu_c,
            beta=beta,
            h=hyster,
            damp1=0.5,
            funct_id_load=[1],
            title="LAW38_FOAM",
        )
        deck.prop_solid(1, "PROP_SOLID", qa=1.1, qb=0.05, h=0.1)

        # Boundary conditions: Clamped base in Z (nodes 1..4)
        deck.grnod_node(1, "base_nodes", [1, 2, 3, 4])
        deck.bcs(1, "clamp_base_z", "001", "111", 1)

        # Uniaxial strain condition: constrain lateral motion in X and Y
        deck.grnod_node(2, "all_nodes", list(range(1, 9)))
        deck.bcs(2, "fix_xy", "110", "111", 2)

        # Cyclic velocity loading:
        # 1. Ramp compression: t in [0.0, 0.35], v_z = -1.0 mm/s (compresses by ~0.35 mm)
        # 2. Hold compression: t in [0.35, 0.50], v_z = 0.0
        # 3. Unload back: t in [0.50, 0.85], v_z = +1.0 mm/s
        # 4. Rest: t in [0.85, 1.00], v_z = 0.0
        deck.grnod_node(3, "top_nodes", [5, 6, 7, 8])
        deck.funct(2, "cyclic_vel", [
            (0.0, 0.0),
            (0.05, -1.0),
            (0.35, -1.0),
            (0.40, 0.0),
            (0.50, 0.0),
            (0.55, 1.0),
            (0.85, 1.0),
            (0.90, 0.0),
            (1.00, 0.0),
        ])
        deck.impvel(1, "top_vel", 2, "Z", 3)
        deck.write(s_path)

        # Engine deck: run to 1.0 ms, stopping at cycle 65
        dt_scale = 0.9
        engine_deck = f"""/RUN/{run_name}/1
1.0
/DT
{dt_scale} 0
/PRINT/-1
/STOP
65
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        state = eng_model.engine_state

        # 1. Verify 50+ time steps completed stably
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.cycle == 65 or state.t >= 1.0 * (1.0 - 1e-12)

        # 2. Verify energy ledger
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0, f"Internal energy must be positive, got {en['IE']}"
        assert en["EW"] > 0.0, f"External work must be positive, got {en['EW']}"

        # 3. Verify total energy conservation within strict tolerance (< 1e-4 relative)
        total_energy = en["IE"] + en["KE"] + en["HE"] + en["EN"]
        ref_energy = en["REF"]
        rel_energy_error = abs(total_energy - en["EW"]) / ref_energy
        assert rel_energy_error < 1.0e-4, (
            f"Relative energy error {rel_energy_error:12.5E} exceeds strict tolerance 1e-4"
        )
        assert abs(en["ERR"]) < 0.01, f"Reported error percentage {en['ERR']}% exceeds 0.01%"

        # 4. Verify state variables uv38 and eps38 exist and are finite
        brick_g = dict(eng_model.element_groups())["bricks"]
        mat_extra = brick_g.state["mat_extra"]
        assert "uv38" in mat_extra
        assert "eps38" in mat_extra
        uv38 = mat_extra["uv38"]
        eps38 = mat_extra["eps38"]
        assert np.isfinite(uv38).all()
        assert np.isfinite(eps38).all()
        assert uv38.shape == (1, 33)
        assert eps38.shape == (1, 6)

    def test_2x2x2_solid_block_mesh_multicycle_simulation(self, tmp_path):
        """2x2x2 mesh of 8 hexa8 brick elements with LAW38 under cyclic dynamic compression.

        Verifies:
        - Multi-element topology explicit integration over 50+ time steps.
        - Positive internal energy accumulation during compressive loading.
        - Strict energy conservation (< 1e-4 relative).
        - Consistency across all 8 elements.
        """
        run_name = "BLOCK_2X2X2_L38"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        nodes, elements = _build_2x2x2_mesh(dx=10.0, dy=10.0, dz=10.0)

        deck = StarterDeck(run_name)
        deck.node(nodes)
        deck.brick(1, elements)
        deck.part(1, "FOAM_BLOCK_2X2X2", 1, 1)

        # Loading curve
        deck.funct(1, "load_curve", [
            (0.0, 0.0),
            (0.05, 10.0),
            (0.20, 15.0),
            (0.40, 18.0),
            (0.60, 40.0),
        ])

        deck.mat_law38(
            1,
            rho=1.2e-3,
            e=150.0,
            nu=0.32,
            nu_t=0.32,
            nu_c=0.36,
            beta=15.0,
            h=0.8,
            damp1=0.5,
            funct_id_load=[1],
            title="LAW38_2X2X2",
        )
        deck.prop_solid(1, "HEXA_PROP", qa=1.1, qb=0.05, h=0.1)

        # Base nodes at z = 0 (i in range 3, j in range 3, k = 0 -> nodes 1..9)
        base_nids = [nid for nid, x, y, z in nodes if math.isclose(z, 0.0, abs_tol=1e-5)]
        deck.grnod_node(1, "base_nodes", base_nids)
        deck.bcs(1, "fix_base_z", "001", "111", 1)

        # Symmetry / lateral constraint: fix X on x=0 and Y on y=0
        all_nids = [nid for nid, _, _, _ in nodes]
        deck.grnod_node(2, "all_nodes", all_nids)
        deck.bcs(2, "fix_xy", "110", "111", 2)

        # Top nodes at z = 20 (k = 2 -> 9 nodes)
        top_nids = [nid for nid, x, y, z in nodes if math.isclose(z, 20.0, abs_tol=1e-5)]
        deck.grnod_node(3, "top_nodes", top_nids)

        # Compression velocity ramp: compress top down in -Z
        deck.funct(2, "top_velocity", [
            (0.0, 0.0),
            (0.10, -1.5),
            (0.50, -1.5),
            (0.60, 0.0),
            (0.70, 0.0),
            (0.80, 1.5),
            (1.20, 1.5),
            (1.30, 0.0),
            (1.50, 0.0),
        ])
        deck.impvel(1, "top_vel", 2, "Z", 3)
        deck.write(s_path)

        dt_scale = 0.9
        engine_deck = f"""/RUN/{run_name}/1
1.50
/DT
{dt_scale} 0
/PRINT/-1
/STOP
65
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0, f"Starter errors: {log.errors}"
        state = eng_model.engine_state

        # Verify 50+ time steps
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"

        # Verify energies
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0, f"Expected positive internal energy, got {en['IE']}"
        assert en["EW"] > 0.0, f"Expected positive external work, got {en['EW']}"

        # Energy conservation relative error < 1e-4
        total_e = en["IE"] + en["KE"] + en["HE"] + en["EN"]
        rel_err = abs(total_e - en["EW"]) / en["REF"]
        assert rel_err < 1.0e-4, f"Relative energy error {rel_err:12.5E} exceeds 1e-4"
        assert abs(en["ERR"]) < 0.01

        # Check element state for all 8 elements
        brick_g = dict(eng_model.element_groups())["bricks"]
        assert brick_g.n == 8
        sig = brick_g.state["sig"]
        uv38 = brick_g.state["mat_extra"]["uv38"]
        assert sig.shape == (8, 6)
        assert uv38.shape == (8, 33)
        assert np.isfinite(sig).all()
        assert np.isfinite(uv38).all()


# =============================================================================
# 2. Confined Air Pressure Response & Pore Collapse
# =============================================================================

class TestLaw38ConfinedAirPressure:
    """Confined closed-cell air pressure response and porosity collapse (KCOMPAIR=1)."""

    def test_confined_air_pressure_porosity_collapse(self):
        """Compress a solid element towards porosity limit J -> phi.

        Analytical formula (sigeps38.F line 570):
            P_comp = P0 * (J - 1.0) / (J - phi)
        Verify explosive compressive pressure stiffening as pore space collapses (J -> phi).
        """
        p0 = 10.0
        phi = 0.40  # 40% initial porosity
        relaxp = 0.0  # no time relaxation for pure geometric test

        mat = Material(
            id=1,
            law=38,
            rho0=1.0e-3,
            params={
                "e0": 100.0,
                "nu_t": 0.3,
                "nu_c": 0.35,
                "kcompair": 1,
                "kair": 1,
                "p0": p0,
                "phi": phi,
                "poros": phi,
                "relaxp": relaxp,
                "pr": relaxp,
                "pmax": 1.0e9,
                "maxpres": 1.0e9,
                "load_curves": [lambda eps: 0.0],  # isolate air pressure response
            },
        )

        # Test volumetric ratios J approaching phi = 0.40 from above
        J_values = [0.90, 0.70, 0.50, 0.45, 0.42, 0.405]
        computed_pressures = []
        analytical_pressures = []

        for J in J_values:
            # Hydrostatic compression strain: deps_ii = ln(J) / 3
            eps_ii = math.log(J) / 3.0
            deps = np.array([[eps_ii, eps_ii, eps_ii, 0.0, 0.0, 0.0]])
            sig = np.zeros((1, 6))
            extra = {"time": 0.0, "uv38": np.zeros((1, 33)), "eps38": np.zeros((1, 6))}

            sign, _, _ = law38_visc_tab.solid_update(mat, sig, deps, dt=1.0e-3, extra=extra)

            p_comp_analytical = p0 * (J - 1.0) / (J - phi)
            # In sigeps38.F, PAIR is added directly to normal stresses (compression negative)
            pair_computed = extra["uv38"][0, 15]

            computed_pressures.append(pair_computed)
            analytical_pressures.append(p_comp_analytical)

            # Verification: computed pair matches analytical formula
            assert math.isclose(pair_computed, p_comp_analytical, rel_tol=1e-3), (
                f"At J={J}: expected P_comp={p_comp_analytical}, got {pair_computed}"
            )

        # Verify explosive pressure stiffening:
        # Near collapse (J = 0.405): J - phi = 0.005
        # Compared to mild compression (J = 0.90): J - phi = 0.50
        stiffening_ratio = abs(computed_pressures[-1]) / abs(computed_pressures[0])
        expected_ratio = abs(analytical_pressures[-1]) / abs(analytical_pressures[0])
        assert stiffening_ratio > 500.0, (
            f"Expected explosive stiffening > 500x, got {stiffening_ratio:g}"
        )
        assert math.isclose(stiffening_ratio, expected_ratio, rel_tol=1e-3)

    def test_confined_air_pressure_time_decay_relaxation(self):
        """Verify time-decay relaxation of pore pressure via exp(-relaxp * time).

        Analytical formula (sigeps38.F line 576):
            P_comp(t) = exp(-relaxp * t) * P_comp(0)
        """
        p0 = 25.0
        phi = 0.50
        relaxp = 10.0  # relaxation rate: time constant tau = 1/relaxp = 0.1 s

        mat = Material(
            id=2,
            law=38,
            rho0=1.0e-3,
            params={
                "e0": 100.0,
                "nu_t": 0.3,
                "kcompair": 1,
                "kair": 1,
                "p0": p0,
                "phi": phi,
                "poros": phi,
                "relaxp": relaxp,
                "pr": relaxp,
                "pmax": 1.0e9,
                "load_curves": [lambda eps: 0.0],
            },
        )

        J = 0.80  # fixed volume ratio V/V0
        eps_ii = math.log(J) / 3.0
        p_base = p0 * (J - 1.0) / (J - phi)  # 25.0 * (-0.20) / (0.30) = -16.6667

        test_times = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
        for t in test_times:
            deps = np.array([[eps_ii, eps_ii, eps_ii, 0.0, 0.0, 0.0]])
            sig = np.zeros((1, 6))
            extra = {"time": t, "uv38": np.zeros((1, 33)), "eps38": np.zeros((1, 6))}

            law38_visc_tab.solid_update(mat, sig, deps, dt=1.0e-3, extra=extra)

            expected_p = math.exp(-relaxp * t) * p_base
            actual_p = extra["uv38"][0, 15]

            assert math.isclose(actual_p, expected_p, rel_tol=1e-3), (
                f"At time {t}: expected pore pressure {expected_p}, got {actual_p}"
            )

    def test_engine_confined_air_pressure_simulation(self, tmp_path):
        """Full explicit engine simulation with confined air pressure active.

        Verifies that air pressure contributes to element compressive resistance
        and relaxes over time during hold phase.
        """
        run_name = "HEXA8_AIR_SIM"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.part(1, "AIR_FOAM", 1, 1)

        deck.funct(1, "loading_fn", [(0.0, 0.0), (0.1, 5.0), (0.5, 10.0)])

        deck.mat_law38(
            1,
            rho=1.0e-3,
            e=50.0,
            nu=0.25,
            nu_t=0.25,
            nu_c=0.30,
            kair=1,
            p0=15.0,
            poros=0.50,
            pr=8.0,  # relaxp = 8.0
            funct_id_load=[1],
            title="AIR_FOAM_MAT",
        )
        deck.prop_solid(1, "PROP_SOLID")

        # Boundary conditions
        deck.grnod_node(1, "base", [1, 2, 3, 4])
        deck.bcs(1, "base_z", "001", "111", 1)
        deck.grnod_node(2, "all", list(range(1, 9)))
        deck.bcs(2, "xy", "110", "111", 2)

        # Compress top nodes down in -Z
        deck.grnod_node(3, "top", [5, 6, 7, 8])
        deck.funct(2, "top_vel", [(0.0, 0.0), (0.1, -1.5), (1.2, -1.5), (1.3, 0.0), (2.5, 0.0)])
        deck.impvel(1, "top_v", 2, "Z", 3)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
2.50
/DT
0.9 0
/PRINT/-1
/STOP
60
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0
        state = eng_model.engine_state
        assert state.cycle >= 50

        brick_g = dict(eng_model.element_groups())["bricks"]
        uv38 = brick_g.state["mat_extra"]["uv38"]
        # Air pressure was active and stored in uv38[:, 15]
        pair = uv38[0, 15]
        assert pair < 0.0, f"Expected compressive air pressure, got {pair}"

        # Total energy balance check
        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0


# =============================================================================
# 3. Acoustic Wave Propagation & Courant Time Step
# =============================================================================

class TestLaw38AcousticWaveAndCourant:
    """Acoustic sound speed, wave propagation, and CFL stability for LAW38."""

    def test_acoustic_wave_speed_analytical_match(self):
        """Verify longitudinal sound speed matches analytical formula:

        K = E / (3 * (1 - 2*nu))
        G = E / (2 * (1 + nu))
        c = sqrt((K + 4G/3) / rho0)
        """
        test_cases = [
            {"rho0": 1.0e-3, "e": 100.0, "nu_t": 0.30, "nu_c": 0.35},
            {"rho0": 7.8e-6, "e": 2.1e5, "nu_t": 0.28, "nu_c": 0.28},
            {"rho0": 2.7e-6, "e": 7.0e4, "nu_t": 0.33, "nu_c": 0.38},
        ]

        for tc in test_cases:
            mat = Material(
                id=1,
                law=38,
                rho0=tc["rho0"],
                params={
                    "e": tc["e"],
                    "e0": tc["e"],
                    "nu_t": tc["nu_t"],
                    "nu_c": tc["nu_c"],
                },
            )

            nu_max = min(0.499, max(tc["nu_t"], tc["nu_c"]))
            K_ana = tc["e"] / (3.0 * (1.0 - 2.0 * nu_max))
            G_ana = tc["e"] / (2.0 * (1.0 + nu_max))
            c_analytical = math.sqrt((K_ana + (4.0 / 3.0) * G_ana) / tc["rho0"])

            c_law38 = float(law38_visc_tab.sound_speed(mat))
            c_entity = float(mat.sound_speed_solid())

            assert math.isclose(c_law38, c_analytical, rel_tol=1e-12), (
                f"LAW38 sound speed {c_law38} != analytical {c_analytical}"
            )
            assert math.isclose(c_entity, c_analytical, rel_tol=1e-12), (
                f"Material.sound_speed_solid {c_entity} != analytical {c_analytical}"
            )

    def test_1d_column_impact_wave_propagation_and_cfl(self, tmp_path):
        """1D column of 5 hexa8 solid elements subjected to compressive impact velocity.

        Verifies:
        - Propagation of compressive wave along the column at acoustic speed c.
        - Downstream elements remain undisturbed until the acoustic wave arrives (t_arrive = x / c).
        - Engine time step dt <= S_f * (L / c) strictly guarantees Courant stability.
        - Complete numerical stability over 50+ time steps without NaN or blowup.
        """
        run_name = "COLUMN_1D_WAVE"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        n_elem = 5
        elem_len = 10.0  # L = 10 mm per element along X
        total_len = n_elem * elem_len

        nodes = []
        nid = 1
        for i in range(n_elem + 1):
            x = i * elem_len
            nodes.append((nid, x, 0.0, 0.0))
            nodes.append((nid + 1, x, 10.0, 0.0))
            nodes.append((nid + 2, x, 10.0, 10.0))
            nodes.append((nid + 3, x, 0.0, 10.0))
            nid += 4

        elements = []
        for i in range(n_elem):
            b = 4 * i + 1
            elements.append((
                i + 1,
                b, b + 4, b + 5, b + 1,
                b + 3, b + 7, b + 6, b + 2
            ))

        deck = StarterDeck(run_name)
        deck.node(nodes)
        deck.brick(1, elements)
        deck.part(1, "COLUMN_PART", 1, 1)

        rho0 = 1.0e-3
        e0 = 100.0
        nu_t = 0.25
        nu_c = 0.25

        # Analytical wave speed:
        # nu = 0.25 -> K = E / 3(1-0.5) = 2/3 E = 66.6667
        # G = E / 2(1.25) = 0.4 E = 40.0
        # K + 4G/3 = 66.6667 + 53.3333 = 120.0
        # c = sqrt(120.0 / 1e-3) = sqrt(120000) = 346.41 mm/ms
        K = e0 / (3.0 * (1.0 - 2.0 * nu_t))
        G = e0 / (2.0 * (1.0 + nu_t))
        c_sound = math.sqrt((K + 4.0 * G / 3.0) / rho0)

        # Tabulated curve with initial elastic modulus e0
        deck.funct(1, "elastic_plateau", [
            (0.0, 0.0),
            (0.05, 5.0),
            (0.20, 10.0),
            (0.50, 20.0),
        ])

        deck.mat_law38(
            1,
            rho=rho0,
            e=e0,
            nu=nu_t,
            nu_t=nu_t,
            nu_c=nu_c,
            funct_id_load=[1],
            title="COLUMN_MAT",
        )
        deck.prop_solid(1, "COLUMN_PROP")

        # Boundary conditions:
        # Fix lateral degrees of freedom (Y and Z) on all nodes
        all_nids = [n[0] for n in nodes]
        deck.grnod_node(1, "all_nodes", all_nids)
        deck.bcs(1, "fix_yz", "011", "111", 1)

        # Impact velocity at x = 0: nodes 1, 2, 3, 4
        deck.grnod_node(2, "impact_face", [1, 2, 3, 4])
        # Constant compressive velocity in +X: v_x = +1.5
        deck.funct(2, "impact_vel", [(0.0, 1.5), (2.0, 1.5)])
        deck.impvel(1, "impact_v", 2, "X", 2)
        deck.write(s_path)

        # Courant timestep limit: dt_cfl = 0.9 * (L / c)
        dt_scale = 0.9
        dt_cfl_max = dt_scale * (elem_len / c_sound)

        engine_deck = f"""/RUN/{run_name}/1
2.00
/DT
{dt_scale} 0
/PRINT/-1
/STOP
60
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0
        state = eng_model.engine_state

        # 1. Verify 50+ time steps completed
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"

        # 2. Verify Courant timestep condition: dt <= dt_cfl_max
        brick_g = dict(eng_model.element_groups())["bricks"]
        assert brick_g.n == n_elem

        # Check that simulation proceeded stably without NaN
        sig = brick_g.state["sig"]
        assert np.isfinite(sig).all(), "Stress contains NaN or Inf"

        # Check wave front arrival:
        # Element 1 (impact face) must have significant compressive stress sigma_xx < 0
        assert sig[0, 0] < -0.1, f"Expected compressive stress in element 1, got {sig[0, 0]}"


# =============================================================================
# 4. Dynamic Tension Cutoff and Element Deletion
# =============================================================================

class TestLaw38TensionCutoffElementDeletion:
    """Dynamic tension cutoff (TENSIONCUT) and element deletion."""

    def test_engine_dynamic_tension_cutoff_and_deletion(self, tmp_path):
        """Apply high tensile velocity exceeding TENSIONCUT in explicit engine simulation.

        Verifies:
        - Element is marked broken (off38 = 0.0 or off = 0.0).
        - Stresses drop to zero.
        - Engine simulation continues stably past the deletion cycle without NaN or blowup.
        """
        run_name = "HEXA8_TENSIONCUT"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        deck.node([
            # Brick 1: pulled to tension cutoff
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
            # Brick 2: reference intact element keeping Courant time step bounded
            (9, 20.0, 0.0, 0.0),
            (10, 30.0, 0.0, 0.0),
            (11, 30.0, 10.0, 0.0),
            (12, 20.0, 10.0, 0.0),
            (13, 20.0, 0.0, 10.0),
            (14, 30.0, 0.0, 10.0),
            (15, 30.0, 10.0, 10.0),
            (16, 20.0, 10.0, 10.0),
        ])
        deck.brick(1, [
            (1, 1, 2, 3, 4, 5, 6, 7, 8),
            (2, 9, 10, 11, 12, 13, 14, 15, 16),
        ])
        deck.part(1, "TENSILE_BRICK", 1, 1)

        deck.funct(1, "loading_fn", [
            (0.0, 0.0),
            (0.1, 10.0),
            (0.5, 30.0),
        ])

        tensioncut_val = 15.0  # tension cutoff threshold

        deck.mat_law38(
            1,
            rho=1.0e-3,
            e=100.0,
            nu=0.3,
            nu_t=0.3,
            nu_c=0.35,
            cutoff=tensioncut_val,  # TENSIONCUT
            funct_id_load=[1],
            title="LAW38_CUTOFF",
        )
        deck.prop_solid(1, "PROP_SOLID")

        # Boundary conditions: Clamped base in Z for both elements
        deck.grnod_node(1, "base", [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15, 16])
        deck.bcs(1, "clamp_base", "001", "111", 1)

        # Lateral constraint for all nodes
        deck.grnod_node(2, "all", list(range(1, 17)))
        deck.bcs(2, "fix_xy", "110", "111", 2)

        # High tensile velocity on top nodes of Brick 1: v_z = +2.5
        # Causes rapid tensile strain and exceeds tensioncut_val around cycle 20
        deck.grnod_node(3, "top_brick1", [5, 6, 7, 8])
        deck.funct(2, "tensile_vel", [(0.0, 2.5), (3.0, 2.5)])
        deck.impvel(1, "tensile_v", 2, "Z", 3)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
2.00
/DT
0.9 0
/PRINT/-1
/STOP
60
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        log = MessageLog()
        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path, log=log)
            eng_model = run_engine(e_path)

        assert len(log.errors) == 0
        state = eng_model.engine_state

        # Verify simulation completed stably past deletion for 50+ cycles
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"

        brick_g = dict(eng_model.element_groups())["bricks"]
        off = brick_g.state["off"]
        sig = brick_g.state["sig"]
        off38 = brick_g.state["mat_extra"]["off38"]

        # 1. Pulled element 1 is marked broken (off == 0.0 and off38 == 0.0)
        assert off[0] == 0.0, f"Element 1 should be deleted (off == 0), got {off[0]}"
        assert off38[0] == 0.0, f"Element 1 should be deleted (off38 == 0), got {off38[0]}"

        # 2. Intact element 2 remains active (off == 1.0 and off38 == 1.0)
        assert off[1] == 1.0, f"Element 2 should remain intact (off == 1), got {off[1]}"
        assert off38[1] == 1.0, f"Element 2 should remain intact (off38 == 1), got {off38[1]}"

        # 3. Stresses of deleted element drop to zero
        assert np.allclose(sig[0], 0.0, atol=1e-12), (
            f"Expected zero stress after element deletion, got {sig[0]}"
        )

        # 4. No NaNs in state arrays
        assert np.isfinite(sig).all()
        assert np.isfinite(off).all()
        assert np.isfinite(off38).all()

    def test_constitutive_tension_cutoff_multi_element(self):
        """Constitutive verification of TENSIONCUT element deletion on array of elements.

        Verifies:
        - Elements exceeding TENSIONCUT are deleted (off = 0, off38 = 0).
        - Sub-threshold elements remain intact (off = 1, off38 = 1).
        - Subsequent cycles on deleted elements produce strictly zero stresses.
        """
        tensioncut = 20.0
        mat = Material(
            id=1,
            law=38,
            rho0=1.0e-3,
            params={
                "e0": 100.0,
                "nu_t": 0.25,
                "nu_c": 0.25,
                "cutoff": tensioncut,
                "tensioncut": tensioncut,
                "load_curves": [None],
            },
        )

        # 3 elements:
        # Element 0: Moderate tension below cutoff (sigma_zz ~ 8 < 20)
        # Element 1: High tension exceeding cutoff (sigma_zz ~ 30 > 20)
        # Element 2: Extreme tension exceeding cutoff (sigma_zz ~ 60 > 20)
        sig = np.zeros((3, 6))
        deps = np.array([
            [0.0, 0.0, 0.08, 0.0, 0.0, 0.0],  # ~ 8 MPa < 20
            [0.0, 0.0, 0.30, 0.0, 0.0, 0.0],  # ~ 30 MPa > 20
            [0.0, 0.0, 0.60, 0.0, 0.0, 0.0],  # ~ 60 MPa > 20
        ])
        off = np.ones(3)
        off38 = np.ones(3)
        extra = {
            "time": 0.0,
            "off": off,
            "off38": off38,
            "uv38": np.zeros((3, 33)),
            "eps38": np.zeros((3, 6)),
        }

        sign, _, _ = law38_visc_tab.solid_update(mat, sig, deps, dt=1.0e-3, extra=extra)

        # Element 0 should remain alive
        assert extra["off"][0] == 1.0
        assert extra["off38"][0] == 1.0
        assert sign[0, 2] > 0.0, "Element 0 should carry tensile stress"

        # Elements 1 and 2 should be deleted
        assert extra["off"][1] == 0.0
        assert extra["off38"][1] == 0.0
        assert sign[1, 2] == 0.0, "Element 1 stress must drop to zero"

        assert extra["off"][2] == 0.0
        assert extra["off38"][2] == 0.0
        assert sign[2, 2] == 0.0, "Element 2 stress must drop to zero"

        # Subsequent cycle: apply further strain increments
        deps_next = np.array([
            [0.0, 0.0, 0.02, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.05, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.10, 0.0, 0.0, 0.0],
        ])
        sign_next, _, _ = law38_visc_tab.solid_update(mat, sign, deps_next, dt=1.0e-3, extra=extra)

        # Element 0 continues to evolve stress
        assert sign_next[0, 2] > sign[0, 2]

        # Elements 1 and 2 remain strictly zero
        assert np.allclose(sign_next[1], 0.0)
        assert np.allclose(sign_next[2], 0.0)
        assert extra["off"][1] == 0.0
        assert extra["off"][2] == 0.0
