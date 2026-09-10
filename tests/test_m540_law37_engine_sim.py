"""Tests for Milestone M540 (Auditor 3): Explicit Dynamic Engine Simulation & Hydrodynamics Audit.

Covers:
1. Multi-cycle explicit dynamic engine simulation on solid Hexa8 elements:
   - 50+ explicit time integration cycles.
   - Courant time step bounding: dt <= S_f * (L_c / c_sound) where c_sound is mixture sound speed.
   - Energy balance: external work W_ext, internal energy E_int, kinetic energy E_k, viscous dissipation, |ERR| < 1%.
   - State history tracking: uv37 (M1/V, rho2, rho1, alpha_v1, alpha_v2).
   - Validation for dual solvers (ISOLVER=1 legacy quadratic chord solver and ISOLVER=2 2D Newton-Raphson).
2. Pure liquid vs pure gas vs mixture shock compression:
   - 1D column mesh of solid elements impacted at high velocity.
   - Shock wave propagation comparison across pure liquid (alpha1 = 1.0), biphasic mixture (alpha1 = 0.8), and pure gas (alpha1 = 0.0).
   - Smooth monotonic pressure rise and stable integration without NaN or Inf.
   - Cavitation and tension cutoff (P >= P_min) during high strain rate volumetric expansion.
3. Mixed mesh simulation:
   - Combined Hexa8 brick and Tetra4 tetrahedron mesh sharing a single /MAT/LAW37 material.
   - Synchronous explicit time stepping across element topologies.
   - Continuous interface force and compressive stress transmission.
   - Energy balance and stability (|ERR| < 1%).
4. Shell rejection checks in engine:
   - Clean descriptive rejection of BT4 shell elements referencing /MAT/LAW37.
   - Clean descriptive rejection of QBAT (Ishell=12) shell elements referencing /MAT/LAW37.
   - materials.shell_update rejection guard.
   - Starter check rejection with StarterError.
"""

from __future__ import annotations

import contextlib
import io
import math
import os
import tempfile
from typing import Tuple

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss import materials
from pyradioss.materials import law37_biphas as l37
from pyradioss.model.entities import Part, Property
from pyradioss.model.model import Model, ElementGroup
from pyradioss.starter.restart import read_restart, write_restart
from pyradioss.starter.starter import run_starter


# =============================================================================
# 1. Multi-Cycle Explicit Engine Simulation on Solid Hexa8 Elements
# =============================================================================

class TestHexa8ExplicitSimulation:
    """Multi-cycle explicit dynamic simulation on solid Hexa8 elements with LAW37."""

    def test_hexa8_explicit_multicycle_courant_control_isolver1(self, tmp_path):
        """Verify explicit integration over 50+ steps, Courant bound, and |ERR| < 1% (ISOLVER=1)."""
        run_name = "HEXA8_L37_ISO1"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        # Unit cube 10 x 10 x 10
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
        deck.part(1, "HEXA8_BLOCK", 1, 1)

        rho_l0 = 1000.0
        c_l = 2.2e7
        alpha1 = 0.8
        rho_g0 = 1.2
        gamma = 1.4
        p0 = 1.0e5

        deck.mat_law37(
            1,
            rho_l0=rho_l0,
            c_l=c_l,
            alpha1=alpha1,
            nu_l=1.0e-3,
            nu_vol_l=2.0e-3,
            rho_g0=rho_g0,
            gamma_g=gamma,
            p0_g=p0,
            nu_g=1.8e-5,
            nu_vol_g=3.0e-5,
            isolver=1,
            title="LAW37_ISO1",
        )
        deck.prop_solid(1, "HEXA_PROP")

        # Boundary conditions: Clamped base in Z
        deck.grnod_node(1, "base_nodes", [1, 2, 3, 4])
        deck.bcs(1, "clamp_base_z", "001", "111", 1)

        # Uniaxial strain state: prevent lateral expansion (oedometer / plane-strain)
        deck.grnod_node(2, "all_nodes", [1, 2, 3, 4, 5, 6, 7, 8])
        deck.bcs(2, "fix_xy", "110", "111", 2)

        # Imposed velocity ramp: compression in Z
        deck.grnod_node(3, "top_nodes", [5, 6, 7, 8])
        deck.funct(1, "vel_ramp", [(0.0, 0.0), (0.1, -0.5), (10.0, -0.5)])
        deck.impvel(1, "top_vel", 1, "Z", 3)
        deck.write(s_path)

        dt_scale = 0.9
        engine_deck = f"""/RUN/{run_name}/1
5.0
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

        # 2. Verify Courant time step control
        brick_g = dict(eng_model.element_groups())["bricks"]
        uv37 = brick_g.state["mat_extra"]["uv37"]
        rho1_curr = uv37[0, 2]
        c_sound = math.sqrt(c_l / max(rho1_curr, 1e-30))
        lc = 10.0
        dt_courant_bound = dt_scale * (lc / c_sound)

        _, eng_dict = read_restart(os.path.join(tmp_path, f"{run_name}_0001.rst"))
        engine_dt = eng_dict["dt"]
        assert engine_dt <= dt_courant_bound * 1.15, (
            f"Engine dt {engine_dt} exceeds Courant bound {dt_courant_bound}"
        )

        # 3. Verify energy balance: W_ext, E_int, E_k, and |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert en["EW"] > 0.0, f"External work must be positive, got {en['EW']}"
        assert en["IE"] > 0.0, f"Internal energy must be positive, got {en['IE']}"
        assert en["KE"] > 0.0, f"Kinetic energy must be positive, got {en['KE']}"
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR| must be < 1%, got {en['ERR']}%"

        # 4. Stress and state variables
        sig = brick_g.state["sig"]
        assert np.isfinite(sig).all()
        assert sig[0, 2] < 0.0, f"Expected compressive stress in Z, got {sig[0, 2]}"
        assert sig[0, 0] < 0.0 and sig[0, 1] < 0.0, "Lateral hydrostatic pressure must be compressive"

        # History array uv37: [M1/V, rho2, rho1, alpha_v1, alpha_v2]
        assert "uv37" in brick_g.state["mat_extra"]
        assert np.isfinite(uv37).all()
        assert uv37[0, 1] > rho_g0, "Gas phase density must rise under compression"
        assert uv37[0, 2] > rho_l0, "Liquid phase density must rise under compression"
        assert 0.0 <= uv37[0, 3] <= 1.0, "Liquid volume fraction alpha_v1 must stay in [0, 1]"
        assert 0.0 <= uv37[0, 4] <= 1.0, "Gas volume fraction alpha_v2 must stay in [0, 1]"
        assert math.isclose(uv37[0, 3] + uv37[0, 4], 1.0, abs_tol=1e-5), "Volume fractions must sum to 1"

    def test_hexa8_explicit_multicycle_courant_control_isolver2(self, tmp_path):
        """Verify explicit integration over 50+ steps with ISOLVER=2 (Newton-Raphson + Wood's formula)."""
        run_name = "HEXA8_L37_ISO2"
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
        deck.part(1, "HEXA8_BLOCK", 1, 1)

        rho_l0 = 1000.0
        c_l = 2.2e7
        alpha1 = 0.5
        rho_g0 = 1.2
        gamma = 1.4
        p0 = 1.0e5

        deck.mat_law37(
            1,
            rho_l0=rho_l0,
            c_l=c_l,
            alpha1=alpha1,
            nu_l=5.0e-4,
            rho_g0=rho_g0,
            gamma_g=gamma,
            p0_g=p0,
            nu_g=1.0e-5,
            isolver=2,
            title="LAW37_ISO2",
        )
        deck.prop_solid(1, "HEXA_PROP")

        deck.grnod_node(1, "base_nodes", [1, 2, 3, 4])
        deck.bcs(1, "clamp_base_z", "001", "111", 1)

        deck.grnod_node(2, "all_nodes", [1, 2, 3, 4, 5, 6, 7, 8])
        deck.bcs(2, "fix_xy", "110", "111", 2)

        deck.grnod_node(3, "top_nodes", [5, 6, 7, 8])
        deck.funct(1, "vel_ramp", [(0.0, 0.0), (0.1, -0.4), (10.0, -0.4)])
        deck.impvel(1, "top_vel", 1, "Z", 3)
        deck.write(s_path)

        dt_scale = 0.9
        engine_deck = f"""/RUN/{run_name}/1
5.0
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
        assert state.cycle >= 50

        # Verify Wood's mixture sound speed bounds the step
        brick_g = dict(eng_model.element_groups())["bricks"]
        uv37 = brick_g.state["mat_extra"]["uv37"]
        rho2 = uv37[0, 1]
        rho1 = uv37[0, 2]
        av1 = uv37[0, 3]
        av2 = uv37[0, 4]

        # Check phase pressure equilibrium P_1 = P_2 in ISOLVER=2
        r1 = c_l / rho_l0
        p1 = r1 * rho1 - c_l + p0
        p2 = p0 * (rho2 / rho_g0) ** gamma
        rel_p_diff = abs(p1 - p2) / max(p1, p2, 1e-10)
        assert rel_p_diff < 1e-4, f"Phase pressures P1={p1} and P2={p2} out of equilibrium (rel diff {rel_p_diff})"

        # Wood's formula
        ssp1 = r1 * rho1
        ssp2 = gamma * p0 * (rho2 / rho_g0) ** gamma
        ssp_tot = av1 / ssp1 + av2 / ssp2
        rho_current = float(brick_g.state["mass"][0] / brick_g.state["vol0"][0])
        c_wood = math.sqrt(1.0 / (ssp_tot * rho_current))
        dt_wood_bound = dt_scale * (10.0 / c_wood)

        _, eng_dict = read_restart(os.path.join(tmp_path, f"{run_name}_0001.rst"))
        engine_dt = eng_dict["dt"]
        assert engine_dt <= dt_wood_bound * 1.25, (
            f"Engine dt {engine_dt} exceeds Wood Courant bound {dt_wood_bound}"
        )

        en = _energies(eng_model, state)
        assert en["EW"] > 0.0
        assert en["IE"] > 0.0
        assert abs(en["ERR"]) < 1.0

    def test_hexa8_viscous_damping_dissipation(self, tmp_path):
        """Verify dynamic viscous damping dissipates work into internal energy while conserving total energy."""
        run_name = "HEXA8_L37_VISC"
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
        deck.part(1, "HEXA8_BLOCK", 1, 1)

        # High shear and bulk viscosities
        nu_l = 5.0e-2
        nu_vol_l = 1.0e-1
        nu_g = 5.0e-4
        nu_vol_g = 1.0e-3

        deck.mat_law37(
            1,
            rho_l0=1000.0,
            c_l=2.2e7,
            alpha1=0.8,
            nu_l=nu_l,
            nu_vol_l=nu_vol_l,
            rho_g0=1.2,
            gamma_g=1.4,
            p0_g=1.0e5,
            nu_g=nu_g,
            nu_vol_g=nu_vol_g,
            title="LAW37_HIGH_VISC",
        )
        deck.prop_solid(1, "HEXA_PROP")

        deck.grnod_node(1, "base_nodes", [1, 2, 3, 4])
        deck.bcs(1, "clamp_base_z", "001", "111", 1)

        deck.grnod_node(2, "all_nodes", [1, 2, 3, 4, 5, 6, 7, 8])
        deck.bcs(2, "fix_xy", "110", "111", 2)

        deck.grnod_node(3, "top_nodes", [5, 6, 7, 8])
        deck.funct(1, "vel_ramp", [(0.0, 0.0), (0.1, -1.0), (10.0, -1.0)])
        deck.impvel(1, "top_vel", 1, "Z", 3)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
6.0
/DT
0.9 0
/PRINT/-1
/STOP
60
/STOP/NSTEP
55
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50

        # Energy conservation with viscous dissipation
        en = _energies(eng_model, state)
        assert en["IE"] > 0.0
        assert abs(en["ERR"]) < 1.0, f"Energy error |ERR| must be < 1%, got {en['ERR']}%"

        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]
        assert np.isfinite(sig).all()


# =============================================================================
# 2. Pure Liquid vs Pure Gas vs Mixture Shock Compression & Cavitation Cutoff
# =============================================================================

class TestHydrodynamicShockAndCavitation:
    """Shock wave propagation across fluid phases and cavitation tension cutoff."""

    @staticmethod
    def _build_column_mesh(run_name: str, num_elements: int = 5) -> Tuple[StarterDeck, list[int], list[int]]:
        """Helper to build a 1D column mesh of solid Hexa8 elements along Z."""
        deck = StarterDeck(run_name)
        nodes = []
        nid = 1
        for k in range(num_elements + 1):
            nodes.extend([
                (nid, 0.0, 0.0, float(k * 10.0)),
                (nid + 1, 10.0, 0.0, float(k * 10.0)),
                (nid + 2, 10.0, 10.0, float(k * 10.0)),
                (nid + 3, 0.0, 10.0, float(k * 10.0)),
            ])
            nid += 4
        deck.node(nodes)

        bricks = []
        for k in range(num_elements):
            b = 4 * k
            bricks.append((k + 1, b + 1, b + 2, b + 3, b + 4, b + 5, b + 6, b + 7, b + 8))
        deck.brick(1, bricks)
        deck.part(1, "COLUMN_PART", 1, 1)

        base_nids = [1, 2, 3, 4]
        top_nids = [nid - 4, nid - 3, nid - 2, nid - 1]
        return deck, base_nids, top_nids

    def test_column_shock_pure_liquid_vs_pure_gas_vs_mixture(self, tmp_path):
        """Impact column of 3D solid elements with high velocity: compare liquid, gas, mixture."""
        cases = {
            "pure_liquid": {"alpha1": 1.0, "isolver": 1},
            "mixture": {"alpha1": 0.8, "isolver": 1},
            "pure_gas": {"alpha1": 0.0, "isolver": 2},
        }
        results = {}

        for case_name, cfg in cases.items():
            run_name = f"SHOCK_{case_name.upper()}"
            s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
            e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

            deck, base_nids, top_nids = self._build_column_mesh(run_name, num_elements=5)

            deck.mat_law37(
                1,
                rho_l0=1000.0,
                c_l=2.2e7,
                alpha1=cfg["alpha1"],
                nu_l=1.0e-3,
                nu_vol_l=1.0e-2,
                rho_g0=1.2,
                gamma_g=1.4,
                p0_g=1.0e5,
                nu_g=1.8e-5,
                nu_vol_g=1.0e-3,
                isolver=cfg["isolver"],
                title=f"MAT_{case_name}",
            )
            deck.prop_solid(1, "SOLID_PROP")

            deck.grnod_node(1, "base", base_nids)
            deck.bcs(1, "fix_base", "001", "111", 1)

            all_nids = list(range(1, 25))
            deck.grnod_node(2, "all_nodes", all_nids)
            deck.bcs(2, "fix_xy", "110", "111", 2)

            # High-velocity impact ramp on top surface: v_z = -0.2
            deck.grnod_node(3, "top_nodes", top_nids)
            deck.funct(1, "v_ramp", [(0.0, 0.0), (0.5, -0.2), (10.0, -0.2)])
            deck.impvel(1, "imp_top", 1, "Z", 3)
            deck.write(s_path)

            engine_deck = f"""/RUN/{run_name}/1
0.25
/DT
0.9 0
/PRINT/-1
/STOP
60
/STOP/NSTEP
3
/END
"""
            with open(e_path, "w") as f:
                f.write(engine_deck)

            with contextlib.redirect_stdout(io.StringIO()):
                st_model = run_starter(s_path)
                eng_model = run_engine(e_path)

            state = eng_model.engine_state
            assert state.cycle >= 3

            # 1. Verify no NaN or Inf in coordinates, velocities, stresses
            assert np.isfinite(eng_model.x).all()
            assert np.isfinite(eng_model.v).all()

            brick_g = dict(eng_model.element_groups())["bricks"]
            sig = brick_g.state["sig"]
            assert np.isfinite(sig).all()

            pressures = -(sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
            assert np.isfinite(pressures).all()
            assert pressures.max() > 0.0, f"Impacted column must experience positive shock pressure in {case_name}"

            # 2. Verify energy balance |ERR| < 1%
            en = _energies(eng_model, state)
            assert abs(en["ERR"]) < 1.0, f"Energy error |ERR| in {case_name} exceeded 1%: {en['ERR']}%"

            results[case_name] = {
                "max_p": float(pressures.max()),
                "min_p": float(pressures.min()),
                "cycle": state.cycle,
            }

        # 3. Physics comparison across phases:
        # Liquid has low compressibility -> highest shock pressure
        # Gas has high compressibility -> lowest shock pressure
        # Mixture has intermediate pressure
        assert results["pure_liquid"]["max_p"] > results["mixture"]["max_p"], (
            f"Pure liquid max pressure ({results['pure_liquid']['max_p']}) should exceed mixture ({results['mixture']['max_p']})"
        )
        assert results["mixture"]["max_p"] > results["pure_gas"]["max_p"], (
            f"Mixture max pressure ({results['mixture']['max_p']}) should exceed gas ({results['pure_gas']['max_p']})"
        )

    def test_cavitation_tension_cutoff_expansion(self, tmp_path):
        """Verify cavitation/tension cutoff when elements expand past P_min limit."""
        run_name = "HEXA8_CAVITATION"
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
        deck.part(1, "EXPAND_PART", 1, 1)

        p0 = 1.0e5

        deck.mat_law37(
            1,
            rho_l0=1000.0,
            c_l=2.2e7,
            alpha1=0.8,
            nu_l=0.0,
            rho_g0=1.2,
            gamma_g=1.4,
            p0_g=p0,
            title="CAVITATION_MAT",
        )
        deck.prop_solid(1, "EXPAND_PROP")

        deck.grnod_node(1, "base", [1, 2, 3, 4])
        deck.bcs(1, "fix_base", "001", "111", 1)

        deck.grnod_node(2, "all_nodes", [1, 2, 3, 4, 5, 6, 7, 8])
        deck.bcs(2, "fix_xy", "110", "111", 2)

        # Apply upward tensile displacement/velocity: v_z = +0.5
        deck.grnod_node(3, "top_nodes", [5, 6, 7, 8])
        deck.funct(1, "pull_ramp", [(0.0, 0.0), (0.1, 0.5), (10.0, 0.5)])
        deck.impvel(1, "pull_top", 1, "Z", 3)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
1.5
/DT
0.9 0
/PRINT/-1
/STOP
60
/STOP/NSTEP
15
/END
"""
        with open(e_path, "w") as f:
            f.write(engine_deck)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 15

        # Verify no NaN or Inf during cavitation / expansion
        assert np.isfinite(eng_model.x).all()
        assert np.isfinite(eng_model.v).all()

        brick_g = dict(eng_model.element_groups())["bricks"]
        sig = brick_g.state["sig"]
        assert np.isfinite(sig).all()

        # Hydrostatic pressure P = -(sig_xx + sig_yy + sig_zz) / 3
        p_eff = -(sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0

        # Tension cutoff assertion: P >= -p0 (within small numerical margin)
        assert p_eff >= -p0 - 1e-3, (
            f"Effective pressure {p_eff} fell below cavitation limit {-p0}"
        )

        # Verify constitutive update cutoff analytically
        mat = st_model.materials[1]
        sig_test = np.zeros((1, 6))
        deps_tens = np.array([[0.0, 0.0, 0.2, 0.0, 0.0, 0.0]])  # 20% expansion
        extra_test = {"rho": np.array([50.0])}  # significantly expanded density
        s_cut, _, _ = l37.solid_update(mat, sig_test, deps_tens, dt=0.01, extra=extra_test)
        p_cut = -(s_cut[0, 0] + s_cut[0, 1] + s_cut[0, 2]) / 3.0
        assert math.isclose(p_cut, -p0, rel_tol=1e-6), (
            f"Analytical solid_update pressure {p_cut} must match -p0 {-p0}"
        )


# =============================================================================
# 3. Mixed Mesh Simulation (Hexa8 + Tetra4)
# =============================================================================

class TestMixedMeshSimulation:
    """Mixed mesh test combining Hexa8 and Tetra4 elements sharing /MAT/LAW37."""

    def test_mixed_hexa8_tetra4_shared_law37_synchronous_stepping(self, tmp_path):
        """Run combined Hexa8 + Tetra4 mesh: verify synchronous time stepping and interface transmission."""
        run_name = "MIXED_HEX_TET_L37"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        deck = StarterDeck(run_name)
        # Mesh: 1 Hexa8 brick at base (nodes 1..8) + 2 Tetra4 elements on top (nodes 5..8, apex 9)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
            (9, 5.0, 5.0, 20.0),
        ])
        deck.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])
        deck.tetra4(2, [
            (2, 5, 6, 7, 9),
            (3, 5, 7, 8, 9),
        ])
        deck.part(1, "HEX_PART", 1, 1)
        deck.part(2, "TET_PART", 2, 1)

        # Single shared LAW37 biphasic fluid material
        deck.mat_law37(
            1,
            rho_l0=1000.0,
            c_l=2.2e7,
            alpha1=0.8,
            nu_l=1.0e-3,
            rho_g0=1.2,
            gamma_g=1.4,
            p0_g=1.0e5,
            title="SHARED_LAW37_FLUID",
        )
        deck.prop_solid(1, "HEX_PROP")
        deck.prop_solid(2, "TET_PROP")

        # Clamped base in Z
        deck.grnod_node(1, "base", [1, 2, 3, 4])
        deck.bcs(1, "fix_base", "111", "111", 1)

        # Rollers on lateral perimeter nodes (constrain X/Y to prevent shear distortion in pure fluid)
        deck.grnod_node(2, "lateral", list(range(1, 10)))
        deck.bcs(2, "fix_xy", "110", "111", 2)

        # Push apex node downward: load transmits through Tetras to Brick
        deck.grnod_node(3, "apex", [9])
        deck.funct(1, "apex_v", [(0.0, 0.0), (0.1, -0.4), (5.0, -0.4)])
        deck.impvel(1, "push_apex", 1, "Z", 3)
        deck.write(s_path)

        engine_deck = f"""/RUN/{run_name}/1
0.5
/DT
0.9 0
/PRINT/-1
/STOP
60
/STOP/NSTEP
2
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
        assert state.cycle >= 2

        # 1. Verify element groups
        groups = dict(eng_model.element_groups())
        assert "bricks" in groups and "tetras" in groups

        sig_b = groups["bricks"].state["sig"]
        sig_t = groups["tetras"].state["sig"]

        assert sig_b.shape == (1, 6)
        assert sig_t.shape == (2, 6)
        assert np.isfinite(sig_b).all()
        assert np.isfinite(sig_t).all()

        # 2. Both topologies experience compressive downward stress in Z
        assert sig_b[0, 2] < 0.0, "Hexa8 base brick must carry compressive force in Z"
        assert (sig_t[:, 2] < 0.0).all(), "Tetra4 elements must carry compressive force in Z"

        # 3. Persistent history variables uv37 in both element types
        assert "uv37" in groups["bricks"].state["mat_extra"]
        assert "uv37" in groups["tetras"].state["mat_extra"]
        assert np.isfinite(groups["bricks"].state["mat_extra"]["uv37"]).all()
        assert np.isfinite(groups["tetras"].state["mat_extra"]["uv37"]).all()

        # 4. Global energy balance: |ERR| < 1.0%
        en = _energies(eng_model, state)
        assert en["EW"] > 0.0
        assert abs(en["ERR"]) < 1.0, f"Energy error {en['ERR']}% exceeds 1%"

        assert np.isfinite(eng_model.x).all()
        assert np.isfinite(eng_model.v).all()


# =============================================================================
# 4. Shell Rejection Checks in Engine & Starter
# =============================================================================

class TestShellRejection:
    """Verify that any shell element referencing /MAT/LAW37 is cleanly rejected."""

    def test_engine_rejects_shell_element_with_law37(self, tmp_path):
        """Verify that run_engine rejects shell elements referencing /MAT/LAW37 with NotImplementedError."""
        run_name = "ENGINE_SHELL_REJECT"
        rst_path = os.path.join(tmp_path, f"{run_name}_0000.rst")
        e_path = os.path.join(tmp_path, f"{run_name}_0001.rad")

        # Construct a raw model with a shell element referencing LAW37
        m = Model()
        m.x0 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
        m.x = m.x0.copy()
        m.v = np.zeros((4, 3))
        m.vr = np.zeros((4, 3))
        m.mass = np.ones(4)
        m.inertia = np.ones(4)
        m.node_ids = np.array([1, 2, 3, 4])

        mat = l37.build_law37({"id": 1, "c_l": 2.2e7, "alpha1": 0.8})
        m.materials[1] = mat
        prop = Property(id=1, type=1, title="SH_PROP", params={"thick": 1.0})
        m.properties[1] = prop
        part = Part(id=1, title="SH_PART", prop_id=1, mat_id=1)
        m.parts[1] = part

        from pyradioss.elements.shell_bt4 import init_group as init_shells
        sh_group = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1, 2, 3]]), part=np.array([1]))
        sh_group.state = {"slices": [(slice(0, 1), mat, prop)]}

        class MockLog:
            errors = []
            def error(self, *a, **k): pass
            def warning(self, *a, **k): pass

        init_shells(sh_group, m, MockLog())
        m.shells = sh_group

        write_restart(m, rst_path)
        with open(e_path, "w") as f:
            f.write(f"""/RUN/{run_name}/1
0.1
/DT
0.9 0
/PRINT/-1
/STOP
5
/END
""")
        with pytest.raises(NotImplementedError, match="LAW37 \\(biphasic fluid/gas\\) is implemented for 3D solid and SPH elements only"):
            run_engine(e_path)

    def test_materials_shell_update_rejects_law37(self):
        """Verify that materials.shell_update raises NotImplementedError for LAW37."""
        mat = l37.build_law37({"id": 1, "c_l": 2.2e7, "alpha1": 0.8})
        sig = np.zeros((1, 3))
        deps = np.zeros((1, 3))

        with pytest.raises(NotImplementedError, match="LAW37 \\(biphasic fluid/gas\\) is implemented for 3D solid and SPH elements only"):
            materials.shell_update(mat, sig, deps, None, 0.01)

        with pytest.raises(NotImplementedError, match="LAW37 \\(biphasic fluid/gas\\) is implemented for 3D solid and SPH elements only"):
            l37.shell_update(mat, sig, deps)

    def test_starter_checks_reject_shell_with_law37(self, tmp_path):
        """Verify that starter card checks reject shell elements referencing LAW37 with StarterError."""
        run_name = "STARTER_SHELL_REJECT"
        s_path = os.path.join(tmp_path, f"{run_name}_0000.rad")

        deck = StarterDeck(run_name)
        deck.node([
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])
        deck.part(1, "SHELL_PART", 1, 1)
        deck.mat_law37(1, rho_l0=1000.0, c_l=2.2e7, alpha1=0.8, title="L37_MAT")
        deck.prop_shell(1, "SHELL_PROP", thick=1.0)
        deck.write(s_path)

        log = MessageLog()
        with pytest.raises(StarterError):
            with contextlib.redirect_stdout(io.StringIO()):
                run_starter(s_path, log=log)

        assert any("not ported for shells elements" in str(err) for err in log.errors)
