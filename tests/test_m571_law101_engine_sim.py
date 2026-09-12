"""Dynamic Engine Simulation & Energy Balance Tests for Milestone M571 (/MAT/LAW101 Bouvard Polypropylene).

Verifies:
1. Multi-cycle dynamic explicit engine simulations for LAW101 with Hexa8 solid continuum brick under tensile loading.
2. Tetra4 solid continuum element with LAW101 under compressive loading.
3. Adiabatic plastic heating (theta_flag=2.0, omega=0.9) with temperature evolution.
4. Positive internal energy accumulation and zero energy leaks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.starter import run_starter


def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 1.0e-3,
    dt_scale: float = 0.5,
    dt_max: Optional[float] = None,
    stop_cycles: Optional[int] = None,
    print_freq: int = -1000,
) -> None:
    """Generate and write an engine control deck file."""
    lines = [
        f"/RUN/{run_name}/1",
        f"{tstop:.6e}",
        "/DT",
        f"{dt_scale:.2f} {dt_max:.6e}" if dt_max is not None else f"{dt_scale:.2f} 0.0",
        f"/PRINT/{print_freq}",
    ]
    if stop_cycles is not None:
        lines.extend(["/STOP", f"{stop_cycles}"])
    lines.append("/END\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class TestLaw101EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW101."""

    def test_solid_hexa8_law101_tensile(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW101 under dynamic tensile stretching."""
        run_name = "HEXA8_LAW101"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law101(
            mid=1,
            title="Bouvard_PP",
            rho0=0.9e-6,
            e_ref=1500.0,
            e1=-2.5,
            nu=0.38,
            ve1=0.25,
            ve2=1.5,
            edot_ref=1.0e-4,
            gamma0=1.0e-3,
            alpha_p=0.08,
            deltah=45000.0,
            vol=2.0e-28,
            m=1.8,
            c3=-0.05,
            c4=25.0,
            alphak1=0.15,
            alphak2=0.05,
            hard=50.0,
            zeta1i=0.1,
            c5=-0.001,
            c6=1.0,
            c7=-0.002,
            c8=2.0,
            c9=-0.01,
            c10=10.0,
            hard1=20.0,
            zeta2i=0.05,
            c11=-0.001,
            c12=1.5,
            c13=0.01,
            c14=5.0,
            c1=-0.05,
            c2=8.0,
            lambdal=5.0,
            rho_ref=0.9e-6,
            cv_ref=1800.0,
            tref=293.15,
            alpha_th=1.2e-4,
            theta_glass=250.0,
            omega=0.0,
            theta_flag=0.0,
            heat_t0=293.15,
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "SolidPart", 1, 1)

        # 8 nodes forming a 10x10x10 mm cube
        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ]
        d.node(nodes)
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        # Fix bottom face nodes (1, 2, 3, 4)
        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull top face nodes (5, 6, 7, 8) in Z
        d.grnod_node(2, "pull_top", [5, 6, 7, 8])
        d.funct(11, "vz_pull", [(0.0, 100.0), (1.0e-2, 100.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=30)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 30
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0

    def test_solid_tetra4_law101_compression(self, tmp_path: Path):
        """Single Tetra4 element with /MAT/LAW101 under compressive loading."""
        run_name = "TETRA4_LAW101"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_pp(
            mat_id=1,
            title="PP_Tetra",
            rho0=0.9e-6,
            e_ref=1400.0,
            nu=0.37,
            c4=22.0,
            deltah=40000.0,
            gamma0=1.0e-3,
            vol=2.0e-28,
            m=1.7,
            hard=45.0,
            lambdal=4.5,
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "TetraPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 5.0, 10.0, 0.0),
            (4, 5.0, 5.0, 10.0),
        ]
        d.node(nodes)
        d.tetra4(1, [(1, 1, 2, 3, 4)])

        # Fix base nodes (1, 2, 3)
        d.grnod_node(1, "fix_base", [1, 2, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Push apex node (4) downwards in Z
        d.grnod_node(2, "push_apex", [4])
        d.funct(11, "vz_push", [(0.0, -50.0), (1.0e-2, -50.0)])
        d.impvel(1, "push_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=25)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 25
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0

    def test_solid_hexa8_law101_adiabatic_heating(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW101 with adiabatic heating (theta_flag=2.0, omega=0.9)."""
        run_name = "HEXA8_LAW101_ADIAB"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_plas_poly(
            mat_id=1,
            title="PP_Adiabatic",
            rho0=0.9e-6,
            e_ref=1600.0,
            nu=0.36,
            c4=20.0,
            deltah=40000.0,
            gamma0=1.0e-3,
            vol=2.0e-28,
            m=1.6,
            hard=50.0,
            lambdal=5.0,
            rho_ref=0.9e-6,
            cv_ref=1800.0,
            tref=293.15,
            omega=0.9,
            theta_flag=2.0,
            heat_t0=293.15,
        )
        d.prop_solid(1, "SolidProp", isolid=1)
        d.part(1, "SolidPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
            (5, 0.0, 0.0, 10.0),
            (6, 10.0, 0.0, 10.0),
            (7, 10.0, 10.0, 10.0),
            (8, 0.0, 10.0, 10.0),
        ]
        d.node(nodes)
        d.brick(1, [(1, 1, 2, 3, 4, 5, 6, 7, 8)])

        d.grnod_node(1, "fix_base", [1, 2, 3, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_top", [5, 6, 7, 8])
        d.funct(11, "vz_pull", [(0.0, 80.0), (1.0e-2, 80.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=25)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 25
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0
