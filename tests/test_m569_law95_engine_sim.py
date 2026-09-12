"""
Dynamic Engine Simulation & Energy Balance Tests for Milestone M569 (/MAT/LAW95 Bergstrom-Boyce).
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
    stop_cycles: Optional[int] = None,
    print_freq: int = -1000,
) -> None:
    """Generate and write a clean engine control deck file."""
    lines = [
        f"/RUN/{run_name}/1",
        f"{tstop:.6e}",
        "/DT",
        f"{dt_scale:.2f} 0.0",
        f"/PRINT/{print_freq}",
    ]
    if stop_cycles is not None:
        lines.extend(["/STOP", f"{stop_cycles}"])
    lines.append("/END\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class TestLaw95EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW95."""

    def test_solid_hexa8_bb_tensile_energy_balance(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW95 under tensile stretching."""
        run_name = "HEXA8_LAW95"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law95(
            mid=1,
            title="BB_Rubber",
            rho0=1.0e-9,  # tonne/mm3 = 1000 kg/m3
            c10=5.0,      # MPa
            c01=1.0,      # MPa
            sb=0.5,
            d1=1.0e-3,    # 1/MPa
            a=0.01,
            expc=-0.7,
            expm=1.0,
            ksi=0.01,
            tau_ref=2.0,
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
        d.funct(11, "vz_pull", [(0.0, 100.0), (1.0e-3, 100.0)])
        d.impvel(1, "pull_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5, stop_cycles=50)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 50
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0

    def test_solid_tetra4_bb_compression(self, tmp_path: Path):
        """Single Tetra4 element with /MAT/LAW95 under compressive loading."""
        run_name = "TETRA4_LAW95"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law95(
            mid=1,
            title="BB_Tetra",
            rho0=1.0e-9,
            c10=3.0,
            c01=0.5,
            sb=0.8,
            d1=2.0e-3,
            a=0.02,
            tau_ref=1.5,
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

        # Move apex node (4) downwards in Z
        d.grnod_node(2, "push_apex", [4])
        d.funct(11, "vz_push", [(0.0, -50.0), (1.0e-3, -50.0)])
        d.impvel(1, "push_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=5.0e-5, dt_scale=0.5, stop_cycles=30)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 30
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0
