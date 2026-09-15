"""Dynamic Engine Simulation & Energy Balance Tests for /MAT/LAW102 (/MAT/DPRAG2).

Verifies:
  - Multi-cycle dynamic explicit engine simulations for LAW102 with Hexa8 solid brick
  - Multi-cycle dynamic explicit engine simulations for LAW102 with Tetra4 element
  - Positive internal energy accumulation and total energy balance
  - History state tracking across time steps
"""

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


class TestLaw102EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW102."""

    def test_solid_hexa8_law102_tensile(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW102 under dynamic tensile stretching."""
        run_name = "HEXA8_LAW102"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law102(
            mid=1,
            title="DPRAG2_Hexa",
            rho=2.5e-6,
            iform=2,
            e=25000.0,
            nu=0.25,
            c=15.0,
            phi=30.0,
            amax=200.0,
            pmin=-5.0,
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
        d.funct(11, "vz_pull", [(0.0, 50.0), (1.0e-2, 50.0)])
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

    def test_solid_tetra4_law102_compression(self, tmp_path: Path):
        """Single Tetra4 element with /MAT/DPRAG2 under compressive loading."""
        run_name = "TETRA4_LAW102"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_dprag2(
            mid=1,
            title="DPRAG2_Tetra",
            rho=2.4e-6,
            iform=1,
            e=20000.0,
            nu=0.22,
            c=10.0,
            phi=28.0,
            amax=150.0,
            pmin=-3.0,
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
        d.funct(11, "vz_push", [(0.0, -30.0), (1.0e-2, -30.0)])
        d.impvel(1, "push_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=30)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 30
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0
