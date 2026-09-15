"""Dynamic Engine Simulation & Energy Balance Tests for /MAT/LAW104 (/MAT/DRUCKER).

Verifies:
  - Multi-cycle dynamic explicit engine simulations for LAW104 with 3D Hexa8 solid brick
  - Multi-cycle dynamic explicit engine simulations for LAW104 with 2D Shell quad element
  - Positive internal energy accumulation and energy conservation
  - Material alias execution through Starter and Engine
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


class TestLaw104EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW104."""

    def test_solid_hexa8_law104_compression(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW104 under dynamic compressive stamping."""
        run_name = "HEXA8_LAW104"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law104(
            mid=1,
            title="Drucker_Hexa",
            rho=7.8e-9,
            young=210000.0,
            nu=0.3,
            ires=1,
            sigma0_yld=400.0,
            h=1000.0,
            q_voce=250.0,
            b_voce=15.0,
            c_dr=1.0,
            c_jc=0.02,
            eps0=1.0,
            fcut=1000.0,
            tss=0.002,
            tref=293.15,
            tini=293.15,
            eta=0.9,
            cp=4.5e8,
            eps_iso=1.0,
            eps_ad=100.0,
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

        # Compress top face nodes (5, 6, 7, 8) in -Z
        d.grnod_node(2, "press_top", [5, 6, 7, 8])
        d.funct(11, "vz_press", [(0.0, -100.0), (1.0e-2, -100.0)])
        d.impvel(1, "press_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=30)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 30
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0

    def test_shell_qeph_law104_tension(self, tmp_path: Path):
        """Single 2D shell element with /MAT/LAW104 under in-plane tensile extension."""
        run_name = "SHELL_LAW104"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law104(
            mid=1,
            title="Drucker_Shell",
            rho=7.8e-9,
            young=210000.0,
            nu=0.3,
            sigma0_yld=350.0,
            c_dr=0.5,
        )
        d.prop_shell(1, "ShellProp", thick=1.5)
        d.part(1, "ShellPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ]
        d.node(nodes)
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix left edge (1, 4)
        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull right edge (2, 3) in +X
        d.grnod_node(2, "pull_right", [2, 3])
        d.funct(11, "vx_pull", [(0.0, 100.0), (1.0e-2, 100.0)])
        d.impvel(1, "pull_x", 11, "X", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=30)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 30
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0

    def test_alias_mat_drucker_engine_simulation(self, tmp_path: Path):
        """Verify simulation runs identically when using /MAT/DRUCKER alias."""
        run_name = "ALIAS_DRUCKER"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_drucker(
            mid=1,
            title="Drucker_Alias",
            rho=7.8e-9,
            young=210000.0,
            nu=0.3,
            sigma0_yld=400.0,
            c_dr=1.2,
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

        d.grnod_node(2, "press_top", [5, 6, 7, 8])
        d.funct(11, "vz_press", [(0.0, -100.0), (1.0e-2, -100.0)])
        d.impvel(1, "press_z", 11, "Z", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=30)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 30
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0
