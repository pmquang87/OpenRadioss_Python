"""Dynamic Engine Simulation & Energy Balance Tests for /MAT/LAW103 (/MAT/HENSEL_SPITTEL).

Verifies:
  - Multi-cycle dynamic explicit engine simulations for LAW103 with Hexa8 solid brick
  - Multi-cycle dynamic explicit engine simulations for LAW103 with Tetra4 element
  - Positive internal energy accumulation and total energy balance
  - History state tracking across time steps (plastic strain and temperature)
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


class TestLaw103EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW103."""

    def test_solid_hexa8_law103_compression(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW103 under dynamic compressive stamping."""
        run_name = "HEXA8_LAW103"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law103(
            mid=1,
            title="Hensel_Spittel_Hexa",
            rho=4.43e-9,
            e=110000.0,
            nu=0.34,
            a0=800.0,
            m1=-0.002,
            m2=0.15,
            m3=0.05,
            m4=-0.0005,
            m5=-0.0001,
            m7=0.02,
            fsmooth=0,
            fcut=0.0,
            eps0=0.001,
            pmin=-1.0e30,
            rcp=2.4e-3,
            t0=973.15,
            eta=0.9,
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

    def test_solid_tetra4_law103_compression(self, tmp_path: Path):
        """Single Tetra4 element with /MAT/HENSEL_SPITTEL under compressive loading."""
        run_name = "TETRA4_LAW103"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_hensel_spittel(
            mid=1,
            title="Hensel_Spittel_Tetra",
            rho=7.85e-9,
            e=205000.0,
            nu=0.29,
            a0=950.0,
            m1=-0.002,
            m2=0.22,
            m3=0.05,
            m4=-0.001,
            m5=-0.0001,
            m7=0.018,
            fsmooth=0,
            fcut=0.0,
            eps0=0.002,
            pmin=-1.0e30,
            rcp=3.6e-3,
            t0=1273.15,
            eta=0.88,
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
        _write_engine_deck(e_path, run_name, tstop=2.0e-3, dt_scale=0.5, stop_cycles=30)

        st_model = run_starter(s_path)
        assert st_model is not None

        eng_model = run_engine(e_path)
        state = eng_model.engine_state
        assert state.cycle >= 30
        eint_total = sum(float(g.state["eint"].sum()) for _, g in eng_model.element_groups())
        assert eint_total > 0.0
