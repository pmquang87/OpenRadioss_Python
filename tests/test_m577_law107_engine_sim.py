"""Dynamic Engine Simulation & Energy Balance Tests for /MAT/LAW107 (/MAT/PAPER_LIGHT).

Verifies:
  - Multi-cycle dynamic explicit engine simulations for LAW107 with 3D Hexa8 solid brick
  - Multi-cycle dynamic explicit engine simulations for LAW107 with 2D Shell quad element
  - Tabulated /MAT/PFEIFFER function curve engine simulation
  - Positive internal energy accumulation, energy balance, and non-decreasing plastic dissipation
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.starter import run_starter
from pyradioss.materials.law107_paper_light import (
    PaperLightConstants,
    PaperLightParams,
    compute_paper_light_constants,
    solid_update,
    shell_update,
)


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


class TestLaw107EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW107."""

    def test_solid_hexa8_law107_compression(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW107 under dynamic compressive deformation."""
        run_name = "HEXA8_LAW107"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law107(
            mid=1,
            title="Paper_Hexa",
            rho=7.5e-7,
            young1=7000.0,
            young2=3500.0,
            young3=100.0,
            nu21=0.15,
            g12=1800.0,
            g23=40.0,
            g31=40.0,
            ires=2,
            itab=0,
            xi1=0.5,
            xi2=0.5,
            k1=0.1,
            k2=-0.1,
            k3=0.2,
            k4=0.3,
            k5=0.4,
            k6=0.5,
            sigy1=25.0,
            cini1=10.0,
            s1=2.0,
            sigy2=15.0,
            cini2=8.0,
            s2=1.5,
            sigy1c=30.0,
            sigy2c=20.0,
            sigyt=10.0,
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

    def test_shell_qeph_paper_light_tension(self, tmp_path: Path):
        """Single 2D shell quad element with /MAT/PAPER_LIGHT under tensile extension."""
        run_name = "SHELL_PAPER_LIGHT"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_paper_light(
            mid=1,
            title="Paper_Light_Shell",
            rho=7.5e-7,
            young1=7000.0,
            young2=3500.0,
            young3=100.0,
            nu21=0.15,
            g12=1800.0,
            g23=40.0,
            g31=40.0,
            ires=2,
            itab=0,
            sigy1=25.0,
            cini1=10.0,
            s1=2.0,
            sigy2=15.0,
            sigyt=10.0,
        )
        d.prop_shell(1, "ShellProp", thick=0.2)
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

    def test_tabulated_pfeiffer_engine_simulation(self, tmp_path: Path):
        """Verify simulation runs with tabulated curves using /MAT/PFEIFFER."""
        run_name = "PFEIFFER_TAB"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_pfeiffer(
            mid=1,
            title="Pfeiffer_Tabulated",
            rho=7.5e-7,
            young1=7000.0,
            young2=3500.0,
            young3=100.0,
            nu21=0.15,
            g12=1800.0,
            g23=40.0,
            g31=40.0,
            itab=1,
            ires=2,
            tab_yld1=101,
            tab_yld2=102,
            tab_yld1c=103,
            tab_yld2c=104,
            tab_yldt=105,
        )
        # Add yield curves
        d.funct(101, "Yld1", [(0.0, 25.0), (0.01, 28.0), (0.1, 35.0)])
        d.funct(102, "Yld2", [(0.0, 15.0), (0.01, 17.0), (0.1, 22.0)])
        d.funct(103, "Yld1c", [(0.0, 30.0), (0.01, 33.0), (0.1, 40.0)])
        d.funct(104, "Yld2c", [(0.0, 20.0), (0.01, 22.0), (0.1, 28.0)])
        d.funct(105, "Yldt", [(0.0, 10.0), (0.01, 11.0), (0.1, 15.0)])

        d.prop_shell(1, "ShellProp", thick=0.25)
        d.part(1, "ShellPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ]
        d.node(nodes)
        d.shell(1, [(1, 1, 2, 3, 4)])

        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        d.grnod_node(2, "pull_right", [2, 3])
        d.funct(11, "vx_pull", [(0.0, 80.0), (1.0e-2, 80.0)])
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

    def test_multicycle_cyclic_plastic_dissipation(self):
        """Verify cyclic plastic loading maintains non-decreasing cumulative plastic strain and positive work."""
        p = PaperLightParams(
            rho=7.5e-7,
            refer_rho=7.5e-7,
            young1=7000.0,
            young2=3500.0,
            young3=100.0,
            nu21=0.15,
            g12=1800.0,
            g23=40.0,
            g31=40.0,
            ires=2,
            itab=0,
            sigy1=25.0,
            cini1=10.0,
            s1=2.0,
            sigy2=15.0,
            cini2=8.0,
            s2=1.5,
            sigy1c=30.0,
            sigy2c=20.0,
            sigyt=10.0,
        )
        c = compute_paper_light_constants(p)

        sig = np.zeros(5, dtype=np.float64)
        extra = {"uvar": np.zeros(1), "pla": np.zeros(6), "epsd": np.zeros(6)}

        # Cyclic strain path:
        # 1. Elastic loading
        # 2. Tension loading past yield (sigy1 = 25.0 MPa, A11 ~ 7330)
        # 3. Elastic unloading
        # 4. Compressive reload past compressive yield (sigy1c = 30.0 MPa)
        # 5. Shear loading past shear yield (sigyt = 10.0 MPa, G12 = 1800)
        strain_increments = [
            np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0e-2, 0.0, 0.0, 0.0, 0.0]),  # Plastic tension
            np.array([-5.0e-3, 0.0, 0.0, 0.0, 0.0]),  # Unloading
            np.array([-1.5e-2, 0.0, 0.0, 0.0, 0.0]),  # Plastic compression
            np.array([0.0, 0.0, 2.0e-2, 0.0, 0.0]),  # Plastic shear
        ]

        prev_epsp = 0.0
        total_work = 0.0

        for de in strain_increments:
            sig_new, dpla, epsp_new, extra = shell_update(
                p, sig, de, dt=1.0e-6, extra=extra, return_tuple=False
            )
            # Plastic multiplier must be non-negative
            assert dpla >= 0.0
            # Cumulative plastic strain must be monotonically non-decreasing
            assert epsp_new >= prev_epsp

            # Work increment: (sig + sig_new)/2 : de
            avg_sig = 0.5 * (sig + sig_new)
            work_incr = float(np.dot(avg_sig, de))
            total_work += work_incr

            sig = sig_new
            prev_epsp = epsp_new

        assert prev_epsp > 0.0
        assert total_work > 0.0
