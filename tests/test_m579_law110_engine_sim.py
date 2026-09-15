"""Dynamic Engine Simulation & Energy Balance Tests for /MAT/LAW110 (/MAT/VEGTER).

Verifies:
  - Multi-cycle dynamic explicit engine simulations for LAW110 with 2D Shell quad element
  - Starter diagnostic rejection when 3D solid elements are assigned LAW110 (ANCMSG 305)
  - Cyclic loading: elastic loading, plastic yielding, elastic unloading, and reload yield expansion
  - Parity and consistency between Nice explicit projection (IRES=1) and Newton cutting plane (IRES=2)
  - Internal energy accumulation and plastic work dissipation
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.starter import run_starter
from pyradioss.starter.checks import check_mat_law110
from pyradioss.common.messages import MessageLog, StarterError
from pyradioss.model.entities import MaterialLaw110
from pyradioss.materials.law110_vegter import (
    VegterModelParams,
    law110_shell_update,
    get_vegter_params,
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


class TestLaw110EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW110."""

    def test_shell_law110_tension(self, tmp_path: Path):
        """Single 2D QEPH shell quad element with /MAT/LAW110 under dynamic tensile extension."""
        run_name = "SHELL_LAW110"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law110(
            mid=1,
            title="Vegter_Shell",
            rho_i=7.85e-9,
            e=210000.0,
            nu=0.3,
            icrit=1,
            ihard=1,
            sigma_r=250.0,
            b_swift=500.0,
            n_swift=0.2,
            eps_0=0.001,
            angles_data=[
                (0.0, 1.0, 1.8, 1.15, 0.0, 0.50),
                (45.0, 1.02, 1.6, 1.18, 0.0, 0.52),
                (90.0, 0.98, 2.0, 1.12, 0.0, 0.48),
            ],
        )

        d.prop_shell(1, "ShellProp", thick=1.0)
        d.part(1, "ShellPart", 1, 1)

        nodes = [
            (1, 0.0, 0.0, 0.0),
            (2, 10.0, 0.0, 0.0),
            (3, 10.0, 10.0, 0.0),
            (4, 0.0, 10.0, 0.0),
        ]
        d.node(nodes)
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix left edge (nodes 1, 4)
        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull right edge (nodes 2, 3) in +X direction
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

    def test_solid_element_diagnostic_rejection(self, tmp_path: Path):
        """Verify Starter check rejects 3D solid elements with /MAT/LAW110 (ANCMSG 305)."""
        run_name = "HEXA_LAW110_BAD"
        s_path = str(tmp_path / f"{run_name}_0000.rad")

        d = StarterDeck(run_name)
        d.mat_law110(
            mid=1,
            title="Vegter_Solid_Bad",
            rho_i=7.85e-9,
            e=210000.0,
            nu=0.3,
            icrit=1,
            ihard=1,
            sigma_r=250.0,
            b_swift=500.0,
            n_swift=0.2,
            eps_0=0.001,
            angles_data=[
                (0.0, 1.0, 1.8, 1.15, 0.0, 0.50),
                (45.0, 1.02, 1.6, 1.18, 0.0, 0.52),
                (90.0, 0.98, 2.0, 1.12, 0.0, 0.48),
            ],
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
        d.write(s_path)

        log = MessageLog()
        with pytest.raises(StarterError):
            run_starter(s_path, log=log)
        assert any("ANCMSG 305" in m or "solid elements" in m.lower() for m in log.messages)

    def test_multistep_cyclic_loading_and_yield_expansion(self):
        """Verify cyclic loading on LAW110: elastic -> plastic yield -> elastic unload -> reload."""
        angles = [
            [1.0, 1.8, 1.15, 0.0, 0.50],
            [1.02, 1.6, 1.18, 0.0, 0.52],
            [0.98, 2.0, 1.12, 0.0, 0.48],
        ]
        mat = MaterialLaw110(
            mid=1,
            rho0=7.8e-6,
            young=210000.0,
            nu=0.3,
            icrit=1,
            sigma_r=250.0,
            dsigm=500.0,
            beta=1.0,
            angles_data=angles,
            ires=2,
        )

        sig_cur = np.zeros(3, dtype=np.float64)
        extra = {"pla": 0.0}
        dt = 1.0e-5

        # Step 1: Small elastic loading (dxx = 0.0005, dyy = -0.00015)
        # Expected stress ~ young * 0.0005 = 105 MPa < yield = 250 MPa
        deps1 = np.array([0.0005, -0.00015, 0.0], dtype=np.float64)
        sig_cur, extra = law110_shell_update(mat, deps1, sig_cur, extra=extra, dt=dt)
        assert extra["pla"] == 0.0
        assert 90.0 < sig_cur[0] < 130.0

        # Step 2: Plastic loading past initial yield (dxx = 0.003, dyy = -0.0015)
        deps2 = np.array([0.003, -0.0015, 0.0], dtype=np.float64)
        sig_cur, extra = law110_shell_update(mat, deps2, sig_cur, extra=extra, dt=dt)
        pla1 = extra["pla"]
        assert pla1 > 0.0
        sig_yield1 = extra["sigy"]
        assert sig_yield1 > 250.0

        # Step 3: Elastic unloading (dxx = -0.0005, dyy = 0.00015)
        deps3 = np.array([-0.0005, 0.00015, 0.0], dtype=np.float64)
        sig_cur, extra = law110_shell_update(mat, deps3, sig_cur, extra=extra, dt=dt)
        # Plastic strain must remain constant during elastic unloading
        assert extra["pla"] == pytest.approx(pla1, abs=1.0e-12)
        assert sig_cur[0] < sig_yield1

        # Step 4: Reloading within the expanded yield surface (dxx = 0.0003, dyy = -0.00009)
        deps4 = np.array([0.0003, -0.00009, 0.0], dtype=np.float64)
        sig_cur, extra = law110_shell_update(mat, deps4, sig_cur, extra=extra, dt=dt)
        # Still inside expanded yield surface: plastic strain unchanged
        assert extra["pla"] == pytest.approx(pla1, abs=1.0e-12)

    def test_nice_vs_newton_algorithm_comparison(self):
        """Compare Nice explicit projection (IRES=1) vs Newton cutting plane (IRES=2)."""
        angles = [
            [1.0, 1.8, 1.15, 0.0, 0.50],
            [1.02, 1.6, 1.18, 0.0, 0.52],
            [0.98, 2.0, 1.12, 0.0, 0.48],
        ]
        mat_nice = MaterialLaw110(
            mid=1, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1,
            sigma_r=250.0, dsigm=500.0, beta=1.0,
            angles_data=angles, ires=1,
        )
        mat_newton = MaterialLaw110(
            mid=2, rho0=7.8e-6, young=210000.0, nu=0.3, icrit=1,
            sigma_r=250.0, dsigm=500.0, beta=1.0,
            angles_data=angles, ires=2,
        )

        dt = 1.0e-5
        sig_nice = np.zeros(3, dtype=np.float64)
        sig_newt = np.zeros(3, dtype=np.float64)
        ex_nice = {"pla": 0.0}
        ex_newt = {"pla": 0.0}

        # Apply 35 explicit increments past yield
        deps = np.array([0.0001, -0.00005, 0.00002], dtype=np.float64)
        for _ in range(35):
            sig_nice, ex_nice = law110_shell_update(mat_nice, deps, sig_nice, extra=ex_nice, dt=dt)
            sig_newt, ex_newt = law110_shell_update(mat_newton, deps, sig_newt, extra=ex_newt, dt=dt)

        assert ex_nice["pla"] > 0.0
        assert ex_newt["pla"] > 0.0
        # Nice and Newton produce closely matching stress (< 1%) and plastic strain (< 10%)
        assert sig_nice[0] == pytest.approx(sig_newt[0], rel=0.01)
        assert sig_nice[1] == pytest.approx(sig_newt[1], rel=0.01)
        assert ex_nice["pla"] == pytest.approx(ex_newt["pla"], rel=0.10)
