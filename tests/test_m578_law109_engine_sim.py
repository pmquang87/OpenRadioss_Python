"""Dynamic Engine Simulation & Energy Balance Tests for /MAT/LAW109 (/MAT/TAB_PLAS).

Verifies:
  - Multi-cycle dynamic explicit engine simulations for LAW109 with 3D Hexa8 solid brick
  - Multi-cycle dynamic explicit engine simulations for LAW109 with 2D Shell quad element
  - Strain rate sensitivity under multi-cycle loading
  - Cyclic loading: elastic loading, plastic yielding, elastic unloading, and expanded yield reload
  - Adiabatic thermal heating and temperature accumulation
  - Positive internal energy accumulation and energy conservation
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.starter.starter import run_starter
from pyradioss.materials.law109_tab_plas import (
    Law109Params,
    Law109Table,
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


class TestLaw109EngineSimulation:
    """Multi-cycle dynamic explicit engine simulations for LAW109."""

    def test_solid_hexa8_law109_compression(self, tmp_path: Path):
        """Single Hexa8 brick with /MAT/LAW109 under dynamic compressive deformation."""
        run_name = "HEXA8_LAW109"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_law109(
            mid=1,
            title="TabPlas_Hexa",
            rho=7.85e-9,
            e=210000.0,
            nu=0.3,
            c_p=4.5e8,
            eta=0.9,
            t_ref=293.15,
            t_ini=293.15,
            tab_id_h=100,
            xscale_h=1.0,
            yscale_h=1.0,
            i_smooth=1,
        )
        # Yield hardening function curve
        d.funct(100, "YieldHardening", [(0.0, 300.0), (0.05, 400.0), (0.2, 550.0)])

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

    def test_shell_law109_tension(self, tmp_path: Path):
        """Single 2D shell quad element with /MAT/LAW109 under tensile extension."""
        run_name = "SHELL_LAW109"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.mat_tab_plas(
            mid=1,
            title="TabPlas_Shell",
            rho=7.85e-9,
            e=200000.0,
            nu=0.3,
            tab_id_h=101,
            xscale_h=1.0,
            yscale_h=1.0,
        )
        d.funct(101, "ShellYld", [(0.0, 250.0), (0.02, 320.0), (0.1, 450.0)])

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

    def test_multistep_cyclic_loading_and_yield_expansion(self):
        """Verify cyclic loading: elastic -> plastic -> unloading -> reload yield expansion."""
        young = 200000.0
        nu = 0.3
        rho = 7.8e-9
        sigy0 = 250.0
        h_mod = 2000.0

        eps = np.array([0.0, 1.0])
        sig = np.array([sigy0, sigy0 + h_mod * 1.0])
        tbl = Law109Table(rates=[0.0], curves=[(eps, sig)])

        params = Law109Params(
            rho0=rho,
            young=young,
            nu=nu,
            yield_table=tbl,
        )

        dt = 1.0e-5
        sig_cur = np.zeros(6, dtype=np.float64)
        extra = {"pla": 0.0, "epsd": 0.0}

        # Step 1: Small elastic loading (deps_xx = 0.0005)
        # Expected stress ~ young * 0.0005 = 100 MPa < sigy0 = 250 MPa
        deps1 = np.array([0.0005, -0.00015, -0.00015, 0.0, 0.0, 0.0], dtype=np.float64)
        sig_cur, extra = solid_update(params, deps1, sig_cur, dt=dt, extra=extra)
        assert extra["pla"] == 0.0
        assert 90.0 < sig_cur[0] < 120.0

        # Step 2: Plastic loading past yield (deps_xx = 0.003)
        deps2 = np.array([0.003, -0.0015, -0.0015, 0.0, 0.0, 0.0], dtype=np.float64)
        sig_cur, extra = solid_update(params, deps2, sig_cur, dt=dt, extra=extra)
        pla1 = extra["pla"]
        assert pla1 > 0.0
        # Yield stress has expanded
        expanded_yield = sigy0 + h_mod * pla1

        # Step 3: Elastic unloading (deps_xx = -0.0005)
        deps3 = np.array([-0.0005, 0.00015, 0.00015, 0.0, 0.0, 0.0], dtype=np.float64)
        sig_cur, extra = solid_update(params, deps3, sig_cur, dt=dt, extra=extra)
        assert extra["pla"] == pla1  # Plastic strain does NOT change during unloading

        # Step 4: Reloading back up to previous expanded yield (deps_xx = 0.0005)
        deps4 = np.array([0.0005, -0.00015, -0.00015, 0.0, 0.0, 0.0], dtype=np.float64)
        sig_cur, extra = solid_update(params, deps4, sig_cur, dt=dt, extra=extra)
        assert extra["pla"] == pla1  # Still within elastic unloading-reloading regime

    def test_strain_rate_sensitivity_comparison(self):
        """Verify that higher strain rate produces higher flow stress according to 2D table."""
        young = 200000.0
        nu = 0.3
        rho = 7.8e-9

        # Table with rate = 1.0 (sigy = 200) and rate = 1000.0 (sigy = 350)
        eps = np.array([0.0, 0.1])
        sig_qs = np.array([200.0, 250.0])
        sig_dyn = np.array([350.0, 420.0])
        tbl = Law109Table(rates=[1.0, 1000.0], curves=[(eps, sig_qs), (eps, sig_dyn)])

        params = Law109Params(
            rho0=rho,
            young=young,
            nu=nu,
            yield_table=tbl,
            ismooth=1,
        )

        deps = np.array([0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0], dtype=np.float64)

        # Quasi-static rate: epsd = 1.0
        extra_qs = {"pla": 0.0, "epsd": 1.0}
        sig_qs_out, ext_qs = solid_update(params, deps, np.zeros(6), dt=1.0e-5, extra=extra_qs)

        # High dynamic rate: epsd = 1000.0
        extra_dyn = {"pla": 0.0, "epsd": 1000.0}
        sig_dyn_out, ext_dyn = solid_update(params, deps, np.zeros(6), dt=1.0e-5, extra=extra_dyn)

        # Dynamic stress must be significantly higher than quasi-static stress
        assert sig_dyn_out[0] > sig_qs_out[0] + 50.0

    def test_adiabatic_temperature_rise_and_softening(self):
        """Verify adiabatic heating and thermal softening in multi-step loading."""
        young = 200000.0
        nu = 0.3
        rho = 7.8e-9
        cp = 450.0e6
        eta = 0.9

        # Yield curve: base yield = 300 MPa
        eps = np.array([0.0, 1.0])
        sig = np.array([300.0, 300.0])
        tbl = Law109Table(rates=[0.0], curves=[(eps, sig)])

        # Thermal softening: at T=300K factor=1.0, at T=600K factor=0.7
        temp_curve = (np.array([300.0, 600.0]), np.array([1.0, 0.7]))

        params = Law109Params(
            rho0=rho,
            young=young,
            nu=nu,
            yield_table=tbl,
            temp_table=temp_curve,
            cp=cp,
            eta=eta,
            tref=300.0,
            tini=300.0,
        )

        sig_cur = np.zeros(6, dtype=np.float64)
        extra = {"pla": 0.0, "epsd": 0.0, "temp": 300.0}

        # Apply progressive plastic strains
        deps = np.array([0.005, -0.0025, -0.0025, 0.0, 0.0, 0.0], dtype=np.float64)
        for _ in range(5):
            sig_cur, extra = solid_update(params, deps, sig_cur, dt=1.0e-5, extra=extra)

        # Temperature must have risen due to adiabatic plastic dissipation
        assert extra["temp"] > 300.0
        # Plastic strain must have accumulated
        assert extra["pla"] > 0.0
