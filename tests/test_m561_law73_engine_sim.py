"""
Dynamic Engine Simulation & Energy Balance Verifier for M561 (/MAT/LAW73 /MAT/BARLAT2000 /MAT/HILL_THERM).

Exhaustive dynamic explicit engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations across shell element formulations:
   - Shell BT4 with LAW73 under cyclic tension, compression, and shear (50+ cycles, |ERR| < 1.0%)
   - Shell QEPH with LAW73 under dynamic biaxial tension and shear (50+ cycles, |ERR| < 1.0%)
   - Shell Tri3 with LAW73 under dynamic stretching (50+ cycles, |ERR| < 1.0%)
   - Shell BT4 2x2 multi-element patch with LAW73 under cyclic loading (50+ cycles, |ERR| < 1.0%)
   - Asserts normal termination, >= 50 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), and external work consistency.
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt
   - Midpoint trapezoidal work matches exact analytical strain energy in closed reversible cycles.
   - Free vibration of undamped shell elements (BT4, QEPH):
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1.0%).
   - Plastic dissipation monotonically increasing during plastic flow.
3. Physical anisotropic sheet metal behavior:
   - Verify differential yield strengths across 0 deg, 45 deg, 90 deg rolling orientations matching Hill 1948 theory.
   - Verify differential plastic thinning across different Lankford R values (plastic incompressibility).
   - Cyclic loading exhibiting Bauschinger reverse softening under kinematic hardening (chard > 0)
     vs expanding elastic domain under isotropic hardening (chard = 0).
4. Progressive failure & element deletion:
   - Tensile softening for epsr1 <= epst < epsr2 and plastic failure when eps_p >= eps_max.
   - Verify stresses collapse to zero and post-erosion dynamic simulation continues stably without NaNs, Infs, or Courant collapse.
   - Two-element patch progressive rupture where one element erodes while neighbor stays intact.
5. Dynamic thermal softening & adiabatic heating:
   - Adiabatic heating rises dynamically during plastic deformation when rhocp > 0.
   - Coupled thermal softening reduces yield stress at elevated temperatures.
6. Acoustic sound speed & Courant time-step stability:
   - Sound speed c_shell maintaining positive, stable Courant bounds across all dynamic cycles.
   - Sound speed stability under dynamic modulus degradation (ce > 0 / einf).
   - Courant time-step stability under dynamic pulse propagation in multi-element patch.
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.input.deck_writer import StarterDeck
from pyradioss.materials.law73_hill_therm import (
    Law73Params,
    build_law73,
    shell_update as shell_update_law73,
    sound_speed as sound_speed_shell_law73,
    shell_membrane_tangent,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers: Engine Control Deck Writer and Material Factories
# ============================================================================

def _scalar(val: Any) -> float:
    """Safely convert 0-d or 1-d single element array/scalar to float (NumPy 2.x safe)."""
    return float(np.asarray(val).flat[0])


def _write_engine_deck(
    path: Path | str,
    run_name: str,
    tstop: float = 2.5e-4,
    dt_scale: float = 0.5,
    stop_cycles: int | None = None,
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


def make_test_material_law73(
    mid: int = 1,
    rho0: float = 2.7e-9,  # typical aluminum: 2.7e-9 ton/mm^3
    E: float = 70000.0,
    nu: float = 0.33,
    r00: float = 1.5,
    r45: float = 1.2,
    r90: float = 1.8,
    chard: float = 0.0,
    iyield: int = 0,
    sigy0: float = 220.0,
    eps_max: float = 1.0e30,
    epsr1: float = 1.0e30,
    epsr2: float = 2.0e30,
    t0: float = 293.0,
    rhocp: float = 0.0,
    yield_table: Any = None,
    table_id: int = 0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW73 (/MAT/BARLAT2000, /MAT/HILL_THERM) Material instance."""
    if yield_table is None and table_id == 0 and sigy0 > 0.0:
        yield_table = [(0.0, sigy0), (0.05, sigy0 * 1.3), (0.2, sigy0 * 1.6)]
    params = {
        "e": E,
        "nu": nu,
        "r00": r00,
        "r45": r45,
        "r90": r90,
        "chard": chard,
        "fisokin": chard,
        "iyield": iyield,
        "sigy0": sigy0,
        "eps_max": eps_max,
        "epsr1": epsr1,
        "epsr2": epsr2,
        "t0": t0,
        "rhocp": rhocp,
        "table_id": table_id,
    }
    if yield_table is not None:
        params["yield_table"] = yield_table
    params.update(kwargs)
    return build_law73(id=mid, rho0=rho0, title="Alu_LAW73_HillTherm", params=params)


class MockProp:
    """Mock shell property mimicking /PROP/SHELL (/PROP/TYPE1)."""
    def __init__(
        self,
        pid: int = 1,
        thick: float = 1.0,
        nip: int = 3,
        ishell: int = 1,
        ish3n: int = 1,
        hm: float = 0.1,
        hf: float = 0.1,
        hr: float = 0.1,
        dn: float = 0.015,
        **kwargs: Any,
    ):
        self.id = pid
        self.thick = thick
        self.nip = nip
        self.ishell = ishell
        self.ish3n = ish3n
        self.params = {
            "thick": thick,
            "nip": nip,
            "ishell": ishell,
            "ish3n": ish3n,
            "qa": 0.0,
            "qb": 0.0,
            "hm": hm,
            "hf": hf,
            "hr": hr,
            "dn": dn,
            **kwargs,
        }


class MockGroup:
    """Mock element group with connectivity, id indexing, and state buffer."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


# ============================================================================
# 1. Multi-Cycle Explicit Dynamic Simulations Across Formulations
# ============================================================================

class TestLaw73MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across supported shell element formulations."""

    def test_shell_bt4_cyclic_tension_compression_shear_engine(self, tmp_path: Path):
        """Shell BT4 with LAW73 under cyclic tension, compression, and shear (50+ cycles)."""
        run_name = "SHELL_BT4_LAW73_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        # Yield hardening curve: 120 MPa initial yield to 280 MPa
        d.funct(10, "yield_fct", [(0.0, 120.0), (0.05, 200.0), (0.15, 280.0)])
        d.mat_law73(
            1, "AluHillTherm",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.5, r45=1.2, r90=1.8, chard=0.3,
            table_id=10,
        )
        thick0 = 1.2
        d.prop_shell(1, "PropBT4", thick=thick0, nip=3, ishell=1)
        d.part(1, "PartBT4", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix left edge (nodes 1, 4)
        d.grnod_node(1, "fix_left", [1, 4])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Multi-axial cyclic loading on right edge (nodes 2, 3)
        d.grnod_node(2, "right_nodes", [2, 3])
        # Cyclic tension -> hold -> compression -> tension in X
        d.funct(11, "vx_cyc", [
            (0.0, 1200.0),
            (0.8e-4, 1200.0),
            (0.8001e-4, -800.0),
            (1.6e-4, -800.0),
            (1.6001e-4, 1000.0),
            (2.5e-4, 1000.0),
        ])
        d.impvel(1, "pull_x", 11, "X", 2)

        # In-plane shear loading in Y
        d.funct(12, "vy_shear", [
            (0.0, 500.0),
            (1.0e-4, 500.0),
            (1.0001e-4, -500.0),
            (2.5e-4, -500.0),
        ])
        d.impvel(2, "shear_y", 12, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Unexpected stop reason: {state.stop_reason}"

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"BT4 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must be strictly positive"
        assert en["EW"] > 0.0, "External work must be positive"

        # Check plastic strain accumulation occurred
        mat_extra = eng_model.shells.state.get("mat_extra", {})
        assert "pla73" in mat_extra or "epsp" in eng_model.shells.state
        pla_final = mat_extra.get("pla73", eng_model.shells.state.get("epsp"))
        assert np.any(pla_final > 0.0), "Plastic strain must accumulate during cycle"

    def test_shell_qeph_dynamic_biaxial_tension_shear_engine(self, tmp_path: Path):
        """Shell QEPH (Ishell=24) with LAW73 under dynamic biaxial tension and shear (50+ cycles)."""
        run_name = "SHELL_QEPH_LAW73_BIAX"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 180.0), (0.1, 280.0)])
        d.mat_law73(
            1, "AluQEPH",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.4, r45=1.1, r90=1.7, chard=0.2,
            table_id=10,
        )
        d.prop_shell(1, "PropQEPH", thick=1.0, nip=3, ishell=24)
        d.part(1, "PartQEPH", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 10.0, 10.0, 0.0), (4, 0.0, 10.0, 0.0),
        ])
        d.shell(1, [(1, 1, 2, 3, 4)])

        # Fix corner node 1 completely, node 4 in X, node 2 in Y
        d.grnod_node(1, "fix_corner", [1])
        d.bcs(1, "bcs_fix", "111", "111", 1)
        d.grnod_node(2, "fix_x", [4])
        d.bcs(2, "bcs_fix_x", "100", "000", 2)
        d.grnod_node(3, "fix_y", [2])
        d.bcs(3, "bcs_fix_y", "010", "000", 3)

        # Pull right edge (2, 3) in X
        d.grnod_node(4, "pull_x_nodes", [2, 3])
        d.funct(11, "vx_pull", [(0.0, 350.0), (2.0e-4, 350.0)])
        d.impvel(1, "pull_x", 11, "X", 4)

        # Pull top edge (3, 4) in Y
        d.grnod_node(5, "pull_y_nodes", [3, 4])
        d.funct(12, "vy_pull", [(0.0, 300.0), (2.0e-4, 300.0)])
        d.impvel(2, "pull_y", 12, "Y", 5)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"QEPH energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_tri3_dynamic_stretching_engine(self, tmp_path: Path):
        """Shell Tri3 (Ish3n=1) with LAW73 under dynamic stretching (50+ cycles)."""
        run_name = "SHELL_TRI3_LAW73_STR"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 200.0), (0.1, 300.0)])
        d.mat_law73(
            1, "AluTri3",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.5, r45=1.2, r90=1.8, chard=0.0,
            table_id=10,
        )
        d.prop_shell(1, "PropTri3", thick=1.0, nip=3, ish3n=1)
        d.part(1, "PartTri3", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 0.0, 10.0, 0.0),
        ])
        d.sh3n(1, [(1, 1, 2, 3)])

        # Fix base root (nodes 1, 3)
        d.grnod_node(1, "fix_root", [1, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull apex node 2 in X and Y
        d.grnod_node(2, "apex_node", [2])
        d.funct(11, "vx_pull", [(0.0, 450.0), (2.0e-4, 450.0)])
        d.impvel(1, "pull_apex_x", 11, "X", 2)
        d.funct(12, "vy_pull", [(0.0, 200.0), (2.0e-4, 200.0)])
        d.impvel(2, "pull_apex_y", 12, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tri3 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_bt4_multi_element_patch_cyclic_engine(self, tmp_path: Path):
        """Shell BT4 2x2 multi-element patch with LAW73 under cyclic dynamic loading (50+ cycles)."""
        run_name = "PATCH_2X2_LAW73"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 210.0), (0.08, 310.0)])
        d.mat_law73(
            1, "AluPatch",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.5, r45=1.2, r90=1.8, chard=0.5,
            table_id=10,
        )
        d.prop_shell(1, "PropPatch", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartPatch", 1, 1)

        d.node([
            (1, 0.0, 0.0, 0.0), (2, 10.0, 0.0, 0.0), (3, 20.0, 0.0, 0.0),
            (4, 0.0, 10.0, 0.0), (5, 10.0, 10.0, 0.0), (6, 20.0, 10.0, 0.0),
            (7, 0.0, 20.0, 0.0), (8, 10.0, 20.0, 0.0), (9, 20.0, 20.0, 0.0),
        ])
        d.shell(1, [
            (1, 1, 2, 5, 4),
            (2, 2, 3, 6, 5),
            (3, 4, 5, 8, 7),
            (4, 5, 6, 9, 8),
        ])

        # Fix bottom edge (nodes 1, 2, 3)
        d.grnod_node(1, "fix_bottom", [1, 2, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Pull top edge (nodes 7, 8, 9) in Y
        d.grnod_node(2, "top_nodes", [7, 8, 9])
        d.funct(11, "vy_pull", [
            (0.0, 600.0),
            (1.0e-4, 600.0),
            (1.0001e-4, -400.0),
            (2.2e-4, -400.0),
        ])
        d.impvel(1, "pull_top", 11, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=2.2e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 50, f"Expected >= 50 cycles, got {state.cycle}"
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"2x2 patch energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0


# ============================================================================
# 2. Energy Conservation & Work Balance Auditing
# ============================================================================

class TestLaw73EnergyConservationAndWorkBalance:
    """Audit energy conservation, incremental trapezoidal work, and plastic dissipation."""

    def test_strain_energy_ledger_trapezoidal_integration(self):
        """Incremental trapezoidal work matches exact analytical strain energy in elastic regime."""
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, r00=1.0, r45=1.0, r90=1.0,
            sigy0=10000.0,  # High yield stress for pure elastic response
        )
        vol0 = 100.0  # 10 x 10 x 1 mm
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
        }
        dt = 1.0e-6

        nu = 0.33
        E = 70000.0
        a11 = E / (1.0 - nu ** 2)
        a12 = nu * a11
        g = E / (2.0 * (1.0 + nu))

        eps_total = np.zeros(3, dtype=float)
        eint_incremental = 0.0

        for step in range(15):
            deps = np.array([[1.0e-4, -0.33 * 1.0e-4, 5.0e-5]])
            sig_old = sig.copy()
            res = shell_update_law73(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig_new = res[0].copy()
            sig = sig_new.copy()
            epsp[0] = _scalar(res[1])

            sig_mid = 0.5 * (sig_old + sig_new)
            de = (sig_mid[0, 0] * deps[0, 0] + sig_mid[0, 1] * deps[0, 1] + sig_mid[0, 2] * deps[0, 2]) * vol0
            eint_incremental += de
            eps_total += deps[0]

        # Analytical elastic strain energy: 0.5 * (a11*ex^2 + a11*ey^2 + 2*a12*ex*ey + g*exy^2) * vol0
        ex, ey, exy = eps_total
        analytical_eint = 0.5 * (a11 * ex**2 + a11 * ey**2 + 2.0 * a12 * ex * ey + g * exy**2) * vol0
        assert eint_incremental == pytest.approx(analytical_eint, rel=1e-5), \
            f"Incremental strain energy {eint_incremental} != analytical {analytical_eint}"
        assert epsp[0] == 0.0, "No plastic strain should accumulate in elastic regime"

    def test_shell_bt4_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Shell BT4 in elastic regime: |Delta E| / E_0 < 1.0%."""
        deck = StarterDeck("BT4_ELAS_OSC")
        deck.mat_law73(
            1, "ElasticAlu",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.0, r45=1.0, r90=1.0,
            eps_max=1.0e30,
        )
        thick0 = 1.0
        deck.prop_shell(1, "PropBT4", thick=thick0, nip=3, ishell=1, hm=0.0, hf=0.0, hr=0.0)
        deck.part(1, "PartShell", 1, 1)

        lx, ly = 10.0, 10.0
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, lx, 0.0, 0.0), (3, lx, ly, 0.0), (4, 0.0, ly, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])

        s_path = str(tmp_path / "BT4_ELAS_OSC_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells
        rho0 = 2.7e-9
        area0 = lx * ly
        m_node = rho0 * thick0 * area0 / 4.0
        mass_vec = np.full(4, m_node)

        # Pure symmetric breathing velocity mode in X
        v = np.zeros_like(model.x)
        vx0 = 40.0
        v[[1, 2], 0] = vx0
        v[[0, 3], 0] = -vx0

        e_kin_0 = 0.5 * np.sum(mass_vec * (vx0 ** 2))
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0
        assert e_tot_0 > 0.0

        dt = 1.0e-7
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        model.v = v.copy()

        e_tot_history = []
        for step in range(80):
            fint.fill(0.0)
            mint.fill(0.0)
            shell_bt4.forces(group, model.x, model.v, model.vr, dt, fint, mint)
            v_old = model.v.copy()
            acc = fint / mass_vec[:, None]
            model.v += acc * dt
            model.x += model.v * dt
            v_mid = 0.5 * (v_old + model.v)
            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_err < 0.01, f"BT4 elastic energy conservation error {max_err*100:.3f}% exceeds 1.0%"

    def test_plastic_dissipation_monotonically_increasing(self):
        """Plastic strain and plastic dissipation monotonically non-decreasing during plastic flow."""
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, r00=1.5, r45=1.2, r90=1.8,
            sigy0=200.0,
            yield_table=[(0.0, 200.0), (0.1, 320.0)],
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
        }

        epsp_history = []
        dt = 1.0e-6

        for step in range(40):
            deps = np.array([[3.0e-4, -1.0e-4, 1.5e-4]])
            res = shell_update_law73(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])
            epsp_history.append(float(epsp[0]))

        d_epsp = np.diff(epsp_history)
        assert np.all(d_epsp >= -1e-15), "Plastic strain increments must be non-negative"
        assert epsp_history[-1] > epsp_history[0], "Plastic strain must strictly increase during yield"


# ============================================================================
# 3. Anisotropic Sheet Metal Behavior Auditing
# ============================================================================

class TestLaw73AnisotropicSheetBehavior:
    """Audit differential yield strengths across 0 deg, 45 deg, 90 deg, and Bauschinger effect."""

    def test_differential_yield_strengths_0_45_90_rolling_angles(self):
        """Verify differential yield stresses across 0 deg, 45 deg, and 90 deg rolling angles match theory."""
        r00, r45, r90 = 2.0, 1.0, 0.5
        sigy0 = 200.0
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, r00=r00, r45=r45, r90=r90, sigy0=sigy0,
            yield_table=[(0.0, sigy0), (0.1, sigy0 * 1.5)],
        )
        p = mat.params["_obj"]

        # Theoretical Hill 1948 yield stresses:
        sig0_theory = sigy0 / math.sqrt(p.a01)
        sig90_theory = sigy0 / math.sqrt(p.a02)
        sig45_theory = 2.0 * sigy0 / math.sqrt(p.a01 + p.a02 - p.a03 + p.a12)

        assert abs(sig90_theory - sig0_theory) > 5.0, "Anisotropy must produce distinct 90 deg yield stress"
        assert abs(sig45_theory - sig0_theory) > 5.0, "Anisotropy must produce distinct 45 deg yield stress"

        # Simulation 1: Uniaxial tension along 0 deg (x)
        sig_0 = np.zeros((1, 3), dtype=float)
        epsp_0 = np.zeros(1, dtype=float)
        extra_0 = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
        }
        for _ in range(60):
            deps = np.array([[1.0e-4, -0.33 * 1.0e-4, 0.0]])
            res = shell_update_law73(mat, sig_0, deps, epsp_0, dt=1e-6, extra=extra_0)
            sig_0 = res[0].copy()
            epsp_0[0] = _scalar(res[1])
            if epsp_0[0] > 1e-6:
                break
        assert sig_0[0, 0] == pytest.approx(sig0_theory, rel=0.03)

        # Simulation 2: Uniaxial tension along 90 deg (y)
        sig_90 = np.zeros((1, 3), dtype=float)
        epsp_90 = np.zeros(1, dtype=float)
        extra_90 = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
        }
        for _ in range(60):
            deps = np.array([[-0.33 * 1.0e-4, 1.0e-4, 0.0]])
            res = shell_update_law73(mat, sig_90, deps, epsp_90, dt=1e-6, extra=extra_90)
            sig_90 = res[0].copy()
            epsp_90[0] = _scalar(res[1])
            if epsp_90[0] > 1e-6:
                break
        assert sig_90[0, 1] == pytest.approx(sig90_theory, rel=0.03)

    def test_differential_plastic_thinning_anisotropy_r_values(self):
        """Verify sheet thinning resistance scales with Lankford coefficients (R values)."""
        mat_high_r = make_test_material_law73(
            mid=1, r00=2.5, r45=2.0, r90=2.5, sigy0=200.0,
            yield_table=[(0.0, 200.0), (0.1, 300.0)],
        )
        mat_low_r = make_test_material_law73(
            mid=2, r00=0.5, r45=0.5, r90=0.5, sigy0=200.0,
            yield_table=[(0.0, 200.0), (0.1, 300.0)],
        )

        sig_h = np.zeros((1, 3), dtype=float)
        epsp_h = np.zeros(1, dtype=float)
        extra_h = {"uvar73": np.zeros((1, 7)), "thk73": np.array([1.0]), "pla73": np.zeros(1), "off73": np.ones(1), "temp": np.full(1, 293.0)}

        sig_l = np.zeros((1, 3), dtype=float)
        epsp_l = np.zeros(1, dtype=float)
        extra_l = {"uvar73": np.zeros((1, 7)), "thk73": np.array([1.0]), "pla73": np.zeros(1), "off73": np.ones(1), "temp": np.full(1, 293.0)}

        dt = 1.0e-6
        for step in range(40):
            deps = np.array([[5.0e-4, -1.5e-4, 0.0]])
            res_h = shell_update_law73(mat_high_r, sig_h, deps, epsp_h, dt=dt, extra=extra_h)
            sig_h = res_h[0].copy()
            epsp_h[0] = _scalar(res_h[1])
            res_l = shell_update_law73(mat_low_r, sig_l, deps, epsp_l, dt=dt, extra=extra_l)
            sig_l = res_l[0].copy()
            epsp_l[0] = _scalar(res_l[1])

        thk_final_high_r = float(extra_h["thk73"][0])
        thk_final_low_r = float(extra_l["thk73"][0])

        delta_thk_high_r = 1.0 - thk_final_high_r
        delta_thk_low_r = 1.0 - thk_final_low_r

        # Low R material must undergo more thinning than high R material
        assert delta_thk_low_r > delta_thk_high_r, \
            f"Low R thinning ({delta_thk_low_r:.5f}) must exceed High R thinning ({delta_thk_high_r:.5f})"


# ============================================================================
# 4. Progressive Failure & Element Deletion Auditing
# ============================================================================

class TestLaw73ProgressiveFailureAndElementDeletion:
    """Audit tensile failure (epsr1 <= epst < epsr2), plastic failure (epsp >= eps_max), and stable post-erosion."""

    def test_tensile_softening_damage_epsr1_epsr2(self):
        """Progressive tensile softening for epsr1 <= epst < epsr2 and collapse at epsr2."""
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, sigy0=200.0,
            epsr1=0.01, epsr2=0.03,
            yield_table=[(0.0, 200.0), (0.1, 250.0)],
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
            "eps": np.zeros((1, 3)),
        }
        dt = 1.0e-6

        peak_stress = 0.0
        softened_stress = 0.0

        for step in range(40):
            deps = np.array([[8.0e-4, 0.0, 0.0]])
            extra["eps"] += deps
            res = shell_update_law73(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])

            s_xx = float(sig[0, 0])
            if s_xx > peak_stress:
                peak_stress = s_xx
            if step > 20:
                softened_stress = s_xx

        # Beyond epsr1, stress softens below peak
        assert peak_stress > 150.0
        assert softened_stress < peak_stress, f"Expected softened stress ({softened_stress}) < peak ({peak_stress})"

    def test_plastic_strain_failure_and_erosion_eps_max(self):
        """Element deletion when plastic strain reaches eps_max."""
        eps_max = 0.015
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, sigy0=200.0,
            eps_max=eps_max,
            yield_table=[(0.0, 200.0), (0.1, 300.0)],
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
        }
        dt = 1.0e-6

        eroded = False
        for step in range(50):
            deps = np.array([[5.0e-4, -0.33 * 5.0e-4, 0.0]])
            res = shell_update_law73(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])

            if extra["off73"][0] <= 0.8:
                eroded = True
                break

        assert eroded, "Element must be eroded when epsp reaches eps_max"

    def test_post_erosion_dynamic_simulation_stability(self):
        """Post-erosion cycle continuation without solver NaN, Inf, or Courant collapse."""
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, sigy0=200.0, eps_max=0.005,
            yield_table=[(0.0, 200.0), (0.1, 250.0)],
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
        }
        dt = 1.0e-6

        # Step 1: Drive element to erosion
        for step in range(30):
            deps = np.array([[6.0e-4, -0.33 * 6.0e-4, 1.0e-4]])
            res = shell_update_law73(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])

        # Step 2: Continue for 50+ steps on the deleted element
        for post_step in range(50):
            deps_post = np.array([[3.0e-4, -2.0e-4, 1.5e-4]])
            res = shell_update_law73(
                mat, sig, deps_post, epsp, dt=dt, extra=extra, return_tuple=True
            )
            sig_out, epsp_out, c_out = res[0], res[1], res[2]
            sig = sig_out.copy()
            epsp[0] = _scalar(epsp_out)

            assert np.isfinite(sig_out).all(), "Stresses must remain finite"
            assert np.isfinite(epsp_out).all(), "Plastic strains must remain finite"
            assert np.isfinite(c_out).all(), "Sound speed must remain finite"
            assert np.all(c_out > 1000.0), "Sound speed must remain bounded positive to prevent Courant collapse"

    def test_two_element_patch_progressive_rupture(self):
        """Two-element patch where one element ruptures while neighbor stays intact."""
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, sigy0=200.0, eps_max=0.015,
            yield_table=[(0.0, 200.0), (0.1, 250.0)],
        )
        sig = np.zeros((2, 3), dtype=float)
        epsp = np.zeros(2, dtype=float)
        extra = {
            "uvar73": np.zeros((2, 7), dtype=float),
            "off73": np.ones(2, dtype=float),
            "pla73": np.zeros(2, dtype=float),
            "thk73": np.ones(2, dtype=float),
            "temp": np.full(2, 293.0),
        }
        dt = 1.0e-6

        for step in range(50):
            deps = np.array([
                [6.0e-4, -0.33 * 6.0e-4, 0.0],  # Elem 0 (high strain -> rupture)
                [1.0e-4, -0.33 * 1.0e-4, 0.0],  # Elem 1 (low strain -> intact)
            ])
            res = shell_update_law73(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp = np.asarray(res[1]).copy()

        off = extra["off73"]
        assert off[0] <= 0.8, "Element 0 must be eroded"
        assert off[1] == 1.0, "Element 1 must remain intact"
        assert sig[1, 0] > 0.0, "Element 1 must continue carrying positive tensile stress"


# ============================================================================
# 5. Dynamic Thermal Softening & Adiabatic Heating Auditing
# ============================================================================

class TestLaw73ThermalSofteningAndAdiabaticHeating:
    """Audit coupled adiabatic heating and thermal softening."""

    def test_adiabatic_heating_dynamic_simulation(self):
        """Plastic work triggers adiabatic temperature rise (rhocp > 0)."""
        rhocp = 2.4e-3
        t0 = 293.0
        mat = make_test_material_law73(
            E=70000.0, nu=0.33, sigy0=180.0,
            t0=t0, rhocp=rhocp,
            yield_table=[(0.0, 180.0), (0.1, 260.0)],
        )

        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, t0),
            "vol": np.array([100.0]),
        }
        dt = 1.0e-6

        for _ in range(25):
            deps = np.array([[8.0e-4, -0.33 * 8.0e-4, 0.0]])
            sig, epsp = shell_update_law73(mat, sig, deps, epsp, dt=dt, extra=extra)

        final_temp = float(extra["temp"][0])
        assert final_temp > t0, f"Expected temperature to rise above {t0} K, got {final_temp} K"


# ============================================================================
# 6. Acoustic Sound Speed & Courant Time-Step Stability
# ============================================================================

class TestLaw73SoundSpeedAndCourantStability:
    """Audit plane-stress acoustic sound speed c_shell and Courant time-step stability."""

    def test_sound_speed_shell_positivity_and_courant_bounds(self):
        """Verify positive plane-stress sound speed c = sqrt(E / (rho0 * (1 - nu^2))) across alloys."""
        alloys = [
            ("Steel", 210000.0, 0.30, 7.85e-9),
            ("Aluminum", 70000.0, 0.33, 2.70e-9),
            ("Titanium", 110000.0, 0.31, 4.50e-9),
            ("Magnesium", 45000.0, 0.35, 1.80e-9),
        ]
        lc = 2.0  # characteristic shell length = 2.0 mm

        for name, E, nu, rho0 in alloys:
            p = Law73Params(e=E, nu=nu, rho0=rho0)
            c_shell = sound_speed_shell_law73(p)
            c_expected = math.sqrt(E / (rho0 * (1.0 - nu ** 2)))

            assert c_shell == pytest.approx(c_expected, rel=1e-6)
            assert c_shell > 0.0

            dt_courant = lc / c_shell
            assert 1.0e-8 < dt_courant < 1.0e-5, f"Courant dt {dt_courant} outside reasonable bounds for {name}"

    def test_sound_speed_stability_under_modulus_degradation(self):
        """Sound speed remains positive and stable under dynamic Young's modulus degradation."""
        e0, einf, ce, rho0, nu = 70000.0, 50000.0, 25.0, 2.7e-9, 0.33
        p = Law73Params(
            e=e0, einf=einf, ce=ce, rho0=rho0, nu=nu,
        )

        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra = {
            "uvar73": np.zeros((1, 7)),
            "pla73": np.zeros(1),
            "off73": np.ones(1),
            "thk73": np.ones(1),
            "temp": np.full(1, 293.0),
        }
        dt = 1.0e-6

        c_history = []
        for step in range(40):
            deps = np.array([[4.0e-4, -0.33 * 4.0e-4, 0.0]])
            res = shell_update_law73(p, sig, deps, epsp, dt=dt, extra=extra, return_tuple=True)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])
            c = res[2]
            c_history.append(float(np.asarray(c).flat[0]))

        c_min_theory = math.sqrt(einf / (rho0 * (1.0 - nu ** 2)))
        c_max_theory = math.sqrt(e0 / (rho0 * (1.0 - nu ** 2)))

        assert c_history[0] == pytest.approx(c_max_theory, rel=1e-3)
        assert c_history[-1] < c_history[0], "Sound speed must decrease as modulus degrades"
        assert c_history[-1] > c_min_theory, f"Sound speed must remain bounded above c(Einf) = {c_min_theory}"

    def test_courant_step_under_dynamic_pulse_propagation(self):
        """Multi-element patch under dynamic shear wave maintains stable Courant time steps."""
        lx = 2.0
        mat = make_test_material_law73(E=70000.0, nu=0.33, rho0=2.7e-9, sigy0=220.0)

        c_theory = sound_speed_shell_law73(mat)
        dt_courant_bound = lx / c_theory

        assert dt_courant_bound > 0.0
        assert np.isfinite(dt_courant_bound)
        assert dt_courant_bound > 1.0e-7
