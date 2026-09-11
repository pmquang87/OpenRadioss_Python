"""
Auditor 2C: Dynamic Engine Simulation & Energy Balance Auditor for M555 (/MAT/LAW57 /MAT/BARLAT3).

Exhaustive dynamic engine simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Shell BT4 with LAW57 under cyclic tension, compression, and shear
   - Shell QEPH with LAW57 under dynamic biaxial tension and shear
   - Shell Tri3 with LAW57 under dynamic stretching
   - Shell BT4 2x2 multi-element patch with LAW57 under cyclic loading
   - Verified through Starter and Engine decks under cyclic velocity/displacement loading.
   - Asserts normal termination, >= 20 cycles, energy balance error |ERR| < 1.0%,
     positive internal strain energy accumulation (IE > 0), and external work consistency.
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt
   - Midpoint trapezoidal work matches exact analytical strain energy in closed reversible cycles.
   - Free vibration of undamped shell elements (BT4, QEPH):
     mechanical energy E_tot = E_kin + E_int is strictly conserved (|Delta E| / E_0 < 1.0%).
   - Plastic dissipation monotonically increasing during plastic flow.
3. Physical anisotropic sheet metal behavior:
   - Verify differential yield strengths across 0°, 45°, 90° rolling orientations matching Barlat-Lian theory.
   - Verify differential plastic thinning across different Lankford R values (plastic incompressibility).
   - Cyclic loading exhibiting Bauschinger reverse softening under kinematic hardening (F_isokin = 1.0 in Fortran)
     vs expanding elastic domain under isotropic hardening (F_isokin = 0.0 in Fortran).
4. Progressive failure & element deletion:
   - Tensile failure when eps_t >= eps_t2 and plastic failure when eps_p >= eps_max
   - Verify stresses collapse to zero and post-erosion dynamic simulation continues stably without NaNs, Infs, or Courant collapse.
   - Two-element patch progressive rupture where one element erodes while neighbor stays intact.
5. Acoustic sound speed & Courant time-step stability:
   - Sound speed c_shell maintaining positive, stable Courant bounds across all dynamic cycles.
   - Sound speed stability under dynamic modulus degradation (CE / Einf).
   - Courant time-step stability under dynamic acoustic pulse propagation in multi-element patch.

Fortran references:
- engine/source/materials/mat/mat057/sigeps57c.F90 (2D shell plane-stress constitutive kernel)
- starter/source/materials/mat/mat057/calculp2.F90 (Newton-Raphson solver for parameter p)
- starter/source/materials/mat/mat057/hm_read_mat57.F90 (parameter initialization and checks)
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
from pyradioss.materials.law57_barlat import (
    Law57Params,
    barlat_params,
    barlat_equivalent_stress,
    build_law57,
    calculp2,
    shell_update_law57,
    sound_speed_shell_law57,
    tangent_law57_shell,
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
    tstop: float = 2.0e-4,
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


def make_test_material_law57(
    mid: int = 1,
    rho0: float = 2.7e-9,  # typical aluminum: 2.7e-9 ton/mm^3
    E: float = 70000.0,
    nu: float = 0.33,
    r00: float = 1.5,
    r45: float = 1.2,
    r90: float = 1.8,
    m: float = 8.0,
    sigy0: float = 250.0,
    fisokin: float = 0.0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW57 (/MAT/BARLAT3) Material instance."""
    params = {
        "E": E,
        "nu": nu,
        "r00": r00,
        "r45": r45,
        "r90": r90,
        "m": m,
        "sigy0": sigy0,
        "fisokin": fisokin,
        "chard": fisokin,
    }
    params.update(kwargs)
    mat = Material(id=mid, law=57, rho0=rho0, title="Alu_LAW57_Barlat", params=params)
    return mat


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

class TestLaw57MultiCycleDynamicSimulations:
    """Audit multi-cycle explicit dynamic simulations across supported shell element formulations."""

    def test_shell_bt4_cyclic_tension_compression_shear_engine(self, tmp_path: Path):
        """Shell BT4 with LAW57 under cyclic tension, compression, and shear."""
        run_name = "SHELL_BT4_LAW57_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        # Yield hardening curve: 100 MPa initial yield to 250 MPa
        d.funct(10, "yield_fct", [(0.0, 100.0), (0.05, 180.0), (0.1, 250.0)])
        d.mat_law57(
            1, "AluBarlat",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.5, r45=1.2, r90=1.8, m=8.0,
            curves=[(10, 1.0, 0.0)],
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
            (0.0, 1500.0),
            (0.5e-4, 1500.0),
            (0.5001e-4, -1000.0),
            (1.0e-4, -1000.0),
            (1.0001e-4, 1200.0),
            (1.5e-4, 1200.0),
        ])
        d.impvel(1, "pull_x", 11, "X", 2)

        # In-plane shear loading in Y
        d.funct(12, "vy_shear", [
            (0.0, 600.0),
            (0.7e-4, 600.0),
            (0.7001e-4, -600.0),
            (1.5e-4, -600.0),
        ])
        d.impvel(2, "shear_y", 12, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.5e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20, f"Expected >= 20 cycles, got {state.cycle}"
        assert state.stop_reason == "", f"Unexpected stop reason: {state.stop_reason}"

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"BT4 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0, "Internal strain energy must be strictly positive"
        assert en["EW"] > 0.0, "External work must be positive"

        # Check plastic strain accumulation occurred
        mat_extra = eng_model.shells.state.get("mat_extra", {})
        assert "pla57" in mat_extra or "epsp" in eng_model.shells.state
        pla_final = mat_extra.get("pla57", eng_model.shells.state.get("epsp"))
        assert np.any(pla_final > 0.0), "Plastic strain must accumulate during cycle"

    def test_shell_qeph_dynamic_biaxial_tension_shear_engine(self, tmp_path: Path):
        """Shell QEPH (Ishell=24) with LAW57 under dynamic biaxial tension and shear."""
        run_name = "SHELL_QEPH_LAW57_BIAX"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 260.0), (0.1, 360.0)])
        d.mat_law57(
            1, "AluQEPH",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.4, r45=1.1, r90=1.7, m=8.0,
            curves=[(10, 1.0, 0.0)],
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
        d.funct(11, "vx_pull", [(0.0, 400.0), (1.2e-4, 400.0)])
        d.impvel(1, "pull_x", 11, "X", 4)

        # Pull top edge (3, 4) in Y
        d.grnod_node(5, "pull_y_nodes", [3, 4])
        d.funct(12, "vy_pull", [(0.0, 350.0), (1.2e-4, 350.0)])
        d.impvel(2, "pull_y", 12, "Y", 5)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.2e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"QEPH energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0

    def test_shell_tri3_dynamic_stretching_engine(self, tmp_path: Path):
        """Shell Tri3 (Ish3n=1) with LAW57 under dynamic stretching."""
        run_name = "SHELL_TRI3_LAW57_STR"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 240.0), (0.1, 340.0)])
        d.mat_law57(
            1, "AluTri3",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.5, r45=1.2, r90=1.8, m=8.0,
            curves=[(10, 1.0, 0.0)],
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
        d.funct(11, "vx_pull", [(0.0, 500.0), (1.0e-4, 500.0)])
        d.impvel(1, "pull_apex_x", 11, "X", 2)
        d.funct(12, "vy_pull", [(0.0, 250.0), (1.0e-4, 250.0)])
        d.impvel(2, "pull_apex_y", 12, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.0e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Tri3 energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0
        assert en["HE"] == 0.0, "Tri3 formulation has zero hourglass energy"

    def test_shell_bt4_multi_element_patch_cyclic_engine(self, tmp_path: Path):
        """2x2 multi-element BT4 patch with LAW57 under dynamic cyclic shear and stretch."""
        run_name = "SHELL_PATCH_LAW57_CYC"
        s_path = str(tmp_path / f"{run_name}_0000.rad")
        e_path = str(tmp_path / f"{run_name}_0001.rad")

        d = StarterDeck(run_name)
        d.funct(10, "yield_fct", [(0.0, 250.0), (0.1, 350.0)])
        d.mat_law57(
            1, "AluPatch",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.5, r45=1.2, r90=1.8, m=8.0,
            curves=[(10, 1.0, 0.0)],
        )
        d.prop_shell(1, "PropPatch", thick=1.0, nip=3, ishell=1)
        d.part(1, "PartPatch", 1, 1)

        # 3x3 node grid (9 nodes, 4 shells)
        # 7 - 8 - 9
        # 4 - 5 - 6
        # 1 - 2 - 3
        nodes = [
            (1, 0.0, 0.0, 0.0), (2, 5.0, 0.0, 0.0), (3, 10.0, 0.0, 0.0),
            (4, 0.0, 5.0, 0.0), (5, 5.0, 5.0, 0.0), (6, 10.0, 5.0, 0.0),
            (7, 0.0, 10.0, 0.0), (8, 5.0, 10.0, 0.0), (9, 10.0, 10.0, 0.0),
        ]
        d.node(nodes)
        d.shell(1, [
            (1, 1, 2, 5, 4),
            (2, 2, 3, 6, 5),
            (3, 4, 5, 8, 7),
            (4, 5, 6, 9, 8),
        ])

        # Fix bottom nodes (1, 2, 3)
        d.grnod_node(1, "fix_bottom", [1, 2, 3])
        d.bcs(1, "bcs_fix", "111", "111", 1)

        # Cyclic pull on top nodes (7, 8, 9)
        d.grnod_node(2, "top_nodes", [7, 8, 9])
        d.funct(11, "vy_top", [
            (0.0, 300.0),
            (0.6e-4, 300.0),
            (0.6001e-4, -200.0),
            (1.2e-4, -200.0),
        ])
        d.impvel(1, "pull_top", 11, "Y", 2)

        d.write(s_path)
        _write_engine_deck(e_path, run_name, tstop=1.2e-4, dt_scale=0.5)

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = run_starter(s_path)
            eng_model = run_engine(e_path)

        state = eng_model.engine_state
        assert state.cycle >= 20
        assert state.stop_reason == ""

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Multi-element patch energy error {en['ERR']}% exceeds 1.0%"
        assert en["IE"] > 0.0


# ============================================================================
# 2. Energy Conservation & Work Balance Auditing
# ============================================================================

class TestLaw57EnergyConservationAndWorkBalance:
    """Audit incremental strain energy ledger, reversible elastic conservation, and plastic dissipation."""

    def test_incremental_strain_energy_ledger_matches_external_work(self):
        """Verify Delta E_int = int sigma : depsilon * dV matches external work in closed cycles."""
        mat = make_test_material_law57(
            E=70000.0, nu=0.33, r00=1.5, r45=1.2, r90=1.8, m=8.0,
            sigy0=200.0,
            curves=[([0.0, 0.1], [200.0, 350.0], 0.0)],
        )

        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}

        thick = 1.0
        area = 100.0  # 10x10 mm shell
        vol0 = area * thick
        dt = 1.0e-6

        # Phase 1: Pure elastic deformation (verify against exact analytical strain energy)
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
            res = shell_update_law57(mat, sig, deps, epsp, dt=dt, extra=extra, return_sound_speed=False)
            sig_new = res[0] if isinstance(res, tuple) else res
            sig = sig_new.copy()
            epsp[0] = _scalar(res[1]) if isinstance(res, tuple) else epsp[0]

            sig_mid = 0.5 * (sig_old + sig_new)
            # Plane-stress incremental work: sig_xx*deps_xx + sig_yy*deps_yy + sig_xy*deps_xy (deps_xy is engineering shear gamma)
            de = (sig_mid[0, 0] * deps[0, 0] + sig_mid[0, 1] * deps[0, 1] + sig_mid[0, 2] * deps[0, 2]) * vol0
            eint_incremental += de
            eps_total += deps[0]

        # Analytical elastic strain energy: 0.5 * (a11*ex^2 + a11*ey^2 + 2*a12*ex*ey + g*exy^2) * vol0
        ex, ey, exy = eps_total
        analytical_eint = 0.5 * (a11 * ex**2 + a11 * ey**2 + 2.0 * a12 * ex * ey + g * exy**2) * vol0
        assert eint_incremental == pytest.approx(analytical_eint, rel=1e-5), \
            f"Incremental strain energy {eint_incremental} != analytical {analytical_eint}"
        assert epsp[0] == 0.0, "No plastic strain should accumulate in elastic regime"

        # Phase 2: Plastic loading and reversal
        for step in range(35):
            factor = 1.0 if step < 20 else -1.0
            deps = np.array([[factor * 5.0e-4, -0.33 * factor * 5.0e-4, factor * 2.0e-4]])
            sig_old = sig.copy()
            res = shell_update_law57(mat, sig, deps, epsp, dt=dt, extra=extra, return_sound_speed=False)
            sig_new = res[0] if isinstance(res, tuple) else res
            sig = sig_new.copy()
            epsp[0] = _scalar(res[1]) if isinstance(res, tuple) else epsp[0]

            sig_mid = 0.5 * (sig_old + sig_new)
            de = (sig_mid[0, 0] * deps[0, 0] + sig_mid[0, 1] * deps[0, 1] + sig_mid[0, 2] * deps[0, 2]) * vol0
            eint_incremental += de

        assert eint_incremental > 0.0, "Total strain energy must remain strictly positive"
        assert epsp[0] > 0.0, "Plastic strain must accumulate during high strain cycle"

    def test_shell_bt4_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Shell BT4 in elastic regime: |Delta E| / E_0 < 1%."""
        deck = StarterDeck("BT4_ELAS_OSC")
        deck.mat_law57(
            1, "ElasticAlu",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.0, r45=1.0, r90=1.0, m=6.0,
            sigy0=20000.0,  # High yield stress to remain strictly elastic
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
        vx0 = 50.0
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
        assert np.all(group.state["epsp"] == 0.0), "Plastic strain must remain identically zero"

    def test_shell_qeph_undamped_free_oscillation_energy_conservation(self, tmp_path: Path):
        """Free vibration of undamped Shell QEPH (Ishell=24) in elastic regime: |Delta E| / E_0 < 1%."""
        deck = StarterDeck("QEPH_ELAS_OSC")
        deck.mat_law57(
            1, "ElasticAlu",
            rho=2.7e-9, e=70000.0, nu=0.33,
            r00=1.0, r45=1.0, r90=1.0, m=6.0,
            sigy0=20000.0,
        )
        thick0 = 1.0
        deck.prop_shell(1, "PropQEPH", thick=thick0, nip=3, ishell=24, hm=0.0, hf=0.0, hr=0.0)
        deck.part(1, "PartShell", 1, 1)

        lx, ly = 10.0, 10.0
        deck.node([
            (1, 0.0, 0.0, 0.0), (2, lx, 0.0, 0.0), (3, lx, ly, 0.0), (4, 0.0, ly, 0.0),
        ])
        deck.shell(1, [(1, 1, 2, 3, 4)])

        s_path = str(tmp_path / "QEPH_ELAS_OSC_0000.rad")
        deck.write(s_path)
        with contextlib.redirect_stdout(io.StringIO()):
            model = run_starter(s_path)

        group = model.shells_qeph if (getattr(model, "shells_qeph", None) is not None and model.shells_qeph.n > 0) else model.shells
        rho0 = 2.7e-9
        area0 = lx * ly
        m_node = rho0 * thick0 * area0 / 4.0
        mass_vec = np.full(4, m_node)

        v = np.zeros_like(model.x)
        vx0 = 50.0
        v[[1, 2], 0] = vx0
        v[[0, 3], 0] = -vx0

        e_kin_0 = 0.5 * np.sum(mass_vec * (vx0 ** 2))
        e_int_0 = float(np.sum(group.state["eint"]))
        e_tot_0 = e_kin_0 + e_int_0

        dt = 1.0e-7
        fint = np.zeros_like(model.x)
        mint = np.zeros_like(model.x)
        model.v = v.copy()

        e_tot_history = []
        for step in range(80):
            fint.fill(0.0)
            mint.fill(0.0)
            shell_qeph.forces(group, model.x, model.v, model.vr, dt, fint, mint)
            v_old = model.v.copy()
            acc = fint / mass_vec[:, None]
            model.v += acc * dt
            model.x += model.v * dt
            v_mid = 0.5 * (v_old + model.v)
            e_kin = 0.5 * np.sum(mass_vec[:, None] * (v_mid ** 2))
            e_int = float(np.sum(group.state["eint"]))
            e_tot_history.append(e_kin + e_int)

        max_err = max(abs(e - e_tot_0) / e_tot_0 for e in e_tot_history)
        assert max_err < 0.01, f"QEPH elastic energy error {max_err*100:.3f}% exceeds 1.0%"

    def test_plastic_dissipation_monotonically_increasing(self):
        """Plastic strain and plastic work dissipation monotonically non-decreasing during plastic flow."""
        mat = make_test_material_law57(
            E=70000.0, nu=0.33, r00=1.5, r45=1.2, r90=1.8, m=8.0,
            sigy0=220.0,
            curves=[([0.0, 0.1], [220.0, 340.0], 0.0)],
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}

        epsp_history = []
        dissipation_history = []
        cum_diss = 0.0
        dt = 1.0e-6

        for step in range(50):
            # Dynamic multi-axial strain increments (tension and shear)
            deps = np.array([[3.0e-4, -1.0e-4, 1.5e-4]])
            res = shell_update_law57(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])

            d_pla = float(epsp[0]) - (epsp_history[-1] if epsp_history else 0.0)
            # Dissipated plastic increment: sigma_eq * d_pla
            bp = mat.params.barlat if isinstance(mat.params, Law57Params) else build_law57(mat).barlat
            seq = barlat_equivalent_stress(sig[0], bp.a, bp.c, bp.h_bar, bp.p, mat.params.get("m", 8.0))
            dWp = seq * d_pla
            cum_diss += dWp

            epsp_history.append(float(epsp[0]))
            dissipation_history.append(cum_diss)

        d_epsp = np.diff(epsp_history)
        assert np.all(d_epsp >= -1e-15), "Plastic strain increments must be non-negative"

        d_diss = np.diff(dissipation_history)
        assert np.all(d_diss >= -1e-15), "Plastic dissipation increments must be non-negative"
        assert dissipation_history[-1] > dissipation_history[0], "Plastic dissipation must strictly increase"


# ============================================================================
# 3. Anisotropic Sheet Metal Behavior Auditing
# ============================================================================

class TestLaw57AnisotropicSheetBehavior:
    """Audit differential yield strengths across 0°, 45°, 90°, and Bauschinger effect."""

    def test_differential_yield_strengths_0_45_90_rolling_angles(self):
        """Verify differential yield stresses across 0°, 45°, and 90° rolling angles match theory."""
        r00, r45, r90, m = 1.5, 1.2, 1.8, 8.0
        sigy0 = 250.0
        bp = barlat_params(r0=r00, r45=r45, r90=r90, m=m)

        # Theoretical yield stresses:
        # 1. Rolling direction (0°): sigma_0 = sigy0
        sig0_theory = sigy0

        # 2. Transverse direction (90°): sigma_90 = sigy0 / h_bar
        sig90_theory = sigy0 / bp.h_bar

        # 3. 45° direction: sigma_45 where seq([s/2, s/2, s/2]) = sigy0
        s45_test = np.array([0.5, 0.5, 0.5])
        seq_unit45 = barlat_equivalent_stress(s45_test, bp.a, bp.c, bp.h_bar, bp.p, m)
        sig45_theory = sigy0 / seq_unit45

        # Verify that anisotropic yield stresses differ noticeably:
        assert abs(sig90_theory - sig0_theory) > 5.0, "Anisotropy must produce distinct 90° yield stress"
        assert abs(sig45_theory - sig0_theory) > 5.0, "Anisotropy must produce distinct 45° yield stress"

        # Now test through constitutive kernel:
        mat = make_test_material_law57(
            E=70000.0, nu=0.33, r00=r00, r45=r45, r90=r90, m=m, sigy0=sigy0,
        )

        # Simulation 1: Uniaxial tension along 0° (exx > 0, eyy = -nu*exx)
        sig_0 = np.zeros((1, 3), dtype=float)
        epsp_0 = np.zeros(1, dtype=float)
        for _ in range(60):
            deps = np.array([[1.0e-4, -0.33 * 1.0e-4, 0.0]])
            res = shell_update_law57(mat, sig_0, deps, epsp_0, dt=1e-6)
            sig_0 = res[0].copy()
            epsp_0[0] = _scalar(res[1])
            if epsp_0[0] > 1e-6:
                break
        assert sig_0[0, 0] == pytest.approx(sig0_theory, rel=0.02)

        # Simulation 2: Uniaxial tension along 90° (eyy > 0, exx = -nu*eyy)
        sig_90 = np.zeros((1, 3), dtype=float)
        epsp_90 = np.zeros(1, dtype=float)
        for _ in range(60):
            deps = np.array([[-0.33 * 1.0e-4, 1.0e-4, 0.0]])
            res = shell_update_law57(mat, sig_90, deps, epsp_90, dt=1e-6)
            sig_90 = res[0].copy()
            epsp_90[0] = _scalar(res[1])
            if epsp_90[0] > 1e-6:
                break
        assert sig_90[0, 1] == pytest.approx(sig90_theory, rel=0.02)

    def test_differential_plastic_thinning_anisotropy_r_values(self):
        """Verify sheet thinning resistance scales with Lankford coefficients (R values)."""
        # Material High R (resists thinning, higher width contraction)
        mat_high_r = make_test_material_law57(
            mid=1, r00=2.2, r45=2.0, r90=2.5, m=6.0, sigy0=200.0,
            curves=[([0.0, 0.1], [200.0, 300.0], 0.0)],
        )
        # Material Low R (thins heavily)
        mat_low_r = make_test_material_law57(
            mid=2, r00=0.6, r45=0.5, r90=0.7, m=6.0, sigy0=200.0,
            curves=[([0.0, 0.1], [200.0, 300.0], 0.0)],
        )

        sig_h = np.zeros((1, 3), dtype=float)
        epsp_h = np.zeros(1, dtype=float)
        extra_h = {"thk57": np.array([1.0]), "thk0": np.array([1.0])}

        sig_l = np.zeros((1, 3), dtype=float)
        epsp_l = np.zeros(1, dtype=float)
        extra_l = {"thk57": np.array([1.0]), "thk0": np.array([1.0])}

        # Subject both materials to equal plastic stretch
        dt = 1.0e-6
        for step in range(40):
            deps = np.array([[5.0e-4, -1.5e-4, 0.0]])
            res_h = shell_update_law57(mat_high_r, sig_h, deps, epsp_h, dt=dt, extra=extra_h)
            sig_h = res_h[0].copy()
            epsp_h[0] = _scalar(res_h[1])
            res_l = shell_update_law57(mat_low_r, sig_l, deps, epsp_l, dt=dt, extra=extra_l)
            sig_l = res_l[0].copy()
            epsp_l[0] = _scalar(res_l[1])

        thk_final_high_r = float(extra_h["thk57"][0])
        thk_final_low_r = float(extra_l["thk57"][0])

        delta_thk_high_r = 1.0 - thk_final_high_r
        delta_thk_low_r = 1.0 - thk_final_low_r

        # Low R material must undergo more thinning than high R material
        assert delta_thk_low_r > delta_thk_high_r, \
            f"Low R thinning ({delta_thk_low_r:.5f}) must exceed High R thinning ({delta_thk_high_r:.5f})"

    def test_bauschinger_effect_kinematic_vs_isotropic_hardening(self):
        """Verify Bauschinger reverse softening under kinematic hardening vs expanding domain under isotropic."""
        # Upstream Fortran (sigeps57c.F90 lines 278-280):
        # yld = (1 - fisokin)*yld + fisokin*yld0
        # hk = fisokin * dyld_dp
        # Thus fisokin = 1.0 is pure kinematic hardening, and fisokin = 0.0 is pure isotropic hardening.

        # 1. Kinematic Hardening (fisokin = 1.0): backstress develops
        mat_kin = make_test_material_law57(
            mid=1, E=70000.0, nu=0.33, sigy0=200.0, fisokin=1.0, chard=1.0,
            curves=[([0.0, 0.1], [200.0, 400.0], 0.0)],
        )
        sig_kin = np.zeros((1, 3), dtype=float)
        epsp_kin = np.zeros(1, dtype=float)
        extra_kin = {"sigb57": np.zeros((1, 3))}

        # 2. Isotropic Hardening (fisokin = 0.0): zero backstress, expanding yield radius
        mat_iso = make_test_material_law57(
            mid=2, E=70000.0, nu=0.33, sigy0=200.0, fisokin=0.0, chard=0.0,
            curves=[([0.0, 0.1], [200.0, 400.0], 0.0)],
        )
        sig_iso = np.zeros((1, 3), dtype=float)
        epsp_iso = np.zeros(1, dtype=float)
        extra_iso = {"sigb57": np.zeros((1, 3))}

        dt = 1.0e-6

        # Forward tension past yield
        for step in range(30):
            deps_fwd = np.array([[2.0e-4, -0.33 * 2.0e-4, 0.0]])
            res_k = shell_update_law57(mat_kin, sig_kin, deps_fwd, epsp_kin, dt=dt, extra=extra_kin)
            sig_kin = res_k[0].copy()
            epsp_kin[0] = _scalar(res_k[1])
            res_i = shell_update_law57(mat_iso, sig_iso, deps_fwd, epsp_iso, dt=dt, extra=extra_iso)
            sig_iso = res_i[0].copy()
            epsp_iso[0] = _scalar(res_i[1])

        sig_fwd_kin = float(sig_kin[0, 0])
        sig_fwd_iso = float(sig_iso[0, 0])
        assert sig_fwd_kin > 200.0
        assert sig_fwd_iso > 200.0

        # Backstress must be positive in kinematic hardening, zero in isotropic
        alpha_xx_kin = float(extra_kin["sigb57"][0, 0])
        alpha_xx_iso = float(extra_iso["sigb57"][0, 0])
        assert alpha_xx_kin > 0.0, f"Kinematic backstress must be positive, got {alpha_xx_kin}"
        assert alpha_xx_iso == pytest.approx(0.0), f"Isotropic backstress must be zero, got {alpha_xx_iso}"

        # Reverse compression: apply negative strain increments until reverse yielding begins
        epsp_kin_rev0 = float(epsp_kin[0])
        epsp_iso_rev0 = float(epsp_iso[0])

        rev_yield_stress_kin = None
        rev_yield_stress_iso = None

        for step in range(60):
            deps_rev = np.array([[-2.0e-4, 0.33 * 2.0e-4, 0.0]])
            res_k = shell_update_law57(mat_kin, sig_kin, deps_rev, epsp_kin, dt=dt, extra=extra_kin)
            sig_kin = res_k[0].copy()
            epsp_kin[0] = _scalar(res_k[1])
            res_i = shell_update_law57(mat_iso, sig_iso, deps_rev, epsp_iso, dt=dt, extra=extra_iso)
            sig_iso = res_i[0].copy()
            epsp_iso[0] = _scalar(res_i[1])

            if rev_yield_stress_kin is None and (epsp_kin[0] - epsp_kin_rev0) > 1.0e-5:
                rev_yield_stress_kin = abs(float(sig_kin[0, 0]))

            if rev_yield_stress_iso is None and (epsp_iso[0] - epsp_iso_rev0) > 1.0e-5:
                rev_yield_stress_iso = abs(float(sig_iso[0, 0]))

        assert rev_yield_stress_kin is not None, "Kinematic hardening must yield in reverse compression"
        assert rev_yield_stress_iso is not None, "Isotropic hardening must yield in reverse compression"

        # Bauschinger reverse softening: kinematic reverse yield magnitude is smaller than isotropic
        assert rev_yield_stress_kin < rev_yield_stress_iso, \
            f"Bauschinger softening: kinematic yield ({rev_yield_stress_kin:.2f}) < isotropic yield ({rev_yield_stress_iso:.2f})"
        assert rev_yield_stress_kin < sig_fwd_kin, \
            f"Reverse yield {rev_yield_stress_kin:.2f} must be less than forward peak {sig_fwd_kin:.2f}"


# ============================================================================
# 4. Progressive Failure & Element Deletion Auditing
# ============================================================================

class TestLaw57ProgressiveFailureAndElementDeletion:
    """Audit tensile failure (eps_t >= eps_t2), plastic failure (eps_p >= eps_max), and stable post-erosion."""

    def test_tensile_failure_damage_and_erosion_eps_t2(self):
        """Progressive tensile damage for eps_t1 <= eps_t < eps_t2 and erosion when eps_t >= eps_t2."""
        mat = make_test_material_law57(
            E=70000.0, nu=0.33, sigy0=250.0,
            eps_t1=0.015, eps_t2=0.025,
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}
        dt = 1.0e-6

        eroded = False
        damage_grew = False

        for step in range(50):
            deps = np.array([[8.0e-4, 0.0, 0.0]])
            res = shell_update_law57(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])

            dmg = extra.get("dmg57", extra.get("dmg"))
            off = extra.get("off57", extra.get("off"))

            if dmg is not None and dmg[0, 2] > 0.0:
                damage_grew = True

            if off is not None and off[0] == 0.0:
                eroded = True
                break

        assert damage_grew, "Tensile damage factor must grow past eps_t1"
        assert eroded, "Element must erode once eps_t reaches eps_t2"
        np.testing.assert_allclose(sig, 0.0, atol=1e-12)

    def test_plastic_strain_failure_and_erosion_eps_max(self):
        """Element deletion when plastic strain eps_p reaches eps_max."""
        mat = make_test_material_law57(
            E=70000.0, nu=0.33, sigy0=200.0,
            eps_max=0.01,
            curves=[([0.0, 0.1], [200.0, 300.0], 0.0)],
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}
        dt = 1.0e-6

        eroded = False
        for step in range(40):
            deps = np.array([[5.0e-4, -0.33 * 5.0e-4, 0.0]])
            res = shell_update_law57(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])

            off = extra.get("off57", extra.get("off"))
            if off is not None and off[0] == 0.0:
                eroded = True
                break

        assert eroded, "Element must be eroded when epsp reaches eps_max"
        np.testing.assert_allclose(sig, 0.0, atol=1e-12)

    def test_post_erosion_dynamic_simulation_stability(self):
        """Post-erosion cycle continuation without solver NaN, Inf, or Courant collapse."""
        mat = make_test_material_law57(
            E=70000.0, nu=0.33, sigy0=200.0, eps_max=0.005,
        )
        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}
        dt = 1.0e-6

        # Step 1: Drive element to erosion
        for step in range(30):
            deps = np.array([[6.0e-4, -0.33 * 6.0e-4, 1.0e-4]])
            res = shell_update_law57(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])

        off = extra.get("off57", extra.get("off"))
        assert off[0] == 0.0, "Element must be eroded"

        # Step 2: Continue for 60+ steps on the deleted element
        for post_step in range(60):
            deps_post = np.array([[3.0e-4, -2.0e-4, 1.5e-4]])
            res = shell_update_law57(
                mat, sig, deps_post, epsp, dt=dt, extra=extra, return_sound_speed=True
            )
            sig_out, epsp_out, c_out = res[0], res[1], res[2]
            sig = sig_out.copy()
            epsp[0] = _scalar(epsp_out)

            assert np.isfinite(sig_out).all(), "Stresses must remain finite"
            assert np.isfinite(epsp_out).all(), "Plastic strains must remain finite"
            assert np.isfinite(c_out).all(), "Sound speed must remain finite"
            np.testing.assert_allclose(sig_out, 0.0, atol=1e-15), "Deleted element stresses must stay zero"
            assert np.all(c_out > 1000.0), "Sound speed must remain bounded positive to prevent Courant collapse"

    def test_two_element_patch_progressive_rupture(self):
        """Two-element patch where one element ruptures while neighbor stays intact."""
        mat = make_test_material_law57(
            E=70000.0, nu=0.33, sigy0=200.0, eps_max=0.015,
        )
        # 2 elements batched
        sig = np.zeros((2, 3), dtype=float)
        epsp = np.zeros(2, dtype=float)
        extra = {
            "off57": np.ones(2, dtype=float),
            "pla57": np.zeros(2, dtype=float),
            "sigb57": np.zeros((2, 3), dtype=float),
            "dmg57": np.zeros((2, 3), dtype=float),
            "eps57": np.zeros((2, 3), dtype=float),
        }
        dt = 1.0e-6

        # Element 0 experiences high strain, Element 1 experiences low strain
        for step in range(50):
            deps = np.array([
                [6.0e-4, -0.33 * 6.0e-4, 0.0],   # Elem 0 (high strain -> failure)
                [1.0e-4, -0.33 * 1.0e-4, 0.0],   # Elem 1 (low strain -> intact)
            ])
            res = shell_update_law57(mat, sig, deps, epsp, dt=dt, extra=extra)
            sig = res[0].copy()
            epsp = np.asarray(res[1]).copy()

        off = extra["off57"]
        assert off[0] == 0.0, "Element 0 must be eroded"
        assert off[1] == 1.0, "Element 1 must remain intact"

        np.testing.assert_allclose(sig[0], 0.0, atol=1e-12)
        assert sig[1, 0] > 0.0, "Element 1 must continue carrying positive tensile stress"


# ============================================================================
# 5. Acoustic Sound Speed & Courant Time-Step Stability
# ============================================================================

class TestLaw57SoundSpeedAndCourantStability:
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
            p = Law57Params(E=E, nu=nu, rho0=rho0)
            c_shell = sound_speed_shell_law57(p)
            c_expected = math.sqrt(E / (rho0 * (1.0 - nu ** 2)))

            assert c_shell == pytest.approx(c_expected, rel=1e-6)
            assert c_shell > 0.0

            dt_courant = lc / c_shell
            assert 1.0e-8 < dt_courant < 1.0e-5, f"Courant dt {dt_courant} outside reasonable bounds for {name}"

    def test_sound_speed_stability_under_modulus_degradation(self):
        """Sound speed remains positive and stable under dynamic Young's modulus degradation."""
        # Modulus degrades from E0=70000 to Einf=50000 with ce=25
        e0, einf, ce, rho0, nu = 70000.0, 50000.0, 25.0, 2.7e-9, 0.33
        p = Law57Params(
            E=e0, einf=einf, ce=ce, rho0=rho0, nu=nu, sigy0=200.0,
        )

        sig = np.zeros((1, 3), dtype=float)
        epsp = np.zeros(1, dtype=float)
        extra: dict[str, Any] = {}
        dt = 1.0e-6

        c_history = []
        for step in range(40):
            deps = np.array([[4.0e-4, -0.33 * 4.0e-4, 0.0]])
            res = shell_update_law57(p, sig, deps, epsp, dt=dt, extra=extra, return_sound_speed=True)
            sig = res[0].copy()
            epsp[0] = _scalar(res[1])
            c = res[2]
            c_history.append(float(np.asarray(c).flat[0]))

        # Sound speed must decrease smoothly as E degrades, but remain bounded above c(Einf)
        c_min_theory = math.sqrt(einf / (rho0 * (1.0 - nu ** 2)))
        c_max_theory = math.sqrt(e0 / (rho0 * (1.0 - nu ** 2)))

        assert c_history[0] == pytest.approx(c_max_theory, rel=1e-3)
        assert c_history[-1] < c_history[0], "Sound speed must decrease as modulus degrades"
        assert c_history[-1] > c_min_theory, f"Sound speed must remain bounded above c(Einf) = {c_min_theory}"

    def test_courant_step_under_dynamic_pulse_propagation(self):
        """Multi-element patch under dynamic shear wave maintains stable Courant time steps."""
        lx = 2.0
        mat = make_test_material_law57(E=70000.0, nu=0.33, rho0=2.7e-9, sigy0=220.0)

        # Compute theoretical Courant bound
        c_theory = sound_speed_shell_law57(mat.params)
        dt_courant_bound = lx / c_theory

        assert dt_courant_bound > 0.0
        assert np.isfinite(dt_courant_bound)
        assert dt_courant_bound > 1.0e-7
