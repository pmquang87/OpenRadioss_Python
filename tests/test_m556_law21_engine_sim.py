"""
Auditor 2C: Dynamic Engine Simulation & Energy Balance Auditor for M556 (/MAT/LAW21, /MAT/DPRAG).

Comprehensive explicit dynamic simulation and energy balance audit suite verifying:
1. Multi-cycle explicit dynamic simulations:
   - Hexa8 brick with LAW21 under cyclic hydrostatic compression and unloading
   - Hexa8 brick with LAW21 under cyclic shear loading with confining pressure
   - Tetra4 solid with LAW21 under dynamic compression and shear
   - Multi-element patch simulations (2x2x2 Hexa8 patch, 5-element Tetra4 patch)
2. Energy conservation & work balance:
   - Incremental strain energy ledger Delta E_int = int sigma : depsilon * dV * dt
     matching external trapezoidal work
   - Total mechanical energy conservation: |Delta E| / E_0 < 1% across cycles under
     reversible elastic vibrations
   - Monotonic plastic dissipation during plastic yielding and compaction
3. Physical soil/concrete compaction & Drucker-Prager cone behavior:
   - Pressure-dependent shear strength: verify that higher confining pressure increases
     shear resistance before yielding
   - Compaction volume reduction and permanent volumetric plastic strain eps_{p,vol} = mu_bak
   - Unloading hysteresis with higher slope K_unload > C_1 dissipating compaction energy
   - Tensile cutoff behavior under triaxial expansion with pressure limited to P_min
4. Acoustic sound speed & Courant time-step stability:
   - Verify sound speed c_solid remains positive, finite, and well-behaved across cycles,
     guaranteeing Courant time step Delta t = L_e / c stability.
5. End-to-end Starter and Engine explicit dynamic runs:
   - Single-element and multi-element patch models running through Starter + Engine,
     achieving >= 20 cycles and energy balance error |ERR| < 1.0%.

Fortran references:
- ``engine/source/materials/mat/mat021/m21law.F`` (solid constitutive kernel)
- ``starter/source/materials/mat/mat021/hm_read_mat21.F`` (starter reader & parameter init)
- ``hm_cfg_files/config/CFG/radioss110/MAT/matl21_dprag.cfg`` (CFG attributes & card format)
"""

from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from pyradioss.elements import solid_hexa8, solid_tetra4
from pyradioss.engine import engine
from pyradioss.engine.engine import run_engine, _energies
from pyradioss.materials.law21_dprag import (
    build_law21,
    solid_update_law21,
    sound_speed_solid_law21,
    tangent_law21_solid,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter import starter
from pyradioss.starter.starter import run_starter


# ============================================================================
# Helpers & Factories
# ============================================================================

def _scalar(val: Any) -> float:
    """Safely convert 0-d or 1-d single element array/scalar to float (NumPy 2.x safe)."""
    return float(np.asarray(val).flat[0])


def make_test_material_law21(
    mid: int = 1,
    rho0: float = 2000.0,
    E: float = 2.0e10,
    nu: float = 0.25,
    c1: float = 1.0e9,
    bunl: float = 2.0e9,
    mumax: float = 0.05,
    a0: float = 1.0e10,
    a1: float = 0.5,
    a2: float = 0.0,
    amax: float = 1.0e20,
    pmin: float = -1.0e30,
    pext: float = 0.0,
    **kwargs: Any,
) -> Material:
    """Factory creating a valid /MAT/LAW21 (/MAT/DPRAG) Material instance."""
    params = {
        "E": E,
        "nu": nu,
        "c1": c1,
        "bunl": bunl,
        "mumax": mumax,
        "a0": a0,
        "a1": a1,
        "a2": a2,
        "amax": amax,
        "pmin": pmin,
        "pext": pext,
        "rho0": rho0,
    }
    params.update(kwargs)
    mat = Material(id=mid, law=21, rho0=rho0, title="Rock_LAW21", params=params)
    return mat


class MockProp:
    """Mock solid property mimicking /PROP/SOLID."""
    def __init__(
        self,
        pid: int = 1,
        qa: float = 1.1,
        qb: float = 0.05,
        hm: float = 0.1,
        **kwargs: Any,
    ):
        self.id = pid
        self.params = {
            "qa": qa,
            "qb": qb,
            "hm": hm,
            **kwargs,
        }


class MockGroup:
    """Mock element group for direct element kernel integration."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_single_hexa8(mat: Material, prop: MockProp | None = None) -> tuple[MockGroup, Model]:
    """Create a single unit-cube Hexa8 element group and model."""
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    if prop is None:
        prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    return group, model


def make_single_tetra4(mat: Material, prop: MockProp | None = None) -> tuple[MockGroup, Model]:
    """Create a single canonical Tetra4 element group and model."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    if prop is None:
        prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    model.x = coords.copy()
    group._model = model

    solid_tetra4.init_group(group, model, None)
    return group, model


# ============================================================================
# 1. Multi-Cycle Explicit Dynamic Simulations
# ============================================================================

class TestLaw21Hexa8DynamicSim:
    """Multi-cycle explicit dynamic simulations for Hexa8 solid elements with LAW21."""

    def test_hexa8_cyclic_hydrostatic_compression_and_unloading(self):
        """Hydrostatic multi-cycle simulation: 35 cycles compression, 35 cycles unloading, 35 cycles reloading."""
        c1 = 1.0e9
        bunl = 2.5e9
        mumax = 0.08
        mat = make_test_material_law21(
            rho0=2000.0,
            c1=c1,
            bunl=bunl,
            mumax=mumax,
            a0=1.0e16,  # High yield envelope so response remains purely volumetric
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 5.0e-5

        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # Phase 1: Inward compression (35 cycles)
        vel_comp = -0.05 * (coords0 - centroid)  # volumetric contraction
        p_history_load = []
        mu_history_load = []

        for _ in range(35):
            curr_x += vel_comp * dt
            solid_hexa8.forces(group, curr_x, vel_comp, None, dt, fint, mint)
            sig = group.state["sig"][0]
            p = -(sig[0] + sig[1] + sig[2]) / 3.0
            p_history_load.append(p)
            mu_history_load.append(float(group.state["mat_extra"]["mu"][0]))

            # Force equilibrium on 8-node brick
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)
            # Normal stresses are compressive (< 0)
            assert sig[0] < 0.0 and sig[1] < 0.0 and sig[2] < 0.0

        p_peak = p_history_load[-1]
        mu_peak = mu_history_load[-1]
        mu_bak_after_load = float(group.state["mat_extra"]["mu_bak"][0])
        assert mu_peak > 0.0
        assert math.isclose(mu_bak_after_load, mu_peak, rel_tol=1e-5)
        assert p_peak > 0.0

        # Phase 2: Outward expansion / unloading (35 cycles)
        vel_unload = 0.04 * (coords0 - centroid)
        p_history_unl = []
        mu_history_unl = []

        for _ in range(35):
            curr_x += vel_unload * dt
            solid_hexa8.forces(group, curr_x, vel_unload, None, dt, fint, mint)
            sig = group.state["sig"][0]
            p = -(sig[0] + sig[1] + sig[2]) / 3.0
            p_history_unl.append(p)
            mu_history_unl.append(float(group.state["mat_extra"]["mu"][0]))

            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        # mu_bak must be preserved at peak historical compaction during unloading
        mu_bak_after_unl = float(group.state["mat_extra"]["mu_bak"][0])
        assert math.isclose(mu_bak_after_unl, mu_peak, rel_tol=1e-5)
        # Pressure dropped significantly during unloading
        assert p_history_unl[-1] < p_peak

        # Phase 3: Re-compression / reloading (35 cycles)
        vel_reload = -0.06 * (coords0 - centroid)
        for _ in range(35):
            curr_x += vel_reload * dt
            solid_hexa8.forces(group, curr_x, vel_reload, None, dt, fint, mint)
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        # Final compaction exceeded earlier peak, mu_bak advanced
        mu_final = float(group.state["mat_extra"]["mu"][0])
        mu_bak_final = float(group.state["mat_extra"]["mu_bak"][0])
        assert mu_final > mu_peak
        assert math.isclose(mu_bak_final, min(mumax, mu_final), rel_tol=1e-5)
        assert group.state["eint"][0] > 0.0

    def test_hexa8_cyclic_shear_with_confining_pressure(self):
        """Cyclic shear under confining pressure: verifies Drucker-Prager yield limit tau_max = sqrt(G0(P))."""
        a0 = 2.0e7
        a1 = 0.4
        a2 = 0.0
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            a0=a0,
            a1=a1,
            a2=a2,
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        dt = 1.0e-5
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))
        curr_x = coords0.copy()

        # Step 1: Pre-confining compression (10 cycles)
        centroid = np.array([0.5, 0.5, 0.5])
        vel_conf = -0.02 * (coords0 - centroid)
        for _ in range(10):
            curr_x += vel_conf * dt
            solid_hexa8.forces(group, curr_x, vel_conf, None, dt, fint, mint)

        p_conf = -(group.state["sig"][0, 0] + group.state["sig"][0, 1] + group.state["sig"][0, 2]) / 3.0
        assert p_conf > 0.0
        expected_g0 = a0 + a1 * p_conf
        expected_tau_yield = math.sqrt(expected_g0)

        # Step 2: Forward shear (30 cycles) on top face (z=1, nodes [4, 5, 6, 7])
        vel_shear_fwd = np.zeros_like(coords0)
        vel_shear_fwd[[4, 5, 6, 7], 0] = 60.0  # +x direction

        for _ in range(30):
            curr_x += vel_shear_fwd * dt
            solid_hexa8.forces(group, curr_x, vel_shear_fwd, None, dt, fint, mint)
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        sig = group.state["sig"][0]
        tau_zx = sig[5]
        # Shear stress should be bounded by Drucker-Prager cone
        assert math.isclose(abs(tau_zx), expected_tau_yield, rel_tol=5e-2)
        epsp_fwd = float(group.state["epsp"][0])
        assert epsp_fwd > 0.0

        # Step 3: Reverse shear (30 cycles) on top face
        vel_shear_rev = np.zeros_like(coords0)
        vel_shear_rev[[4, 5, 6, 7], 0] = -60.0  # -x direction

        for _ in range(30):
            curr_x += vel_shear_rev * dt
            solid_hexa8.forces(group, curr_x, vel_shear_rev, None, dt, fint, mint)
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        epsp_rev = float(group.state["epsp"][0])
        # Plastic strain must accumulate across reverse cycles
        assert epsp_rev > epsp_fwd
        assert group.state["eint"][0] > 0.0


# ============================================================================
# 2. Tetra4 Solid Element Dynamic Simulations
# ============================================================================

class TestLaw21Tetra4DynamicSim:
    """Multi-cycle explicit dynamic simulations for Tetra4 elements with LAW21."""

    def test_tetra4_dynamic_compression_and_shear(self):
        """Tetra4 element under 35 cycles compression followed by 35 cycles shear."""
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=2.0e9,
            a0=1.0e8,
            a1=0.3,
            a2=0.0,
        )
        group, model = make_single_tetra4(mat)
        coords0 = model.x0.copy()
        centroid = coords0.mean(axis=0)
        dt = 2.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))

        # Phase 1: Compression (35 cycles)
        vel_comp = -0.04 * (coords0 - centroid)
        dtc_history = []
        for _ in range(35):
            curr_x += vel_comp * dt
            dtc = solid_tetra4.forces(group, curr_x, vel_comp, None, dt, fint, mint)
            dtc_history.append(float(dtc[0]))
            assert dtc[0] > 0.0 and np.isfinite(dtc[0])
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        sig = group.state["sig"][0]
        p = -(sig[0] + sig[1] + sig[2]) / 3.0
        assert p > 0.0
        assert group.state["eint"][0] > 0.0
        assert group.state["mat_extra"]["mu_bak"][0] > 0.0

        # Phase 2: Shear (35 cycles) by moving apex node 3 along x
        vel_shear = np.zeros_like(coords0)
        vel_shear[3, 0] = 40.0
        for _ in range(35):
            curr_x += vel_shear * dt
            dtc = solid_tetra4.forces(group, curr_x, vel_shear, None, dt, fint, mint)
            assert dtc[0] > 0.0 and np.isfinite(dtc[0])
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        assert group.state["epsp"][0] > 0.0
        assert float(group.state["mat_extra"]["epxe"][0]) > 0.0


# ============================================================================
# 3. Multi-Element Patch Simulations
# ============================================================================

class TestLaw21MultiElementPatches:
    """Multi-element patch simulations (2x2x2 Hexa8 patch and 5-element Tetra4 patch)."""

    def test_hexa8_2x2x2_patch_dynamic_compression_and_shear(self):
        """2x2x2 Hexa8 patch (8 brick elements, 27 nodes) under dynamic compression and shear."""
        coords = np.array([
            [x, y, z] for z in range(3) for y in range(3) for x in range(3)
        ], dtype=float)

        def n_idx(i: int, j: int, k: int) -> int:
            return i + 3 * j + 9 * k

        bricks = []
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    bricks.append([
                        n_idx(i, j, k), n_idx(i + 1, j, k),
                        n_idx(i + 1, j + 1, k), n_idx(i, j + 1, k),
                        n_idx(i, j, k + 1), n_idx(i + 1, j, k + 1),
                        n_idx(i + 1, j + 1, k + 1), n_idx(i, j + 1, k + 1),
                    ])
        conn = np.array(bricks, dtype=np.int64)
        assert len(conn) == 8

        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=2.0e9,
            a0=1.0e8,
            a1=0.2,
        )
        prop = MockProp()

        group = MockGroup(conn, slices=[(slice(0, 8), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        model.x = coords.copy()
        group._model = model

        solid_hexa8.init_group(group, model, None)

        dt = 2.0e-5
        curr_x = coords.copy()
        fint = np.zeros((27, 3))
        mint = np.zeros((27, 3))

        # Dynamic compression and shear: affine velocity field across 2x2x2 mesh
        vel = np.zeros_like(coords)
        vel[:, 2] = -10.0 * (coords[:, 2] / 2.0)  # compression along z
        vel[:, 0] = 5.0 * (coords[:, 2] / 2.0)   # shear along x

        for _ in range(40):
            curr_x += vel * dt
            dtc = solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

            assert np.all(dtc > 0.0)
            assert np.all(np.isfinite(dtc))
            # Global force equilibrium across the 2x2x2 mesh
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-4)

        # All 8 elements remain alive and accumulating energy
        assert np.all(group.state["off"] == 1.0)
        assert np.all(group.state["eint"] > 0.0)
        assert np.all(group.state["mat_extra"]["mu_bak"] > 0.0)

    def test_tetra4_5_element_patch_dynamic_shear(self):
        """5-element Tetra4 patch (Kuhn's triangulation of unit cube) under dynamic compression/shear."""
        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ], dtype=float)

        tets = np.array([
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 4, 5, 6],
            [3, 4, 6, 7],
            [1, 3, 4, 6],
        ], dtype=np.int64)

        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=2.0e9,
            a0=2.0e8,
            a1=0.3,
        )
        prop = MockProp()

        group = MockGroup(tets, slices=[(slice(0, 5), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        model.x = coords.copy()
        group._model = model

        solid_tetra4.init_group(group, model, None)

        dt = 1.0e-5
        curr_x = coords.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # Dynamic shear: top face nodes (4, 5, 6, 7) move in +x
        vel = np.zeros_like(coords)
        vel[[4, 5, 6, 7], 0] = 30.0

        for _ in range(40):
            curr_x += vel * dt
            dtc = solid_tetra4.forces(group, curr_x, vel, None, dt, fint, mint)

            assert np.all(dtc > 0.0)
            assert np.all(np.isfinite(dtc))
            assert np.allclose(fint.sum(axis=0), 0.0, atol=1e-5)

        assert np.all(group.state["off"] == 1.0)
        assert np.all(group.state["eint"] > 0.0)


# ============================================================================
# 4. Energy Conservation & Work Balance Auditing
# ============================================================================

class TestLaw21EnergyConservationAndWorkBalance:
    """Auditing strain energy ledger, reversible elastic conservation, and monotonic dissipation."""

    def test_incremental_strain_energy_ledger_matches_external_work(self):
        """Verify Delta E_int = int sigma : depsilon * dV matches trapezoidal work increment."""
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=2.0e9,
            a0=1.0e16,  # High yield envelope so response is elastic
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        dt = 5.0e-5

        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # Multi-axis velocity field producing normal and shear deformation
        vel = np.zeros_like(coords0)
        vel[[1, 2, 5, 6], 0] = 10.0
        vel[[4, 5, 6, 7], 2] = -15.0
        vel[[2, 3, 6, 7], 1] = 8.0

        total_work = 0.0

        for _ in range(25):
            sig_prev = group.state["sig"][0].copy()
            curr_x += vel * dt

            solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

            sig_curr = group.state["sig"][0].copy()
            sig_mid = 0.5 * (sig_prev + sig_curr)

            # Engineering strain increment for unit cube
            d_eps = np.zeros(6)
            d_eps[0] = 10.0 * dt
            d_eps[1] = 8.0 * dt
            d_eps[2] = -15.0 * dt
            vol = float(group.state["vol0"][0])
            dw = vol * np.sum(sig_mid * d_eps)
            total_work += dw

        internal_energy = float(group.state["eint"][0])
        # Verify relative agreement between cumulative trapezoidal work and internal energy ledger
        assert math.isclose(internal_energy, total_work, rel_tol=1e-2)

    def test_reversible_elastic_undamped_vibration_energy_conservation(self):
        """Free vibration of undamped Hexa8 solid: mechanical energy E_tot = E_kin + E_int conserved (|Delta E|/E0 < 1%)."""
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=1.0e9,
            a0=1.0e20,  # Purely elastic
        )
        # qa=0, qb=0, hm=0: zero artificial damping and zero hourglass dissipation
        prop = MockProp(qa=0.0, qb=0.0, hm=0.0)
        group, model = make_single_hexa8(mat, prop)

        coords0 = model.x0.copy()
        vol0 = 1.0
        rho0 = 2000.0
        m_node = rho0 * vol0 / 8.0
        mass = np.full((8, 1), m_node)

        # Symmetric breathing mode in X
        x = coords0.copy()
        v = np.zeros_like(x)
        vx0 = 1.0
        v[[1, 2, 5, 6], 0] = vx0
        v[[0, 3, 4, 7], 0] = -vx0

        e0 = float(0.5 * np.sum(mass * v**2))
        assert e0 > 0.0

        dt = 1.0e-5
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))
        e_tots = []

        # Standard explicit leapfrog initialization: v^{1/2} = v^0 + 0.5 * a^0 * dt
        fint.fill(0.0)
        solid_hexa8.forces(group, x, v, None, 0.0, fint, mint)
        acc = fint / mass
        v = v + 0.5 * acc * dt

        for _ in range(100):
            x = x + v * dt
            fint.fill(0.0)
            solid_hexa8.forces(group, x, v, None, dt, fint, mint)
            acc = fint / mass
            # Centered velocity at full time step t_{n+1}
            v_full = v + 0.5 * acc * dt
            ke = float(0.5 * np.sum(mass * v_full**2))
            ie = float(group.state["eint"][0])
            e_tots.append(ke + ie)
            v = v + acc * dt

        max_dev = max(abs(e - e0) / e0 for e in e_tots)
        assert max_dev < 0.01, f"Energy conservation error {max_dev*100:.3f}% exceeds 1.0% limit"

    def test_tetra4_reversible_elastic_undamped_vibration_energy_conservation(self):
        """Free vibration of undamped Tetra4 solid: |Delta E| / E_0 < 1.0%."""
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=1.0e9,
            a0=1.0e20,
        )
        prop = MockProp(qa=0.0, qb=0.0)
        group, model = make_single_tetra4(mat, prop)

        coords0 = model.x0.copy()
        vol0 = 1.0 / 6.0
        rho0 = 2000.0
        m_node = rho0 * vol0 / 4.0
        mass = np.full((4, 1), m_node)

        x = coords0.copy()
        v = np.zeros_like(x)
        v[1, 0] = 1.0
        v[0, 0] = -1.0
        e0 = float(0.5 * np.sum(mass * v**2))

        dt = 1.0e-5
        fint = np.zeros((4, 3))
        mint = np.zeros((4, 3))
        e_tots = []

        fint.fill(0.0)
        solid_tetra4.forces(group, x, v, None, 0.0, fint, mint)
        acc = fint / mass
        v = v + 0.5 * acc * dt

        for _ in range(100):
            x = x + v * dt
            fint.fill(0.0)
            solid_tetra4.forces(group, x, v, None, dt, fint, mint)
            acc = fint / mass
            v_full = v + 0.5 * acc * dt
            ke = float(0.5 * np.sum(mass * v_full**2))
            ie = float(group.state["eint"][0])
            e_tots.append(ke + ie)
            v = v + acc * dt

        max_dev = max(abs(e - e0) / e0 for e in e_tots)
        assert max_dev < 0.01, f"Tetra4 energy conservation error {max_dev*100:.3f}% exceeds 1.0%"

    def test_monotonic_plastic_dissipation_during_yielding_and_compaction(self):
        """Verify that equivalent plastic strain epxe and compaction mu_bak are monotonically non-decreasing."""
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=2.0e9,
            a0=1.0e7,
            a1=0.2,
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        dt = 2.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        epxe_history = []
        mubak_history = []

        # Alternating compression and shear loading/unloading cycles
        for cycle in range(60):
            factor = math.sin(cycle * 0.3)
            centroid = np.array([0.5, 0.5, 0.5])
            # Radial velocity + shear
            vel = -0.05 * factor * (coords0 - centroid)
            vel[[4, 5, 6, 7], 0] += 40.0 * factor

            curr_x += vel * dt
            solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

            epxe = float(group.state["mat_extra"]["epxe"][0])
            mubak = float(group.state["mat_extra"]["mu_bak"][0])
            epxe_history.append(epxe)
            mubak_history.append(mubak)

        # Monotonicity check: plastic strain never decreases
        for i in range(1, len(epxe_history)):
            assert epxe_history[i] >= epxe_history[i - 1] - 1e-14, \
                f"Plastic strain decreased from {epxe_history[i-1]} to {epxe_history[i]}"

        # Monotonicity check: historical compaction never decreases
        for i in range(1, len(mubak_history)):
            assert mubak_history[i] >= mubak_history[i - 1] - 1e-14, \
                f"Historical compaction decreased from {mubak_history[i-1]} to {mubak_history[i]}"


# ============================================================================
# 5. Physical Soil/Concrete Compaction & Drucker-Prager Cone Behavior
# ============================================================================

class TestLaw21CompactionAndDruckerPragerCone:
    """Audit pressure-dependent shear strength, volumetric compaction, hysteresis, and tensile cutoff."""

    def test_pressure_dependent_shear_strength_confinement_effect(self):
        """Drucker-Prager cone: higher confining pressure increases shear resistance before yielding."""
        a0 = 1.0e7
        a1 = 5.0
        a2 = 0.0

        # Run Case 1: Low confining pressure P1
        mat1 = make_test_material_law21(a0=a0, a1=a1, a2=a2, c1=1.0e9)
        group1, model1 = make_single_hexa8(mat1)
        coords1 = model1.x0.copy()
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 1.0e-5
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # Compress to P1
        vel_c1 = -5.0 * (coords1 - centroid)
        for _ in range(10):
            coords1 += vel_c1 * dt
            solid_hexa8.forces(group1, coords1, vel_c1, None, dt, fint, mint)

        p1 = -(group1.state["sig"][0, 0] + group1.state["sig"][0, 1] + group1.state["sig"][0, 2]) / 3.0
        # Now apply large shear
        vel_s = np.zeros_like(coords1)
        vel_s[[4, 5, 6, 7], 0] = 50.0
        for _ in range(30):
            coords1 += vel_s * dt
            solid_hexa8.forces(group1, coords1, vel_s, None, dt, fint, mint)
        tau_yield_1 = abs(float(group1.state["sig"][0, 5]))

        # Run Case 2: High confining pressure P2
        mat2 = make_test_material_law21(a0=a0, a1=a1, a2=a2, c1=1.0e9)
        group2, model2 = make_single_hexa8(mat2)
        coords2 = model2.x0.copy()

        # Compress to P2
        vel_c2 = -40.0 * (coords2 - centroid)
        for _ in range(10):
            coords2 += vel_c2 * dt
            solid_hexa8.forces(group2, coords2, vel_c2, None, dt, fint, mint)

        p2 = -(group2.state["sig"][0, 0] + group2.state["sig"][0, 1] + group2.state["sig"][0, 2]) / 3.0
        assert p2 > p1

        # Apply same shear
        for _ in range(30):
            coords2 += vel_s * dt
            solid_hexa8.forces(group2, coords2, vel_s, None, dt, fint, mint)
        tau_yield_2 = abs(float(group2.state["sig"][0, 5]))

        # Higher confining pressure MUST produce higher shear yield strength
        assert tau_yield_2 > 1.5 * tau_yield_1
        assert math.isclose(tau_yield_1, math.sqrt(a0 + a1 * p1), rel_tol=0.01)
        assert math.isclose(tau_yield_2, math.sqrt(a0 + a1 * p2), rel_tol=0.01)

    def test_compaction_volume_reduction_and_permanent_plastic_volumetric_strain(self):
        """Verify compaction volume reduction and permanent plastic volumetric strain upon unloading."""
        c1 = 1.0e9
        bunl = 3.0e9
        mumax = 0.0002
        mat = make_test_material_law21(
            rho0=2000.0,
            c1=c1,
            bunl=bunl,
            mumax=mumax,
            a0=1.0e16,  # purely volumetric
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        vol0 = 1.0
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 5.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # 1. Compact element: volume decreases from 1.0
        vel_comp = -0.06 * (coords0 - centroid)
        for _ in range(25):
            curr_x += vel_comp * dt
            solid_hexa8.forces(group, curr_x, vel_comp, None, dt, fint, mint)

        # Volume reduction
        _, vol_compacted = solid_hexa8._geometry(curr_x[group.conn])
        assert vol_compacted[0] < vol0
        mu_peak = float(group.state["mat_extra"]["mu_bak"][0])
        assert mu_peak > 0.0

        # 2. Unload back until pressure drops to zero
        vel_unl = 0.06 * (coords0 - centroid)
        vol_at_zero_p = None
        for _ in range(35):
            curr_x += vel_unl * dt
            solid_hexa8.forces(group, curr_x, vel_unl, None, dt, fint, mint)
            sig = group.state["sig"][0]
            p = -(sig[0] + sig[1] + sig[2]) / 3.0
            _, vol_cur = solid_hexa8._geometry(curr_x[group.conn])
            if p <= 0.0:
                vol_at_zero_p = float(vol_cur[0])
                break

        # Element at zero pressure retains permanent volumetric reduction
        assert vol_at_zero_p is not None
        assert vol_at_zero_p < vol0
        # mu_bak remains at peak compaction
        assert math.isclose(float(group.state["mat_extra"]["mu_bak"][0]), mu_peak, rel_tol=1e-5)

    def test_unloading_hysteresis_and_compaction_energy_dissipation(self):
        """Verify steeper unloading slope K_unload > C1 traces a hysteresis loop dissipating compaction energy."""
        c1 = 1.0e9
        bunl = 4.0e9
        mat = make_test_material_law21(
            rho0=2000.0,
            c1=c1,
            bunl=bunl,
            mumax=0.0002,
            a0=1.0e16,
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 5.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        mu_records = []
        p_records = []

        # Loading phase (30 steps)
        vel_comp = -0.04 * (coords0 - centroid)
        for _ in range(30):
            curr_x += vel_comp * dt
            solid_hexa8.forces(group, curr_x, vel_comp, None, dt, fint, mint)
            sig = group.state["sig"][0]
            p_records.append(-(sig[0] + sig[1] + sig[2]) / 3.0)
            mu_records.append(float(group.state["mat_extra"]["mu"][0]))

        # Unloading phase (30 steps)
        vel_unl = 0.03 * (coords0 - centroid)
        for _ in range(30):
            curr_x += vel_unl * dt
            solid_hexa8.forces(group, curr_x, vel_unl, None, dt, fint, mint)
            sig = group.state["sig"][0]
            p_records.append(-(sig[0] + sig[1] + sig[2]) / 3.0)
            mu_records.append(float(group.state["mat_extra"]["mu"][0]))

        # Calculate enclosed area oint P d(mu) via numerical integration
        mu_arr = np.array(mu_records)
        p_arr = np.array(p_records)

        # Work during loading vs work recovered during unloading
        w_diss = np.trapezoid(p_arr, mu_arr)
        # Clockwise loop in (mu, P) space represents positive dissipated energy
        assert w_diss > 0.0, f"Compaction hysteresis loop must dissipate energy (got {w_diss})"

    def test_tensile_cutoff_triaxial_expansion(self):
        """Under triaxial volumetric expansion, pressure is clamped to P_min and shear strength collapses."""
        pmin = -2.0e7
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            pmin=pmin,
            a0=1.0e10,
            a1=0.5,
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 5.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # Outward expansion velocity (triaxial tension)
        vel_exp = 30.0 * (coords0 - centroid)
        # Combine with shear velocity on top face
        vel_exp[[4, 5, 6, 7], 0] += 50.0

        for _ in range(25):
            curr_x += vel_exp * dt
            solid_hexa8.forces(group, curr_x, vel_exp, None, dt, fint, mint)

        sig = group.state["sig"][0]
        p = -(sig[0] + sig[1] + sig[2]) / 3.0

        # Pressure clamped to Pmin
        assert math.isclose(p, pmin, rel_tol=1e-3)
        # Yield envelope G0 collapsed to zero, wiping shear stress to zero
        g0 = float(group.state["mat_extra"]["g0"][0])
        assert g0 == 0.0
        assert math.isclose(sig[3], 0.0, abs_tol=1e-5)
        assert math.isclose(sig[4], 0.0, abs_tol=1e-5)
        assert math.isclose(sig[5], 0.0, abs_tol=1e-5)


# ============================================================================
# 6. Acoustic Sound Speed & Courant Time-Step Stability
# ============================================================================

class TestLaw21AcousticAndCourantStability:
    """Audit sound speed positivity and Courant time step bounds across dynamic cycles."""

    def test_acoustic_sound_speed_positive_and_finite_across_cycles(self):
        """Verify sound speed c_solid = sqrt(|4/3 G + K_eff| / rho0) remains positive and finite."""
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            bunl=3.0e9,
            mumax=0.05,
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 2.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        c_history = []
        dtc_history = []

        # Run 80 dynamic cycles cycling through compression and expansion
        for cycle in range(80):
            amp = math.sin(cycle * 0.2)
            vel = -0.05 * amp * (coords0 - centroid)
            curr_x += vel * dt
            dtc = solid_hexa8.forces(group, curr_x, vel, None, dt, fint, mint)

            c_val = float(group.state["mat_extra"]["c_solid"][0])
            c_history.append(c_val)
            dtc_history.append(float(dtc[0]))

            assert c_val > 1000.0, f"Sound speed {c_val} fell below physical bound"
            assert np.isfinite(c_val)
            assert dtc[0] > 0.0 and np.isfinite(dtc[0])

        assert min(c_history) > 0.0
        assert min(dtc_history) > 0.0

    def test_courant_time_step_stability_under_dynamic_compaction(self):
        """During compaction, evolving bulk modulus K_eff stiffens, keeping Courant dt safely bounded."""
        c1 = 1.0e9
        bunl = 5.0e9  # 5x stiffening upon compaction
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=c1,
            bunl=bunl,
            mumax=0.0002,
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 2.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        vel_comp = -0.08 * (coords0 - centroid)
        curr_x += vel_comp * dt
        dtc_start = float(solid_hexa8.forces(group, curr_x, vel_comp, None, dt, fint, mint)[0])

        # Compact element to full mumax
        for _ in range(29):
            curr_x += vel_comp * dt
            dtc = solid_hexa8.forces(group, curr_x, vel_comp, None, dt, fint, mint)

        dtc_compacted = float(dtc[0])
        # Stiffened compacted material has higher sound speed, therefore slightly smaller critical dt
        assert dtc_compacted < dtc_start
        assert dtc_compacted > 0.3 * dtc_start  # Remains well-behaved, not collapsing

    def test_sound_speed_under_triaxial_expansion(self):
        """Under triaxial expansion to P_min, sound speed remains positive and does not collapse or diverge."""
        mat = make_test_material_law21(
            rho0=2000.0,
            E=2.0e10,
            nu=0.25,
            c1=1.0e9,
            pmin=-1.0e7,
        )
        group, model = make_single_hexa8(mat)
        coords0 = model.x0.copy()
        centroid = np.array([0.5, 0.5, 0.5])
        dt = 5.0e-5
        curr_x = coords0.copy()
        fint = np.zeros((8, 3))
        mint = np.zeros((8, 3))

        # Pull outward beyond cutoff
        vel_exp = 0.20 * (coords0 - centroid)
        for _ in range(25):
            curr_x += vel_exp * dt
            dtc = solid_hexa8.forces(group, curr_x, vel_exp, None, dt, fint, mint)
            c_val = float(group.state["mat_extra"]["c_solid"][0])
            assert c_val > 0.0 and np.isfinite(c_val)
            assert dtc[0] > 0.0 and np.isfinite(dtc[0])


# ============================================================================
# 7. End-to-End Starter and Engine Dynamic Simulations
# ============================================================================

class TestLaw21StarterEngineEndToEnd:
    """Full Starter and Engine execution verifying energy balance and normal termination."""

    def test_full_engine_run_hexa8_compression(self, tmp_path: Path):
        """Execute full Starter + Engine on Hexa8 cube with LAW21 under dynamic compression."""
        run_name = "HEXA8_LAW21_COMP"
        deck_0000 = f"""/BEGIN
{run_name}_0000
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
Cube_Solid
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW21/1
Soil_DPRAG
2000.0
2.0e10 0.25
1.0e10 0.5 0.0 1.0e20
0 1.0e9 1.0
-1.0e30
2.0e9 0.05
/GRNOD/NODE/1
Bottom_Nodes
1 2 3 4
/GRNOD/NODE/2
Top_Nodes
5 6 7 8
/BCS/1
Fix_Bottom
111 111 0 1
/FUNCT/1
Constant_Vel
0.0 1.0
1.0 1.0
/IMPVEL/1
Push_Z
1 Z 2 -5.0
/END
"""
        deck_0001 = f"""/RUN/{run_name}/1
5.0e-3
/DT
0.67 0
/PRINT/-100
/END
"""
        f0 = tmp_path / f"{run_name}_0000.rad"
        f1 = tmp_path / f"{run_name}_0001.rad"
        f0.write_text(deck_0000, encoding="ascii")
        f1.write_text(deck_0001, encoding="ascii")

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = starter.run_starter(str(f0))
            assert st_model.bricks.n == 1
            eng_model = engine.run_engine(str(f1))

        assert eng_model is not None
        state = eng_model.engine_state
        assert state.cycle >= 20
        assert eng_model.bricks.state["off"][0] == 1.0
        assert eng_model.bricks.state["eint"][0] > 0.0

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy balance error {en['ERR']}% exceeds 1.0%"

    def test_full_engine_run_hexa8_cyclic_shear(self, tmp_path: Path):
        """Execute full Starter + Engine on Hexa8 cube with LAW21 under dynamic shear."""
        run_name = "HEXA8_LAW21_SHEAR"
        deck_0000 = f"""/BEGIN
{run_name}_0000
/NODE
1 0.0 0.0 0.0
2 1.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 0.0 0.0 1.0
6 1.0 0.0 1.0
7 1.0 1.0 1.0
8 0.0 1.0 1.0
/BRICK/1
1 1 2 3 4 5 6 7 8
/PART/1
Cube_Solid
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW21/1
Soil_DPRAG
2000.0
2.0e10 0.25
5.0e8 0.4 0.0 1.0e20
0 1.0e9 1.0
-1.0e30
2.0e9 0.05
/GRNOD/NODE/1
Bottom_Nodes
1 2 3 4
/GRNOD/NODE/2
Top_Nodes
5 6 7 8
/BCS/1
Fix_Bottom
111 111 0 1
/FUNCT/1
Shear_Vel
0.0 1.0
1.0 1.0
/IMPVEL/1
Shear_X
1 X 2 10.0
/END
"""
        deck_0001 = f"""/RUN/{run_name}/1
5.0e-3
/DT
0.67 0
/PRINT/-100
/END
"""
        f0 = tmp_path / f"{run_name}_0000.rad"
        f1 = tmp_path / f"{run_name}_0001.rad"
        f0.write_text(deck_0000, encoding="ascii")
        f1.write_text(deck_0001, encoding="ascii")

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = starter.run_starter(str(f0))
            assert st_model.bricks.n == 1
            eng_model = engine.run_engine(str(f1))

        assert eng_model is not None
        state = eng_model.engine_state
        assert state.cycle >= 20
        assert eng_model.bricks.state["eint"][0] > 0.0

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy balance error {en['ERR']}% exceeds 1.0%"

    def test_full_engine_run_patch_2x2x2(self, tmp_path: Path):
        """Execute full Starter + Engine on 2x2x2 Hexa8 patch (8 elements, 27 nodes)."""
        run_name = "PATCH_LAW21_RUN"
        nodes_str = "\n".join([
            f"{i + 1} {x:.1f} {y:.1f} {z:.1f}"
            for i, (x, y, z) in enumerate(
                [(x, y, z) for z in range(3) for y in range(3) for x in range(3)]
            )
        ])

        def n_idx(i: int, j: int, k: int) -> int:
            return i + 3 * j + 9 * k + 1

        bricks = []
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    eid = len(bricks) + 1
                    bricks.append(
                        f"{eid} {n_idx(i, j, k)} {n_idx(i+1, j, k)} "
                        f"{n_idx(i+1, j+1, k)} {n_idx(i, j+1, k)} "
                        f"{n_idx(i, j, k+1)} {n_idx(i+1, j, k+1)} "
                        f"{n_idx(i+1, j+1, k+1)} {n_idx(i, j+1, k+1)}"
                    )
        bricks_str = "\n".join(bricks)

        deck_0000 = f"""/BEGIN
{run_name}_0000
/NODE
{nodes_str}
/BRICK/1
{bricks_str}
/PART/1
Patch_Solid
1 1
/PROP/SOLID/1
Solid_Prop
1.1 0.05 0.1
/MAT/LAW21/1
Soil_DPRAG
2000.0
2.0e10 0.25
1.0e10 0.5 0.0 1.0e20
0 1.0e9 1.0
-1.0e30
2.0e9 0.05
/GRNOD/NODE/1
Bottom_Nodes
1 2 3 4 5 6 7 8 9
/GRNOD/NODE/2
Top_Nodes
19 20 21 22 23 24 25 26 27
/BCS/1
Fix_Bottom
111 111 0 1
/FUNCT/1
Constant_Vel
0.0 1.0
1.0 1.0
/IMPVEL/1
Push_Z
1 Z 2 -5.0
/END
"""
        deck_0001 = f"""/RUN/{run_name}/1
5.0e-3
/DT
0.67 0
/PRINT/-100
/END
"""
        f0 = tmp_path / f"{run_name}_0000.rad"
        f1 = tmp_path / f"{run_name}_0001.rad"
        f0.write_text(deck_0000, encoding="ascii")
        f1.write_text(deck_0001, encoding="ascii")

        with contextlib.redirect_stdout(io.StringIO()):
            st_model = starter.run_starter(str(f0))
            assert st_model.bricks.n == 8
            eng_model = engine.run_engine(str(f1))

        assert eng_model is not None
        state = eng_model.engine_state
        assert state.cycle >= 20
        assert np.all(eng_model.bricks.state["off"] == 1.0)
        assert np.all(eng_model.bricks.state["eint"] > 0.0)

        en = _energies(eng_model, state)
        assert abs(en["ERR"]) < 1.0, f"Energy balance error {en['ERR']}% exceeds 1.0%"
