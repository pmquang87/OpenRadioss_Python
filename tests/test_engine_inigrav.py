"""
Tests for /INIGRAV geostatic gravity stress & pore pressure initialization.

Ported from OpenRadioss Fortran:
- starter/source/initial_conditions/inigrav/hm_read_inigrav.F
- starter/source/initial_conditions/inigrav/inigrav_load.F
- starter/source/initial_conditions/inigrav/inigrav_m37.F
- starter/source/initial_conditions/inigrav/inigrav_m51.F
- starter/source/initial_conditions/inigrav/inigrav_eos.F
"""

import math
from types import SimpleNamespace
import numpy as np
import pytest

from pyradioss.engine.inigrav import (
    IniGravParams,
    SoilLayer,
    GeostaticStressResult,
    compute_geostatic_stress,
    compute_depth_along_gravity,
    apply_inigrav,
    check_geostatic_equilibrium,
    k0_from_phi,
    k0_from_nu,
    build_inigrav,
)
from pyradioss.model.entities import IniGrav


def test_vertical_stress_linear_with_depth():
    """Verify vertical stress increases linearly with depth: sigma_v = rho * g * h."""
    rho = 2000.0  # kg/m^3
    g = 9.81      # m/s^2
    z_surf = 10.0 # ground surface elevation

    # Test column of element centroids at depths h = 0, 2, 5, 10 meters
    z_coords = np.array([10.0, 8.0, 5.0, 0.0])
    centroids = np.zeros((len(z_coords), 3))
    centroids[:, 2] = z_coords

    params = IniGravParams(
        z_surface=z_surf,
        gravity=(0.0, 0.0, -g),
        k0=0.5,
        pref=0.0,
    )

    res = compute_geostatic_stress(centroids, rho, params)

    expected_depths = np.array([0.0, 2.0, 5.0, 10.0])
    np.testing.assert_allclose(res.depth, expected_depths, atol=1e-8)

    expected_sigma_v = rho * g * expected_depths
    np.testing.assert_allclose(res.sigma_v, expected_sigma_v, rtol=1e-6)

    # Radioss Cauchy stress convention: compression is negative
    np.testing.assert_allclose(res.sigma_total[:, 2], -expected_sigma_v, rtol=1e-6)


def test_lateral_stress_ratio_k0():
    """Verify lateral stress ratio K0 = sigma_h / sigma_v."""
    rho = 1800.0
    g = 9.81
    k0_val = 0.45

    centroids = np.array([
        [0.0, 0.0, -2.0],
        [0.0, 0.0, -5.0],
        [0.0, 0.0, -10.0],
    ])

    params = IniGravParams(
        z_surface=0.0,
        gravity=(0.0, 0.0, -g),
        k0=k0_val,
    )

    res = compute_geostatic_stress(centroids, rho, params)

    # Check horizontal stress equals K0 * sigma_v
    np.testing.assert_allclose(res.sigma_hx, k0_val * res.sigma_v, rtol=1e-6)
    np.testing.assert_allclose(res.sigma_hy, k0_val * res.sigma_v, rtol=1e-6)

    # Check ratio explicitly
    ratio_x = res.sigma_hx / res.sigma_v
    ratio_y = res.sigma_hy / res.sigma_v
    np.testing.assert_allclose(ratio_x, k0_val, rtol=1e-6)
    np.testing.assert_allclose(ratio_y, k0_val, rtol=1e-6)

    # Check Radioss tensor components
    np.testing.assert_allclose(res.sigma_total[:, 0], -k0_val * res.sigma_v, rtol=1e-6)
    np.testing.assert_allclose(res.sigma_total[:, 1], -k0_val * res.sigma_v, rtol=1e-6)
    np.testing.assert_allclose(res.sigma_total[:, 3:], 0.0, atol=1e-12)


def test_anisotropic_lateral_ratios():
    """Verify directional lateral ratios K0_x != K0_y."""
    rho = 2200.0
    params = IniGravParams(
        z_surface=0.0,
        gravity=(0.0, 0.0, -9.81),
        k0_x=0.4,
        k0_y=0.6,
    )
    centroids = np.array([[0.0, 0.0, -5.0]])
    res = compute_geostatic_stress(centroids, rho, params)

    assert abs(res.sigma_hx - 0.4 * res.sigma_v) < 1e-4
    assert abs(res.sigma_hy - 0.6 * res.sigma_v) < 1e-4
    assert abs(res.sigma_total[0, 0] - (-0.4 * res.sigma_v)) < 1e-4
    assert abs(res.sigma_total[0, 1] - (-0.6 * res.sigma_v)) < 1e-4


def test_hydrostatic_pore_pressure_below_water_table():
    """Verify hydrostatic pore water pressure u_w and Terzaghi effective stress."""
    rho_bulk = 2000.0  # Total saturated soil density
    rho_w = 1000.0     # Water density
    g = 9.81
    z_surf = 10.0      # Ground surface at z = 10 m
    z_water = 6.0      # Water table at z = 6 m (4 m depth below surface)

    # 3 points:
    # 1. Above water table: z = 8.0 m (h = 2 m, h_w = 0)
    # 2. At water table: z = 6.0 m (h = 4 m, h_w = 0)
    # 3. Below water table: z = 2.0 m (h = 8 m, h_w = 4 m)
    centroids = np.array([
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 6.0],
        [0.0, 0.0, 2.0],
    ])

    params = IniGravParams(
        z_surface=z_surf,
        z_water=z_water,
        rho_w=rho_w,
        gravity=(0.0, 0.0, -g),
        k0=0.5,
        use_effective_stress_k0=True,
    )

    res = compute_geostatic_stress(centroids, rho_bulk, params)

    # Depths below ground surface
    np.testing.assert_allclose(res.depth, [2.0, 4.0, 8.0], atol=1e-8)

    # Total vertical stresses: sigma_v = rho_bulk * g * h
    expected_sigma_v = rho_bulk * g * np.array([2.0, 4.0, 8.0])
    np.testing.assert_allclose(res.sigma_v, expected_sigma_v, rtol=1e-6)

    # Pore water pressures: u_w = max(0, rho_w * g * (z_water - z))
    expected_uw = np.array([0.0, 0.0, rho_w * g * 4.0])
    np.testing.assert_allclose(res.pore_pressure, expected_uw, rtol=1e-6)

    # Effective vertical stresses: sigma'_v = sigma_v - u_w
    expected_sigma_v_eff = expected_sigma_v - expected_uw
    np.testing.assert_allclose(res.sigma_v_eff, expected_sigma_v_eff, rtol=1e-6)

    # Terzaghi effective lateral stress: sigma'_h = K0 * sigma'_v
    # Total lateral stress: sigma_h = sigma'_h + u_w
    expected_sigma_h_eff = 0.5 * expected_sigma_v_eff
    expected_sigma_h = expected_sigma_h_eff + expected_uw
    np.testing.assert_allclose(res.sigma_hx_eff, expected_sigma_h_eff, rtol=1e-6)
    np.testing.assert_allclose(res.sigma_hx, expected_sigma_h, rtol=1e-6)


def test_geostatic_equilibrium_stress_divergence():
    """Test equilibrium: internal vertical stress divergence d(sigma_v)/dz balances rho * g."""
    rho = 2100.0
    g = 9.81
    z_surf = 20.0

    # 10 vertical stations through soil deposit
    z_stations = np.linspace(0.0, 20.0, 21)
    centroids = np.zeros((len(z_stations), 3))
    centroids[:, 2] = z_stations

    params = IniGravParams(
        z_surface=z_surf,
        gravity=(0.0, 0.0, -g),
        k0=0.5,
    )

    res = compute_geostatic_stress(centroids, rho, params)

    # Check equilibrium via finite difference divergence balance
    is_balanced, max_res = check_geostatic_equilibrium(
        z_coords=res.centroids[:, 2],
        sigma_v=res.sigma_v,
        densities=rho,
        gravity=g,
        tolerance=1e-6,
    )
    assert is_balanced
    assert max_res < 1e-6


def test_multi_layer_soil_stratigraphy():
    """Verify overburden integration across multiple soil layers with different densities."""
    g = 9.81
    # Layer 1: Sand from z = 20 to 15 m (thick = 5 m), density = 1800 kg/m^3
    # Layer 2: Clay from z = 15 to 8 m (thick = 7 m), density = 2000 kg/m^3
    # Layer 3: Bedrock from z = 8 to 0 m (thick = 8 m), density = 2500 kg/m^3
    layers = [
        SoilLayer(part_id=1, name="Sand", z_top=20.0, z_bottom=15.0, density=1800.0),
        SoilLayer(part_id=2, name="Clay", z_top=15.0, z_bottom=8.0, density=2000.0),
        SoilLayer(part_id=3, name="Rock", z_top=8.0, z_bottom=0.0, density=2500.0),
    ]

    centroids = np.array([
        [0.0, 0.0, 18.0],  # Mid Layer 1: 2 m into Sand
        [0.0, 0.0, 15.0],  # Interface 1/2: 5 m Sand
        [0.0, 0.0, 10.0],  # Inside Layer 2: 5 m Sand + 5 m Clay
        [0.0, 0.0, 4.0],   # Inside Layer 3: 5 m Sand + 7 m Clay + 4 m Rock
    ])

    params = IniGravParams(
        z_surface=20.0,
        gravity=(0.0, 0.0, -g),
        layers=layers,
        k0=0.5,
    )

    res = compute_geostatic_stress(centroids, densities=2000.0, params=params)

    # Expected vertical stresses:
    # Point 1 (z=18): 1800 * 9.81 * 2 = 35316 Pa
    exp_1 = 1800.0 * g * 2.0
    # Point 2 (z=15): 1800 * 9.81 * 5 = 88290 Pa
    exp_2 = 1800.0 * g * 5.0
    # Point 3 (z=10): 1800 * 9.81 * 5 + 2000 * 9.81 * 5 = 186390 Pa
    exp_3 = 1800.0 * g * 5.0 + 2000.0 * g * 5.0
    # Point 4 (z=4): 1800 * 9.81 * 5 + 2000 * 9.81 * 7 + 2500 * 9.81 * 4 = 88290 + 137340 + 98100 = 323730 Pa
    exp_4 = 1800.0 * g * 5.0 + 2000.0 * g * 7.0 + 2500.0 * g * 4.0

    expected = np.array([exp_1, exp_2, exp_3, exp_4])
    np.testing.assert_allclose(res.sigma_v, expected, rtol=1e-5)


def test_k0_formulae_jaky_and_elastic():
    """Test Jaky's K0 = 1 - sin(phi) and elastic K0 = nu / (1 - nu)."""
    # Jaky formula
    phi = 30.0  # sin(30 deg) = 0.5 => K0 = 1 - 0.5 = 0.5
    assert abs(k0_from_phi(phi) - 0.5) < 1e-6

    phi45 = 45.0
    assert abs(k0_from_phi(phi45) - (1.0 - math.sqrt(2.0) / 2.0)) < 1e-6

    # Elastic formula
    nu = 0.25  # nu / (1 - nu) = 0.25 / 0.75 = 1/3
    assert abs(k0_from_nu(nu) - 1.0 / 3.0) < 1e-6

    nu3 = 0.3  # 0.3 / 0.7
    assert abs(k0_from_nu(nu3) - 3.0 / 7.0) < 1e-6

    # Automated derivation inside IniGravParams
    p_phi = IniGravParams(phi=30.0)
    assert abs(p_phi.k0 - 0.5) < 1e-6

    p_nu = IniGravParams(nu=0.25)
    assert abs(p_nu.k0 - 1.0 / 3.0) < 1e-6


def test_arbitrary_gravity_direction_and_surface_projection():
    """Test 3D projection matching inigrav_load.F with inclined surface and gravity direction."""
    # Let gravity be along negative Y: g = (0, -9.81, 0)
    # Let ground surface be horizontal in X-Z at y = 15 m: B = (0, 15, 0), normal N = (0, 1, 0)
    centroids = np.array([
        [0.0, 10.0, 0.0],  # 5 m depth along Y
        [0.0, 5.0, 0.0],   # 10 m depth along Y
    ])

    params = IniGravParams(
        gravity=(0.0, -9.81, 0.0),
        surface_point=(0.0, 15.0, 0.0),
        surface_normal=(0.0, 1.0, 0.0),
        k0=0.5,
    )

    res = compute_geostatic_stress(centroids, densities=2000.0, params=params)

    np.testing.assert_allclose(res.depth, [5.0, 10.0], atol=1e-6)
    expected_sig_v = 2000.0 * 9.81 * np.array([5.0, 10.0])
    np.testing.assert_allclose(res.sigma_v, expected_sig_v, rtol=1e-5)


def test_build_inigrav_from_card_and_entity():
    """Verify build_inigrav converts cards, dicts, and IniGrav entities."""
    # From IniGrav dataclass
    ent = IniGrav(
        id=5,
        title="SoilIni",
        grpart_id=10,
        surf_id=20,
        grav_id=30,
        pref=101325.0,
        bx=0.0,
        by=0.0,
        bz=12.5,
    )
    p = build_inigrav(ent)
    assert p.id == 5
    assert p.title == "SoilIni"
    assert p.pref == 101325.0
    assert p.z_surface == 12.5
    assert p.surface_point == (0.0, 0.0, 12.5)

    # From dictionary
    d = {
        "id": 2,
        "pref": 5000.0,
        "z_surface": 8.0,
        "phi": 30.0,
        "z_water": 4.0,
    }
    p2 = build_inigrav(d)
    assert p2.id == 2
    assert p2.pref == 5000.0
    assert p2.z_surface == 8.0
    assert abs(p2.k0 - 0.5) < 1e-6
    assert p2.z_water == 4.0


def test_apply_inigrav_to_model():
    """Verify apply_inigrav populates solid elements initial_stress."""
    elem1 = SimpleNamespace(id=1, node_ids=[1, 2, 3, 4, 5, 6, 7, 8], mat_id=1)
    elem2 = SimpleNamespace(id=2, node_ids=[1, 2, 3, 4, 5, 6, 7, 8], mat_id=1)
    model = SimpleNamespace(
        solid_elements={1: elem1, 2: elem2},
    )

    centroids = np.array([
        [0.0, 0.0, -2.0],
        [0.0, 0.0, -4.0],
    ])
    params = IniGravParams(z_surface=0.0, k0=0.5)

    res = apply_inigrav(model, params=params, elem_centroids=centroids, elem_densities=2000.0)

    assert hasattr(elem1, "initial_stress")
    assert hasattr(elem2, "initial_stress")
    assert elem1.initial_stress.shape == (6,)
    assert elem1.initial_stress[2] < 0.0  # Compression negative
    assert elem2.initial_stress[2] < elem1.initial_stress[2]  # Deeper = more compressive
