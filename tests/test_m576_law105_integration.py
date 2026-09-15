"""Central materials integration and registry tests for /MAT/LAW105 (M576).

Validates:
  - Central registration in MAT_PHYSICS_REGISTRY.
  - Central dispatch through pyradioss.materials.solid_update.
  - Central dispatch through pyradioss.materials.sound_speed.
  - Central dispatch through pyradioss.materials.solid_tangent.
  - Central dispatch through pyradioss.materials.resolve_curves for /FUNCT curves.
"""

import numpy as np
import pytest

from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.materials import (
    build_law105,
    resolve_curves,
    solid_tangent,
    solid_update,
    sound_speed,
)
from pyradioss.common.tables import FunctTable
from pyradioss.model.entities import MaterialLaw105
from pyradioss.model.model import Model


def _make_sample_mat(law_id: int = 105, law_name: str = "LAW105") -> MaterialLaw105:
    m = MaterialLaw105(
        id=law_id,
        title="IntegrationMat",
        rho0=1.65e-6,
        bulk=14000.0,
        p0=0.2,
        psh=0.0,
        gas_d=0.5e-6,
        gas_eg=4200.0,
        gr=0.07,
        c=0.8,
        alpha=1.0,
        c1=2200.0,
        func_b=10,
    )
    m.law = law_id
    m.law_name = law_name
    return m


def test_law105_mat_physics_registry():
    """Verify all LAW105 aliases are properly wired in MAT_PHYSICS_REGISTRY."""
    expected_keys = [
        105, "105", "LAW105", "POWDER_BURN", "POWDERBURN",
        "MAT_POWDER_BURN", "MAT_POWDERBURN", "MAT_105", "MAT_LAW105", "LAW105_POWDER_BURN",
    ]
    for key in expected_keys:
        assert key in MAT_PHYSICS_REGISTRY, f"Key {key!r} missing in MAT_PHYSICS_REGISTRY"
        builder = MAT_PHYSICS_REGISTRY[key]
        assert callable(builder)


def test_law105_central_solid_update():
    """Verify pyradioss.materials.solid_update dispatches correctly to law105."""
    m = _make_sample_mat()
    sig = np.zeros(6, dtype=np.float64)
    deps = np.zeros(6, dtype=np.float64)
    extra = {"initial_step": True, "volume": 50.0}

    sig_out, epsp_out, c = solid_update(m, sig, deps, dt=0.0, extra=extra)

    assert c == pytest.approx((14000.0 / 1.65e-6) ** 0.5)
    assert np.allclose(sig_out[:3], -0.2)
    assert np.allclose(sig_out[3:], 0.0)


def test_law105_central_sound_speed():
    """Verify pyradioss.materials.sound_speed dispatches correctly."""
    m = _make_sample_mat(law_name="POWDER_BURN")
    c_val = sound_speed(m)
    assert c_val == pytest.approx((14000.0 / 1.65e-6) ** 0.5)


def test_law105_central_solid_tangent():
    """Verify pyradioss.materials.solid_tangent dispatches correctly."""
    m = _make_sample_mat()
    c_tan = solid_tangent(m)
    assert c_tan.shape == (6, 6)
    assert c_tan[0, 0] == pytest.approx(14000.0)
    assert c_tan[3, 3] == 0.0


def test_law105_central_resolve_curves():
    """Verify resolve_curves hooks function curve references."""
    m = _make_sample_mat()
    model = Model()
    curve_10 = FunctTable(10, [0.0, 10.0, 100.0], [0.0, 5.0, 25.0], title="BurnRateP")
    model.functions[10] = curve_10

    resolve_curves(m, model)

    assert m.params.get("curve_b") is curve_10 or getattr(m, "curve_b", None) is curve_10
