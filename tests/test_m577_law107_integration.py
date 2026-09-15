"""Central materials integration and registry tests for /MAT/LAW107 (M577).

Validates:
  - Central registration in MAT_PHYSICS_REGISTRY.
  - Central dispatch through pyradioss.materials.solid_update.
  - Central dispatch through pyradioss.materials.shell_update.
  - Central dispatch through pyradioss.materials.sound_speed.
  - Central dispatch through pyradioss.materials.solid_tangent.
  - Central dispatch through pyradioss.materials.consistent_shell_tangent.
  - Central dispatch through pyradioss.materials.extra_shapes.
  - Central dispatch through pyradioss.materials.needs_env.
  - Central dispatch through pyradioss.materials.resolve_curves for tabulated curves.
"""

import numpy as np
import pytest

from pyradioss.input.mat_reader import MAT_PHYSICS_REGISTRY
from pyradioss.materials import (
    build_law107,
    consistent_shell_tangent,
    extra_shapes,
    needs_env,
    resolve_curves,
    shell_update,
    solid_tangent,
    solid_update,
    sound_speed,
)
from pyradioss.common.tables import FunctTable
from pyradioss.model.entities import MaterialLaw107
from pyradioss.model.model import Model


def _make_sample_mat(law_id: int = 107, law_name: str = "LAW107", itab: int = 0) -> MaterialLaw107:
    m = MaterialLaw107(
        id=law_id,
        title="IntegrationMat107",
        rho0=7.5e-7,
        rhor=7.5e-7,
        e1=7000.0,
        e2=3500.0,
        e3=100.0,
        nu21=0.15,
        g12=1800.0,
        g23=40.0,
        g13=40.0,
        ires=2,
        itab=itab,
        ismooth=1,
        xi1=0.5,
        xi2=0.5,
        g1c=1.0,
        d1=0.1,
        d2=0.2,
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
        cini1c=12.0,
        s1c=2.5,
        sigy2c=20.0,
        cini2c=10.0,
        s2c=2.0,
        sigyt=10.0,
        cinit=5.0,
        st=1.0,
        tab_yld1=101 if itab != 0 else 0,
        tab_yld2=102 if itab != 0 else 0,
        tab_yld1c=103 if itab != 0 else 0,
        tab_yld2c=104 if itab != 0 else 0,
        tab_yldt=105 if itab != 0 else 0,
    )
    m.law = 107
    m.law_name = law_name
    m.params = {
        "MAT_RHO": 7.5e-7,
        "MAT_E1": 7000.0,
        "MAT_E2": 3500.0,
        "MAT_E3": 100.0,
        "MAT_NU21": 0.15,
        "MAT_G12": 1800.0,
        "MAT_G23": 40.0,
        "MAT_G13": 40.0,
        "MAT_IRES": 2,
        "MAT_ITAB": itab,
        "MAT_SMOOTH": 1,
        "MAT_XI1": 0.5,
        "MAT_XI2": 0.5,
        "MAT_G1C": 1.0,
        "MAT_D1": 0.1,
        "MAT_D2": 0.2,
        "MAT_K1": 0.1,
        "MAT_K2": -0.1,
        "MAT_K3": 0.2,
        "MAT_K4": 0.3,
        "MAT_K5": 0.4,
        "MAT_K6": 0.5,
        "MAT_SIGY1": 25.0,
        "MAT_CINI1": 10.0,
        "MAT_S1": 2.0,
        "MAT_SIGY2": 15.0,
        "MAT_CINI2": 8.0,
        "MAT_S2": 1.5,
        "MAT_SIGY1C": 30.0,
        "MAT_CINI1C": 12.0,
        "MAT_S1C": 2.5,
        "MAT_SIGY2C": 20.0,
        "MAT_CINI2C": 10.0,
        "MAT_S2C": 2.0,
        "MAT_SIGYT": 10.0,
        "MAT_CINIT": 5.0,
        "MAT_ST": 1.0,
        "TAB_YLD1": 101 if itab != 0 else 0,
        "TAB_YLD2": 102 if itab != 0 else 0,
        "TAB_YLD1C": 103 if itab != 0 else 0,
        "TAB_YLD2C": 104 if itab != 0 else 0,
        "TAB_YLDT": 105 if itab != 0 else 0,
    }
    return m


def test_law107_mat_physics_registry():
    """Verify all LAW107 aliases are properly wired in MAT_PHYSICS_REGISTRY."""
    expected_keys = [
        107, "107", "LAW107", "PAPER_LIGHT", "PLAS_PAPER_LIGHT",
        "LAW107_PAPER_LIGHT", "PFEIFFER", "MAT_PFEIFFER",
        "MAT_PAPER_LIGHT", "MAT_PLAS_PAPER_LIGHT", "MAT_107", "MAT_LAW107"
    ]
    for key in expected_keys:
        assert key in MAT_PHYSICS_REGISTRY, f"Key {key!r} missing in MAT_PHYSICS_REGISTRY"
        builder = MAT_PHYSICS_REGISTRY[key]
        assert callable(builder)


def test_law107_central_solid_update():
    """Verify pyradioss.materials.solid_update dispatches correctly to law107."""
    m = _make_sample_mat()
    sig = np.zeros(6, dtype=np.float64)
    # Small strain increment within elastic limit
    deps = np.array([1.0e-4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {"uvar": np.zeros(1), "pla": np.zeros(6), "epsd": np.zeros(6)}

    sig_out, epsp_out, c = solid_update(m, sig, deps, dt=1.0e-6, extra=extra)

    assert c > 0.0
    assert sig_out[0] > 0.0  # Tension in direction 1
    assert epsp_out == pytest.approx(0.0)  # Still elastic


def test_law107_central_shell_update():
    """Verify pyradioss.materials.shell_update dispatches correctly to law107."""
    m = _make_sample_mat(law_name="PAPER_LIGHT")
    sig = np.zeros(5, dtype=np.float64)
    deps = np.array([2.0e-4, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    extra = {"uvar": np.zeros(1), "pla": np.zeros(6), "epsd": np.zeros(6)}

    sig_out, epsp_out = shell_update(m, sig, deps, dt=1.0e-6, extra=extra)

    assert sig_out[0] > 0.0
    assert len(sig_out) == 5


def test_law107_central_sound_speed():
    """Verify pyradioss.materials.sound_speed dispatches correctly."""
    m = _make_sample_mat(law_name="PLAS_PAPER_LIGHT")
    c_val = sound_speed(m)
    assert c_val > 0.0
    assert np.isfinite(c_val)


def test_law107_central_solid_and_shell_tangent():
    """Verify pyradioss.materials tangent dispatches correctly."""
    m = _make_sample_mat()
    c_sol = solid_tangent(m)
    assert c_sol.shape == (6, 6)
    assert c_sol[0, 0] > 0.0
    assert c_sol[1, 1] > 0.0

    c_sh = consistent_shell_tangent(m)
    assert c_sh.shape == (3, 3)
    assert c_sh[0, 0] > 0.0


def test_law107_central_extra_shapes_and_needs_env():
    """Verify extra_shapes and needs_env return expected metadata."""
    m = _make_sample_mat()
    shapes_none = extra_shapes(m, nip=None)
    assert "uvar" in shapes_none
    assert "pla" in shapes_none
    assert "epsd" in shapes_none
    assert shapes_none["uvar"] == (1,)
    assert shapes_none["pla"] == (6,)

    shapes_5 = extra_shapes(m, nip=5)
    assert shapes_5["uvar"] == (5, 1)
    assert shapes_5["pla"] == (5, 6)

    assert needs_env(m) is True


def test_law107_central_resolve_curves():
    """Verify resolve_curves hooks function curve references."""
    m = _make_sample_mat(itab=1)
    model = Model()
    c101 = FunctTable(101, [0.0, 0.01, 0.05], [25.0, 30.0, 35.0], title="Yld1")
    c102 = FunctTable(102, [0.0, 0.01, 0.05], [15.0, 18.0, 22.0], title="Yld2")
    c103 = FunctTable(103, [0.0, 0.01, 0.05], [30.0, 35.0, 40.0], title="Yld1c")
    c104 = FunctTable(104, [0.0, 0.01, 0.05], [20.0, 24.0, 28.0], title="Yld2c")
    c105 = FunctTable(105, [0.0, 0.01, 0.05], [10.0, 12.0, 15.0], title="Yldt")
    model.functions[101] = c101
    model.functions[102] = c102
    model.functions[103] = c103
    model.functions[104] = c104
    model.functions[105] = c105

    resolve_curves(m, model)

    assert m.params.get("curve_yld1") is c101
    assert m.params.get("curve_yld2") is c102
    assert m.params.get("curve_yld1c") is c103
    assert m.params.get("curve_yld2c") is c104
    assert m.params.get("curve_yldt") is c105
