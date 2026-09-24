"""
Unit tests for pyradioss user-defined subroutine framework.

Tests:
1. UserMaterial ABC enforces solid_update and shell_update abstract methods.
2. Example elastic user material produces correct stress (compared to law01_elastic).
3. Registration and lookup of user materials, user springs, user sensors, and user outputs.
4. Dynamic module loading via load_user_library and auto_discover_user_modules.
5. User material, spring, and sensor stubs raise NotImplementedError with helpful guidance.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyradioss.materials import law01_elastic
from pyradioss.user import (
    MaterialState,
    SpringState,
    UserMaterial,
    UserMaterialLaw29,
    UserMaterialLaw99,
    UserOutput,
    UserSensor,
    UserSensorType29,
    UserSensorType30,
    UserSensorType31,
    UserSpring,
    UserSpringType29,
    UserSpringType30,
    UserSpringType31,
    auto_discover_user_modules,
    clear_user_outputs,
    get_active_user_outputs,
    get_user_material,
    get_user_output,
    get_user_sensor,
    get_user_spring,
    load_user_library,
    register_user_material,
    register_user_output,
    register_user_sensor,
    register_user_spring,
    unregister_user_material,
    unregister_user_sensor,
    unregister_user_spring,
)
from pyradioss.user.examples.user_mat_elastic import UserElasticMaterial


# ----------------------------------------------------------------------------
# 1. ABC Enforcement Tests
# ----------------------------------------------------------------------------


def test_user_material_abc_enforcement():
    """UserMaterial cannot be instantiated without solid_update and shell_update."""

    class IncompleteMaterial(UserMaterial):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteMaterial()  # type: ignore

    class OnlySolidMaterial(UserMaterial):
        def solid_update(self, state, dt):
            return state.sig

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        OnlySolidMaterial()  # type: ignore

    class ConcreteMaterial(UserMaterial):
        def solid_update(self, state, dt):
            return state.sig

        def shell_update(self, state, dt):
            return state.sig

    mat = ConcreteMaterial()
    assert isinstance(mat, UserMaterial)


def test_user_spring_abc_enforcement():
    """UserSpring cannot be instantiated without forces method."""

    class IncompleteSpring(UserSpring):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteSpring()  # type: ignore

    class ConcreteSpring(UserSpring):
        def forces(self, state, dt):
            return 10.0 * state.disp

    spring = ConcreteSpring()
    assert isinstance(spring, UserSpring)
    assert spring.stiffness(None) == 0.0


def test_user_sensor_abc_enforcement():
    """UserSensor cannot be instantiated without evaluate method."""

    class IncompleteSensor(UserSensor):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteSensor()  # type: ignore

    class ConcreteSensor(UserSensor):
        def evaluate(self, model, time):
            return time > 0.01

    sensor = ConcreteSensor()
    assert isinstance(sensor, UserSensor)
    assert sensor.evaluate(None, 0.02) is True
    assert sensor.evaluate(None, 0.005) is False


def test_user_output_abc_enforcement():
    """UserOutput cannot be instantiated without write method."""

    class IncompleteOutput(UserOutput):
        pass

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteOutput()  # type: ignore

    class ConcreteOutput(UserOutput):
        def __init__(self):
            super().__init__()
            self.written = []

        def write(self, model, time, cycle):
            self.written.append((time, cycle))

    out = ConcreteOutput()
    assert isinstance(out, UserOutput)
    out.write(None, 0.001, 10)
    assert out.written == [(0.001, 10)]


# ----------------------------------------------------------------------------
# 2. Example Elastic User Material Physics Tests
# ----------------------------------------------------------------------------


def test_example_elastic_user_material_solid():
    """UserElasticMaterial matches law01_elastic for 3D solid elements."""
    E = 210000.0
    nu = 0.3
    rho0 = 7.85e-9

    user_mat = UserElasticMaterial(E=E, nu=nu, rho0=rho0)

    # Reference law01 material mock
    G_ref = E / (2.0 * (1.0 + nu))
    K_ref = E / (3.0 * (1.0 - 2.0 * nu))
    ref_mat = SimpleNamespace(E=E, nu=nu, G=G_ref, K=K_ref, rho0=rho0)

    # Multi-element batch: shape (3, 6)
    sig_user = np.zeros((3, 6), dtype=float)
    sig_ref = np.zeros((3, 6), dtype=float)

    deps = np.array([
        [0.001, -0.0003, -0.0003, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.002, 0.0, 0.0],
        [0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001],
    ], dtype=float)

    state = MaterialState(sig=sig_user, deps=deps, rho=rho0)
    user_mat.solid_update(state, dt=1.0e-6)
    law01_elastic.solid_update(ref_mat, sig_ref, deps)

    np.testing.assert_allclose(sig_user, sig_ref, rtol=1e-12, atol=1e-12)


def test_example_elastic_user_material_shell():
    """UserElasticMaterial matches law01_elastic for 2D plane-stress shells."""
    E = 210000.0
    nu = 0.3
    rho0 = 7.85e-9

    user_mat = UserElasticMaterial(E=E, nu=nu, rho0=rho0)

    G_ref = E / (2.0 * (1.0 + nu))
    ref_mat = SimpleNamespace(E=E, nu=nu, G=G_ref, rho0=rho0)

    # Multi-element batch: shape (2, 3) = [xx, yy, xy]
    sig_user = np.zeros((2, 3), dtype=float)
    sig_ref = np.zeros((2, 3), dtype=float)

    deps = np.array([
        [0.001, 0.0002, 0.0005],
        [-0.0005, 0.001, 0.002],
    ], dtype=float)

    state = MaterialState(sig=sig_user, deps=deps, rho=rho0)
    user_mat.shell_update(state, dt=1.0e-6)
    law01_elastic.shell_update(ref_mat, sig_ref, deps)

    np.testing.assert_allclose(sig_user, sig_ref, rtol=1e-12, atol=1e-12)


def test_example_elastic_user_material_sound_speed_and_tangent():
    """Check sound speed and tangent modulus calculations."""
    E = 210000.0
    nu = 0.3
    rho0 = 7.85e-9

    user_mat = UserElasticMaterial(E=E, nu=nu, rho0=rho0)
    state = MaterialState(sig=np.zeros(6), deps=np.zeros(6), rho=rho0)

    c = user_mat.sound_speed(state)
    G = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    expected_c = np.sqrt((K + 4.0 / 3.0 * G) / rho0)
    assert np.isclose(c, expected_c)

    C = user_mat.tangent_modulus(state)
    assert C.shape == (6, 6)
    assert np.isclose(C[0, 0], user_mat.lam + 2.0 * user_mat.G)
    assert np.isclose(C[0, 1], user_mat.lam)
    assert np.isclose(C[3, 3], user_mat.G)


# ----------------------------------------------------------------------------
# 3. Registration and Lookup Tests
# ----------------------------------------------------------------------------


def test_user_material_registration():
    """Register, lookup, and unregister user materials."""
    class CustomMat(UserMaterial):
        def solid_update(self, state, dt):
            return state.sig

        def shell_update(self, state, dt):
            return state.sig

    try:
        register_user_material(42, CustomMat)
        assert get_user_material(42) is CustomMat

        # Invalid registration type raises TypeError
        with pytest.raises(TypeError, match="subclass of UserMaterial"):
            register_user_material(43, dict)  # type: ignore
    finally:
        unregister_user_material(42)
        assert get_user_material(42) is None


def test_user_spring_registration():
    """Register, lookup, and unregister user springs."""
    class CustomSpring(UserSpring):
        def forces(self, state, dt):
            return 50.0 * state.disp

    try:
        register_user_spring(99, CustomSpring)
        assert get_user_spring(99) is CustomSpring

        with pytest.raises(TypeError, match="subclass of UserSpring"):
            register_user_spring(100, str)  # type: ignore
    finally:
        unregister_user_spring(99)
        assert get_user_spring(99) is None


def test_user_sensor_registration():
    """Register, lookup, and unregister user sensors."""
    class CustomSensor(UserSensor):
        def evaluate(self, model, time):
            return time >= 1.0

    try:
        register_user_sensor(CustomSensor, type_number=55)
        assert get_user_sensor(55) is CustomSensor

        with pytest.raises(TypeError, match="subclass of UserSensor"):
            register_user_sensor(int)  # type: ignore
    finally:
        unregister_user_sensor(55)
        assert get_user_sensor(55) is None


def test_user_output_registration():
    """Register and retrieve user output hooks."""
    class CustomOutput(UserOutput):
        def write(self, model, time, cycle):
            pass

    try:
        register_user_output(CustomOutput, name="custom_logger")
        assert get_user_output("custom_logger") is CustomOutput

        instance = CustomOutput(name="active_inst")
        register_user_output(instance)
        assert instance in get_active_user_outputs()

        with pytest.raises(TypeError, match="UserOutput subclass or instance"):
            register_user_output(12345)  # type: ignore
    finally:
        clear_user_outputs()
        assert len(get_active_user_outputs()) == 0


# ----------------------------------------------------------------------------
# 4. Dynamic Module Loading Tests
# ----------------------------------------------------------------------------


def test_dynamic_loader_single_file(tmp_path: Path):
    """Dynamically load a user module from a Python file."""
    code = """
from pyradioss.user import UserMaterial, register_user_material

class DynamicMaterial(UserMaterial):
    def solid_update(self, state, dt):
        state.sig[:] += 1.0
        return state.sig

    def shell_update(self, state, dt):
        state.sig[:] += 2.0
        return state.sig

register_user_material(77, DynamicMaterial)
"""
    module_file = tmp_path / "user_dyn_mat.py"
    module_file.write_text(code, encoding="utf-8")

    try:
        mod = load_user_library(module_file)
        assert hasattr(mod, "DynamicMaterial")
        registered_cls = get_user_material(77)
        assert registered_cls is not None
        assert registered_cls.__name__ == "DynamicMaterial"

        # Instantiate and test
        mat = registered_cls()
        sig = np.zeros(6)
        state = MaterialState(sig=sig, deps=np.zeros(6))
        mat.solid_update(state, dt=1.0)
        assert np.allclose(sig, 1.0)
    finally:
        unregister_user_material(77)


def test_dynamic_loader_file_not_found(tmp_path: Path):
    """load_user_library raises FileNotFoundError for missing files."""
    missing = tmp_path / "non_existent_module.py"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_user_library(missing)


def test_auto_discover_user_modules(tmp_path: Path):
    """auto_discover_user_modules finds and loads all user_*.py files in directory."""
    code1 = """
from pyradioss.user import UserSpring, register_user_spring
class AutoSpring1(UserSpring):
    def forces(self, state, dt): return 100.0
register_user_spring(81, AutoSpring1)
"""
    code2 = """
from pyradioss.user import UserSpring, register_user_spring
class AutoSpring2(UserSpring):
    def forces(self, state, dt): return 200.0
register_user_spring(82, AutoSpring2)
"""
    # Non-matching file that should NOT be loaded
    ignored_code = "raise RuntimeError('Should not be executed')"

    (tmp_path / "user_spring1.py").write_text(code1, encoding="utf-8")
    (tmp_path / "user_spring2.py").write_text(code2, encoding="utf-8")
    (tmp_path / "other_spring.py").write_text(ignored_code, encoding="utf-8")

    try:
        loaded = auto_discover_user_modules(tmp_path)
        assert len(loaded) == 2
        assert get_user_spring(81) is not None
        assert get_user_spring(82) is not None
    finally:
        unregister_user_spring(81)
        unregister_user_spring(82)


# ----------------------------------------------------------------------------
# 5. Stubs NotImplementedError Guidance Tests
# ----------------------------------------------------------------------------


def test_user_material_stubs():
    """UserMaterialLaw29 and UserMaterialLaw99 raise informative NotImplementedError."""
    stub29 = UserMaterialLaw29()
    state = MaterialState(sig=np.zeros(6), deps=np.zeros(6))

    with pytest.raises(NotImplementedError, match="LAW29"):
        stub29.solid_update(state, dt=1.0e-5)

    with pytest.raises(NotImplementedError, match="LAW29"):
        stub29.shell_update(state, dt=1.0e-5)

    stub99 = UserMaterialLaw99()
    with pytest.raises(NotImplementedError, match="LAW99"):
        stub99.solid_update(state, dt=1.0e-5)

    with pytest.raises(NotImplementedError, match="LAW99"):
        stub99.shell_update(state, dt=1.0e-5)


def test_user_spring_stubs():
    """UserSpringType29, 30, 31 raise informative NotImplementedError."""
    state = SpringState(disp=0.01, vel=1.0)

    stub29 = UserSpringType29()
    with pytest.raises(NotImplementedError, match="TYPE29.*PROP/USER1"):
        stub29.forces(state, dt=1.0e-5)

    stub30 = UserSpringType30()
    with pytest.raises(NotImplementedError, match="TYPE30.*PROP/USER2"):
        stub30.forces(state, dt=1.0e-5)

    stub31 = UserSpringType31()
    with pytest.raises(NotImplementedError, match="TYPE31.*PROP/USER3"):
        stub31.forces(state, dt=1.0e-5)


def test_user_sensor_stubs():
    """UserSensorType29, 30, 31 raise informative NotImplementedError."""
    stub29 = UserSensorType29()
    with pytest.raises(NotImplementedError, match="USER SENSOR 29"):
        stub29.evaluate(None, 0.0)

    stub30 = UserSensorType30()
    with pytest.raises(NotImplementedError, match="USER SENSOR 30"):
        stub30.evaluate(None, 0.0)

    stub31 = UserSensorType31()
    with pytest.raises(NotImplementedError, match="USER SENSOR 31"):
        stub31.evaluate(None, 0.0)
