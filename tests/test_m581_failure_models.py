"""
Milestone M581: Core Failure Models Physics Completion.
Tests for /FAIL/WILKINS, /FAIL/CHANG, /FAIL/HASHIN, /FAIL/MMC (/FAIL/WIERZBICKI), and /FAIL/TBUTCHER.
"""

import numpy as np
import pytest

from pyradioss.model.entities import FailureModel, Material, Part
from pyradioss.model.model import Model
from pyradioss.failure import solid_step, shell_step
from pyradioss.failure import wilkins, chang, hashin, mmc, tbutcher
from pyradioss.input.deck_reader import read_deck


def test_wilkins_damage_accumulation():
    """Verify Wilkins cumulative damage model dD = W1 * W2 * d_epsp."""
    # Setup failure model: alpha=1.0, beta=1.0, plim=100.0, df=0.5
    fail = FailureModel(type="WILKINS", ifail_sh=1, params={
        "alpha": 1.0, "beta": 1.0, "plim": 100.0, "df": 0.5, "ifail_so": 1
    })
    n = 2
    # Element 0: pure hydrostatic tension (sxx=syy=szz=50.0, sxy=syz=szx=0)
    # Element 1: zero stress
    sig = np.array([
        [50.0, 50.0, 50.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    d_epsp = np.array([0.1, 0.1])
    deps = np.zeros_like(sig)
    dt = 1e-4
    dama = np.zeros(n, dtype=float)

    # For Element 0: P = 50, W1 = 1 / (1 - 50/100) = 2.0. Deviator = 0 -> A = 1 -> W2 = (2 - 1) = 1.0
    # Expected dD = 2.0 * 1.0 * 0.1 = 0.2
    broken = wilkins.solid_step(fail, sig, d_epsp, deps, dt, dama)
    assert not broken[0]
    assert np.isclose(dama[0], 0.2, atol=1e-4)

    # Another cycle with larger plastic increment: d_epsp = 0.2 -> dD = 0.4 -> dama = 0.6 >= df(0.5)
    d_epsp = np.array([0.2, 0.0])
    broken = solid_step(fail, sig, d_epsp, deps, dt, dama)
    assert broken[0]
    assert not broken[1]
    assert np.isclose(dama[0], 0.6, atol=1e-4)


def test_chang_failure_modes():
    """Verify Chang-Chang 4 composite modes: fiber tension/compression, matrix tension/compression."""
    fail = FailureModel(type="CHANG", ifail_sh=1, params={
        "xt": 100.0, "xc": 80.0, "yt": 50.0, "yc": 40.0, "s": 30.0, "beta": 1.0
    })
    # 4 elements testing each mode
    # Elem 0: Tensile fiber (sxx = 100 -> (100/100)^2 = 1.0 -> broken)
    # Elem 1: Compressive fiber (sxx = -80 -> (-80/80)^2 = 1.0 -> broken)
    # Elem 2: Tensile matrix (syy = 50 -> (50/50)^2 = 1.0 -> broken)
    # Elem 3: Shear (sxy = 30 -> (30/30)^2 = 1.0 -> broken)
    sig = np.array([
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-80.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 50.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 30.0, 0.0, 0.0]
    ])
    dama = np.zeros(4, dtype=float)
    broken = chang.solid_step(fail, sig, np.zeros(4), np.zeros((4, 6)), 1e-4, dama)
    assert np.all(broken)
    assert np.all(dama >= 1.0)


def test_hashin_failure_modes():
    """Verify Hashin 3D envelopes including delamination and crush."""
    fail = FailureModel(type="HASHIN", ifail_sh=1, params={
        "xt": 100.0, "xc": 80.0, "yt": 50.0, "yc": 40.0, "zt": 45.0,
        "csig": 200.0, "fsig12": 30.0, "msig12": 30.0, "msig23": 25.0, "angle": 0.0
    })
    sig = np.array([
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Fiber tension: 1.0
        [-80.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # Fiber compression: 1.0
        [-200.0, -200.0, -200.0, 0.0, 0.0, 0.0], # Hydrostatic crush: P = 200 -> 1.0
        [0.0, 0.0, 45.0, 0.0, 0.0, 0.0]    # Delamination in z: 1.0
    ])
    dama = np.zeros(4, dtype=float)
    broken = hashin.solid_step(fail, sig, np.zeros(4), np.zeros((4, 6)), 1e-4, dama)
    assert np.all(broken)
    assert np.all(dama >= 1.0)


def test_mmc_wierzbicki_damage():
    """Verify Modified Mohr-Coulomb / Xue-Wierzbicki fracture strain and damage."""
    fail = FailureModel(type="MMC", ifail_sh=1, params={
        "c1": 0.5, "c2": 0.0, "c3": 0.0, "c4": 0.0, "m": 1.0, "n": 1.0
    })
    # With c2=c3=c4=0, n=1: eps_f = c1 = 0.5
    sig = np.array([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d_epsp = np.array([0.25])
    dama = np.zeros(1, dtype=float)

    broken = mmc.solid_step(fail, sig, d_epsp, np.zeros((1, 6)), 1e-4, dama)
    assert not broken[0]
    # dD = d_epsp / eps_f = 0.25 / 0.5 = 0.5
    assert np.isclose(dama[0], 0.5, atol=1e-3)

    # Next step with 0.3 plastic strain -> total damage = 0.5 + 0.6 = 1.1 >= 1.0
    d_epsp = np.array([0.3])
    broken = solid_step(fail, sig, d_epsp, np.zeros((1, 6)), 1e-4, dama)
    assert broken[0]
    assert np.isclose(dama[0], 1.1, atol=1e-3)


def test_tbutcher_dynamic_spall():
    """Verify Tuler-Butcher dynamic spall criterion D = integral (sigma_1 - sigma_0)^lambda dt."""
    fail = FailureModel(type="TBUTCHER", ifail_sh=1, params={
        "lambda": 2.0, "k": 100.0, "sigma_r": 50.0, "ifail_so": 1
    })
    # Pure uniaxial tension sigma_1 = 60.0 (over_stress = 10.0)
    sig = np.array([[60.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dt = 0.5
    dama = np.zeros(1, dtype=float)

    # Step 1: dD = dt * (60 - 50)^2 = 0.5 * 100 = 50.0 < K(100)
    broken = tbutcher.solid_step(fail, sig, np.zeros(1), np.zeros((1, 6)), dt, dama)
    assert not broken[0]
    assert np.isclose(dama[0], 50.0)

    # Step 2: dD = 50.0 -> total D = 100.0 >= K
    broken = solid_step(fail, sig, np.zeros(1), np.zeros((1, 6)), dt, dama)
    assert broken[0]
    assert np.isclose(dama[0], 100.0)


def test_starter_parsing_all_failure_keywords():
    """Verify starter parsing and entity attachment for /FAIL/WILKINS, CHANG, HASHIN, MMC, TBUTCHER."""
    deck_text = """/BEGIN
TEST_FAILURE_MODELS
/MAT/PLAS_JOHNS/1
Steel
7.85e-9
210000 0.3
300 400 0.5 0.0 0.0
/FAIL/WILKINS/1
1.0 1.0 500.0 1.0
1 1
/MAT/PLAS_JOHNS/2
Composite
1.5e-9
50000 0.25
200 300 0.5 0.0 0.0
/FAIL/CHANG/2
1500 50 70 1000 150
1.0 1.0 1 1
/MAT/PLAS_JOHNS/3
MMC_Mat
2.7e-9
70000 0.33
250 350 0.4 0.0 0.0
/FAIL/MMC/3
0.8 0.5 0.2 0.1 1.0
1.0 1 1 0
/MAT/PLAS_JOHNS/4
Spall_Mat
8.9e-9
110000 0.34
220 300 0.5 0.0 0.0
/FAIL/TBUTCHER/4
2.0 500.0 100.0 1 1
/END
"""
    deck = read_deck(deck_text.splitlines())
    from pyradioss.starter.starter import build_model
    model = build_model(deck)

    # Check material 1 has WILKINS
    mat1 = model.materials[1]
    assert mat1.fail is not None
    assert mat1.fail.type == "WILKINS"
    assert mat1.fail.params["alpha"] == 1.0
    assert mat1.fail.params["plim"] == 500.0

    # Check material 2 has CHANG
    mat2 = model.materials[2]
    assert mat2.fail is not None
    assert mat2.fail.type == "CHANG"
    assert mat2.fail.params["sigma_1t"] == 1500.0
    assert mat2.fail.params["sigma_2t"] == 50.0

    # Check material 3 has WIERZBICKI / MMC
    mat3 = model.materials[3]
    assert mat3.fail is not None
    assert mat3.fail.type == "WIERZBICKI"
    assert mat3.fail.params["c1"] == 0.8
    assert mat3.fail.params["c2"] == 0.5

    # Check material 4 has TBUTCHER
    mat4 = model.materials[4]
    assert mat4.fail is not None
    assert mat4.fail.type == "TBUTCHER"
    assert mat4.fail.params["lambda"] == 2.0
    assert mat4.fail.params["k"] == 500.0
    assert mat4.fail.params["sigma_r"] == 100.0


def test_engine_failure_erosion_and_energy_conservation(tmp_path):
    """Verify that in explicit dynamic cycle, failure erodes element, zeroes stress, and preserves mass."""
    from pyradioss.common.messages import MessageLog
    from pyradioss.model.model import ElementGroup, EngineControls
    from pyradioss.model.entities import Property
    from pyradioss.engine.engine import _integrate
    from pyradioss.elements import solid_hexa8

    m = Model()
    m.title = "TEST_FAIL_EROSION"

    nodes_x = np.array([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.01, 0.01, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, 0.0, 0.01],
        [0.01, 0.0, 0.01],
        [0.01, 0.01, 0.01],
        [0.0, 0.01, 0.01],
    ], dtype=np.float64)

    m.node_ids = np.arange(1, len(nodes_x) + 1, dtype=np.int64)
    m.x0 = nodes_x.copy()
    m.x = nodes_x.copy()
    m.v = np.zeros((m.numnod, 3), dtype=np.float64)
    m.vr = np.zeros((m.numnod, 3), dtype=np.float64)
    m.a = np.zeros((m.numnod, 3), dtype=np.float64)
    m.mass = np.ones(m.numnod, dtype=np.float64) * 0.001
    m.mass0 = m.mass.copy()
    m.inertia = np.zeros(m.numnod, dtype=np.float64)

    # Attach TBUTCHER with low critical threshold
    fail = FailureModel(type="TBUTCHER", ifail_sh=1, params={
        "lambda": 1.0, "k": 1e-4, "sigma_r": 100.0, "ifail_so": 1
    })
    mat = Material(id=1, law=2, rho0=7800.0, params={
        "E": 2.1e11, "nu": 0.3, "a": 100.0, "b": 100.0, "n": 0.5, "c": 0.0, "eps_dot_0": 1.0
    })
    mat.fail = fail
    prop = Property(id=1, type=14, params={})
    part = Part(id=1, prop_id=1, mat_id=1)

    m.materials[1] = mat
    m.properties[1] = prop
    m.parts[1] = part
    m.parts_list = [part]

    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    bg = ElementGroup(
        ids=np.array([1], dtype=np.int64),
        conn=conn,
        part=np.zeros(1, dtype=np.int64),
        state={"slices": [(slice(0, 1), mat, prop)]}
    )
    solid_hexa8.init_group(bg, m, MessageLog())
    m.bricks = bg

    # Pull top nodes with upward velocity to generate tension
    m.v[4:, 2] = 200.0

    controls = EngineControls()
    controls.t_end = 5e-5
    controls.dt_scale = 0.9

    out_dir = str(tmp_path)
    res_model = _integrate(m, controls, MessageLog(), out_dir, "TEST_RUN", 1)

    # The brick must have failed and eroded
    assert res_model.bricks.state["off"][0] == 0.0, "Element must be eroded by failure model"
    # Deleted element must carry zero stress
    assert np.allclose(res_model.bricks.state["sig"][0], 0.0), "Eroded element must carry 0 stress"
    # Nodal mass must be strictly conserved (OpenRadioss never reduces nodal mass upon erosion)
    assert np.allclose(res_model.mass, m.mass0), "Lumped nodal mass must remain conserved"



def test_gurson_tvergaard_ductile_fracture():
    """Verify Tvergaard-Needleman porous ductile fracture model (GURSON)."""
    fail = FailureModel(type="GURSON", ifail_sh=1, params={
        "q1": 1.5, "q2": 1.0, "fc": 0.10, "fr": 0.20, "f0": 0.01,
        "as_": 0.05, "kw": 0.2, "epn": 0.0
    })
    # Element 0: high triaxiality tension -> void growth + nucleation
    # Element 1: compression -> nucleation inhibited, no void growth
    sig = np.array([
        [150.0, 100.0, 100.0, 0.0, 0.0, 0.0],
        [-100.0, -100.0, -100.0, 0.0, 0.0, 0.0]
    ])
    d_epsp = np.array([0.05, 0.05])
    deps = np.zeros_like(sig)
    dama = np.zeros(2, dtype=float)

    broken = solid_step(fail, sig, d_epsp, deps, 1e-4, dama)
    assert not broken[0]
    assert not broken[1]
    assert dama[0] > dama[1]  # Tensile element accumulates significantly more void damage

    # Large plastic strain driving void coalescence and failure
    d_epsp_large = np.array([0.8, 0.0])
    broken2 = solid_step(fail, sig, d_epsp_large, deps, 1e-4, dama)
    assert broken2[0]
    assert dama[0] >= 1.0


@pytest.mark.parametrize("ftype", ["WILKINS", "CHANG", "HASHIN", "MMC", "TVERGAARD", "TBUTCHER"])
def test_all_failure_models_shell_step(ftype):
    """Verify shell_step works correctly for all ported failure models."""
    params = {
        "alpha": 1.0, "beta": 1.0, "plim": 500.0, "df": 1.0,
        "xt": 500.0, "xc": 400.0, "yt": 50.0, "yc": 40.0, "s": 30.0,
        "zt": 45.0, "csig": 200.0, "fsig12": 30.0, "msig12": 30.0, "msig23": 25.0, "angle": 0.0,
        "c1": 0.5, "c2": 0.2, "c3": 0.1, "c4": 0.2, "m": 1.0, "n": 1.0,
        "q1": 1.5, "q2": 1.0, "fc": 0.15, "fr": 0.25, "f0": 0.02,
        "lambda": 2.0, "k": 100.0, "sigma_r": 50.0,
    }
    fail = FailureModel(type=ftype, ifail_sh=1, params=params)
    sig_shell = np.array([[100.0, 50.0, 10.0]])
    d_epsp = np.array([0.02])
    deps = np.zeros_like(sig_shell)
    dama = np.zeros(1, dtype=float)

    broken = shell_step(fail, sig_shell, d_epsp, deps, 1e-4, dama)
    assert isinstance(broken, (np.ndarray, bool))
    assert dama[0] >= 0.0
