"""Unit tests for /INITEMP initial temperature field generator.

Verifies:
- Uniform temperature on parts, node groups, and full models
- 3D linear spatial gradient fields T(x) = T0 + G . (x - x0)
- Shell through-thickness temperature gradient (T_top, T_mid, T_bot, layers)
- Solid element averaging from vertex nodes (Hexa8 1/8 sum, Tetra4 1/4 sum)
- Nodal table overrides (fld_type=1)
- Thermal strain Delta eps_th = alpha * (T - T_ref)
- Johnson-Cook thermal softening C_T = 1 - (T*)^m (LAW02, LAW04)
- Full Model integration through run_starter
- Deck card parser
"""

import pytest
import numpy as np

from pyradioss.common.messages import MessageLog
from pyradioss.model.entities import InitialTemperature
from pyradioss.model.model import Model, ElementGroup
from pyradioss.starter.starter import run_starter
from pyradioss.starter.initemp import (
    InitempRecord,
    InitempParams,
    apply_initemp,
    compute_thermal_strain,
    johnson_cook_thermal_softening,
    build_uniform_initemp,
    build_gradient_initemp,
    build_shell_gradient_initemp,
    build_nodal_table_initemp,
    parse_initemp_deck_cards,
)


# ============================================================================
# 1. Uniform Temperature Field
# ============================================================================

def test_initemp_uniform_all_nodes():
    """Verify uniform T0 applied across all nodes when no target group/part is set."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    rec = build_uniform_initemp(t0=350.0, id=1, title="Uniform350")
    t_arr = apply_initemp(model, [rec])

    assert len(t_arr) == 4
    np.testing.assert_allclose(t_arr, 350.0)
    assert model.temperature is model.temperatures
    np.testing.assert_allclose(model.temperature, 350.0)


def test_initemp_uniform_part():
    """Verify uniform T0 mapped specifically to elements of a designated part."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    model._id2idx = {nid: i for i, nid in enumerate(model.node_ids)}
    model.x0 = np.zeros((6, 3), dtype=np.float64)

    # Two shell elements: elem 1 in part 10, elem 2 in part 20
    # Elem 1: nodes [0, 1, 2, 3] -> Part 10
    # Elem 2: nodes [2, 3, 4, 5] -> Part 20
    conn = np.array([
        [0, 1, 2, 3],
        [2, 3, 4, 5],
    ], dtype=np.int64)
    group = ElementGroup(ids=np.array([101, 102]), conn=conn, part=np.array([0, 1]))
    group.state["part_ids"] = np.array([10, 20], dtype=np.int64)
    model.shells = group

    rec = build_uniform_initemp(t0=450.0, part_id=10, id=1)
    t_arr = apply_initemp(model, [rec], default_t0=300.0)

    # Nodes 0, 1, 2, 3 belong to part 10 -> 450 K
    # Nodes 4, 5 belong only to part 20 -> default 300 K
    assert t_arr[0] == 450.0
    assert t_arr[1] == 450.0
    assert t_arr[2] == 450.0
    assert t_arr[3] == 450.0
    assert t_arr[4] == 300.0
    assert t_arr[5] == 300.0

    # Check element state
    assert "temp" in group.state
    elem_temp = group.state["temp"]
    # Element 1 (part 10): mean(450, 450, 450, 450) = 450 K
    assert elem_temp[0] == 450.0
    # Element 2 (part 20): mean(450, 450, 300, 300) = 375 K
    assert elem_temp[1] == 375.0


# ============================================================================
# 2. 3D Linear Spatial Gradient Field
# ============================================================================

def test_initemp_linear_spatial_gradient():
    """Verify 3D linear spatial gradient T(x) = T0 + G . (x - x0)."""
    # G = (100, -50, 20) [K/m], x0 = (1.0, 2.0, 3.0), T0 = 300 K
    rec = build_gradient_initemp(
        t0=300.0,
        gradient=[100.0, -50.0, 20.0],
        x0=[1.0, 2.0, 3.0],
        id=2,
    )

    pts = np.array([
        [1.0, 2.0, 3.0],       # At reference point: dx = 0 -> T = 300
        [2.0, 2.0, 3.0],       # dx = (1, 0, 0) -> T = 300 + 100 = 400
        [1.0, 3.0, 3.0],       # dy = (0, 1, 0) -> T = 300 - 50 = 250
        [1.0, 2.0, 4.0],       # dz = (0, 0, 1) -> T = 300 + 20 = 320
        [2.0, 4.0, 5.0],       # dx = (1, 2, 2) -> T = 300 + 100 - 100 + 40 = 340
    ], dtype=np.float64)

    t_calc = rec.compute_nodal_temperature(pts)
    expected = np.array([300.0, 400.0, 250.0, 320.0, 340.0], dtype=np.float64)
    np.testing.assert_allclose(t_calc, expected)

    # Test via apply_initemp on model
    model = Model()
    model.node_ids = np.arange(1, 6)
    model.x0 = pts
    apply_initemp(model, [rec])
    np.testing.assert_allclose(model.temperature, expected)


# ============================================================================
# 3. Shell Through-Thickness Temperature Gradient
# ============================================================================

def test_initemp_shell_through_thickness_gradient():
    """Verify shell layer temperature distribution through thickness."""
    # Top = 400 K, Bottom = 300 K, Mid = 350 K (symmetric)
    rec = build_shell_gradient_initemp(t_mid=350.0, t_top=400.0, t_bot=300.0)

    # 5 integration points through thickness
    nip = 5
    xi = np.linspace(-1.0, 1.0, nip)  # [-1, -0.5, 0, 0.5, 1]
    t_layers = rec.compute_shell_layer_temperatures(nip, coords_xi=xi)

    expected = np.array([300.0, 325.0, 350.0, 375.0, 400.0])
    np.testing.assert_allclose(t_layers, expected)

    # Test asymmetric mid-temperature (piecewise linear):
    # bot = 300 K, mid = 380 K, top = 400 K
    rec_asym = build_shell_gradient_initemp(t_mid=380.0, t_top=400.0, t_bot=300.0)
    t_asym = rec_asym.compute_shell_layer_temperatures(nip, coords_xi=xi)
    # xi = -1 -> 300
    # xi = -0.5 -> 380 - 0.5*(380 - 300) = 340
    # xi = 0 -> 380
    # xi = 0.5 -> 380 + 0.5*(400 - 380) = 390
    # xi = 1 -> 400
    expected_asym = np.array([300.0, 340.0, 380.0, 390.0, 400.0])
    np.testing.assert_allclose(t_asym, expected_asym)


def test_initemp_shell_group_integration():
    """Verify shell element group receives multi-layer temperature array."""
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    model._id2idx = {nid: i for i, nid in enumerate(model.node_ids)}
    model.x0 = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["part_ids"] = np.array([1], dtype=np.int64)
    # 3 integration points (sig shape (1, 3, 3))
    group.state["sig"] = np.zeros((1, 3, 3))
    group.state["mat_extra"] = {"temp": np.zeros((1, 3))}
    model.shells = group

    rec = build_shell_gradient_initemp(
        t_mid=350.0, t_top=400.0, t_bot=300.0, part_id=1
    )
    apply_initemp(model, [rec])

    # Check that group.state['temp'] has shape (1, 3)
    st_temp = group.state["temp"]
    assert st_temp.shape == (1, 3)
    # Bot (xi < 0) < Mid (xi = 0) < Top (xi > 0)
    assert st_temp[0, 0] < st_temp[0, 1] < st_temp[0, 2]
    # Check mat_extra['temp'] synchronization
    np.testing.assert_allclose(group.state["mat_extra"]["temp"], st_temp)


# ============================================================================
# 4. Solid Element Vertex Averaging (sinit3.F:275-280)
# ============================================================================

def test_initemp_solid_element_hexa8_average():
    """Verify Hexa8 element temperature is exactly 1/8 sum of vertex node temperatures."""
    model = Model()
    model.node_ids = np.arange(1, 9, dtype=np.int64)
    model._id2idx = {nid: i for i, nid in enumerate(model.node_ids)}
    model.x0 = np.zeros((8, 3))

    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["part_ids"] = np.array([1], dtype=np.int64)
    model.bricks = group

    # Assign distinct temperatures to 8 vertices
    nodal_overrides = {
        1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0,
        5: 500.0, 6: 600.0, 7: 700.0, 8: 800.0,
    }
    rec = build_nodal_table_initemp(t0=0.0, nodal_temps=nodal_overrides)
    apply_initemp(model, [rec])

    expected_mean = np.mean(list(nodal_overrides.values()))  # 450.0
    assert group.state["temp"][0] == pytest.approx(expected_mean)


def test_initemp_solid_element_tetra4_average():
    """Verify Tetra4 element temperature is 1/4 sum of vertex node temperatures."""
    model = Model()
    model.node_ids = np.arange(1, 5, dtype=np.int64)
    model._id2idx = {nid: i for i, nid in enumerate(model.node_ids)}
    model.x0 = np.zeros((4, 3))

    conn = np.array([[0, 1, 2, 3]], dtype=np.int64)
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    group.state["part_ids"] = np.array([1], dtype=np.int64)
    model.tetras = group

    nodal_overrides = {1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0}
    rec = build_nodal_table_initemp(t0=0.0, nodal_temps=nodal_overrides)
    apply_initemp(model, [rec])

    expected_mean = np.mean([100.0, 200.0, 300.0, 400.0])  # 250.0
    assert group.state["temp"][0] == pytest.approx(expected_mean)


# ============================================================================
# 5. Nodal Table Overrides (fld_type=1)
# ============================================================================

def test_initemp_nodal_table_overrides():
    """Verify fld_type=1 applies base T0 then overrides specific nodes."""
    model = Model()
    model.node_ids = np.array([101, 102, 103, 104], dtype=np.int64)
    model._id2idx = {nid: i for i, nid in enumerate(model.node_ids)}
    model.x0 = np.zeros((4, 3))

    # Base T0 = 293.15, overrides on 101 -> 310, 103 -> 330
    rec = build_nodal_table_initemp(
        t0=293.15,
        nodal_temps={101: 310.0, 103: 330.0},
        id=1,
    )
    apply_initemp(model, [rec])

    assert model.temperature[0] == 310.0
    assert model.temperature[1] == 293.15
    assert model.temperature[2] == 330.0
    assert model.temperature[3] == 293.15


# ============================================================================
# 6. Thermal Strain Delta eps_th = alpha * (T - T_ref) * I
# ============================================================================

def test_thermal_strain_solid_and_shell():
    """Verify compute_thermal_strain for 3D solid and 2D plane-stress shell."""
    alpha = 1.2e-5  # Steel thermal expansion
    t_ref = 293.15  # Room temp
    temp = 393.15   # Delta T = 100 K

    # Solid (6 components: exx, eyy, ezz, exy, eyz, ezx)
    eps_solid = compute_thermal_strain(temp, t_ref=t_ref, alpha=alpha, elem_type="SOLID")
    expected_val = alpha * 100.0  # 1.2e-3
    assert eps_solid.shape == (6,)
    np.testing.assert_allclose(eps_solid[:3], expected_val)
    np.testing.assert_allclose(eps_solid[3:], 0.0)

    # Shell (3 in-plane components: exx, eyy, exy)
    eps_shell = compute_thermal_strain(temp, t_ref=t_ref, alpha=alpha, elem_type="SHELL")
    assert eps_shell.shape == (3,)
    np.testing.assert_allclose(eps_shell[:2], expected_val)
    assert eps_shell[2] == 0.0


# ============================================================================
# 7. Johnson-Cook Thermal Softening (LAW02, LAW04)
# ============================================================================

def test_johnson_cook_thermal_softening():
    """Verify C_T = 1 - (T*)^m with T* = (T - T_room) / (T_melt - T_room)."""
    t_room = 293.15
    t_melt = 1793.15

    # At room temperature: T* = 0 -> C_T = 1.0
    assert johnson_cook_thermal_softening(t_room, t_room, t_melt, m=1.0) == pytest.approx(1.0)

    # Below room temperature: clipped to T* = 0 -> C_T = 1.0
    assert johnson_cook_thermal_softening(200.0, t_room, t_melt, m=1.0) == pytest.approx(1.0)

    # At melting temperature: T* = 1 -> C_T = 0.0
    assert johnson_cook_thermal_softening(t_melt, t_room, t_melt, m=1.0) == pytest.approx(0.0)

    # Above melting temperature: clipped to T* = 1 -> C_T = 0.0
    assert johnson_cook_thermal_softening(2500.0, t_room, t_melt, m=1.0) == pytest.approx(0.0)

    # Intermediate temperature: T = 1043.15 (Delta T = 750, denom = 1500) -> T* = 0.5
    # For m = 1.0 -> C_T = 0.5
    assert johnson_cook_thermal_softening(1043.15, t_room, t_melt, m=1.0) == pytest.approx(0.5)
    # For m = 2.0 -> C_T = 1 - 0.5^2 = 0.75
    assert johnson_cook_thermal_softening(1043.15, t_room, t_melt, m=2.0) == pytest.approx(0.75)


def test_johnson_cook_material_coupling():
    """Verify coupling of initial temperature with law02_johnson_cook."""
    from pyradioss.materials import law02_johnson_cook
    from pyradioss.model.entities import Material

    mat = Material(id=1, title="Steel", law=2, rho0=7.8e-9)
    mat.params = {
        "A": 500.0,
        "B": 300.0,
        "n": 0.5,
        "T_i": 293.15,
        "T_melt": 1793.15,
        "mT": 1.0,
    }

    # Element at 1043.15 K -> temp rise above T_i is 750.0 K
    extra = {"temp": np.array([750.0])}
    ct, _ = law02_johnson_cook._thermal_factor(mat, extra)
    # T* = 750 / (1793.15 - 293.15) = 750 / 1500 = 0.5 -> C_T = 0.5
    assert pytest.approx(ct[0]) == 0.5


# ============================================================================
# 8. Starter Integration with /INITEMP Deck
# ============================================================================

def test_starter_run_with_initemp(tmp_path):
    """Verify run_starter parses /INITEMP and initializes model.temperature."""
    deck = """\
#---1---+----2---+----3---+----4---+----5---+----6---+----7---+----8---+----9---+---10---+
/BEGIN
TEST_STARTER_INITEMP
      2021         0
/MAT/LAW1/1
Elastic
              7.8e-9
            210000.0                 0.3
/PROP/TYPE1/1
Shell_Prop
         1         1         1         0         0         0         0
                 1.0                 1.0                 1.0
                 1.0            0.833333
/PART/1
Part_one
         1         1
/NODE
       101                 0.0                 0.0                 0.0
       102                10.0                 0.0                 0.0
       103                10.0                10.0                 0.0
       104                 0.0                10.0                 0.0
/SHELL/1
         1       101       102       103       104
/GRNOD/NODE/1
AllNodes
       101       102       103       104
/INITEMP/1
InitialTemp_Uniform
#                 T0   grnd_ID  fld_type
              373.15         1         0
/END
"""
    p = tmp_path / "TEST_INITEMP_0000.rad"
    p.write_text(deck, encoding="utf-8")
    log = MessageLog()
    model = run_starter(str(p), log)

    assert len(log.errors) == 0
    assert hasattr(model, "temperature")
    assert model.temperature is not None
    assert len(model.temperature) == 4
    np.testing.assert_allclose(model.temperature, 373.15)

    # Shell element temperature should be 373.15
    assert hasattr(model, "shells")
    assert "temp" in model.shells.state
    np.testing.assert_allclose(model.shells.state["temp"], 373.15)


# ============================================================================
# 9. Deck Cards Parser
# ============================================================================

def test_parse_initemp_deck_cards():
    """Verify parse_initemp_deck_cards parses standard and extended directives."""
    cards = [
        "/INITEMP/1",
        "Spatial_Gradient_Field",
        "300.0   1   0",
        "GRADIENT 10.0 20.0 -5.0 1.0 2.0 3.0",
        "/INITEMP/2",
        "Shell_Gradient_Field",
        "SHELL 400.0 350.0 300.0",
        "/INITEMP/3",
        "Nodal_Table",
        "293.15  1   1",
        "310.0   101",
        "320.0   102",
    ]

    records = parse_initemp_deck_cards(cards)
    assert len(records) == 3

    # Record 1: Gradient
    assert records[0].id == 1
    assert records[0].t0 == 300.0
    assert records[0].gradient == (10.0, 20.0, -5.0)
    assert records[0].x0 == (1.0, 2.0, 3.0)

    # Record 2: Shell
    assert records[1].id == 2
    assert records[1].t_top == 400.0
    assert records[1].t_mid == 350.0
    assert records[1].t_bot == 300.0

    # Record 3: Nodal table
    assert records[2].id == 3
    assert records[2].fld_type == 1
    assert records[2].t0 == 293.15
    assert records[2].nodal_temps[101] == 310.0
    assert records[2].nodal_temps[102] == 320.0


# ============================================================================
# 10. Additive Mode
# ============================================================================

def test_initemp_additive_mode():
    """Verify additive=True adds multiple temperature fields together."""
    model = Model()
    model.node_ids = np.array([1, 2], dtype=np.int64)
    model.x0 = np.zeros((2, 3))

    rec1 = build_uniform_initemp(t0=300.0, id=1)
    rec2 = build_uniform_initemp(t0=50.0, id=2)
    rec2.additive = True

    apply_initemp(model, [rec1, rec2])
    np.testing.assert_allclose(model.temperature, 350.0)
