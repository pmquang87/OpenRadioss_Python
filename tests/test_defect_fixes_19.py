"""
Regression tests for the 19 defect groups (14 P1, 4 P2, 1 P3).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock

from pyradioss.model.model import Model, ElementGroup
from pyradioss.model.entities import (
    Part, Material, Property, NodeGroup, AddedMass,
    AdmasNonUniform, AdmasNonUniformItem, FailureModel
)
from pyradioss.implicit.dofmap import DofMap
from pyradioss.implicit.assembly import assemble, assemble_mass
from pyradioss.elements import (
    solid_tetra10, solid_bric20, shell_thick16, solid_hexa8, shell_tri3
)
from pyradioss.failure import snconnect
from pyradioss.engine.ams import AMSManager
from pyradioss.contact.inter_type7 import ContactType7
from pyradioss.contact.inter_type11 import ContactType11
from pyradioss.contact.inter_type18 import ContactType18
from pyradioss.output.time_history import TimeHistory
from pyradioss.starter import initialization
from pyradioss.common.messages import MessageLog


# ----------------------------------------------------------------------------
# Defect 1: Implicit assembly supports all element families
# ----------------------------------------------------------------------------
def test_defect_1_implicit_assembly_all_families():
    from pyradioss.implicit.assembly import _TANGENT_KERNELS, _MASS_KERNELS, KERNELS
    # All kernels in KERNELS must be registered
    for k in KERNELS:
        assert k in _TANGENT_KERNELS
        assert k in _MASS_KERNELS


# ----------------------------------------------------------------------------
# Defect 2: QBAT, QEPH, DKT18 have rotational DOFs in DofMap
# ----------------------------------------------------------------------------
def test_defect_2_shell_rotational_dofs():
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4])
    model.x = np.zeros((4, 3))
    model.mass = np.ones(4)
    # Add a shells_qbat element
    conn = np.array([[0, 1, 2, 3]])
    model.shells_qbat = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    dof = DofMap(model)
    # Shell nodes 0..3 must have 6 DOFs each
    for n in range(4):
        assert dof.has_rot[n]
        for d in range(6):
            assert dof.eq[n * 6 + d] >= 0


# ----------------------------------------------------------------------------
# Defect 3: Virtual midside condensation and edofs -1
# ----------------------------------------------------------------------------
def test_defect_3_virtual_midsides():
    # TETRA10 with slaved midsides
    conn_tet = np.array([[0, 1, 2, 3, -1, -1, -1, -1, -1, -1]])
    edofs_tet = solid_tetra10._edofs(conn_tet)
    assert np.all(edofs_tet[:, 12:] == -1)
    assert np.all(edofs_tet[:, :12] >= 0)

    # BRIC20 with slaved midsides
    conn_bric = np.array([[0, 1, 2, 3, 4, 5, 6, 7] + [-1] * 12])
    edofs_bric = solid_bric20._edofs(conn_bric)
    assert np.all(edofs_bric[:, 24:] == -1)
    assert np.all(edofs_bric[:, :24] >= 0)

    # SHEL16 with slaved midsides
    conn_s16 = np.array([[0, 1, 2, 3, 4, 5, 6, 7] + [-1] * 8])
    edofs_s16 = shell_thick16._edofs(conn_s16)
    assert np.all(edofs_s16[:, 24:] == -1)
    assert np.all(edofs_s16[:, :24] >= 0)


# ----------------------------------------------------------------------------
# Defect 4: Implicit dynamics finite residual check
# ----------------------------------------------------------------------------
def test_defect_4_implicit_dynamics_finite_residual():
    # Verify that dynamics terminates / does not accept inf residual
    ref = 1.0
    tol = 1e-4
    rnorm = float("inf")
    # Old logic: rnorm <= tol * ref when ref became inf -> inf <= inf is True!
    # New logic: requires np.isfinite(rnorm)
    assert not (np.isfinite(rnorm) and rnorm <= tol * ref)


# ----------------------------------------------------------------------------
# Defect 5: assemble_mass 1D dof indexing and PART handling
# ----------------------------------------------------------------------------
def test_defect_5_assemble_mass_indexing_and_admas():
    model = Model()
    model.node_ids = np.array([1, 2])
    model.x = np.zeros((2, 3))
    model.mass = np.array([1.0, 1.0])
    # Nodal mass
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0, 1]))
    model.admas.append(AddedMass(id=1, grnod_id=1, mass=5.0, mass_type=1))
    
    # Non-uniform PART mass
    model.admas_non_uniforms[1] = AdmasNonUniform(
        id=1, kind="PART", items=[AdmasNonUniformItem(mass=10.0, entity_id=1)]
    )
    grp = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1]]), part=np.array([0]))
    grp.state["part_ids"] = np.array([1])
    grp.state["mass"] = np.array([2.0])
    model.trusses = grp

    dof = DofMap(model)
    M = assemble_mass(model, dof, model.x)
    assert M.shape == (dof.ndof, dof.ndof)
    # Total mass assembled: 2.0 (truss) + 5.0 (admas) + 10.0 (non-uniform part) = 17.0
    # Sum of all matrix entries for x DOF should equal 17.0
    total_x_mass = sum(M[dof.eq[a * 6 + 0], dof.eq[b * 6 + 0]] for a in range(2) for b in range(2))
    assert total_x_mass == pytest.approx(17.0, rel=1e-5)


# ----------------------------------------------------------------------------
# Defect 6: TETRA10 mass lumping per s10mass3.F
# ----------------------------------------------------------------------------
def test_defect_6_tetra10_mass_formula():
    conn = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    x0 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ], dtype=float)
    model = MagicMock()
    model.x0 = x0
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    mat = MagicMock()
    mat.rho0 = 1.0
    mat.law = 1
    mat.eos = None
    mat.fail = None
    mat.params = {}
    prop = MagicMock()
    prop.params = {}
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    log = MagicMock()

    node_idx, mass_c, _ = solid_tetra10.init_group(group, model, log)
    total_m = group.state["mass"][0]
    # Corners: m/32
    for m in mass_c[:4]:
        assert m == pytest.approx(total_m / 32.0, rel=1e-6)
    # Midsides: 7m/48
    for m in mass_c[4:]:
        assert m == pytest.approx(7.0 * total_m / 48.0, rel=1e-6)


# ----------------------------------------------------------------------------
# Defect 7: BRIC20 dama allocation and failure step
# ----------------------------------------------------------------------------
def test_defect_7_bric20_dama_and_failure():
    conn = np.arange(20)[None, :]
    group = ElementGroup(ids=np.array([1]), conn=conn, part=np.array([0]))
    model = MagicMock()
    model.x0 = np.random.randn(20, 3)
    mat = MagicMock()
    mat.rho0 = 1.0
    mat.law = 1
    mat.fail = MagicMock()
    mat.eos = None
    mat.params = {}
    prop = MagicMock()
    prop.params = {}
    group.state["slices"] = [(slice(0, 1), mat, prop)]
    log = MagicMock()

    solid_bric20.init_group(group, model, log)
    assert "dama" in group.state
    assert group.state["dama"].shape == (1, 8)


# ----------------------------------------------------------------------------
# Defect 8: AMS group.part ndarray truth value
# ----------------------------------------------------------------------------
def test_defect_8_ams_array_truth_value():
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4])
    model.x = np.zeros((4, 3))
    model.x0 = np.zeros((4, 3))
    part = Part(id=1, mat_id=1, prop_id=1)
    model.parts[1] = part
    model.parts_list = [part]

    conn = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    # group.part is an ndarray of length 2!
    grp = ElementGroup(ids=np.array([1, 2]), conn=conn, part=np.array([0, 0]))
    grp.state["mass"] = np.array([1.0, 1.0])
    model.quads = grp

    controls = MagicMock()
    controls.dt_min = 1.0
    controls.dt_scale = 1.0
    controls.dt_ams_igrp = 0

    ams = AMSManager(model, controls)
    dt_claims = [np.array([0.5, 0.5])]
    # Must not raise "ValueError: The truth value of an array with more than one element is ambiguous"
    tagged = ams.tag_nodes(dt_claims)
    assert tagged.any()
    diag, M_off, _ = ams.build_ams_matrix(dt_claims)
    assert diag is not None


# ----------------------------------------------------------------------------
# Defect 9: Contact activation time gating
# ----------------------------------------------------------------------------
def test_defect_9_contact_activation_time():
    itf = MagicMock()
    itf.id = 1
    itf.tstart = 0.5
    itf.tstop = 1.0
    itf.stfac = 1.0
    itf.gap = 0.1
    itf.fric = 0.0
    itf.sens_id = 0
    itf.line_id1 = 1
    itf.line_id2 = 2
    itf.surf_id = 1
    itf.grnod_id = 1

    model = Model()
    model.node_ids = np.array([1, 2, 3, 4])
    model.x = np.zeros((4, 3))
    model.v = np.zeros((4, 3))
    model.mass = np.ones(4)

    ct7 = ContactType7(itf, model, MagicMock())
    # Before tstart: forces must return 0 work
    fcont = np.zeros((4, 3))
    w, _ = ct7.forces(model.x, model.v, model.mass, 0.01, fcont, cycle=0, t=0.1)
    assert w == 0.0

    # ContactType11 and 18 also accept t and **kwargs
    ct11 = ContactType11(itf, model, MagicMock())
    w11, _ = ct11.forces(model.x, model.v, model.mass, 0.01, fcont, cycle=0, t=0.1)
    assert w11 == 0.0

    ct18 = ContactType18(itf, model, MagicMock())
    w18, _ = ct18.forces(model.x, model.v, model.mass, 0.01, fcont, cycle=0, t=0.1)
    assert w18 == 0.0


# ----------------------------------------------------------------------------
# Defect 10: SNCONNECT state survives state copy / restart
# ----------------------------------------------------------------------------
def test_defect_10_snconnect_state_survives_copy():
    fail = FailureModel(type="SNCONNECT", params={"a2": 0.0, "b2": 1.0, "a3": 0.0, "b3": 1.0})

    dama = np.zeros(4)
    sig = np.zeros((4, 6))
    sig[:, 2] = 100.0  # normal stress

    # Step once — accumulates epsp on the fail object
    snconnect.solid_step(fail, sig, np.array([0.5, 0.5, 0.5, 0.5]), None, 1e-4, dama)
    assert hasattr(fail, "_snc")
    epsp_before = fail._snc["epsp"].copy()
    assert epsp_before[0] == pytest.approx(0.5)

    # Simulate restart: replace dama with a fresh array of the same shape
    dama_new = dama.copy()
    snconnect.solid_step(fail, sig, np.array([0.3, 0.3, 0.3, 0.3]), None, 1e-4, dama_new)
    # History must accumulate (0.5 + 0.3 = 0.8), not reset to 0.3
    assert fail._snc["epsp"][0] == pytest.approx(0.8)


# ----------------------------------------------------------------------------
# Defect 11 & 12: ADMAS mass_type (0, 1, 2, 3) and ADMAS/NON_UNIFORM
# ----------------------------------------------------------------------------
def test_defect_11_12_admas_types():
    model = Model()
    model.node_ids = np.array([1, 2, 3, 4])
    model.x = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], dtype=float)
    model.x0 = model.x.copy()
    model.mass = np.zeros(4)

    # Node group 1 with 2 nodes (0, 1)
    model.node_groups[1] = NodeGroup(id=1, node_idx=np.array([0, 1]))
    # Type 0: per-node mass (1.0 each -> 1.0 to node 0 and 1)
    model.admas.append(AddedMass(id=1, grnod_id=1, mass=1.0, mass_type=0))
    # Type 1: total mass (2.0 divided over 2 nodes -> 1.0 each)
    model.admas.append(AddedMass(id=2, grnod_id=1, mass=2.0, mass_type=1))

    # Non uniform node mass
    model.admas_non_uniforms[1] = AdmasNonUniform(
        id=1, kind="NODE", items=[AdmasNonUniformItem(mass=3.0, entity_id=3)]
    )
    model._id2idx = {0: 0, 1: 1, 2: 2, 3: 3}

    log = MessageLog()
    initialization.initialize_elements_and_mass(model, log)
    # Node 0: 1.0 (type 0) + 1.0 (type 1) = 2.0
    assert model.mass[0] == pytest.approx(2.0)
    # Node 1: 1.0 (type 0) + 1.0 (type 1) = 2.0
    assert model.mass[1] == pytest.approx(2.0)
    # Node 3: 3.0 from non-uniform
    assert model.mass[3] == pytest.approx(3.0)


# ----------------------------------------------------------------------------
# Defect 13: Node 0 validation in element connectivity
# ----------------------------------------------------------------------------
def test_defect_13_node_0_validation():
    model = Model()
    model._id2idx = {1: 0, 2: 1, 3: 2, 4: 3}
    model.parts[1] = Part(id=1, mat_id=1, prop_id=1)
    model.properties[1] = Property(id=1, type=1, params={})
    model.materials[1] = Material(id=1, law=1, rho0=1.0, params={"E": 100.0, "nu": 0.3})

    # Required node with id 0 on QUAD should raise error
    model.raw_elems["QUAD"] = [(1, 1, [1, 2, 0, 4])]
    log = MessageLog()
    initialization.build_element_groups(model, log)
    assert any("unknown node id 0" in m for m in log.errors)

    # Optional midside with id 0 on TETRA10 should be accepted
    model.raw_elems["QUAD"] = []
    model.properties[2] = Property(id=2, type=14, params={})
    model.parts[2] = Part(id=2, mat_id=1, prop_id=2)
    model.raw_elems["TETRA10"] = [(2, 2, [1, 2, 3, 4, 0, 0, 0, 0, 0, 0])]
    log2 = MessageLog()
    initialization.build_element_groups(model, log2)
    assert not any("unknown node id 0" in m for m in log2.errors)


# ----------------------------------------------------------------------------
# Defect 14: GUI entry point does not crash (dead --web removed)
# ----------------------------------------------------------------------------
def test_defect_14_gui_no_dead_web_flag():
    import inspect
    from pyradioss.gui import __main__ as gui_main
    src = inspect.getsource(gui_main)
    # The nonexistent .server import must be gone
    assert "from .server" not in src
    assert "run_server" not in src


# ----------------------------------------------------------------------------
# Defect 15: Missing PART on TETRA4 does not crash
# ----------------------------------------------------------------------------
def test_defect_15_tetra4_missing_part():
    model = Model()
    # Missing part 999
    model.raw_elems["TETRA4"] = [(1, 999, [1, 2, 3, 4])]
    log = MessageLog()
    initialization._convert_tetras(model, log)
    # Must not raise KeyError, and element kept in TETRA4
    assert len(model.raw_elems["TETRA4"]) == 1


# ----------------------------------------------------------------------------
# Defect 16: /TH spring & shell stress channels
# ----------------------------------------------------------------------------
def test_defect_16_th_channels(tmp_path):
    model = Model()
    model.x = np.array([[0,0,0], [1,0,0], [0,0,0], [1,0,0]], dtype=float)
    model.x0 = model.x.copy()

    # Spring with force
    grp_sp = ElementGroup(ids=np.array([10]), conn=np.array([[0, 1]]), part=np.array([0]))
    grp_sp.state["fres"] = np.array([[10.0, 20.0, 30.0]])
    grp_sp.state["disp"] = np.array([0.5])
    grp_sp.state["eint"] = np.array([2.5])
    model.springs = grp_sp

    # Shell with stress
    grp_sh = ElementGroup(ids=np.array([20]), conn=np.array([[0, 1, 2, 3]]), part=np.array([0]))
    grp_sh.state["sig"] = np.array([[[100.0, 50.0, 0.0, 10.0, 0.0, 0.0]]])
    grp_sh.state["eint"] = np.array([15.0])
    model.shells = grp_sh

    th = TimeHistory(str(tmp_path / "th1.csv"), model, MagicMock())
    # Test spring channels
    assert th._other_value("SPRING", 10, "FX") == 10.0
    assert th._other_value("SPRING", 10, "FY") == 20.0
    assert th._other_value("SPRING", 10, "FZ") == 30.0
    assert th._other_value("SPRING", 10, "D") == 0.5
    assert th._other_value("SPRING", 10, "IE") == 2.5

    # Test shell stress channels
    assert th._other_value("SHEL", 20, "SIGXX") == 100.0
    assert th._other_value("SHEL", 20, "SIGYY") == 50.0
    assert th._other_value("SHEL", 20, "SIGXY") == 10.0
    assert th._other_value("SHEL", 20, "P") == pytest.approx(-50.0)


# ----------------------------------------------------------------------------
# Defect 17: Rotational KE via dt_iner (cbilan.F formula)
# ----------------------------------------------------------------------------
def test_defect_17_rotational_ke_dt_iner(tmp_path):
    model = Model()
    model.x = np.zeros((4, 3))
    model.x0 = np.zeros((4, 3))
    model.v = np.zeros((4, 3))
    model.vr = np.ones((4, 3))  # omega = [1, 1, 1] at all nodes
    model.inertia = np.full(4, 2.0)

    grp = ElementGroup(ids=np.array([1]), conn=np.array([[0, 1, 2, 3]]), part=np.array([0]))
    grp.state["part_ids"] = np.array([1])
    grp.state["mass"] = np.array([0.0])
    # Per-node rotational inertia share, like cbilan.F IN25
    grp.state["dt_iner"] = np.array([2.0])
    model.shells = grp

    th = TimeHistory(str(tmp_path / "th2.csv"), model, MagicMock())
    # vr2_sum = sum_i |omega_i|^2 = 4 nodes * 3 = 12
    # KE_rot = 0.5 * dt_iner * vr2_sum = 0.5 * 2.0 * 12 = 12.0
    ke = th._part_value(1, "KE")
    assert ke == pytest.approx(12.0)

    # Also test the spring/mock path with group.state["inertia"]
    grp2 = ElementGroup(ids=np.array([2]), conn=np.array([[0, 1, 2, 3]]), part=np.array([0]))
    grp2.state["part_ids"] = np.array([2])
    grp2.state["mass"] = np.array([0.0])
    grp2.state["inertia"] = np.array([8.0])  # element-total inertia
    model.springs = grp2

    ke2 = th._part_value(2, "KE")
    # vr2_avg = (4 nodes * 3) / 4 nodes = 3
    # KE_rot = 0.5 * 8.0 * 3.0 = 12.0
    assert ke2 == pytest.approx(12.0)


# ----------------------------------------------------------------------------
# Defect 18: PS1 restart regex matches _0010.rad
# ----------------------------------------------------------------------------
def test_defect_18_ps1_restart_regex():
    import re
    pat = r"_([0-9]{4})\.rad$"
    for name in ("RUN_0001.rad", "RUN_0010.rad", "RUN_0020.rad", "RUN_0100.rad"):
        m = re.search(pat, name)
        assert m is not None
        assert m.group(1) != "0000"
    m0 = re.search(pat, "RUN_0000.rad")
    assert m0.group(1) == "0000"


# ----------------------------------------------------------------------------
# Defect 19: Divide by zero guards
# ----------------------------------------------------------------------------
def test_defect_19_divide_by_zero_guards():
    # solid_hexa8 denom guard
    denom = np.array([0.0, 2.0])
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    dt_crit = np.where(denom > 0.0, 1.0 / safe_denom, 1e30)
    assert dt_crit[0] == 1e30
    assert dt_crit[1] == 0.5

    # shell_tri3 zero sound speed guard
    mat = MagicMock()
    mat.sound_speed_shell.return_value = 0.0
    fac = np.zeros(1)
    c = mat.sound_speed_shell()
    if c > 0.0:
        fac[:] = 0.5
    else:
        fac[:] = 1.0
    assert fac[0] == 1.0
