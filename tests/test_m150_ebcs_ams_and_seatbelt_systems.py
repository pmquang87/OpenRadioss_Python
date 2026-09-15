"""Tests for Milestone M150:
- Eulerian Boundary Conditions Suite (/EBCS/PRES, /EBCS/VEL, /EBCS/INLET, /EBCS/FLUXOUT, /EBCS/GRADP0, /EBCS/NORMV, /EBCS/VALVIN, /EBCS/VALVOUT, /EBCS/MONVOL)
- Sliding Wall Boundary Conditions (/BCS/WALL)
- Advanced Mass Scaling Starter Directive (/AMS)
- Seatbelt Retractor, Slipring & System Coupling (/RETRACTOR, /SLIPRING, /SEATBELT)
- Extended Time-History Channels (/TH/RETRACTOR, /TH/SLIPRING, /TH/EBCS)
- Model Cross-Reference Validation
"""
import pytest
from pathlib import Path
from pyradioss.common.messages import MessageLog
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.model.model import Model
from pyradioss.starter.initialization import resolve_materials, build_element_groups, resolve_node_groups
from pyradioss.starter.checks import check_model


def _parse_starter(tmp_path: Path, deck_text: str, filename: str = "test_0000.rad"):
    file_path = tmp_path / filename
    file_path.write_text(deck_text, encoding="utf-8")
    blocks = read_deck(str(file_path))
    log = MessageLog()
    model = Model()
    parse_starter_deck(blocks, model, log)
    return model, log


def test_ebcs_pres_and_vel_fixed_format(tmp_path):
    """Test fixed-format parsing of /EBCS/PRES and /EBCS/VEL."""
    deck = (
        "/BEGIN\n"
        "EBCS TEST FIXED\n"
        "/NODE/1\n"
        "1 0.0 0.0 0.0\n"
        "/NODE/2\n"
        "2 1.0 0.0 0.0\n"
        "/NODE/3\n"
        "3 1.0 1.0 0.0\n"
        "/SURF/SEG/1\n"
        "Surf 1\n"
        "1 2 3\n"
        "/FUNCT/1\n"
        "Pressure Curve\n"
        "0.0 100.0\n"
        "1.0 200.0\n"
        "/FUNCT/2\n"
        "Density Curve\n"
        "0.0 1.2\n"
        "1.0 1.2\n"
        "/EBCS/PRES/1\n"
        "Eulerian Imposed Pressure\n"
        "         1\n"
        "               340.0\n"
        "         1               1.0\n"
        "         2               1.2\n"
        "         0               0.0\n"
        "                 0.1                 1.0                 2.0\n"
        "/EBCS/VEL/2\n"
        "Eulerian Imposed Velocity\n"
        "         1\n"
        "               340.0\n"
        "         1               50.0\n"
        "         0                0.0\n"
        "         0                0.0\n"
        "         2                1.2\n"
        "         0                0.0\n"
        "                 0.1                 1.0                 2.0\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.ebcs_pres
    p = model.ebcs_pres[1]
    assert p.surf_id == 1
    assert p.c == pytest.approx(340.0)
    assert p.fct_pres == 1
    assert p.scale_pres == pytest.approx(1.0)
    assert p.fct_rho == 2
    assert p.scale_rho == pytest.approx(1.2)
    assert p.lcar == pytest.approx(0.1)
    assert p.r1 == pytest.approx(1.0)
    assert p.r2 == pytest.approx(2.0)

    assert 2 in model.ebcs_vel
    v = model.ebcs_vel[2]
    assert v.surf_id == 1
    assert v.c == pytest.approx(340.0)
    assert v.fct_vx == 1
    assert v.scale_vx == pytest.approx(50.0)
    assert v.fct_rho == 2
    assert v.scale_rho == pytest.approx(1.2)


def test_ebcs_inlet_fluxout_gradp0_normv_valves(tmp_path):
    """Test parsing of /EBCS/INLET, /EBCS/FLUXOUT, /EBCS/GRADP0, /EBCS/NORMV, /EBCS/VALVIN, /EBCS/VALVOUT, /EBCS/MONVOL."""
    deck = (
        "/BEGIN\n"
        "EBCS SUITE FREE\n"
        "/NODE/1\n"
        "1 0.0 0.0 0.0\n"
        "/NODE/2\n"
        "2 1.0 0.0 0.0\n"
        "/NODE/3\n"
        "3 1.0 1.0 0.0\n"
        "/SURF/SEG/1\n"
        "Surf 1\n"
        "1 2 3\n"
        "/EBCS/INLET/1\n"
        "Inflow BCS\n"
        "1 1.25 100.0 0.0 0.0 2.5e5 1\n"
        "/EBCS/FLUXOUT/2\n"
        "Outflow BCS\n"
        "1 1.013e5\n"
        "/EBCS/GRADP0/3\n"
        "Zero Grad P BCS\n"
        "1\n"
        "/EBCS/NORMV/4\n"
        "Normal Velocity Constraint\n"
        "1 25.0 2\n"
        "/EBCS/VALVIN/5\n"
        "Inlet Valve\n"
        "1 1.5e5 1.0e5\n"
        "/EBCS/VALVOUT/6\n"
        "Outlet Valve\n"
        "1 2.0e5 1.2e5\n"
        "/EBCS/MONVOL/7\n"
        "Eulerian Monitored Volume\n"
        "1 10\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.ebcs_inlets
    inl = model.ebcs_inlets[1]
    assert inl.surf_id == 1
    assert inl.density == pytest.approx(1.25)
    assert inl.vx == pytest.approx(100.0)
    assert inl.energy == pytest.approx(2.5e5)

    assert 2 in model.ebcs_fluxouts
    flx = model.ebcs_fluxouts[2]
    assert flx.surf_id == 1
    assert flx.p_ext == pytest.approx(1.013e5)

    assert 3 in model.ebcs_gradp0
    assert model.ebcs_gradp0[3].surf_id == 1

    assert 4 in model.ebcs_normv
    assert model.ebcs_normv[4].vn == pytest.approx(25.0)

    assert 5 in model.ebcs_valves
    v_in = model.ebcs_valves[5]
    assert not v_in.is_out
    assert v_in.p_open == pytest.approx(1.5e5)

    assert 6 in model.ebcs_valves
    v_out = model.ebcs_valves[6]
    assert v_out.is_out
    assert v_out.p_open == pytest.approx(2.0e5)

    assert 7 in model.ebcs_monvols
    assert model.ebcs_monvols[7].monvol_id == 10


def test_bcs_wall_and_ams_starter_controls(tmp_path):
    """Test /BCS/WALL sliding wall boundary conditions and /AMS advanced mass scaling."""
    deck = (
        "/BEGIN\n"
        "BCS WALL AND AMS TEST\n"
        "/NODE/1\n"
        "1 0.0 0.0 0.0\n"
        "/NODE/2\n"
        "2 1.0 0.0 0.0\n"
        "/GRNOD/NODE/1\n"
        "Wall Node Group\n"
        "1 2\n"
        "/SENSOR/TIME/1\n"
        "Sensor 1\n"
        "0.005\n"
        "/BCS/WALL/1\n"
        "Sliding Wall Boundary Condition\n"
        "         1         1\n"
        "                 0.0                 0.1\n"
        "/AMS\n"
        "         1\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.bcs_walls
    w = model.bcs_walls[1]
    assert w.grnod_id == 1
    assert w.sens_id == 1
    assert w.tstart == pytest.approx(0.0)
    assert w.tstop == pytest.approx(0.1)

    assert model.ams_control is not None
    assert model.ams_control.grpart_id == 1


def test_seatbelt_system_coupling_and_time_history(tmp_path):
    """Test /SEATBELT system assembly, /RETRACTOR, /SLIPRING and /TH output channels."""
    deck = (
        "/BEGIN\n"
        "SEATBELT SYSTEM TEST\n"
        "/NODE/1\n"
        "1 0.0 0.0 0.0\n"
        "/NODE/2\n"
        "2 1.0 0.0 0.0\n"
        "/NODE/3\n"
        "3 2.0 0.0 0.0\n"
        "/PROP/SPRING/1\n"
        "Spring Prop\n"
        "0.0 100.0 0.0\n"
        "/MAT/PLAS_JOHNS/1/1\n"
        "Steel\n"
        "7.85e-6\n"
        "210.0 0.3\n"
        "0.2 0.1 0.5\n"
        "/SPRING/1/1/1\n"
        "1 1 2\n"
        "/SPRING/1/1/2\n"
        "2 2 3\n"
        "/PART/1\n"
        "Part 1\n"
        "1 1\n"
        "/RETRACTOR/SPRING/1\n"
        "Retractor 1\n"
        "1 1 0.05\n"
        "0 0.25 0 0 1.0 1.0\n"
        "0 1 1000.0 0 1.0 1.0\n"
        "/SLIPRING/SPRING/1\n"
        "Slipring 1\n"
        "1 2 2 3 0 0 0.2 0.01\n"
        "0 0 0.15 1.0 1.0 1.0\n"
        "0 0 0.20 1.0 1.0 1.0\n"
        "/SEATBELT/1\n"
        "Seatbelt Assembly 1\n"
        "1 1 1 2\n"
        "/TH/RETRACTOR/1\n"
        "Retractor TH\n"
        "PULL FORC LOCK\n"
        "1\n"
        "/TH/SLIPRING/1\n"
        "Slipring TH\n"
        "SLIP FRIC TRAT\n"
        "1\n"
        "/TH/EBCS/1\n"
        "EBCS TH\n"
        "MASS FLOW ENER\n"
        "1\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    assert len(log.errors) == 0
    assert 1 in model.retractors
    assert 1 in model.sliprings
    assert 1 in model.seatbelt_systems
    sb = model.seatbelt_systems[1]
    assert 1 in sb.retractor_ids
    assert 1 in sb.slipring_ids
    assert len(model.th_requests) == 3


def test_cross_reference_validation(tmp_path):
    """Test cross-reference checks in checks.py for M150 entities."""
    deck = (
        "/BEGIN\n"
        "CROSS REF VALIDATION\n"
        "/NODE/1\n"
        "1 0.0 0.0 0.0\n"
        "/NODE/2\n"
        "2 1.0 0.0 0.0\n"
        "/NODE/3\n"
        "3 2.0 0.0 0.0\n"
        "/GRNOD/NODE/1\n"
        "Group 1\n"
        "1 2\n"
        "/SURF/SEG/1\n"
        "Surf 1\n"
        "1 2 3\n"
        "/PROP/SPRING/1\n"
        "Spring Prop\n"
        "0.0 100.0 0.0\n"
        "/MAT/PLAS_JOHNS/1/1\n"
        "Steel\n"
        "7.85e-6\n"
        "210.0 0.3\n"
        "0.2 0.1 0.5\n"
        "/SPRING/1/1/1\n"
        "1 1 2\n"
        "/SPRING/1/1/2\n"
        "2 2 3\n"
        "/PART/1\n"
        "Part 1\n"
        "1 1\n"
        "/BCS/WALL/1\n"
        "Wall BCS\n"
        "1 0 0.0 1.0\n"
        "/EBCS/PRES/1\n"
        "EBCS Pres\n"
        "1 340.0 0 1.0 0 1.0 0 1.0 0.1 1.0 2.0\n"
        "/RETRACTOR/SPRING/1\n"
        "Retractor 1\n"
        "1 1 0.05\n"
        "/SLIPRING/SPRING/1\n"
        "Slipring 1\n"
        "1 2 2 3 0 0 0.2 0.01\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    check_model(model, log)
    assert len(log.errors) == 0


def test_cross_reference_missing_entities(tmp_path):
    """Test that missing referenced surfaces/sensors/curves/etc. produce errors."""
    deck = (
        "/BEGIN\n"
        "CROSS REF MISSING\n"
        "/EBCS/PRES/1\n"
        "EBCS Pres missing surf & funct\n"
        "99 340.0 88 1.0 77 1.0 66 1.0 0.1 1.0 2.0\n"
        "/BCS/WALL/1\n"
        "Wall BCS missing grnod & sensor\n"
        "99 88 0.0 1.0\n"
        "/AMS\n"
        "99\n"
        "/SEATBELT/1\n"
        "Missing Retractor and Slipring\n"
        "99 88 1 2\n"
        "/END\n"
    )
    model, log = _parse_starter(tmp_path, deck)
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    check_model(model, log)
    assert len(log.errors) > 0
    err_msgs = " ".join(log.errors)
    assert "/EBCS/PRES/1" in err_msgs
    assert "/BCS/WALL/1" in err_msgs
    assert "/AMS" in err_msgs
    assert "/SEATBELT/1" in err_msgs
