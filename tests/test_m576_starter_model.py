"""Starter diagnostics and model entity unit tests for /MAT/LAW105 (M576).

Validates:
  - MaterialLaw105 entity, properties, aliases (MatLaw105, MatPowderBurn, MaterialPowderBurn).
  - Model container alias model.mat_powder_burns.
  - Diagnostic validation checks (hm_read_mat105.F90):
      - RHO0 <= 0 -> ANCMSG 1514
      - BULK <= 0 -> ANCMSG 856
      - GAS_D <= 0 -> ANCMSG 856
      - GAS_EG <= 0 -> ANCMSG 856
      - Gr <= 0 -> ANCMSG 856
      - C1 <= 0 -> ANCMSG 856
      - Shell element compatibility rejection -> ANCMSG 305
      - 1D element compatibility rejection -> ANCMSG 306
"""

import pytest

from pyradioss.model.entities import (
    MaterialLaw105,
    MatLaw105,
    MatPowderBurn,
    MaterialPowderBurn,
    Part,
    Property,
)
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.checks import check_mat_law105, check_materials


def test_law105_model_entities_and_properties():
    """Verify MaterialLaw105 creation, aliases, and physical property shortcuts."""
    assert MatLaw105 is MaterialLaw105
    assert MatPowderBurn is MaterialLaw105
    assert MaterialPowderBurn is MaterialLaw105

    m = MaterialLaw105(
        id=105,
        title="BlackPowder",
        rho0=1.5e-6,
        bulk=15000.0,
        p0=0.1,
        psh=0.0,
        gas_d=0.5e-6,
        gas_eg=4000.0,
        gr=0.08,
        c=0.75,
        alpha=1.0,
        c1=2500.0,
    )

    assert m.id == 105
    assert m.title == "BlackPowder"
    assert m.rho == pytest.approx(1.5e-6)
    assert m.K == pytest.approx(15000.0)
    assert m.young == pytest.approx(1.2 * 15000.0)
    assert m.nu == pytest.approx(0.3)
    assert m.G == pytest.approx((6.0 / 13.0) * 15000.0)
    assert m.sound_speed == pytest.approx((15000.0 / 1.5e-6) ** 0.5)

    model = Model()
    model.mat_law105s[105] = m
    assert 105 in model.mat_powder_burns
    assert model.mat_powder_burns[105] is m


def test_law105_starter_checks_ancmsg856_parameters():
    """Verify ANCMSG 856 triggers for missing/non-positive required parameters."""
    log = MessageLog()

    # 1. Missing BULK
    m_bad_bulk = MaterialLaw105(id=1, rho0=1.6e-6, bulk=0.0, gas_d=1e-6, gas_eg=1000.0, gr=0.1, c1=100.0)
    check_mat_law105(m_bad_bulk, log)
    assert any("BULK MODULUS MUST BE DEFINED" in str(e) and "856" in str(e) for e in log.errors)

    # 2. Missing GAS_D
    log = MessageLog()
    m_bad_d = MaterialLaw105(id=2, rho0=1.6e-6, bulk=1000.0, gas_d=0.0, gas_eg=1000.0, gr=0.1, c1=100.0)
    check_mat_law105(m_bad_d, log)
    assert any("GAS EOS PARAMETER D MUST BE DEFINED" in str(e) and "856" in str(e) for e in log.errors)

    # 3. Missing GAS_EG
    log = MessageLog()
    m_bad_eg = MaterialLaw105(id=3, rho0=1.6e-6, bulk=1000.0, gas_d=1e-6, gas_eg=0.0, gr=0.1, c1=100.0)
    check_mat_law105(m_bad_eg, log)
    assert any("GAS EOS PARAMETER EG MUST BE DEFINED" in str(e) and "856" in str(e) for e in log.errors)

    # 4. Missing Gr
    log = MessageLog()
    m_bad_gr = MaterialLaw105(id=4, rho0=1.6e-6, bulk=1000.0, gas_d=1e-6, gas_eg=1000.0, gr=0.0, c1=100.0)
    check_mat_law105(m_bad_gr, log)
    assert any("GROWTH PARAMETER Gr MUST BE DEFINED" in str(e) and "856" in str(e) for e in log.errors)

    # 5. Missing C1
    log = MessageLog()
    m_bad_c1 = MaterialLaw105(id=5, rho0=1.6e-6, bulk=1000.0, gas_d=1e-6, gas_eg=1000.0, gr=0.1, c1=0.0)
    check_mat_law105(m_bad_c1, log)
    assert any("BURNING VELOCITY C1 MUST BE DEFINED" in str(e) and "856" in str(e) for e in log.errors)


def test_law105_starter_checks_density():
    """Verify zero or negative density triggers ANCMSG 1514."""
    log = MessageLog()
    m_bad_rho = MaterialLaw105(id=6, rho0=0.0, bulk=1000.0, gas_d=1e-6, gas_eg=1000.0, gr=0.1, c1=100.0)
    check_mat_law105(m_bad_rho, log)
    assert any("ANCMSG 1514" in str(e) for e in log.errors)


def test_law105_starter_checks_element_compatibility():
    """Verify rejection of incompatible 2D shells (ANCMSG 305) and 1D elements (ANCMSG 306)."""
    # 1. Shell part rejection
    model_shell = Model()
    m = MaterialLaw105(id=1, rho0=1.5e-6, bulk=5000.0, gas_d=0.5e-6, gas_eg=3000.0, gr=0.1, c1=1000.0)
    model_shell.materials[1] = m
    model_shell.mat_law105s[1] = m
    p_shell = Part(id=1, mat_id=1, prop_id=1)
    p_shell.element_type = "SHELL"
    model_shell.parts[1] = p_shell
    model_shell.properties[1] = Property(id=1, type="SHELL")

    log_shell = MessageLog()
    check_mat_law105(m, model_shell, log_shell)
    assert any("ANCMSG 305" in str(e) and "shell" in str(e).lower() for e in log_shell.errors)

    # 2. 1D truss/beam/spring rejection
    model_1d = Model()
    model_1d.materials[1] = m
    model_1d.mat_law105s[1] = m
    p_beam = Part(id=2, mat_id=1, prop_id=2)
    p_beam.element_type = "BEAM"
    model_1d.parts[2] = p_beam
    model_1d.properties[2] = Property(id=2, type="TYPE3")

    log_1d = MessageLog()
    check_mat_law105(m, model_1d, log_1d)
    assert any("ANCMSG 306" in str(e) and "1d" in str(e).lower() for e in log_1d.errors)


def test_law105_starter_checks_valid_solid():
    """Verify valid solid element with LAW105 passes checks without errors."""
    model_solid = Model()
    m = MaterialLaw105(id=1, rho0=1.5e-6, bulk=5000.0, gas_d=0.5e-6, gas_eg=3000.0, gr=0.1, c1=1000.0)
    model_solid.materials[1] = m
    model_solid.mat_law105s[1] = m
    p_solid = Part(id=1, mat_id=1, prop_id=1)
    p_solid.element_type = "SOLID"
    model_solid.parts[1] = p_solid
    model_solid.properties[1] = Property(id=1, type="TYPE14")

    log = MessageLog()
    check_mat_law105(m, model_solid, log)
    assert not log.errors
