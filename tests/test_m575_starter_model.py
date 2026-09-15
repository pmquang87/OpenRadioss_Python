"""
Tests for /MAT/LAW106 starter model entity, aliases, and checks (Milestone M575).
"""

import pytest
from pyradioss.model.entities import (
    MaterialLaw106,
    MatLaw106,
    MatJCookAlm,
    MatJohnsCookAlm,
    Part,
)
from pyradioss.model.model import Model
from pyradioss.common.messages import MessageLog
from pyradioss.starter.checks import check_mat_law106


def test_mat_law106_entity_and_aliases():
    """Verify entity creation, aliases, and computed properties."""
    assert MatLaw106 is MaterialLaw106
    assert MatJCookAlm is MaterialLaw106
    assert MatJohnsCookAlm is MaterialLaw106

    m = MaterialLaw106(
        id=10,
        title="Ti6Al4V_ALM",
        rho0=4.43e-6,
        rhor=4.43e-6,
        young=110000.0,
        nu=0.34,
        sigy=850.0,
        beta=400.0,
        hard_n=0.45,
        ep_max=0.18,
        sig_max=1200.0,
        fcut=0.0,
        vp=2,
        nmax=5,
        tol=1e-5,
        cjc=0.015,
        deps0=1.0,
        m=1.05,
        tmelt=1923.0,
        spheat=2.35e6,
        eta=0.9,
        t0=293.0,
        tr=293.0,
    )

    assert m.id == 10
    assert m.title == "Ti6Al4V_ALM"
    assert m.rho == 4.43e-6
    assert m.e == 110000.0
    assert m.E == 110000.0
    assert m.a == 850.0
    assert m.b == 400.0
    assert m.n == 0.45
    assert m.eps_max == 0.18
    assert m.sigma_max == 1200.0
    assert m.cs == 2.35e6
    assert m.tref == 293.0
    assert m.G > 0.0
    assert m.K > 0.0
    assert m.sound_speed > m.sound_speed_shell


def test_model_dict_aliases():
    """Verify Model has mat_jcook_alms and mat_johns_cook_alms pointing to mat_law106s."""
    model = Model()
    assert model.mat_jcook_alms is model.mat_law106s
    assert model.mat_johns_cook_alms is model.mat_law106s

    m = MaterialLaw106(id=1, young=100000.0, nu=0.3, rho0=7.8e-6)
    model.mat_law106s[1] = m
    assert 1 in model.mat_jcook_alms
    assert 1 in model.mat_johns_cook_alms


def test_check_mat_law106_valid():
    """Verify no errors or warnings for a completely valid LAW106 material."""
    model = Model()
    log = MessageLog()

    m = MaterialLaw106(
        id=1,
        title="Valid_Law106",
        rho0=7.8e-6,
        rhor=7.8e-6,
        young=210000.0,
        nu=0.3,
        sigy=400.0,
        tmelt=1800.0,
        tr=300.0,
    )
    model.mat_law106s[1] = m
    model.materials[1] = m

    check_mat_law106(m, model, log)
    assert not log.has_errors
    assert not log.has_warnings


def test_check_mat_law106_ancmsg_diagnostics():
    """Verify ANCMSG diagnostics for invalid density, young, nu, and temperature."""
    # 1. Invalid density (ANCMSG 1514)
    model = Model()
    log = MessageLog()
    m_bad_rho = MaterialLaw106(id=1, rho0=-1.0, young=200000.0, nu=0.3)
    check_mat_law106(m_bad_rho, model, log)
    assert any("ANCMSG 1514" in str(err) or "density" in str(err).lower() for err in log.errors)

    # 2. Invalid Young's modulus (ANCMSG 276)
    model = Model()
    log = MessageLog()
    m_bad_e = MaterialLaw106(id=2, rho0=7.8e-6, young=0.0, nu=0.3)
    check_mat_law106(m_bad_e, model, log)
    assert any("ANCMSG 276" in str(err) or "young" in str(err).lower() for err in log.errors)

    # 3. Invalid Poisson's ratio (ANCMSG 300)
    model = Model()
    log = MessageLog()
    m_bad_nu = MaterialLaw106(id=3, rho0=7.8e-6, young=200000.0, nu=0.55)
    check_mat_law106(m_bad_nu, model, log)
    assert any("ANCMSG 300" in str(err) or "poisson" in str(err).lower() for err in log.errors)

    # 4. 1D truss/beam element rejection (ANCMSG 306)
    model = Model()
    log = MessageLog()
    m_valid = MaterialLaw106(id=4, rho0=7.8e-6, young=200000.0, nu=0.3)
    model.mat_law106s[4] = m_valid
    part_truss = Part(id=10, mat_id=4, prop_id=100)
    part_truss.element_type = "TRUSS"
    model.parts[10] = part_truss
    check_mat_law106(m_valid, model, log)
    assert any("ANCMSG 306" in str(err) or "1d" in str(err).lower() for err in log.errors)
