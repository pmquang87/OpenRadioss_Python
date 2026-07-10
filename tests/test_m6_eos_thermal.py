"""M6 validations: /EOS (polynomial + ideal gas, E-p coupling), the LAW2
adiabatic thermal terms and /FAIL/JOHNSON's D5 (see PORTING_GUIDE.md)."""

import contextlib
import io
import re

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.materials import eos as eos_mod
from pyradioss.materials import law02_johnson_cook as law02
from pyradioss.model.entities import EquationOfState, Material
from pyradioss.starter.starter import run_starter


def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model, e.replace("_0001.rad", "_0001.out")


def _gas(gamma=1.4, p0=0.1e-3, rho0=1.2e-9):
    """An ideal-gas EOS entity (units mm/ms/kg: p in GPa)."""
    return EquationOfState(
        kind="IDEAL-GAS", rho0=rho0,
        params={"c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0,
                "c4": gamma - 1.0, "c5": gamma - 1.0,
                "e0": p0 / (gamma - 1.0), "gamma": gamma})


# ----------------------------------------------------------------------------
# EOS unit tests (closed-form)
# ----------------------------------------------------------------------------

def test_eos_update_solves_the_implicit_pair_exactly():
    """The implicit E-p update must satisfy BOTH of its defining
    equations algebraically: E = E0 + dE - (p0+p)/2 dv and p = A + B E."""
    eos = EquationOfState(
        kind="POLYNOMIAL", rho0=7.8e-6,
        params={"c0": 0.01, "c1": 100.0, "c2": 50.0, "c3": 20.0,
                "c4": 0.5, "c5": 0.3, "e0": 0.002})
    mu = np.array([0.08, -0.03, 0.2])
    dv = np.array([-0.01, 0.005, -0.03])
    e_old = np.array([0.002, 0.001, 0.01])
    p_old = np.array([0.02, 0.005, 0.15])
    de = np.array([1e-4, 0.0, 5e-4])
    p_new, e_new, c2 = eos_mod.update(eos, mu, dv, e_old, p_old, de)
    A, B = eos_mod.coefficients(eos, mu)
    assert np.allclose(p_new, A + B * e_new, rtol=1e-14)
    assert np.allclose(
        e_new, e_old + de - 0.5 * dv * (p_old + p_new), rtol=1e-12)
    assert np.all(c2 >= 0.0)
    # tension drops the quadratic term: A(mu<0) has no C2 contribution
    A_t, _ = eos_mod.coefficients(eos, np.array([-0.03]))
    assert A_t[0] == pytest.approx(0.01 + 100.0 * -0.03 + 20.0 * (-0.03) ** 3)


def test_eos_ideal_gas_isentrope_and_sound_speed():
    """Integrating the implicit update along a slow compression must
    track the ideal-gas adiabat p V^gamma = const, and the returned
    sound speed must equal gamma p / rho."""
    gamma = 1.4
    eos = _gas(gamma=gamma, p0=0.1e-3)
    e = np.array([eos.params["e0"]])
    p = np.array([0.1e-3])
    n = 2000
    js = np.linspace(1.0, 0.5, n + 1)          # compress to half volume
    for k in range(n):
        j_new = js[k + 1]
        mu = 1.0 / j_new - 1.0
        dv = np.array([j_new - js[k]])
        p, e, c2 = eos_mod.update(eos, np.array([mu]), dv, e, p,
                                  np.zeros(1))
    p_exact = 0.1e-3 * (1.0 / 0.5) ** gamma    # p0 (V0/V)^gamma
    assert p[0] == pytest.approx(p_exact, rel=1e-5)
    rho = eos.rho0 / 0.5
    assert c2[0] == pytest.approx(gamma * p[0] / rho, rel=1e-9)


def test_eos_hugoniot_point_consistency():
    """Rankine-Hugoniot check: for a jump to compression mu, the energy
    E_H = (p+p0)(v0-v)/2 (per V0: (p+p0)(1-J)/2) and the EOS p(mu, E_H)
    must agree with the closed-form ideal-gas Hugoniot pressure

        p_H = p0 [ (gamma+1) - (gamma-1) J' ] / [ (gamma+1) J' - (gamma-1) ]

    with J' = V/V0. The EOS evaluated AT the Hugoniot energy must return
    exactly the Hugoniot pressure (this is what the q-heating buys)."""
    gamma, p0 = 1.4, 0.1e-3
    eos = _gas(gamma=gamma, p0=p0)
    J = 0.7
    mu = 1.0 / J - 1.0
    p_h = p0 * ((gamma + 1) - (gamma - 1) * J) / ((gamma + 1) * J
                                                  - (gamma - 1))
    e_h = eos.params["e0"] + 0.5 * (p_h + p0) * (1.0 - J)
    A, B = eos_mod.coefficients(eos, np.array([mu]))
    assert A[0] + B[0] * e_h == pytest.approx(p_h, rel=1e-12)


# ----------------------------------------------------------------------------
# LAW2 thermal terms (closed-form unit tests)
# ----------------------------------------------------------------------------

def _thermal_steel():
    return Material(
        id=1, law=2, rho0=7.8e-6,
        params={"E": 210.0, "nu": 0.3, "A": 0.2, "B": 0.0, "n": 1.0,
                "eps_p_max": 1e30, "sig_max": 1e30, "c": 0.0,
                "eps_dot_0": 1.0, "mT": 1.0, "T_melt": 1798.0,
                "rho_cp": 3.6e-6, "T_i": 298.0})


def test_law2_adiabatic_heating_rate():
    """dT/d(eps_p) = sigma_y / rho_Cp during plastic flow (the closed
    form of the adiabatic update): pull one element far into the flow
    and compare the accumulated temperature rise."""
    mat = _thermal_steel()
    n = 1
    sig = np.zeros((n, 6))
    epsp = np.zeros(n)
    temp = np.zeros(n)
    deps = np.zeros((n, 6))
    d = 1e-4
    deps[:, 0], deps[:, 1], deps[:, 2] = d, -d / 2, -d / 2  # deviatoric
    for _ in range(200):
        law02.solid_update(mat, sig, deps, epsp, 1.0, {"temp": temp})
    assert epsp[0] > 0.01
    # perfect plasticity (B = 0), small T*: T ~ (A/rho_cp) * eps_p
    # (the softening feeds back at O(T*) — evaluate the exact integral:
    #  dT/dep = A(1 - T/(Tm-Ti))/rho_cp  ->  T = DT*(1 - exp(-A ep/(rho_cp DT))))
    DT = mat.params["T_melt"] - mat.params["T_i"]
    t_exact = DT * (1.0 - np.exp(-mat.params["A"] * epsp[0]
                                 / (mat.params["rho_cp"] * DT)))
    assert temp[0] == pytest.approx(t_exact, rel=2e-3)


def test_law2_thermal_softening_factor():
    """At a prescribed temperature rise the yield stress must read
    sigma_y = A (1 - T*^m) exactly (m = 1 here): load one element to
    yield at T* = 0.5 and check the flow stress halves."""
    mat = _thermal_steel()
    n = 1
    sig = np.zeros((n, 6))
    epsp = np.zeros(n)
    DT = mat.params["T_melt"] - mat.params["T_i"]
    temp = np.array([0.5 * DT])                 # T* = 0.5
    deps = np.zeros((n, 6))
    d = 1e-3
    deps[:, 0], deps[:, 1], deps[:, 2] = d, -d / 2, -d / 2
    for _ in range(50):
        law02.solid_update(mat, sig, deps, epsp, 1.0,
                           {"temp": temp.copy()})   # freeze T
    s = sig[0]
    p = (s[0] + s[1] + s[2]) / 3
    vm = np.sqrt(1.5 * ((s[0] - p) ** 2 + (s[1] - p) ** 2
                        + (s[2] - p) ** 2))
    assert vm == pytest.approx(0.5 * mat.params["A"], rel=1e-6)


# ----------------------------------------------------------------------------
# Full starter+engine validation: quasi-static ideal-gas compression
# ----------------------------------------------------------------------------

def test_ideal_gas_piston_matches_adiabat(make_deck):
    """A gas-filled brick compressed to 60% volume by an /IMPDISP piston
    (all other faces confined): the pressure must land on the analytic
    adiabat p = p0 (V0/V)^gamma and the balance must close. This is the
    'ideal-gas compression matching the analytic p(V)' acceptance test
    of the milestone."""
    gamma, p0 = 1.4, 1.0e-4          # 1 bar in GPa
    starter = (
        "/BEGIN\ngas piston\n"
        "/NODE\n"
        "1 0 0 0\n2 10 0 0\n3 10 10 0\n4 0 10 0\n"
        "5 0 0 10\n6 10 0 10\n7 10 10 10\n8 0 10 10\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ngas\n1 1\n"
        # a soft LAW1 host: the EOS replaces its pressure; tiny E keeps a
        # sliver of shear stiffness for stability of the box modes
        "/MAT/LAW1/1\ngas host\n1.2e-9\n1e-6 0.\n"
        f"/EOS/IDEAL-GAS/1\n{gamma} {p0}\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/NODE/1\nall\n1 2 3 4 5 6 7 8\n"
        "/BCS/1\nconfine yz\n011 000 0 1\n"
        "/GRNOD/NODE/2\nfixed face\n1 4 5 8\n"
        "/BCS/2\nx0\n100 000 0 2\n"
        "/GRNOD/NODE/3\npiston\n2 3 6 7\n"
        "/FUNCT/1\nramp then hold\n0.0 0.0\n8.0 -4.0\n100.0 -4.0\n"
        "/IMPDISP/1\npush\n1 X 3\n"
        "/END\n"
    )
    engine = "/RUN/GAS/1\n10.0\n/DT\n0.9 0\n/PRINT/-20000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "GAS", starter, engine)
    st = model.bricks.state
    J = 0.6                                       # 10->6 mm
    p_exact = p0 * (1.0 / J) ** gamma
    assert st["p_eos"][0] == pytest.approx(p_exact, rel=2e-3)
    # the stress state is -p on the diagonal (the tiny shear host adds
    # its deviator, orders of magnitude smaller)
    assert -st["sig"][0, :3].mean() == pytest.approx(p_exact, rel=2e-3)
    # energy: E per V0 on the adiabat = e0 * (V0/V)^(gamma-1) / ... =
    # p/(gamma-1) * J  (E_total/V0 with p at J)
    e_exact = p_exact * J / (gamma - 1.0)
    assert st["e_eos"][0] == pytest.approx(e_exact, rel=2e-3)
    text = open(out).read()
    err = float(re.search(r"ENERGY ERROR\s*\.[ .]*:\s*([-\d.Ee+]+)",
                          text).group(1))
    assert "ENGINE TERMINATION : NORMAL" in text
    assert abs(err) < 1.0


def test_fail_johnson_d5_raises_failure_strain():
    """D5 > 0 with a hot point must scale eps_f by (1 + D5 T*): drive
    identical damage steps with T* = 0 and T* = 0.5 and compare the
    accumulated damage ratio."""
    from pyradioss.failure import johnson
    from pyradioss.model.entities import FailureModel
    fail = FailureModel(type="JOHNSON", params={
        "D1": 0.2, "D2": 0.0, "D3": 0.0, "D4": 0.0, "D5": 0.8,
        "eps_dot_0": 1.0})
    sig = np.array([[0.3, 0.0, 0.0, 0.0, 0.0, 0.0]])
    deps = np.zeros((1, 6))
    d_ep = np.array([0.01])
    dama_cold = np.zeros(1)
    dama_hot = np.zeros(1)
    johnson.solid_step(fail, sig, d_ep, deps, 1.0, dama_cold, None)
    johnson.solid_step(fail, sig, d_ep, deps, 1.0, dama_hot,
                       np.array([0.5]))
    assert dama_cold[0] == pytest.approx(0.01 / 0.2, rel=1e-12)
    assert dama_hot[0] == pytest.approx(0.01 / (0.2 * 1.4), rel=1e-12)
