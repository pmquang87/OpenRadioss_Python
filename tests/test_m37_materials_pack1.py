"""M37 material physics pack 1: /MAT/VOID, /MAT/GAS, LAW70 (tabulated
visco-elastic foam), LAW35 (visco-elastic foam), LAW40 (/MAT/KELVINMAX
generalized Kelvin-Maxwell — the law the 'Kelvin-Maxwell' corpus decks
actually use) and LAW44 (Cowper-Symonds).

Analytic single-element checks against closed forms, frame-indifference
(objectivity) spins, sound-speed/dt sanity, cfg-driven deck parsing and
the official-corpus decks (skipped when the corpus extract is absent).
"""

import os

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.common.tables import FunctTable
from pyradioss.elements import solid_hexa8
from pyradioss.input.deck_reader import read_deck
from pyradioss.input import mat_reader
from pyradioss.input.mat_reader import GenericMaterialRecord
from pyradioss.input.starter_keywords import parse_starter_deck

# Deck-level tests here parse /MAT cards through the cfg-driven generic
# reader, which needs OpenRadioss's hm_cfg_files (not vendored — license;
# CI fetches a sparse checkout, ci.yml). Skip loudly without them.
pytestmark = pytest.mark.skipif(
    mat_reader.catalogue().schema("FABRI") is None,
    reason="hm_cfg_files CFG tree not found — set PYRADIOSS_HM_CFG "
           "(see ci.yml / PORTING_GUIDE M37)")
from pyradioss.materials import (eos as eos_mod, law35_kelvinmax,
                                 law40_kelvinmax, law44_cowper,
                                 law70_tabfoam, mat_gas, mat_void)
from pyradioss.model.model import Model
from pyradioss.starter.checks import check_model
from pyradioss.starter.initialization import (build_element_groups,
                                              initialize_elements_and_mass,
                                              resolve_materials,
                                              resolve_node_groups,
                                              resolve_surfaces)

# Official-deck corpus. The repo vendors the small decks these tests need
# under tests/data/rd_decks (see the README there for provenance); point
# PYRADIOSS_RD_DECKS at a full corpus extract to also run the tests whose
# decks are too large to vendor (e.g. the 15 MB RD-E-1300 blast deck).
RD_DECKS = os.environ.get(
    "PYRADIOSS_RD_DECKS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "data", "rd_decks"))
_HAS_CORPUS = os.path.isdir(RD_DECKS)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _f20(*vals):
    """One fixed-format card: 20-char right-justified fields."""
    return "".join(f"{v:>20}" for v in vals)


def _i10(*vals):
    return "".join(f"{v:>10d}" for v in vals)


def _build(deck_text, tmp_path, expect_clean=True):
    f = tmp_path / "K_0000.rad"
    f.write_text(deck_text)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    build_element_groups(model, log)
    resolve_node_groups(model, log)
    resolve_surfaces(model, log)
    initialize_elements_and_mass(model, log)
    if expect_clean:
        assert not log.errors, log.errors
    return model, log


_CUBE = (
    "/NODE\n"
    "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
    "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
    "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
    "/PART/1\ncube\n1 1\n"
    "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
)


class _FnModel:
    """Minimal stand-in for Model in law-level resolve() calls."""

    def __init__(self, functions):
        self.functions = functions


def _rotmat(axis, angle):
    a = np.asarray(axis, dtype=float)
    a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _v2t(v, half_shear):
    s = 0.5 if half_shear else 1.0
    return np.array([[v[0], s * v[3], s * v[5]],
                     [s * v[3], v[1], s * v[4]],
                     [s * v[5], s * v[4], v[2]]])


def _t2v(T, double_shear):
    d = 2.0 if double_shear else 1.0
    return np.array([T[0, 0], T[1, 1], T[2, 2],
                     d * T[0, 1], d * T[1, 2], d * T[0, 2]])


def _rot_deps(R, v):
    """Rotate a Voigt strain (engineering shear)."""
    return _t2v(R @ _v2t(v, True) @ R.T, True)


def _rot_sig(R, v):
    """Rotate a Voigt stress (plain shear)."""
    return _t2v(R @ _v2t(v, False) @ R.T, False)


# ============================================================================
# /MAT/VOID — law 0
# ============================================================================

def _void_mat(rho=7.8e-9, e=210.0, nu=0.3):
    rec = GenericMaterialRecord(law_name="VOID", law_number=0, id=1,
                                density=rho, params={"MAT_E": e, "nu": nu})
    return mat_void.build_void(rec)


def test_void_zero_stress_any_deformation_and_spin():
    """A void element NEVER carries stress — whatever the kernel's
    Jaumann pre-rotation or the strain increment did."""
    mat = _void_mat()
    sig = np.random.default_rng(0).normal(size=(4, 6))   # pre-rotated junk
    out = mat_void.solid_update(mat, sig, None)
    assert np.all(out == 0.0)
    sigsh = np.ones((3, 3))
    mat_void.shell_update(mat, sigsh, None)
    assert np.all(sigsh == 0.0)
    # the card's E/nu exist ONLY for the dt/contact estimate
    K, G = mat.K, mat.G
    assert mat.sound_speed_solid() == pytest.approx(
        np.sqrt((K + 4.0 * G / 3.0) / mat.rho0), rel=1e-12)


def test_void_hexa_deck_mass_but_no_forces(tmp_path):
    """Deck-level: a VOID cube gets its mass (rho*V) but produces zero
    stress and zero internal force under arbitrary straining."""
    deck = (
        "/BEGIN\nvoid cube\n" + _CUBE +
        "/MAT/VOID/1\ndummy\n" + _f20("7.8e-9", "210.0", "0.3") + "\n"
        "/END\n")
    model, _ = _build(deck, tmp_path)
    g = model.bricks
    mat = g.state["slices"][0][1]
    assert mat.law == 0 and not getattr(mat, "inactive", False)
    assert g.state["mass"][0] == pytest.approx(7.8e-9, rel=1e-12)

    v = np.zeros_like(model.x)
    v[:, 0] = 0.1 * model.x[:, 0]           # uniaxial stretching
    v[:, 1] = -0.05 * model.x[:, 2]         # plus some shear
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    for _ in range(10):
        dtc = solid_hexa8.forces(g, model.x, v, model.vr, 1e-4, fint, mint)
    assert np.all(g.state["sig"] == 0.0)
    assert np.abs(fint).max() < 1e-14
    assert np.isfinite(dtc[0]) and dtc[0] > 0.0


# ============================================================================
# /MAT/GAS — law 999 (thermodynamics + EOS semantics on solids)
# ============================================================================

def _gas_rec(subtype, params):
    return GenericMaterialRecord(law_name="GAS", law_number=None, id=1,
                                 density=0.0, subtype=subtype,
                                 params=params)


def test_gas_gamma_all_subtypes():
    """gamma = cp/(cp - R/MW) for MASS; MOLE coefficients are per-mole
    (divided by MW at read time); CSTA derives MW = R/(cp-cv); PREDEF
    uses the hm_read_matgas table."""
    R = mat_gas.R_IGC_SI
    m = mat_gas.build_gas(_gas_rec("MASS", {"MASS": 0.02896,
                                            "ABG_cpai": 1005.0}))
    assert m.gamma(300.0) == pytest.approx(
        1005.0 / (1005.0 - R / 0.02896), rel=1e-13)

    mol = mat_gas.build_gas(_gas_rec("MOLE", {"MASS": 0.02896,
                                              "ABG_cpai": 29.1048}))
    # 29.1048 J/mol/K / 0.02896 kg/mol = 1005.0 J/kg/K: same gas
    assert mol.cp(300.0) == pytest.approx(1005.0, rel=1e-12)
    assert mol.gamma(300.0) == pytest.approx(m.gamma(300.0), rel=1e-12)

    cst = mat_gas.build_gas(_gas_rec("CSTA", {"MAT_BSAT": 1005.0,
                                              "MAT_RSAT": 718.0}))
    assert cst.mw == pytest.approx(R / (1005.0 - 718.0), rel=1e-13)
    assert cst.gamma(300.0) == pytest.approx(1005.0 / 718.0, rel=1e-13)

    air = mat_gas.build_gas(_gas_rec("PREDEF", {"GAS": "AIR"}))
    # NIST-ish air at 300 K: gamma close to 1.4
    assert 1.35 < air.gamma(300.0) < 1.42
    with pytest.raises(ValueError):
        mat_gas.build_gas(_gas_rec("PREDEF", {"GAS": "UNOBTAINIUM"}))


def test_gas_isentropic_compression_exact_relation():
    """Adiabatic compression of the gas EOS must follow the exact
    isentrope p = p0 (V0/V)^gamma, and the sound speed must be
    c^2 = gamma p / rho — the 'EOS semantics' contract of /MAT/GAS on
    solid elements (the element kernels' /EOS block runs exactly this
    update; see solid_hexa8.forces)."""
    gm = mat_gas.build_gas(_gas_rec("MASS", {
        "MASS": 0.02896, "ABG_cpai": 1005.0,
        "P0": 1.0e5, "T0": 293.0, "RHO0": 1.2}))
    mat_gas.resolve_gas(gm)
    assert gm.eos is not None and gm.rho0 == 1.2
    gam = gm.gamma(293.0)
    assert gm.eos.params["gamma"] == pytest.approx(gam, rel=1e-13)

    e, p = np.array([eos_mod.initial_state(gm.eos)[0]]), np.array([1.0e5])
    J = 1.0
    nsteps = 4000
    dJ = (0.6 - 1.0) / nsteps
    for _ in range(nsteps):
        J += dJ
        p, e, c2 = eos_mod.update(gm.eos, np.array([1.0 / J - 1.0]),
                                  np.array([dJ]), e, p, np.array([0.0]))
    p_exact = 1.0e5 * (1.0 / 0.6) ** gam
    assert p[0] == pytest.approx(p_exact, rel=1e-6)
    assert c2[0] == pytest.approx(gam * p[0] / (1.2 / 0.6), rel=1e-9)


def test_gas_zero_deviator():
    """A gas carries no shear: the law strips the deviator and keeps
    only the isotropic part (which the EOS block then replaces)."""
    gm = mat_gas.build_gas(_gas_rec("MASS", {"MASS": 0.028,
                                             "ABG_cpai": 1000.0}))
    sig = np.array([[1.0, 2.0, 3.0, 0.5, -0.5, 0.25]])
    mat_gas.solid_update(gm, sig)
    assert np.allclose(sig[0, :3], 2.0)          # mean kept
    assert np.all(sig[0, 3:] == 0.0)             # shear gone


def test_gas_eos_card_supplies_density_and_state(tmp_path):
    """Fixed-dialect /MAT/GAS + /EOS/IDEAL-GAS: the EOS card's RHO_0
    supplies the gas density (the /MAT/GAS card has none) and gamma/P0
    the initial state — resolve_materials wires it all up."""
    deck = (
        "/BEGIN\ngas\n"
        + f"{2021:>10d}{0:>10d}\n"
        + " " * 18 + "kg" + " " * 19 + "m" + " " * 19 + "s\n"
        + " " * 18 + "kg" + " " * 19 + "m" + " " * 19 + "s\n"
        + "/MAT/GAS/MASS/1\nair\n"
        + _f20("0.02896") + "\n"
        + _f20("1005.0", "0.0", "0.0", "0.0", "0.0") + "\n"
        + _f20("0.0") + "\n"
        + "/EOS/IDEAL-GAS/1\ngas eos\n"
        + _f20("1.4", "1.0e5", "0.0", "293.0", "1.2") + "\n"
        + "/END\n")
    f = tmp_path / "GASEOS_0000.rad"
    f.write_text(deck)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    assert not log.errors, log.errors
    mat = model.materials[1]
    assert mat.law == 999 and not getattr(mat, "inactive", False)
    assert mat.rho0 == pytest.approx(1.2)        # from the EOS RHO_0
    assert mat.eos is not None
    assert mat.eos.params["gamma"] == pytest.approx(1.4)
    assert mat.eos.params["e0"] == pytest.approx(1.0e5 / 0.4)
    assert mat.eos.rho0 == pytest.approx(1.2)
    # the Starter-side elastic ESTIMATE maps the gas bulk gamma*P0
    assert mat.K == pytest.approx(1.4 * 1.0e5, rel=2e-3)


def test_gas_on_elements_requires_eos(tmp_path):
    """Starter check: /MAT/GAS on solid elements without an /EOS is a
    model ERROR (no pressure, no stiffness, no dt claim possible)."""
    deck = (
        "/BEGIN\ngas cube\n" + _CUBE +
        "/MAT/GAS/MASS/1\nair\n" + _f20("0.02896") + "\n"
        + _f20("1005.0", "0.0", "0.0", "0.0", "0.0") + "\n"
        + _f20("0.0") + "\n"
        "/END\n")
    model, log = _build(deck, tmp_path, expect_clean=False)
    check_model(model, log)
    assert any("/EOS/IDEAL-GAS" in e for e in log.errors), log.errors


# ============================================================================
# LAW70 — tabulated visco-elastic foam
# ============================================================================

def _law70_mat(functions, nload=1, nun=1, iflag=0, shape=0.0, hys=0.0,
               e0=50.0, emax=100.0, epsmax=1.0, nu=0.0,
               load=(1,), load_rates=(0.0,), load_scales=(0.0,),
               unload=(2,), unload_rates=(0.0,), unload_scales=(0.0,)):
    rec = GenericMaterialRecord(
        law_name="LAW70", law_number=70, id=1, density=1e-9,
        params={"MAT_E0": e0, "MAT_NU": nu, "E_Max": emax,
                "MAT_EPS": epsmax, "MAT_asrate": 0.0, "ISRATE": 0,
                "NRATEP": nload, "NRATEN": nun, "MAT_Iflag": iflag,
                "MAT_SHAPE": shape, "MAT_HYST": hys, "Itens": 0,
                "FUN_LOAD": list(load), "STRAINRATE_LOAD": list(load_rates),
                "SCALE_LOAD": list(load_scales),
                "FUN_UNLOAD": list(unload),
                "STRAINRATE_UNLOAD": list(unload_rates),
                "SCALE_UNLOAD": list(unload_scales)})
    mat = law70_tabfoam.build_law70(rec)
    law70_tabfoam.resolve(mat, _FnModel(functions), MessageLog())
    return mat


def _law70_extra(n=1, rho=1e-9):
    return {"eps70": np.zeros((n, 6)), "uv70": np.zeros((n, 10)),
            "epsd70": np.zeros(n), "rho": np.full(n, rho)}


def test_law70_loading_unloading_curve_interpolation():
    """Prescribed uniaxial strain path: on loading the stress magnitude
    sits EXACTLY on the (piecewise-linear) loading curve at the reached
    strain norm; after reversal it lands EXACTLY on the unloading
    curve — the hand-checkable core of sigeps70's spherical projection."""
    fns = {1: FunctTable(1, [0.0, 0.1, 0.3, 1.0], [0.0, 0.5, 0.8, 1.5]),
           2: FunctTable(2, [0.0, 0.1, 0.3, 1.0], [0.0, 0.25, 0.4, 0.75])}
    mat = _law70_mat(fns)
    sig = np.zeros((1, 6))
    extra = _law70_extra()
    dt = 1e-3
    deps = np.zeros((1, 6))
    deps[0, 0] = 1e-3
    for _ in range(200):                        # load to eps_xx = 0.2
        sig, c = law70_tabfoam.solid_update(mat, sig, deps, dt, extra)
    # curve value at 0.2 (between the 0.1 and 0.3 knots): 0.65
    assert sig[0, 0] == pytest.approx(fns[1].eval(0.2), rel=1e-9)
    assert np.abs(sig[0, 1:]).max() < 1e-12     # uniaxial, nu = 0

    deps[0, 0] = -1e-3
    for _ in range(100):                        # unload to eps_xx = 0.1
        sig, c = law70_tabfoam.solid_update(mat, sig, deps, dt, extra)
    assert sig[0, 0] == pytest.approx(fns[2].eval(0.1), rel=1e-9)


def test_law70_strain_rate_interpolation_exact():
    """Two loading curves at rates 0 and 10 driven at exactly rate 5:
    the stress must be the linear blend of the two curves (the 2-D
    table interpolation of TABLE_MAT_VINTERP)."""
    fns = {3: FunctTable(3, [0.0, 1.0], [0.0, 0.4]),
           4: FunctTable(4, [0.0, 1.0], [0.0, 0.8])}
    mat = _law70_mat(fns, nload=2, nun=0, load=(3, 4),
                     load_rates=(0.0, 10.0), load_scales=(0.0, 0.0),
                     unload=(), unload_rates=(), unload_scales=(),
                     e0=100.0, emax=200.0)
    sig = np.zeros((1, 6))
    extra = _law70_extra()
    deps = np.zeros((1, 6))
    deps[0, 0] = 5e-3                            # |deps|/dt = 5.0
    for _ in range(40):
        sig, c = law70_tabfoam.solid_update(mat, sig, deps, 1e-3, extra)
    eps = extra["eps70"][0, 0]
    blend = 0.5 * (fns[3].eval(eps) + fns[4].eval(eps))
    assert sig[0, 0] == pytest.approx(blend, rel=1e-9)


def test_law70_sound_speed_and_dt_claim():
    """c = sqrt(AA1/rho0) with AA1 the P-wave modulus of the CURRENT
    evolving E: at virgin state E = E0 exactly; after loading E can only
    grow, so the dt claim can only tighten."""
    fns = {1: FunctTable(1, [0.0, 0.1, 0.3, 1.0], [0.0, 0.5, 0.8, 1.5]),
           2: FunctTable(2, [0.0, 0.1, 0.3, 1.0], [0.0, 0.25, 0.4, 0.75])}
    mat = _law70_mat(fns, nu=0.2)
    sig = np.zeros((1, 6))
    extra = _law70_extra()
    deps = np.zeros((1, 6))
    sig, c = law70_tabfoam.solid_update(mat, sig, deps, 1e-3, extra)
    e0, nu = 50.0, 0.2
    aa1 = e0 * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    assert c[0] == pytest.approx(np.sqrt(aa1 / 1e-9), rel=1e-12)
    deps[0, 0] = 1e-3
    for _ in range(300):
        sig, c2 = law70_tabfoam.solid_update(mat, sig, deps, 1e-3, extra)
    assert c2[0] >= c[0]                        # E only stiffens


def test_law70_objectivity_spin_and_isotropy():
    """(a) pure spin from any state adds no strain (deps = 0) and leaves
    the stress magnitude untouched; (b) the law is isotropic: rotating
    the strain path rotates the stress (single-step frame check)."""
    fns = {1: FunctTable(1, [0.0, 0.1, 0.3, 1.0], [0.0, 0.5, 0.8, 1.5]),
           2: FunctTable(2, [0.0, 0.1, 0.3, 1.0], [0.0, 0.25, 0.4, 0.75])}
    mat = _law70_mat(fns, nu=0.1)
    dt = 1e-3
    deps = np.zeros((1, 6))
    deps[0] = [2e-2, -1e-2, 0.5e-2, 3e-2, -1e-2, 0.4e-2]

    sig1 = np.zeros((1, 6))
    ex1 = _law70_extra()
    sig1, _ = law70_tabfoam.solid_update(mat, sig1, deps.copy(), dt, ex1)

    R = _rotmat([1.0, 2.0, 0.5], 0.7)
    depsR = np.zeros((1, 6))
    depsR[0] = _rot_deps(R, deps[0])
    sig2 = np.zeros((1, 6))
    ex2 = _law70_extra()
    sig2, _ = law70_tabfoam.solid_update(mat, sig2, depsR, dt, ex2)
    assert np.allclose(sig2[0], _rot_sig(R, sig1[0]), rtol=1e-9,
                       atol=1e-12)

    # pure spin: zero strain increment leaves the state exactly in place
    before = sig1.copy()
    sig1, _ = law70_tabfoam.solid_update(mat, sig1, np.zeros((1, 6)),
                                         dt, ex1)
    assert np.allclose(sig1, before, rtol=1e-12, atol=1e-15)


def test_law70_iflag4_hysteresis_bounds():
    """Iflag=4: unloading scales the WHOLE tensor by
    R = 1-(1-Hys)*(1-(E/E_hist)^Shape) in [Hys, 1] — the unloading
    stress is bracketed by Hys*loading-curve and the loading curve."""
    fns = {1: FunctTable(1, [0.0, 0.1, 0.3, 1.0], [0.0, 0.5, 0.8, 1.5])}
    hys = 0.2
    mat = _law70_mat(fns, nun=0, iflag=4, shape=1.5, hys=hys,
                     unload=(), unload_rates=(), unload_scales=())
    sig = np.zeros((1, 6))
    extra = _law70_extra()
    dt = 1e-3
    deps = np.zeros((1, 6))
    deps[0, 0] = 1e-3
    for _ in range(200):
        sig, _ = law70_tabfoam.solid_update(mat, sig, deps, dt, extra)
    deps[0, 0] = -1e-3
    for _ in range(50):                          # unload to eps = 0.15
        sig, _ = law70_tabfoam.solid_update(mat, sig, deps, dt, extra)
    load_val = fns[1].eval(0.15)
    assert hys * load_val - 1e-12 <= sig[0, 0] <= load_val + 1e-12
    assert sig[0, 0] < load_val                  # strict hysteresis


def test_law70_hexa_deck_on_curve(tmp_path):
    """End-to-end through the cfg-driven reader and the brick kernel:
    compress a LAW70 cube and the axial stress must sit on the loading
    curve at the accumulated strain norm."""
    deck = (
        "/BEGIN\nfoam cube\n" + _CUBE +
        "/MAT/LAW70/1\nfoam\n"
        + _f20("1e-9") + "\n"
        + _f20("50.0", "0.0", "100.0", "1.0") + f"{0:>10d}" + "\n"
        + _f20("0.0") + _i10(0, 1, 1, 0) + _f20("1.0", "1.0") + "\n"
        + f"{1:>10d}" + _f20("0.0", "0.0") + "\n"
        + f"{2:>10d}" + _f20("0.0", "0.5") + "\n"
        "/FUNCT/1\nload\n0.0 0.0\n0.1 0.5\n0.3 0.8\n1.0 1.5\n"
        "/FUNCT/2\nunload\n0.0 0.0\n0.1 0.25\n0.3 0.4\n1.0 0.75\n"
        "/END\n")
    model, log = _build(deck, tmp_path)
    g = model.bricks
    mat = g.state["slices"][0][1]
    assert mat.law == 70 and not getattr(mat, "inactive", False)
    assert "eps70" in g.state["mat_extra"]

    v = np.zeros_like(model.x)
    v[:, 0] = -0.1 * model.x[:, 0]              # compressing
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    dt = 1e-3
    for _ in range(1000):                        # eps_xx -> -0.1
        dtc = solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
    epst = abs(g.state["mat_extra"]["eps70"][0, 0])
    assert epst == pytest.approx(0.1, rel=1e-12)
    assert g.state["sig"][0, 0] == pytest.approx(-0.5, rel=1e-9)
    assert np.isfinite(dtc[0]) and dtc[0] > 0.0


def test_law70_physical_hourglass_stiffness_is_elastic(tmp_path):
    """M38 regression for the LAW70 physical-hourglass fix.

    RD-V-0220 c46/c47/c49 died on hourglass-energy INJECTION at their
    densification lock-up: the deck's /PROP/SOLID Isolid=24 asks for the
    HEPH physically-stabilized brick, but the port's default brick carries
    only VISCOUS hourglass control, which resists hourglass VELOCITY, never
    DEFORMATION — so past EPS_max the zero-energy modes ran away (Fortran
    holds HOURGLASS ENERGY = 0 the whole run). The M38 fix
    (solid_hexa8._phys_hourglass_law70) adds the missing Belytschko-Bindeman
    hourglass STIFFNESS for LAW70 bricks.

    This drives ONE hourglass mode UP then back DOWN at a densified state and
    asserts the hourglass-energy ledger is STORED on the way up and RETURNED
    on the way down — i.e. the added control is an ELASTIC restoring force. A
    viscous-only control (the pre-fix behaviour) could only ever DISSIPATE, so
    its ledger would keep GROWING on the reverse leg and never come back;
    hence ``ehour_end < ehour_peak`` fails without the fix and holds with it.
    """
    deck = (
        "/BEGIN\nfoam hg\n" + _CUBE +
        "/MAT/LAW70/1\nfoam\n"
        + _f20("1.0") + "\n"                              # heavy rho -> slow c
        + _f20("25.0", "0.0", "2500.0", "1.0") + f"{0:>10d}" + "\n"
        + _f20("0.0") + _i10(0, 1, 1, 0) + _f20("1.0", "1.0") + "\n"
        + f"{1:>10d}" + _f20("0.0", "0.0") + "\n"
        + f"{2:>10d}" + _f20("0.0", "0.5") + "\n"
        "/FUNCT/1\nload\n0.0 0.0\n0.1 0.4\n0.9 1.0\n1.0 7.0\n"
        "/FUNCT/2\nunload\n0.0 0.0\n0.1 0.2\n0.9 0.5\n1.0 3.5\n"
        "/END\n")
    model, log = _build(deck, tmp_path)
    g = model.bricks
    assert g.state["has_law70"]                    # fix wired for this group
    fint = np.zeros_like(model.x)
    mint = np.zeros_like(model.x)
    dt = 1e-3

    # ---- 1. compress uniformly to a stiffened (densifying) state ----------
    v = np.zeros_like(model.x)
    v[:, 2] = -0.8 * model.x[:, 2]
    for _ in range(1000):
        solid_hexa8.forces(g, model.x, v, model.vr, dt, fint, mint)
    E = g.state["mat_extra"]["uv70"][0, 2]
    assert E > 1500.0                              # modulus climbed toward E_max
    # uniform compression excites no hourglass -> ledger still ~0
    ehour0 = float(g.state["ehour"][0])

    conn = g.conn[0]                               # the element's 8 node ids
    pat = solid_hexa8._H[0]                        # a pure hourglass mode
    amp = 1e-3

    # ---- 2. LOAD the hourglass mode: the ledger must climb (energy stored) -
    def drive(sign, nstep):
        vh = np.zeros_like(model.x)
        vh[conn, 2] = sign * amp * pat
        last = None
        for _ in range(nstep):
            last = solid_hexa8.forces(g, model.x, vh, model.vr, dt, fint, mint)
        return last

    dtc = drive(+1.0, 150)
    ehour_peak = float(g.state["ehour"][0])
    assert ehour_peak > ehour0 + 1e-9             # STORED on loading
    assert np.all(np.isfinite(dtc)) and dtc[0] > 0.0   # dt stays finite/coupled

    # ---- 3. UNLOAD the mode back to the start: the ledger must come DOWN ---
    drive(-1.0, 150)
    ehour_end = float(g.state["ehour"][0])
    assert ehour_end < ehour_peak - 1e-9          # RETURNED on unloading (elastic)
    assert np.isfinite(ehour_end)                 # bounded throughout (no runaway)


@pytest.mark.skipif(not _HAS_CORPUS, reason="official-deck corpus extract "
                    "not present on this machine")
def test_law70_oracle_deck_material_resolves():
    """RD-V-0220_Foam_LAW70 (official oracle): the /MAT/LAW70 card of
    every variant must parse to an ACTIVE material with resolved
    tables (variant 0: Iflag 0 + unload curve; variants 1-3: Iflag 4
    with Shape/Hys and Itens)."""
    base = os.path.join(RD_DECKS, "rd_v_material", "RD-V-0220_Foam_LAW70",
                        "0220_foam_LAW70")
    for k in range(4):
        path = os.path.join(base, f"0220_foam_LAW70_{k}",
                            "BLOCK_H8_0000.rad")
        if not os.path.isfile(path):
            pytest.skip("RD-V-0220 deck not in corpus extract")
        model = Model()
        log = MessageLog()
        parse_starter_deck(read_deck(path), model, log)
        resolve_materials(model, log)
        mat = model.materials[1]
        assert mat.law == 70 and not getattr(mat, "inactive", False)
        assert "xg_load" in mat.params, (k, log.errors)
        mat_errors = [e for e in log.errors if "LAW70" in e]
        assert not mat_errors, mat_errors


# ============================================================================
# M38 / M37-BUG-2 — null-density MAT CHECK vs multi-material ALE laws
# ============================================================================

@pytest.mark.skipif(not _HAS_CORPUS, reason="official-deck corpus extract "
                    "not present on this machine")
def test_law151_multimaterial_ale_exempt_from_null_density_check():
    """M38 / M37-BUG-2: the multimaterial ALE family (LAW51, LAW151/
    MULTIFLUID) carries its initial density on the SUBMATERIAL references +
    volume fractions, not the top-level RHO0 — the card's first density
    field is legitimately blank (RD-E-1300 blast_experiment: an empty first
    card, then ``mat_ID_01 / Vfrac_01``). The port's null-density MAT CHECK
    must NOT fatal-error them (the upstream Starter does not). Here the two
    LAW151 air materials parse with RHO0 = 0, land on brick element groups,
    and check_model must raise NO 'initial density' error for them."""
    path = os.path.join(RD_DECKS, "rd_e", "RD-E-1300_Shock_tube",
                        "13_Shock_tube", "Blast_experiment",
                        "blast_experiment_0000.rad")
    if not os.path.isfile(path):
        pytest.skip("blast_experiment deck not in corpus extract")
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(path), model, log)
    resolve_materials(model, log)
    build_element_groups(model, log)
    law151 = [m for m in model.materials.values() if m.law == 151]
    assert law151, "blast_experiment should define /MAT/LAW151 materials"
    assert all(getattr(m, "rho0", 0.0) == 0.0 for m in law151), \
        "a multimaterial LAW151's top-level RHO0 is blank/zero"
    on_group = [mat.id for _, g in model.element_groups()
                for _, mat, _ in g.state["slices"] if mat.law == 151]
    assert on_group, "LAW151 materials must reach the MAT CHECK on a group"
    log.errors.clear()
    check_model(model, log)
    dens_err = [e for e in log.errors if "initial density" in e]
    assert not dens_err, dens_err              # EXEMPT — no fatal density error


def test_null_density_check_still_fires_for_non_ale_law(tmp_path):
    """The M38 exemption is NARROW: a genuine null-density material on a
    non-ALE law (only LAW51/LAW151 keep density on submaterials) is still a
    fatal MAT CHECK error — masses cannot be initialized without it."""
    deck = (
        "/BEGIN\nnull dens\n" + _CUBE +
        "/MAT/LAW1/1\nsteel\n"
        + _f20("0.0") + "\n"                       # RHO_I = 0 (the real bug)
        + _f20("210000.0", "0.3") + "\n"
        "/END\n")
    model = Model()
    log = MessageLog()
    f = tmp_path / "K_0000.rad"
    f.write_text(deck)
    parse_starter_deck(read_deck(str(f)), model, log)
    resolve_materials(model, log)
    build_element_groups(model, log)
    log.errors.clear()
    check_model(model, log)
    assert any("initial density" in e for e in log.errors), log.errors


# ============================================================================
# LAW35 — visco-elastic foam (standard linear solid deviator)
# ============================================================================

def _law35_mat(**over):
    params = {"MAT_RHO": 2e-9, "MAT_E": 10.0, "MAT_NU": 0.0,
              "MAT_ETAN": 4.0, "MAT_NUt": 0.0, "MAT_ETA2": 2.0,
              "MAT_ETA1": 0.0, "MAT_CO1": 0.0, "MAT_CO2": 0.0,
              "MAT_CO3": 0.0}
    params.update(over)
    rec = GenericMaterialRecord(law_name="LAW35", law_number=35, id=1,
                                density=2e-9, params=params)
    return law35_kelvinmax.build_law35(rec)


def _law35_extra(n=1, rho=2e-9):
    return {"eps35": np.zeros((n, 6)), "sigair35": np.zeros(n),
            "edot35": np.zeros(n), "rho": np.full(n, rho)}


def test_law35_relaxation_time_constant_closed_form():
    """Deviatoric standard-linear-solid: held at constant shear strain
    the stress relaxes toward s_inf = G2*Gt2/(G2+Gt2)*e with the exact
    time constant tau = 2*mu/(G2+Gt2) = mu/(G+Gt).  The trapezoidal
    (MIDSTEP) update is the exact Crank-Nicolson map, so the discrete
    decay factor rho = (1-dt/2tau)/(1+dt/2tau) must be reproduced to
    machine precision."""
    mat = _law35_mat()
    G2, GT2, vmu2 = 10.0, 4.0, 4.0               # E/(1+nu), Et/(1+nut), 2mu
    k = (G2 + GT2) / vmu2                        # 1/tau
    assert 1.0 / k == pytest.approx(2.0 * 2.0 / (G2 + GT2), rel=1e-14)

    sig = np.zeros((1, 6))
    extra = _law35_extra()
    dt = 1e-2
    deps = np.zeros((1, 6))
    deps[0, 3] = 0.01                            # engineering shear step
    sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt, extra)
    s0 = sig[0, 3]

    e_xy = 0.005                                 # tensor shear strain
    q = (G2 * GT2 / vmu2) * e_xy
    s_inf = q / k                                # = G2*Gt2/(G2+Gt2)*e
    rho_f = (1 - k * dt / 2) / (1 + k * dt / 2)  # exact CN decay factor

    deps[:] = 0.0
    m = 80
    for _ in range(m):
        sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt, extra)
    assert sig[0, 3] == pytest.approx(s_inf + (s0 - s_inf) * rho_f ** m,
                                      rel=1e-12)
    # relax essentially to the long-term spring: tau = 2/7, m*dt = 2.8*tau
    for _ in range(4000):
        sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt, extra)
    assert sig[0, 3] == pytest.approx(s_inf, rel=1e-10)
    # sound speed: sqrt(((2/3)G2 + BULK3/3)/rho) with E_new = E
    assert c[0] == pytest.approx(
        np.sqrt(((2.0 / 3.0) * G2 + 10.0 / 3.0) / 2e-9), rel=1e-12)


def test_law35_pressure_and_closed_cell_air():
    """C1=1, C2=C3=0 makes the pressure rate PDOT = K*tr(deps)/dt: the
    mean stress integrates K*eps_v exactly; the closed-cell air term
    subtracts SIGAIR = P0*delta/(1-delta) (phi = gama0 = 0) from the
    diagonal."""
    p0 = 0.05
    mat = _law35_mat(MAT_CO1=1.0, MAT_P0=p0)
    n_st = 100
    d = -1e-4                                    # volumetric compression
    dt = 1e-3
    sig = np.zeros((1, 6))
    extra = _law35_extra()
    deps = np.zeros((1, 6))
    deps[0, :3] = d
    J = 1.0
    for _ in range(n_st):
        J *= (1.0 + 3.0 * d)                     # dV/V = tr(deps)
        extra["rho"][0] = 2e-9 / J
        sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt, extra)
    delta = 1.0 - J                              # compression fraction
    sigair = p0 * delta / (1.0 - delta)
    assert extra["sigair35"][0] == pytest.approx(sigair, rel=1e-9)
    # mean total stress = K*eps_v - sigair (deviator zero by symmetry)
    K = 10.0 / 3.0
    eps_v = n_st * 3.0 * d
    pm = (sig[0, 0] + sig[0, 1] + sig[0, 2]) / 3.0
    assert pm == pytest.approx(K * eps_v - sigair, rel=1e-6)


def test_law35_isotropy_single_step():
    """Frame indifference of the isotropic rate equations: a rotated
    strain increment produces the rotated stress (single step)."""
    mat = _law35_mat(MAT_NU=0.2, MAT_NUt=0.1)
    dt = 1e-3
    deps = np.zeros((1, 6))
    deps[0] = [1e-2, -0.5e-2, 0.2e-2, 2e-2, -0.7e-2, 0.3e-2]
    sig1 = np.zeros((1, 6))
    sig1, _ = law35_kelvinmax.solid_update(mat, sig1, deps.copy(), dt,
                                           _law35_extra())
    R = _rotmat([0.3, -1.0, 0.8], 1.1)
    depsR = np.zeros((1, 6))
    depsR[0] = _rot_deps(R, deps[0])
    sig2 = np.zeros((1, 6))
    sig2, _ = law35_kelvinmax.solid_update(mat, sig2, depsR, dt,
                                           _law35_extra())
    assert np.allclose(sig2[0], _rot_sig(R, sig1[0]), rtol=1e-9,
                       atol=1e-14)


# ============================================================================
# LAW40 — /MAT/KELVINMAX generalized Kelvin-Maxwell
# ============================================================================

def _law40_mat(k=66.67, gi=10.0, g1=90.0, beta1=0.01):
    rec = GenericMaterialRecord(
        law_name="KELVINMAX", law_number=40, id=1, density=2e-9,
        params={"MAT_BULK": k, "MAT_GI": gi, "MAT_G0": g1,
                "MAT_DECAY": beta1})
    return law40_kelvinmax.build_law40(rec)


def _law40_extra(n=1, rho=2e-9):
    return {"eps40": np.zeros((n, 6)), "uv40": np.zeros((n, 40)),
            "rho": np.full(n, rho)}


def test_law40_relaxation_time_constant_exact():
    """A pre-stressed Maxwell branch held at constant strain relaxes
    EXACTLY exponentially with tau = 1/beta (the branch integration is
    closed-form in the Fortran), leaving the long-term spring stress
    2*G_inf*e."""
    beta = 0.01
    mat = _law40_mat(beta1=beta)
    sig = np.zeros((1, 6))
    extra = _law40_extra()
    extra["eps40"][0, 3] = 0.01                 # gamma_xy held constant
    v0 = 0.03
    extra["uv40"][0, 13] = v0                   # branch-1 xy stress
    dt, m = 1e-2, 400
    deps = np.zeros((1, 6))
    for _ in range(m):
        sig, c = law40_kelvinmax.solid_update(mat, sig, deps, dt, extra)
    s_spring = 2.0 * 10.0 * 0.005               # 2*G_inf*e_xy
    assert sig[0, 3] == pytest.approx(
        s_spring + v0 * np.exp(-beta * m * dt), rel=1e-12)
    # the dt claim, verbatim upstream: sqrt(K/rho + 4*GT/(3 rho)) with
    # GT the DOUBLED total shear modulus (deliberate over-estimate)
    gt = 2.0 * (10.0 + 90.0)
    assert c[0] == pytest.approx(np.sqrt((66.67 + 4.0 * gt / 3.0) / 2e-9),
                                 rel=1e-12)


def test_law40_bulk_pressure_incremental():
    """Mean stress integrates K*tr(deps) exactly (hypoelastic bulk)."""
    mat = _law40_mat()
    sig = np.zeros((1, 6))
    extra = _law40_extra()
    deps = np.zeros((1, 6))
    deps[0, :3] = -1e-4
    for _ in range(50):
        sig, _ = law40_kelvinmax.solid_update(mat, sig, deps, 1e-3, extra)
    pm = sig[0, :3].mean()
    assert pm == pytest.approx(66.67 * 50 * 3 * (-1e-4), rel=1e-9)


@pytest.mark.skipif(not _HAS_CORPUS, reason="official-deck corpus extract "
                    "not present on this machine")
def test_law40_corpus_kelvinmax_deck_builds():
    """RD-E-5200 creep decks: /MAT/KELVINMAX/3 parses and builds an
    ACTIVE LAW40 material (K=66.67, G_inf=10, G1=90, beta1=.01)."""
    path = os.path.join(RD_DECKS, "rd_e", "RD-E-5200_Creep",
                        "52_cylinder_creep", "cylinder_creep_beta_001",
                        "foam_relax_0000.rad")
    if not os.path.isfile(path):
        pytest.skip("RD-E-5200 deck not in corpus extract")
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(path), model, log)
    resolve_materials(model, log)
    mat = model.materials[3]
    assert mat.law == 40 and not getattr(mat, "inactive", False)
    assert mat.params["K40"] == pytest.approx(66.67)
    assert mat.params["G_inf"] == pytest.approx(10.0)
    assert mat.params["G"][0] == pytest.approx(90.0)
    assert mat.params["beta"][0] == pytest.approx(0.01)


# ============================================================================
# LAW44 — Cowper-Symonds
# ============================================================================

def _law44_mat(**over):
    params = {"MAT_E": 210.0, "MAT_NU": 0.3, "MAT_SIGY": 0.4,
              "MAT_B": 0.0, "MAT_N": 0.0, "MAT_SRC": 100.0,
              "MAT_SRE": 2.0}
    params.update(over)
    rec = GenericMaterialRecord(law_name="LAW44", law_number=44, id=1,
                                density=7.8e-9, params=params)
    return law44_cowper.build_law44(rec)


def test_law44_rate_hardening_factor_exact_solid():
    """Sustained plastic flow at a constant strain rate: the von Mises
    stress must equal A * (1 + (epsdot/C)^(1/p)) EXACTLY (B = 0, so no
    hardening and the radial return lands on the rate-scaled surface).
    epsdot is the VP=2 measure: the tensor norm of the strain rate."""
    mat = _law44_mat()
    assert mat.params["cc"] == pytest.approx(1.0 / 100.0, rel=1e-14)
    assert mat.params["cp"] == pytest.approx(0.5, rel=1e-14)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    extra = {"rho": np.full(1, 7.8e-9)}
    d, dt = 5e-4, 1e-4
    deps = np.zeros((1, 6))
    deps[0] = [d, -d / 2, -d / 2, 0, 0, 0]       # pure deviatoric
    for _ in range(300):
        sig, epsp, c = law44_cowper.solid_update(mat, sig, deps, epsp,
                                                 dt, extra)
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    epsdot = np.sqrt(d ** 2 + 2 * (d / 2) ** 2) / dt
    assert vm == pytest.approx(0.4 * (1.0 + (epsdot / 100.0) ** 0.5),
                               rel=1e-12)
    assert epsp[0] > 0.0
    # sound speed: sqrt((K + 4G/3)/rho0), constant
    E, nu = 210.0, 0.3
    K = E / (3 * (1 - 2 * nu))
    G = E / (2 * (1 + nu))
    assert c[0] == pytest.approx(np.sqrt((K + 4 * G / 3) / 7.8e-9),
                                 rel=1e-12)


def test_law44_linear_hardening_lands_on_curve():
    """n = 1 (linear hardening, quasi-static): the upstream IPLA=0
    return scales the stress to the yield of the PREVIOUS plastic
    strain while pla advances by dpla — the discrete relation
    vm_n = A + B*pla_{n-1} must hold EXACTLY (sigeps44's one-step
    radial return, ported verbatim)."""
    mat = _law44_mat(MAT_B=2.0, MAT_N=1.0, MAT_SRC=0.0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    extra = {"rho": np.full(1, 7.8e-9)}
    deps = np.zeros((1, 6))
    deps[0] = [1e-4, -5e-5, -5e-5, 0, 0, 0]
    for _ in range(599):
        sig, epsp, c = law44_cowper.solid_update(mat, sig, deps, epsp,
                                                 1e-3, extra)
    ep_prev = epsp[0]
    sig, epsp, c = law44_cowper.solid_update(mat, sig, deps, epsp,
                                             1e-3, extra)
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert epsp[0] > 0.01
    assert vm == pytest.approx(0.4 + 2.0 * ep_prev, rel=1e-9)
    # the lag vanishes as dt -> 0: within one increment of the curve
    assert vm == pytest.approx(0.4 + 2.0 * epsp[0], rel=1e-3)


def test_law44_rate_factor_exact_shell():
    """Shell (plane-stress) flow: sigma_eq = A*(1+(epsdot/C)^(1/p)) with
    the sigeps44c in-plane deviatoric rate measure."""
    mat = _law44_mat()
    sig = np.zeros((1, 3))
    epsp = np.zeros(1)
    d, dt = 5e-4, 1e-4
    deps = np.zeros((1, 3))
    deps[0] = [d, -d, 0.0]
    for _ in range(300):
        sig, epsp = law44_cowper.shell_update(mat, sig, deps, epsp, dt, {})
    seq = np.sqrt(sig[0, 0] ** 2 - sig[0, 0] * sig[0, 1] + sig[0, 1] ** 2
                  + 3 * sig[0, 2] ** 2)
    dav = (d + (-d)) / 3.0 / dt
    d1, d2, d3 = d / dt - dav, -d / dt - dav, -dav
    epsdot = np.sqrt(3.0 * 0.5 * (d1 ** 2 + d2 ** 2 + d3 ** 2)) / 1.5
    assert seq == pytest.approx(0.4 * (1.0 + (epsdot / 100.0) ** 0.5),
                                rel=1e-9)
    assert epsp[0] > 0.0


def test_law44_total_pressure_from_density():
    """The solid pressure is TOTAL: P = K*(rho/rho0 - 1) whatever the
    increment history (sigeps44's EOS-style bulk response)."""
    mat = _law44_mat(MAT_SRC=0.0)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    mu = 0.05                                    # 5% compressed
    extra = {"rho": np.full(1, 7.8e-9 * (1 + mu))}
    sig, epsp, c = law44_cowper.solid_update(
        mat, sig, np.zeros((1, 6)), epsp, 1e-3, extra)
    K = 210.0 / (3 * (1 - 0.6))
    assert sig[0, 0] == pytest.approx(-K * mu, rel=1e-12)
    assert sig[0, 0] == sig[0, 1] == sig[0, 2]
    assert np.all(sig[0, 3:] == 0.0)


def test_law44_sig_max_cap_and_eps_max_kill():
    """The yield saturates at sig_max (via EPSGM) and drops to zero
    beyond eps_max — the deviator vanishes, and the kernels' generic
    eps_p_max plumbing deletes the element (params['eps_p_max'])."""
    mat = _law44_mat(MAT_B=2.0, MAT_N=1.0, MAT_SRC=0.0, MAT_SIG=0.45,
                     MAT_EPS=0.5)
    assert mat.params["eps_p_max"] == pytest.approx(0.5)
    sig = np.zeros((1, 6))
    epsp = np.zeros(1)
    extra = {"rho": np.full(1, 7.8e-9)}
    deps = np.zeros((1, 6))
    deps[0] = [1e-4, -5e-5, -5e-5, 0, 0, 0]
    for _ in range(1500):
        sig, epsp, _ = law44_cowper.solid_update(mat, sig, deps, epsp,
                                                 1e-3, extra)
    vm = np.sqrt(0.5 * ((sig[0, 0] - sig[0, 1]) ** 2
                        + (sig[0, 1] - sig[0, 2]) ** 2
                        + (sig[0, 2] - sig[0, 0]) ** 2))
    assert vm == pytest.approx(0.45, rel=1e-9)   # capped
    epsp[0] = 0.6                                # beyond eps_max
    for _ in range(5):
        sig, epsp, _ = law44_cowper.solid_update(mat, sig, deps, epsp,
                                                 1e-3, extra)
    assert np.abs(sig[0]).max() < 1e-12          # yld = 0: stress gone


def test_law44_isotropy_single_step():
    """Frame indifference: a rotated strain increment gives the rotated
    stress (single plastic step from virgin state)."""
    mat = _law44_mat()
    dt = 1e-4
    deps = np.zeros((1, 6))
    deps[0] = [3e-3, -1e-3, -0.5e-3, 2e-3, -1e-3, 0.7e-3]
    extra = {"rho": np.full(1, 7.8e-9)}
    sig1 = np.zeros((1, 6))
    ep1 = np.zeros(1)
    sig1, ep1, _ = law44_cowper.solid_update(mat, sig1, deps.copy(),
                                             ep1, dt, extra)
    R = _rotmat([0.2, 0.5, -1.0], 0.9)
    depsR = np.zeros((1, 6))
    depsR[0] = _rot_deps(R, deps[0])
    sig2 = np.zeros((1, 6))
    ep2 = np.zeros(1)
    sig2, ep2, _ = law44_cowper.solid_update(mat, sig2, depsR, ep2, dt,
                                             {"rho": np.full(1, 7.8e-9)})
    assert ep2[0] == pytest.approx(ep1[0], rel=1e-12)
    assert np.allclose(sig2[0], _rot_sig(R, sig1[0]), rtol=1e-9,
                       atol=1e-13)


def test_law44_deck_parse_both_spellings(tmp_path):
    """/MAT/LAW44 and /MAT/COWPER both build active LAW44 materials with
    the starter transforms CC = 1/C, CP = 1/p."""
    card = (
        _f20("7.8e-9") + "\n"
        + _f20("210.0", "0.3") + "\n"
        + _f20("0.4", "0.5", "1.0", "0.0", "0.0") + "\n"
        + _f20("100.0", "2.0") + _i10(0, 0) + _f20("0.0")
        + " " * 10 + _i10(0) + "\n"
        + _f20("0.0", "0.0", "0.0") + "\n")
    deck = ("/BEGIN\nlaw44\n" + _CUBE
            + "/MAT/LAW44/1\nsteel\n" + card + "/END\n")
    model, _ = _build(deck, tmp_path)
    mat = model.materials[1]
    assert mat.law == 44 and not getattr(mat, "inactive", False)
    assert mat.params["cc"] == pytest.approx(0.01, rel=1e-12)
    assert mat.params["cp"] == pytest.approx(0.5, rel=1e-12)
    assert mat.params["A"] == pytest.approx(0.4)
    assert mat.params["B"] == pytest.approx(0.5)

    deck2 = ("/BEGIN\ncowper\n" + _CUBE
             + "/MAT/COWPER/1\nsteel\n" + card + "/END\n")
    model2, _ = _build(deck2, tmp_path)
    assert model2.materials[1].law == 44
