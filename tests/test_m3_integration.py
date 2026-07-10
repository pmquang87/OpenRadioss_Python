"""M3 analytic validations: full Starter+Engine runs for the new
material laws, the /FAIL element deletion and the plastic beams
(see PORTING_GUIDE.md §6 — every new feature gets at least one
closed-form validation)."""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter


def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model


def _bar_bricks(nel):
    """Nodes + brick cards for a nel x 1 x 1 mm bar along x."""
    nodes, bricks = [], []
    nid = 0
    for i in range(nel + 1):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {float(i)} {float(y)} {float(z)}")
    for e in range(nel):
        b = 4 * e
        bricks.append(f"{e + 1} {b + 1} {b + 2} {b + 3} {b + 4} "
                      f"{b + 5} {b + 6} {b + 7} {b + 8}")
    return nodes, bricks, nid


# ============================================================================
# LAW36 — tabulated plasticity, full run
# ============================================================================

def test_law36_tensile_bar_stress_on_curve(make_deck):
    """A bar of LAW36 bricks pulled in uniaxial stress: after sustained
    plastic flow the axial stress must sit on the tabulated hardening
    curve sy = 0.4 + 1.0*eps_p at the reached plastic strain — the
    tensile_bar validation of M1, tabulated flavour."""
    nel = 4
    nodes, bricks, nid = _bar_bricks(nel)
    fixed = "1 2 3 4"
    pulled = " ".join(str(4 * nel + k) for k in (1, 2, 3, 4))
    starter = (
        "/BEGIN\nlaw36 tension\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\nbar\n1 1\n"
        "/MAT/LAW36/1\ntabulated steel\n7.8e-6\n210. 0.3\n1 0\n11\n"
        "/FUNCT/11\nhardening sy = 0.4 + ep\n0.0 0.4\n1.0 1.4\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/1\nfixed\n{fixed}\n"
        f"/GRNOD/NODE/2\npulled\n{pulled}\n"
        "/BCS/1\nhold x\n100 000 0 1\n"
        "/FUNCT/1\nramp\n0.0 0.0\n0.02 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npull\n1 X 2 0.2\n"
        "/END\n"
    )
    engine = "/RUN/L36/1\n0.8\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n15.0\n"
    model = _run(make_deck, "L36", starter, engine)
    st = model.bricks.state
    ep = st["epsp"].mean()
    assert ep > 0.01                       # well into the plastic regime
    assert st["sig"][:, 0].mean() == pytest.approx(0.4 + 1.0 * ep, rel=0.01)


# ============================================================================
# LAW42 — Ogden rubber, quasi-static closed form + /DT 0.9 stability
# ============================================================================

_RUBBER_CUBE = (
    "/NODE\n"
    "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
    "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
    "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
    "/PART/1\nrubber\n1 1\n"
    "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
    "/GRNOD/NODE/1\nfixed x=0\n1 4 5 8\n"
    "/GRNOD/NODE/2\npulled x=1\n2 3 6 7\n"
    "/BCS/1\nhold x\n100 000 0 1\n"
)


def test_law42_rubber_uniaxial_closed_form(make_deck):
    """A rubber cube pulled slowly to lambda ~ 1.45 (Mooney-Rivlin,
    nearly incompressible): the axial Cauchy stress must match the
    incompressible closed form sum mu_p (lam^a_p - lam^(-a_p/2)) at the
    MEASURED stretch — quasi-static because the pull velocity is ~1e-3
    of the material sound speed and ramped."""
    mu, al = [0.0008, -0.0002], [2.0, -2.0]
    starter = (
        "/BEGIN\nrubber pull\n" + _RUBBER_CUBE +
        "/MAT/LAW42/1\nrubber MR\n1.0e-6\n0.0008 -0.0002\n2.0 -2.0\n0.495\n"
        "/FUNCT/1\nsmooth ramp\n0.0 0.0\n0.4 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npull\n1 X 2 0.25\n"
        "/END\n"
    )
    engine = "/RUN/RUB/1\n2.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n15.0\n"
    model = _run(make_deck, "RUB", starter, engine)
    x_pulled = model.x[model.node_indices([2, 3, 6, 7]), 0].mean()
    x_fixed = model.x[model.node_indices([1, 4, 5, 8]), 0].mean()
    lam = x_pulled - x_fixed               # initial length = 1
    assert lam > 1.3                       # genuinely large stretch
    sig_exact = sum(m * (lam ** a - lam ** (-a / 2.0))
                    for m, a in zip(mu, al))
    sxx = model.bricks.state["sig"][0, 0]
    assert sxx == pytest.approx(sig_exact, rel=0.05)
    # lateral stress ~ 0 (uniaxial stress state, free faces)
    assert abs(model.bricks.state["sig"][0, 1]) < 0.05 * sxx


def test_law42_stability_at_large_stretch_dt09(make_deck):
    """THE time-step requirement of M3: hold an Ogden element (alpha = 3,
    tangent shear ~ 8x the ground state at lambda = 2) at large stretch
    for thousands of cycles at /DT 0.9. With the sound speed frozen at
    the ground-state value this configuration IS unstable (0.9 * the
    true frequency ratio > 1); the law's nonlinear sound speed keeps the
    Courant step a bound — the run must stay clean to the end."""
    starter = (
        "/BEGIN\nrubber stretch hold\n" + _RUBBER_CUBE +
        # single-term Ogden, alpha = 3: strong tension stiffening;
        # nu = 0.45 so the (constant) bulk term does not mask it
        "/MAT/LAW42/1\nrubber ogden a3\n1.0e-6\n0.000667\n3.0\n0.45\n"
        # pull to ~2x, then stop and HOLD stretched
        "/FUNCT/1\npull then hold\n0.0 0.0\n0.2 1.0\n2.1 1.0\n2.3 0.0\n"
        "100.0 0.0\n"
        "/IMPVEL/1\npull\n1 X 2 0.5\n"
        "/END\n"
    )
    engine = "/RUN/ROG/1\n40.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n14.9\n"
    model = _run(make_deck, "ROG", starter, engine)
    # held at ~2x stretch...
    assert model.x[model.node_index(2), 0] > 1.9
    # ...and still healthy after the long hold: finite, bounded velocities
    # (an unstable run blows past this within a few hundred cycles)
    v = model.v[model.mass < 1e29]
    assert np.all(np.isfinite(v))
    assert np.abs(v).max() < 0.5


# ============================================================================
# /FAIL — weak-link bar, closed-form failure strain
# ============================================================================

def test_fail_johnson_weak_link_bar(make_deck):
    """A bar with one WEAKER element (lower yield, both perfectly
    plastic): plastic flow localizes there, and with D2 = 0 the
    Johnson-Cook failure strain is exactly D1 regardless of
    triaxiality. The weak element must be deleted at eps_p ~= D1, the
    strong ones must stay elastic and alive, and the run must finish
    cleanly (the deleted element's energy is booked, not lost)."""
    nel = 4
    nodes, bricks, nid = _bar_bricks(nel)
    weak = bricks[0]
    strong = bricks[1:]
    fixed = "1 2 3 4"
    pulled = " ".join(str(4 * nel + k) for k in (1, 2, 3, 4))
    starter = (
        "/BEGIN\nweak link\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/2\n" + weak + "\n"
        "/BRICK/1\n" + "\n".join(strong) + "\n"
        "/PART/1\nstrong\n1 1\n"
        "/PART/2\nweak\n1 2\n"
        "/MAT/LAW2/1\nstrong steel\n7.8e-6\n210. 0.3\n0.40 0.0 1.0\n"
        "/MAT/LAW2/2\nweak steel\n7.8e-6\n210. 0.3\n0.36 0.0 1.0\n"
        "/FAIL/JOHNSON/2\n0.10 0 0 0 0\n"
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/1\nfixed\n{fixed}\n"
        f"/GRNOD/NODE/2\npulled\n{pulled}\n"
        "/BCS/1\nhold x\n100 000 0 1\n"
        "/FUNCT/1\nramp\n0.0 0.0\n0.05 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npull\n1 X 2 0.2\n"
        "/END\n"
    )
    engine = "/RUN/WLNK/1\n0.9\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n15.0\n"
    model = _run(make_deck, "WLNK", starter, engine)
    st = model.bricks.state
    part_ids = st["part_ids"]
    weak_el = part_ids == 2
    # exactly the weak element was deleted, at eps_p ~ D1 = 0.10
    assert np.all(st["off"][weak_el] == 0.0)
    assert np.all(st["off"][~weak_el] == 1.0)
    assert st["epsp"][weak_el][0] == pytest.approx(0.10, rel=0.05)
    assert st["dama"][weak_el][0] >= 1.0
    # strong elements stayed essentially elastic — the steady flow stress
    # 0.36 never reaches their 0.40 yield; only the start-up stress ring
    # of the imposed velocity can touch it, leaving a e-4-order transient
    # plastic strain, 3 orders below the weak element's 0.10
    assert np.abs(st["epsp"][~weak_el]).max() < 1e-3
    # the run stayed healthy after the rupture
    v = model.v[model.mass < 1e29]
    assert np.all(np.isfinite(v))


# ============================================================================
# LAW27 — brittle shell strip cracks through
# ============================================================================

def test_law27_shell_strip_cracks_and_deletes(make_deck):
    """A strip of brittle LAW27 shells pulled past the rupture strain:
    at least one element must crack through (all layers broken -> element
    deleted, LAW27 rule), the crack releases the strip and the run stays
    clean. Membrane tension strains every layer identically, so layer
    bookkeeping and the all-layers deletion rule are both exercised."""
    nel = 4
    nodes = []
    for i in range(nel + 1):
        nodes.append(f"{2 * i + 1} {float(i)} 0.0 0.0")
        nodes.append(f"{2 * i + 2} {float(i)} 1.0 0.0")
    shells = [f"{i + 1} {2 * i + 1} {2 * i + 3} {2 * i + 4} {2 * i + 2}"
              for i in range(nel)]
    pulled = f"{2 * nel + 1} {2 * nel + 2}"
    starter = (
        "/BEGIN\nbrittle strip\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/SHELL/1\n" + "\n".join(shells) + "\n"
        "/PART/1\nglass\n1 1\n"
        "/MAT/LAW27/1\nglass\n2.5e-6\n70. 0.2\n"
        "0.001 0.002 0.999 0.0025\n"
        "/PROP/SHELL/1\nglass\n1 0 0 0\n0.01 0.01 0.01 0 0\n3 0 0.1\n"
        "/GRNOD/NODE/1\nfixed\n1 2\n"
        f"/GRNOD/NODE/2\npulled\n{pulled}\n"
        "/BCS/1\nhold x\n100 000 0 1\n"
        "/FUNCT/1\nramp\n0.0 0.0\n0.05 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npull\n1 X 2 0.05\n"
        "/END\n"
    )
    engine = "/RUN/BRIT/1\n1.0\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n15.0\n"
    model = _run(make_deck, "BRIT", starter, engine)
    st = model.shells.state
    deleted = st["off"] == 0.0
    assert np.any(deleted)                       # the strip cracked through
    # deleted elements carry no stress at all
    assert np.abs(st["sig"][deleted]).max() == 0.0
    v = model.v[model.mass < 1e29]
    assert np.all(np.isfinite(v))


# ============================================================================
# Plastic beams — axial saturation, full run
# ============================================================================

def test_plastic_beam_bar_axial_saturation(make_deck):
    """Two elastic-perfectly-plastic beams pulled axially past yield:
    the axial resultant must saturate at exactly N = A * sigma_y (the
    global model's closed form) and the plastic strain must localize the
    difference between total and elastic stretch."""
    starter = (
        "/BEGIN\nplastic beam bar\n"
        "/NODE\n1 0 0 0\n2 10 0 0\n3 20 0 0\n9 0 50 0\n"
        "/BEAM/1\n1 1 2 9\n2 2 3 9\n"
        "/PART/1\nbeam\n1 1\n"
        "/MAT/LAW2/1\nepp steel\n7.8e-6\n210. 0.3\n0.4 0.0 1.0\n"
        "/PROP/BEAM/1\nsquare 5x5\n25.0 52.0833333 52.0833333 88.0\n"
        "/GRNOD/NODE/1\nroot\n1\n"
        "/GRNOD/NODE/2\ntip\n3\n"
        "/BCS/1\nclamp\n111 111 0 1\n"
        "/FUNCT/1\nramp\n0.0 0.0\n0.05 1.0\n100.0 1.0\n"
        "/IMPVEL/1\npull\n1 X 2 0.5\n"
        "/END\n"
    )
    engine = "/RUN/PBEA/1\n0.8\n/DT\n0.9 0\n/PRINT/-100000\n/STOP\n15.0\n"
    model = _run(make_deck, "PBEA", starter, engine)
    st = model.beams.state
    # total pull ~0.5*(0.8-0.025) = 0.39 mm on 20 mm -> ~2% >> yield 0.19%
    assert st["epsp"].min() > 0.005
    assert st["fres"][:, 0] == pytest.approx(25.0 * 0.4, rel=1e-6)
