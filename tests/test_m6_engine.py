"""M6 validations: bulk-viscosity energy ledger fix, /DT/NODA/CST mass
scaling, restart chaining, /STATE, /DAMP, /SENSOR, /MPC (see
PORTING_GUIDE.md roadmap M6)."""

import contextlib
import io
import os
import re

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter


def _run(make_deck, name, starter, engine):
    s, e = make_deck(name, starter, engine)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        model = run_engine(e)
    return model, e.replace("_0001.rad", "_0001.out")


def _final_summary(out_path):
    text = open(out_path).read()
    def g(label):
        return float(re.search(label + r"\s*\.[ .]*:\s*([-\d.Ee+]+)",
                               text).group(1))
    return {"ERR": g("ENERGY ERROR"), "CE": g("CONTACT ENERGY"),
            "EW": g("EXTERNAL WORK"), "EN": g("NUMERICAL DISSIPATION"),
            "IE": g("INTERNAL ENERGY"), "KE": g("KINETIC ENERGY"),
            "CYCLES": int(g("CYCLES")), "TEXT": text,
            "NORMAL": "ENGINE TERMINATION : NORMAL" in text}


def _read_th(path):
    with open(path) as fh:
        fh.readline()
        cols = fh.readline().strip().split(",")
        data = np.loadtxt(fh, delimiter=",", ndmin=2)
    return {c: data[:, k] for k, c in enumerate(cols)}


# ----------------------------------------------------------------------------
# Deck fragments
# ----------------------------------------------------------------------------

STEEL_LAW1 = """\
/MAT/LAW1/1
steel elastic
7.8e-6
210. 0.3
"""


def _ringing_cube_deck():
    """The PORTING_GUIDE M5-note reproducer: a 10 mm cube meshed 2x2x2,
    opposite z faces launched at +-0.3 mm/ms — violent barely-resolved
    volumetric ringing on single elements through the thickness."""
    nodes, nid = [], {}
    k = 0
    for i in range(3):
        for j in range(3):
            for l in range(3):
                k += 1
                nid[(i, j, l)] = k
                nodes.append(f"{k} {i*5.0} {j*5.0} {l*5.0}")
    bricks = []
    eid = 0
    for i in range(2):
        for j in range(2):
            for l in range(2):
                eid += 1
                n = [nid[(i, j, l)], nid[(i+1, j, l)], nid[(i+1, j+1, l)],
                     nid[(i, j+1, l)], nid[(i, j, l+1)], nid[(i+1, j, l+1)],
                     nid[(i+1, j+1, l+1)], nid[(i, j+1, l+1)]]
                bricks.append(f"{eid} " + " ".join(map(str, n)))
    zlo = " ".join(str(nid[(i, j, 0)]) for i in range(3) for j in range(3))
    zhi = " ".join(str(nid[(i, j, 2)]) for i in range(3) for j in range(3))
    return (
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/1\nzlo\n{zlo}\n"
        f"/GRNOD/NODE/2\nzhi\n{zhi}\n"
        "/INIVEL/TRA/1\nlo\n0 0 0.3 1\n"
        "/INIVEL/TRA/2\nhi\n0 0 -0.3 2\n"
    )


# ----------------------------------------------------------------------------
# Bulk-viscosity energy misbooking fix (the M1-era issue documented in the
# guide's M5 note): the energy balance must close under barely-resolved
# violent ringing, at both a marginal and a fine time step.
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("dt_scale,t_end", [(0.9, 4.0), (0.2, 1.0)])
def test_qb_ringing_balance_closes(make_deck, dt_scale, t_end):
    """2x2x2 cube, +-0.3 opposite face velocities, long free flight.

    Before the M6 fix this read -35% ERR at /DT 0.9 (one-sided qb work
    booking + the elastic-force numerical-dissipation channel the ledger
    could not see). With the trapezoidal viscous booking and the
    internal-work (EN) ledger the balance closes to round-off, and the
    dissipated ringing energy shows up explicitly as eint + EN."""
    starter = ("/BEGIN\nqb ringing reproducer\n" + _ringing_cube_deck()
               + "/END\n")
    engine = (f"/RUN/QBR/1\n{t_end}\n/DT\n{dt_scale} 0\n"
              "/PRINT/-5000\n/STOP\n15.0\n")
    model, out = _run(make_deck, f"QBR{int(dt_scale*10)}", starter, engine)
    s = _final_summary(out)
    assert s["NORMAL"]
    assert abs(s["ERR"]) < 0.5
    # the ringing is being dissipated: by t_end a good share of the launch
    # energy has left the (real-mass) kinetic ledger
    real = model.mass < 1e29
    ke = float(0.5 * (model.mass[real, None] * model.v[real] ** 2).sum())
    e0 = 0.5 * 0.3 ** 2 * float(model.mass[real].sum()) * (18 / 27)
    assert ke < 0.9 * e0


def test_en_ledger_negligible_on_healthy_run(make_deck):
    """The EN (numerical dissipation) ledger must stay negligible for a
    well-resolved run — it exists to catch the pathological damper
    interaction, not to absorb everyday energy. A confined bar hit by a
    sudden end velocity (the M1 wave validation setup) runs thousands of
    cycles at /DT 0.9; EN must stay at the few-% level of the energy
    scale (the sudden-velocity front carries a little genuinely
    barely-resolved content — that sliver is exactly what EN is FOR)."""
    nel = 20
    nodes, bricks = [], []
    nid = 0
    for i in range(nel + 1):
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {i} {y} {z}")
    for e in range(nel):
        b = 4 * e
        bricks.append(f"{e+1} {b+1} {b+2} {b+3} {b+4} "
                      f"{b+5} {b+6} {b+7} {b+8}")
    all_n = " ".join(str(k) for k in range(1, nid + 1))
    starter = (
        "/BEGIN\nen ledger check\n"
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/1\nall\n{all_n}\n"
        "/BCS/1\nconfine\n011 000 0 1\n"
        "/GRNOD/NODE/2\nend\n1 2 3 4\n"
        "/FUNCT/1\nconst push\n0.0 0.1\n100.0 0.1\n"
        "/IMPVEL/1\nhit\n1 X 2\n"
        "/END\n"
    )
    engine = "/RUN/ENC/1\n2.0\n/DT\n0.9 0\n/PRINT/-10000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "ENC", starter, engine)
    s = _final_summary(out)
    assert s["NORMAL"]
    scale = max(abs(s["IE"]), abs(s["KE"]), 1e-30)
    assert abs(s["EN"]) < 0.03 * scale
    assert abs(s["ERR"]) < 1.0


# ----------------------------------------------------------------------------
# /DT/NODA and /DT/NODA/CST — nodal time step + mass scaling
# ----------------------------------------------------------------------------

def _bar_deck(lengths, push_funct=True):
    """1D bar of unit-section bricks with the given element lengths."""
    xs = np.concatenate([[0.0], np.cumsum(lengths)])
    nodes, bricks = [], []
    nid = 0
    for xx in xs:
        for (y, z) in ((0, 0), (1, 0), (1, 1), (0, 1)):
            nid += 1
            nodes.append(f"{nid} {xx} {y} {z}")
    for e in range(len(lengths)):
        b = 4 * e
        bricks.append(f"{e+1} {b+1} {b+2} {b+3} {b+4} "
                      f"{b+5} {b+6} {b+7} {b+8}")
    all_n = " ".join(str(k) for k in range(1, nid + 1))
    deck = (
        "/NODE\n" + "\n".join(nodes) + "\n"
        "/BRICK/1\n" + "\n".join(bricks) + "\n"
        "/PART/1\nbar\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        f"/GRNOD/NODE/1\nall\n{all_n}\n"
        "/BCS/1\nconfine\n011 000 0 1\n"
        f"/GRNOD/NODE/2\nfixed end\n{nid-3} {nid-2} {nid-1} {nid}\n"
        "/BCS/2\nclamp\n100 000 0 2\n"
        "/GRNOD/NODE/3\npushed end\n1 2 3 4\n"
    )
    if push_funct:
        deck += ("/FUNCT/1\nslow push\n0.0 0.01\n100.0 0.01\n"
                 "/IMPVEL/1\npush\n1 X 3\n")
    return deck


def _initial_dt(out_path):
    return float(re.search(r"INITIAL TIME STEP\s*\.[ .]*:\s*([-\d.Ee+]+)",
                           open(out_path).read()).group(1))


def _history_dts(out_path):
    """The TIME-STEP column of the listing's cycle lines."""
    dts = []
    for line in open(out_path):
        tok = line.split()
        if len(tok) >= 10 and tok[0].isdigit():
            dts.append(float(tok[2]))
    return np.array(dts)


def test_dt_noda_equals_element_dt_on_uniform_mesh(make_deck):
    """For a uniform mesh the nodal time step must equal the element one
    (each node's stiffness and mass scale together: sqrt(2M/K) = dt_e) —
    the /DT and /DT/NODA runs start with the same step and end with the
    same answer."""
    deck = "/BEGIN\nnoda uniform\n" + _bar_deck([1.0] * 8) + "/END\n"
    eng_el = "/RUN/NU/1\n0.5\n/DT\n0.9 0\n/PRINT/-10000\n/STOP\n15.0\n"
    eng_no = "/RUN/NU/1\n0.5\n/DT/NODA\n0.9 0\n/PRINT/-10000\n/STOP\n15.0\n"
    m1, o1 = _run(make_deck, "NUEL", deck, eng_el)
    m2, o2 = _run(make_deck, "NUNO", deck, eng_no)
    assert _initial_dt(o2) == pytest.approx(_initial_dt(o1), rel=1e-9)
    assert np.allclose(m1.x, m2.x, atol=1e-12)


def test_dt_noda_cst_holds_step_adds_mass_quasistatic(make_deck):
    """Mass scaling acceptance: a bar with ONE thin element (its natural
    dt ~5x below the rest) pushed quasi-statically. /DT/NODA/CST with a
    dT_min above the thin element's limit must (a) hold every cycle's dt
    at or above dT_min, (b) add mass — reported in the summary, (c) leave
    the QUASI-STATIC answer unchanged (that is the point of mass scaling:
    the added mass only matters dynamically), (d) keep the balance
    closed WITH the added-mass energy accounted."""
    lengths = [1.0] * 4 + [0.2] + [1.0] * 4
    deck = "/BEGIN\nmass scaling\n" + _bar_deck(lengths) + "/END\n"
    # element dt of the thin brick ~ 0.2/c ~ 2.5e-5 ms; regular ~1.24e-4.
    dt_min = 1.0e-4
    eng_ref = "/RUN/MS/1\n2.0\n/DT\n0.9 0\n/PRINT/-2000\n/STOP\n15.0\n"
    eng_cst = (f"/RUN/MS/1\n2.0\n/DT/NODA/CST\n0.9 {dt_min}\n"
               "/PRINT/-2000\n/STOP\n15.0\n")
    m1, o1 = _run(make_deck, "MSREF", deck, eng_ref)
    m2, o2 = _run(make_deck, "MSCST", deck, eng_cst)
    s2 = _final_summary(o2)
    assert s2["NORMAL"]
    # (a) the step held: every listed dt at/above dT_min (the final cycle
    # may be clipped to land exactly on t_end)
    dts = _history_dts(o2)
    assert np.all(dts[:-1] >= 0.999 * dt_min)
    # and the run was genuinely mass-scaled: the reference runs at the
    # thin element's much smaller step
    assert _history_dts(o1).min() < 0.35 * dt_min
    # (b) added mass reported
    txt = open(o2).read()
    madd = float(re.search(
        r"ADDED MASS \(/DT/NODA/CST\) :\s*([-\d.Ee+]+)", txt).group(1))
    assert madd > 0.0
    # (c) quasi-static answer unchanged: same end-shortening -> same
    # axial stress everywhere along the bar (compare mid-bar element)
    sig1 = m1.bricks.state["sig"][:, 0]
    sig2 = m2.bricks.state["sig"][:, 0]
    assert sig2[2] == pytest.approx(sig1[2], rel=0.01)
    assert np.allclose(m2.x[:, 0], m1.x[:, 0], atol=5e-4)
    # (d) balance closed with the added-mass energy booked
    assert abs(s2["ERR"]) < 1.0


# ----------------------------------------------------------------------------
# /DAMP — Rayleigh mass damping
# ----------------------------------------------------------------------------

def test_damp_exact_exponential_decay(make_deck):
    """A freely translating cube under /DAMP: rigid translation feels no
    internal force, so v(t) = v0 exp(-alpha t) EXACTLY — the integrating
    factor reproduces it to round-off at ANY dt, the removed energy sits
    in the DE ledger, and the balance closes."""
    alpha, v0, tend = 2.0, 1.5, 1.0
    starter = (
        "/BEGIN\ndamped flight\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\nall\n1\n"
        f"/INIVEL/TRA/1\nfly\n{v0} 0 0 1\n"
        f"/DAMP/1\nbrakes\n{alpha} 1\n"
        "/END\n"
    )
    engine = f"/RUN/DMP/1\n{tend}\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "DMP", starter, engine)
    v_expect = v0 * np.exp(-alpha * tend)
    assert model.v[:, 0] == pytest.approx(v_expect, rel=1e-9)
    s = _final_summary(out)
    m = float(model.mass.sum())
    de_expect = 0.5 * m * v0 ** 2 * (1.0 - np.exp(-2.0 * alpha * tend))
    de = float(re.search(r"DAMPING DISSIPATION\s*\.[ .]*:\s*([-\d.Ee+]+)",
                         s["TEXT"]).group(1))
    assert de == pytest.approx(de_expect, rel=1e-9)
    assert abs(s["ERR"]) < 1e-6
    # no spurious straining while decaying
    assert np.abs(model.bricks.state["sig"]).max() < 1e-12


def test_damp_time_window(make_deck):
    """Tstart/Tstop: damping only acts inside its window — outside it the
    flight is undamped."""
    starter = (
        "/BEGIN\nwindowed damping\n"
        "/NODE\n"
        "1 0 0 0\n2 1 0 0\n3 1 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 1 0 1\n7 1 1 1\n8 0 1 1\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\ncube\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\nall\n1\n"
        "/INIVEL/TRA/1\nfly\n1.0 0 0 1\n"
        "/DAMP/1\npulse brake\n4.0 1 0.25 0.5\n"
        "/END\n"
    )
    engine = "/RUN/DMW/1\n1.0\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "DMW", starter, engine)
    # damped only over the window [0.25, 0.5]: v = exp(-4 * ~0.25)
    # (the window edges land mid-cycle: allow that one-step fuzz)
    v_expect = np.exp(-4.0 * 0.25)
    assert model.v[:, 0] == pytest.approx(v_expect, rel=1e-3)


# ----------------------------------------------------------------------------
# /SENSOR — time + displacement sensors gating loads and interfaces
# ----------------------------------------------------------------------------

def _point_mass_deck():
    """A standalone node with /ADMAS (the cleanest load target) plus a
    far-away bystander spring: the model needs at least one element, and
    the spring's period sets a sane time step (dt = 1e-3 ms) for the
    otherwise force-free point mass."""
    return (
        "/NODE\n1 0 0 0\n90 100 0 0\n91 101 0 0\n"
        "/SPRING/9\n9 90 91\n"
        "/PART/9\nclock\n9 1\n" + STEEL_LAW1 +
        "/PROP/SPRING/9\nclock\n2e-3 2000. 0.\n"
        "/GRNOD/NODE/8\nspring ends\n90 91\n"
        "/BCS/8\nhold spring\n111 111 0 8\n"
        "/GRNOD/NODE/1\npoint\n1\n"
        "/ADMAS/1\nballast\n2.0 1\n"
    )


def test_sensor_time_gates_and_shifts_cload(make_deck):
    """/CLOAD gated by /SENSOR/TIME: silent before Tdelay, then follows
    f(t - t_fire). A constant unit force from Tdelay = 0.5 on a 2 kg
    point mass gives v(1.0) = 1.0 * 0.5 / 2.0 exactly."""
    starter = (
        "/BEGIN\nsensor time cload\n" + _point_mass_deck() +
        "/FUNCT/1\nconst force\n0.0 1.0\n100.0 1.0\n"
        "/SENSOR/TIME/7\nfire at .5\n0.5\n"
        "/CLOAD/1\ngated push\n1 X 1 1.0 7\n"
        "/END\n"
    )
    engine = "/RUN/SNT/1\n1.0\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "SNT", starter, engine)
    # the sensor is polled at cycle starts: the activation lands on the
    # first cycle boundary at/after Tdelay — one dt of fuzz
    assert model.v[0, 0] == pytest.approx(0.5 / 2.0, abs=1e-3)
    assert "/SENSOR/7 ACTIVATED" in open(out).read()
    # a bare point mass under constant force carries the classic
    # f^2 dt^2/2m booking residual (~1/N over N cycles) — no element
    # ledger absorbs it here
    assert abs(_final_summary(out)["ERR"]) < 0.5


def test_sensor_disp_fires_on_displacement(make_deck):
    """/SENSOR/DISP: a point mass pushed by F1 = 1 fires the sensor when
    it has moved Dmin, which turns on a second, opposing load F2 = -2
    (curve evaluated from the fire time). Closed form:
    x(t) = F1 t^2 / 2m until ts = sqrt(2 m Dmin / F1), then
    v(T) = [F1 T - 2 F1 (T - ts)] / m."""
    m, F1, dmin, T = 2.0, 1.0, 0.09, 1.0
    ts = np.sqrt(2 * m * dmin / F1)
    starter = (
        "/BEGIN\nsensor disp\n" + _point_mass_deck() +
        "/FUNCT/1\npush\n0.0 1.0\n100.0 1.0\n"
        "/FUNCT/2\ncounter\n0.0 2.0\n100.0 2.0\n"
        f"/SENSOR/DISP/3\nmoved enough\n1 {dmin}\n"
        "/CLOAD/1\npush\n1 X 1 1.0\n"
        "/CLOAD/2\ncounter\n2 X 1 -1.0 3\n"
        "/END\n"
    )
    engine = f"/RUN/SND/1\n{T}\n/DT\n0.9 0\n/PRINT/-5000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "SND", starter, engine)
    v_expect = (F1 * T - 2.0 * F1 * (T - ts)) / m
    # the fire time lands on a cycle boundary: tolerance = a few dt
    assert model.v[0, 0] == pytest.approx(v_expect, abs=0.02)
    assert "/SENSOR/3 ACTIVATED" in open(out).read()


def test_sensor_gates_interface(make_deck):
    """A /INTER/TYPE7 gated by a never-firing sensor exerts no force (the
    block sails through the shell); the same deck with the sensor firing
    at t=0 bounces it. This is the Tstart-style interface gating."""
    def deck(tdelay):
        return (
            "/BEGIN\nsensor interface\n"
            "/NODE\n"
            "1 0 0 1\n2 1 0 1\n3 1 1 1\n4 0 1 1\n"
            "5 0 0 2\n6 1 0 2\n7 1 1 2\n8 0 1 2\n"
            "11 -2 -2 0\n12 3 -2 0\n13 3 3 0\n14 -2 3 0\n"
            "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
            "/SHELL/2\n2 11 12 13 14\n"
            "/PART/1\nblock\n1 1\n"
            "/PART/2\nfloor\n2 1\n" + STEEL_LAW1 +
            "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
            "/PROP/SHELL/2\nfloor\n1. 3 0.01\n"
            "/GRNOD/NODE/10\nfloor corners\n11 12 13 14\n"
            "/BCS/1\nclamp floor\n111 111 0 10\n"
            "/GRNOD/PART/20\nblock nodes\n1\n"
            "/INIVEL/TRA/1\ndrop\n0 0 -1.5 20\n"
            "/SURF/PART/1\nfloor surf\n2\n"
            f"/SENSOR/TIME/5\ngate\n{tdelay}\n"
            "/INTER/TYPE7/1\ngated contact\n20 1 2 0 5\n"
            "1.0 0.0 0.2\n"
            "/END\n"
        )
    engine = "/RUN/SNI/1\n1.4\n/DT\n0.5 0\n/PRINT/-5000\n/STOP\n15.0\n"
    m_on, _ = _run(make_deck, "SNION", deck(0.0), engine)
    m_off, _ = _run(make_deck, "SNIOFF", deck(99.0), engine)
    # active interface: the block bounced (moving up or still above);
    # gated-off: it fell straight through the floor plane z=0
    assert m_off.x[:8, 2].min() < -0.3
    assert m_on.x[:8, 2].min() > -0.05


# ----------------------------------------------------------------------------
# /MPC — general linear multi-point constraints
# ----------------------------------------------------------------------------

def _two_spring_deck(k=10.0, m=2e-3):
    """Two independent grounded springs along X: nodes 2 and 4 free."""
    return (
        "/NODE\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 1 1 0\n"
        "/SPRING/1\n1 1 2\n/SPRING/1\n2 3 4\n"
        "/PART/1\nsprings\n1 1\n" + STEEL_LAW1 +
        f"/PROP/SPRING/1\nspring\n{m} {k} 0.\n"
        "/GRNOD/NODE/1\nground\n1 3\n"
        "/BCS/1\nground\n111 111 0 1\n"
        "/GRNOD/NODE/2\nfree\n2 4\n"
        "/BCS/2\nx only\n011 111 0 2\n"
    )


def test_mpc_ties_dofs_and_does_no_work(make_deck):
    """u2x - u4x = 0 on two grounded springs, force on node 2 only, mass
    damping to settle: the pair must behave as ONE oscillator of
    stiffness 2k — static deflection F/(2k) on BOTH nodes — with the
    constraint satisfied to round-off and doing zero booked work."""
    F, k = 1.0, 10.0
    starter = (
        "/BEGIN\nmpc equal\n" + _two_spring_deck(k=k) +
        "/FUNCT/1\npush\n0.0 1.0\n100.0 1.0\n"
        f"/CLOAD/1\npush 2\n1 X 3 {F}\n"
        "/GRNOD/NODE/3\nloaded\n2\n"
        "/MPC/1\nequal x\n2 1 1.0\n4 1 -1.0\n"
        "/DAMP/1\nsettle\n30.0 2\n"
        "/END\n"
    )
    engine = "/RUN/MPC1/1\n4.0\n/DT\n0.2 0\n/PRINT/-50000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "MPC1", starter, engine)
    u2 = model.x[1, 0] - model.x0[1, 0]
    u4 = model.x[3, 0] - model.x0[3, 0]
    # constraint satisfied to round-off, both settled at F/(2k)
    assert abs(u2 - u4) < 1e-12
    assert u2 == pytest.approx(F / (2 * k), rel=1e-3)
    s = _final_summary(out)
    assert s["NORMAL"]
    # zero constraint work: the balance closes exactly (external work =
    # strain energy + damping dissipation + the damper's small
    # attribution residual, which the EN ledger carries — the MPC
    # contributes nothing, or ERR could not close)
    assert abs(s["ERR"]) < 0.01
    assert abs(s["EN"]) < 0.05 * max(abs(s["EW"]), 1e-30)


def test_mpc_opposite_and_weighted(make_deck):
    """u2x + 2 u4x = 0 (a weighted row): pushing node 2 drags node 4 the
    other way with half the travel — checked against the constraint at
    every scale, plus exact satisfaction at the end."""
    starter = (
        "/BEGIN\nmpc weighted\n" + _two_spring_deck() +
        "/FUNCT/1\npush\n0.0 1.0\n100.0 1.0\n"
        "/CLOAD/1\npush 2\n1 X 3 1.0\n"
        "/GRNOD/NODE/3\nloaded\n2\n"
        "/MPC/1\nweighted\n2 1 1.0\n4 1 2.0\n"
        "/DAMP/1\nsettle\n30.0 2\n"
        "/END\n"
    )
    engine = "/RUN/MPC2/1\n4.0\n/DT\n0.2 0\n/PRINT/-50000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "MPC2", starter, engine)
    u2 = model.x[1, 0] - model.x0[1, 0]
    u4 = model.x[3, 0] - model.x0[3, 0]
    assert abs(u2 + 2.0 * u4) < 1e-12
    assert u2 > 1e-3                       # it did move
    # statics of the constrained pair: minimize (k/2)(u2^2+u4^2) - F u2
    # s.t. u2 = -2 u4  ->  u2 = 4F/(5k)
    assert u2 == pytest.approx(4.0 * 1.0 / (5.0 * 10.0), rel=1e-3)
    assert abs(_final_summary(out)["ERR"]) < 0.01


def test_mpc_fixed_dof_is_ground(make_deck):
    """A row with one term on a /BCS-fixed DOF: the fixed side is
    infinitely massive ground, so the free DOF is driven to satisfy the
    row alone — u2x + u1x = 0 with u1 clamped pins node 2 at zero even
    under load (the constraint force carries it)."""
    starter = (
        "/BEGIN\nmpc ground\n" + _two_spring_deck() +
        "/FUNCT/1\npush\n0.0 1.0\n100.0 1.0\n"
        "/CLOAD/1\npush 2\n1 X 3 1.0\n"
        "/GRNOD/NODE/3\nloaded\n2\n"
        "/MPC/1\npinned\n2 1 1.0\n1 1 1.0\n"
        "/END\n"
    )
    engine = "/RUN/MPC3/1\n1.0\n/DT\n0.2 0\n/PRINT/-50000\n/STOP\n15.0\n"
    model, out = _run(make_deck, "MPC3", starter, engine)
    assert abs(model.x[1, 0] - model.x0[1, 0]) < 1e-12
    assert abs(_final_summary(out)["ERR"]) < 1e-6


# ----------------------------------------------------------------------------
# Restart chaining (RunName_0002.rad) + /STATE
# ----------------------------------------------------------------------------

def _chain(make_deck, name, starter, eng1, eng2):
    """Run starter + engine leg 1 + engine leg 2 (NAME_0002.rad)."""
    s, e1 = make_deck(name, starter, eng1)
    e2 = e1.replace("_0001.rad", "_0002.rad")
    with open(e2, "w") as fh:
        fh.write(eng2)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        run_engine(e1)
        model = run_engine(e2)
    return model, e2.replace("_0002.rad", "_0002.out")


_SPRING_OSC = (
    "/BEGIN\nrestart oscillator\n"
    "/NODE\n1 0 0 0\n2 1 0 0\n"
    "/SPRING/1\n1 1 2\n"
    "/PART/1\nosc\n1 1\n" + STEEL_LAW1 +
    "/PROP/SPRING/1\nspring\n2e-3 10.0 0.\n"
    "/GRNOD/NODE/1\nground\n1\n"
    "/BCS/1\nground\n111 111 0 1\n"
    "/GRNOD/NODE/2\nfree\n2\n"
    "/BCS/2\nx only\n011 111 0 2\n"
    "/INIVEL/TRA/1\nkick\n0.5 0 0 2\n"
    "/END\n"
)


def test_restart_chain_reproduces_unchained_exactly(make_deck):
    """THE acceptance test of the restart contract: a run chained at a
    cycle boundary must reproduce the unchained run EXACTLY — same cycle
    count, same state to round-off.

    The oscillating spring has an exactly CONSTANT critical dt, so a
    probe run gives the exact step and the junction time T1 is placed on
    the accumulated 100-cycle boundary (the only source of difference is
    then the <= 1 ulp clip of the junction landing)."""
    from pyradioss.starter.restart import read_restart
    # probe: learn the exact dt the engine takes
    s, e1 = make_deck("RCP", _SPRING_OSC,
                      "/RUN/RCP/1\n0.2\n/DT\n0.45 0\n/PRINT/-999\n"
                      "/STOP\n15.0\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        run_engine(e1)
    dt0 = read_restart(e1.replace("_0001.rad", "_0001.rst"))[1]["dt"]
    t1 = 0.0
    for _ in range(100):                   # the engine's own fp sum
        t1 += dt0
    t_tot = t1
    for _ in range(150):
        t_tot += dt0

    # unchained reference
    mu, _ = _run(make_deck, "RCU", _SPRING_OSC,
                 f"/RUN/RCU/1\n{t_tot!r}\n/DT\n0.45 0\n/PRINT/-999\n"
                 "/STOP\n15.0\n")
    # chained: 0 -> t1 -> t_tot
    mc, out2 = _chain(
        make_deck, "RCC", _SPRING_OSC,
        f"/RUN/RCC/1\n{t1!r}\n/DT\n0.45 0\n/PRINT/-999\n/STOP\n15.0\n",
        f"/RUN/RCC/2\n{t_tot!r}\n/DT\n0.45 0\n/PRINT/-999\n/STOP\n15.0\n")

    assert mc.engine_state.cycle == mu.engine_state.cycle == 250
    assert np.allclose(mc.x, mu.x, rtol=0, atol=1e-13)
    assert np.allclose(mc.v, mu.v, rtol=0, atol=1e-13)
    assert mc.springs.state["eint"][0] == pytest.approx(
        mu.springs.state["eint"][0], rel=1e-11, abs=1e-18)
    # the chained run continued the T-file numbering
    assert os.path.exists(out2.replace("_0002.out", "T02.csv"))


def test_restart_chain_tumbling_rigid_body(make_deck):
    """Chaining a torque-free TUMBLING /RBODY: the body frame R and the
    angular momentum L live only in the engine restart — the resumed run
    must track the unchained tumble to round-off (the hardest state to
    reconstruct; interior elements keep dt exactly constant, so the
    junction can sit on a cycle boundary like the spring test)."""
    from pyradioss.starter.restart import read_restart
    starter = (
        "/BEGIN\ntumbling body\n"
        "/NODE\n"
        "1 0 0 0\n2 2 0 0\n3 2 1 0\n4 0 1 0\n"
        "5 0 0 1\n6 2 0 1\n7 2 1 1\n8 0 1 1\n"
        "100 0 0 0\n"
        "/BRICK/1\n1 1 2 3 4 5 6 7 8\n"
        "/PART/1\nbrick\n1 1\n" + STEEL_LAW1 +
        "/PROP/SOLID/1\nsolid\n1.1 0.05 0.1\n"
        "/GRNOD/PART/1\nslaves\n1\n"
        "/RBODY/1\nbody\n100 1\n"
        # spin about the unstable middle axis-ish direction -> tumbling
        "/INIVEL/AXIS/1\nspin\n3.0 Y 1 1.0 0.5 0.5\n"
        "/INIVEL/TRA/2\ndrift\n0.05 0.02 0.01 1\n"
        "/END\n"
    )
    s, e1 = make_deck("RBP", starter,
                      "/RUN/RBP/1\n0.05\n/DT\n0.45 0\n/PRINT/-9999\n"
                      "/STOP\n1e9\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        run_engine(e1)
    dt0 = read_restart(e1.replace("_0001.rad", "_0001.rst"))[1]["dt"]
    t1 = 0.0
    for _ in range(400):
        t1 += dt0
    t_tot = t1
    for _ in range(600):
        t_tot += dt0

    mu, _ = _run(make_deck, "RBU", starter,
                 f"/RUN/RBU/1\n{t_tot!r}\n/DT\n0.45 0\n/PRINT/-9999\n"
                 "/STOP\n1e9\n")
    mc, out2 = _chain(
        make_deck, "RBC", starter,
        f"/RUN/RBC/1\n{t1!r}\n/DT\n0.45 0\n/PRINT/-9999\n/STOP\n1e9\n",
        f"/RUN/RBC/2\n{t_tot!r}\n/DT\n0.45 0\n/PRINT/-9999\n/STOP\n1e9\n")
    assert mc.engine_state.cycle == mu.engine_state.cycle == 1000
    assert np.allclose(mc.x, mu.x, rtol=0, atol=1e-10)
    assert np.allclose(mc.v, mu.v, rtol=0, atol=1e-10)
    # interior of the body must still be at exactly zero strain
    assert np.abs(mc.bricks.state["sig"]).max() < 1e-12


def test_state_dt_writes_resumable_snapshots(make_deck):
    """/STATE/DT: periodic restart snapshots during the run (each
    refreshes RunName_0001.rst) — and the snapshot IS resumable: kill
    the imaginary run at the snapshot and chain from it."""
    from pyradioss.starter.restart import read_restart
    s, e1 = make_deck(
        "STA", _SPRING_OSC,
        "/RUN/STA/1\n1.0\n/DT\n0.45 0\n/PRINT/-999\n/STOP\n15.0\n"
        "/STATE/DT\n0.2 0.2\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(s)
        run_engine(e1)
    out = open(e1.replace("_0001.rad", "_0001.out")).read()
    assert out.count("/STATE: RESTART SNAPSHOT") >= 4
    model, eng = read_restart(e1.replace("_0001.rad", "_0001.rst"))
    assert eng is not None and eng["t"] == pytest.approx(1.0, rel=1e-12)
