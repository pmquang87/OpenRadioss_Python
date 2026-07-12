"""
M16 validations: CONSISTENT element mass + MODAL (free-vibration) eigenvalue
analysis (/IMPL/EIGV).

Every new capability gets at least one ANALYTIC check (the port's philosophy):

CONSISTENT ELEMENT MASS (per family)
* partition of unity: the translational consistent mass rows sum to the
  element mass (the lumped nodal mass), so the total is the element mass;
* rigid-body translational kinetic energy 1/2 v^T M v = 1/2 m |v|^2 EXACT;
* symmetry and positive (semi-)definiteness of every element mass;
* the consistent_mass() call NEVER mutates element state (the M7 parity
  contract — like test_m14/test_m15);
* the LUMPED-mass path (model.mass/inertia, dynamics._lumped_mass_eq) is
  BIT-IDENTICAL with the consistent mass present — the mass analogue of the
  M13 mu=0 / M15 Ifric=0 contracts.

MODAL EIGENVALUE EXTRACTION
* longitudinal bar frequencies vs (2n-1) c / (4L) (fixed-free) and n c/(2L)
  (free-free, the rigid mode filtered);
* cantilever bending vs the Euler-Bernoulli (beta_n L) roots
  (1.875, 4.694, 7.855 -> f_n) to a few percent;
* a simply-supported plate fundamental vs pi/a^2 sqrt(D/(rho t))
  (documented: the BT4 SHEAR-LOCKS in thin bending, a pre-existing element
  property, so the check uses a genuine a/t = 20 plate and shows the error
  shrinking on refinement — the MASS is validated by consistent ~= lumped);
* mode-shape M-orthogonality phi_i^T M phi_j = delta_ij to round-off;
* the consistent-vs-lumped BRACKET: lumped under-predicts, consistent
  over-predicts, both straddle the exact frequency;
* a CONSTRAINT-condensed case — a rigid link doubling a tip mass halves the
  frequency (sqrt(k/(m1+m2))), and the rigid-body condensation of a
  distributed consistent mass reproduces the continuum parallel-axis inertia;
* a PRESTRESSED case — a truss string has NO transverse stiffness until it
  is tensioned, so its transverse modes are PURE geometric-stiffness modes
  at the exact taut-string frequencies n/(2L) sqrt(T/mu);
* the /IMPL/EIGV card mirror (a PORT card — freimpl.F has no modal branch)
  and its /STRS prestressed variant.

See PORTING_GUIDE.md roadmap M16.
"""

import contextlib
import io

import numpy as np
import pytest

from pyradioss.engine.engine import run_engine
from pyradioss.starter.starter import run_starter

pytest.importorskip("scipy")   # implicit requires scipy (optional otherwise)

import scipy.linalg as sla                                     # noqa: E402

from pyradioss.implicit.assembly import assemble, assemble_mass  # noqa: E402
from pyradioss.implicit.dofmap import DofMap                      # noqa: E402
from pyradioss.implicit.dynamics import _lumped_mass_eq           # noqa: E402
from pyradioss.implicit.modal import modal_frequencies            # noqa: E402


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _starter(text):
    import tempfile
    import os
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M16_0000.rad")
    with open(sp, "w") as f:
        f.write(text)
    with contextlib.redirect_stdout(io.StringIO()):
        return run_starter(sp)


def _run(starter_text, engine_text):
    import tempfile
    import os
    d = tempfile.mkdtemp()
    sp = os.path.join(d, "M16_0000.rad")
    ep = os.path.join(d, "M16_0001.rad")
    with open(sp, "w") as f:
        f.write(starter_text)
    with open(ep, "w") as f:
        f.write(engine_text)
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(sp)
        return run_engine(ep)


STEEL = """\
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.3
"""


# ----------------------------------------------------------------------------
# element-level decks (one small group per family)
# ----------------------------------------------------------------------------

def _bar_deck(N=8, L=100.0, A=1.0, fixed_free=True):
    """Axial chain of N trusses along x; node 1 pinned. ``fixed_free``:
    interior/tip free along x only (fixed-free bar). Else the tip is free in
    x too (free-free)."""
    dx = L / N
    nodes = "\n".join(f"{i+1:10d}{i*dx:20.10f}{0.0:20.10f}{0.0:20.10f}"
                      for i in range(N + 1))
    trs = "\n".join(f"{i+1:10d}{i+1:10d}{i+2:10d}" for i in range(N))
    alln = "\n".join(str(i + 1) for i in range(N + 1))
    pin = ("/BCS/1\npin\n       111       000         0         1\n"
           if fixed_free else "")
    return f"""\
#RADIOSS STARTER
/BEGIN
BAR
/NODE
{nodes}
/TRUSS/1
{trs}
/PART/1
bar
         1         1
/MAT/LAW1/1
steel
   7.8e-6
     210.0       0.0
/PROP/TRUSS/1
bar
       {A}
/GRNOD/NODE/1
pin
1
/GRNOD/NODE/2
all
{alln}
{pin}/BCS/2
transverse yz fixed
       011       000         0         2
/END
"""


def _cantilever_shell_strip(nx=16, L=100.0, b=10.0, t=1.0):
    """Shell cantilever strip (1 element wide), clamped at x = 0."""
    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = "\n".join(
        f"{nid(i,j):10d}{i*L/nx:20.10f}{j*b:20.10f}{0.0:20.10f}"
        for j in range(2) for i in range(nx + 1))
    els = "\n".join(
        f"{i+1:10d}{nid(i,0):10d}{nid(i+1,0):10d}{nid(i+1,1):10d}{nid(i,1):10d}"
        for i in range(nx))
    fixed = "\n".join(str(nid(0, j)) for j in range(2))
    return f"""\
#RADIOSS STARTER
/BEGIN
STRIP
/NODE
{nodes}
/SHELL/1
{els}
/PART/1
p
         1         1
{STEEL}/PROP/SHELL/1
sh
       {t}         5      0.833
/GRNOD/NODE/1
clamp
{fixed}
/BCS/1
clamp
       111       111         0         1
/END
"""


def _plate_deck(nx, a=100.0, t=5.0):
    """Square plate, simply supported (uz = 0 on the edges), in-plane and
    drilling fixed everywhere to isolate the transverse bending spectrum."""
    ny = nx

    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = "\n".join(
        f"{nid(i,j):10d}{i*a/nx:20.10f}{j*a/ny:20.10f}{0.0:20.10f}"
        for j in range(ny + 1) for i in range(nx + 1))
    els = []
    eid = 0
    for j in range(ny):
        for i in range(nx):
            eid += 1
            c = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            els.append(f"{eid:10d}" + "".join(f"{v:10d}" for v in c))
    edge = [nid(i, j) for j in range(ny + 1) for i in range(nx + 1)
            if i in (0, nx) or j in (0, ny)]
    alln = [nid(i, j) for j in range(ny + 1) for i in range(nx + 1)]
    return f"""\
#RADIOSS STARTER
/BEGIN
PLATE
/NODE
{nodes}
/SHELL/1
{chr(10).join(els)}
/PART/1
p
         1         1
{STEEL}/PROP/SHELL/1
sh
       {t}         5      0.833
/GRNOD/NODE/1
edge
{chr(10).join(str(n) for n in edge)}
/GRNOD/NODE/2
all
{chr(10).join(str(n) for n in alln)}
/BCS/1
ss edge uz
       001       000         0         1
/BCS/2
in-plane + drill fixed
       110       001         0         2
/END
"""


# ============================================================================
# CONSISTENT ELEMENT MASS — per-family unit properties
# ============================================================================

def _each_group(model):
    for name, group in model.element_groups():
        yield name, group


def test_consistent_mass_partition_of_unity_and_rigid_ke():
    """For every family, the translational consistent mass conserves mass
    (total = element mass) and is EXACT for rigid-body translational kinetic
    energy: 1/2 v^T M v = 1/2 m |v|^2. Also symmetry and PSD."""
    # one small model per family (solids, shells, beam, truss, spring)
    decks = {
        "trusses": _bar_deck(N=3),
        "shells": _cantilever_shell_strip(nx=3),
        "sh3n": _tri_strip_deck(nx=3),
        "bricks": _brick_deck(),
        "tetras": _tetra_deck(),
        "beams": _beam_deck(),
        "springs": _spring_deck(),
    }
    from pyradioss.elements import KERNELS
    for fam, deck in decks.items():
        model = _starter(deck)
        group = getattr(model, fam)
        assert group is not None and group.n > 0, fam
        me, edofs = KERNELS[fam].consistent_mass(group, model.x0)
        # symmetry
        assert np.allclose(me, np.transpose(me, (0, 2, 1)), atol=1e-12), fam
        # positive semidefinite (mass never removes energy)
        for k in range(group.n):
            w = np.linalg.eigvalsh(me[k])
            assert w.min() > -1e-9 * max(w.max(), 1.0), fam
        # per-element mass and rigid KE, in each of the 3 global translations
        # (the element mass is model-dependent; use the state mass array)
        m_el = group.state["mass"]
        d = me.shape[1]
        ndof_node = 6 if fam in ("shells", "sh3n", "beams") else 6
        nnode = d // ndof_node if fam in ("shells", "sh3n", "beams") \
            else d // 3
        for comp in range(3):
            v = np.zeros((group.n, d))
            if fam in ("shells", "sh3n", "beams"):
                for i in range(nnode):
                    v[:, i * 6 + comp] = 1.0
            else:
                for i in range(nnode):
                    v[:, i * 3 + comp] = 1.0
            ke = np.einsum("ni,nij,nj->n", v, me, v)
            # spring mass is a point property; the beam mass is on N1,N2 only
            assert np.allclose(ke, m_el, rtol=1e-9), (fam, comp)


def test_consistent_mass_does_not_mutate_state():
    """consistent_mass() reads geometry/mass and NEVER writes element state —
    the M7 force-path parity contract (like test_m14/test_m15). Snapshot every
    state array, call the mass, assert nothing changed."""
    from pyradioss.elements import KERNELS
    for fam, deck in {
            "trusses": _bar_deck(N=3), "shells": _cantilever_shell_strip(3),
            "beams": _beam_deck(), "bricks": _brick_deck()}.items():
        model = _starter(deck)
        group = getattr(model, fam)
        before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                  for k, v in group.state.items()}
        KERNELS[fam].consistent_mass(group, model.x0)
        for k, v in before.items():
            if isinstance(v, np.ndarray):
                assert np.array_equal(group.state[k], v), (fam, k)


# ============================================================================
# LUMPED-mass path stays BIT-IDENTICAL (the M16 opt-in contract)
# ============================================================================

def test_lumped_path_bit_identical_with_consistent_present():
    """Building the consistent mass NEVER perturbs the lumped path: model.mass
    / model.inertia and dynamics._lumped_mass_eq are bit-identical before and
    after assemble_mass runs (the mass analogue of the M13 mu=0 contract — the
    consistent mass is a NEW opt-in operator, never the default)."""
    model = _cantilever_shell_strip(nx=6)
    m = _starter(model)
    mass0 = m.mass.copy()
    inertia0 = m.inertia.copy()
    dof = DofMap(m)
    M_lumped0 = _lumped_mass_eq(m, dof)
    # now build the consistent mass (the new operator)
    _ = assemble_mass(m, dof, m.x0)
    # nothing on the lumped path moved
    assert np.array_equal(m.mass, mass0)
    assert np.array_equal(m.inertia, inertia0)
    assert np.array_equal(_lumped_mass_eq(m, dof), M_lumped0)


# ============================================================================
# MODAL — longitudinal bar
# ============================================================================

def test_bar_fixed_free_frequencies():
    """Fixed-free axial bar: f_n = (2n-1) c / (4L), c = sqrt(E/rho). The
    consistent mass over-predicts by < 1% at 20 elements."""
    L, E, rho, A, N = 100.0, 210.0, 7.8e-6, 1.0, 20
    c = np.sqrt(E / rho)
    m = _starter(_bar_deck(N=N, L=L, A=A, fixed_free=True))
    f, _, _ = modal_frequencies(m, nev=3)
    fn = np.array([(2 * n - 1) * c / (4 * L) for n in range(1, 4)])
    assert np.all(np.abs(f[:3] / fn - 1.0) < 0.01)


def test_bar_free_free_frequencies():
    """Free-free axial bar: f_n = n c / (2L). The single rigid-body
    translation mode is correctly filtered (lambda ~ 0)."""
    L, E, rho, N = 100.0, 210.0, 7.8e-6, 20
    c = np.sqrt(E / rho)
    m = _starter(_bar_deck(N=N, L=L, fixed_free=False))
    f, _, _ = modal_frequencies(m, nev=3)
    fn = np.array([n * c / (2 * L) for n in range(1, 4)])
    assert np.all(np.abs(f[:3] / fn - 1.0) < 0.02)


def test_consistent_vs_lumped_bracket():
    """The exact frequency is BRACKETED: the lumped mass under-predicts, the
    consistent mass over-predicts (the classic result — both converge on
    refinement). Fixed-free axial bar, first three modes."""
    L, E, rho, N = 100.0, 210.0, 7.8e-6, 20
    c = np.sqrt(E / rho)
    m = _starter(_bar_deck(N=N, L=L, fixed_free=True))
    dof = DofMap(m)
    K = assemble(m, dof, m.x0).toarray()
    Mc = assemble_mass(m, dof, m.x0).toarray()
    Ml = np.diag(_lumped_mass_eq(m, dof))

    def freqs(M):
        lam = sla.eigh(K, M, eigvals_only=True)
        lp = lam[lam > 1e-8 * np.median(lam[lam > 0])]
        return np.sqrt(lp) / (2 * np.pi)
    fc, fl = freqs(Mc), freqs(Ml)
    fex = np.array([(2 * n - 1) * c / (4 * L) for n in range(1, 4)])
    assert np.all(fl[:3] <= fex + 1e-9)          # lumped under-predicts
    assert np.all(fex <= fc[:3] + 1e-9)          # consistent over-predicts


# ============================================================================
# MODAL — cantilever bending (beam consistent mass)
# ============================================================================

def test_cantilever_beam_bending_euler_bernoulli():
    """Cantilever beam bending vs the Euler-Bernoulli roots beta_n L =
    1.875, 4.694, ... -> f_n = (beta_n L)^2/(2 pi) sqrt(EI/(rho A))/L^2. The
    Rayleigh (Hermite-cubic + rotary) consistent beam mass matches to < 2% at
    10 elements — validated on the antenna_mast example deck."""
    with contextlib.redirect_stdout(io.StringIO()):
        m = run_starter("examples/antenna_mast/MAST_0000.rad")
    f, _, _ = modal_frequencies(m, nev=4)
    E, I, rho, A, L = 210.0, 2.818e5, 7.8e-6, 707.5, 2000.0
    beta = np.array([1.875104, 4.694091])
    fn = beta ** 2 / (2 * np.pi) * np.sqrt(E * I / (rho * A)) / L ** 2
    # Iyy = Izz so the bending modes come in x/y pairs: f[0]==f[1], f[2]==f[3]
    assert abs(f[0] / fn[0] - 1.0) < 0.02
    assert abs(f[2] / fn[1] - 1.0) < 0.02
    assert abs(f[0] - f[1]) < 1e-6 * f[0]        # the degenerate pair


# ============================================================================
# MODAL — simply-supported plate fundamental
# ============================================================================

def test_plate_simply_supported_fundamental():
    """SS square plate fundamental vs pi/a^2 sqrt(D/(rho t)), D = E t^3/
    (12(1-nu^2)). The BT4 SHEAR-LOCKS in thin bending (a pre-existing element
    property, PORTING_GUIDE M16), so this uses a genuine a/t = 20 plate where
    the shell is accurate and shows the error SHRINKING on refinement toward
    the closed form."""
    a, t, E, nu, rho = 100.0, 5.0, 210.0, 0.3, 7.8e-6
    D = E * t ** 3 / (12 * (1 - nu ** 2))
    f_ss = np.pi / a ** 2 * np.sqrt(D / (rho * t))
    f_coarse = modal_frequencies(_starter(_plate_deck(12, a, t)), nev=1)[0][0]
    f_fine = modal_frequencies(_starter(_plate_deck(20, a, t)), nev=1)[0][0]
    # converging toward the closed form from above (discretization stiffer)
    assert f_fine < f_coarse
    assert abs(f_fine / f_ss - 1.0) < 0.08       # a few percent at nx=20


# ============================================================================
# MODAL — mode-shape M-orthogonality
# ============================================================================

def test_mode_shape_mass_orthogonality():
    """The extracted modes are M-orthonormal: phi_i^T M phi_j = delta_ij to
    round-off (eigh returns mass-normalized eigenvectors; the port expands
    them through the DofMap and reports them mass-normalized)."""
    L, N = 100.0, 12
    m = _starter(_bar_deck(N=N, L=L, fixed_free=True))
    dof = DofMap(m)
    M = assemble_mass(m, dof, m.x0)
    f, modes, _ = modal_frequencies(m, nev=5)
    # rebuild the equation-space mode vectors and check M-orthonormality
    vecs = []
    for du, dur in modes:
        vecs.append(dof.gather_residual(du, dur))
    V = np.array(vecs).T                          # (ndof, nmode)
    G = V.T @ (M @ V)
    assert np.allclose(G, np.eye(len(modes)), atol=1e-6)


# ============================================================================
# MODAL — constraint-condensed spectrum
# ============================================================================

def test_rigid_link_frequency_shift():
    """A rigid link doubling the oscillating tip mass shifts the frequency by
    sqrt(m1/(m1+m2)): a spring-mass (k, m1) whose tip is /RBODY-tied to a
    second mass m2 rings at sqrt(k/(m1+m2))/2pi, EXACTLY (the T^T M T
    condensation carrying the slave's mass to the master)."""
    K, M, M2 = 4.0, 1.0e-3, 3.0e-3
    # /PROP/SPRING card is "Mass K C"
    starter = f"""\
#RADIOSS STARTER
/BEGIN
LINK
/NODE
         1                 0.0   0.0   0.0
         2                10.0   0.0   0.0
         3                10.0   5.0   0.0
         4                20.0   5.0   0.0
/SPRING/1
         1         1         2
/SPRING/2
         2         3         4
/PART/1
s1
         1         1
/PART/2
s2
         2         1
{STEEL}/PROP/SPRING/1
k
     {2*M}       {K}       0.0
/PROP/SPRING/2
k0
     {2*M2}     1e-08       0.0
/GRNOD/NODE/1
slave
3
/GRNOD/NODE/3
master
2
/GRNOD/NODE/2
ends
1 4
/RBODY/1
link
         2         1       0.0         0
/BCS/1
ends fixed
       111       111         0         2
/BCS/2
master axial only, rotations fixed
       011       111         0         3
/END
"""
    m = _starter(starter)
    f, _, _ = modal_frequencies(m, nev=1)
    f_link = np.sqrt(K / (M + M2)) / (2 * np.pi)
    assert abs(f[0] / f_link - 1.0) < 1e-4


def test_rigid_body_condensation_parallel_axis_inertia():
    """The CONSISTENT element mass, condensed to a rigid body, reproduces the
    continuum parallel-axis inertia: a uniform truss bar (mass m, length L)
    rigidly rotated about its end has rotary inertia m L^2/3 (about its centre
    m L^2/12) — sum_ij M_ij r_i r_j = integral rho x^2 dV, exact because a
    rigid rotation velocity is linear in position (in the linear element's
    shape space). This is the T^T M T block /IMPL/DYNA rides on, now fed by
    the consistent mass."""
    E, rho, A, L, N = 210.0, 7.8e-6, 1.0, 100.0, 4
    m_tot = rho * A * L
    dx = L / N
    nodes = "\n".join(f"{i+1:10d}{i*dx:20.10f}{0.0:20.10f}{0.0:20.10f}"
                      for i in range(N + 1))
    trs = "\n".join(f"{i+1:10d}{i+1:10d}{i+2:10d}" for i in range(N))
    # a FULLY FREE bar (no /BCS): every translational DOF is numbered, so the
    # z-translations exist for the rotary-inertia projection below
    m = _starter(f"""\
#RADIOSS STARTER
/BEGIN
FREEBAR
/NODE
{nodes}
/TRUSS/1
{trs}
/PART/1
b
         1         1
/MAT/LAW1/1
s
   {rho}
     {E}       0.0
/PROP/TRUSS/1
b
       {A}
/GRNOD/NODE/1
n
1
/END
""")
    dof = DofMap(m)
    M = assemble_mass(m, dof, m.x0).toarray()
    x = m.x0[:, 0]
    zeq = np.array([dof.eq[i * 6 + 2] for i in range(m.numnod)])  # z-trans eq

    def rotary(about):
        arm = x - about
        val = 0.0
        for i in range(m.numnod):
            for j in range(m.numnod):
                if zeq[i] >= 0 and zeq[j] >= 0:
                    val += M[zeq[i], zeq[j]] * arm[i] * arm[j]
        return val
    assert abs(rotary(0.0) - m_tot * L ** 2 / 3.0) < 1e-9         # about end
    assert abs(rotary(L / 2) - m_tot * L ** 2 / 12.0) < 1e-9      # about COG


# ============================================================================
# MODAL — prestressed (geometric stiffness) spectrum
# ============================================================================

def test_prestressed_taut_string():
    """A truss string has NO transverse material stiffness — its transverse
    modes exist ONLY through the geometric (initial-stress) stiffness. Inject
    a uniform tension T = sigma A and the prestressed eigensolver returns the
    exact taut-string frequencies f_n = n/(2L) sqrt(T/mu), mu = rho A. This
    exercises modal_frequencies(prestress=True): K = K_mat + K_geo."""
    E, rho, A, L, N = 210.0, 7.8e-6, 1.0, 100.0, 40
    dx = L / N
    nodes = "\n".join(f"{i+1:10d}{i*dx:20.10f}{0.0:20.10f}{0.0:20.10f}"
                      for i in range(N + 1))
    trs = "\n".join(f"{i+1:10d}{i+1:10d}{i+2:10d}" for i in range(N))
    interior = "\n".join(str(i + 1) for i in range(1, N))
    starter = f"""\
#RADIOSS STARTER
/BEGIN
STRING
/NODE
{nodes}
/TRUSS/1
{trs}
/PART/1
b
         1         1
/MAT/LAW1/1
s
   {rho}
     {E}       0.0
/PROP/TRUSS/1
b
       {A}
/GRNOD/NODE/1
n1
1
/GRNOD/NODE/3
interior
{interior}
/GRNOD/NODE/2
last
{N+1}
/BCS/1
node1 all
       111       000         0         1
/BCS/3
interior only y free
       101       000         0         3
/BCS/2
last all
       111       000         0         2
/END
"""
    m = _starter(starter)
    sigma = 1.05
    m.trusses.state["sig"][:] = sigma            # inject the pre-tension
    T, mu = sigma * A, rho * A
    f, _, _ = modal_frequencies(m, nev=3, prestress=True)
    fn = np.array([n / (2 * L) * np.sqrt(T / mu) for n in range(1, 4)])
    assert np.all(np.abs(f[:3] / fn - 1.0) < 0.01)


# ============================================================================
# /IMPL/EIGV engine card
# ============================================================================

def test_impl_eigv_card_end_to_end():
    """The /IMPL/EIGV card (a PORT card — freimpl.F has no modal branch) runs
    the modal eigensolver after the static solve and reports frequencies /
    modes / effective mass on model.implicit_result (mirroring /IMPL/BUCKL)."""
    import shutil
    import tempfile
    import os
    d = tempfile.mkdtemp()
    shutil.copy("examples/antenna_mast/MAST_0000.rad",
                os.path.join(d, "M_0000.rad"))
    with open(os.path.join(d, "M_0001.rad"), "w") as f:
        f.write("#\n/RUN/M/1\n1.0\n/IMPL\n/IMPL/EIGV\n4\n/END\n")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_starter(os.path.join(d, "M_0000.rad"))
        model = run_engine(os.path.join(d, "M_0001.rad"))
    r = model.implicit_result
    assert r.modal_frequencies is not None
    assert len(r.modal_frequencies) == 4
    assert len(r.modal_modes) == 4
    assert r.modal_effective_mass.shape == (4, 6)
    assert np.all(np.diff(r.modal_frequencies) >= -1e-9)     # ascending
    assert "NATURAL FREQUENCIES COMPUTATION" in buf.getvalue()


def test_impl_eigv_strs_card_prestressed():
    """/IMPL/EIGV/STRS mirrors the prestressed variant: it sets the K_geo
    path (impl_nlgeom) and extracts the stress-stiffened spectrum. On the
    unstressed antenna_mast (no load) the /STRS spectrum coincides with the
    plain one (K_geo = 0 at zero stress — the M8 path invariance)."""
    import shutil
    import tempfile
    import os
    d = tempfile.mkdtemp()
    shutil.copy("examples/antenna_mast/MAST_0000.rad",
                os.path.join(d, "M_0000.rad"))
    # a bare NONLIN run with no external load leaves zero stress
    with open(os.path.join(d, "M_0001.rad"), "w") as f:
        f.write("#\n/RUN/M/1\n1.0\n/IMPL/NONLIN\n1.0\n/IMPL/EIGV/STRS\n2\n"
                "/END\n")
    with contextlib.redirect_stdout(io.StringIO()):
        run_starter(os.path.join(d, "M_0000.rad"))
        model = run_engine(os.path.join(d, "M_0001.rad"))
    r = model.implicit_result
    assert r.modal_frequencies is not None and len(r.modal_frequencies) == 2
    # zero stress -> K_geo = 0 -> the plain cantilever bending fundamental
    E, I, rho, A, L = 210.0, 2.818e5, 7.8e-6, 707.5, 2000.0
    f1 = 1.875104 ** 2 / (2 * np.pi) * np.sqrt(E * I / (rho * A)) / L ** 2
    assert abs(r.modal_frequencies[0] / f1 - 1.0) < 0.02


# ----------------------------------------------------------------------------
# small element decks for the per-family mass tests
# ----------------------------------------------------------------------------

def _tri_strip_deck(nx=3, L=100.0, b=10.0, t=1.0):
    def nid(i, j):
        return 1 + i + j * (nx + 1)
    nodes = "\n".join(
        f"{nid(i,j):10d}{i*L/nx:20.10f}{j*b:20.10f}{0.0:20.10f}"
        for j in range(2) for i in range(nx + 1))
    els = []
    eid = 0
    for i in range(nx):
        eid += 1
        els.append(f"{eid:10d}{nid(i,0):10d}{nid(i+1,0):10d}{nid(i+1,1):10d}")
        eid += 1
        els.append(f"{eid:10d}{nid(i,0):10d}{nid(i+1,1):10d}{nid(i,1):10d}")
    return f"""\
#RADIOSS STARTER
/BEGIN
TRISTRIP
/NODE
{nodes}
/SH3N/1
{chr(10).join(els)}
/PART/1
p
         1         1
{STEEL}/PROP/SHELL/1
sh
       {t}         3      0.833
/GRNOD/NODE/1
n
1
/END
"""


def _brick_deck():
    """Two stacked unit bricks (a small solid column)."""
    nodes = []
    nid = 0
    coords = {}
    k = 1
    for z in (0.0, 1.0, 2.0):
        for y in (0.0, 1.0):
            for x in (0.0, 1.0):
                coords[(x, y, z)] = k
                nodes.append(f"{k:10d}{x*10:20.10f}{y*10:20.10f}{z*10:20.10f}")
                k += 1

    def c(x, y, z):
        return coords[(x, y, z)]
    els = []
    for e, z0 in enumerate((0.0, 1.0)):
        z1 = z0 + 1.0
        conn = [c(0, 0, z0), c(1, 0, z0), c(1, 1, z0), c(0, 1, z0),
                c(0, 0, z1), c(1, 0, z1), c(1, 1, z1), c(0, 1, z1)]
        els.append(f"{e+1:10d}" + "".join(f"{v:10d}" for v in conn))
    return f"""\
#RADIOSS STARTER
/BEGIN
BRICK
/NODE
{chr(10).join(nodes)}
/BRICK/1
{chr(10).join(els)}
/PART/1
p
         1         1
{STEEL}/PROP/SOLID/1
sol
         1
/GRNOD/NODE/1
n
1
/END
"""


def _tetra_deck():
    """A single unit tetra."""
    nodes = "\n".join([
        f"{1:10d}{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}",
        f"{2:10d}{10.0:20.10f}{0.0:20.10f}{0.0:20.10f}",
        f"{3:10d}{0.0:20.10f}{10.0:20.10f}{0.0:20.10f}",
        f"{4:10d}{0.0:20.10f}{0.0:20.10f}{10.0:20.10f}"])
    return f"""\
#RADIOSS STARTER
/BEGIN
TET
/NODE
{nodes}
/TETRA4/1
         1         1         2         3         4
/PART/1
p
         1         1
{STEEL}/PROP/SOLID/1
sol
         1
/GRNOD/NODE/1
n
1
/END
"""


def _beam_deck():
    """A single beam N1-N2 with orientation node N3."""
    nodes = "\n".join([
        f"{1:10d}{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}",
        f"{2:10d}{100.0:20.10f}{0.0:20.10f}{0.0:20.10f}",
        f"{99:10d}{0.0:20.10f}{100.0:20.10f}{0.0:20.10f}"])
    return f"""\
#RADIOSS STARTER
/BEGIN
BEAM
/NODE
{nodes}
/BEAM/1
         1         1         2        99
/PART/1
p
         1         1
{STEEL}/PROP/BEAM/1
sec
     707.5 2.818e+05 2.818e+05 5.636e+05
/GRNOD/NODE/1
n
1
/END
"""


def _spring_deck():
    """A single TYPE4 spring (Mass K C)."""
    nodes = "\n".join([
        f"{1:10d}{0.0:20.10f}{0.0:20.10f}{0.0:20.10f}",
        f"{2:10d}{10.0:20.10f}{0.0:20.10f}{0.0:20.10f}"])
    return f"""\
#RADIOSS STARTER
/BEGIN
SPRING
/NODE
{nodes}
/SPRING/1
         1         1         2
/PART/1
p
         1         1
{STEEL}/PROP/SPRING/1
sp
     0.002       4.0       0.0
/GRNOD/NODE/1
n
1
/END
"""
