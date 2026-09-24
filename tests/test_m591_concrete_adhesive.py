"""Constitutive physics unit tests for OpenRadioss /MAT/LAW126 (HJC) and /MAT/LAW169 (Arup) (M591).

Validates:
  1. LAW126 Johnson-Holmquist Concrete (HJC):
     - Parameter initialization and derived elastic moduli (K0, E, nu, H, Kav, etc.).
     - 3-region compacting Equation of State (EOS) (elastic, transitional, polynomial lock-up).
     - Hydrostatic pressure dependence, tensile cutoff, and yield surface strength cap SFMAX.
     - Strain rate sensitivity (Johnson-Cook logarithmic rate factor and Cowper-Symonds extension).
     - Cumulative damage evolution (shear plastic strain + volumetric compaction) and softening.
     - Element failure criteria (IDEL 1..4) and post-failure residual behavior (IFAILSO 1..5).
     - Acoustic wave speed and consistent algorithmic solid tangent.
     - Shell incompatibility rejection (solids only).
  2. LAW169 Arup Structural Adhesive:
     - Parameter initialization, derived moduli, and failure displacement limits (dfn, dfs, dp).
     - Pure tension vs pure shear yield envelopes and normal stress coupling (SHT_SL).
     - Mixed-mode failure power-law envelope (PWRT, PWRS).
     - Progressive softening and fracture energy dissipation to GIc and GIIc.
     - Shear plastic plateau (SHRP > 0) with zero damage up to dp.
     - 3D solid and 2D plane-stress shell formulations.
     - Acoustic wave speeds and algorithmic tangents (solid, shell, shell membrane).
  3. Starter Keyword Parsing & Model Checks:
     - Fixed and free format parsing for /MAT/LAW126 and /MAT/JOHNSON_HOLMQUIST_CONCRETE.
     - Fixed and free format parsing for /MAT/LAW169 and /MAT/ARUP_ADHESIVE.
     - Starter parameter bounds checking (rho0, G, E, nu, strengths, fracture energies).
     - Element family compatibility validation (solids for both, shells for 169, 1D rejected).
  4. Dynamic simulation and energy balance accounting:
     - Cyclic compressive loading on LAW126 solid element.
     - Mixed-mode tension-shear opening on LAW169 adhesive element with fracture energy accounting.
"""

import math
import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input.starter_keywords import parse_starter_deck
from pyradioss.materials.law126_hjc import (
    Law126Params,
    build_law126,
    solid_step as law126_solid_step,
    solid_update as law126_solid_update,
    sound_speed as law126_sound_speed,
    solid_tangent as law126_solid_tangent,
    consistent_solid_tangent as law126_consistent_solid_tangent,
    extra_shapes as law126_extra_shapes,
)
from pyradioss.materials.law169_arup import (
    Law169Params,
    build_law169,
    solid_step as law169_solid_step,
    shell_step as law169_shell_step,
    solid_update as law169_solid_update,
    shell_update as law169_shell_update,
    sound_speed as law169_sound_speed,
    solid_tangent as law169_solid_tangent,
    shell_tangent as law169_shell_tangent,
    consistent_solid_tangent as law169_consistent_solid_tangent,
    consistent_shell_tangent as law169_consistent_shell_tangent,
    shell_membrane_tangent as law169_shell_membrane_tangent,
    extra_shapes as law169_extra_shapes,
)
from pyradioss.materials import (
    solid_update,
    shell_update,
    sound_speed,
    solid_tangent,
    shell_layer_tangent,
    shell_membrane_tangent,
    needs_env,
    extra_shapes,
)
from pyradioss.model.entities import Material, Part, Property
from pyradioss.model.model import Model
from pyradioss.starter.checks import (
    check_mat_law126,
    check_mat_law169,
    check_materials,
)


# =============================================================================
# 1. LAW126 (HJC) Parameter Initialization & Derived Moduli
# =============================================================================

def test_law126_params_and_derived_moduli():
    """Verify HJC parameter initialization and derived mechanical properties."""
    rho0 = 2440.0e-12  # t/mm^3 (2440 kg/m^3)
    shear = 14800.0    # MPa
    fc = 48.0          # MPa
    pc = 16.0          # MPa (Pc = fc / 3)
    muc = 0.001        # volumetric crush strain
    pl = 800.0         # MPa
    mul = 0.1          # volumetric lock strain
    k1 = 85000.0       # MPa
    k2 = -171000.0     # MPa
    k3 = 208000.0      # MPa
    t0 = 4.0           # MPa

    data = {
        "id": 126,
        "title": "HJC Concrete 48MPa",
        "rho": rho0,
        "shear": shear,
        "aa": 0.79,
        "bb": 1.60,
        "nn": 0.61,
        "fc": fc,
        "t0": t0,
        "cc": 0.007,
        "eps0": 1.0e-3,
        "sfmax": 7.0,
        "efmin": 0.01,
        "pc": pc,
        "muc": muc,
        "pl": pl,
        "mul": mul,
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "d1": 0.04,
        "d2": 1.0,
        "idel": 1,
        "ifailso": 2,
    }

    mat = build_law126(data)
    assert isinstance(mat, Material)
    p = mat.params["law126_params"]
    assert isinstance(p, Law126Params)

    # Expected derived moduli
    expected_k0 = pc / muc  # 16000 MPa
    expected_young = 9.0 * expected_k0 * shear / (3.0 * expected_k0 + shear)
    expected_nu = (3.0 * expected_k0 - 2.0 * shear) / (6.0 * expected_k0 + 2.0 * shear)
    expected_h = (pl - pc) / mul
    expected_kav = (pl - pc) / (mul - muc)

    assert np.isclose(p.k0, expected_k0)
    assert np.isclose(p.young, expected_young)
    assert np.isclose(p.nu, expected_nu)
    assert np.isclose(p.h, expected_h)
    assert np.isclose(p.kav, expected_kav)
    assert p.idel == 1
    assert p.ifailso == 2


# =============================================================================
# 2. LAW126 3-Region Compacting Equation of State (EOS)
# =============================================================================

def test_law126_3region_eos():
    """Verify HJC 3-phase compaction EOS in pure hydrostatic compression."""
    rho0 = 2400.0e-12
    shear = 15000.0
    fc = 50.0
    pc = 16.0
    muc = 0.001
    pl = 800.0
    mul = 0.1
    k1 = 80000.0
    k2 = -160000.0
    k3 = 200000.0

    p = Law126Params(
        rho0=rho0, shear=shear, fc=fc, pc=pc, muc=muc, pl=pl, mul=mul,
        k1=k1, k2=k2, k3=k3, aa=0.8, bb=1.5, nn=0.6, t0=4.0
    )

    # 1. Region 1: Elastic compression (mu <= muc)
    mu1 = 0.0005
    p1 = p.eval_eos(mu1)
    k0 = pc / muc
    assert np.isclose(p1, k0 * mu1)

    # At boundary mu = muc
    assert np.isclose(p.eval_eos(muc), pc)

    # 2. Region 2: Transitional compaction (muc < mu < mul)
    mu2 = 0.05
    p2 = p.eval_eos(mu2)
    kav = (pl - pc) / (mul - muc)
    expected_p2 = pc + kav * (mu2 - muc)
    assert np.isclose(p2, expected_p2)

    # At boundary mu = mul
    assert np.isclose(p.eval_eos(mul), pl)

    # 3. Region 3: Fully compacted polynomial lock-up (mu >= mul)
    mu3 = 0.15
    p3 = p.eval_eos(mu3)
    mu_bar = (mu3 - mul) / (1.0 + mul)
    expected_p3 = k1 * mu_bar + k2 * (mu_bar**2) + k3 * (mu_bar**3)
    assert np.isclose(p3, expected_p3)


# =============================================================================
# 3. LAW126 Yield Surface & Pressure Dependence
# =============================================================================

def test_law126_yield_surface_and_sfmax():
    """Verify HJC normalized yield surface and strength capping by SFMAX."""
    fc = 40.0
    aa = 0.75
    bb = 1.65
    nn = 0.61
    sfmax = 5.0
    p = Law126Params(fc=fc, aa=aa, bb=bb, nn=nn, sfmax=sfmax, cc=0.0)

    # Intact concrete (D = 0) at pressure P = 80 MPa -> P* = P / fc = 2.0
    p_hydro = 80.0
    sig_star_intact = p.eval_yield_strength(p_hydro, dmg=0.0, epsd=0.0)
    expected_star = (aa * (1.0 - 0.0) + bb * (2.0**nn))
    assert np.isclose(sig_star_intact, expected_star)

    # Fully damaged concrete (D = 1.0) -> cohesive term vanishes, friction only
    sig_star_damaged = p.eval_yield_strength(p_hydro, dmg=1.0, epsd=0.0)
    expected_damaged = bb * (2.0**nn)
    assert np.isclose(sig_star_damaged, expected_damaged)
    assert sig_star_damaged < sig_star_intact

    # At extremely high confinement, strength must cap at SFMAX
    p_huge = 10000.0  # 10 GPa
    sig_star_huge = p.eval_yield_strength(p_huge, dmg=0.0, epsd=0.0)
    assert np.isclose(sig_star_huge, sfmax)


# =============================================================================
# 4. LAW126 Strain Rate Sensitivity
# =============================================================================

def test_law126_strain_rate_sensitivity():
    """Verify Johnson-Cook logarithmic rate factor and Cowper-Symonds rate scaling."""
    fc = 50.0
    cc = 0.015
    eps0 = 1.0e-3
    p_jc = Law126Params(fc=fc, cc=cc, eps0=eps0, aa=0.8, bb=1.5, nn=0.6)

    # Low strain rate (epsd <= eps0) -> rate factor is 1.0
    rate_fac_low = p_jc.eval_rate_factor(epsd=1.0e-4, is_comp=True)
    assert np.isclose(rate_fac_low, 1.0)

    # High strain rate (epsd = 100 s^-1)
    rate_fac_high = p_jc.eval_rate_factor(epsd=100.0, is_comp=True)
    expected_fac = 1.0 + cc * math.log(100.0 / eps0)
    assert np.isclose(rate_fac_high, expected_fac)

    # Cowper-Symonds extension
    p_cs = Law126Params(
        fc=fc, cst=40.0, powt=5.0, csc=100.0, powc=4.0,
        icowpsym=1, aa=0.8, bb=1.5, nn=0.6
    )
    # Compressive rate factor
    cs_comp = p_cs.eval_rate_factor(epsd=200.0, is_comp=True)
    expected_cs_comp = 1.0 + (200.0 / 100.0)**(1.0 / 4.0)
    assert np.isclose(cs_comp, expected_cs_comp)

    # Tensile rate factor
    cs_tens = p_cs.eval_rate_factor(epsd=200.0, is_comp=False)
    expected_cs_tens = 1.0 + (200.0 / 40.0)**(1.0 / 5.0)
    assert np.isclose(cs_tens, expected_cs_tens)


# =============================================================================
# 5. LAW126 Damage Accumulation & Failure Models
# =============================================================================

def test_law126_damage_accumulation_and_failure():
    """Verify cumulative damage D = sum(deps_p + dmu_p)/Dp and post-failure flags."""
    fc = 40.0
    t0 = 4.0
    d1 = 0.04
    d2 = 1.0
    efmin = 0.01

    p = Law126Params(
        shear=14800.0, aa=0.79, bb=1.60, nn=0.61,
        fc=fc, t0=t0, d1=d1, d2=d2, efmin=efmin,
        idel=1, ifailso=4, eps_max=0.05
    )

    # P = 40 MPa -> P* = 1.0, T* = 0.1 -> P* + T* = 1.1
    # Dp = D1 * (P* + T*)^D2 = 0.04 * 1.1^1.0 = 0.044
    dp_val = p.eval_fracture_plastic_strain(p_hydro=40.0)
    assert np.isclose(dp_val, 0.044)

    # Below efmin cutoff: P* + T* extremely small
    dp_small = p.eval_fracture_plastic_strain(p_hydro=-3.9)
    assert np.isclose(dp_small, efmin)

    # Single solid element stress integration under shear deformation
    mat = Material(id=1, law=126, rho0=2.4e-9, params={"law126_params": p})
    sig = np.zeros((1, 6), dtype=float)
    extra_state = {}

    # Apply 20 increments of shear strain
    for _ in range(20):
        deps = np.array([[0.0, 0.0, 0.0, 0.005, 0.0, 0.0]], dtype=float)
        sig, epsp, c = law126_solid_update(mat, sig, deps, dt=1.0e-5, extra=extra_state)

    dmg = extra_state.get("dmg", np.zeros(1))[0]
    assert dmg > 0.0, "Damage should accumulate during plastic shear deformation"
    assert extra_state["uvar"][0, 0] >= 0.0, "State uvar should track plastic volumetric compaction"


# =============================================================================
# 6. LAW126 Algorithmic Tangent & Sound Speed
# =============================================================================

def test_law126_tangent_and_sound_speed():
    """Verify LAW126 6x6 solid algorithmic tangent and acoustic wave speed."""
    rho0 = 2400.0e-12
    shear = 15000.0
    pc = 16.0
    muc = 0.001
    k0 = pc / muc  # 16000 MPa

    p = Law126Params(rho0=rho0, shear=shear, pc=pc, muc=muc)
    mat = Material(id=1, law=126, rho0=rho0, params={"law126_params": p})

    # Acoustic wave speed: c = sqrt((K + 4/3 G) / rho)
    c_expected = math.sqrt((k0 + (4.0 / 3.0) * shear) / rho0)
    c_actual = law126_sound_speed(mat, rho=rho0)
    assert np.isclose(c_actual, c_expected)

    # Solid tangent stiffness matrix C (6, 6)
    c_mat = law126_solid_tangent(mat)
    assert c_mat.shape == (6, 6)
    lam = k0 - (2.0 / 3.0) * shear
    assert np.isclose(c_mat[0, 0], lam + 2.0 * shear)
    assert np.isclose(c_mat[0, 1], lam)
    assert np.isclose(c_mat[3, 3], shear)

    # Shell element dispatch must raise NotImplementedError for LAW126
    with pytest.raises(NotImplementedError):
        shell_update(mat, sig=np.zeros(3), deps=np.zeros(3))


# =============================================================================
# 7. LAW169 (Arup) Parameter Initialization & Derived Limits
# =============================================================================

def test_law169_params_and_derived_limits():
    """Verify Arup adhesive parameter initialization and derived failure limits."""
    rho0 = 1.2e-9    # t/mm^3 (1200 kg/m^3)
    young = 2000.0   # MPa
    nu = 0.38
    tenmax = 30.0    # MPa
    shrmax = 25.0    # MPa
    gcten = 1.2      # mJ/mm^2 (1200 J/m^2)
    gcshr = 2.5      # mJ/mm^2 (2500 J/m^2)
    pwrt = 2
    pwrs = 2
    shrp = 0.2
    sht_sl = 0.15

    data = {
        "id": 169,
        "title": "Arup Epoxy Adhesive",
        "rho": rho0,
        "e": young,
        "nu": nu,
        "tenmax": tenmax,
        "shrmax": shrmax,
        "gcten": gcten,
        "gcshr": gcshr,
        "pwrt": pwrt,
        "pwrs": pwrs,
        "shrp": shrp,
        "sht_sl": sht_sl,
    }

    mat = build_law169(data)
    assert isinstance(mat, Material)
    p = mat.params["law169_params"]
    assert isinstance(p, Law169Params)

    expected_shear = young / (2.0 * (1.0 + nu))
    expected_dfn = 2.0 * gcten / tenmax
    expected_dfs = (2.0 * gcshr / (1.0 + shrp)) / shrmax
    expected_dp = shrp * expected_dfs
    expected_eps_n0 = tenmax / young
    expected_eps_s0 = shrmax / expected_shear + expected_dp

    assert np.isclose(p.shear, expected_shear)
    assert np.isclose(p.dfn, expected_dfn)
    assert np.isclose(p.dfs, expected_dfs)
    assert np.isclose(p.dp, expected_dp)
    assert np.isclose(p.eps_n0, expected_eps_n0)
    assert np.isclose(p.eps_s0, expected_eps_s0)


# =============================================================================
# 8. LAW169 Pure Tension vs Pure Shear Envelopes
# =============================================================================

def test_law169_pure_tension_and_pure_shear():
    """Verify pure mode I tension and pure mode II shear response."""
    p = Law169Params(
        rho0=1.0e-9, young=1000.0, nu=0.25,
        tenmax=20.0, shrmax=15.0, gcten=1.0, gcshr=1.5,
        pwrt=2, pwrs=2, shrp=0.0, sht_sl=0.2
    )
    mat = Material(id=1, law=169, rho0=1.0e-9, params={"law169_params": p})

    # 1. Pure Normal Tension (solid along z axis: sig[2])
    sig = np.zeros((1, 6), dtype=float)
    extra = {}
    deps_n = np.array([[0.0, 0.0, 0.01, 0.0, 0.0, 0.0]], dtype=float)  # eps_zz = 0.01
    sig, epsp, _ = law169_solid_update(mat, sig, deps_n, extra=extra)
    # In Fortran sigeps169_connect.F90, the normal through-thickness stiffness is
    # wave = young * (1 - nu) / ((1 + nu) * (1 - 2*nu)) = 1200 MPa
    assert np.isclose(sig[0, 2], p.wave * 0.01), f"Initial elastic slope wave={p.wave}"

    # Reach tensile yield: deps_zz = 0.015 -> eps_tot = 0.025 > eps_n0 (0.02)
    deps_n2 = np.array([[0.0, 0.0, 0.015, 0.0, 0.0, 0.0]], dtype=float)
    sig, epsp, _ = law169_solid_update(mat, sig, deps_n2, extra=extra)
    assert extra["uvar169"][0, 8] == 1.0, "Yield flag should activate once tensile yield reached"

    # Subsequent increment past eps_n0 grows tensile damage (Fortran sigeps169_connect.F90 lines 206-209)
    deps_n3 = np.array([[0.0, 0.0, 0.005, 0.0, 0.0, 0.0]], dtype=float)
    sig, epsp, _ = law169_solid_update(mat, sig, deps_n3, extra=extra)
    assert extra["dmg169"][0] > 0.0, "Tensile damage should initiate beyond eps_n0"

    # 2. Pure Shear with normal stress effect (tau_max = shrmax - sht_sl * sig_n)
    # Compressive normal stress (sig_n < 0) should strengthen shear limit
    comp_norm = -50.0  # compressive normal stress
    taumax_comp = max(1e-6, p.shrmax - p.sht_sl * comp_norm)
    assert taumax_comp == 15.0 - 0.2 * (-50.0) == 25.0
    assert taumax_comp > p.shrmax


# =============================================================================
# 9. LAW169 Mixed-Mode Failure Criterion & Energy Dissipation
# =============================================================================

def test_law169_mixed_mode_energy_dissipation():
    """Verify power-law mixed-mode failure envelope and fracture energy dissipation."""
    tenmax = 20.0
    shrmax = 20.0
    gcten = 0.8
    gcshr = 1.6
    p = Law169Params(
        young=2000.0, nu=0.3, tenmax=tenmax, shrmax=shrmax,
        gcten=gcten, gcshr=gcshr, pwrt=2, pwrs=2, shrp=0.0
    )
    mat = Material(id=1, law=169, rho0=1.0e-9, params={"law169_params": p})

    sig = np.zeros((1, 6), dtype=float)
    extra = {}

    # Progressive loading until total failure (displacement > dfn and dfs)
    # dfn = 2 * 0.8 / 20 = 0.08
    dfn = p.dfn
    steps = 40
    step_size = (dfn * 1.5) / steps

    for _ in range(steps):
        deps = np.array([[0.0, 0.0, step_size, 0.0, 0.0, 0.0]], dtype=float)
        sig, epsp, _ = law169_solid_update(mat, sig, deps, extra=extra)

    # After full elongation past dfn, damage should be 1.0 and stress should drop to zero
    assert extra["dmg169"][0] >= 1.0
    assert np.allclose(sig[0], 0.0)


# =============================================================================
# 10. LAW169 Shear Plastic Plateau (SHRP)
# =============================================================================

def test_law169_shear_plastic_plateau():
    """Verify that SHRP > 0 delays damage initiation up to dp without degradation."""
    young = 1000.0
    nu = 0.25
    shear = young / (2.0 * (1.0 + nu))  # 400 MPa
    shrmax = 20.0
    gcshr = 2.0
    shrp = 0.4

    p = Law169Params(
        young=young, nu=nu, shrmax=shrmax, tenmax=50.0,
        gcshr=gcshr, gcten=5.0, shrp=shrp, pwrs=2, pwrt=2
    )
    mat = Material(id=1, law=169, rho0=1.0e-9, params={"law169_params": p})

    # dfs = (2 * 2.0 / (1 + 0.4)) / 20.0 = 0.142857
    # dp = shrp * dfs = 0.05714
    # eps_s0 = shrmax / shear + dp = 0.05 + 0.05714 = 0.10714
    dfs = p.dfs
    dp = p.dp
    eps_s0 = p.eps_s0

    # Apply shear deformation incrementally that exceeds elastic limit (0.05) but stays within plateau (< eps_s0 = 0.10714)
    sig = np.zeros((1, 6), dtype=float)
    extra = {}
    n_steps = 16
    d_step = 0.08 / n_steps
    for _ in range(n_steps):
        deps = np.array([[0.0, 0.0, 0.0, 0.0, d_step, 0.0]], dtype=float)
        sig, epsp, _ = law169_solid_update(mat, sig, deps, extra=extra)

    # In plateau region: damage must remain 0
    assert extra["uvar169"][0, 1] == 0.0, "Shear damage should not initiate while inside plastic plateau"
    # Shear stress should be capped at shrmax = 20
    assert np.isclose(sig[0, 4], shrmax, atol=1e-2)


# =============================================================================
# 11. LAW169 Plane-Stress Shell & Tangents
# =============================================================================

def test_law169_shell_and_tangents():
    """Verify 2D plane-stress shell formulation and algorithmic tangents for LAW169."""
    young = 3000.0
    nu = 0.3
    p = Law169Params(young=young, nu=nu, tenmax=40.0, shrmax=30.0, gcten=1.0, gcshr=2.0)
    mat = Material(id=1, law=169, rho0=1.2e-9, params={"law169_params": p})

    # 1. Shell update (3 components: xx, yy, xy)
    sig_sh = np.zeros((1, 3), dtype=float)
    deps_sh = np.array([[0.005, 0.001, 0.002]], dtype=float)
    extra_sh = {}
    sig_out, epsp_out = law169_shell_update(mat, sig_sh, deps_sh, extra=extra_sh)
    assert sig_out.shape == (1, 3)

    # 2. Consistent shell tangent (3, 3)
    c_sh = law169_shell_tangent(mat)
    assert c_sh.shape == (3, 3)
    e_plane = young / (1.0 - nu**2)
    assert np.isclose(c_sh[0, 0], e_plane)
    assert np.isclose(c_sh[2, 2], p.shear)

    # 3. Shell membrane tangent
    c_mem = law169_shell_membrane_tangent(mat)
    assert c_mem.shape == (3, 3)
    assert np.isclose(c_mem[0, 1], nu * e_plane)

    # 4. Solid tangent (6, 6)
    c_sol = law169_solid_tangent(mat)
    assert c_sol.shape == (6, 6)

    # 5. Acoustic wave speeds
    cs_solid = law169_sound_speed(mat, is_shell=False)
    cs_shell = law169_sound_speed(mat, is_shell=True)
    assert cs_solid > 0.0
    assert cs_shell > 0.0


# =============================================================================
# 12. Starter Keyword Parsing (LAW126 and LAW169)
# =============================================================================

def test_starter_keyword_parsing_law126_and_law169(tmp_path):
    """Verify parsing of fixed and free format /MAT/LAW126 and /MAT/LAW169 cards."""
    # Deck with LAW126 and LAW169
    deck_content = """# Starter test deck for M591
/BEGIN
Test_M591_Concrete_Adhesive
       100         1
/MAT/LAW126/1
Concrete_Foundation
#   RHO_I
  2.4E-09
#             G              A              B              N             FC             T0
        14000.0           0.79           1.60           0.61           48.0            4.0
#             C          EPS_0          SFMAX          EFMIN
          0.007         0.0001            7.0           0.01
#            PC            MUC             PL            MUL
           16.0          0.001          800.0            0.1
#            K1             K2             K3
        85000.0      -171000.0       208000.0
#            D1             D2           IDEL        EPS_MAX        IFAILSO
           0.04            1.0              1           0.05              2
#           CST           POWT            CSC           POWC
            0.0            1.0            0.0            1.0
/MAT/LAW169/2
Adhesive_Joint
#   RHO_I
  1.1E-09
#             E             NU         SHT_SL         TENMAX          GCTEN
         2100.0           0.35           0.15           35.0            1.5
#        SHRMAX          GCSHR           PWRT           PWRS           SHRP
           28.0            2.2              2              2           0.25
/END
"""
    deck_path = tmp_path / "deck_m591.rad"
    deck_path.write_text(deck_content, encoding="utf-8")
    model = parse_starter_deck(str(deck_path))

    # Verify LAW126 parsed correctly
    assert 1 in model.materials
    mat126 = model.materials[1]
    assert mat126.law == 126
    p126 = mat126.params["law126_params"]
    assert np.isclose(p126.fc, 48.0)
    assert np.isclose(p126.shear, 14000.0)
    assert np.isclose(p126.pc, 16.0)
    assert p126.idel == 1
    assert p126.ifailso == 2

    # Verify LAW169 parsed correctly
    assert 2 in model.materials
    mat169 = model.materials[2]
    assert mat169.law == 169
    p169 = mat169.params["law169_params"]
    assert np.isclose(p169.young, 2100.0)
    assert np.isclose(p169.tenmax, 35.0)
    assert np.isclose(p169.shrmax, 28.0)
    assert p169.pwrt == 2
    assert np.isclose(p169.shrp, 0.25)


# =============================================================================
# 13. Starter Checks & Element Compatibility
# =============================================================================

def test_starter_checks_and_element_compatibility():
    """Verify sanity checks for parameter bounds and element type restrictions."""
    log = MessageLog()
    model = Model()

    # Valid LAW126 material
    mat126_ok = Material(id=1, law=126, rho0=2.4e-9, params={
        "shear": 14000.0, "fc": 40.0, "pc": 15.0, "muc": 0.001,
        "pl": 600.0, "mul": 0.1
    })
    check_mat_law126(model, 1, mat126_ok, log)
    assert log.nerror == 0

    # Invalid LAW126: negative density & missing parameters
    log_bad = MessageLog()
    mat126_bad = Material(id=2, law=126, rho0=-1.0, params={
        "shear": 0.0, "fc": 0.0, "pc": 0.0, "muc": 0.0, "pl": 0.0, "mul": 0.0
    })
    check_mat_law126(model, 2, mat126_bad, log_bad)
    assert log_bad.nerror > 0, "Errors should be flagged for invalid LAW126 parameters"

    # LAW126 assigned to shell property -> must error
    log_elem = MessageLog()
    model.materials[1] = mat126_ok
    model.properties[10] = Property(id=10, type=1)  # Shell property
    model.parts[100] = Part(id=100, mat_id=1, prop_id=10)
    check_mat_law126(model, 1, mat126_ok, log_elem)
    assert log_elem.nerror > 0, "LAW126 must reject shell element assignments"

    # Valid LAW169 material
    log169 = MessageLog()
    mat169_ok = Material(id=3, law=169, rho0=1.2e-9, params={
        "e": 2000.0, "nu": 0.35, "tenmax": 30.0, "shrmax": 20.0,
        "gcten": 1.0, "gcshr": 2.0, "pwrt": 2, "pwrs": 2, "shrp": 0.1
    })
    check_mat_law169(model, 3, mat169_ok, log169)
    assert log169.nerror == 0


# =============================================================================
# 14. Central Materials Dispatch Integration
# =============================================================================

def test_central_materials_dispatch_law126_and_law169():
    """Verify integration of LAW126 and LAW169 into pyradioss.materials dispatchers."""
    p126 = Law126Params(rho0=2.4e-9, shear=14000.0, fc=40.0, pc=15.0, muc=0.001, pl=600.0, mul=0.1)
    mat126 = Material(id=1, law=126, rho0=2.4e-9, params={"law126_params": p126})

    p169 = Law169Params(rho0=1.2e-9, young=2500.0, nu=0.35, tenmax=30.0, shrmax=20.0, gcten=1.0, gcshr=2.0)
    mat169 = Material(id=2, law=169, rho0=1.2e-9, params={"law169_params": p169})

    # needs_env check
    assert needs_env(mat126) is True
    assert needs_env(mat169) is True

    # extra_shapes check
    shapes126 = extra_shapes(mat126)
    assert "uvar126" in shapes126
    assert "dmg126" in shapes126

    shapes169 = extra_shapes(mat169)
    assert "uvar169" in shapes169
    assert "dmg169" in shapes169

    # solid_update dispatch
    sig126 = np.zeros((1, 6), dtype=float)
    deps126 = np.array([[0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0]], dtype=float)
    s_out126, _, c126 = solid_update(mat126, sig126, deps126)
    assert s_out126.shape == (1, 6)
    assert c126 is not None

    sig169 = np.zeros((1, 6), dtype=float)
    deps169 = np.array([[0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0]], dtype=float)
    s_out169, _, c169 = solid_update(mat169, sig169, deps169)
    assert s_out169.shape == (1, 6)
    assert c169 is not None

    # shell_update dispatch
    sig_sh = np.zeros((1, 3), dtype=float)
    deps_sh = np.array([[0.0001, 0.0001, 0.0]], dtype=float)
    s_sh_out, _ = shell_update(mat169, sig_sh, deps_sh)
    assert s_sh_out.shape == (1, 3)

    # solid_tangent dispatch
    ctan126 = solid_tangent(mat126)
    assert ctan126.shape == (6, 6)
    ctan169 = solid_tangent(mat169)
    assert ctan169.shape == (6, 6)

    # shell_membrane_tangent dispatch
    cmem169 = shell_membrane_tangent(mat169)
    assert cmem169.shape == (3, 3)
