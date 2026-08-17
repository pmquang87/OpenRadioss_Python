"""M37 tests: the cfg-driven generic /MAT reader (mat_reader.py).

Every snippet in MAT_SNIPPETS is a REAL card block extracted from the
official OpenRadioss demo/validation corpus (rd_e / rd_v / tutorial
decks — the source deck is named above each snippet), so these tests
fail exactly when the parser regresses on real decks.

Covered here:

* all 28 /MAT families of the coverage ranked_gaps parse error-free into
  InactiveMaterial records with the right density (+ /ALE/MAT,
  /EULER/MAT, /HEAT/MAT as parse-only notes = the 31 gap families);
* cfg-exact field values (fixed-column slicing incl. LAW37's jammed
  fields, blank cards, CARD_LIST arrays, GAS subtypes, LAW51
  subobjects, /MAT/<law>/<id>/<unit_id> headers);
* the InactiveMaterial flow: Starter accepts (mass init from the parsed
  density, listing note), Engine REFUSES with an error naming the law;
* the MAT_PHYSICS_REGISTRY hook the next-phase physics builders use.
"""

import os

import numpy as np
import pytest

from pyradioss.common.messages import MessageLog
from pyradioss.input import mat_reader
from pyradioss.input.deck_reader import read_deck
from pyradioss.input.mat_reader import (
    GenericMaterialRecord, InactiveMaterial, InactiveMaterialError,
    MAT_PHYSICS_REGISTRY, refuse_inactive_materials)
from pyradioss.input.starter_keywords import parse_starter_deck

# The generic reader parses its schemas from OpenRadioss's hm_cfg_files —
# not vendored (license); local installs find C:/OpenRadioss, CI fetches a
# sparse checkout and sets PYRADIOSS_HM_CFG (ci.yml). Anywhere else: skip,
# loudly, instead of failing on all-zero parameters.
pytestmark = pytest.mark.skipif(
    mat_reader.catalogue().schema("FABRI") is None,
    reason="hm_cfg_files CFG tree not found — set PYRADIOSS_HM_CFG "
           "(see ci.yml / PORTING_GUIDE M37)")
from pyradioss.model.entities import Material
from pyradioss.model.model import Model

HAVE_CFG = mat_reader.catalogue().root is not None

needs_cfg = pytest.mark.skipif(
    not HAVE_CFG, reason="hm_cfg_files tree not installed "
                         "(set PYRADIOSS_HM_CFG)")


# ============================================================================
# Real corpus snippets — one per unsupported /MAT family (ranked_gaps)
# ============================================================================

MAT_SNIPPETS = {
    # rd_e/RD-E-1900_Wave/19_Wave_propagation/ALE_formulation/WAVE_0000.rad
    "BOUND": """\
/MAT/BOUND/2
bound silent material
#              RHO_I               RHO_0
                2842                2842
#     Ityp                          P_sh            Fscale_T
         3                             0                   0
# node_IDV                             c                 I_c
         0                          5068                9.31
#fun_IDRHO
         0
#funct_IDp                     Fscale_p0
         0                             0
#funct_IDE                      Fscale_E
         0                             0
# Blank card

# Blank card

#funct_IDT funct_IDQ
         0         0
""",
    # rd_e/RD-E-4701_Kupfer/Kupfer_tests/LAW24/C000/C000_0000.rad
    "CONC": """\
/MAT/CONC/1
Concrete
#              RHO_I
               .0022                   0
#                E_c                  NU      Icap
               31700                 .22         0
#                 fc            ft_on_fc            fb_on_fc            f2_on_fc            s0_on_fc
               32.22                 0.1                1.15
#                H_t               D_sup             EPS_max
                   0                   0                   0
#                k_y                 r_t                 r_c                H_bp
                 .35                   0                   0                   0
#            ALPHA_y             ALPHA_F               V_max
                -0.6                 0.2                   0
#                f_k                 f_0                H_v0
                   0                   0                   0
#                  E             sigma_y                 E_t
                   0                   0                   0
#             ALPHA1              ALPHA2              ALPHA3
                   0                   0                   0
""",
    # tutorial/RD-HWX-T-1060/solved/data/3PB_completed_0000.rad
    "CONNECT": """\
/MAT/CONNECT/3
HC hexa spotweld_M59_CONNECT_Inc0
#              RHO_I
7.80000000000000E-09
#                  E                   G     Imass     Icomp               Ecomp
            210000.0            210000.0
# NB_funct   Fsmooth                Fcut
         1         1                 0.0
# YFun_IDN  YFun_IDT        SR_reference        Fscale_yield
        15        16                 0.0                 0.0
""",
    # rd_e/RD-E-0400_Airbag/04_Airbag/driver/driver_airbag_0000.rad
    "FABRI": """\
/MAT/FABRI/2
FABRIC
#              RHO_I               RHO_O
             8.5E-10                   0
#                E11                 E22                NU12
                 500                 500                   0
#                G12                 G23                 G31
                  10                  10                  10
#                R_E                              ZEROSTRESS          FSCALE_POR   SENS_ID
                .001                                       1                   0         1
""",
    # rd_e/RD-E-0400_Airbag/04_Airbag/driver/driver_airbag_0000.rad
    "GAS": """\
/MAT/GAS/MASS/1001
AIR
#                 MW
             2.89E-5
#                Cpa                 Cpb                 Cpc                 Cpd                 Cpe
         928474283.7                   0                   0                   0                   0
#                Cpf
                   0
""",
    # rd_e/RD-E-2500_Spring_back/25_Spring-back/Explicit_spring-back/DBEND_44_0000.rad
    "HILL_TAB": """\
/MAT/HILL_TAB/1
Matflan
#              RHO_I
                  80                  80
#                  E                  NU
              206000                  .3
#FUNCT_IDE                          EINF                  CE
         0                             0                   0
#                r00                 r45                 r90              C_hard
                1.73                1.34                2.24                   0
#           EPSP_max              EPS_t1               EPS_m
                   0                   0                   0
# func_IDi                      Fscale_i           EPS_dot_i
         5                             0                   0
""",
    # rd_e/RD-E-4601_Lagrange/Lagrange/MAT_PROP.inc
    "HYD_JCOOK": """\
/MAT/HYD_JCOOK/1
Copper
#              RHO_O
                8.96
#                 E                   nu
                1.24                 .35
#                  A                   B                   n              epsmax              sigmax
                9E-4              .00292                 .31                   0               .0066
#               Pmin
               -1E30
#                  C           EPS_DOT_0                   M               Tmelt               Tmax
                .025                1E-6                1.09                1656                1E30
#              RHOCP                                                          Tr
            3.461E-5                                                           0
""",
    # rd_e/RD-E-1300_Shock_tube/13_Shock_tube/Blast_experiment/blast_experiment_0000.rad
    "HYD_VISC": """\
/MAT/HYD_VISC/1
AIR_HIGH_PRESSURE
#              RHO_0
            7.963e-6
#                Knu                Pmin
                   0                   0
""",
    # rd_e/RD-E-4601_Lagrange/Lagrange/LAW05.txt
    "JWL": """\
/MAT/JWL/2/123
TNT - data from example 46 with unit: (g-cm-mus) - Standard JWL , No Afterburning
#              RHO_I
                1.63
#                  A                   B                  R1                  R2               OMEGA
              3.7121               .0323                4.15                 .95                  .3
#                  D                P_CJ                  E0                Eadd   I_BFRAC     Q_OPT
                .693                 .21                 .07                   0         0         0
#                 P0                 Psh                B_un
                   0                   0                   0
""",
    # rd_e/RD-E-5200_Creep/52_cylinder_creep/cylinder_creep_beta_001/foam_relax_0000.rad
    "KELVINMAX": """\
/MAT/KELVINMAX/3
LAW40 elastic rubber
#              RHO_I
                2E-9                   0
#                  K               G_inf              Astass              Bstass                 Kvm
               66.67                  10                   0                   0                   0
#                 G1                  G2                  G3                  G4                  G5
                  90                   0                   0                   0                   0
#              BETA1               BETA2               BETA3               BETA4               BETA5
                 .01                   0                   0                   0                   0
""",
    # rd_e/RD-E-1300_Shock_tube/13_Shock_tube/Blast_experiment/blast_experiment_0000.rad
    "LAW151": """\
/MAT/LAW151/2
Multimaterial AIR high
#        BLANK CARD

#mat_ID_01            Vfrac_01
         1                 1.0
""",
    # rd_e/RD-E-0601_Fuel_tank/1-Tank_sloshing/data/TANK_0000.rad
    # (note the JAMMED 20-char fields on the liquid card: '0.0' touches
    # '1.00000000000000E-03' — only fixed-column slicing reads it right)
    "LAW37": """\
/MAT/LAW37/2
AIR
#                Psh

#             RHO_l0                 C_l             ALPHA_l                NU_l            NU_VOL_l
1.00000000000000E-03              2089.0                 0.01.00000000000000E-03                 0.0
#             RHO_G0               GAMMA                  P0                NU_g            NU_VOL_g
1.22000000000000E-06                 1.4                 0.1             0.00143                 0.0
""",
    # rd_e/RD-E-3900_Biomedical_valve/39_Bio_Valve/BIO_VALVE/VALVE_0000.rad
    "LAW46": """\
/MAT/LAW46/2
BLOOLIKE
#              RHO_I              RHO_0
                 960                   0
#                 c                  Nu
                  50             5.45E-5
#     Isgs                 C_s                Csp
         0                  .1                  .1
""",
    # rd_e/RD-E-2201_ALE/Ditching_Mono_Domain_ALE/data/mat_law51_n.inc
    "LAW51": """\
/MAT/LAW51/1
non reflecting BC

#    Iform
         6
#               Pext                           Tcp            Tc_alpha
                 0.0                           0.0                 0.0
#            ALPHA_1              RHO_01                E_01              P_min1                P_01
                 0.0                 0.0                 0.0                 0.0                 0.0
#             SSP_01
                 0.0
#    BLANK CARD

#            ALPHA_1              RHO_01                E_01              P_min1                P_01
                 0.0                 0.0                 0.0                 0.0                 0.0
#             SSP_01
                 0.0
#    BLANK CARD

#            ALPHA_1              RHO_01                E_01              P_min1                P_01
                 0.0                 0.0                 0.0                 0.0                 0.0
#             SSP_01
                 0.0
#    BLANK CARD

""",
    # rd_e/RD-E-4802_Solid_spring/solid_spot_law59/FRAME_MODIFIED_0000.rad
    "LAW59": """\
/MAT/LAW59/100026
new Material
#              RHO_I
              7.9E-9
#                  E                   G
               21000               21000
# NB_funct   Fsmooth                Fcut
         1         1                   0
# YFun_IDN  YFun_IDT        SR_reference        Fscale_yield
         1         2                   0                   0
""",
    # rd_e/RD-E-1601_Explicit/EXPLICIT_solver/ADYREL/data/SEAT_ADYREL_0000.rad
    "LAW62": """\
/MAT/LAW62/11
Foam
#              RHO_I
               5E-11                   0
#                 Nu         N         M              mu_max Flag_Visc      Form
                   0         2         0                   0         0         0
#               mu_i
                .015                .005
#            alpha_i
                   2                  -2
#               Nu_i
                   0                   0
""",
    # rd_e/RD-E-4400_Blow_molding_AMS/44_blow_moding_ams/E4_66_AMS/EXAMPLE4_66_0000.rad
    "LAW66": """\
/MAT/LAW66/3
Bottle
#              RHO_I
                1E-9
#                  E                  Nu              C_hard               F_cut  F_smooth Iyld_rate
                   4                .475                   0                   1         0         4
#                P_c                 P_t
                   0                   0
#   NFUNCC    NFUNCT
         5         5
#funct_IDc                    Episilon_c             Fscalec
         1                            .1                   0
         2                             1                   0
         3                            10                   0
         4                           100                   0
         5                          1000                   0
#funct_IDt                    Episilon_t             Fscalet
         1                            .1                   0
         2                             1                   0
         3                            10                   0
         4                           100                   0
         5                          1000                   0
""",
    # rd_e/RD-E-5600_Hyperelastic_material/.../LAW69_extend_to_compression/LAW69.txt
    "LAW69": """\
/MAT/LAW69/1/1
rubber LAW69
#              RHO_I
                1E-9
#   LAW_ID    FCT_ID                  NU              FSCALE    N_PAIR    ICHECK
         1         0                .495                   0         2         3
#  FCT_ID1
         3
""",
    # rd_v_material/RD-V-0220_Foam_LAW70/0220_foam_LAW70/0220_foam_LAW70_0/BLOCK_H8_0000.rad
    "LAW70": """\
/MAT/LAW70/1
Foam material
#              RHO_I
               1E-10
#                 EO                  NU               E_max             EPS_max     Itens
                  25                   0                2500                   0         0
#              F_cut   Ismooth     Nload   Nunload     Iflag               Shape                 Hys
                1000         1         1         1         0                   0                   0
#  fct_IDL          Eps_._load          Fscaleload
         1                   0                   0
#  fct_IDL          Eps_._load          Fscaleload
         1                   0                  .2
""",
    # rd_e/RD-E-4701_Kupfer/Kupfer_tests/LAW81/C000/LAW81_fit.txt
    "LAW81": """\
/MAT/LAW81/1
 Concret
#              Rho_i
   0.002200000000000
#                 K0                  G0
    18869.0476190476    12991.8032786885
#               BETA                 PSI
       68.7903728190
#              ALPHA             EPS_MAX               EPS_0
        0.0736137946
#  Fct_IDK   Fct_IDG   Fct_IDc  Fct_IDPb     Isoft
         0         0        23        24         0
""",
    # rd_e/RD-E-5600_Hyperelastic_material/.../Ogden_model/LAW82/LAW82_N2.txt
    "LAW82": """\
/MAT/LAW82/1
rubber LAW82
#              RHO_I
                1E-9                   0
#        N                            Nu
         2                         .4997
#               Mu_i
0.000045637449070023   0.547913433558156                   0                   0                   0
#            Alpha_i
      7.168617832124     -4.158214786551                   0                   0                   0
#                D_i
                   0                   0
""",
    # rd_e/RD-E-4801_Solid_spotweld/One_solid_element_validation_law83/run01_no_failure/MAT83_solid_spotweld.inc
    "LAW83": """\
/MAT/LAW83/5/2
solid spotweld
#              RHO_I
              7.8E-6
#                  E                         Imass
                1.18                             0
#  Fct_ID1                      Y_scale1            X_scale1               ALPHA                BETA
       200                             1                   1               0.627               1.434
#                 RN                  RS   Fsmooth                Fcut
                 .17                .248         0                   0
#  Fct_IDN   Fct_IDS              XSCALE
         0         0                   0
""",
    # rd_e/RD-E-5600_Hyperelastic_material/.../Ogden_model/LAW88/LAW88.txt
    "LAW88": """\
/MAT/LAW88/1
rubber
#              RHO_I
                1E-9
#                 NU                   K               F_cut  F_smooth       N_L
               .4997      913.0824653215                   0                   1
#fctID_Unl                 Fscale_unload                 HYs               Shape   Tension
         0                            1.                  0.                  0.         0
#fctID_l                     Fscale_load          Eps_._load
         3                            1.                  0.
""",
    # rd_e/RD-E-5600_Hyperelastic_material/.../Arruda_Boyce_model/LAW92_UT/LAW92_UT.txt
    "LAW92": """\
/MAT/LAW92/1/1
rubber
#              RHO_I
            1.000E-9
#                 mu                   D                 LAM
                   0                   0                   0
#    IType    fct_ID                  NU
         1         3               .4997
""",
    # rd_e/RD-E-5600_Hyperelastic_material/.../Yeoh_model/LAW94.txt
    "LAW94": """\
/MAT/LAW94/1
plastic
#              RHO_I
              1.0E-9
#Blank

#                C10                 C20                 C30
   0.184342573377574  -0.002127301973852   0.000049595456375
#                 D1                  D2                  D3
   0.054428256220026
""",
    # rd_v_blast/RD-V-0500_Shyue_Shock_Tube/JWL_shock_tube_MUSCL/data/material_MUSCL.inc
    "MULTIFLUID": """\
/MAT/MULTIFLUID/10
LEFT 151

#   mat_ID            Vol_frac
         1                 1.0
         2                   0
""",
    # rd_e/RD-E-5000_Inivol_FSI/50_inivol_and_fluid_structure/data/fsi_drop_container_0000.rad
    "PLAS_PREDEF": """\
/MAT/PLAS_PREDEF/7
Plastic
#       Material Name
PA6GF30
""",
    # rd_e/RD-E-0400_Airbag/04_Airbag/driver/driver_airbag_0000.rad
    "VOID": """\
/MAT/VOID/100001
void
#                RHO                   E                  NU
                1E-9                 500                   0
""",
}

#: family -> (expected material id, expected density)
EXPECTED = {
    "BOUND": (2, 2842.0),
    "CONC": (1, 0.0022),
    "CONNECT": (3, 7.8e-9),
    "FABRI": (2, 8.5e-10),
    "GAS": (1001, 0.0),          # /MAT/GAS has no density card
    "HILL_TAB": (1, 80.0),
    "HYD_JCOOK": (1, 8.96),
    "HYD_VISC": (1, 7.963e-6),
    "JWL": (2, 1.63),            # /MAT/JWL/2/123: id 2, unit 123
    "KELVINMAX": (3, 2e-9),
    "LAW151": (2, 0.0),
    "LAW37": (2, 1e-3),          # liquid-phase density estimate
    "LAW46": (2, 960.0),
    "LAW51": (1, 0.0),           # non-reflecting boundary: all-zero phases
    "LAW59": (100026, 7.9e-9),
    "LAW62": (11, 5e-11),
    "LAW66": (3, 1e-9),
    "LAW69": (1, 1e-9),
    "LAW70": (1, 1e-10),
    "LAW81": (1, 0.0022),
    "LAW82": (1, 1e-9),
    "LAW83": (5, 7.8e-6),        # /MAT/LAW83/5/2: id 5, unit 2
    "LAW88": (1, 1e-9),
    "LAW92": (1, 1e-9),
    "LAW94": (1, 1e-9),
    "MULTIFLUID": (10, 0.0),
    "PLAS_PREDEF": (7, None),    # density from the predef table
    "VOID": (100001, 1e-9),
}


def _parse_block(text, tmp_path, name="SNIP"):
    f = tmp_path / f"{name}_0000.rad"
    f.write_text(text)
    blocks = read_deck(str(f))
    assert blocks, "snippet produced no keyword block"
    return blocks[0]


def _parse_deck(text, tmp_path, name="SNIP"):
    f = tmp_path / f"{name}_0000.rad"
    f.write_text(text)
    model = Model()
    log = MessageLog()
    parse_starter_deck(read_deck(str(f)), model, log)
    return model, log


# ============================================================================
# 1. Every family parses error-free into an InactiveMaterial with density
# ============================================================================

#: families whose PHYSICS is registered by the M37 packs — they now
#: build LIVE materials instead of InactiveMaterial (their constructors
#: need the cfg-named params, so without the cfg tree they are skipped)
ACTIVE_FAMILIES = {"VOID", "GAS", "KELVINMAX", "LAW70",       # pack 1
                   "CONC", "FABRI", "LAW62", "LAW81",         # pack 2
                   "LAW83", "HYD_VISC", "LAW66",
                   "LAW88", "LAW92", "LAW94", "LAW46", "LAW69",
                   "LAW124", "LAW126", "LAW125", "LAW127", "LAW130",
                   "LAW128", "LAW129", "LAW123", "LAW132", "LAW134",
                   "LAW104", "LAW105", "LAW106", "LAW107", "LAW110", "LAW115",
                   "LAW109", "LAW111", "LAW112", "LAW116", "LAW122", "LAW158",
                   "BOUND", "LAW151", "MULTIFLUID",
                   "LAW59", "CONNECT", "LAW52", "GURSON", "LAW16", "GRAY",
                   "LAW14", "COMPSO", "LAW64", "MARTENSITE",
                   "LAW68", "COSSER", "COSSERAT", "LAW72", "HILL_MMC",
                   "LAW65", "ELASTOMER", "LAW58", "FABR_A", "LAW20", "BIMAT",
                   "LAW38", "VISC_TAB", "LAW29", "FEM", "LAW34", "BOLTZMAN",
                   "LAW23", "PLAS_DAMA", "LAW78",
                   "LAW100", "SPOTWELD", "LAW97", "EXPLOSIVE_JWLS", "JWLS",
                   "LAW71", "SUPER_ELAS", "NITINOL", "LAW73", "THERM_HILL", "HILL_THERM",
                   "LAW84", "SWIFT_VOCE", "PLAS_SWIFT_VOCE", "LAW93", "ORTH_HILL",
                   "LAW133", "GRANULAR", "LAW101", "PLAS_POLY", "LAW43", "HILL_TAB"}


@pytest.mark.parametrize("family", sorted(MAT_SNIPPETS))
def test_family_parses_clean(family, tmp_path):
    """Real corpus card -> parse_starter_deck -> material, no errors,
    right id, right density (mass init depends on it).  Laws whose
    physics the M37 packs registered come out LIVE; the rest are
    InactiveMaterial with usable elastic fallbacks."""
    if family in ACTIVE_FAMILIES and not HAVE_CFG:
        pytest.skip("physics constructors need the cfg-named params")
    model, log = _parse_deck(MAT_SNIPPETS[family], tmp_path, family)
    assert not log.errors, log.errors
    mat_id, density = EXPECTED[family]
    assert mat_id in model.materials, \
        f"/MAT/{family} did not produce material {mat_id}"
    mat = model.materials[mat_id]
    if getattr(mat, "inactive", False):
        assert isinstance(mat, InactiveMaterial)
        # the kernel-facing elastic fallbacks must exist and be usable
        assert mat.E > 0.0
        assert 0.0 <= mat.nu < 0.5
        assert mat.G > 0.0
    else:
        # an M37 physics pack owns this law now — LIVE material built
        # from the same record (parse contract unchanged)
        assert family in ACTIVE_FAMILIES, \
            f"{family} unexpectedly active — update ACTIVE_FAMILIES"
        assert not isinstance(mat, InactiveMaterial)
    if HAVE_CFG and density is not None:
        assert mat.rho0 == pytest.approx(density), \
            f"{family}: density {mat.rho0} != {density}"


@needs_cfg
@pytest.mark.parametrize("family", sorted(MAT_SNIPPETS))
def test_family_has_cfg_schema(family):
    """Every ranked-gap family resolves to a real cfg schema (no
    heuristic fallback) with a law number where one exists."""
    law = MAT_SNIPPETS[family].splitlines()[0].split("/")[2]
    schema = mat_reader.catalogue().schema(law)
    assert schema is not None, f"no cfg schema for /MAT/{law}"
    assert schema.statements, f"{schema.path}: empty FORMAT program"


# ============================================================================
# 2. cfg-exact field values (schema round-trips on real cards)
# ============================================================================

@needs_cfg
def test_fabri_params(tmp_path):
    # the READER contract is on the record (LAW19 physics is registered
    # now, so the model-side material is the pack-2 live object)
    block = _parse_block(MAT_SNIPPETS["FABRI"], tmp_path)
    rec = mat_reader.parse_generic_mat(block, MessageLog())
    p = rec.params
    assert p["MAT_EA"] == pytest.approx(500.0)
    assert p["MAT_EB"] == pytest.approx(500.0)
    assert p["MAT_GAB"] == pytest.approx(10.0)
    assert p["MAT_REDFACT"] == pytest.approx(0.001)
    assert p["ISENSOR"] == 1
    assert rec.law_number == 19


@needs_cfg
def test_law37_jammed_fixed_columns(tmp_path):
    """The tank-sloshing LAW37 card jams '0.0' against the next 20-char
    field ('...2089.0            0.01.00000000000000E-03') — a
    whitespace tokenizer CANNOT read it; the cfg column widths can."""
    model, log = _parse_deck(MAT_SNIPPETS["LAW37"], tmp_path)
    p = model.materials[2].params
    assert p["Lqud_Rho_l"] == pytest.approx(1e-3)     # liquid density
    assert p["C_l"] == pytest.approx(2089.0)          # liquid sound speed
    assert p["Lqud_Rho_g"] == pytest.approx(1.22e-6)  # gas density
    assert p["Lqud_Gamma_bulk"] == pytest.approx(1.4)
    # the blank Psh card (a whitespace-only line) must NOT shift the
    # card cursor
    assert p["Nu_g"] == pytest.approx(0.00143)


@needs_cfg
def test_law94_blank_card_statement(tmp_path):
    """matl94_94.cfg has an explicit BLANK; between rho and C10 — the
    deck's whitespace-only line must be consumed by it, keeping the
    C10/C20/C30 and D1 cards aligned."""
    model, log = _parse_deck(MAT_SNIPPETS["LAW94"], tmp_path)
    p = model.materials[1].params
    assert p["LAW94_C01"] == pytest.approx(0.184342573377574)
    assert p["LAW94_C02"] == pytest.approx(-0.002127301973852)
    assert p["LAW94_C03"] == pytest.approx(4.9595456375e-5)
    assert p["LAW94_D1"] == pytest.approx(0.054428256220026)


@needs_cfg
def test_gas_subtype_and_fields(tmp_path):
    """/MAT/GAS/MASS: the HEADER %s capture must yield the MASS subtype
    (MGAS_TYPE=1) and the MW/Cp cards (record-level: GAS physics is
    registered by pack 1, the model-side object is a GasMaterial)."""
    block = _parse_block(MAT_SNIPPETS["GAS"], tmp_path)
    rec = mat_reader.parse_generic_mat(block, MessageLog())
    assert rec.subtype == "MASS"
    assert rec.params["MGAS_TYPE"] == 1
    assert rec.params["MASS"] == pytest.approx(2.89e-5)
    assert rec.params["ABG_cpai"] == pytest.approx(928474283.7)
    # and the model-side object carries the derived thermodynamics
    model, log = _parse_deck(MAT_SNIPPETS["GAS"], tmp_path)
    mat = model.materials[1001]
    assert not isinstance(mat, InactiveMaterial)
    assert mat.mw == pytest.approx(2.89e-5)


@needs_cfg
def test_law66_card_list_arrays(tmp_path):
    """ISRATE=4 selects the NFUNC/TFUNC CARD_LIST branch: 5 compression
    + 5 tension (funct_ID, rate) rows into ARRAY attributes."""
    model, log = _parse_deck(MAT_SNIPPETS["LAW66"], tmp_path)
    p = model.materials[3].params
    assert p["ISRATE"] == 4
    assert p["NFUNC"] == 5 and p["TFUNC"] == 5
    assert p["ABG_IPt"] == [1, 2, 3, 4, 5]
    assert p["K_A1"] == pytest.approx([0.1, 1.0, 10.0, 100.0, 1000.0])
    assert p["ABG_IPdel"] == [1, 2, 3, 4, 5]


@needs_cfg
def test_law51_subobject_iform6(tmp_path):
    """LAW51 Iform=6 routes through SUBOBJECTS(/SUBOBJECT/LAW51_IFLAG_6)
    — the per-phase cards live in law51_Iflag_6.cfg and must be read on
    the same card cursor."""
    model, log = _parse_deck(MAT_SNIPPETS["LAW51"], tmp_path)
    p = model.materials[1].params
    assert p["MAT_Iflag"] == 6
    assert p["PEXT"] == pytest.approx(0.0)
    # three phases of (ALPHA, RHO, E, Pmin, P0) + SSP0 were consumed
    assert len(p["MAT_RHO_Iflg0_phas"]) == 3
    assert len(p["MAT_SSP0_Iflg0_phas"]) == 3


@needs_cfg
def test_multifluid_free_card_list(tmp_path):
    """LAW151/MULTIFLUID: FREE_CARD_LIST reads (mat_ID, Vfrac) rows
    until the block ends and sets the size attribute."""
    model, log = _parse_deck(MAT_SNIPPETS["MULTIFLUID"], tmp_path)
    p = model.materials[10].params
    assert p["MAT_ID_ARRAY"] == [1, 2]
    assert p["VOL_FRAC_ARRAY"] == pytest.approx([1.0, 0.0])
    assert p["NIP"] == 2


@needs_cfg
def test_law82_cell_arrays(tmp_path):
    model, log = _parse_deck(MAT_SNIPPETS["LAW82"], tmp_path)
    p = model.materials[1].params
    assert p["ORDER"] == 2
    assert p["MAT_NU"] == pytest.approx(0.4997)
    assert p["Mu_arr"][:2] == pytest.approx([4.5637449070023e-05,
                                             0.547913433558156])
    assert p["Alpha_arr"][:2] == pytest.approx([7.168617832124,
                                                -4.158214786551])


def test_mat_id_with_unit_id(tmp_path):
    """/MAT/LAW92/1/1 (id 1, unit 1) and /MAT/JWL/2/123: the FIRST
    trailing integer is the material id — the unit id must not steal
    it."""
    model, log = _parse_deck(MAT_SNIPPETS["LAW92"], tmp_path)
    assert not log.errors
    assert 1 in model.materials
    assert model.materials[1].record.unit_id == 1
    model, log = _parse_deck(MAT_SNIPPETS["JWL"], tmp_path)
    assert 2 in model.materials
    assert model.materials[2].record.unit_id == 123


@needs_cfg
def test_predef_material_name_string(tmp_path):
    """/MAT/PLAS_PREDEF carries the material NAME as a string card."""
    model, log = _parse_deck(MAT_SNIPPETS["PLAS_PREDEF"], tmp_path)
    mat = model.materials[7]
    assert mat.params.get("Material_Name_Str") == "PA6GF30"


# ============================================================================
# 3. /ALE/MAT, /EULER/MAT, /HEAT/MAT — parse-only notes
# ============================================================================

def test_ale_euler_heat_mat_notes(tmp_path):
    deck = MAT_SNIPPETS["HYD_VISC"] + """\
/ALE/MAT/1
                 0.5
/EULER/MAT/1
                 0.7
/HEAT/MAT/1
                 293             3.5E-6                 0.5                   0         0
                   0                   0                   0
/END
"""
    model, log = _parse_deck(deck, tmp_path)
    assert not log.errors, log.errors
    assert len(model.raw_mat_notes) == 3
    kinds = sorted(k for k, _i, _p, _s in model.raw_mat_notes)
    assert kinds == ["ALE/MAT", "EULER/MAT", "HEAT/MAT"]
    # the resolve step attaches them to the material's params
    from pyradioss.starter.initialization import resolve_materials
    resolve_materials(model, log)
    p = model.materials[1].params
    assert "ale_mat_note" in p
    assert "euler_mat_note" in p
    assert "heat_mat_note" in p
    if HAVE_CFG:
        # radioss2022 dropped the Flrd card from /ALE/MAT and /EULER/MAT
        # (header-only blocks) — the notes still record the defaults;
        # /HEAT/MAT keeps its T0/rho0Cp cards
        assert p["heat_mat_note"]["HEAT_T0"] == pytest.approx(293.0)
        assert p["heat_mat_note"]["HEAT_RHocp"] == pytest.approx(3.5e-6)


def test_other_ale_options_still_skipped(tmp_path):
    model, log = _parse_deck("/ALE/UNPORTED/DONEA\n0.1 0.2\n/END\n", tmp_path)
    assert not log.errors
    assert not model.raw_mat_notes
    assert any("not ported" in w for w in log.warnings)


# ============================================================================
# 4. The InactiveMaterial flow: Starter accepts, Engine refuses
# ============================================================================

VOID_BRICK_DECK = """\
/BEGIN
M37VOID
/NODE
        1               0.0               0.0               0.0
        2               1.0               0.0               0.0
        3               1.0               1.0               0.0
        4               0.0               1.0               0.0
        5               0.0               0.0               1.0
        6               1.0               0.0               1.0
        7               1.0               1.0               1.0
        8               0.0               1.0               1.0
/BRICK/1
        1         1         2         3         4         5         6         7         8
/PART/1
void part
         1         1
/PROP/SOLID/1
solid prop
         1
                 1.1                0.05                 0.1
/MAT/VOID/1
void material
#                RHO                   E                  NU
                1E-9                 500                   0
/END
"""


def test_starter_accepts_inactive_material(tmp_path, clean_registry):
    """Full Starter run: /PART -> /MAT cross-ref to an INACTIVE law must
    NOT error; mass init works from the parsed density (rho*V/8 per
    brick node); the listing carries the per-instance M37 note.  VOID
    physics is registered since pack 1 — popped here so the test keeps
    exercising the inactive flow itself."""
    clean_registry.pop("VOID", None)
    clean_registry.pop("LAW0", None)
    from pyradioss.starter.starter import run_starter
    f = tmp_path / "M37VOID_0000.rad"
    f.write_text(VOID_BRICK_DECK)
    model = run_starter(str(f))              # raises on any starter error
    mat = model.materials[1]
    assert isinstance(mat, InactiveMaterial)
    assert mat.rho0 == pytest.approx(1e-9)
    # unit cube, rho = 1e-9 -> element mass 1e-9, 1/8 per node
    assert model.mass[:8] == pytest.approx(np.full(8, 1.25e-10))
    listing = (tmp_path / "M37VOID_0000.out").read_text(errors="replace")
    assert "parsed, physics not implemented (M37)" in listing


def test_engine_refuses_inactive_material(tmp_path, clean_registry):
    """The Engine must refuse to run an element group whose part
    references an InactiveMaterial — clear error naming the law (VOID
    physics popped: the test exercises the refusal machinery)."""
    clean_registry.pop("VOID", None)
    clean_registry.pop("LAW0", None)
    from pyradioss.engine.engine import run_engine
    from pyradioss.starter.starter import run_starter
    (tmp_path / "M37VOID_0000.rad").write_text(VOID_BRICK_DECK)
    (tmp_path / "M37VOID_0001.rad").write_text("/RUN/M37VOID/1\n1e-6\n")
    run_starter(str(tmp_path / "M37VOID_0000.rad"))
    with pytest.raises(InactiveMaterialError) as exc:
        run_engine(str(tmp_path / "M37VOID_0001.rad"))
    msg = str(exc.value)
    assert "VOID" in msg
    assert "physics not implemented" in msg
    assert "bricks" in msg


def test_refusal_only_for_referenced_materials(tmp_path, clean_registry):
    """A merely DEFINED inactive material (no part references it — the
    /MAT/GAS of an airbag) must not block the Engine (GAS physics
    popped: the test exercises the unreferenced-inactive flow)."""
    clean_registry.pop("GAS", None)
    deck = VOID_BRICK_DECK.replace(
        "/MAT/VOID/1\nvoid material\n"
        "#                RHO                   E                  NU\n"
        "                1E-9                 500                   0\n",
        "/MAT/ELAST/1\nsteel\n7.8e-9\n210000 0.3\n"
        + MAT_SNIPPETS["GAS"])
    f = tmp_path / "M37GAS_0000.rad"
    f.write_text(deck.replace("M37VOID", "M37GAS"))
    from pyradioss.starter.starter import run_starter
    model = run_starter(str(f))
    assert isinstance(model.materials[1001], InactiveMaterial)
    refuse_inactive_materials(model)          # must NOT raise


def test_fabri_on_shells_starter_accepts(tmp_path, clean_registry):
    """/PART cross-ref to an InactiveMaterial on SHELL elements: parse +
    group build + checks run clean (warning, never an error) — FABRI
    physics popped, the test exercises the inactive-on-shells flow."""
    clean_registry.pop("FABRI", None)
    clean_registry.pop("LAW19", None)
    deck = """\
/BEGIN
M37FAB
/NODE
        1               0.0               0.0               0.0
        2               1.0               0.0               0.0
        3               1.0               1.0               0.0
        4               0.0               1.0               0.0
/SHELL/1
        1         1         2         3         4
/PART/1
fabric part
         1         2
/PROP/SHELL/1
shell prop
0.5
""" + MAT_SNIPPETS["FABRI"] + "/END\n"
    from pyradioss.starter.starter import run_starter
    f = tmp_path / "M37FAB_0000.rad"
    f.write_text(deck)
    model = run_starter(str(f))               # raises on starter error
    assert isinstance(model.materials[2], InactiveMaterial)
    assert model.mass[:4].sum() == pytest.approx(8.5e-10 * 0.5)


# ============================================================================
# 5. The physics registry hook (the next-phase builders' contract)
# ============================================================================

@pytest.fixture
def clean_registry():
    saved = dict(MAT_PHYSICS_REGISTRY)
    yield MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY.clear()
    MAT_PHYSICS_REGISTRY.update(saved)


def test_registry_constructor_is_used(tmp_path, clean_registry):
    """A registered constructor receives the GenericMaterialRecord and
    its return object lands in model.materials (no InactiveMaterial)."""
    seen = {}

    def build_void(rec):
        seen["rec"] = rec
        return Material(id=rec.id, law=1, rho0=rec.density,
                        title=rec.title,
                        params={"E": float(rec.params.get("MAT_E", 1.0)),
                                "nu": 0.0})

    clean_registry["VOID"] = build_void
    model, log = _parse_deck(MAT_SNIPPETS["VOID"], tmp_path)
    assert not log.errors
    mat = model.materials[100001]
    assert not isinstance(mat, InactiveMaterial)
    assert isinstance(seen["rec"], GenericMaterialRecord)
    assert seen["rec"].law_name == "VOID"
    assert mat.rho0 == pytest.approx(1e-9)
    if HAVE_CFG:
        assert mat.E == pytest.approx(500.0)


def test_registry_law_number_alias(tmp_path, clean_registry):
    """Registering under LAW19 must catch the /MAT/FABRI spelling too
    (the number alias resolves through the cfg catalogue)."""
    if not HAVE_CFG:
        pytest.skip("number aliasing needs the cfg catalogue")
    calls = []

    def build(rec):
        calls.append(rec.law_name)
        return Material(id=rec.id, law=1, rho0=rec.density,
                        params={"E": 1.0, "nu": 0.3})

    clean_registry.pop("FABRI", None)      # pack 2 owns it — out of the way
    clean_registry["LAW19"] = build
    model, log = _parse_deck(MAT_SNIPPETS["FABRI"], tmp_path)
    assert calls == ["FABRI"]
    assert not isinstance(model.materials[2], InactiveMaterial)


def test_registry_failure_is_an_error_not_a_crash(tmp_path, clean_registry):
    def broken(rec):
        raise ValueError("bad constants")

    clean_registry["VOID"] = broken
    model, log = _parse_deck(MAT_SNIPPETS["VOID"], tmp_path)
    assert any("constructor failed" in e for e in log.errors)
    assert 100001 not in model.materials


@needs_cfg
def test_registry_pending_laws_have_schemas():
    """The next-phase physics builders register LAW19/24/35/44/70/81,
    VOID and GAS — every spelling must already resolve to a cfg schema
    (name AND number alias) so their records carry full params."""
    cat = mat_reader.catalogue()
    for key in ("LAW19", "FABRI", "LAW24", "CONC", "LAW35", "FOAM_VISC",
                "LAW44", "COWPER", "LAW70", "FOAM_TAB", "LAW81",
                "VOID", "LAW0", "GAS"):
        assert cat.schema(key) is not None, f"no cfg schema for {key}"


@needs_cfg
def test_catalogue_covers_the_full_mat_family():
    """The user asked for ALL material cards: the catalogue must know
    (far) more than the 31 coverage gaps — the whole 2022 MAT tree."""
    assert len(mat_reader.catalogue().known_laws()) >= 150


# ============================================================================
# 6. Restart round-trip: InactiveMaterial survives the rst pickle
# ============================================================================

def test_inactive_material_survives_restart(tmp_path, clean_registry):
    clean_registry.pop("VOID", None)       # exercise the INACTIVE flow
    clean_registry.pop("LAW0", None)
    from pyradioss.starter.restart import read_restart
    from pyradioss.starter.starter import run_starter
    f = tmp_path / "M37VOID_0000.rad"
    f.write_text(VOID_BRICK_DECK)
    run_starter(str(f))
    model, saved = read_restart(str(tmp_path / "M37VOID_0000.rst"))
    mat = model.materials[1]
    assert isinstance(mat, InactiveMaterial)
    assert getattr(mat, "inactive", False)
    assert mat.rho0 == pytest.approx(1e-9)
    with pytest.raises(InactiveMaterialError):
        refuse_inactive_materials(model)
