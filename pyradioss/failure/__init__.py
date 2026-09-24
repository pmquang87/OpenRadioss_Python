"""
pyradioss.failure — failure models (/FAIL cards) and element deletion.

Fortran origin: ``engine/source/materials/fail/`` — one sub-directory per
criterion (johnson_cook, biquad, tab, ...). In OpenRadioss a /FAIL option
attaches to a material (the keyword carries the mat_ID:
``/FAIL/JOHNSON/mat_ID``); every element of a part using that material
accumulates a damage variable D, and when D reaches 1 at an integration
point the point is 'broken'. Element deletion follows:

* solids — the single integration point of the ported hexa/tetra breaks
  -> the element is deleted (GBUF%OFF = 0);
* shells — points break layer by layer (the ``layfail`` array of the
  shell kernels); the element is deleted according to the card's
  Ifail_sh flag: 1 = when ONE layer is broken (default), 2 = when ALL
  layers are broken.

A deleted element keeps its nodal mass (like the original) but carries no
stress, no bulk viscosity, no hourglass force, and no longer constrains
the time step. Its stored elastic energy at the moment of deletion simply
disappears from the system while remaining counted in the internal-energy
history — the energy balance therefore stays consistent (deletion is an
energy sink booked as internal energy, exactly the original's behaviour).

The generic ``eps_p_max`` element deletion of the material cards
(/MAT/LAW2, /MAT/LAW36) is handled by the same kernel plumbing but needs
no model here: the kernels compare the equivalent plastic strain to the
material's threshold directly.

Dispatch contract (mirrors the material-law dispatch):

    solid_step(fail, sig, d_epsp, deps, dt, dama, tstar)  -> broken mask
    shell_step(fail, sig, d_epsp, deps, dt, dama, tstar, eps_tot)  -> broken mask

with sig/deps the (m, 6) or (m, 3) slice arrays of the group, d_epsp the
plastic-strain increment of this cycle, dama the persistent damage
array (in-place) and tstar the homologous temperature of the points.
All vectorized over the element slice.
"""

from . import (  # noqa: F401
    alter,
    biquad,
    brokmann,
    chang,
    cockcroft,
    connect,
    emc,
    energy,
    fabric,
    fail_composite,
    fail_gurson,
    fail_ladeveze,
    fail_rtcl,
    failwave,
    fld,
    fractal,
    gene1,
    hashin,
    hc_dsse,
    hoffman,
    inicrack,
    inievo,
    johnson,
    lemaitre,
    max_strain,
    mmc,
    mullins,
    nxt,
    orthbiquad,
    orthenerg,
    orthstrain,
    puck,
    sahraei,
    snconnect,
    spalling,
    syazwan,
    tab,
    tab1,
    tab2,
    tbutcher,
    tensstrain,
    tsaihill,
    tsaiwu,
    tvergaard,
    user,
    visual,
    wilkins,
)


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance the damage of a solid slice; returns the broken mask."""
    ftype = fail.type.upper()
    if ftype == "JOHNSON":
        return johnson.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "BIQUAD":
        return biquad.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "TAB1":
        return tab1.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("TAB2", "FAIL_TAB2"):
        return tab2.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("TAB", "TAB_OLD", "FAIL_TAB"):
        return tab.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "SNCONNECT":
        return snconnect.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "FLD":
        return fld.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "WILKINS":
        return wilkins.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("CHANG", "CHANGCHANG"):
        return chang.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "HASHIN":
        return hashin.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("MMC", "WIERZBICKI"):
        return mmc.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("TBUTCHER", "TULER-BUTCHER", "TULER_BUTCHER"):
        return tbutcher.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("GURSON", "GURSON_MODEL", "GURSON_DAMAGE"):
        return fail_gurson.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("TVERGAARD", "GTN"):
        return tvergaard.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("PUCK", "PUCK_COMPOSITE"):
        return puck.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("COCKCROFT", "COCKCROFT_LATHAM", "COCKCROFT-LATHAM", "COCKROFT", "COCKROFT_LATHAM"):
        return cockcroft.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("HC_DSSE", "HC-DSSE", "HCDSSE", "HC/DSSE", "HC"):
        return hc_dsse.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("LADEVEZE", "LAD_DAMA", "LADEVEZE_COMPOSITE"):
        return fail_ladeveze.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("RTCL", "RTCL_MODEL", "RTCL_LAW"):
        return fail_rtcl.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "HOFFMAN":
        return hoffman.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("TSAIHILL", "TSAI-HILL", "TSAI_HILL"):
        return tsaihill.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("TSAIWU", "TSAI-WU", "TSAI_WU"):
        return tsaiwu.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("MAXSTRAIN", "MAX_STRAIN"):
        return max_strain.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("COMPOSITE", "COMPOSITE_FAILURE"):
        return fail_composite.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("ORTHSTRAIN", "ORTH_STRAIN"):
        return orthstrain.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("ORTHBIQUAD", "ORTH_BIQUAD"):
        return orthbiquad.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("ORTHENERG", "ORTH_ENERG"):
        return orthenerg.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "ENERGY":
        return energy.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "LEMAITRE":
        return lemaitre.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("SPALLING", "SPALL"):
        return spalling.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "EMC":
        return emc.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("TENSSTRAIN", "TENSTRAIN"):
        return tensstrain.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "CONNECT":
        return connect.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "FABRIC":
        return fabric.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("MULLINS", "MULLINS_OR", "MULLINS-OR"):
        return mullins.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("ALTER", "WINDSHIELD", "WINDSHIELD_ALTER"):
        return alter.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("BROKMANN", "FAIL_BROKMANN", "ALTER_BROKMANN"):
        return brokmann.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("GENE1", "FAIL_GENE1"):
        return gene1.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("INIEVO", "FAIL_INIEVO"):
        return inievo.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "NXT":
        return nxt.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "SAHRAEI":
        return sahraei.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "SYAZWAN":
        return syazwan.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype == "VISUAL":
        return visual.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("FRACTAL", "FRACTAL_DMG"):
        return fractal.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    if ftype in ("USER", "USER1", "USER2", "USER3", "FAIL_USER"):
        return user.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar)
    raise NotImplementedError(f"/FAIL/{fail.type} not ported")


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None):
    """Advance the damage of one shell layer; returns the broken mask."""
    ftype = fail.type.upper()
    if ftype == "JOHNSON":
        return johnson.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "BIQUAD":
        return biquad.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "TAB1":
        return tab1.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("TAB2", "FAIL_TAB2"):
        return tab2.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("TAB", "TAB_OLD", "FAIL_TAB"):
        return tab.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "SNCONNECT":
        return snconnect.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "FLD":
        return fld.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "WILKINS":
        return wilkins.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("CHANG", "CHANGCHANG"):
        return chang.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "HASHIN":
        return hashin.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("MMC", "WIERZBICKI"):
        return mmc.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("TBUTCHER", "TULER-BUTCHER", "TULER_BUTCHER"):
        return tbutcher.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("GURSON", "GURSON_MODEL", "GURSON_DAMAGE"):
        return fail_gurson.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("TVERGAARD", "GTN"):
        return tvergaard.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("PUCK", "PUCK_COMPOSITE"):
        return puck.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("COCKCROFT", "COCKCROFT_LATHAM", "COCKCROFT-LATHAM", "COCKROFT", "COCKROFT_LATHAM"):
        return cockcroft.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("HC_DSSE", "HC-DSSE", "HCDSSE", "HC/DSSE", "HC"):
        return hc_dsse.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("LADEVEZE", "LAD_DAMA", "LADEVEZE_COMPOSITE"):
        return fail_ladeveze.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("RTCL", "RTCL_MODEL", "RTCL_LAW"):
        return fail_rtcl.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "HOFFMAN":
        return hoffman.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("TSAIHILL", "TSAI-HILL", "TSAI_HILL"):
        return tsaihill.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("TSAIWU", "TSAI-WU", "TSAI_WU"):
        return tsaiwu.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("MAXSTRAIN", "MAX_STRAIN"):
        return max_strain.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("COMPOSITE", "COMPOSITE_FAILURE"):
        return fail_composite.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("ORTHSTRAIN", "ORTH_STRAIN"):
        return orthstrain.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("ORTHBIQUAD", "ORTH_BIQUAD"):
        return orthbiquad.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("ORTHENERG", "ORTH_ENERG"):
        return orthenerg.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "ENERGY":
        return energy.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "LEMAITRE":
        return lemaitre.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("SPALLING", "SPALL"):
        return spalling.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "EMC":
        return emc.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("TENSSTRAIN", "TENSTRAIN"):
        return tensstrain.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "CONNECT":
        return connect.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "FABRIC":
        return fabric.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("MULLINS", "MULLINS_OR", "MULLINS-OR"):
        return mullins.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("ALTER", "WINDSHIELD", "WINDSHIELD_ALTER"):
        return alter.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("BROKMANN", "FAIL_BROKMANN", "ALTER_BROKMANN"):
        return brokmann.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("GENE1", "FAIL_GENE1"):
        return gene1.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("INIEVO", "FAIL_INIEVO"):
        return inievo.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "NXT":
        return nxt.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "SAHRAEI":
        return sahraei.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "SYAZWAN":
        return syazwan.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype == "VISUAL":
        return visual.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("FRACTAL", "FRACTAL_DMG"):
        return fractal.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    if ftype in ("USER", "USER1", "USER2", "USER3", "FAIL_USER"):
        return user.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)
    raise NotImplementedError(f"/FAIL/{fail.type} not ported")


def beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=None, tstar=None, **kwargs):
    """Advance the damage of a beam element slice; returns the broken mask.

    Fortran origin: engine/source/elements/beam/fail_beam3.F
    """
    ftype = fail.type.upper()
    mod = FAILURE_MODELS.get(ftype)
    if mod is not None and hasattr(mod, "beam_step"):
        return mod.beam_step(fail, svm, pressure, d_epsp, deps, dt, dama, length=length, tstar=tstar, **kwargs)
    raise NotImplementedError(f"/FAIL/{fail.type} beam failure not ported")


def integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=None, tstar=None, ip=0, npg=1, **kwargs):
    """Advance the damage of an integrated beam integration point; returns broken mask.

    Fortran origin: engine/source/elements/beam/fail_beam18.F
    """
    ftype = fail.type.upper()
    mod = FAILURE_MODELS.get(ftype)
    if mod is not None and hasattr(mod, "integrated_beam_step"):
        return mod.integrated_beam_step(fail, sig, d_epsp, deps, dt, dama, length=length, tstar=tstar, ip=ip, npg=npg, **kwargs)
    raise NotImplementedError(f"/FAIL/{fail.type} integrated beam failure not ported")


def thick_shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None, eps_tot=None, pla=None):
    """Advance the damage of a thick shell layer; returns the broken mask.

    Fortran origin: engine/source/materials/fail/fld/fail_fld_tsh.F
    """
    ftype = fail.type.upper()
    if ftype == "FLD":
        return fld.thick_shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot, pla=pla)
    mod = FAILURE_MODELS.get(ftype)
    if mod is not None and hasattr(mod, "thick_shell_step"):
        return mod.thick_shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot, pla=pla)
    return shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=tstar, eps_tot=eps_tot)


def xfem_step(fail, sig, d_epsp, deps, dt, dama, elcrkini, tstar=None, eps_tot=None, dadv=1.0, is_phantom=False):
    """Advance damage and update crack propagation state for XFEM.

    Fortran origin: engine/source/materials/fail/*/*_xfem.F
    """
    ftype = fail.type.upper()
    mod = FAILURE_MODELS.get(ftype)
    if mod is not None and hasattr(mod, "xfem_step"):
        return mod.xfem_step(fail, sig, d_epsp, deps, dt, dama, elcrkini, tstar=tstar, eps_tot=eps_tot, dadv=dadv, is_phantom=is_phantom)
    raise NotImplementedError(f"/FAIL/{fail.type} XFEM failure not ported")



FAILURE_MODELS: dict[str, object] = {
    "ALTER": alter,
    "WINDSHIELD": alter,
    "WINDSHIELD_ALTER": alter,
    "BIQUAD": biquad,
    "BROKMANN": brokmann,
    "FAIL_BROKMANN": brokmann,
    "ALTER_BROKMANN": brokmann,
    "CHANG": chang,
    "CHANGCHANG": chang,
    "COCKCROFT": cockcroft,
    "COCKCROFT_LATHAM": cockcroft,
    "CONNECT": connect,
    "EMC": emc,
    "ENERGY": energy,
    "FABRIC": fabric,
    "COMPOSITE": fail_composite,
    "COMPOSITE_FAILURE": fail_composite,
    "GURSON": fail_gurson,
    "GURSON_MODEL": fail_gurson,
    "LADEVEZE": fail_ladeveze,
    "LAD_DAMA": fail_ladeveze,
    "RTCL": fail_rtcl,
    "RTCL_MODEL": fail_rtcl,
    "FAILWAVE": failwave,
    "FLD": fld,
    "FRACTAL": fractal,
    "GENE1": gene1,
    "HASHIN": hashin,
    "HC_DSSE": hc_dsse,
    "HC-DSSE": hc_dsse,
    "HOFFMAN": hoffman,
    "INIEVO": inievo,
    "JOHNSON": johnson,
    "LEMAITRE": lemaitre,
    "MAXSTRAIN": max_strain,
    "MAX_STRAIN": max_strain,
    "MMC": mmc,
    "MULLINS": mullins,
    "NXT": nxt,
    "ORTHBIQUAD": orthbiquad,
    "ORTHENERG": orthenerg,
    "ORTHSTRAIN": orthstrain,
    "PUCK": puck,
    "SAHRAEI": sahraei,
    "SNCONNECT": snconnect,
    "SPALLING": spalling,
    "SYAZWAN": syazwan,
    "TAB": tab,
    "TAB1": tab1,
    "TAB2": tab2,
    "TBUTCHER": tbutcher,
    "TENSSTRAIN": tensstrain,
    "TSAIHILL": tsaihill,
    "TSAIWU": tsaiwu,
    "TVERGAARD": tvergaard,
    "USER": user,
    "VISUAL": visual,
    "WILKINS": wilkins,
}


def register_failure_model(name: str, module: object) -> None:
    """Register a failure model module in the dispatcher table."""
    FAILURE_MODELS[name.upper()] = module
