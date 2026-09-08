"""
pyradioss.materials — constitutive laws.

Fortran origin: ``engine/source/materials/mat/matXXX/sigepsXX.F``. In
OpenRadioss every material law is a routine ``SIGEPS<law>`` receiving the
strain-rate components and the old stress of a *group* of elements and
returning the new stress (plus updated internal variables and the sound
speed). The element kernels call it once per integration point per cycle.

The port keeps that contract exactly (vectorized over the group):

    law01_elastic.solid_update / shell_update        (LAW1)
    law02_johnson_cook.solid_update / shell_update   (LAW2)
    law36_tabulated.solid_update / shell_update      (LAW36)
    law27_brittle.shell_update                       (LAW27, shells only)
    law42_ogden.solid_update                         (LAW42, solids only)

Stress storage convention (Voigt, engineering shear):
    solids : (n, 6)  = [xx, yy, zz, xy, yz, zx]
    shells : (n, 3)  = [xx, yy, xy]   (in-plane; transverse shear handled
                                       elastically by the shell kernel)
Strain increments use the same ordering, with *engineering* shear
(gamma = 2*eps) exactly like the Fortran EPSPXX/EPSPXY arguments.

The objective (Jaumann) rotation of the old stress is done by the element
kernel BEFORE calling the law — same split as the Fortran, where SROTA3.F
rotates the stress and SIGEPS only integrates the constitutive rate.
(Hyperelastic laws — LAW42 — overwrite the stress from the deformation
gradient instead, so the pre-rotation is harmless for them.)

The ``extra`` argument (M3)
---------------------------
Some laws carry state beyond (sig, epsp), or need kinematic context
beyond the strain increment. The Fortran passes dozens of UVAR/extra
arguments; the port passes one ``extra`` dict of per-slice array views:

* ``extra_shapes(mat, nip)`` tells the element kernels which persistent
  per-element (per-layer for shells) arrays a law needs; the kernels
  allocate them in the group state under ``mat_extra`` and slice them
  into ``extra`` at call time.
* ``needs_defgrad(mat)`` asks for the deformation gradient: the solid
  kernels then compute F exactly from the stored initial shape-function
  gradients each cycle and pass it as ``extra["F"]``.
* shell kernels always add ``extra["layfail"]`` (the shared layer-failure
  array — see pyradioss.failure) so a law can break layers (LAW27).

``solid_update`` returns (sig, epsp, c) where c is a per-element sound
speed array, or None to keep the kernel's constant elastic estimate.
This is the Fortran SOUNDSP output: laws whose tangent stiffness can
exceed the ground-state one (LAW42 at large stretch!) MUST return the
true current sound speed or the Courant time step is not a bound.
"""

from . import (eos, law01_elastic, law02_johnson_cook, law06_hyd_visc,  # noqa: F401
               law19_fabric, law24_concrete, law27_brittle,
               law35_kelvinmax, law36_tabulated, law40_kelvinmax,
               law42_ogden, law44_cowper, law62_hypervisco,
               law70_tabfoam, law81_druckerprager, law83_spotweld,
               law114_seatbelt, law120_advanced,
               mat_gas, mat_void)


def register_materials():
    eos._register()
    law01_elastic._register()
    law02_johnson_cook._register()
    law06_hyd_visc._register()
    law19_fabric._register()
    law24_concrete._register()
    law35_kelvinmax._register()
    law40_kelvinmax._register()
    law44_cowper._register()
    law62_hypervisco._register()
    law70_tabfoam._register()
    law81_druckerprager._register()
    law83_spotweld._register()
    law114_seatbelt._register()
    law120_advanced._register()
    mat_gas._register()
    mat_void._register()


def extra_shapes(mat, nip=None):
    """Per-element persistent state a law needs beyond (sig, epsp).

    Returns {name: trailing_shape}; the kernels allocate arrays of shape
    (n, *trailing_shape) for solids and (n, nip, *trailing_shape[1:])...
    — in practice the shapes below already include the layer dimension
    for shell laws (nip is the layer count of the property; ``nip=None``
    for solids, whose per-point state is per-element)."""
    shapes = {}
    if mat.law == 27:
        shapes.update(eps27=(nip, 3), crk27=(nip,), ang27=(nip,),
                      dmg27=(nip, 2))
    if mat.law == 36:
        if mat.params.get("c_hard", 0.0) > 0.0:
            shapes.update(sigb36=(nip, 3) if nip else (6,))
        if mat.params.get("f_cut", 0.0) > 0.0:
            shapes.update(epsd36=(nip,) if nip else ())
    if mat.law == 19:
        # M37 pack 2: total strain, zerostress reference stress SIGI and
        # the law's own time accumulator (see law19_fabric docstring)
        shapes.update(eps19=(nip, 3), sigi19=(nip, 3), t19=(nip,))
    if mat.law == 24:
        # M37 pack 2: the LBUF state of conc24.F (law24_concrete docstring)
        # Includes steel reinforcement state (siga24, epsa24)
        shapes.update(strain24=(6,), sigc24=(6,), crak24=(3,),
                      dam24=(3,), ang24=(6,), epsf24=(3,),
                      siga24=(3,), epsa24=(3,),
                      vk024=(), vk24=(), rob24=(), off24=(), ini24=())
    if mat.law == 81:
        # M37 pack 2: the defp(nel, 2) plastic strains of sigeps81.F90
        shapes.update(epspd81=(), epspv81=())
    if mat.law == 62 and mat.params.get("NPRONY", 0) > 0:
        # M37 pack 2: Prony history of sigeps62.F (UVAR 1:6 = previous
        # global deviatoric PK2, then 6 per Prony term)
        shapes.update(sdg62=(6,), h62=(mat.params["NPRONY"], 6))
    if mat.law == 70:
        # M37 pack 1: total strain, the 10 UVARs of sigeps70.F and the
        # filtered strain rate (law70_tabfoam docstring; solids only)
        shapes.update(eps70=(6,), uv70=(10,), epsd70=())
    if mat.law == 35:
        # M37 pack 1: total strain, closed-cell air pressure (UVAR1) and
        # the filtered strain rate (UVAR4) of sigeps35.F (solids only)
        shapes.update(eps35=(6,), sigair35=(), edot35=())
    if mat.law == 40:
        # M37 pack 1: total strain + the 40 UVARs of sigeps40.F (Stassi
        # criteria, EDRV rate memory, 5x6 Prony branch stresses)
        shapes.update(eps40=(6,), uv40=(40,))
    if mat.law == 44:
        # M37 pack 1 (law44_cowper): the filtered strain-rate state is
        # only needed when filtering / VP=1 is active; the total strain
        # only when the tension softening (eps_t1) is finite
        if mat.params.get("ismooth", 0) or mat.params.get("vflag") == 1:
            shapes["epsd44"] = (nip,) if nip is not None else ()
        if mat.params.get("epsr1", 1e30) < 1e30:
            shapes["eps44"] = (nip, 3) if nip is not None else (6,)
    if mat.law == 2 and "mT" in mat.params:
        # adiabatic temperature RISE above T_i (M6 thermal terms)
        shapes["temp"] = (nip,) if nip is not None else ()
    if getattr(mat, "law", None) == 83 or type(mat).__name__ == "Law83":
        shapes.update(epsp=(), asrate=())
    if getattr(mat, "fail", None) is not None and mat.fail.type == "FLD":
        shapes["eps_fld"] = (nip, 3) if nip is not None else (3,)
    return shapes


def needs_defgrad(mat) -> bool:
    """True if the law is total-strain and needs F from the kernel
    (LAW42 Ogden; LAW62 hyper-visco foam since M37 pack 2)."""
    return mat.law in (42, 62)


def needs_env(mat) -> bool:
    """True if the law wants the kernel's per-cycle environment views in
    ``extra`` — current density ``rho`` and internal energy ``eint``
    (M37 pack 2: LAW24's dilatancy gates ALPHA on EINT <= 0 and
    RHO < RHO0, LAW81's maximum-dilatancy clamp on RHO; M37 pack 1:
    LAW35's relative volume / air pressure, LAW44's total pressure
    P = K*(rho/rho0 - 1) and LAW70's Itens tension scale all need
    ``rho``; LAW62's CIMAX sound-speed bound divides by the current
    density; LAW40's sound speed too; M40: LAW36 solids use the same
    total pressure as LAW44 — sigeps36.F P = BULK*AMU)."""
    return mat.law in (2, 6, 24, 35, 36, 40, 44, 62, 70, 81)


def solid_update(mat, sig, deps, epsp, dt, extra=None):
    """Dispatch a solid stress update to the material's law.

    Returns (sig, epsp, c): c is the law's current sound speed array or
    None (constant elastic estimate is a bound). ``epsp`` may be None for
    laws without plasticity."""
    if mat.law == 1:
        return law01_elastic.solid_update(mat, sig, deps), epsp, None
    if mat.law == 2:
        sig, epsp = law02_johnson_cook.solid_update(mat, sig, deps, epsp,
                                                    dt, extra)
        return sig, epsp, None
    if mat.law == 36:
        sig, epsp = law36_tabulated.solid_update(mat, sig, deps, epsp, dt,
                                                 extra)
        return sig, epsp, None
    if mat.law == 42:
        return law42_ogden.solid_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 24:
        return law24_concrete.solid_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 81:
        return law81_druckerprager.solid_update(mat, sig, deps, epsp, dt,
                                                extra)
    if mat.law == 83:
        return law83_spotweld.solid_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 62:
        return law62_hypervisco.solid_update(mat, sig, deps, epsp, dt,
                                             extra)
    if mat.law == 0:
        return mat_void.solid_update(mat, sig, deps), epsp, None
    if mat.law == 6:
        sig, epsp, c = law06_hyd_visc.solid_update(mat, sig, deps, epsp, dt, extra)
        return sig, epsp, c
    if mat.law == 999:
        # /MAT/GAS: zero deviator; the pressure and the sound speed come
        # from the attached IDEAL-GAS /EOS through the kernels' EOS block
        return mat_gas.solid_update(mat, sig), epsp, None
    if mat.law == 70:
        sig, c = law70_tabfoam.solid_update(mat, sig, deps, dt, extra)
        return sig, epsp, c
    if mat.law == 35:
        sig, c = law35_kelvinmax.solid_update(mat, sig, deps, dt, extra)
        return sig, epsp, c
    if mat.law == 40:
        sig, c = law40_kelvinmax.solid_update(mat, sig, deps, dt, extra)
        return sig, epsp, c
    if mat.law == 44:
        return law44_cowper.solid_update(mat, sig, deps, epsp, dt, extra)
    raise NotImplementedError(f"material LAW{mat.law} not ported for solids")


def shell_update(mat, sig, deps, epsp, dt, extra=None):
    """Dispatch a plane-stress (shell) update to the material's law."""
    if mat.law == 1:
        return law01_elastic.shell_update(mat, sig, deps), epsp
    if mat.law == 2:
        return law02_johnson_cook.shell_update(mat, sig, deps, epsp, dt,
                                               extra)
    if mat.law == 36:
        return law36_tabulated.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 27:
        return law27_brittle.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 19:
        return law19_fabric.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 0:
        return mat_void.shell_update(mat, sig, deps), epsp
    if mat.law == 44:
        return law44_cowper.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 35:
        return law35_kelvinmax.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 40:
        return law40_kelvinmax.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 62:
        return law62_hypervisco.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 83:
        return law83_spotweld.shell_update(mat, sig, deps, epsp, dt, extra)
    if mat.law == 6:
        return law06_hyd_visc.shell_update(mat, sig, deps, epsp, dt, extra)
    raise NotImplementedError(f"material LAW{mat.law} not ported for shells")


# ----------------------------------------------------------------------------
# Consistent tangents for the implicit solver (M8)
# ----------------------------------------------------------------------------

def solid_tangent(mat, sig, epsp, epsp_incr, extra=None):
    """Dispatch the (n, 6, 6) consistent solid tangent for the implicit
    solve. LAW1 returns the constant elastic C broadcast over the group;
    LAW2 returns the CONSISTENT (algorithmic) elastoplastic tangent of the
    radial return (see law02.consistent_solid_tangent for the derivation);
    LAW36 (M13) the same algebra with the hardening slope from the table's
    local segment (law36.consistent_solid_tangent); LAW42 (M14) the exact
    spectral SPATIAL tangent of the total-form Ogden stress, built from
    the trial deformation gradient the element passes in ``extra["F"]``
    (law42.consistent_solid_tangent — pairs with the assembler's K_geo)."""
    n = sig.shape[0]
    if mat.law == 1:
        import numpy as np
        return np.broadcast_to(law01_elastic.solid_tangent(mat),
                               (n, 6, 6)).copy()
    if mat.law == 2:
        return law02_johnson_cook.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 36:
        return law36_tabulated.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 42:
        if extra is None or "F" not in extra:
            raise NotImplementedError(
                "LAW42 implicit tangent needs the deformation gradient — "
                "supported for the solid kernels (hexa8/tetra4) under "
                "/IMPL/NONLIN only (PORTING_GUIDE M14)")
        return law42_ogden.consistent_solid_tangent(mat, extra["F"])
    if mat.law == 24:
        return law24_concrete.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 35:
        return law35_kelvinmax.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 40:
        return law40_kelvinmax.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 44:
        return law44_cowper.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 81:
        return law81_druckerprager.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 62:
        return law62_hypervisco.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 83:
        return law83_spotweld.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra)
    if mat.law == 6:
        return law06_hyd_visc.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr, extra=extra)
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit solid tangent (LAW1 "
        f"elastic, LAW2, LAW6, LAW24, LAW35, LAW36, LAW40, LAW44, LAW62, LAW81 and LAW83, LAW42 hyperelastic "
        f"are ported; LAW27 is deferred — see PORTING_GUIDE M14)")


def shell_membrane_tangent(mat):
    """(3, 3) plane-stress membrane/bending tangent for the shell implicit
    tangent — the constant elastic matrix (LAW1 shells use it for every
    layer; the shell kernels take this fast path so the M8 results stay
    byte-identical). Elastoplastic shells go through the per-layer
    ``shell_layer_tangent`` instead (M11)."""
    if mat.law == 1:
        return law01_elastic.shell_membrane_tangent(mat)
    if mat.law == 19:
        return law19_fabric.shell_membrane_tangent(mat)
    if mat.law == 44:
        return law44_cowper.shell_membrane_tangent(mat)
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit shell tangent (LAW1 elastic, "
        f"LAW19 fabric and LAW2/44 elastoplastic are ported; see PORTING_GUIDE)")


def shell_layer_tangent(mat, sig, epsp, epsp_incr, extra=None):
    """Dispatch the (n, 3, 3) consistent PLANE-STRESS tangent of one
    through-thickness layer for the implicit shell tangents (M11). LAW1
    broadcasts the elastic matrix; LAW2 returns the CONSISTENT (algorithmic)
    tangent of the Iplas=2 radial projection (see
    law02.consistent_shell_tangent for the derivation); LAW36 (M13) the
    same projection tangent with the table's local hardening slope;
    LAW27 (M15) the damaged fixed-crack unilateral tangent built from the
    layer's trial crack state passed in ``extra`` (the eps27/crk27/ang27/
    dmg27/layfail views — law27.consistent_shell_tangent for the
    per-branch derivation: uncracked / open-frozen / open-growing /
    closed / broken); LAW19 (M524) the orthotropic fabric tangent with
    RCOMP and beta compression scaling."""
    n = sig.shape[0]
    if mat.law == 1:
        import numpy as np
        return np.broadcast_to(law01_elastic.shell_membrane_tangent(mat),
                               (n, 3, 3)).copy()
    if mat.law == 2:
        return law02_johnson_cook.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 36:
        return law36_tabulated.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr)
    if mat.law == 27:
        if extra is None:
            raise NotImplementedError(
                "LAW27 implicit tangent needs the layer crack state "
                "(the shell kernels pass it since M15)")
        return law27_brittle.consistent_shell_tangent(mat, extra)
    if mat.law == 19:
        return law19_fabric.consistent_shell_tangent(mat, extra)
    if mat.law == 44:
        return law44_cowper.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr, extra)
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit shell tangent (LAW1 "
        f"elastic, LAW2, LAW36 and LAW44 elastoplastic, LAW27 brittle cracking, "
        f"LAW19 fabric are ported — see PORTING_GUIDE M15)")
