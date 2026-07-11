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

from . import (eos, law01_elastic, law02_johnson_cook,  # noqa: F401
               law27_brittle, law36_tabulated, law42_ogden)


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
    if mat.law == 2 and "mT" in mat.params:
        # adiabatic temperature RISE above T_i (M6 thermal terms)
        shapes["temp"] = (nip,) if nip is not None else ()
    return shapes


def needs_defgrad(mat) -> bool:
    """True if the law is total-strain and needs F from the kernel."""
    return mat.law == 42


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
        sig, epsp = law36_tabulated.solid_update(mat, sig, deps, epsp, dt)
        return sig, epsp, None
    if mat.law == 42:
        return law42_ogden.solid_update(mat, sig, deps, epsp, dt, extra)
    raise NotImplementedError(f"material LAW{mat.law} not ported for solids")


def shell_update(mat, sig, deps, epsp, dt, extra=None):
    """Dispatch a plane-stress (shell) update to the material's law."""
    if mat.law == 1:
        return law01_elastic.shell_update(mat, sig, deps), epsp
    if mat.law == 2:
        return law02_johnson_cook.shell_update(mat, sig, deps, epsp, dt,
                                               extra)
    if mat.law == 36:
        return law36_tabulated.shell_update(mat, sig, deps, epsp, dt)
    if mat.law == 27:
        return law27_brittle.shell_update(mat, sig, deps, epsp, dt, extra)
    raise NotImplementedError(f"material LAW{mat.law} not ported for shells")


# ----------------------------------------------------------------------------
# Consistent tangents for the implicit solver (M8)
# ----------------------------------------------------------------------------

def solid_tangent(mat, sig, epsp, epsp_incr):
    """Dispatch the (n, 6, 6) consistent solid tangent for the implicit
    solve. LAW1 returns the constant elastic C broadcast over the group;
    LAW2 returns the CONSISTENT (algorithmic) elastoplastic tangent of the
    radial return (see law02.consistent_solid_tangent for the derivation)."""
    n = sig.shape[0]
    if mat.law == 1:
        import numpy as np
        return np.broadcast_to(law01_elastic.solid_tangent(mat),
                               (n, 6, 6)).copy()
    if mat.law == 2:
        return law02_johnson_cook.consistent_solid_tangent(
            mat, sig, epsp, epsp_incr)
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit solid tangent (M8 supports "
        f"LAW1 elastic and LAW2 elastoplastic)")


def shell_membrane_tangent(mat):
    """(3, 3) plane-stress membrane/bending tangent for the shell implicit
    tangent — the constant elastic matrix (LAW1 shells use it for every
    layer; the shell kernels take this fast path so the M8 results stay
    byte-identical). Elastoplastic shells go through the per-layer
    ``shell_layer_tangent`` instead (M11)."""
    if mat.law == 1:
        return law01_elastic.shell_membrane_tangent(mat)
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit shell tangent (LAW1 elastic "
        f"and LAW2 elastoplastic are ported; see PORTING_GUIDE)")


def shell_layer_tangent(mat, sig, epsp, epsp_incr):
    """Dispatch the (n, 3, 3) consistent PLANE-STRESS tangent of one
    through-thickness layer for the implicit shell tangents (M11). LAW1
    broadcasts the elastic matrix; LAW2 returns the CONSISTENT (algorithmic)
    tangent of the Iplas=2 radial projection (see
    law02.consistent_shell_tangent for the derivation)."""
    n = sig.shape[0]
    if mat.law == 1:
        import numpy as np
        return np.broadcast_to(law01_elastic.shell_membrane_tangent(mat),
                               (n, 3, 3)).copy()
    if mat.law == 2:
        return law02_johnson_cook.consistent_shell_tangent(
            mat, sig, epsp, epsp_incr)
    raise NotImplementedError(
        f"material LAW{mat.law} has no implicit shell tangent (LAW1 elastic "
        f"and LAW2 elastoplastic are ported; see PORTING_GUIDE)")
