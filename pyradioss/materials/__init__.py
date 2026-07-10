"""
pyradioss.materials — constitutive laws.

Fortran origin: ``engine/source/materials/mat/matXXX/sigepsXX.F``. In
OpenRadioss every material law is a routine ``SIGEPS<law>`` receiving the
strain-rate components and the old stress of a *group* of elements and
returning the new stress (plus updated internal variables and the sound
speed). The element kernels call it once per integration point per cycle.

The port keeps that contract exactly (vectorized over the group):

    law01_elastic.solid_update / shell_update
    law02_johnson_cook.solid_update / shell_update

Stress storage convention (Voigt, engineering shear):
    solids : (n, 6)  = [xx, yy, zz, xy, yz, zx]
    shells : (n, 3)  = [xx, yy, xy]   (in-plane; transverse shear handled
                                       elastically by the shell kernel)
Strain increments use the same ordering, with *engineering* shear
(gamma = 2*eps) exactly like the Fortran EPSPXX/EPSPXY arguments.

The objective (Jaumann) rotation of the old stress is done by the element
kernel BEFORE calling the law — same split as the Fortran, where SROTA3.F
rotates the stress and SIGEPS only integrates the constitutive rate.
"""

from . import law01_elastic, law02_johnson_cook  # noqa: F401


def solid_update(mat, sig, deps, epsp, dt):
    """Dispatch a solid stress update to the material's law.

    Returns the updated (sig, epsp). ``epsp`` may be None for laws without
    plasticity."""
    if mat.law == 1:
        return law01_elastic.solid_update(mat, sig, deps), epsp
    if mat.law == 2:
        return law02_johnson_cook.solid_update(mat, sig, deps, epsp, dt)
    raise NotImplementedError(f"material LAW{mat.law} not ported")


def shell_update(mat, sig, deps, epsp, dt):
    """Dispatch a plane-stress (shell) update to the material's law."""
    if mat.law == 1:
        return law01_elastic.shell_update(mat, sig, deps), epsp
    if mat.law == 2:
        return law02_johnson_cook.shell_update(mat, sig, deps, epsp, dt)
    raise NotImplementedError(f"material LAW{mat.law} not ported")
