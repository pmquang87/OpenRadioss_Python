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
    shell_step(fail, sig, d_epsp, deps, dt, dama, tstar)  -> broken mask

with sig/deps the (m, 6) or (m, 3) slice arrays of the group, d_epsp the
plastic-strain increment of this cycle, dama the persistent damage
array (in-place) and tstar the homologous temperature of the points
(M6, None for materials without the thermal card — only /FAIL/JOHNSON's
D5 term reads it). All vectorized over the element slice.
"""

from . import biquad, johnson  # noqa: F401


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance the damage of a solid slice; returns the broken mask."""
    if fail.type == "JOHNSON":
        return johnson.solid_step(fail, sig, d_epsp, deps, dt, dama, tstar)
    if fail.type == "BIQUAD":
        return biquad.solid_step(fail, sig, d_epsp, deps, dt, dama)
    raise NotImplementedError(f"/FAIL/{fail.type} not ported")


def shell_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """Advance the damage of one shell layer; returns the broken mask."""
    if fail.type == "JOHNSON":
        return johnson.shell_step(fail, sig, d_epsp, deps, dt, dama, tstar)
    if fail.type == "BIQUAD":
        return biquad.shell_step(fail, sig, d_epsp, deps, dt, dama)
    raise NotImplementedError(f"/FAIL/{fail.type} not ported")
