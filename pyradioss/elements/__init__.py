"""
pyradioss.elements — element kernels.

Fortran origin: ``engine/source/elements/`` — one sub-directory per element
family (``solid/solide`` 8-node bricks, ``shell/coque`` 4-node shells,
``truss``, ``spring`` ...). In the Engine's main loop these are called from
the force driver (ALEMAIN/FORINT) for one *group* of same-type elements at
a time; the group loop is `pyradioss.engine.engine`.

Every kernel module exposes the same two entry points:

``init_group(group, model, log)``
    Called once by the Starter: builds the per-element ``state`` arrays
    (the *element buffer*, Fortran ``ELBUF_TAB``: stress, plastic strain,
    volumes...), computes element masses and returns the nodal mass (and
    inertia) contributions used to build the lumped mass matrix.

``forces(group, x, v, vr, dt, fint, mint)``
    Called every cycle by the Engine: computes internal + hourglass forces,
    scatters them into the global arrays ``fint`` (translations) and
    ``mint`` (rotations, shells only), updates the element state (stresses,
    energies) and returns the per-element critical time step array.

Sign convention: kernels ACCUMULATE the internal force with a MINUS sign
into ``fint`` — i.e. after all kernels ran, ``fint`` is the net force such
that  a = (fext + fint) / m  (this matches the Fortran A(3,*) accumulation
where internal forces enter negated).
"""

from . import (beam_type3, shell_bt4, shell_tri3, solid_hexa8,  # noqa: F401
               solid_tetra4, spring, truss)

KERNELS = {
    "bricks": solid_hexa8,
    "tetras": solid_tetra4,
    "shells": shell_bt4,
    "sh3n": shell_tri3,
    "trusses": truss,
    "springs": spring,
    "beams": beam_type3,
}
