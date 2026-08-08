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

from . import (beam_type3, shell_bt4, shell_dkt18, shell_qbat, shell_qeph,  # noqa: F401
               shell_thick16, shell_tri3, solid_heph, solid_hexa8, solid_quad,
               solid_tetra4, spring, truss)

KERNELS = {
    "bricks": solid_hexa8,
    "bricks_heph": solid_heph,
    "quads": solid_quad,
    "tetras": solid_tetra4,
    "shells": shell_bt4,
    "shells_qbat": shell_qbat,
    "shells_qeph": shell_qeph,
    "sh3n": shell_tri3,
    "sh3n_dkt18": shell_dkt18,
    "shel16s": shell_thick16,
    "trusses": truss,
    "springs": spring,
    "beams": beam_type3,
}

#: /PROP/SHELL Ishell -> dedicated element-technology group. Parts whose
#: property carries one of these Ishell values are SPLIT out of the
#: generic "shells" (Belytschko-Tsay) group by the starter's
#: shell-formulation dispatch (starter/initialization.py) and routed to
#: their own kernel; every other Ishell keeps the BT kernel untouched.
#: 12 = QBAT (fully integrated Batoz — cbaforc3.F, M41). The engine
#: starter folds nothing into 12 (hm_read_prop01.F keeps 12 distinct).
#: 22/23/24 = QEPH (physically-stabilized 1-point — czforc3.F, M41); the
#: starter folds 22/23 into 24 (hm_read_prop01.F lines 185-192).
SHELL_ISHELL_GROUPS = {
    12: "shells_qbat",
    22: "shells_qeph",
    23: "shells_qeph",
    24: "shells_qeph",
}

#: /PROP/SHELL Ish3n -> dedicated element-technology group for 3-node shells.
#: 2 = DKT18 (Discrete Kirchhoff Triangle — cdkforc3.F).
SH3N_ISHELL_GROUPS = {
    2: "sh3n_dkt18",
}

#: /PROP/SOLID Isolid -> dedicated element-technology group.
#: 24 = HEPH (physically-stabilized 8-node hexahedral element).
SOLID_ISOLID_GROUPS = {
    24: "bricks_heph",
}
