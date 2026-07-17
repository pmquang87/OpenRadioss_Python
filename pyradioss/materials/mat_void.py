"""
/MAT/VOID — law 0, the void (dummy) material.

Fortran origin: ``starter/source/materials/mat/mat000/hm_read_mat00.F``.
There is NO engine kernel for law 0 — that is the point: a void element
carries **mass and geometry but no stress**, ever.  The material exists so
that dummy parts (contact skins over a rigid body, airbag reference
geometry, seat foams replaced by rigids in a sled run...) keep their
inertia and their contact/time-step stiffness *estimate* without adding
any structural force.

The card still reads a Young modulus and a Poisson ratio: the reference
Starter uses them ONLY for derived quantities — the interface stiffness
and the element time-step estimate (``hm_read_mat00`` computes
``SSP = sqrt(E/rho)`` and stores the elastic constants in PM(20)/PM(21),
never a yield or a stress).  The port keeps that split exactly:

* :func:`solid_update` / :func:`shell_update` zero the stress uncondi-
  tionally (they also erase whatever the kernel's Jaumann pre-rotation
  produced — a void element cannot accumulate ANY stress state);
* the sound speed is left to the element kernel's elastic estimate from
  the card's E/nu (``sqrt((K + 4G/3)/rho)`` for solids, the plane-stress
  modulus for shells), which is exactly what those constants are for.
  With ``E = 0`` (legal — some decks give a bare density) the element
  claims no time step at all, again like the original.

Element families: the reference accepts VOID on every family
(``SOLID_ISOTROPIC / SHELL_ISOTROPIC / SPRING / BEAM / TRUSS / SPH`` in
``hm_read_mat00``); the port wires it into the families whose kernels
dispatch through ``pyradioss.materials`` — solids and shells (the corpus
uses it on contact-skin shells).  Trusses/beams/springs keep their own
direct-E kernels and refuse law 0 in the Starter check.
"""

from __future__ import annotations

from ..model.entities import Material


def build_void(rec) -> Material:
    """Physics constructor for the cfg-parsed /MAT/VOID record
    (``mat_reader.MAT_PHYSICS_REGISTRY['VOID']``).

    cfg attributes (``matl_void0.cfg``): MAT_RHO, MAT_E, nu.  E and nu
    are stiffness ESTIMATES (contact, time step) — never a stress.
    """
    p = rec.params
    e = float(p.get("MAT_E") or 0.0)
    nu = float(p.get("nu") or 0.0)
    if e < 0.0:
        raise ValueError(f"/MAT/VOID/{rec.id}: negative Young modulus "
                         f"E={e:g}")
    if not (-1.0 < nu < 0.5):
        raise ValueError(f"/MAT/VOID/{rec.id}: Poisson ratio nu={nu:g} "
                         f"outside (-1, 0.5)")
    return Material(id=rec.id, law=0, rho0=rec.density, title=rec.title,
                    params={"E": e, "nu": nu})


def solid_update(mat, sig, deps):
    """Void solids: no stress, ever (mass + contact stiffness only)."""
    sig[:] = 0.0
    return sig


def shell_update(mat, sig, deps):
    """Void shells: no stress, ever."""
    sig[:] = 0.0
    return sig


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["VOID"] = build_void
    MAT_PHYSICS_REGISTRY["LAW0"] = build_void


_register()
