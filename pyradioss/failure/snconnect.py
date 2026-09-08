"""
Cohesive failure model for solid spotwelds (/FAIL/SNCONNECT).

Fortran origin: ``engine/source/materials/fail/snconnect/fail_snconnect.F``

History variables (epsp, pla1, pla2) are stored as companion arrays
alongside the ``dama`` group-state array.  The Fortran keeps them in
restartable element-variable slots; here we store them on the ``fail``
object (a model-level FailureModel that persists across state copies).
The arrays are sized to match ``dama``'s base shape on first contact and
are re-allocated only when the shape changes (a restart with different
element count — never in normal operation).
"""

import numpy as np

_TINY = 1e-20


def _get_state(fail, dama):
    """Retrieve or allocate persistent SNCONNECT state arrays.

    Stored on ``fail._snc`` as a dict ``{'epsp': ..., 'pla1': ...,
    'pla2': ...}`` with flat arrays matching the full (base) ``dama``
    shape.  The returned views are sliced to match the caller's
    ``dama`` view.

    Unlike the old id()-keyed cache, this survives state copies because
    ``fail`` is a model-definition object (not part of the copied
    engine state), and the arrays are re-attached by shape, not by
    Python object identity.
    """
    base = dama.base if dama.base is not None else dama
    n = len(base)

    snc = getattr(fail, "_snc", None)
    if snc is None or snc["epsp"].shape[0] != n:
        # First call, or shape mismatch after restart with changed mesh
        snc = {
            "epsp": np.zeros(n, dtype=np.float64),
            "pla1": np.zeros(n, dtype=np.float64),
            "pla2": np.zeros(n, dtype=np.float64),
        }
        if fail is not None:
            fail._snc = snc

    # Compute the slice corresponding to this dama view
    if dama.base is not None:
        offset = (
            dama.__array_interface__["data"][0]
            - base.__array_interface__["data"][0]
        ) // dama.itemsize
        sl = slice(offset, offset + len(dama))
    else:
        sl = slice(None)

    return snc["epsp"][sl], snc["pla1"][sl], snc["pla2"][sl]


def solid_step(fail, sig, d_epsp, deps, dt, dama, tstar=None):
    """3-D damage step for SNCONNECT."""
    p = fail.params
    a2, b2 = p.get("a2", 0.0), p.get("b2", 1.0)
    a3, b3 = p.get("a3", 0.0), p.get("b3", 1.0)
    isym = p.get("isym", 0)

    epsp, pla1, pla2 = _get_state(fail, dama)

    # accumulate local epsp
    epsp += np.maximum(d_epsp, 0.0)

    # normal and shear stresses
    signzz = sig[:, 2]
    signyz = sig[:, 4]
    signzx = sig[:, 5]

    ssym = 0.0
    svmn = np.abs(signzz)
    svmt = np.sqrt(signyz ** 2 + signzx ** 2)
    phi = np.arctan2(svmn, np.maximum(svmt, _TINY))
    sphi = np.sin(phi)
    cphi = np.cos(phi)

    # Phase 1: No damage yet (pla1 == 0)
    mask1 = pla1 == 0.0
    if np.any(mask1):
        t1 = np.where(
            (isym == 1) & (signzz <= 0.0), 0.0, sphi / (1.0 - a2 * ssym)
        )
        t2 = cphi

        ttn = t1[mask1] * epsp[mask1]
        tts = t2[mask1] * epsp[mask1]
        fct = (ttn ** b2 + tts ** b2) ** (1.0 / b2)

        start_dmg = fct > 1.0
        if np.any(start_dmg):
            idx = np.where(mask1)[0][start_dmg]
            pla1[idx] = (t1[idx] ** b2 + t2[idx] ** b2) ** (-1.0 / b2)

            t1_3 = np.where(
                (isym == 1) & (signzz[idx] <= 0.0),
                0.0,
                sphi[idx] / (1.0 - a3 * ssym),
            )
            t2_3 = cphi[idx]

            pla2[idx] = (t1_3 ** b3 + t2_3 ** b3) ** (-1.0 / b3)
            d_val = (epsp[idx] - pla1[idx]) / np.maximum(
                _TINY, pla2[idx] - pla1[idx]
            )
            dama[idx] = np.minimum(d_val, 1.0)

    # Phase 2: Damage is progressing (pla1 > 0)
    mask2 = pla1 > 0.0
    if np.any(mask2):
        t1_3 = np.where(
            (isym == 1) & (signzz[mask2] <= 0.0),
            0.0,
            sphi[mask2] / (1.0 - a3 * ssym),
        )
        t2_3 = cphi[mask2]

        ttn3 = t1_3 * epsp[mask2]
        tts3 = t2_3 * epsp[mask2]
        fct3 = (ttn3 ** b3 + tts3 ** b3) ** (1.0 / b3)

        pla2[mask2] = (t1_3 ** b3 + t2_3 ** b3) ** (-1.0 / b3)
        d_val = (epsp[mask2] - pla1[mask2]) / np.maximum(
            _TINY, pla2[mask2] - pla1[mask2]
        )
        dama[mask2] = np.minimum(d_val, 1.0)

        rupture = fct3 > 1.0
        if np.any(rupture):
            idx = np.where(mask2)[0][rupture]
            dama[idx] = 1.0

    return dama >= 1.0
