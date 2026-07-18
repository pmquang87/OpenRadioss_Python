"""
Unit systems: /BEGIN work units + local /UNIT blocks (M37).

Fortran origin
--------------
``starter/source/general_controls/computation/unit_code.F`` parses one
unit *field* — either a plain number (the SI factor itself) or a code
``<prefix><base>`` where the base letter is ``g``/``m``/``s`` (gram,
meter, second) and the prefix a metric multiplier (``m`` = 1e-3, ``k`` =
1e3, ``M`` = 1e6, ...).  The SI mass unit being the kg while the base
letter is the gram, MASS codes carry an extra 1e-3 (``UNIT_CODE``:
``IF (KEY(1:4) == 'MASS') FAC=FAC*EM03``) — so ``g`` -> 1e-3 kg, ``Mg``
-> 1e3 kg (the tonne), ``mm`` -> 1e-3 m, ``ms`` -> 1e-3 s.

``hm_read_unit.F`` reads the /BEGIN input/work unit cards and every
``/UNIT/<id>`` block into UNITAB; ``hm_get_floatv.F`` then converts each
quantity by ``FAC_M^a * FAC_L^b * FAC_T^c`` with (a, b, c) the
quantity's mass/length/time dimension from the cfg files.  The **net
effect**, verified against the real Windows starter on the RD-E-2601
``main_TEST4`` deck (/MAT in a local g-mm-ms system, work units
Mg-mm-s): every value of a block that references ``/UNIT/<uid>`` is
multiplied by

    (fac_m_local / fac_m_work)^a  *  (fac_l_local / fac_l_work)^b
                                  *  (fac_t_local / fac_t_work)^c

i.e. converted INTO the /BEGIN *work* unit system — the density 0.0028
g/mm3 lands in the listing as 2.8000E-09 Mg/mm3 while the modulus 71000
(dimension M L^-1 T^-2, whose g-mm-ms -> Mg-mm-s ratio is exactly 1)
stays 71000.

Port scope (documented deviation)
---------------------------------
The port converts the quantities it actually reads — the named /MAT law
parameters and /PROP geometry parameters, via per-name dimension tables
below — and **warns loudly** about any non-default parameter it cannot
classify (generic cfg-parsed material laws) and about any OTHER keyword
carrying a local unit id, instead of converting silently wrong or, as
before M37, ignoring /UNIT altogether.  When the local system equals the
work system (every ratio 1.0) conversion is exact by construction and no
warning is emitted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common.messages import MessageLog

# ----------------------------------------------------------------------------
# unit_code.F: the metric prefix table (SI factors)
# ----------------------------------------------------------------------------

_PREFIXES: Dict[str, float] = {
    "y": 1e-24, "z": 1e-21, "a": 1e-18, "f": 1e-15, "p": 1e-12,
    "n": 1e-9, "mu": 1e-6, "u": 1e-6, "m": 1e-3, "c": 1e-2, "d": 1e-1,
    "": 1.0, "da": 1e1, "h": 1e2, "k": 1e3, "K": 1e3, "M": 1e6,
    "G": 1e9, "T": 1e12, "P": 1e15, "E": 1e18, "Z": 1e21, "Y": 1e24,
}

#: the base letter of each unit kind (unit_code.F rejects anything else)
_BASE = {"MASS": "g", "LENGTH": "m", "TIME": "s"}


def parse_unit_code(field: str, kind: str) -> float:
    """One unit field -> SI factor (unit_code.F).

    ``field`` is either a plain number (the factor itself, in SI) or a
    code ``<prefix><base>`` with base g/m/s.  ``kind`` is 'MASS',
    'LENGTH' or 'TIME'; MASS codes get the g -> kg 1e-3 (numeric fields
    do NOT — they are already the SI factor).  Raises ValueError on an
    unknown code, mirroring the reference MSGID 573 error.
    """
    tok = field.strip()
    if not tok:
        raise ValueError(f"empty {kind} unit field")
    try:
        return float(tok.replace("D", "E").replace("d", "e"))
    except ValueError:
        pass
    base = _BASE[kind]
    if not tok.endswith(base):
        raise ValueError(f"unknown {kind} unit code '{tok}' "
                         f"(expected ...{base} or a number)")
    prefix = tok[:-1]
    # the prefix match is case-sensitive except k/K (unit_code.F lists
    # both); 'mu' is the two-letter micro spelling
    if prefix not in _PREFIXES:
        raise ValueError(f"unknown metric prefix '{prefix}' in {kind} "
                         f"unit code '{tok}'")
    fac = _PREFIXES[prefix]
    if kind == "MASS":
        fac *= 1e-3                       # the SI mass unit is kg, not g
    return fac


def parse_unit_triple(fields: List[str], source: str,
                      log: MessageLog) -> Optional[Tuple[float, float, float]]:
    """(MUNIT, LUNIT, TUNIT) fields -> (fac_m, fac_l, fac_t) SI factors,
    or None (with errors logged) when any field fails to parse."""
    out = []
    ok = True
    for field, kind in zip(fields, ("MASS", "LENGTH", "TIME")):
        try:
            out.append(parse_unit_code(field, kind))
        except (ValueError, KeyError) as exc:
            log.error(f"unit card: {exc}", source)
            ok = False
    return tuple(out) if ok else None


# ----------------------------------------------------------------------------
# Dimension tables: (mass, length, time) exponents of the port-read fields
# ----------------------------------------------------------------------------

_DENSITY = (1, -3, 0)
_STRESS = (1, -1, -2)     # pressure / modulus / energy per volume
_RATE = (0, 0, -1)
_NONE = (0, 0, 0)

#: /MAT params of the natively-ported laws (1/2/27/36/42), by name.
#: Anything NOT listed here and non-zero is reported unconverted.
MAT_PARAM_DIMS: Dict[str, Tuple[int, int, int]] = {
    "E": _STRESS, "nu": _NONE,
    # LAW2 Johnson-Cook
    "A": _STRESS, "B": _STRESS, "n": _NONE, "sig_max": _STRESS,
    "eps_p_max": _NONE, "c": _NONE, "eps_dot_0": _RATE,
    # LAW2 thermal card: temperatures are NOT covered by the M/L/T unit
    # system (no temperature unit in /UNIT) — left as written, like the
    # reference; rho_cp = energy/(volume.K) converts as a stress per K.
    "mT": _NONE, "T_melt": _NONE, "T_i": _NONE, "rho_cp": _STRESS,
    # LAW27 damage strains (dimensionless)
    "eps_t1": _NONE, "eps_m1": _NONE, "dmax1": _NONE, "eps_f1": _NONE,
    "eps_t2": _NONE, "eps_m2": _NONE, "dmax2": _NONE, "eps_f2": _NONE,
    # LAW36 (curves are referenced by id; the rates are strain rates)
    "rates": _RATE,
    # LAW42 Ogden
    "mu": _STRESS, "alpha": _NONE,
}

#: /MAT params that are ids/lists-of-ids or resolve-time products —
#: never converted, never reported.  LAW36 "yfac" (per-curve Fscale_i,
#: M40) is skipped too: it scales the /FUNCT ordinates, whose physical
#: unit conversion is carried by the function itself.
_MAT_SKIP = {"funct_ids", "curve_x", "curve_y", "curve_s", "yfac"}

#: read-time DEFAULT sentinels: the parsers inject these when the card
#: field is blank/zero (e.g. LAW2 eps_dot_0 = 0 -> 1.0, sig_max = 0 ->
#: 1e30).  The reference injects its defaults AFTER unit conversion (a
#: blank field converts as 0), so an injected default is already a
#: work-unit value and must NOT be re-scaled here.
_MAT_DEFAULT_SENTINELS = {"eps_dot_0": 1.0, "sig_max": 1e30,
                          "eps_p_max": 1e30, "eps_f1": 1e30,
                          "eps_f2": 1e30}

#: /PROP params by property type.
PROP_PARAM_DIMS: Dict[int, Dict[str, Tuple[int, int, int]]] = {
    1: {"thick": (0, 1, 0), "nip": _NONE,
        "hm": _NONE, "hf": _NONE, "hr": _NONE},
    2: {"area": (0, 2, 0)},
    3: {"area": (0, 2, 0), "iyy": (0, 4, 0), "izz": (0, 4, 0),
        "ixx": (0, 4, 0)},
    4: {"mass": (1, 0, 0), "k": (1, 0, -2), "c": (1, 0, -1)},
    14: {"qa": _NONE, "qb": _NONE, "h": _NONE},
}


def _ratios(local: Tuple[float, float, float],
            work: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """local -> work conversion ratios per base quantity."""
    return (local[0] / work[0], local[1] / work[1], local[2] / work[2])


def _factor(r: Tuple[float, float, float],
            dim: Tuple[int, int, int]) -> float:
    return r[0] ** dim[0] * r[1] ** dim[1] * r[2] ** dim[2]


def _convert_params(params: dict, dims: Dict[str, Tuple[int, int, int]],
                    skip: set, r: Tuple[float, float, float],
                    sentinels: Optional[dict] = None):
    """Convert the named entries of ``params`` in place; return the list
    of non-default names the table does not classify (unconverted)."""
    unknown = []
    for name, val in list(params.items()):
        if name in skip:
            continue
        if sentinels and sentinels.get(name) == val:
            continue                 # read-time default: already work units
        dim = dims.get(name)
        if dim is None:
            # report only values that actually carry information
            try:
                nonzero = bool(np.any(np.asarray(val, dtype=float) != 0.0))
            except (TypeError, ValueError):
                nonzero = True
            if nonzero:
                unknown.append(name)
            continue
        if dim == _NONE:
            continue
        f = _factor(r, dim)
        if f == 1.0:
            continue
        if isinstance(val, (list, tuple)):
            params[name] = type(val)(v * f for v in val)
        elif isinstance(val, np.ndarray):
            params[name] = val * f
        else:
            params[name] = val * f
    return unknown


def apply_unit_conversions(model, log: MessageLog) -> None:
    """Convert every block that referenced a LOCAL /UNIT into the /BEGIN
    work unit system (the resolve-time half of the M37 unit machinery;
    runs BEFORE resolve_materials so LAW36 curve copies etc. see the
    converted parameters).

    Follows the reference semantics established above: value_work =
    value_local * rM^a * rL^b * rT^c with r* the local/work SI-factor
    ratios.  Unknown /UNIT ids, missing /BEGIN work units and
    unclassifiable parameters are ERRORS/WARNINGS — nothing is silently
    ignored any more (the pre-M37 behaviour this replaces).
    """
    # /BEGIN input units different from work units: the reference reads
    # uid-0 blocks in the INPUT system and converts to WORK — the port
    # reads everything as written, so a differing pair deviates. Warn.
    if (model.unit_input is not None and model.unit_work is not None
            and tuple(model.unit_input) != tuple(model.unit_work)):
        log.warning("/BEGIN: input unit system differs from the work "
                    "unit system — the port reads unit-less blocks AS "
                    "WRITTEN (work units); quantities meant in input "
                    "units are NOT converted", "UNIT CHECK")

    if not model.raw_unit_refs:
        return
    if model.unit_work is None:
        log.error("blocks reference local /UNIT systems but /BEGIN "
                  "declares no work unit system — units cannot be "
                  "converted", "UNIT CHECK")
        return

    for key0, user_id, unit_id, source in model.raw_unit_refs:
        local = model.units.get(unit_id)
        if local is None:
            log.error(f"/{key0}/{user_id}: unknown unit system "
                      f"/UNIT/{unit_id}", source)
            continue
        r = _ratios(local, model.unit_work)
        identity = (r == (1.0, 1.0, 1.0))

        if key0 == "MAT":
            mat = model.materials.get(user_id)
            if mat is None:
                continue
            if identity:
                continue
            mat.rho0 *= _factor(r, _DENSITY)
            unknown = _convert_params(mat.params, MAT_PARAM_DIMS,
                                      _MAT_SKIP, r,
                                      _MAT_DEFAULT_SENTINELS)
            if unknown:
                log.warning(f"/MAT .../{user_id}: local /UNIT/{unit_id} "
                            f"conversion applied to the ported fields, "
                            f"but these parameters have no dimension "
                            f"entry and were left AS WRITTEN: "
                            f"{', '.join(sorted(unknown))}", source)
        elif key0 == "PROP":
            prop = model.properties.get(user_id)
            if prop is None:
                continue
            if identity:
                continue
            dims = PROP_PARAM_DIMS.get(prop.type)
            if dims is None:
                log.warning(f"/PROP .../{user_id}: local /UNIT/{unit_id} "
                            f"differs from the work units but the "
                            f"property type has no dimension table — "
                            f"values left AS WRITTEN", source)
                continue
            unknown = _convert_params(prop.params, dims, set(), r)
            if unknown:
                log.warning(f"/PROP .../{user_id}: parameters without a "
                            f"dimension entry left AS WRITTEN: "
                            f"{', '.join(sorted(unknown))}", source)
        else:
            # /FAIL, /EOS, /FUNCT, loads ... — accepted only when the
            # local system IS the work system (conversion factor 1);
            # otherwise the values would be silently wrong: warn LOUDLY.
            if not identity:
                log.warning(f"/{key0} .../{user_id}: references local "
                            f"/UNIT/{unit_id} whose system differs from "
                            f"the work units — /{key0} quantities are "
                            f"NOT unit-converted by the port, values "
                            f"left AS WRITTEN", source)
