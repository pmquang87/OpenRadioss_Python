"""
/MAT/GAS — gas thermodynamics (upstream ``ilaw = 999``).

Fortran origin: ``starter/source/materials/mat/matgas/hm_read_matgas.F``
(the four subtypes MASS / MOLE / CSTA / PREDEF) and the airbag machinery
that consumes the stored constants (``engine/source/airbag/airbag1.F``,
``fvinjt8.F``, ``fvtemp.F``).  Upstream a /MAT/GAS is a *thermodynamic
definition* — molecular weight MW and the mass-specific heat polynomial

    cp(T) = CPA + CPB*T + CPC*T^2 + CPD*T^3 + CPE/T^2 + CPF*T^4

(the enthalpy integral the FVM injectors use is
``h(T) = CPA*T + CPB*T^2/2 + CPC*T^3/3 + CPD*T^4/4 - CPE/T + CPF*T^5/5``,
fvinjt8.F) — referenced by monitored-volume injectors, never assigned to
elements.  The port stores exactly those constants (MOLE and PREDEF
coefficients are divided by MW at read time like the original, so params
always carry the MASS-specific form) and exposes the derived ideal-gas
relations:

    r      = R / MW               (specific gas constant)
    cv(T)  = cp(T) - r
    gamma(T) = cp(T) / cv(T)

``R`` is the universal gas constant in *model units* — the port has no
unit system (consistent-units philosophy), so it defaults to the SI
kg-m-s value ``8.314472`` (upstream ``R_IGC``) and can be overridden per
material with ``params['R_igc']`` (e.g. 8314.472 in a g-mm-ms deck).
CSTA gives cp and cv directly and derives MW = R/(cp - cv), like the
original.

Elements — EOS semantics (port extension)
-----------------------------------------
Placed on solid elements, a gas has **no shear strength and an ideal-gas
pressure**: the stress update zeroes the deviator and the pressure comes
from the attached IDEAL-GAS equation of state through the element
kernels' /EOS block (``materials/eos.py`` — the energy-integrated form
whose isentrope is p = p0 (rho/rho0)^gamma), which also supplies the
sound speed c^2 = gamma p / rho for the time-step claim.  The initial
state must come from an ``/EOS/IDEAL-GAS`` card attached to the material
(or the programmatic ``P0 / T0 / RHO0`` params, from which
``resolve_gas`` builds the EOS with gamma evaluated at T0); a /MAT/GAS
card alone has no pressure or density and the Starter refuses to put it
on elements (upstream cannot at all — it has no law-999 element kernel).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.entities import EquationOfState, Material

#: upstream constant_mod.F R_IGC (SI, J/(mol K)) — override per material
#: with params['R_igc'] in non-SI unit systems.
R_IGC_SI = 8.314472

#: predefined gases of hm_read_matgas.F: name -> (MW [kg/mol], molar cp
#: polynomial coefficients CPA..CPF [J/(mol K)]); divided by MW at build
#: time exactly like IMOLE=1.  SI values — a non-SI deck using PREDEF
#: must supply consistent overrides (documented deviation: the original
#: rescales with the /BEGIN unit factors the port does not carry).
PREDEF_GASES = {
    "N2O": (0.04401, (27.67988, 5.1149e-2, -3.0645e-5, 6.8479e-9,
                      -1.5791e5, 0.0)),
    "N2": (0.02801, (26.0920000, 8.2188e-3, -1.9761e-6, 1.5927e-10,
                     4.4434e4, 0.0)),
    "O2": (0.032, (29.659, 6.1373e-3, -1.1865e-6, 9.5780e-11,
                   -2.1966e5, 0.0)),
    "CO2": (0.04401, (24.997350, 5.5187e-2, -3.3691e-5, 7.9484e-9,
                      -1.3664e5, 0.0)),
    "CO": (0.02801, (25.567590, 6.0961e-3, 4.0547e-6, -2.6713e-9,
                     1.3102e5, 0.0)),
    "AR": (0.03995, (20.786, 2.8259e-10, -1.4642e-13, 1.0921e-17,
                     -3.6614e-2, 0.0)),
    "NE": (0.02018, (20.786030, 4.8506e-13, -1.5829e-16, 1.5251e-20,
                     3.1963e-5, 0.0)),
    "HE": (0.004, (20.786030, 4.8506e-13, -1.5829e-16, 1.5251e-20,
                   3.1963e-5, 0.0)),
    "H2O": (0.01802, (30.092, 6.8325e-3, 6.7934e-6, -2.5345e-9,
                      8.2139e4, 0.0)),
    "H2": (0.00202, (33.066178, -1.1363e-2, 1.1433e-5, -2.7729e-9,
                     -1.5856e5, 0.0)),
    "NH3": (0.01703, (19.995630, 4.9771e-2, -1.5376e-5, 1.9212e-9,
                      1.8917e5, 0.0)),
    "H2S": (0.03408, (26.884120, 1.8678e-2, 3.4342e-6, -3.3787e-9,
                      1.3588e5, 0.0)),
    "C6H6": (0.07811, (-36.220000, 4.8475e-1, -3.1570e-4, 7.7620e-8,
                       0.0, 0.0)),
    "AIR": (0.02896, (26.789065, 7.7213e-3, -1.8027e-6, 1.4705e-10,
                      1.1359e4, 0.0)),
}


@dataclass
class GasMaterial(Material):
    """A /MAT/GAS: thermodynamic constants + the ideal-gas relations.

    ``params`` carries MW, CPA..CPF (mass-specific), R_igc; the generic
    elastic properties E/nu are zero (a gas has no elastic skeleton —
    stiffness and sound speed come from the attached EOS)."""

    subtype: str = ""

    #: Starter listings/messages name this law by keyword, not a number
    law_name = "GAS"

    @property
    def mw(self) -> float:
        return self.params["MW"]

    @property
    def r_spec(self) -> float:
        """Specific gas constant r = R / MW."""
        return self.params["R_igc"] / self.params["MW"]

    def cp(self, T: float) -> float:
        """Mass-specific heat at constant pressure at temperature T
        (fvinjt8.F polynomial; the CPE/T^2 term is singular at T = 0)."""
        p = self.params
        cp = (p["CPA"] + p["CPB"] * T + p["CPC"] * T ** 2
              + p["CPD"] * T ** 3 + p["CPF"] * T ** 4)
        if p["CPE"] != 0.0:
            cp += p["CPE"] / (T * T)
        return cp

    def cv(self, T: float) -> float:
        """cv = cp - r (ideal gas / Mayer relation, airbag1.F CVG)."""
        return self.cp(T) - self.r_spec

    def gamma(self, T: float) -> float:
        """Isentropic exponent gamma = cp/cv at temperature T."""
        return self.cp(T) / self.cv(T)


def build_gas(rec) -> GasMaterial:
    """Physics constructor for /MAT/GAS/<MASS|MOLE|CSTA|PREDEF>/id
    (``mat_reader.MAT_PHYSICS_REGISTRY['GAS']``) — hm_read_matgas.F."""
    p = rec.params
    sub = rec.subtype.upper() or "MASS"
    if sub == "PREDEFINED":
        sub = "PREDEF"
    r_igc = float(p.get("R_igc") or R_IGC_SI)

    if sub in ("MASS", "MOLE"):
        mw = float(p.get("MASS") or 0.0)
        if mw <= 0.0:
            raise ValueError(f"/MAT/GAS/{sub}/{rec.id}: molecular weight "
                             f"MW must be > 0 (upstream error 710)")
        coeffs = [float(p.get(k) or 0.0) for k in
                  ("ABG_cpai", "ABG_cpbi", "ABG_cpci", "MAT_D", "MAT_E1",
                   "MAT_F")]
        if sub == "MOLE":
            # molar input: /MW at read time (IMOLE=1), CPF not read
            coeffs = [c / mw for c in coeffs[:5]] + [0.0]
    elif sub == "CSTA":
        cp = float(p.get("MAT_BSAT") or 0.0)
        cv = float(p.get("MAT_RSAT") or 0.0)
        if cp <= 0.0 or cv <= 0.0:
            raise ValueError(f"/MAT/GAS/CSTA/{rec.id}: cp and cv must be "
                             f"> 0 (upstream error 916)")
        if cp <= cv:
            raise ValueError(f"/MAT/GAS/CSTA/{rec.id}: cp must exceed cv "
                             f"(upstream error 917)")
        mw = r_igc / (cp - cv)
        coeffs = [cp, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif sub == "PREDEF":
        name = str(p.get("GAS") or "").strip().upper()
        # upstream matches on the leading characters (N2O before N2...)
        key = next((k for k in sorted(PREDEF_GASES, key=len, reverse=True)
                    if name.startswith(k)), None)
        if key is None:
            raise ValueError(f"/MAT/GAS/PREDEF/{rec.id}: unknown gas "
                             f"'{name}' (upstream error 722)")
        mw, molar = PREDEF_GASES[key]
        coeffs = [c / mw for c in molar]
    else:
        raise ValueError(f"/MAT/GAS/{rec.id}: unknown subtype '{sub}' "
                         f"(MASS, MOLE, CSTA, PREDEF)")

    params = {"MW": mw, "CPA": coeffs[0], "CPB": coeffs[1],
              "CPC": coeffs[2], "CPD": coeffs[3], "CPE": coeffs[4],
              "CPF": coeffs[5], "R_igc": r_igc,
              # no elastic skeleton: stiffness comes from the EOS
              "E": 0.0, "nu": 0.0}
    # programmatic element-path initial state (see resolve_gas)
    for k in ("P0", "T0", "RHO0"):
        if k in p:
            params[k] = float(p[k])
    return GasMaterial(id=rec.id, law=999, rho0=rec.density,
                       title=rec.title, params=params, subtype=sub)


def resolve_gas(mat: GasMaterial, log=None) -> None:
    """Post-deck resolution: build the element-path IDEAL-GAS EOS from
    the programmatic P0/T0/RHO0 params when no /EOS card attached one
    (gamma evaluated at T0 — constant-gamma isentrope, the same
    approximation the /EOS/IDEAL-GAS card makes), and pick up the
    density.  Called from starter resolve_materials."""
    p = mat.params
    if mat.rho0 == 0.0 and p.get("RHO0", 0.0) > 0.0:
        mat.rho0 = p["RHO0"]
    if mat.eos is None and p.get("P0", 0.0) > 0.0:
        t0 = p.get("T0", 295.0)
        g = mat.gamma(t0)
        mat.eos = EquationOfState(
            kind="IDEAL-GAS",
            params={"c0": 0.0, "c1": 0.0, "c2": 0.0, "c3": 0.0,
                    "c4": g - 1.0, "c5": g - 1.0,
                    "e0": p["P0"] / (g - 1.0), "gamma": g},
            rho0=mat.rho0)
    # Starter-side elastic ESTIMATE (contact stiffness, the exact-dt
    # correction factor): map the initial gas bulk modulus K = gamma*P0
    # through a near-incompressible nu so the spurious shear stays
    # negligible (G ~ 1e-3 K).  The RUNTIME stiffness and sound speed
    # come from the EOS block of the element kernels, not from this.
    if mat.eos is not None and p.get("E", 0.0) == 0.0:
        gam = mat.eos.params.get("gamma", 1.4)
        p0 = mat.eos.params["e0"] * (gam - 1.0)
        if p0 > 0.0:
            nu_est = 0.4995
            p["nu"] = nu_est
            p["E"] = 3.0 * (1.0 - 2.0 * nu_est) * gam * p0


def solid_update(mat, sig):
    """Gas on solids: zero deviator, keep the (isotropic) pressure part —
    the /EOS block of the element kernel then REPLACES that pressure by
    the ideal-gas value and claims the c^2 = gamma p/rho sound speed.
    Without an EOS the stress is identically zero (and the Starter has
    already refused the model)."""
    pm = (sig[:, 0] + sig[:, 1] + sig[:, 2]) / 3.0
    sig[:, 0] = pm
    sig[:, 1] = pm
    sig[:, 2] = pm
    sig[:, 3:] = 0.0
    return sig


def _register():
    from ..input.mat_reader import MAT_PHYSICS_REGISTRY
    MAT_PHYSICS_REGISTRY["GAS"] = build_gas


_register()
