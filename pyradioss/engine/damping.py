"""
/DAMP — Rayleigh mass damping (M6).

Fortran origin: ``engine/source/assembly/damping.F``, subroutine
``damping51`` (lines 100–170 for the mass-proportional branch in global
coordinates, lines 175–228 for the rotational-DOF branch).  The Fortran
uses an implicit-trapezoidal acceleration correction::

    OMEGA = 1/(1 + 0.5*DAMP_A*DT1)
    DA = (A - DAMP_A*V - BETASDT*(A - A_old)) * OMEGA - A
    A = A + DA

with energy booked as  DW += m * DA * (V + 0.5*A*DT1) * DT12.  The port
replaces this with the exact integrating factor (see below), which gives
identical physics (both are first-order-accurate mass damping) but is
unconditionally stable and books energy exactly from the KE identity.

A mass-proportional damping force  f_i = -alpha * m_i * v_i  acts on
the nodes of a group, optionally windowed in time (Tstart/Tstop). The
stiffness-proportional (beta) branch of full Rayleigh damping needs K*v
products the explicit port does not assemble — not ported (documented).

Implementation — the exact integrating factor
---------------------------------------------
Mass damping decouples per node:  m dv/dt = -alpha m v  has the exact
solution  v(t+dt) = v(t) * exp(-alpha dt).  Instead of adding the force
-alpha m v^{n-1/2} explicitly (which is half-step LAGGED — exactly the
phase defect the M6 bulk-viscosity investigation showed feeding phantom
energy channels at marginal time steps, and which would also erode the
stable dt for large alpha), the port applies the integrating factor to
the freshly updated velocities:

    v <- v * exp(-alpha dt)          (translations, and rotations where
                                      the node carries inertia)

This is unconditionally stable for ANY alpha*dt, exact for the damping
ODE, claims no time step, and its dissipation is booked EXACTLY from the
kinetic-energy identity:

    E_damp += sum 1/2 m (|v_before|^2 - |v_after|^2)

so the constraint of the milestone — "book its dissipation, the balance
must stay honest" — holds to round-off by construction. The dissipation
goes to its own ledger (DE in the T01, 'DAMPING DISSIPATION' in the
listing) and enters the energy balance.

The damper runs right after the acceleration update, BEFORE the
kinematic conditions: a /BCS, /IMPVEL or rigid-body enforcement then
overwrites the damped velocity like it overwrites the accelerated one
(damping never fights a kinematic condition; on such nodes it
effectively does nothing — the booking uses the velocities it actually
changed, so no phantom energy is booked either way... with one bounded
exception: a node corrected by a WALL in the same cycle books the
damping share of its arrest twice, once here and once in the wall's
injection identity; the overlap is one force phase of the few cycles a
node spends being arrested and vanishes with dt).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..model.model import Model


class Dampers:
    """Engine-side /DAMP list."""

    def __init__(self, model: Model, log):
        self.items = []          # (idx, alpha, tstart, tstop)
        self.all_idx = np.zeros(0, dtype=np.int64)   # union, for the
        # attribution correction the Engine books (see engine step 4b)
        # Build rigid-body slave set once — Fortran damping.F:150 excludes
        # nodes with TAGSLV_RBY != 0 because rigid_body.advance() will
        # overwrite their velocities, making any damping booking phantom.
        rb_slaves = set()
        for rb in getattr(model, 'rbodies', []):
            slaves = getattr(rb, 'slaves', None)
            if slaves is not None:
                rb_slaves.update(int(s) for s in slaves)
        if rb_slaves:
            rb_arr = np.array(sorted(rb_slaves), dtype=np.int64)
        else:
            rb_arr = np.zeros(0, dtype=np.int64)
        for dp in model.damps:
            g = model.node_groups.get(dp.grnod_id)
            if g is None or g.node_idx is None or g.node_idx.size == 0:
                log.error(f"/DAMP/{dp.id}: node group {dp.grnod_id} is "
                          f"missing or empty", "DAMP CHECK")
                continue
            idx = g.node_idx[model.mass[g.node_idx] < 1e29]
            # Exclude rigid-body slave nodes (damping.F:150 TAGSLV_RBY check)
            if len(rb_arr):
                idx = idx[~np.isin(idx, rb_arr)]
            if len(idx) == 0:
                continue
            self.items.append((idx, dp.alpha, dp.tstart, dp.tstop))
            self.all_idx = np.unique(np.concatenate([self.all_idx, idx]))
            log.info(f"     /DAMP/{dp.id}: ALPHA = {dp.alpha:12.5E} ON "
                     f"{len(idx)} NODE(S)"
                     + (f", ACTIVE {dp.tstart:g} TO {dp.tstop:g}"
                        if dp.tstop < 1e30 or dp.tstart > 0 else ""))

    def __len__(self):
        return len(self.items)

    # ------------------------------------------------------------------
    def apply(self, t: float, dt: float, v: np.ndarray, vr: np.ndarray,
              mass: np.ndarray, inertia: np.ndarray) -> float:
        """Damp the group velocities (exact integrating factor) and
        return the kinetic energy removed this cycle.

        Fortran: ``damping.F`` lines 129–171 (translational, ISK<=1, global
        coords) and lines 175–228 (rotational DOFs, IRODDL branch).

        The Fortran applies  DA = (A - alpha*V) * omega - A  per axis and
        books  DW += m * DA * (V + 0.5*A*DT1) * DT12.  The port replaces
        this with the exact ODE solution  v *= exp(-alpha*dt)  and books
        the KE drop  de += 0.5 * m * (|v_old|^2 - |v_new|^2)  — identical
        to roundoff for the damping ODE and unconditionally stable.
        """
        de = 0.0
        for idx, alpha, tstart, tstop in self.items:
            if dt <= 0.0 or t < tstart or t > tstop:
                continue
            fac = np.exp(-alpha * dt)
            de += float(0.5 * (1.0 - fac * fac)
                        * (mass[idx, None] * v[idx] ** 2).sum())
            v[idx] *= fac
            has_in = inertia[idx] > 0.0
            if np.any(has_in):
                ir = idx[has_in]
                de += float(0.5 * (1.0 - fac * fac)
                            * (inertia[ir, None] * vr[ir] ** 2).sum())
                vr[ir] *= fac
        return de


def butterworth_filter(dt: float, freq: float, x2: float, x1: float, x: float,
                       fx2: float, fx1: float) -> float:
    """Second-order Butterworth low-pass digital filter.

    Upstream Fortran reference:
    ``engine/source/tools/univ/butterworth.F``
    """
    dt2 = dt / 2.0
    wd = np.sqrt(2.0) * np.pi * freq * (5.0 / 3.0)
    cos_val = np.cos(wd * dt2)
    if abs(cos_val) < 1e-15:
        wa = 1e15
    else:
        wa = np.sin(wd * dt2) / cos_val
    wa2 = wa * wa
    c1 = 1.0 + np.sqrt(2.0) * wa + wa2
    if abs(c1) < 1e-20:
        return float(x)
    a0 = wa2 / c1
    a1 = 2.0 * a0
    a2 = a0
    b1 = -2.0 * (wa2 - 1.0) / c1
    b2 = (-1.0 + np.sqrt(2.0) * wa - wa2) / c1
    fx = a0 * x + a1 * x1 + a2 * x2 + b1 * fx1 + b2 * fx2
    return float(fx)


class DynamicRelaxation:
    """Dynamic Relaxation solver controls (/DYREL, /KEREL, /ADYREL, /RELAX).

    Fortran origin: ``engine/source/general_controls/damping/static.F`` (subroutines
    ``STATIC``, ``E_PERIOD``, and ``ENER_W0`` called from ``resol.F:7321`` and ``resol.F:8294``).

    Supports:
    1. /DYREL (ISTAT=1): Viscous relaxation damping on velocities matching static.F:
           omega = betate * dt
           omega2 = max(0.0, (1.0 - 2.0 * omega) ** 2)
           v *= (1.0 - 2.0 * omega)
       Dissipated energy is booked exactly into DE (damping dissipation).
    2. /KEREL (ISTAT=2): Kinetic energy peak zeroing. Monitors the system kinetic
       energy on the relaxation node group; when a peak is detected (ke < ke_prev),
       all velocities on the group are reset to zero, freezing the structure at
       the static equilibrium point.
    3. /ADYREL (ISTAT=3): Adaptive dynamic relaxation. Uses a 2nd-order Butterworth
       low-pass filter on kinetic and internal energy to estimate the fundamental
       oscillation period and automatically adapts betate.
    4. /RELAX: Canonical Radioss relaxation directive.
    """

    def __init__(self, model: Model, controls, log=None):
        self.model = model
        self.controls = controls
        self.log = log

        self.dyrel_active = bool(getattr(controls, "dyrel_active", False))
        self.dyrel_beta = float(getattr(controls, "dyrel_beta", 1.0))
        self.dyrel_period = float(getattr(controls, "dyrel_period", 0.0))
        self.dyrel_istatg = int(getattr(controls, "dyrel_istatg", 0))

        if self.dyrel_period > 0.0:
            self.betate = self.dyrel_beta / self.dyrel_period
        else:
            self.betate = self.dyrel_beta

        self.kerel_active = bool(getattr(controls, "kerel_active", False))
        self.kerel_tstart = float(getattr(controls, "kerel_tstart", 0.0))
        self.kerel_tstop = float(getattr(controls, "kerel_tstop", 0.0))
        if self.kerel_tstop <= 0.0:
            self.kerel_tstop = float(getattr(controls, "t_end", 0.0) or 1e30)
        self.kerel_istatg = int(getattr(controls, "kerel_istatg", 0))

        self.adyrel_active = bool(getattr(controls, "adyrel_active", False))
        self.adyrel_freq_c = float(getattr(controls, "adyrel_freq_c", 0.0))
        self.adyrel_tstart = float(getattr(controls, "adyrel_tstart", 0.0))
        self.adyrel_tstop = float(getattr(controls, "adyrel_tstop", 0.0))
        if self.adyrel_tstop <= 0.0:
            self.adyrel_tstop = float(getattr(controls, "t_end", 0.0) or 1e30)
        self.adyrel_istatg = int(getattr(controls, "adyrel_istatg", 0))

        # ADYREL frequency adaptation state
        self.fil_ke = np.zeros(4, dtype=np.float64)
        self.fil_ie = np.zeros(4, dtype=np.float64)
        self.pcin = 0.0
        self.pint = 0.0
        self.pcmax = 0.0
        self.pimax = 0.0
        self.encin_0 = 0.0
        self.eint_0 = 0.0
        self.ifirst = 0
        self.adyrel_betate = 0.0

        self.ke_prev = 0.0

        self.dyrel_idx = self._resolve_nodes(self.dyrel_istatg)
        self.kerel_idx = self._resolve_nodes(self.kerel_istatg)
        self.adyrel_idx = self._resolve_nodes(self.adyrel_istatg)

        if log is not None:
            if self.dyrel_active:
                log.info(f"     /DYREL: BETA = {self.dyrel_beta:g}, "
                         f"PERIOD = {self.dyrel_period:g} (BETATE = {self.betate:12.5E}) "
                         f"ON {len(self.dyrel_idx)} NODE(S)")
            if self.kerel_active:
                log.info(f"     /KEREL: ACTIVE {self.kerel_tstart:g} TO {self.kerel_tstop:g} "
                         f"ON {len(self.kerel_idx)} NODE(S)")
            if self.adyrel_active:
                log.info(f"     /ADYREL: ACTIVE {self.adyrel_tstart:g} TO {self.adyrel_tstop:g} "
                         f"(FREQ_C = {self.adyrel_freq_c:g}) ON {len(self.adyrel_idx)} NODE(S)")

    def _resolve_nodes(self, istatg: int) -> np.ndarray:
        if istatg != 0 and hasattr(self.model, "node_groups"):
            target_id = abs(istatg)
            g = self.model.node_groups.get(target_id) or self.model.node_groups.get(istatg)
            if g is not None and getattr(g, "node_idx", None) is not None and len(g.node_idx) > 0:
                idx = g.node_idx[self.model.mass[g.node_idx] < 1e29]
                return idx
        if hasattr(self.model, "mass") and self.model.mass is not None:
            return np.where(self.model.mass < 1e29)[0]
        n = getattr(self.model, "numnod", 0)
        return np.arange(n, dtype=np.int64)

    @property
    def active(self) -> bool:
        return bool(self.dyrel_active or self.kerel_active or self.adyrel_active)

    def __len__(self) -> int:
        return 1 if self.active else 0

    def e_period(self, dt: float, encin: float, enint: float, t: float) -> tuple[int, int]:
        """Port of E_PERIOD from static.F:211.
        Returns (ipi, ipc) peak detection indicators.
        """
        f_fil = self.adyrel_freq_c
        if t == 0.0 or (self.fil_ke[0] == 0.0 and self.fil_ie[0] == 0.0 and encin > 0.0):
            self.fil_ke[:] = encin
            self.fil_ie[:] = enint

        if f_fil > 0.0 and dt > 0.0:
            dt2 = dt / 2.0
            fv_ke = butterworth_filter(
                dt2, f_fil,
                self.fil_ke[1], self.fil_ke[0], encin,
                self.fil_ke[3], self.fil_ke[2]
            )
            self.fil_ke[1] = self.fil_ke[0]
            self.fil_ke[0] = encin
            self.fil_ke[3] = self.fil_ke[2]
            self.fil_ke[2] = fv_ke
            encint = fv_ke

            fv_ie = butterworth_filter(
                dt2, f_fil,
                self.fil_ie[1], self.fil_ie[0], enint,
                self.fil_ie[3], self.fil_ie[2]
            )
            self.fil_ie[1] = self.fil_ie[0]
            self.fil_ie[0] = enint
            self.fil_ie[3] = self.fil_ie[2]
            self.fil_ie[2] = fv_ie
            eint = fv_ie
        else:
            encint = encin
            eint = enint

        self.pcin += dt
        self.pint += dt
        self.pcmax = max(self.pcmax, self.pcin)
        self.pimax = max(self.pimax, self.pint)

        if encint < self.encin_0 and encint >= 0.0:
            self.encin_0 = 0.0
            self.pcin = 0.0
            ipc = 1
        else:
            ipc = 0
            self.encin_0 = encint

        if eint < 0.0:
            self.eint_0 = 0.0
            self.pint = 0.0
            ipi = -2
        elif eint < self.eint_0 and eint >= 0.0:
            self.eint_0 = 0.0
            self.pint = 0.0
            ipi = 1
        else:
            self.eint_0 = eint
            ipi = 0

        return ipi, ipc

    def update_adaptive_frequency(self, t: float, dt: float, cycle: int,
                                  e_int: float, e_kin: float):
        """Port of ENER_W0 from static.F:312.
        Adapts self.adyrel_betate at each cycle.
        """
        if not self.adyrel_active or dt <= 0.0:
            return

        ipi, ipc = self.e_period(dt, e_kin, e_int, t)
        if cycle == 0:
            return

        nc_act = 200
        ei_tol = 1.0e-12
        f_max = 0.01 / dt
        if self.adyrel_freq_c < 0.0:
            self.adyrel_freq_c = -self.adyrel_freq_c * f_max
        f_0 = 0.01 * f_max
        betate_n = f_max

        if ipi == 1:
            fi = 1.0 / max(1e-20, self.pimax)
            betate_n = min(f_max, fi)
            if self.adyrel_betate == 0.0 and self.eint_0 > ei_tol:
                if cycle >= nc_act:
                    self.adyrel_betate = min(f_0, betate_n)
                    self.ifirst = 1
            elif self.ifirst == 1:
                self.adyrel_betate = betate_n
                self.ifirst += 1
            else:
                self.adyrel_betate = min(self.adyrel_betate, betate_n)

        if ipc == 1:
            fc = 1.0 / max(1e-20, self.pcmax)
            betate_n = min(f_max, fc)
            if self.adyrel_betate == 0.0 and self.eint_0 > ei_tol:
                if cycle >= nc_act:
                    self.adyrel_betate = min(f_0, betate_n)
                    self.ifirst = 1
            elif self.ifirst == 1:
                self.adyrel_betate = betate_n
                self.ifirst += 1
            else:
                ratio = self.encin_0 / max(1e-20, self.eint_0)
                if ratio > 1.0e-3 and self.adyrel_betate < betate_n * 1.5:
                    betate_m = 0.5 * self.adyrel_betate
                    self.adyrel_betate = min(self.adyrel_betate, betate_n)
                    self.adyrel_betate = max(self.adyrel_betate, betate_m)

        if self.adyrel_betate == 0.0 and self.eint_0 > ei_tol and cycle >= nc_act:
            self.adyrel_betate = f_0
            self.ifirst = 1

        if self.ifirst >= 1 and (ipc + ipi) == 0:
            fi = 1.0 / max(1e-20, max(self.pimax, self.pcmax))
            if self.adyrel_betate > 1.1 * fi:
                self.adyrel_betate = fi

    def apply(self, t: float, dt: float, v: np.ndarray, vr: Optional[np.ndarray],
              mass: np.ndarray, inertia: Optional[np.ndarray]) -> float:
        """Apply dynamic relaxation (DYREL, KEREL, and/or ADYREL) and return dissipated energy."""
        de = 0.0
        if dt <= 0.0 or not self.active:
            return de

        # 1. /DYREL viscous relaxation damping
        if self.dyrel_active and len(self.dyrel_idx) > 0:
            idx = self.dyrel_idx
            omega = self.betate * dt
            fac = max(0.0, 1.0 - 2.0 * omega)
            omega2 = fac * fac
            de += float(0.5 * (1.0 - omega2) * np.sum(mass[idx, None] * (v[idx] ** 2)))
            v[idx] *= fac
            if vr is not None and inertia is not None:
                has_in = inertia[idx] > 0.0
                if np.any(has_in):
                    ir = idx[has_in]
                    de += float(0.5 * (1.0 - omega2) * np.sum(inertia[ir, None] * (vr[ir] ** 2)))
                    vr[ir] *= fac

        # 2. /KEREL kinetic energy peak zeroing
        if self.kerel_active and (self.kerel_tstart <= t <= self.kerel_tstop) and len(self.kerel_idx) > 0:
            idx = self.kerel_idx
            ke = float(0.5 * np.sum(mass[idx, None] * (v[idx] ** 2)))
            has_in = inertia is not None and vr is not None and (inertia[idx] > 0.0)
            if vr is not None and inertia is not None and np.any(has_in):
                ir = idx[has_in]
                ke += float(0.5 * np.sum(inertia[ir, None] * (vr[ir] ** 2)))

            if ke < self.ke_prev and self.ke_prev > 0.0:
                v[idx] = 0.0
                if vr is not None and inertia is not None and np.any(has_in):
                    vr[ir] = 0.0
                de += ke
                self.ke_prev = 0.0
            else:
                self.ke_prev = ke

        # 3. /ADYREL adaptive dynamic relaxation damping
        if self.adyrel_active and (self.adyrel_tstart <= t <= self.adyrel_tstop) and len(self.adyrel_idx) > 0 and self.adyrel_betate > 0.0:
            idx = self.adyrel_idx
            omega = self.adyrel_betate * dt
            fac = max(0.0, 1.0 - 2.0 * omega)
            omega2 = fac * fac
            de += float(0.5 * (1.0 - omega2) * np.sum(mass[idx, None] * (v[idx] ** 2)))
            v[idx] *= fac
            if vr is not None and inertia is not None:
                has_in = inertia[idx] > 0.0
                if np.any(has_in):
                    ir = idx[has_in]
                    de += float(0.5 * (1.0 - omega2) * np.sum(inertia[ir, None] * (vr[ir] ** 2)))
                    vr[ir] *= fac

        return de

    def apply_acceleration_damping(
        self,
        a: np.ndarray,
        v: np.ndarray,
        dt: float,
        dt12: float,
        ar: Optional[np.ndarray] = None,
        vr: Optional[np.ndarray] = None,
    ) -> None:
        """Apply acceleration damping coupling matching static.F:83-109 / 135-165.

        A <- -2*beta*V + (1 - beta*dt12)*A
        AR <- -2*beta*VR + (1 - beta*dt12)*AR
        """
        if self.dyrel_active:
            beta = self.betate
            target_idx = self.dyrel_idx
        elif self.adyrel_active:
            beta = self.adyrel_betate
            target_idx = self.adyrel_idx
        else:
            beta = self.betate
            target_idx = self.dyrel_idx

        if beta <= 0.0:
            return

        omega = beta * dt12
        uomega = 1.0 - omega
        domega = 2.0 * beta

        if a.ndim > 1 and len(target_idx) > 0 and np.max(target_idx) < a.shape[0]:
            a[target_idx] = -domega * v[target_idx] + uomega * a[target_idx]
        else:
            a[:] = -domega * v + uomega * a

        if ar is not None and vr is not None:
            if ar.ndim > 1 and len(target_idx) > 0 and np.max(target_idx) < ar.shape[0]:
                ar[target_idx] = -domega * vr[target_idx] + uomega * ar[target_idx]
            else:
                ar[:] = -domega * vr + uomega * ar

    def compute_timestep_reduction(
        self,
        dt: float,
        dt12: float,
        beta: Optional[float] = None,
    ) -> float:
        """Compute time-step stability factor matching dtnodarayl.F:168-188.

        Dampa3 = 2 * beta / (1 + beta * dt12)
        BB = 0.5 * Dampa3 * dt^2
        dt_reduced = sqrt(BB^2 + dt^2) - BB
        """
        if beta is None:
            if self.adyrel_active and self.adyrel_betate > 0.0:
                beta = self.adyrel_betate
            elif self.dyrel_active and self.betate > 0.0:
                beta = self.betate
            else:
                beta = self.betate

        if beta <= 0.0 or dt <= 0.0:
            return float(dt)

        dampa3 = 2.0 * beta / (1.0 + beta * dt12)
        bb = 0.5 * dampa3 * (dt * dt)
        dt_reduced = float(np.sqrt(bb * bb + dt * dt) - bb)
        return dt_reduced

    def check_convergence(
        self,
        e_kin: float,
        e_int: float,
        e_kin_max: float,
        tol: float = 1e-3,
    ) -> bool:
        """Check static relaxation convergence."""
        return check_convergence(e_kin, e_int, e_kin_max, tol=tol)


def check_convergence(
    e_kin: float,
    e_int: float,
    e_kin_max: float,
    tol: float = 1e-3,
) -> bool:
    """Check static relaxation convergence.

    Signals that static equilibrium has been reached when either:
    1. E_kin / E_int < tol (default 1e-3)
    2. E_kin / E_kin_max < tol
    """
    if e_int > 0.0 and (e_kin / e_int) < tol:
        return True
    if e_kin_max > 0.0 and (e_kin / e_kin_max) < tol:
        return True
    return False


