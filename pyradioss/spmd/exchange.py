"""
Engine side of the SPMD domain decomposition: the per-cycle frontier
exchanges, the global time-step packet, the weighted global sums and the
rank-0 gathers that feed the listing / time history / animation outputs.

Fortran origin
--------------
* ``engine/source/mpi/forces/spmd_exch_a.F`` — the frontier force /
  stiffness sum (``MSGOFF = 120``): every domain posts one ``IRECV`` per
  neighbour, packs the rows of its frontier nodes (``FR_ELEM`` in
  ``IAD_ELEM`` order) node-interleaved ``A(1:3), AR(1:3), STIFN, STIFR``
  into one buffer per neighbour, ``ISEND``\\ s it, waits, and ACCUMULATES
  the received partial sums into its own arrays — so every holder of a
  frontier node ends the assembly with the complete nodal force.
* ``engine/source/mpi/generic/spmd_exch_v.F`` (``MSGOFF = 7000``) — the
  velocity exchange used by the Fortran where a kinematic condition is
  computed on one domain only; the port replicates kinematic entities
  (see ``domdec.py``) so the explicit cycle needs no velocity exchange,
  the tag is kept for the generic point-to-point helper.
* ``engine/source/mpi/generic/spmd_glob_min5.F`` + ``generic/glob_min.F``
  — the per-cycle global time-step reduction (the 10-slot packet: DT2
  min, critical element type/number following the winner, summed and
  max'ed stop flags); the operator itself lives in ``comm.Comm.glob_min``.
* ``engine/source/mpi/generic/spmd_glob_rsum_poff.F`` /
  ``spmd_exsum_fb6.F`` — global sums of partial per-domain quantities
  (energies, works, masses).
* ``engine/source/mpi/output/spmd_collect.F`` (``MSGOFF0/MSGOFF`` 176 /
  177) and ``spmd_gather.F`` — the gathers to domain 0: each domain sends
  the rows of the nodes it OWNS (``WEIGHT(I) == 1``) with their
  ``NODGLOB`` global index, domain 0 scatters them into the global
  array (``RECGLOB(:, NODGLOB(I))``).
* ``engine/source/mpi/kinematic_conditions/spmd_exch_a_rb6.F`` — rigid
  body resultants; the port replicates each rigid body on every domain
  holding one of its nodes, the frontier force exchange already gives
  every replica the complete forces, so the resultant is identical
  everywhere without a separate exchange.

WEIGHT
------
``WEIGHT(N) = 1`` on the node's main domain, ``0`` on every other holder
(``ecrit.F``, ``gravit.F``, ``fixvel.F`` multiply once-only sums by it).
:attr:`SpmdContext.weight` is that array (``None`` in a serial run, which
every caller treats as "all ones" through a separate, untouched branch —
the serial path pays nothing).

Determinism of the frontier sum
-------------------------------
``spmd_exch_a.F`` accumulates into the domain's own partial sum in the
order the neighbour receives complete; for a node shared by three or more
domains the floating-point result can then differ in the last bit between
holders.  The port relies on every holder integrating EXACTLY the same
nodal force (velocities are never exchanged), so the frontier sum is
formed in ASCENDING RANK ORDER over all holders on every holder — the
same deterministic ordering ``/PARITH/ON`` buys in the Fortran — which
makes the assembled force bitwise identical on all holders.
"""

from __future__ import annotations

import copy
from collections import namedtuple
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .comm import SERIAL, SPMD_MAX, Comm

# message types (the Fortran MSGOFF of the routine each exchange ports)
MSGOFF_EXCH_A = 120      # spmd_exch_a.F      DATA MSGOFF/120/
MSGOFF_EXCH_V = 7000     # spmd_exch_v.F      DATA MSGOFF/7000/
MSGOFF_COLLECT = 177     # spmd_collect.F     DATA MSGOFF/177/

EP30 = 1.0e30

#: element-group state keys that are NOT per-element data (index
#: permutations, colouring tables, local node connectivities) and must
#: never be scattered into the global view
_STATE_SKIP = frozenset({"color_indices", "color_offsets", "slices",
                         "mass_conn", "conn"})

#: nodal model arrays mirrored into the global output view
NODAL_KEYS = ("x", "v", "vr", "a", "mass", "inertia", "fint", "fext")

GlobMin = namedtuple("GlobMin", "dt crit_type crit_id stop timet sums")

_CRIT_TABLE: Optional[List[str]] = None


def crit_type_table() -> List[str]:
    """The ``ITYPTS`` code table: critical-entity type name <-> integer
    (slot 2 of the ``spmd_glob_min5.F`` packet carries a NUMBER, like the
    Fortran element type code).  Built deterministically (sorted element
    group names of the kernel registry + the non-element entities) so every
    domain — thread or MPI process — uses the same table."""
    global _CRIT_TABLE
    if _CRIT_TABLE is None:
        try:
            from ..elements import KERNELS
            names = sorted(str(k) for k in KERNELS)
        except Exception:   # pragma: no cover - registry import failure
            names = []
        _CRIT_TABLE = ["", "NODE", "INTERFACE", "RBODY"] + \
            [n for n in names if n not in ("", "NODE", "INTERFACE", "RBODY")]
    return _CRIT_TABLE


def crit_type_code(name) -> int:
    """Name -> code (unknown names map to 0, the empty type)."""
    if isinstance(name, (int, np.integer)):
        return int(name)
    tab = crit_type_table()
    try:
        return tab.index(str(name))
    except ValueError:
        return 0


def crit_type_name(code) -> str:
    """Code -> name (out-of-range codes map to '')."""
    tab = crit_type_table()
    c = int(round(float(code)))
    return tab[c] if 0 <= c < len(tab) else ""


class _RBodyView:
    """Picklable output view of one engine rigid body (the attributes the
    time history reads: ``output/time_history.py`` RBODY requests)."""

    __slots__ = ("id", "xg", "x_cg0", "v_ref", "w", "f_res", "m_res")

    def __init__(self, rb):
        self.id = int(rb.rb.id) if hasattr(rb, "rb") else int(getattr(rb, "id", 0))
        self.xg = np.array(rb.x_cg, dtype=np.float64, copy=True)
        self.x_cg0 = np.array(getattr(rb, "x_cg0", rb.x_cg), dtype=np.float64,
                              copy=True)
        self.v_ref = np.array(rb.v_cg, dtype=np.float64, copy=True)
        self.w = np.array(getattr(rb, "w", np.zeros(3)), dtype=np.float64,
                          copy=True)
        self.f_res = np.array(getattr(rb, "f_res", np.zeros(3)),
                              dtype=np.float64, copy=True)
        self.m_res = np.array(getattr(rb, "m_res", np.zeros(3)),
                              dtype=np.float64, copy=True)

    @property
    def x_cg(self):
        return self.xg

    @property
    def v_cg(self):
        return self.v_ref


class SpmdContext:
    """The engine's handle on the decomposition: communicator + the
    ``DomainInfo`` the Starter attached to the local model
    (``model.spmd``).

    ``SpmdContext(comm, info)`` with ``info is None`` (or ``nspmd == 1``, or
    a size-1 communicator) is the SERIAL context: ``active`` is False and
    every method is a cheap identity, so the engine can call the hooks
    unconditionally — but engine.py guards them with ``spmd.active``
    anyway to keep the serial path bitwise untouched."""

    def __init__(self, comm: Optional[Comm] = None, info: Any = None):
        self.comm = comm if comm is not None else SERIAL
        self.info = info
        self.active = bool(self.comm.active and info is not None
                           and int(getattr(info, "nspmd", 1)) > 1)
        self.rank = self.comm.rank
        self.size = self.comm.size
        if not self.active:
            self.weight = None
            self._nbrs: List[int] = []
            return
        if int(info.nspmd) != self.comm.size or \
                int(info.ispmd) != self.comm.rank:
            raise RuntimeError(
                f"SPMD: domain restart of ISPMD={info.ispmd} (NSPMD="
                f"{info.nspmd}) opened by rank {self.comm.rank} of a "
                f"{self.comm.size}-process communicator")
        self.weight = np.asarray(info.weight, dtype=np.float64)
        self.nodglob = np.asarray(info.nodglob, dtype=np.int64)
        self.main_proc = np.asarray(info.main_proc, dtype=np.int64)
        self.numnod_glob = int(info.numnod_glob)
        self._own = np.nonzero(self.weight > 0.0)[0]
        # neighbour table (IAD_ELEM / FR_ELEM): only non-empty lists,
        # ascending rank
        self._front: Dict[int, np.ndarray] = {}
        for r, idx in (info.frontier or {}).items():
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx) and int(r) != self.rank:
                self._front[int(r)] = idx
        self._nbrs = sorted(self._front)
        if self._nbrs:
            self._fall = np.unique(np.concatenate(
                [self._front[r] for r in self._nbrs]))
            self._pos = {r: np.searchsorted(self._fall, self._front[r])
                         for r in self._nbrs}
            self._order = sorted(self._nbrs + [self.rank])
        else:
            self._fall = np.zeros(0, dtype=np.int64)
            self._pos = {}
            self._order = [self.rank]

    # ------------------------------------------------------------------
    @property
    def is_root(self) -> bool:
        return self.rank == 0

    @property
    def neighbours(self) -> List[int]:
        return list(self._nbrs)

    # ------------------------------------------------------------------
    def exch_forces(self, fint: Optional[np.ndarray],
                    fext: Optional[np.ndarray] = None,
                    fcont: Optional[np.ndarray] = None,
                    mint: Optional[np.ndarray] = None,
                    stifn: Optional[np.ndarray] = None,
                    stifr: Optional[np.ndarray] = None) -> None:
        """Frontier sum of the partial nodal arrays, IN PLACE
        (``spmd_exch_a.F``).  ``None`` arrays are skipped — the set of
        non-None arguments must be the same on every domain (the message
        size is the product of the frontier length and the packed width,
        exactly like the Fortran ``SIZE`` argument)."""
        if not self.active or not self._nbrs:
            return
        arrays = [a for a in (fint, fext, fcont, mint, stifn, stifr)
                  if a is not None]
        self._exchange_sum(arrays, MSGOFF_EXCH_A)

    def _exchange_sum(self, arrays: Sequence[np.ndarray], tag: int) -> None:
        cols = []
        for a in arrays:
            k = 1 if a.ndim == 1 else int(np.prod(a.shape[1:]))
            cols.append(k)
        width = int(sum(cols))
        comm = self.comm
        # 1. post every receive (the IRECV loop of spmd_exch_a.F)
        rbufs, rreqs = {}, {}
        for r in self._nbrs:
            buf = np.empty((len(self._front[r]), width), dtype=np.float64)
            rbufs[r] = buf
            rreqs[r] = comm.irecv(buf, r, tag)
        # 2. pack node-interleaved per neighbour in frontier order + ISEND
        sreqs = []
        for r in self._nbrs:
            idx = self._front[r]
            sbuf = np.empty((len(idx), width), dtype=np.float64)
            c = 0
            for a, k in zip(arrays, cols):
                sbuf[:, c:c + k] = a[idx].reshape(len(idx), k)
                c += k
            sreqs.append(comm.isend(sbuf, r, tag))
        # 3. wait for every receive, then ACCUMULATE — in ascending rank
        # order over all holders (module docstring: bitwise-identical
        # sums on every holder)
        for r in self._nbrs:
            rreqs[r].wait()
        F = self._fall
        c = 0
        for a, k in zip(arrays, cols):
            shape = (len(F),) + a.shape[1:]
            acc = np.zeros(shape, dtype=np.float64)
            for r in self._order:
                if r == self.rank:
                    acc += a[F]
                else:
                    acc[self._pos[r]] += rbufs[r][:, c:c + k].reshape(
                        (len(self._pos[r]),) + a.shape[1:])
            a[F] = acc
            c += k
        for req in sreqs:
            req.wait()

    # ------------------------------------------------------------------
    def glob_min(self, dt: float, crit_type="", crit_id: int = 0,
                 stop: bool = False, *sums: float,
                 timet: bool = False, tstop: float = EP30) -> GlobMin:
        """The ``spmd_glob_min5.F`` packet: global minimum time step with
        the critical element of the winning domain, the stop flags
        (``MSTOP1``/``MSTOP2`` MAX slots — here the local-stop flag and
        the wall-clock ``/STOP/TIMET`` flag) and up to three SUMMED values
        (the ``IEXICODT``/``IMSCH``/``IWIOUT`` slots, used by the port for
        the global added mass of /DT/NODA/CST).  Returns a
        :data:`GlobMin` namedtuple; the serial context returns its inputs."""
        if len(sums) > 3:
            raise ValueError("glob_min carries at most 3 summed values")
        s = list(sums) + [0.0] * (3 - len(sums))
        if not self.active:
            return GlobMin(float(dt),
                           crit_type if isinstance(crit_type, str)
                           else crit_type_name(crit_type),
                           int(crit_id), bool(stop), bool(timet),
                           tuple(float(x) for x in sums))
        pk = np.array([float(dt), float(crit_type_code(crit_type)),
                       float(crit_id), float(s[0]), float(s[1]),
                       float(tstop), float(s[2]),
                       1.0 if stop else 0.0, 1.0 if timet else 0.0, 0.0],
                      dtype=np.float64)
        out = self.comm.glob_min(pk)
        gs = (out[3], out[4], out[6])[:len(sums)]
        return GlobMin(float(out[0]), crit_type_name(out[1]),
                       int(round(float(out[2]))), bool(out[7] > 0.0),
                       bool(out[8] > 0.0), tuple(float(x) for x in gs))

    # ------------------------------------------------------------------
    def sum(self, values):
        """Global sum (``spmd_glob_dsum``/``spmd_glob_rsum_poff.F``) of a
        float or a (small) ndarray; identity in serial.

        The partial values are all-gathered and added in RANK ORDER on
        every domain, so the result is bitwise identical everywhere and
        independent of the MPI library's reduction tree — the energy /
        mass-error stop tests evaluated on it then take the same branch
        on every domain (a divergent branch would dead-lock the next
        collective)."""
        if not self.active:
            return values
        scalar = np.isscalar(values)
        a = np.array(values, dtype=np.float64, copy=True).reshape(-1)
        parts = self.comm.allgather(a)
        tot = np.zeros_like(a)
        for pa in parts:
            tot += pa
        if scalar:
            return float(tot[0])
        return tot.reshape(np.shape(values))

    def max(self, values):
        """Global maximum (``SPMD_MAX`` all-reduce)."""
        if not self.active:
            return values
        if np.isscalar(values):
            return self.comm.allreduce_scalar(float(values), SPMD_MAX)
        return self.comm.allreduce(np.asarray(values, dtype=np.float64),
                                   SPMD_MAX)

    def any_stop(self, flag: bool) -> bool:
        """True on every domain when ANY domain raises ``flag`` (the MAX of
        the ``MSTOP`` flags)."""
        if not self.active:
            return bool(flag)
        return self.comm.allreduce_scalar(1.0 if flag else 0.0,
                                          SPMD_MAX) > 0.0

    def allgather(self, obj) -> list:
        if not self.active:
            return [obj]
        return self.comm.allgather(obj)

    def bcast(self, obj, root: int = 0):
        if not self.active:
            return obj
        return self.comm.bcast(obj, root)

    def bcast_str(self, s: str, root: int = 0) -> str:
        return str(self.bcast(str(s), root))

    def barrier(self) -> None:
        if self.active:
            self.comm.barrier()

    # ------------------------------------------------------------------
    # rank-0 gathers (spmd_collect.F / spmd_gather.F)
    # ------------------------------------------------------------------
    def gather_nodal(self, local_arr: np.ndarray) -> Optional[np.ndarray]:
        """The full ``(numnod_glob, ...)`` array on domain 0 (``None``
        elsewhere): every domain contributes the rows of the nodes it owns
        (``WEIGHT == 1``) at their ``NODGLOB`` position — ``spmd_collect.F``.
        Serial: the input itself."""
        if not self.active:
            return local_arr
        a = np.asarray(local_arr)
        own = self._own
        parts = self.comm.gather((self.nodglob[own], a[own]), root=0)
        if parts is None:
            return None
        out = np.zeros((self.numnod_glob,) + a.shape[1:], dtype=a.dtype)
        for gidx, vals in parts:
            out[gidx] = vals
        return out

    def gather_group_state(self, gname: str,
                           arr: np.ndarray) -> Optional[np.ndarray]:
        """The global ``(numel_glob[gname], ...)`` array of one per-element
        quantity on domain 0 via ``elem_glob`` (elements are partitioned,
        not replicated: every local row is written exactly once)."""
        if not self.active:
            return arr
        a = np.asarray(arr)
        eg = np.asarray(self.info.elem_glob.get(gname, np.zeros(0)),
                        dtype=np.int64)
        parts = self.comm.gather((eg, a.copy()), root=0)   # copy: see
        # _group_payload (thread backend hands objects over by reference)
        if parts is None:
            return None
        n = int(self.info.numel_glob.get(gname, 0))
        out = np.zeros((n,) + a.shape[1:], dtype=a.dtype)
        for gidx, vals in parts:
            if len(gidx):
                out[gidx] = vals
        return out

    @staticmethod
    def _group_payload(group, n: int) -> dict:
        # COPIES: the thread backend hands the objects themselves to domain
        # 0, and the sender resumes integrating (in place) as soon as the
        # collective returns — a view would race with the next cycle
        pay = {}
        for key, val in group.state.items():
            if key in _STATE_SKIP:
                continue
            if isinstance(val, np.ndarray):
                if val.ndim >= 1 and val.shape[0] == n:
                    pay[key] = val.copy()
            elif isinstance(val, dict):
                sub = {k2: v2.copy() for k2, v2 in val.items()
                       if isinstance(v2, np.ndarray) and v2.ndim >= 1
                       and v2.shape[0] == n}
                if sub:
                    pay[key] = sub
        return pay

    @staticmethod
    def _scatter_state(gstate: dict, key, gidx, vals, nglob: int) -> None:
        garr = gstate.get(key)
        if not isinstance(garr, np.ndarray) or garr.shape[0] != nglob or \
                garr.shape[1:] != vals.shape[1:] or garr.dtype != vals.dtype:
            garr = np.zeros((nglob,) + vals.shape[1:], dtype=vals.dtype)
            gstate[key] = garr
        if len(gidx):
            garr[gidx] = vals

    def gather_model_view(self, model, gmodel, nodal: bool = True,
                          groups: bool = True,
                          extra_nodal: Optional[Dict[str, np.ndarray]] = None,
                          rigid_bodies: bool = True) -> Optional[dict]:
        """Refresh domain 0's GLOBAL output model ``gmodel`` from every
        domain's local model in ONE gather: the nodal arrays of
        :data:`NODAL_KEYS` (owned rows at ``NODGLOB``), every per-element
        ``group.state`` array (and one level of nested dicts such as
        ``mat_extra``) at ``elem_glob`` rows, and the rigid-body output
        views.  ``extra_nodal`` ({name: local nodal array}) is gathered too
        and returned as {name: global array} on domain 0.  Collective:
        every domain must call it at the same point.  Returns ``None`` on
        the other domains."""
        if not self.active:
            if gmodel is not None and gmodel is not model:
                for key in NODAL_KEYS:
                    val = getattr(model, key, None)
                    if isinstance(val, np.ndarray):
                        setattr(gmodel, key, val.copy())
            return dict(extra_nodal or {})
        own = self._own
        payload = {"nod": self.nodglob[own]}
        if nodal:
            nd = {}
            n_loc = len(self.nodglob)
            for key in NODAL_KEYS:
                val = getattr(model, key, None)
                if isinstance(val, np.ndarray) and val.ndim >= 1 and \
                        val.shape[0] == n_loc:
                    nd[key] = val[own]
            payload["nodal"] = nd
        if extra_nodal:
            payload["extra"] = {k: np.asarray(v)[own]
                                for k, v in extra_nodal.items()}
        if groups:
            gp = {}
            eglob = getattr(self.info, "elem_glob", {}) or {}
            for gname, group in model.element_groups():
                if gname not in eglob:
                    continue
                n = len(eglob[gname])
                gp[gname] = (np.asarray(eglob[gname], dtype=np.int64),
                             self._group_payload(group, n))
            payload["groups"] = gp
        if rigid_bodies:
            rbs = getattr(model, "rigid_bodies", None) or {}
            payload["rb"] = {int(k): _RBodyView(v) for k, v in rbs.items()}
        parts = self.comm.gather(payload, root=0)
        if parts is None:
            return None
        nglob = self.numnod_glob
        extra_out = {}
        if nodal:
            for key in NODAL_KEYS:
                vals = [(p["nod"], p["nodal"][key]) for p in parts
                        if key in p.get("nodal", {})]
                if not vals:
                    continue
                proto = vals[0][1]
                garr = getattr(gmodel, key, None)
                if not isinstance(garr, np.ndarray) or \
                        garr.shape != (nglob,) + proto.shape[1:]:
                    garr = np.zeros((nglob,) + proto.shape[1:],
                                    dtype=proto.dtype)
                for gidx, v in vals:
                    garr[gidx] = v
                setattr(gmodel, key, garr)
        if extra_nodal:
            for key in extra_nodal:
                vals = [(p["nod"], p["extra"][key]) for p in parts]
                proto = vals[0][1]
                garr = np.zeros((nglob,) + proto.shape[1:], dtype=proto.dtype)
                for gidx, v in vals:
                    garr[gidx] = v
                extra_out[key] = garr
        if groups:
            gg = dict(gmodel.element_groups())
            nel = getattr(self.info, "numel_glob", {}) or {}
            for p in parts:
                for gname, (gidx, pay) in p.get("groups", {}).items():
                    ggroup = gg.get(gname)
                    if ggroup is None:
                        continue
                    nglob_e = int(nel.get(gname, len(ggroup.conn)))
                    for key, val in pay.items():
                        if isinstance(val, dict):
                            sub = ggroup.state.get(key)
                            if not isinstance(sub, dict):
                                sub = {}
                                ggroup.state[key] = sub
                            for k2, v2 in val.items():
                                self._scatter_state(sub, k2, gidx, v2,
                                                    nglob_e)
                        else:
                            self._scatter_state(ggroup.state, key, gidx, val,
                                                nglob_e)
        if rigid_bodies:
            merged = {}
            for p in parts:              # lowest rank wins (identical copies)
                for k, v in p.get("rb", {}).items():
                    merged.setdefault(k, v)
            gmodel.rigid_bodies = merged
        return extra_out

    # ------------------------------------------------------------------
    def reduce_state(self, state, keys: Sequence[str] = (
            "wext", "econt", "e_num", "e_damp", "e_madd", "ek_ams")):
        """A copy of the engine state with the PARTIAL per-domain ledgers
        summed (one all-reduce): every domain books only its own share
        (elements are partitioned, nodal bookings carry WEIGHT), so the
        global ledger is the plain sum.  Global quantities (``t``,
        ``cycle``, ``epeak``, ``e0``, ``dt_prev`` ...) are copied as they
        are.  The copy is detached from the context (``spmd = None``) so
        it can be handed to serial post-processing."""
        red = copy.copy(state)
        if hasattr(red, "spmd"):
            red.spmd = None
        if not self.active:
            return red
        vals = np.array([float(getattr(state, k, 0.0) or 0.0) for k in keys]
                        + [float(getattr(state, "ndel", 0) or 0)],
                        dtype=np.float64)
        g = self.sum(vals)
        for k, v in zip(keys, g[:-1]):
            setattr(red, k, float(v))
        if hasattr(state, "ndel"):
            # the deleted-element count (elements are partitioned)
            red.ndel = int(round(float(g[-1])))
        return red


def serial_context() -> SpmdContext:
    """The no-op context of a serial run."""
    return SpmdContext(SERIAL, None)
