"""
``pyradioss.spmd`` — the port of the OpenRadioss SPMD (Single Program,
Multiple Data) domain-decomposition and message-passing layer.

Fortran origin
--------------
* ``engine/source/mpi/`` (289 files) — the ``SPMD_*`` wrappers and every
  per-subsystem exchange routine (``spmd_exch_a.F``, ``spmd_glob_min5.F``,
  ``spmd_exch_v.F``, ``spmd_collect.F`` ...);
* ``starter/source/spmd/`` (11 files) — the domain decomposition
  (``domdec1.F``, ``domdec2.F``, ``initwg.F``, ``frontplus.F``,
  ``ddtools.F``) and the per-domain restart split (``ddsplit.F``).

Layout of the port
------------------
* :mod:`pyradioss.spmd.comm`     — communicator backends (serial, threads,
  mpi4py) and the ``glob_min`` custom reduction;
* :mod:`pyradioss.spmd.domdec`   — Starter side: element weights, weighted
  RCB partition, frontier nodes (``IFRONT``/``IFRONTPLUS``), ``NODGLOB``,
  ``WEIGHT``, entity ownership rules and the per-domain model slice;
* :mod:`pyradioss.spmd.exchange` — Engine side: the per-cycle frontier
  force/stiffness sum, the global time-step packet, the weighted global
  sums and the rank-0 gathers for the listing / TH / ANIM outputs;
* :mod:`pyradioss.spmd.driver`   — the ``-np N`` launcher (mpi4py under
  ``mpirun``, otherwise one thread per domain in this process).
"""

from .comm import (  # noqa: F401
    SPMD_MAX, SPMD_MIN, SPMD_SUM, SPMD_PROD,
    Comm, Request, SerialComm, SERIAL, ThreadComm, Mpi4pyComm,
    mpi_world_size,
)
from .domdec import (  # noqa: F401
    DomainInfo, Decomposition, element_weights, partition_elements,
    check_spmd_support, decompose, slice_model, domain_restart_path,
    decomposition_table, write_domain_restarts,
)
from .exchange import (  # noqa: F401
    MSGOFF_EXCH_A, MSGOFF_EXCH_V, MSGOFF_COLLECT,
    SpmdContext, GlobMin, serial_context,
    crit_type_table, crit_type_code, crit_type_name,
)
from .driver import count_domain_restarts, run_engine_spmd  # noqa: F401

__all__ = [
    # comm
    "SPMD_MAX", "SPMD_MIN", "SPMD_SUM", "SPMD_PROD",
    "Comm", "Request", "SerialComm", "SERIAL", "ThreadComm", "Mpi4pyComm",
    "mpi_world_size",
    # domdec (Starter side)
    "DomainInfo", "Decomposition", "element_weights", "partition_elements",
    "check_spmd_support", "decompose", "slice_model", "domain_restart_path",
    "decomposition_table", "write_domain_restarts",
    # exchange (Engine side)
    "MSGOFF_EXCH_A", "MSGOFF_EXCH_V", "MSGOFF_COLLECT",
    "SpmdContext", "GlobMin", "serial_context",
    "crit_type_table", "crit_type_code", "crit_type_name",
    # driver
    "count_domain_restarts", "run_engine_spmd",
]
