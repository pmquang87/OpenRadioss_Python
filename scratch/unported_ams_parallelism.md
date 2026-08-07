# Research Report: Advanced Mass Scaling (AMS) and MPI/SMP Parallelism in OpenRadioss

## 1. Advanced Mass Scaling (AMS)
### Fortran Implementation
- **Location**: `C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\ams\`
- **Key Files**: 
  - `sms_init.F`: Initializes AMS, determining which nodes and parts require artificial mass to raise the timestep above the minimum $\Delta t_{min}$.
  - `sms_build_mat_2.F`, `sms_build_diag.F`: Constructs the scaled, non-diagonal mass matrix and the preconditioner.
  - `sms_pcg.F`: Implements the Preconditioned Conjugate Gradient (PCG) iterative solver.
- **Mechanism**: Normally, explicit dynamics uses a lumped (diagonal) mass matrix, which is trivial to invert. AMS selectively adds non-diagonal mass terms to damp high-frequency modes without affecting the low-frequency physical response. Because the mass matrix is no longer diagonal, the momentum equation $[M] \{a\} = \{F\}$ becomes a coupled linear system that must be solved at every time step using the PCG solver.

### Porting Plan to Python (`pyradioss`)
- **Detection & Setup**: Implement logic to detect $\Delta t < \Delta t_{min}$ and identify target elements/nodes.
- **Solver Choice**: Python’s `scipy.sparse.linalg.cg` is the standard equivalent, but the overhead of calling Scipy inside the critical per-timestep inner loop would likely bottleneck performance.
- **Numba Preconditioned Conjugate Gradient**: A high-performance port requires writing a custom CSR-format PCG solver or a matrix-free PCG solver in Numba (`@njit`), mirroring the logic in `sms_pcg.F` to ensure parity and minimal overhead.

## 2. SMP Parallelism (Shared Memory)
### Fortran Implementation
- **Location**: Throughout `engine/source/` (e.g., `sms_build_mat_2.F`, `sms_pcg.F`).
- **Mechanism**: OpenRadioss uses OpenMP compiler pragmas (e.g., `!$OMP SINGLE`, `!$OMP PARALLEL DO`) for loop-level parallelism across elements and nodes. To prevent race conditions (e.g., when accumulating forces on a shared node from multiple adjacent elements concurrently), OpenRadioss uses element chunking/coloring or thread-local accumulators.

### Porting Plan to Python (`pyradioss`)
- **Tooling**: Python’s standard `multiprocessing` is ill-suited for fine-grained loop-level parallelism due to data serialization overhead. Instead, Numba’s `parallel=True` with `numba.prange` is the exact conceptual equivalent of OpenMP.
- **Race Condition Mitigation**: Just like the Fortran implementation, using `prange` over elements requires either thread-local NumPy arrays that are reduced after the loop, or adopting a node-coloring algorithm to ensure independent elements are processed concurrently without stepping on each other's nodal force accumulations.

## 3. MPI Parallelism (SPMD Distributed Memory)
### Fortran Implementation
- **Location**: `C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\mpi\`
- **Key Files**: `spmd_mod.F90`, `spmd_comm_world.F90`, `spmd_allreduce.F90`, `spmd_exch_sub.F`.
- **Mechanism**: Uses standard Message Passing Interface (MPI) for Single Program Multiple Data (SPMD) execution. The starter decomposes the mesh domain into $N$ partitions. Each MPI rank runs its own instance of the engine, computing forces on its local elements. Halo/boundary nodes are synchronized across ranks via `Allreduce`, `Isend`, and `Irecv`.

### Porting Plan to Python (`pyradioss`)
- **Tooling**: Use the `mpi4py` library. It offers zero-copy MPI bindings over contiguous memory like NumPy arrays.
- **Data Exchange**: Pass NumPy buffers directly to `mpi4py` operations (e.g., `comm.Irecv(buf, source)`). This avoids any Python serialization (pickle) overhead.
- **Launch Mechanism**: The engine entry point will need to detect if it's launched via `mpiexec -n <N> python -m pyradioss.engine` and handle the corresponding partition input files produced by a domain-decomposing starter step.
