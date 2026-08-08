"""
8-node hexahedral solid element, one-point integration with
Flanaganâ€“Belytschko hourglass control (/BRICK + /PROP/SOLID, Isolid=1).

Fortran origin: ``engine/source/elements/solid/solide/`` â€” the cycle path is

    sforc3.F   driver: gather coords/velocities, call the chain below
    srcoor3.F  geometry (Jacobian at the centroid, volume)
    sdefo3.F   velocity gradient  ->  rate of deformation D, spin W
    srota3.F   Jaumann rotation of the old stress by the spin increment
    smalla3.F / mmain.F   call the material law (SIGEPS..)
    sbulk3.F   bulk viscosity (shock damping) pressure
    shour3.F   hourglass (zero-energy mode) control forces
    sfint3.F   internal nodal forces  f_i = V * sigma . gradN_i
    sdlen3.F   characteristic length -> critical time step

Theory notes (kept close to Belytschko, Liu & Moran, "Nonlinear Finite
Elements for Continua and Structures", ch. 8, and Flanagan & Belytschko
IJNME 1981):

* One integration point at the element centroid: the strain field is
  evaluated with the "uniform gradient" B-matrix. Cheap and robust for
  crash/impact, but admits 12 zero-energy ("hourglass") deformation modes
  which must be stabilized â€” that is the role of shour3/this file's
  hourglass block.

* The **strain rate** is the symmetric part of the spatial velocity
  gradient L = dv/dx; the skew part W (spin) drives the **Jaumann
  objective rate**: rigid rotation must rotate the stress without changing
  it, so the stress update is
      sigma <- sigma + (W.sigma - sigma.W) dt   (rotation, srota3)
      sigma <- sigma + C : D dt                 (material law, sigeps)

* **Bulk viscosity** (sbulk3): explicit codes smear shocks over a few
  elements by adding a viscous pressure in compression
      q = rho * lc * (qa^2 * lc * trD^2 - qb * c * trD),   trD < 0
  (qa quadratic, qb linear coefficient â€” /PROP/SOLID defaults 1.1, 0.05).

* **Critical time step** (sdlen3): Courant condition on the P-wave,
      dt = lc / (Q + sqrt(Q^2 + c^2)),   Q = qb*c + qa*lc*|trD^-|
  with lc = V / A_max the volume over the largest face area (a safe
  generalization of "smallest height" to distorted hexas).

  IMPORTANT refinement over the textbook estimate: lc/c is NOT a strict
  bound for the one-point hexa â€” the exact maximum eigenfrequency of the
  element (a free single cube, nu = 0.3) is ~1.36x higher than 2c/lc, so
  a run at 0.9 * lc/c can be genuinely unstable (we verified this both by
  eigenanalysis and by watching round-off grow in a rigid-body-motion
  test). The element stiffness is K = V B^T C B with a CONSTANT B, so
  its nonzero eigenvalues are those of the 6x6 matrix C.(B B^T) â€” cheap
