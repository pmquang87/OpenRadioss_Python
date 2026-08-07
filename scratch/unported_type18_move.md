# Research Report: Fortran Implementation of `INTER/TYPE18` and `MOVE_FUNCT`

## 1. `INTER/TYPE18` (Fluid-Structure Interaction Contact)
**Location:** `C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\interfaces\int18\`

**Key Findings:**
*   **Purpose:** `TYPE18` is a Fluid-Structure Interaction (FSI) contact interface used to couple a Lagrangian structural surface to a fluid domain represented by ALE or Eulerian brick elements. It uses a penalty formulation.
*   **Key Files:**
    *   `i18for3.F`: This is the core explicit routine that computes reaction forces. It handles local coordinate interpolations on faces, computes the penetration (`PP1 = MAX(ZERO, GAP - D1)`), relative velocities, and applies penalty stiffness forces.
    *   `i18main_kine.F`: Contains an experimental kinematic formulation (`/INTER/TYPE18/KINE`) where structural velocity imposes fluid velocity. A comment in the Fortran source explicitly states this version is "abandoned" and "never released", meaning the standard penalty `TYPE18` is the primary focus.
*   **Mechanics:** Contact relies on projecting Lagrangian nodes onto ALE/Eulerian brick facets (triangles/quads), calculating normal vectors, tracking gap closure, and applying opposing forces on both the fluid and structure nodes based on a defined stiffness.

## 2. `MOVE_FUNCT` (`/IMPVEL` and `/IMPDISP` Kinematic Constraints)
**Location:** `C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\constraints\general\impvel\` and `...\bcs\`

**Key Findings:**
*   **Purpose:** In OpenRadioss, "MOVE_FUNCT" maps to the `/IMPDISP` (imposed displacement) and `/IMPVEL` (imposed velocity) keywords. These apply a user-defined time-history curve (`/FUNCT`) to specific nodal degrees of freedom.
*   **Key Files:**
    *   `fixvel.F`: Evaluates the explicit imposed velocity. It iterates over constrained nodes, interpolates the velocity curve at the current time `TT` (using `vinterdp` or `vinter_smooth`), and scales it by a factor `FACX`.
    *   `fixfingeo.F`: Evaluates imposed displacements (Fixed Geometry / `FGEO`). It computes target positions and derives the required velocities and accelerations to reach them.
    *   `fv_imp0.F` / `bc_imp0.F`: Handles the implicit solver matrix adjustments by forcing zeroes or modifying the global stiffness `K` matrix for the fixed degrees of freedom.
*   **Mechanics:** At every explicit cycle, the solver overrides the computed velocity or displacement for the targeted DOF with the function's value, zeros out invalid DOFs, and computes reaction forces.

---

## 3. Proposed Python Porting Plan

To implement these features faithfully in `OpenRadioss_Python`, the following phased plan is recommended:

### Phase 1: Kinematic Movements (`/IMPVEL` & `/IMPDISP`)
1.  **Starter/Input Parsing:**
    *   Extend `pyradioss/input/` to parse `/IMPVEL` and `/IMPDISP` block formats, mapping node groups to function (`/FUNCT`) objects.
    *   Support scaling factors (`Ascale`, `Fscale`).
2.  **Engine Kinematic Hook:**
    *   Create a new constraints module (e.g., `pyradioss.engine.constraints`).
    *   Inject an update step in the explicit time loop (mirroring `fixvel.F`) right after acceleration calculation.
    *   For constrained nodes, evaluate the function for the current time `t`, set the nodal velocity `V`, and back-compute the necessary reaction forces.
3.  **Energy Accounting:**
    *   Crucially, update the energy balance ledger. Imposed kinematics do external work on the system, which must be booked properly to prevent energy leak test failures.

### Phase 2: `TYPE18` Contact Foundation
1.  **Data Structures & Parsing:**
    *   Add `/INTER/TYPE18` to the starter readers. Collect `surf_ID`, `grbric_ID`, gap formulations (`I_gap`), and stiffness modes (`I_stf`).
2.  **Neighbor Search:**
    *   Implement an efficient spatial search (e.g., bounding volume hierarchies) to identify which Lagrangian nodes are near which ALE brick facets, as a naive $O(N^2)$ search will fail on performance tests.

### Phase 3: `TYPE18` Penalty Physics
1.  **Force Computation:**
    *   Port the math from `i18for3.F` into `pyradioss.contact.int18`.
    *   Vectorize the node-to-facet distance and penetration logic using `numpy`.
    *   Compute normal/shear penalty forces and scatter them back to the nodes (`numpy.add.at`).
2.  **Validation:**
    *   Use the `validation-compare` skill and `lspp-check` on a reference d3plot. Ensure Fortran parity for energy and T01 channel outputs. Run with `PYRADIOSS_BACKEND=numpy` to isolate physics differences from numba jitter.
