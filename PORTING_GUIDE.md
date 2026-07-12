# Porting guide — OpenRadioss (Fortran) → `pyradioss` (Python)

This document is the working map of the port: how the original source tree
translates to this repository, what is already ported, the conventions used,
and the roadmap for the remaining work. **Read this first if you plan to
extend the code.**

---

## 1. Architecture recap

OpenRadioss is two programs sharing data through a *restart file*:

```
            MYRUN_0000.rad                MYRUN_0001.rad
                 │                              │
                 ▼                              ▼
   ┌──────────────────────┐   restart   ┌──────────────────────┐
   │        STARTER        │ ─────────► │        ENGINE         │
   │  read + check + init  │  .rst      │  explicit time loop   │
   └──────────────────────┘             └──────────────────────┘
        MYRUN_0000.out                MYRUN_0001.out, T01, ANIM
```

The Engine's heart (`engine/source/engine/resol.F`, ~5000 lines) performs, per
cycle: element internal forces → contact forces → external loads → kinematic
conditions → acceleration/velocity/position update → time-step computation →
output. `pyradioss/engine/engine.py` reproduces exactly this sequence with the
same names in comments.

## 2. Source-tree map

| OpenRadioss (Fortran) | pyradioss (Python) | Notes |
|---|---|---|
| `starter/source/starter/lectur.F` (deck reading driver) | `pyradioss/starter/starter.py` | orchestration |
| `starter/source/reader/*` + `hm_reader` (Altair reader lib) | `pyradioss/input/deck_reader.py` | block/keyword lexer, `#include` |
| `starter/source/elements/reader` per-keyword `hm_read_*.F` | `pyradioss/input/starter_keywords.py` | one function per keyword |
| Fortran derived types / common blocks (`common_source/modules`) | `pyradioss/model/*.py` | dataclasses + NumPy arrays |
| `starter/source/initial_conditions`, `inimass` etc. | `pyradioss/starter/initialization.py` | lumped mass, volumes |
| restart write `starter/source/restart/ddsplit/wrrest.F` | `pyradioss/starter/restart.py` | pickle instead of binary |
| `engine/source/engine/resol.F` | `pyradioss/engine/engine.py` | main loop |
| `engine/source/engine/lectur.F` + `hm_read_*` (engine cards) | `pyradioss/input/engine_keywords.py` | `/RUN /DT /TFILE /ANIM …` |
| `engine/source/assembly/asspar*.F` | `pyradioss/engine/engine.py` (`np.add.at` scatter) | force assembly |
| `engine/source/constraints/general/bcs` + `impvel/fixvel.F` | `pyradioss/engine/kinematics.py` | `/BCS`, `/IMPVEL`, `/IMPDISP` |
| `engine/source/constraints/general/rwall` (`rgwal0/s/c/t.F`) | `pyradioss/engine/rigid_wall.py` | kinematic wall: plane/sphere/cylinder, moving (M5) |
| `starter/.../rbody/hm_read_rbody.F`, `rbyini.F` + `engine/.../rbody/rbyfor.F`, `rbycor.F` (and `rbe2/`) | `pyradioss/engine/rigid_body.py` + `starter/initialization.py` (`initialize_rigid_bodies`) | `/RBODY`, `/RBE2` (M5) |
| `starter/.../rbe3/hm_read_rbe3.F` + `engine/.../rbe3/rbe3f.F`, `rbe3v.F` | `pyradioss/engine/rbe3.py` | `/RBE3` interpolation constraint (M5) |
| `engine/source/loads/general/pload/pload.F` | `pyradioss/engine/kinematics.py` (`external_forces`) | `/PLOAD` follower pressure (M5) |
| `engine/source/tools/sect/` (`section.F`, `forint.F`) | `pyradioss/engine/sections.py` | `/SECT` via the side-sum identity (M5) |
| `starter/source/tools/admas/` | `pyradioss/starter/initialization.py` | `/ADMAS` (M5) |
| `engine/source/elements/solid/solide/` (`sforc3.F`, `srota3.F`, `shour3.F`…) | `pyradioss/elements/solid_hexa8.py` | 1-pt + FB hourglass |
| `engine/source/elements/solid/solide4/` (`s4forc3.F`…) | `pyradioss/elements/solid_tetra4.py` | constant-strain tetra |
| `engine/source/elements/shell/coque/` (`cforc3.F`, `czforc3.F`…) | `pyradioss/elements/shell_bt4.py` | Belytschko–Tsay, BLT84 stiffness hourglass |
| `engine/source/elements/sh3n/coque3n/` (`c3forc3.F`…) | `pyradioss/elements/shell_tri3.py` | C0 triangle |
| `engine/source/elements/beam/` (`pforc3.F`, `pdefo3.F`…) | `pyradioss/elements/beam_type3.py` | corotational Timoshenko |
| `engine/source/elements/truss/` (`tforc3.F`) | `pyradioss/elements/truss.py` | |
| `engine/source/elements/spring/` (`rforc3.F`) | `pyradioss/elements/spring.py` | TYPE4 |
| `engine/source/materials/mat/mat001/sigeps01.F` | `pyradioss/materials/law01_elastic.py` | |
| `engine/source/materials/mat/mat002/sigeps02.F` | `pyradioss/materials/law02_johnson_cook.py` | |
| `engine/source/materials/mat/mat027/sigeps27c.F` | `pyradioss/materials/law27_brittle.py` | shells only, like the original |
| `engine/source/materials/mat/mat036/sigeps36.F` (+ `36c`) | `pyradioss/materials/law36_tabulated.py` | |
| `engine/source/materials/mat/mat042/sigeps42.F` | `pyradioss/materials/law42_ogden.py` | solids; returns its own SOUNDSP |
| `engine/source/materials/fail/johnson_cook/`, `fail/biquad/` | `pyradioss/failure/` | /FAIL cards + GBUF%OFF element deletion |
| `engine/source/elements/beam/pmat3.F` (global plasticity) | `pyradioss/elements/beam_type3.py` | LAW2 resultant-space return |
| `engine/source/interfaces/int07/` (`i7dst3.F`, `i7for3.F`) + `intsort/i7buce.F` | `pyradioss/contact/inter_type7.py` | penalty node↔segment, voxel broad phase |
| `engine/source/interfaces/int02/` (`i2for3.F`, `i2vit3.F`) | `pyradioss/contact/inter_type2.py` | tied contact (kinematic) |
| `engine/source/interfaces/int11/` (`i11dst3.F`, `i11for3.F`) | `pyradioss/contact/inter_type11.py` | penalty edge↔edge |
| `starter/source/interfaces/inter3d1/` (`i7sti3.F`, `i11sti3.F`, gap setup) | `pyradioss/contact/stiffness.py` | element-based penalty stiffness + gaps |
| interface `IDEL` bookkeeping vs `GBUF%OFF` | `pyradioss/contact/tracking.py` | deleted elements drop out of contact |
| `starter/source/model/sets/hm_read_lines.F` (IGRSLIN) | `pyradioss/starter/initialization.py` (`resolve_lines`) | /LINE edge sets |
| `engine/source/output/` (`ecrit.F`, `sortie_main.F`, TH, ANIM) | `pyradioss/output/*.py` | CSV + VTK |
| `engine/source/time_step/` (`dtnoda.F`, STIFN accumulation) | `pyradioss/engine/mass_scaling.py` | `/DT/NODA[/CST]` nodal dt + mass scaling (M6) |
| `engine/source/output/restart/` (`wrrestp.F`, `rdresb.F`) | `pyradioss/starter/restart.py` + engine resume | engine restarts, `_0002.rad` chaining, /STATE (M6) |
| `engine/source/assembly/damping*.F` | `pyradioss/engine/damping.py` | `/DAMP` mass damping (M6) |
| `starter+engine/source/tools/sensor/` | `pyradioss/engine/sensors.py` | `/SENSOR/TIME`, `/SENSOR/DISP` (M6) |
| `starter+engine/source/constraints/general/mpc/` | `pyradioss/engine/mpc.py` | `/MPC` Lagrange treatment (M6) |
| `starter/source/materials/eos/` + `engine/source/materials/eos/eosmain.F` | `pyradioss/materials/eos.py` + solid kernels | `/EOS` polynomial & ideal gas, implicit E-p (M6) |
| `common_source/` (constants, tables) | `pyradioss/common/*.py` | |
| — (OpenRadioss speed = compiled Fortran + OpenMP/MPI, out of scope) | `pyradioss/accel/` | M7: optional numba backend behind the same kernel API (see the package docstring for architecture + parity contract) |
| — | `pyradioss/common/fastmath.py` | M7: small-array NumPy primitives (bitwise-documented replacements for np.cross / norm / det / inv / add.at) |
| — | `tools/benchmark.py` | M7: NumPy vs numba wall-clock benchmark over the examples |
| `engine/source/implicit/ind_glob_k.F` (equation numbering) | `pyradioss/implicit/dofmap.py` | M8: assign each free nodal DOF an index; /BCS-fixed DOFs condensed (removed, not penalized); shell rotations numbered where they carry stiffness |
| `engine/source/implicit/imp_glob_k.F` / `imp_fsa_inv.F` (sparse assembly) | `pyradioss/implicit/assembly.py` | M8: element tangents → COO triplets → scipy CSR (scipy guarded inside the package) |
| `engine/source/implicit/imp_solv.F` (implicit driver + Newton loop) | `pyradioss/implicit/statics.py` | M8: load stepping, residual R = f_ext − f_int (reusing the explicit kernels), K Δu = R, convergence norms, iteration cap |
| `engine/source/implicit/imp_dsolv*.F` (direct-solver interface) | `pyradioss/implicit/linsolve.py` | M8: `solve(K,R)` behind the M7 backend pattern — SuperLU default, optional CHOLMOD / MUMPS wrapped (not ported) with fallback |
| element `KE` routines (e.g. `s8eoff.F` / shell `cmalpha`) + material `TANGENT` | `solid_hexa8.tangent`, `shell_bt4.tangent`, `materials.*_tangent` | M8: element tangent stiffness + the LAW1 elastic and LAW2 CONSISTENT (algorithmic) tangents (a NEW method alongside `forces()`; does not perturb the M7 force path) |
| `engine/source/input/*` `/IMPL*` cards | `pyradioss/input/engine_keywords.py` (`/IMPL`) | M8: minimal implicit-static control (final load factor, increment size, Newton tolerances, linear-solver choice); M9: `/IMPL/NONLIN[/SMDISP]` + `/IMPL/ARCL` |
| `imp_solv.F` /IMPL/NONLIN branch (updated-Lagrangian step) + the `imp_kgeo` geometric-stiffness assembly in `imp_glob_k.F` | `statics.py` (nlgeom path) + `kgeo()` in `solid_hexa8` / `shell_bt4` / `truss` | M9: committed frame advances per increment; residual = midpoint (Hughes–Winget) stress update + end-configuration assembly (`static_internal_forces`); tangent += K_geo = ∫G^T[σ]G dV at the trial geometry |
| the arc-length continuation of the implicit nonlinear driver (`imp_solv.F` family) | `statics.py` (`_run_arclength`) | M9: spherical Riks/Crisfield constraint, root selection by path continuation, adaptive radius, sign-following predictor; `/IMPL/ARCL` |
| `engine/source/implicit/imp_buck.F` (/IMPL/BUCKL buckling eigensolver) | `pyradioss/implicit/buckling.py` | M9: (K_mat + μ K_geo)φ = 0 generalized eigenproblem on a pre-stressed state (library function; the engine card itself deferred) |
| `engine/source/implicit/imp_dyna.F` (implicit dynamics: DYNA_INI scheme setup, DYNA_INA initial acceleration, IMP_DYNAM effective-stiffness diagonal, IMP_DYNAR/IMP_FHHT dynamic residual + HHT weighting, INTE_DYNA a/v recovery, DYNA_WEX work ledger) + the /IMPL/DYNA read of `engine/source/input/freimpl.F` | `pyradioss/implicit/dynamics.py` + `/IMPL/DYNA` in `engine_keywords.py` | M10: Newmark-β/HHT-α implicit time integration on top of the M8/M9 statics core — lumped M (model.mass + model.inertia) condensed through the DofMap, R = (1+α)(f_ext+f_int)ₙ₊₁ − α(…)ₙ − M aₙ₊₁, K_eff = (1+α)K_T + M/(β dt²), both geometry modes; rate devices disabled explicitly (see the module docstring) |
| the `IDY_DAMP` branch of `imp_dyna.F` (`IMP_DYKS`/`IMP_DYKV` damping force DY_DAM = a·M·v + b·K·v with the step-start K, the IDY_DAMP branch of `IMP_DYNAM`, the DY_EDAMP ledger of `DYNA_WEX`) + the /IMPL/DYNA/DAMP read of `freimpl.F` | `implicit/dynamics.py` (damping blocks) + `/IMPL/DYNA/DAMP` in `engine_keywords.py` | M11: Rayleigh damping C = a·M + b·K in the implicit system — damping force HHT-weighted in the residual, (1+α)γ/(βdt)·C in K_eff, trapezoidal dissipation booked into its own `edamp` ledger channel |
| `engine/source/implicit/imp_dt.F` (`IMP_DTN`: cut on IMCONV < 0 bounded by DT_MIN, IDTC = 1 growth toward DT_MAX after ≤ NL_DTP iterations) + the /IMPL/DT reads of `freimpl.F` | `implicit/statics.py` (`StepControl`) + `/IMPL/DT/STOP`, `/IMPL/DT/1` in `engine_keywords.py` | M11: automatic implicit step control for BOTH statics load increments and dynamics time steps — always on, the cards tune it; IDTC 2/3 deferred |
| `engine/source/implicit/imp_buck.F` engine-card path (the /IMPL/BUCKL read of `freimpl.F`, the "BUCKLING MODES COMPUTATION" listing block) | `statics.py` (`_run_buckling`) + `/IMPL/BUCKL/1|2` in `engine_keywords.py` | M11: the card runs the static prestress increments then the M9 eigensolver, reports factors/modes in the listing and on `implicit_result` |
| element `KE` routines of the remaining families (solide4, coque3n/sh3n, beam, spring) + their `imp_kgeo` branches | `tangent()`/`kgeo()`/`static_internal_forces()` in `solid_tetra4`, `shell_tri3`, `beam_type3`; `tangent()`/`kgeo()`/`implicit_internal_forces()` in `spring` and the truss's `implicit_internal_forces` | M11: element tangent COMPLETENESS — every element family of the port is implicit-capable (the spring/truss get their own implicit residual: total-form / iterated-return, see the module notes) |
| `engine/source/constraints/general/rbody/rby_imp0.F` (RBY_IMP1/IMPR1/IMPR2), `rbe2/rbe2_imp0.F`, `rbe3/rbe3_imp0.F`, `engine/source/interfaces/interf/i2_imp1.F` (I2UPDK0) — the constraint condensations `imp_solv.F`/`imp_dyna.F` call around every assembly | `pyradioss/implicit/constraints.py` | M12: /RBODY, /RBE2, /INTER/TYPE2 tied, /RBE3 and /MPC in the implicit system by CONDENSATION — one sparse transform `u_full = T u_red`, `K_red = T^T K T`, `R_red = T^T R` (the *_IMP1/*_IMPR1 block algebra; dependent DOFs eliminated, never penalized); rebuilt per committed frame + exact Rodrigues re-placement under /IMPL/NONLIN; `T^T M T` = the exact rigid 6-DOF mass at the master under /IMPL/DYNA |
| `engine/source/implicit/imp_int_k.F` (IMP_INT_K forcing IMP_INT7 = 3) + `engine/source/interfaces/int07/i7ke3.F` / `i7keg3.F` (the per-pair contact stiffness blocks) | `pyradioss/implicit/contact.py` | M12: /INTER/TYPE7 penalty contact in the Newton loop — frictionless force at the trial configuration in the residual, exact gap tangent K·g gᵀ **plus the closest-point curvature −K·p·∇²d** (the original assembles the n nᵀ blocks only) in K/K_eff, active set re-evaluated every iteration, i7sti3 stiffness/gap machinery (`contact/stiffness.py`) reused unchanged |
| the FRIC blocks of `i7keg3.F` (I7KEG3's µ-scaled tangential-plane spring, I7FRF3's tangential increment spring, **I7KFOR3's incremental return mapping** — `BETA = MIN(ONE, XMU*SQRT(FN/FT))` on the anchored force CAND_F + STIF0·D) | `implicit/contact.py` (friction blocks of `ImplicitContact7`) | M13: Coulomb friction in the implicit loop — the incremental tangential RETURN MAPPING at the frozen projection (stick = tangential penalty spring K_t = K on the slip increment, slip = radial return to the cone) with the CONSISTENT nonsymmetric tangent µK·t nᵀ + (µf_n K/\|f_tr\|)(I − nnᵀ − ttᵀ) instead of I7KEG3's always-stick µK(I−nnᵀ) spring; anchors committed once per converged increment (the CAND_F save), slip work in the `efric` dynamics ledger channel |
| `engine/source/interfaces/int11/i11ke3.F` / `i11keg3.F` (I11KE3/I11KEG3 — the TYPE11 edge-to-edge implicit stiffness; IMP_INT7 = 3 forced around it too) | `implicit/contact.py` (`ImplicitContact11`) | M13: /INTER/TYPE11 under implicit — penalty force at the segment-segment closest points in the residual (the explicit i11dst3 port reused read-only), exact gap tangent K·g gᵀ (g from the two edges' parameters — the original's HS·HM blocks are its diagonal-boosted variant) plus the EXACT edge-edge closest-point curvature (the 2×2 optimality-system linearization, every projection region); near-parallel overlaps get a two-point trapezoid quadrature (the single closest point of parallel edges is non-unique and flips ends — a period-2 Newton cycle, the M13 lesson) |
| `engine/source/implicit/imp_glob_k.F` — `IMP_KPRES` + `KPQUAD`/`KPTRIA` (pressure load stiffness: Gauss-integrated, only the OFF-diagonal blocks, forced antisymmetric, scaled SCALN = HALF) | `pyradioss/implicit/followerload.py` | M13: /PLOAD follower-load stiffness under /IMPL/NONLIN — the port linearizes ITS OWN lumped area-vector pressure force EXACTLY (nonsymmetric −∂(p·w·½ d13×d24)/∂x per segment; documented deviation: Newton cares about consistency with the residual actually iterated), and the NLGEOM residual now evaluates /PLOAD at the TRIAL configuration |
| `engine/source/materials/mat/mat036/sigeps36.F` (+ `36c`) — the return the tangent must be consistent with | `materials/law36_tabulated.py` (`consistent_solid_tangent` / `consistent_shell_tangent`) | M13: LAW36 consistent tangents — the LAW2 algorithmic-tangent algebra with H from the TABLE's local segment slope; the piecewise-linear return is already EXACT at implicit increment sizes (measured — no iterated-return upgrade needed); rate-curve families truncated to the static curve under implicit (warned) |
| `starter/source/constraints/general/rbody/rbody_part_modif.F90` (the PARENT_OF hierarchy resolving nested part-rigid-bodies in the Starter) | `implicit/constraints.py` (`_fully_resolve` + the topological `commit_placement`) | M14: constraint CHAINS — a master DOF of one constraint DEPENDENT in another (rigid-on-rigid, /MPC on rigid slaves, /RBE3 masters in a body, ties on body slaves) resolved by transform SUBSTITUTION (= the product T = T1·T2·… in topological order, never formed); CIRCULAR chains refused as a DFS back-edge; conflicts (one DOF, two rows) still refused; the explicit engine refuses chains loudly (its per-body integrator has no nesting order — the original's engine never sees one) |
| the FRIC blocks of `i11keg3.F` (I11KEG3's always-stick µ-scaled plane spring; I11KFOR3's UNCAPPED tangential spring `FTN = -FRIC*STIF*DXT` — no CAND_F anchor, no cone) | `implicit/contact.py` (friction blocks of `ImplicitContact11`) | M14: TYPE11 Coulomb friction under implicit — the TYPE7/I7KFOR3 incremental RETURN MAPPING generalized to edge pairs (slip = relative motion of the closest MATERIAL points at frozen parameters, tangential plane ⊥ n — which CONTAINS both edge directions at a crossing), consistent stick/slip tangents on the [(1-s), s, -(1-t), -t] pattern, anchors keyed 2*(pair)+overlap-end (the documented near-parallel keying); a deliberate deviation from the original's uncapped spring, same rationale as M13's TYPE7 |
| `imp_solv.F`'s Riks machinery (IDTC = 3 of `imp_dt.F` + the BFAC load rescaling; arc metric UL2 = full-field `PRODUT_UHP0` norm AFTER the dependent-motion recovery) | `statics.py` (`_run_arclength` / `_solve_increment_arc` / `_arc_tangent`) | M14: /IMPL/ARCL WITH constraints & contact — both auxiliary solves, the spherical metric and the root selection in the REDUCED space (documented deviation: the original's metric is the full recovered field — the two differ by the fixed SPD reweighting TᵀT, both valid Crisfield parametrizations); contact force in the corrector residual + active-set tangent in both solves, anchors/placement committed per arc increment |
| `imp_buck.F` (UPD_GLOB_K condensing BOTH matrices before EIGBUCKP, RECUKIN mode recovery; NO contact assembly — NDDLI7 forced 0; IMP_KPRES into KG) | `implicit/buckling.py` | M14: /IMPL/BUCKL WITH constraints & contact — the reduced pencil Tᵀ(K_mat)T + µ Tᵀ(K_geo)T exactly as the original; the CONVERGED contact active set's tangent added to K_MAT (documented deviation: the original omits contact entirely — a column resting on a stop would report the free-column factor; the stop's stiffness does not scale with µ, so K_geo would be wrong), friction blocks symmetrized for eigh, frozen-set/bilateral caveats documented |
| `sigeps42.F`'s implicit branch (IMPL_S > 0: a SCALAR stiffness ratio ET scaling a linear elastic D — a secant modified Newton) | `materials/law42_ogden.py` (`consistent_solid_tangent`) + the total-form re-evaluation in `solid_hexa8/tetra4.static_internal_forces` | M14: LAW42 CONSISTENT tangent — the exact spectral SPATIAL elasticity of the port's own Ogden stress (Bonet & Wood §6.6/ch.8: c_aabb from ∂²W/∂lnλ², the (σ_a λ_b² − σ_b λ_a²)/(λ_a² − λ_b²) shear terms, the equal-stretch L'Hôpital limit — in the shifted convention (c_aaaa − c_aabb)/2 WITHOUT the extra −σ_a of the textbook form, verified by the coalescent FD identity); the NLGEOM residual re-evaluates total-form stress at the END configuration; /IMPL/NONLIN REQUIRED (refused on the frozen frame — F never sees the trial displacement there) |
| the MFROT µ(p, v) blocks + IFQ filtering of `i7for3.F` ("Friction coefficient computation" / "TANGENT FORCE CALCULATION"), the `Ifric/Ifiltr/Xfreq/C1–C6` reads + XFILTR mapping of `hm_read_inter_type07.F` | `pyradioss/contact/friction.py` (laws + filter) wired into `inter_type7.py`/`inter_type11.py` (explicit) | M15: friction MODELS — MFROT 1 (generalized viscous polynomial), 2 (Darmstadt), 3 (Renard piecewise), 4 (exponential decay) with p = f_n/AREA(main segment) and the EM30 floor; IFQ 1/2/3 first-order (CAND_F EMA) filter with the reader's exact XFILTR mapping (IFQ 3's per-cycle α = min(1, XFILTR·dt) — the DOCUMENTED deviation from the fetched source's `MAX(ONE, …)`, which disables the filter it was asked for); TYPE11 = a documented PORT EXTENSION (the original's i11mainf.F forces MFROT = 0 — checked) with the edge pressure DEFINED p = f_n/(L_main·gap); Ifric = 0 bit-identical, numba mirror untouched (the models live downstream of the mirrored narrow phase) |
| the MFROT/IFQ blocks of `i7keg3.F`'s I7KFOR3 (µ(p, v) fed the increment pseudo-rates) + I7KEG3's `FACT(I)=FRIC` always-stick spring (constant µ in the matrix even when MFROT > 0) | `implicit/contact.py` (`_cone` + the µ_t coupling in both classes' `_friction_state`/`triplets`) | M15: friction MODELS under implicit — the Coulomb cone radius becomes µ(p)·f_n at the STATIC LIMIT µ(p, v=0) (`friction.mu_static`; rate devices reduce loudly, never fed du/1 — the original feeds I7KFOR3's µ law the increment fields, a step-size-dependent pseudo-rate the port deliberately does not reproduce), and the slip tangent's t nᵀ block carries the derived µ_t = µ + f_n µ′(p)/A coupling slope (frozen area, the frozen-weight class) — a documented deviation from I7KEG3's constant-FRIC spring, the M13 IMP_KPRES pattern; IFQ ignored with a warning (a time device: its DC limit is the unfiltered force); mfrot = 0 bit-identical to M13/M14 |
| — (no LAW27 implicit tangent exists in OpenRadioss: the law is an explicit crash material) | `materials/law27_brittle.py` (`consistent_shell_tangent`) + the `extra` hook of `materials.shell_layer_tangent` and both shell kernels' tangent layer loops | M15: LAW27 CONSISTENT shell tangent — the exact derivative of the port's own fixed-crack unilateral law per branch: uncracked = elastic C; open + FROZEN damage = the (1−d) secant rows; open + GROWING damage = plus the softening −cps(en_i+ν en_j)·dd_i/den_i and the shear-row −G g12 dd_i terms (derived, not bounded — the M12 lesson; nonsymmetric like every softening tangent); CLOSED crack = full elastic rows (the unilateral switch — the M13 line search is the non-smooth backstop); broken = zero. Assembled in the frozen crack frame, rotated with the Voigt pair Tεᵀ C Tε |
| the LAW2 global resultant-plasticity return of the beam (the port's M3 model; note the ACTUAL `pmat3.F` is pke3.F's ELASTIC shear-stiffness setup — the original's implicit beam KE has no plasticity linearization to mirror) | `elements/beam_type3.py` (`tangent` LAW2 branch + `implicit_internal_forces`) | M15: LAW2 BEAM consistent tangent — the algorithmic derivative of the radial resultant return, C_alg = s·C + [(H/(E+H) − s)/seq_tr]·R_tr (qᵀC), built from the POST-return state via the homogeneity identities (seq degree-1, q degree-0 ⇒ seq_tr = sy + E·dλ, R_tr = R·seq_tr/sy); the implicit residual runs its own ITERATED consistency solve (`implicit_internal_forces`, the M11 truss lesson MEASURED again in resultant space: 5 iterations are exact at n = 0.5 but leave O(1) residual at n = 0.2 virgin yield; the explicit kernel keeps its bit-identical 5); elastic beams route through the new hook bit-identically |
| — (NO modal-superposition, frequency-response or harmonic path exists in the open-source engine: `engine/source/input/freimpl.F` reads only DYNA / BUCKL / DT / NONLIN / ARCL — OpenRadioss is a time-domain crash/impact code) | `pyradioss/implicit/modal_response.py` + `/IMPL/MODAL/DYNA`, `/IMPL/FREQ` in `engine_keywords.py` | M17: MODAL-SUPERPOSITION dynamics — consumes the M16 real eigenpairs (u = Σφᵢqᵢ, each qᵢ a decoupled damped SDOF q̈ᵢ + 2ζᵢωᵢq̇ᵢ + ωᵢ²qᵢ = φᵢᵀf). MODAL DAMPING (uniform ζ / (freq,ζ) table / the Rayleigh map ζᵢ = ½(α/ωᵢ + βωᵢ) — the SAME α,β as the M11 direct /IMPL/DYNA/DAMP, so both solvers carry identical physical damping); MODAL TRANSIENT (the EXACT Nigam–Jennings piecewise-linear-forcing recurrence per mode — no dispersion, unlike the M10 direct Newmark — with an optional mode-ACCELERATION / residual-flexibility static correction K⁻¹−Σφφᵀ/ωᵢ² for the truncated tail); HARMONIC / FREQUENCY RESPONSE (the complex FRF qᵢ(Ω) = φᵢᵀF/(ωᵢ²−Ω²+2iζᵢωᵢΩ), swept, amplitude/phase, base-excitation feeding the M16 effective-mass participation). PORT cards, library-first exactly like M16's /IMPL/EIGV; the M10 Newmark integrator and the M16 eigensolver stay bit-identical (modal superposition is a NEW parallel path) |
| — (NO complex/damped eigensolver, NO assembled viscous C beyond the on-the-fly Rayleigh of imp_dyna.F, NO state-space / QEP path anywhere in the open-source engine — the frequency domain is not part of the time-domain solver) | `pyradioss/implicit/damping_matrix.py` + `pyradioss/implicit/complex_modal.py` + `/IMPL/CEIGV` in `engine_keywords.py`; `spring.damping_matrix` | M18: COMPLEX / DAMPED eigenvalues + NON-CLASSICALLY-damped complex-mode superposition. ASSEMBLED C (the C analogue of M16's `assemble_mass`) = Rayleigh αM + βK (M11-consistent) + discrete dashpots (the /PROP/SPRING `c` term c·aaᵀ, revived from its M11 implicit deferral; per-node /DAMP mass dampers), COO→CSR through the DofMap, condensed TᵀCT. COMPLEX EIGENVALUES: the QEP (λ²M + λC + K)φ = 0 via the SYMMETRIC state-space linearization A z = λB z, A = [[0,K],[K,C]], B = [[K,0],[0,−M]], `scipy.linalg.eig` on the reduced pencil → λᵢ = −ζᵢωᵢ ± iωd,ᵢ (decay + damped freq) and COMPLEX mode shapes (the DOF phase lag). COMPLEX-MODE SUPERPOSITION: the state-space decoupling ẋᵢ = λᵢxᵢ + pᵢ(t) (first-order exact recurrence) + the damped complex FRF (matches (K−Ω²M+iΩC)⁻¹ on the full basis). PORT card, library-first exactly like M16/M17; the M10 integrator, M16 REAL eigensolver and M17 REAL-mode superposition stay bit-identical (a NEW parallel path) |
| — (NO frequency-domain / random-vibration / PSD / response-spectrum path anywhere in the open-source engine — `freimpl.F` re-read line by line for M19: only DYNA / BUCKL / DT / NONLIN / ARCL + solver housekeeping, the sole `PSD` token being `IMUMPSD`, a MUMPS flag) | `pyradioss/implicit/random_response.py` + `pyradioss/implicit/response_spectrum.py` + `/IMPL/PSD`, `/IMPL/RSPEC` in `engine_keywords.py` | M19: RANDOM / SPECTRAL (PSD) response + RESPONSE SPECTRA — the stochastic and envelope analyses that BUILD on the M17 real-mode FRF and the M18 complex FRF. RANDOM (PSD): the stationary response PSD S_uu(Ω) = H(Ω)S_ff(Ω)H(Ω)* through the modal transfer function (the M17 real FRF for classical damping, the M18 complex FRF for non-classical — FRF-source-agnostic), reporting the RMS σ_u = √m₀, the spectral moments m₀/m₁/m₂ (mₙ = (1/π)∫₀^∞ Ωⁿ S_uu dΩ, the Wiener–Khinchin / task ∫S dΩ/2π convention), the mean zero-crossing rate ν₀ = (1/2π)√(m₂/m₀) and peak rate ν_p = (1/2π)√(m₄/m₂) (Rice), plus the base-excitation (support-motion) PSD feed via the M16 participation. RESPONSE SPECTRA: the per-mode peaks rᵢ = Γᵢ Sa(ωᵢ,ζᵢ)/ωᵢ² φᵢ (participation-scaled spectral ordinate) combined by SRSS and CQC (Der Kiureghian 1981 closed-form ρᵢⱼ). PORT cards, library-first exactly like M16/M17/M18; the M10 integrator, M16/M17/M18 paths stay bit-identical (a NEW parallel path consuming their FRFs read-only). Theory: Newland; Wirsching/Paez/Ortiz; Vanmarcke; Chopra ch. 13; Der Kiureghian 1981 |
| — (NO frequency-domain / spectral-fatigue solver anywhere in the open-source engine — `freimpl.F` re-read line by line for M20: no /FATIG, no S-N / Miner branch, no Dirlik / rainflow / narrow-band estimator; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag) | `pyradioss/implicit/spectral_fatigue.py` + the stress-recovery section of `pyradioss/implicit/random_response.py` + `/IMPL/FATIG` in `engine_keywords.py` | M20: RANDOM-VIBRATION (SPECTRAL) FATIGUE — the stress-life damage estimate computed DIRECTLY from the M19 stress-PSD spectral moments. STRESS-PSD RECOVERY: the per-mode stress modes σᵢ = (element stress operator)·φᵢ recovered by running the SAME force kernels the M8 static solve uses on each mode shape (read-only, σ = C:B:u on the zero-stress reference), the stress FRF H_σ(Ω) = Σᵢ σᵢ qᵢ(Ω) and the stress response PSD S_σσ(Ω) = \|H_σ\|² S_ff with its own moments m₀..m₄. DAMAGE MODELS for an S-N curve N = C·S^−m under a Miner sum: NARROW-BAND (Bendat 1964, the closed-form Rayleigh-range Γ(1+m/2) damage), DIRLIK (1985, the empirical three-term rainflow-range PDF from m₀,m₁,m₂,m₄), WIRSCHING–LIGHT (1980) and TOVO–BENASCIUTTI (2005) — the damage rate, equivalent stress and time-to-failure, plus a seeded Monte-Carlo rainflow (ASTM E1049) cross-check. PORT card, library-first exactly like M16–M19; the M10 integrator and the M16/M17/M18/M19 paths stay bit-identical (a NEW parallel path consuming the stress moments read-only). Theory: Bendat 1964; Dirlik 1985; Wirsching & Light 1980; Benasciutti & Tovo 2005; Newland; Bishop & Sherratt (NAFEMS); Palmgren–Miner; ASTM E1049 |
| — (NO frequency-domain / spectral-fatigue solver of ANY kind — scalar OR multiaxial — anywhere in the open-source engine — `freimpl.F` re-read line by line for M21: no /FATIG, no von-Mises / critical-plane / cross-PSD machinery; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag) | `pyradioss/implicit/multiaxial_fatigue.py` + the per-element Voigt-block helpers & `_run_multiaxial` driver in `pyradioss/implicit/random_response.py` + `/IMPL/FATIG/MULT` in `engine_keywords.py` | M21: MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE — the damage estimate of a MULTIAXIAL stress STATE, the natural consumer of the M20 scalar-channel fatigue and vector stress FRF. STRESS-TENSOR CROSS-PSD: the M20 vector stress modes give the FULL 6-Voigt stress FRF H_σ(Ω) (nf,6) per solid/shell element (read-only); the stress-tensor cross-PSD S_σσ(Ω) = H_σ S_ff H_σᴴ (a 6×6 Hermitian matrix per frequency per element) — its diagonal = the M20 per-component channel PSDs, uniaxial → the M20 scalar σₓₓ. EQUIVALENT-STRESS METHODS: reduce the 6×6 to a scalar and run the M20 estimators — the frequency-domain EQUIVALENT VON MISES S_vm = trace(Q·S_σσ) (Preumont & Piéfort 1994 / Pitoiset & Preumont 2000, the von Mises quadratic operator Q; rank-1 form HᴴQH), plus MAX-NORMAL and MAX-SHEAR CRITICAL PLANE (Carpinteri–Spagnoli / Cristofori–Susmel–Tovo, the projection-by-direction spectral form σₙ=pₙᵀσ, τ=pₛᵀσ over a searched candidate-plane set), reporting the critical-plane damage/equivalent-stress/life. A MULTIVARIATE Monte-Carlo cross-check (Cholesky/eigen of S_σσ → correlated Gaussian component histories → project onto the critical plane → ASTM E1049 rainflow → Miner, seeded). PORT sub-card, library-first exactly like M16–M20; the M10 integrator and the M16–M20 paths (incl. the M20 SCALAR fatigue) stay bit-identical (a NEW parallel path consuming the vector stress modes read-only). Theory: Preumont & Piéfort 1994; Pitoiset & Preumont 2000; Carpinteri–Spagnoli 2001; Cristofori/Susmel–Tovo 2008; Socie & Marquis; Shinozuka & Deodatis (multivariate synthesis) |
| — (NO frequency-domain / spectral / critical-plane / path-counting fatigue solver of ANY kind anywhere in the open-source engine — `freimpl.F` re-read line by line for M22: no /FATIG, no critical-plane / MCC / Findley / Fatemi-Socie machinery; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag) | `pyradioss/implicit/nonproportional_fatigue.py` + the `_run_nonproportional` extension of the `_run_multiaxial` driver (`_report_nonproportional`) in `pyradioss/implicit/random_response.py` + `/IMPL/FATIG/MULT/NPROP` in `engine_keywords.py` | M22: NON-PROPORTIONAL MULTIAXIAL FATIGUE — the critical-plane TIME-DOMAIN path-counting damage of a rotating (non-proportional) shear path, the FIRST item deferred out of M21 that BUILDS on it (the rotating-shear PATH the M21 projected-equivalent spectral methods deferred), the natural consumer of the M21 multivariate synthesiser + candidate-plane machinery. SHEAR-PATH AMPLITUDE OPERATORS: on a candidate plane, the 2-D resolved-shear locus (τ_a(t),τ_b(t)) in the M21 in-plane basis (a `shear_projection` per in-plane axis of a synthesised stress-component history) — the MINIMUM CIRCUMSCRIBED CIRCLE (Papadopoulos 1998, radius = τ_a; Welzl on the convex hull), the LONGEST CHORD / longest projection (the diameter), and the MAXIMUM RECTANGULAR / PRISMATIC HULL (Mamiya–Araújo–Castro 2009, max_θ √(a1²+a2²)). All collapse to the scalar M21 amplitude for a proportional line; for a circle MCC=r, chord=2r, MRH=r√2. A non-proportionality factor F_np = √(λ₂/λ₁) of the 2-D shear-path covariance (the Itoh–Kanazawa aspect-ratio form: 0 for a line, 1 for a circle). CRITICAL-PLANE TIME-DOMAIN DAMAGE: over the M21-synthesised correlated stress-component histories (seeded), resolve σ_n(t) and the shear PATH on each candidate plane, rainflow (ASTM E1049) the dominant resolved shear scaled by the non-proportional path factor g=MRH/scalar, and apply FINDLEY (τ_a + k σ_n,max) and FATEMI–SOCIE (γ_a(1 + k σ_n,max/σ_y)) carrying the per-plane MAX NORMAL stress (which folds in that plane's mean — a piece of the M20/M21 mean-stress deferral). Search the critical plane (max criterion parameter — the textbook Findley/FS definition), report the plane, F_np, damage rate / life. For PROPORTIONAL loading the path count reduces to the M21 max-shear rainflow (g=1, F_np=0); for a 90°-out-of-phase case the non-proportional damage is g^m higher than the scalar projection (F_np≈1) — the extra damage the spectral method misses. PORT sub-flag, library-first exactly like M16–M21; the M10 integrator, the M16–M20 paths AND the M21 SPECTRAL reductions stay bit-identical (a NEW parallel path — the M21 spectral answer is byte-identical whether or not /NPROP runs). Theory: Papadopoulos 1998; Mamiya–Araújo–Castro 2009; Findley 1959; Fatemi–Socie 1988; Matake 1977; Carpinteri–Spagnoli 2001; Itoh–Sakane–Socie 1995; Socie & Marquis ch. 2–4 |
| — (NO frequency-domain / spectral / critical-plane / non-proportional fatigue solver of ANY kind anywhere in the open-source engine — `freimpl.F` re-read line by line for M23: no /FATIG, no frequency-domain non-proportionality / modified-Wöhler / spectral-invariant machinery; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag) | `pyradioss/implicit/spectral_nonproportional_fatigue.py` + the `_run_spectral_nonproportional` extension of the `_run_multiaxial` driver (`_report_spectral_nonproportional`) in `pyradioss/implicit/random_response.py` + `/IMPL/FATIG/MULT/NPROP/SPEC` in `engine_keywords.py` | M23: SPECTRAL NON-PROPORTIONAL MULTIAXIAL FATIGUE — the FREQUENCY-DOMAIN non-proportionality factor and critical-plane damage estimated DIRECTLY from the stress-tensor cross-PSD spectral-MOMENT matrices, with NO synthesised history (the Cristofori–Susmel–Tovo / Pitoiset spectral method M22's time-domain path count deferred), the SPECTRAL sibling of M22. SPECTRAL SHEAR-PATH / F_np: on a candidate plane, the 2×2 in-plane shear cross-spectral moment matrix Σ_τ (the 2×2 block of the M21 M₀ = E[σσᵀ] built from `shear_projection` along each in-plane axis — NO synthesis), and F_np = √(λ₂/λ₁) of its eigenvalues — EQUALS the M22 TIME-DOMAIN F_np (the identity M₀ = E[σσᵀ] = the covariance the synthesiser reproduces); F_np = 0 for a proportional (rank-1) state, → 1 for the 90°-out-of-phase circle. SPECTRAL CRITICAL-PLANE DAMAGE: the dominant-shear PSD moments p_domᵀMₙp_dom and resolved-normal p_nᵀMₙp_n (from p^T Mₙ p, no history), the closed-form non-proportional amplitude correction g² = 1 + F_np² = trace(Σ_τ)/λ₁ (the exact MRH/MCC ratio of an elliptical path — the M22 prismatic-hull effect read spectrally), the Susmel–Tovo modified-Wöhler stress ratio ρ = σ_a/τ_a, and the equivalent-stress moments per model (FINDLEY as the spectral-invariant linear combination g·τ + k·σ_n, FATEMI–SOCIE, SHEAR-PATH) run through the four M20 estimators. For PROPORTIONAL loading the spectral shear-path damage reduces EXACTLY to the M21 max-shear spectral answer (F_np=0); for 90°-out-of-phase it is (1+F_np²)^(m/2)=2^(m/2) higher than the M21 projected scalar AND agrees with the M22 time-domain path count within scatter (the two non-proportional methods converge). PORT sub-flag, library-first exactly like M16–M22; the M10 integrator, the M16–M20 paths, the M21 SPECTRAL reductions AND the M22 TIME-DOMAIN path count stay bit-identical (a NEW parallel path — both byte-identical whether or not /SPEC runs). Theory: Pitoiset & Preumont 2000; Cristofori–Susmel–Tovo 2008; Susmel & Lazzarin 2002; Bäckström & Marquis 2001; Carpinteri–Spagnoli–Vantadori 2013 |
| — (NO frequency-domain / spectral / non-Gaussian fatigue solver of ANY kind anywhere in the open-source engine — `freimpl.F` re-read line by line for M24: no /FATIG, no non-Gaussian / kurtosis / Hermite-moment / Winterstein machinery; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag) | `pyradioss/implicit/nongaussian_fatigue.py` + the `_run_nongaussian` / `_run_nongaussian_multiaxial` extensions of the `run_fatigue` / `_run_multiaxial` drivers (`_report_nongaussian` / `_report_nongaussian_multiaxial`) in `pyradioss/implicit/random_response.py` + `/IMPL/FATIG/NGAUSS` in `engine_keywords.py` | M24: NON-GAUSSIAN / KURTOSIS SPECTRAL FATIGUE — the frequency-domain damage of a stationary but NON-GAUSSIAN random-vibration response, computed by CORRECTING the M20–M23 Gaussian spectral estimators for a specified kurtosis (and skewness). WINTERSTEIN HERMITE MODEL: g(u) = κ[u + h₃(u²−1) + h₄(u³−3u)] maps a standard Gaussian u to the non-Gaussian process for a target γ₄/γ₃ (the softening fit h₄ = (√(1+1.5(γ₄−3))−1)/18; κ = 1/√(1+2h₃²+6h₄²) preserves the mean/variance EXACTLY). CORRECTION FACTOR: λ_ng = E[g(V)ᵐ]/E[Vᵐ] over the Rayleigh amplitude V (Benasciutti–Braccesi / Rizzi–Kihm), scaling the Gaussian narrow-band / Dirlik / Wirsching-Light / Tovo-Benasciutti damage, with an optional Benasciutti–Tovo bandwidth attenuation γ₄,eff = 3 + (γ₄−3)α₂. NON-GAUSSIAN MONTE-CARLO: the M20 Gaussian history pushed through the memoryless Hermite transform to (γ₃, γ₄), ASTM E1049 rainflow + Miner (`synthesize_nongaussian_history` / `nongaussian_monte_carlo_damage` / `_projected`). λ_ng = 1 for Gaussian (M20 recovered EXACTLY), > 1 leptokurtic, < 1 platykurtic; composes with /MULT, /NPROP, /SPEC (a scalar correction on the equivalent-stress PSD the M21/M23 reductions produce). PORT sub-flag, library-first exactly like M16–M23; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions stay bit-identical (a NEW parallel path — byte-identical whether or not /NGAUSS runs). Theory: Winterstein 1988; Winterstein & MacKenzie 1997; Benasciutti & Tovo 2005/2006; Braccesi–Cianetti–Lori–Pioli 2009; Rizzi–Kihm–Ferguson; Kihm & Rizzi 2013 |
| — (NO frequency-domain / spectral / non-stationary fatigue solver of ANY kind anywhere in the open-source engine — `freimpl.F` re-read line by line for M25: no /FATIG, no non-stationary / evolutionary / spectrogram / mission-profile machinery; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag) | `pyradioss/implicit/nonstationary_fatigue.py` + the `_run_nonstationary` / `_run_nonstationary_multiaxial` extensions of the `run_fatigue` / `_run_multiaxial` drivers (`_report_nonstationary` / `_report_nonstationary_multiaxial`) in `pyradioss/implicit/random_response.py` + `/IMPL/FATIG/NSTAT` in `engine_keywords.py` | M25: NON-STATIONARY / EVOLUTIONARY-PSD SPECTRAL FATIGUE — the frequency-domain damage of a random-vibration response whose PSD / RMS VARIES WITH TIME, computed by extending the M20–M24 STATIONARY estimators to a non-stationary process. PIECEWISE-STATIONARY / BLOCK ("mission profile") MODEL: partition the load into stationary blocks (each an RMS scaling a_i of a shared PSD shape — moments scale as m_n → a_i²m_n — over a duration T_i), run the M20 estimators PER BLOCK and Palmgren–Miner SUM D = Σ (E[D]/T)_i T_i (`block_fatigue_summary`; each estimator's damage scales EXACTLY as a_i^m, the shape coefficients scale-invariant). AMPLITUDE-MODULATED / EVOLUTIONARY MODEL: Priestley's separable S(ω,t) = |A(t)|² S(ω), the stationary damage INTEGRATED over the RMS distribution p(a) — the closed-form E[a^m]-weighted damage (`amplitude_modulated_summary`; = the block Miner-sum for equal-shape blocks). M25↔M24 BRIDGE: a varying RMS makes the Gaussian carrier leptokurtic, γ₄ = 3·E[a⁴]/E[a²]² (the Wolfsteiner–Trapp / Kihm–Rizzi non-stationary→kurtosis link), so the non-stationary amplification κ_ns = E[a^m]/E[a²]^(m/2) (`nonstationary_amplification`) AGREES with the M24 λ_ng at that kurtosis (`bridge_to_nongaussian`, reusing `nongaussian_fatigue` read-only) — EXACTLY 1 in the constant-modulation limit, to leading order otherwise (differing by the documented (m−2)/(m−1) scale-mixture-vs-Hermite factor). NON-STATIONARY MONTE-CARLO: the M20 Gaussian carrier × a time-varying RMS envelope, ASTM E1049 rainflow + Miner (`synthesize_nonstationary_history` / `nonstationary_monte_carlo_damage`); the block-boundary-rainflow caveat documented; the constant-modulation limit reduces to the M20 Gaussian MC bit-identically. Composes with /MULT, /NPROP, /SPEC, /NGAUSS (a scaling on the equivalent-stress PSD the M21/M23 reductions produce). PORT sub-flag, library-first exactly like M16–M24; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions AND the M24 NON-GAUSSIAN correction stay bit-identical (a NEW parallel path — byte-identical whether or not /NSTAT runs). Theory: Priestley 1965; Bendat & Piersol; Wolfsteiner & Breuer / Wolfsteiner & Trapp; Braccesi–Cianetti–Lori–Pioli; Kihm–Ferguson–Antoni; Rychlik (switching process); Palmgren–Miner |
| — (NO frequency-domain / spectral / non-stationary / evolutionary fatigue solver of ANY kind anywhere in the open-source engine — `freimpl.F` re-read line by line for M26: no /FATIG, no non-stationary / evolutionary / spectrogram / time-frequency machinery; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag at line 269) | `pyradioss/implicit/evolutionary_fatigue.py` + the `_run_evolutionary` / `_run_evolutionary_multiaxial` extensions of the `run_fatigue` / `_run_multiaxial` drivers (`_report_evolutionary` / `_report_evolutionary_multiaxial`) in `pyradioss/implicit/random_response.py` + `/IMPL/FATIG/EVOL` in `engine_keywords.py` | M26: FULLY EVOLUTIONARY / NON-SEPARABLE-PSD SPECTRAL FATIGUE — the frequency-domain damage of a random-vibration response whose spectral SHAPE (not merely its RMS level) VARIES WITH TIME, computed by extending the M25 piecewise-stationary / amplitude-modulated estimators to a genuinely NON-SEPARABLE evolutionary spectrum S(ω,t) / spectrogram. SPECTROGRAM / PER-WINDOW-FULL-PSD MODEL: partition into time-windows each with its OWN complete stress PSD (hence its own full moments m₀..m₄ — different bandwidth / rates / centre frequency per window, NOT a shared shape), run the M20 estimators PER WINDOW and Palmgren–Miner SUM the window damages duration-weighted (`evolutionary_fatigue_summary`, driven read-only through the M25 `block_fatigue_summary` which already accepts per-window distinct moments). TIME-FREQUENCY / DRIFTING-SHAPE MODEL: an evolutionary spectrum from a smoothly time-varying parameter — a swept centre frequency f_c(t) (a chirp) and/or a broadening bandwidth b(t) — sampled into windows whose full moments are recomputed FROM the drifting shape (`drifting_shape_spectrogram` applies a swept/broadening Gaussian window W_i(f) to the recovered stress PSD, commuting with |H|²; `gaussian_evolutionary_spectrogram` builds one analytically). NON-SEPARABLE MONTE-CARLO with a TIME-VARYING FILTER: per-window spectral-representation blocks concatenated (`synthesize_evolutionary_history` / `evolutionary_monte_carlo_damage`), whose short-time spectrogram (per-window RMS + ν₀) tracks S(ω,t); the constant-shape limit DELEGATES to the M25 `synthesize_nonstationary_history` bit-identically (the built-in M26↔M25 reduction). Validated: a single window = M20 EXACTLY; a constant-shape spectrogram = M25 amplitude-modulated EXACTLY; the window Miner-sum = the duration-weighted per-window damages; a two-window narrow→wide shape-swap differing from any single-shape M25 scaling; the swept f_c drifting ν₀, the broadening b drifting α₂; the non-separable MC matching the window estimate within scatter and reducing to the M25 / M20 MC bit-identically in the constant-shape / single-window limit. Composes with /MULT, /NPROP, /SPEC, /NGAUSS, /NSTAT (a drifting window on the equivalent-stress PSD the M21/M23 reductions produce). PORT sub-flag, library-first exactly like M16–M25; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions, the M24 NON-GAUSSIAN correction AND the M25 NON-STATIONARY correction stay bit-identical (a NEW parallel path — byte-identical whether or not /EVOL runs). Theory: Priestley 1965/1967 (evolutionary / non-separable spectra); Mark 1970 / Hammond (non-stationary spectral analysis); Newland (the spectrogram / short-time method); Bendat & Piersol; the Wigner–Ville time-frequency view; Palmgren–Miner |
| — (NO frequency-domain / spectral / multiaxial / evolutionary fatigue solver of ANY kind anywhere in the open-source engine — `freimpl.F` re-read line by line for M27: no /FATIG, no critical-plane / stress-tensor cross-PSD / joint-tensor / spectrogram machinery; the sole `PSD` token is still `IMUMPSD`, a MUMPS flag at line 269) | `pyradioss/implicit/joint_evolutionary_fatigue.py` + the `_run_joint_evolutionary` extension of the `_run_multiaxial` driver (`_report_joint_evolutionary`) in `pyradioss/implicit/random_response.py` + `/IMPL/FATIG/MULT/EVOL/JOINT` in `engine_keywords.py` | M27: FULLY NON-STATIONARY / EVOLUTIONARY MULTIAXIAL (JOINT-TENSOR) FATIGUE — the frequency-domain damage of a MULTIAXIAL random-vibration response whose full 6×6 stress-TENSOR cross-PSD S_σσ(ω,t) (not just an equivalent scalar) VARIES WITH TIME, computed by forming the per-window tensor cross-PSD and reducing it PER WINDOW with the M21/M23 critical-plane machinery — the critical-plane ORIENTATION and the M23 non-proportionality factor F_np RE-SEARCHED from the WINDOW's OWN tensor (so the plane may ROTATE and F_np may DRIFT window to window), Palmgren–Miner-summing the per-window MULTIAXIAL damages. The item M25/M26 deferred (M25/M26 applied the block / evolutionary scaling to the EQUIVALENT SCALAR the M21/M23 reductions produce, a FIXED reduction; M27 lifts exactly that — the JOINT tensor evolves so the reduction itself drifts). PER-WINDOW TENSOR CROSS-PSD: the M26 drifting-shape window W_i(f) multiplies the full 6×6 S_σσ(ω) per window (it commutes with the reduction, as M26 established for |H|²), the per-window 6×6 spectral-MOMENT matrices M_{n,i} = a_i²·(1/π)∫ω^n W_i(f) S_σσ dω recomputed (`windowed_tensor_moment_matrices`), and EACH window reduced by the M21 equivalent von Mises + max-normal / max-shear critical-plane search + the M23 F_np — the plane RE-SEARCHED per window (`reduce_window_tensor`, `joint_evolutionary_fatigue_summary`). Reports the per-window critical-plane DRIFT (plane normal / F_np / equivalent-stress RMS per window) + the plane ROTATION angle + the F_np drift. NON-STATIONARY MULTIVARIATE MONTE-CARLO: per-window multivariate spectral-representation blocks of the WINDOWED tensor (the M21 per-bin eigen/Cholesky synthesiser with a time-varying tensor cross-PSD), projected onto the WINDOW's OWN critical plane, ASTM E1049 rainflow + Miner (`synthesize_joint_evolutionary_history` / `joint_evolutionary_monte_carlo_damage`); the constant-shape limit DELEGATES to a single multivariate carrier × envelope, and a single unit window reduces to the M21 multivariate MC bit-identically. Validated: a STATIONARY tensor / single window recovering the M21 spectral multiaxial answer EXACTLY (all three reductions); a FIXED critical plane recovering the M26 scalar per-window spectrogram EXACTLY; a CONSTANT-tensor-shape / RMS-only drift recovering the M25 multiaxial block answer EXACTLY; a rotating-principal-axes case whose per-window critical plane genuinely DRIFTS and whose Miner-sum DIFFERS from the M26 fixed reduction (the M27↔M26 boundary); the windowed tensor moments commuting with the reduction; the multivariate MC tracking the spectrogram and matching the window estimate within scatter. Composes with /NPROP, /SPEC, /NGAUSS, /NSTAT; /JOINT (or /TENSOR) implies MULT + EVOL. PORT sub-flag, library-first exactly like M16–M26; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions, the M24 NON-GAUSSIAN correction, the M25 NON-STATIONARY correction AND the M26 EVOLUTIONARY-SCALAR correction stay BIT-IDENTICAL (a NEW parallel path — the M20–M26 answers byte-identical whether or not /JOINT runs). Theory: Priestley 1965/1967 (evolutionary spectra, here the matrix-valued tensor cross-PSD); Preumont & Piéfort 1994 / Pitoiset & Preumont 2000 (spectral von Mises); Carpinteri–Spagnoli / Cristofori–Susmel–Tovo (critical plane); Bäckström & Marquis (non-proportional); Palmgren–Miner window sum |

## 3. Conventions used in this port

1. **Every module docstring names its Fortran origin** and explains the theory
   (with textbook references: Belytschko–Liu–Moran for the corotational shell
   and hourglass control, Wilkins for the radial return, etc.).
2. **Structure of arrays, not array of structures.** Elements of one type are
   processed as NumPy arrays over the whole group — the Python analogue of the
   Fortran element-group loops (`NGROUP`/`MVSIZ` blocks). Scalar per-element
   Python loops are avoided except in setup code.
3. **IDs vs indices.** Decks use arbitrary user IDs; the Starter converts all
   of them to dense 0-based indices (the original does the same via `USR2SYS`
   tables). Arrays in the Engine are indexed, never keyed by user ID; the
   mapping is kept for output.
4. **Explicit is explicit.** No hidden state: each cycle recomputes forces
   from (x, v, internal state). All mutable element state (stress, plastic
   strain, hourglass forces…) lives in per-group `state` dictionaries created
   by the Starter.
5. **Units**: consistent-unit philosophy identical to Radioss — the solver
   never converts units; the deck must be consistent (e.g. mm, ms, kg → kN).
6. **Failure behaviour**: readable exceptions from the Starter for model
   errors (`StarterError`, with the same "** ERROR" flavour as the listing);
   the Engine protects against divergence with an energy-error stop criterion
   like the original (`/STOP` defaults).

## 4. Feature matrix (Milestones 1–6)

Legend: ✅ ported (functional), 🟡 simplified (functional but reduced options), ❌ not yet.

### Input keywords

| Keyword | Status | Notes |
|---|---|---|
| `/BEGIN`, `/END`, `/TITLE` | ✅ | |
| `#include` | ✅ | recursive |
| `/NODE` | ✅ | |
| `/BRICK` | ✅ | 8-node hexa; 4-distinct-node degenerates auto-convert to `/TETRA4`, penta/pyramid rejected with a clear error |
| `/TETRA4` | ✅ | 4-node constant-strain tetra |
| `/SHELL` | ✅ | 4-node Belytschko–Tsay |
| `/SH3N` | ✅ | 3-node C0 triangle (plain C0; DKT-flavoured Ish3n variants ❌) |
| `/TRUSS`, `/SPRING` | ✅ | |
| `/BEAM` | ✅ | N1 N2 + orientation node N3 |
| `/PART`, `/SUBSET` | ✅ / ❌ | |
| `/MAT/LAW1` (`/MAT/ELAST`) | ✅ | |
| `/MAT/LAW2` (`/MAT/PLAS_JOHNS`) | ✅ | εp-rate & hardening, eps_p_max element deletion (M3), beams via the global-plasticity model (M3); since M6 the full thermal terms in the ADIABATIC approximation (optional card 6 `m T_melt rho_Cp T_i`): plastic work heats the point, dT = σy·dεp/ρCp, and T*^m softens the yield — per-point temperature state on solids and shell layers |
| `/MAT/LAW27` (`/MAT/PLAS_BRIT`) | 🟡 | brittle tensile cracking with fixed crack direction, unilateral damage, layer rupture + element deletion; since M15 the CONSISTENT implicit shell tangent (per-branch — see law27_brittle.py); the plastic block of the original ❌ (shells only, like the original) |
| `/MAT/LAW36` (`/MAT/PLAS_TAB`) | ✅ | tabulated hardening from /FUNCT curves, strain-rate curve family (linear rate interpolation), eps_p_max deletion; Fsmooth/Chard/Fcut and Fscale ❌ |
| `/MAT/LAW42` (`/MAT/OGDEN`) | 🟡 | Ogden/Mooney-Rivlin, incompressible + K(J-1) bulk penalty, exact F from initial gradients, **nonlinear sound speed feeds the time step** (the law stiffens with stretch — verified by a long /DT 0.9 hold at λ≈2); solids only, no shell variant, no Prony viscosity |
| `/FAIL/JOHNSON` | ✅ | D1–D4 + rate term; thermal D5 since M6 (needs the LAW2 thermal card, warned otherwise); Ifail_sh 1/2; element deletion (stress zeroing, dt release, OFF in ANIM) |
| `/FAIL/BIQUAD` | 🟡 | explicit c1–c5 input (two-parabola εf(σ*) fit); M-flag material presets and S-flag ❌ |
| `/PROP/TYPE1` (`SHELL`) | 🟡 | thickness, N integration points, hourglass coeffs (Ishell fixed = BT for quads, C0 for `/SH3N`) |
| `/PROP/TYPE2` (`TRUSS`) | ✅ | area |
| `/PROP/TYPE3` (`BEAM`) | 🟡 | A, Iyy, Izz, Ixx; Timoshenko with full-section shear (no shear factor / Ishear variants), LAW1 only |
| `/PROP/TYPE4` (`SPRING`) | 🟡 | linear k, c, mass |
| `/PROP/TYPE14` (`SOLID`) | 🟡 | qa/qb bulk viscosity, hourglass coeff (Isolid fixed = 1-pt+FB) |
| `/BCS` | ✅ | translation + rotation fixities; on a rigid-body master it becomes a body-level condition (full 111 translations = pivot) |
| `/INIVEL/TRA`, `/INIVEL/AXIS` | ✅ / ✅ | AXIS since M5: rigid-rotation field ω·d×(x−P) (the way a spinning /RBODY is set up); the translational Vt fields of the full AXIS card via an extra /INIVEL/TRA |
| `/IMPVEL` | ✅ | via `/FUNCT`, fixed direction; on a rigid-body master it drives the body (moving rigid die); on a moving-wall node it drives the wall |
| `/IMPDISP` | ✅ | M5 — kinematic like /IMPVEL but enforced at the *position* level (the node lands exactly at x0 + d(t), no velocity-integration drift); work booked from the constraint impulse like /IMPVEL |
| `/GRAV` | ✅ | |
| `/CLOAD` | ✅ | optional /SENSOR gating since M6 (waits for the sensor, then follows f(t − t_fire)) |
| `/EOS/POLYNOMIAL`, `/EOS/IDEAL-GAS` | ✅ | M6 — attaches to LAW1/2/36 like /FAIL; the EOS pressure replaces the law's (deviator stays with the law); implicit E-p coupling per element with relative-volume state, viscous shock heating into E, EOS sound speed feeds the time step (see materials/eos.py) |
| `/DAMP` | 🟡 | M6 — Rayleigh MASS damping (α), Tstart/Tstop window; applied as the exact per-cycle integrating factor (unconditionally stable, claims no dt) with the dissipation booked exactly into the DE ledger; the stiffness (β) branch ❌ (needs K·v products) |
| `/SENSOR/TIME`, `/SENSOR/DISP` | ✅ | M6 — latching sensors gating /CLOAD, /PLOAD and /INTER/TYPE7/11 (fire time survives restarts); other sensor types ❌ |
| `/MPC` | ✅ | M6 (deferred from M5) — general linear rows on translations (+ rotations where the node carries inertia), solved together via the nc×nc Lagrange system on accelerations + a velocity cleanup; zero work by construction, /BCS-fixed DOFs act as ground; redundant row sets fall back to least-squares multipliers |
| `/PLOAD` | ✅ | M5 — follower pressure on a /SURF (current segment normal, p·A lumped to corners, triangles 1/3); segments of /FAIL-deleted elements stop carrying pressure |
| `/ADMAS` | 🟡 | M5 — per-node added mass (Radioss type-0 semantics only); also the way a moving /RWALL gets its inertia |
| `/FUNCT` | ✅ | piecewise-linear tables |
| `/GRNOD/NODE`, `/GRNOD/PART`, `/GRNOD/BOX` | ✅ | |
| `/BOX/RECTA` | ✅ | |
| `/RWALL/PLANE`, `/RWALL/SPHER`, `/RWALL/CYL` | ✅ | M5: three geometries, sliding/tied/friction, and MOVING walls tied to a carrier node (free with /ADMAS+/INIVEL — impulses react on the node, momentum-exact; or /IMPVEL-driven — the drive absorbs the reaction and books external work). Walls do not rotate; containment (nodes inside a sphere/cyl) ❌ |
| `/RBODY` | ✅ | M5 — master + slave node set as one rigid body: starter assembles mass/COG/inertia tensor (point masses + nodal inertias + added Mass/Jxx-Jzz), ICoG=1 master relocation; engine integrates the 6-DOF Newton-Euler EOM (angular-momentum update + exponential-map rotation — L conserved by construction). Sensors, skew/spherical inertia, IKREM, surface envelope ❌ |
| `/RBE2` | 🟡 | M5 — rigid link: same mechanics with a structural master kept at its own position; full 6-DOF tie only (per-DOF flags ❌) |
| `/RBE3` | 🟡 | M5 — interpolation constraint (least-squares rigid fit + its virtual-work dual force distribution — no stiffening, no spurious work); one master group with uniform weights (per-set weights/DOF flags ❌) |
| `/SECT` | 🟡 | M5 — section force/moment time history through a cut, computed by the side-sum identity over one side's node set (see engine/sections.py); output via /TH/SECT. The frame/element-set input of the full card ❌ |
| `/INTER/TYPE7` | ✅ | penalty node↔surface (M4): Istf 0–5 stiffness variants, Igap 0/1 (constant / variable from shell thicknesses) with Gap_min/Gap_max, self-impact (`grnod_ID = 0`), Coulomb friction, voxel broad phase; /SENSOR gating since M6 (the Tstart/Tstop role); since M15 the Ifric > 0 friction MODELS (MFROT 1–4: generalized viscous / Darmstadt / Renard / exponential decay µ(p, v) with C1–C6, `contact/friction.py`) and the Ifiltr = 1/2/3 IFQ tangential-force filter with the reader's exact XFILTR mapping; Inacti, Igap 2/3, Ifiltr ≥ 10 (MODFR 2, refused loudly), /FRICTION per-part-pair sets, orthotropic friction ❌ |
| `/INTER/TYPE2` | 🟡 | tied contact (M4): kinematic secondary→main gluing, constant-weight projection with co-rotating offset, lumped mass/force transfer, deletion release; rotational-DOF tying (Spotflag) and offset moment redistribution ❌ |
| `/INTER/TYPE11` | ✅ | edge↔edge penalty (M4): /LINE edge sets, Istf/Igap as TYPE7, exact segment-segment closest points; parallel-overlap force distribution simplified to the closest-point pair; since M15 the Ifric > 0 friction models + IFQ as a documented PORT EXTENSION (the original TYPE11 never evaluates MFROT — i11mainf.F forces MFROT = 0, checked; edge-pair pressure DEFINED p = f_n/(L_main·gap), see contact/friction.py) |
| `/LINE/SURF`, `/LINE/SEG` | ✅ | edge sets for TYPE11 (M4), with element provenance for deletion |
| `/SURF/PART`, `/SURF/SEG` | ✅ | for contact; since M4 every segment carries its parent-element provenance (deletion, stiffness, gap) |
| `/TH/NODE`, `/TH/PART`, `/TH/SECT` | ✅ | SECT since M5: FX FY FZ MX MY MZ |
| Engine: `/RUN`, `/VERS`, `/TFILE`, `/ANIM/DT`, `/ANIM/VECT|ELEM`, `/DT`, `/PRINT`, `/STOP` | ✅/🟡 | |
| Engine: `/DT/NODA`, `/DT/NODA/CST` | ✅ | M6 — nodal time step dt_i = √(2Mᵢ/Kᵢ) with the element stiffness derived from the kernels' own dt claims (kᵢᵉ = 2mᵢᵉ/dt_e², = the element dt on uniform meshes) and the contact NEAR-spring stiffness accumulated in; CST adds mass to hold dT_min — added mass, its momentum and its kinetic energy are tracked, reported (1%-step announcements + termination summary) and the energy enters the balance; prescribed nodes (rigid-body members, tied secondaries, RBE3 dependents) are excluded (see engine/mass_scaling.py) |
| Engine: `/STATE/DT` | 🟡 | M6 — periodic full restart snapshots refreshing `RunName_{nn}.rst` (crash recovery / early chaining); the original's .sta ASCII format ❌ (the pickle restart plays that role) |
| Engine: `/IMPL` (+ `/IMPL/DTINI`, `/IMPL/NEWTON`, `/IMPL/LSOLVER`) | 🟡 | M8 — switches the run to the implicit-STATIC Newton driver (final /RUN "time" = load factor, increment size, Newton tolerance + iteration cap, direct-solver choice); unknown sub-cards warn and skip like the rest of the reader |
| Engine: `/IMPL/DYNA/1` (HHT), `/IMPL/DYNA/2` (Newmark) | ✅ | M10 — implicit DYNAMICS: /RUN "time" and /IMPL/DTINI become PHYSICAL again. The card mirror follows the SOURCE (freimpl.F + imp_dyna.F), checked, not the docs: `/1` reads the HHT **alpha itself** (HHT_A — *not* a spectral radius; γ = ½−α, β = ¼(1−α)² derived), `/2` reads **gamma then beta** (NM_A → DY_G, NM_B → DY_B), defaults γ=½ β=¼ (trapezoidal); a bare `/IMPL/DYNA` = `/2` defaults (a port convenience). Warns outside the HHT range [−1/3, 0] and outside 2β ≥ γ ≥ ½ |
| Engine: `/IMPL/DYNA/DAMP` | ✅ | M11 — Rayleigh damping in the implicit system, card = `a b` (freimpl.F reads DAMPA_IMP then DAMPB_IMP; the card alone IMPLIES dynamics, mirroring `IF (IDYNA==0) IDYNA=1`): C = a·M + b·K(step start), damping force HHT-weighted in the residual, (1+α)γ/(βdt)·C in K_eff, DY_EDAMP-style trapezoidal dissipation ledger. Validated vs the closed-form damped SDOF (a-only / b-only / mixed) |
| Engine: `/IMPL/DT/STOP`, `/IMPL/DT/1` | ✅ | M11 — automatic implicit step control (imp_dt.F, IDTC = 1): ALWAYS ON with documented port defaults (target 6 iterations, grow ×1.1 toward the /IMPL/DTINI step, cut ×0.5 on non-convergence, dt_min = 1e-4·dtini); the cards tune it. Works for statics load increments AND dynamics time steps; IDTC 2/3 (displacement-norm/Riks step controls) and /IMPL/DT/FIXP deferred |
| Engine: `/IMPL/BUCKL/1\|2` | ✅ | M11 — the engine card for the M9 buckling eigensolver: runs the static prestress increments, then reports the critical-load multipliers + modes in the listing (imp_buck.F flavour) and on `implicit_result.buckling_factors/modes`; card = `Emin Emax Nmode …` (Nmode = NBUCK drives the dense eigh; the ARPACK range/controls accepted, unused); bare `/IMPL/BUCKL` errors as OBSOLETE like the original reader; refused with /IMPL/DYNA |
| Engine: `/IMPL/NONLIN[/N]` (`/SMDISP`), `/IMPL/ARCL` | ✅ | M9 — NONLIN switches the implicit run to NONLINEAR GEOMETRY (updated-Lagrangian frame + K_geo; the original card's large-displacement default), `/IMPL/NONLIN/SMDISP` keeps the M8 small-displacement path (the original's SMDISP), a numeric /N (solver-strategy pick) is accepted (the port always runs full Newton); ARCL card `dl max_inc it_des` (all optional) selects the Riks/Crisfield arc-length continuation and implies NONLIN |
| Engine restart chaining (`RunName_0002.rad`) | ✅ | M6 — the Engine ALWAYS writes `RunName_{nn}.rst` at termination; run nn+1 resumes it: clock/ledgers/next-dt/output numbering restored, rigid-body R & L and sensor fire-times carried, everything else deliberately reconstructed from the model arrays (tied projections, contact candidates, fix masks). Acceptance: a chained run reproduces the unchained one exactly — same cycle count, state to round-off (asserted for a spring oscillator and a tumbling /RBODY) |

### Solver features

| Feature | Status |
|---|---|
| Explicit central-difference integration | ✅ |
| Element time step with scale factor, bulk viscosity | ✅ |
| Exact per-element eigenvalue bound on dt (port improvement: the classic lc/c estimate is up to ~40% above the true one-point-element stability limit; the Starter computes the exact eigenvalue correction once per element — 6×6 for solids, membrane **and** bending/shear branches for shells since M2, the full 12×12 for beams — see `solid_hexa8._exact_dt_factor`, `shell_bt4._bend_shear_omega2`, `beam_type3._exact_dt`) | ✅ |
| Solid hexa8, 1-point, Flanagan–Belytschko hourglass control | ✅ |
| Solid tetra4, constant strain (no hourglass modes; plain formulation — nodal-pressure Itetra variants ❌) | ✅ |
| Belytschko–Tsay 4-node shell (memb/bend/shear, BLT84 **stiffness** hourglass control since M2 — M1's viscous form artificially damped coarse dynamic bending) | ✅ |
| C0 3-node triangle shell (CST membrane + Mindlin plate, no hourglass modes) | ✅ |
| Corotational Timoshenko beam (axial/2×shear/torsion/2×bending resultants) | ✅ (LAW1 elastic; LAW2 via the global resultant-plasticity model since M3 — yields at exactly W·σy, no elastic-core spread to the 1.5·W·σy hinge; consistent implicit tangent + iterated implicit return since M15; fiber-integrated TYPE18 beam is a roadmap item) |
| Degenerated /BRICK → tetra conversion, penta/pyramid clear check | 🟡 |
| Truss, linear spring | ✅ |
| Jaumann objective stress update | ✅ |
| LAW1, LAW2 (3D + plane stress) | ✅ |
| LAW36 tabulated plasticity (3D + plane stress, rate curve family) | ✅ |
| LAW27 brittle cracking (fixed smeared crack, unilateral damage; consistent implicit tangent since M15) | 🟡 (elastic-brittle; original's plastic block ❌) |
| LAW42 Ogden hyperelasticity (total-strain from exact F, nonlinear SOUNDSP → dt) | 🟡 (solids only) |
| /FAIL element deletion plumbing (per-layer for shells, GBUF%OFF, deleted elements keep mass, drop stress/hourglass/dt claim, OFF field in ANIM, deletion count in the listing) | ✅ |
| Equations of state (/EOS) for solids (M6): polynomial + ideal gas, implicit E-p update per element (closed form — p linear in E), relative-volume state, q-work shock heating into E, EOS sound speed → dt; validated against the exact ideal-gas isentrope pV^γ, the Rankine–Hugoniot identity and a quasi-static piston compression | ✅ |
| LAW2 adiabatic thermal terms + /FAIL/JOHNSON D5 (M6): per-point temperature rise from plastic work, (1−T*^m) softening (validated against the closed-form heating ODE and the softened flow stress) | ✅ |
| /DT/NODA/CST mass scaling with honest added-mass accounting (M6) | ✅ |
| /DAMP mass damping via the exact integrating factor, dissipation booked from the KE identity (M6) | ✅ |
| /SENSOR gating with latching + time-shifted load curves (M6) | ✅ |
| /MPC general linear constraints (Lagrange on accelerations + velocity cleanup, zero work by construction) (M6) | ✅ |
| Engine restart chaining, bit-reproducing the unchained run (M6) | ✅ |
| Numerical-dissipation ledger EN (M6): the exact internal-force midstep work measured per cycle against the state-function bookings — closes the balance under damped barely-resolved ringing (the M1-era qb misbooking), stays negligible on healthy runs (asserted), and its strongly-negative excursions are the new energy-INJECTION divergence stop (|ERR| alone is blind to instability once EN is in the balance) | ✅ |
| Rigid wall (kinematic, slide/tied/friction; plane, sphere, cylinder; fixed, free-with-mass, velocity-driven — M5) | ✅ |
| Rigid bodies /RBODY + /RBE2 (6-DOF Newton-Euler: gather → L += T dt → w = (R J0 Rᵀ)⁻¹L → rigid scatter → exponential-map placement; exact L conservation for torque-free bodies, pivot mode from master /BCS, /IMPVEL body drive; slaves coexist with contact and the TYPE2 effective-mass machinery; deletion never changes the inertia — masses stay; elements interior to a body carry exactly zero strain because the enforcement re-scatters the rigid velocity field at the placed positions) | ✅ |
| /RBE3 interpolation constraint (weighted least-squares rigid fit + dual force distribution — transmits force and moment exactly, adds no stiffness, does no work; lumped mass transfer like TYPE2) | 🟡 (uniform weights) |
| /SECT section resultants (side-sum identity: element self-equilibrium cancels everything interior to the side, leaving the through-cut force/moment; no per-side reassembly needed) | ✅ |
| Free moving wall: IMPLICIT joint momentum solve of the carrier node with all hit nodes (an M5 lesson: correcting the nodes against the pre-recoil wall velocity and recoiling afterwards feeds every riding node a one-cycle-stale, faster wall — an energy injection that does **not** vanish with dt, measured at +33% before the fix; the joint 3×3 solve reproduces the exact perfectly-inelastic collision in one cycle, asserted by a closed-form test) | ✅ |
| Kinematic-wall energy booking on the per-node injection identity U = ΔKE − f·v_old dt (books the arrest, the within-cycle acquired velocity, tied drag and friction in one expression; free-wall carrier KE change booked exactly from the joint solve; driven-wall injection booked as external work) | ✅ |
| TYPE7 penalty contact + friction (Istf variants, Igap, self-impact, voxel search) | ✅ |
| TYPE2 tied contact (kinematic; zero-work by construction — asserted in tests) | ✅ (translations; no rotation tying) |
| TYPE11 edge-to-edge penalty contact | ✅ |
| Contact ⇄ element deletion (M3⇄M4): segments/edges of `off == 0` elements drop out per cycle, secondary nodes with no surviving element stop being tracked, tied pairs release | ✅ |
| Interface time step: static node-on-spring bound + per-cycle accumulation of NEAR candidate spring stiffness per node (dt ≤ √(2m/ΣK) — springs stack at corners and in self-impact; proven by /DT 0.9 long-run impacts for Istf 0/2/5) | ✅ |
| Contact energy booked at the leapfrog midstep velocity (an M4 lesson: booking f·v at the pre-update velocity leaves a positive-definite f²dt²/2m residual per cycle that reads as energy creation when penalty springs dominate the scale — the midstep booking closes the balance to round-off) | ✅ |
| Energy balance (int/kin/hourglass/contact/external work), error % | ✅ |
| T01 time history, ANIM (as VTK), listings | ✅ |
| M7 NumPy fast paths (`common/fastmath.py`: formula-identical cross/norm, bincount scatter assembly, cofactor 3×3 det/inv; fused shell rate/hourglass matmuls; LAW36 exact-fixed-point Newton exit) | ✅ |
| Optional numba backend (`pyradioss/accel`): jit mirrors of the measured hotspots — hexa8 pre/post, BT4 shell pre/post, TYPE7 narrow phase — behind the same kernel API, `PYRADIOSS_BACKEND=numba` or `pyradioss-engine -backend numba`, NumPy fallback with a warning when numba is absent; parity asserted at kernel level and on full runs, restart chain bit-match asserted under numba (M7) | ✅ |
| JAX backend | ❌ (deferred — see the M7 roadmap note) |
| MPI/domain decomposition, SMP | ❌ (out of scope) |
| **Implicit STATIC analysis (M8)**: Newton–Raphson equilibrium, load stepping, global equation numbering with /BCS condensation, sparse tangent assembly (scipy CSR), reuse of the explicit force kernels for the residual, direct linear solve. Elements: 8-node solid (hexa8) + 4-node shell (BT4); materials: LAW1 elastic + LAW2 with the CONSISTENT elastoplastic tangent. Load control AND /IMPDISP displacement control. Validated: single-element Hooke (exact, 1-step), multi-element patch test (exact), shell cantilever tip deflection vs beam theory (<1%), LAW2 uniaxial vs the closed-form Johnson–Cook curve with quadratic Newton convergence, reaction/energy balance | ✅ (small-strain linear geometry) |
| Direct linear solver behind `solve(K,R)`: SuperLU default (SciPy), optional CHOLMOD (scikit-sparse, SPD) and MUMPS (python-mumps, wrapped not ported), env/CLI-selected with SuperLU fallback + warning — mirrors the M7 compute-backend pattern | ✅ |
| **Implicit NONLINEAR GEOMETRY (M9)**: geometric (initial-stress) stiffness K_geo = ∫G^T[σ]G dV for hexa8 / BT4 / truss, updated-Lagrangian reference frame (committed geometry advances per increment; midpoint Hughes–Winget stress update + end-configuration force assembly), exact corotational truss tangent (LAW1). Validated: Euler buckling of a shell-strip column vs π²EI/(4L²) (<3%), Euler's RELATION on a hexa column with the mesh's measured EI (<1% — see the hourglass note in the M9 roadmap entry), FD-exact residual/tangent consistency, small-strain limit reproduces M8 | ✅ |
| Linearized buckling eigensolver (K_mat + μ K_geo)φ = 0 (`implicit/buckling.py`, imp_buck.F analogue; dense eigh — fine at this port's model sizes) | ✅ (library function; /IMPL/BUCKL card deferred) |
| Arc-length continuation (spherical Riks/Crisfield: predictor sign-following, Crisfield root selection by path continuation, adaptive radius with cut-on-failure, exact landing on the final load factor). Validated: von Mises two-bar truss traced through BOTH limit points against the closed form (sampled peaks ≥ 99% of ±P_max, never exceeding them; far-branch equilibrium to 0.1%) | ✅ (`/IMPL/ARCL`; proportional force loading only, no /IMPDISP) |
| Statics bulk-viscosity fix (M9): the implicit driver zeroes the solid qa/qb — a rate device that leaked a spurious viscous pressure into COMPRESSIVE increments of the pseudo-velocity residual (latent in M8: all its validations were tensile); uniaxial compression is now exactly Hooke (asserted) | ✅ |
| **Implicit DYNAMICS (M10)**: Newmark-β time integration with HHT-α numerical dissipation (`/IMPL/DYNA/1|2`, imp_dyna.F) — lumped (diagonal) mass in equation space from the starter's model.mass/model.inertia (no consistent mass, like the original's MS/IN), dynamic residual R = (1+α)(f_ext+f_int)ₙ₊₁ − α(f_ext+f_int)ₙ − M aₙ₊₁, effective tangent K_eff = (1+α)K_T + M/(β dt²), Newmark a/v recovery per converged step; BOTH geometry modes (M8 linear + M9 /IMPL/NONLIN updated-Lagrangian); /IMPDISP at physical time; per-step energy ledger (KE/IE/W_ext/balance) in the result history. Validated: SDOF period elongation MATCHING the closed-form (ω dt)²/12 dispersion (and its 4× drop when dt halves), exact amplitude, energy conservation to round-off (trapezoidal), stability at 20× the leapfrog limit (leapfrog divergence asserted on the same system), HHT high-mode dissipation vs trapezoidal conservation on a bar, quasi-static limit = the M8 static answer, transient cross-checked against the EXPLICIT solver at the response peak (0.5%), large-rotation pendulum vs the elliptic-integral period (0.5%) with quadratic Newton | ✅ |
| Rate effects under implicit dynamics: the kernels are driven with the step increment as a pseudo-velocity at dt = 1 (what the tangents linearize), so the rate devices are DISABLED explicitly — bulk viscosity zeroed (as statics), LAW2 strain-rate term zeroed with a warning — never silently fed du/1 | ✅ (deliberate deferral of rate-dependent plasticity — see dynamics.py) |
| **Implicit COMPLETENESS (M11)**: element tangents for EVERY family — tetra4 (full-rank V·BᵀDB, no hourglass block needed), sh3n C0 triangle (membrane/bending/shear blocks + drilling penalty, per-layer LAW2 integration), corotational Timoshenko beam 12×12 (the exact-dt eigenproblem's own L·BᵀCB rotated by the frame; axial K_geo — the consistent initial-stress operator of the LINEAR element), spring TYPE4 (total-form element with its OWN implicit residual: incremental on the committed force under linear geometry, exact total form under NLGEOM) — plus beam rotational DOFs in the equation numbering and the zero-mass orientation-node exclusion. FD residual/tangent consistency exact at zero stress for all four (beam/spring exact including prestress); closed-form checks: tetra constant-strain patch (exact, one step), sh3n membrane patch (exact) + thick cantilever vs Timoshenko (< 2%), beam cantilever vs FL³/3EI + FL/GA (< 0.5%), spring u = F/k (exact); a mixed model with all seven families converging in one Newton step | ✅ |
| LAW2 consistent PLANE-STRESS (shell) tangent (M11 — the Iplas=2 radial projection's algorithmic tangent, mildly nonsymmetric rank-one update, per-layer thickness integration with the A/B/D moment matrices — the B coupling block carries a plastified stack's neutral-surface shift) + the elastoplastic TRUSS (implicit path upgraded to the ITERATED 1-D consistency solve — the explicit kernel's one-step return veers off the hardening curve at implicit increment sizes, measured and documented; consistent modulus E·H/(E+H)). Validated: BT4 + sh3n uniaxial past yield matching the Iplas=2 algorithm's own closed form u/L = σ/E + (3G/E)·εp with εp exactly on the JC curve and QUADRATIC Newton tails; truss vs the 1-D closed form | ✅ |
| Rayleigh damping in the implicit system (M11, /IMPL/DYNA/DAMP — imp_dyna.F IDY_DAMP): C = a·M + b·K(step start), damping force HHT-weighted like f_int (IMP_DYNAR), exact velocity linearization (1+α)γ/(βdt)·C in K_eff (the IMP_DYNAM BDT/S0 algebra), trapezoidal DY_EDAMP dissipation ledger in the energy balance. Validated: damped SDOF vs exp(−ζωt)·sin(ω_d t) and the damped period for mass-only / stiffness-only / mixed Rayleigh; balance closes to round-off WITH the ledger; dissipated fraction matches 1 − exp(−2ζωt) | ✅ |
| Automatic implicit step control (M11, imp_dt.F IMP_DTN): cut-and-RETRY on non-convergence (rollback is free — failed increments never touch model.x and the element buffers re-base from the committed snapshots), growth back toward /IMPL/DTINI after ≤-target-iteration steps; statics AND dynamics. Validated: a one-increment deep-elastica run that fails its Newton budget now completes through cuts and matches the fine fixed-increment answer to 0.5%; smooth runs take zero cuts and reproduce the fixed-dt stepping exactly | ✅ |
| /IMPL/BUCKL engine card (M11): prestress increments → (K_mat + μ K_geo)φ = 0 → factors/modes reported (listing + result object). Validated: the M9 shell Euler column through the CARD path (π²EI/4L² < 3%) | ✅ |
| Implicit↔explicit switching mid-run; /IMPVEL under implicit dynamics (refused — use /IMPDISP); rate devices under implicit (LAW2 strain-rate term, bulk viscosity, spring dashpot — all disabled loudly) | ❌ (deferred — see the M10/M11 roadmap notes) |
| **Implicit KINEMATIC CONSTRAINTS by condensation (M12)**: /RBODY, /RBE2, /INTER/TYPE2 tied, /RBE3 and /MPC in the implicit system — dependent DOFs eliminated through the sparse transform K_red = TᵀKT, R_red = TᵀR (the rby_imp0.F / rbe2_imp0.F / rbe3_imp0.F / i2_imp1.F block condensations; never penalized), in BOTH geometry modes (T rebuilt per committed frame, rigid bodies re-placed exactly with the Rodrigues map at each NLGEOM commit) and under /IMPL/DYNA (TᵀMT carries the exact rigid 6-DOF mass at the master — total mass, parallel-axis inertia, COG-coupling blocks — and initial velocities project onto the constraint manifold mass-weighted = the explicit momentum projection). Standalone frozen masters unfrozen and force-numbered; /BCS on a master = the body-level condition (all-translations-fixed = the PIVOT); /BCS on dependents warned, constraint wins. Validated: RBE2 rigid-lever closed form exact in one Newton step (both modes), the condensed master 6×6 mass block vs the parallel-axis closed form, /MPC equality split exact, /RBE3 dual lever rule exact, /RBODY pivot static rotation exact + the implicit-dynamic physical pendulum on the elliptic-integral quarter period (< 0.5%, amplitude preserved, arms exact), spot-weld lap joint implicit = explicit quasi-static (< 2%) | ✅ |
| **Implicit PENALTY CONTACT (M12)**: /INTER/TYPE7 in the Newton loop — frictionless contact force at the TRIAL configuration in the residual, exact gap tangent K·g gᵀ + the closest-point curvature −K·p·∇²d (region-wise: zero on faces, the point/edge lateral terms at vertices/edges — the i7keg3.F blocks omit it; an M12 lesson) in K/K_eff, ACTIVE-SET Newton (pairs enter/leave per iteration; a set that will not settle lands in the M11 StepControl cut), Istf/Igap stiffness+gap machinery of contact/stiffness.py reused unchanged, stored spring energy ½Kp² in its own dynamics ledger channel. Validated: two blocks pressed = series-springs closed form EXACT with the active set entering mid-run, FD residual/tangent consistency (exact for secondary-side directions; ≤ 5% full-direction, the documented weight-variation omission), implicit punch = explicit damped steady state (< 2%) and the dead-load closed form (1e-6) | ✅ |
| A constraint-condensation predictor lesson (M12, recorded in `constraints.make_consistent`): the dynamics predictor must be PROJECTED onto the constraint manifold (u = T u_red always) — the node-space extrapolation violates the constraint by O(θ²)·arm, Newton cannot remove what Tᵀ annihilates, and the commit placement silently converts the violation into energy (found by a pendulum that gained 20× its drop energy and circulated) | ✅ |
| **Implicit CONTACT & LOAD COMPLETION (M13)**: /INTER/TYPE7 COULOMB FRICTION in the Newton loop (the i7kfor3.F incremental return mapping — stick/slip with the consistent nonsymmetric slip tangent, anchors committed per increment, `efric` slip-work ledger channel, µ = 0 bit-identical to M12); /INTER/TYPE11 edge-to-edge under implicit (exact segment-segment gap tangent + the EXACT edge-edge closest-point curvature in every projection region; near-parallel overlaps by two-point trapezoid quadrature — the period-2 lesson); /PLOAD follower-load stiffness under /IMPL/NONLIN (trial-configuration pressure residual + the exact nonsymmetric −∂f_ext/∂x; IMP_KPRES analogue, documented deviation) with /PLOAD + /IMPL/ARCL refused; LAW36 consistent tangents (solids + shells, table-slope H; the piecewise-linear return measured EXACT at implicit increments; rate families truncated to the static curve, warned). Two SOLVER lessons recorded in the code: the imp_solv.F-style backtracking LINE SEARCH (engages only when the residual GROWS — smooth runs bit-identical) that breaks non-smooth assignment cycles, and the PERSISTENT implicit hourglass state `hgq` (the incremental static stabilization forgot accumulated hourglass deformation at every commit and the modes ratcheted — latent since M8, exposed by moment-loaded corner forces) | ✅ |
| **Implicit GENERALITY (M14)**: constraint CHAINS resolved by transform substitution (rigid-on-rigid, /MPC rows on rigid slaves, /RBE3 masters/ties inside bodies — the rbody_part_modif.F90 hierarchy expressed as T = T1·T2·…; CIRCULAR chains refused; conflicts refused; the explicit engine refuses chains loudly), with the topological commit placement under NLGEOM and the chained TᵀMT mass under /IMPL/DYNA (chained pendulum on the elliptic-integral period); /INTER/TYPE11 COULOMB FRICTION under implicit (the M13 TYPE7 return mapping generalized to edge pairs, consistent stick/slip tangents, anchors keyed per (edge, edge, overlap-end) — the original's I11KFOR3 has NO cone cap: documented deviation); /IMPL/ARCL WITH constraints & contact (reduced-space corrector solves + spherical metric — documented deviation from the full-field PRODUT_UHP0 Riks norm; contact active set re-evaluated inside the corrector, validated by the snap-catch); /IMPL/BUCKL WITH constraints & contact (the reduced pencil, as imp_buck.F's UPD_GLOB_K; the converged contact tangent in K_mat — imp_buck.F omits contact, documented deviation); LAW42 CONSISTENT spectral tangent (uniaxial/equibiaxial exact with quadratic tails; the coalescent-stretch L'Hôpital branch FD-verified; /IMPL/NONLIN required, frozen-frame runs refused) | ✅ |
| **Friction MODELS (M15)**: MFROT 1–4 µ(p, v) + IFQ filtering in the EXPLICIT TYPE7 (and TYPE11 as a documented port extension), the STATIC-LIMIT µ(p) Coulomb cone with the consistent µ′(p) coupling tangent in the IMPLICIT TYPE7/TYPE11 return mapping; LAW27 implicit shell tangent (per-branch: uncracked/open-frozen/open-growing/closed/broken); LAW2 BEAM resultant-plasticity consistent tangent + iterated implicit return. Validated: exact MFROT formula/branch checks, kernel-level transmitted-force closed forms, the exact discrete IFQ step response, implicit stick/slip closed forms with µ(p), FD tangent consistency in every regime (µ′ block included), pre-crack = LAW1 exact, crack-closure stiffness recovery, notched-strip implicit-vs-explicit crack pattern, the beam hinge at EXACTLY the resultant limit load with the root element ON the yield surface to 1e-9, quadratic tails, cross-solver checks, bit-identity of every switched-off path (monkeypatch-asserted) | ✅ |
| **CONSISTENT ELEMENT MASS + MODAL analysis (M16)**: a `consistent_mass()` per family alongside the lumped mass — the ∫ρ Nᵀ N dV translational brick/tetra mass (analytic tet, 2×2×2 Gauss brick), the bilinear/CST membrane+bending+rotary shell mass (isotropic ρt / ρt³/12 blocks, drilling inertia included so the reduced mass stays PD), the 12×12 Rayleigh–Timoshenko beam mass (axial + torsion + two-plane Hermite-cubic bending with rotary-inertia coupling, rotated by the corotational frame), the exact truss/spring mass — each carrying its Fortran-origin + shape-integral docstring; the global consistent M assembled COO→CSR through the DofMap and condensed TᵀMT under every M12/M14 constraint (the exact rigid-body parallel-axis block, now fed by the consistent mass). MODAL EIGENVALUE extraction (`/IMPL/EIGV`, a PORT card — freimpl.F has no modal branch, so ported library-first like the M9 buckling eigensolver): (K − ω²M)φ = 0 for the lowest N natural frequencies + mass-normalized mode shapes + modal effective mass, in the REDUCED (constrained) space via `buckling.py`'s dense `eigh` (Lanczos/subspace deferred, documented), on a PRESTRESSED state too (K = K_mat + K_geo, `/IMPL/EIGV/STRS`). Validated: longitudinal bar vs (2n−1)c/4L and nc/2L, cantilever bending vs the Euler–Bernoulli βₙL roots (< 0.3% at 20 beams), a simply-supported plate fundamental (a/t = 20, the BT4 thin-plate shear-lock documented + convergence shown), mode M-orthogonality φᵢᵀMφⱼ = δᵢⱼ to round-off, the consistent-over/lumped-under BRACKET around the exact frequency, the rigid-link spectrum shift sqrt(k/(m₁+m₂)) + the rigid-bar parallel-axis inertia mL²/3, the taut-string prestressed modes n/2L·sqrt(T/μ) (pure geometric-stiffness modes); the LUMPED path (explicit leapfrog + M10 implicit dynamics) BIT-IDENTICAL, consistent_mass() never mutating element state (monkeypatch-asserted, the M7 parity contract) | ✅ |
| /RWALL under implicit (refused loudly); /IMPDISP on constraint nodes; /PLOAD with /IMPL/ARCL (refused); MFROT velocity terms + IFQ under implicit (static limit, loudly); Ifiltr ≥ 10 / MODFR 2 (refused); /FRICTION per-part-pair sets; orthotropic friction | ❌ (deferred — see the M12/M13/M14/M15 roadmap notes) |
| **MODAL-SUPERPOSITION dynamics (M17)**: mode-superposition TRANSIENT (`/IMPL/MODAL/DYNA`) + harmonic / frequency response (`/IMPL/FREQ`) + modal damping, consuming the M16 real eigenpairs (u = Σφᵢqᵢ, decoupled damped SDOFs). MODAL DAMPING — uniform ζ, a (freq, ζ) table interpolated onto the ωᵢ, and the Rayleigh map ζᵢ = ½(α/ωᵢ + βωᵢ) so a /IMPL/DYNA/DAMP α,β becomes modal damping CONSISTENT with the M11 direct integrator (validated against the direct damped-decay envelope). MODAL TRANSIENT — the EXACT Nigam–Jennings piecewise-linear-forcing recurrence per mode (step/impulse response matched POINTWISE + bit-close to the DIRECT M10 Newmark at a peak on a spring-mass sharing K,M; modal-truncation convergence to DAF = 2; the mode-ACCELERATION / residual-flexibility static correction recovering the EXACT static tail from one mode — a documented deviation), the M10 energy ledger in modal coordinates. HARMONIC RESPONSE — the complex FRF qᵢ(Ω) = φᵢᵀF/(ωᵢ²−Ω²+2iζᵢωᵢΩ), swept, amplitude/phase (SDOF peak = 1/(2ζ), half-power Δω/ω = 2ζ; resonances coinciding with the M16 natural frequencies; a driven-cantilever tip amplitude; base-excitation feeding the M16 effective-mass participation). PORT cards, library-first like /IMPL/EIGV; the M10 Newmark integrator + M16 eigensolver stay BIT-IDENTICAL (monkeypatch-asserted parity contract) | ✅ |
| **COMPLEX / DAMPED eigenvalues + NON-CLASSICAL damping (M18)**: the ASSEMBLED damping C (`implicit/damping_matrix.py`) = Rayleigh αM + βK (M11-consistent) + discrete dashpots (the /PROP/SPRING `c` term c·aaᵀ, revived from its M11 implicit deferral; per-node /DAMP mass dampers), COO→CSR through the DofMap and condensed TᵀCT like M16's mass. The COMPLEX eigenproblem (λ²M + λC + K)φ = 0 (`implicit/complex_modal.py`, `/IMPL/CEIGV`) via the SYMMETRIC state-space linearization A z = λB z (A = [[0,K],[K,C]], B = [[K,0],[0,−M]]; no mass inverse, symmetric so the biorthogonality is z_iᵀBz_j = 0), `scipy.linalg.eig` on the reduced pencil → λᵢ = −ζᵢωᵢ ± iωd,ᵢ (decay rate + damped frequency) + COMPLEX mode shapes (the DOF phase lag of non-proportional damping). COMPLEX-MODE SUPERPOSITION — the state-space decoupling into first-order complex modal equations ẋᵢ = λᵢxᵢ + pᵢ(t) (an EXACT piecewise-linear recurrence, the first-order analogue of Nigam–Jennings) + the damped complex FRF. Validated: a classically-damped (Rayleigh) system reducing EXACTLY to −ζᵢωᵢ ± iωd,ᵢ with the M16 ωᵢ and M17 ζᵢ (real-up-to-phase shapes); a 2-DOF one-dashpot system matching the closed-form complex roots + a genuine phase lag; the state-space biorthogonality; the discrete-dashpot C contribution + pure-Rayleigh C reproducing the M17 ζᵢ; the complex-mode transient matching a DIRECT Newmark march of (K,C,M) where the M17 REAL-mode superposition provably errs (gap asserted); reduction to the M17 answer when damping IS classical; the complex FRF matching (K−Ω²M+iΩC)⁻¹ on the full basis. PORT card, library-first like M16/M17; the M10 integrator, M16 REAL eigensolver and M17 REAL-mode superposition stay BIT-IDENTICAL | ✅ |
| **RANDOM / SPECTRAL (PSD) response + RESPONSE SPECTRA (M19)**: the stationary random-vibration and design-envelope analyses that build on the M17/M18 FRFs (`implicit/random_response.py`, `implicit/response_spectrum.py`, `/IMPL/PSD`, `/IMPL/RSPEC`). RANDOM (PSD) — the response PSD S_uu(Ω) = \|H(Ω)\|² S_ff(Ω) through the modal transfer function (FRF-source-agnostic: the M17 real FRF for classical damping, the M18 complex FRF for non-classical), reporting the RMS σ_u = √m₀, the spectral moments m₀/m₁/m₂ (mₙ = (1/π)∫₀^∞ Ωⁿ S_uu dΩ), the mean zero-crossing rate ν₀ = (1/2π)√(m₂/m₀) and peak rate ν_p = (1/2π)√(m₄/m₂) (Rice), and the base-excitation (support-motion) PSD feed via the M16 participation. RESPONSE SPECTRA — the per-mode peaks rᵢ = Γᵢ Sa(ωᵢ,ζᵢ)/ωᵢ² φᵢ combined by SRSS and CQC (Der Kiureghian 1981 closed-form ρᵢⱼ). Validated: a white-noise SDOF's σ² = S₀/(2ck) closed form; the multi-DOF response PSD from the modal FRF matching the DIRECT (K−Ω²M+iΩC)⁻¹ inversion; the spectral-moment / Parseval identity m₀(velocity) = m₂(displacement); the RMS reducing to the static σ_f/k for a quasi-static band; the CQC closed-form ρᵢⱼ; a single-mode spectrum recovering Γ Sa/ω²; SRSS ≈ CQC for well-separated modes and the CQC-vs-SRSS GAP on a closely-spaced (near-degenerate tuning-fork) pair; the complex-FRF PSD reducing to the real-FRF PSD when damping is classical. PORT cards, library-first like M16-M18; the M10 integrator, M16/M17/M18 paths stay BIT-IDENTICAL (a NEW parallel path consuming their FRFs read-only — monkeypatch-asserted parity) | ✅ |
| **RANDOM-VIBRATION (SPECTRAL) FATIGUE (M20)**: the stress-life fatigue-damage estimate computed DIRECTLY from the M19 stress-PSD spectral moments (`implicit/spectral_fatigue.py`, `implicit/random_response.py` stress recovery, `/IMPL/FATIG`). STRESS-PSD RECOVERY — the per-mode STRESS modes σᵢ = (element stress operator)·φᵢ recovered by running the SAME force kernels the M8 static solve uses on each mode shape (read-only; sigma = C:B:u the linear stress operator on the zero-stress reference), the stress FRF H_σ(Ω) = Σᵢ σᵢ qᵢ(Ω) (stress commutes with the modal superposition) and the stress response PSD S_σσ(Ω) = \|H_σ\|² S_ff with its own moments m₀..m₄. DAMAGE MODELS — for an S-N curve N = C·S^−m under a Miner sum: the NARROW-BAND (Bendat 1964) closed-form Rayleigh-range Gamma-function damage E[D]/T = (ν₀/C)(2√2σ)^m Γ(1+m/2), DIRLIK (1985) (the empirical three-term rainflow-range PDF from m₀,m₁,m₂,m₄), WIRSCHING–LIGHT (1980) and TOVO–BENASCIUTTI (2005) as wide-band cross-checks — reporting the damage rate, equivalent stress and time-to-failure, plus a seeded Monte-Carlo rainflow (ASTM E1049) cross-check (synthesise a Gaussian history from the PSD, count, Miner-sum). Validated: the narrow-band closed form vs a hand Gamma evaluation; Dirlik → narrow-band as the bandwidth → 0 and the correct wide-band bias on a bimodal spectrum; the ASTM E1049 counter on its canonical example; the Monte-Carlo damage matching Dirlik within the documented scatter; the recovered static stress = the M8 static stress; the stress FRF = the direct stress operator on U(Ω). A PORT card, library-first like M16–M19; the M10 integrator, M16/M17/M18/M19 paths stay BIT-IDENTICAL (a NEW parallel path consuming the stress moments read-only — monkeypatch-asserted parity) | ✅ |
| **MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE (M21)**: the fatigue-damage estimate of a MULTIAXIAL stress STATE (`implicit/multiaxial_fatigue.py`, the per-element Voigt-block helpers + `_run_multiaxial` in `implicit/random_response.py`, `/IMPL/FATIG/MULT`). STRESS-TENSOR CROSS-PSD — the M20 vector stress modes give the FULL 6-Voigt stress FRF H_σ(Ω) (nf,6) per solid/shell element (read-only, the M20 recovery reused unchanged), and the stress-tensor cross-PSD S_σσ(Ω) = H_σ S_ff H_σᴴ (a 6×6 Hermitian matrix per frequency per element). EQUIVALENT-STRESS METHODS — reduce the 6×6 to a scalar EQUIVALENT-stress PSD and run the four M20 estimators on its moments: the frequency-domain EQUIVALENT VON MISES S_vm(Ω) = trace(Q·S_σσ) (Preumont & Piéfort 1994 / Pitoiset & Preumont 2000, the von Mises quadratic operator Q; evaluated through the cheap rank-1 HᴴQH form), the MAX-NORMAL-stress and MAX-SHEAR-stress CRITICAL PLANE (Carpinteri–Spagnoli / Cristofori–Susmel–Tovo, the projection σₙ = pₙᵀσ, τ = pₛᵀσ over a searched candidate-plane set; the shear direction from the leading eigenvector of the 2×2 in-plane covariance) — reporting the damage/equivalent-stress/life and the critical-plane orientation, picking the critical ELEMENT by von Mises Dirlik damage. A seeded MULTIVARIATE Monte-Carlo cross-check (per-bin eigendecomposition / Cholesky of S_σσ → correlated Gaussian component histories → project onto the critical plane → ASTM E1049 rainflow → Miner). Validated: uniaxial → the M20 scalar σₓₓ (cross-PSD collapse, von Mises reduction); the cross-PSD diagonal = the M20 per-component channel PSDs, Hermitian; the trace / quadratic-operator identity (trace(Q Mₙ) = the direct S_vm moments); the von Mises operator on the textbook invariants; pure shear recovering the 45° max-normal plane and the coordinate max-shear plane, both at the shear amplitude; p^T Mₙ p = the |H·p|² S scalar moments; the multivariate synthesis reproducing the full covariance (variances AND cross-covariances = M₀); the Monte-Carlo critical-plane damage matching Dirlik within scatter. PORT sub-card, library-first like M16–M20; the M10 integrator and the M16–M20 paths (incl. the M20 SCALAR fatigue) stay BIT-IDENTICAL (a NEW parallel path consuming the vector stress modes read-only — asserted). Theory: Preumont & Piéfort 1994; Pitoiset & Preumont 2000; Carpinteri–Spagnoli 2001; Cristofori/Susmel–Tovo 2008; Socie & Marquis; Shinozuka & Deodatis | ✅ |
| **NON-PROPORTIONAL MULTIAXIAL FATIGUE (M22)**: the CRITICAL-PLANE, TIME-DOMAIN, PATH-COUNTING damage of a rotating (non-proportional) shear path (`implicit/nonproportional_fatigue.py`, the `_run_nonproportional` extension of `_run_multiaxial` in `implicit/random_response.py`, `/IMPL/FATIG/MULT/NPROP`) — the rotating-shear PATH the M21 projected-equivalent spectral methods deferred, consuming the M21 multivariate synthesiser + candidate-plane machinery read-only. SHEAR-PATH AMPLITUDE OPERATORS — the 2-D resolved-shear locus (τ_a(t),τ_b(t)) on each candidate plane, and its amplitude by the MINIMUM CIRCUMSCRIBED CIRCLE (Papadopoulos 1998, Welzl on the hull), the LONGEST CHORD (the diameter) and the MAXIMUM RECTANGULAR / PRISMATIC HULL (Mamiya–Araújo–Castro 2009, max_θ √(a1²+a2²)), plus the non-proportionality factor F_np = √(λ₂/λ₁) of the shear-path covariance (Itoh–Kanazawa aspect-ratio form: 0 for a line, 1 for a circle). CRITICAL-PLANE TIME-DOMAIN DAMAGE — over the M21-synthesised correlated stress-component histories (seeded), rainflow (ASTM E1049) the dominant resolved shear scaled by the non-proportional path factor g = MRH/scalar, and apply FINDLEY (τ_a + k σ_n,max) and FATEMI–SOCIE (γ_a(1 + k σ_n,max/σ_y)) carrying the per-plane MAX NORMAL stress (folding in the plane mean), searching the critical plane by the criterion parameter and reporting the plane / F_np / damage rate / life. Validated: proportional line → all three operators collapse to the scalar M21 amplitude, F_np = 0; circle → MCC = r, chord = 2r, MRH = r√2, F_np = 1; ellipse MRH = √(p²+q²), MCC = major semi-axis, F_np = q/p; the off-centre-circle MCC recovering radius + centre; proportional path count → the M21 max-shear rainflow within scatter (g = 1); 90°-out-of-phase → g^m ≈ 2^(m/2) higher than the scalar projection with F_np ≈ 1; a two-channel-sinusoid hand check of MCC / MRH / Findley / Fatemi-Socie. PORT sub-flag, library-first like M16–M21; the M10 integrator, the M16–M20 paths AND the M21 SPECTRAL reductions stay BIT-IDENTICAL (a NEW parallel path — the M21 spectral damage rates AND critical-plane orientations byte-identical whether or not /NPROP runs — asserted). Theory: Papadopoulos 1998; Mamiya–Araújo–Castro 2009; Findley 1959; Fatemi–Socie 1988; Matake 1977; Carpinteri–Spagnoli 2001; Itoh–Sakane–Socie 1995; Socie & Marquis ch. 2–4 | ✅ |
| **SPECTRAL NON-PROPORTIONAL MULTIAXIAL FATIGUE (M23)**: the FREQUENCY-DOMAIN non-proportionality factor and critical-plane damage estimated DIRECTLY from the stress-tensor cross-PSD spectral-MOMENT matrices, with NO synthesised history (`implicit/spectral_nonproportional_fatigue.py`, the `_run_spectral_nonproportional` extension of `_run_multiaxial` in `implicit/random_response.py`, `/IMPL/FATIG/MULT/NPROP/SPEC`) — the spectral estimator the M22 time-domain path count deferred, the SPECTRAL sibling of M22, both built on the M21 moment-matrix / candidate-plane machinery (read-only). SPECTRAL SHEAR-PATH COVARIANCE + F_np — the 2×2 in-plane shear block Σ_τ of M₀ = E[σσᵀ] (`inplane_shear_covariance`, from `pₐᵀM₀pₐ`, `pₐᵀM₀p_b`, `p_bᵀM₀p_b`), and F_np = √(λ₂/λ₁) of its eigenvalues (`spectral_nonproportionality_factor`) — EQUALS the M22 TIME-DOMAIN F_np (the identity M₀ = E[σσᵀ] = the covariance the synthesiser reproduces), computed with no synthesis. SPECTRAL CRITICAL-PLANE DAMAGE — the dominant-shear PSD moments mₙ^τ = p_domᵀMₙp_dom and resolved-normal moments p_nᵀMₙp_n (from p^T Mₙ p, no history), the closed-form non-proportional amplitude correction g² = 1 + F_np² = trace(Σ_τ)/λ₁ (the exact MRH/MCC ratio of an elliptical path), the Susmel–Tovo modified-Wöhler stress ratio ρ = σ_a/τ_a, the Findley (spectral-invariant linear combination g·τ + k·σ_n) / Fatemi–Socie / shear-path models, searched by the amplitude-only criterion parameter and run through the four M20 estimators on the corrected moments. Validated: the 2×2 shear block = the M22 time-domain shear-path covariance; spectral F_np = M22 time-domain F_np; F_np = 0 for a proportional (rank-1) state, → 1 for the 90°-out-of-phase circle (closed forms); the resolved amplitudes matching the M22 synthesised RMS; proportional → the M21 max-shear spectral answer EXACTLY (no correction); 90°-out-of-phase → g^m = 2^(m/2) higher than the M21 projected scalar AND agreeing with the M22 time-domain path count within scatter; a two-channel-cross-PSD hand check of F_np / ρ / g. PORT sub-flag, library-first like M16–M22; the M10 integrator, the M16–M20 paths, the M21 SPECTRAL reductions AND the M22 TIME-DOMAIN path count stay BIT-IDENTICAL (a NEW parallel path — both byte-identical whether or not /SPEC runs — asserted). Theory: Pitoiset & Preumont 2000; Cristofori–Susmel–Tovo 2008; Susmel & Lazzarin (modified Wöhler curve) 2002; Bäckström & Marquis 2001; Carpinteri–Spagnoli–Vantadori 2013 | ✅ |
| **NON-GAUSSIAN / KURTOSIS SPECTRAL FATIGUE (M24)**: the frequency-domain damage of a stationary but NON-GAUSSIAN random-vibration response, computed by CORRECTING the M20–M23 Gaussian spectral estimators for a specified kurtosis / skewness (`implicit/nongaussian_fatigue.py`, the `_run_nongaussian` / `_run_nongaussian_multiaxial` extensions of the M20/M21 drivers in `implicit/random_response.py`, `/IMPL/FATIG/NGAUSS`) — the FIRST fatigue item deferred out of M20–M23 that BUILDS on the spectral-fatigue machinery (every prior estimator assumed a stationary GAUSSIAN response; M24 lifts exactly that). WINTERSTEIN HERMITE-MOMENT MODEL — g(u) = κ[u + h₃(u²−1) + h₄(u³−3u)] (`hermite_coefficients` / `hermite_transform`) maps a standard Gaussian u to the non-Gaussian process for a target γ₄/γ₃; Gaussian-orthogonality gives EXACTLY zero mean and (κ = 1/√(1+2h₃²+6h₄²)) unit variance, the softening fit h₄ = (√(1+1.5(γ₄−3))−1)/18, h₃ = γ₃/(6(1+6h₄)). CORRECTION FACTOR — λ_ng = E[g(V)ᵐ]/E[Vᵐ] over the Rayleigh(1) amplitude V (`nongaussian_correction_factor`, Benasciutti–Braccesi / Rizzi–Kihm), scaling the Gaussian narrow-band / Dirlik / Wirsching-Light / Tovo-Benasciutti damage (`nongaussian_summary`), with an optional Benasciutti–Tovo bandwidth attenuation γ₄,eff = 3 + (γ₄−3)α₂. NON-GAUSSIAN MONTE-CARLO — the M20 Gaussian history pushed through the memoryless Hermite transform to (γ₃, γ₄) (`synthesize_nongaussian_history`), ASTM E1049 rainflow + Miner (`nongaussian_monte_carlo_damage`; `_projected` for the multiaxial critical-plane scalar). Validated: λ_ng = 1 for a Gaussian process (γ₄ = 3, γ₃ = 0) — the M20 answer recovered EXACTLY; λ_ng > 1 leptokurtic (γ₄ > 3), < 1 platykurtic (γ₄ < 3), monotone in γ₄ and m; the Hermite transform preserving the mean/variance EXACTLY and hitting the target kurtosis (exact-moment + large-sample hand checks); the non-Gaussian Monte-Carlo damage matching the λ_ng-corrected spectral estimate within the seeded scatter on a narrow band, and reducing EXACTLY to the M20 Gaussian Monte-Carlo in the Gaussian limit; composes with /MULT, /NPROP, /SPEC (a scalar correction on the equivalent-stress PSD the M21/M23 reductions produce). PORT sub-flag, library-first like M16–M23; the M10 integrator, the M16–M20 paths AND the M21–M23 MULTIAXIAL reductions stay BIT-IDENTICAL (a NEW parallel path — the M20–M23 answers byte-identical whether or not /NGAUSS runs — asserted). Theory: Winterstein 1988; Winterstein & MacKenzie 1997; Benasciutti & Tovo 2005/2006; Braccesi–Cianetti–Lori–Pioli 2009; Rizzi–Kihm–Ferguson; Kihm & Rizzi 2013 | ✅ |
| **NON-STATIONARY / EVOLUTIONARY-PSD SPECTRAL FATIGUE (M25)**: the frequency-domain damage of a random-vibration response whose PSD / RMS VARIES WITH TIME, computed by extending the M20–M24 STATIONARY estimators to a non-stationary process (`implicit/nonstationary_fatigue.py`, the `_run_nonstationary` / `_run_nonstationary_multiaxial` extensions of the M20/M21 drivers in `implicit/random_response.py`, `/IMPL/FATIG/NSTAT`) — the FIRST fatigue item deferred out of M20–M24 that BUILDS on the spectral machinery (every prior estimator, Gaussian OR non-Gaussian, assumed a single time-invariant PSD; M25 lifts exactly that stationarity, the M24 sibling). PIECEWISE-STATIONARY / BLOCK ("mission profile") MODEL — partition into stationary blocks (RMS scaling a_i of a shared shape, moments m_n → a_i²m_n, duration T_i), run the M20 estimators per block and Palmgren–Miner SUM D = Σ (E[D]/T)_i T_i (`block_fatigue_summary`; every estimator's damage scales EXACTLY as a_i^m). AMPLITUDE-MODULATED / EVOLUTIONARY MODEL — Priestley's separable S(ω,t) = |A(t)|² S(ω), the stationary damage integrated over the RMS distribution, the closed-form E[a^m]-weighted damage (`amplitude_modulated_summary`). M25↔M24 BRIDGE — γ₄ = 3·E[a⁴]/E[a²]² (Wolfsteiner–Trapp / Kihm–Rizzi), κ_ns = E[a^m]/E[a²]^(m/2) (`nonstationary_amplification`) agreeing with the M24 λ_ng at that kurtosis (`bridge_to_nongaussian`, `nongaussian_fatigue` read-only). NON-STATIONARY MONTE-CARLO — the Gaussian carrier × a time-varying RMS envelope, ASTM E1049 rainflow + Miner (`synthesize_nonstationary_history` / `nonstationary_monte_carlo_damage`). Validated: the constant-modulation limit recovering the M20 answer EXACTLY (E[a^m] = 1, κ_ns = 1, MC bit-identical); the block Miner-sum = the duration-weighted per-block damages (hand check) and = the amplitude-modulated E[a^m] damage for a shared shape; κ_ns agreeing with λ_ng at the induced kurtosis (both directions, the constant limit exact); the non-stationary MC hitting the induced sample kurtosis / schedule RMS and matching the block estimate within scatter; composes with /MULT, /NPROP, /SPEC, /NGAUSS. PORT sub-flag, library-first like M16–M24; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions AND the M24 NON-GAUSSIAN correction stay BIT-IDENTICAL (a NEW parallel path — the M20–M24 answers byte-identical whether or not /NSTAT runs — asserted). Theory: Priestley 1965; Bendat & Piersol; Wolfsteiner & Breuer / Wolfsteiner & Trapp; Braccesi–Cianetti–Lori–Pioli; Kihm–Ferguson–Antoni; Rychlik; Palmgren–Miner | ✅ |
| **FULLY EVOLUTIONARY / NON-SEPARABLE-PSD SPECTRAL FATIGUE (M26)**: the frequency-domain damage of a random-vibration response whose spectral SHAPE (not merely its RMS level) VARIES WITH TIME, computed by extending the M25 piecewise-stationary / amplitude-modulated estimators to a genuinely NON-SEPARABLE evolutionary spectrum `S(ω,t)` / spectrogram (`implicit/evolutionary_fatigue.py`, the `_run_evolutionary` / `_run_evolutionary_multiaxial` extensions of the M20/M21 drivers in `implicit/random_response.py`, `/IMPL/FATIG/EVOL`) — the FIRST fatigue item deferred out of M25 that BUILDS on the non-stationary spectral machinery (M25 lifted stationarity only in the RMS ENVELOPE, the separable `|A(t)|²S(ω)` case with a FIXED shape; M26 lifts SEPARABILITY itself, so each time-window carries its OWN full moment set m₀..m₄, not a scaling of a shared shape). SPECTROGRAM / PER-WINDOW-FULL-PSD MODEL — partition into time-windows each with its OWN complete stress PSD (different bandwidth / rates / centre frequency per window), run the M20 estimators PER WINDOW and Palmgren–Miner SUM the window damages duration-weighted (`evolutionary_fatigue_summary`, driven read-only through the M25 `block_fatigue_summary` which already accepts per-window distinct moments — M26 supplies the real per-window PSDs of a drifting shape, of which M25's shared-shape scaling is the constant-shape reduction). TIME-FREQUENCY / DRIFTING-SHAPE MODEL — an evolutionary spectrum built from a smoothly time-varying parameter (a swept centre frequency `f_c(t)` — a "chirp" — and/or a broadening bandwidth `b(t)`), sampled into windows whose full moments are recomputed FROM the drifting shape (`drifting_shape_spectrogram` applies a swept/broadening Gaussian window `W_i(f)` to the recovered stress PSD — commutes with `\|H\|²`; `gaussian_evolutionary_spectrogram` builds one analytically). NON-SEPARABLE MONTE-CARLO with a TIME-VARYING FILTER — per-window spectral-representation blocks concatenated (`synthesize_evolutionary_history` / `evolutionary_monte_carlo_damage`), whose short-time spectrogram tracks `S(ω,t)`; in the constant-shape limit it DELEGATES to the M25 `synthesize_nonstationary_history` (bit-identical). Validated: a single window recovering the M20 stationary answer EXACTLY; a constant-shape spectrogram recovering the M25 amplitude-modulated answer EXACTLY (the built-in M26↔M25 reduction); the window Miner-sum = the duration-weighted per-window damages (hand check); a two-window shape-swap (narrow→wide) differing from any single-shape M25 scaling (the non-separable point); the swept `f_c` drifting the per-window ν₀, the broadening `b` drifting the per-window α₂; the non-separable MC tracking the spectrogram and matching the window estimate within scatter, and reducing to the M25 (hence M20) MC bit-identically in the constant-shape (single-window) limit; composes with /MULT, /NPROP, /SPEC, /NGAUSS, /NSTAT. PORT sub-flag, library-first like M16–M25; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions, the M24 NON-GAUSSIAN correction AND the M25 NON-STATIONARY correction stay BIT-IDENTICAL (a NEW parallel path — the M20–M25 answers byte-identical whether or not /EVOL runs — asserted). Theory: Priestley 1965/1967 (evolutionary / non-separable spectra); Mark 1970 / Hammond (non-stationary spectral analysis); Newland (the spectrogram / short-time method); Bendat & Piersol; Wigner–Ville time-frequency view; Palmgren–Miner window sum | ✅ |
| **FULLY NON-STATIONARY / EVOLUTIONARY MULTIAXIAL (JOINT-TENSOR) FATIGUE (M27)**: the frequency-domain damage of a MULTIAXIAL random-vibration response whose full 6×6 stress-TENSOR cross-PSD `S_σσ(ω,t)` (not just an equivalent scalar) VARIES WITH TIME (`implicit/joint_evolutionary_fatigue.py`, the `_run_joint_evolutionary` extension of the M21 `_run_multiaxial` driver in `implicit/random_response.py`, `/IMPL/FATIG/MULT/EVOL/JOINT`) — the item M25/M26 DEFERRED (M25/M26 applied the block / evolutionary scaling to the EQUIVALENT SCALAR the M21/M23 reductions produce, a FIXED reduction; M27 lifts exactly that — the JOINT tensor evolves so the critical-plane orientation and F_np are RE-SEARCHED per window and may DRIFT). PER-WINDOW TENSOR CROSS-PSD — the M26 drifting-shape window `W_i(f)` multiplies the full 6×6 `S_σσ(ω)` per window (commutes with the reduction, as M26 established for `\|H\|²`), the per-window 6×6 spectral-MOMENT matrices `M_{n,i} = a_i²·(1/π)∫ω^n W_i(f) S_σσ dω` recomputed (`windowed_tensor_moment_matrices`), EACH window reduced by the M21 equivalent von Mises + max-normal / max-shear critical-plane search + the M23 `F_np`, the plane RE-SEARCHED from the window's OWN tensor (`reduce_window_tensor`, `joint_evolutionary_fatigue_summary`), and the per-window multiaxial damages Palmgren–Miner-summed. Reports the per-window critical-plane DRIFT (plane normal / `F_np` / equivalent-stress RMS), the plane ROTATION angle and the `F_np` drift. NON-STATIONARY MULTIVARIATE MONTE-CARLO — per-window multivariate spectral-representation blocks of the WINDOWED tensor (the M21 per-bin eigen/Cholesky synthesiser with a time-varying tensor cross-PSD), projected onto the window's OWN critical plane, ASTM E1049 rainflow + Miner (`synthesize_joint_evolutionary_history` / `joint_evolutionary_monte_carlo_damage`); the constant-shape limit delegates to a single multivariate carrier × envelope, a single unit window reducing to the M21 multivariate MC bit-identically. Validated: a STATIONARY tensor / single window recovering the M21 spectral multiaxial answer EXACTLY (all three reductions); a FIXED critical plane recovering the M26 scalar per-window spectrogram EXACTLY; a CONSTANT-tensor-shape / RMS-only drift recovering the M25 multiaxial block answer EXACTLY; a rotating-principal-axes case whose per-window critical plane genuinely DRIFTS and whose Miner-sum DIFFERS from the M26 fixed reduction (the M27↔M26 boundary made explicit); the windowed tensor moments commuting with the reduction; the MC tracking the spectrogram within scatter; composes with /NPROP, /SPEC, /NGAUSS, /NSTAT (/JOINT implies MULT+EVOL). PORT sub-flag, library-first like M16–M26; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions, the M24 NON-GAUSSIAN correction, the M25 NON-STATIONARY correction AND the M26 EVOLUTIONARY-SCALAR correction stay BIT-IDENTICAL (a NEW parallel path — the M20–M26 answers byte-identical whether or not /JOINT runs — asserted). Theory: Priestley 1965/1967 (evolutionary spectra, matrix-valued); Preumont & Piéfort 1994 / Pitoiset & Preumont 2000; Carpinteri–Spagnoli / Cristofori–Susmel–Tovo; Bäckström & Marquis; Palmgren–Miner | ✅ |
| **MULTI-INPUT / PARTIALLY-COHERENT RANDOM-VIBRATION RESPONSE & FATIGUE (M28)**: the stationary response (and stress-tensor) cross-PSD driven by SEVERAL simultaneous random inputs with a full Hermitian input cross-spectral matrix `S_ff(ω) = [√(G_a G_b) γ_ab e^{iθ_ab}]` (auto-PSDs on the diagonal, coherence γ_ab ∈ [0,1] + phase θ_ab off-diagonal), propagated through the VECTOR FRF by the MIMO relation `S_uu = H S_ff Hᴴ` / `S_σσ = H_σ S_ff H_σᴴ` and reduced by the whole M20–M27 estimator family UNCHANGED (`implicit/multi_input_response.py` + `implicit/multi_input_fatigue.py`, the `_run_multi_input_fatigue` / `_run_multi_input_response` extensions of the M19/M21 drivers in `implicit/random_response.py`, `/IMPL/PSD/MULTI` + `/IMPL/FATIG/MULT/MINPUT`) — the LAST item on the recurring deferral tail carried since M19/M20/M21 (the multi-input cross-PSD with coherence deferred in every spectral milestone). Where M19/M20/M21 assumed ONE scalar input and formed the RANK-1 `H S_ff Hᴴ` with a scalar `S_ff`, M28 generalises `S_ff` to a NON-DIAGONAL ninput×ninput Hermitian matrix and contracts it with a per-input FRF COLUMN stack `H(ω)` (nf, ndof, ninput). INPUT CROSS-PSD MODEL — `input_cross_psd_matrix` assembles the Hermitian `S_ff` from per-input auto-PSDs and a coherence model (`constant_coherence` OR the exponential/decay `exponential_coherence` for distributed loads), projected onto the nearest Hermitian PSD matrix (`nearest_psd`, clip negative eigenvalues, Higham 1988, a no-op on a valid matrix). The per-input FRF columns (`stress_frf_columns` / `displacement_frf_columns`, reusing the M17/M19 machinery read-only) and the batched triple products (`response_cross_psd(_diagonal)`, `stress_tensor_cross_psd_multi`); the single-input (ninput = 1) case delegates to the M21 rank-1 routines (bit-identical). `multi_input_multiaxial_summary` reduces `S_σσ` by the SAME M21 machinery (returning the SAME dict shape, so the M22–M27 corrections compose UNCHANGED). MULTI-INPUT MONTE-CARLO — synthesise the ninput CORRELATED input histories from the per-bin eigen/Cholesky factor of `S_ff` (the M21 synthesiser applied to the INPUT matrix, `synthesize_multi_input_forces` / `synthesize_multi_input_stress`), drive each through its stress FRF column, sum, ASTM E1049 rainflow + Miner (`monte_carlo_multi_input_damage`); the single-input / fully-coherent (rank-1) cases DELEGATE to the M21 MC (bit-identical), the partially-coherent case runs the input-level synthesis and is cross-checked against the stress-level (M21-on-`S_σσ`) answer within scatter; `measure_coherence` confirms the synthesised inputs' coherence matches γ_ab. Validated: the single-input cross-PSD BIT-IDENTICAL to the M21 rank-1 / M19 |U|²G answers; the coherence identity |S_ff[a,b]|² = γ_ab²G_aG_b; the PSD projection a no-op on a valid matrix; a diagonal `S_ff` = the SUM of the per-input answers EXACTLY; a rank-1 coherent `S_ff` = the single-input answer for the combined pattern EXACTLY; partial coherence interpolating monotonically; a two-input SDOF closed form; the M21 reductions byte-identical given the same `S_σσ`; the fully-coherent MC reducing to the M21 MC bit-identically; the incoherent variance additive + the input-level MC matching the stress-level MC within scatter; the synthesised coherence matching γ_ab. Composes with /NPROP, /SPEC, /NGAUSS, /NSTAT, /EVOL, /JOINT (the multi-input `S_σσ` flows into them unchanged), reported ALONGSIDE the single-input numbers (a `multi_input` sub-entry). PORT sub-flag, library-first like M16–M27; the M10 integrator, the M16–M20 paths, the M21–M23 MULTIAXIAL reductions and the M24–M27 corrections stay BIT-IDENTICAL (a NEW parallel path — the single-input answers byte-identical whether or not the multi-input path runs — asserted). Theory: Newland ch. 6–8 (multiple correlated inputs, the coherence function); Bendat & Piersol ch. 5–7 (the cross-spectral / coherence matrix, the `H S_ff Hᴴ` MIMO relation); Wirsching–Paez–Ortiz (multi-input random fatigue); Higham 1988 (nearest PSD); Shinozuka & Deodatis (multivariate spectral representation) | ✅ |
| Lanczos/subspace for large models; AMLS / substructuring; a continuous Wigner–Ville / Loève INSTANTANEOUS-spectrum formulation (scalar M26 OR joint-tensor M27 — beyond the windowed spectrogram); a full NON-GAUSSIAN JOINT-TENSOR evolutionary distribution (beyond the M27 Gaussian joint-tensor + the M24 equivalent-scalar kurtosis correction); a fully NON-STATIONARY / EVOLUTIONARY MULTI-INPUT cross-PSD (a time-varying coherence, beyond the M28 stationary multi-input) and frequency-dependent coherence beyond the M28 constant + exponential models; multi-directional 100-30-30 response spectra; non-proportional HARDENING as a material model; mean-stress beyond the per-plane normal / basic Goodman option; crack-growth / fracture-mechanics fatigue; the complex-FRF stress recovery; gyroscopic / circulatory (non-symmetric C/K) systems | ❌ (deferred — see the M18/M19/M20/M21/M22/M23/M24/M25/M26/M27/M28 roadmap notes) |

## 5. Roadmap (next milestones)

1. **M2 — element completeness** ✅ (done): `/SH3N` (C0 triangle), 4-node
   tetra, beams (`/BEAM` + `/PROP/TYPE3`, elastic), degenerate-brick
   handling, shell hourglass upgraded from viscous to BLT84 stiffness
   type, exact-dt coverage extended to the shell bending/shear branch and
   the beam 12×12 eigenproblem. Deferred to later milestones:
   fully-integrated solid (Isolid=17 equivalent), QEPH shell, DKT18
   triangle, plastic beams (global plasticity model), nodal-pressure
   tetra variants.
2. **M3 — materials** ✅ (done): LAW36 tabulated plasticity (/FUNCT
   hardening curves + strain-rate curve family), LAW27 brittle cracking
   (shells), LAW42 Ogden hyperelasticity (solids, with the law feeding
   its **nonlinear sound speed** into the element time step — the M2
   time-step lesson applied to a stiffening material, proven by a long
   /DT 0.9 hold at λ≈2), /FAIL/JOHNSON and /FAIL/BIQUAD with full
   element-deletion plumbing (per-layer shell failure, OFF in ANIM,
   deletion messages), eps_p_max deletion for LAW2/LAW36, and the
   deferred-from-M2 plastic beams (LAW2 global resultant plasticity).
   Deferred out of M3, explicitly:
   * **/EOS (equations of state)** — energy-dependent pressure
     integration (E-p coupling per element, relative-volume state) is a
     solver-loop change, not just a material: it moves to M6 together
     with the thermal Johnson–Cook terms it usually accompanies;
   * thermal terms of LAW2//FAIL/JOHNSON (no thermal solution yet);
   * the plastic block of LAW27, shell LAW42, LAW42 Prony viscosity;
   * /FAIL/BIQUAD M-flag presets and S-flag; LAW36 Fsmooth/Fscale;
   * fiber-integrated beams (/PROP/TYPE18) for true plastic-hinge
     spread.
3. **M4 — contact** ✅ (done): TYPE7 full options — Istf 0–5 stiffness
   variants from the element formulas of i7sti3 (shell 0.5·E·t, solid
   B·A²/V), Igap constant/variable gaps with Gap_min/Gap_max, self-impact
   (grnod = 0), and a voxel broad phase replacing M1's all-pairs box
   test; TYPE2 tied contact (kinematic constraint: constant-weight
   projection with a co-rotating offset frame, lumped mass/force
   transfer — momentum-exact and zero-work by construction, both
   asserted); TYPE11 edge-to-edge penalty on /LINE edge sets (exact
   segment-segment closest points). Contact ⇄ /FAIL deletion is fully
   plumbed through per-segment element provenance: crack faces stop
   carrying contact, orphaned nodes stop being tracked, tied pairs
   release (the notched-plate example now runs a self-impact interface
   through full ligament tearing at ~0% energy error). Two hard-won
   solver lessons are recorded in the code: the interface dt must see
   the SUM of the candidate spring stiffnesses per node (springs stack
   at corners/self-impact — i7's STIFN accumulation), and contact work
   must be booked at the leapfrog midstep velocity or the balance reads
   spurious energy creation. Deferred out of M4, explicitly:
   * Inacti initial-penetration treatments and the stiffening
     K·p/(gap−p) near-crossing guard of the original force law;
   * Igap 2/3 (mesh-size-scaled gaps), Tstart/Tstop, sensors,
     Ifric > 0 friction models, thermal contact;
   * TYPE2 rotational-DOF tying (Spotflag) and the moment
     redistribution of offset ties; penalty-formulation TYPE2;
   * TYPE19/24/25 style combined interfaces;
   * parallel-edge overlap force distribution for TYPE11 (resultant is
     right, distribution acts at the closest-point pair).
4. **M5 — constraints & loads** ✅ (done): /RBODY and /RBE2 rigid bodies
   (starter-assembled mass/COG/inertia tensor; a 6-DOF Newton-Euler
   engine update that integrates the ANGULAR MOMENTUM — not the spin —
   and rotates the body frame with the exponential map, so a torque-free
   body conserves L by construction and the finite rotation is stable at
   any step; pivot mode when the master's /BCS clamps all translations —
   the physical-pendulum configuration, inertia transported by
   parallel-axis; /IMPVEL on the master = moving rigid die), /RBE3
   (least-squares rigid fit + its virtual-work dual force distribution,
   following the ContactType2 lumped pattern hook for hook), /SECT
   section resultants (via the side-sum identity — element
   self-equilibrium does the bookkeeping), /PLOAD follower pressure,
   /IMPDISP position-level imposed displacement, /ADMAS,
   /INIVEL/AXIS, and rigid walls completed: SPHER/CYL geometries and
   MOVING walls (free with a carrier-node mass — momentum-exact
   impulse exchange — or /IMPVEL-driven). Two M5 solver lessons are
   recorded in the code:
   * elements fully interior to a rigid body must see the rigid
     velocity field evaluated at their PLACED positions (the enforce()
     re-scatter) or their hypoelastic stress ratchets at O(w^2 dt) per
     cycle — the original deactivates such elements, the port keeps
     them alive at exactly zero strain so their faces stay available to
     contact;
   * a free moving wall must solve its recoil IMPLICITLY with the node
     corrections (one 3×3 system per cycle): the explicit-lag variant
     feeds riding nodes a one-cycle-stale wall velocity and injects
     energy at a rate that does not vanish with dt; the wall energy is
     then booked from the per-node injection identity
     U = dKE − f·v_old dt (rigid_wall.py).
   A pre-existing (M1-era) issue was *isolated* during M5 and left
   for M6 (where it was FIXED — see the M6 entry), documented here
   honestly: the solid **bulk-viscosity work
   under barely-resolved ringing** misbooks — a coarse block left
   ringing violently (strain rates ~1/ms on a single element through
   the thickness) drifts the energy balance over long free flights,
   with the half-step-lagged linear (qb) damper as the isolated
   culprit; it is invisible in normal meshes/loadings (all M1–M5
   validations and examples), and none of the M5 modules touch it —
   reproduce with a 2×2×2 cube given ±0.3 opposite face velocities.
   Deferred out of M5, explicitly:
   * **/MPC** — a general multi-point constraint row couples arbitrary
     DOFs and needs a small implicit solve per constraint (or mass
     redistribution à la Lagrange/penalty) that shares nothing with the
     lumped one-way patterns used here; it moves to M6;
   * per-DOF flags of /RBE2 and /RBE3, RBE3 per-set weights;
   * /RBODY sensors, IKREM, skew/spherical inertia input, the surface
     envelope, merged bodies (/RBODY of /RBODY);
   * wall rotation (a moving wall translates only), containment
     (nodes inside a sphere/cylinder), the Dist search band;
   * /SECT frame output and element-set input; distributed /PLOAD on
     solids' internal faces (only surface segments), /PLOAD Ipinch etc.
5. **M6 — engine niceties** ✅ (done): /DT/NODA + /DT/NODA/CST mass
   scaling (nodal dt from the kernels' own claims + contact NEAR
   springs; added mass/momentum/energy tracked, reported and kept in
   the balance), engine restart chaining (`_0002.rad` resumes
   `_0001.rst`; a chain reproduces the unchained run exactly — the
   acceptance test — with rigid-body R/L and sensor latch state carried
   and everything else deliberately reconstructed), /STATE/DT restart
   snapshots, /DAMP mass damping (exact integrating factor, exact
   KE-identity booking, DE ledger), /SENSOR/TIME + /SENSOR/DISP gating
   loads (time-shifted curves) and interfaces, /MPC (deferred from M5:
   the coupled nc×nc Lagrange solve on accelerations — zero work by
   construction, fixed DOFs as ground), /EOS polynomial + ideal gas with
   the implicit E-p coupling and EOS sound speed (deferred from M3),
   the LAW2 adiabatic thermal terms and /FAIL/JOHNSON D5, and the FIX
   of the M1-era bulk-viscosity misbooking documented in the M5 note.
   The M6 solver lessons, recorded in the code:
   * the qb misbooking was only half the story: booking the damper's
     work trapezoidally (the SDOF-exact midstep booking) still left the
     reproducer at −66%, because a LAGGED damper acting on
     barely-resolved content (ω·dt → 2) drains energy through the
     DISCRETE ELASTIC FORCE — real numerical dissipation that no state
     function can book. The consistent treatment measures the exact
     internal-force midstep work each cycle (one einsum — the same
     identity as the M4 contact lesson) and books the residual into the
     reported EN ledger: the reproducer closes to −0.00% at /DT 0.9 AND
     0.2, healthy runs keep EN ≲ few % (asserted);
   * a balance that CONTAINS its own residual ledger can no longer flag
     instability through |ERR| — divergence now drives EN hard negative
     instead, so the Engine stops on ERRN < −2×limit (energy injection)
     and on time-step collapse (dt < 1e-9 of its running peak — a dead
     run must never spin forever at dt ≈ 1e-18);
   * /DAMP's integrating factor rescales the very velocities the other
     ledgers book work with — the O(α·dt) attribution residual is
     measured on the damped nodes and moved into EN (the /MPC settle
     test read a spurious frozen +3.2% before that correction).
   Deferred out of M6, explicitly:
   * the β (stiffness) branch of /DAMP (needs K·v products);
   * /SENSOR types beyond TIME/DISP, sensor-driven /RBODY activation;
   * /STATE .sta ASCII output (the pickle restart is the state file);
   * Gruneisen/tabulated EOS, Psh/tension cutoffs, EOS on shells;
   * per-DOF /MPC skew frames; heat conduction (thermal stays adiabatic);
   * ALE/CFD (long term).
6. **M7 — performance** ✅ (done): profile first, optimize second, and
   keep the solver architecture (and every result) unchanged.
   * **The profile drove everything** (numbers in PR #7): at this
     port's typical model sizes (10²–10³ elements — the examples) the
     cycle cost is NumPy *per-call overhead*, not flops. On
     rigid_impactor, `np.cross` alone was ~20% of the runtime (its
     moveaxis/axis bookkeeping), stacked `np.linalg.det/inv` on the
     (n,3,3) Jacobians ~90 µs/call through LAPACK dispatch, `np.add.at`
     4× slower than a bincount, ~60 einsum dispatches per cycle; on
     notched_plate the TYPE7 narrow phase (~30 vectorized where/maximum
     passes over the candidate set) was the hottest single block.
   * **Pure-NumPy cheap wins**, benefiting every install:
     `common/fastmath.py` (cross3/norm3 — bitwise identical to the
     NumPy calls they replace; scatter_add3 — bincount assembly,
     bitwise identical into a zero target, reassociated at ulp level
     into an accumulated one; det_inv33 — explicit cofactor 3×3, ~5×
     faster than LAPACK here and machine-precision close); the shell
     kernel's ten per-rate einsums fused into ONE stacked matmul (and
     the five hourglass-mode einsums into one); all-faces-at-once
     characteristic lengths; LAW36's Newton loop exits on its exact
     fixed point (bitwise-identical results, ~2.5× fewer table walks);
     an integer `np.clip` (a hidden np.finfo per call in NumPy 2)
     removed from the LAW36 curve walk. **Preallocated scratch buffers
     were profiled and rejected**: allocations measured < 1% of the
     cycle — call-count reduction is where the time was.
   * **The optional numba backend** (`pyradioss/accel`, the package
     docstring is the specification): `forces()` of solid_hexa8 and
     shell_bt4 split into pre/post array blocks around the pure-Python
     material/failure loop, each block plus the TYPE7 narrow phase
     mirrored as an `@njit(cache=True)` kernel — exactly the three
     measured hotspots, nothing else. Selected explicitly
     (`PYRADIOSS_BACKEND=numba`, `pyradioss-engine -backend numba`, or
     `accel.select_backend`); numba stays an optional dependency
     (`pip install -e ".[accel]"`) and a missing/unknown backend falls
     back to NumPy with a warning. No `fastmath`, no `parallel`: the
     mirrors reproduce the reference math element for element, so the
     backends agree bitwise except for reassociated short reductions
     (documented, ≲1e-15/call). tests/test_m7_backends.py asserts the
     contract at three levels: single-call kernel parity (rtol 1e-12),
     a full starter+engine impact run compared state array by state
     array, and the M6 restart-chain bit-match rerun UNDER numba — the
     canary that would instantly catch nondeterministic reductions.
   * **Measured speedups** (tools/benchmark.py; engine wall clock, one
     clean run on the same machine, numba JIT cache warm — the compile
     itself is a one-time ~10 s paid on the very first run; energy error
     identical between backends to the reordered-reduction ulp drift):

     | example (engine s) | M7 NumPy | M7 numba | numba/NumPy |
     |---|---|---|---|
     | notched_plate  | 57.6 | 24.4 | 2.36× |
     | rigid_impactor | 56.7 | 25.8 | 2.20× |
     | box_beam_impact | 3.88 | 2.19 | 1.77× |
     | spot_weld      | 5.38 | 3.22 | 1.67× |
     | tensile_bar    | 1.11 | 0.68 | 1.63× |
     | edge_impact    | 33.7 | 22.9 | 1.47× |
     | rubber_block   | 1.07 | 0.75 | 1.42× |
     | gas_piston     | 0.40 | 0.36 | 1.10× |
     | antenna_mast   | 1.27 | 1.22 | 1.05× |

     The biggest wins are the compute-heavy solid/shell + contact runs
     (notched_plate 2.4×, rigid_impactor 2.2×); the small quick examples
     sit near 1× — fixed per-cycle Python overhead (the engine loop,
     kinematics, output) that neither backend touches dominates them, so
     the accelerated kernels are a small slice of their wall clock.
     Against the M6 (pre-M7) code the long examples are ~3× faster
     end-to-end (NumPy fast paths + numba stacked): rigid_impactor
     78.0 s → 25.8 s, notched_plate 67.7 s → 24.4 s.
   * Deferred out of M7, explicitly:
     - **the JAX backend** (stretch scope, not started — reasons on
       record): the engine cycle is built on in-place scatter into
       shared force arrays, per-cycle Python branching (sensors,
       deletion, EOS, /DT/NODA) and stateful dict-of-arrays element
       buffers; none of that maps to `jax.jit` without rewriting the
       engine in functional style — a fork, not a backend — while
       un-jitted `jax.numpy` would only add dispatch overhead at these
       model sizes. Revisit only after (if ever) a functional-core
       engine refactor;
     - numba mirrors beyond the measured hotspots: the TYPE11 narrow
       phase, the material laws (LAW36's table walk is the visible
       next candidate on notched_plate), tri3/tetra4/beam kernels;
     - threading: numba `parallel=True` breaks the determinism/parity
       contract (scatter order); MPI/domain decomposition stays out of
       scope for the port.
7. **M8 — implicit analysis (implicit STATICS)** ✅ (done): the port
   was fully EXPLICIT before this milestone (leap-frog, lumped mass, no
   global matrix). M8 adds a *parallel* implicit-static driver
   (`pyradioss/implicit/`, mapped to `engine/source/implicit/`) that
   REUSES the explicit element force kernels for the internal-force
   residual and adds ONE new piece per element — the tangent stiffness —
   assembled into a global sparse matrix and factorized by a direct
   solver. The explicit loop, all M1–M7 tests and all nine examples are
   untouched (re-verified). Delivered:
   * **DOF management** (`implicit/dofmap.py`): a global equation
     numbering (node*6 + component) that assigns each free nodal DOF an
     index, CONDENSES /BCS-fixed and /IMPDISP-prescribed DOFs (removed,
     not penalized), and numbers shell rotations only where they carry
     stiffness (shell nodes) — solids get translations only.
   * **Sparse tangent assembly** (`implicit/assembly.py`): each element
     returns its tangent as COO triplets in global-DOF space; they
     scatter into a `scipy.sparse` CSR K. SciPy is imported *inside* the
     implicit package (`require_scipy`), so the base explicit install
     stays NumPy-only and an implicit run without SciPy fails with one
     clear message (SciPy is required for implicit, optional otherwise).
   * **Consistent tangents**: LAW1 elastic (the material tangent is just
     C) for hexa8 and BT4, and — the piece that governs Newton's
     quadratic convergence — the LAW2 radial-return CONSISTENT
     (algorithmic) elastoplastic tangent (de Souza Neto Box 7.3, derived
     in `materials/law02_johnson_cook.consistent_solid_tangent`; NOT the
     continuum tangent, which would only converge linearly — asserted by
     the observed quadratic tail). The element tangents live alongside
     `forces()` and never touch the force path or the M7 numba parity
     contract (asserted). The one-point solid needs a genuine STIFFNESS
     hourglass for statics (the explicit VISCOUS hourglass is ~1e-4 of
     the physical stiffness and cannot control hourglass under load): a
     Flanagan–Belytschko stiffness-hourglass term is added to BOTH the
     tangent and the residual (`solid_hexa8.static_stabilization`),
     consistently, so Newton keeps its quadratic rate; it is orthogonal
     to the constant-strain modes, so uniform-strain states (patch test,
     uniaxial) stay EXACT. The BT4 shell's BLT84 hourglass is already
     stiffness-type and needs no addition.
   * **Newton–Raphson** (`implicit/statics.py`): load stepping
     (increments), residual R = f_ext − f_int with f_int from the
     EXISTING kernels (driven at the committed reference geometry with
     the displacement increment as a pseudo-velocity at dt=1, snapshot/
     restore around each residual evaluation so the rate kernels behave
     as pure functions of u), K Δu = R, residual + displacement
     convergence norms, an iteration cap with a clear non-convergence
     stop. Load control AND /IMPDISP displacement control.
   * **Direct linear solver** (`implicit/linsolve.py`): a `solve(K,R)`
     interface behind the EXACT M7 backend pattern — `splu` (SuperLU)
     default (no extra dependency), optional CHOLMOD (scikit-sparse, SPD)
     and MUMPS (python-mumps, *wrapped* — not ported), selected by
     `PYRADIOSS_LINSOLVE` / `-linsolve`, each falling back to SuperLU
     with a warning when its library is absent. Optional solvers stay
     optional dependencies; the base install keeps working.
   * **Engine input**: a minimal `/IMPL` control card (final /RUN "time"
     reinterpreted as the load factor; `/IMPL/DTINI`, `/IMPL/NEWTON`,
     `/IMPL/LSOLVER`); unknown sub-cards warn and skip; `/IMPL/DYNA`
     warns (implicit dynamics not ported) and runs static.
   * **Validation** (`tests/test_m8_implicit.py`, the port's philosophy —
     an analytic check per capability): single-element uniaxial pull =
     Hooke exactly with ONE-step (quadratic) convergence; multi-element
     constant-stress PATCH test exact; shell cantilever tip deflection
     within <1% of Euler–Bernoulli; LAW2 uniaxial matching the
     closed-form Johnson–Cook curve AND the explicit solver driven
     quasi-statically, with the plastic increments showing QUADRATIC
     convergence; DOF condensation, the reaction (equilibrium) balance
     and the strain-energy balance; the linear-solver fallback and the
     SciPy guard.
   Deferred out of M8, EXPLICITLY (not half-implemented):
   * **geometric / initial-stress stiffness** (large-displacement K_geo):
     M8 lands small-strain LINEAR geometry. The residual keeps the full
     corotational kernels (so moderate rotations enter f_int), but the
     tangent omits the stress-dependent geometric term and the reference
     frame stays at x0 for the whole run — so a genuinely large-rotation
     or buckling problem is out of scope until K_geo lands;
   * **implicit DYNAMICS** (Newmark / HHT / generalized-α) — the mass
     matrix, the a/v update and the effective dynamic stiffness are a
     separate build on top of this statics core;
   * **contact, rigid bodies and general constraints in the tangent
     system** (/INTER, /RBODY, /RBE2/3, /MPC): these are kinematic /
     penalty in the explicit port and contribute nothing to K here — an
     implicit run must not use them (the assembler errors on un-ported
     element groups; constraint contributions to K are a follow-on);
   * **arc-length / snap-through continuation** — plain load control
     only, so a limit point (softening past the peak) stops Newton
     rather than turning the corner;
   * tetra4 / sh3n / beam / truss / spring element tangents (only hexa8
     and BT4 in M8) and the LAW2 SHELL consistent tangent (LAW1 shells
     only); LAW36/27/42 implicit tangents.
8. **M9 — implicit NONLINEAR GEOMETRY** ✅ (done): geometric stiffness
   first, mapped to the /IMPL/NONLIN branch of `imp_solv.F` (the
   updated-Lagrangian step), the `imp_kgeo` geometric-stiffness assembly
   inside `imp_glob_k.F`, and `imp_buck.F` (buckling). M8's implicit
   statics was small-strain LINEAR geometry: tangent without K_geo,
   reference frame frozen at x0. M9 delivers, all opt-in behind
   `/IMPL/NONLIN` so the M8 path stays byte-identical (asserted by the
   unchanged M8 suite):
   * **K_geo** (initial-stress stiffness) for hexa8, BT4 and (new) the
     truss: K_geo = ∫G^T[σ]G dV — δ_ij ∇N_a·σ·∇N_b for the one-point
     solid, the membrane-resultant von-Kármán form for the shell, the
     exact (F/L)(I − aa^T) for the corotational truss. K_geo → 0 at zero
     stress, so the small-strain limit IS the M8 path.
   * **Updated-Lagrangian frame**: the committed geometry advances to the
     deformed configuration each increment; within an increment the
     stress integrates at the MIDPOINT geometry (Hughes–Winget: the
     midpoint gradient of a finite rigid-rotation increment is the exact
     Cayley skew, so rigid increments produce identically zero strain —
     the end-point evaluation would leak 1−cosθ spurious strain per
     increment, measured fatal for the elastica) and the force is
     re-assembled on the END configuration (`static_internal_forces` per
     element), where equilibrium is stated. Tangent = material +
     hourglass + K_geo at the trial geometry.
   * **Arc-length continuation** (`/IMPL/ARCL`): spherical Riks/Crisfield
     constraint ‖Δu‖² + w·Δλ² = dl² with w = ‖K₀⁻¹q‖² (the pure
     cylindrical form let λ jump arbitrarily on the stiff post-snap
     branch — measured, and fixed by the spherical metric), Crisfield
     root selection by path continuation, sign-following predictor,
     radius adaptation + halving on failure, and a trailing
     load-controlled step landing exactly on the final load factor.
     Proportional force loading required (asserted); /IMPDISP refused.
   * **Linearized buckling** (`implicit/buckling.py`): dense generalized
     eigenproblem (K_mat + μ K_geo)φ = 0 on a pre-stressed state.
   * **Statics bulk-viscosity fix**: the implicit driver now zeroes the
     solid qa/qb rate coefficients — the pseudo-velocity residual gave
     compressive increments a spurious viscous pressure (latent in M8,
     whose validations were all tensile). Compression is now exactly
     Hooke (asserted).
   * **Validation** (tests/test_m9_geomnl.py, one analytic check per
     capability): Euler buckling of a clamped-free BT4 strip column vs
     π²EI/(4L²) (<3%, ν=0 so plate=beam); Euler's RELATION on a hexa8
     column — P_cr = π²(EI)_eff/(4L²) with the mesh's own bending
     stiffness measured by a static tip-load bend (<1% — see the honest
     note below); the large-deflection cantilever at PL²/EI = 2 vs the
     Bisshopp–Drucker elastica (tip position within 2% of L, far from
     the linear w = αL/3); the von Mises two-bar truss traced by arc
     length through BOTH limit points against the closed form
     P(y) = −2EA·ln(L/L0)·y/L (exact for the corotational log-strain
     truss: sampled peaks reach 99% of ±P_max and never exceed them,
     the far-branch landing matches to 0.1%, and the load factor goes
     NEGATIVE mid-trace — the signature no load control can produce);
     finite-difference residual/tangent consistency (exact for the truss
     including prestress, exact for the hexa at zero stress); the
     small-strain /IMPL/NONLIN limit reproducing Hooke with a quadratic
     Newton tail.
   * An HONEST discretization note recorded in the hexa buckling test:
     pure bending excites the one-point element's hourglass pattern in
     every element, so the M8 FB *stiffness* stabilization (HG_STIFF,
     sized for robust static hourglass control) over-stiffens
     coarse-section solid bending (measured 3.4× on a 3×3-element
     section). That is a K_mat property of the 1-pt element, not a K_geo
     error — the buckling eigenvalue obeys Euler's relation with the
     mesh's own EI to <1%, which is exactly what K_geo controls. Shell
     bending is physical, hence the strip column validates against the
     continuum Euler load directly.
   Deferred out of M9, explicitly:
   * **implicit DYNAMICS** (Newmark / HHT / generalized-α) — the natural
     M10: mass matrix, a/v updates, effective dynamic stiffness on top of
     this statics core (DONE in M10 — see the next entry);
   * contact, rigid bodies and /MPC in the implicit tangent system
     (unchanged from M8);
   * **follower-load (pressure) stiffness**: /PLOAD is evaluated at the
     committed frame each increment; its configuration dependence is not
     linearized into K (quadratic convergence degrades gracefully if
     used);
   * the /IMPL/BUCKL engine card (the eigensolver is a library function),
     sparse (shift-invert) buckling eigensolves for large models;
   * LAW2 (elastoplastic) truss and shell consistent tangents; tetra4 /
     sh3n / beam / spring element tangents; LAW36/27/42 implicit
     tangents;
   * line search for the Newton corrector (the arc radius adaptation
     covered every validation case).
9. **M10 — implicit DYNAMICS** ✅ (done): Newmark-β time integration with
   HHT-α numerical dissipation, mapped to the /IMPL/DYNA branch of the
   implicit driver — `engine/source/implicit/imp_dyna.F` (DYNA_INI scheme
   setup, DYNA_INA initial acceleration, IMP_DYNAM effective-stiffness
   diagonal, IMP_DYNAR + IMP_FHHT dynamic residual with the HHT weighting,
   INTE_DYNA acceleration/velocity recovery, DYNA_WEX work ledger) and the
   /IMPL/DYNA read of `engine/source/input/freimpl.F`. Implicit was
   STATICS ONLY through M9 (/IMPL/DYNA warned and ran static). Delivered
   (`pyradioss/implicit/dynamics.py`; statics stays the default for a
   bare /IMPL):
   * **The lumped mass in equation space**: diagonal M from the starter's
     `model.mass` (translations) and `model.inertia` (shell rotations),
     condensed through the existing DofMap — exactly the arrays the
     explicit leapfrog divides by, and exactly the original's lumped
     MS/IN use in IMP_DYNAM. NO consistent-mass option (the original has
     none here either); a zero-inertia rotational equation keeps M = 0
     and stays well-posed through K_eff.
   * **The Newmark/HHT stepper on the Newton machinery**: each time step
     is a statics increment plus inertia. Newmark makes (a, v) pure
     KINEMATIC functions of the step displacement increment
     (a = Δu/(βdt²) − v/(βdt) − (1/2β − 1)a_n); the HHT residual
     R = (1+α)(f_ext + f_int)_{n+1} − α(f_ext + f_int)_n − M a_{n+1}
     REUSES `statics._internal_forces` for f_int (BOTH geometry modes:
     the M8 frozen-frame path and the M9 /IMPL/NONLIN updated-Lagrangian
     midpoint/end evaluation, whose committed frame advances per step);
     the effective tangent K_eff = (1+α) K_T + M/(βdt²) reuses
     `assembly.assemble` (+K_geo under NONLIN). The previous level's
     converged force is stored, never recomputed. /RUN "time" and
     /IMPL/DTINI are PHYSICAL again; /IMPDISP drives at real time;
     /IMPVEL is REFUSED with a pointer to /IMPDISP (silently ignoring a
     real dynamic BC would be worse). /IMPL/ARCL + /IMPL/DYNA is refused
     as the contradiction it is.
   * **The /IMPL/DYNA card, mirrored from the SOURCE** (an M10 check the
     task asked for: the original does NOT take a spectral-radius input):
     `/IMPL/DYNA/1` reads the HHT **alpha itself** (freimpl.F HHT_A;
     γ = 1/2 − α, β = (1−α)²/4 derived exactly as DYNA_INI), and
     `/IMPL/DYNA/2` reads **gamma, beta in that order** (NM_A → DY_G,
     NM_B → DY_B). Defaults = the trapezoidal rule (γ = 1/2, β = 1/4,
     α = 0), unconditionally stable and non-dissipative. Out-of-range
     values warn (α outside [−1/3, 0]; 2β ≥ γ ≥ 1/2 violated).
   * **Rate handling DECIDED and documented** (the pseudo-velocity trick
     changes meaning under dynamics): the kernels stay driven with the
     step increment at dt = 1 — that is what the element tangents
     (including the solid hourglass consistency) linearize — so every
     strain-rate device is disabled EXPLICITLY rather than silently fed
     du/1: bulk viscosity qa/qb zeroed (the statics rationale, plus an
     implicit dynamic step is far above the shock-resolving scale, and
     the original's implicit branch runs without it), the LAW2 rate term
     (c > 0) zeroed with a WARNING — rate-dependent plasticity under
     implicit dynamics is a deliberate deferral.
   * **Energy ledger** (DYNA_WEX analogue): per-step KE (½vMv + ½wIw),
     IE (+hourglass) from the element bookings, trapezoidal external
     work (+ the /IMPDISP constraint-reaction work), and the balance —
     recorded in `model.implicit_result.history` with displacement
     snapshots (the validation instrument), printed at termination.
   * **Validation** (tests/test_m10_impdyn.py — an analytic check per
     capability): the equation-space mass matrix (values + condensation);
     an SDOF truss free vibration whose measured period elongation
     MATCHES Newmark's closed-form (ω dt)²/12 dispersion to 5% AND drops
     4× when dt halves (the O(dt²) signature), amplitude exact to 1e-3,
     energy balance < 1e-10 (the trapezoidal rule's exact conservation on
     linear systems); UNCONDITIONAL stability at 20× the explicit
     critical step over 100 steps — with the leapfrog's divergence at the
     same dt asserted by direct recursion on the same discrete system;
     HHT α = −0.3 draining the unresolvable high modes of an 8-element
     bar (jump excitation) below 0.92·E0 while never exceeding E0 and
     while the trapezoidal run conserves to 1e-9 — with the honest note
     that the strictly monotone HHT quantity is the ALGORITHMIC energy
     (the physical KE+IE wiggles at the 0.1% level while decaying);
     the quasi-static limit reproducing the M8 implicit-static cantilever
     to 0.2%; a step-force transient matching (F/k)(1 − cos ωt) pointwise
     AND the EXPLICIT solver on the same shell-cantilever deck at the
     response peak (0.5% — sampled where v = 0 so end-time granularity
     cannot leak in); the /IMPL/NONLIN large-rotation pendulum crossing
     the vertical at the elliptic-integral quarter period to 0.5% (7.3%
     away from linear theory at 60°), bar length preserved to 1e-4
     through the 120° swing, closed energy, quadratic Newton tails; card
     parsing vs the freimpl.F semantics; the rate-term and /IMPVEL
     deferrals firing. Example: `examples/implicit_pendulum`.
   Deferred out of M10, explicitly (not half-implemented):
   * **consistent (element) mass matrix** — the original's implicit is
     lumped here too; nothing to mirror until a consistent-mass source
     path exists to port;
   * **Rayleigh damping in the implicit system** (/IMPL/DYNA/DAMP,
     IDY_DAMP/DAMPA_IMP/DAMPB_IMP): needs the damping force AND its
     tangent blended into residual/K_eff — a self-contained follow-on;
     the card warns and skips;
   * **rate-dependent plasticity under implicit dynamics** (the LAW2 c
     term): feeding the true velocity to the kernels would break the
     tangent/hourglass consistency the statics drive guarantees; doing it
     right means threading the real dt through the kernel drive and
     linearizing the rate term — deferred, the term is zeroed loudly;
   * **automatic implicit time-step control** (imp_dt.F): the port stops
     on non-convergence with a clear message instead of cutting dt;
   * **modal / eigenvalue dynamics** (frequency extraction beyond the M9
     buckling eigensolver), **implicit↔explicit switching mid-run**
     (/IMPL/SWITCH family), **/IMPVEL under dynamics** (use /IMPDISP),
     the **QSTAT_*** quasi-static-initialization branch of imp_dyna.F;
   * contact / rigid bodies / /MPC in the dynamic tangent, consistent
     with their M8/M9 statics deferral — an implicit dynamic run must not
     use them either.
10. **M11 — implicit COMPLETENESS** ✅ (done): the implicit solver
    accepted only hexa8/BT4/truss + LAW1 (and LAW2 solids), fixed steps,
    no damping, and the buckling eigensolver had no engine card. M11
    closes those gaps (each mapped to the checked Fortran —
    imp_dyna.F's IDY_DAMP blocks, imp_dt.F, imp_buck.F + the freimpl.F
    reads — not to docs from memory):
    * **Element tangent completeness**: `tangent()` (+ `kgeo()` +
      `static_internal_forces()`) for tetra4 (full integration — no
      hourglass block to stabilize), sh3n (the BT4 construction with
      triangle operators, no hourglass), beam (the 12×12 L·BᵀCB the
      exact-dt eigenproblem already built, frame-rotated; K_geo = the
      truss-form axial operator, which IS consistent for the linear
      one-point element), spring TYPE4. The `_TANGENT_KERNELS` gate now
      admits every family (and stays as the guard for future ones); the
      DofMap numbers beam rotations and excludes zero-mass orientation
      nodes. TWO elements needed their OWN implicit residual
      (`implicit_internal_forces`, dispatched by
      `statics._internal_forces` instead of `forces()`):
      - the SPRING is total-form (forces() rebuilds F = k(L−L0) from
        geometry, so the frozen-frame pseudo-velocity drive would feed
        the elastic term NOTHING and the dashpot the increment) —
        incremental on the committed force state under linear geometry,
        exact total form at the end configuration under NLGEOM. An M11
        lesson recorded in the code: the first cut evaluated
        k(L(x_ref)−L0) + k·a·Δu with x_ref frozen at x0 — correct for
        ONE increment, silently losing all accumulated displacement on
        the next (caught by the multi-increment regression test and a
        −67% dynamics ledger);
      - the TRUSS's explicit LAW2 return is a SINGLE linearized step
        (H frozen at the committed εp) — exact at explicit step sizes,
        but at implicit increments it barely flows (measured: εp 0.006
        instead of 0.04 at σ = 0.5) because the virgin JC slope
        diverges; the implicit path runs the ITERATED consistency solve
        (forces() untouched — the M7 parity contract).
    * **LAW2 consistent tangents for shells and the truss** (the M8/M9
      deferrals): the plane-stress Iplas=2 radial projection's
      algorithmic tangent (derived in
      `law02.consistent_shell_tangent`; mildly NONSYMMETRIC — the price
      of the projection; reconstructed entirely from the converged
      state like the solid one), integrated per layer with the force
      path's own quadrature (A/B/D thickness moments — the coupling
      block activates when the stack yields asymmetrically); LAW1
      shells keep the closed-form elastic path bit for bit. The
      validations pin the ALGORITHM's own closed form: the radial
      projection makes the axial plastic flow (3G/E)·εp, not εp — a
      documented property of the ported Iplas=2 variant, asserted
      exactly, with quadratic Newton tails.
    * **/IMPL/DYNA/DAMP** (imp_dyna.F IDY_DAMP): C = a·M + b·K with the
      STEP-START tangent (the IMP_DYKS save; reassembled per committed
      frame under NLGEOM), damping force at the current velocity
      iterate folded into the HHT weighting exactly like f_int
      (IMP_DYNAR), K_eff += (1+α)γ/(βdt)·C — algebraically the
      original's BDT/S0 form times (1+α) — and the DY_EDAMP trapezoidal
      dissipation ledger (booked on the energy side, as the source
      comments explain, in its own `edamp` channel). The card alone
      implies dynamics (`IF (IDYNA==0) IDYNA=1`).
    * **Automatic step control** (imp_dt.F IMP_DTN): cut-and-retry on
      non-convergence (the source's TT/NCYCLE rollback + SCAL_DTN cut,
      stop at DT_MIN), IDTC = 1 growth toward DT_MAX after easy steps —
      one `StepControl` object driving statics increments AND dynamics
      steps. Always on (the source cuts regardless of /IMPL/DT);
      /IMPL/DT/STOP and /IMPL/DT/1 tune it. The original's defaults
      live in an init routine outside the reader, so the port's
      defaults are its own, documented: target 6, grow ×1.1, cut ×0.5,
      dt_max = the /IMPL/DTINI step, dt_min = 1e-4 of it.
    * **/IMPL/BUCKL/1|2**: prestress increments, then the M9
      eigensolver, factors/modes in the listing (the imp_buck.F
      "BUCKLING MODES COMPUTATION" block) and on the result object;
      bare /IMPL/BUCKL errors as OBSOLETE exactly like the reader.
    * **Validation** (tests/test_m11_implcomp.py — an analytic check per
      capability, 26 tests): FD residual/tangent consistency for all
      four new elements (exact at zero stress; beam/spring exact with
      prestress; tetra/sh3n prestressed at the documented σ/E scale of
      the omitted spin terms); the tetra patch test and sh3n membrane
      patch EXACT in one Newton step; the thick sh3n cantilever vs
      Timoshenko (2% — the THIN strip shear-locks: −8% even at 32×4,
      the honest C0 note); the beam cantilever vs FL³/3EI + FL/GA
      (0.5%); spring u = F/k exact incl. multi-increment; a
      seven-family mixed model in one Newton step; plane-stress and
      truss JC closed forms with quadratic tails; the damped-SDOF
      trilogy vs exp(−ζωt) envelopes, damped periods and the
      dissipation fraction 1 − exp(−2ζωt), balance at round-off; the
      elastica cut-and-complete + smooth-run-reproduction pair; the
      Euler column through the BUCKL card. Example:
      `examples/implicit_ringdown` (mixed beam+sh3n+truss+spring mast,
      Rayleigh ring-down onto its own static answer at 0.00% balance).
    Deferred out of M11, explicitly (not half-implemented):
    * **contact, /RBODY, /RBE2/3 and /MPC in the implicit tangent
      system** — still THE structural gap and the natural M12: penalty
      contact stiffness and constraint condensation in K, so implicit
      models can carry joints and interfaces;
    * consistent (element) mass; modal / eigenvalue dynamics;
      implicit↔explicit switching (/IMPL/SWITCH); QSTAT_*;
    * LAW27/36/42 implicit tangents (LAW36's tabulated consistent
      tangent is the cheapest next candidate); the LAW2 BEAM tangent
      (linearizing the global resultant-space return is its own
      derivation — plastic beams refuse rather than run elastic);
    * rate devices under implicit stay OFF loudly: LAW2 strain-rate
      term, bulk viscosity, and now the spring DASHPOT (its force needs
      the true velocity; under statics it is meaningless);
    * the IDTC = 2/3 step controls (displacement-norm / Riks coupling)
      and /IMPL/DT/FIXP fix points; line search;
    * beam K_geo shear/moment frame-coupling terms (the axial term is
      the consistent one for the linear element; the omitted couplings
      are O(Q/N, M/NL) at a buckling state);
    * exact plane-stress LAW2 return (Iplas=1) — the port mirrors the
      Iplas=2 radial projection, and its tangent differentiates THAT
      algorithm (validated against its own closed form, documented
      3G/E flow factor and all).
11. **M12 — implicit CONSTRAINTS & CONTACT** ✅ (done): the M11 deferral
    list's structural gap — an implicit run silently ignored /RBODY,
    /RBE2, /RBE3, /MPC and every /INTER (they contributed nothing to K
    or the residual). M12 closes it, each piece mapped to the CHECKED
    Fortran (fetched, not recalled): the rby_imp0.F / rbe2_imp0.F /
    rbe3_imp0.F / i2_imp1.F condensations, imp_int_k.F + i7ke3.F /
    i7keg3.F for the contact stiffness, imp_solv.F's ICONTA bookkeeping.
    * **Kinematic constraints by CONDENSATION**
      (`implicit/constraints.py`): the original transforms every
      dependent DOF block in place (`UPDKB_RB`: K' = CDIᵀ·K·CDI with
      CDI = [[I, R],[0, I]] built from the CURRENT arm, residual
      B_M += CDIᵀ·B_s, slave equations marked IKC and skipped); the port
      expresses the whole family as ONE sparse transform u = T·u_red
      around each Newton solve — algebraically the same condensation,
      dependent DOFs eliminated, never penalized. /RBODY + /RBE2 (the
      rigid map u_s = u_M + θ_M × r_s), /INTER/TYPE2 (the Starter
      projection weights — the constraint layer literally reuses the
      explicit ContactType2 search, so implicit and explicit ties are
      identical; the rotational tie / offset-moment branch of I2UPDK0
      stays unported like the explicit side), /RBE3 (the least-squares
      fit linearized — Tᵀ IS rbe3f's virtual-work dual), /MPC
      (column-pivoted QR elimination; fixed DOFs read as ground,
      redundant rows dropped with a warning). The DofMap gained the
      constraint hooks: force-numbered master blocks (a standalone
      /RBODY master is a frozen placeholder that must be UNFROZEN),
      dependent slots always numbered so their element stiffness/loads
      are captured before elimination, /BCS on a master = the body-level
      condition (the PIVOT when all translations are fixed), /BCS on
      dependents warned away (the constraint wins — the explicit
      convention). Under /IMPL/NONLIN, T is rebuilt on every committed
      frame and the rigid bodies are RE-PLACED exactly (Rodrigues of the
      increment rotation; tied nodes on their co-rotated segment) — the
      linearized map would stretch a body O(θ²) per increment. Under
      /IMPL/DYNA nothing extra is assembled: Tᵀ M T IS the exact
      rigid-body 6-DOF mass at the master (parallel-axis inertia and
      COG-coupling blocks included), the initial velocities project onto
      the constraint manifold mass-weighted (= the explicit momentum
      projection), and the initial acceleration solves the condensed
      M_red a_red = R_red. TWO M12 lessons are recorded in the code:
      - the dynamics PREDICTOR must be projected onto the constraint
        manifold (`constraints.make_consistent`): the node-space
        constant-acceleration extrapolation violates u = T u_red by
        O(θ²)·arm per step, Newton cannot remove what Tᵀ annihilates,
        and the commit placement silently converts the violation into
        energy — found by a physical pendulum that gained 20× its drop
        energy and swung over the top;
      - convergence is measured on the REDUCED residual (the only one
        that must vanish — a dependent row's out-of-balance is by
        construction carried by its masters).
    * **Penalty contact in the Newton loop** (`implicit/contact.py`,
      /INTER/TYPE7): the contact force of every active pair at the TRIAL
      configuration (model.x + u — contact is geometric in both element
      modes) joins the residual, and the tangent gets the exact gap
      linearization K·g gᵀ (g = [n, −H₁n … −H₄n]) PLUS the closest-point
      curvature −K·p·∇²d, region-wise exact: zero for face-interior
      projections, the point-tie lateral term (I−nnᵀ)/d at vertices, the
      line-tie term at real boundary edges (i7keg3.F assembles only the
      n nᵀ blocks with the corner weights un-squared — a diagonal-boosted
      modified Newton; IMP_INT_K forces IMP_INT7 = 3, the CONSTANT-spring
      branch, which is exactly the port's explicit force law, so
      residual/tangent/explicit-cross-check are mutually consistent).
      The curvature term is an M12 lesson: a punch whose corner nodes
      land on the pad's grid leaves Newton in a LIMIT CYCLE without it
      (the missing lateral stiffness was 23% of K at p/d ≈ 0.23). The
      ACTIVE SET is simply re-evaluated at every residual/tangent call
      (the penalty force is continuous at p = 0); an increment whose set
      refuses to settle fails its Newton budget into the M11 StepControl
      cut — the imp_dt.F coupling. Istf/Igap stiffness and gap machinery
      reused verbatim from contact/stiffness.py; the i7 normal VISCOUS
      damper is a rate device and does not exist here; under /IMPL/DYNA
      the contact force is HHT-weighted like f_int and the stored spring
      energy ½Kp² gets its own `econt` ledger channel. A DOCUMENTED
      non-smoothness: a converged state sitting exactly on a projection-
      region boundary (node laterally on a main-mesh grid line, or on
      the crease of a warped quad's triangle split — the medial axis) can
      leave Newton cycling at ~1e-5 relative residual; deliberate mesh
      alignment provokes it (the press example documents the geometry),
      breaking the alignment or loosening /IMPL/NEWTON's tolerance clears
      it — node-to-segment contact in the original has the same
      non-smooth set behind a looser default tolerance.
    * **Refusals, never silence** (an implicit run used to IGNORE all of
      this): /RWALL under implicit refuses (a real BC of the explicit
      update — use TYPE7 against a meshed surface), /INTER/TYPE11
      refuses, /IMPL/ARCL and /IMPL/BUCKL refuse when combined with
      constraints or contact, /IMPDISP on constraint nodes refuses,
      constraint CHAINS (a dependent DOF of one constraint appearing in
      another) refuse, TYPE7 friction warns and runs frictionless.
    * **Validation** (tests/test_m12_implconstr.py — an analytic check
      per capability, 17 tests): the RBE2 rigid-lever closed form exact
      in ONE Newton step (both geometry modes, frozen master unfrozen);
      the condensed master 6×6 mass vs the parallel-axis block (1e-12);
      /MPC equality split and /RBE3 dual lever rule exact; the /RBODY
      pivot static rotation exact and the implicit-dynamic physical
      pendulum crossing the vertical at the elliptic-integral quarter
      period (< 0.5%, amplitude preserved < 1°, arm lengths exact,
      ledger closed); the spot-weld lap joint solved implicitly matching
      the explicit damped quasi-static answer (< 2% — the tie searches
      are shared code, so the comparison isolates the condensation); two
      blocks pressed = the series-springs closed form EXACT with the
      active set entering mid-run and uniform patch stress; FD
      residual/tangent consistency at an active contact state (exact for
      secondary-side directions, ≤ 5% full — the documented
      weight-variation omission); the implicit punch vs the explicit
      damped steady state (< 2%) AND its dead-load closed form (1e-6);
      the parity contract (builds mutate nothing shared); every refusal
      fires. Example: `examples/implicit_press` (RBE2 ram + TYPE7 pad,
      10 increments × 3 iterations).
    Deferred out of M12, explicitly (not half-implemented):
    * **friction in the implicit loop** (the FRIC blocks of i7keg3.F):
      the tangential force needs a slip/stick decision and its own
      consistent tangent — frictionless first, warned loudly;
    * **/INTER/TYPE11 edge-to-edge under implicit**: its segment-segment
      narrow phase shares nothing with the node-segment machinery here
      — refused, not approximated;
    * Inacti initial-penetration treatments, Igap 2/3, sensor gating of
      interfaces under the implicit clock, the IMP_INT7 = 0/1 stiffening
      (gap-scaled) tangent branches of i7keg3.F;
    * /RWALL under implicit (refused — model a wall as TYPE7 against a
      fixed meshed surface);
    * constraint CHAINS (rigid-on-rigid, /MPC rows on rigid slaves,
      /RBE3 masters inside bodies…): the original resolves some
      orderings; the port refuses them loudly. **Removed in M14.**;
    * /IMPDISP on constraint nodes (drive a free master with forces
      instead); /IMPVEL stays refused as before;
    * /IMPL/ARCL and /IMPL/BUCKL combined with constraints or contact
      (the arc-length metric and the buckling eigenproblem would need
      the reduction threaded through — refused for now). **Removed in
      M14.**;
    * the TYPE2 rotational tie / offset-moment redistribution (the
      UPDKB_RB arm branch of I2UPDK0 for 6-DOF mains) — same deferral as
      the explicit port, so the two solvers stay comparable;
    * follower-load (/PLOAD) stiffness: still evaluated at the committed
      frame, its configuration dependence not linearized into K (slows
      Newton on pressure-dominated NLGEOM runs, never changes the
      answer). **Removed in M13.**
12. **M13 — implicit FRICTION, TYPE11, FOLLOWER LOADS, LAW36 tangent** ✅
    (done): the head of the M12 deferral list — friction and the
    remaining contact/load pieces of the implicit system, each mapped to
    the CHECKED Fortran (fetched, not recalled): the FRIC blocks of
    i7keg3.F (I7KEG3/I7FRF3/I7KFOR3), i11ke3.F/i11keg3.F, imp_glob_k.F's
    IMP_KPRES/KPQUAD/KPTRIA, sigeps36.F.
    * **/INTER/TYPE7 COULOMB FRICTION in the Newton loop**
      (`implicit/contact.py`): the port implements the incremental
      tangential RETURN MAPPING of I7KFOR3's incremental branch (the
      stored force CAND_F plus STIF0·D radially returned to the cone by
      `BETA = MIN(1, µ√(FN/FT))` — also the textbook static stick/slip
      algorithm), with K_t = K (the original's own choice) and the
      CONSISTENT tangent per regime: stick K_t(I − nnᵀ), slip the
      nonsymmetric µK·t nᵀ + (µf_n K_t/|f_tr|)(I − nnᵀ − ttᵀ) — where
      the original's I7KEG3 assembles a µ-scaled always-stick spring (a
      modified Newton). Anchored tangential forces are COMMITTED once
      per converged increment (projected onto the current tangential
      plane — the FTN removal), every Newton residual is a pure function
      of the trial displacement from the committed anchor, and the slip
      work µf_n·dγ lands in its own `efric` dynamics ledger channel. The
      explicit port's velocity-regularized KINETIC friction is a rate
      device and is NOT fed the pseudo-velocity — the two agree where
      they must (steady sliding: both µN, cross-validated). µ = 0 stays
      bit-identical to M12.
    * **/INTER/TYPE11 edge-to-edge under implicit** (`ImplicitContact11`):
      penalty-in-residual + gap-tangent with the segment-segment
      closest-point kinematics (the explicit i11dst3 port reused
      read-only), and the M12 curvature lesson WORKED OUT for the
      edge-edge map instead of bounded: the exact Hessian of the
      closest-point distance from the 2×2 optimality-system
      linearization — exact in EVERY projection region
      (interior-interior, clamped point-segment INCLUDING the foot
      variation the TYPE7 edge term omits, point-point), FD-exact in all
      directions (asserted). Friction for TYPE11 warns and runs
      frictionless.
    * **/PLOAD follower-load stiffness under /IMPL/NONLIN**
      (`implicit/followerload.py`): the NLGEOM residual now evaluates
      /PLOAD at the TRIAL configuration and K gains the exact
      nonsymmetric load stiffness −∂(p·w·½d13×d24)/∂x. The original DOES
      have a pressure stiffness (IMP_KPRES — verified in imp_glob_k.F):
      a Gauss-integrated, skew-symmetrized variant of ITS
      shape-function-consistent force, off-diagonal blocks only, scaled
      HALF; the port deliberately linearizes ITS OWN lumped area-vector
      force EXACTLY instead (Newton cares about consistency with the
      residual actually iterated — documented deviation). The
      small-displacement path keeps the dead x0 pressure (byte-identical
      M8). /PLOAD + /IMPL/ARCL is REFUSED (a follower pressure violates
      the proportional-loading assumption).
    * **LAW36 consistent tangents** (solids + shells): the LAW2
      algorithmic-tangent algebra with the hardening slope H from the
      TABLE's local segment at the end-of-increment plastic strain. NO
      iterated-return upgrade is needed (the M11 truss lesson does not
      transfer): between knots the consistency condition is LINEAR, so
      the existing fixed-point return lands EXACTLY on the curve at any
      increment size — measured, asserted to 1e-9. Multi-rate curve
      families run on the STATIC curve under implicit (truncated with a
      warning — the pseudo-velocity drive must never feed the rate
      interpolation); softening tables use the true (negative) H with a
      floored denominator (documented).
    * **Two SOLVER lessons**, recorded in the code:
      - the STATICS Newton loop (`statics._solve_increment`) gained the
        imp_solv.F-style backtracking LINE SEARCH (the ILINE branch,
        IMCONV = −1): the
        full Newton step is accepted whenever it does not grow the
        residual norm — every monotone (smooth) run is bit-identical to
        plain Newton — and a growing step (the signature of the
        non-smooth assignment cycles a contact active set or a
        stick/slip boundary falls into; a mixed stick-slip state with
        pairs parked exactly on the cone cycled with period 2) is
        backtracked by halving, keeping the best trial. The DYNAMIC
        Newton keeps plain full steps: its M/(β dt²) diagonal already
        regularizes the assignment cycling the static loop is exposed
        to;
      - `solid_hexa8.static_stabilization` gained a PERSISTENT hourglass
        modal state (`hgq`, committed/restored with the stress): the
        incremental form forgot the accumulated hourglass deformation at
        every commit — each increment's converged hourglass content
        became a permanent out-of-balance jump and the modes RATCHETED
        (latent since M8: every earlier validation loads solids
        symmetrically enough to keep the term invisible; a moment-loaded
        block's unequal corner forces exposed it). Inert in the explicit
        engine.
    * **TYPE11 near-parallel overlap treatment** (an M13 lesson of the
      same family as the M12 curvature term): the closest-point pair of
      parallel overlapping edges is non-unique — the clamped solver
      picks an END of the overlap and the pick flips with the tilt sign
      as the structure deforms, a genuinely DISCONTINUOUS residual that
      left Newton in an exact period-2 cycle. Near-parallel pairs with a
      genuine overlap are split into TWO sub-pairs at the overlap ends
      (half stiffness each, own penetrations/normals — the trapezoid
      quadrature of the line contact, exact resultant for a linear
      penetration profile); threshold sin²θ < 1e-4, crossing it
      redistributes the force between two nearby points
      (resultant-continuous; the moment jump is O(K·L·θ_c) — documented,
      not hidden).
    * **Validation** (tests/test_m13_implfric.py, 18 tests — an analytic
      check per capability): friction stick below the cone (transmitted
      shear exact, per-node micro-slip F/(4K_t) exact on a quasi-rigid
      base), slip capped at exactly µN (displacement-driven), the
      inclined-load stick/slip transition (below: exact transmission;
      above: NO static equilibrium — fails loudly), FD stick/slip
      consistency (secondary rows EXACT at a face-interior projection,
      corner rows at the documented weight-variation tolerance), µ > 0
      with pure normal load reproducing the M12 frictionless closed form
      to 1e-9, and the explicit cross-check in steady sliding (friction/
      normal ratios agree within the explicit law's regularization
      factor); TYPE11 crossed edges on grounded springs = the 1-D closed
      form to 1e-8, FD consistency exact in ALL directions + tangent
      symmetry, the quasi-static edge_impact geometry implicit = explicit
      damped steady state (< 2%); /PLOAD load-stiffness FD-exact on a
      warped segment, the soft warped-brick pressure run converging in
      strictly FEWER iterations with the load stiffness ON and the SAME
      answer to 1e-6, the strip's small-pressure limit on beam theory
      (< 3%); LAW36 solid + shell uniaxial pulls landing on the table to
      1e-9/1e-6 with quadratic Newton tails, the rate family truncated
      with a warning; the parity contract (M13 builds/commits mutate
      nothing shared); /PLOAD + ARCL refused. Example:
      `examples/implicit_clamp` (the press punch pushed sideways below
      the cone — the pad's integrated shear resultant equals the applied
      push exactly).
    Deferred out of M13, explicitly (not half-implemented):
    * **Ifric > 0 friction models** under implicit (MFROT 1/2/3 —
      viscous/Darmstadt/Renard pressure- and velocity-dependent µ — and
      the IFQ friction filtering);
    * **TYPE11 friction under implicit** (warned, runs frictionless —
      **removed in M14**); thermal contact; TYPE19/24/25 combined
      interfaces;
    * Inacti initial-penetration treatments, Igap 2/3, sensor gating of
      interfaces under the implicit clock, the IMP_INT7 = 0/1 stiffening
      tangent branches;
    * /RWALL under implicit (refused); constraint CHAINS (refused —
      **removed in M14**); /IMPDISP on constraint nodes (refused);
      /IMPL/ARCL and /IMPL/BUCKL with constraints or contact (refused —
      **removed in M14**); **/PLOAD with /IMPL/ARCL** (refused — the
      follower pressure breaks proportional loading);
    * LAW27 and LAW42 implicit tangents (LAW42's **removed in M14**),
      the LAW2 BEAM (global resultant plasticity) tangent; LAW36 rate
      families under implicit (static curve only, warned); rate devices
      under implicit stay disabled loudly;
    * the NLGEOM (updated-Lagrangian) hourglass memory: the `hgq` fix
      covers the small-displacement path; under /IMPL/NONLIN the
      committed hourglass deformation lives in the advanced frame itself
      (the classic UL one-point-element ratcheting, shared with the
      original);
    * the BT4 drilling-row residual floor under NLGEOM (neighboring
      elements' rotated frames leave a tiny irreducible rotational
      residual, ~1e-4 of the load scale on the strip that measured it):
      runs converge through the displacement-correction criterion;
      documented in the M13 validation rather than hidden.
13. **M14 — implicit GENERALITY: chains, TYPE11 friction, ARCL/BUCKL with
    constraints & contact, the LAW42 tangent** ✅ (done): the remaining
    STRUCTURAL refusals of the implicit system, each mapped to the
    CHECKED Fortran (fetched, not recalled): rbody_part_modif.F90 (the
    starter's PARENT_OF rigid-body hierarchy), i11keg3.F's FRIC blocks
    (I11KEG3/I11KFOR3), imp_solv.F's Riks machinery (IDTC = 3 of
    imp_dt.F + PRODUT_UHP0), imp_buck.F (UPD_GLOB_K on both matrices, no
    contact assembly, IMP_KPRES into KG), sigeps42.F (the scalar-ET
    implicit branch).
    * **Constraint CHAINS** (`constraints.py`): a master DOF of one
      constraint DEPENDENT in another resolves by recursive transform
      SUBSTITUTION — exactly the product T = T1·T2·… in topological
      order, never formed; the DFS back-edge = a CIRCULAR chain, refused
      loudly; one DOF claimed by two rows = a CONFLICT, refused (the
      original MERGES part-body overlaps — merging would silently change
      the model). /MPC rows pre-substitute their dependent columns before
      the pivoting. Under NLGEOM the whole substitution re-runs per
      committed frame (all factors linearized at ONE configuration) and
      the commit placement walks the chain topologically — parents place
      their slaves (including a child's master) exactly before the child
      places its own, arms measured on the pre-increment snapshot. The
      EXPLICIT engine refuses chains loudly (its per-body 6-DOF
      integrator has no nesting order; the original's engine never sees
      one). Starter check relaxed to the slave/master-overlap case only.
    * **TYPE11 COULOMB FRICTION** (`contact.py`): the M13 TYPE7 return
      mapping generalized to edge pairs — the slip increment is the
      relative motion of the two closest MATERIAL points at frozen
      parameters, projected onto the plane orthogonal to n (which
      CONTAINS both edge directions at a crossing: axial sliding of
      either edge is genuine rubbing); stick/slip consistent tangents on
      the [(1-s), s, -(1-t), -t] pattern; anchors committed per converged
      increment, keyed 2*(i*n_main + j) + k with k the overlap-end index
      (the documented near-parallel keying: the LOW-s sub-pair inherits
      the single-point anchor across the threshold, the HIGH end starts
      fresh). What the ORIGINAL does (fetched): I11KEG3 assembles the
      always-stick µ-scaled plane spring and I11KFOR3 applies an UNCAPPED
      tangential spring with no stored anchor — the port's return mapping
      is a deliberate deviation, the same one M13 made for TYPE7 and for
      the same reason. µ = 0 stays bit-identical to M13.
    * **/IMPL/ARCL with constraints & contact** (`statics.py`): both
      corrector auxiliary solves, the spherical metric, the Crisfield
      root selection and the predictor-sign rule live in the REDUCED
      space; T is rebuilt and the exact placement runs per committed arc
      increment. The ORIGINAL's Riks control measures its arc metric on
      the FULL recovered nodal field (PRODUT_UHP0-style sum over all
      nodes — verified) — the port's reduced metric is a documented
      deviation (the constraint quadratic must live where the solves
      live; the two differ by the fixed SPD reweighting TᵀT, both valid
      Crisfield parametrizations). Contact joins the corrector residual
      at the trial configuration and both solves' matrix; the active set
      re-evaluates per iteration (the residual is continuous), and the
      radius-halving cut remains the non-smooth backstop — the corrector
      keeps plain Newton steps (no M13 line search; measured unnecessary
      on the snap-catch, where the set changes inside increments without
      a single cut).
    * **/IMPL/BUCKL with constraints & contact** (`buckling.py`): the
      reduced pencil Tᵀ(K_mat)T + µ Tᵀ(K_geo)T — exactly the original
      (imp_buck.F runs UPD_GLOB_K on BOTH assemblies before EIGBUCKP and
      recovers modes with RECUKIN). Contact: the original assembles NONE
      (no IMP_INT_K call — a column on a stop reports the free-column
      factor); the port DEVIATES, documented: the CONVERGED prestressed
      active set's gap (+friction, symmetrized) tangent joins K_MAT — the
      µ-independent side, because a closed stop is a physical support
      whose stiffness does not scale with the load multiplier (the one
      configuration stiffness the original DOES include, IMP_KPRES, goes
      into its KG correctly: follower pressure does scale). Frozen-set /
      bilateral-linearization caveats documented, not hidden.
    * **LAW42 consistent tangent** (`materials/law42_ogden.py`): the
      exact spectral SPATIAL elasticity of the port's own Ogden stress —
      c_aabb = β_ab/J − 2σ_a δ_ab with β from ∂²W/∂lnλ², the
      (σ_a λ_b² − σ_b λ_a²)/(λ_a² − λ_b²) shear terms, and the
      equal-stretch L'HÔPITAL limit which in THIS (shifted) convention is
      (c_aaaa − c_aabb)/2 — the textbook "− σ_a" form applies to the
      UNSHIFTED entries and would subtract σ_a twice (caught by the
      coalescent FD identity; recorded in the module). The NLGEOM
      residual re-evaluates total-form stress at the END configuration
      (the midpoint value is the right objective INCREMENT for
      hypoelastic laws but simply the wrong configuration for a pure
      function of F); tangent() feeds the trial F through the new
      ``extra`` hook of materials.solid_tangent. The ORIGINAL's implicit
      branch only scales a linear D by the scalar ET of sigeps42.F — a
      secant modified Newton; the port's exact tangent is the documented
      deviation (the M13 IMP_KPRES pattern). LAW42 under implicit
      REQUIRES /IMPL/NONLIN — on the frozen frame F never sees the trial
      displacement; refused loudly, statics and dynamics.
    * **Validation** (tests/test_m14_implgen.py, 19 tests): the two-body
      chain lever closed form EXACT in one Newton step (both geometry
      modes), the /MPC-on-rigid-slave equality closed form, the CHAINED
      pendulum on the elliptic-integral quarter period with BOTH arm
      lengths exact (the topological placement), the circular/conflict/
      explicit-engine refusals; TYPE11 friction stick (u = F/(2k + K_t))
      and slip (f_t = µN, u = (F − µN)/2k) closed forms EXACT, µ = 0 and
      pure-normal-load bit-consistency, FD in both regimes at a committed
      anchor, the explicit kinetic law evaluated ON the implicit converged
      slip state landing on µ·v/(v+eps); the snap-catch (free-path peak
      sampled from below exactly as M9, arrested equilibrium on the stop
      to 1e-5) and the RBE2-loaded snap through BOTH limit points; the
      RBE2-capped Euler column = the plain factor (1e-9 relative) on
      π²EI/4L², the pinned column on a contact stop = π²EI/L² (< 3%);
      LAW42: the FD Truesdell identity exact in every regime INCLUDING
      coalescent stretches, full-path FD exact with the hourglass frozen
      (the live-hourglass residue measured IDENTICAL for LAW1 — a
      pre-existing NLGEOM omission, not a LAW42 term), uniaxial (λ≈1.70)
      and equibiaxial closed forms to 1e-8 with quadratic tails, the
      explicit /IMPDISP quasi-static cross-check (< 2%), the
      incompressible-limit 1/K scaling of J−1, the frozen-frame refusal;
      the parity contract. Example: `examples/snap_catch`.
    Deferred out of M14, explicitly (not half-implemented):
    * **Ifric > 0 friction models** under implicit (MFROT 1/2/3 and IFQ
      filtering) — for TYPE7 AND TYPE11 (**removed in M15** — both
      solvers); thermal contact; TYPE19/24/25 combined interfaces;
    * Inacti initial-penetration treatments, Igap 2/3, sensor gating of
      interfaces under the implicit clock, the IMP_INT7 = 0/1 stiffening
      tangent branches;
    * /RWALL under implicit (refused); /IMPDISP on constraint nodes
      (refused); /PLOAD with /IMPL/ARCL (refused — the follower pressure
      breaks proportional loading, and the arc metric would need the
      configuration-dependent pattern re-sampled);
    * LAW27 and the LAW2 BEAM (global resultant plasticity) tangents
      (**removed in M15**); LAW42 SHELLS (no explicit shell variant
      exists to be consistent with) and LAW42 Prony viscosity; rate
      devices under implicit stay disabled loudly;
    * buckling with the arc-length traced state uses the same
      linearized pencil (no extended-system / branch-switching
      continuation — the classical eigenvalue estimate only);
    * the IDTC 2/3 step controls of imp_dt.F (the original's own Riks
      flavor lives there — the port keeps its Crisfield corrector);
    * the UL hourglass memory under NLGEOM and the hourglass-operator
      geometry variation in the tangent (the O(k_hg·|u|) omission the
      M14 FD validation measures and documents — shared by every
      material since M9, invisible at the u = 0 states the M9 FD tests
      probe);
    * the BT4 drilling-row residual floor under NLGEOM (unchanged M13
      note).

14. **M15 — friction MODELS + the last material tangents** ✅ (done):
    the head of the M14 deferral list, each piece mapped to the CHECKED
    Fortran (fetched, not recalled): the MFROT/IFQ blocks of i7for3.F
    and the hm_read_inter_type07.F reader (Ifric → MFROT, Ifiltr → IFQ,
    Xfreq → XFILTR with the exact 1/2/3 mapping, C1–C5 for Ifric > 0 and
    C6 for Ifric > 1), i11mainf.F (which HARDCODES MFROT = 0 — the
    original TYPE11 never evaluates friction models; I11FOR3 receives no
    FRIC_COEFS), the I7KFOR3/I7KEG3 implicit friction blocks of
    i7keg3.F, sigeps27c.F (whose damage/crack machinery lives in
    M27ELAS/M27CRAK — the port's total-strain law is what the tangent
    must be consistent with), and pmat3.F (which turns out to be pke3.F's
    ELASTIC shear-stiffness setup: the original's implicit beam has no
    resultant-plasticity linearization to mirror).
    * **Friction MODELS, explicit** (`contact/friction.py` +
      `inter_type7/11.py`): MFROT 1 (generalized viscous polynomial),
      2 (Darmstadt), 3 (Renard piecewise in v), 4 (exponential decay),
      p = f_n over the CURRENT main-segment area, the EM30 floor; the
      IFQ 1/2/3 first-order filter on the tangential force (the CAND_F
      exponential moving average, per-pair anchors rebuilt from the
      active set — the IFPEN scope; the anchor store resets across a
      restart chain, documented). ONE documented deviation: the fetched
      source computes IFQ 3's per-cycle coefficient as
      `MAX(ONE, ALPHA0*DT12)` — identically 1, i.e. the filter the card
      asked for never engages; the port uses min(1, XFILTR·dt), the
      documented cutoff-frequency behaviour. TYPE11 is a documented
      PORT EXTENSION (see the i11mainf.F finding above) with the edge
      pressure DEFINED p = f_n/(L_main·gap_pair) — stated in
      contact/friction.py so a calibrated C1 can be converted; decks
      that must match OpenRadioss keep Ifric = 0 on TYPE11. Ifric = 0
      is bit-identical (x + (−a) ≡ x − a; monkeypatch-asserted), and
      the numba mirror is untouched — the models sit downstream of the
      mirrored narrow phase in the NumPy path both backends share (the
      M7 boundary, stated in the module docstrings, parity asserted at
      the force-call level).
    * **Friction MODELS, implicit** (`implicit/contact.py`): the
      Coulomb cone radius becomes µ(p)·f_n with µ at the STATIC LIMIT
      µ(p, v = 0) (`friction.mu_static` — the M10 rate-device
      convention; the ORIGINAL's I7KFOR3 feeds its µ(p, v) the
      increment fields DX/DY/DZ, a step-size-dependent pseudo-rate the
      port deliberately does not reproduce, warned per interface when
      the deck's coefficients actually carry velocity terms; MFROT 3
      lands on its static coefficient C1, MFROT 4 on the card Fric).
      IFQ under statics is a TIME device whose DC fixed point is the
      unfiltered force — ignored with a warning. The consistent tangent
      gains the µ′(p) COUPLING BLOCK: d(µ f_n) = [µ + f_n µ′(p)/A] df_n
      =: µ_t df_n at frozen area (the frozen-weight class), so the slip
      block's t nᵀ factor is µ_t while the in-plane rotation term keeps
      the current cone radius — derived, not bounded (the M12 lesson),
      and a documented deviation from I7KEG3, which assembles its
      always-stick spring with the CONSTANT card FRIC even when
      MFROT > 0 (`FACT(I)=FRIC` — no µ(p) in the original's matrix at
      all). mfrot = 0 keeps every M13/M14 expression verbatim
      (bit-identical, monkeypatch-asserted). TYPE11: the same cone and
      coupling with the port-extension pressure definition.
    * **LAW27 implicit shell tangent**
      (`law27_brittle.consistent_shell_tangent`, dispatched through the
      new `extra` hook of `materials.shell_layer_tangent` and both
      shell kernels' layer loops): the EXACT derivative of the port's
      own total-strain fixed-crack unilateral law, per branch —
      uncracked = elastic C; crack OPEN with FROZEN damage = the (1−d)
      secant rows; OPEN with GROWING damage = plus the softening
      −cps(en_i + ν en_j)·dd_i/den_i normal terms and the shear-row
      −G·g12·dd_i terms of the ruling direction (growth detected by the
      stored-equals-drive identity, the loading branch taken at the
      corner — the plasticity convention; mildly nonsymmetric, as every
      softening tangent); CLOSED (elastic normal stress ≤ 0) = full
      stiffness rows — the unilateral switch, with the M13 line search
      as the non-smooth backstop (checked by the load-reversal
      validation); BROKEN = zero. Assembled in the frozen crack frame,
      rotated by the Voigt strain/stress pair.
    * **LAW2 BEAM consistent tangent** (`beam_type3.py`): the
      algorithmic derivative of the global resultant return in
      resultant space, C_alg = s·C + [(H/(E+H) − s)/seq_tr]·R_tr (qᵀC)
      — reconstructed from the POST-return state via the homogeneity
      identities (seq is degree-1 homogeneous and its gradient q
      degree-0, so seq_tr = sy + E·dλ and R_tr = R·seq_tr/sy with dλ
      from the epsp_incr plumbing); q carries the extreme-fiber |·|
      sign pattern (non-smooth at resultant sign changes like the yield
      function itself). The INCREMENTAL treatment was decided by
      MEASUREMENT (the M11 truss lesson replayed): the explicit
      kernel's historical 5-iteration consistency Newton is exact at
      n = 0.5 but leaves an O(1) residual at n = 0.2 virgin yield (the
      e^(n−1) slope needs O(1/n) iterations to escape), so the implicit
      residual runs its own ITERATED solve
      (`implicit_internal_forces`, 60 iterations) while forces() keeps
      its bit-identical 5 (the M7 contract — LAW1 beams route through
      the new hook reproducing the old path bit for bit, asserted).
      Two DRIVER lessons this milestone exposed (statics.py): a
      NON-FINITE residual/unorm must fail the increment (the perfectly
      plastic H = 0 plateau overflowed u and the relative-displacement
      test compared against tol·inf — accepting a NaN state as
      "converged"), and an EXACTLY SINGULAR trial factorization must
      fail the increment for the StepControl to cut (a too-large
      increment spuriously yields enough H = 0 elements to form a
      transient mechanism; smaller increments keep the intermediate
      states regular).
    * **Validation** (tests/test_m15_fricmat.py, 28 tests): exact MFROT
      formula/branch/continuity/floor checks + FD of the static-limit
      µ′(p); the reader's XFILTR mapping and refusals (Ifiltr ≥ 10,
      Ifric = 5, out-of-range Xfreq); kernel-level transmitted-force
      closed forms for MFROT 1 (every factor recomputed independently)
      and the Renard µ(v) curve through all three branches; the exact
      discrete first-order IFQ step response (1 − (1−α)^k for IFQ 1 AND
      the IFQ 3 cutoff form); the TYPE11 pressure-definition closed
      form; force-call numba parity; implicit µ(p) slip and stick
      closed forms; FD tangent consistency in both regimes with the
      coupling block active; static-limit and IFQ-ignored warnings with
      the converged force provably free of velocity terms (and the
      IFQ run equal to the unfiltered one bitwise); the
      implicit-vs-explicit steady-sliding identity at matched pressure;
      LAW27 pre-crack = LAW1 EXACTLY, law-level FD in every branch,
      the crack-closure stiffness recovery against the LAW1 twin, the
      notched-strip implicit-vs-explicit crack pattern/angle/damage;
      the beam hinge at EXACTLY the closed-form resultant limit load
      (root element ON the yield surface to 1e-9), hardening quadratic
      tails ON the JC curve to 1e-9, the iterated-return measurement
      (n = 0.2), the LAW1 hook-route bit-identity, the quasi-static
      explicit cross-check (0.1% measured), the NLGEOM plastic-beam
      branch; the parity/no-shared-mutation contract; Ifric = 0 /
      mfrot = 0 monkeypatch bit-identity in both solvers. Example:
      `examples/brake_pad` (µ(p) clamp that holds ONLY because the
      press pressure lifts the cone — the constant-µ0 twin fails
      loudly, verified).
    Deferred out of M15, explicitly (not half-implemented):
    * the IFQ ≥ 10 / MODFR 2 incremental (stiffness) EXPLICIT
      tangential formulation (refused at the reader; its return-mapping
      mechanics is exactly what the implicit port already implements);
      /FRICTION per-part-pair friction sets (INTFRIC) and orthotropic
      friction (IORTHFRIC); friction µ(T) thermal dependence (IFRICTH);
    * MFROT velocity terms and IFQ under implicit DYNAMICS (rate
      devices stay off — the M10 convention, warned);
    * the IFQ anchor store across restart CHAINS (rebuilt empty — a
      chained IFQ run re-converges its filter within ~1/α cycles; the
      M6 bit-match contract holds for Ifiltr = 0 decks, documented in
      inter_type7.py);
    * the LAW27 plastic block (M27PLAS — unchanged M3 deferral: the
      port's law is elastic-brittle, and the tangent is consistent with
      THE PORT'S law), LAW27 solids (the original refuses them too);
    * the fiber-integrated TYPE18 beam (the global resultant model's
      W·σy hinge convention is documented — the elastic-core spread to
      1.5·W·σy needs fibers); beam K_geo beyond the axial term
      (unchanged M11 note);
    * thermal contact, TYPE19/24/25, Inacti, Igap 2/3, rate devices
      under implicit dynamics, LAW42 shells/Prony, IDTC 2/3, /RWALL
      under implicit, the UL hourglass memory and the hourglass-operator
      geometry variation in the NLGEOM tangent, the BT4 drilling floor
      (all unchanged from the M10–M14 lists).

15. **M16 — CONSISTENT element mass + MODAL / eigenvalue analysis** ✅
    (done): the largest standing item of the M10–M15 deferral tail —
    free-vibration natural frequencies and mode shapes. The port was
    LUMPED-mass everywhere (the explicit leapfrog and the M10 implicit
    dynamics divide by the diagonal `model.mass`/`model.inertia`, exactly
    the original's MS/IN in IMP_DYNAM); M16 adds a *parallel* CONSISTENT
    mass operator — the mass analogue of `tangent()` sitting alongside
    `forces()` — feeding a modal eigensolver, without touching the lumped
    path (which stays bit-identical, asserted).

    * **Consistent element mass** (`consistent_mass()` per family,
      alongside the lumped mass of each `init_group`): the ∫ρ Nᵀ N dV
      shape-function integral. SOLIDS — the analytic V/20·[[2,1,1,1]…]
      tetra mass (constant Jacobian, exact) and the 2×2×2-Gauss brick
      mass (a one-point evaluation would be rank-1 and singular); SHELLS —
      the bilinear-quad (parallelogram-exact A/36·[[4,2,1,2]…]) and CST
      (A/12·[[2,1,1]…]) area integrals, isotropic ρt on the translations
      and ρt³/12 on the rotations (the PHYSICAL bending rotary inertia
      WITHOUT the lumped path's "+A" time-step boost, applied to all three
      rotations incl. drilling so the reduced mass stays PD); the BEAM —
      the 12×12 Rayleigh–Timoshenko mass (Przemieniecki ch. 11): axial
      ρAL/6·[[2,1],[1,2]], torsion with the mass polar moment ρ(Iyy+Izz),
      and the Hermite-cubic bending mass ρAL/420·[156,22L,…] PLUS the
      rotary-inertia mass ρI/30L·[36,3L,…] with the translation↔rotation
      coupling, in both principal planes (the x–z plane sign-flipped for
      θy = −w′), rotated to global axes by the SAME corotational frame as
      its stiffness; the TRUSS bar mass ρAL/6·[[2I,I],[I,2I]] and the
      spring's exact point mass (= its lumped mass — a discrete property,
      no interior to integrate). Fortran origin: the *mass3.F lumped
      family (smass3.F/s4mass3.F/cmass3.F/c3mass3.F/pmass3.F/tmass3.F —
      the open source ships only the lumped form); the consistent operator
      is derived per family from the shape integral and ported library-
      first, exactly as M9 ported the buckling eigensolver before M11's
      thin card.
    * **Mass assembly + constraint condensation** (`assembly.assemble_mass`
      → COO/CSR through the DofMap; `constraints.reduce_matrix` for TᵀMT):
      the global consistent M, condensed under every M12/M14 constraint
      (/RBODY, /RBE2, tied, /RBE3, /MPC and chains) — the same congruence
      transform as TᵀKT, so a rigid body carries its EXACT parallel-axis
      6-DOF mass at the master, now with the CONSISTENT element mass
      feeding it (validated: a uniform rigid bar reproduces the continuum
      mL²/3 end / mL²/12 centre inertia, exact because a rigid rotation is
      linear in the shape space).
    * **Modal eigenvalue extraction** (`implicit/modal.py`, `/IMPL/EIGV`):
      the generalized eigenproblem (K − ω²M)φ = 0 for the lowest N natural
      frequencies (Hz) + MASS-NORMALIZED mode shapes + the (N,6) modal
      effective mass, in the REDUCED (constrained) space — reusing
      `buckling.py`'s reduced-pencil + dense `scipy.linalg.eigh` (fine at
      this port's sizes; the original's Lanczos/subspace deferred, a
      documented deviation exactly as buckling did). A robust rigid-body
      filter (median-scaled, immune to the penalty-DOF spectrum inflation
      a max-relative threshold suffers) skips the ≤6 mechanism modes. On a
      PRESTRESSED state too (`/IMPL/EIGV/STRS`: K = K_mat + K_geo — the
      stress-stiffened spectrum; a tension-tuned string, a compressive
      column softening toward its buckling load where the modal and
      buckling eigenproblems meet). `/IMPL/EIGV` is a PORT card
      (freimpl.F reads only BUCKL and DYNA); it reports factors/modes on
      `model.implicit_result` (mirroring `buckling_factors`/`modes`) and
      in the listing.
    * **Validated** (`tests/test_m16_modal.py`): per-family mass partition
      of unity (row-sums to the lumped nodal mass), rigid-body KE
      ½vᵀMv = ½m|v|² EXACT, symmetry + PSD, no state mutation
      (monkeypatch-asserted — the M7 parity contract), the lumped path
      bit-identical with the consistent mass present; longitudinal bar
      vs (2n−1)c/4L (fixed-free) and nc/2L (free-free, the rigid mode
      filtered); cantilever bending vs the Euler–Bernoulli βₙL roots
      (1.875, 4.694 → < 0.3% at 20 beams — the antenna_mast example deck);
      a simply-supported plate fundamental; mode M-orthogonality
      φᵢᵀMφⱼ = δᵢⱼ to round-off; the consistent-over/lumped-under BRACKET
      around the exact frequency; the rigid-link spectrum shift
      sqrt(k/(m₁+m₂)); the taut-string prestressed modes n/2L·sqrt(T/μ)
      (transverse modes that exist ONLY through the geometric stiffness).
      Example: `examples/modal_mast` (/IMPL/EIGV natural frequencies of
      the cantilever mast).

    A deliberate, documented finding: the BT4 shell SHEAR-LOCKS in thin
    (a/t ≫ 20) 2D plate bending — a pre-existing element property, NOT an
    M16 mass error (consistent and lumped masses give the same locked
    frequency; the plate validation therefore uses a genuine a/t = 20
    plate and shows the error shrinking on refinement). Recorded so a
    future milestone (an assumed-strain / MITC4 shear treatment) can lift
    it.

    Deferred out of M16, explicitly (not half-implemented):
    * modal SUPERPOSITION transient and frequency response (/FREQ) — the
      response-history / steady-state build on these eigenpairs;
    * complex / damped (quadratic) eigenvalues — the real symmetric
      eigenproblem is ported; a damped structure's complex modes need the
      state-space or QEP formulation;
    * the Lanczos / subspace-iteration sparse eigensolver of the original
      (EIGBUCKP family) for large models — the dense `eigh` is the
      documented library choice at this port's sizes; a shift-invert
      `scipy.sparse.linalg.eigsh` is the upgrade path;
    * AMLS / component-mode substructuring; random & spectral (PSD)
      response;
    * the BT4 thin-plate shear-lock (an assumed-strain shell — a future
      element milestone, above);
    * the M10–M15 deferral tail unchanged: IFQ ≥ 10 / MODFR 2, /FRICTION
      per-part-pair sets, orthotropic / thermal friction, the fiber
      TYPE18 beam, the LAW27 plastic block / solids, thermal contact,
      TYPE19/24/25, Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3,
      /RWALL under implicit, the UL hourglass memory, the NLGEOM
      hourglass-operator geometry variation, the BT4 drilling floor.

16. **M17 — MODAL-SUPERPOSITION dynamics (transient + frequency response
    + modal damping)** ✅ (done): the FIRST item deferred out of M16 — the
    response-history and steady-state analyses that BUILD on the M16 natural
    frequencies + mode shapes. Modal superposition is the ALTERNATIVE to the
    M10 DIRECT Newmark/HHT time integration: expand u(t) = Σ φᵢ qᵢ(t) in the
    M16 mass-normalized modes, and the M- and K-orthogonality of the modes
    DECOUPLES the equation of motion into independent damped SDOFs, one per
    mode, q̈ᵢ + 2ζᵢωᵢq̇ᵢ + ωᵢ²qᵢ = φᵢᵀf(t). A NEW, parallel path
    (`implicit/modal_response.py`) that CONSUMES the M16 eigenpairs and never
    touches the M10 integrator or the M16 eigensolver (both stay
    bit-identical, monkeypatch-asserted). Theory: Clough & Penzien /
    Chopra / Craig & Kurdila (mode superposition), Nigam & Jennings 1969
    (the exact recurrence), the damped-SDOF complex FRF.

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (read
    line by line for M17) has no /FREQ, no /IMPL/MODAL, no mode-superposition
    or harmonic-response branch anywhere in the open tree; OpenRadioss is a
    time-domain crash/impact code. So M17 ports these as clean LIBRARY
    capabilities with minimal PORT cards, exactly as M16 ported the
    eigensolver behind /IMPL/EIGV and M9/M11 ported buckling behind
    /IMPL/BUCKL.

    * **Modal damping** (`modal_damping` + `rayleigh_ratios` /
      `table_ratios`): per-mode ζᵢ from (a) a uniform ζ, (b) a table
      (frequency, ζ) linearly interpolated onto the extracted ωᵢ, or (c) the
      Rayleigh map ζᵢ = ½(α/ωᵢ + β ωᵢ). The Rayleigh map uses the SAME
      (α, β) the M11 direct-integration /IMPL/DYNA/DAMP feeds C = αM + βK, so
      the modal and direct solvers describe IDENTICAL physical damping —
      validated against the direct-integration damped-decay envelope (the
      measured decay rate = ζωₙ). Over-critical ζ ≥ 1 refused loudly.
    * **Modal transient** (`modal_transient`, `/IMPL/MODAL/DYNA`): project
      the deck loads onto the modes (rᵢ(t) = φᵢᵀf(t) — the same
      /CLOAD//GRAV//PLOAD machinery evaluated at PHYSICAL time), march each
      decoupled SDOF with the EXACT Nigam–Jennings piecewise-linear-forcing
      recurrence (Chopra Table 5.2.1 — eight closed-form coefficients per
      mode; exact for a sampled load at ANY step, so NO period elongation,
      unlike the M10 Newmark's (ωdt)²/12 dispersion — which is why the
      step/impulse response matches the closed form POINTWISE), recombine
      u(t) = Σ φᵢ qᵢ(t). The M10 energy ledger is reproduced in modal
      coordinates (KE = ½Σq̇ᵢ², IE = ½Σωᵢ²qᵢ², edamp = Σ2ζᵢωᵢ∫q̇ᵢ²dt,
      balance closed to round-off). An optional MODE-ACCELERATION /
      residual-flexibility correction (`mode_acceleration=True`, /MACC on the
      card) adds the truncated tail's quasi-static response
      [K⁻¹ − Σᵢ₌₁..N φᵢφᵢᵀ/ωᵢ²]f(t) — one extra linear solve per step,
      recovering the EXACT static answer from a SINGLE mode (a documented
      ADDITION, off by default). Validated: a step-loaded spring-mass SDOF
      matching (F/k)(1−cos ωt) pointwise; the modal-vs-DIRECT-Newmark
      cross-check bit-close at the response peak (the spring's consistent
      mass EQUALS its lumped mass, so both solvers share K and M — the mass
      analogue of the M10 explicit-vs-implicit check); modal-truncation
      convergence to DAF = 2 as N grows; the mode-acceleration correction
      recovering the exact static tip; a uniform-ζ decay on the exp(−ζωt)
      envelope.
    * **Harmonic / frequency response** (`modal_frequency_response`,
      `/IMPL/FREQ`): for f(t) = F e^{iΩt} the complex modal FRF is
      qᵢ(Ω) = (φᵢᵀF)/(ωᵢ² − Ω² + 2iζᵢωᵢΩ), recombined to the complex
      transfer function u(Ω) = Σ φᵢ qᵢ(Ω), swept over a band and reported as
      amplitude and phase. Validated: the SDOF FRF peak = 1/(2ζ) at
      resonance with the half-power bandwidth Δω/ω = 2ζ; the FRF resonances
      COINCIDING with the M16 natural frequencies (the /IMPL/FREQ card on
      the antenna_mast cantilever); a driven cantilever's tip amplitude vs
      the closed-form modal FRF; base excitation (a shaker table) feeding
      the M16 effective-mass participation rᵢ = −φᵢᵀM r_d = −Γᵢ into the FRF.
    * **Engine cards + reporting** (`statics._run_modal_transient` /
      `_run_freqresponse`, mirroring `_run_modal`): both cards run AFTER the
      (usually zero-load, or prestress under /STRS) static solve, extract the
      modes on the committed state (rest at x0 for a pure transient) and
      drive the transient / sweep, storing `modal_damping`,
      `modal_transient_history` and `freq_response` on
      `model.implicit_result` and printing the "MODAL TRANSIENT" /
      "FREQUENCY RESPONSE" listing blocks. /IMPL/MODAL/DYNA card:
      `t_end dt [nmode]` (+ /MACC, /STRS); /IMPL/MODAL/DAMP: `zeta`
      (uniform); /IMPL/FREQ: `fmin fmax nf [zeta] [nmode]` — PORT cards,
      minimal like /IMPL/EIGV.
    * **Example**: `examples/modal_frf` (a shaker-driven cantilever mast,
      /IMPL/FREQ sweeping the first two bending resonances — the tip spikes
      to 1/(2ζ) = 50× its static deflection at each ωᵢ).
    * **Validated** (`tests/test_m17_modalresp.py`): all of the above plus
      the card mirror and the parity contract (modal superposition never
      mutating the element state or the eigensolver output, the direct
      /IMPL/DYNA answer byte-identical whether or not the modal path runs).

    Deferred out of M17, explicitly (not half-implemented):
    * complex / damped (state-space / quadratic-eigenvalue) eigenvalues and
      NON-CLASSICAL damping — the port superposes the M16 REAL modes and
      assumes classical (diagonal modal) damping; a structure whose C is not
      αM + βK has coupled modal equations the SDOF decoupling cannot
      capture. This is the M16-deferred complex-modes item, now reached and
      re-deferred: it needs the state-space / QEP formulation;
    * random & spectral (PSD) response and RESPONSE SPECTRA — the
      frequency-domain statistics / envelope analyses that build on this
      same real-modes FRF machinery;
    * base excitation beyond the simple participation-factor feed (multiple
      support motions, large rigid-base rotations);
    * the Lanczos / subspace-iteration sparse eigensolver for large models
      (unchanged from M16 — the dense `eigh` is the documented library
      choice at this port's sizes; `scipy.sparse.linalg.eigsh` is the
      upgrade path); AMLS / component-mode substructuring;
    * the BT4 thin-plate shear-lock (an assumed-strain / MITC4 shell — a
      future element milestone);
    * the M10–M16 deferral tail unchanged: IFQ ≥ 10 / MODFR 2, /FRICTION
      per-part-pair sets, orthotropic / thermal friction, the fiber TYPE18
      beam, the LAW27 plastic block / solids, thermal contact, TYPE19/24/25,
      Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under implicit,
      the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 drilling floor.

17. **M18 — COMPLEX / DAMPED eigenvalues + NON-CLASSICALLY-damped mode
    superposition** ✅ (done): the FIRST item deferred out of M17 (and the
    complex-modes item M16 deferred before it) — the state-space /
    quadratic-eigenvalue (QEP) analysis for structures whose damping is NOT
    classical, where the M16 real symmetric eigensolver and the M17
    real-mode superposition (classical damping only) both break down. When
    the damping matrix C is not proportional to M or K (a local dashpot,
    damping on part of the structure), the M16 undamped modes no longer
    diagonalize C: the free-vibration modes become COMPLEX (a DOF-to-DOF
    phase lag) and the SDOF decoupling M17 relies on no longer holds. M18
    generalizes the eigenproblem to the damped case and superposes the
    COMPLEX modes. A NEW, parallel path (`implicit/damping_matrix.py` +
    `implicit/complex_modal.py`) that never touches the M10 integrator, the
    M16 eigensolver or the M17 superposition (all stay bit-identical,
    asserted). Theory: Géradin & Rixen "Mechanical Vibrations" (ch. 3, 5),
    Meirovitch "Principles and Techniques of Vibrations" (ch. 9), Tisseur &
    Meerbergen "The Quadratic Eigenvalue Problem" (SIAM Review 2001), Clough
    & Penzien / Caughey & O'Kelly (classical vs non-classical damping).

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (read
    line by line for M18) has no /CEIGV, no complex/damped eigensolver, and
    no state-space / QEP branch anywhere in the open tree; the only damping
    the implicit path knows is the on-the-fly Rayleigh force C v = a M v +
    b K v of imp_dyna.F's IMP_DYKV (M11), never an assembled C. OpenRadioss
    is a time-domain crash/impact code. So M18 ports these as clean LIBRARY
    capabilities behind a minimal PORT card, exactly as M16 ported the real
    eigensolver behind /IMPL/EIGV and M17 the superposition behind
    /IMPL/MODAL, /IMPL/FREQ.

    * **Assembled damping matrix C** (`damping_matrix.assemble_damping`, the
      C analogue of M16's `assemble_mass`): C = Rayleigh αM + βK (the SAME
      α, β the M11 /IMPL/DYNA/DAMP feeds, with the CONSISTENT mass and the
      tangent K) PLUS the discrete viscous dashpots — the /PROP/SPRING `c`
      term revived as the element damping matrix c·aaᵀ
      (`spring.damping_matrix`, the velocity analogue of the elastic k·aaᵀ,
      after the dashpot was DEFERRED from the M11 implicit residual as a rate
      device) and the per-node /DAMP mass damper α_dp·m as its diagonal C
      contribution. Assembled COO→CSR through the DofMap and condensed TᵀCT
      under the M12/M14 constraints exactly like TᵀMT. Validated: the
      discrete dashpot contributes exactly c·aaᵀ on the free DOF; a
      pure-Rayleigh C reproduces the M11/M17 modal ratios ζᵢ =
      ½(α/ωᵢ + βωᵢ) in the M16 real modal basis (diag(ΦᵀCΦ) = 2ζᵢωᵢ, and the
      off-diagonal ΦᵀCΦ is zero — Rayleigh IS classical). Documented: the
      Rayleigh C uses the CONSISTENT mass (the modal pencil), not the lumped
      mass the M11 DIRECT integrator uses; the /DAMP time window has no
      meaning in a linearized analysis (constant α used, flagged).
    * **Complex / damped eigenvalues** (`complex_modal.build_complex_basis`,
      `/IMPL/CEIGV`): the QEP (λ²M + λC + K)φ = 0 solved through the
      SYMMETRIC state-space linearization A z = λB z with z = [φ; λφ],
      A = [[0,K],[K,C]], B = [[K,0],[0,−M]] — chosen over the M⁻¹ companion
      form [[0,I],[−M⁻¹K,−M⁻¹C]] because it needs NO mass inverse (the
      reduced consistent M is dense) and A, B inherit the symmetry of K, C, M
      so the left/right eigenvectors coincide and the biorthogonality is the
      clean symmetric-bilinear (transpose, not conjugate) relation
      z_iᵀBz_j = 0. `scipy.linalg.eig` (non-symmetric generalized) on the
      REDUCED pencil (dense, the documented M16 choice — Lanczos/subspace and
      a structure-preserving QEP solver deferred). Reports λᵢ = −ζᵢωᵢ ±
      iωᵢ√(1−ζᵢ²): the decay rate −Re(λ), the damped frequency Im(λ), the
      natural frequency |λ|, the ratio ζ = −Re(λ)/|λ|, and the COMPLEX mode
      shapes (phase lag). The rigid/spurious modes filtered exactly as M16
      filters its near-zero ω²; the FULL surviving reduced modal set retained
      for an EXACT (non-truncated) superposition, only the reporting limited
      to the lowest N under-damped modes. Validated: a classically-damped
      (Rayleigh) system reducing EXACTLY to −ζᵢωᵢ ± iωd,ᵢ with the M16 ωᵢ and
      the M17 ζᵢ, its mode shapes real-up-to-phase; a 2-DOF chain with a
      dashpot on ONE mass matching the closed-form roots of
      det(λ²M + λC + K) = 0 AND showing a DOF phase lag (neither in phase nor
      exactly out of phase); the state-space biorthogonality z_iᵀBz_j = 0.
    * **Complex-mode superposition** (`complex_modal_transient`,
      `complex_frf`): the state-space form B ẇ = A w + P (w = [u; u̇],
      P = [0; −f]) decouples in the complex modal basis into 2n INDEPENDENT
      FIRST-ORDER complex modal equations ẋₖ = λₖxₖ + pₖ(t), pₖ =
      (zₖᵀP)/(zₖᵀBzₖ) — each marched by the EXACT piecewise-linear-forcing
      recurrence (the first-order analogue of M17's Nigam–Jennings) and
      recombined u(t) = Σ φₖxₖ(t) (real to round-off, the modes/forces
      conjugate-closed). The damped complex FRF xₖ(Ω) = pₖᶠ/(iΩ − λₖ),
      U(Ω) = Σ φₖxₖ, exact for non-classical damping. Validated: the
      complex-mode transient of a non-classically-damped chain matching a
      DIRECT Newmark march of the assembled (K, C, M) BIT-CLOSE, where the
      M17 real-mode (classical-damping) superposition — even fed the best
      diagonal modal damping — is provably WRONG (the gap asserted, > 20×);
      reduction to the M17 answer when the damping IS classical; the
      complex-mode FRF matching the direct inversion (K − Ω²M + iΩC)⁻¹F on
      the full basis to round-off; the first-order recurrence exact (DC limit
      −p/λ + free decay e^{λt}).
    * **Engine card + reporting** (`statics._run_complex_modal`, mirroring
      `_run_modal` / `run_modal_transient`): /IMPL/CEIGV runs after the
      (usually zero-load, or /STRS prestress) static solve, assembles
      (K, C, M) — the Rayleigh α, β from /IMPL/DYNA/DAMP folded into C —
      extracts the complex modes and prints the "COMPLEX / DAMPED
      EIGENVALUES" block (frequency / damped frequency / ζ / decay rate per
      mode), storing `complex_eigenvalues`, `complex_natural_freqs`,
      `complex_damped_freqs`, `complex_decay_rates`, `complex_damping_ratios`
      and `complex_modes` on `model.implicit_result`. /IMPL/CEIGV/TRAN drives
      the complex-mode transient (card: t_end dt); /IMPL/CEIGV/FRF the damped
      complex FRF sweep (card: fmin fmax nf). PORT cards, minimal like
      /IMPL/EIGV.
    * **Example**: `examples/complex_modes` (a fixed-free spring-mass chain
      with ONE localized dashpot — /IMPL/CEIGV extracting the complex modes,
      whose ζ is NON-monotone in frequency (the localized dashpot damps modes
      unequally, the non-classical fingerprint), + /IMPL/CEIGV/FRF sweeping
      the decaying complex FRF).
    * **Validated** (`tests/test_m18_cmplxmodes.py`): all of the above plus
      the /IMPL/CEIGV (+ /STRS, /TRAN, /FRF) card mirror and the parity
      contract (the complex path never mutating the M16 eigensolver / M17
      superposition / the element state, the direct /IMPL/DYNA answer
      byte-identical whether or not it runs).

    Deferred out of M18, explicitly (not half-implemented):
    * the Lanczos / subspace-iteration sparse eigensolver for large models
      (unchanged from M16 — the dense `scipy.linalg.eig` is the documented
      library choice at this port's sizes; a structure-preserving QEP solver
      — SOAR / second-order Arnoldi, `polyeig` — is the upgrade path); AMLS /
      component-mode substructuring;
    * random & spectral (PSD) response and RESPONSE SPECTRA — the
      frequency-domain statistics / envelope analyses that build on this same
      FRF machinery (unchanged from M17);
    * GYROSCOPIC / circulatory systems — a non-symmetric damping C (from a
      rotating frame's Coriolis term) or a non-symmetric stiffness K (a
      follower / circulatory force): the symmetric state-space linearization
      above assumes symmetric K, C, M, so these are OUT of scope (the
      non-symmetric pencil and its distinct left/right eigenvectors are the
      generalization);
    * base excitation beyond the participation-factor feed (multiple support
      motions, large rigid-base rotations — unchanged from M17);
    * the M10–M17 deferral tail unchanged: IFQ ≥ 10 / MODFR 2, /FRICTION
      per-part-pair sets, orthotropic / thermal friction, the fiber TYPE18
      beam, the LAW27 plastic block / solids, thermal contact, TYPE19/24/25,
      Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under implicit,
      the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

18. **M19 — RANDOM / SPECTRAL (PSD) response + RESPONSE SPECTRA** ✅ (done):
    the FIRST frequency-domain-statistics item deferred out of M17 AND M18 —
    the stochastic (random-vibration) and envelope (response-spectrum)
    analyses that BUILD on the M17 real-mode FRF and the M18 complex FRF (both
    explicitly deferred these). A NEW, parallel path
    (`implicit/random_response.py` + `implicit/response_spectrum.py`) that
    never touches the M10 integrator, the M16 eigensolver, the M17
    superposition or the M18 complex path (all stay bit-identical, asserted).
    Theory: Newland, "An Introduction to Random Vibrations, Spectral & Wavelet
    Analysis" (ch. 5-7); Wirsching, Paez & Ortiz, "Random Vibrations: Theory
    and Practice"; Vanmarcke, "Random Fields" (spectral moments / crossing
    rates); the Wiener-Khinchin theorem; Chopra, "Dynamics of Structures"
    (ch. 13); Der Kiureghian 1981 (CQC).

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read
    line by line for M19) has no /PSD, no /RSPEC, no random-response or
    spectral-moment or SRSS/CQC branch anywhere in the open tree (the sole
    `PSD` token is `IMUMPSD`, a MUMPS-solver flag — not a power spectral
    density). OpenRadioss is a time-domain crash/impact code — the frequency-
    domain statistics simply are not part of the open-source solver (the same
    finding M16 made for the real eigensolver, M17 for superposition, M18 for
    the complex modes). So M19 ports these as clean LIBRARY capabilities behind
    minimal PORT cards, exactly as M16-M18 did.

    * **RANDOM / SPECTRAL (PSD) RESPONSE** (`random_response.py`, /IMPL/PSD):
      given an input force- or base-acceleration PSD S_ff(Ω) (a /FUNCT table),
      the stationary response PSD is S_uu(Ω) = H(Ω) S_ff(Ω) H(Ω)* through the
      modal transfer function H(Ω) — the |H|² S law (Newland eq. 6.31). The
      module is FRF-SOURCE-AGNOSTIC: it consumes the M17 real-mode FRF (for
      classical damping) or the M18 complex FRF (`/IMPL/PSD/CPLX`, for
      non-classical damping — the assembled C = Rayleigh + discrete dashpots),
      read-only. Reports the RMS σ_u = √m₀ (the zeroth spectral moment / the
      response variance) per DOF, the spectral moments mₙ = (1/π)∫₀^∞ Ωⁿ S_uu
      dΩ (m₀/m₁/m₂; the 1/π folds the Wiener-Khinchin 1/2π with the even-
      function factor 2 — EXACTLY the task's σ² = ∫S dΩ/2π convention), and the
      mean zero-crossing rate ν₀ = (1/2π)√(m₂/m₀) + peak rate ν_p =
      (1/2π)√(m₄/m₂) (Rice's formula) + the Vanmarcke bandwidth. The BASE-
      excitation (support-motion) PSD feed reuses the M16 participation
      (`/IMPL/PSD/BASE`, a shaker table). Validated: a white-noise-driven
      SDOF's σ² = S₀/(2ck) closed form (the Lorentzian ∫|H|²dΩ = π/(ck)); a
      multi-DOF response PSD from the modal FRF matching the DIRECT
      (K−Ω²M+iΩC)⁻¹ inversion; the spectral-moment / Parseval identity
      m₀(velocity PSD) = m₂(displacement PSD); the RMS reducing to the static
      σ_f/k for a quasi-static (low-frequency-band) input; the complex-FRF PSD
      reducing to the real-FRF PSD when the damping IS classical.
    * **RESPONSE SPECTRA** (`response_spectrum.py`, /IMPL/RSPEC): given a
      design response spectrum Sa(ω,ζ) (a /FUNCT of frequency), the per-mode
      peak rᵢ = Γᵢ Sa(ωᵢ,ζᵢ)/ωᵢ² φᵢ (the participation-scaled spectral
      ordinate, Γᵢ = φᵢᵀM ιₐ the SIGNED M16 participation — the effective mass
      Γ² loses the sign, recovered here) is combined by SRSS (well-separated
      modes) and CQC (the Der Kiureghian 1981 complete-quadratic-combination
      with the closed-form modal-correlation ρᵢⱼ = 8√(ζᵢζⱼ)(ζᵢ+rζⱼ)r^{3/2} /
      [(1−r²)²+4ζᵢζⱼr(1+r²)+4(ζᵢ²+ζⱼ²)r²], r = ωⱼ/ωᵢ — the general unequal-
      damping form, reducing to the familiar equal-ζ one). Both reported so the
      gap is visible. Validated: the CQC ρᵢⱼ closed form (unit diagonal, →0
      well-separated, →1 for r→1); SRSS ≈ CQC for well-separated modes (ρ→I);
      the CQC-vs-SRSS GAP on CLOSELY-SPACED modes (a near-degenerate tuning-
      fork pair both participating under base excitation — the CQC analogue of
      M18's classical-vs-complex gap, asserted > 5 %); a single-mode spectrum
      recovering Γ Sa/ω²; the flat-spectrum limit computed exactly.
    * **Engine cards + reporting** (`statics._run_random_response` /
      `_run_response_spectrum`, mirroring `_run_freqresponse` / M18's
      `_run_complex_modal`): /IMPL/PSD (+ /BASE, /CPLX, /STRS) extracts the
      modes (real or complex), builds the FRF, reads the input PSD /FUNCT,
      forms the response PSD / spectral moments / RMS and reports them on
      `model.implicit_result.random_response`; /IMPL/RSPEC (+ /STRS) reads the
      design spectrum /FUNCT, combines the SRSS/CQC peaks and reports them on
      `model.implicit_result.response_spectrum` (+ the participation table in
      the listing). PORT cards, minimal like /IMPL/EIGV.
    * **Example**: `examples/random_vibration` (a base-excited 5-mass
      instrument stack — /IMPL/PSD/BASE reporting the RMS relative
      displacement + spectral moments + zero-crossing/peak rate under a flat
      band-limited base-acceleration PSD, and /IMPL/RSPEC combining a plateau
      design spectrum by SRSS and CQC).
    * **Validated** (`tests/test_m19_random.py`): all of the above plus the
      /IMPL/PSD (+ /BASE, /CPLX) and /IMPL/RSPEC card mirror and the parity
      contract (the random/spectral path never mutating the M16 eigensolver /
      M17 / M18 transfer functions / the element state, the direct /IMPL/DYNA
      answer byte-identical whether or not it runs).

    Deferred out of M19, explicitly (not half-implemented):
    * NON-STATIONARY / evolutionary PSD (a time-varying spectrum — an
      earthquake's build-up/decay envelope, a run-up transient): the port does
      the STATIONARY response only (the |H|² S law assumes a stationary input
      and a settled response);
    * MULTI-INPUT cross-PSD with coherence (a full S_ff MATRIX with
      off-diagonal cross-spectra between multiple correlated inputs): the port
      does a SINGLE scalar input process (a force pattern OR one base
      direction), the |H|² S law; the matrix H S H* triple product and a
      coherence model are the generalization;
    * FATIGUE DAMAGE from the spectral moments (Dirlik / rainflow / narrow-band
      Miner) and the full PEAK-FACTOR / extreme-value distribution beyond the
      mean crossing/peak rate — they build on the moments this milestone
      reports;
    * MULTI-DIRECTIONAL response-spectrum combination (the 100-30-30 /
      SRSS-of-directions rules): a straightforward post-combination of
      per-direction /IMPL/RSPEC runs, deferred;
    * the complex-FRF BASE-excitation feed (the participation projected through
      the state-space biorthogonality) — real-mode base PSD and complex-mode
      force PSD are both supported, their combination is not;
    * GYROSCOPIC / circulatory (non-symmetric C/K) systems unchanged from M18;
      the Lanczos / subspace-iteration sparse eigensolver + AMLS / substructur-
      ing unchanged from M16;
    * the M10–M18 deferral tail unchanged: IFQ ≥ 10 / MODFR 2, /FRICTION
      per-part-pair sets, orthotropic / thermal friction, the fiber TYPE18
      beam, the LAW27 plastic block / solids, thermal contact, TYPE19/24/25,
      Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under implicit,
      the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

19. **M20 — RANDOM-VIBRATION (SPECTRAL) FATIGUE** ✅ (done):
    the FIRST item deferred out of M19 that BUILDS on it — the stress-life
    fatigue-damage estimate computed DIRECTLY from the M19 stress-PSD spectral
    moments (m₀..m₄). A NEW, parallel path (`implicit/spectral_fatigue.py` +
    the stress-recovery section of `implicit/random_response.py`) that never
    touches the M10 integrator, the M16 eigensolver, the M17/M18 superposition
    or the M19 PSD / response-spectrum paths (all stay bit-identical,
    asserted). Theory: Bendat, "Probability Functions for Random Responses"
    (NASA CR-33, 1964 — the narrow-band estimate); Dirlik, PhD thesis
    (Warwick, 1985 — the empirical rainflow-range PDF); Wirsching & Light
    (ASCE 1980); Benasciutti & Tovo (Int. J. Fatigue 2005); Newland; Bishop &
    Sherratt (NAFEMS 2000); the Rayleigh / rainflow-range spectral connection;
    Palmgren–Miner linear damage; the ASTM E1049-85 rainflow standard.

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read
    line by line for M20) has no /FATIG, no S-N / Miner branch, no Dirlik /
    rainflow / narrow-band estimator anywhere in the open tree (the sole `PSD`
    token is still `IMUMPSD`, a MUMPS-solver flag). OpenRadioss is a
    time-domain crash/impact code — the stationary random-vibration fatigue
    analysis simply is not part of the open-source solver (the same finding
    M16 made for the real eigensolver, M17–M19 for their frequency-domain
    analyses). So M20 ports spectral fatigue as a clean LIBRARY capability
    behind a minimal PORT card, exactly as M16–M19 did.

    * **STRESS-PSD RECOVERY** (`random_response.py`): the M19 response path is
      extended so the transfer function can target a STRESS (or
      stress-resultant) component, not just a displacement DOF. Because the
      element stress is a LINEAR operator on the nodal displacement for a
      small-strain element on the reference configuration (σ = C:B:u), the
      per-mode STRESS MODES σᵢ = (element stress operator)·φᵢ are recovered by
      running the SAME force kernels the M8 static solve uses on each mode
      shape (`recover_element_stresses` / `stress_modes`), on the ZERO-stress
      reference at x0, restoring the buffers afterward (the M14–M19 read-only
      parity contract — the element/stress kernels are reused READ-ONLY, never
      perturbed). The stress FRF is the modal combination H_σ(Ω) = Σᵢ σᵢ qᵢ(Ω)
      (stress commutes with the superposition — one small (nf,nmode)×(nmode,
      nchan) product, no per-frequency kernel calls), and the stress response
      PSD S_σσ(Ω) = \|H_σ\|² S_ff has its own moments m₀..m₄. Truss/solid/shell
      elements expose their `sig` (axial / 6-Voigt); the spring exposes its
      axial FORCE resultant (its natural stress-resultant channel). Built on
      the REAL-mode FRF (classical damping — the standard random-vibration-
      fatigue assumption). Validated: the recovered STATIC stress = the M8
      implicit-static stress (= P/A closed form); the stress FRF = the direct
      element-stress operator applied to the physical FRF U(Ω).
    * **SPECTRAL FATIGUE-DAMAGE MODELS** (`spectral_fatigue.py`): for an S-N
      curve N = C·S^−m (a C, m input) under a Palmgren–Miner sum, the expected
      damage rate E[D]/T = (ν/C)·E[Sᵐ]. The NARROW-BAND (Bendat) estimate — the
      rate ν₀, the Rayleigh peak/range distribution, the closed-form
      Gamma-function damage (2√2σ)ᵐΓ(1+m/2). The WIDE-BAND corrections — DIRLIK
      (1985)'s empirical three-distribution rainflow-range PDF from
      m₀,m₁,m₂,m₄ (one exponential + two Rayleigh, the closed-form Eq. (9)
      moment), plus WIRSCHING–LIGHT (1980) and TOVO–BENASCIUTTI (2005) as
      cross-checks — each reporting the damage rate, equivalent constant-
      amplitude stress range and time-to-failure. A basic Goodman mean-stress
      option (C_eff = C(1−σ_m/S_u)ᵐ). An ASTM E1049-85 rainflow counter and a
      spectral-representation Gaussian-history synthesiser (an inverse-rFFT of
      the PSD with seeded random phases) drive a MONTE-CARLO cross-check.
      Validated: the narrow-band closed form vs a hand Gamma evaluation; Dirlik
      → narrow band as the bandwidth → 0 (D1,D2 → 0, D3 → 1) and the correct
      wide-band BIAS (less conservative) on a bimodal spectrum; the rainflow
      counter on the canonical ASTM E1049 example; the synthesised RMS = √m₀;
      the Monte-Carlo damage matching Dirlik within the documented scatter
      (~20 % here); Wirsching–Light / Tovo–Benasciutti reducing to narrow band
      in the narrow-band limit; the T_f = 1/rate and S_eq reporting triple.
    * **Engine card + reporting** (`statics._run_fatigue`,
      `random_response.run_fatigue`, mirroring M19's `_run_random_response`):
      /IMPL/FATIG (+ /BASE, /STRS) builds the real-mode FRF, recovers the
      stress modes, forms the stress PSD / moments, picks the CRITICAL channel
      (highest Dirlik damage), evaluates the four estimators + the optional
      Monte-Carlo, and reports the damage rate / equivalent stress / life on
      `model.implicit_result.fatigue` and in the listing. Two card lines: the
      PSD sweep (fmin fmax nf funct [dir] [nmode]) and the S-N + options (m C
      [zeta] [mean ult] [mcdur seed]). A PORT card, minimal like /IMPL/EIGV.
    * **Example**: `examples/spectral_fatigue` — the M19 base-driven 5-mass
      instrument stack re-run under /IMPL/FATIG/BASE to report the fatigue life
      of the critical element (the base spring) under its base-acceleration
      PSD: the four estimators + the Monte-Carlo cross-check, the wide-band
      Dirlik relaxation of the conservative narrow-band life.
    * **Validated** (`tests/test_m20_fatigue.py`): all of the above plus the
      /IMPL/FATIG (+ /BASE, /STRS) card mirror and the parity contract (the
      fatigue path never mutating the M16 eigensolver / M17–M19 transfer
      functions / the element state; the direct /IMPL/DYNA answer byte-
      identical whether or not it runs).

    Deferred out of M20, explicitly (not half-implemented):
    * MULTIAXIAL / critical-plane fatigue and stress-INVARIANT (von Mises /
      signed-von-Mises) equivalent PSDs: the port computes damage from a SINGLE
      scalar stress (or stress-resultant) component's PSD (the critical
      element/component is auto-selected). The full stress-tensor cross-PSD
      matrix and a critical-plane search are the generalization;
    * MEAN-STRESS corrections beyond the basic Goodman intercept option
      (Gerber / Soderberg / Walker, and a per-cycle mean from the rainflow
      pairing);
    * NON-STATIONARY / evolutionary-PSD fatigue and non-Gaussian (kurtosis)
      corrections (the |H|² S law and every PDF assume a STATIONARY GAUSSIAN
      process) — they build on these same moments;
    * CYCLE-COUNTING of a full MULTI-INPUT cross-PSD (needs the M19-deferred
      cross-PSD response first);
    * CRACK-GROWTH / fracture-mechanics fatigue (Paris law, spectral da/dN) — a
      different analysis entirely;
    * the COMPLEX-FRF stress recovery (the stress path is built on the REAL-mode
      FRF; recovering stress through the M18 complex state-space modes is
      deferred, as is M19's complex-FRF base-excitation feed);
    * the unchanged M10–M19 deferral tail: non-stationary / evolutionary PSD,
      multi-input cross-PSD with coherence, multi-directional 100-30-30 response
      spectra, the complex-FRF base-excitation feed, gyroscopic / circulatory
      systems, Lanczos / subspace + AMLS, IFQ ≥ 10 / MODFR 2, /FRICTION
      per-part-pair sets, orthotropic / thermal friction, the fiber TYPE18
      beam, the LAW27 plastic block / solids, thermal contact, TYPE19/24/25,
      Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under implicit,
      the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

20. **M21 — MULTIAXIAL / CRITICAL-PLANE SPECTRAL FATIGUE** ✅ (done):
    the FIRST item deferred out of M20 that BUILDS on it — the frequency-domain
    fatigue-damage estimate of a MULTIAXIAL stress STATE, the natural consumer
    of the M20 scalar-channel fatigue and the vector stress FRF M20 delivered.
    A NEW, parallel path (`implicit/multiaxial_fatigue.py` + the per-element
    Voigt-block helpers and `_run_multiaxial` driver in
    `implicit/random_response.py`) that never touches the M10 integrator, the
    M16 eigensolver, the M17/M18 superposition, the M19 PSD path or the M20
    SCALAR-channel fatigue (all stay bit-identical, asserted). Theory:
    Preumont & Piéfort, "Predicting random high-cycle fatigue life with finite
    elements" (J. Sound Vib. 168, 1994 — the equivalent-von-Mises frequency-
    domain projection); Pitoiset & Preumont, "Spectral methods for multiaxial
    random fatigue analysis" (Int. J. Fatigue 22, 2000 — the trace(Q S) form);
    Carpinteri & Spagnoli (Int. J. Fatigue 23, 2001); Cristofori, Susmel & Tovo
    (Int. J. Fatigue 30, 2008 — the projection-by-direction critical plane);
    Socie & Marquis, "Multiaxial Fatigue" (SAE 2000); Shinozuka & Deodatis (the
    multivariate spectral-representation synthesis).

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read
    line by line for M21) has no /FATIG, no S-N / Miner branch, and NO
    von-Mises / critical-plane / stress-tensor cross-PSD machinery of any kind
    (the sole `PSD` token is still `IMUMPSD`, a MUMPS-solver flag). OpenRadioss
    is a time-domain crash/impact code — the stationary random-vibration
    multiaxial fatigue analysis simply is not part of the open-source solver
    (the same finding M16 made for the real eigensolver, M17–M19 for the
    frequency-domain analyses, M20 for the scalar spectral fatigue). So M21
    ports multiaxial spectral fatigue as a clean LIBRARY capability behind a
    minimal PORT sub-card (/IMPL/FATIG/MULT), exactly as M16–M20 did.

    * **STRESS-TENSOR CROSS-PSD** (`random_response.element_voigt_blocks` /
      `element_voigt_frf` + `multiaxial_fatigue.stress_tensor_cross_psd`): the
      M20 stress recovery already returns the FULL 6-component Voigt stress per
      solid/shell element, so the vector stress FRF H_σ(Ω) = Σᵢ σᵢ qᵢ(Ω)
      (nf, 6) per element is available with NO new kernel work — the M20 vector
      stress modes are reused READ-ONLY, the modal coordinates q sliced by the
      element's 6 channel columns. For a SINGLE scalar random input the
      stress-tensor cross-PSD is the rank-1 Hermitian S_σσ(Ω) = H_σ S_ff H_σᴴ
      (a 6×6 matrix per frequency per element). Validated: for a UNIAXIAL state
      the 6×6 collapses to the M20 scalar σₓₓ channel; the DIAGONAL S[c,c] =
      |H_c|² S_ff equals the M20 per-component channel PSDs; the matrix is
      Hermitian.
    * **EQUIVALENT-STRESS SPECTRAL METHODS** (`multiaxial_fatigue.py`): reduce
      the 6×6 cross-PSD to a scalar EQUIVALENT-stress PSD, then run the four M20
      estimators (narrow-band / Dirlik / Wirsching–Light / Tovo–Benasciutti) on
      its moments. The frequency-domain EQUIVALENT VON MISES (Preumont &
      Piéfort 1994; Pitoiset & Preumont 2000): S_vm(Ω) = trace(Q·S_σσ) with the
      von Mises quadratic operator Q (σ_vm² = σᵀQσ), evaluated through the cheap
      rank-1 HᴴQH form (no 6×6 formed at run time). At least one CRITICAL-PLANE
      method — BOTH the MAX-NORMAL-stress (σₙ = pₙᵀσ, pₙ = [nₓ², …, 2nₓnᵧ, …])
      and the MAX-SHEAR-stress (τ = pₛᵀσ; on each plane the in-plane direction
      of maximum shear variance from the 2×2 in-plane covariance) over a
      searched candidate-plane set (Carpinteri–Spagnoli / Cristofori–
      Susmel–Tovo, the projection-by-direction spectral form) — reporting the
      damage / equivalent stress / life on the critical plane and its normal.
      A seeded MULTIVARIATE Monte-Carlo cross-check: synthesise the CORRELATED
      Gaussian stress-component histories from the cross-PSD via a per-bin
      eigendecomposition / Cholesky of S_σσ (the M20 seeded synthesiser
      generalised to a vector process, batched over the FFT grid), project onto
      the critical plane's LINEAR scalar, ASTM E1049 rainflow, Miner. Validated:
      the equivalent-von-Mises PSD reducing to the M20 scalar answer for a
      uniaxial state; the trace / quadratic-operator identity (trace(Q Mₙ) =
      the direct S_vm moments); the von Mises operator on the textbook
      invariants (uniaxial → 1, pure shear → 3, hydrostatic → 0); the
      critical-plane search recovering the 45° max-normal plane AND the
      coordinate max-shear plane for pure shear, both at the shear amplitude;
      the projection-by-direction moments p^T Mₙ p = the |H·p|² S scalar
      moments; the multivariate synthesis reproducing the full covariance
      (variances AND cross-covariances = M₀); the Monte-Carlo critical-plane
      damage matching Dirlik within the documented scatter.
    * **Engine sub-card + reporting** (`random_response._run_multiaxial`, a
      branch of `run_fatigue`): /IMPL/FATIG/MULT (composing with /BASE, /STRS)
      builds the real-mode FRF, recovers the vector stress modes, groups them
      into per-element 6-Voigt blocks, forms the cross-PSD, reduces to the
      three equivalent-stress PSDs, evaluates the M20 damage models + the
      Monte-Carlo, picks the critical ELEMENT (highest von Mises Dirlik damage)
      and reports the multiaxial damage rate / equivalent stress / critical-
      plane life on `model.implicit_result.fatigue` (the M20 dict, extended
      with `multiaxial`) and in the listing. Two card lines like M20: the PSD
      sweep (fmin fmax nf funct [dir] [nmode]) and the S-N + options
      (m C [zeta] [mean ult] [mcdur seed]). A PORT sub-card, minimal like
      /IMPL/FATIG.
    * **Example**: `examples/multiaxial_fatigue` — a base-clamped SOLID-brick
      cantilever driven by a skew (bending + shear) random tip-force PSD, so
      the clamped-root elements see a genuinely MULTIAXIAL stress state; the
      critical root element's von Mises + max-normal + max-shear critical-plane
      life, the TILTED critical plane (not a coordinate axis), and the
      multivariate Monte-Carlo cross-check.
    * **Validated** (`tests/test_m21_multiaxfatig.py`): all of the above plus
      the /IMPL/FATIG/MULT (+ /BASE) card mirror and the parity contract (the
      multiaxial path never mutating the M16 eigensolver / M17–M19 transfer
      functions / the M20 SCALAR fatigue / the element state; the direct
      /IMPL/DYNA answer byte-identical whether or not it runs).

    Deferred out of M21, explicitly (not half-implemented):
    * NON-PROPORTIONAL multiaxial cycle-counting beyond the projected-equivalent
      methods: the full TENSOR rainflow / minimum-circumscribed-circle
      (Papadopoulos) shear-amplitude path counting and the rotating-principal-
      axes non-proportional-hardening correction — the port counts damage on a
      PROJECTED scalar (von Mises or a critical-plane resolved stress), which is
      exact for proportional loading and the standard spectral approximation
      otherwise;
    * the Monte-Carlo cross-check validates the LINEAR critical-plane
      projection; a VON-MISES (quadratic, hence non-Gaussian) time-domain
      rainflow cross-check is DEFERRED (the spectral von-Mises PDF is itself an
      approximation — Pitoiset & Preumont 2000 discuss the bias);
    * MEAN-STRESS corrections beyond the basic M20 Goodman intercept (carried
      through to the equivalent scalar) — Gerber / Soderberg / Walker and a
      per-plane mean remain DEFERRED (M20);
    * a full MULTI-INPUT cross-PSD with coherence (a non-diagonal S_ff and a
      coherence model — the same M19/M20 deferral): the cross-PSD here is the
      rank-1 H S_ff H^H of a SINGLE scalar input process;
    * NON-STATIONARY / evolutionary-PSD fatigue, NON-GAUSSIAN (kurtosis)
      corrections, CRACK-GROWTH / fracture-mechanics fatigue, and the
      COMPLEX-FRF stress recovery (the stress path is built on the REAL-mode
      FRF) — the unchanged M20 tail;
    * the unchanged M10–M20 deferral tail: non-stationary / evolutionary PSD,
      multi-input cross-PSD with coherence, multi-directional 100-30-30 response
      spectra, the complex-FRF base-excitation feed, gyroscopic / circulatory
      systems, Lanczos / subspace + AMLS, IFQ ≥ 10 / MODFR 2, /FRICTION
      per-part-pair sets, orthotropic / thermal friction, the fiber TYPE18
      beam, the LAW27 plastic block / solids, thermal contact, TYPE19/24/25,
      Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under implicit,
      the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

21. **M22 — NON-PROPORTIONAL MULTIAXIAL FATIGUE (critical-plane time-domain
    path counting)** ✅ (done): the FIRST item deferred out of M21 that BUILDS on
    it — the CRITICAL-PLANE, TIME-DOMAIN, PATH-COUNTING damage of a multiaxial
    stress state whose principal axes ROTATE (a non-proportional load path), the
    rotating-shear PATH the M21 projected-equivalent (von Mises / resolved-scalar)
    spectral methods deliberately deferred. The natural consumer of the M21
    multivariate synthesiser (`synthesize_multiaxial_history`) and candidate-plane
    machinery (`candidate_normals` / `_inplane_basis` / `normal_projection` /
    `shear_projection`). A NEW, parallel path (`implicit/nonproportional_fatigue.py`
    + the `_run_nonproportional` extension of the `_run_multiaxial` driver in
    `implicit/random_response.py`) that never touches the M10 integrator, the M16
    eigensolver, the M17/M18 superposition, the M19 PSD path, the M20 SCALAR
    fatigue or the M21 MULTIAXIAL SPECTRAL reductions (all stay bit-identical —
    the M21 spectral answer is byte-identical whether or not /NPROP runs,
    asserted). Theory: Papadopoulos, "Critical plane approaches in high-cycle
    fatigue: on the definition of the amplitude and mean value of the shear
    stress" (FFEMS 21, 1998 — the minimum circumscribed circle); Mamiya, Araújo &
    Castro, "Prismatic hull: a new measure of shear stress amplitude" (Int. J.
    Fatigue 31, 2009 — the maximum rectangular hull); Findley 1959 (τ_a + k
    σ_n,max); Fatemi & Socie 1988 (γ_a(1 + k σ_n,max/σ_y)); Matake 1977;
    Carpinteri & Spagnoli 2001; Itoh, Sakane, Ohnami & Socie 1995 (the F_np
    non-proportionality factor); Socie & Marquis, "Multiaxial Fatigue" ch. 2–4
    (the rotating-principal-axes / non-proportional-hardening background).

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read line
    by line for M22) has no /FATIG, no S-N / Miner branch, and NO critical-plane /
    MCC / Findley / Fatemi-Socie / path-counting machinery of any kind (the sole
    `PSD` token is still `IMUMPSD`, a MUMPS-solver flag). OpenRadioss is a
    time-domain crash/impact code — the stationary random-vibration MULTIAXIAL
    fatigue analysis, proportional (M21) OR non-proportional (M22), simply is not
    part of the open-source solver (the same finding M16–M21 made). So M22 ports
    non-proportional multiaxial fatigue as a clean LIBRARY capability behind a
    minimal PORT sub-flag (/IMPL/FATIG/MULT/NPROP), exactly as M16–M21 did.

    * **SHEAR-PATH AMPLITUDE OPERATORS** (`nonproportional_fatigue.py`): on a
      candidate plane of normal n, the 2-D resolved-shear locus (τ_a(t), τ_b(t))
      in the M21 in-plane basis (`resolved_shear_path` = a `shear_projection` per
      in-plane axis applied to a synthesised stress-component history), and its
      "amplitude" three standard ways — the MINIMUM CIRCUMSCRIBED CIRCLE
      (`min_circumscribed_circle`, Papadopoulos; Welzl on the convex hull), the
      LONGEST CHORD / longest projection (`longest_chord`, the diameter), and the
      MAXIMUM RECTANGULAR / PRISMATIC HULL (`max_rectangular_hull`,
      Mamiya–Araújo, max_θ √(a1²+a2²)). Plus the non-proportionality factor
      `nonproportionality_factor` F_np = √(λ₂/λ₁) of the 2-D shear-path covariance
      (the Itoh–Kanazawa aspect-ratio form). Validated: for a PROPORTIONAL
      (in-phase) line all three collapse to the scalar M21 amplitude and F_np = 0;
      for a CIRCULAR (90°-out-of-phase, equal-amplitude) path the MCC radius = the
      component amplitude, the longest chord = the diameter (2r) and the MRH =
      r√2, with F_np = 1 (closed forms); an ellipse's MRH = √(p²+q²), MCC = the
      major semi-axis, F_np = q/p; the MCC on an off-centre circle recovers the
      radius and centre (amplitude mean-independent — Papadopoulos separates the
      shear amplitude from its mean).
    * **CRITICAL-PLANE TIME-DOMAIN DAMAGE** (`nonproportional_fatigue.py`): over
      the M21-synthesised correlated stress-component histories (seeded), for each
      candidate plane resolve σ_n(t) and the shear PATH, rainflow (ASTM E1049, the
      M20 counter) the DOMINANT resolved shear scaled by the non-proportional path
      factor g = MRH/scalar (1 for a line), and apply FINDLEY (τ_a + k σ_n,max)
      and FATEMI–SOCIE (γ_a(1 + k σ_n,max/σ_y), the shear modulus cancelling in the
      stress-based port) carrying the per-plane MAX NORMAL stress (which folds in
      that plane's MEAN normal stress — a piece of the M20/M21 mean-stress
      deferral). The critical plane is searched by the model's CRITERION PARAMETER
      (the textbook Findley/FS definition — amplitude-only, so the O(n_planes) scan
      costs no rainflow; only the winning plane is rainflowed), reporting the
      critical plane, F_np, damage rate / life. Validated: for PROPORTIONAL
      loading the path count (shear-path model, g = 1) reduces to the M21 max-shear
      critical-plane rainflow within the seeded Monte-Carlo scatter; for a
      90°-OUT-OF-PHASE (rotating) case the non-proportional damage is g^m ≈
      2^(m/2) HIGHER than the scalar projection (the M21 spectral method's linear
      projection) at the same channel amplitudes — the extra damage the spectral
      method misses — with F_np ≈ 1; a hand check of the MCC radius / MRH /
      Findley / Fatemi-Socie damage on a synthetic two-channel sinusoid with a
      constant normal stress.
    * **Engine sub-flag + reporting** (`random_response._run_nonproportional`,
      inside `_run_multiaxial`): /IMPL/FATIG/MULT/NPROP (implies /MULT; composes
      with /BASE / /STRS) runs everything /MULT does — the M21 spectral reductions,
      byte-identical — and THEN, ALONGSIDE, synthesises the critical element's
      correlated stress-component histories (seeded via the M20/M21 mcdur/seed
      args), searches the Findley / Fatemi-Socie / shear-path critical plane and
      reports the non-proportional damage rate / F_np / critical-plane life on
      `model.implicit_result.fatigue['nprop_result']` and in the listing —
      ALONGSIDE the M21 spectral von-Mises / critical-plane numbers, so the listing
      shows the non-proportional correction explicitly. Card line 2 optionally
      appends the Findley/FS normal-sensitivity k and the yield stress σ_y
      (m C zeta mean ult mcdur seed [k σ_y]). A PORT sub-flag, minimal like
      /IMPL/FATIG/MULT.
    * **Example**: `examples/nonproportional_fatigue` — the M21 solid-brick
      cantilever re-run with a RECTANGULAR (1×2 mm) cross-section (so the two
      transverse bending modes SPLIT to ~33 and ~60 kHz) driven in BOTH transverse
      directions by one random PSD: the split modes decorrelate the root shears
      σ_xy (~33 kHz) and σ_zx (~60 kHz) in time, so the resolved shear traces a
      rotating near-circular PATH — a genuinely NON-PROPORTIONAL state (F_np ≈
      0.89–0.97), MRH ≈ √2·MCC, and the non-proportional critical-plane life
      ≈ g^m shorter than the M21 proportional-spectral projection of the same
      state.
    * **Validated** (`tests/test_m22_nproptfatig.py`): all of the above plus the
      /IMPL/FATIG/MULT/NPROP card mirror (a PORT sub-flag, composes in any order,
      implies /MULT) and the parity contract (the non-proportional path never
      mutating the M16 eigensolver / M17–M19 transfer functions / the M20 SCALAR
      fatigue / the M21 SPECTRAL reductions / the element state; the M21 spectral
      damage rates AND critical-plane orientations byte-identical whether or not
      /NPROP runs; the direct /IMPL/DYNA answer byte-identical).

    Deferred out of M22, explicitly (not half-implemented):
    * a purely FREQUENCY-DOMAIN non-proportionality estimator that avoids a
      synthesised history — the Cristofori–Susmel–Tovo / Pitoiset spectral
      non-proportionality factor: M22 is a TIME-DOMAIN path count over the M21
      multivariate synthesised history (seeded), not a spectral estimator;
    * MEAN-STRESS beyond the per-plane MAX normal stress σ_n,max (which folds in
      the plane's own mean) and the basic M20/M21 Goodman intercept — Gerber /
      Soderberg / Walker remain DEFERRED (M20/M21);
    * NON-PROPORTIONAL HARDENING as a MATERIAL model (the extra cyclic hardening a
      rotating path induces in the flow curve): F_np is reported and available as a
      Carpinteri–Spagnoli damage multiplier, but the port's S-N curve is the
      proportional one (the standard high-cycle-fatigue assumption — the hardening
      matters mainly in LOW-cycle fatigue, Socie & Marquis);
    * a MULTI-INPUT cross-PSD with coherence (a non-diagonal S_ff and a coherence
      model): the cross-PSD here is the rank-1 H S_ff H^H of a SINGLE scalar input
      process (the non-proportionality of the TIME history comes from the
      frequency-dependent multi-mode FRF phase, not from multiple inputs) — the
      same M19–M21 deferral;
    * CRACK-GROWTH / fracture-mechanics fatigue; NON-STATIONARY / evolutionary-PSD
      and NON-GAUSSIAN (kurtosis) corrections; the COMPLEX-FRF stress recovery
      (the stress path is built on the REAL-mode FRF) — the unchanged M20/M21 tail;
    * the unchanged M10–M21 deferral tail: multi-input cross-PSD with coherence,
      multi-directional 100-30-30 response spectra, the complex-FRF base-excitation
      feed, gyroscopic / circulatory systems, Lanczos / subspace + AMLS, IFQ ≥ 10
      / MODFR 2, /FRICTION per-part-pair sets, orthotropic / thermal friction, the
      fiber TYPE18 beam, the LAW27 plastic block / solids, thermal contact,
      TYPE19/24/25, Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under
      implicit, the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

22. **M23 — SPECTRAL NON-PROPORTIONAL MULTIAXIAL FATIGUE (frequency-domain F_np +
    critical-plane damage from the cross-PSD moment matrices)** ✅ (done): the
    FIRST item deferred out of M22 that BUILDS on it — the FREQUENCY-DOMAIN
    non-proportionality factor and critical-plane damage estimated DIRECTLY from
    the stress-tensor cross-PSD spectral-MOMENT matrices, with NO synthesised
    history: the Cristofori–Susmel–Tovo / Pitoiset SPECTRAL non-proportional
    method the M22 time-domain path count deliberately deferred. The SPECTRAL
    sibling of M22 (the same F_np and critical-plane concepts, read spectrally off
    the M21 moment matrices instead of counted over a synthesised history), the
    natural consumer of the M21 cross-PSD / tensor-moment machinery
    (`tensor_moment_matrices` / `normal_projection` / `shear_projection` /
    `_inplane_basis` / `candidate_normals`) and the M22 F_np / critical-plane /
    shear-path concepts. A NEW, parallel path
    (`implicit/spectral_nonproportional_fatigue.py` + the
    `_run_spectral_nonproportional` extension of the `_run_multiaxial` driver in
    `implicit/random_response.py`) that never touches the M10 integrator, the M16
    eigensolver, the M17/M18 superposition, the M19 PSD path, the M20 SCALAR
    fatigue, the M21 MULTIAXIAL SPECTRAL reductions OR the M22 NON-PROPORTIONAL
    TIME-DOMAIN path (all stay bit-identical — the M21 spectral AND the M22
    time-domain answers are byte-identical whether or not /SPEC runs, asserted).
    Theory: Pitoiset & Preumont, "Spectral methods for multiaxial random fatigue
    analysis of metallic structures" (Int. J. Fatigue 22, 2000 — the frequency-
    domain multiaxial projection + the cross-PSD eigenstructure); Cristofori,
    Susmel & Tovo, "A stress-invariant based spectral method to estimate fatigue
    life under multiaxial random loading" (Int. J. Fatigue 30, 2008 — the spectral
    critical-plane / modified-Wöhler estimator this module ports); Susmel &
    Lazzarin, "A bi-parametric Wöhler curve for high cycle multiaxial fatigue
    assessment" (FFEMS 25, 2002 — the modified Wöhler curve method + the stress
    ratio ρ = σ_a/τ_a); Bäckström & Marquis 2001; Carpinteri, Spagnoli & Vantadori
    2013 (the spectral critical-plane weighting).

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read line
    by line for M23, 639 lines fetched from raw.githubusercontent.com) has no
    /FATIG, no S-N / Miner branch, and NO frequency-domain non-proportionality /
    modified-Wöhler / spectral-invariant / critical-plane machinery of any kind
    (the sole `PSD` token is still `IMUMPSD`, a MUMPS-solver flag at line 269).
    OpenRadioss is a time-domain crash/impact code — the stationary random-
    vibration MULTIAXIAL fatigue analysis, proportional (M21) OR non-proportional
    time-domain (M22) OR non-proportional spectral (M23), simply is not part of the
    open-source solver (the same finding M16–M22 made). So M23 ports the spectral
    non-proportional estimator as a clean LIBRARY capability behind a minimal PORT
    sub-flag (/IMPL/FATIG/MULT/NPROP/SPEC), exactly as M16–M22 did.

    * **SPECTRAL SHEAR-PATH / NON-PROPORTIONALITY FACTOR**
      (`spectral_nonproportional_fatigue.py`): on a candidate plane of normal n,
      the 2×2 IN-PLANE SHEAR cross-spectral moment matrix Σ_τ =
      [[pₐᵀM₀pₐ, pₐᵀM₀p_b],[·, p_bᵀM₀p_b]] (`inplane_shear_covariance`, the 2×2
      block of the M21 M₀ = E[σσᵀ] built from `shear_projection` along each
      in-plane axis — NO synthesis), and the SPECTRAL non-proportionality factor
      F_np = √(λ₂/λ₁) of its eigenvalues (`spectral_nonproportionality_factor`),
      plus the spectral resolved-shear / resolved-normal moments from p^T Mₙ p.
      Validated: the 2×2 shear block EQUALS the M22 time-domain shear-path
      covariance (np.cov of the synthesised path) — the M₀ = E[σσᵀ] identity; the
      SPECTRAL F_np EQUALS the M22 TIME-DOMAIN F_np on the same cross-PSD; F_np = 0
      for a proportional (rank-1) state and → 1 for the equal-amplitude
      90°-out-of-phase circle (closed forms); the spectral resolved amplitudes
      matching the M22 synthesised RMS within the seeded scatter.
    * **SPECTRAL CRITICAL-PLANE DAMAGE** (`spectral_nonproportional_fatigue.py`):
      over the M21 moment matrices (no history), for each candidate plane the
      dominant-shear PSD moments mₙ^τ = p_domᵀMₙp_dom (p_dom the leading
      eigenvector of Σ_τ), the resolved-normal moments p_nᵀMₙp_n, the CLOSED-FORM
      non-proportional amplitude correction g² = 1 + F_np² = trace(Σ_τ)/λ₁ =
      (MRH/MCC)² of an elliptical path (so the effective shear variance is
      λ₁+λ₂ = trace, the M22 prismatic-hull effect read spectrally), the
      Susmel–Tovo modified-Wöhler stress ratio ρ = σ_a/τ_a, and the equivalent-
      stress moments per model — FINDLEY (the spectral-invariant LINEAR
      combination g·τ + k·σ_n, itself a linear projection so its moments come
      straight from Mₙ), FATEMI–SOCIE ((1 + k σ_n,a/σ_y)²·(1+F_np²)·mₙ^τ) and
      SHEAR-PATH ((1+F_np²)·mₙ^τ) — searched by the amplitude-only criterion
      parameter and run through the four M20 estimators (Dirlik etc.). Reports the
      critical plane, F_np, ρ, damage rate / equivalent stress / life. Validated:
      PROPORTIONAL loading (F_np = 0) → the spectral shear-path damage reduces
      EXACTLY to the M21 max-shear critical-plane spectral answer (no correction);
      90°-out-of-phase → the spectral non-proportional damage is (1+F_np²)^(m/2) =
      2^(m/2) HIGHER than the uncorrected M21 projected-scalar answer AND AGREES
      with the M22 TIME-DOMAIN path-counting damage within the seeded Monte-Carlo
      scatter (the two non-proportional methods — spectral estimator vs time-domain
      path count — converge, the whole point); a hand check of the spectral F_np /
      ρ / g on a synthetic two-channel cross-PSD.
    * **Engine sub-flag + reporting** (`random_response._run_spectral_nonproportional`,
      inside `_run_multiaxial`): /IMPL/FATIG/MULT/NPROP/SPEC (implies /NPROP hence
      /MULT; composes with /BASE / /STRS) runs everything /MULT (M21 spectral) and
      /NPROP (M22 time-domain path count) do — both byte-identical — and THEN,
      ALONGSIDE, computes the spectral non-proportional critical-plane damage on
      the critical element from the SAME 6×6 moment matrices (`summ["Mmats"]`)
      WITHOUT synthesising a history, reporting the spectral non-proportional
      damage rate / F_np / ρ / critical-plane life on
      `model.implicit_result.fatigue['nprop_result']['spectral']` and in the
      listing — ALONGSIDE the M21 spectral von-Mises / critical-plane numbers AND
      the M22 time-domain path-counting numbers, so the listing shows ALL THREE
      (proportional-spectral, non-proportional time-domain, non-proportional
      spectral) side by side. A PORT sub-flag, minimal like /IMPL/FATIG/MULT/NPROP.
    * **Example**: `examples/spectral_nonproportional_fatigue` — the M22
      rectangular-section (1×2 mm) solid-brick cantilever re-run to report its
      SPECTRAL non-proportional critical-plane life ALONGSIDE its M22 TIME-DOMAIN
      path-counting life and its M21 proportional-spectral life, the three side by
      side: the split bending modes (~33 / ~60 kHz) make the root shear path a
      rotating near-circle (F_np ≈ 0.89–0.97 — the SPECTRAL F_np matching the M22
      TIME-DOMAIN F_np to ~1e-4), the closed-form path factor g = √(1+F_np²) ≈
      1.34–1.41, and the spectral non-proportional lives (~2.4×10⁹ ms) CLOSE to
      the M22 time-domain lives (~2.1×10⁹ ms) — the two non-proportional methods
      converge — both ~g^m ≈ 5× SHORTER than the M21 proportional-spectral
      projection (~1.3×10¹⁰ ms).
    * **Validated** (`tests/test_m23_specnprop.py`): all of the above plus the
      /IMPL/FATIG/MULT/NPROP/SPEC card mirror (a PORT sub-flag, composes in any
      order, implies /NPROP hence /MULT) and the parity contract (the spectral
      path never mutating the M16 eigensolver / M17–M19 transfer functions / the
      M20 SCALAR fatigue / the M21 SPECTRAL reductions / the M22 TIME-DOMAIN path
      count / the element state; the M21 spectral damage rates AND the M22
      time-domain damage rates + critical-plane orientations byte-identical whether
      or not /SPEC runs; the direct /IMPL/DYNA answer byte-identical).

    Deferred out of M23, explicitly (not half-implemented):
    * the non-proportional amplitude correction uses the EXACT-ellipse factor
      g = √(1 + F_np²) (the MRH/MCC closed form) and REUSES the dominant-shear PSD
      spectral SHAPE for the higher moments; M22 measures the actual MRH of the
      synthesised hull and rainflow-counts the real history — the two converge for
      a clean ellipse and agree within the M20 rainflow-vs-Dirlik + path scatter
      otherwise (the exact-ellipse shape reuse is the documented spectral
      approximation);
    * the normal-stress AMPLITUDE is a peak-stress factor × RMS (default psf = √2,
      the equivalent-sinusoid amplitude of a narrow-band Gaussian), NOT a
      per-record σ_n,max (a time-domain object); MEAN-STRESS beyond the per-plane
      normal and the basic M20/M21 Goodman intercept — Gerber / Soderberg / Walker
      — remain DEFERRED (M20/M21/M22);
    * the spectral Findley uses the LINEAR-COMBINATION (spectral-invariant) form
      g·τ + k·σ_n (a single linear projection whose moments come straight from Mₙ,
      folding in the shear/normal phase), which differs from M22's ADDITIVE
      record-max form — the spectral form is the natural frequency-domain object;
    * NON-PROPORTIONAL HARDENING as a MATERIAL model (F_np is reported and drives
      the amplitude correction, but the port's S-N curve is the proportional one —
      the standard high-cycle-fatigue assumption, Socie & Marquis);
    * a MULTI-INPUT cross-PSD with coherence (the cross-PSD here is the rank-1
      H S_ff Hᴴ of a SINGLE scalar input — the non-proportionality of the response
      comes from the frequency-dependent multi-mode FRF phase, not multiple
      inputs) — the same M19–M22 deferral;
    * CRACK-GROWTH / fracture-mechanics fatigue; NON-STATIONARY / evolutionary-PSD
      and NON-GAUSSIAN (kurtosis) corrections; the COMPLEX-FRF stress recovery —
      the unchanged M20/M21/M22 tail;
    * the unchanged M10–M22 deferral tail: multi-input cross-PSD with coherence,
      multi-directional 100-30-30 response spectra, the complex-FRF base-excitation
      feed, gyroscopic / circulatory systems, Lanczos / subspace + AMLS, IFQ ≥ 10
      / MODFR 2, /FRICTION per-part-pair sets, orthotropic / thermal friction, the
      fiber TYPE18 beam, the LAW27 plastic block / solids, thermal contact,
      TYPE19/24/25, Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under
      implicit, the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

23. **M24 — NON-GAUSSIAN / KURTOSIS SPECTRAL FATIGUE** ✅ (done):
    the FIRST item deferred out of M20–M23 that BUILDS on the spectral-fatigue
    machinery — the frequency-domain damage of a stationary but NON-GAUSSIAN
    random-vibration response, computed by CORRECTING the M20–M23 Gaussian
    spectral estimators for a specified kurtosis (and skewness), with the
    correction cross-validated against a non-Gaussian time-domain Monte-Carlo.
    Every spectral estimator so far assumed a stationary GAUSSIAN response (the
    |H|²S law of a Gaussian input, the Rayleigh peak / rainflow-range PDFs); M24
    lifts EXACTLY that Gaussian assumption while keeping the process stationary —
    the natural consumer of the M20 scalar / M21–M23 multiaxial moment + Dirlik
    machinery. A NEW, parallel path (`implicit/nongaussian_fatigue.py` + the
    `_run_nongaussian` / `_run_nongaussian_multiaxial` extensions of the M20/M21
    drivers in `implicit/random_response.py`) that never touches the M10
    integrator, the M16 eigensolver, the M17/M18 superposition, the M19 PSD path,
    the M20 SCALAR fatigue, the M21 MULTIAXIAL SPECTRAL path, the M22
    NON-PROPORTIONAL TIME-DOMAIN path OR the M23 SPECTRAL NON-PROPORTIONAL path
    (all stay bit-identical — the M20–M23 answers are byte-identical whether or
    not /NGAUSS runs, asserted). Theory: Winterstein, "Nonlinear vibration models
    for extremes and fatigue" (J. Eng. Mech. 114, 1988 — the Hermite-moment
    transformation g(u)); Winterstein & MacKenzie 1997 / Winterstein & Kashef 2000
    (the refined softening coefficients); Benasciutti & Tovo, "Cycle distribution
    and fatigue damage assessment in broad-band non-Gaussian random processes"
    (Prob. Eng. Mech. 20, 2005; Int. J. Fatigue 2006 — the bandwidth-dependent
    non-Gaussian rainflow correction); Braccesi, Cianetti, Lori & Pioli, "Fatigue
    behaviour analysis of mechanical components subject to random bimodal stress
    process: frequency domain approach" (Int. J. Fatigue 31, 2009 — the closed-form
    non-Gaussian damage correction coefficient); Rizzi, Kihm, Ferguson, Przekop et
    al. (the kurtosis-corrected spectral damage); Kihm & Rizzi, "Understanding how
    kurtosis is transferred from input acceleration to stress response" (2013 —
    non-Gaussian random fatigue); Ochi & Ahn 1994.

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read line
    by line for M24, 639 lines fetched from raw.githubusercontent.com) has no
    /FATIG, no S-N / Miner branch, and NO non-Gaussian / kurtosis / Hermite-moment
    / Winterstein machinery of any kind (the sole `PSD` token is still `IMUMPSD`, a
    MUMPS-solver flag at line 269). OpenRadioss is a time-domain crash/impact code
    — the stationary random-vibration fatigue analysis, Gaussian (M20–M23) OR
    non-Gaussian (M24), simply is not part of the open-source solver (the same
    finding M16–M23 made). So M24 ports the non-Gaussian / kurtosis correction as a
    clean LIBRARY capability behind a minimal PORT sub-flag (/IMPL/FATIG/NGAUSS —
    the non-Gaussian analogue of the M20 /IMPL/FATIG Gaussian card), exactly as
    M16–M23 did.

    * **NON-GAUSSIAN CORRECTION FACTOR + HERMITE MOMENTS**
      (`nongaussian_fatigue.py`): the Winterstein Hermite-moment model
      g(u) = κ[u + h₃(u²−1) + h₄(u³−3u)] (`hermite_coefficients` /
      `hermite_transform`) maps a standard Gaussian u to the non-Gaussian process
      for a target kurtosis γ₄ and skewness γ₃; because the Hermite polynomials are
      Gaussian-orthogonal the transform has EXACTLY zero mean and (with
      κ = 1/√(1 + 2h₃² + 6h₄²)) EXACTLY unit variance, and the coefficients hit the
      target moments (the refined softening fit h₄ = (√(1 + 1.5(γ₄−3)) − 1)/18,
      h₃ = γ₃/(6(1 + 6h₄)) for γ₄ ≥ 3; the first-order h₄ = (γ₄−3)/24 for a
      platykurtic target). The closed-form correction coefficient (Benasciutti–
      Braccesi / Rizzi–Kihm) is the ratio of the non-Gaussian to the Gaussian
      m-th range moment, the Winterstein amplitude transform integrated over the
      Rayleigh(1) amplitude: λ_ng = E[g(V)ᵐ]/E[Vᵐ] (`nongaussian_correction_
      factor`), which SCALES the Gaussian narrow-band / Dirlik / Wirsching-Light /
      Tovo-Benasciutti damage (`nongaussian_summary`), with an optional
      Benasciutti–Tovo bandwidth attenuation γ₄,eff = 3 + (γ₄−3)α₂ (a wide-band
      process gets less correction). Validated: λ_ng = 1 for a GAUSSIAN process
      (γ₄ = 3, γ₃ = 0) — the M20 answer recovered EXACTLY (the identity transform);
      λ_ng > 1 for a LEPTOKURTIC (γ₄ > 3, spiky) process and < 1 for a PLATYKURTIC
      (γ₄ < 3) one (closed forms), monotone in γ₄ and the slope m; the Hermite
      transform preserving the mean/variance EXACTLY and hitting the target
      kurtosis (an exact-moment hand check + a large-sample check).
    * **NON-GAUSSIAN MONTE-CARLO CROSS-CHECK** (`nongaussian_fatigue.py`): a
      non-Gaussian time-domain synthesiser — the M20 Gaussian spectral-
      representation history pushed through the memoryless Hermite / power-law
      transform g(u) to the target (γ₃, γ₄) (`synthesize_nongaussian_history`) —
      rainflow-counted (the M20 ASTM E1049 counter) and Miner-summed
      (`nongaussian_monte_carlo_damage`, and `nongaussian_monte_carlo_projected`
      for the multiaxial critical-plane projection). The static/memoryless
      transform preserves the PSD SHAPE only approximately (a cubic nonlinearity
      injects harmonics — the documented bandwidth caveat: the correction is exact
      for a narrow band). Validated: the synthesised history hitting the target
      sample kurtosis/skewness; the non-Gaussian Monte-Carlo damage matching the
      λ_ng-corrected spectral damage within the seeded scatter on a NARROW-band
      process (where the amplitude and instantaneous transforms coincide); the
      Gaussian limit (γ₄ = 3) reducing the non-Gaussian Monte-Carlo EXACTLY
      (bit-identical) to the M20 Gaussian Monte-Carlo.
    * **Engine sub-flag + reporting** (`random_response._run_nongaussian` /
      `_run_nongaussian_multiaxial`): /IMPL/FATIG/NGAUSS (composing with /MULT,
      /NPROP, /SPEC — it is ORTHOGONAL, a scalar correction on whatever
      equivalent-stress PSD the M20–M23 path produces) recovers the (scalar OR
      multiaxial von-Mises / critical-plane) stress PSD as M20–M23 do, computes
      λ_ng and the corrected damage rate / equivalent stress / life, runs the
      non-Gaussian Monte-Carlo cross-check, and reports the non-Gaussian damage /
      kurtosis / λ_ng ALONGSIDE the M20 Gaussian numbers on
      `model.implicit_result.fatigue['nongaussian']` and in the listing (a new
      NON-GAUSSIAN / KURTOSIS block showing the Gaussian and the kurtosis-corrected
      answers side by side). The kurtosis (and optional skewness) go on a THIRD
      card line: kurt [skew]. A PORT sub-flag, minimal like /IMPL/FATIG.
    * **Example**: `examples/nongaussian_fatigue` — the M20 `spectral_fatigue`
      base-excited instrument stack re-run under /IMPL/FATIG/NGAUSS/BASE with a
      leptokurtic input (γ₄ = 6): the M20 Gaussian block UNCHANGED, then the
      NON-GAUSSIAN / KURTOSIS block with λ_ng ≈ 2.85 and the ~3× shorter
      kurtosis-corrected life, plus the non-Gaussian Monte-Carlo cross-check.
    * **Validated** (`tests/test_m24_ngfatig.py`): all of the above plus the
      /IMPL/FATIG/NGAUSS (composing with /MULT, /NPROP, /SPEC) card mirror and the
      parity contract (the non-Gaussian path never mutating the M16 eigensolver /
      M17–M19 transfer functions / the M20 SCALAR fatigue / the M21–M23 MULTIAXIAL
      reductions / the element state; the M20 SCALAR AND the M21/M22/M23
      MULTIAXIAL damage rates byte-identical whether or not /NGAUSS runs; the
      direct /IMPL/DYNA answer byte-identical).

    Deferred out of M24, explicitly (not half-implemented):
    * NON-STATIONARY / evolutionary-PSD fatigue — the kurtosis here is a
      STATIONARY non-Gaussian correction (a fixed one-point marginal), NOT a
      time-varying PSD; a genuinely non-stationary (burst / run-up) process is a
      different analysis and remains DEFERRED;
    * a full non-Gaussian MULTIAXIAL JOINT distribution (a vector Hermite transform
      of the correlated stress tensor): M24 applies the SCALAR kurtosis correction
      to the (von Mises / critical-plane) equivalent scalar the M21/M23 reductions
      already produce, so it COMPOSES with the multiaxial / non-proportional paths
      but does not model a joint non-Gaussian tensor;
    * the WIDE-BAND correction uses the narrow-band-exact Winterstein amplitude
      transform with a first-order α₂ bandwidth attenuation (Benasciutti–Tovo), NOT
      the exact empirical Braccesi 2009 fit constants (DEFERRED — the Hermite-
      derived form is more principled and matches the Monte-Carlo directly); the
      memoryless transform preserves the PSD shape only approximately (documented);
    * MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
      fracture-mechanics fatigue, the COMPLEX-FRF stress recovery and a MULTI-INPUT
      cross-PSD with coherence — the unchanged M20–M23 tail;
    * the unchanged M10–M23 deferral tail: multi-input cross-PSD with coherence,
      multi-directional 100-30-30 response spectra, the complex-FRF base-excitation
      feed, gyroscopic / circulatory systems, Lanczos / subspace + AMLS, IFQ ≥ 10
      / MODFR 2, /FRICTION per-part-pair sets, orthotropic / thermal friction, the
      fiber TYPE18 beam, the LAW27 plastic block / solids, thermal contact,
      TYPE19/24/25, Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under
      implicit, the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

24. **M25 — NON-STATIONARY / EVOLUTIONARY-PSD SPECTRAL FATIGUE** ✅ (done):
    the FIRST item deferred out of M20–M24 that BUILDS on the spectral-fatigue
    machinery — the frequency-domain damage of a random-vibration response whose
    PSD (or its variance / RMS) VARIES WITH TIME, computed by extending the
    M20–M24 STATIONARY spectral estimators to a time-varying spectrum, and
    cross-validated against a non-stationary time-domain Monte-Carlo. Every
    spectral estimator so far assumed a STATIONARY process — a single,
    time-invariant PSD: M20 (scalar), M21–M23 (multiaxial) and M24 (non-Gaussian)
    all evaluate ONE fixed set of moments. M25 lifts EXACTLY that stationarity
    assumption — the M24 SIBLING (M24 lifted the Gaussian assumption at fixed
    stationarity; M25 lifts the stationarity assumption). A NEW, parallel path
    (`implicit/nonstationary_fatigue.py` + the `_run_nonstationary` /
    `_run_nonstationary_multiaxial` extensions of the M20/M21 drivers in
    `implicit/random_response.py`) that never touches the M10 integrator, the M16
    eigensolver, the M17/M18 superposition, the M19 PSD path, the M20 SCALAR
    fatigue, the M21 MULTIAXIAL SPECTRAL path, the M22 NON-PROPORTIONAL
    TIME-DOMAIN path, the M23 SPECTRAL NON-PROPORTIONAL path OR the M24
    NON-GAUSSIAN correction (all stay bit-identical — the M20–M24 answers are
    byte-identical whether or not /NSTAT runs, asserted). Theory: Priestley,
    "Evolutionary spectra and non-stationary processes" (J. Roy. Statist. Soc. B
    27, 1965 — the evolutionary spectrum S(ω,t)); Bendat & Piersol, "Random Data"
    ch. 12 (non-stationary random data); Wolfsteiner & Breuer / Wolfsteiner &
    Trapp, "Fatigue life due to non-stationary vibration" (the spectrogram /
    RMS-mixture method); Braccesi, Cianetti, Lori & Pioli (non-stationary
    frequency-domain fatigue); Kihm, Ferguson & Antoni (non-stationary /
    non-Gaussian random-vibration fatigue); Rychlik (the switching-process /
    Markov-modulated stationary model); Palmgren–Miner (the block-damage sum);
    Newland ch. 5–7.

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read line
    by line for M25) has no /FATIG, no S-N / Miner branch, and NO non-stationary /
    evolutionary / spectrogram / mission-profile machinery of any kind (the sole
    `PSD` token is still `IMUMPSD`, a MUMPS-solver flag at line 269). OpenRadioss
    is a time-domain crash/impact code — the random-vibration fatigue analysis,
    stationary Gaussian (M20–M23), stationary non-Gaussian (M24) OR non-stationary
    (M25), simply is not part of the open-source solver (the same finding M16–M24
    made). So M25 ports non-stationary spectral fatigue as a clean LIBRARY
    capability behind a minimal PORT sub-flag (/IMPL/FATIG/NSTAT — the
    non-stationary analogue of the M20 /IMPL/FATIG and M24 /IMPL/FATIG/NGAUSS
    cards), exactly as M16–M24 did.

    * **NON-STATIONARY DAMAGE MODELS** (`nonstationary_fatigue.py`): (a) the
      PIECEWISE-STATIONARY / BLOCK ("mission profile") model — partition the load
      into stationary blocks (each an RMS scaling a_i of a shared PSD shape, so
      the moments scale as m_n → a_i²m_n — `block_moments` — over a duration T_i),
      run the M20 estimators PER BLOCK and Palmgren–Miner SUM the block damages
      duration-weighted, D = Σ (E[D]/T)_i T_i (`block_fatigue_summary`); because
      the RMS scaling scales every estimator's damage EXACTLY as a_i^m (the shape
      coefficients — Dirlik D1..D3/Q/R, the rates, the width factors — are
      scale-invariant), the block sum is the standard non-stationary-as-sequence-
      of-stationary answer. (b) the AMPLITUDE-MODULATED / evolutionary model —
      Priestley's separable S(ω,t) = |A(t)|²S(ω), the stationary damage INTEGRATED
      over the RMS distribution p(a): the closed-form E[a^m]-weighted damage
      (`amplitude_modulated_summary`; = the block Miner-sum for a shared shape with
      the durations as p(a)). The RMS-modulation ↔ kurtosis link (the M25↔M24
      bridge): a varying RMS makes the Gaussian carrier leptokurtic, γ₄ =
      3·E[a⁴]/E[a²]² (the Wolfsteiner–Trapp / Kihm–Rizzi relation,
      `rms_modulation_kurtosis`), and the non-stationary amplification κ_ns =
      E[a^m]/E[a²]^(m/2) (`nonstationary_amplification`) is the normalization-
      invariant counterpart of the M24 λ_ng. Validated: the non-stationary damage
      REDUCING to the M20 stationary answer for a constant modulation / single
      block (recovered EXACTLY, E[a^m] = 1); the block Miner-sum equalling the
      duration-weighted per-block damages (hand check) and the amplitude-modulated
      E[a^m] damage for a shared shape; κ_ns AGREEING with the M24 λ_ng at the
      induced kurtosis (`bridge_to_nongaussian`, reusing `nongaussian_fatigue`
      read-only — EXACTLY 1 in the constant limit both directions, to leading order
      otherwise; the (m−2)/(m−1) scale-mixture-vs-Hermite structural difference is
      documented, not hidden).
    * **NON-STATIONARY MONTE-CARLO CROSS-CHECK** (`nonstationary_fatigue.py`): a
      non-stationary time-domain synthesiser — the M20 Gaussian spectral-
      representation carrier MULTIPLIED by a time-varying RMS envelope a(t) (a
      concatenation of stationary blocks / a modulation schedule) —
      (`synthesize_nonstationary_history`), rainflow-counted (the M20 ASTM E1049
      counter) and Miner-summed (`nonstationary_monte_carlo_damage`). The
      block-boundary rainflow caveat (cycles straddling a block boundary, which the
      concatenated MC counts but the closed-form block Miner-sum does not) is
      documented. Validated: the synthesised history's time-varying RMS / sample
      kurtosis matching the target modulation (γ₄ = 3·E[a⁴]/E[a²]²); the
      non-stationary Monte-Carlo damage matching the block / modulated spectral
      estimate within the seeded scatter; the constant-modulation limit reducing to
      the M20 Gaussian Monte-Carlo BIT-IDENTICALLY (the envelope is unity, x = u).
    * **Engine sub-flag + reporting** (`random_response._run_nonstationary` /
      `_run_nonstationary_multiaxial`): /IMPL/FATIG/NSTAT (composing with /MULT,
      /NPROP, /SPEC, /NGAUSS — orthogonal, a scaling on whatever equivalent-stress
      PSD the M20–M24 path produces) recovers the (scalar OR multiaxial von-Mises /
      critical-plane) stress PSD as M20–M24 do, reads an RMS scale-vs-time
      modulation /FUNCT (the mission profile — run-up / dwell / run-down — sampled
      into blocks), evaluates the block Miner-sum + amplitude-modulated damage /
      equivalent stress / life, runs the non-stationary Monte-Carlo cross-check,
      and reports the non-stationary damage / induced kurtosis / block breakdown /
      the M25↔M24 bridge ALONGSIDE the M20 stationary numbers on
      `model.implicit_result.fatigue['nonstationary']` and in the listing (a new
      NON-STATIONARY / EVOLUTIONARY-PSD block showing the stationary and the
      non-stationary answers side by side). The modulation /FUNCT id (and optional
      block count) go on a card line AFTER the sweep / S-N (and the M24 kurtosis, if
      /NGAUSS) lines: modfunct [nseg]. A PORT sub-flag, minimal like /IMPL/FATIG.
    * **Example**: `examples/nonstationary_fatigue` — the M20 `spectral_fatigue`
      base-excited instrument stack re-run under /IMPL/FATIG/NSTAT/BASE with a
      run-up / dwell / run-down RMS mission profile (/FUNCT/20, sampled into 12
      blocks): the M20 stationary block UNCHANGED, then the NON-STATIONARY block
      with the induced kurtosis γ₄ ≈ 3.86, the M25↔M24 bridge (κ_ns ≈ 1.51 vs
      λ_ng ≈ 1.86), the block Miner-sum = the amplitude-modulated E[a^m] ≈ 4.07
      damage and the ~4× shorter mission life (the dwell at RMS × 1.5 dominates),
      plus the non-stationary Monte-Carlo cross-check.
    * **Validated** (`tests/test_m25_nstatfatig.py`): all of the above plus the
      /IMPL/FATIG/NSTAT (composing with /MULT, /NPROP, /SPEC, /NGAUSS) card mirror
      and the parity contract (the non-stationary path never mutating the M16
      eigensolver / M17–M19 transfer functions / the M20 SCALAR fatigue / the
      M21–M23 MULTIAXIAL reductions / the M24 NON-GAUSSIAN correction / the element
      state; the M20 SCALAR AND the M21/M22/M23 MULTIAXIAL AND the M24 non-Gaussian
      damage rates byte-identical whether or not /NSTAT runs; the direct /IMPL/DYNA
      answer byte-identical).

    Deferred out of M25, explicitly (not half-implemented):
    * a FULLY evolutionary NON-SEPARABLE S(ω,t) with a time-varying spectral SHAPE
      (beyond the separable |A(t)|²S(ω) / piecewise-stationary block model, where
      each block MAY carry a different shape but the amplitude-modulation is
      separable) — DEFERRED (a genuinely non-separable evolutionary spectrum is a
      different analysis);
    * the amplitude-modulated damage and the M25↔M24 bridge use the SCALE-MIXTURE
      amplitude model; the leading-order (m−2)/(m−1) structural difference from the
      M24 Hermite amplitude is DOCUMENTED, not hidden — both are principled
      kurtosis-driven amplifications, and the port validates their agreement within
      tolerance rather than claiming an identity that does not hold;
    * a full non-stationary MULTIAXIAL JOINT treatment (a time-varying stress-tensor
      cross-PSD): M25 applies the block/modulation scaling to the (von Mises /
      critical-plane) equivalent scalar the M21/M23 reductions already produce, so it
      COMPOSES with the multiaxial / non-proportional paths but does not model a
      jointly evolutionary tensor — DEFERRED;
    * MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
      fracture-mechanics fatigue, the COMPLEX-FRF stress recovery and a MULTI-INPUT
      cross-PSD with coherence — the unchanged M20–M24 tail;
    * the unchanged M10–M24 deferral tail: multi-input cross-PSD with coherence,
      multi-directional 100-30-30 response spectra, the complex-FRF base-excitation
      feed, gyroscopic / circulatory systems, Lanczos / subspace + AMLS, IFQ ≥ 10
      / MODFR 2, /FRICTION per-part-pair sets, orthotropic / thermal friction, the
      fiber TYPE18 beam, the LAW27 plastic block / solids, thermal contact,
      TYPE19/24/25, Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under
      implicit, the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

25. **M26 — FULLY EVOLUTIONARY / NON-SEPARABLE-PSD SPECTRAL FATIGUE** ✅ (done):
    the FIRST item deferred out of M25 that BUILDS on the non-stationary spectral
    machinery — the frequency-domain damage of a random-vibration response whose
    spectral SHAPE (not merely its RMS level) VARIES WITH TIME, computed by
    extending the M25 piecewise-stationary / amplitude-modulated estimators to a
    genuinely NON-SEPARABLE evolutionary spectrum S(ω,t) (a time-frequency
    "spectrogram" whose bandwidth / rates / centre frequency DRIFT with time), and
    cross-validated against a non-stationary time-domain Monte-Carlo with a
    time-varying filter. M25 lifted stationarity only in the RMS ENVELOPE
    (Priestley's SEPARABLE S(ω,t) = |A(t)|²S(ω) — a FIXED spectral shape whose
    amplitude is modulated, so the rates ν₀/ν_p and width factors α₁/α₂ stay
    fixed); M26 lifts exactly that SEPARABILITY assumption — the shape itself
    evolves, so each time-window carries its OWN complete moment set m₀..m₄, not
    just its own scaling of a shared shape. M26's non-separable spectrogram is the
    GENERAL per-window-full-PSD block, of which M25's RMS-scaled block is the
    shared-shape SPECIAL CASE (the built-in M26↔M25 reduction: a non-separable
    spectrogram whose per-window shape is CONSTANT recovers the M25
    amplitude-modulated answer EXACTLY; a single window / a time-invariant shape
    recovers the M20 stationary answer EXACTLY). A NEW, parallel path
    (`implicit/evolutionary_fatigue.py` + the `_run_evolutionary` /
    `_run_evolutionary_multiaxial` extensions of the M20/M21 drivers in
    `implicit/random_response.py`) that never touches the M10 integrator, the M16
    eigensolver, the M17/M18 superposition, the M19 PSD path, the M20 SCALAR
    fatigue, the M21 MULTIAXIAL SPECTRAL path, the M22 NON-PROPORTIONAL TIME-DOMAIN
    path, the M23 SPECTRAL NON-PROPORTIONAL path, the M24 NON-GAUSSIAN correction
    OR the M25 NON-STATIONARY correction (all stay bit-identical — the M20–M25
    answers are byte-identical whether or not /EVOL runs, asserted). Theory:
    Priestley, "Evolutionary spectra and non-stationary processes" (J. Roy.
    Statist. Soc. B 27, 1965) AND "Power spectral analysis of non-stationary random
    processes" (J. Sound Vib. 6, 1967 — the GENERAL non-separable oscillatory
    amplitude A(ω,t), of which M25 used the separable A(t)A(ω) subcase); Mark 1970
    / Hammond (non-stationary spectral analysis, the instantaneous spectrum);
    Newland (the short-time / windowed spectral method — the spectrogram — and the
    time-frequency view); Bendat & Piersol ch. 12; the Wigner–Ville / Loève
    time-frequency distribution as the continuous instantaneous-spectrum limit;
    Palmgren–Miner (the window sum); the M25 piecewise-stationary basis.

    Fortran origin: there is NONE — `engine/source/input/freimpl.F` (re-read line
    by line for M26, fetched from raw.githubusercontent.com) has no /FATIG, no
    S-N / Miner branch, and NO non-stationary / evolutionary / spectrogram /
    time-frequency machinery of any kind (the sole `PSD` token is still `IMUMPSD`,
    a MUMPS-solver flag at line 269). OpenRadioss is a time-domain crash/impact
    code — the random-vibration fatigue analysis, stationary Gaussian (M20–M23),
    stationary non-Gaussian (M24), separable non-stationary (M25) OR fully
    evolutionary / non-separable (M26), simply is not part of the open-source
    solver (the same finding M16–M25 made). So M26 ports evolutionary /
    non-separable-PSD fatigue as a clean LIBRARY capability behind a minimal PORT
    sub-flag (/IMPL/FATIG/EVOL — the evolutionary analogue of the M20 /IMPL/FATIG,
    M24 /IMPL/FATIG/NGAUSS and M25 /IMPL/FATIG/NSTAT cards), exactly as M16–M25 did.

    * **EVOLUTIONARY / NON-SEPARABLE DAMAGE MODEL** (`evolutionary_fatigue.py`):
      (a) the SPECTROGRAM / per-window-full-PSD model — partition the loading into
      time-windows, each with its OWN complete stress PSD (hence its own full
      moment set m₀..m₄ — a genuinely different bandwidth / rates / shape per
      window), run the M20 estimators PER WINDOW and Palmgren–Miner SUM the window
      damages duration-weighted (`evolutionary_fatigue_summary`, the general
      non-separable extension of the M25 block model — driven read-only through the
      M25 `block_fatigue_summary`, which already accepts per-block distinct
      moments, with the real per-window PSDs of a drifting shape). (b) the
      TIME-FREQUENCY / drifting-shape model — an evolutionary spectrum built from a
      smoothly time-varying spectral parameter (a swept centre frequency f_c(t) — a
      "chirp-like" random process — and/or a broadening bandwidth b(t)), sampled
      into windows whose full moments are recomputed FROM the drifting shape
      (`drifting_shape_spectrogram` applies a swept/broadening Gaussian window
      W_i(f) = exp(−(f−f_c,i)²/(2b_i²)) to the recovered stress PSD — the window
      commutes with the linear |H_σ|² map, so windowing the stress PSD = windowing
      the input; `gaussian_evolutionary_spectrogram` builds a self-contained
      analytic spectrogram). Validated: the evolutionary damage REDUCING to the M25
      amplitude-modulated answer for a CONSTANT-SHAPE spectrogram (only the RMS
      drifts — recovered EXACTLY, the window Miner-sum over shared-shape windows IS
      the M25 block Miner-sum); REDUCING to the M20 stationary answer for a single
      window / a time-invariant shape (EXACTLY); the window Miner-sum equalling the
      duration-weighted per-window damages (hand check); a two-window "shape-swap"
      (narrow-band → wide-band) whose Miner-sum DIFFERS from any single-shape M25
      scaling (the point of a NON-separable spectrum — the M26↔M25 boundary made
      explicit); the swept f_c drifting the per-window ν₀, the broadening b drifting
      the per-window α₂.
    * **NON-SEPARABLE MONTE-CARLO CROSS-CHECK** (time-varying filter,
      `evolutionary_fatigue.py`): a non-separable synthesiser — per-window
      spectral-representation blocks (the M20 spectral-representation history of
      each window's OWN PSD, seeded deterministically) CONCATENATED in time so the
      instantaneous spectrum tracks S(ω,t) window by window (the
      short-time-stationary approximation — a slowly time-varying filter on the
      carrier) — (`synthesize_evolutionary_history`), rainflow-counted (the M20
      ASTM E1049 counter) and Miner-summed (`evolutionary_monte_carlo_damage`). The
      window-boundary rainflow caveat (carried from M25) and the
      instantaneous-vs-windowed-spectrum distinction are documented. Validated: the
      synthesised history's short-time spectrogram (per-window RMS + zero-crossing
      rate) matching the target evolutionary spectrum (the centre-frequency drift);
      the non-separable Monte-Carlo damage matching the window spectral estimate
      within the seeded scatter; the constant-shape limit DELEGATING to the M25
      `synthesize_nonstationary_history` and reducing to the M25 non-stationary
      Monte-Carlo BIT-IDENTICALLY (and a single unit window to the M20 Gaussian
      Monte-Carlo bit-identically).
    * **Engine sub-flag + reporting** (`random_response._run_evolutionary` /
      `_run_evolutionary_multiaxial`): /IMPL/FATIG/EVOL (composing with /MULT,
      /NPROP, /SPEC, /NGAUSS, /NSTAT — a drifting-shape window on whatever
      equivalent-stress PSD the M20–M25 path produces) recovers the (scalar OR
      multiaxial von-Mises / critical-plane) stress PSD as M20–M25 do, reads a
      drifting-shape schedule (a swept centre frequency f_c0→f_c1 and broadening
      bandwidth b_0→b_1 over nwin windows) plus the RMS level schedule from the
      shared modulation /FUNCT (so /EVOL composes with /NSTAT), evaluates the
      window Miner-sum damage / equivalent stress / life PER estimator, runs the
      non-separable Monte-Carlo cross-check, and reports the evolutionary damage /
      per-window shape breakdown (α₂ / ν₀ / RMS drift) ALONGSIDE the M20 stationary
      and M25 non-stationary numbers on
      `model.implicit_result.fatigue['evolutionary']` and in the listing (a new
      FULLY EVOLUTIONARY / NON-SEPARABLE-PSD block showing the stationary,
      RMS-non-stationary and shape-evolutionary answers side by side). The
      drifting-shape line (fc0 fc1 bw0 bw1 nwin) goes on a card line AFTER the
      sweep / S-N (and the M24 kurtosis / M25 modulation, if /NGAUSS / /NSTAT)
      lines. A PORT sub-flag, minimal like /IMPL/FATIG.
    * **Example**: `examples/evolutionary_fatigue` — the M20 `spectral_fatigue`
      base-excited instrument stack re-run under /IMPL/FATIG/EVOL/NSTAT/BASE with a
      RESONANCE SWEEP (the excitation centre frequency sweeps f_c 31 → 200 Hz
      through the five modes near 31/91/142/179/199 Hz, broadening b 12 → 30,
      12 windows) on top of the M25 RMS mission profile: the M20 stationary block
      UNCHANGED (Dirlik life ~7.4×10⁵ ms), the M25 non-stationary block UNCHANGED
      (mission life ~1.8×10⁵ ms), then the M26 FULLY EVOLUTIONARY block with the
      per-window shape drift (α₂ 0.65…0.98, ν₀ 31.5…169.7 Hz — the sweep made
      explicit), the induced γ₄ ≈ 3.86 and the much LONGER shape-evolutionary life
      (Dirlik ~1.2×10⁸ ms — the sweep dwells only transiently on each mode, doing
      far less damage than the fixed excitation sitting on the worst mode forever),
      plus the non-separable Monte-Carlo cross-check (~8.0×10⁻⁹ vs the ~8.6×10⁻⁹
      window estimate).
    * **Validated** (`tests/test_m26_evolfatig.py`): all of the above plus the
      /IMPL/FATIG/EVOL (composing with /MULT, /NPROP, /SPEC, /NGAUSS, /NSTAT) card
      mirror and the parity contract (the evolutionary path never mutating the M16
      eigensolver / M17–M19 transfer functions / the M20 SCALAR fatigue / the
      M21–M23 MULTIAXIAL reductions / the M24 NON-GAUSSIAN correction / the M25
      NON-STATIONARY correction / the element state; the M20 SCALAR AND the
      M21/M24/M25 MULTIAXIAL / non-Gaussian / non-stationary damage rates
      byte-identical whether or not /EVOL runs; the direct /IMPL/DYNA answer
      byte-identical).

    Deferred out of M26, explicitly (not half-implemented):
    * a continuous WIGNER–VILLE / Loève INSTANTANEOUS-spectrum (Priestley's
      oscillatory A(ω,t)) formulation — the M26 windowed SPECTROGRAM is the
      engineering standard and the fine-window limit of the continuous
      distribution; the cross-term-laden continuous time-frequency distribution is
      DEFERRED (a different analysis);
    * the SHORT-TIME-STATIONARY approximation is used within each window (the M20
      estimators assume local stationarity): the window must be long relative to
      the carrier period and short relative to the shape drift — the standard
      spectrogram trade-off, DOCUMENTED not hidden; the window-boundary rainflow
      caveat (cycles straddling a window boundary, which the concatenated MC counts
      but the window Miner-sum does not) is DOCUMENTED (carried from M25);
    * a full non-stationary MULTIAXIAL JOINT treatment of the evolutionary stress
      TENSOR (a jointly time-varying tensor cross-PSD): M26 applies the
      drifting-shape window to the (von Mises / critical-plane) equivalent scalar
      the M21/M23 reductions already produce, so it COMPOSES with the multiaxial /
      non-proportional paths but does not model a jointly evolutionary tensor —
      DEFERRED;
    * MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
      fracture-mechanics fatigue, the COMPLEX-FRF stress recovery, a MULTI-INPUT
      cross-PSD with coherence and the full non-Gaussian MULTIAXIAL joint
      distribution — the unchanged M20–M25 tail;
    * the unchanged M10–M25 deferral tail: multi-input cross-PSD with coherence,
      multi-directional 100-30-30 response spectra, the complex-FRF base-excitation
      feed, gyroscopic / circulatory systems, Lanczos / subspace + AMLS, IFQ ≥ 10
      / MODFR 2, /FRICTION per-part-pair sets, orthotropic / thermal friction, the
      fiber TYPE18 beam, the LAW27 plastic block / solids, thermal contact,
      TYPE19/24/25, Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under
      implicit, the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

26. **M27 — FULLY NON-STATIONARY / EVOLUTIONARY MULTIAXIAL (JOINT-TENSOR)
    FATIGUE** ✅ (done): the item M25/M26 DEFERRED — the frequency-domain damage
    of a MULTIAXIAL random-vibration response whose FULL 6×6 stress-TENSOR
    cross-PSD S_σσ(ω,t) (a jointly evolutionary tensor, not just an equivalent
    scalar) VARIES WITH TIME, computed by forming the per-time-window stress-tensor
    cross-PSD, reducing it PER WINDOW with the M21/M23 critical-plane machinery
    (equivalent von Mises + max-normal / max-shear critical plane), letting the
    CRITICAL PLANE itself DRIFT window to window, and Palmgren–Miner-summing the
    per-window MULTIAXIAL damages — cross-validated against a non-stationary
    MULTIVARIATE time-domain Monte-Carlo. M25/M26 applied the block / evolutionary
    scaling to the EQUIVALENT SCALAR the M21/M23 reductions produce (a scalar
    per-window spectrogram, a FIXED reduction — one critical plane held fixed);
    M27 lifts exactly that — the JOINT tensor evolves, so the critical-plane
    orientation AND the M23 non-proportionality factor F_np are RE-SEARCHED per
    window from the window's OWN tensor cross-PSD (the plane may ROTATE, F_np may
    DRIFT), NOT held fixed. This is the natural consumer of BOTH the M21/M23
    multiaxial machinery AND the M25/M26 non-stationary / evolutionary machinery.
    A NEW, parallel path (`implicit/joint_evolutionary_fatigue.py` + the
    `_run_joint_evolutionary` extension of the M21 `_run_multiaxial` driver in
    `implicit/random_response.py`, `/IMPL/FATIG/MULT/EVOL/JOINT`) that never touches
    the M10 integrator, the M16 eigensolver, the M17/M18 superposition, the M19 PSD
    path, the M20 SCALAR fatigue, the M21 MULTIAXIAL SPECTRAL path, the M22
    NON-PROPORTIONAL TIME-DOMAIN path, the M23 SPECTRAL NON-PROPORTIONAL path, the
    M24 NON-GAUSSIAN correction, the M25 NON-STATIONARY correction OR the M26
    EVOLUTIONARY-SCALAR correction (all stay bit-identical — the M20–M26 answers
    byte-identical whether or not /JOINT runs, asserted). The BUILT-IN reductions:
    a STATIONARY tensor recovers the M21 spectral answer EXACTLY (all three
    reductions — one flat unit window gives the full-tensor moment matrices, so the
    per-window plane search IS `critical_plane_search`); the equivalent-scalar
    reduction of the evolutionary tensor with a FIXED critical plane recovers the
    M26 scalar-spectrogram answer EXACTLY (the `drift=False` branch projects onto
    the stationary plane and windows the scalar — the M26 route); a
    constant-tensor-shape / RMS-only drift recovers the M25 multiaxial answer
    EXACTLY (a flat window with only the level drifting keeps the plane fixed, so
    the window Miner-sum IS the M25 block Miner-sum). The POINT of M27 over M26 is
    the case none of these cover — a rotating-principal-axes tensor whose per-window
    critical plane genuinely DRIFTS (the physics a fixed reduction is blind to: in
    a real structure S_σσ(f) has a DIFFERENT tensor orientation near each mode, so a
    swept window rotates the critical plane), whose Miner-sum DIFFERS from the M26
    fixed-reduction scalar spectrogram.

    * **EVOLUTIONARY JOINT-TENSOR DAMAGE MODEL** (`joint_evolutionary_fatigue.py`):
      (a) the per-window TENSOR cross-PSD — the M26 drifting-shape window W_i(f)
      multiplies the full 6×6 S_σσ(ω) per window (it commutes with the reduction,
      as M26 established for |H|²), the per-window 6×6 spectral-MOMENT matrices
      M_{n,i} = a_i²·(1/π)∫ω^n W_i(f) S_σσ dω recomputed
      (`windowed_tensor_moment_matrices`), and EACH window reduced by the M21
      equivalent von Mises + M21/M23 critical-plane search — the plane / F_np
      RE-SEARCHED from the window's own tensor (`reduce_window_tensor`,
      `joint_evolutionary_fatigue_summary`; `drift=True` re-searches, `drift=False`
      is the M26 fixed-plane reduction), Miner-summed duration-weighted; (b) the
      per-window critical-plane DRIFT report (plane normal / F_np / equivalent-stress
      RMS per window) + the plane ROTATION angle + the F_np drift — the point of a
      JOINT evolutionary tensor over the M26 fixed-reduction scalar spectrogram.
    * **NON-STATIONARY MULTIVARIATE MONTE-CARLO** (`joint_evolutionary_fatigue.py`):
      per-window multivariate spectral-representation blocks of the WINDOWED tensor
      (the M21 per-bin eigen/Cholesky synthesiser with a time-varying tensor
      cross-PSD) concatenated, each window's segment projected onto that WINDOW's
      OWN critical plane (the windowed critical-plane path), ASTM E1049 rainflow +
      Miner (`synthesize_joint_evolutionary_history` /
      `joint_evolutionary_monte_carlo_damage`); the constant-shape limit DELEGATES
      to a single multivariate carrier × the piecewise RMS envelope (the M25
      multivariate lift), and a single unit window reduces to the M21 multivariate
      MC (`monte_carlo_multiaxial_damage`) BIT-IDENTICALLY. The window-boundary
      rainflow caveat (carried from M25/M26) and the rotating-vs-windowed
      critical-plane distinction are DOCUMENTED.
    * **ENGINE CARD + reporting** (`/IMPL/FATIG/MULT/EVOL/JOINT` — or `/TENSOR`,
      implying MULT + EVOL; composes with /NPROP, /SPEC, /NGAUSS, /NSTAT): recovers
      the full stress-TENSOR cross-PSD PER WINDOW, evaluates the joint-tensor window
      Miner-sum multiaxial damage / life with the per-window critical-plane search,
      runs the non-stationary multivariate Monte-Carlo, and reports the joint-tensor
      damage / per-window critical-plane drift (plane normal, F_np, RMS per window)
      + the plane rotation + F_np drift ALONGSIDE the M21 stationary, M25
      non-stationary and M26 scalar-evolutionary numbers on
      `model.implicit_result.fatigue` (a `joint_evolutionary` sub-entry) and in the
      listing — so the listing shows the STATIONARY, the RMS-NON-STATIONARY, the
      SCALAR-SHAPE-EVOLUTIONARY and the JOINT-TENSOR-EVOLUTIONARY answers side by
      side.
    * **Example** (`examples/joint_evolutionary_fatigue`): the M21 solid-brick
      cantilever (a rectangular section + an OFFSET SKEW tip load exciting bending
      AND torsion) RE-RUN under an EVOLVING TENSOR — a resonance sweep (fc 40 → 600
      Hz) drifting the excitation band through the modes so the root tensor
      orientation drifts; the listing shows a ~90° max-normal-plane ROTATION and a
      ~0.63 F_np drift, and reports the joint-tensor-evolutionary life alongside the
      M21 stationary, M25 non-stationary and M26 scalar-evolutionary lives.

    Validated (`tests/test_m27_jointevolfatig.py`): the stationary tensor / single
    window recovering the M21 spectral answer EXACTLY (all three reductions); the
    fixed critical plane recovering the M26 scalar spectrogram EXACTLY; the
    constant-tensor-shape / RMS-only drift recovering the M25 multiaxial block
    answer EXACTLY; the window Miner-sum = the duration-weighted per-window
    multiaxial damages (hand check); a rotating-principal-axes case whose per-window
    critical plane DRIFTS (>45° swing, x-face → y-face) and whose max-normal
    Miner-sum DIFFERS from the M26 fixed reduction (the M27↔M26 boundary); the
    windowed tensor moments commuting with the reduction; the multivariate MC
    tracking the swept spectrogram (per-window ν₀ rising) and matching the window
    estimate within scatter; the single-unit-window MC reducing to the M21
    multivariate MC BIT-IDENTICALLY; the card mirror (`/JOINT` / `/TENSOR` implying
    MULT + EVOL, composing with /NSTAT / /NGAUSS); and the M7 parity contract (the
    M21/M25/M26 damage rates byte-identical whether or not /JOINT runs; no mutation
    of the M16 eigensolver / element state; the direct /IMPL/DYNA answer
    byte-identical).

    Deferred out of M27, explicitly (not half-implemented):
    * a continuous WIGNER–VILLE / Loève INSTANTANEOUS-tensor-spectrum formulation
      (the matrix-valued analogue of the M26 scalar Wigner–Ville deferral, beyond
      the M27 windowed tensor spectrogram) — a different, cross-term-laden
      matrix-valued time-frequency distribution — DEFERRED;
    * the SHORT-TIME-STATIONARY approximation is used within each window (the
      M20/M21 estimators assume local stationarity with the window's tensor
      cross-PSD): the window must be long relative to the carrier period and short
      relative to the tensor drift — the standard spectrogram trade-off, DOCUMENTED;
      the window-boundary rainflow caveat and the rotating-vs-windowed
      critical-plane distinction are DOCUMENTED (carried from M25/M26);
    * a full NON-GAUSSIAN JOINT-TENSOR evolutionary distribution (a jointly
      time-varying, non-Gaussian tensor marginal): M27 is Gaussian per window and
      composes with the M24 non-Gaussian correction on the equivalent scalar, but
      does not model a non-Gaussian joint-tensor evolutionary distribution —
      DEFERRED;
    * MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
      fracture-mechanics fatigue, the COMPLEX-FRF stress recovery and a MULTI-INPUT
      cross-PSD with coherence (the rank-1 single-input tensor) — the unchanged
      M20–M26 tail;
    * the unchanged M10–M26 deferral tail: multi-input cross-PSD with coherence,
      multi-directional 100-30-30 response spectra, the complex-FRF base-excitation
      feed, gyroscopic / circulatory systems, Lanczos / subspace + AMLS, IFQ ≥ 10
      / MODFR 2, /FRICTION per-part-pair sets, orthotropic / thermal friction, the
      fiber TYPE18 beam, the LAW27 plastic block / solids, thermal contact,
      TYPE19/24/25, Inacti, Igap 2/3, LAW42 shells/Prony, IDTC 2/3, /RWALL under
      implicit, the UL hourglass memory, the NLGEOM hourglass-operator geometry
      variation, the BT4 thin-plate shear-lock / drilling floor.

27. **M28 — MULTI-INPUT / PARTIALLY-COHERENT RANDOM-VIBRATION RESPONSE &
    FATIGUE** ✅ (done): the stationary response (and stress-tensor) cross-PSD
    driven by SEVERAL simultaneous random input processes with a full Hermitian
    input cross-spectral matrix S_ff(ω), propagated through the VECTOR FRF by the
    MIMO relation and reduced by the whole M20–M27 estimator family UNCHANGED.
    Where M19/M20/M21 assumed ONE scalar random input f(t) of PSD S_ff(Ω) and
    formed the RANK-1 Hermitian cross-PSD H S_ff Hᴴ, M28 generalises S_ff to a
    NON-DIAGONAL ninput×ninput Hermitian matrix — auto-PSDs G_a(Ω) on the
    diagonal, the ordinary coherence γ_ab(Ω) ∈ [0,1] and the phase θ_ab off the
    diagonal, S_ff[a,b] = √(G_a G_b) γ_ab e^{iθ_ab} — and contracts it with a
    per-input FRF COLUMN stack H(Ω) (nf, ndof, ninput), one column per input load
    pattern, by the matrix triple products S_uu = H S_ff Hᴴ and S_σσ = H_σ S_ff
    H_σᴴ. This is the LAST item on the recurring deferral tail carried since
    M19/M20/M21 (the multi-input cross-PSD with coherence deferred in EVERY
    spectral milestone since M19). The M19/M20/M21 single-input answer is EXACTLY
    the 1×1 (diagonal / rank-1) special case of this matrix path — asserted, and
    delegated to the M21 rank-1 routines so it is bit-identical. Upstream check
    (re-run for M28): the open-source OpenRadioss engine has NO frequency-domain /
    spectral / random-vibration / cross-spectral / coherence / MIMO solver of any
    kind — `engine/source/implicit/imp_solv.F` and `input/freform.F` carry no
    `coherence` / `cross_psd` / multi-input token, and the sole `PSD` token
    anywhere is `IMUMPSD`, a MUMPS-solver flag (exactly the finding M16–M27
    recorded line by line). So M28, like every spectral milestone since M16,
    ports multi-input coherent random vibration as a clean LIBRARY capability and
    drives it with minimal PORT engine sub-flags (`/IMPL/PSD/MULTI` and
    `/IMPL/FATIG/MINPUT`). A NEW, parallel path (`implicit/multi_input_response.py`
    + `implicit/multi_input_fatigue.py` + the `_run_multi_input_fatigue` /
    `_run_multi_input_response` extensions of the M19/M21 drivers in
    `implicit/random_response.py`) that never touches the M10 integrator, the M16
    eigensolver, the M17/M18 superposition, the M19 PSD path, the M20 SCALAR
    fatigue, the M21 MULTIAXIAL SPECTRAL path OR the M22–M27 corrections (all stay
    bit-identical — the single-input answers byte-identical whether or not the
    multi-input path runs, asserted). The multi-input path is reported ALONGSIDE
    the single-input numbers (a `multi_input` sub-entry on
    `model.implicit_result.fatigue` / `.random_response`), so the listing shows
    the single-input and multi-input answers side by side.

    * **INPUT CROSS-PSD MODEL** (`multi_input_response.py`): (a) the Hermitian
      input cross-spectral matrix S_ff(Ω) (ninput×ninput per frequency,
      `input_cross_psd_matrix`) assembled from per-input auto-PSDs G_a(Ω) and a
      coherence model γ_ab(Ω) with phase θ_ab — a CONSTANT coherence table
      (`constant_coherence`) OR an optional EXPONENTIAL / decay coherence for
      spatially distributed loads (`exponential_coherence`, γ_ab(f) = exp(−decay·
      |x_a−x_b|·f/speed), the Davenport-type model) — then projected onto the
      NEAREST Hermitian PSD matrix (`nearest_psd`, clip negative eigenvalues,
      Higham 1988), a documented no-op on an already-valid matrix. (b) the
      per-input VECTOR FRF column stack H(Ω) (`stress_frf_columns` /
      `displacement_frf_columns`, one modal-superposition FRF per input pattern,
      reusing the M17/M19 machinery read-only) and the MIMO triple products
      `response_cross_psd(_diagonal)` (S_uu = U S_ff Uᴴ) and
      `stress_tensor_cross_psd_multi` (S_σσ = H_σ S_ff H_σᴴ), batched over
      frequency; the single-input (ninput = 1) case delegates to the M21 rank-1
      routines (bit-identical). (c) `multi_input_multiaxial_summary` reduces the
      multi-input S_σσ by the SAME M21 machinery (equivalent von Mises + max-
      normal / max-shear critical plane), returning the SAME dict shape so the
      M22–M27 corrections consume it UNCHANGED.
    * **MULTI-INPUT FATIGUE + MONTE-CARLO** (`multi_input_fatigue.py`): the multi-
      input S_σσ / its 6×6 moment matrices feed the M20 scalar and the M21–M27
      multiaxial reductions UNCHANGED (asserted byte-identical given the same
      S_σσ). The INDEPENDENT cross-check is a MULTI-INPUT multivariate spectral-
      representation Monte-Carlo: synthesise the ninput CORRELATED input histories
      from the per-bin eigen/Cholesky factor of S_ff (the M21 synthesiser applied
      to the INPUT matrix, `synthesize_multi_input_forces` /
      `synthesize_multi_input_stress`), drive each through its stress FRF column,
      SUM, ASTM E1049 rainflow + Miner (`monte_carlo_multi_input_damage`). The
      single-input and fully-coherent (rank-1 S_ff) cases DELEGATE to the M21
      single-input Monte-Carlo on the effective S_σσ (bit-identical); the
      partially-coherent / incoherent case runs the input-level synthesis and is
      cross-checked against the stress-level (M21-on-S_σσ) answer within scatter.
      `measure_coherence` recovers the synthesised inputs' coherence (Welch
      averaging) to confirm it matches the target γ_ab.
    * **ENGINE CARD + reporting** (`/IMPL/PSD/MULTI`, `/IMPL/FATIG/MULT/MINPUT` —
      composing with /NPROP / /SPEC / /NGAUSS / /NSTAT / /EVOL / /JOINT): reads
      the input-pattern TABLE (a header `ninput cohmodel γ θ [decay speed]` then a
      `cload_funct psd_funct [x y z]` row per input — each input names the /CLOAD
      /FUNCT id identifying its spatial pattern and the auto-PSD /FUNCT), assembles
      S_ff, recovers S_uu / S_σσ, runs the requested reductions and the multi-
      input Monte-Carlo, and reports the multi-input response RMS / cross-spectral
      / coherence diagnostics and the damage / life ALONGSIDE the single-input
      numbers (the `multi_input` sub-entry). The single-input /MULT (or scalar
      /IMPL/FATIG, or /IMPL/PSD) result is fully formed and left byte-identical.
    * **Example** (`examples/multi_input_random`): the M21 solid-brick cantilever
      RE-RUN under TWO partially-coherent random inputs — a transverse z-load
      (input 1, auto-PSD G₁ = 2) and a lateral y-load (input 2, auto-PSD
      G₂ = 1.5), coherence γ = 0.5 — reporting its multi-input life alongside its
      single-input life, with the two Monte-Carlo estimates (stress-level /
      input-level) agreeing within scatter.

    Validated (`tests/test_m28_multiinput.py`): the single-input (ninput = 1)
    cross-PSD BIT-IDENTICAL to the M21 rank-1 cross-PSD and the M19 |U|²G response
    PSD; the assembled S_ff Hermitian and obeying the coherence identity
    |S_ff[a,b]|² = γ_ab² G_a G_b; the Hermitian-PSD projection a NO-OP on an
    already-valid matrix (and clipping a constructed non-PSD one); a DIAGONAL
    (incoherent) S_ff giving the SUM of the per-input single-input cross-PSDs
    EXACTLY; a RANK-1 fully-coherent S_ff reducing to the single-input answer for
    the effective combined pattern EXACTLY; partial coherence interpolating
    MONOTONICALLY between the two; a two-input SDOF response cross-PSD matching the
    hand-derived closed form S_uu = |H₁|²G₁ + |H₂|²G₂ + 2Re(H₁H̄₂ S_ff,12); the
    M21 reductions byte-identical given the same S_σσ; the single-input /
    fully-coherent Monte-Carlo reducing to the M21 single-input MC BIT-IDENTICALLY
    (delegation); the incoherent projected variance being the SUM of the per-input
    projected variances (exact) and the input-level MC matching the stress-level
    MC within scatter; the synthesised inputs' MEASURED coherence matching the
    target γ_ab; the card mirror (`/MULT/MINPUT` implying MULT, `/PSD/MULTI`, the
    constant + exponential coherence models, composing with /NGAUSS); the solid-
    brick end to end (multi-input life alongside single-input); and the M7 parity
    contract (the single-input M19/M20/M21 reductions AND the M22–M27 corrections
    byte-identical whether or not the multi-input path runs; the recovery
    read-only; the direct /IMPL/DYNA answer byte-identical).

    Deferred out of M28, explicitly (not half-implemented):
    * a fully NON-STATIONARY / EVOLUTIONARY MULTI-INPUT cross-PSD (a
      TIME-VARYING input coherence matrix S_ff(ω, t)): M28 is STATIONARY multi-
      input, and composes with the M25/M26/M27 corrections by modulating the
      stationary multi-input S_σσ with a scalar RMS mission profile / drifting-
      shape window — but the COHERENCE itself stays stationary; a jointly
      time-varying coherence is a different object — DEFERRED;
    * frequency-dependent coherence BEYOND the supported constant + exponential/
      decay models (a measured γ_ab(f) table per pair, or a propagation-phase
      θ_ab(f) = 2πf d_ab/speed beyond the constant lag) — DEFERRED (the library
      accepts a full (nf, n, n) coherence stack, but the card exposes only the two
      built-in models);
    * multi-directional 100-30-30 (multi-component) response spectra — a
      response-SPECTRUM (peak-combination) multi-input analogue, distinct from the
      PSD cross-spectral path — DEFERRED (carried from M19);
    * the base-acceleration MULTI-INPUT feed (per-direction participation column
      stack) — M28 supports FORCE-pattern multi-input; the base-input columns are
      DEFERRED (the /MINPUT/BASE card warns and falls back);
    * MEAN-STRESS beyond the basic M20/M21 Goodman intercept, CRACK-GROWTH /
      fracture-mechanics fatigue, the COMPLEX-FRF stress recovery and the complex-
      FRF base-excitation feed — the unchanged M20–M27 tail;
    * the unchanged M10–M27 deferral tail: multi-directional 100-30-30 response
      spectra, the continuous Wigner–Ville / Loève instantaneous-tensor spectrum,
      the full non-Gaussian joint-tensor evolutionary distribution, non-
      proportional hardening, gyroscopic / circulatory systems, Lanczos /
      subspace + AMLS, IFQ ≥ 10 / MODFR 2, /FRICTION per-part-pair sets,
      orthotropic / thermal friction, the fiber TYPE18 beam, the LAW27 plastic
      block / solids, thermal contact, TYPE19/24/25, Inacti, Igap 2/3, LAW42
      shells/Prony, IDTC 2/3, /RWALL under implicit, the UL hourglass memory, the
      NLGEOM hourglass-operator geometry variation, the BT4 thin-plate shear-lock
      / drilling floor.

## 6. Validation strategy

`tests/` contains two layers:

* **Unit tests** — parser round-trips, single-element material/element checks
  (e.g. one brick under uniaxial strain must return the analytic stress).
* **Analytic validations** — full starter+engine runs compared to closed-form
  results: longitudinal wave speed in a bar, cantilever/plate vibration
  frequency, Johnson–Cook uniaxial yield curve, energy conservation of a
  block bouncing on a rigid wall.
* **Random / spectral (PSD) & response-spectrum validations (M19)** — the
  white-noise SDOF variance against the closed form σ² = S₀/(2ck) (the
  Lorentzian ∫|H|²dΩ = π/(ck)); the multi-DOF response PSD from the modal
  FRF matching the DIRECT (K−Ω²M+iΩC)⁻¹ inversion (the |H|² S law is the
  same whichever transfer function feeds it); the spectral-moment / Parseval
  identity m₀(velocity PSD) = m₂(displacement PSD); the RMS reducing to the
  static σ_f/k for a quasi-static band; the mean zero-crossing rate ≈ the
  natural frequency for a narrow-band SDOF; the CQC correlation ρᵢⱼ against
  the Der Kiureghian equal-damping closed form (unit diagonal, →0 well-
  separated, →1 for r→1); a single-mode spectrum recovering Γ Sa/ω²; SRSS ≈
  CQC for well-separated modes and the CQC-vs-SRSS GAP on a closely-spaced
  (near-degenerate tuning-fork) pair; the complex-FRF PSD reducing to the
  real-FRF PSD when damping is classical; the card mirror and the parity
  contract (no mutation of the M16-M18 solvers / element state; the direct
  /IMPL/DYNA answer byte-identical whether or not the random path runs).
* **Random-vibration (spectral) fatigue validations (M20)** — the recovered
  STATIC stress equal to the M8 implicit-static stress (= the P/A closed form),
  the stress FRF H_σ(Ω) = Σᵢσᵢqᵢ(Ω) equal to the direct element-stress operator
  applied to U(Ω) (a |H_σ|² S stress-PSD identity), and the recovery leaving
  the element state untouched (read-only); the narrow-band (Bendat) closed form
  against a hand Gamma-function evaluation; Dirlik reducing to narrow band as
  the bandwidth → 0 (D1,D2 → 0, D3 → 1) and giving the correct wide-band bias
  (less conservative than narrow band) on a bimodal spectrum; the ASTM
  E1049-85 rainflow counter on its canonical worked example; a Gaussian time
  history synthesised from the PSD (RMS = √m₀) rainflow-counted and matching
  the Dirlik estimate within the documented scatter (a seeded Monte-Carlo
  cross-check); Wirsching–Light / Tovo–Benasciutti reducing to narrow band in
  the narrow-band limit; the card mirror (/IMPL/FATIG + /BASE + /STRS) and the
  parity contract (no mutation of the M16-M19 solvers / element state; the
  direct /IMPL/DYNA answer byte-identical whether or not the fatigue path runs).
* **Multiaxial / critical-plane spectral fatigue validations (M21)** — for a
  UNIAXIAL stress state the 6×6 stress-tensor cross-PSD collapsing to the M20
  scalar σₓₓ channel (the equivalent-von-Mises PSD = |H_xx|² S_ff) and its
  DIAGONAL equal to the M20 per-component channel PSDs, Hermitian; the von Mises
  operator Q on the textbook invariants (uniaxial → 1, pure shear → 3,
  hydrostatic → 0); the trace / quadratic-operator identity trace(Q Mₙ) = the
  moments of the direct S_vm = HᴴQH PSD; the MAX-NORMAL-stress critical plane
  recovering the 45° principal plane and the MAX-SHEAR-stress critical plane the
  coordinate plane for a pure-shear state, both at the shear amplitude; the
  projection-by-direction moments p^T Mₙ p equal to the |H·p|² S scalar
  moments; the MULTIVARIATE spectral-representation synthesis (per-bin
  eigendecomposition / Cholesky of the cross-PSD) reproducing the full
  covariance matrix (component variances AND cross-covariances = M₀); the
  seeded Monte-Carlo critical-plane rainflow damage matching the spectral Dirlik
  estimate within the documented scatter; the /IMPL/FATIG/MULT (+ /BASE) card
  mirror and the parity contract (no mutation of the M16-M20 solvers / the M20
  SCALAR fatigue / element state; the direct /IMPL/DYNA answer byte-identical
  whether or not the multiaxial path runs).
* **Non-proportional multiaxial fatigue validations (M22)** — for a PROPORTIONAL
  (in-phase) LINE shear path the three amplitude operators (Papadopoulos MCC /
  longest chord / Mamiya MRH) collapsing to the scalar M21 amplitude with the
  non-proportionality factor F_np = 0; for a CIRCULAR (90°-out-of-phase,
  equal-amplitude) path the MCC radius = the component amplitude, the longest
  chord = the diameter (2r), the MRH = r√2 and F_np = 1 (the closed forms); an
  ellipse's MRH = √(p²+q²), MCC = the major semi-axis, F_np = q/p; the MCC on an
  off-centre circle recovering the radius AND centre (the amplitude
  mean-independent); for PROPORTIONAL loading the time-domain path-counting
  (shear-path) damage on the critical plane reducing to the M21 max-shear
  critical-plane rainflow within the seeded Monte-Carlo scatter (g = 1, F_np ≈ 0
  — no spurious non-proportional damage on a proportional path); for a
  90°-OUT-OF-PHASE (rotating) case the non-proportional path-counting damage
  HIGHER than the scalar projection (the M21 spectral method's linear projection)
  at the same channel amplitudes by g^m ≈ 2^(m/2) with F_np ≈ 1 (the extra damage
  the projection misses), and the line-vs-circle 2^(m/2) closed form; a hand check
  of the MCC / MRH / Findley (τ_a + k σ_n,max) / Fatemi-Socie (γ_a(1 + k
  σ_n,max/σ_y)) damage on a synthetic two-channel sinusoid with a constant normal
  stress; the /IMPL/FATIG/MULT/NPROP card mirror (a PORT sub-flag composing in any
  order, implying /MULT) and the parity contract (no mutation of the M16-M20
  solvers / the M21 SPECTRAL reductions — damage rates AND critical-plane
  orientations byte-identical whether or not /NPROP runs — / element state; the
  direct /IMPL/DYNA answer byte-identical).
* **Spectral non-proportional multiaxial fatigue validations (M23)** — the 2×2
  in-plane shear cross-spectral moment matrix (from M₀) EQUALLING the M22
  TIME-DOMAIN shear-path covariance (np.cov of the synthesised path — the M₀ =
  E[σσᵀ] identity); the SPECTRAL F_np EQUALLING the M22 TIME-DOMAIN F_np on the
  same cross-PSD; F_np = 0 on EVERY plane for a PROPORTIONAL (rank-1) state and
  F_np = 1 for the equal-amplitude 90°-out-of-phase circle (closed forms); the
  spectral resolved-shear / resolved-normal amplitudes (from p^T Mₙ p) matching
  the RMS of the M22 synthesised histories; a two-channel-cross-PSD hand check of
  F_np = 1/a (the aspect ratio), the path factor g = √(1+F_np²) and the effective
  shear amplitude; for PROPORTIONAL loading the spectral non-proportional damage
  reducing EXACTLY to the M21 max-shear critical-plane spectral answer (F_np = 0,
  no correction); for a 90°-OUT-OF-PHASE case the spectral non-proportional damage
  (1+F_np²)^(m/2) = 2^(m/2) HIGHER than the uncorrected M21 projected-scalar answer
  AND AGREEING with the M22 TIME-DOMAIN path-counting damage within the seeded
  Monte-Carlo scatter (the two non-proportional methods converge); the
  /IMPL/FATIG/MULT/NPROP/SPEC card mirror (a PORT sub-flag composing in any order,
  implying /NPROP hence /MULT) and the parity contract (no mutation of the M16-M20
  solvers / the M21 SPECTRAL reductions / the M22 TIME-DOMAIN path count — the M21
  damage rates AND the M22 damage rates + critical-plane orientations byte-identical
  whether or not /SPEC runs — / element state; the direct /IMPL/DYNA answer
  byte-identical).
* **Non-Gaussian / kurtosis spectral fatigue validations (M24)** — the Winterstein
  Hermite transform g(u) = κ[u + h₃(u²−1) + h₄(u³−3u)] being the IDENTITY for the
  Gaussian target (γ₄ = 3, γ₃ = 0) and preserving the mean (0) and variance (κ
  makes Var(g) = 1 EXACTLY, the Hermite-orthogonality identity — checked
  analytically AND on a large sample) while hitting the target kurtosis to the
  Winterstein-fit accuracy (an exact-moment hand check via `hermite_kurtosis` + a
  large-sample check); the non-Gaussian correction factor λ_ng = E[g(V)ᵐ]/E[Vᵐ]
  being EXACTLY 1 for a Gaussian process (the M20 answer recovered to machine
  precision), > 1 for a LEPTOKURTIC (γ₄ > 3, spiky) process and < 1 for a
  PLATYKURTIC (γ₄ < 3) one (the closed-form Rayleigh-amplitude integral), monotone
  in γ₄ and the S-N slope m; the Benasciutti–Tovo bandwidth attenuation moving
  λ_ng toward 1 for a wide band and leaving it unchanged for a narrow band; the
  corrected damage of each estimator being EXACTLY λ_ng × the Gaussian damage; on a
  NARROW-band process the non-Gaussian Monte-Carlo (Gaussian history → Hermite
  transform → ASTM E1049 rainflow → Miner) hitting the target sample kurtosis and
  matching the λ_ng-corrected spectral (Dirlik) estimate within the seeded scatter;
  the Gaussian limit (γ₄ = 3) reducing the non-Gaussian Monte-Carlo EXACTLY
  (bit-identical) to the M20 Gaussian Monte-Carlo; the /IMPL/FATIG/NGAUSS card
  mirror (a PORT sub-flag with the kurtosis / skewness on card line 3, composing
  with /MULT, /NPROP, /SPEC) and the parity contract (no mutation of the M16-M19
  solvers / the M20 SCALAR fatigue / the M21-M23 MULTIAXIAL reductions / element
  state — the M20 SCALAR AND the M21/M22/M23 MULTIAXIAL damage rates byte-identical
  whether or not /NGAUSS runs — ; the direct /IMPL/DYNA answer byte-identical).
* **Non-stationary / evolutionary-PSD spectral fatigue validations (M25)** — a
  CONSTANT modulation (any level) inducing kurtosis EXACTLY 3 and amplification
  EXACTLY 1 (the stationary limit — short-circuited to defeat float round-off) and
  a single unit block / constant modulation recovering the M20 stationary answer
  EXACTLY (E[a^m] = 1) for every estimator; any VARYING modulation being
  leptokurtic (γ₄ = 3·E[a⁴]/E[a²]² > 3, Jensen) and amplifying (κ_ns > 1), both
  monotone in the modulation depth and the S-N slope m; the block Miner-sum
  EQUALLING the duration-weighted per-block damages (a hand check) and — for a
  SHARED spectral shape — EQUALLING the amplitude-modulated E[a^m]-weighted damage
  EXACTLY (the block moments scaling as a², the rates / width factors invariant);
  the amplitude-modulated damage of each estimator being EXACTLY E[a^m] × the
  stationary damage; the M25↔M24 BRIDGE — κ_ns = E[a^m]/E[a²]^(m/2) and the M24
  λ_ng at the induced kurtosis being EXACTLY 1 in the constant limit (both
  directions) and agreeing within tolerance for mild kurtosis (differing only by
  the documented (m−2)/(m−1) scale-mixture-vs-Hermite factor), both > 1 and
  growing together with the kurtosis; the non-stationary Monte-Carlo (the M20
  Gaussian carrier × a time-varying RMS envelope → ASTM E1049 rainflow → Miner)
  hitting the induced sample kurtosis and the per-block schedule RMS, matching the
  block / modulated spectral estimate within the seeded scatter (the block-
  boundary rainflow caveat documented), and reducing EXACTLY (bit-identical) to
  the M20 Gaussian Monte-Carlo in the constant-modulation limit; the
  /IMPL/FATIG/NSTAT card mirror (a PORT sub-flag with the modulation /FUNCT id +
  block count on the card line after the sweep / S-N / kurtosis lines, composing
  with /MULT, /NPROP, /SPEC, /NGAUSS) and the parity contract (no mutation of the
  M16-M19 solvers / the M20 SCALAR fatigue / the M21-M23 MULTIAXIAL reductions /
  the M24 NON-GAUSSIAN correction / element state — the M20-M24 damage rates
  byte-identical whether or not /NSTAT runs; the direct /IMPL/DYNA answer
  byte-identical).
* **Fully evolutionary / non-separable-PSD spectral fatigue validations (M26)** —
  a SINGLE window / a time-invariant shape recovering the M20 stationary answer
  EXACTLY (the window Miner-sum of one block) and a CONSTANT-SHAPE spectrogram
  (a flat window, only the RMS level drifting) recovering the M25
  amplitude-modulated answer EXACTLY (the window Miner-sum over shared-shape
  windows IS the M25 block Miner-sum — the built-in M26↔M25 reduction); the window
  Miner-sum EQUALLING the duration-weighted per-window damages with genuinely
  DIFFERENT per-window shapes (a hand check); a two-window narrow→wide SHAPE-SWAP
  whose Miner-sum DIFFERS from any single-shape M25 scaling and lies BETWEEN the
  two single-shape rates (the non-separable point — the per-window α₂ genuinely
  differing, ~1 narrow, <0.9 wide); a swept centre frequency drifting the
  per-window ν₀ UP (a chirp) and a broadening bandwidth drifting the per-window α₂
  toward 0 (narrow→wide); the non-separable Monte-Carlo (per-window
  spectral-representation blocks concatenated — a time-varying filter → ASTM E1049
  rainflow → Miner) whose short-time spectrogram (per-window RMS + zero-crossing
  rate) tracks the swept centre frequency (ν₀ rising through the record), matching
  the window spectral estimate within the seeded scatter (the window-boundary
  rainflow caveat documented), reducing EXACTLY (bit-identical) to the M25
  non-stationary Monte-Carlo in the constant-shape limit (the synthesiser
  delegating to the M25 single-carrier-times-envelope path) and to the M20
  Gaussian Monte-Carlo for a single unit window; the /IMPL/FATIG/EVOL card mirror
  (a PORT sub-flag with the drifting-shape schedule fc0 fc1 bw0 bw1 nwin on the
  card line after the sweep / S-N / kurtosis / modulation lines, composing with
  /MULT, /NPROP, /SPEC, /NGAUSS, /NSTAT) and the parity contract (no mutation of
  the M16-M19 solvers / the M20 SCALAR fatigue / the M21-M23 MULTIAXIAL reductions
  / the M24 NON-GAUSSIAN correction / the M25 NON-STATIONARY correction / element
  state — the M20-M25 damage rates byte-identical whether or not /EVOL runs; the
  direct /IMPL/DYNA answer byte-identical).
* **Fully non-stationary / evolutionary multiaxial (joint-tensor) fatigue
  validations (M27)** — a STATIONARY tensor / SINGLE flat window recovering the M21
  spectral multiaxial answer EXACTLY (all three reductions: the single-window plane
  search IS `critical_plane_search`, the vM moments ARE `equivalent_vonmises_
  moments`); the equivalent-scalar reduction with a FIXED critical plane recovering
  the M26 scalar per-window spectrogram EXACTLY (the `drift=False` branch projects
  onto the stationary plane and windows the scalar); a CONSTANT-tensor-shape /
  RMS-only drift recovering the M25 multiaxial block Miner-sum EXACTLY (the plane
  does not rotate); the window Miner-sum EQUALLING the duration-weighted per-window
  MULTIAXIAL damages (a hand check on a genuinely drifting tensor); a
  rotating-principal-axes case whose per-window critical plane genuinely DRIFTS
  (>45° swing, the max-normal normal x-face → y-face) and whose Miner-sum DIFFERS
  from the M26 fixed reduction (the M27↔M26 boundary made explicit — while the vM
  reduction, a fixed quadratic, does NOT drift); the windowed tensor moments
  COMMUTING with the reduction (`windowed_tensor_moment_matrices` =
  `tensor_moment_matrices` of the windowed tensor); the non-stationary MULTIVARIATE
  Monte-Carlo (per-window multivariate spectral-representation blocks of the
  windowed tensor → per-window critical-plane projection → ASTM E1049 rainflow →
  Miner) tracking the swept spectrogram (per-window ν₀ rising through the record),
  matching the joint-tensor window estimate within the seeded scatter, and reducing
  BIT-IDENTICALLY to the M21 multivariate Monte-Carlo for a single unit window; the
  /IMPL/FATIG/MULT/EVOL/JOINT card mirror (a PORT sub-flag — /JOINT or /TENSOR
  implying MULT + EVOL, reusing the M26 drifting-shape schedule, composing with
  /NPROP, /SPEC, /NGAUSS, /NSTAT) and the parity contract (no mutation of the
  M16-M19 solvers / the M20 SCALAR fatigue / the M21-M23 MULTIAXIAL reductions / the
  M24 NON-GAUSSIAN correction / the M25 NON-STATIONARY correction / the M26
  EVOLUTIONARY-SCALAR correction / element state — the M20-M26 damage rates
  byte-identical whether or not /JOINT runs; the direct /IMPL/DYNA answer
  byte-identical).
* **Multi-input / partially-coherent random-vibration validations (M28)** — the
  single-input (ninput = 1) stress-tensor cross-PSD BIT-IDENTICAL to the M21
  rank-1 cross-PSD and the M19 |U|²G response PSD (the diagonal / rank-1 special
  case); the assembled input cross-PSD S_ff Hermitian and obeying the coherence
  identity |S_ff[a,b]|² = γ_ab² G_a G_b, with its diagonal exactly the auto-PSDs;
  the nearest-Hermitian-PSD projection a NO-OP on an already-valid matrix (and
  clipping a constructed non-PSD one, negative eigenvalue → zero); a DIAGONAL
  (incoherent) S_ff giving the SUM of the per-input single-input cross-PSDs
  EXACTLY; a RANK-1 fully-coherent S_ff reducing to the single-input answer for
  the effective combined pattern H₁+H₂ EXACTLY; partial coherence interpolating
  MONOTONICALLY between the incoherent SUM and the coherent combination; a
  two-input SDOF response cross-PSD matching the hand-derived closed form S_uu =
  |H₁|²G₁ + |H₂|²G₂ + 2Re(H₁H̄₂ S_ff,12); the M21 reductions byte-identical given
  the same multi-input S_σσ (the multi-input summary of a rank-1 S_σσ = the M21
  single-input summary, same M_n matrices); the single-input / fully-coherent
  Monte-Carlo DELEGATING to the M21 single-input MC BIT-IDENTICALLY; the
  incoherent projected variance being the SUM of the per-input projected
  variances (exact) and the input-level MC matching the stress-level
  (M21-on-S_σσ) MC within scatter; the synthesised inputs' MEASURED coherence
  (Welch-averaged) matching the target γ_ab; the card mirror (/IMPL/FATIG/MULT/
  MINPUT implying MULT, /IMPL/PSD/MULTI, the constant + exponential coherence
  models, composing with /NGAUSS); the solid-brick cantilever under two
  partially-coherent inputs end to end (the multi-input life alongside the
  single-input life, both MC formulations agreeing); and the M7 parity contract
  (the single-input M19/M20/M21 reductions AND the M22-M27 corrections
  byte-identical whether or not the multi-input path runs; the recovery
  read-only; the direct /IMPL/DYNA answer byte-identical).
* **Friction-model / material-tangent validations (M15)** — exact MFROT
  formula, branch-joint and floor checks with the FD-verified static
  µ′(p); kernel-level transmitted-force closed forms (every factor
  recomputed independently) and the Renard µ(v) curve; the exact
  discrete IFQ first-order step response; the implicit µ(p) stick/slip
  closed forms, FD tangent consistency with the µ′(p) coupling block,
  and the cross-solver steady-sliding identity at matched pressure;
  LAW27 pre-crack = LAW1 exact, per-branch law-level FD, crack-closure
  stiffness recovery, and the notched-strip implicit-vs-explicit crack
  pattern; the beam hinge at exactly the resultant limit load with
  quadratic tails on the JC curve, the measured iterated-return
  decision, and the monkeypatch-asserted bit-identity of every
  switched-off path.
* **Implicit generality validations (M14)** — the two-body rigid chain
  lever and /MPC-on-slave closed forms (exact, one Newton step), the
  chained pendulum on the elliptic-integral period with exact arm
  lengths, the circular-chain / conflict / explicit-engine refusals; the
  TYPE11 friction stick and slip closed forms (exact), FD consistency in
  both regimes, µ = 0 preservation, and the explicit kinetic law
  evaluated on the implicit converged slip state; the snap-catch (arc
  length + contact: free-path peak as M9, arrested closed form on the
  stop) and the RBE2-loaded snap (arc length + constraints, both limit
  points); the RBE2-capped and contact-stopped Euler columns; the LAW42
  Truesdell FD identity (including coalescent stretches), uniaxial/
  equibiaxial closed forms with quadratic tails, the explicit
  quasi-static cross-check, the incompressible 1/K scaling and the
  frozen-frame refusal; the parity contract.
* **Implicit friction / TYPE11 / follower-load / LAW36 validations
  (M13)** — the friction stick (transmitted shear + micro-slip
  F/(4K_t)) and slip (exactly µN) closed forms with the inclined-load
  stick/slip transition (above the cone: loud failure — no static
  equilibrium exists), FD stick/slip tangent consistency, the
  frictionless closed form preserved at µ > 0 under pure normal load,
  the explicit-kinetic cross-check in steady sliding; the TYPE11
  crossed-edges 1-D closed form, all-direction FD exactness of the
  edge-edge curvature (+ symmetry), and the quasi-static edge_impact
  geometry against the explicit damped steady state; the /PLOAD
  load-stiffness FD exactness, the iteration-count drop at an identical
  answer, and the small-pressure beam-theory limit; the LAW36 uniaxial
  pulls landing exactly on the tabulated curve with quadratic tails and
  the rate-family truncation warning; the parity contract and the
  refusal battery.
* **Implicit constraints & contact validations (M12)** — the RBE2
  rigid-lever and /RBODY-pivot closed forms (exact), the condensed
  rigid-body 6×6 mass at the master against the parallel-axis block, the
  /MPC equality split and /RBE3 dual lever rule (exact), the implicit-
  dynamic physical pendulum against the elliptic-integral period, the
  spot-weld lap joint implicit-vs-explicit cross-check, the two-block
  series-springs contact closed form (exact, active set entering
  mid-run), FD residual/tangent consistency at an active contact state,
  the implicit punch against the explicit damped steady state, the
  no-shared-mutation parity assertion and the loud-refusal battery.
* **Implicit completeness validations (M11)** — finite-difference
  residual/tangent consistency for every new element tangent, the tetra and
  sh3n membrane patch tests (exact), the thick-sh3n and beam cantilevers
  against their Timoshenko closed forms, the plane-stress/truss Johnson–Cook
  closed forms with quadratic Newton tails, the damped SDOF against the
  exponential envelope + damped period with the dissipation ledger closing
  the balance at round-off, the elastica step-cut/step-grow pair for the
  automatic dt control, and the Euler column through the /IMPL/BUCKL card.
* **Implicit dynamics validations (M10)** — the SDOF free vibration against
  Newmark's closed-form period dispersion (ω dt)²/12 (quantitatively, with
  the 4× error drop when dt halves), unconditional stability far beyond the
  explicit critical step (leapfrog divergence asserted on the same discrete
  system), trapezoidal energy conservation vs HHT high-mode dissipation on
  a bar, the quasi-static limit reproducing the implicit-static answer, a
  transient cross-checked against the explicit solver at the response peak,
  and the large-rotation pendulum against the elliptic-integral period.
* **Implicit nonlinear-geometry validations (M9)** — Euler buckling
  (shell column vs the continuum formula; hexa column vs Euler's relation
  with the mesh's measured EI), the elastica large-deflection cantilever,
  the von Mises truss snap-through traced through both limit points by arc
  length against its closed form, finite-difference residual/tangent
  consistency, and the small-strain limit reproducing the M8 answers.
* **Implicit validations (M8)** — the same philosophy for the Newton solver:
  a single-element uniaxial pull reproducing Hooke's law exactly (with
  one-step, i.e. quadratic, convergence), a multi-element constant-stress
  patch test, a shell cantilever tip deflection against Euler–Bernoulli beam
  theory, and a LAW2 monotonic pull matching both the closed-form Johnson–Cook
  curve and the explicit solver driven quasi-statically — the last also
  asserting the QUADRATIC Newton convergence that only the consistent
  (algorithmic) elastoplastic tangent delivers. Cross-checks: the reaction
  (equilibrium) balance and the strain-energy balance close to round-off.

When porting new features, always add at least one analytic validation — this
is how the port stays trustworthy without bit-for-bit comparison against the
Fortran solver (which the different output formats make impractical).
