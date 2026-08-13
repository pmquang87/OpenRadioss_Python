# OpenRadioss LAW24 (Concrete with Reinforcement) Steel Reinforcement (ARM1, ARM2, ARM3) Technical Report

A detailed analysis of the Fortran source code for **LAW24 (MAT24)** under `C:\OpenRadioss\source\OpenRadioss-latest-20260520\engine\source\materials\mat\mat024\` (`carm24.F`, `gloa24.F`, `conc24.F`, `m24law.F`, `rotloc.F`) and starter (`hm_read_mat24.F`) was conducted. Below are the findings for each of your three questions.

---

### 1. Mathematical Formulas and Constitutive Behavior of Steel Reinforcement

Each reinforcement direction $k \in \{1, 2, 3\}$ is modeled as an **independent 1D elastoplastic bar with linear isotropic hardening** aligned along the local reinforcement axes.

#### Input Material Parameters (`hm_read_mat24.F`)
- `YMS` ($E_s$): Young's modulus of steel (`PM(50)`)
- `Y0S` ($\sigma_{y,0}$): Initial yield stress of steel (`PM(51)`)
- `ETS` ($E_{ts}$): Tangent modulus of steel (`PM(52)`)
- `ARM1`, `ARM2`, `ARM3` ($\eta_1, \eta_2, \eta_3$): Volume fraction / reinforcement ratio in directions 1, 2, 3 (`PM(53..55)`)

#### Hardening Modulus Calculation (`carm24.F`)
The plastic hardening modulus $H_s$ is derived from $E_s$ and $E_{ts}$:
$$H_s = \frac{E_s \cdot E_{ts}}{\max(E_s - E_{ts}, 10^{-20})}$$

#### Incremental Return Mapping Algorithm (`CARM24`)
For each direction $k \in \{1, 2, 3\}$:
1. **Elastic Trial Stress**:
   $$S_k^{trial} = \sigma_{A,k}^{n} + E_s \cdot \Delta\epsilon_k$$
2. **Current Yield Stress (Isotropic Hardening)**:
   $$\sigma_{y,k}^{n} = \sigma_{y,0} + H_s \cdot |\epsilon_{p,k}^{n}|$$
3. **Yield Condition Check**:
   $$\text{SCLE}_k = \frac{1}{2} + \frac{1}{2} \operatorname{sign}\left(|S_k^{trial}| - \sigma_{y,k}^{n}\right) \quad (\text{0 if elastic, 1 if plastic})$$
4. **Plastic Strain Increment ($\Delta\epsilon_{p,k}$)**:
   $$\text{SCAL}_k = \frac{|S_k^{trial} - \operatorname{sign}(S_k^{trial})\,\sigma_{y,k}^{n}|}{\max(|E_s \Delta\epsilon_k|, 10^{-20})}$$
   $$\Delta\epsilon_{p,k} = \text{SCLE}_k \cdot \text{SCAL}_k \cdot \left(1 - \frac{E_{ts}}{E_s + 10^{-10}}\right) \cdot \Delta\epsilon_k$$
   $$\epsilon_{p,k}^{n+1} = \epsilon_{p,k}^{n} + \Delta\epsilon_{p,k}$$
5. **Updated Stress**:
   $$\sigma_{y,k}^{n+1} = \sigma_{y,k}^{n} + H_s \cdot \Delta\epsilon_{p,k}$$
   $$\sigma_{A,k}^{n+1} = (1 - \text{SCLE}_k) \, S_k^{trial} + \text{SCLE}_k \cdot \operatorname{sign}(S_k^{trial}) \, \sigma_{y,k}^{n+1}$$

---

### 2. Concrete-Steel Coupling Mechanism

The coupling follows a **Rule-of-Mixtures (Voigt parallel spring model)** with strain compatibility in the local reinforcement frame $\mathbf{R}$.

#### A. Transverse Strain Kinematic Reduction (`GLOA24`)
Before computing steel stresses, the effective reinforcement ratios $\eta_k$ are dynamically adjusted based on lateral strains to account for cross-sectional area changes:
$$\eta_1^{eff} = \eta_1 \cdot \max(0, 1 - \epsilon_{22} - \epsilon_{33})$$
$$\eta_2^{eff} = \eta_2 \cdot \max(0, 1 - \epsilon_{11} - \epsilon_{33})$$
$$\eta_3^{eff} = \eta_3 \cdot \max(0, 1 - \epsilon_{11} - \epsilon_{22})$$

#### B. Stress Combination (`CONC24`)
In the local orthotropic reinforcement frame, normal stresses are combined linearly according to the effective volume fraction $\eta_k^{eff}$:
$$\sigma_{11}^{total} = (1 - \eta_1^{eff}) \cdot \sigma_{concrete,11} + \eta_1^{eff} \cdot \sigma_{steel,1}$$
$$\sigma_{22}^{total} = (1 - \eta_2^{eff}) \cdot \sigma_{concrete,22} + \eta_2^{eff} \cdot \sigma_{steel,2}$$
$$\sigma_{33}^{total} = (1 - \eta_3^{eff}) \cdot \sigma_{concrete,33} + \eta_3^{eff} \cdot \sigma_{steel,3}$$

Shear stress components are unaffected by the reinforcement:
$$\sigma_{ij}^{total} = \sigma_{concrete,ij} \quad \text{for } i \neq j$$

Finally, the total stress tensor $\boldsymbol{\sigma}^{total}$ is rotated back to the global coordinate system (`AGLO24`):
$$\boldsymbol{\sigma}_{global} = \mathbf{R} \, \boldsymbol{\sigma}^{total} \, \mathbf{R}^T$$

---

### 3. Required State Variables for Steel Reinforcement

Yes, LAW24 allocates **6 additional state history variables** specifically for the reinforcement steel (3 per integration point direction):

- `SIGA(NEL, 3)` (`L_SIGA = 3` in `hm_read_mat24.F`): Current 1D normal stress $\sigma_{A,k}$ for steel in directions 1, 2, and 3.
- `EPXA(NEL, 3)` (`L_EPSA = 3` in `hm_read_mat24.F`): Accumulated 1D plastic strain $\epsilon_{p,k}$ for steel in directions 1, 2, and 3.

Both state variable tensors are stored in the element buffer (`LBUF%SIGA` and `LBUF%EPSA`) and passed into `CONC24` / `CARM24` every timestep.