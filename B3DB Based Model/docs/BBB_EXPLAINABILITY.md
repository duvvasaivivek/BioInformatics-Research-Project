# Blood-Brain Barrier (BBB) Permeability Explainability Framework

## 1. Overview & Theoretical Framework
The Blood-Brain Barrier (BBB) represents the primary mechanical and enzymatic interface separating the systemic circulation from the central nervous system (CNS) parenchyma. Passive transcellular permeability across the tightly apposed brain capillary endothelial cells is fundamentally governed by thermodynamics:
1. **Desolvation Penalty:** A polar molecule dissolved in aqueous blood must break its hydrogen-bonding hydration shell before it can dissolve into the hydrophobic lipid core of the endothelial plasma membrane.
2. **Lipophilic Membrane Partitioning:** Lipophilicity ($\text{MolLogP}$) drives the thermodynamic equilibrium toward membrane dissolution.
3. **Steric Transcellular Diffusion:** Molecular size ($\text{MolWt}$) and conformational flexibility ($\text{RotBonds}$) dictate the kinetic diffusion velocity through the dense phospholipid bilayer.

---

## 2. Model Architecture & Feature Engineering
Our BBB classifier ($\mathcal{M}_{\text{BBB}}$) is trained on the curated **B3DB benchmark** ($\sim 7,800+$ molecules, Martins et al., 2012) using a Bayesian Optuna-tuned XGBoost ensemble.
- **Topological Features:** 2048-bit Morgan Fingerprints (radius 2 / ECFP4) capturing local atomic environments.
- **Continuous Physicochemical Descriptors (Standardized):**
  $$\mathbf{x}_{\text{2D}} = \left[ \text{MolWt}, \text{LogP}, \text{TPSA}, \text{HDonors}, \text{HAcceptors}, \text{RotBonds} \right]$$
  Standardization: $z_j = \frac{x_j - \mu_j}{\sigma_j}$ using parameters serialized in `bbb_scaler_metadata.json`.

---

## 3. TreeSHAP Margin Attribution Formulation
For an individual input feature vector $\mathbf{x}$, TreeSHAP decomposes the model output in **log-odds margin space**:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{M} \phi_i(\mathbf{x})$$
where:
- $\phi_0 = -0.3121$ is the expected baseline margin across the training background.
- $\phi_i(\mathbf{x})$ is the Shapley attribution of feature $i$.
- Output probability relates to margin via the logistic sigmoid function:
  $$P(\text{BBB}^+) = \frac{1}{1 + e^{-f(\mathbf{x})}}$$

### Interpretation of Attribution Signs:
- For a **BBB+** prediction ($P \ge 0.50$, $f(x) \ge 0$):
  - $\phi_i > 0$: **Supporting features** that drive transcellular membrane permeation.
  - $\phi_i < 0$: **Opposing features** that penalize permeation.
- For a **BBB-** prediction ($P < 0.50$, $f(x) < 0$):
  - $\phi_i < 0$: **Supporting features** that explain the permeation barrier.
  - $\phi_i > 0$: **Opposing features** that weakly favor crossing.

---

## 4. Empirical Global Feature Rankings
Computed over held-out test partitions ($N=500$):
1. **Topological Polar Surface Area (TPSA):** Mean $|\text{SHAP}| = 1.0775$ (Primary desolvation filter).
2. **Molecular Weight (MolWt):** Mean $|\text{SHAP}| = 0.7107$ (Steric diffusion retardant).
3. **Lipophilicity (MolLogP):** Mean $|\text{SHAP}| = 0.4087$ (Membrane partition driver).
4. **Hydrogen Bond Donors (HDonors):** Mean $|\text{SHAP}| = 0.2962$.
5. **Rotatable Bonds (RotBonds):** Mean $|\text{SHAP}| = 0.1653$.
6. **Hydrogen Bond Acceptors (HAcceptors):** Mean $|\text{SHAP}| = 0.1063$.

---

## 5. Substructural Fingerprint Deconvolution
Using `MorganSubstructureMapper`, active Morgan bits in the top attributions are mapped to exact atomic environments and functional group tags (e.g. halogenated aryl cores, basic amine centers, carbonyl oxygen motifs), ensuring that individual fingerprint bit attributions are grounded in verifiable chemical fragments.
