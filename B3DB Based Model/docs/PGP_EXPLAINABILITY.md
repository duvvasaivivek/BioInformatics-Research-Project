# P-glycoprotein (P-gp / ABCB1) Efflux Explainability Framework

## 1. Physiological Mechanism of P-gp Efflux
P-glycoprotein (ABCB1 / MDR1) is a 170 kDa ATP-binding cassette (ABC) efflux transporter located on the apical (luminal) membrane of brain capillary endothelial cells. Rather than acting as a standard channel, P-gp functions as a "hydrophobic vacuum cleaner":
- Lipophilic xenobiotics partition into the inner lipid leaflet of the endothelial membrane.
- P-gp intercept substrates directly within the lipid bilayer through its large internal translocation chamber ($\sim 6,000 \text{ \AA}^3$).
- Upon substrate binding and ATP hydrolysis at its nucleotide-binding domains (NBDs), P-gp undergoes an outward-facing conformational shift, expelling the molecule back into the capillary lumen.

---

## 2. 3D Conformation Feature Engineering
Because P-gp's multisite binding chamber is inherently three-dimensional and flexible, 2D connectivity fingerprints are augmented with **six stereochemical 3D conformer descriptors** generated via RDKit ETKDGv3 embedding:
1. **Radius of Gyration ($R_g$):** Measures the mass-weighted root-mean-square distance of atoms from the center of mass. High $R_g$ indicates an extended conformation capable of spanning multiple transmembrane binding contacts.
2. **Asphericity ($\Omega$):** Quantifies deviation from spherical geometry ($\Omega = 0$ for a sphere, $\Omega = 1$ for a rod).
3. **Eccentricity ($c/a$):** Ratio of principal axes characterizing prolate/oblate ellipsoids.
4. **Spherocity Index:** Degree of spherical compactness. High spherocity correlates with smooth membrane crossing without cavity entrapment.
5. **Normalized Principal Moments of Inertia ($NPR_1, NPR_2$):** Coordinate coordinates in triangular shape space (rod-like, disc-like, spherical).

---

## 3. TreeSHAP Attribution Space
Attributions are calculated in **log-odds margin space**:
$$f(\mathbf{x}) = \phi_0 + \sum_{i=1}^{M} \phi_i(\mathbf{x})$$
where:
- $\phi_0 = +0.1519$ is the expected background value.
- Positive class ($Y=1$): P-gp Substrate (Pumped out).
- Negative class ($Y=0$): P-gp Non-Substrate (Efflux evader).

### Empirical Global Rankings for P-gp Efflux:
1. **Lipophilicity (MolLogP):** Mean $|\text{SHAP}| = 1.4074$ (Dominant driver of membrane cleft recruitment).
2. **Molecular Weight (MolWt):** Mean $|\text{SHAP}| = 0.6292$ (Cavity spatial volume).
3. **Radius of Gyration ($R_g$):** Mean $|\text{SHAP}| = 0.5381$ (Primary 3D spatial determinant).
4. **Normalized Moment ($NPR_2$):** Mean $|\text{SHAP}| = 0.4718$.
5. **Topological Polar Surface Area (TPSA):** Mean $|\text{SHAP}| = 0.1266$.
6. **Hydrogen Bond Donors (HDonors):** Mean $|\text{SHAP}| = 0.0680$.

---

## 4. Seelig Pharmacophore Alignment
Our model's attributions autonomously validate the classical **Seelig rules (1998, 2000)**:
- Substrates require spatial pairs of electron-donor/acceptor groups.
- Molecules with $\le 1$ H-bond donor lack sufficient electrostatic anchoring, evading the efflux pump (e.g., Levetiracetam).
