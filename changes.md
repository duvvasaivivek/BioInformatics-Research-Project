# Required Research-Paper Revisions for Prism

Use this file as the revision brief for `main.tex`. Make the changes in the manuscript and any directly related tables, figures, captions, and references. Preserve the scientific scope and use repository/source artifacts rather than inventing results. Where a point is marked **VERIFY**, do not guess: flag it for the authors and avoid presenting an unverified interpretation as fact.

## 1. Resolve the P-gp endpoint before editing conclusions

This is the highest-priority issue and affects the paper's central interpretation.

- The project file is `Novelty Upgrade/data/pgp_broccatelli.tab`, with `Drug_ID`, `Drug` (SMILES), and `Y` columns. The preprocessing code passes `Y` through as the label.
- The cited Broccatelli 2011 article is titled *A Novel Approach for Predicting P-Glycoprotein (ABCB1) Inhibition Using Molecular Interaction Fields* (DOI: `10.1021/jm101421d`).
- TDC describes its `Pgp_Broccatelli` task as **P-gp inhibition prediction**, not substrate prediction.
- Therefore, do not call `Y=1` an effluxed/transported substrate unless the exact raw-file provenance and positive-class mapping have been independently verified.

**VERIFY with the project authors:** the exact TDC task/version, origin of the committed raw file, the meaning and direction of `Y`, and whether/how the inhibition labels were transformed. The manuscript currently says “following the authors’ clarification,” but author clarification alone does not establish that the upstream labels mean substrate.

If confirmed as inhibition:

- Reframe the endpoint consistently as **P-gp inhibition** throughout abstract, methods, results, tables, figures, quadrant definitions, discussion, and conclusion.
- Remove or rewrite interpretations that equate P-gp-positive with “pumped back out,” “efflux liability,” “substrate,” “efflux resistance,” or “brain exposure.” Inhibiting P-gp and being transported by P-gp are different phenomena.
- Rename quadrants and decision metrics so they accurately describe the labels. Do not retain the “favorable CNS Q1” and “paradox leakage” interpretations without a biological basis for that endpoint.
- Reassess whether docking into the P-gp cavity answers the inhibition-label question. Binding/docking alone cannot distinguish inhibitor from substrate; present docking only as exploratory unless independently justified.

If the intended endpoint is substrate/efflux:

- Do not merely relabel the current TDC inhibition task. Obtain a verified substrate-specific benchmark, document assay and class semantics, rerun preprocessing/training/evaluation/overlap/SHAP/screens and regenerate all affected values and figures.
- Clearly distinguish any new substrate dataset from the current inhibition dataset.

Until this is resolved, describe outputs as **project P-gp benchmark labels** and avoid substrate/efflux biological conclusions.

## 2. Reconcile data counts and definitions

- State that the raw P-gp table has 1,219 rows, the descriptor preprocessing retains 1,205, and the GNN corpus assembly uses 1,212 P-gp-labeled structures. Explain that these differ because the pipelines apply different validity/feature requirements and record handling; verify exact counts from the current scripts/artifacts before publication.
- B3DB contains 7,807 raw rows and the descriptor pipeline retains 7,805.
- The merged GNN corpus has 8,801 unique structures, with 7,805 BBB labels, 1,212 P-gp labels, and 216 shared labels.
- In the corpus table, mark “Shared structures” as an **overlap subset**, not an additional stratum to be added to the totals.
- Distinguish raw source labels from derived model features. For P-gp, the raw columns are name/identifier, SMILES, and target label; fingerprints and descriptors are calculated by this project.
- Do not call the 216 overlap “experimental confirmation” of efflux. Describe it as overlap in benchmark annotations, with the exact endpoint semantics stated.

## 3. Reconcile all numerical values with the exact artifacts

- Replace stale SHAP values using the selected, final result artifact. The current repository result `Novelty Upgrade/results/shared_properties_quantitative_comparison.csv` reports:
  - BBB TPSA mean |SHAP| = 1.1825
  - P-gp MolLogP mean |SHAP| = 1.3949
  - BBB MolWt mean |SHAP| = 0.5366
  - P-gp MolWt mean |SHAP| = 0.5783
- The Overleaf export contains an older SHAP file with different values (e.g. TPSA 0.9875, LogP 1.4074, MolWt 0.6955/0.6292). Select the intended analysis version and synchronize manuscript text, tables, plots, and supplements. Do not mix results from the Overleaf snapshot with current repository results.
- The GNN OOF values in the paper appear consistent with repository reports: single-task BBB/P-gp ROC-AUC 0.9379/0.9011; multi-task 0.9363/0.9056. Keep these separate from the descriptor/XGBoost AUCs (random BBB/P-gp 0.9522/0.9587; scaffold 0.9048/0.9631).
- Keep evaluation protocols and sample sizes beside every headline number. Do not imply XGBoost and GNN values are a controlled head-to-head comparison.
- Docking data differ across snapshots: current repository `cryoem_pgp_docking_validation.csv` contains 22 rows; the Overleaf export contains 21. Use one version, state the complete-case n used for each analysis, and regenerate tables/plots/statistics from that same version.
- The Overleaf `extra_analysis.json` says n=21. Its size-adjusted analysis reports heavy-atom-count correlations with docking score and P-gp model score, and a near-zero partial rank association after controlling for heavy-atom count. Retain this only if the matching script and exact data are included and reproducible. Do not present raw docking separation as independent confirmation.
- The legacy Zhao/Adenot/Li corpus has inconsistent counts in repository documents (1,890 vs 1,990). Verify raw records, filtering, duplicates, and actual model inputs; state each count with its provenance rather than choosing one silently.
- Run the repository's document consistency check after choosing final artifacts, and use its matching results as the source of reported values.

## 4. Correct methods wording and scope

- In `Novelty Upgrade/core/features.py`, ETKDGv3 conformers are embedded to calculate 3D shape descriptors; this descriptor path does not force-field-minimize them. Do not claim minimization for the P-gp descriptor model unless another invoked method actually does it. Describe docking ligand preparation separately if it includes minimization.
- Explain that TreeSHAP explains the repository's XGBoost models, not the GNN. Do not imply the GNN itself has been explained by these SHAP results.
- Describe SHAP as model attribution, not causal effect or proof of transporter mechanism.
- Describe the approved-drug screen as computational prioritization/hypothesis generation, not validated repurposing or external validation. State benchmark overlap.
- Describe docking scores as computed scoring-function outputs, not experimentally measured binding free energies, affinities, transport rates, or substrate/inhibitor proof.
- Keep the limitation that the five-seed AUC comparisons are underpowered; do not conclude equivalence or that multi-task learning cannot help.
- Avoid “proves,” “first,” “clinical validation,” and other stronger claims than the data support.

## 5. Improve the flowchart and figure communication

Review every schematic/flowchart for scientific correctness, readability, and agreement with the final endpoint semantics. Redraw or simplify any confusing diagrams; do not only change colors or styling.

- **Dual-barrier concept diagram:** show passive BBB permeation and P-gp transport as distinct processes. If the dataset is confirmed to be inhibition, depict inhibitor activity correctly (inhibition of transporter activity), not substrate efflux. If substrate labels are confirmed, show transport back toward blood only where that label is validated. Add a clear note that these are two predicted labels, not a sequential experimental verification.
- **Data-flow diagram:** name the exact source datasets and their label endpoints; show raw data → parsing/canonicalization → feature generation/graph construction → single-task and multi-task models → held-out/OOF evaluation. Clearly mark masks for missing labels and the 216-molecule overlap. Do not visually suggest the shared rows are all training data for each independent evaluation.
- **Model-journey diagram:** make the stages and model families explicit: legacy BBB GCN reference, separate endpoint models, then shared-encoder GNN experiment. Do not imply that XGBoost is a GNN or that P-gp validates BBB predictions.
- Ensure arrows have unambiguous direction and do not cross/overlap labels. Use consistent BBB/P-gp names, positive-class definitions, quadrant colors and legends across every figure.
- Each figure should remain understandable when viewed alone: define abbreviations, indicate cohort/split where relevant, report n, and distinguish prediction from measurement.
- Replace any stale or misleading docking-correlation figure if size-adjusted analysis changes its interpretation. Prefer a plot that shows both raw association and the heavy-atom-adjusted result, with caveats.

## 6. References, figures, and reproducible build

- Keep `references.bib` synchronized with the final citations. Verify the exact TDC dataset task citation and dataset version; do not cite the inhibition paper as a substrate-data source without explaining the discrepancy.
- Confirm every `\cite{...}` key exists in the bibliography and every listed reference is cited.
- Check that all figure filenames referenced by `main.tex` exist under `figures/`, and that their values/labels match the chosen results snapshot.
- Confirm the requested BibTeX style is available in the Overleaf environment; if `unsrturl` is unavailable, select a supported style that follows the target journal requirements.
- Compile the complete Overleaf project and fix missing files, unresolved citations/references, overfull tables, unreadable figure labels, and broken layout before finalizing.
- Follow the selected journal’s current author instructions/template; do not assume a generic `article` class is guaranteed to satisfy submission requirements.

## Order of work

1. Verify P-gp endpoint provenance and label meaning.
2. Decide whether the paper is about P-gp inhibition or P-gp substrate transport.
3. Recompute or reframe all dependent results and biological interpretations.
4. Select one canonical results/figures snapshot and reconcile counts and values.
5. Correct methods, diagrams, tables, citations, and build issues.
6. Recompile and perform a final consistency pass against the repository artifacts.
