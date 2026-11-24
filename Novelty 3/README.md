# Novelty 3 — Metabolite-aware CNS screening

Screen documented drug parents and their metabolites through **BBB → P-gp**. Each molecule gets its own BBB decision. A molecule that passes BBB is checked for P-gp substrate liability. A rejected parent still has its metabolites screened. The research question is whether prioritising metabolites by their scores relative to their parents recovers more favourable candidates at the same follow-up assay budget.

## Run in PowerShell

From the repository root, use the existing virtual environment.

```powershell
Set-Location -LiteralPath 'Novelty 3'
$py = '..\venv\Scripts\python.exe'
& $py cli.py environment
& $py -u cli.py data fetch
& $py -u cli.py data audit
& $py -u cli.py data prepare
```

The audit returns **2** for a completed evidence review report and **1** on failure. Continue to preparation after exit 2. Preparation verifies the audit artifacts and builds a reusable graph cache. It groups structures by connected metabolic family, connectivity and nonempty Bemis–Murcko scaffold. Groups containing annotated metabolites are reserved across both endpoints. Each model independently splits its remaining labelled data into training, early stopping, calibration and test sets. A new environment needs the dependencies in `requirements.txt` and a CUDA-enabled PyTorch build to use the GPU.

Run the training commands in order. The default is five seeds per endpoint. To train one seed during a shorter session, append `--seeds 11` to either training command and later run the same command with all seeds. Completed seeds are skipped. Ctrl+C stops foreground training. Rerun the same command with the same settings to continue from the last saved epoch. Run one training session at a time per seed. Device selection defaults to CUDA when available. Use `--device cpu` if needed.

```powershell
& $py -u cli.py train bbb
& $py -u cli.py train pgp
& $py -u cli.py calibrate bbb
& $py -u cli.py calibrate pgp
& $py -u cli.py evaluate
```

Both GINE models use three graph layers, 128 hidden channels, mean pooling, dropout, AdamW and validation average precision for early stopping. The defaults are batch size 32, a maximum of 200 epochs and patience of 20 epochs. For a one-seed evaluation, use `--seeds 11` on **both calibration commands**. Calibration fits a sigmoid on the mean ensemble logit using its own holdout. The default cutoffs target at least 90% sensitivity for the positive endpoint label on that holdout, including P-gp **substrates**. Adjust the target through `--min-sensitivity` before examining test results.

## Compare the P-gp models

After training and calibrating the GNNs, run these from the active virtual environment in `Novelty 3`.

```powershell
python -u cli.py fit-pgp-baseline
python -u cli.py compare-pgp
```

The first command fits one CPU XGBoost model using 167-bit MACCS fingerprints. It uses the same P-gp training, early stopping and calibration sets as the GNN. The baseline cutoff targets the sensitivity stored in the GNN calibration, currently 90%. The baseline weights and calibration are saved together in `models/pgp/maccs_xgboost.ubj`.

The second command writes `results/pgp_comparison.json`. Compare the substrate detection and nonsubstrate false alarm counts side by side. The two thresholds use the same calibration target, while the actual detection rates are reported separately on each cohort. The after-BBB rows use the frozen BBB gate and exclude groups used in its training, early stopping and calibration. The existing test results are retrospective. Confirm a model change on new source-verified families. The standard screening commands still use the existing GNNs.

### Metabolite-specific P-gp cutoffs

```powershell
python -u cli.py pgp-metabolites audit
python -u cli.py pgp-metabolites calibrate
python -u cli.py pgp-metabolites evaluate
```

The audit checks `data/input/pgp_metabolite_benchmark.json` for structural identity, substrate assay evidence, and cohort overlap. Each distinct metabolite has one adjudicated record. Record the parent, metabolite, original study, assay and human metabolism citation. Use `1` for a measured substrate, `0` for a measured nonsubstrate, and `null` for an unknown or conflicting result. A study's `source_overlap` must be `clear` with a documented review to qualify. Source clearance requires checking the experiments used by the training datasets, including the [upstream Han et al. P-gp data](https://doi.org/10.1021/acs.jcim.4c02122). Use `unresolved` when the evidence does not establish independence. An `identity_note` keeps an uncertain structure out of both cohorts.

Assign cleared records to `development` or `test` before inspecting model predictions. Related drug families, shared scaffolds, the same study and near analogues (Morgan similarity ≥ 0.80) must stay together. Groups related to prior endpoint data or either metabolite benchmark are excluded. The bundled evidence is a starter set marked `exploratory`; an eligible development and test cohort still need to be curated.

Once the data qualify, `calibrate` selects the highest cutoffs that catch at least 90% of development substrates after the existing BBB gate. It saves the cutoffs with fingerprints of the cohort and models as `models/pgp/metabolite_cutoffs.json`. `evaluate` checks those fingerprints and writes `results/pgp_metabolite_evaluation.json`. It compares the two new cutoffs and the current GNN on test metabolites, and reports paired 95% bootstrap intervals obtained by resampling whole groups. XGBoost is a replacement candidate only when it retains substrate detection and reduces false flags, with both conclusions supported by the intervals. A one-sided sign test at 0.05 also checks whether the false-flag improvement extends across enough groups. After reviewing test outcomes, use a fresh test cohort to validate any change to the method.

## Results

- `results/phase1/feasibility_report.md` documents the source coverage and evidence gaps. `candidate_coverage.csv` in the same directory maps relationship identities to structures, available labels and source evidence.
- `models/bbb/` and `models/pgp/` each hold one resumable checkpoint per seed and one calibration file. Checkpoints store the latest epoch and the best validation weights. Calibration records the checkpoint fingerprints.
- `results/screening.csv` contains the parent and metabolite decisions and scores, keyed by full InChIKey. Blank P-gp predictions indicate BBB rejection. `screen` produces this file on its own, and `evaluate` also generates it.
- `results/evaluation.json` reports test and held-out metabolite metrics independently for BBB and P-gp. Fixed-budget comparisons use only reserved groups. The methods compare parent-relative prioritisation, independent metabolite prioritisation, and parent-guided selection. Each can select at most one metabolite per family, and unknown outcomes are counted explicitly. Budgets default to 5, 10, 20 and 50 and can be set with `evaluate --budgets 5 10 20`.

For compounds that pass both stages, ranking support is the smaller of their BBB-positive score and P-gp nonsubstrate score after subtracting one seed standard deviation from each, with a floor of zero. Parent-relative ranking subtracts the parent's support. Rejected compounds have zero support for ranking purposes. This is a prioritisation heuristic. Seed standard deviation reflects variation across models. The parent-guided comparator orders candidates by the parent score and breaks ties by identity, using no metabolite prediction. The joint outcome requires labels for both endpoints. The standalone P-gp benchmark also evaluates P-gp-labelled compounds that the sequential screen would reject at BBB.

The bundled benchmark uses retrospective source annotations. B3DB BBB labels combine heterogeneous endpoints. SF-GA gives binary substrate labels with limited per-record assay context. The metabolism sources lack complete study-level provenance for many relationships. The few candidate BBB-negative parent to BBB-positive metabolite reversals require original study review for any human exposure claim.

Reviewed experimental records can be added to `data/input/metabolite_measurements.csv`. Supply a unique record ID, exact structure, endpoint, 0/1 label, measurement type, species, assay, source URL or DOI, and a label definition. BBB accepts `bbb_annotation`, `logbb` and `kp_uu_brain`. P-gp accepts `substrate_assay` and `efflux_ratio`. Ratio and logBB records require finite values and units. Provide the experimental classification cutoff in the definition. Curated metabolite results are reported separately. After changing curation, rerun audit and preparation and move old checkpoints aside before starting a new run.

## Verification and sources

The published-measurement pilot runs with the existing checkpoints.

```powershell
& $py -u cli.py validate
```

`data/input/paired_validation.json` records the original studies, compound identities, exact assay conditions and paired measurements. The command checks those identities, adds the verified relationships in memory and recomputes family and scaffold groups. It excludes groups used to train, tune or calibrate either model from the fixed-budget comparison. It then applies the original BBB → P-gp cutoffs and writes `results/paired_validation.json`. The training data, checkpoints and retrospective report are separate from this pilot.

The budget comparison counts picks with a higher measured brain-to-systemic concentration ratio than their parent in the same study. For repeated time points it uses the median of the matched fold changes and reports each time point. It allows one pick per merged group and reports how many slots each policy filled. Species and assays differ, so the report uses within-study directions and treats them descriptively. P-gp knockout findings serve as separate controls for the animal experiments.

With the current five-seed ensembles, six paired measurements were reviewed and five pairs across three groups qualified. Clozapine and its metabolite were excluded from the budget comparison because their group overlapped both training sets. Both venlafaxine metabolites had higher median brain-to-serum ratios relative to their parent across the measured time points. Both were rejected at BBB. Risperidone and paliperidone were both flagged at P-gp, consistent with the mouse knockout study. All three policies selected zero candidates at budgets 1, 2 and 3. The original retrospective benchmark showed a small early advantage for parent-relative ranking. The paired pilot exposed gate misses and leaves the ranking advantage unconfirmed on measured data. [Venlafaxine measurements](https://pubmed.ncbi.nlm.nih.gov/20466523/) · [Risperidone transport study](https://pubmed.ncbi.nlm.nih.gov/15683552/)

Review of the original three annotation reversals found one name–structure mismatch, one metabolite formation study without comparable BBB measurements, and one predicted pass for a metabolite whose animal brain distribution study reported restricted exposure. These three cases were kept out of the paired pilot. See the [famotidine identity record](https://precision.fda.gov/uniisearch/srs/unii/5qzo15j2z8), [salicylic acid study](https://pubmed.ncbi.nlm.nih.gov/25840124/), and [6-mercaptopurine brain distribution study](https://pubmed.ncbi.nlm.nih.gov/10794520/).

Run the implementation tests with:

```powershell
& $py -m pytest tests -q
```

Pinned source data and notices are downloaded to `data/raw/`. Graph caches, processed artifacts, checkpoints, and outputs are local and excluded from version control.

- [B3DB BBB annotations](https://doi.org/10.1038/s41597-021-01069-5)
- [SF-GA P-gp substrate data](https://doi.org/10.1021/acs.jcim.6c01938)
- [GLORYx metabolism data](https://doi.org/10.1021/acs.chemrestox.0c00224)
- [GLORYx dataset files and notices](https://github.com/christinadebruynkops/GLORYx/tree/master/datasets)
