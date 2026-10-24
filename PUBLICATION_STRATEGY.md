# Publication Strategy: Finding a Defendable Novelty

**Prepared:** 2026-10-24
**Repo state at time of writing:** HEAD `e13f3a5 MDs Added`, working tree clean. That commit added
`ARCHITECTURE_FLOW_BLOCK_DIAGRAM.md` and `NOVELTY_ASSESSMENT.md` only. No Python has changed since
`a53789f`, so every defect listed in `NOVELTY_ASSESSMENT.md` section 6 is still live.
**Method:** 14 literature searches plus full-text retrieval of the two most threatening papers.

---

## 1. Headline: the landscape moved, and it moved against the current framing

`NOVELTY_ASSESSMENT.md` concluded the dual-barrier framing was scooped by Adenot & Lahana (2004).
That was right but incomplete. A deeper sweep found **five more relevant papers from 2025-2026**, the
newest three months old. The "combine BBB prediction with efflux prediction" lane is now thoroughly
occupied:

| Paper | What it did | Why it threatens the current draft |
|---|---|---|
| **BBB-Nuke**, bioRxiv 2026.07.13.738280 (Jul 2026) | Trains on **B3DB (7,807)** + 2 benchmarks = 9,262 compounds. Predicts substrate probability for **7 efflux transporters** (MDR1, BCRP, MRP1, MRP2, MRP4, MATE1, OAT3) and feeds them as features into a gradient-boosted BBB classifier. AUROC 0.933 CV. Ships as pip package, REST API and MCP server. Screened 1B+ compounds. | **This is your architecture, done more comprehensively** (7 transporters vs your 1) on the same dataset, with tooling. Your two-model setup is now a strict subset. |
| **Daood et al.**, *Mol. Pharmaceutics* 2025, 22, 7022 (Oct 2025) | Hand-curated >24,000 bioactivity records from >900 sources into 8 ABC-transporter datasets (~8,800 chemicals). Links predicted efflux to brain exposure on 85 measured Kp,uu compounds. CC-BY 4.0. | Occupies "predicted efflux substrates have lower brain exposure." |
| **JCIM 2024**, 10.1021/acs.jcim.4c02212 | 154 drugs, 3-class model separating **efflux substrates from CNS+ and CNS−** (AUC 0.82). Beats CNS-MPO (0.53), BBB-Score (0.68). | Occupies "efflux deserves its own class, not a merged label." |
| **Dolgikh et al.**, *JCIM* 2016, 56, 2225 | QSAR for Kp,uu,brain **incorporating P-gp efflux as a variable**; ~1,000 compounds; adding P-gp improved test performance (rho 0.73, n=91). | Occupies "efflux improves brain-exposure prediction" — since 2016. |
| **JCIM 2026**, 10.1021/acs.jcim.6c00819 + bioRxiv 2026.02.26.708193 | Audits of TDC ADMET leaderboards: leakage, duplicated compounds with conflicting labels, `bbb_martins` rank 8 to 1 by deliberate overfitting. | Occupies the generic "the benchmark is flawed" critique. |

Also relevant: *Sci. Rep.* 2024 (s41598-024-59734-9) handles the efflux problem by **deleting P-gp
substrates** from its BBB training set "as these are likely to be removed from the CNS."

**Read plainly: a paper whose contribution is "we built a BBB model and a P-gp model and combined
them" is not publishable in 2026.** That is settled. Stop defending it.

---

## 2. But the sweep also found the gap — and it is sharper than before

Two facts, both verified from full text, not snippets:

**Fact 1 — BBB-Nuke never checks its target against reality.** In the full text, `Kp,uu` appears
exactly once, in a figure caption describing a feedback loop the paper never implements. "Unbound"
and "Friden" do not appear at all. Brain-to-plasma ratio appears only as a stated limitation and a
future direction. The paper **treats B3DB as ground truth and never analyses the label** — no
discussion of label definition, threshold, source disagreement or noise.

**Fact 2 — Daood et al. never touch a BBB benchmark.** Their brain-exposure analysis uses 85 Fridén
rat Kp,uu compounds against *predicted* transporter status. B3DB and every other BBB permeability
benchmark are absent from the paper. And their headline "twice or more likely" claim is reported
**with no statistical test and no confidence interval.**

So the field has split into two camps that never meet:

- Camp A (BBB-Nuke, LightBBB, ADMETlab, your repo): optimize accuracy **on the BBB± label**.
- Camp B (Dolgikh, Daood, Loryan): model **actual brain exposure**, on small or proprietary data.

**Nobody has asked whether Camp A's target predicts Camp B's endpoint.** That question is open, it is
cheap to answer with public data, and your measurement from `NOVELTY_ASSESSMENT.md` section 3 is the
reason to suspect the answer is no.

---

## 3. The recommended claim

> **Accuracy on BBB permeability benchmarks does not transfer to net brain exposure, and the
> benchmark label is why.** The BBB± label carries no usable information about active efflux, so
> models that improve on it — including transport-aware ones — cannot be improving prediction of
> brain exposure. We quantify the label's efflux blindness, demonstrate the failure to transfer
> against measured Kp,uu, and release a cross-labelled resource and an exposure-validated protocol.

### Three legs

**Leg 1 — Label mis-specification (you already have the seed).**
B3DB's BBB± label is uninformative about ground-truth P-gp substrate status: n = 216,
Cramer's V = 0.064, OR 1.42 [0.76, 2.67], p = 0.346; 32.1% [24.5-40.6%] of BBB+ molecules are
confirmed substrates. **This needs scaling up** — see section 5.

**Leg 2 — The failure to transfer (the killer experiment, and the actual novelty).**
Assemble the public measured-Kp,uu compounds and run a head-to-head: how well do BBB-label-trained
predictors recover *measured net brain exposure*? Baselines that are all freely available: your BBB
model, CNS-MPO, BBB-Score, LightBBB, ADMETlab, and **BBB-Nuke itself** (pip-installable, so it costs
you an afternoon to include the current state of the art as a baseline). Then show whether an
explicit efflux axis closes the gap.
*This is the part nobody has done.* Verified: BBB-Nuke has no Kp,uu validation; Daood's version has
no statistics and no BBB benchmark.

**Leg 3 — Resource and protocol.**
Release the cross-labelled set plus a recommendation: report efflux-stratified and
exposure-validated metrics alongside BBB AUC. Dataset-and-protocol papers are exactly what
*Journal of Cheminformatics* and *Scientific Data* publish, and the artifact is what gets cited.

### Why each competitor cannot claim this

| Competitor | Why it does not block Leg 2 |
|---|---|
| BBB-Nuke 2026 | No Kp,uu validation anywhere; treats B3DB as ground truth; never questions the label |
| Daood 2025 | Never uses B3DB or any BBB benchmark; "2x" claim has no statistical test |
| Dolgikh 2016 | Proprietary in-house data, regression model, not a benchmark critique, not reproducible |
| JCIM 2024 | 154 drugs, 3-class label; no Kp,uu; does not question B3DB |
| TDC audits 2026 | Audit label **correctness** and leakage. This is label **mis-specification** — the label is correct and still measures the wrong thing. Different axis. |
| Adenot 2004 | Built the dual model; never tested label adequacy or brain exposure |

**Bonus finding worth a paragraph of its own.** BBB-Nuke labels a compound a substrate if
**"IC50 is below 10 uM or the efflux ratio is above 2.0."** IC50 is an *inhibition* measure, not
transport. Their seven "substrate" models are therefore partly inhibition models — a real
methodological error in the current state of the art, and exactly the kind of thing a careful
cross-labelled resource exists to prevent. Their external test set also appears in their own
training list (Benchmark2.5 is both "never used in training" and among the 9,262 CV compounds), a
contradiction the preprint does not address. Both are fair, citable observations.

---

## 4. What to abandon

Being blunt, because the sunk cost is the enemy here:

- **The current title and framing.** Dead.
- **3D conformer descriptors as a selling point.** Your own ablation refutes it (`Morgan+2D` 0.9509
  beats `Morgan+2D+3D` 0.9481) and *Pharmaceutics* 2026, 18, 1018 published the same conclusion.
- **Any claim to competitive BBB accuracy.** 0.9522 is mid-pack; BBB-Nuke reports 0.933 CV on a
  larger set and a 2025 GNN reports 0.9685 under scaffold splits.
- **The GNN arm** (`Initial Model/`). Discards 269 Li molecules silently; no metrics persisted.
- **"We built a dual-barrier framework."** Adenot 2004, and now BBB-Nuke with 7 transporters.

## What to keep, repurposed

- **The two trained models** — demoted from "the contribution" to "one baseline among six."
- **The 216-molecule overlap** — promoted from a footnote to the seed of the released resource.
- **The cryo-EM 6QEX docking** — still genuinely unoccupied as an ML-vs-dG correlation, but only
  after the hard-coded probabilities are fixed. Belongs in Paper 2 as mechanism.
- **The FDA pharmacopeia screen** — Paper 2's lead.
- **`core/features.py`** — the shared feature kernel is reusable for the resource build.

---

## 5. Concrete next steps, in order

**Step 0 — fix the invalidating defects first (days).**
A1 (unscaled SHAP), A2 (hard-coded docking probabilities), A3 (two legacy eval scripts) from
`NOVELTY_ASSESSMENT.md`. Everything downstream inherits these bugs otherwise. No Python has changed
since `a53789f`, so none of this is done yet.

**Step 1 — build the cross-labelled resource (1-2 weeks).**
Download Daood et al.'s supporting file `mp5c01065_si_002.xlsx` (458 KB, **CC-BY 4.0**, free from the
ACS article page) — it contains transporter interactions for ~8,800 compounds across eight sets.
Intersect each with B3DB by InChIKey, as already done for Broccatelli.

Realistic expectations, so you plan with the right number. Their *substrate* sets are smaller than
the "8,800" headline suggests, because that figure is dominated by inhibition data:

| Daood substrate set | Size | Split |
|---|:---:|---|
| P-gp | 607 | 297 sub / 310 non-sub |
| BCRP | 507 | 253 / 254 |
| MRP2 | 253 | 134 / 119 |
| MRP1 | 182 | 94 / 88 |

Combined with Broccatelli (1,205, giving 216 B3DB matches), a realistic yield is **roughly 350-600
molecules carrying a BBB label plus at least one transporter label**, some with two or three
transporters. That is 2-3x your current n, enough to tighten the OR confidence interval
substantially, and a **tri-axis (BBB x P-gp x BCRP) resource does not currently exist**. Report the
yield honestly whatever it turns out to be.

**Step 2 — assemble the public Kp,uu evaluation set (1 week).**
- Fridén et al. rat Kp,uu, 85 compounds after dedup (the set Daood used — cite them).
- A Frontiers 2024 compilation (~157 compounds, stated as publicly disclosed).
- A separate 88-compound literature compilation.
Dedupe by InChIKey; expect **~150-250 unique compounds**. Use the established Kp,uu >= 0.1 threshold,
and report the 0.3 threshold as a sensitivity analysis. **State the rat-vs-human caveat prominently**
— most public data is rodent.

**Step 3 — run the head-to-head (1-2 weeks). This is the paper.**
Predict measured Kp,uu class with: your BBB model; CNS-MPO; BBB-Score; LightBBB; ADMETlab;
BBB-Nuke; and your two-axis score. Report AUC/MCC per method on the *exposure* endpoint, next to
each method's published BBB-benchmark AUC. The expected headline is a scatter where benchmark
performance and exposure performance do not line up.

**Step 4 — Leg 1 at the new scale, plus efflux-stratified error analysis** (E1 from
`NOVELTY_ASSESSMENT.md`, requires K-fold retraining for out-of-fold predictions).

**Step 5 — write Paper 1**, then Paper 2 unchanged from `NOVELTY_ASSESSMENT.md` section 5.

---

## 6. Risks, stated honestly

1. **The window is closing.** Five relevant papers in 18 months, the newest in July 2026. If someone
   runs the Kp,uu head-to-head first, this plan is gone. This is the main argument for moving now and
   for preprinting early.
2. **Leg 1 alone is not a paper.** At n = 216 it is a null result on a small sample, and a reviewer
   can call it underpowered or obvious. It only works as the *explanation* for Leg 2. Do not lead
   with it unscaled.
3. **Public Kp,uu data is small and mostly rodent.** 150-250 compounds, rat-dominated. This caps the
   claim's strength and must be disclosed, not buried.
4. **Fridén's 85 compounds are not fresh** — Daood used them. Cite and differentiate (they tested
   predicted substrate status; you test BBB-benchmark-trained models against the exposure endpoint).
5. **Leg 2 might come back positive** — BBB models might predict Kp,uu acceptably. Then the thesis
   inverts and the paper becomes "BBB benchmarks are a better proxy for exposure than their label
   composition suggests," which is publishable but weaker. Write both framings before running it.
6. **Baseline fairness.** If you benchmark BBB-Nuke or LightBBB on Kp,uu compounds that overlap their
   training data, the comparison is void. Check overlap by InChIKey for every baseline and report it.
   BBB-Nuke trains on all of B3DB, so any Kp,uu compound also in B3DB is contaminated for that
   baseline — this will shrink your usable set and you must report the clean subset separately.
7. **A chemistry reviewer will push back on the independence result** by arguing the 216 overlap
   molecules are a biased subset (drugs studied for both endpoints). Pre-empt it: compare their
   property distributions against full B3DB and report the comparison.

---

## 7. If you want the fastest publishable thing instead

If the above is more work than you want, the minimum viable honest paper is **Paper 2 only** from
`NOVELTY_ASSESSMENT.md` section 5: the FDA pharmacopeia two-axis screen plus the 6QEX docking, in
*Molecular Pharmaceutics* or *Pharmaceutics*, framed as a resource and not as a methods advance. It
needs defects A2, A3, B6 fixed and the clinical claims toned down. Novelty is moderate — the
ML-vs-docking-energy correlation in 6QEX is the one piece nobody has published — but it is
achievable in weeks rather than months, and it does not require the new data work.

**My recommendation is still the Kp,uu route.** It is the only path found in 14 searches that leads
to a claim no 2025-2026 paper has made, and it is defendable precisely because the two camps in this
field have not yet been made to talk to each other.

---

## 8. Verification notes

- BBB-Nuke and Daood et al. claims come from **full-text retrieval**, not search snippets. Everything
  else in section 1 comes from search results and should be re-verified against the primary source
  before citation — several are preprints and at least one search result conflated authors
  (ci034205d is Adenot & Lahana, not Muehlbacher).
- Dataset sizes in section 5 are quoted from Daood et al. Table 2 as retrieved; confirm against the
  supporting file once downloaded.
- The 216-molecule statistics were computed in-session from the repo's own files and reconcile
  (45+95+19+57 = 216).
- Yield estimates in Step 1 and Step 2 are **projections, not measurements.** Measure them before
  committing to a framing.
