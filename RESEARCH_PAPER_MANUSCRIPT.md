# Manuscript transcribed from main.pdf

> Source: main.pdf. Text and reported results follow the PDF; no newer repository results have been substituted. Figure artwork and page layout remain in the source PDF; captions and extractable labels are included where available.

<!-- PDF page 1 -->
From Single-Task to Multi-Task Learning for
Dual-Barrier CNS Prediction: A Decision-Aware
Evaluation
Sai Vivek Duvva 1, Nagireddi Yashaswini 1, Rishidhar Reddy 1,
Chandra Mohan Dasari 1,∗
1Department of Computer Science and Engineering,
Indian Institute of Information Technology Design and Manufacturing Kurnool,
Jagannathagattu Hill, Kurnool, Andhra Pradesh 518008, India
∗Supervisor and corresponding author: chandramohan.d@iiitk.ac.in
Abstract
Predicting brain exposure from a blood–brain barrier (BBB) permeability label alone
omits active transporter recognition, including P-glycoprotein (P-gp) status. We compared
independent graph models with a shared-encoder multi-task graph neural network (GNN)
using the project’s BBB and P-gp benchmarks. The datasets overlap for 216 molecules, with
weak label association (Cram´ er’sV= 0.064; Fisher odds ratio 1.42, 95% CI 0.76–2.67). In
five-fold out-of-fold evaluation, single-task models achieved ROC-AUCs of 0.9379 (BBB) and
0.9011 (P-gp), compared with 0.9363 and 0.9056 for the multi-task model; these results do
not demonstrate a reliable transfer benefit. On the dual-labeled molecules, quadrant exact
match was 57.9% for composed single-task models and 56.0% for the multi-task model.
A BBB-only rule sent 42 of 45 true BBB +/P-gp-positive cases to the favorable quadrant.
Repository XGBoost screening, feature attribution and docking analyses provide additional
context, but are not independent clinical or experimental validation. The P-gp endpoint’s
substrate interpretation also requires source-level confirmation.
Keywords:blood–brain barrier; P-glycoprotein; multi-task learning; graph neural network;
CNS drug discovery; decision-aware evaluation

## 1 Introduction

The blood–brain barrier (BBB) regulates molecular access from blood to the central nervous
system through tightly connected endothelial cells and selective transport processes. Predicting
whether a compound is likely to cross this barrier is an established computational task, and
graph neural networks (GNNs) have been applied to represent molecules directly as graphs of
atoms and bonds. The GCN-BBB study supplied with this project provides the starting point
for the initial graph-model replication [1]. Curated datasets such as B3DB provide a larger
collection of BBB permeability labels for model development [2].
Permeability does not by itself determine brain exposure. P-glycoprotein (P-gp/ABCB1),
expressed at the BBB, can transport recognized substrates back toward the circulation [3].
Loperamide illustrates the relevance of transporter-mediated restriction: P-gp inhibition can

<!-- PDF page 2 -->
increase its delivery to the brain [4]. Passive permeation and active efflux are distinct processes;
neither a BBB+ call nor a P-gp classifier output alone establishes unbound brain concentration,
therapeutic exposure or clinical efficacy.
The project therefore frames CNS prediction as a two-axis classification problem. Each
molecule is assigned a BBB label and a P-gp label; their combinations form four quadrants. For
this paper, consistent with the authors’ clarification, P-gp + denotes a molecule annotated as a
substrate that is pumped back toward the blood, while P-gp − denotes the non-substrate/efflux-
negative label. BBB +/P-gp-negative is a useful screening hypothesis, not a sufficient condition
for CNS drug suitability. It does not capture unbound brain exposure, other transporters,
metabolism or target engagement.
Figure 1:Four-quadrant representation combining BBB permeability and the project’s P-gp pipeline
labels.

### 1.1 Study motivation and questions

A BBB model can perform well on its own label while remaining unable to distinguish a permeant
P-gp substrate from a permeant non-substrate. The two benchmark labels overlap for 216
molecules, allowing a direct, though limited, examination of this issue. Prior work has also
described combined BBB/P-gp classification, so the present study does not claim to introduce
the dual-barrier concept [5].
We ask: (i) how much association is observed between the two labels in the shared set; (ii)
whether hard parameter sharing improves per-task or joint decision performance over separate
models; and (iii) what is lost when a single-axis BBB decision is used in place of a joint clas-
sification. We additionally report explainability, approved-drug screening and docking outputs
from the repository as supporting analyses, with their scope and limitations stated.

<!-- PDF page 3 -->
P-gp is an ATP-dependent transporter with two transmembrane domains and a broad ligand-
binding cavity. Structural and pharmacological studies describe recognition as multi-specific
rather than governed by a single universal motif; molecular size, hydrophobic interactions and
other ligand features can contribute in context-dependent ways [6, 7]. This helps explain why
passive membrane partitioning cannot be treated as a surrogate for transporter recognition.
The present classifiers nevertheless learn benchmark labels, not transporter kinetics or net brain
exposure. The project-specific interpretation of the positive class is stated above; the cited
Broccatelli publication itself is titled as an inhibitor-prediction study, so exact source-label
lineage remains a reproducibility caveat.
Compound in blood BBB endothelium Brain compartment
P-gp recognition
and transport
passive
permeation
transport
back
Figure 2:Conceptual distinction between passive permeation and transporter-mediated return.

### 1.2 Related computational approaches

Early BBB classifiers used molecular descriptors and statistical learning to distinguish com-
pounds reported to penetrate the brain from compounds that did not [8–10]. Fingerprints and
physicochemical descriptors remain practical baselines because they encode local substructures
and global properties while allowing efficient training on curated data [11, 12]. B3DB sub-
sequently assembled a diverse collection of BBB measurements and labels, creating a widely
used resource for prediction and benchmarking [2]. These binary labels are useful for ranking,
but they simplify a continuum of permeability and often combine experimental protocols and
thresholds.
Graph neural networks offer a complementary representation: message passing updates atom
states using neighboring atom and bond attributes, then pools atom states to represent the
molecule. The supplied GCN-BBB paper motivated the initial project replication [1]; the present
graph arm uses an edge-aware GINE-style network rather than claiming an exact reimplemen-
tation of every architectural or data choice in that study. General graph-learning work provides
the methodological context for this representation [13, 14].
Transporter-aware prediction is not new. Earlier permeability models explicitly distinguished
potential CNS drugs from P-gp substrates [5], and structural studies have since refined under-
standing of the transporter cavity and its substrate/inhibitor discrimination [7]. The project
contribution is therefore not the invention of a dual-barrier concept. Its focused question is em-
pirical: does a shared graph encoder improve the two tasks in this particular, sparsely co-labeled
benchmark, and do task-level scores translate into reliable joint quadrant decisions?

### 1.3 Study objectives and hypotheses

The analysis has four objectives. First, quantify the observed relationship between the two labels
among molecules measured on both endpoints, without treating a nonsignificant association

<!-- PDF page 4 -->
test as proof of independence. Second, compare separate graph models with a hard-parameter-
sharing model under the same task definitions and report the limited power of repeated-seed
comparisons. Third, evaluate compound-level decision quality on the complete dual-labeled
out-of-fold subset, emphasizing errors that advance a BBB-positive efflux substrate as if it were
an unimpeded CNS candidate. Fourth, place those results in the context of the repository’s
independently developed XGBoost screening system, feature ablations, model explanations and
exploratory docking.
The motivating transfer hypothesis was that the smaller P-gp task might benefit from a
representation learned on the much larger BBB dataset. The counter-hypothesis is that if the
measured labels encode weakly associated properties, shared parameters may offer little trans-
ferable signal or may create negative transfer. The work treats both as testable expectations,
not as established biological explanations.

## 2 Data Sources and Preparation

The graph-learning benchmark combined the B3DB classification data with the project’s P-
gp table, distributed through the Therapeutics Data Commons (TDC) ecosystem [2, 15, 16].
Canonical molecular identity was used to merge records, and each task retained an explicit
label mask. The graph corpus contained 7,805 BBB-labeled molecules and 1,212 P-gp-labeled
molecules, with 216 molecules measured on both tasks, yielding 8,801 unique structures. The
validated descriptor/XGBoost branch uses its own processed files (7,805 BBB and 1,205 P-gp
structures); those counts differ from the GNN assembly and are kept separate rather than silently
combined.
For this paper, positive P-gp labels are interpreted as substrate/efflux-positive and negative
labels as non-substrate/efflux-negative, following the project authors’ clarification. This is the
intended meaning of the repository’s task label; the scientific source attribution should still be
checked against the exact TDC version and label mapping because the cited Broccatelli article
describes inhibitor prediction rather than substrate identification. Results should be read as
performance against the project-defined annotation, not as a new experimental measurement of
transport.
Input structures were parsed and canonicalized; invalid or unlabelled records were excluded,
and duplicate structures with conflicting labels were not resolved by row order. The dual-
task assembly represented absent labels with masks rather than imputing them as negative.
The graph corpus recorded 1,058 available LogBB values, but continuous LogBB regression was
outside the reported experiments. This avoids conflating a thresholded BBB classification target
with a continuous permeability estimate.
Dataset partitions were molecule-level: a structure was assigned as a whole and never split
into different task-specific train/test identities. Random partitions assess interpolation across
a largely shared chemical space; Bemis–Murcko scaffold partitions test transfer to frameworks
absent from training [17]. For the GNN single-run comparison, the common random protocol
allocated 6,160/1,320/1,321 molecules to train/validation/test, with 33 dual-labeled test cases;
the scaffold protocol allocated the same totals, with 48 dual-labeled test cases and no scaffold
overlap between train and test. The five-fold random out-of-fold (OOF) analysis was used for
joint decision metrics because it supplied held-out predictions for all 216 shared molecules.

<!-- PDF page 5 -->
T able 1:Graph-learning corpus composition and the dual-labeled contingency table.
Source/stratum BBB labels P-gp labels Unique molecules
B3DB benchmark 7,805 – 7,805
P-gp benchmark – 1,212 1,212
Shared structures 216 216 216
Outer-joined corpus 7,805 1,212 8,801
Dual-labeled group P-gp + P-gp− Total
BBB+ 45 95 140
BBB− 19 57 76
Total 64 152 216
B3DB
7,805 BBB-
labeled structures
Project P-gp
benchmark
1,212 labeled
structuresSMILES valida-
tion, canonical
identity, dupli-
cate/conflict handling
Outer join with
task masks
8,801 structures;
216 dual-labeled
66 atom / 14
bond features
and graph construction
Independent
single-task
GNNs
Shared GINE encoder
with two task heads
Five-fold OOF
predictions
and joint de-
cision analysis
Figure 3:Data assembly and evaluation flow. Every arrow terminates at a box border; task-specific
labels remain masked when unobserved.

<!-- PDF page 6 -->

### 2.1 Sources of heterogeneity and scope of labels

BBB measurements can reflect different experimental conditions, classification thresholds and
compound provenance. B3DB is a curated resource rather than a single uniform permeability
experiment; a binary class necessarily discards continuous variation and protocol metadata [2].
P-gp annotations also depend on assay type and curation. A substrate assay, an inhibition
assay, a cellular accumulation assay and an in vivo brain-to-plasma measurement are related
but not interchangeable endpoints. The project-level label mapping is used consistently in
this analysis, but external validation against a second, explicitly documented substrate dataset
remains necessary.
The outer join contains 7,589 BBB-only structures and 996 P-gp-only structures in addition
to the 216 shared structures. Thus, approximately 97.6% of corpus records have only one
observed endpoint. This imbalance is not a missing-at-random guarantee: the two benchmarks
were curated separately, and chemical coverage can differ. The masked loss prevents an absent
label from becoming a false negative, but it cannot remove dataset-specific selection bias or
assay heterogeneity.

## 3 Model Selection: Development Journey

Model selection followed a sequence of increasingly joint formulations, rather than assuming
that the most complex architecture would be best.
Starting point: graph-based BBB prediction.The project began by reproducing a
GCN-style BBB classifier based on the supplied GCN-BBB paper and the repository’s legacy
dataset/code. This established a graph-learning baseline and informed the subsequent expansion
from one endpoint to two [1].
Independent endpoint models.The project then developed separate BBB and P-gp
classifiers. The screening workflow uses 2,048-bit Morgan fingerprints with six 2D descriptors
for BBB, and adds six 3D conformer descriptors for the full P-gp feature set, with XGBoost
classifiers producing separate endpoint scores that can be composed into a four-quadrant result
[18]. These models provide an independent-task comparison and a practical screening system.
Shared-encoder hypothesis.Because only 216 molecules have both labels, the two tasks
were assembled into a molecule-level corpus with a shared graph encoder and task-specific output
heads. A masked loss lets each head learn from all molecules labeled for its endpoint without
treating a missing label as a negative. This hard-parameter-sharing multi-task GNN adds a
second prediction head to a graph representation [19].
Selection outcome.The multi-task model was retained as an experimental comparison,
not selected as a superior replacement. Its per-task differences from single-task GNNs were small
and inconsistent across random and scaffold splits; increasing the P-gp task weight did not yield
a stable improvement. The journey thus motivates reporting both the simpler independent
models and the multi-task result, including its null finding.

### 3.1 Descriptor-based screening models

The repository’s deployable screening system uses two independently trained XGBoost classi-
fiers [18]. Both begin with a 2,048-bit Morgan fingerprint (radius 2, conventionally referred to
as ECFP4) and six 2D descriptors: molecular weight, calculated LogP, topological polar surface

<!-- PDF page 7 -->
GCN-BBB
replication
single BBB endpoint
Separate classifiers
BBB + P-gp
Shared graph
encoder
two task-
specific heads
Decision-aware
comparison
no clear multi-
task gain
Figure 4:Model-selection path from the single-task baseline to the tested multi-task architecture.
area (TPSA), hydrogen-bond donors, hydrogen-bond acceptors and rotatable bonds. The BBB
vector therefore contains 2,054 model features. The P-gp vector adds six geometry descriptors
from a generated 3D conformer—radius of gyration, asphericity, eccentricity, spherocity index
and the two normalized principal moments of inertia—for 2,060 features. Conformers are gen-
erated deterministically using ETKDGv3 with a fixed random seed, then minimized using the
project pipeline’s force-field workflow [20–22].
The BBB XGBoost model was tuned using Optuna within a train/validation/test design;
the final model uses early stopping and evaluates a held-out test partition. The separate P-gp
XGBoost model uses the processed P-gp descriptor set and its own training protocol. The two
models are not assumed to form a statistically calibrated joint probability. Their probability
scores are combined only to construct predicted binary quadrants at the selected threshold, and
their validation populations differ from the graph OOF comparison. For that reason, XGBoost
values are reported as a separate project arm rather than a controlled architecture contest.

### 3.2 Graph featurization

For graph learning, each molecule is represented as an undirected chemical graph encoded as
directed edges for message passing. The encoded feature mode used in the reported experiments
represents each atom by 66 values: one-hot blocks for element, degree, formal charge, hydrogen
count, hybridization, chirality and minimum ring size, followed by binary and bounded con-
tinuous flags. Each bond has 14 values describing bond type, stereo configuration and binary
flags. The graph encoder uses these categorical and local chemical attributes directly; it does
not receive the XGBoost Morgan fingerprint vector.
The earlier reference implementation used a legacy 9-atom/7-bond encoding. Experiments in
this paper use the newer 66/14 encoded mode, and the parameter count and feature dimensions
are recorded with the run reports. This distinction matters for replication: the baseline is
inspired by the supplied GCN-BBB reference, but it is not a claim of exact feature parity with
its legacy graph input.

### 3.3 Single-task and multi-task graph architectures

The graph networks use three edge-aware GINE message-passing layers with 256 hidden chan-
nels. Each layer applies an MLP-based neighborhood update, batch normalization, ReLU and
dropout (0.30); a global mean-pooling operation maps the atom representations to one 256-
dimensional graph embedding. A task-specific readout maps this vector through a 32-unit

<!-- PDF page 8 -->
hidden layer and a two-class output. The single-task models have 364,608 trainable parameters;
the two-head shared model has 372,898, a 2.3% increase over the single-head network. The
second head branches from the pooled graph embedding, so the two tasks share message passing
but retain distinct output layers.
For moleculei, letm it ∈ {0,1}indicate whether taskthas an observed label, and letℓ it be
its class-weighted cross-entropy. The optimized multi-task objective is
L(θ) =
X
t∈{BBB,Pgp}
αt
P
i mitℓit(θ)P
i mit
,(1)
with default task weightsα t = 1. Class weights are computed from the training partition
using inverse class frequency. The denominator is task-specific, so the larger BBB set does not
dominate solely by contributing more labeled rows. Missing labels are never converted into
negative labels. The shared trunk is therefore exposed to each task’s labeled subset, while only
observed target values contribute gradients to the corresponding head.
Molecular graph
66 atom / 14
bond features
Three edge-aware
GINE layers
256 hidden
channels
Global mean
pooling
256-dimensional
graph vector
BBB read-
out head
2-class logits
P-gp read-
out head
2-class logits
Masked, class-
weighted
cross-entropy on
observed labels
Figure 5:Shared-encoder multi-task GNN. The arrows route from the graph input through the shared
trunk into task-specific heads and the masked objective; lines terminate at node borders.

### 3.4 Optimization and model selection

Training used AdamW with learning rate 5×10 −4 and weight decay 5×10 −4, batch size 64,
at most 200 epochs, and early stopping after 20 validation epochs without improvement. A
ReduceLROnPlateau scheduler halves the learning rate after eight stagnant epochs, with a floor

<!-- PDF page 9 -->
of 10 −6; gradient norm is clipped at 2.0. The validation selection score is the mean F1 across
active tasks. The best validation checkpoint is restored before test evaluation. These values
are fixed in the shared training loop so single-task and multi-task experiments use the same
implementation.
T able 2:Core graph-network training configuration recorded in the experiment reports.
Setting Value
Graph encoder 3-layer edge-aware GINE
Hidden width / readout width 256 / 32
Dropout 0.30
Optimizer AdamW
Learning rate / weight decay 5×10 −4 / 5×10 −4
Batch size / maximum epochs 64 / 200
Early-stopping patience 20 epochs
Validation score Mean F1 over active task heads
Task weights (baseline MTL) BBB 1.0; P-gp 1.0
Gradient clipping Maximum norm 2.0

## 4 Model Evaluation

Per-task ranking was evaluated using ROC-AUC and area under the precision–recall curve (PR-
AUC); thresholded performance was summarized using accuracy, balanced accuracy, precision,
recall, specificity, F1 and Matthews correlation coefficient (MCC). ROC-AUC measures ranking
over thresholds, while PR-AUC is sensitive to class prevalence. Accuracy alone is insufficient
under class imbalance, so threshold-dependent metrics are provided alongside both curves.

### 4.1 Out-of-fold protocol and uncertainty

For pooled graph-model metrics, every labeled structure received one prediction from a model
that was not trained on that structure. Five-fold evaluation covered all 7,805 BBB labels and all
1,212 P-gp labels and produced predictions for all 216 dual-labeled molecules. Within each fold,
the training partition was further divided to select the checkpoint using the mean validation F1;
predictions were generated on the held-out fold only. Integrity checks confirmed unique OOF
assignment and no train/test prediction leakage. The pooled OOF results were bootstrapped
1,000 times for 95% confidence intervals; the intervals quantify uncertainty in the held-out
molecule sample and do not capture all sources of model or dataset uncertainty.
The separate random/scaffold XGBoost validation uses its own split-generation and fitting
procedure. The scaffold run trains a fresh model and is not a nested, retuned replica of the
random-split Optuna configuration. Consequently, random-to-scaffold differences describe the
saved project evaluations, not a perfectly controlled split-only experiment. The GNN single-split
scaffold values are another distinct evaluation and must not be numerically compared as though
they used the XGBoost partitions.

### 4.2 Joint endpoint and decision metrics

The dual-label contingency table was tested using Pearson’s chi-squared statistic, Fisher’s exact
odds ratio and Cram´ er’sV. The odds-ratio interval was calculated on the 2×2 table. A non-

<!-- PDF page 10 -->
significantp-value is not interpreted as proof that the endpoints are independent. For quadrant
evaluation, the binary predictions on the two axes were combined into Q1 (BBB +/P-gp−), Q2
(BBB+/P-gp+), Q3 (BBB −/P-gp−) and Q4 (BBB −/P-gp+). Exact match requires both axis
predictions to be correct; Cohen’sκsummarizes chance-corrected four-class agreement.
Paradox recall is the fraction of observed Q2 compounds predicted as Q2. Paradox leakage
is the fraction of true Q2 compounds predicted as Q1, the progression-eligible quadrant; leakage
is therefore a specific consequential error, not the complement of recall. The BBB-only baseline
supplies no P-gp prediction and assigns the negative P-gp class by assumption. Confidence
intervals for quadrant and paradox metrics were obtained from 1,000 bootstrap resamples of the
216 molecule-level records.
To make the implicit progression decision explicit, we also report a simple cost-sensitive
regret:
R(λ) = λNfalse progress +N missed Q1
N ,(2)
whereλis the relative cost of advancing a non-Q1 molecule versus missing a true Q1 molecule.
This is a transparent sensitivity analysis, not a clinical utility estimate; the repository contains
no elicited clinical cost ratio. Reporting a curve overλavoids presenting one arbitrary cost as
universal.
In five-fold out-of-fold evaluation, the single-task GNN reached BBB ROC-AUC 0.9379 (95%
CI 0.9323–0.9433) and P-gp ROC-AUC 0.9011 (0.8838–0.9190). The multi-task GNN reached
0.9363 (0.9312–0.9416) and 0.9056 (0.8885–0.9226), respectively. The intervals overlap sub-
stantially. For the 216 dual-labeled compounds, decision-aware evaluation additionally mea-
sured quadrant exact match, Cohen’sκ, paradox recall and the leakage rate into the favorable
BBB+/P-gp non-substrate quadrant.
The following confusion matrices are from the XGBoost models’ random held-out test sets;
they are not the GNN out-of-fold predictions summarized above.
Figure asset unavailable:
bbb_test_confusion_matrix.png
Figure asset unavailable:
pgp_test_confusion_matrix.png
Figure 6:XGBoost random held-out test-set confusion matrices for BBB (left) and P-gp (right).
T able 3:Pooled five-fold OOF performance for independent and multi-task GNNs. ROC-AUC confi-
dence intervals use 1,000 bootstrap resamples; other values are point estimates.
Endpoint ModelnROC-AUC (95% CI) PR-AUC Bal. Acc. Specificity MCC
BBB Single-task 7,805 0.9379 (0.9323–0.9433) 0.9571 0.8594 0.8192 0.7197
BBB Multi-task 7,805 0.9363 (0.9312–0.9416) 0.9587 0.8523 0.8077 0.7065
P-gp Single-task 1,212 0.9011 (0.8838–0.9190) 0.9076 0.8493 0.8407 0.6984
P-gp Multi-task 1,212 0.9056 (0.8885–0.9226) 0.9181 0.8337 0.8018 0.6698

<!-- PDF page 11 -->
Figure 7:P-gp ROC curve from the repository’s XGBoost evaluation. This is distinct from the graph-
model out-of-fold results in Table 3.

## 5 Results Across Models


### 5.1 Association between measured endpoint labels

Among the 216 dual-labeled molecules, 45 were labeled BBB +/P-gp+, 95 BBB +/P-gp−, 19
BBB−/P-gp+ and 57 BBB −/P-gp−. The observed proportion carrying a positive P-gp label
was 32.1% among BBB-positive compounds (45/140; exact binomial 95% CI 24.5–40.6%) and
25.0% among BBB-negative compounds (19/76). The table is imbalanced in a way consistent
with only weak association:χ 2 = 0.887,p= 0.346, Fisher odds ratio 1.42 (95% CI 0.76–2.67),
and Cram´ er’sV= 0.064. The interval remains wide enough to include moderate associations;
the data support the claim that BBB status is not a reliable substitute for the P-gp label, not
a claim of proven independence.

### 5.2 Single-split graph comparisons

The graph experiments evaluated random and scaffold-separated partitions with the same archi-
tecture family. On the random split, BBB ROC-AUC was 0.9388 for the single-task model and
0.9404 for multi-task learning, while P-gp ROC-AUC was 0.9390 and 0.9301, respectively. On
the scaffold split, BBB ROC-AUC was 0.7583 versus 0.7467, and P-gp ROC-AUC was 0.8227
versus 0.8171. Thus, hard sharing produced no consistent gain across tasks or partition types.
The scaffold GNN test sets contained 1,036 BBB labels and 333 P-gp labels, larger than the
48 dual-labeled records in that test set; joint quadrant metrics for this scaffold split were not
computed.
T able 4:Single-split graph-model ROC-AUC comparisons from the encoded-feature runs.
Task Split Single-task Multi-task Difference
BBB Random 0.9388 0.9404 +0.0016
P-gp Random 0.9390 0.9301−0.0089
BBB Scaffold 0.7583 0.7467−0.0116
P-gp Scaffold 0.8227 0.8171−0.0056
The random-to-scaffold decrease in the single-task GNN was 0.1805 ROC-AUC points for
BBB and 0.1163 for P-gp. This indicates that both axes become harder under the scaffold

<!-- PDF page 12 -->
protocol, with a larger observed loss for BBB. These split-specific results are separate from the
pooled random OOF metrics and from the descriptor-model scaffold results below; they do not
establish that P-gp generalizes universally better across chemical space.

### 5.3 Descriptor-model validation

Table 5 summarizes the repository’s descriptor-based XGBoost evaluations. The BBB model
retains a high ROC-AUC under the saved scaffold split, but its specificity falls to 0.5795 and
balanced accuracy to 0.7605. This illustrates why ranking metrics and the chosen classification
threshold answer different questions. The P-gp scaffold result is also dataset- and protocol-
specific and should not be generalized beyond the evaluated held-out set. Since the scaffold
evaluation used a fresh untuned model, the random/scaffold contrast is descriptive rather than
a strictly matched comparison.
T able 5:Selected metrics for the repository’s XGBoost validation splits. Each cell is a point estimate;
ROC-AUC 95% bootstrap CIs are shown in brackets.
Endpoint Split ROC-AUC [95% CI] PR-AUC Bal. Acc. Specificity MCC
BBB Random 0.9522 [0.9410–0.9629] 0.9724 0.8775 0.8478 0.7535
BBB Scaffold 0.9048 [0.8835–0.9240] 0.9668 0.7605 0.5795 0.5718
P-gp Random 0.9587 [0.9330–0.9806] 0.9685 0.9287 0.9189 0.8580
P-gp Scaffold 0.9631 [0.9359–0.9846] 0.9801 0.8978 0.9800 0.7843

### 5.4 Repeated-seed analysis and task weighting

To assess run-to-run variability, five random seeds were paired across single- and multi-task
training. Mean ROC-AUC differences favored the single-task model in both endpoints, but
confidence intervals on the mean differences overlapped zero: BBB ∆ =−0.0094 (95% CI
−0.0234 to 0.0045; pairedp= 0.133) and P-gp ∆ =−0.0027 (95% CI−0.0348 to 0.0294; paired
p= 0.826). With only five paired seeds, these comparisons are underpowered to resolve small
AUC changes. The same runs showed a consistent threshold-metric disadvantage for shared
learning on BBB: paired differences in MCC, F1 and balanced accuracy were negative, with
paired testsp= 0.0059, 0.0081 and 0.0097, respectively. Those threshold results are exploratory
and should be interpreted alongside the small seed count and the validation-selected thresholds.
T able 6:Five-seed paired random-split ROC-AUC contrasts (multi-task minus single-task).
Endpoint Mean difference 95% CI PairedpMDEP(MT>ST)
BBB−0.00944 [−0.02337,0.00449] 0.1331 0.01865 0.20
P-gp−0.00272 [−0.03479,0.02936] 0.8256 0.04295 0.54
The minimum detectable effect (MDE) is the estimated difference resolvable with 80% power
under the recorded paired-seed variability and the stated testing assumptions. It is a design
diagnostic, not a post-hoc guarantee. The estimated MDE exceeded the absolute observed
AUC difference for both tasks, so a failure to reject a difference cannot establish equivalence.
Conversely, the repeated-seed threshold metrics suggest the BBB decision boundary can be more
sensitive to parameter sharing than its rank ordering.

<!-- PDF page 13 -->
The P-gp task was upweighted to test whether its smaller sample size explained the absence
of transfer. Increasing its loss weight to 3 or 5 did not rescue P-gp ROC-AUC and reduced BBB
performance. A zero-weight sanity run left the P-gp head near chance while retaining BBB
performance, supporting the claim that task weighting was active as configured.
T able 7:P-gp loss-weight sensitivity on the random GNN split. The zero-weight run is an implementa-
tion check, not a valid P-gp predictor.
P-gp task weight BBB ROC-AUC P-gp ROC-AUC Interpretation
0 0.9411 0.5626 P-gp head receives no task loss
1 0.9404 0.9301 Equal-weight multi-task baseline
3 0.9310 0.9295 No P-gp recovery; BBB declines
5 0.9086 0.9299 Greater BBB degradation

### 5.5 Feature ablation

The descriptor ablation tested whether added feature blocks improved discrimination in the
XGBoost experiments. For BBB, combining Morgan fingerprints with six 2D descriptors gave
ROC-AUC 0.9467, exceeding either the 2D-only model (0.9174) or Morgan-only model (0.9393).
For P-gp, Morgan plus 2D features reached 0.9509; the full 2D+3D+fingerprint representation
reached 0.9481. Three-dimensional shape descriptors alone were informative (0.8425) but did
not improve the full P-gp result when appended to the fingerprint-plus-2D baseline. Differences
are modest relative to their reported fold variation; these values do not prove that 3D shape has
no value for any compound or use case.
T able 8:Feature-ablation ROC-AUC (mean±standard deviation) from the repository experiments.
Endpoint Feature configuration Features ROC-AUC
BBB 2D descriptors 6 0.9174±0.0081
BBB Morgan fingerprints 2,048 0.9393±0.0054
BBB Morgan + 2D 2,054 0.9467±0.0035
P-gp 2D descriptors 6 0.9185±0.0107
P-gp 3D shape descriptors 6 0.8425±0.0243
P-gp Morgan + 2D 2,054 0.9509±0.0070
P-gp Morgan + 2D + 3D 2,060 0.9481±0.0076
T able 9:Joint decision results on 216 dual-labeled molecules using random-split OOF predictions.
Bootstrap intervals use 1,000 resamples.
Decision strategy Exact match
(95% CI)
Q2 recall (95%
CI)
Q2 leakage (95%
CI)
BBB-only; P-gp as-
signed negative
55.6% (47.7–62.0) 0.0% (0.0–0.0) 93.3% (86.5–
100)
Composed indepen-
dent GNNs
57.9% (50.9–64.8) 44.4% (29.5–
60.5)
48.9% (34.2–
63.9)
Shared-encoder
multi-task GNN
56.0% (49.5–62.5) 51.1% (36.6–
66.7)
42.2% (26.7–
56.8)

<!-- PDF page 14 -->

### 5.6 Four-quadrant error structure

Overall exact match obscures which decision is wrong. Table 10 therefore reports the full four-
class confusion matrices, with rows representing observed quadrants and columns representing
predictions. The BBB-only baseline cannot predict Q2 or Q4 by construction. It assigns every
molecule a negative P-gp label, so its 42 Q2-to-Q1 errors are the direct consequence of sub-
stituting permeability for efflux. The two-axis models reduce that failure but still misclassify
approximately half of Q2: 22 cases for composed independent models and 19 for the shared
model. The multi-task model’s slightly lower leakage comes with a lower exact match than the
composed single-task predictions; the results do not establish an overall superiority of either
two-axis architecture.
T able 10:Out-of-fold four-quadrant confusion matrices on the shared set. Rows are observed quadrants;
columns are predicted quadrants.
Strategy Observed Pred Q1 Pred Q2 Pred Q3 Pred Q4
BBB-only
Q1 84 0 11 0
Q2 42 0 3 0
Q3 21 0 36 0
Q4 5 0 14 0
Composed
Q1 66 18 8 3
Q2 22 20 3 0
Q3 16 5 33 3
Q4 2 3 8 6
Multi-task
Q1 64 18 11 2
Q2 19 23 2 1
Q3 19 2 29 7
Q4 4 5 5 5
The BBB score was further stratified by the observed P-gp class. For the composed indepen-
dent models, BBB ROC-AUC was 0.7990 among P-gp-negative molecules (n= 152) and 0.9368
among P-gp-positive molecules (n= 64), a difference of−0.1379 (95% bootstrap CI−0.2406
to−0.0410 when computed as non-substrate minus substrate). For the multi-task model the
corresponding values were 0.7904 and 0.8760, with a difference of−0.0856 (95% CI−0.1992 to
0.0247). This exploratory analysis runs counter to the simple expectation that efflux-positive
molecules must be harder for a BBB classifier to rank. Label-stratified performance is influenced
by chemical composition and the small number of shared compounds; it does not measure net
exposure or establish a causal effect of efflux on the permeability label.
In the dual-labeled set, the BBB-positive subset carrying a positive P-gp label had higher
mean molecular weight (370.9 versus 281.4 Da), higher LogP (3.8 versus 2.2), lower TPSA
(47.8 versus 60.3 ˚A2), fewer hydrogen-bond donors (0.8 versus 1.4) and more rotatable bonds
(5.2 versus 3.9) than the P-gp-negative subset. These descriptive differences help interpret the
stratified score result but are not adjusted causal estimates. They also illustrate why a model
can correctly separate a permeability label and still omit the active-efflux decision.

### 5.7 Decision regret under varying progression costs

Equation 2 makes the trade-off between false progression and missed Q1 opportunities explicit.
Atλ= 1, regret was 0.3657 for BBB-only, 0.3194 for the composed model and 0.3380 for
multi-task learning. Asλincreases, the cost of advancing a false candidate rises, and the two-
axis strategies become preferable. The crossover with BBB-only occurs nearλ= 0.643 for the

<!-- PDF page 15 -->
composed strategy and 0.769 for the multi-task strategy in this sample. These crossovers are
conditional on the stated binary loss and observed error counts; they must not be interpreted
as clinically estimated cost ratios.
T able 11:Normalized decision regret at selected false-progression cost ratios. Lower values are better
only under the stated cost function.
λBBB-only Composed Multi-task
0.25 0.1296 0.1806 0.1921
0.50 0.2083 0.2269 0.2407
1.00 0.3657 0.3194 0.3380
2.00 0.6806 0.5046 0.5324
4.00 1.3102 0.8750 0.9213
8.00 2.5694 1.6157 1.6991
16.00 5.0880 3.0972 3.2546
Repository TreeSHAP summaries ranked TPSA highest among the listed shared descriptors
for the BBB model (mean absolute SHAP 0.9875) and MolLogP highest for the P-gp model
(1.4074); MolWt was also attributed in both (BBB 0.6955, P-gp 0.6292). These are model-
specific feature attributions on the reported scale, not causal effects or evidence of transporter
mechanism.
The repository also screened 2,405 approved compounds and assigned 273 non-CNS-labeled
drugs to the predicted BBB +/P-gp-negative quadrant. These are in-silico prioritization results,
not validated repurposing candidates; substantial overlap with benchmark data means the screen
is not an external test. The full predicted quadrant distribution and its overlap with the training
benchmarks are reported below.

## 6 Visual Results

The following panels reproduce the project’s current figure set. They show model explanations,
feature ablation, classification diagnostics, the approved-drug landscape and the structural dock-
ing analysis. Plots are descriptive: explanation scores are not causal effects, and docking scores
are not experimental binding affinities.
Figure 8:Global feature attributions for the BBB
model.
Figure 9:Global feature attributions for the P-
gp model.

<!-- PDF page 16 -->
Figure 10:Comparison of shared molecular de-
scriptors across tasks.
Figure 11:Feature-set ablation results.

### 6.1 Screening and structural-analysis figures

Predicted quadrants and disease-area distributions from the 2,405-compound repository screen
are shown below. The docking figures summarize the repository’s calculations against human
P-gp structure 6QEX [7, 23, 24]. They are computational comparisons and do not establish
transporter activity in a biological assay.
Figure 12:Joint BBB/P-gp quadrant distribu-
tion.
Figure 13:Predicted dual-barrier landscape of
approved compounds.

<!-- PDF page 17 -->
Figure asset unavailable:repurposing_
disease_area_distribution.png
Figure 14:Disease-area distribution in the repos-
itory screening results.
Q2 (BBB+/P-gp+)
n=10
Q1 (BBB+/P-gp−)
n=8
Q3 (BBB−/P-gp−)
n=3
−11
−10
−9
−8
−7
−6
−5
Vina ΔGbind (kcal/mol)
Figure 15:Docking-score distribution for the se-
lected compounds.
0.0 0.2 0.4 0.6 0.8 1.0
ML P-gp probability
−11
−10
−9
−8
−7
−6
−5
Vina ΔGbind (kcal/mol)
Levetiracetam
LoperamidePenicillin G
Vilazodone
Betamethasone
Spearman ρ = -0.596, p = 0.0043 (n = 21)
Q1 (BBB+/P-gp−)
Q2 (BBB+/P-gp+)
Q3 (BBB−/P-gp−)
Figure 16:Relationship between model output
and docking score.

### 6.2 Screening output and dataset overlap

The virtual screen assigned 710 compounds to Q1, 240 to Q2, 1,064 to Q3 and 391 to Q4. Of
these, 273 compounds annotated as non-CNS approved drugs were included in Q1 and retained
as hypotheses for further evaluation. The result is a prioritization list rather than a repurposing
claim: benchmark training overlap, domain distance and known pharmacology must be checked
compound by compound.
The repository audit reports that 1,330 screened molecules occur in B3DB and 354 occur in
the P-gp model’s training data. Among the Q1 non-CNS leads, 160 overlap the B3DB bench-
mark; among the 105 marketed neuropsychiatric compounds assigned to Q2, 27 occur in the P-gp
training set. These overlaps limit the screen’s evidential independence. The classifications can
still be operationally useful for inspecting pipeline behavior, but a prospective or truly external
candidate list must exclude known benchmark compounds and be evaluated with experimentally

<!-- PDF page 18 -->
T able 12:Predicted quadrants in the 2,405-compound approved-drug screen.
Quadrant Predicted interpretation Count Fraction
Q1 BBB +/P-gp− 710 29.5%
Q2 BBB +/P-gp+ 240 10.0%
Q3 BBB −/P-gp− 1,064 44.2%
Q4 BBB −/P-gp+ 391 16.3%
Total 2,405 100.0%
independent measurements.
T able 13:Known overlap between the approved-drug screen and model benchmark sets, as recorded in
the repository audit.
Screen subset Benchmark overlap Total subset
All approved compounds in B3DB 1,330 2,405
All approved compounds in P-gp training data 354 2,405
Predicted Q1 non-CNS leads in B3DB 160 273
Predicted Q2 neuropsychiatric compounds in P-gp training data 27 105

### 6.3 Feature attribution and interpretation

TreeSHAP values explain the trained tree ensemble’s predictions in its output-margin space [25].
The global BBB summary ranks TPSA among the leading descriptors, whereas the P-gp model
places LogP prominently. In the repository’s quantitative comparison of shared properties, mean
absolute SHAP was 0.9875 for BBB TPSA and 1.4074 for P-gp MolLogP; the corresponding
values for MolWt were 0.6955 and 0.6292. These summary magnitudes are model- and scale-
dependent. They do not constitute causal effects, identify a unique transporter-recognition
motif, or say that lowering TPSA or changing LogP will improve a compound without affecting
other properties.
The feature-attribution analysis also has the usual limitations of correlated descriptors and
fingerprint features. When correlated inputs share predictive information, attribution can be
distributed among them in a model-dependent way. A global mean absolute attribution removes
the sign and averages across molecules, so it cannot reveal whether a feature raises or lowers a
prediction in a particular chemical series. Local explanation and a substructure mapping can
support medicinal-chemistry review, but any proposed edit remains a computational hypothesis
requiring synthesis and measurement.

### 6.4 Structural docking as exploratory analysis

The repository docked a small set of selected molecules against the human P-gp cryo-electron
microscopy structure 6QEX [7, 23, 24, 26]. Docking scores and residue-contact counts summarize
poses under a prepared, largely fixed receptor model. They are not experimentally measured
binding free energies, transport rates or proof that a molecule is a substrate; substrate and
inhibitor binding also cannot be separated solely by a favorable score in one receptor conforma-
tion.
The raw docking CSV contains 22 rows; the accompanying complete-case analysis reports

<!-- PDF page 19 -->
n= 21. In that audit, heavy-atom count was strongly associated with more favorable docking
score (Spearmanρ=−0.817,p <10 −5) and with model P-gp probability (ρ= 0.757,p <
10−4). After controlling for heavy-atom count using the reported partial-rank analysis, the
association between predicted P-gp probability and docking energy was near zero (ρ= 0.060,
p= 0.798). Accordingly, the apparent raw relationship cannot be presented as independent
structural confirmation of the classifier.
The chosen receptor conformation, protonation and charge assignment, ligand conformer
generation, search box and scoring function all influence the resulting score. The analysis also
has small sample size, possible benchmark overlap, and a size-composition imbalance between
predicted groups. A stronger structural study would use an independently selected and size-
matched compound set, multiple receptor conformations, replicate docking settings and experi-
mental transporter assays. Docking is retained here as a pipeline demonstration and a cautionary
example of confounding, not as validation.

## 7 Discussion


### 7.1 Two distinct biological axes should remain distinct in the model

The dual-label subset contains clear counterexamples to the idea that BBB-positive status is
interchangeable with efflux-negative status: 45 of 140 BBB-positive molecules (32.1%) are also
annotated P-gp-positive. The weak association statistics reinforce that the two benchmark labels
should be measured and reported separately. The label count does not establish independence in
the population, and the limited overlap leaves wide uncertainty; nevertheless, the practical find-
ing is robustly framed as a prevalence in this sample rather than an overinterpreted significance
test.
The user-confirmed interpretation of P-gp-positive as an efflux substrate is biologically co-
herent with the four-quadrant motivation. However, the exact benchmark lineage remains a
central reproducibility issue: the referenced Broccatelli publication describes inhibition pre-
diction, whereas the project’s task is presented as substrate status. This paper reports the
project-defined label semantics and the computed results, but it cannot independently certify
that the raw TDC target was mapped correctly. A conference submission should cite the exact
TDC task/version and preserve the upstream label mapping or identify a substrate assay source.

### 7.2 Per-task discrimination does not equal joint decision accuracy

The graph models attain high pooled OOF ROC-AUC on each individual endpoint, yet the
two-axis exact-match rate on the 216 shared molecules is only 55.6–57.9%. These are not
contradictory measures: ROC-AUC evaluates ranking on one axis over thresholds, whereas
exact match requires two binary calls on the same molecule to be simultaneously correct. A
favorable BBB score therefore cannot answer whether the compound is effluxed.
The BBB-only baseline makes the implication explicit. By assigning P-gp-negative to every
molecule, it has zero recall for Q2 and leaks 42 of 45 Q2 molecules into Q1. The two-model and
shared-encoder strategies reduce leakage to 22 and 19 molecules respectively, but their confidence
intervals overlap and their overall exact-match estimates differ by only a few percentage points.
The central practical implication is to report the efflux axis and the joint error profile alongside
permeability metrics, rather than to claim that a second classifier solves the decision problem.

<!-- PDF page 20 -->
Quadrant calls should also be presented with their component probabilities and, where pos-
sible, uncertainty or applicability-domain flags. The binary cut at the default prediction thresh-
old can obscure borderline molecules, and the current cross-task system has not been calibrated
jointly against an in vivo exposure endpoint. A Q1 call is thus a screening category, not a clinical
recommendation.

### 7.3 Interpretation of the multi-task result

The transfer hypothesis was plausible because the BBB-labeled corpus is substantially larger
than the P-gp-labeled corpus and both tasks are defined over molecules. Empirically, pooled
OOF ROC-AUC changed by−0.0016 for BBB and +0.0045 for P-gp under multi-task learn-
ing. Single-split results show the absence of a consistent advantage across random and scaffold
settings, while five-seed comparisons show that the AUC contrasts remain imprecise. The de-
fensible conclusion is that these experiments did not demonstrate a reliable ROC-AUC benefit
from hard parameter sharing.
The lower BBB MCC, F1 and balanced accuracy across five paired seeds raise a separate,
threshold-sensitive concern: shared learning may make the BBB decision boundary worse even
when AUC is similar. Five seeds are not enough to establish the magnitude or universality of
that effect, but this evidence cautions against using a single ranking metric to conclude that
the models are interchangeable. More task weighting did not restore P-gp performance and
coincided with lower BBB AUC. The zero-weight control produced a near-chance P-gp score
while retaining BBB performance, reducing concern that the task-weight implementation was
inert.
The near-zero label association is a possible explanation for limited transfer, but it does
not prove a mechanism of negative transfer. Label correlation is only one factor in multi-task
learning; shared low-level chemistry, task noise, differences in assay provenance, representation
capacity, optimization, and sampling can also determine whether shared parameters help. A
future design could test soft sharing, task-specific adapters, uncertainty-weighted losses, or a
larger curated dual-label set, but those alternatives were not evaluated here.

### 7.4 Generalization and threshold behavior

The scaffold experiments emphasize the importance of evaluation protocol. For the single-
task graph models, scaffold-split ROC-AUC was lower than random-split ROC-AUC for both
endpoints, more strongly for BBB. The XGBoost BBB scaffold ROC-AUC is still relatively high,
but the thresholded specificity falls to 0.5795. That combination demonstrates how a model can
preserve ranking while assigning too many negatives to the positive class at a fixed threshold.
The random and scaffold comparisons are not fully matched because the descriptor validation
trains a fresh model and the graph test partitions are independent; they should be reported as
separate pieces of evidence.
The single-split scaffold P-gp test set includes 333 labeled structures and the BBB test set
1,036; the dual-labeled subset is only 48. The joint OOF analysis, by contrast, uses a random
five-fold protocol. Thus, the paper does not claim that its quadrant leakage figures persist under
scaffold shift. Future work must compute the same joint decision metrics on scaffold-held-out
dual-labeled compounds and preferably use an external chemical series.

<!-- PDF page 21 -->

### 7.5 Feature representation and explanations

The feature ablation supports a practical, not mechanistic, conclusion: Morgan fingerprints
plus simple 2D descriptors form a strong baseline for both tasks, and the additional 3D shape
descriptors did not improve the reported P-gp ROC-AUC. The fact that 3D-only descriptors
achieved 0.8425 suggests that shape encodes some signal in this benchmark, but the full feature-
set comparison does not justify claiming that 3D modeling improved prediction. Nor does the
comparison rule out value from conformer ensembles, transporter-specific structural features or
improved assay labels.
The SHAP summary suggests different predictive associations across the two tree models,
with TPSA prominent in the BBB attribution and LogP prominent for P-gp. This pattern
is broadly compatible with the conceptual distinction between desolvation/permeability and
transporter recognition, but the model explanation is not a mechanistic experiment. Correlated
descriptor effects, training-set composition and label provenance limit biological interpretation.
The SHAP figures are best used to audit model behavior and formulate hypotheses for subse-
quent testing.

### 7.6 Virtual screening and structural analyses

The approved-drug screen demonstrates that the trained classifiers can be applied at scale
and organize compounds into four interpretable categories. Its value is hypothesis generation;
the large benchmark overlap means it is not an independent validation of the candidate list.
A stronger repurposing analysis would conduct a leave-known-training-compounds-out screen,
state the applicability-domain status, compare against established CNS annotations and evalu-
ate a blinded set experimentally.
Similarly, exploratory docking should not be used as a second line of evidence for the classifier
when heavy-atom count explains a substantial part of the raw score association. The size-
adjusted correlation reported by the repository is close to zero. The use of one prepared structure
and approximate docking score adds structural context, but does not discriminate substrate
from inhibitor or establish active transport. This more restrained interpretation is important
for conference review because the raw group-separation numbers can appear persuasive while
being confounded.

## 8 Limitations and Future Work


### 8.1 Endpoint mapping and assay harmonization

The most important limitation is the provenance of the P-gp label. Although the project authors
define P-gp-positive as efflux/substrate-positive, the cited Broccatelli paper’s title describes
inhibitor prediction. The manuscript therefore distinguishes project-specific label semantics
from independently verified assay semantics. The precise raw file, TDC release, target name,
positive-class conversion and assay definition must accompany any public release of the 216-
molecule overlap set. A carefully curated substrate-specific benchmark should be used to validate
the key result before the physiological interpretation is generalized.
The BBB and P-gp datasets were curated separately and may differ in assay systems, pro-
tocols, compound sourcing and class thresholds. A binary model cannot recover quantitative

<!-- PDF page 22 -->
permeability or transport kinetics from labels alone. The small dual-labeled set is vulnerable
to sampling variation and represents only a narrow intersection of the two source collections.
Larger, harmonized data with experimental metadata and matched compounds are required.

### 8.2 Evaluation design and statistical power

The OOF analysis covers all molecules under a random split, but the resulting folds may share
close analogs across train and test. The scaffold test reduces this risk for per-task model evalu-
ation, yet the paper lacks scaffold OOF quadrant metrics and a prospective external test. Five
repeated seeds provide only limited power to resolve small AUC differences; the MDE estimates
illustrate this limitation. Bootstrap intervals on the same test compounds quantify sampling
uncertainty but do not replace independent validation, correct for all model-selection choices,
or account for dataset shift.
The association analysis is likewise underpowered to establish a small relationship or equiva-
lence. The odds-ratio interval from 0.76 to 2.67 includes both near-null and moderately elevated
associations. Accordingly, we do not use the phrase “statistically independent.” The observed
proportions and effect sizes should be re-estimated on a larger, independently curated overlap.

### 8.3 Biological scope and clinical translation

BBB permeability and P-gp substrate status are components of brain exposure, not complete
determinants of pharmacological efficacy. Transport by BCRP and other ABC transporters,
passive influx, plasma and brain binding, metabolism, molecular ionization, dosing, species
differences and target potency all contribute. The project does not measure unbound brain-
to-plasma partition coefficient (K p,uu), transporter kinetics, exposure-response or therapeutic
outcome. A Q1 prediction cannot establish that a compound reaches an effective concentration
or is safe for a CNS indication.
No external in vitro transport experiment was performed in this study, and docking is not a
substitute for such an experiment. The screened compounds may be outside model applicability
domains or may have adverse properties unrelated to BBB penetration. Candidate-specific
claims should be deferred until those issues are evaluated.

### 8.4 Prioritized next steps

The next analysis should verify the P-gp source and class mapping; produce a versioned, dedupli-
cated release of the 216 shared molecules; and compute scaffold-held-out quadrant metrics with
fold integrity checks. A second priority is an external test using an explicitly substrate-labeled
P-gp dataset and measured BBB endpoints. The model comparison should expand paired seeds
and include calibration, threshold selection confined to validation folds, and pre-specified joint
costs. Finally, candidate compounds should be tested in transporter assays and, where feasible,
against unbound brain exposure; docking should be repeated on size-matched compounds and
multiple receptor conformations only if it informs a clearly stated structural hypothesis.

## 9 Conclusion

This paper evaluates a two-axis CNS screening problem using project-defined BBB and P-gp
labels, compares separate and shared-encoder graph models, and measures the compound-level

<!-- PDF page 23 -->
decisions obscured by single-axis scores. The two labels show weak observed association in
the 216-molecule intersection, and hard parameter sharing does not show a reliable pooled OOF
ROC-AUC advantage. Per-axis discrimination nevertheless coexists with modest exact quadrant
accuracy and substantial Q2-to-Q1 leakage under the BBB-only baseline. Reporting the P-gp
axis alongside BBB permeability therefore makes a materially different decision visible.
The conclusion is deliberately bounded: the data do not establish label independence, a
clinical exposure prediction, or a superiority of one two-axis model. The project-defined P-
gp semantics are accepted here following the authors’ clarification, but the source mapping
remains a reproducibility caveat. Larger dual-labeled benchmarks, external scaffold validation
and experimental transporter/brain-exposure measurements are necessary before repurposing or
clinical claims can be made.
Data and code availability
Project code, benchmark-processing scripts, model artifacts, figures and result summaries are
available in the public repository [27]. The repository includes the OOF prediction files and
integrity reports needed to reproduce the joint-decision calculations. Raw upstream data remain
subject to their original source licenses and terms. The manuscript’s numerical claims should
be read alongside the specific saved artifact and split protocol named in the corresponding table
caption.
A Metric definitions and quadrant accounting
Lety i and ˆyi denote the observed and predicted binary outcomes for a task. Sensitivity is
TP/(TP + FN), specificity is TN/(TN + FP), balanced accuracy is the average of sensitivity
and specificity, and MCC summarizes all four entries of the binary confusion matrix. ROC-
AUC estimates the probability that a randomly selected positive receives a higher score than
a randomly selected negative. PR-AUC summarizes the precision–recall curve and depends on
the prevalence of positives in the evaluated sample.
For the joint evaluation, quadrant exact match is
QEM = 1
N
NX
i=1
⊮[ˆqi =q i],
whereq i is the observed combination of the two binary labels. Q2 recall is TP Q2/NQ2. Q2
leakage is
LeakQ2→Q1 = #{i:q i =Q2,ˆq i =Q1}
NQ2
.
Leakage is a subset of Q2 false negatives, but not all missed Q2 predictions are Q1: some are
assigned to Q3 or Q4. This distinction explains why paradox recall and leakage need not sum
to one.
B Reproducibility and experiment provenance
The principal graph experiments used the encoded 66/14 atom/bond featurization, 256 hidden
channels, three GINE layers and seed 42 for the primary fixed-split and five-fold runs. The OOF

<!-- PDF page 24 -->
T able 14:Operational definitions of the four quadrants used throughout the analysis.
Quadrant BBB label P-gp label Screening interpretation
Q1 Positive Negative Permeable; no labeled efflux liability
Q2 Positive Positive Permeable; labeled efflux substrate
Q3 Negative Negative Permeability-limited under these labels
Q4 Negative Positive Both permeability-negative and efflux-positive
report records fold assignments, best epochs, training configuration, class weights, hardware and
data cardinalities. Five-seed variance analysis used paired seeds 42–46 and a random split. The
reported task-weight sensitivity runs altered only the P-gp loss multiplier in the corresponding
training configuration.
The repository’s five-fold OOF integrity report records 8,801 molecules scored exactly once,
all 216 dual-labeled compounds scored on both axes, and zero train/test prediction leaks. For
the single-task runs, all 7,805 BBB labels and 1,212 P-gp labels were covered. These checks
address bookkeeping integrity; they do not guarantee absence of close analogs across random
folds or remove overlap between the approved-drug screen and model training data.
T able 15:Primary result artifacts and the claim each supports.
Artifact Manuscript use
gnn_single_bbb_kfold5_
encoded_report.json
Single-task BBB pooled OOF metrics
gnn_single_pgp_kfold5_
encoded_report.json
Single-task P-gp pooled OOF metrics
gnn_multitask_kfold5_
encoded_report.json
Shared-encoder pooled OOF metrics and integrity checks
gnn_decision_metrics_
report.json
Quadrant, leakage, regret and stratified AUC summaries
gnn_experiment_comparison.
csv
Fixed random/scaffold graph comparisons
gnn_seedvar_random_report.
json
Paired five-seed uncertainty and MDE summaries
validation_metrics_summary.
csv
Descriptor-model random/scaffold metrics
ablation_study_results.csvDescriptor/fingerprint/3D feature ablations
shared_properties_
quantitative_comparison.csv
Shared-descriptor SHAP summaries
cryoem_pgp_docking_
validation.csv;extra_
analysis.json
Exploratory docking and size-adjusted association
References
[1] Yankang Jing, Guangyi Zhao, Yuanyuan Xu, Terence McGuire, Ganqian Hou, Jack Zhao, Maozi
Chen, Oscar Lopez, Ying Xue, and Xiang-Qun Xie. GCN-BBB: Deep learning blood-brain barrier
(BBB) permeability PharmacoAnalytics with graph convolutional neural (GCN) network.The AAPS
Journal, 27:73, 2025.
[2] Fanwang Meng, Yang Xi, Jinfeng Huang, and Paul W. Ayers. A curated diverse molecular database
of blood-brain barrier permeability with chemical descriptors.Scientific Data, 8:289, 2021.
[3] Wolfgang L¨ oscher and Heidrun Potschka. Blood-brain barrier active efflux transporters: ATP-
binding cassette gene family.NeuroRx, 2(1):86–98, 2005.

<!-- PDF page 25 -->
[4] A. J. M. Sadeque, C. Wandel, H. He, S. Shah, and A. J. J. Wood. Increased drug delivery to the
brain by P-glycoprotein inhibition.Clinical Pharmacology & Therapeutics, 68(3):231–237, 2000.
[5] Marc Adenot and Roger Lahana. Blood-brain barrier permeation models: Discriminating between
potential CNS and non-CNS drugs including P-glycoprotein substrates.Journal of Chemical Infor-
mation and Computer Sciences, 44:239–248, 2004.
[6] Anna Seelig. A general pattern for substrate recognition by P-glycoprotein.European Journal of
Biochemistry, 251(1–2):252–261, 1998.
[7] Amer Alam, Julia Kowal, Eugenia Broude, Igor Roninson, and Kaspar P. Locher. Structural insight
into substrate and inhibitor discrimination by human P-glycoprotein.Science, 363(6428):753–756,
2019.
[8] Yuan H. Zhao, Michael H. Abraham, Adam Ibrahim, Paul V. Fish, Susan Cole, Mark L. Lewis,
Marcel J. de Groot, and Derek P. Reynolds. Predicting penetration across the blood-brain barrier
from simple descriptors and fragmentation schemes.Journal of Chemical Information and Modeling,
47:170–175, 2007.
[9] Hu Li, Chun Wei Yap, Choong Yong Ung, Ying Xue, Zhi Wei Cao, and Yu Zong Chen. Effect
of selection of molecular descriptors on the prediction of blood-brain barrier penetrating and non-
penetrating agents by statistical learning methods.Journal of Chemical Information and Modeling,
45:1376–1384, 2005.
[10] Ines Filipa Martins, Ana L. Teixeira, Luis Pinheiro, and Andre O. Falcao. A Bayesian approach to
in silico blood-brain barrier penetration modeling.Journal of Chemical Information and Modeling,
52:1686–1697, 2012.
[11] Yaxia Yuan, Fang Zheng, and Chang-Guo Zhan. Improved prediction of blood-brain barrier per-
meability through machine learning with combined use of molecular property-based descriptors and
fingerprints.The AAPS Journal, 20:54, 2018.
[12] David Rogers and Mathew Hahn. Extended-connectivity fingerprints.Journal of Chemical Infor-
mation and Modeling, 50(5):742–754, 2010.
[13] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural net-
works? InInternational Conference on Learning Representations, 2019. arXiv:1810.00826.
[14] Matthias Fey and Jan E. Lenssen. Fast graph representation learning with PyTorch Geometric. In
ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019. arXiv:1903.02428.
[15] Fabio Broccatelli, Emanuele Carosati, Annalisa Neri, Maria Frosini, Laura Goracci, Tudor I. Oprea,
and Gabriele Cruciani. A novel approach for predicting P-glycoprotein (ABCB1) inhibition using
molecular interaction fields.Journal of Medicinal Chemistry, 54(6):1740–1751, 2011.
[16] Kexin Huang, Tianfan Fu, Wenhao Gao, Yue Zhao, Yusuf Roohani, Jure Leskovec, Connor W. Coley,
Cao Xiao, Jimeng Sun, and Marinka Zitnik. Therapeutics data commons: Machine learning datasets
and tasks for drug discovery and development. InProceedings of the Neural Information Processing
Systems Track on Datasets and Benchmarks, 2021. arXiv:2102.09548.
[17] Guy W. Bemis and Mark A. Murcko. The properties of known drugs. 1. molecular frameworks.
Journal of Medicinal Chemistry, 39(15):2887–2893, 1996.
[18] Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. InProceedings of
the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages
785–794, 2016.
[19] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure
Leskovec. Strategies for pre-training graph neural networks. InInternational Conference on Learning
Representations, 2020. arXiv:1905.12265.
[20] Sereina Riniker and Gregory A. Landrum. Better informed distance geometry: Using what we know
to improve conformation generation.Journal of Chemical Information and Modeling, 55(12):2562–
2574, 2015.
[21] Shuzhe Wang, Jagna Witek, Gregory A. Landrum, and Sereina Riniker. Improving conformer gener-
ation for small rings and macrocycles based on distance geometry and experimental torsional-angle
preferences.Journal of Chemical Information and Modeling, 60(4):2044–2058, 2020.
[22] Thomas A. Halgren. Merck molecular force field. I. basis, form, scope, parameterization, and per-
formance of MMFF94.Journal of Computational Chemistry, 17(5–6):490–519, 1996.
[23] RCSB Protein Data Bank. PDB entry 6qex: human p-glycoprotein, 2019.
[24] Oleg Trott and Arthur J. Olson. AutoDock Vina: Improving the speed and accuracy of docking
with a new scoring function, efficient optimization, and multithreading.Journal of Computational
Chemistry, 31(2):455–461, 2010.
[25] Scott M. Lundberg, Gabriel Erion, Hugh Chen, Alex DeGrave, Jordan M. Prutkin, Bala Nair,
Ronit Katz, Jonathan Himmelfarb, Nisha Bansal, and Su-In Lee. From local explanations to global
understanding with explainable AI for trees.Nature Machine Intelligence, 2(1):56–67, 2020.
[26] Jerome Eberhardt, Diogo Santos-Martins, Andreas F. Tillack, and Stefano Forli. AutoDock Vina

<!-- PDF page 26 -->
1.2.0: New docking methods, expanded force field, and Python bindings.Journal of Chemical
Information and Modeling, 61(8):3891–3898, 2021.
[27] Duvva et al. Dual-barrier cns drug prediction research project repository.
