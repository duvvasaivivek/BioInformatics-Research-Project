"""
Four-Quadrant Pharmacological Decision Matrix and Clinical Reference Benchmarks.
Decouples passive BBB membrane partition from active P-gp transporter efflux.
"""

from typing import Tuple, Dict, Any


class Colors:
    """ANSI color codes for formatted terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


BENCHMARK_DRUGS: Dict[str, Dict[str, Any]] = {
    "Levetiracetam (Keppra)": {
        "smiles": "CC[C@@H](C(=O)N)N1CCCC1=O",
        "ground_truth": "BBB+ / P-gp- (Optimal CNS Candidate)",
        "expected_quadrant": 1,
        "notes": "Refractory antiepileptic drug that evades P-gp efflux entirely (Löscher & Potschka 2005)."
    },
    "Loperamide (Imodium)": {
        "smiles": "CN(C)C(=O)CC(c1ccccc1)(c2ccccc2)CCN3CCC(CC3)(c4ccc(cc4)Cl)O",
        "ground_truth": "BBB+ / P-gp+ (Paradoxical Drug / High Efflux Risk)",
        "expected_quadrant": 2,
        "notes": "Classic paradoxical drug: readily crosses membranes (BBB+) but is actively extruded by luminal P-gp."
    },
    "Caffeine": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "ground_truth": "BBB+ / P-gp- (Optimal CNS Candidate)",
        "expected_quadrant": 1,
        "notes": "Classic CNS psychostimulant with high transcellular permeability and negligible P-gp recognition."
    },
    "Diazepam (Valium)": {
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "ground_truth": "BBB+ / P-gp- (Optimal CNS Candidate)",
        "expected_quadrant": 1,
        "notes": "Lipophilic benzodiazepine sedative that rapidly distributes into brain tissue."
    },
    "Penicillin G": {
        "smiles": "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        "ground_truth": "BBB- / P-gp- (Peripheral Low-Permeation Agent)",
        "expected_quadrant": 3,
        "notes": "Peripheral antibiotic; poor transcellular BBB penetration due to ionization and high desolvation penalty."
    }
}


def determine_dual_barrier_quadrant(bbb_pred: int, pgp_pred: int) -> Tuple[str, str]:
    """
    Classifies candidate compound into one of four pharmacologically grounded quadrants:
    1. BBB+ / P-gp- : Optimal CNS Drug Candidate (Passive Influx, No Pump Extrusion)
    2. BBB+ / P-gp+ : Paradoxical Drug Candidate (Lipophilic Influx, Violent Efflux - e.g. Loperamide)
    3. BBB- / P-gp- : Peripheral Low-Permeation Agent (Fails Passive Influx, Evades Pump)
    4. BBB- / P-gp+ : Restricted Effluxed Peripheral Compound (Poor Permeation & Active Clearance)
    """
    if bbb_pred == 1 and pgp_pred == 0:
        return (
            "Quadrant 1: BBB+ / P-gp- [Optimal CNS Candidate]",
            "Compound possesses favorable lipid partition to cross brain capillary endothelial membranes passively, "
            "and does not engage the ABCB1 (P-gp) ATPase pump. Like Levetiracetam, it maintains sustained brain exposure."
        )
    elif bbb_pred == 1 and pgp_pred == 1:
        return (
            "Quadrant 2: BBB+ / P-gp+ [Paradoxical Drug / High Efflux Risk]",
            "Compound readily partitions into endothelial membranes (BBB+), but high spatial bulk or Seelig H-bonding motifs "
            "trigger active luminal extrusion by P-glycoprotein. Resembles Loperamide: high in vitro lipophilicity but negligible brain exposure."
        )
    elif bbb_pred == 0 and pgp_pred == 0:
        return (
            "Quadrant 3: BBB- / P-gp- [Peripheral Low-Permeation Agent]",
            "Compound lacks sufficient lipid solubility or possesses excessive polar surface area to cross the BBB passively. "
            "Although it does not trigger P-gp efflux, lack of passive permeation prevents CNS penetration."
        )
    else:
        return (
            "Quadrant 4: BBB- / P-gp+ [Restricted Effluxed Peripheral Compound]",
            "Compound is doubly impeded: it fails passive BBB permeation and is actively recognized by P-gp as an efflux substrate. "
            "Action is strictly confined to peripheral compartments."
        )
