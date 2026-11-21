"""
Sequential Cascaded Sieve Engine for CNS Drug Discovery (Novelty 2).

Replaces simultaneous joint modeling with a biologically grounded hierarchical pipeline:
  Stage 1: High-Throughput Blood-Brain Barrier (BBB) Permeability Sieve (B3DB, N=7,805)
  Stage 2: Targeted P-glycoprotein (P-gp) Active Efflux Gatekeeper (Broccatelli, N=1,205)

Compounds that fail the Stage 1 BBB sieve (BBB-) are early-gated as peripheral agents,
completely avoiding the costly 3D conformer generation and inference required for Stage 2.
Permeant compounds (BBB+) proceed sequentially to Stage 2 to screen for active efflux liabilities.
"""

import os
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import xgboost as xgb
import numpy as np
from rdkit import Chem

from .features import (
    extract_features_bbb,
    extract_features_pgp,
    prepare_dataframe,
    load_scaler_metadata,
)

BENCHMARK_DRUGS: Dict[str, Dict[str, Any]] = {
    "Levetiracetam (Keppra)": {
        "smiles": "CC[C@@H](C(=O)N)N1CCCC1=O",
        "expected_stage1": "PASS (BBB+)",
        "expected_stage2": "PASS (P-gp- Evader)",
        "expected_disposition": "CNS_PENETRANT",
        "notes": "Optimal CNS candidate: high transcellular permeation, evades efflux pump."
    },
    "Loperamide (Imodium)": {
        "smiles": "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1",
        "expected_stage1": "PASS (BBB+)",
        "expected_stage2": "REJECT (P-gp+ Substrate)",
        "expected_disposition": "PARADOXICAL_EFFLUX",
        "notes": "Paradoxical drug: enters lipid membrane but is violently expelled by luminal P-gp."
    },
    "Caffeine": {
        "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "expected_stage1": "PASS (BBB+)",
        "expected_stage2": "PASS (P-gp- Evader)",
        "expected_disposition": "CNS_PENETRANT",
        "notes": "Classic psychostimulant with high permeability and pump evasion."
    },
    "Diazepam (Valium)": {
        "smiles": "CN1C(=O)CN=C(c2ccccc2)c3cc(Cl)ccc13",
        "expected_stage1": "PASS (BBB+)",
        "expected_stage2": "PASS (P-gp- Evader)",
        "expected_disposition": "CNS_PENETRANT",
        "notes": "Lipophilic sedative with sustained brain partition."
    },
    "Penicillin G": {
        "smiles": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
        "expected_stage1": "REJECT (BBB-)",
        "expected_stage2": "BYPASS (Stage 2 Skipped)",
        "expected_disposition": "PERIPHERAL_LOW_PERMEATION",
        "notes": "Peripheral antibiotic: fails Stage 1 BBB sieve, skipping Stage 2 compute entirely."
    }
}


@dataclass
class SequentialResult:
    """Detailed record of a compound's evaluation through the sequential cascade."""
    smiles: str
    stage1_evaluated: bool
    stage1_pass: bool
    stage1_prob_bbb: float
    stage1_threshold: float
    stage2_evaluated: bool
    stage2_pass: Optional[bool]
    stage2_prob_pgp: Optional[float]
    final_disposition: str
    composite_cns_score: float
    compute_saved: bool
    stage1_latency_ms: float
    stage2_latency_ms: float
    total_latency_ms: float
    details: Dict[str, Any]


class SequentialCascadedEngine:
    """
    Two-Stage Sequential Cascaded Sieve for CNS Drug Prioritization.

    Stage 1: BBB Permeability Sieve (threshold tau_bbb)
    Stage 2: P-gp Efflux Gatekeeper (threshold tau_pgp)
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        tau_bbb: float = 0.50,
        tau_pgp: float = 0.50,
    ) -> None:
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.models_dir = models_dir
        self.tau_bbb = tau_bbb
        self.tau_pgp = tau_pgp

        # Load Stage 1 BBB Model & Scaler
        bbb_model_path = os.path.join(models_dir, "bbb_xgboost.json")
        bbb_scaler_path = os.path.join(models_dir, "bbb_scaler_metadata.json")
        self.bbb_model = xgb.XGBClassifier()
        self.bbb_model.load_model(bbb_model_path)
        self.bbb_means, self.bbb_stds = load_scaler_metadata(bbb_scaler_path)

        # Load Stage 2 P-gp Model & Scaler
        pgp_model_path = os.path.join(models_dir, "pgp_xgboost.json")
        pgp_scaler_path = os.path.join(models_dir, "pgp_scaler_metadata.json")
        self.pgp_model = xgb.XGBClassifier()
        self.pgp_model.load_model(pgp_model_path)
        self.pgp_means, self.pgp_stds = load_scaler_metadata(pgp_scaler_path)

    def evaluate_molecule(self, smiles: str) -> SequentialResult:
        """
        Executes the hierarchical sequential sieve on a single SMILES string.
        """
        t0 = time.perf_counter()

        # -------------------------------------------------------------
        # STAGE 1: BBB Permeability Sieve (Fast 2D + FP)
        # -------------------------------------------------------------
        t_s1_start = time.perf_counter()
        bbb_feats, raw_bbb, mol_bbb = extract_features_bbb(smiles)
        if bbb_feats is None:
            raise ValueError(f"Invalid SMILES structure: {smiles}")

        df_bbb = prepare_dataframe(
            bbb_feats[0], bbb_feats[1], self.bbb_means, self.bbb_stds, is_pgp=False
        )
        p_bbb = float(self.bbb_model.predict_proba(df_bbb)[0, 1])
        stage1_pass = bool(p_bbb >= self.tau_bbb)
        t_s1 = (time.perf_counter() - t_s1_start) * 1000.0

        # -------------------------------------------------------------
        # STAGE 2: Conditional P-gp Efflux Gatekeeper (3D ETKDGv3 Conformer + Shape)
        # -------------------------------------------------------------
        stage2_evaluated = False
        stage2_pass = None
        p_pgp = None
        t_s2 = 0.0

        if not stage1_pass:
            # Molecule failed passive permeation.
            # STAGE 2 IS COMPLETELY BYPASSED: Saving conformer generation & inference!
            final_disp = "PERIPHERAL_LOW_PERMEATION"
            composite_score = 0.0
            compute_saved = True
        else:
            # Molecule passed Stage 1: Proceed to Stage 2 P-gp gate
            stage2_evaluated = True
            t_s2_start = time.perf_counter()
            pgp_feats, raw_pgp, mol_pgp = extract_features_pgp(smiles)

            if pgp_feats is None:
                # 3D conformer generation failed: flag as conformer error
                p_pgp = 0.50
                stage2_pass = False
                final_disp = "CONFORMER_GENERATION_FAILED"
                composite_score = 0.0
                compute_saved = False
            else:
                df_pgp = prepare_dataframe(
                    pgp_feats[0], pgp_feats[1], self.pgp_means, self.pgp_stds, is_pgp=True
                )
                p_pgp = float(self.pgp_model.predict_proba(df_pgp)[0, 1])
                is_evader = bool(p_pgp < self.tau_pgp)
                stage2_pass = is_evader

                if is_evader:
                    # Passes both Stage 1 and Stage 2
                    final_disp = "CNS_PENETRANT"
                    composite_score = float(p_bbb * (1.0 - p_pgp))
                else:
                    # Passes Stage 1 but caught by Stage 2 P-gp efflux
                    final_disp = "PARADOXICAL_EFFLUX"
                    composite_score = 0.0  # Rejected by efflux gatekeeper

                compute_saved = False
            t_s2 = (time.perf_counter() - t_s2_start) * 1000.0

        total_lat = (time.perf_counter() - t0) * 1000.0

        return SequentialResult(
            smiles=smiles,
            stage1_evaluated=True,
            stage1_pass=stage1_pass,
            stage1_prob_bbb=p_bbb,
            stage1_threshold=self.tau_bbb,
            stage2_evaluated=stage2_evaluated,
            stage2_pass=stage2_pass,
            stage2_prob_pgp=p_pgp,
            final_disposition=final_disp,
            composite_cns_score=composite_score,
            compute_saved=compute_saved,
            stage1_latency_ms=t_s1,
            stage2_latency_ms=t_s2,
            total_latency_ms=total_lat,
            details={
                "p_permeant": p_bbb,
                "p_efflux": p_pgp,
                "raw_bbb": raw_bbb,
            }
        )

    def evaluate_library(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Executes batch sequential screening, collecting cascade attrition statistics.
        """
        results: List[SequentialResult] = []
        n_total = len(smiles_list)
        n_stage1_pass = 0
        n_stage2_evaluated = 0
        n_cns_penetrants = 0
        n_paradoxical = 0
        n_peripheral = 0
        total_stage1_time = 0.0
        total_stage2_time = 0.0

        for smi in smiles_list:
            res = self.evaluate_molecule(smi)
            results.append(res)
            total_stage1_time += res.stage1_latency_ms
            total_stage2_time += res.stage2_latency_ms

            if res.stage1_pass:
                n_stage1_pass += 1
                n_stage2_evaluated += 1
                if res.final_disposition == "CNS_PENETRANT":
                    n_cns_penetrants += 1
                elif res.final_disposition == "PARADOXICAL_EFFLUX":
                    n_paradoxical += 1
            else:
                n_peripheral += 1

        stage1_retention_rate = n_stage1_pass / n_total if n_total > 0 else 0.0
        stage1_attrition_rate = 1.0 - stage1_retention_rate
        stage2_retention_rate = n_cns_penetrants / n_stage2_evaluated if n_stage2_evaluated > 0 else 0.0
        stage2_attrition_rate = 1.0 - stage2_retention_rate
        overall_yield = n_cns_penetrants / n_total if n_total > 0 else 0.0
        early_compute_saving_ratio = n_peripheral / n_total if n_total > 0 else 0.0

        return {
            "n_total": n_total,
            "n_stage1_pass": n_stage1_pass,
            "n_stage1_rejected": n_peripheral,
            "n_stage2_evaluated": n_stage2_evaluated,
            "n_cns_leads": n_cns_penetrants,
            "n_paradoxical_effluxed": n_paradoxical,
            "stage1_attrition_rate": stage1_attrition_rate,
            "stage2_attrition_rate": stage2_attrition_rate,
            "overall_yield": overall_yield,
            "early_compute_saving_ratio": early_compute_saving_ratio,
            "total_stage1_time_ms": total_stage1_time,
            "total_stage2_time_ms": total_stage2_time,
            "results": results
        }
