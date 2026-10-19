"""
Applicability Domain and Prediction Uncertainty Assessment Module.
Evaluates chemical similarity to training distributions (Tanimoto max-similarity)
and continuous descriptor leverage (Z-score Mahalanobis/Euclidean distance)
to assess whether a test compound falls within the model's reliable chemical domain.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


class ApplicabilityDomainEstimator:
    """
    Assesses chemical domain validity and uncertainty for candidate molecules.
    """

    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
        self.data_dir = data_dir

        self.bbb_fps: List[Any] = []
        self.pgp_fps: List[Any] = []
        self._load_reference_fingerprints()

    def _load_reference_fingerprints(self, max_samples: int = 1000):
        """Loads a representative sample of training fingerprints for fast Tanimoto screening."""
        # 1. BBB reference
        bbb_path = os.path.join(self.data_dir, "B3DB_classification.tsv")
        if os.path.exists(bbb_path):
            try:
                df_bbb = pd.read_csv(bbb_path, sep='\t')
                smiles_list = df_bbb['SMILES'].dropna().tolist()
                step = max(1, len(smiles_list) // max_samples)
                sample_smiles = smiles_list[::step]
                for s in sample_smiles:
                    m = Chem.MolFromSmiles(s)
                    if m is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
                        self.bbb_fps.append(fp)
            except Exception as e:
                print(f"[WARN] Failed to load BBB reference for applicability domain: {e}")

        # 2. P-gp reference
        pgp_path = os.path.join(self.data_dir, "pgp_broccatelli.tab")
        if os.path.exists(pgp_path):
            try:
                df_pgp = pd.read_csv(pgp_path, sep='\t')
                smiles_col = 'Drug' if 'Drug' in df_pgp.columns else df_pgp.columns[1]
                smiles_list = df_pgp[smiles_col].dropna().tolist()
                for s in smiles_list:
                    m = Chem.MolFromSmiles(s)
                    if m is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
                        self.pgp_fps.append(fp)
            except Exception as e:
                print(f"[WARN] Failed to load P-gp reference for applicability domain: {e}")

    def evaluate_compound(
        self,
        mol: Chem.Mol,
        scaled_descriptors: List[float],
        model_name: str,
        prob_pos: float,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Assesses applicability domain status for a single molecule.
        
        Args:
            mol: RDKit molecule object.
            scaled_descriptors: Standardized continuous descriptor vector.
            model_name: "BBB" or "Pgp".
            prob_pos: Predicted probability for positive class.
            threshold: Decision threshold.
            
        Returns:
            Dictionary with max Tanimoto similarity, descriptor leverage,
            domain status, and uncertainty metrics.
        """
        is_pgp = (model_name.upper() == "PGP")
        ref_fps = self.pgp_fps if is_pgp else self.bbb_fps

        # 1. Structural Similarity via Tanimoto
        test_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        max_tanimoto = 0.0
        if ref_fps:
            sims = DataStructs.BulkTanimotoSimilarity(test_fp, ref_fps)
            max_tanimoto = float(max(sims)) if sims else 0.0

        # 2. Continuous Descriptor Leverage (Normalized Euclidean distance from centroid)
        desc_arr = np.array(scaled_descriptors)
        norm_dist = float(np.sqrt(np.mean(desc_arr ** 2)))

        # 3. Domain Categorization
        # Standard cheminformatics thresholds:
        # High confidence: Tanimoto >= 0.40 and norm_dist <= 2.5
        # Borderline: Tanimoto 0.30 - 0.40 or norm_dist 2.5 - 3.5
        # Outside Domain: Tanimoto < 0.30 or norm_dist > 3.5
        if max_tanimoto >= 0.40 and norm_dist <= 2.5:
            domain_status = "Inside Domain"
            domain_warning = None
        elif max_tanimoto >= 0.28 and norm_dist <= 3.5:
            domain_status = "Borderline Domain"
            domain_warning = "Moderate chemical divergence from training set; interpret predictions with caution."
        else:
            domain_status = "Outside Applicability Domain"
            domain_warning = (
                "High chemical novelty relative to training benchmarks. "
                "The model's empirical confidence may not reflect true biological activity."
            )

        # 4. Uncertainty Quantification
        # Margin confidence: distance from decision threshold [0, 1]
        decision_margin = abs(prob_pos - threshold)
        normalized_confidence = min(1.0, decision_margin / 0.5)
        uncertainty_score = 1.0 - normalized_confidence

        return {
            "model_name": model_name,
            "max_tanimoto_similarity": round(max_tanimoto, 4),
            "descriptor_leverage_distance": round(norm_dist, 3),
            "domain_status": domain_status,
            "domain_warning": domain_warning,
            "decision_margin": round(decision_margin, 4),
            "prediction_confidence": round(normalized_confidence, 4),
            "uncertainty_score": round(uncertainty_score, 4),
        }
