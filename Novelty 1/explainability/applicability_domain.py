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
from rdkit.Chem import DataStructs, rdFingerprintGenerator

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class ApplicabilityDomainEstimator:
    """
    Assesses chemical domain validity and uncertainty for candidate molecules.
    """

    def __init__(self, data_dir: Optional[str] = None, max_samples: Optional[int] = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base_dir, "data")
        self.data_dir = data_dir

        self.bbb_fps: List[Any] = []
        self.pgp_fps: List[Any] = []
        self._load_reference_fingerprints(max_samples=max_samples)

    def _load_reference_fingerprints(self, max_samples: Optional[int] = None):
        """Loads reference training fingerprints for fast Tanimoto screening across full dataset."""
        from core.artifacts import training_smiles
        self.reference_errors = {}
        for task, destination in [('bbb', self.bbb_fps), ('pgp', self.pgp_fps)]:
            try:
                smiles = training_smiles(self.data_dir, task)
                if max_samples is not None:
                    if max_samples < 1:
                        raise ValueError('max_samples must be positive')
                    ids = np.linspace(0, len(smiles)-1, min(max_samples, len(smiles)), dtype=int)
                    smiles = [smiles[i] for i in ids]
                for value in smiles:
                    mol = Chem.MolFromSmiles(value)
                    if mol is not None:
                        destination.append(_MORGAN_GEN.GetFingerprint(mol))
            except (OSError, ValueError, KeyError) as exc:
                self.reference_errors[task] = str(exc)

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
        test_fp = _MORGAN_GEN.GetFingerprint(mol)
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

        if not ref_fps:
            domain_status = 'Domain unavailable'
            domain_warning = self.reference_errors.get('pgp' if is_pgp else 'bbb', 'No training references')

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
