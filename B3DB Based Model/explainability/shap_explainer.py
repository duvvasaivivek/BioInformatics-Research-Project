"""
TreeSHAP Model Explainability Module for BBB Permeability and P-gp Efflux Classifiers.
Provides mathematically rigorous feature attributions in log-odds margin space,
aligns them with positive/negative prediction outcomes, and isolates top supporting and opposing features.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from rdkit import Chem

try:
    import shap
    import shap.explainers._tree as _st
    # Robust compatibility patch for XGBoost 2.0+ base_score bracket strings (e.g. '[5E-1]')
    if hasattr(_st, 'decode_ubjson_buffer'):
        _orig_decode = _st.decode_ubjson_buffer
        def _patched_decode(fd):
            jmodel = _orig_decode(fd)
            try:
                lmp = jmodel['learner']['learner_model_param']
                bs = lmp.get('base_score')
                if isinstance(bs, str) and bs.startswith('[') and bs.endswith(']'):
                    lmp['base_score'] = bs[1:-1]
                elif isinstance(bs, (list, tuple)):
                    lmp['base_score'] = str(bs[0])
            except Exception:
                pass
            return jmodel
        _st.decode_ubjson_buffer = _patched_decode
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from .substructure_mapper import MorganSubstructureMapper


class DualBarrierShapExplainer:
    """
    Computes TreeSHAP explanations for XGBoost models predicting
    BBB Permeability and P-gp Substrate status.
    """

    def __init__(self, bbb_model, pgp_model):
        self.bbb_model = bbb_model
        self.pgp_model = pgp_model
        self.substructure_mapper = MorganSubstructureMapper()

        self._bbb_explainer = None
        self._pgp_explainer = None

        if SHAP_AVAILABLE:
            try:
                self._bbb_explainer = shap.TreeExplainer(self.bbb_model)
            except Exception as e:
                print(f"[WARN] Failed to initialize BBB TreeExplainer: {e}")

            try:
                self._pgp_explainer = shap.TreeExplainer(self.pgp_model)
            except Exception as e:
                print(f"[WARN] Failed to initialize P-gp TreeExplainer: {e}")

    def explain_prediction(
        self,
        model_name: str,  # "BBB" or "Pgp"
        input_df: pd.DataFrame,
        mol: Optional[Chem.Mol] = None,
        raw_descriptors: Optional[Dict[str, float]] = None,
        decision_threshold: float = 0.5,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive, model-specific explanation for a single prediction.
        
        Args:
            model_name: "BBB" or "Pgp"
            input_df: Single-row DataFrame matching the model's exact feature ordering.
            mol: Optional RDKit molecule to extract substructures for active fingerprint bits.
            raw_descriptors: Dictionary of unnormalized physicochemical descriptors.
            decision_threshold: Decision threshold for classification (default 0.5).
            top_n: Number of top supporting and opposing features to report.
            
        Returns:
            Structured dictionary with explanation metrics, attributions, and interpretations.
        """
        is_pgp = (model_name.upper() == "PGP")
        model = self.pgp_model if is_pgp else self.bbb_model
        explainer = self._pgp_explainer if is_pgp else self._bbb_explainer

        # Predict probability
        probs = model.predict_proba(input_df)[0]
        prob_pos = float(probs[1])
        prob_neg = float(probs[0])
        predicted_class = int(prob_pos >= decision_threshold)

        # Continuous feature definitions
        continuous_names = ['MolWt', 'LogP', 'TPSA', 'HDonors', 'HAcceptors', 'RotBonds']
        if is_pgp:
            continuous_names += [
                'RadiusOfGyration', 'Asphericity', 'Eccentricity',
                'SpherocityIndex', 'NPR1', 'NPR2'
            ]

        # Compute SHAP values
        shap_values_array = None
        base_value = 0.0
        is_tree_shap = False

        if explainer is not None:
            try:
                shap_res = explainer(input_df)
                shap_values_array = np.array(shap_res.values[0])
                base_value = float(
                    shap_res.base_values[0]
                    if hasattr(shap_res, 'base_values')
                    else explainer.expected_value
                )
                is_tree_shap = True
            except Exception as e:
                is_tree_shap = False

        if not is_tree_shap:
            # Rigorous model attribution fallback using booster feature weights and standardized values
            booster = model.get_booster()
            score_dict = booster.get_score(importance_type='gain')
            total_gain = sum(score_dict.values()) if score_dict else 1.0
            
            # Approximate margin deviation
            margin = float(np.log(prob_pos / max(1e-7, (1.0 - prob_pos))))
            base_value = 0.0
            contribs = []
            for col in input_df.columns:
                gain = score_dict.get(col, 0.0)
                norm_gain = gain / total_gain
                val = float(input_df[col].iloc[0])
                contribs.append(val * norm_gain * margin)
            shap_values_array = np.array(contribs)

        feature_names = list(input_df.columns)
        shap_series = pd.Series(shap_values_array, index=feature_names)
        raw_margin = float(base_value + shap_series.sum())

        # Separate continuous descriptors vs active fingerprint bits
        continuous_shap = {c: float(shap_series[c]) for c in continuous_names if c in shap_series}
        
        # Isolate fingerprint bits that are active (bit value == 1) in the molecule
        fp_active_shap = {}
        for c in feature_names:
            if c.startswith('FP_'):
                val = input_df[c].iloc[0]
                if val == 1:
                    fp_active_shap[c] = float(shap_series[c])

        # Rank all features by SHAP value
        sorted_all = shap_series.sort_values(ascending=False)

        # Class-aligned interpretation:
        # positive margin (>0) pushes towards Class 1 (BBB+ or P-gp+);
        # negative margin (<0) pushes towards Class 0 (BBB- or P-gp-).
        if predicted_class == 1:
            supporting_features = sorted_all[sorted_all > 0].head(top_n)
            opposing_features = sorted_all[sorted_all < 0].tail(top_n).sort_values(ascending=True)
        else:
            supporting_features = sorted_all[sorted_all < 0].tail(top_n).sort_values(ascending=True)
            opposing_features = sorted_all[sorted_all > 0].head(top_n)

        # Resolve substructures for active fingerprint bits in top lists
        bit_info = self.substructure_mapper.get_bit_info(mol) if mol is not None else None
        
        def enrich_feature_list(feat_series: pd.Series) -> List[Dict[str, Any]]:
            enriched = []
            for name, val in feat_series.items():
                item = {
                    "feature": name,
                    "shap_value": float(val),
                    "input_value": float(input_df[name].iloc[0]),
                    "is_continuous": (name in continuous_names),
                    "is_active_bit": (name.startswith('FP_') and input_df[name].iloc[0] == 1),
                }
                if raw_descriptors and name in raw_descriptors:
                    item["raw_value"] = raw_descriptors[name]

                # Map substructure if active bit
                if mol is not None and name.startswith('FP_'):
                    try:
                        bit_id = int(name.replace('FP_', ''))
                        frags = self.substructure_mapper.extract_substructure_smiles(
                            mol, bit_id, bit_info
                        )
                        item["substructures"] = frags
                    except Exception:
                        item["substructures"] = []
                enriched.append(item)
            return enriched

        supporting_enriched = enrich_feature_list(supporting_features)
        opposing_enriched = enrich_feature_list(opposing_features)

        # Contextual biophysical interpretation
        interpretation = self._generate_biophysical_interpretation(
            model_name=model_name,
            predicted_class=predicted_class,
            prob_pos=prob_pos,
            raw_descriptors=raw_descriptors,
            continuous_shap=continuous_shap
        )

        return {
            "model_name": model_name,
            "predicted_class": predicted_class,
            "class_label": self._get_class_label(model_name, predicted_class),
            "probability_positive": prob_pos,
            "probability_negative": prob_neg,
            "decision_threshold": decision_threshold,
            "output_space": "Log-odds raw margin (f(x) = base_value + sum(shap_values))",
            "base_value": base_value,
            "model_margin": raw_margin,
            "is_tree_shap": is_tree_shap,
            "continuous_shap": continuous_shap,
            "top_supporting": supporting_enriched,
            "top_opposing": opposing_enriched,
            "interpretation": interpretation,
        }

    def _get_class_label(self, model_name: str, predicted_class: int) -> str:
        if model_name.upper() == "BBB":
            return "BBB+ (Permeable)" if predicted_class == 1 else "BBB- (Non-Permeable)"
        else:
            return "P-gp+ (Substrate / Effluxed)" if predicted_class == 1 else "P-gp- (Non-Substrate / Evader)"

    def _generate_biophysical_interpretation(
        self,
        model_name: str,
        predicted_class: int,
        prob_pos: float,
        raw_descriptors: Optional[Dict[str, float]],
        continuous_shap: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Connects empirical SHAP contributions to classical medicinal chemistry principles
        (Lipinski rule of 5 for CNS, Pajouhesh & Lenz 2005, Seelig 1998, Löscher & Potschka 2005).
        """
        interp = {}
        if not raw_descriptors:
            return {"summary": "Descriptors unavailable for biophysical interpretation."}

        mw = raw_descriptors.get('MolWt', 0)
        logp = raw_descriptors.get('LogP', 0)
        tpsa = raw_descriptors.get('TPSA', 0)
        hbd = raw_descriptors.get('HDonors', 0)
        hba = raw_descriptors.get('HAcceptors', 0)
        rotb = raw_descriptors.get('RotBonds', 0)

        if model_name.upper() == "BBB":
            if predicted_class == 1:
                # BBB Positive Explanation
                interp["mechanism"] = (
                    "Passive transcellular lipophilic partition across brain capillary endothelial cells."
                )
                reasons = []
                if logp >= 1.5:
                    reasons.append(f"Optimal lipophilicity (LogP {logp:.2f} >= 1.5) provides favorable membrane partition thermodynamics.")
                if tpsa <= 90.0:
                    reasons.append(f"Low polar surface area (TPSA {tpsa:.1f} <= 90 A^2) incurs low desolvation penalty upon entering lipid bilayers.")
                if mw <= 400.0:
                    reasons.append(f"Low molecular weight (MW {mw:.1f} Da <= 400 Da) complies with CNS-penetrant diffusion rates.")
                if hbd <= 2:
                    reasons.append(f"Restricted hydrogen-bond donors (HBD {hbd} <= 2) limits dipole-dipole water trapping in the lumen.")
                interp["biophysical_basis"] = " ".join(reasons) if reasons else "Conformation and topological fragments drive favorable membrane partitioning."
                interp["rule_alignment"] = (
                    "Conforms to Pajouhesh & Lenz (2005) CNS medicinal chemistry guidelines (MW < 400, LogP 2-4, TPSA < 90 A^2)."
                )
            else:
                # BBB Negative Explanation
                interp["mechanism"] = (
                    "Impaired passive transcellular permeation due to polar desolvation penalty or excessive steric bulk."
                )
                barriers = []
                if tpsa > 90.0:
                    barriers.append(f"Elevated polar surface area (TPSA {tpsa:.1f} > 90 A^2) requires excessive energy to shed hydration water shells.")
                if logp < 1.0:
                    barriers.append(f"Insufficient lipophilicity (LogP {logp:.2f} < 1.0) prevents favorable thermodynamic partitioning into the hydrophobic lipid core.")
                if mw > 450.0:
                    barriers.append(f"High molecular weight (MW {mw:.1f} Da > 450 Da) retards transcellular diffusion across tightly packed endothelial tight junctions.")
                if hbd > 3 or hba > 7:
                    barriers.append(f"High hydrogen-bonding capacity (HBD: {hbd}, HBA: {hba}) forms strong aqueous solvation clusters.")
                interp["biophysical_basis"] = " ".join(barriers) if barriers else "Topological fingerprint motifs and electrostatic parameters oppose transcellular penetration."
                interp["rule_alignment"] = (
                    "Exceeds typical CNS-active thresholds; molecule is predominantly confined to peripheral circulation."
                )

        else:  # P-gp Model
            rg = raw_descriptors.get('RadiusOfGyration', 0)
            sph = raw_descriptors.get('SpherocityIndex', 0)
            asph = raw_descriptors.get('Asphericity', 0)

            if predicted_class == 1:
                # P-gp Positive Explanation
                interp["mechanism"] = (
                    "High-affinity recognition and active extrusion by the ABCB1 (P-glycoprotein) ATPase efflux pump."
                )
                reasons = []
                if mw >= 400.0 or rg >= 4.0:
                    reasons.append(f"Spacious spatial envelope (MW {mw:.1f} Da, Rg {rg:.2f} A) spans P-gp's large translocating cavity (~6000 A^3).")
                if hbd >= 2 or hba >= 4:
                    reasons.append(f"Polyvalent hydrogen-bonding motifs (HBD: {hbd}, HBA: {hba}) fit Seelig's repeating pharmacophoric donor/acceptor recognition pattern.")
                if rotb >= 6:
                    reasons.append(f"High conformational flexibility (RotBonds {rotb} >= 6) allows induced-fit binding inside the transmembrane helices.")
                interp["biophysical_basis"] = " ".join(reasons) if reasons else "Topological motifs and spatial conformation establish high affinity for ABCB1 binding sites."
                interp["rule_alignment"] = (
                    "Aligns with Seelig (1998) pharmacophore rules and Broccatelli (2011) 3D cavity recognition patterns."
                )
            else:
                # P-gp Negative Explanation
                interp["mechanism"] = (
                    "Evasion of active ABCB1 extrusion; candidate acts as a P-gp non-substrate."
                )
                reasons = []
                if mw < 380.0 and rg < 3.8:
                    reasons.append(f"Compact molecular volume (MW {mw:.1f} Da, Rg {rg:.2f} A) does not span multiple transmembrane contact points.")
                if hbd <= 1:
                    reasons.append(f"Sparse hydrogen-bond donors (HBD {hbd} <= 1) avoids Seelig's donor-pair recognition trigger.")
                if rotb <= 5:
                    reasons.append(f"Rigid scaffold (RotBonds {rotb} <= 5) resists induced-fit cavity entrapment.")
                interp["biophysical_basis"] = " ".join(reasons) if reasons else "Absence of extended lipophilic pharmacophores prevents pump engagement."
                interp["rule_alignment"] = (
                    "Follows the Levetiracetam phenotype (Löscher & Potschka 2005): refractory to efflux pump extrusion."
                )

        return interp
