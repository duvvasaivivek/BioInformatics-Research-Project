"""
Explainability and Biophysical Analysis Package for Dual-Barrier CNS Drug Predictor.
Provides TreeSHAP attribution, substructure mapping, shared properties analysis,
applicability domain assessment, counterfactual generation, and scientific report exports.
"""

from .shap_explainer import DualBarrierShapExplainer
from .substructure_mapper import MorganSubstructureMapper
from .applicability_domain import ApplicabilityDomainEstimator
from .shared_properties import SharedPropertiesAnalyzer
from .counterfactual import MolecularCounterfactualEngine
from .report_generator import ScientificReportGenerator

__all__ = [
    "DualBarrierShapExplainer",
    "MorganSubstructureMapper",
    "ApplicabilityDomainEstimator",
    "SharedPropertiesAnalyzer",
    "MolecularCounterfactualEngine",
    "ScientificReportGenerator",
]
