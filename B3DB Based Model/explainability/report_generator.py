"""
Scientific Report Generation Module.
Exports comprehensive dual-barrier predictions, TreeSHAP attributions,
shared-properties tables, and applicability domain assessments into
structured CSV, JSON, and publication-ready HTML reports.
"""

import json
import csv
import datetime
import os
from typing import Dict, Any, Optional


class ScientificReportGenerator:
    """
    Serializes dual-barrier predictions and explanations into CSV, JSON, and standalone HTML reports.
    """

    @staticmethod
    def export_json(report_data: Dict[str, Any], output_path: str):
        """Saves complete analysis dictionary to formatted JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)

    @staticmethod
    def export_csv(report_data: Dict[str, Any], output_path: str):
        """Exports key predictions, descriptors, and top feature attributions to CSV."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        flat_record = {
            "SMILES": report_data.get("smiles", ""),
            "Timestamp": report_data.get("timestamp", ""),
            "DualBarrier_Category": report_data.get("quadrant_category", ""),
            "BBB_Prediction": report_data.get("bbb_class_label", ""),
            "BBB_Probability": report_data.get("bbb_prob", 0.0),
            "Pgp_Prediction": report_data.get("pgp_class_label", ""),
            "Pgp_Probability": report_data.get("pgp_prob", 0.0),
            "Applicability_Domain_BBB": report_data.get("applicability_domain_bbb", {}).get("domain_status", ""),
            "Applicability_Domain_Pgp": report_data.get("applicability_domain_pgp", {}).get("domain_status", ""),
            "Max_Tanimoto_BBB": report_data.get("applicability_domain_bbb", {}).get("max_tanimoto_similarity", 0.0),
            "Max_Tanimoto_Pgp": report_data.get("applicability_domain_pgp", {}).get("max_tanimoto_similarity", 0.0),
        }

        # Add continuous descriptors
        raw_desc = report_data.get("raw_descriptors", {})
        for k, v in raw_desc.items():
            flat_record[f"Desc_{k}"] = v

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_record.keys()))
            writer.writeheader()
            writer.writerow(flat_record)

    @staticmethod
    def export_html(report_data: Dict[str, Any], output_path: str):
        """Generates a publication-grade standalone HTML scientific report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        smiles = report_data.get("smiles", "")
        timestamp = report_data.get("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        quadrant = report_data.get("quadrant_category", "Unknown")
        quadrant_desc = report_data.get("quadrant_description", "")
        bbb_label = report_data.get("bbb_class_label", "")
        bbb_prob = report_data.get("bbb_prob", 0.0)
        pgp_label = report_data.get("pgp_class_label", "")
        pgp_prob = report_data.get("pgp_prob", 0.0)

        bbb_expl = report_data.get("bbb_explanation", {})
        pgp_expl = report_data.get("pgp_explanation", {})
        shared_prop = report_data.get("shared_properties", {})
        app_domain_bbb = report_data.get("applicability_domain_bbb", {})
        app_domain_pgp = report_data.get("applicability_domain_pgp", {})
        counterfactuals = report_data.get("counterfactuals", [])

        # Helpers for HTML tables
        def render_feature_table(feat_list):
            if not feat_list:
                return "<p style='color:#718096;'>No dominant features in this direction.</p>"
            rows = []
            for item in feat_list:
                f_name = item.get("feature", "")
                shap_val = item.get("shap_value", 0.0)
                raw_v = item.get("raw_value", item.get("input_value", "-"))
                if isinstance(raw_v, float):
                    raw_str = f"{raw_v:.3f}"
                else:
                    raw_str = str(raw_v)
                
                # Substructure fragments
                subs = item.get("substructures", [])
                if subs:
                    sub_info = f"<code>{subs[0].get('fragment_smiles', '')}</code> ({subs[0].get('functional_group', '')})"
                else:
                    sub_info = "<span style='color:#a0aec0;'>Continuous / Global</span>"

                shap_color = "#38a169" if shap_val > 0 else "#e53e3e"
                rows.append(f"""
                <tr>
                    <td><strong>{f_name}</strong></td>
                    <td>{raw_str}</td>
                    <td style='color:{shap_color}; font-weight:bold;'>{shap_val:+.4f}</td>
                    <td>{sub_info}</td>
                </tr>
                """)
            return f"""
            <table class='data-table'>
                <thead>
                    <tr><th>Feature</th><th>Value</th><th>TreeSHAP Attribution (Margin)</th><th>Substructural Motif</th></tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
            """

        # Shared Continuous Table
        shared_rows = []
        for row in shared_prop.get("continuous_table", []):
            shared_rows.append(f"""
            <tr>
                <td><strong>{row.get('property_name')}</strong></td>
                <td>{row.get('molecular_value', 0):.2f}</td>
                <td style='font-weight:bold;'>{row.get('bbb_contribution', 0):+.4f}</td>
                <td style='font-weight:bold;'>{row.get('pgp_contribution', 0):+.4f}</td>
                <td><span class='badge'>{row.get('task_relationship')}</span></td>
                <td><small>{row.get('scientific_interpretation')}</small></td>
            </tr>
            """)
        shared_table_html = "".join(shared_rows)

        # Counterfactuals HTML
        cf_html = ""
        if counterfactuals:
            cf_cards = []
            for cf in counterfactuals:
                cf_cards.append(f"""
                <div class='cf-card'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <span class='badge' style='background:#2b6cb0;'>Tanimoto: {cf.get('tanimoto_similarity')}</span>
                        <span style='font-size:0.85rem; color:#4a5568;'>{cf.get('transformation')}</span>
                    </div>
                    <p style='margin:8px 0; font-family:monospace; word-break:break-all;'><strong>Analog SMILES:</strong> {cf.get('analog_smiles')}</p>
                    <p style='color:#2f855a; margin:4px 0;'><strong>Effect:</strong> {cf.get('objective_note')}</p>
                    <small>BBB Prob: {cf.get('bbb_prob')*100:.1f}% | P-gp Prob: {cf.get('pgp_prob')*100:.1f}%</small>
                </div>
                """)
            cf_html = f"""
            <div class='card'>
                <h2>6. Chemically Valid Counterfactual Analogs (Molecular Optimization)</h2>
                {''.join(cf_cards)}
            </div>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dual-Barrier CNS Drug Report - {smiles}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f7fafc;
            color: #2d3748;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1a365d, #2b6cb0);
            color: white;
            padding: 24px 30px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 1.8rem; }}
        .header p {{ margin: 4px 0; font-size: 0.95rem; opacity: 0.9; }}
        .card {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            border: 1px solid #e2e8f0;
        }}
        .card h2 {{
            color: #1a365d;
            margin-top: 0;
            font-size: 1.3rem;
            border-bottom: 2px solid #edf2f7;
            padding-bottom: 10px;
        }}
        .quadrant-box {{
            padding: 18px 24px;
            border-radius: 8px;
            margin: 15px 0;
            border-left: 6px solid #3182ce;
            background: #ebf8ff;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .metric-badge {{
            display: inline-block;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
        }}
        .badge-green {{ background: #c6f6d5; color: #22543d; }}
        .badge-red {{ background: #fed7d7; color: #742a2a; }}
        .badge-yellow {{ background: #fefcbf; color: #744210; }}
        .badge {{
            display: inline-block;
            background: #edf2f7;
            color: #4a5568;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9rem;
        }}
        .data-table th, .data-table td {{
            padding: 10px 12px;
            border: 1px solid #e2e8f0;
            text-align: left;
        }}
        .data-table th {{
            background-color: #f7fafc;
            color: #4a5568;
            font-weight: 600;
        }}
        .cf-card {{
            background: #f8fafc;
            border: 1px solid #cbd5e0;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 12px;
        }}
        .disclaimer {{
            background: #fffaf0;
            border-left: 4px solid #dd6b20;
            padding: 12px 16px;
            font-size: 0.85rem;
            color: #7b341e;
            border-radius: 4px;
            margin-top: 20px;
        }}
        code {{
            background: #edf2f7;
            padding: 2px 5px;
            border-radius: 3px;
            font-size: 0.88rem;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Dual-Barrier CNS Drug Candidate Evaluation Report</h1>
        <p><strong>Candidate SMILES:</strong> <code>{smiles}</code></p>
        <p><strong>Generated on:</strong> {timestamp} | <strong>Pipeline Version:</strong> Dual-Barrier XAI v2.0</p>
    </div>

    <!-- 1. Dual-Barrier Classification -->
    <div class="card">
        <h2>1. Dual-Barrier Four-Quadrant Assessment</h2>
        <div class="quadrant-box">
            <h3 style="margin:0 0 8px 0; font-size:1.2rem;">Classification: {quadrant}</h3>
            <p style="margin:0; font-size:0.95rem;">{quadrant_desc}</p>
        </div>
        <div class="grid-2" style="margin-top:15px;">
            <div style="background:#f7fafc; padding:15px; border-radius:6px;">
                <p style="margin:0 0 5px 0;"><strong>Blood-Brain Barrier (BBB) Permeability:</strong></p>
                <span class="metric-badge {'badge-green' if bbb_prob > 0.5 else 'badge-red'}">{bbb_label}</span>
                <span style="margin-left:10px; font-weight:600;">{bbb_prob*100:.1f}% Confidence</span>
                <p style="font-size:0.85rem; color:#718096; margin-top:8px;">Model: Bayesian Optuna-XGBoost (B3DB Benchmark, 7,807 molecules)</p>
            </div>
            <div style="background:#f7fafc; padding:15px; border-radius:6px;">
                <p style="margin:0 0 5px 0;"><strong>P-glycoprotein (ABCB1) Efflux Status:</strong></p>
                <span class="metric-badge {'badge-red' if pgp_prob > 0.5 else 'badge-green'}">{pgp_label}</span>
                <span style="margin-left:10px; font-weight:600;">{pgp_prob*100:.1f}% Confidence</span>
                <p style="font-size:0.85rem; color:#718096; margin-top:8px;">Model: 3D Conformation-Augmented XGBoost (Broccatelli TDC Benchmark)</p>
            </div>
        </div>
    </div>

    <!-- 2. BBB Detailed Explainability -->
    <div class="card">
        <h2>2. Blood-Brain Barrier (BBB) Permeability Explainability (TreeSHAP)</h2>
        <p><strong>Model Base Value (&phi;<sub>0</sub>):</strong> {bbb_expl.get('base_value', 0):+.4f} | <strong>Total Margin Output f(x):</strong> {bbb_expl.get('model_margin', 0):+.4f} (Log-odds space)</p>
        <p><strong>Biophysical Basis:</strong> {bbb_expl.get('interpretation', {}).get('biophysical_basis', 'N/A')}</p>
        <p><strong>Medicinal Chemistry Reference:</strong> {bbb_expl.get('interpretation', {}).get('rule_alignment', 'N/A')}</p>
        
        <h4 style="margin:18px 0 8px 0; color:#2c5282;">Top Features Supporting Predicted BBB Outcome</h4>
        {render_feature_table(bbb_expl.get('top_supporting', []))}

        <h4 style="margin:18px 0 8px 0; color:#742a2a;">Top Features Opposing Predicted BBB Outcome</h4>
        {render_feature_table(bbb_expl.get('top_opposing', []))}
    </div>

    <!-- 3. P-gp Detailed Explainability -->
    <div class="card">
        <h2>3. P-glycoprotein (P-gp / ABCB1) Efflux Explainability (TreeSHAP & 3D Conformation)</h2>
        <p><strong>Model Base Value (&phi;<sub>0</sub>):</strong> {pgp_expl.get('base_value', 0):+.4f} | <strong>Total Margin Output f(x):</strong> {pgp_expl.get('model_margin', 0):+.4f}</p>
        <p><strong>Efflux Cavity Mechanism:</strong> {pgp_expl.get('interpretation', {}).get('biophysical_basis', 'N/A')}</p>
        <p><strong>Pharmacological Reference:</strong> {pgp_expl.get('interpretation', {}).get('rule_alignment', 'N/A')}</p>

        <h4 style="margin:18px 0 8px 0; color:#2c5282;">Top Features Supporting P-gp Outcome</h4>
        {render_feature_table(pgp_expl.get('top_supporting', []))}

        <h4 style="margin:18px 0 8px 0; color:#742a2a;">Top Features Opposing P-gp Outcome</h4>
        {render_feature_table(pgp_expl.get('top_opposing', []))}
    </div>

    <!-- 4. Shared Properties Analysis -->
    <div class="card">
        <h2>4. BBB - P-gp Shared Molecular Properties Analysis</h2>
        <p style="font-size:0.95rem; color:#4a5568;">Comparative assessment of 6 common physicochemical descriptors across both barrier mechanisms:</p>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                    <th>BBB Model Impact (&phi;)</th>
                    <th>P-gp Model Impact (&phi;)</th>
                    <th>Task Relationship</th>
                    <th>Scientific Interpretation</th>
                </tr>
            </thead>
            <tbody>
                {shared_table_html}
            </tbody>
        </table>
        <div style="background:#f7fafc; padding:12px; border-radius:6px; font-size:0.9rem;">
            <strong>Cross-Task Synthesis:</strong> {shared_prop.get('task_divergence_summary', '')}
        </div>
    </div>

    <!-- 5. Applicability Domain -->
    <div class="card">
        <h2>5. Applicability Domain & Uncertainty Assessment</h2>
        <div class="grid-2">
            <div style="background:#f7fafc; padding:14px; border-radius:6px;">
                <h4 style="margin:0 0 8px 0;">BBB Chemical Domain</h4>
                <p style="margin:4px 0;">Status: <strong>{app_domain_bbb.get('domain_status')}</strong></p>
                <p style="margin:4px 0;">Max Tanimoto to Training: <code>{app_domain_bbb.get('max_tanimoto_similarity')}</code></p>
                <p style="margin:4px 0;">Descriptor Leverage: <code>{app_domain_bbb.get('descriptor_leverage_distance')}</code></p>
            </div>
            <div style="background:#f7fafc; padding:14px; border-radius:6px;">
                <h4 style="margin:0 0 8px 0;">P-gp Chemical Domain</h4>
                <p style="margin:4px 0;">Status: <strong>{app_domain_pgp.get('domain_status')}</strong></p>
                <p style="margin:4px 0;">Max Tanimoto to Training: <code>{app_domain_pgp.get('max_tanimoto_similarity')}</code></p>
                <p style="margin:4px 0;">Descriptor Leverage: <code>{app_domain_pgp.get('descriptor_leverage_distance')}</code></p>
            </div>
        </div>
    </div>

    <!-- 6. Counterfactuals (if any) -->
    {cf_html}

    <div class="disclaimer">
        <strong>Scientific and Regulatory Disclaimer:</strong> This computational report is designed for preclinical screening, lead optimization, and research prioritization. A predicted BBB+ / P-gp- profile does not guarantee in vivo CNS efficacy or safety, as total brain exposure depends upon free unbound fraction (f_u,brain), metabolic stability, plasma protein binding, and non-Pgp clearance pathways.
    </div>
</div>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
