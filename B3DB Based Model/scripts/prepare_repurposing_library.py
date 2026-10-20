"""
Download and inspect Broad Institute Drug Repurposing Hub dataset.
Filters for Launched / Approved small molecules and prepares screening candidate table.
"""

import os
import urllib.request
import pandas as pd
from rdkit import Chem

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "repurposing")
os.makedirs(DATA_DIR, exist_ok=True)

DRUGS_URL = "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt"
SAMPLES_URL = "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_samples_20200324.txt"

DRUGS_FILE = os.path.join(DATA_DIR, "repurposing_drugs_20200324.txt")
SAMPLES_FILE = os.path.join(DATA_DIR, "repurposing_samples_20200324.txt")


def download_file(url: str, dest: str):
    if not os.path.exists(dest):
        print(f"Downloading {url} to {dest}...")
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(dest, 'wb') as out:
            out.write(resp.read())
        print(f"Saved {dest} ({os.path.getsize(dest)} bytes)")
    else:
        print(f"File already exists: {dest}")


def load_and_merge():
    download_file(DRUGS_URL, DRUGS_FILE)
    download_file(SAMPLES_URL, SAMPLES_FILE)

    # Read tables skipping metadata comment lines starting with '!'
    df_drugs = pd.read_csv(DRUGS_FILE, sep='\t', comment='!')
    df_samples = pd.read_csv(SAMPLES_FILE, sep='\t', comment='!')

    print(f"Loaded {len(df_drugs)} drug annotation rows.")
    print(f"Loaded {len(df_samples)} sample structure rows.")

    # Merge on pert_iname
    df_merged = pd.merge(
        df_samples[['pert_iname', 'smiles']].dropna(subset=['smiles']).drop_duplicates(subset=['pert_iname']),
        df_drugs[['pert_iname', 'clinical_phase', 'moa', 'target', 'disease_area', 'indication']],
        on='pert_iname',
        how='inner'
    )

    print(f"Merged dataset: {len(df_merged)} unique compounds with valid SMILES.")
    print("Clinical Phase Distribution:")
    print(df_merged['clinical_phase'].value_counts())

    # Focus on Launched (Approved) drugs
    df_launched = df_merged[df_merged['clinical_phase'] == 'Launched'].copy()
    print(f"\nTotal Launched (FDA/EMA Approved) Drugs: {len(df_launched)}")

    # Clean and validate SMILES with RDKit
    valid_rows = []
    for idx, row in df_launched.iterrows():
        smi = str(row['smiles'])
        try:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                can_smi = Chem.MolToSmiles(m)
                valid_rows.append({
                    'drug_name': row['pert_iname'],
                    'smiles': can_smi,
                    'clinical_phase': row['clinical_phase'],
                    'moa': row['moa'],
                    'target': row['target'],
                    'disease_area': row['disease_area'],
                    'indication': row['indication']
                })
        except Exception:
            pass

    df_clean = pd.DataFrame(valid_rows).drop_duplicates(subset=['smiles'])
    out_csv = os.path.join(DATA_DIR, "fda_approved_repurposing_compounds.csv")
    df_clean.to_csv(out_csv, index=False)
    print(f"Saved {len(df_clean)} verified FDA-approved small molecule drugs to {out_csv}")


if __name__ == "__main__":
    load_and_merge()
