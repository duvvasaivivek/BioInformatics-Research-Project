"""Model-bound preprocessing metadata and molecular identity safeguards."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.model_selection import train_test_split


def clean_structures(df, smiles_col='SMILES', label_col='Label'):
    """Drop invalid structures and conflicting labels; retain one row per identity."""
    frame = df.copy()
    def identity(value):
        mol = Chem.MolFromSmiles(str(value))
        return Chem.MolToInchiKey(mol) if mol is not None and mol.GetNumAtoms() else None
    frame['_identity'] = frame[smiles_col].map(identity)
    frame = frame.dropna(subset=['_identity', label_col])
    if not frame[label_col].isin([0, 1]).all():
        raise ValueError('Labels must be binary')
    counts = frame.groupby('_identity')[label_col].nunique()
    frame = frame[frame['_identity'].isin(counts[counts == 1].index)]
    return frame.drop_duplicates('_identity').drop(columns='_identity').reset_index(drop=True)


def save_model_bundle(model, model_path, metadata):
    """Atomically publish a model with its scaler and training identities inside it."""
    model.get_booster().set_attr(preprocessing_metadata=json.dumps(metadata))
    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(suffix='.json', dir=path.parent)
    os.close(fd)
    try:
        model.save_model(temp)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def read_metadata(scaler_path):
    path = Path(scaler_path)
    model_path = path.with_name(path.name.replace('_scaler_metadata', '_xgboost'))
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding='utf-8'))
        embedded = model['learner'].get('attributes', {}).get('preprocessing_metadata')
        if embedded:
            return json.loads(embedded)
    if not path.exists():
        raise FileNotFoundError(f'Required scaler metadata missing: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('model_sha256') and model_path.exists():
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != data['model_sha256']:
            raise ValueError('Scaler metadata does not belong to the saved model')
    return data


def training_smiles(data_dir, task):
    """Use persisted identities, or verify reconstruction for historical models."""
    base = Path(data_dir).parent
    metadata = read_metadata(base/'models'/f'{task}_scaler_metadata.json')
    if 'training_smiles' in metadata:
        return metadata['training_smiles']
    filename = 'processed_b3db_data_advanced.csv' if task == 'bbb' else 'processed_pgp_data.csv'
    df = pd.read_csv(Path(data_dir)/filename)
    train, _ = train_test_split(df, test_size=.15 if task == 'bbb' else .2,
                               random_state=42, stratify=df.Label)
    if task == 'bbb':
        train, _ = train_test_split(train, test_size=.15/.85, random_state=42, stratify=train.Label)
    values = train[metadata['features']].to_numpy(float)
    std = values.std(axis=0)
    std[std == 0] = 1
    if not (np.allclose(values.mean(axis=0), metadata['means'], atol=1e-7) and
            np.allclose(std, metadata['stds'], atol=1e-7)):
        raise ValueError('Cannot verify historical training partition; retrain or restore original data')
    return train.SMILES.tolist()
