import numpy as np
from collections import defaultdict

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

def generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def scaffold_split(smiles_list, test_size=0.2, val_size=0.2, n_splits=10, random_state=0):
    np.random.seed(random_state)

    scaffolds = defaultdict(list)
    
    for idx, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles)
        scaffolds[scaffold].append(idx)

    scaffold_sets = list(scaffolds.values())

    splits = {'train': [], 'valid': [], 'test': []}

    for s in range(n_splits):
        np.random.shuffle(scaffold_sets)
        
        train_idx, test_idx, val_idx = [], [], []
    
        for scaffold_set in scaffold_sets:
            if len(test_idx) < test_size * len(smiles_list):
                test_idx.extend(scaffold_set)
            elif len(val_idx) < val_size * len(smiles_list):
                val_idx.extend(scaffold_set)
            else:
                train_idx.extend(scaffold_set)

        splits['train'].append(train_idx)
        splits['valid'].append(val_idx)
        splits['test'].append(test_idx)

    return splits

def random_split(n, test_size=0.2, val_size=0.2, n_splits=10, random_state=0):
    np.random.seed(random_state)

    splits = {'train': [], 'valid': [], 'test': []}

    idxs = np.arange(n)
    for s in range(n_splits): 
        np.random.shuffle(idxs)
        
        splits['train'].append(idxs[:int(n * (1 - val_size - test_size))].copy())
        splits['valid'].append(idxs[int(n * (1 - val_size - test_size)):int(n * (1 - val_size))].copy())
        splits['test'].append(idxs[int(n * (1 - val_size)):].copy())

    return splits

def get_ids(ls, ids): 
    return [ls[i] for i in ids]



