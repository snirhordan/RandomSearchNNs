import os
import argparse
import pandas as pd
import numpy as np
import regex as re
import pickle as pkl

import torch
from rdkit import Chem
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')

from utils import *
from scaffold_split import *

def process_tok(components): 
    modified_components = []
    for component in components:
        if component.startswith('[') and ':' in component:
            # Extract the map number
            atom_number = int(component.split(':')[1][:-1]) - 1  # Extracts 1 from "[C:1]"
            # Remove the atom number from the component
            base_component = component.split(':')[0][1:] # Removes map number, e.g., "[C:1]" becomes "[C]"
            modified_components.append(base_component)

    return modified_components

if __name__ == '__main__': 
    # argparse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx_y', type=int, default=2) # y index for dataset
    parser.add_argument('--idx_smiles', type=int, default=0) # smiles index for dataset
    parser.add_argument('--l_max', type=int, default=-1) # maximum string length
    parser.add_argument('--scaffold', type=int, default=1)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--save_directory', type=str)
    
    args = parser.parse_args()
    idx_y = args.idx_y
    idx_smiles = args.idx_smiles
    l_max = args.l_max # TODO: IMPLEMENT MAX LEN CUTOFF! 
    scaffold = args.scaffold
    data_name = args.data_name
    save_directory = args.save_directory

    data_path = f'/home/mbito/data/benchmarks/MoleculeNet/{data_name}.csv'
    df = pd.read_csv(data_path, sep=',')
    data = df.to_numpy()

    PATTERN = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex_tokenizer = re.compile(PATTERN)
    
    vocab = []
    for idx, smiles in enumerate(data[:, idx_smiles]): 
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and np.isnan(data[idx, idx_y]) == False: 
            if mol.GetNumAtoms() > l_max:
                l_max = mol.GetNumAtoms()
            mol = get_canonical_molecule(mol)
            
            smiles = Chem.MolToSmiles(mol, doRandom=False, canonical=True, allHsExplicit=False, ignoreAtomMapNumbers=False)
            smiles_tok = process_tok(regex_tokenizer.findall(smiles))
            vocab += smiles_tok
    
    vocab_d = {}
    vocab = list(np.unique(vocab))
    vocab.append('PAD')
    for i, key in enumerate(vocab): 
        vocab_d[key] = i
    
    vocab = vocab_d
    
    ys_data, graphs, smiles_scaffold = [], [], []
    for idx, smiles in enumerate(data[:, idx_smiles]): 
        mol = Chem.MolFromSmiles(smiles)
        if mol != None and np.isnan(data[idx, idx_y]) == False: 
            mol = get_canonical_molecule(mol)
            ys_data.append(torch.from_numpy(np.array(data[idx, idx_y])).float())
            graphs.append(mol2graph(mol, tokenizer=regex_tokenizer, vocab=vocab))
            smiles_scaffold.append(smiles)

    if scaffold: 
        splits = scaffold_split(smiles_scaffold, test_size=0.2, val_size=0.2, random_state=0)
    else: 
        splits = random_split(len(graphs), test_size=0.2, val_size=0.2, random_state=0)

    # save raw dataset and tokenizer
    data_dict = {}
    data_dict['l_max'] = l_max
    data_dict['vocab'] = vocab
    data_dict['graphs'] = graphs
    data_dict['ys_data'] = ys_data
    data_dict['splits'] = splits
    
    data_name = f'{data_name}_{idx_y}_rws.pkl'
        
    with open(os.path.join(save_directory, data_name), 'wb') as file: 
        pkl.dump(data_dict, file, protocol=pkl.HIGHEST_PROTOCOL)

    print(f'Processed dataset {data_name}!')

    


