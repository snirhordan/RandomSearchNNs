import numpy as np
import networkx as nx

import torch
import torch_geometric

from torch_geometric.data import Data

from rdkit import Chem

global allowable_features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'misc'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ], 
    'possible_is_conjugated_list': [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def atom_to_feature_vector(atom):
    """
    OGB atom to veature vector preprocessing 
    
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            safe_index(allowable_features['possible_chirality_list'], str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature

def get_canonical_molecule(mol):
    """
    Takes an RDKit molecule as input and returns a new molecule
    with atoms in canonical order and the atom map numbers set to their
    canonical indices.

    Parameters:
    - mol: RDKit Mol object

    Returns:
    - A new RDKit Mol object with canonical atom order and annotated map numbers
    """
    # Generate canonical SMILES from the input molecule
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    # Create a new molecule from the canonical SMILES
    canonical_mol = Chem.MolFromSmiles(canonical_smiles)

    # Set atom map numbers to correspond to canonical indices
    for atom in canonical_mol.GetAtoms():
        canonical_idx = atom.GetIdx()  # This is already the canonical index
        atom.SetAtomMapNum(canonical_idx+1)

    return canonical_mol

def get_emb(mol, tokenizer, vocab): 
    smiles = Chem.MolToSmiles(mol, doRandom=False, canonical=True, allHsExplicit=False, ignoreAtomMapNumbers=False)
    smiles_tok = tokenizer.findall(smiles)

    modified_components = torch.empty((mol.GetNumAtoms(), ), dtype=torch.int)
    for component in smiles_tok:
        if component.startswith('[') and ':' in component:
            # Extract the map number
            atom_number = int(component.split(':')[1][:-1]) - 1  # Extracts 1 from "[C:1]"
            # Remove the atom number from the component
            base_component = component.split(':')[0][1:] # Removes map number, e.g., "[C:1]" becomes "[C]"
            modified_components[atom_number] = vocab[base_component]

    return modified_components

def mol2graph(mol, tokenizer=None, vocab=None):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # mol = Chem.MolFromSmiles(smiles_string)
    # # mol = Chem.RemoveAllHs(mol)
    # mol = mol if removeHs else Chem.AddHs(mol)
    mol = get_canonical_molecule(mol)
    
    # featurize atoms
    atom_features_list = []
    atom_labels_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # featurize bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    edge_index = torch_geometric.utils.to_undirected(torch.tensor(edge_index, dtype=torch.int64)) # makes sure edge indices are undirected
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if vocab != None and tokenizer != None: 
        x_emb = get_emb(mol, tokenizer, vocab)
        data.x_emb = x_emb

    data = get_neighbor_dict(data)

    return data 

def get_ids(ls, ids): 
    return [ls[i] for i in ids]

def add_laplacian_eigs(data, k):
    '''
    Adds eigenvector and eigenvalue positional encoding to pytorch data object

    Code from https://github.com/cptq/SignNet-BasisNet/blob/main/LearningFilters/utils.py
    '''
    # A = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index).todense()
    # A = torch_geometric.utils.to_dense_adj(data.edge_index).squeeze().numpy()
    A = torch.zeros(data.x.shape[0], data.x.shape[0])
    A[data.edge_index[0, :], data.edge_index[1, :]] = 1
    A = A.numpy()
    nnodes = A.shape[0]
    D_vec = np.sum(A, axis=1)
    D_vec[D_vec == 0] = 1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = np.diag(D_vec_invsqrt_corr)
    L = np.eye(nnodes)-D_invsqrt_corr @ A @ D_invsqrt_corr
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    eig_min = np.min(eigenvectors, axis=0)
    eig_max = np.max(eigenvectors, axis=0)
    eig_max[eig_max == eig_min] = 1
    eig_min[eig_max == eig_min] = -1
    eigenvectors_norm = (eigenvectors - eig_min)/(eig_max - eig_min) * (1 + 1) - 1

    # zero-pad if number of eigenvalues is less than k
    if eigenvalues.shape[0] < k: 
        eigenvalues = np.concatenate([eigenvalues, np.zeros((k - eigenvalues.shape[0], ))], axis=0)
        eigenvectors = np.concatenate([eigenvectors, np.zeros((eigenvectors.shape[0], k - eigenvectors.shape[1]))], axis=1)
        eigenvectors_norm = np.concatenate([eigenvectors_norm, np.zeros((eigenvectors_norm.shape[0], k - eigenvectors_norm.shape[1]))], axis=1)

    data.eigenvectors_norm = torch.from_numpy(eigenvectors_norm).float()[:, :k]
    data.eigenvectors = torch.from_numpy(eigenvectors).float()[:, :k]
    data.eigenvalues = torch.from_numpy(eigenvalues).float()[:k][None, :k]

    return data

def add_rwse(data, m): 
    """
    Initializing positional encoding with RWSE (diagonal of m-step random walk matrix) (Dwivedi et al., 2022, Rampasek et al., 2022, Mueller et al., 2024)

    Code adapted from https://github.com/vijaydwivedi75/gnn-lspe/blob/main/data/molecules.py
    """
    # construct random walk matrix: RW := D^-1 x A (since we are only concerned with diagonals, rowwise is equivalent to colwise)
    # A = np.array(torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index).todense())
    # A = torch_geometric.utils.to_dense_adj(data.edge_index).numpy()
    A = torch.zeros(data.x.shape[0], data.x.shape[0])
    A[data.edge_index[0, :], data.edge_index[1, :]] = 1
    A = A.numpy()
    D_vec = np.sum(A, axis=1).flatten()
    D_vec[D_vec == 0] = 1
    D_vec_inv = 1 / D_vec
    D_inv = np.diag(D_vec_inv)
    RW = A @ D_inv
    RW_k = RW.copy()

    # iterate m-steps of the random walk matrix and add diagonals to random_walks
    random_walks = [RW.diagonal()[:, None]]
    for k in range(m - 1): 
        RW_k = RW_k @ RW
        random_walks.append(RW_k.diagonal()[:, None])

    # concatenate the encodings and add to pytorch dataset
    data.random_walks = torch.from_numpy(np.concatenate(random_walks, axis=-1)).float()
    
    return data

def get_neighbor_dict(data):
    """
    Computes and returns the neighbor dictionary for a graph represented by data.edge_index.
    The neighbor lists are stored as sets for fast membership testing.
    The dictionary is stored in a private attribute data._neighbor_dict so that it is not included
    in the collate operation.
    """
    if '_neighbor_dict' in data.__dict__:
        return data._neighbor_dict

    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    neighbor_dict = {i: set() for i in range(num_nodes)}
    edge_index = data.edge_index
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        neighbor_dict[src].add(dst)
        
    data._neighbor_dict = neighbor_dict
    return data


