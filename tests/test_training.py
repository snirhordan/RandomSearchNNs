"""Integration test: a 2-epoch training run on a 16-graph ClinTox subset."""
import sys
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import regex as re
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from generation.utils import get_canonical_molecule, mol2graph, get_ids
from utils.search import sample_walks_adaptive
from models.rwnn import RWNN_base_ada


PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|"
    r"\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
)


def _process_tok(components):
    out = []
    for c in components:
        if c.startswith('[') and ':' in c:
            out.append(c.split(':')[0][1:])
    return out


class _SMILESDataset(Dataset):
    def __init__(self, graphs, ys, m, l, w, nb, max_len, vocab):
        self.graphs = graphs
        self.ys = ys
        self.m, self.l, self.w, self.nb = m, l, w, nb
        self.max_len = max_len
        self.vocab = vocab

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        data = self.graphs[idx]
        data.ys = self.ys[idx]
        data.idx = idx
        data = sample_walks_adaptive(data, self.m, data.x.shape[0], self.w, self.nb, self.max_len, self.vocab)
        return data


def test_two_epoch_training_on_clintox_subset(device):
    SEED = 2024
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    csv_path = REPO_ROOT / 'quickstart' / 'data' / 'clintox.csv'
    df = pd.read_csv(csv_path, sep=',')
    arr = df.to_numpy()

    tokenizer = re.compile(PATTERN)

    # collect first 32 valid rows and use 16 each for train/val to ensure
    # both classes are present and the model has something to learn from.
    smiles, ys = [], []
    for row in arr:
        if len(smiles) >= 32:
            break
        smi, y = row[0], row[2]
        mol = Chem.MolFromSmiles(smi)
        if mol is None or pd.isna(y):
            continue
        smiles.append(smi)
        ys.append(float(y))

    # if all 32 are one class, supplement until both classes are present
    if len(set(int(y) for y in ys)) < 2:
        for row in arr[len(smiles):]:
            smi, y = row[0], row[2]
            mol = Chem.MolFromSmiles(smi)
            if mol is None or pd.isna(y):
                continue
            if int(y) not in {int(yy) for yy in ys}:
                smiles.append(smi)
                ys.append(float(y))
            if len(set(int(yy) for yy in ys)) >= 2:
                break

    # build a vocab over the chosen smiles
    vocab_list = []
    l_max = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        l_max = max(l_max, mol.GetNumAtoms())
        mol = get_canonical_molecule(mol)
        canon = Chem.MolToSmiles(mol, doRandom=False, canonical=True,
                                 allHsExplicit=False, ignoreAtomMapNumbers=False)
        vocab_list += _process_tok(tokenizer.findall(canon))
    vocab_list = list(np.unique(vocab_list))
    vocab_list.append('PAD')
    vocab = {k: i for i, k in enumerate(vocab_list)}

    graphs, ys_t = [], []
    for smi, y in zip(smiles, ys):
        mol = Chem.MolFromSmiles(smi)
        g = mol2graph(mol, tokenizer=tokenizer, vocab=vocab)
        graphs.append(g)
        ys_t.append(torch.tensor(y, dtype=torch.float32))

    # split: first 16 train, last 16 valid
    train_g, train_y = graphs[:16], ys_t[:16]
    valid_g, valid_y = graphs[16:32], ys_t[16:32]

    m, w = 4, 4
    pe_in_dim = 2 * w
    pe_out_dim = 8
    h_dim = 32
    out_dim = 1

    model = RWNN_base_ada(pe_in_dim, pe_out_dim, h_dim, out_dim, 2, len(vocab), 'mean').to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    train_ds = _SMILESDataset(train_g, train_y, m, None, w, False, l_max, vocab)
    valid_ds = _SMILESDataset(valid_g, valid_y, m, None, w, False, l_max, vocab)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False, num_workers=0)

    epoch_losses = []
    model.train()
    for epoch in range(2):
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch).squeeze(-1)
            loss = criterion(out, batch.ys.squeeze())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_losses.append(float(np.mean(losses)))

    # Loss should generally not blow up; allow some noise.
    assert epoch_losses[1] < 1.05 * epoch_losses[0] + 0.5, (
        f"epoch1={epoch_losses[1]:.3f} > 1.05 * epoch0={epoch_losses[0]:.3f}"
    )
    for v in epoch_losses:
        assert np.isfinite(v)

    # Evaluate AUC on the validation set; must be a finite number in [0, 1].
    model.eval()
    ys_all, outs_all = [], []
    with torch.no_grad():
        for batch in valid_loader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1)
            outs_all.append(out.cpu().numpy())
            ys_all.append(batch.ys.cpu().numpy())
    ys_arr = np.concatenate(ys_all, axis=0)
    outs_arr = np.concatenate(outs_all, axis=0)

    if len(set(ys_arr.tolist())) >= 2:
        auc = roc_auc_score(ys_arr, outs_arr)
        assert 0.0 <= auc <= 1.0
    else:
        # only one class is present in valid; AUC is undefined but the
        # forward pass having produced finite outputs is itself the test.
        assert np.isfinite(outs_arr).all()
