"""PDBbind (GET-processed) preprocessing for atom-level RSNN.

This module is purely additive: it does NOT import anything under ``GET/``
(GET's own dataset code pulls in ``Bio.PDB``, which is absent from the
``rwnn`` conda env) and does NOT alter any existing pipeline component. It
consumes the ``.pkl`` files produced by
``GET/scripts/data_process/process_PDBbind_benchmark.py`` directly, decoding
GET's compact block/atom encoding with a self-contained periodic-table
lookup instead of GET's ``Vocab`` class.

Exposes:

- ``ATOMS`` / ``IDX2ATOM``  -- periodic-table element list and the index
                               table GET's ``A`` array indexes into.
- ``element_to_z``           -- element symbol -> atomic number.
- ``decode_item``            -- one raw pickle item -> plain-array dict
                               (global nodes dropped, coords/elements/
                               segment/residue_id/label/pdb_id).
- ``build_pdbbind_vocab``    -- element-symbol vocabulary over decoded items.
- ``save_vocab`` / ``load_vocab`` -- JSON (de)serialization of the vocab.
- ``build_data``             -- decoded dict + vocab -> ``torch_geometric``
                               ``Data`` implementing the radius-graph /
                               inverse-distance-weight / one-hot recipe.
- ``preprocess_split``       -- decode+build a whole split pickle, cache as
                               ``list[Data]`` via ``torch.save``.
- ``load_pdbbind_cache``     -- load a cached split.

Graph-construction recipe (atom-level RSNN, see task spec): nodes are atoms
(protein pocket + ligand, hydrogens already stripped by GET); edges connect
every atom pair with Euclidean distance strictly less than ``cutoff``
(default 4.5 Angstrom), undirected, no self-loops; edge weight is
``1 / d_ij + eps``; node features are one-hot over the element vocabulary.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from torch_cluster import radius_graph as _radius_graph
    _HAS_TORCH_CLUSTER = True
except ImportError:  # pragma: no cover -- exercised only when torch_cluster is absent
    _HAS_TORCH_CLUSTER = False


# ---------------------------------------------------------------------------
# Element vocabulary (verbatim periodic table, GET/data/pdb_utils.py:44-67)
# ---------------------------------------------------------------------------

ATOMS = [  # Periodic Table
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og',
]

# GET's per-atom ``A`` ids index into [PAD, MASK, GLOBAL] + ATOMS (GET's
# ``Vocab.idx2atom``). Ids 0/1/2 never denote a real atom; 2 ('g') marks the
# artificial per-segment global node, which we always drop.
IDX2ATOM = ['p', 'm', 'g'] + ATOMS
assert len(IDX2ATOM) == 121, f"expected 121 idx2atom entries, got {len(IDX2ATOM)}"

_GLOBAL_ATOM_TOKEN = IDX2ATOM[2]  # 'g'

# GET's per-atom ``atom_positions`` ids index into a Greek-level / role code
# table (mirrors IDX2ATOM's placement/verification convention). Ids 0/1/2
# ('p'/'m'/'g') never denote a real atom's position code; '' (id 3) marks
# "no side-chain level" (used by backbone N/C/O); 'A'..'H' + 'XT'/'P' are
# increasing side-chain Greek levels (+ the rare C-terminal OXT / proline
# marker); 'sm'/"'" are ligand-only codes (never observed on protein atoms
# in identity30, see PROTEIN TORSION ORDER ablation confirmation notes).
IDX2POS = ['p', 'm', 'g', ''] + ['A', 'B', 'G', 'D', 'E', 'Z', 'H', 'XT', 'P'] + ['sm'] + ["'"]
assert len(IDX2POS) == 15, f"expected 15 idx2pos entries, got {len(IDX2POS)}"


def element_to_z(sym: str) -> int:
    """Atomic number of an element symbol (``ATOMS`` is in periodic order)."""
    return ATOMS.index(sym) + 1


def z_to_element(z: int) -> str:
    """Inverse of ``element_to_z``: atomic number -> element symbol."""
    return ATOMS[z - 1]


# ---------------------------------------------------------------------------
# Per-item decoding: drop global nodes, slice by block_lengths/segment_ids
# ---------------------------------------------------------------------------


def decode_item(item: dict) -> dict:
    """Decode one raw GET PDBbind pickle item into plain per-atom arrays.

    ``item`` has the shape ``{'id': str, 'affinity': {'neglog_aff': float},
    'data': DATA}`` where ``DATA`` is GET's ``blocks_to_data`` output (see
    ``GET/data/dataset.py``). Blocks within each segment are laid out as
    ``[GLOBAL_block, real_block_1, real_block_2, ...]``; the global block
    (block_length 1, atom id 'g') is dropped here. Segment 0 (protein) real
    blocks are residues in sequence order; segment 1 (ligand) real blocks
    are single-atom blocks.

    Returns a dict with:

    - ``coords``     (N, 3) float64 ndarray
    - ``elements``   list[str], length N, element symbols
    - ``segment``    (N,) int ndarray, 0=protein / 1=ligand
    - ``residue_id`` (N,) int ndarray; 0,1,2,... per protein residue in
                     sequence order, -1 for every ligand atom
    - ``label``      float, ``affinity.neglog_aff``
    - ``pdb_id``      str, ``item['id']``
    - ``atom_positions`` list[str], length N, Greek-level/role code per atom
                     (``IDX2POS[data['atom_positions'][k]]``), row-aligned
                     with ``elements``/``coords``/``segment``/``residue_id``
                     via the SAME global-node-skip + ``lo:hi`` slicing as
                     every other per-atom field (PROTEIN TORSION ORDER
                     ablation; additive, unused unless
                     ``build_protein_torsion_order`` / ``augment_cache_with_torsion``
                     are invoked).
    """
    data = item['data']
    X = np.asarray(data['X'])
    if X.ndim == 3:
        # (Natom, n_channel, 3) -- take the first channel.
        X = X[:, 0, :]
    A = data['A']
    raw_atom_positions = data['atom_positions']
    block_lengths = data['block_lengths']
    segment_ids = data['segment_ids']
    if len(block_lengths) != len(segment_ids):
        raise ValueError(
            f"block_lengths ({len(block_lengths)}) and segment_ids "
            f"({len(segment_ids)}) length mismatch"
        )

    coords: List[np.ndarray] = []
    elements: List[str] = []
    segment: List[int] = []
    residue_id: List[int] = []
    atom_positions: List[str] = []

    cursor = 0
    prev_seg = None
    res_counter = -1
    for blen, seg in zip(block_lengths, segment_ids):
        lo, hi = cursor, cursor + blen
        is_segment_head = seg != prev_seg
        if is_segment_head:
            # First block of a new segment is the artificial global node.
            if blen != 1:
                raise ValueError(
                    f"expected global-node block_length==1, got {blen} "
                    f"(segment {seg})"
                )
            if IDX2ATOM[A[lo]] != _GLOBAL_ATOM_TOKEN:
                raise ValueError(
                    f"expected global-node atom id at segment {seg} head, "
                    f"got element {IDX2ATOM[A[lo]]!r}"
                )
            prev_seg = seg
            res_counter = -1
            cursor = hi
            continue

        for k in range(lo, hi):
            elem = IDX2ATOM[A[k]]
            if elem in ('p', 'm', _GLOBAL_ATOM_TOKEN):
                raise ValueError(f"unexpected non-atom token {elem!r} at atom {k}")
            elements.append(elem)
            coords.append(X[k])
            segment.append(seg)
            atom_positions.append(IDX2POS[raw_atom_positions[k]])

        if seg == 0:
            res_counter += 1
            residue_id.extend([res_counter] * blen)
        else:
            residue_id.extend([-1] * blen)

        cursor = hi

    return {
        'coords': np.asarray(coords, dtype=np.float64),
        'elements': elements,
        'segment': np.asarray(segment, dtype=np.int64),
        'residue_id': np.asarray(residue_id, dtype=np.int64),
        'label': float(item['affinity']['neglog_aff']),
        'pdb_id': str(item['id']),
        'atom_positions': atom_positions,
    }


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def build_pdbbind_vocab(list_of_decoded: List[dict], min_count: int = 1) -> Dict[str, int]:
    """Build an element-symbol vocabulary over decoded PDBbind items.

    ``UNK`` is index 0, then element symbols (occurrence count >=
    ``min_count``) in **alphabetical** order (chosen over frequency order for
    determinism independent of dataset statistics), and ``PAD`` is placed
    LAST. PAD-last is required: ``RSNN_TRSF_Reg`` / ``RSNN_LSTM_Reg`` hardcode
    ``nn.Embedding(n_emb, hid_dim, padding_idx=n_emb-1)`` (quickstart/
    train_qm9.py), so the PAD token must sit at index ``len(vocab)-1`` for its
    embedding to be zeroed/ignored -- this matches QM9's own vocab convention.
    The mapping must stay fixed once cached splits exist alongside it.
    """
    from collections import Counter

    counts: Counter = Counter()
    for decoded in list_of_decoded:
        counts.update(decoded['elements'])

    elements = sorted(e for e, c in counts.items() if c >= min_count)
    vocab = {'UNK': 0}
    for i, elem in enumerate(elements):
        vocab[elem] = i + 1
    vocab['PAD'] = len(vocab)  # last index == n_emb-1 (embedding padding_idx)
    return vocab


def save_vocab(vocab: Dict[str, int], path: str) -> None:
    with open(path, 'w') as f:
        json.dump(vocab, f, indent=2)


def load_vocab(path: str) -> Dict[str, int]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Radius-graph construction (torch_cluster if available, else cdist fallback)
# ---------------------------------------------------------------------------


def _radius_graph_cdist(pos: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Pure-torch radius graph: all pairs with d < cutoff, no self-loops."""
    n = pos.size(0)
    if n < 2:
        return torch.empty((2, 0), dtype=torch.long)
    dmat = torch.cdist(pos, pos)
    mask = dmat < cutoff
    mask.fill_diagonal_(False)
    row, col = mask.nonzero(as_tuple=True)
    return torch.stack([row, col], dim=0).long()


def _build_edges(pos: torch.Tensor, cutoff: float, eps: float,
                  max_num_neighbors: Optional[int] = None):
    """Undirected radius graph + ``1/d + eps`` edge weights, strict ``d < cutoff``.

    Candidate pairs come from ``torch_cluster.radius_graph`` when available
    (falling back to a ``cdist`` scan otherwise); either way we re-derive
    exact pairwise distances from ``pos`` and re-filter with a strict ``<``
    comparison, so the returned edge set does not depend on the underlying
    library's own boundary convention (``<`` vs ``<=`` at ``d == cutoff``).
    ``max_num_neighbors`` defaults to ``n`` (no truncation) so no true edge
    within ``cutoff`` is ever dropped, however dense the local neighborhood.
    """
    n = pos.size(0)
    if n < 2:
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float32))

    if _HAS_TORCH_CLUSTER:
        mnn = max_num_neighbors if max_num_neighbors is not None else n
        edge_index = _radius_graph(pos, r=cutoff, loop=False, max_num_neighbors=mnn)
    else:
        edge_index = _radius_graph_cdist(pos, cutoff)

    if edge_index.numel() == 0:
        return (torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float32))

    row, col = edge_index[0], edge_index[1]
    d = (pos[row] - pos[col]).norm(dim=-1)
    keep = d < cutoff
    row, col, d = row[keep], col[keep], d[keep]
    edge_index = torch.stack([row, col], dim=0)
    edge_attr = (1.0 / d + eps).unsqueeze(-1)
    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# decoded dict -> torch_geometric Data
# ---------------------------------------------------------------------------


def build_data(decoded: dict, vocab: Dict[str, int], cutoff: float = 4.5,
               eps: float = 1e-5, max_num_neighbors: Optional[int] = None) -> Data:
    """Build the atom-level RSNN ``Data`` object for one decoded complex.

    Field names ``x_emb``, ``pos``, ``edge_index`` are kept exactly as-is --
    ``utils/search.py`` consumes these on any ``Data`` passed to the sampler.
    """
    elements = decoded['elements']
    n = len(elements)
    unk = vocab['UNK']

    pos = torch.as_tensor(np.asarray(decoded['coords'], dtype=np.float32))
    x_emb = torch.tensor([vocab.get(e, unk) for e in elements], dtype=torch.long)
    z = torch.tensor([element_to_z(e) for e in elements], dtype=torch.long)

    vocab_size = len(vocab)
    x = torch.zeros((n, vocab_size), dtype=torch.float32)
    if n > 0:
        x[torch.arange(n), x_emb] = 1.0

    edge_index, edge_attr = _build_edges(pos, cutoff, eps, max_num_neighbors)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.x_emb = x_emb
    data.pos = pos
    data.z = z
    data.segment = torch.as_tensor(decoded['segment'], dtype=torch.long)
    data.residue_id = torch.as_tensor(decoded['residue_id'], dtype=torch.long)
    data.y = torch.tensor([decoded['label']], dtype=torch.float32)
    data.num_nodes = n
    data.pdb_id = decoded['pdb_id']
    return data


# ---------------------------------------------------------------------------
# Split-level (de)serialization
# ---------------------------------------------------------------------------


def preprocess_split(pkl_path: str, out_pt_path: str, vocab: Dict[str, int],
                      cutoff: float = 4.5, eps: float = 1e-5) -> None:
    """Decode+build every item in ``pkl_path``, cache ``list[Data]`` to ``out_pt_path``.

    Items that fail to decode or build are skipped with a logged reason;
    processing continues for the rest of the split.
    """
    with open(pkl_path, 'rb') as f:
        items = pickle.load(f)

    data_list = []
    n_failed = 0
    for item in items:
        try:
            decoded = decode_item(item)
            data = build_data(decoded, vocab, cutoff=cutoff, eps=eps)
        except Exception as e:  # noqa: BLE001 -- one bad complex must not abort the split
            n_failed += 1
            print(f"[pdbbind] skip {item.get('id', '?')}: {e}")
            continue
        data_list.append(data)

    torch.save(data_list, out_pt_path)
    print(f"[pdbbind] {out_pt_path}: kept {len(data_list)}/{len(items)} "
          f"complexes ({n_failed} skipped)")


def load_pdbbind_cache(pt_path: str) -> List[Data]:
    return torch.load(pt_path, weights_only=False)


# ---------------------------------------------------------------------------
# Ligand bond-graph extraction (ABLATION option; purely additive -- consumed
# only by ``utils.search.sample_protein_ligand``'s ``ligand_graph='bond'``
# path, which itself defaults to the existing distance-graph behavior).
#
# Raw ligand files live at
# ``GET/datasets/PDBBind/pdbbind/pdb_files/<pdb_id>/<pdb_id>_ligand.sdf``
# (``raw_dir`` == the ``pdb_files`` directory). RDKit's heavy-atom coordinates
# match our processed ligand atoms exactly (verified: max nearest-neighbor
# distance ~1e-6 A on a sample complex), which is what lets nearest-coordinate
# matching translate RDKit bond pairs into our global node indexing.
# ---------------------------------------------------------------------------


def extract_ligand_bond_edges(pdb_id: str, our_ligand_global_indices, our_ligand_coords,
                              raw_dir: str) -> torch.Tensor:
    """Ligand-ligand heavy-atom bond edges, in OUR global node indexing.

    Parameters
    ----------
    pdb_id : str
        Complex id (matches ``data.pdb_id``); raw files are looked up at
        ``<raw_dir>/<pdb_id>/<pdb_id>_ligand.{sdf,mol2}``.
    our_ligand_global_indices : sequence[int], length n_lig
        Our global node index for each ligand atom, in the SAME order as
        ``our_ligand_coords`` rows (e.g. ``torch.nonzero(data.segment == 1)``).
    our_ligand_coords : array-like (n_lig, 3)
        Our ligand atom coordinates, row-aligned with
        ``our_ligand_global_indices`` (e.g. ``data.pos[data.segment == 1]``).
    raw_dir : str
        Directory containing one subdirectory per pdb_id with the raw
        ligand sdf/mol2 file (GET's ``pdb_files`` layout).

    Returns
    -------
    torch.LongTensor (2, Eb)
        Undirected ligand-ligand heavy-atom bond edges (both ``(i, j)`` and
        ``(j, i)`` present), in our global indexing. ``(2, 0)`` only if the
        molecule truly has zero heavy-heavy bonds (e.g. a single atom).

    Raises
    ------
    ValueError
        If the raw file is missing, RDKit fails to load a molecule from it,
        or any RDKit heavy atom's nearest our-ligand-atom coordinate is
        >= 0.5 A away (indexing/atom-count mismatch) or the nearest-atom
        mapping is not injective (would silently merge two ligand atoms).
    """
    from rdkit import Chem  # local import: rdkit not required for the rest of this module

    our_ligand_global_indices = list(our_ligand_global_indices)
    n_lig = len(our_ligand_global_indices)
    our_coords = np.asarray(our_ligand_coords, dtype=np.float64)
    if our_coords.ndim != 2 or our_coords.shape[-1] != 3 or our_coords.shape[0] != n_lig:
        raise ValueError(
            f"{pdb_id}: our_ligand_coords shape {our_coords.shape} incompatible with "
            f"{n_lig} our_ligand_global_indices"
        )
    if n_lig == 0:
        return torch.empty((2, 0), dtype=torch.long)

    sdf_path = os.path.join(raw_dir, pdb_id, f'{pdb_id}_ligand.sdf')
    mol2_path = os.path.join(raw_dir, pdb_id, f'{pdb_id}_ligand.mol2')

    mol = None
    if os.path.exists(sdf_path):
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
        for candidate in supplier:
            if candidate is not None:
                mol = candidate
                break
    if mol is None and os.path.exists(mol2_path):
        mol = Chem.MolFromMol2File(mol2_path, removeHs=True, sanitize=False)
    if mol is None:
        raise ValueError(
            f"{pdb_id}: rdkit failed to load a ligand molecule from "
            f"{sdf_path!r} (fallback {mol2_path!r})"
        )

    # Heavy atoms only -- ``removeHs`` does not guarantee every explicit H is
    # stripped when sanitize=False, so filter explicitly by atomic number.
    heavy_rdkit_idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(heavy_rdkit_idx) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    conf = mol.GetConformer()
    rdkit_coords = np.array(
        [list(conf.GetAtomPosition(i)) for i in heavy_rdkit_idx], dtype=np.float64
    )

    # Nearest-coordinate match: each rdkit heavy atom -> our ligand atom slot.
    dmat = np.linalg.norm(
        rdkit_coords[:, None, :] - our_coords[None, :, :], axis=-1
    )  # (n_rdkit_heavy, n_lig)
    nearest_slot = dmat.argmin(axis=1)
    nearest_dist = dmat[np.arange(len(heavy_rdkit_idx)), nearest_slot]
    worst = int(nearest_dist.argmax())
    if nearest_dist[worst] >= 0.5:
        raise ValueError(
            f"{pdb_id}: nearest-atom coordinate distance "
            f"{nearest_dist[worst]:.3f} A >= 0.5 A for rdkit heavy atom "
            f"{heavy_rdkit_idx[worst]} -- coordinate/indexing mismatch"
        )
    if len(set(nearest_slot.tolist())) != len(heavy_rdkit_idx):
        raise ValueError(
            f"{pdb_id}: nearest-atom mapping is not injective (two rdkit "
            f"heavy atoms matched to the same our-ligand-atom slot)"
        )

    rdkit_to_global = {
        rd_idx: our_ligand_global_indices[nearest_slot[k]]
        for k, rd_idx in enumerate(heavy_rdkit_idx)
    }

    pairs = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in rdkit_to_global and a2 in rdkit_to_global:
            g1 = rdkit_to_global[a1]
            g2 = rdkit_to_global[a2]
            pairs.append((g1, g2))
            pairs.append((g2, g1))

    if len(pairs) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def augment_cache_with_bonds(pt_path: str, raw_dir: str, out_path: Optional[str] = None) -> dict:
    """Attach ``data.ligand_bond_edge_index`` to every ``Data`` in a cache.

    Loads ``pt_path`` (``list[Data]``), and for each complex extracts ligand
    bond edges via ``extract_ligand_bond_edges``. On ANY per-complex failure
    (missing raw file, rdkit returning None, coord-match failure, etc.) that
    complex's ``ligand_bond_edge_index`` is set to an empty ``(2, 0)`` tensor
    and it is counted as skipped -- augmentation never aborts the whole
    split. Saves the augmented ``list[Data]`` to ``out_path`` (default:
    overwrite ``pt_path``) and returns a ``{"kept", "skipped", "total"}``
    summary dict.
    """
    data_list = load_pdbbind_cache(pt_path)
    n_kept = 0
    n_skipped = 0
    for data in data_list:
        pdb_id = getattr(data, 'pdb_id', '?')
        try:
            lig_mask = data.segment == 1
            lig_global_idx = torch.nonzero(lig_mask, as_tuple=False).view(-1).tolist()
            lig_coords = data.pos[lig_mask].detach().cpu().numpy()
            edge_index = extract_ligand_bond_edges(pdb_id, lig_global_idx, lig_coords, raw_dir)
            n_kept += 1
        except Exception as e:  # noqa: BLE001 -- one bad complex must not abort the split
            print(f"[pdbbind] ligand-bond skip {pdb_id}: {e}")
            edge_index = torch.empty((2, 0), dtype=torch.long)
            n_skipped += 1
        data.ligand_bond_edge_index = edge_index

    save_path = out_path if out_path is not None else pt_path
    torch.save(data_list, save_path)
    print(f"[pdbbind] ligand-bond augmentation {pt_path} -> {save_path}: "
          f"kept {n_kept}, skipped {n_skipped} (total {len(data_list)})")
    return {"kept": n_kept, "skipped": n_skipped, "total": len(data_list)}


# ---------------------------------------------------------------------------
# PROTEIN TORSION ORDER (ABLATION; purely additive -- consumed only by
# ``utils.search.sample_protein_ligand``'s ``protein_order='torsion'`` path,
# which itself defaults to the existing flat ``(residue_id, atom_index)``
# protein-prefix behavior). See PROTEIN TORSION ORDER ablation confirmation
# notes for the empirical facts this design is grounded in (role recipe,
# fragment statistics, Greek-code coverage on the identity30 test split).
# ---------------------------------------------------------------------------

PROTEIN_BOND_CUTOFF = 1.8  # Angstrom; strict covalent C-N peptide-bond cutoff.

# Side-chain Greek levels considered for the chi-path, in increasing order
# (i.e. chi1, chi2, chi3, chi4-ish continuation). Codes outside this set
# (``XT``, ``P``, ``sm``, ``'``, and any duplicate/branch atom at a level
# already consumed) always fall through to the per-residue tail -- nothing
# is ever silently dropped, only reordered.
_CHI_LEVELS = ['B', 'G', 'D', 'E', 'Z', 'H']


def build_protein_torsion_order(global_idx, element, greek, residue_id, coords,
                                bond_cutoff: float = PROTEIN_BOND_CUTOFF):
    """Torsion-aware ordering of one complex's PROTEIN atoms (segment == 0).

    Produces consecutive-atom quadruplets that are real torsions instead of
    the flat ``(residue_id, atom_index)`` order's frequent non-bonded
    sibling-atom adjacencies (e.g. Tyr's stored ``CD,CD,CE,CE``): a
    backbone SPINE (``N, CA, C`` per residue, concatenated across a
    peptide-bonded run of residues) so consecutive spine quadruplets are
    real phi/psi/omega backbone dihedrals; a per-residue CHI PATH
    (``N, CA`` + the first atom at each increasing Greek level in
    ``{B, G, D, E, Z, H}``) so consecutive chi-path quadruplets are real
    chi1/chi2/chi3/chi4 side-chain dihedrals; and a per-residue TAIL
    (everything else in the residue -- ``O``, duplicate/branch side-chain
    atoms, rare codes like ``XT``) emitted as plain tokens with no
    dihedral-contiguity claim.

    Trade-off (see ablation confirmation notes, section on phi/psi vs chi
    contiguity): because ``O`` and the side chain always intervene between
    ``C(r)`` and ``N(r+1)`` in ANY single linear token order, a single
    scheme cannot make both phi/psi-contiguity and chi-contiguity hold at
    every position simultaneously (CA has two branch children -- the
    C-continuation and the CB side chain). This function resolves that by
    giving each of {spine, chi-path, tail} its own contiguous block and its
    own segment id, so phi/psi ARE captured (within a spine block) and chi
    dihedrals ARE captured (within a chi-path block) -- just never both at
    the same consecutive-quadruplet position. Segment ids are consumed by
    ``utils.search.sample_protein_ligand``'s ``protein_order='torsion'``
    gating to hard-reset angle/dihedral/edge-window features at every
    spine/chi-path/tail/ligand-walk boundary.

    Parameters
    ----------
    global_idx : Sequence[int], length n
        This complex's global row-index (into ``data.pos`` / ``data.x_emb``
        / ``data.edge_index``) for each PROTEIN atom, in ANY order (the
        function only uses ``residue_id``/``element``/``greek`` to
        determine structure, not input order) -- but conventionally the
        caller passes GET's original flat per-atom storage order (already
        residue-major, N/CA/C/O-then-sidechain-by-level within a residue),
        since that is what ``element``/``greek``/``residue_id`` are
        row-aligned against and it also gives a deterministic tiebreak
        ("first" atom at a role/level) when a residue has duplicates.
    element : Sequence[str], length n
        Element symbol per atom (row-aligned with ``global_idx``).
    greek : Sequence[str], length n
        Greek-level/role code per atom (``decode_item``'s
        ``atom_positions``), row-aligned with ``global_idx``.
    residue_id : Sequence[int], length n
        Residue id per atom (N->C sequence order), row-aligned with
        ``global_idx``.
    coords : (n, 3) array-like
        Atom coordinates, row-aligned with ``global_idx`` (used only for
        the C(i)-N(i+1) peptide-bond fragment-break test).
    bond_cutoff : float
        Strict ``< bond_cutoff`` peptide-bond distance test (Angstrom).

    Returns
    -------
    order : list[int]
        A COVERING sequence of ``global_idx`` (as plain ints): every input
        atom appears AT LEAST once (nothing is ever silently dropped), but
        NOT necessarily exactly once -- N/CA are deliberately duplicated
        across the spine and chi-path blocks of the same residue whenever
        both are emitted (chi1 = N-CA-CB-CG needs all four atoms
        explicitly; this mirrors GET's own raw per-residue storage already
        containing adjacent duplicate/branch atoms). Never raises on
        missing backbone atoms (N/CA/C/O): a residue missing a role atom
        simply contributes no spine/chi-path slot for that role (skip
        slot, forces a fragment break), and the one atom this can strand
        (a residue with CA+C but no N) is the sole scenario where an atom
        is not re-homed to the tail -- 0/14,446 observed on
        identity30/test (see edge-case notes).
    seg : list[int]
        Same length as ``order``; a fresh non-negative integer per
        (fragment | residue chi-path | residue tail) block, i.e. every
        entry of ``order`` carries the segment id of the contiguous block
        it belongs to. Consumers should treat these ids as opaque/local to
        this complex (not comparable across complexes) and reserve
        negative ids for non-protein (e.g. ligand-walk) segments.
    """
    n = len(global_idx)
    if not (len(element) == len(greek) == len(residue_id) == n and len(coords) == n):
        raise ValueError(
            "build_protein_torsion_order: global_idx/element/greek/residue_id/"
            f"coords must all have the same length, got "
            f"{n}/{len(element)}/{len(greek)}/{len(residue_id)}/{len(coords)}"
        )
    if n == 0:
        return [], []

    coords = np.asarray(coords, dtype=np.float64)

    # Group LOCAL positions (0..n-1, i.e. indices into the input arrays) by
    # residue_id, preserving the input's own per-residue atom order (GET's
    # storage order is N, CA, C, O, then side chain by increasing level;
    # this is what makes "first atom at role/level X" a correct, sane
    # tiebreak for duplicates).
    by_res: Dict[int, List[int]] = {}
    for i in range(n):
        by_res.setdefault(int(residue_id[i]), []).append(i)
    residue_ids_sorted = sorted(by_res.keys())

    N_idx: Dict[int, Optional[int]] = {}
    CA_idx: Dict[int, Optional[int]] = {}
    C_idx: Dict[int, Optional[int]] = {}
    spine_ok: Dict[int, bool] = {}
    chi_path_atoms: Dict[int, List[int]] = {}
    tail_atoms_map: Dict[int, List[int]] = {}

    for r in residue_ids_sorted:
        atoms = by_res[r]
        n_i = next((i for i in atoms if element[i] == 'N' and greek[i] == ''), None)
        ca_i = next((i for i in atoms if element[i] == 'C' and greek[i] == 'A'), None)
        if ca_i is not None:
            ca_pos = atoms.index(ca_i)
            c_i = next(
                (i for i in atoms[ca_pos + 1:] if element[i] == 'C' and greek[i] == ''),
                None,
            )
        else:
            c_i = None

        N_idx[r], CA_idx[r], C_idx[r] = n_i, ca_i, c_i
        spine_ok[r] = n_i is not None and ca_i is not None and c_i is not None
        chi_ok = n_i is not None and ca_i is not None  # chi-path needs only N, CA

        # chi_path DELIBERATELY re-includes N/CA even when they are ALSO
        # emitted by the spine (see docstring): chi1 = N-CA-CB-CG needs all
        # four atoms, so N/CA are intentionally duplicated across the spine
        # and chi-path blocks -- this is a token-sequence design (like
        # GET's own raw per-residue storage already having adjacent
        # duplicate/branch atoms), NOT a bug. Only the atom's OWN
        # completeness (every input atom appears at least once) is
        # guaranteed, not exactly-once uniqueness.
        chi_used = set()
        chi_atoms: List[int] = []
        if chi_ok:
            chi_atoms.extend([n_i, ca_i])
            chi_used.update([n_i, ca_i])
            for level in _CHI_LEVELS:
                cand = next(
                    (i for i in atoms if i not in chi_used and greek[i] == level), None
                )
                if cand is not None:
                    chi_atoms.append(cand)
                    chi_used.add(cand)

        # Tail = O (if present) + every side-chain atom not already claimed
        # by the chi-path (duplicate/branch atoms at an already-visited
        # level, and rare codes like XT/P/'sm'/"'"). Backbone N/CA/C are
        # NEVER placed in the tail: they belong to the spine and/or
        # chi-path only, so a residue whose spine role is incomplete
        # (missing N or C -- 0/14,446 observed on identity30/test) may
        # lose that one backbone atom's slot entirely ("skip slot" per the
        # ablation's documented graceful-degradation policy) rather than
        # spuriously duplicating it into the tail.
        backbone_set = {x for x in (n_i, ca_i, c_i) if x is not None}
        exclude = backbone_set | chi_used
        tail_atoms = [i for i in atoms if i not in exclude]

        chi_path_atoms[r] = chi_atoms
        tail_atoms_map[r] = tail_atoms

    # --- Fragments: consecutive spine_ok residues bonded via a strict
    # C(prev)-N(r) < bond_cutoff test. A residue with an incomplete spine
    # role (missing N/CA/C) contributes no spine atoms and forces a break
    # on both sides (there is no atom pair left to bond-test across it).
    fragments: List[List[int]] = []
    current: List[int] = []
    for r in residue_ids_sorted:
        if not spine_ok[r]:
            if current:
                fragments.append(current)
                current = []
            continue
        if current:
            prev_r = current[-1]
            d = float(np.linalg.norm(coords[C_idx[prev_r]] - coords[N_idx[r]]))
            if d >= bond_cutoff:
                fragments.append(current)
                current = [r]
            else:
                current.append(r)
        else:
            current = [r]
    if current:
        fragments.append(current)

    order: List[int] = []
    seg: List[int] = []
    seg_counter = 0

    # 1) All fragment spines first, one segment id per fragment.
    for frag in fragments:
        for r in frag:
            for i in (N_idx[r], CA_idx[r], C_idx[r]):
                order.append(int(global_idx[i]))
                seg.append(seg_counter)
        seg_counter += 1

    # 2) Per residue (residue_id order): chi-path, then tail -- each its
    # own segment id, emitted only if non-empty.
    for r in residue_ids_sorted:
        if chi_path_atoms[r]:
            for i in chi_path_atoms[r]:
                order.append(int(global_idx[i]))
                seg.append(seg_counter)
            seg_counter += 1

        tail_atoms = tail_atoms_map[r]
        if tail_atoms:
            for i in tail_atoms:
                order.append(int(global_idx[i]))
                seg.append(seg_counter)
            seg_counter += 1

    return order, seg


def augment_cache_with_torsion(pkl_path: str, pt_path: str,
                               out_path: Optional[str] = None,
                               bond_cutoff: float = PROTEIN_BOND_CUTOFF) -> dict:
    """Attach ``data.torsion_order`` + ``data.torsion_seg`` to every ``Data``
    in a cache (PROTEIN TORSION ORDER ablation; purely additive, consumed
    only by ``utils.search.sample_protein_ligand``'s ``protein_order='torsion'``
    path).

    Loads ``pt_path`` (``list[Data]``) and the matching raw GET pickle
    ``pkl_path``, decodes each raw item (to recover the Greek-level codes
    that ``build_data`` currently drops), and calls
    ``build_protein_torsion_order`` over that complex's protein atoms
    (``data.segment == 0``). Complexes are matched primarily by
    ``data.pdb_id == item['id']`` (falling back to positional/index
    alignment, with a printed warning, only if a cache item lacks
    ``pdb_id`` or its id isn't found in the pickle -- identity30 caches are
    known 0-skip index-aligned with their pickle, so this fallback should
    never actually trigger there).

    On ANY per-complex failure (id not found, atom-count mismatch,
    ``build_protein_torsion_order`` silently dropping a protein atom, etc.)
    that complex's ``torsion_order``/``torsion_seg``
    are set to empty ``(0,)`` LongTensors and it is counted as skipped --
    augmentation never aborts the whole split (matches
    ``augment_cache_with_bonds``'s convention). Saves the augmented
    ``list[Data]`` to ``out_path`` (default: overwrite ``pt_path``) and
    returns a ``{"kept", "skipped", "total"}`` summary dict.
    """
    data_list = load_pdbbind_cache(pt_path)
    with open(pkl_path, 'rb') as f:
        pkl_items = pickle.load(f)

    by_id: Dict[str, dict] = {}
    for item in pkl_items:
        by_id.setdefault(str(item.get('id')), item)

    n_kept = 0
    n_skipped = 0
    for idx, data in enumerate(data_list):
        pdb_id = getattr(data, 'pdb_id', None)
        try:
            item = by_id.get(str(pdb_id)) if pdb_id is not None else None
            if item is None:
                if idx >= len(pkl_items):
                    raise ValueError(
                        f"no pkl item for cache index {idx} (pdb_id={pdb_id!r}); "
                        f"positional fallback out of range against {len(pkl_items)} "
                        f"pkl items"
                    )
                item = pkl_items[idx]
                print(f"[pdbbind] torsion: pdb_id {pdb_id!r} not found by id in "
                      f"{pkl_path}, falling back to positional index {idx} "
                      f"(pkl id={item.get('id')!r})")

            decoded = decode_item(item)
            if len(decoded['elements']) != data.num_nodes:
                raise ValueError(
                    f"atom-count mismatch: decoded={len(decoded['elements'])} "
                    f"cache={data.num_nodes}"
                )

            prot_mask = data.segment == 0
            prot_local = torch.nonzero(prot_mask, as_tuple=False).view(-1).tolist()
            g_idx = prot_local  # global_index == row index into data.pos/.z
            elems = [z_to_element(int(data.z[i])) for i in prot_local]
            greeks = [decoded['atom_positions'][i] for i in prot_local]
            res_ids = [int(data.residue_id[i]) for i in prot_local]
            coords = data.pos[prot_mask].detach().cpu().numpy()

            order, seg = build_protein_torsion_order(
                g_idx, elems, greeks, res_ids, coords, bond_cutoff=bond_cutoff
            )
            if len(order) != len(seg):
                raise ValueError(
                    f"build_protein_torsion_order returned mismatched "
                    f"order/seg lengths ({len(order)} vs {len(seg)})"
                )
            missing = set(g_idx) - set(order)
            if missing:
                raise ValueError(
                    f"build_protein_torsion_order dropped {len(missing)} protein "
                    f"atom(s) entirely (not a covering sequence): {sorted(missing)[:10]}"
                    + (" ..." if len(missing) > 10 else "")
                )
            data.torsion_order = torch.tensor(order, dtype=torch.long)
            data.torsion_seg = torch.tensor(seg, dtype=torch.long)
            n_kept += 1
        except Exception as e:  # noqa: BLE001 -- one bad complex must not abort the split
            print(f"[pdbbind] torsion skip {pdb_id}: {e}")
            data.torsion_order = torch.empty((0,), dtype=torch.long)
            data.torsion_seg = torch.empty((0,), dtype=torch.long)
            n_skipped += 1

    save_path = out_path if out_path is not None else pt_path
    torch.save(data_list, save_path)
    print(f"[pdbbind] torsion augmentation {pt_path} -> {save_path}: "
          f"kept {n_kept}, skipped {n_skipped} (total {len(data_list)})")
    return {"kept": n_kept, "skipped": n_skipped, "total": len(data_list)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main():
    parser = argparse.ArgumentParser(
        description='Preprocess GET-processed PDBbind pickles into PyG Data caches, '
                    'or (--mode augment_bonds) attach ligand_bond_edge_index to '
                    'existing caches, or (--mode augment_torsion) attach '
                    'torsion_order/torsion_seg to existing caches.'
    )
    parser.add_argument('--mode', choices=['preprocess', 'augment_bonds', 'augment_torsion'],
                        default='preprocess',
                        help="'preprocess' (default, original behavior): pkl -> "
                             "vocab.json + *.pt caches. 'augment_bonds': attach "
                             "data.ligand_bond_edge_index to existing *.pt caches. "
                             "'augment_torsion': attach data.torsion_order/"
                             "torsion_seg to existing *.pt caches (PROTEIN TORSION "
                             "ORDER ablation).")
    # --- preprocess-mode args (unchanged defaults/semantics) ---
    parser.add_argument('--pkl_dir',
                        help='[preprocess, augment_torsion] Directory containing '
                             'train.pkl / valid.pkl / test.pkl')
    parser.add_argument('--out_dir',
                        help='[preprocess] Output directory for vocab.json and *.pt caches')
    parser.add_argument('--cutoff', type=float, default=4.5)
    parser.add_argument('--eps', type=float, default=1e-5)
    parser.add_argument('--min_count', type=int, default=1)
    # --- augment_bonds-mode args ---
    parser.add_argument('--pkl',
                        help='[augment_bonds] Single cache .pt path to augment.')
    parser.add_argument('--cache_dir',
                        help='[augment_bonds, augment_torsion] Directory with '
                             'train.pt/valid.pt/test.pt to augment (use with --splits).')
    parser.add_argument('--raw_dir',
                        help="[augment_bonds] Directory with <pdb_id>/<pdb_id>_ligand.sdf "
                             "raw files (GET's pdb_files layout), e.g. "
                             "GET/datasets/PDBBind/pdbbind/pdb_files")
    parser.add_argument('--splits', nargs='+', default=['train', 'valid', 'test'],
                        help='[augment_bonds, augment_torsion] Splits to process '
                             'under --cache_dir.')
    parser.add_argument('--aug_out_dir',
                        help='[augment_bonds, augment_torsion] Output directory for '
                             'augmented caches (default: overwrite in place).')
    parser.add_argument('--bond_cutoff', type=float, default=PROTEIN_BOND_CUTOFF,
                        help='[augment_torsion] Strict peptide-bond distance cutoff '
                             '(Angstrom) for the backbone fragment-break test.')
    args = parser.parse_args()

    if args.mode == 'augment_torsion':
        if not args.pkl_dir or not args.cache_dir:
            parser.error('--mode augment_torsion requires --pkl_dir and --cache_dir')
        if args.aug_out_dir:
            os.makedirs(args.aug_out_dir, exist_ok=True)
        for split in args.splits:
            pt_path = os.path.join(args.cache_dir, f'{split}.pt')
            pkl_path = os.path.join(args.pkl_dir, f'{split}.pkl')
            if not os.path.exists(pt_path):
                print(f"[pdbbind] {pt_path} not found, skipping")
                continue
            if not os.path.exists(pkl_path):
                print(f"[pdbbind] {pkl_path} not found, skipping")
                continue
            out_path = (os.path.join(args.aug_out_dir, f'{split}.pt')
                       if args.aug_out_dir else None)
            augment_cache_with_torsion(pkl_path, pt_path, out_path=out_path,
                                       bond_cutoff=args.bond_cutoff)
        return

    if args.mode == 'augment_bonds':
        if not args.raw_dir:
            parser.error('--raw_dir is required for --mode augment_bonds')
        if args.aug_out_dir:
            os.makedirs(args.aug_out_dir, exist_ok=True)
        if args.pkl:
            out_path = (os.path.join(args.aug_out_dir, os.path.basename(args.pkl))
                       if args.aug_out_dir else None)
            augment_cache_with_bonds(args.pkl, args.raw_dir, out_path=out_path)
        elif args.cache_dir:
            for split in args.splits:
                pt_path = os.path.join(args.cache_dir, f'{split}.pt')
                if not os.path.exists(pt_path):
                    print(f"[pdbbind] {pt_path} not found, skipping")
                    continue
                out_path = (os.path.join(args.aug_out_dir, f'{split}.pt')
                           if args.aug_out_dir else None)
                augment_cache_with_bonds(pt_path, args.raw_dir, out_path=out_path)
        else:
            parser.error('--mode augment_bonds requires --pkl or --cache_dir')
        return

    if not args.pkl_dir or not args.out_dir:
        parser.error('--pkl_dir and --out_dir are required for --mode preprocess')

    os.makedirs(args.out_dir, exist_ok=True)

    train_pkl = os.path.join(args.pkl_dir, 'train.pkl')
    with open(train_pkl, 'rb') as f:
        train_items = pickle.load(f)

    train_decoded = []
    for item in train_items:
        try:
            train_decoded.append(decode_item(item))
        except Exception as e:  # noqa: BLE001
            print(f"[pdbbind] skip {item.get('id', '?')} while building vocab: {e}")

    vocab = build_pdbbind_vocab(train_decoded, min_count=args.min_count)
    vocab_path = os.path.join(args.out_dir, 'vocab.json')
    save_vocab(vocab, vocab_path)
    print(f"[pdbbind] vocab ({len(vocab)} tokens) -> {vocab_path}")

    for split in ('train', 'valid', 'test'):
        pkl_path = os.path.join(args.pkl_dir, f'{split}.pkl')
        if not os.path.exists(pkl_path):
            print(f"[pdbbind] {pkl_path} not found, skipping")
            continue
        out_pt = os.path.join(args.out_dir, f'{split}.pt')
        preprocess_split(pkl_path, out_pt, vocab, cutoff=args.cutoff, eps=args.eps)

        data_list = load_pdbbind_cache(out_pt)
        n_atoms = [d.num_nodes for d in data_list]
        if n_atoms:
            print(f"[pdbbind] {split}: {len(data_list)} complexes, "
                  f"atoms/complex min={min(n_atoms)} max={max(n_atoms)} "
                  f"mean={sum(n_atoms) / len(n_atoms):.1f}")


if __name__ == '__main__':
    _main()


__all__ = [
    'ATOMS',
    'IDX2ATOM',
    'IDX2POS',
    'element_to_z',
    'z_to_element',
    'decode_item',
    'build_pdbbind_vocab',
    'save_vocab',
    'load_vocab',
    'build_data',
    'preprocess_split',
    'load_pdbbind_cache',
    'extract_ligand_bond_edges',
    'augment_cache_with_bonds',
    'PROTEIN_BOND_CUTOFF',
    'build_protein_torsion_order',
    'augment_cache_with_torsion',
]
