import torch
import random
from collections import deque


# ---------------------------------------------------------------------------
# Quadruplet geometric features (bond angle, dihedral) for sample_dfs.
# ---------------------------------------------------------------------------

def _bond_angle(p_prev2, p_prev1, p_curr):
    """Bond angle θ at p_prev1 between (p_prev2, p_prev1, p_curr).

    Returns θ ∈ [0, π] as a scalar tensor. Clamps cosine to [-1, 1] to
    suppress floating-point noise without biasing the answer at endpoints
    (torch.acos handles exact ±1 correctly).
    """
    a = p_prev2 - p_prev1
    b = p_curr - p_prev1
    a_norm = a.norm()
    b_norm = b.norm()
    eps = 1e-8
    cos_t = (a * b).sum() / (a_norm * b_norm + eps)
    cos_t = cos_t.clamp(-1.0, 1.0)
    return torch.acos(cos_t)


def _dihedral(p0, p1, p2, p3):
    """Dihedral φ ∈ [-π, π] for the four points (p0, p1, p2, p3).

    Uses the standard atan2-based formula to avoid arccos sign loss:
        b1 = p1 - p0;  b2 = p2 - p1;  b3 = p3 - p2
        n1 = b1 × b2;  n2 = b2 × b3
        m1 = n1 × (b2 / ‖b2‖)
        x = n1 · n2;  y = m1 · n2
        φ = atan2(y, x)
    Degenerate (collinear three-of-four) returns 0.
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2_norm = b2.norm()
    if b2_norm < 1e-8:
        return torch.zeros((), dtype=p0.dtype, device=p0.device)
    b2_hat = b2 / b2_norm
    m1 = torch.cross(n1, b2_hat, dim=-1)
    x = (n1 * n2).sum()
    y = (m1 * n2).sum()
    return torch.atan2(y, x)


def _angle_basis(theta, K):
    """Bond-angle Fourier basis: cos(l θ) for l = 1..K, returns (K,) tensor."""
    ls = torch.arange(1, K + 1, dtype=theta.dtype, device=theta.device)
    return torch.cos(ls * theta)


def _dihedral_basis(phi, K):
    """Dihedral Fourier basis: (sin(l φ), cos(l φ)) for l = 1..K, returns (2K,)."""
    ls = torch.arange(1, K + 1, dtype=phi.dtype, device=phi.device)
    return torch.cat([torch.sin(ls * phi), torch.cos(ls * phi)])


# ---------------------------------------------------------------------------
# Batched (vectorized) versions of the above. Numerically equivalent to the
# scalar helpers element-by-element; eliminates Python per-step call overhead
# inside sample_dfs when many quadruplets need to be computed.
# ---------------------------------------------------------------------------


def _batch_bond_angle(p_prev2, p_prev1, p_curr):
    """Vectorized _bond_angle. Inputs are (M, 3) tensors; returns (M,)."""
    a = p_prev2 - p_prev1
    b = p_curr - p_prev1
    a_norm = a.norm(dim=-1)
    b_norm = b.norm(dim=-1)
    eps = 1e-8
    cos_t = (a * b).sum(dim=-1) / (a_norm * b_norm + eps)
    cos_t = cos_t.clamp(-1.0, 1.0)
    return torch.acos(cos_t)


def _batch_dihedral(p0, p1, p2, p3):
    """Vectorized _dihedral. Inputs are (M, 3) tensors; returns (M,)."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = torch.cross(b1, b2, dim=-1)
    n2 = torch.cross(b2, b3, dim=-1)
    b2_norm = b2.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b2_hat = b2 / b2_norm
    m1 = torch.cross(n1, b2_hat, dim=-1)
    x = (n1 * n2).sum(dim=-1)
    y = (m1 * n2).sum(dim=-1)
    return torch.atan2(y, x)


def _batch_angle_basis(theta, K):
    """Vectorized _angle_basis. theta is (M,); returns (M, K)."""
    ls = torch.arange(1, K + 1, dtype=theta.dtype, device=theta.device)
    return torch.cos(theta.unsqueeze(-1) * ls)


def _batch_dihedral_basis(phi, K):
    """Vectorized _dihedral_basis. phi is (M,); returns (M, 2K)."""
    ls = torch.arange(1, K + 1, dtype=phi.dtype, device=phi.device)
    lp = phi.unsqueeze(-1) * ls
    return torch.cat([torch.sin(lp), torch.cos(lp)], dim=-1)

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
    return neighbor_dict
    
def sample_bfs(data, nw, s, max_len, vocab, add_edge_feat=None):
    """
    Performs optimized BFS-based searches on the graph and computes the edge encoding on the fly.
    
    For each BFS search:
      - A random starting node is chosen.
      - A BFS is performed (without revisiting nodes) to obtain an ordering of nodes.
      - As each node is visited, its embedding id (obtained via vocab[data.x_emb[node]])
        is recorded in the search order, and the raw node id is also recorded.
      - For each new node (at position j in the BFS order), for each offset d in [1, min(s, j)],
        if there is an edge between the node at position j and the node at position j-d (in the
        original graph), the corresponding entry in the edge encoding is set to 1.
      - The BFS order is stored in a tensor of shape (nw, max_len), where unfilled positions
        remain padded with vocab['PAD'].
      - The actual length (number of nodes visited) is recorded.
    
    Parameters:
        data (torch_geometric.data.Data): Graph data object with 'edge_index' and 'x_emb' attributes.
        nw (int): Number of BFS searches to perform.
        max_len (int): Maximum length of each BFS search order; shorter searches are padded with vocab['PAD'].
        s (int): Window size for the edge encoding.
        vocab (dict): Mapping from tokens to embedding ids; must include an entry for 'PAD'.
    
    Returns:
        data (torch_geometric.data.Data): The input data object with four new attributes:
            - walk_emb: A tensor of shape (nw, max_len) with the BFS search order in embedding ids.
            - walk_ids: A tensor of shape (1, nw, max_len) with the raw node ids in BFS order.
            - walk_pe: A tensor of shape (nw, max_len, s) with the computed edge encoding.
            - lengths: A tensor of shape (nw,) where each entry is the number of nodes visited in the BFS search.
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use precomputed neighbor dictionary (or compute and store it if not present).
    neighbor_dict = get_neighbor_dict(data)

    # Pre-allocate tensors for BFS searches.
    searches_emb = torch.full((nw, max_len), vocab['PAD'], dtype=torch.long)
    searches = torch.full((nw, max_len), -1, dtype=torch.long)
    encoding_edge = torch.zeros((nw, max_len, s), dtype=torch.float)
    lengths = []

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, max_len, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # For each BFS search:
    for i in range(nw):
        start_node = random.randint(0, num_nodes - 1)
        visited = set()
        queue = deque([start_node])
        order = []  # Raw node indices in BFS order.
        pos = 0     # Current position in the BFS order.
        
        while queue and pos < max_len:
            node = queue.popleft()  # FIFO for BFS.
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Record the embedding id and raw node id.
            searches_emb[i, pos] = data.x_emb[node]
            searches[i, pos] = node
            
            # Compute edge encoding on the fly:
            for d in range(1, min(s, pos) + 1):
                prev_node = order[pos - d]
                # Using sets for fast membership tests.
                if (node in neighbor_dict.get(prev_node, set())) or (prev_node in neighbor_dict.get(node, set())):
                    encoding_edge[i, pos, d - 1] = 1


            # Per-step edge feature (BFS): incoming edge is from prev node
            # in BFS order; if not adjacent in graph, leave zeros.
            if walk_pe_extra is not None and pos > 0:
                prev_in_order = order[pos - 1]
                if (node in neighbor_dict.get(prev_in_order, set())) or (prev_in_order in neighbor_dict.get(node, set())):
                    walk_pe_extra[i, pos] = add_edge_feat[prev_in_order, node]
            pos += 1

            # Append unvisited neighbors to the queue (randomized order for variability).
            neighbors = list(neighbor_dict.get(node, set()))
            random.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited:
                    queue.append(nb)
                    
        lengths.append(pos)

    data.walk_emb = searches_emb
    data.walk_ids = searches[None, :, :]
    if walk_pe_extra is not None:
        data.walk_pe = torch.cat([encoding_edge, walk_pe_extra], dim=-1)
    else:
        data.walk_pe = encoding_edge
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    
    return data

def sample_dfs(data, nw, s, max_len, vocab, add_edge_feat=None,
               max_search_len=None, angles=False, dihedrals=False,
               angle_K=8, dihedral_K=4, vectorize=False):
    """
    Performs DFS-based searches on the graph and computes the edge encoding on the fly.

    New optional features (default off, fully backward-compatible):
      max_search_len : int or None
        If set, terminates each DFS at this position instead of max_len.
        Effective cap is min(max_len, max_search_len).
      angles : bool
        If True, append bond-angle Fourier features (cos(l θ), l=1..angle_K)
        at each step pos >= 2. Steps with pos < 2 are zero-filled.
      dihedrals : bool
        If True, append dihedral Fourier features (sin/cos(l φ), l=1..dihedral_K)
        at each step pos >= 3. Steps with pos < 3 are zero-filled.
      angle_K, dihedral_K : int
        Basis sizes (default 8 / 4 per Gasteiger et al. DimeNet defaults).

    Requires data.pos (N, 3) for angle/dihedral features. Without those flags,
    behaviour is byte-for-byte identical to the original function.

    For each DFS search:
      - A random starting node is chosen.
      - A DFS is performed (without revisiting nodes) to obtain an ordering of nodes.
      - As each node is visited, its embedding id (obtained via vocab[data.x_emb[node]])
        is recorded in the search order.
      - For each new node (at position j in the DFS order), for each offset d in [1, min(s, j)],
        if there is an edge between the node at position j and the node at position j-d (in the
        original graph), the corresponding entry in the edge encoding is set to 1.
      - The DFS order is stored in a tensor of shape (nw, max_len), where unfilled positions
        remain padded with vocab['PAD'].
      - The actual length (number of nodes visited) is recorded.

    Parameters:
        data (torch_geometric.data.Data): Graph data object with 'edge_index' and 'x_emb' attributes.
        nw (int): Number of DFS searches to perform.
        max_len (int): Maximum length of each DFS search order; shorter searches are padded with vocab['PAD'].
        s (int): Window size for the edge encoding.
        vocab (dict): Mapping from tokens to embedding ids; must include an entry for 'PAD'.

    Returns:
        data (torch_geometric.data.Data): The input data object with three new attributes:
            - walk_emb: A tensor of shape (nw, max_len) with the DFS search order in embedding ids.
            - walk_pe: A tensor of shape (nw, max_len, s) with the computed edge encoding.
            - lengths: A tensor of shape (nw,) where each entry is the number of nodes visited in the DFS search.
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use the precomputed neighbor dictionary (or compute & store it if not present).
    neighbor_dict = get_neighbor_dict(data)  # get_neighbor_dict returns a dict with sets.

    # Pre-allocate tensors for DFS searches.
    searches_emb = torch.full((nw, max_len), vocab['PAD'], dtype=torch.long)
    searches = torch.full((nw, max_len), -1, dtype=torch.long)
    encoding_edge = torch.zeros((nw, max_len, s), dtype=torch.float)
    lengths = []

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, max_len, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # Quadruplet geometric features (per Gasteiger et al. DimeNet). Allocated
    # only when requested; require data.pos.
    if angles or dihedrals:
        if not hasattr(data, "pos") or data.pos is None:
            raise ValueError("angles/dihedrals=True requires data.pos (N, 3) coordinates.")
        pos_xyz = data.pos
    walk_pe_angle = (
        torch.zeros((nw, max_len, angle_K), dtype=torch.float)
        if angles else None
    )
    walk_pe_dihedral = (
        torch.zeros((nw, max_len, 2 * dihedral_K), dtype=torch.float)
        if dihedrals else None
    )

    # Effective per-walk length cap (search-length, separate from max_len padding).
    cap = max_len if max_search_len is None else min(int(max_search_len), max_len)

    # When vectorize=True, collect quadruplet INDICES per step and batch-compute
    # all the angle/dihedral features in a single tensor op after the loop.
    # Eliminates Python-level per-step call overhead. Numerically equivalent to
    # the scalar path (same eps, same clamp, same atan2-based formula).
    angle_idx = [] if (vectorize and walk_pe_angle is not None) else None
    dihedral_idx = [] if (vectorize and walk_pe_dihedral is not None) else None

    # For each DFS search:
    for i in range(nw):
        start_node = random.randint(0, num_nodes - 1)
        visited = set()
        stack = [start_node]
        order = []  # To store the raw node indices in the DFS order.
        pos = 0     # Current position in the DFS order.

        while stack and pos < cap:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Record the embedding id (from data.x_emb via vocab) and the raw node id.
            searches_emb[i, pos] = data.x_emb[node]
            searches[i, pos] = node
            
            # Compute edge encoding on the fly for this node:
            for d in range(1, min(s, pos) + 1):
                prev_node = order[pos - d]
                # Check if there's an edge between the current node and the previous node (both directions).
                if (node in neighbor_dict[prev_node]) or (prev_node in neighbor_dict[node]):
                    encoding_edge[i, pos, d - 1] = 1


            # Per-step edge feature (DFS): always populate from the (prev,
            # node) entry of add_edge_feat regardless of graph-adjacency.
            # When prev and node are not bonded, edge_attr_to_dense yields
            # zeros for the bond-type slice, so the bond-type semantics
            # remain "zero unless bonded" — but the distance slice (RBF of
            # the pairwise Euclidean distance) is always populated. This
            # ensures DFS stack-jumps still carry the geometric distance
            # signal to the LSTM.
            if walk_pe_extra is not None and pos > 0:
                prev_in_order = order[pos - 1]
                walk_pe_extra[i, pos] = add_edge_feat[prev_in_order, node]

            # Quadruplet geometric features. DFS guarantees four distinct
            # nodes at pos >= 3 (no revisits), so bond angle / dihedral are
            # well-defined by construction (modulo collinearity, handled
            # inside _bond_angle / _dihedral with eps safeguards).
            # Two paths: scalar (default, one tensor op per step) or vectorize
            # (collect indices, batched compute after the loop).
            if walk_pe_angle is not None and pos >= 2:
                v0 = order[pos - 2]
                v1 = order[pos - 1]
                if vectorize:
                    angle_idx.append((i, pos, v0, v1, node))
                else:
                    theta = _bond_angle(pos_xyz[v0], pos_xyz[v1], pos_xyz[node])
                    walk_pe_angle[i, pos] = _angle_basis(theta, angle_K)
            if walk_pe_dihedral is not None and pos >= 3:
                u0 = order[pos - 3]
                u1 = order[pos - 2]
                u2 = order[pos - 1]
                if vectorize:
                    dihedral_idx.append((i, pos, u0, u1, u2, node))
                else:
                    phi = _dihedral(pos_xyz[u0], pos_xyz[u1], pos_xyz[u2], pos_xyz[node])
                    walk_pe_dihedral[i, pos] = _dihedral_basis(phi, dihedral_K)
            pos += 1

            # Push unvisited neighbors onto the stack in randomized order.
            neighbors = list(neighbor_dict[node])
            random.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited:
                    stack.append(nb)
                    
        lengths.append(pos)

    # Vectorized quadruplet compute (only when vectorize=True and there is at
    # least one valid step to score). Index arrays were accumulated above.
    if angle_idx is not None and len(angle_idx) > 0:
        idx_a = torch.tensor(angle_idx, dtype=torch.long)  # (M, 5)
        p_prev2 = pos_xyz[idx_a[:, 2]]
        p_prev1 = pos_xyz[idx_a[:, 3]]
        p_curr = pos_xyz[idx_a[:, 4]]
        thetas = _batch_bond_angle(p_prev2, p_prev1, p_curr)  # (M,)
        basis_a = _batch_angle_basis(thetas, angle_K)         # (M, angle_K)
        walk_pe_angle[idx_a[:, 0], idx_a[:, 1]] = basis_a.to(walk_pe_angle.dtype)
    if dihedral_idx is not None and len(dihedral_idx) > 0:
        idx_d = torch.tensor(dihedral_idx, dtype=torch.long)  # (M, 6)
        p0 = pos_xyz[idx_d[:, 2]]
        p1 = pos_xyz[idx_d[:, 3]]
        p2 = pos_xyz[idx_d[:, 4]]
        p3 = pos_xyz[idx_d[:, 5]]
        phis = _batch_dihedral(p0, p1, p2, p3)                # (M,)
        basis_d = _batch_dihedral_basis(phis, dihedral_K)     # (M, 2*dihedral_K)
        walk_pe_dihedral[idx_d[:, 0], idx_d[:, 1]] = basis_d.to(walk_pe_dihedral.dtype)

    data.walk_emb = searches_emb
    data.walk_ids = searches[None, :, :]
    parts = [encoding_edge]
    if walk_pe_extra is not None:
        parts.append(walk_pe_extra)
    if walk_pe_angle is not None:
        parts.append(walk_pe_angle)
    if walk_pe_dihedral is not None:
        parts.append(walk_pe_dihedral)
    data.walk_pe = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    return data

def sample_walks(data, nw, l, s, non_backtracking, add_edge_feat=None):
    """
    Samples random walks from a graph and computes two encodings on the fly:
    
    1. Repeat Encoding: For each position j in a walk, marks if the node at j equals 
       any of the nodes at j-d for 1 ≤ d ≤ min(s, j).
    2. Edge Encoding: For each position j in a walk, marks if the node at j is connected 
       (in the original graph) to any of the nodes at j-d for 1 ≤ d ≤ min(s, j).

    Additionally, if non_backtracking is True, the random walk will avoid returning 
    immediately to the previous node if other neighbors exist.

    Parameters:
        data (torch_geometric.data.Data): Graph data object (must have an 'edge_index' attribute).
        nw (int): Number of random walks.
        l (int): Length of each walk (number of nodes in the walk).
        s (int): Window size for the encodings.
        non_backtracking (bool): If True, prevents immediate backtracking to the previous node 
                                 unless it is the only option.
    
    Returns:
        data (torch_geometric.data.Data): The input data object with three new attributes:
            - walk_ids: A tensor of shape (nw, l) with the node indices for each random walk.
            - walk_emb: A tensor of shape (nw, l) with the node embeddings for each random walk.
            - walk_pe: A tensor of shape (nw, l, 2*s) for the repeat and edge connectivity encoding.
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use the precomputed neighbor dictionary (or compute & store it if not present).
    neighbor_dict = get_neighbor_dict(data)  # neighbor_dict: {node: set(neighbors)}

    # Pre-allocate tensors for the random walks and the encodings.
    walk_ids = torch.empty((nw, l), dtype=torch.long)
    walk_emb = torch.empty((nw, l), dtype=torch.long)
    encoding_repeat = torch.zeros((nw, l, s), dtype=torch.float)
    encoding_edge = torch.zeros((nw, l, s), dtype=torch.float)

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, l, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # Randomly select starting nodes for each walk.
    start_nodes = torch.randint(0, num_nodes, (nw,), dtype=torch.long)
    
    for i in range(nw):
        current_node = start_nodes[i].item()
        walk_ids[i, 0] = current_node
        walk_emb[i, 0] = data.x_emb[current_node]
        # For the first node there is no previous context; encodings remain zeros.
        for j in range(1, l):
            # Get neighbors from the set (if any)
            neighbors = neighbor_dict.get(current_node, set())
            if neighbors:
                if non_backtracking and j > 0:
                    prev_node = walk_ids[i, j - 1].item()
                    # Filter out the immediate previous node.
                    filtered_neighbors = [n for n in neighbors if n != prev_node]
                    if filtered_neighbors:
                        next_node = random.choice(filtered_neighbors)
                    else:
                        next_node = random.choice(list(neighbors))
                else:
                    next_node = random.choice(list(neighbors))
            else:
                next_node = current_node
            walk_ids[i, j] = next_node
            walk_emb[i, j] = data.x_emb[next_node]
            
            # Update encodings on the fly for offsets within the window (up to s)
            for d in range(1, min(s, j) + 1):
                previous_node = walk_ids[i, j - d].item()
                # Repeat encoding: mark if next_node is exactly the same as the previous node.
                if next_node == previous_node:
                    encoding_repeat[i, j, d - 1] = 1

                # Edge encoding: mark if there is an edge between next_node and previous_node.
                if (next_node in neighbor_dict[previous_node]) or (previous_node in neighbor_dict[next_node]):
                    encoding_edge[i, j, d - 1] = 1


            # Per-step edge feature: edge (current_node -> next_node).
            if walk_pe_extra is not None:
                walk_pe_extra[i, j] = add_edge_feat[current_node, next_node]
            current_node = next_node

    # Attach the results to the data object.
    data.walk_emb = walk_emb
    data.walk_ids = walk_ids[None, :, :]
    walk_pe = torch.cat([encoding_repeat, encoding_edge], dim=-1)
    if walk_pe_extra is not None:
        walk_pe = torch.cat([walk_pe, walk_pe_extra], dim=-1)
    data.walk_pe = walk_pe
    
    return data

def sample_walks_mdlr(data, nw, l, s, non_backtracking, add_edge_feat=None):
    """
    Samples random walks from a graph using the minimum-degree local rule (MDLR) for neighbor sampling.
    Each walk is of length l, and neighbors are chosen with probability proportional to
        c(u, x) = 1/min(deg(u), deg(x))
    where deg(·) is computed from the precomputed neighbor dictionary.
    
    If non_backtracking is True, the walk avoids immediately returning to the previous node
    when possible.
    
    Additionally, for each random walk, an anonymization vector is computed.
    This vector records for each step in the walk an anonymized ID that reflects the order
    in which unique nodes were encountered.
    For example, if the walk is [1, 2, 1, 3], then the anonymized walk becomes [0, 1, 0, 2].
    
    Parameters:
        data (torch_geometric.data.Data): Graph data object; must have attributes 'edge_index'
                                            and 'x_emb'.
        nw (int): Number of random walks.
        l (int): Length of each walk (number of nodes in the walk).
        s (int): Unused here (kept for compatibility); originally was the window size for encodings.
        non_backtracking (bool): If True, prevents immediate backtracking (i.e. returning to the node
                                 visited in the previous step), unless it is the only option.
    
    Returns:
        data (torch_geometric.data.Data): The input data object is updated with the following new attributes:
            - walk_ids: A tensor of shape (1, nw, l) containing the raw node indices for each random walk.
            - walk_emb: A tensor of shape (nw, l) containing the corresponding node embeddings (via data.x_emb).
            - walk_anonym: A tensor of shape (nw, l) with the anonymized walk (each element is an integer representing
                           the order of first appearance of that node in the walk).
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use the precomputed neighbor dictionary (or compute & store it if not present).
    # get_neighbor_dict should store neighbor information as {node: set(neighbors)}
    neighbor_dict = get_neighbor_dict(data)

    # Pre-allocate tensors to store the random walks:
    # walk_ids will store the raw node indices.
    walk_ids = torch.empty((nw, l), dtype=torch.long)
    # walk_emb stores the corresponding node embedding ID (via data.x_emb).
    walk_emb = torch.empty((nw, l), dtype=torch.long)
    # walk_anonym will store the anonymized version of each walk.
    walk_anonym = torch.empty((nw, l), dtype=torch.long)

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    # MDLR baseline does not produce walk_pe; when add_edge_feat is None we
    # preserve that, otherwise we attach a (nw, l, B) walk_pe tensor.
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, l, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # Randomly select starting nodes for each walk.
    start_nodes = torch.randint(0, num_nodes, (nw,), dtype=torch.long)

    for i in range(nw):
        current_node = start_nodes[i].item()
        # Set the first node in the walk.
        walk_ids[i, 0] = current_node
        walk_emb[i, 0] = data.x_emb[current_node]
        # For anonymization, start by assigning the first node an anonymized label 0.
        anon_mapping = {current_node: 0}  # Maps each encountered node to its anonymized label.
        anon_counter = 1
        walk_anonym[i, 0] = 0

        for j in range(1, l):
            # Get the neighbors of the current node (as a set) using the precomputed dictionary.
            neighbors = neighbor_dict.get(current_node, set())
            if neighbors:
                # If non-backtracking, filter out the immediate previous node when possible.
                if non_backtracking and j > 0:
                    prev_node = walk_ids[i, j - 1].item()
                    filtered_neighbors = [n for n in neighbors if n != prev_node]
                    candidate_neighbors = filtered_neighbors if filtered_neighbors else list(neighbors)
                else:
                    candidate_neighbors = list(neighbors)
                # MDLR weighting: weight for neighbor x = 1 / min(deg(current_node), deg(x))
                deg_current = len(neighbor_dict[current_node])
                weights = []
                for x in candidate_neighbors:
                    deg_x = len(neighbor_dict[x])
                    weight = 1.0 / min(deg_current, deg_x)
                    weights.append(weight)
                # Sample one neighbor based on the computed weights.
                next_node = random.choices(candidate_neighbors, weights=weights, k=1)[0]
            else:
                next_node = current_node  # If no neighbor, stay at the current node.

            # Record the chosen neighbor.
            walk_ids[i, j] = next_node
            walk_emb[i, j] = data.x_emb[next_node]

            # Update the anonymization vector.
            # If the node has been seen before in this walk, reuse its anonymized label.
            # Otherwise, assign the next available label.
            if next_node in anon_mapping:
                walk_anonym[i, j] = anon_mapping[next_node]
            else:
                anon_mapping[next_node] = anon_counter
                walk_anonym[i, j] = anon_counter
                anon_counter += 1


            # Per-step edge feature: edge (current_node -> next_node).
            if walk_pe_extra is not None:
                walk_pe_extra[i, j] = add_edge_feat[current_node, next_node]
            current_node = next_node

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations
    if walk_pe_extra is not None:
        data.walk_pe = walk_pe_extra

    return data

def sample_walks_rum(data, nw, l, s, non_backtracking, add_edge_feat=None):
    """
    Samples random walks from a graph using the minimum-degree local rule (MDLR) for neighbor sampling.
    Each walk is of length l, and neighbors are chosen with probability proportional to
        c(u, x) = 1/min(deg(u), deg(x))
    where deg(·) is computed from the precomputed neighbor dictionary.
    
    If non_backtracking is True, the walk avoids immediately returning to the previous node
    when possible.
    
    Additionally, for each random walk, an anonymization vector is computed.
    This vector records for each step in the walk an anonymized ID that reflects the order
    in which unique nodes were encountered.
    For example, if the walk is [1, 2, 1, 3], then the anonymized walk becomes [0, 1, 0, 2].
    
    Parameters:
        data (torch_geometric.data.Data): Graph data object; must have attributes 'edge_index'
                                            and 'x_emb'.
        nw (int): Number of random walks.
        l (int): Length of each walk (number of nodes in the walk).
        s (int): Unused here (kept for compatibility); originally was the window size for encodings.
        non_backtracking (bool): If True, prevents immediate backtracking (i.e. returning to the node
                                 visited in the previous step), unless it is the only option.
    
    Returns:
        data (torch_geometric.data.Data): The input data object is updated with the following new attributes:
            - walk_ids: A tensor of shape (1, nw, l) containing the raw node indices for each random walk.
            - walk_emb: A tensor of shape (nw, l) containing the corresponding node embeddings (via data.x_emb).
            - walk_anonym: A tensor of shape (nw, l) with the anonymized walk (each element is an integer representing
                           the order of first appearance of that node in the walk).
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use the precomputed neighbor dictionary (or compute & store it if not present).
    # get_neighbor_dict should store neighbor information as {node: set(neighbors)}
    neighbor_dict = get_neighbor_dict(data)

    # Pre-allocate tensors to store the random walks:
    # walk_ids will store the raw node indices.
    walk_ids = torch.empty((nw, l), dtype=torch.long)
    # walk_emb stores the corresponding node embedding ID (via data.x_emb).
    walk_emb = torch.empty((nw, l), dtype=torch.long)
    # walk_anonym will store the anonymized version of each walk.
    walk_anonym = torch.empty((nw, l), dtype=torch.long)

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, l, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # Randomly select starting nodes for each walk.
    start_nodes = torch.randint(0, num_nodes, (nw,), dtype=torch.long)

    for i in range(nw):
        current_node = start_nodes[i].item()
        # Set the first node in the walk.
        walk_ids[i, 0] = current_node
        walk_emb[i, 0] = data.x_emb[current_node]
        # For anonymization, start by assigning the first node an anonymized label 0.
        anon_mapping = {current_node: 0}  # Maps each encountered node to its anonymized label.
        anon_counter = 1
        walk_anonym[i, 0] = 0

        for j in range(1, l):
            # Get the neighbors of the current node (as a set) using the precomputed dictionary.
            neighbors = neighbor_dict.get(current_node, set())
            # if neighbors:
            #     # If non-backtracking, filter out the immediate previous node when possible.
            #     if non_backtracking and j > 0:
            #         prev_node = walk_ids[i, j - 1].item()
            #         filtered_neighbors = [n for n in neighbors if n != prev_node]
            #         candidate_neighbors = filtered_neighbors if filtered_neighbors else list(neighbors)
            #     else:
            #         candidate_neighbors = list(neighbors)
            #     # MDLR weighting: weight for neighbor x = 1 / min(deg(current_node), deg(x))
            #     deg_current = len(neighbor_dict[current_node])
            #     weights = []
            #     for x in candidate_neighbors:
            #         deg_x = len(neighbor_dict[x])
            #         weight = 1.0 / min(deg_current, deg_x)
            #         weights.append(weight)
            #     # Sample one neighbor based on the computed weights.
            #     next_node = random.choices(candidate_neighbors, weights=weights, k=1)[0]
            # else:
            #     next_node = current_node  # If no neighbor, stay at the current node.

            if neighbors:
                if non_backtracking and j > 0:
                    prev_node = walk_ids[i, j - 1].item()
                    # Filter out the immediate previous node.
                    filtered_neighbors = [n for n in neighbors if n != prev_node]
                    if filtered_neighbors:
                        next_node = random.choice(filtered_neighbors)
                    else:
                        next_node = random.choice(list(neighbors))
                else:
                    next_node = random.choice(list(neighbors))
            else:
                next_node = current_node

            # Record the chosen neighbor.
            walk_ids[i, j] = next_node
            walk_emb[i, j] = data.x_emb[next_node]

            # Update the anonymization vector.
            # If the node has been seen before in this walk, reuse its anonymized label.
            # Otherwise, assign the next available label.
            if next_node in anon_mapping:
                walk_anonym[i, j] = anon_mapping[next_node]
            else:
                anon_mapping[next_node] = anon_counter
                walk_anonym[i, j] = anon_counter
                anon_counter += 1


            # Per-step edge feature: edge (current_node -> next_node).
            if walk_pe_extra is not None:
                walk_pe_extra[i, j] = add_edge_feat[current_node, next_node]
            current_node = next_node

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations
    if walk_pe_extra is not None:
        data.walk_pe = walk_pe_extra

    return data

def sample_walks_adaptive(data, nw, l, s, non_backtracking, max_len, vocab, add_edge_feat=None):
    """
    Performs DFS-based searches on the graph and computes the edge encoding on the fly.
    
    For each DFS search:
      - A random starting node is chosen.
      - A DFS is performed (without revisiting nodes) to obtain an ordering of nodes.
      - As each node is visited, its embedding id (obtained via vocab[data.x_emb[node]])
        is recorded in the search order.
      - For each new node (at position j in the DFS order), for each offset d in [1, min(s, j)],
        if there is an edge between the node at position j and the node at position j-d (in the
        original graph), the corresponding entry in the edge encoding is set to 1.
      - The DFS order is stored in a tensor of shape (nw, max_len), where unfilled positions
        remain padded with vocab['PAD'].
      - The actual length (number of nodes visited) is recorded.
    
    Parameters:
        data (torch_geometric.data.Data): Graph data object with 'edge_index' and 'x_emb' attributes.
        nw (int): Number of DFS searches to perform.
        max_len (int): Maximum length of each DFS search order; shorter searches are padded with vocab['PAD'].
        s (int): Window size for the edge encoding.
        vocab (dict): Mapping from tokens to embedding ids; must include an entry for 'PAD'.
    
    Returns:
        data (torch_geometric.data.Data): The input data object with three new attributes:
            - walk_emb: A tensor of shape (nw, max_len) with the DFS search order in embedding ids.
            - walk_pe: A tensor of shape (nw, max_len, s) with the computed edge encoding.
            - lengths: A tensor of shape (nw,) where each entry is the number of nodes visited in the DFS search.
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use the precomputed neighbor dictionary (or compute & store it if not present).
    neighbor_dict = get_neighbor_dict(data)  # get_neighbor_dict returns a dict with sets.

    # Pre-allocate tensors for DFS searches.
    # searches_emb = torch.full((nw, max_len), vocab['PAD'], dtype=torch.long)
    # searches = torch.full((nw, max_len), -1, dtype=torch.long)
    # encoding_edge = torch.zeros((nw, max_len, s), dtype=torch.float)
    # Pre-allocate tensors for the random walks and the encodings.
    walk_ids = torch.full((nw, max_len), -1, dtype=torch.long)
    walk_emb = torch.full((nw, max_len), vocab['PAD'], dtype=torch.long)
    encoding_repeat = torch.zeros((nw, max_len, s), dtype=torch.float)
    encoding_edge = torch.zeros((nw, max_len, s), dtype=torch.float)
    lengths = []

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, max_len, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # For each DFS search:
    # Randomly select starting nodes for each walk.
    start_nodes = torch.randint(0, num_nodes, (nw,), dtype=torch.long)
    
    for i in range(nw):
        current_node = start_nodes[i].item()
        walk_ids[i, 0] = current_node
        walk_emb[i, 0] = data.x_emb[current_node]
        # For the first node there is no previous context; encodings remain zeros.
        for j in range(1, l):
            # Get neighbors from the set (if any)
            neighbors = neighbor_dict.get(current_node, set())
            if neighbors:
                if non_backtracking and j > 0:
                    prev_node = walk_ids[i, j - 1].item()
                    # Filter out the immediate previous node.
                    filtered_neighbors = [n for n in neighbors if n != prev_node]
                    if filtered_neighbors:
                        next_node = random.choice(filtered_neighbors)
                    else:
                        next_node = random.choice(list(neighbors))
                else:
                    next_node = random.choice(list(neighbors))
            else:
                next_node = current_node
            walk_ids[i, j] = next_node
            walk_emb[i, j] = data.x_emb[next_node]
            
            # Update encodings on the fly for offsets within the window (up to s)
            for d in range(1, min(s, j) + 1):
                previous_node = walk_ids[i, j - d].item()
                # Repeat encoding: mark if next_node is exactly the same as the previous node.
                if next_node == previous_node:
                    encoding_repeat[i, j, d - 1] = 1

                # Edge encoding: mark if there is an edge between next_node and previous_node.
                if (next_node in neighbor_dict[previous_node]) or (previous_node in neighbor_dict[next_node]):
                    encoding_edge[i, j, d - 1] = 1


            # Per-step edge feature: edge (current_node -> next_node).
            if walk_pe_extra is not None:
                walk_pe_extra[i, j] = add_edge_feat[current_node, next_node]
            current_node = next_node
                    
        lengths.append(l)

    data.walk_emb = walk_emb
    data.walk_ids = walk_ids[None, :, :]
    walk_pe = torch.cat([encoding_repeat, encoding_edge], dim=-1)
    if walk_pe_extra is not None:
        walk_pe = torch.cat([walk_pe, walk_pe_extra], dim=-1)
    data.walk_pe = walk_pe
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    return data

def sample_walks_mdlr_adaptive(data, nw, l, s, non_backtracking, max_len, vocab, add_edge_feat=None):
    """
    Samples random walks from a graph using the minimum-degree local rule (MDLR) for neighbor sampling.
    Each walk is of length l, and neighbors are chosen with probability proportional to
        c(u, x) = 1/min(deg(u), deg(x))
    where deg(·) is computed from the precomputed neighbor dictionary.
    
    If non_backtracking is True, the walk avoids immediately returning to the previous node
    when possible.
    
    Additionally, for each random walk, an anonymization vector is computed.
    This vector records for each step in the walk an anonymized ID that reflects the order
    in which unique nodes were encountered.
    For example, if the walk is [1, 2, 1, 3], then the anonymized walk becomes [0, 1, 0, 2].
    
    Parameters:
        data (torch_geometric.data.Data): Graph data object; must have attributes 'edge_index'
                                            and 'x_emb'.
        nw (int): Number of random walks.
        l (int): Length of each walk (number of nodes in the walk).
        s (int): Unused here (kept for compatibility); originally was the window size for encodings.
        non_backtracking (bool): If True, prevents immediate backtracking (i.e. returning to the node
                                 visited in the previous step), unless it is the only option.
    
    Returns:
        data (torch_geometric.data.Data): The input data object is updated with the following new attributes:
            - walk_ids: A tensor of shape (1, nw, l) containing the raw node indices for each random walk.
            - walk_emb: A tensor of shape (nw, l) containing the corresponding node embeddings (via data.x_emb).
            - walk_anonym: A tensor of shape (nw, l) with the anonymized walk (each element is an integer representing
                           the order of first appearance of that node in the walk).
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use the precomputed neighbor dictionary (or compute & store it if not present).
    # get_neighbor_dict should store neighbor information as {node: set(neighbors)}
    neighbor_dict = get_neighbor_dict(data)

    # Pre-allocate tensors to store the random walks:
    # walk_ids will store the raw node indices.
    # walk_ids = torch.empty((nw, l), dtype=torch.long)
    # walk_emb stores the corresponding node embedding ID (via data.x_emb).
    # walk_emb = torch.empty((nw, l), dtype=torch.long)
    # walk_anonym will store the anonymized version of each walk.
    # walk_anonym = torch.empty((nw, l), dtype=torch.long)

    # Pre-allocate tensors for the random walks and the encodings.
    walk_ids = torch.full((nw, max_len), -1, dtype=torch.long)
    walk_emb = torch.full((nw, max_len), vocab['PAD'], dtype=torch.long)
    walk_anonym = torch.empty((nw, max_len), dtype=torch.long)
    lengths = []

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, max_len, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # Randomly select starting nodes for each walk.
    start_nodes = torch.randint(0, num_nodes, (nw,), dtype=torch.long)

    for i in range(nw):
        current_node = start_nodes[i].item()
        # Set the first node in the walk.
        walk_ids[i, 0] = current_node
        walk_emb[i, 0] = data.x_emb[current_node]
        # For anonymization, start by assigning the first node an anonymized label 0.
        anon_mapping = {current_node: 0}  # Maps each encountered node to its anonymized label.
        anon_counter = 1
        walk_anonym[i, 0] = 0

        for j in range(1, l):
            # Get the neighbors of the current node (as a set) using the precomputed dictionary.
            neighbors = neighbor_dict.get(current_node, set())
            if neighbors:
                # If non-backtracking, filter out the immediate previous node when possible.
                if non_backtracking and j > 0:
                    prev_node = walk_ids[i, j - 1].item()
                    filtered_neighbors = [n for n in neighbors if n != prev_node]
                    candidate_neighbors = filtered_neighbors if filtered_neighbors else list(neighbors)
                else:
                    candidate_neighbors = list(neighbors)
                # MDLR weighting: weight for neighbor x = 1 / min(deg(current_node), deg(x))
                deg_current = len(neighbor_dict[current_node])
                weights = []
                for x in candidate_neighbors:
                    deg_x = len(neighbor_dict[x])
                    weight = 1.0 / min(deg_current, deg_x)
                    weights.append(weight)
                # Sample one neighbor based on the computed weights.
                next_node = random.choices(candidate_neighbors, weights=weights, k=1)[0]
            else:
                next_node = current_node  # If no neighbor, stay at the current node.

            # Record the chosen neighbor.
            walk_ids[i, j] = next_node
            walk_emb[i, j] = data.x_emb[next_node]

            # Update the anonymization vector.
            # If the node has been seen before in this walk, reuse its anonymized label.
            # Otherwise, assign the next available label.
            if next_node in anon_mapping:
                walk_anonym[i, j] = anon_mapping[next_node]
            else:
                anon_mapping[next_node] = anon_counter
                walk_anonym[i, j] = anon_counter
                anon_counter += 1


            # Per-step edge feature: edge (current_node -> next_node).
            if walk_pe_extra is not None:
                walk_pe_extra[i, j] = add_edge_feat[current_node, next_node]
            current_node = next_node

        lengths.append(l)

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    if walk_pe_extra is not None:
        data.walk_pe = walk_pe_extra

    return data

def sample_walks_rum_adaptive(data, nw, l, s, non_backtracking, max_len, vocab, add_edge_feat=None):
    """
    Samples random walks from a graph using the minimum-degree local rule (MDLR) for neighbor sampling.
    Each walk is of length l, and neighbors are chosen with probability proportional to
        c(u, x) = 1/min(deg(u), deg(x))
    where deg(·) is computed from the precomputed neighbor dictionary.
    
    If non_backtracking is True, the walk avoids immediately returning to the previous node
    when possible.
    
    Additionally, for each random walk, an anonymization vector is computed.
    This vector records for each step in the walk an anonymized ID that reflects the order
    in which unique nodes were encountered.
    For example, if the walk is [1, 2, 1, 3], then the anonymized walk becomes [0, 1, 0, 2].
    
    Parameters:
        data (torch_geometric.data.Data): Graph data object; must have attributes 'edge_index'
                                            and 'x_emb'.
        nw (int): Number of random walks.
        l (int): Length of each walk (number of nodes in the walk).
        s (int): Unused here (kept for compatibility); originally was the window size for encodings.
        non_backtracking (bool): If True, prevents immediate backtracking (i.e. returning to the node
                                 visited in the previous step), unless it is the only option.
    
    Returns:
        data (torch_geometric.data.Data): The input data object is updated with the following new attributes:
            - walk_ids: A tensor of shape (1, nw, l) containing the raw node indices for each random walk.
            - walk_emb: A tensor of shape (nw, l) containing the corresponding node embeddings (via data.x_emb).
            - walk_anonym: A tensor of shape (nw, l) with the anonymized walk (each element is an integer representing
                           the order of first appearance of that node in the walk).
    """
    # Determine the total number of nodes.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    elif hasattr(data, 'x'):
        num_nodes = data.x.size(0)
    else:
        num_nodes = int(data.edge_index.max()) + 1

    # Use the precomputed neighbor dictionary (or compute & store it if not present).
    # get_neighbor_dict should store neighbor information as {node: set(neighbors)}
    neighbor_dict = get_neighbor_dict(data)

    # Pre-allocate tensors to store the random walks:
    # walk_ids will store the raw node indices.
    # walk_ids = torch.empty((nw, l), dtype=torch.long)
    # walk_emb stores the corresponding node embedding ID (via data.x_emb).
    # walk_emb = torch.empty((nw, l), dtype=torch.long)
    # walk_anonym will store the anonymized version of each walk.
    # walk_anonym = torch.empty((nw, l), dtype=torch.long)

    # Pre-allocate tensors for the random walks and the encodings.
    walk_ids = torch.full((nw, max_len), -1, dtype=torch.long)
    walk_emb = torch.full((nw, max_len), vocab['PAD'], dtype=torch.long)
    walk_anonym = torch.empty((nw, max_len), dtype=torch.long)
    lengths = []

    # Optional per-step edge-feature stream appended to walk_pe (variant A).
    if add_edge_feat is not None:
        d_edge = int(add_edge_feat.shape[-1])
        # Match dtype/device of add_edge_feat so e.g. CUDA tensors stay on GPU
        # and bf16/half are preserved through the per-step scatter.
        walk_pe_extra = torch.zeros(
            (nw, max_len, d_edge),
            dtype=add_edge_feat.dtype,
            device=add_edge_feat.device,
        )
    else:
        walk_pe_extra = None

    # Randomly select starting nodes for each walk.
    start_nodes = torch.randint(0, num_nodes, (nw,), dtype=torch.long)

    for i in range(nw):
        current_node = start_nodes[i].item()
        # Set the first node in the walk.
        walk_ids[i, 0] = current_node
        walk_emb[i, 0] = data.x_emb[current_node]
        # For anonymization, start by assigning the first node an anonymized label 0.
        anon_mapping = {current_node: 0}  # Maps each encountered node to its anonymized label.
        anon_counter = 1
        walk_anonym[i, 0] = 0

        for j in range(1, l):
            # Get the neighbors of the current node (as a set) using the precomputed dictionary.
            neighbors = neighbor_dict.get(current_node, set())
            # if neighbors:
            #     # If non-backtracking, filter out the immediate previous node when possible.
            #     if non_backtracking and j > 0:
            #         prev_node = walk_ids[i, j - 1].item()
            #         filtered_neighbors = [n for n in neighbors if n != prev_node]
            #         candidate_neighbors = filtered_neighbors if filtered_neighbors else list(neighbors)
            #     else:
            #         candidate_neighbors = list(neighbors)
            #     # MDLR weighting: weight for neighbor x = 1 / min(deg(current_node), deg(x))
            #     deg_current = len(neighbor_dict[current_node])
            #     weights = []
            #     for x in candidate_neighbors:
            #         deg_x = len(neighbor_dict[x])
            #         weight = 1.0 / min(deg_current, deg_x)
            #         weights.append(weight)
            #     # Sample one neighbor based on the computed weights.
            #     next_node = random.choices(candidate_neighbors, weights=weights, k=1)[0]
            # else:
            #     next_node = current_node  # If no neighbor, stay at the current node.

            if neighbors:
                if non_backtracking and j > 0:
                    prev_node = walk_ids[i, j - 1].item()
                    # Filter out the immediate previous node.
                    filtered_neighbors = [n for n in neighbors if n != prev_node]
                    if filtered_neighbors:
                        next_node = random.choice(filtered_neighbors)
                    else:
                        next_node = random.choice(list(neighbors))
                else:
                    next_node = random.choice(list(neighbors))
            else:
                next_node = current_node

            # Record the chosen neighbor.
            walk_ids[i, j] = next_node
            walk_emb[i, j] = data.x_emb[next_node]

            # Update the anonymization vector.
            # If the node has been seen before in this walk, reuse its anonymized label.
            # Otherwise, assign the next available label.
            if next_node in anon_mapping:
                walk_anonym[i, j] = anon_mapping[next_node]
            else:
                anon_mapping[next_node] = anon_counter
                walk_anonym[i, j] = anon_counter
                anon_counter += 1


            # Per-step edge feature: edge (current_node -> next_node).
            if walk_pe_extra is not None:
                walk_pe_extra[i, j] = add_edge_feat[current_node, next_node]
            current_node = next_node
            
        lengths.append(l)

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    if walk_pe_extra is not None:
        data.walk_pe = walk_pe_extra

    return data


def dfs_edges(data):
    """
    Perform a single depth-first search on a connected graph `data`
    (torch_geometric.data.Data) and return the list of edges traversed
    by the DFS, in the order they are first explored.

    Returns
    -------
    edges : List[Tuple[int,int]]
        Each tuple (u, v) is a tree-edge discovered by the DFS, where
        `u` is the parent and `v` is the newly visited child.
    """
    # --- build a neighbor dictionary (node -> set(neighbors)) ---
    if '_neighbor_dict' in data.__dict__:        # reuse cached one if present
        nbr = data._neighbor_dict
    else:
        if hasattr(data, 'num_nodes'):
            num_nodes = data.num_nodes
        elif hasattr(data, 'x'):
            num_nodes = data.x.size(0)
        else:
            num_nodes = int(data.edge_index.max()) + 1

        nbr = {i: set() for i in range(num_nodes)}
        ei = data.edge_index
        for k in range(ei.size(1)):
            u, v = ei[0, k].item(), ei[1, k].item()
            nbr[u].add(v)
            nbr[v].add(u)           # undirected; drop this line if the graph is directed
        data._neighbor_dict = nbr   # cache privately

    # --- DFS traversal ---
    start = random.randint(0, len(nbr) - 1)
    visited = set()
    stack   = [(start, None)]   # (current_node, parent)
    edges   = []

    while stack:
        node, parent = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        if parent is not None:
            edges.append((parent, node))

        nxt = list(nbr[node])
        random.shuffle(nxt)     # randomize order, optional
        for nb in nxt:
            if nb not in visited:
                stack.append((nb, node))

    return edges
