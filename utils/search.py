import torch
import random
from collections import deque

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
    
def sample_bfs(data, nw, s, max_len, vocab):
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
    data.walk_pe = encoding_edge
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    
    return data

def sample_dfs(data, nw, s, max_len, vocab):
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
    searches_emb = torch.full((nw, max_len), vocab['PAD'], dtype=torch.long)
    searches = torch.full((nw, max_len), -1, dtype=torch.long)
    encoding_edge = torch.zeros((nw, max_len, s), dtype=torch.float)
    lengths = []

    # For each DFS search:
    for i in range(nw):
        start_node = random.randint(0, num_nodes - 1)
        visited = set()
        stack = [start_node]
        order = []  # To store the raw node indices in the DFS order.
        pos = 0     # Current position in the DFS order.
        
        while stack and pos < max_len:
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

            pos += 1

            # Push unvisited neighbors onto the stack in randomized order.
            neighbors = list(neighbor_dict[node])
            random.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited:
                    stack.append(nb)
                    
        lengths.append(pos)

    data.walk_emb = searches_emb
    data.walk_ids = searches[None, :, :]
    data.walk_pe = encoding_edge
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    return data

def sample_walks(data, nw, l, s, non_backtracking):
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

            current_node = next_node

    # Attach the results to the data object.
    data.walk_emb = walk_emb
    data.walk_ids = walk_ids[None, :, :]
    data.walk_pe = torch.cat([encoding_repeat, encoding_edge], dim=-1)
    
    return data

def sample_walks_mdlr(data, nw, l, s, non_backtracking):
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

            current_node = next_node

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations

    return data

def sample_walks_rum(data, nw, l, s, non_backtracking):
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

            current_node = next_node

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations

    return data

def sample_walks_adaptive(data, nw, l, s, non_backtracking, max_len, vocab):
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

            current_node = next_node
                    
        lengths.append(l)

    data.walk_emb = walk_emb
    data.walk_ids = walk_ids[None, :, :]
    data.walk_pe = torch.cat([encoding_repeat, encoding_edge], dim=-1)
    data.lengths = torch.tensor(lengths, dtype=torch.long)
    return data

def sample_walks_mdlr_adaptive(data, nw, l, s, non_backtracking, max_len, vocab):
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

            current_node = next_node

        lengths.append(l)

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations
    data.lengths = torch.tensor(lengths, dtype=torch.long)

    return data

def sample_walks_rum_adaptive(data, nw, l, s, non_backtracking, max_len, vocab):
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

            current_node = next_node
            
        lengths.append(l)

    # Attach the results to the data object.
    # We store the raw node indices with an extra batch dimension for consistency.
    data.walk_ids = walk_ids[None, :, :]
    data.walk_emb = walk_emb  # (nw, l)
    data.walk_anonym = walk_anonym  # (nw, l) anonymized walk representations
    data.lengths = torch.tensor(lengths, dtype=torch.long)

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
