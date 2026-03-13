"""
Structural Positional Encoding (SPE) for circuit graph nodes.

Computes topology-based features that give each node a unique identity:
- Node degree on the raw bipartite graph (normalized)
- Shortest path distance to output net on the net-level graph
- Random Walk PE on the net-level graph (RWPE)

The net-level graph contracts devices: two nets are connected if they
share a device (MOSFET, resistor, etc.). This avoids the bipartite
oscillation problem of the raw terminal-net graph.

These features are transferable across different circuit topologies.
"""

import torch
import numpy as np
from collections import deque, defaultdict


def _build_net_level_graph(edge_index_np, num_nodes, num_terminals, node_names=None):
    """
    Build a net-level adjacency from the bipartite terminal-net graph.

    Two nets are neighbors if they belong to the same device. We group
    terminals by device (using node name prefix, e.g., 'Xm1_drain' -> 'Xm1'),
    find which net each terminal connects to, then connect all nets
    within each device.
    """
    src, dst = edge_index_np[0], edge_index_np[1]

    # Map each terminal to its connected net
    terminal_to_net = {}
    for s, d in zip(src, dst):
        if s < num_terminals and d >= num_terminals:
            terminal_to_net[s] = d - num_terminals
        elif d < num_terminals and s >= num_terminals:
            terminal_to_net[d] = s - num_terminals

    # Group terminals by device using name prefix
    device_terminals = defaultdict(list)
    if node_names is not None:
        for t_idx in range(num_terminals):
            name = node_names[t_idx]
            # Device prefix: everything before the last underscore
            # e.g., 'Xm1_drain' -> 'Xm1', 'Cc_p' -> 'Cc'
            parts = name.rsplit('_', 1)
            device_name = parts[0] if len(parts) > 1 else name
            device_terminals[device_name].append(t_idx)
    else:
        # Fallback: group every 4 terminals as one device (MOSFET assumption)
        for i in range(0, num_terminals, 4):
            device_terminals[f'dev_{i//4}'] = list(range(i, min(i + 4, num_terminals)))

    # Build net-level adjacency: nets sharing a device are connected
    num_nets = num_nodes - num_terminals
    net_adj = defaultdict(set)

    for device_name, terminals in device_terminals.items():
        # Find nets connected to this device's terminals
        device_nets = set()
        for t in terminals:
            if t in terminal_to_net:
                device_nets.add(terminal_to_net[t])

        # All nets in this device form a clique
        device_nets = list(device_nets)
        for i, n1 in enumerate(device_nets):
            for n2 in device_nets[i + 1:]:
                net_adj[n1].add(n2)
                net_adj[n2].add(n1)

    return net_adj, num_nets, terminal_to_net


def compute_structural_pe(
    edge_index: torch.Tensor,
    num_nodes: int,
    output_node_mask: torch.Tensor,
    rwpe_steps: int = 16,
    num_terminals: int = None,
    node_names: list = None,
) -> torch.Tensor:
    """
    Compute structural positional encoding for all nodes.

    Args:
        edge_index: [2, num_edges] edge index tensor
        num_nodes: Total number of nodes in the graph
        output_node_mask: Boolean mask indicating output node(s)
        rwpe_steps: Number of random walk steps for RWPE (K)
        num_terminals: Number of terminal nodes (first num_terminals nodes).
                      If None, inferred from output_node_mask.
        node_names: Optional list of node names to identify output net.

    Returns:
        [num_nodes, 2 + rwpe_steps] tensor:
            - dim 0: normalized node degree (raw graph)
            - dim 1: normalized shortest path distance to output (net-level)
            - dims 2..K+2: RWPE on net-level graph (mapped to all nodes)
    """
    device = edge_index.device
    edge_index_np = edge_index.cpu().numpy()
    src, dst = edge_index_np[0], edge_index_np[1]

    # Infer num_terminals: first contiguous block of non-net nodes
    if num_terminals is None:
        # Net nodes are at the end; terminals come first
        # Use output_node_mask: net nodes that are outputs start after terminals
        output_indices = output_node_mask.cpu().nonzero(as_tuple=True)[0].tolist()
        if output_indices:
            # First output node index gives us approximate boundary
            # But there may be non-output nets too. Use min output index.
            num_terminals = min(output_indices)
        else:
            num_terminals = num_nodes  # no nets identified

    num_nets = num_nodes - num_terminals

    # 1. Node degree on raw bipartite graph (normalized, net nodes only)
    degree = np.zeros(num_nodes, dtype=np.float32)
    for s in src:
        degree[s] += 1
    max_deg = max(degree.max(), 1.0)
    degree_norm = degree / max_deg
    # Zero out terminal nodes — they already have unique device features
    degree_norm[:num_terminals] = 0.0

    # 2 & 3. Build net-level graph for distance and RWPE
    net_adj, _, terminal_to_net = _build_net_level_graph(
        edge_index_np, num_nodes, num_terminals, node_names
    )

    # Find actual output net (vout) — use node_names if available, else heuristic
    output_net_local = None
    if node_names is not None:
        for i, name in enumerate(node_names):
            if name == 'vout':
                output_net_local = i - num_terminals
                break

    if output_net_local is None:
        # Heuristic: use the output_node_mask, pick the net with highest degree
        # as likely output (vout connects to load, feedback, output stage)
        output_nets = [i - num_terminals for i in
                       output_node_mask.cpu().nonzero(as_tuple=True)[0].tolist()
                       if i >= num_terminals]
        if output_nets:
            # Pick the one with most connections in net-level graph
            output_net_local = max(output_nets, key=lambda n: len(net_adj.get(n, set())))
        else:
            output_net_local = 0

    # BFS on net-level graph from output net
    net_dist = np.full(num_nets, -1, dtype=np.float32)
    queue = deque()
    net_dist[output_net_local] = 0
    queue.append(output_net_local)

    while queue:
        node = queue.popleft()
        for neighbor in net_adj.get(node, set()):
            if net_dist[neighbor] < 0:
                net_dist[neighbor] = net_dist[node] + 1
                queue.append(neighbor)

    # Unreachable nets
    max_net_dist = max(net_dist[net_dist >= 0].max(), 1.0) if (net_dist >= 0).any() else 1.0
    net_dist[net_dist < 0] = max_net_dist + 1
    net_dist_norm = net_dist / (max_net_dist + 1)

    # Map net distances to all nodes (net nodes only, terminals stay zero)
    dist_all = np.zeros(num_nodes, dtype=np.float32)
    dist_all[num_terminals:] = net_dist_norm

    # 3. RWPE on net-level graph
    from scipy.sparse import csr_matrix

    # Build net-level sparse adjacency
    net_rows, net_cols = [], []
    for n, neighbors in net_adj.items():
        for nb in neighbors:
            net_rows.append(n)
            net_cols.append(nb)

    if net_rows:
        net_rows = np.array(net_rows)
        net_cols = np.array(net_cols)

        # Row-normalize
        net_degree = np.zeros(num_nets, dtype=np.float32)
        for r in net_rows:
            net_degree[r] += 1
        net_degree = np.maximum(net_degree, 1.0)

        data_vals = np.array([1.0 / net_degree[r] for r in net_rows], dtype=np.float32)
        rw_matrix = csr_matrix((data_vals, (net_rows, net_cols)), shape=(num_nets, num_nets))

        # Compute RW^k diagonal
        net_rwpe = np.zeros((num_nets, rwpe_steps), dtype=np.float32)
        rw_power = rw_matrix.copy()
        for k in range(rwpe_steps):
            net_rwpe[:, k] = rw_power.diagonal()
            if k < rwpe_steps - 1:
                rw_power = rw_power @ rw_matrix
    else:
        net_rwpe = np.zeros((num_nets, rwpe_steps), dtype=np.float32)

    # Map RWPE to all nodes (net nodes only, terminals stay zero)
    rwpe_all = np.zeros((num_nodes, rwpe_steps), dtype=np.float32)
    rwpe_all[num_terminals:] = net_rwpe

    # Stack all features
    features = np.concatenate([
        degree_norm[:, None],     # [N, 1]
        dist_all[:, None],        # [N, 1]
        rwpe_all,                 # [N, K]
    ], axis=1)

    return torch.tensor(features, dtype=torch.float32, device=device)
