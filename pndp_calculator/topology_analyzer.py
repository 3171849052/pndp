import math
from typing import List, Tuple
import networkx as nx
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import cvxpy as cp


def build_random_walk_matrix(graph: nx.Graph) -> np.ndarray:
    matrix = nx.to_numpy_array(graph)
    n = graph.number_of_nodes()
    degrees = matrix.sum(axis=1)
    mh_matrix = np.zeros_like(matrix)
    
    for i in range(n):
        for j in range(n):
            if i != j and matrix[i, j] > 0:
                mh_matrix[i, j] = 1.0 / max(degrees[i], degrees[j])
        mh_matrix[i, i] = 1.0 - mh_matrix[i].sum()
        
    return mh_matrix


def build_W_powers(W: np.ndarray, max_power: int) -> List[np.ndarray]:
    powers = [np.eye(W.shape[0])]
    for _ in range(max_power):
        powers.append(powers[-1] @ W)
    return powers


def build_B_a_sparse(W_powers: List[np.ndarray], R: int, T: int, K: int, target_node: int, graph: nx.Graph) -> sp.csc_matrix:
    n = W_powers[0].shape[0]
    observed_nodes = [target_node] + list(graph.neighbors(target_node))
    observed_nodes = sorted(set(observed_nodes))
    num_obs = len(observed_nodes)
    
    row_indices = []
    col_indices = []
    data = []
    
    for r in range(R):
        for k in range(K):
            row_v = r * K + k
            for q in range(r + 1): 
                power_idx = (r - q) * K + k
                block = W_powers[power_idx]
                block_sub = block[observed_nodes, :]
                
                row_start = row_v * num_obs
                col_start = (q * T) * n
                
                nz_rows, nz_cols = np.nonzero(block_sub)
                vals = block_sub[nz_rows, nz_cols]
                
                for t in range(T):
                    t_col_start = col_start + t * n
                    row_indices.extend(nz_rows + row_start)
                    col_indices.extend(nz_cols + t_col_start)
                    data.extend(vals)
                
    B_a = sp.csc_matrix((data, (row_indices, col_indices)), shape=(R * K * num_obs, R * T * n))
    return B_a


def _compute_kb_pairwise_variance_exact_sdp_optimized(
    A: sp.csc_matrix, A_inv: np.ndarray, n_nodes: int, total_steps: int, k: int, b: int
) -> np.ndarray:
    max_variances = np.zeros(n_nodes)
    actual_k = min(k, 1 + (total_steps - 1) // b) if b > 0 else 1

    H_param = cp.Parameter((actual_k, actual_k), symmetric=True)
    X = cp.Variable((actual_k, actual_k), PSD=True)
    objective = cp.Maximize(cp.trace(H_param @ X))
    constraints = [cp.diag(X) <= 1.0]
    prob = cp.Problem(objective, constraints)

    for node_idx in range(n_nodes):
        node_max_var = 0.0
        for start_step in range(total_steps - (actual_k - 1) * b):
            pi_indices = []
            for step_offset in range(actual_k):
                current_step = start_step + step_offset * b
                col_idx = current_step * n_nodes + node_idx
                pi_indices.append(col_idx)
            
            A_sub = A[:, pi_indices].toarray()
            A_inv_sub = A_inv[pi_indices, :]
            H_sub = A_inv_sub @ A_sub
            H_sub = (H_sub + H_sub.T) / 2.0 
            
            upper_bound = np.sum(np.abs(H_sub))
            if upper_bound <= node_max_var:
                continue
            
            H_param.value = H_sub
            try:
                prob.solve(warm_start=True, verbose=False)
                if prob.status in ["optimal", "optimal_inaccurate"] and prob.value is not None:
                    current_variance = prob.value
                else:
                    current_variance = upper_bound
            except cp.error.SolverError:
                current_variance = upper_bound
            
            if current_variance > node_max_var:
                node_max_var = current_variance
                
        max_variances[node_idx] = float(node_max_var)
        
    return max_variances


def compute_victim_centric_effective_variances(
    W: np.ndarray, R: int, T: int, K: int, graph: nx.Graph,
    k_participation: int, b_interval: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = W.shape[0]
    total_steps = R * T
    W_powers = build_W_powers(W, R * K)
    
    M = np.zeros((n_nodes, n_nodes))
    
    for attacker in tqdm(range(n_nodes), desc="Computing Exact SDP (Victim-centric)"):
        B_a = build_B_a_sparse(W_powers, R, T, K, attacker, graph)
        
        col_indices = np.arange(total_steps) * n_nodes + attacker
        mask = np.ones(B_a.shape[1])
        mask[col_indices] = 0.0
        Mask_mat = sp.diags(mask)
        A_cleaned = B_a @ Mask_mat
        
        A_dense = A_cleaned.toarray()
        A_inv = np.linalg.pinv(A_dense)
        
        variances_for_attacker = _compute_kb_pairwise_variance_exact_sdp_optimized(
            A_cleaned, A_inv, n_nodes, total_steps, k_participation, b_interval
        )
        M[attacker, :] = variances_for_attacker
        
    victim_variances = np.zeros(n_nodes)
    worst_attackers = np.zeros(n_nodes, dtype=int)
    worst_distances = np.zeros(n_nodes, dtype=int)
    for u in range(n_nodes):
        best_val = -1.0
        best_attacker = -1
        for a in range(n_nodes):
            if a == u:
                continue
            if M[a, u] > best_val:
                best_val = M[a, u]
                best_attacker = a
        victim_variances[u] = best_val
        worst_attackers[u] = best_attacker
        worst_distances[u] = nx.shortest_path_length(graph, source=u, target=best_attacker)
        
    return victim_variances, worst_attackers, worst_distances, M


def compute_standard_ldp_effective_variance(R: int, T: int, b: int) -> float:
    total_steps = R * T
    k_participation = math.ceil(total_steps / b)
    return k_participation / T
