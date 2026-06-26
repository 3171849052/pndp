import math
import numpy as np
import networkx as nx

from .topology_analyzer import (
    build_random_walk_matrix,
    compute_victim_centric_effective_variances,
    compute_standard_ldp_effective_variance,
)
from .core_accountant import calculate_optimal_sigma_rdp, calculate_optimal_sigma_gdp


class PNDPAccountant:
    def __init__(self, N_samples: int, batch_size: int, T_local_steps: int, R_rounds: int, K_gossip: int = 1):
        self.N_samples = N_samples
        self.batch_size = batch_size
        self.T_local_steps = T_local_steps
        self.R_rounds = R_rounds
        self.K_gossip = K_gossip
        
        self.b_interval = int(math.ceil(N_samples / batch_size))
        self.total_steps = int(R_rounds * T_local_steps)
        self.k_participation = int(math.ceil(self.total_steps / self.b_interval))
        self.graph = None

    def set_graph(self, graph: nx.Graph):
        self.graph = nx.convert_node_labels_to_integers(graph)
        
    def get_noise_multiplier(self, target_epsilon: float, target_delta: float, algorithm="Average", framework="RDP"):
        if algorithm == "LDP-per-round":
            eff_var_mult = compute_standard_ldp_effective_variance(
                self.R_rounds, self.T_local_steps, self.b_interval
            )
            nm, metric = self._compute_final_multiplier(target_epsilon, target_delta, eff_var_mult, framework)
            return nm, metric
        elif algorithm in ["All Numeric", "Average"]:
            if self.graph is None:
                raise ValueError("Graph must be set using set_graph() for decentralized algorithms.")
            
            W = build_random_walk_matrix(self.graph)
            n_nodes = self.graph.number_of_nodes()
            _, _, _, M = compute_victim_centric_effective_variances(
                W, self.R_rounds, self.T_local_steps, self.K_gossip, self.graph, 
                self.k_participation, self.b_interval
            )
            
            if algorithm == "All Numeric":
                node_variances = np.max(M, axis=0)
                personalized_noise_multipliers = {}
                for node_id, var_mult in enumerate(node_variances):
                    nm, _ = self._compute_final_multiplier(target_epsilon, target_delta, float(var_mult), framework)
                    personalized_noise_multipliers[node_id] = nm
                return personalized_noise_multipliers
            elif algorithm == "Average":
                attacker_avg_variances = np.zeros(n_nodes)
                for a in range(n_nodes):
                    sum_variance = 0.0
                    for u in range(n_nodes):
                        if a != u:
                            sum_variance += M[a, u]
                    attacker_avg_variances[a] = sum_variance / n_nodes
                eff_var_mult = float(np.max(attacker_avg_variances))
                nm, metric = self._compute_final_multiplier(target_epsilon, target_delta, eff_var_mult, framework)
                return nm, metric
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _compute_final_multiplier(self, target_epsilon, target_delta, eff_var_mult, framework):
        if framework == "RDP":
            noise_mult, metric = calculate_optimal_sigma_rdp(target_epsilon, target_delta, eff_var_mult)
        elif framework == "GDP":
            noise_mult, metric = calculate_optimal_sigma_gdp(target_epsilon, target_delta, eff_var_mult)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        return noise_mult, metric
