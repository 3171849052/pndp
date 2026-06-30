import sys
sys.path.insert(0, ".")

import networkx as nx
from pndp_calculator import PNDPAccountant

# python example.py
# ==================== 统一参数 ====================
N_samples = 4490
Batch_size = 16
T_local_steps = 10
R_rounds = 50
K_gossip = 1
epsilon = 8.0
delta = 1e-5
# =================================================

acc = PNDPAccountant(
    N_samples=N_samples,
    batch_size=Batch_size,
    T_local_steps=T_local_steps,
    R_rounds=R_rounds,
    K_gossip=K_gossip,
)
print(f"b_interval={acc.b_interval}, total_steps={acc.total_steps}, k_participation={acc.k_participation}")

# ===== LDP-per-round（不需要图）=====
nm, m = acc.get_noise_multiplier(epsilon, delta, algorithm="LDP-per-round", framework="RDP")
print(f"LDP-per-round (RDP): noise_multiplier={nm:.4f}, metric={m:.4f}")

nm, m = acc.get_noise_multiplier(epsilon, delta, algorithm="LDP-per-round", framework="GDP")
print(f"LDP-per-round (GDP): noise_multiplier={nm:.4f}, metric={m:.4f}")

# ===== 需要图的算法 =====
G = nx.florentine_families_graph()
acc.set_graph(G)

nm, m = acc.get_noise_multiplier(epsilon, delta, algorithm="Average", framework="RDP")
print(f"Average       (RDP): noise_multiplier={nm:.4f}, metric={m:.4f}")

nm, m = acc.get_noise_multiplier(epsilon, delta, algorithm="Average", framework="GDP")
print(f"Average       (GDP): noise_multiplier={nm:.4f}, metric={m:.4f}")

all_nm = acc.get_noise_multiplier(epsilon, delta, algorithm="All Numeric", framework="RDP")
print(f"All Numeric (RDP) — per-node noise_multipliers:")
for node_id, nm in sorted(all_nm.items()):
    print(f"  Node {node_id}: {nm:.4f}")

all_nm = acc.get_noise_multiplier(epsilon, delta, algorithm="All Numeric", framework="GDP")
print(f"All Numeric (GDP) — per-node noise_multipliers:")
for node_id, nm in sorted(all_nm.items()):
    print(f"  Node {node_id}: {nm:.4f}")
