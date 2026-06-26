import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import networkx as nx
import copy
import numpy as np
import os
import sys
import json
from datetime import datetime
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from pndp_calculator import PNDPAccountant

DATASET = "CIFAR100"
GRAPH = "florentine_families"
BATCH_SIZE = 64
T_LOCAL_STEPS = 50
R_ROUNDS = 10
K_GOSSIP = 1
EPSILON = 3.0
DELTA = 1e-5
CLIP_NORM = 1.0
LR = 0.01
GPU = 3
FRAMEWORK = "GDP"

# python train_example.py

def get_graph(name):
    graph_fns = {
        "florentine_families": nx.florentine_families_graph,
    }
    if name not in graph_fns:
        raise ValueError(f"Unknown graph: {name}")
    return graph_fns[name]()


def get_dataset_config(name):
    configs = {
        "CIFAR100": {
            "num_classes": 100,
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
        },
        "CIFAR10": {
            "num_classes": 10,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2023, 0.1994, 0.2010),
        },
    }
    if name not in configs:
        raise ValueError(f"Unknown dataset: {name}")
    return configs[name]


def load_dataset(name, root, train, transform):
    dataset_fns = {
        "CIFAR100": torchvision.datasets.CIFAR100,
        "CIFAR10": torchvision.datasets.CIFAR10,
    }
    if name not in dataset_fns:
        raise ValueError(f"Unknown dataset: {name}")
    return dataset_fns[name](root=root, train=train, download=True, transform=transform)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_resnet_model(num_classes=100):
    model = torchvision.models.resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


class DecentralizedNode:
    def __init__(self, node_id, data_indices, dataset, noise_multiplier, num_classes=100):
        self.node_id = node_id
        self.model = ModuleValidator.fix(get_resnet_model(num_classes)).to(DEVICE)
        self.optimizer = optim.SGD(self.model.parameters(), lr=LR, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

        local_subset = Subset(dataset, data_indices)
        self.dataloader = DataLoader(local_subset, batch_size=BATCH_SIZE, shuffle=True)

        self.noise_multiplier = noise_multiplier

        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.dataloader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=CLIP_NORM,
        )
        self.data_iterator = iter(self.dataloader)

    def get_next_batch(self):
        try:
            inputs, targets = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.dataloader)
            inputs, targets = next(self.data_iterator)
        return inputs.to(DEVICE), targets.to(DEVICE)

    def local_update(self, local_steps):
        self.model.train()
        for step in range(local_steps):
            inputs, targets = self.get_next_batch()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
        return loss.item()


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100.0 * correct / total


def daemonize(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    pid = os.fork()
    if pid > 0:
        return pid
    os.setsid()
    pid = os.fork()
    if pid > 0:
        sys.exit(0)
    log_file = open(log_path, "w")
    sys.stdout = log_file
    sys.stderr = log_file


def main():
    foreground = "--foreground" in sys.argv

    if GPU is not None:
        torch.cuda.set_device(GPU)

    G = get_graph(GRAPH)
    nodes_list = list(G.nodes())
    num_nodes = len(nodes_list)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        "exps",
        f"{timestamp}_{DATASET}_{GRAPH}_e{EPSILON}_d{DELTA}_R{R_ROUNDS}_N{num_nodes}_K{K_GOSSIP}_T{T_LOCAL_STEPS}_B{BATCH_SIZE}_LR{LR}_CN{CLIP_NORM}_F{FRAMEWORK}"
    )
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "train.log")

    if not foreground:
        print(f"tail -f {log_path}")
        pid = daemonize(log_path)
        print(f"kill {pid}")
        sys.exit(0)
    else:
        print(f"[Foreground] Log: {log_path}")

    print(f"Using device: {DEVICE}")

    cfg = get_dataset_config(DATASET)

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    train_dataset = load_dataset(DATASET, root='./data', train=True, transform=transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    test_dataset = load_dataset(DATASET, root='./data', train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    samples_per_node = len(train_dataset) // num_nodes
    acc = PNDPAccountant(
        N_samples=samples_per_node,
        batch_size=BATCH_SIZE,
        T_local_steps=T_LOCAL_STEPS,
        R_rounds=R_ROUNDS,
        K_gossip=K_GOSSIP,
    )
    acc.set_graph(G)

    nm, m = acc.get_noise_multiplier(EPSILON, DELTA, algorithm="Average", framework=FRAMEWORK)
    print(f"[Privacy] Calculated Noise Multiplier: {nm:.4f}")

    params = {
        "timestamp": timestamp,
        "device": str(DEVICE),
        "DATASET": DATASET,
        "GRAPH": GRAPH,
        "num_nodes": num_nodes,
        "num_classes": cfg["num_classes"],
        "N_SAMPLES_TOTAL": len(train_dataset),
        "BATCH_SIZE": BATCH_SIZE,
        "T_LOCAL_STEPS": T_LOCAL_STEPS,
        "R_ROUNDS": R_ROUNDS,
        "K_GOSSIP": K_GOSSIP,
        "EPSILON": EPSILON,
        "DELTA": DELTA,
        "FRAMEWORK": FRAMEWORK,
        "CLIP_NORM": CLIP_NORM,
        "LR": LR,
        "noise_multiplier": float(nm),
        "samples_per_node": samples_per_node,
    }
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Setup] Output directory: {out_dir}")

    all_indices = np.random.permutation(len(train_dataset))
    node_objects = {}

    for i, node_name in enumerate(nodes_list):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node
        indices = all_indices[start_idx:end_idx]

        node_objects[node_name] = DecentralizedNode(
            node_id=node_name,
            data_indices=indices,
            dataset=train_dataset,
            noise_multiplier=nm,
            num_classes=cfg["num_classes"],
        )

    print("Starting Decentralized Training...")
    round_mean_accs = []
    for round_idx in tqdm(range(R_ROUNDS), desc="Rounds"):
        print(f"\n--- Round {round_idx + 1}/{R_ROUNDS} ---")

        round_losses = []
        for node_name, node_obj in tqdm(node_objects.items(), desc=f"Round {round_idx+1} Train", leave=False):
            loss = node_obj.local_update(T_LOCAL_STEPS)
            round_losses.append(loss)
        print(f"Average Local Loss: {np.mean(round_losses):.4f}")

        weights_snapshot = {}
        for node_name, node_obj in node_objects.items():
            weights_snapshot[node_name] = {k: v.clone() for k, v in node_obj.model.state_dict().items()}

        for node_name in nodes_list:
            neighbors = list(G.neighbors(node_name))
            aggregate_set = neighbors + [node_name]
            num_participants = len(aggregate_set)

            new_state_dict = {k: torch.zeros_like(v) for k, v in weights_snapshot[node_name].items()}

            for neighbor_name in aggregate_set:
                neighbor_weights = weights_snapshot[neighbor_name]
                for k in new_state_dict.keys():
                    new_state_dict[k] += neighbor_weights[k]

            for k in new_state_dict.keys():
                new_state_dict[k] = new_state_dict[k] / num_participants

            node_objects[node_name].model.load_state_dict(new_state_dict)

        accuracies = []
        for node_name, node_obj in tqdm(node_objects.items(), desc=f"Round {round_idx+1} Eval", leave=False):
            acc = evaluate_model(node_obj.model, test_loader, DEVICE)
            accuracies.append(acc)
        mean_acc = np.mean(accuracies)
        max_acc = np.max(accuracies)
        min_acc = np.min(accuracies)
        round_mean_accs.append(mean_acc)
        print(f"[Eval] Avg Acc: {mean_acc:.2f}% | Min: {min_acc:.2f}% | Max: {max_acc:.2f}% | Consensus Gap: {max_acc - min_acc:.2f}%")

        csv_path = os.path.join(out_dir, "accuracy.csv")
        if round_idx == 0:
            with open(csv_path, "w") as f:
                f.write("round,mean_acc,max_acc,min_acc,consensus_gap,avg_loss\n")
        with open(csv_path, "a") as f:
            f.write(f"{round_idx + 1},{mean_acc:.4f},{max_acc:.4f},{min_acc:.4f},{max_acc - min_acc:.4f},{np.mean(round_losses):.4f}\n")
            f.flush()

    print("\nTraining Completed.")
    print(f"\n[Summary] Round Average Accuracies: {[f'{a:.2f}%' for a in round_mean_accs]}")
    print(f"[Summary] Final vs Initial: {round_mean_accs[-1]:.2f}% vs {round_mean_accs[0]:.2f}% (Δ={round_mean_accs[-1] - round_mean_accs[0]:+.2f}%)")


if __name__ == "__main__":
    main()
