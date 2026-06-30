import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import networkx as nx
import copy
import numpy as np
import sys
import json
from datetime import datetime
from tqdm import tqdm
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

from pndp_calculator import PNDPAccountant

DATASET = "SST2"
GRAPH = "florentine_families"
BATCH_SIZE = 16
MAX_PHYSICAL_BATCH_SIZE = 16
T_LOCAL_STEPS = 10
R_ROUNDS = 50
K_GOSSIP = 1
EPSILON = 8.0
DELTA = 1e-5
CLIP_NORM = 1
LR = 1e-3 
GPU = 1
FRAMEWORK = "GDP"   # "GDP" or "RDP"
ALGORITHM = "Average"   # "Average" "LDP-per-round" "All Numeric"
ENABLE_PRIVACY = False   # 控制是否开启差分隐私训练
SET_NM = None

# python train_example.py

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_graph(name):
    graph_fns = {
        "florentine_families": nx.florentine_families_graph,
    }
    if name not in graph_fns:
        raise ValueError(f"Unknown graph: {name}")
    return graph_fns[name]()


def get_roberta_lora_model(num_classes=2):
    model = AutoModelForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=num_classes
    )
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, peft_config)
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False
    return model


class DecentralizedNode:
    def __init__(self, node_id, data_indices, dataset, noise_multiplier, num_classes=2, enable_privacy=True, init_state_dict=None):
        self.node_id = node_id
        self.enable_privacy = enable_privacy
        self.model = ModuleValidator.fix(get_roberta_lora_model(num_classes))
        if init_state_dict is not None:
            self.model.load_state_dict(init_state_dict, strict=False)
        self.model = self.model.to(DEVICE)
        self.optimizer = AdamW(self.model.parameters(), lr=LR, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss()

        local_subset = Subset(dataset, data_indices)
        self.dataloader = DataLoader(local_subset, batch_size=BATCH_SIZE, shuffle=True)

        self.noise_multiplier = noise_multiplier

        if enable_privacy:
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.dataloader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=CLIP_NORM,
            )

    def local_update(self, local_steps, desc=None):
        self.model.train()
        accumulation_steps = BATCH_SIZE // MAX_PHYSICAL_BATCH_SIZE
        physical_steps_needed = local_steps * accumulation_steps
        physical_steps_done = 0
        step_pbar = tqdm(total=physical_steps_needed, desc=desc, leave=False)

        if self.enable_privacy:
            while physical_steps_done < physical_steps_needed:
                with BatchMemoryManager(
                    data_loader=self.dataloader,
                    max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
                    optimizer=self.optimizer
                ) as memory_safe_data_loader:
                    for batch in memory_safe_data_loader:
                        batch = {k: v.to(DEVICE) for k, v in batch.items()}
                        self.optimizer.zero_grad()
                        outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                        loss = self.criterion(outputs.logits, batch["labels"])
                        loss.backward()
                        self.optimizer.step()

                        physical_steps_done += 1
                        step_pbar.update(1)
                        step_pbar.set_postfix(loss=f"{loss.item():.4f}")
                        if physical_steps_done >= physical_steps_needed:
                            break
        else:
            data_iterator = iter(self.dataloader)
            self.optimizer.zero_grad()

            while physical_steps_done < physical_steps_needed:
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.dataloader)
                    batch = next(data_iterator)

                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

                loss = self.criterion(outputs.logits, batch["labels"])
                scaled_loss = loss / accumulation_steps
                scaled_loss.backward()

                physical_steps_done += 1

                if physical_steps_done % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                step_pbar.update(1)
                step_pbar.set_postfix(loss=f"{loss.item():.4f}")

        step_pbar.close()
        return loss.item()


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def daemonize(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # 第一次 fork
    pid = os.fork()
    if pid > 0:
        # 父进程直接退出
        sys.exit(0)
        
    os.setsid() # 脱离控制终端
    
    # 第二次 fork
    pid = os.fork()
    if pid > 0:
        # 第一个子进程：打印真正的守护进程 PID（即第二个子进程）并退出
        print(f"kill {pid}")
        sys.exit(0)
        
    # 第二个子进程（真正的守护进程）继续往下执行
    # 刷新缓冲区，防止重定向前的残留内容被重复打印
    sys.stdout.flush()
    sys.stderr.flush()
    
    # 将标准输出和标准错误重定向到日志文件（推荐追加模式 "a" 结合 flush 或 "w"）
    log_file = open(log_path, "w")
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

    pid_file = os.path.join(os.path.dirname(log_path), "daemon.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    # 守护进程返回，准备执行接下来的模型训练代码


def main():
    if GPU is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU)
    foreground = "--foreground" in sys.argv

    if os.name == 'nt':
        print("[Warning] Windows system detected, forcing foreground mode.")
        foreground = True 

    G = get_graph(GRAPH)
    nodes_list = list(G.nodes())
    num_nodes = len(nodes_list)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if ENABLE_PRIVACY:
        dp_str = f"DP_True_e{EPSILON}_d{DELTA}_CN{CLIP_NORM}"
    else:
        dp_str = "DP_False"
    out_dir = os.path.join(
        "exps",
        f"{timestamp}_{DATASET}_{GRAPH}_{dp_str}_R{R_ROUNDS}_N{num_nodes}_K{K_GOSSIP}_T{T_LOCAL_STEPS}_B{BATCH_SIZE}_LR{LR}_F{FRAMEWORK}"
    )
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "train.log")

    if not foreground:
        print(f"tail -f {log_path}")
        daemonize(log_path) 
    else:
        print(f"[Foreground] Log: {log_path}")

    set_seed(42)
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    hf_dataset = load_dataset("stanfordnlp/sst2")

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = hf_dataset.map(tokenize_fn, batched=True)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["validation"]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    samples_per_node = len(train_dataset) // num_nodes
    if ENABLE_PRIVACY:
        acc = PNDPAccountant(
            N_samples=samples_per_node,
            batch_size=BATCH_SIZE,
            T_local_steps=T_LOCAL_STEPS,
            R_rounds=R_ROUNDS,
            K_gossip=K_GOSSIP,
        )
        acc.set_graph(G)
        if SET_NM is None:
            nm, m = acc.get_noise_multiplier(EPSILON, DELTA, algorithm=ALGORITHM, framework=FRAMEWORK)
            print(f"[Privacy] Calculated Noise Multiplier: {nm:.4f}")
        else:
            nm = SET_NM
            print(f"[Privacy] Using Set Noise Multiplier: {nm:.4f}")
    else:
        nm = 0.0

    params = {
        "timestamp": timestamp,
        "device": str(DEVICE),
        "DATASET": DATASET,
        "GRAPH": GRAPH,
        "num_nodes": num_nodes,
        "num_classes": 2,
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
        "ENABLE_PRIVACY": ENABLE_PRIVACY,
    }
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"[Setup] Output directory: {out_dir}")

    all_indices = np.random.permutation(len(train_dataset))
    node_objects = {}

    print("Initializing global model for consistent starting weights...")
    global_model = get_roberta_lora_model(num_classes=2)
    global_state_dict = {k: v.cpu().clone() for k, v in global_model.state_dict().items()}
    del global_model

    for i, node_name in enumerate(nodes_list):
        start_idx = i * samples_per_node
        end_idx = (i + 1) * samples_per_node
        indices = all_indices[start_idx:end_idx]

        node_objects[node_name] = DecentralizedNode(
            node_id=node_name,
            data_indices=indices,
            dataset=train_dataset,
            noise_multiplier=nm,
            num_classes=2,
            enable_privacy=ENABLE_PRIVACY,
            init_state_dict=global_state_dict,
        )

    print("Starting Decentralized Training...")
    round_mean_accs = []
    for round_idx in tqdm(range(R_ROUNDS), desc="Rounds"):
        print(f"\n--- Round {round_idx + 1}/{R_ROUNDS} ---")

        round_losses = []
        pbar = tqdm(enumerate(node_objects.items()), desc=f"Round {round_idx+1} Train", leave=False)
        for i, (node_name, node_obj) in pbar:
            pbar.set_postfix(node=i)
            loss = node_obj.local_update(T_LOCAL_STEPS, desc=f"  Node {i}")
            round_losses.append(loss)
            torch.cuda.empty_cache()
        print(f"Average Local Loss: {np.mean(round_losses):.4f}")

        weights_snapshot = {}
        for node_name, node_obj in node_objects.items():
            weights_snapshot[node_name] = {
                name: param.detach().cpu().clone()
                for name, param in node_obj.model.named_parameters()
                if param.requires_grad
            }

        degrees = {n: G.degree(n) for n in G.nodes()}

        for node_name in nodes_list:
            neighbors = list(G.neighbors(node_name))
            aggregate_set = neighbors + [node_name]

            new_state_dict = {}
            total_weight = 0.0

            for neighbor_name in aggregate_set:
                weight = 1.0 / (max(degrees[node_name], degrees[neighbor_name]) + 1)
                total_weight += weight
                neighbor_weights = weights_snapshot[neighbor_name]
                for k, v in neighbor_weights.items():
                    if k not in new_state_dict:
                        new_state_dict[k] = weight * v.float()
                    else:
                        new_state_dict[k] += weight * v.float()

            for k in new_state_dict:
                new_state_dict[k] = (new_state_dict[k] / total_weight).to(DEVICE)

            node_objects[node_name].model.load_state_dict(new_state_dict, strict=False)

        accuracies = []
        pbar = tqdm(enumerate(node_objects.items()), desc=f"Round {round_idx+1} Eval", leave=False)
        for i, (node_name, node_obj) in pbar:
            pbar.set_postfix(node=i)
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
