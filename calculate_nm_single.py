import sys
sys.path.insert(0, ".")

from opacus.accountants.utils import get_noise_multiplier

# ==================== 参数（根据需要修改）====================
N_SAMPLES = 67349        # SST2 训练集样本数
BATCH_SIZE = 1000         # 逻辑 batch size
EPOCHS = 3              # 训练轮数
EPSILON = 3.0            # 目标隐私预算 ε
DELTA = 1e-5             # 目标 δ
# ============================================================

# python calculate_nm_single.py

sample_rate = BATCH_SIZE / N_SAMPLES
total_steps = EPOCHS * (N_SAMPLES // BATCH_SIZE)

print(f"sample_rate = {sample_rate:.6f} ({BATCH_SIZE}/{N_SAMPLES})")
print(f"steps_per_epoch = {N_SAMPLES // BATCH_SIZE}")
print(f"total_steps = {total_steps} ({EPOCHS} epochs x {N_SAMPLES // BATCH_SIZE} steps)")
print(f"target: ε={EPSILON}, δ={DELTA}")

noise_multiplier = get_noise_multiplier(
    target_epsilon=EPSILON,
    target_delta=DELTA,
    sample_rate=sample_rate,
    steps=total_steps,
)

print(f"\nnoise_multiplier = {noise_multiplier:.4f}")
print(f"→ 训练时传入: privacy_engine.make_private(..., noise_multiplier={noise_multiplier:.4f}, ...)")
