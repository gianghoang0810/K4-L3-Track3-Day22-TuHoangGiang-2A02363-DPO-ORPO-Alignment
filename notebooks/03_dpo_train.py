# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — Huấn luyện DPO (notebook chính)
#
# **Công nghệ:** TRL 1.13 `DPOTrainer`, LoRA mới trên mô hình SFT đã gộp, β=0.1, lr=5e-6.
#
# > **Mục tiêu:** huấn luyện adapter DPO, vẽ **riêng** hai đường `rewards/chosen` và
# > `rewards/rejected` trên cả tập huấn luyện và tập held-out, rồi tự chẩn đoán xem margin
# > tăng theo kiểu nào (NB0 §5).
#
# **Ba thay đổi so với lab cũ, đều ảnh hưởng tới kết quả:**
# 1. **Mô hình tham chiếu (reference) = mô hình SFT.** Lab cũ chồng LoRA DPO lên LoRA SFT rồi để TRL tắt
#    adapter để lấy reference, tức là so với *mô hình gốc*. Ở đây mô hình đang học (policy) là
#    `models/sft-merged` + LoRA mới (khởi tạo bằng 0), và
#    `precompute_ref_log_probs=True` chấm mọi cặp trước bước cập nhật đầu tiên.
# 2. **lr = 5e-6.** 5e-7 là mức cho tinh chỉnh toàn bộ; với LoRA và ~100 bước, reward gần như đứng yên.
# 3. **Eval held-out.** Cặp eval không trùng câu hỏi với huấn luyện (NB2).

# %%
import sys
from pathlib import Path

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lab22" / "config.py").exists())
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401
import torch

from lab22 import config as C
from lab22 import data as D
from lab22 import modeling as MD

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."
torch.cuda.reset_peak_memory_stats()  # VRAM đỉnh chỉ tính NB3 (Colab chạy mọi NB trong một kernel)
assert C.SFT_MERGED.exists(), f"Run NB1 first: {C.SFT_MERGED} missing"
assert (C.PREF_DIR / "train.parquet").exists(), "Run NB2 first"
C.ensure_dirs()
print(C.summary())

# %% [markdown]
# ## 1. Mô hình đang học (policy) = SFT đã gộp + LoRA mới
#
# Không có mô hình thứ hai trong VRAM. Log-prob của reference được tính một lần
# trước khi huấn luyện (lúc LoRA còn bằng 0) rồi lưu lại, nên phần VRAM tăng thêm so
# với SFT chủ yếu đến từ việc giữ cả câu chosen lẫn rejected trong một batch.

# %%
model, tokenizer = MD.load_model(C.SFT_MERGED)
model = MD.add_lora(model)

from datasets import Dataset

train_ds = Dataset.from_parquet(str(C.PREF_DIR / "train.parquet"))
eval_ds = Dataset.from_parquet(str(C.PREF_DIR / "eval.parquet"))
print(f"train={len(train_ds)} eval={len(eval_ds)}  columns={train_ds.column_names}")

# %% [markdown]
# ## 2. Huấn luyện

# %%
from trl import DPOTrainer

args = MD.dpo_config(C.ADAPTERS / "dpo-checkpoints")
print(f"loss_type={args.loss_type} beta={args.beta} lr={args.learning_rate} "
      f"precompute_ref={args.precompute_ref_log_probs} max_length={args.max_length}")

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)
result = trainer.train()
final_eval = trainer.evaluate()
print(f"train loss {result.training_loss:.4f} · held-out reward accuracy "
      f"{final_eval.get('eval_rewards/accuracies', float('nan')):.3f}")

# %% [markdown]
# ## 3. Đường cong reward (sản phẩm nộp `03-dpo-reward-curves.png`)
#
# Implicit reward = β·log(π/π_ref) nên bắt đầu ở 0. Đọc đường *chosen*, không chỉ margin:
# - chosen ↑, rejected ↓ → đúng ý đồ;
# - chosen ↓, rejected ↓ nhanh hơn → **likelihood displacement** (Razin et al. 2024).
#
# Loss ở bước đầu phải gần 0.693 (NB0 §3). Nếu không, reference đang sai.

# %%
train_hist = MD.reward_history(trainer.state.log_history)
eval_hist = MD.reward_history(trainer.state.log_history, prefix="eval_")
title = f"DPO · {C.COMPUTE_TIER} · β={C.DPO_BETA} · lr={C.DPO_LR} · {'+'.join(C.DPO_LOSS)}"
MD.plot_rewards(train_hist, eval_hist, title, C.SCREENSHOTS / "03-dpo-reward-curves.png")

first_loss = next((r["loss"] for r in trainer.state.log_history if "loss" in r), None)
print(f"first logged loss: {first_loss}")

# %%
label, message = MD.diagnose(eval_hist if len(eval_hist) >= 2 else train_hist)
print(f"[{label}] {message}")

# %% [markdown]
# ## 4. Lưu adapter + metrics

# %%
import json

trainer.model.save_pretrained(str(C.DPO_ADAPTER))
tokenizer.save_pretrained(str(C.DPO_ADAPTER))
D.save_split_fingerprint(C.PREF_DIR, C.DPO_ADAPTER)  # NB4 refuses a held-out set this adapter saw


def last(df, col):
    return float(df[col].iloc[-1]) if not df.empty and col in df else None


metrics = {
    "compute_tier": C.COMPUTE_TIER,
    "base_model": C.BASE_MODEL,
    "reference": "models/sft-merged (precomputed)",
    "pref_dataset": C.PREF_DATASET,
    "beta": C.DPO_BETA,
    "lr": C.DPO_LR,
    "loss_type": C.DPO_LOSS,
    "epochs": C.DPO_EPOCHS,
    "final_train_loss": float(result.training_loss),
    "first_logged_loss": first_loss,
    "end_chosen_reward": last(train_hist, "rewards/chosen"),
    "end_rejected_reward": last(train_hist, "rewards/rejected"),
    "end_reward_gap": last(train_hist, "rewards/margins"),
    "eval_chosen_reward": final_eval.get("eval_rewards/chosen"),
    "eval_rejected_reward": final_eval.get("eval_rewards/rejected"),
    "eval_reward_gap": final_eval.get("eval_rewards/margins"),
    "eval_reward_accuracy": final_eval.get("eval_rewards/accuracies"),
    "diagnosis": label,
    "gpu": torch.cuda.get_device_name(0),
    "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    "train_runtime_s": result.metrics.get("train_runtime"),
    "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
}
(C.DPO_ADAPTER / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# %% [markdown]
# ## 5. Ghi chú vibe-coding: quét β (+6 độ chặt chẽ)
#
# `make beta-sweep` chạy `scripts/train_dpo.py` với β ∈ {0.05, 0.1, 0.5} và lưu vào
# `adapters/dpo-b*/`. `python scripts/eval_judge.py --plot-sweep` vẽ β theo margin held-out.
#
# **Đoán trước khi xem:** β lớn giữ mô hình đang học (policy) gần reference hơn. Margin tính bằng β·log-ratio
# sẽ thay đổi thế nào? Còn độ chính xác reward?
#
# **Tiếp theo:** NB3b (so sánh RPO / SimPO-norm / LD-DPO / ORPO) hoặc NB4 (đánh giá).
