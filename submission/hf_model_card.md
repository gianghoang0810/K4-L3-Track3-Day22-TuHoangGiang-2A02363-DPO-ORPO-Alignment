---
base_model: unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit
library_name: peft
license: cc-by-nc-4.0
language: [vi]
tags: [dpo, trl, unsloth, lora, experimental]
datasets: [saillab/alpaca-vietnamese-cleaned, sailor2/sea-ultrafeedback-onpolicy]
---
# Lab 22 — Qwen3-4B SFT + DPO LoRA tiếng Việt (v0, thử nghiệm)

Adapter LoRA huấn luyện trong lab Day 22 (VinUni AICB, Track 3) trên Colab T4. Bản học tập, **không** dùng cho môi trường thật.

## Thành phần
- Thư mục gốc: adapter LoRA **DPO** (r=16, alpha=32, q/k/v/o/gate/up/down_proj), huấn luyện trên mô hình SFT đã gộp.
- `sft-mini/`: adapter LoRA **SFT** gắn lên mô hình gốc.
- Mô hình SFT đã gộp (16-bit, ~8 GB) **không** được đẩy lên. `adapter_config.json` của adapter DPO trỏ tới đường dẫn cục bộ `/content/lab22/models/sft-merged`. Cách dựng lại: nạp mô hình gốc, gắn `sft-mini/`, gộp (`merge_and_unload`), rồi gắn adapter DPO ở thư mục gốc.

## Huấn luyện
| Mục | Giá trị |
|---|---|
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch, lr 2e-4 |
| DPO | `sailor2/sea-ultrafeedback-onpolicy` (tiếng Việt), 800 cặp train / 100 held-out tách theo câu hỏi; β 0.1, lr 5e-06, 1.0 epoch, loss sigmoid; reference = mô hình SFT (log-prob tính trước) |
| Phần cứng | Tesla T4, 35.2 phút, VRAM đỉnh 8.04 GB |

## Kết quả
| Chỉ số (held-out) | Giá trị |
|---|---|
| Loss bước đầu | 0.6933 (≈ log 2) |
| Reward accuracy | 0.68 |
| Margin (chosen − rejected) | 0.0861 |
| Chẩn đoán tự động | INTENDED (chosen và rejected cùng tăng, chosen tăng nhanh hơn) |

Win rate SFT+DPO so với SFT trên 50 câu held-out:

| Giám khảo | DPO / SFT / hoà | Win rate (CI 95%) |
|---|---|---|
| RM `Skywork/Skywork-Reward-V2-Llama-3.2-3B` | 8 / 9 / 33 | 0.49 [0.41, 0.57] |
| `openai/gpt-4.1-mini` (chấm 2 thứ tự A/B) | 6 / 1 / 43 | 0.55 [0.50, 0.60] |

Khoảng tin cậy chứa hoặc chạm 0.5: **chưa đủ bằng chứng** DPO tốt hơn SFT.

## Hạn chế
- 40/58 câu trả lời của SFT và DPO trùng nhau từng ký tự: DPO dịch chuyển mô hình rất ít.
- Mọi câu trả lời sinh ra đều mở đầu bằng token `<tool_call>` hoặc `</tool_call>`; nguyên nhân chưa xác định.
- Dữ liệu SFT có giấy phép CC BY-NC nên adapter chỉ dùng cho học tập và nghiên cứu.

## Nguồn
Mã và báo cáo lab: https://github.com/gianghoang0810/K4-L3-Track3-Day22-TuHoangGiang-2A02363-DPO-ORPO-Alignment
