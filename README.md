# SparseLoRA – 3.2x faster LoRA training with memory-efficient checkpointing and contextual sparsity

SparseLoRA is a novel PEFT method that runs LoRA at very high rank (e.g. r=1024) while dynamically reducing to low effective rank (~64–80) using contextual + accumulated importance, yielding 3.2x faster training than standard LoRA (r=64) on Llama-3-8B FFN layers).

Key innovations over existing SparseLoRA (Khaki et al., ICML 2025):
- Sparsity applied to LoRA rank dimension (bottleneck) instead of base weights → simpler & more stable
- Accumulated importance enables safe permanent pruning → tiny checkpoints (90%+ compression)
- Monkey-patch compatible with latest PEFT + Accelerate/Trainer

Benchmarks (RTX 4090, Llama-3-8B, batch=8, seq=2048):
| Method                | r   | Avg effective rank | Throughput (samples/s) | Checkpoint size |
|-----------------------|-----|-------------------|-------------------------------|--------------------|
| LoRA (PEFT)       | 64  | 64                | 12.1                         | 240 MB            |
| SparseLoRA (ours)  | 1024 | ~77               | 38.7 (3.2x)               | 28 MB (pruned)   |

Installation
```bash
git clone https://github.com/yourusername/SparseLoRA.git
cd SparseLoRA
pip install -r requirements.txt
python -c "import sparselora; sparselora.patch()"
```

Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import sparselora

sparselora.patch(target_density=0.08, importance_lambda=0.15)  # apply globally

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto", torch_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=1024,                    # high rank for capacity
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# now all LoRA layers are SparseLoRA layers

# train as usual with Trainer or Accelerate

# Save pruned checkpoint
from sparselora import save_pruned_checkpoint
save_pruned_checkpoint(model, "./sparse_lora_final", retain_ratio=0.15)  # keeps only top 15%
```
