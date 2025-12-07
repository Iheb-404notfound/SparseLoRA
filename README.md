# SkipLoRA: Contextual Gradient Zeroing for Accelerated LoRA Fine-Tuning

Welcome to the official GitHub repository for **SkipLoRA**, a novel parameter-efficient fine-tuning method that accelerates the backward pass by dynamically skipping redundant gradient computations. Built on PyTorch, SkipLoRA introduces **Contextual Gradient Zeroing (CGZ)** to reduce FLOPs during training without sacrificing model quality. This repo provides a minimal, standalone implementation compatible with Hugging Face Transformers.

Inspired by efficient LoRA variants like QLoRA, but focused on compute savings rather than memory quantization. Ideal for resource-constrained environments where backward pass dominates training time.

## Quick Start

1. Clone the repo:
   ```
   git clone https://github.com/NanoTensor/skiplora.git
   cd skiptora
   ```

2. Install dependencies:
   ```
   pip install -e .
   ```

3. Fine-tune a model (example with GPT-2):
   ```
   python examples/train.py --model_name gpt2 --dataset glue --task mrpc --epochs 3 --threshold 0.05
   ```

## Key Features
- **Dynamic Impact Assessment**: Computes layer-wise efficiency metric \(\mathcal{M}_l\) in forward pass.
- **Adaptive Thresholding**: Configurable \(\tau_l\) for skipping inactive layers.
- **Gradient Zeroing**: Bypasses backward propagation for low-impact adapters, saving ~20-40% FLOPs (empirical on Llama-7B).
- **Seamless Integration**: Hooks into existing LoRA setups; orthogonal to mixed precision.
- **Tested on**: PyTorch 2.1+, Transformers 4.35+.

## Methodology
SkipLoRA operates in two phases:

1. **Forward Pass**: For each LoRA-adapted layer, compute \(\mathcal{M}_l = \frac{||\Delta h||_2}{||h||_2}\), where \(\Delta h\) is the adapter delta.
2. **Backward Pass**: If \(\mathcal{M}_l < \tau_l\), detach outputs and zero gradients for \(\mathbf{A}, \mathbf{B}\).

This targets redundancy in near-zero gradients, complementing techniques like FP16.

## Results
- **Speedup**: 1.8x faster backward on Alpaca dataset (Llama-7B, r=16).
- **Memory**: No additional overhead; uses ~1MB extra for metrics.

See `experiments/` (coming soon) for benchmarks.

## Citation
```
@misc{skiplora2025,
  title={SkipLoRA: Accelerating PEFT via Contextual Gradient Zeroing},
  author={Iheb Gafsi, Alex Kuchynka},
  year={2025},
  url={https://github.com/NanoTensor/skiplora}
}
```

## License
MIT. See LICENSE for details.

---

## Repository File Structure
```
.
├── README.md                 # This file
├── LICENSE                   # MIT License
├── setup.py                  # Package installation
├── requirements.txt          # Dependencies
├── skiptora/                 # Core library
│   ├── __init__.py
│   ├── layer.py              # SkipLoRA module definition
│   └── hooks.py              # Forward/backward hooks
├── examples/                 # Usage scripts
│   └── train.py              # Fine-tuning example
└── tests/                    # Unit tests
    └── test_layer.py         # Basic tests
```
