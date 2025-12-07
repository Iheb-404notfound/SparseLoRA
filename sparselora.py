import torch
from torch import nn
from peft.tuners.lora.layer import Linear as PeftLoraLinear
from peft.tuners.lora.layer import LoraLayer

class SparseLoraLayer(LoraLayer):
    def __init__(self, base_layer, **kwargs):
        super().__init__(base_layer, **kwargs)
        r = self.r
        device = self.weight.device
        self.importance = nn.Parameter(torch.zeros(r, device=device), requires_grad=False)
        self.target_density = getattr(kwargs, "target_density", 0.2)
        self.importance_lambda = getattr(kwargs, "importance_lambda", 0.1)
        self.accum_steps = 0

    def forward(self, x):
        if self.active_adapter not in self.lora_A:
            return self.base_layer(x)

        previous_dtype = x.dtype

        # standard LoRA dropout + down projection
        hidden = self.lora_dropout[self.active_adapter](x)
        hidden = self.lora_A[self.active_adapter](hidden)

        if self.training:
            # Contextual + accumulated importance
            score = hidden.abs().mean(dim=[0,1])  # (r,)
            score += self.importance_lambda * self.importance

            k = max(1, int(self.r * self.target_density))
            if k < self.r:
                topk_vals, active_indices = torch.topk(score, k)
                hidden = hidden[..., active_indices]  # (b, s, k)
                B_active = self.lora_B[self.active_adapter].weight[:, active_indices]  # (out, k)
                hidden = hidden @ B_active.t() * self.scaling[self.active_adapter]
                # Accumulate importance (simple EMA)
                with torch.no_grad():
                    contrib = topk_vals / (topk_vals.sum() + 1e-8)
                    self.importance.scatter_add_(0, active_indices, contrib)
                self.accum_steps += 1
            else:
                hidden = self.lora_B[self.active_adapter](hidden) * self.scaling[self.active_adapter]
                with torch.no_grad():
                    self.importance.add_(0.01 * score)
        else:
            # inference uses all ranks (or pruned version)
            hidden = self.lora_B[self.active_adapter](hidden) * self.scaling[self.active_adapter]

        result = self.base_layer(x) + hidden.to(previous_dtype)

        return result

def patch(target_density=0.2, importance_lambda=0.1):
    from peft.tuners.lora.layer import Linear as PeftLoraLinear
    global SparseLoraLayer
    SparseLoraLayer.target_density = target_density
    SparseLoraLayer.importance_lambda = importance_lambda
    PeftLoraLinear = type("SparseLoraLinear", (PeftLoraLinear, SparseLoraLayer), {})
    # Monkey-patch
    import peft.tuners.lora.layer as layer_module
    layer_module.Linear = PeftLoraLinear
    layer_module.LoraLayer = SparseLoraLayer  # also patch base class

print("SparseLoRA patch applied — use LoraConfig(r=1024, ...) as normal")
