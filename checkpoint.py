import torch
from peft.utils import get_peft_model_state_dict

def save_pruned_checkpoint(model, path: str, retain_ratio: float = 0.15):
    # Prune low-importance ranks before saving
    for name, module in model.named_modules():
        if hasattr(module, "importance"):
            if module.importance is not None and module.r > 8:
                # Sort by importance (descending)
                sorted_imp, indices = torch.sort(module.importance, descending=True)
                keep_num = max(8, int(module.r * retain_ratio))
                keep_indices = indices[:keep_num]

                # Prune lora_A (rows = ranks)
                module.lora_A[module.active_adapter].weight.data = torch.nn.Parameter(
                    module.lora_A[module.active_adapter].weight.data[keep_indices]
                )
                # Prune lora_B (columns = ranks)
                module.lora_B[module.active_adapter].weight.data = torch.nn.Parameter(
                    module.lora_B[module.active_adapter].weight.data[:, keep_indices]
                )
                module.r = keep_num
                module.importance = torch.nn.Parameter(torch.zeros(keep_num, device=module.weight.device), requires_grad=False)

    state_dict = get_peft_model_state_dict(model)
    torch.save(state_dict, f"{path}/adapter_model.bin")
    model.config.save_pretrained(path)
    print(f"Pruned checkpoint saved to {path} (r reduced to ~{retain_ratio*100:.1f}%)")
