import math
import os
import re
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Directly load Corridor's main model class resposible for keying.
from CorridorKeyModule.core.model_transformer import GreenFormer

# ---------------------------------------------------------
# PATCH: Fix timm's in-place shape operations for Nuke
# ---------------------------------------------------------
def patch_timm_for_nuke():
    """
    Nuke's C++ LibTorch interpreter crashes on in-place shape operations (aten::mul_)
    on dynamically traced shape variables. The timm Hiera models use operations like
    `B *= self.q_stride` internally. This dynamically rewrites those classes to use
    out-of-place math (e.g. `B = B * self.q_stride`) before tracing.
    """
    import timm.models.hiera
    
    def replacer(match):
        indent = match.group(1)
        var = match.group(2)
        op = match.group(3)
        expr = match.group(4)
        return f"{indent}{var} = {var} {op} ({expr})"
        
    for cls_name in ['Unroll', 'Reroll', 'MaskUnitAttention']:
        if not hasattr(timm.models.hiera, cls_name):
            continue
            
        cls = getattr(timm.models.hiera, cls_name)
        try:
            src = inspect.getsource(cls)
        except TypeError:
            continue
            
        # Match operations like `var *= expr`, `var += expr`, etc.
        patched_src = re.sub(
            r'^(\s*)([a-zA-Z0-9_]+)\s*([\*\+\-\/]|//)\=\s*(.+)$', 
            replacer, 
            src, 
            flags=re.MULTILINE
        )
        
        # Execute the patched code back into the timm namespace
        exec(patched_src, timm.models.hiera.__dict__)


# ---------------------------------------------------------
# 1. The Core Adapter
# ---------------------------------------------------------
class CoreAdapter(nn.Module):
    """
    Wraps the GreenFormer to return a Tuple instead of a Dict.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        return out['fg'], out['alpha']

# ---------------------------------------------------------
# 2. The Nuke Wrapper
# ---------------------------------------------------------
class NukeWrapper(nn.Module):
    def __init__(self, traced_core: nn.Module, img_size: int = 2048):
        super().__init__()
        self.keyer = traced_core
        self.img_size = img_size
        
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nuke Input: [Batch, 4, H, W]
        H, W = x.size(2), x.size(3)
        
        # Nuke inferences frame-by-frame. Enforce batch dimension = 1 natively.
        rgb = x[0:1, :3, :, :]
        mask = x[0:1, 3:, :, :]
        
        # 1. Resize to fixed 2048x2048
        rgb_2048 = F.interpolate(rgb, size=[self.img_size, self.img_size], mode='bilinear', align_corners=False)
        mask_2048 = F.interpolate(mask, size=[self.img_size, self.img_size], mode='bilinear', align_corners=False)
        
        # 2. Normalize
        rgb_norm = (rgb_2048 - self.mean) / self.std
        inp = torch.cat([rgb_norm, mask_2048], dim=1)
        
        # Tell the TorchScript compiler this is definitively a 1-batch tensor
        inp = inp.view(1, 4, self.img_size, self.img_size)
        
        # 3. Run Traced Core
        fg, alpha = self.keyer(inp)
        
        # 4. Resize back to original resolution
        fg_out = F.interpolate(fg, size=[H, W], mode='bicubic', align_corners=False)
        alpha_out = F.interpolate(alpha, size=[H, W], mode='bicubic', align_corners=False)
        
        # 5. Combine
        return torch.cat([fg_out, alpha_out], dim=1)

def export_for_nuke():
    # 1. RUN THE TIMM PATCH FIRST!
    patch_timm_for_nuke()

    # NOTE: hardcoded path assumes you've downloaded their saved model weights.
    checkpoint_path = r"CorridorKeyModule\checkpoints\CorridorKey_v1.0.pth"
    save_path = "CorridorKey_Nuke_Dynamic_v1.0.pt"
    img_size = 2048
    use_refiner = True
    device = torch.device('cpu') 

    print(f"Initializing GreenFormer (img_size={img_size}) on {device}...")
    base_model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k", 
        img_size=img_size, 
        use_refiner=use_refiner
    )
    base_model = base_model.to(device)
    base_model.eval()

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    new_state_dict = {}
    model_state = base_model.state_dict()

    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            k = k[10:]
        if "pos_embed" in k and k in model_state:
            if v.shape != model_state[k].shape:
                N_src, N_dst, C = v.shape[1], model_state[k].shape[1], v.shape[2]
                grid_src, grid_dst = int(math.sqrt(N_src)), int(math.sqrt(N_dst))
                v_img = v.permute(0, 2, 1).view(1, C, grid_src, grid_src)
                v_resized = F.interpolate(v_img, size=(grid_dst, grid_dst), mode="bicubic", align_corners=False)
                v = v_resized.flatten(2).transpose(1, 2)
        new_state_dict[k] = v

    base_model.load_state_dict(new_state_dict, strict=False)

    print("Tracing the core model...")
    adapter = CoreAdapter(base_model).eval().to(device)
    dummy_input = torch.randn(1, 4, img_size, img_size, device=device)
    
    with torch.no_grad():
        traced_core = torch.jit.trace(adapter, dummy_input)
        
    print("Scripting the Nuke wrapper...")
    nuke_wrapper = NukeWrapper(traced_core, img_size=img_size).eval().to(device)
    
    # Using Script preserves the H/W shape logic so it works on any frame resolution.
    scripted_model = torch.jit.script(nuke_wrapper)
    
    scripted_model.save(save_path)
    print(f"Success! Model saved to: {save_path}")

if __name__ == "__main__":
    export_for_nuke()