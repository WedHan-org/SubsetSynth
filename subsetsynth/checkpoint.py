from __future__ import annotations

import torch


def load_v2_for_inference(model, checkpoint_path: str, device, use_ema: bool = True):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_physics = bool(ckpt.get("physics_enabled", False))
    model_physics = bool(getattr(model, "physics_enabled", False))
    if ckpt_physics != model_physics:
        raise RuntimeError(f"Checkpoint physics_enabled={ckpt_physics} but model physics_enabled={model_physics}")
    state = ckpt.get("model")
    if state is None:
        raise KeyError("Checkpoint does not contain 'model'")
    model.load_state_dict(state)
    if use_ema:
        ema_state = ckpt.get("ema")
        if ema_state is None:
            raise KeyError("EMA inference requested but checkpoint has no 'ema'")
        shadow = ema_state.get("shadow", ema_state)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in shadow:
                    p.copy_(shadow[name].to(device=p.device, dtype=p.dtype))
    model.eval()
    return ckpt
