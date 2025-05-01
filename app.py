# app.py ───────────────────────────────────────────────────────────────────
from pathlib import Path
import os, numpy as np
import gradio as gr
from PIL import Image
import torch

from src.model_new import load_trainer                     # your new helper
from src.utils.img_utils import load_pretrained_dino   # unchanged util

# ------------------------------------------------------------------------
# 0)  Global device
# ------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------------
# 1)  Build ONE DINO backbone and freeze it
# ------------------------------------------------------------------------
TORCH_HOME = Path(__file__).parent / "torch_home"   # adapt if needed
print(f"⇢ Loading DINO-v2 backbone on {DEVICE} …")
dino = load_pretrained_dino(
    torch_path=str(TORCH_HOME),
    model_type="dinov2_vits14",
    use_registers=True,
    device=DEVICE
).to(DEVICE).eval()

for p in dino.parameters():
    p.requires_grad_(False)
print("✓ DINO ready")

# ------------------------------------------------------------------------
# 2)  Discover checkpoints in ./ckpts
# ------------------------------------------------------------------------
CKPT_DIR = Path(__file__).parent / "checkpoints" / "objaverse"
CKPTS = {p.stem: p for p in CKPT_DIR.glob("*.pth")}
DEFAULT_LABEL = next(iter(CKPTS))  # first one

# ------------------------------------------------------------------------
# 3)  Cache Trainers (share the *same* DINO instance)
# ------------------------------------------------------------------------
_loaded_trainers = {}   # label → Trainer

def get_trainer(label: str):
    if label not in _loaded_trainers:
        print(f"⇢ Loading FiLM head: {CKPTS[label].name}")
        _loaded_trainers[label] = load_trainer(
            dino,                       # shared backbone
            CKPTS[label],               # checkpoint path
            device=DEVICE
        )
    return _loaded_trainers[label]

# ------------------------------------------------------------------------
# 4)  Inference wrapper
# ------------------------------------------------------------------------
def predict(img_pil: Image.Image, text: str, ckpt_label: str):
    trainer = get_trainer(ckpt_label)
    mask = trainer.inference(np.array(img_pil), text, thresh=0.5)
    return Image.fromarray((mask * 255).astype(np.uint8))

# ------------------------------------------------------------------------
# 5)  Gradio UI
# ------------------------------------------------------------------------
with gr.Blocks(title="Affordance Heat-map Demo") as demo:
    ckpt_dd = gr.Dropdown(
        label="Checkpoint",
        choices=list(CKPTS.keys()),
        value=DEFAULT_LABEL,
        interactive=True
    )

    img_in   = gr.Image(type="pil", label="Input RGB")
    text_in  = gr.Textbox(label="Affordance Query", value="cup handle")
    mask_out = gr.Image(type="pil", label="Heat-map")

    run_btn = gr.Button("Run")
    run_btn.click(
        predict,
        inputs=[img_in, text_in, ckpt_dd],
        outputs=mask_out
    )

demo.launch()
