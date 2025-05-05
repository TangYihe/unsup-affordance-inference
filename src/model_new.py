# Minimal Trainer for Hugging Face deployment
# -------------------------------------------------
# This slimmed‑down version keeps **ONLY** what is needed
# to load a trained checkpoint and run inference.
# It intentionally drops all training utilities (dataset handling, 
# optimisation, WandB logging, etc.) so the file stays light‑weight
# for an HF Space or model repo.

from pathlib import Path
import os
from typing import Optional

import numpy as np
from PIL import Image
import os.path as osp

import torch
import torch.nn.functional as F
import torchvision.transforms as T

# ---- project imports ---------------------------------------------------
from .network import Conv2DFiLMNet  # same as before
from .utils.img_utils import (
    load_pretrained_dino,
    transform_imgs_for_dino,
    get_dino_features,
    get_text_embedding,
)

# -----------------------------------------------------------------------
DEFAULT_MODEL_CFG = dict(
    in_channels=384,
    filters=[256, 64, 1],
    kernel_sizes=[3, 3, 1],
    strides=[1, 1, 1],
    norm=None,
    activation="lrelu",
    lang_emb_dim=1024,
    film_mode="zero",
)


class Trainer:
    """Light‑weight wrapper that owns the visual backbone (DINO),
    the FiLM network, and an *inference‑only* forward pass.

    Parameters
    ----------
    ckpt_path : str | Path
        Path to a `.pth` file that stores `{"model": state_dict}`.
    device : torch.device | str | None, optional
        Manually force a device (e.g. "cpu" for HF CPU Spaces).  If *None*,
        we pick CUDA when available.
    model_cfg : dict | None, optional
        Override the default Conv2DFiLMNet hyper‑parameters.
    """

    def __init__(self, ckpt_path: os.PathLike, dino, *, device: Optional[torch.device] = None, model_cfg: Optional[dict] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = model_cfg or DEFAULT_MODEL_CFG

        # 1)   Vision backbone (frozen DINO‑v2)
        torch_path = osp.join(osp.dirname(osp.dirname(__file__)), "data/torch_home")
        # self.dino = load_pretrained_dino(torch_path=torch_path, model_type='dinov2_vits14', use_registers=True, device=self.device).to(self.device).eval()
        self.dino = dino
        for p in self.dino.parameters():
            p.requires_grad_(False)

        # 2)   FiLM‑conditioned ConvNet (learned)
        self.model = Conv2DFiLMNet(**self.cfg)
        self.model.build()
        self.model.to(self.device).eval()

        # 3)   Load weights
        ckpt = torch.load(Path(ckpt_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model"], strict=True)
        print(f"✓ Loaded weights from {ckpt_path}")

    # -------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, img_np: np.ndarray, query: str, *, thresh: Optional[float] = None) -> np.ndarray:
        """Run a forward pass and return a heat‑map aligned to *img_np*.

        Parameters
        ----------
        img_np : np.ndarray
            H×W×3 uint8 or float‑in‑[0,1] RGB image.
        query : str
            Natural‑language text prompt describing the affordance.
        thresh : float | None, default 0.5
            If provided, binarises the probability map at this threshold.

        Returns
        -------
        np.ndarray
            Heat‑map (float32, H×W) scaled back to the original resolution.
        """
        # a) Pre‑process RGB exactly like during training
        proc = transform_imgs_for_dino(img_np, blur=False)[0]  # Tensor (3,H',W')
        proc = proc.unsqueeze(0).to(self.device)

        # b) Text → embedding (1024‑D)
        lang_emb = torch.from_numpy(get_text_embedding(query, dim=self.cfg["lang_emb_dim"]))
        lang_emb = lang_emb.unsqueeze(0).to(self.device, dtype=torch.float32)

        # c) Dense visual features via DINO
        feat = get_dino_features(self.dino, proc, repeat_to_orig_size=False)
        feat = feat.permute(0, 3, 1, 2)  # (B,C,H',W') for Conv2DFiLMNet

        # d) FiLM‑Net forward pass → logits
        logits = self.model(feat, lang_emb).squeeze(0).squeeze(0)  # (h',w')

        # e) σ → probability; optional threshold
        prob = torch.sigmoid(logits)
        if thresh is not None:
            prob = (prob > thresh).float()

        prob_np = prob.cpu().numpy().astype(np.float32)  # (h',w')

        # f) Resize back to original image size
        H, W = img_np.shape[:2]
        prob_img = Image.fromarray(prob_np, mode="F")
        prob_resized = np.array(
            T.functional.resize(prob_img, size=(H, W), interpolation=T.InterpolationMode.BILINEAR)
        )
        return prob_resized


# Convenience factory ----------------------------------------------------

def load_trainer(dino, ckpt_path: str | Path, device: Optional[str] = None) -> Trainer:
    """Single‑line helper so `app.py` can do:

    ```python
    trainer = load_trainer("ckpts/best.pth", device="cpu")
    mask = trainer.inference(img, "cup handle")
    ```
    """
    return Trainer(ckpt_path=ckpt_path, dino=dino, device=device)
