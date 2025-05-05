import argparse, numpy as np, cv2, matplotlib.pyplot as plt
from PIL import Image
from src.model_new import load_trainer          # ← comes from our minimal file
from src.utils.img_utils import load_pretrained_dino

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",  required=True)
parser.add_argument("--img",   required=True)
parser.add_argument("--query", required=True)
args = parser.parse_args()

# 1) build the model (auto-chooses GPU if available)
dino = load_pretrained_dino(torch_path="/viscam/u/yihetang/unsup-affordance-inference/data/torch_home",model_type="dinov2_vits14", use_registers=True)
trainer = load_trainer(ckpt_path=args.ckpt, dino=dino)
trainer.model.eval()

# 2) run inference
img_np = np.array(Image.open(args.img).convert("RGB"))
heat = trainer.inference(img_np, args.query, thresh=None)   # 0-1 float map

# 3) visualise overlay
overlay = (cv2.applyColorMap((heat*255).astype(np.uint8),
                             cv2.COLORMAP_JET)[:, :, ::-1] * 0.6
           + img_np * 0.4).astype(np.uint8)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(img_np);   plt.title("RGB");   plt.axis("off")
plt.subplot(1,3,2); plt.imshow(heat, cmap="inferno");         plt.title("Mask"); plt.axis("off")
plt.subplot(1,3,3); plt.imshow(overlay);  plt.title("Overlay");plt.axis("off")
plt.suptitle(f"Query: {args.query}")
plt.tight_layout(); plt.savefig("test.png")
