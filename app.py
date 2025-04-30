"""
Interface for HuggingFace deployment
"""

import gradio as gr
import numpy as np
# from src.model import AffordanceModel
from src.model_new import load_trainer
from src.utils.argument_utils import get_yaml_config
import cv2

# print("Loading config...")
# config = get_yaml_config("checkpoints/gemini/config.yaml")
# print("Building model...")
# model = AffordanceModel(config)
# print("Model built successfully!")

print("Loading model...")
model = load_trainer("checkpoints/objaverse/0429_mixedbg_05.pth", device="cuda")
print("Model loaded successfully!")

def predict(image, text):
    """
    Gradio inference function
    Args:
        image: PIL Image (Gradio's default image input format)
        text: str
    Returns:
        visualization of the heatmap
    """
    # Convert PIL image to numpy array
    image = np.array(image)
    
    # Run model inference
    heatmap = model.inference(image, text)  # Returns (H, W) array
    
    # Visualize heatmap (convert to RGB for display)
    # Scale to 0-255 and apply colormap
    heatmap_vis = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    return heatmap_colored

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Input Image"),  # Accepts uploaded images
        gr.Textbox(label="Text Query", placeholder="Enter text description...")
    ],
    outputs=gr.Image(label="Affordance Heatmap"),
    title="Affordance Detection",
    description="Upload an image and provide a text query to detect affordances.",
    # examples=[
    #     ["examples/test.png", "rim"]  # Add your test image and query
    # ],
    # cache_examples=True
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()