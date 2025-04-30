"""
Image related util functions
"""

import cv2
import numpy as np
import os

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from openai import OpenAI

##############################################
### Image processing
##############################################

def rescale_img(img, max_size=448):
    """
    Rescale image such that largest dimension is max_size
    img: np.ndarray, (H, W, C)
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    
    scale = max_size / max(h, w)
    new_h, new_w = int(h*scale), int(w*scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


##############################################
### DINO features
##############################################

# def load_pretrained_dino(torch_path, model_type='dinov2_vitl14', device=None):
#     '''
#     Args:
#         torch_path: path to download pretrained model weights
#         model_type: in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
#         device: torch.device to load the model on. If None, will automatically detect
#     '''
#     # specify path to download pretrained model weights
#     os.environ['TORCH_HOME'] = torch_path

#     # Automatically determine device if not specified
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # load model
#     if model_type not in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']:
#         raise ValueError('Invalid model type')
    
#     dinov2 = torch.hub.load('facebookresearch/dinov2', model_type).eval()
#     dinov2 = dinov2.to(device)
    
#     return dinov2

def load_pretrained_dino(torch_path, model_type='dinov2_vitl14', use_registers=False, device=None):
    '''
    model_type: in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
    use_registers: bool, whether to use registers
    '''
    # specify path to download pretrained model weights
    os.environ['TORCH_HOME'] = torch_path

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    if model_type not in ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg']:
        raise ValueError('Invalid model type', model_type)
    
    if use_registers and 'reg' not in model_type:
        model_type = model_type + '_reg'
    
    dinov2 = torch.hub.load('facebookresearch/dinov2', model_type).eval().to(device)
    print(f"Loaded {model_type} model")
    
    return dinov2

def get_dino_features(dinov2, imgs, repeat_to_orig_size=False):
    """
    Get features from DINO model
    ::param dinov2:: DINO model
    ::param imgs:: tensor of shape (bs, C, H, W)
    """
    bs, C, H, W = imgs.shape
    patch_h = H // 14
    patch_w = W // 14

    with torch.no_grad():
        features_dict = dinov2.forward_features(imgs)
        features = features_dict['x_norm_patchtokens']
        features = features.reshape(bs, patch_h, patch_w, -1)
    
    if not repeat_to_orig_size:
        return features # (bs, patch_h, patch_w, n_features)
    else:
        # repeat on batched dims to original size
        ratio = H // (patch_h*2)
        features = F.interpolate(features.permute(0, 3, 1, 2), scale_factor=ratio, mode='bilinear').permute(0, 2, 3, 1)
        return features

def transform_imgs_for_dino(imgs, blur=True):
    """
    Transform image before passing to DINO model
    ::param imgs:: np.array of shape (H, W, C) or (bs, H, W, C)
    ::param blur:: bool, whether to apply Gaussian blur before resizing

    ::return:: list of transformed images
    """
    # handles both single image and batch of images
    if len(imgs.shape) == 3:
        H, W, C = imgs.shape
        imgs = imgs[None, ...]
        bs = 1
    else:
        bs, H, W, C = imgs.shape

    H *= 2
    W *= 2

    patch_h = H // 14
    patch_w = W // 14

    if blur:
        transform_lst = [T.GaussianBlur(9, sigma=(1.0, 2.0))]
    else:
        transform_lst = []
    transform_lst += [
        T.Resize((patch_h * 14, patch_w * 14)),
        T.CenterCrop((patch_h * 14, patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]

    transform = T.Compose(transform_lst)
    
    transformed_imgs = []
    for i in range(bs):
        temp = imgs[i].copy()
        if temp.max() <= 1.1: # handle images with values in [0, 1]
            temp = (temp * 255)
        temp = temp.astype(np.uint8).clip(0, 255)
        transformed_imgs.append(transform(Image.fromarray(temp)))
    
    return transformed_imgs


##############################################
### Preparing data for model
##############################################

def get_text_embedding(text, model="text-embedding-3-large", dim=1024):
    """
    Get text embedding with specified dimension
    """
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    embedding = np.array(client.embeddings.create(input=[text], model=model, dimensions=dim).data[0].embedding)
    return embedding

def preprocess_data(img, text, lang_emb_dim=1024):
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    processed_img = transform_imgs_for_dino(img)
    lang_emb = torch.tensor(get_text_embedding(text, dim=lang_emb_dim)).to(torch.float32)

    return processed_img, lang_emb

##############################################
### Visualization
##############################################

def vis_heatmap(img, aff):
    plt.imshow(img) 
    plt.imshow(aff, cmap="hot", alpha=np.clip(aff*2, 0.3, 0.9))
    plt.axis('off')
    plt.savefig("heatmap.png")
 
def overlay_heatmap(img, aff):
    """Compute the heatmap overlay on the image"""
    # Convert heatmap to colors using the 'hot' colormap
    cmap = plt.cm.hot
    heatmap_colored = cmap(aff)[:, :, :3]  # Remove alpha channel
    
    # Apply alpha blending
    alpha = np.clip(aff*2, 0.3, 0.9)
    alpha = alpha[:, :, None]  # Add channel dimension for broadcasting
    
    # Blend the original image with the heatmap
    blended = img * (1-alpha) + heatmap_colored * alpha
    
    # Ensure output is in uint8 range
    return (blended * 255).astype(np.uint8)

def resize_image(image, target_shape=None):
    """
    Resizes an image to the target shape using bilinear interpolation.
    
    Parameters:
        image (np.ndarray): The input image as a HxW numpy array.
        target_shape (tuple): The target shape (height, width).
    
    Returns:
        np.ndarray: The resized image as a H'xW' numpy array.
    """
    # Convert the numpy image to a torch tensor and add batch and channel dimensions
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
    image_tensor = torch.from_numpy(image).float().unsqueeze(0).permute(0, 3, 1, 2)  # Shape: [1, 1, H, W]

    # resize to nearest multiple of 14
    if target_shape is None:
        target_shape = (image.shape[0]//14*14, image.shape[1]//14*14)
    else:
        target_shape = (target_shape, target_shape)

    # Resize the image using interpolate
    resized_tensor = F.interpolate(image_tensor, size=target_shape, mode='bilinear', align_corners=False)

    resized_tensor = resized_tensor.permute(0, 2, 3, 1)
    
    # Remove the batch and channel dimensions and convert back to numpy
    resized_image = resized_tensor.squeeze().numpy()
    return resized_image