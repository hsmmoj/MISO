import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Load model once at module level
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16", torch_dtype=torch.float32, use_safetensors=True).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

def get_clip_patches(image_path):
    """Original method - returns (196, 768)"""
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output = clip_model.vision_model(**inputs)
        patch_feats = output.last_hidden_state[:, 1:, :]  # Remove CLS token
        return patch_feats.squeeze(0)

def get_clip_embeddings(image_path, method='original', **kwargs):
    """
    Get CLIP embeddings using different methods and return in consistent 2D grid format
    """
    if method == 'original':
        return _get_original_clip_patches(image_path)
    
    elif method == 'highres':
        patch_size = kwargs.get('patch_size', 224)
        overlap = kwargs.get('overlap', 0.1)
        grid_size = kwargs.get('grid_size', (14, 14))  # Target grid size
        return _get_highres_clip_patches(image_path, patch_size, overlap, grid_size)
    
    elif method == 'multiscale':
        scales = kwargs.get('scales', [0.5, 0.75, 1.0])
        return _get_multiscale_clip_patches(image_path, scales)
    
    elif method == 'smart_crops':
        num_crops = kwargs.get('num_crops', 5)
        grid_size = kwargs.get('grid_size', (14, 14))
        return _get_smart_crops_clip_patches(image_path, num_crops, grid_size)
    
    elif method == 'adaptive':
        target_patches = kwargs.get('target_patches', 9)
        return _get_adaptive_clip_patches(image_path, target_patches)
    
    else:
        raise ValueError(f"Unknown CLIP method: {method}")

def _get_original_clip_patches(image_path):
    """Original method - returns (14, 14, 768)"""
    clip_emb = get_clip_patches(image_path)  # (196, 768)
    return clip_emb.view(14, 14, 768)

def _get_highres_clip_patches(image_path, patch_size=224, overlap=0.1, grid_size=(14, 14)):
    """High-resolution sliding window method"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    stride = int(patch_size * (1 - overlap))
    
    all_patches = []
    positions = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            
            inputs = clip_processor(images=patch, return_tensors="pt")
            with torch.no_grad():
                output = clip_model.vision_model(**inputs)
                patch_feats = output.last_hidden_state[:, 1:, :]  # Remove CLS token
                all_patches.append(patch_feats.squeeze(0))  # (196, 768)
                positions.append((x, y))
    all_patches = torch.stack(all_patches, dim=0)  # (num_windows, 196, 768)
    # Average pool across windows for each spatial location
    aggregated = torch.mean(all_patches, dim=0)  # (196, 768)
    return aggregated.view(14, 14, 768)

def _get_multiscale_clip_patches(image_path, scales=[0.5, 0.75, 1.0]):
    """Multi-scale method"""
    image = Image.open(image_path).convert("RGB")
    all_features = []
    
    for scale in scales:
        new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
        scaled_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        inputs = clip_processor(images=scaled_image, return_tensors="pt")
        with torch.no_grad():
            output = clip_model.vision_model(**inputs)
            patch_feats = output.last_hidden_state[:, 1:, :]  # Remove CLS token
            all_features.append(patch_feats.squeeze(0))
    # Average across scales
    multiscale_features = torch.stack(all_features, dim=0)  # (num_scales, 196, 768)
    averaged_features = torch.mean(multiscale_features, dim=0)  # (196, 768)
    return averaged_features.view(14, 14, 768)

def _get_smart_crops_clip_patches(image_path, num_crops=5, grid_size=(14, 14)):
    """Smart crops method"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    crop_size = min(w, h) // 2
    if crop_size < 224:
        crop_size = min(w, h)
    crops = []
    center_x, center_y = w // 2, h // 2
    crops.append(image.crop((
        max(0, center_x - crop_size//2), max(0, center_y - crop_size//2),
        min(w, center_x + crop_size//2), min(h, center_y + crop_size//2)
    )))
    if num_crops > 1:
        positions = [
            (0, 0), (max(0, w-crop_size), 0), 
            (0, max(0, h-crop_size)), (max(0, w-crop_size), max(0, h-crop_size))
        ]
        for i, (x, y) in enumerate(positions[:num_crops-1]):
            crops.append(image.crop((x, y, min(w, x+crop_size), min(h, y+crop_size))))
    all_features = []
    for crop in crops:
        inputs = clip_processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            output = clip_model.vision_model(**inputs)
            patch_feats = output.last_hidden_state[:, 1:, :]  # Remove CLS token
            all_features.append(patch_feats.squeeze(0))
    crop_features = torch.stack(all_features, dim=0)  # (num_crops, 196, 768)
    averaged_features = torch.mean(crop_features, dim=0)  # (196, 768)
    return averaged_features.view(14, 14, 768)

def _get_adaptive_clip_patches(image_path, target_patches=9):
    """Adaptive grid method"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    aspect_ratio = w / h
    if aspect_ratio > 1:
        grid_w = int(np.sqrt(target_patches * aspect_ratio))
        grid_h = int(target_patches / grid_w)
    else:
        grid_h = int(np.sqrt(target_patches / aspect_ratio))
        grid_w = int(target_patches / grid_h)
    
    patch_w = w // grid_w
    patch_h = h // grid_h
    all_patches = []
    for i in range(grid_h):
        for j in range(grid_w):
            x1 = j * patch_w
            y1 = i * patch_h
            x2 = min(w, x1 + patch_w + 50)  # Small overlap
            y2 = min(h, y1 + patch_h + 50)
            
            patch = image.crop((x1, y1, x2, y2))
            
            inputs = clip_processor(images=patch, return_tensors="pt")
            with torch.no_grad():
                output = clip_model.vision_model(**inputs)
                patch_feats = output.last_hidden_state[:, 1:, :]
                all_patches.append(patch_feats.squeeze(0))
    all_patches = torch.stack(all_patches, dim=0)  # (grid_h*grid_w, 196, 768)
    averaged_features = torch.mean(all_patches, dim=0)  # (196, 768)
    return averaged_features.view(14, 14, 768)