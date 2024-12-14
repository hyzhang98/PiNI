import cv2
import torch
from clip import clip
import numpy as np
import sys
from torchvision.utils import save_image  # Save a given Tensor into an image file.
import torchvision.transforms as transforms

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root=cfg.MODEL.ROOT)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())
    return model

def heatmap_show(images, mus, variances, labels, heatmap_dir, class_names, heatmapNumList):
    """draw heatmap for mu and variance"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    reverse_normalize_image = transforms.Normalize([-m / s for m, s in zip(mean, std)], [1 / s for s in std])
    images = reverse_normalize_image(images)

    for (image, mu, variance, label) in zip(images, mus, variances, labels):
        class_idx = int(label)
        if heatmapNumList[class_idx] > 100:
            continue
        dir = heatmap_dir + r'/' + class_names[class_idx]
        save_image(image, dir + r'/' + str(heatmapNumList[class_idx]) + r'_image.jpg')
        mu = mu.cpu().detach().numpy()
        variance = variance.cpu().detach().numpy()
        
        variance = np.uint8(variance / max(variance.max(), sys.float_info.epsilon) * 256)
        mu = np.uint8(mu / max(mu.max(), sys.float_info.epsilon) * 256)
        heatmap_variance = cv2.applyColorMap(variance, cv2.COLORMAP_WINTER)
        heatmap_mu = cv2.applyColorMap(mu, cv2.COLORMAP_WINTER)
        file_mu = dir + r'/' + str(heatmapNumList[class_idx]) + r'_mu.jpg'
        file_variance = dir + r'/' + str(heatmapNumList[class_idx]) + r'_variance.jpg'
        cv2.imwrite(file_mu, heatmap_mu)
        cv2.imwrite(file_variance, heatmap_variance)
        heatmapNumList[class_idx] += 1

