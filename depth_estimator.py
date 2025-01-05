import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

depth_estimator = None
feature_extractor = None

DEVICE = None
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

def init():
    global depth_estimator, feature_extractor

    print("### ComfyUI-Background-Replacement: Initializing depth estimator...")

    depth_estimator = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas").to(DEVICE)
    feature_extractor = DPTFeatureExtractor.from_pretrained(
        "Intel/dpt-hybrid-midas")


def get_depth_map(image):
    original_size = image.size

    image = feature_extractor(
        images=image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad(), torch.autocast(DEVICE):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=original_size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

    return image
