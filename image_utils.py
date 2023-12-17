import math
from PIL import Image

from .upscaler import upscale

UPSCALE_PIXEL_THRESHOLD = 1
DOWNSCALE_PIXEL_THRESHOLD = 1


def maybe_upscale(original, megapixels=1.0):
    original_width, original_height = original.size
    original_pixels = original_width * original_height
    target_pixels = megapixels * 1024 * 1024

    if (original_pixels < target_pixels):
        scale_by = math.sqrt(target_pixels / original_pixels)
        target_width = original_width * scale_by
        target_height = original_height * scale_by

        if (target_width - original_width >= 1 or target_height - original_height >= UPSCALE_PIXEL_THRESHOLD):
            print("Upscaling...")

            upscaled = upscale(original)

            print("Upscaled size:", upscaled.size)

            return upscaled

    print("Not upscaling")
    return original


def maybe_downscale(original, megapixels=1.0):
    original_width, original_height = original.size
    original_pixels = original_width * original_height
    target_pixels = megapixels * 1024 * 1024

    if (original_pixels > target_pixels):
        scale_by = math.sqrt(target_pixels / original_pixels)
        target_width = original_width * scale_by
        target_height = original_height * scale_by

        if (original_width - target_width >= 1 or original_height - target_height >= DOWNSCALE_PIXEL_THRESHOLD):
            print("Downscaling...")

            target_width = round(target_width)
            target_height = round(target_height)

            downscaled = original.resize(
                (target_width, target_height), Image.LANCZOS)

            print("Downscaled size:", downscaled.size)

            return downscaled

    print("Not downscaling")
    return original


def ensure_resolution(original, megapixels=1.0):
    return maybe_downscale(maybe_upscale(original, megapixels), megapixels)
