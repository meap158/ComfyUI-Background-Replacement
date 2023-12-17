import cv2
import numpy as np
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

upsampler = None


def init(model_path):
    global upsampler

    print("### ComfyUI-Background-Replacement: Initializing upscaler...")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)

    upsampler = RealESRGANer(scale=2, model_path=model_path, model=model, device="cuda")


def upscale(image):
    original_opencv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    output, _ = upsampler.enhance(original_opencv, outscale=2)
    upscaled = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    return upscaled
