import os
import torch
from torchvision import transforms
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation
import numpy as np

from .captioner import derive_caption
from .segmenter import segment
from .depth_estimator import get_depth_map
from .image_utils import ensure_resolution

background_replacement_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def print_green_text(text, color='\033[92m'):
    print(color + text + '\033[0m')


class BackgroundReplacement:
    # Define the expected input types for the node
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # Input image
                "depth_map_feather_threshold": ("INT", {"default": 128, "min": 0, "max": 255}),
                "depth_map_dilation_iterations": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF}),
                "depth_map_blur_radius": ("INT", {"default": 5, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("depth_image", "cropped_image")
    FUNCTION = "replace_background"  # Function name
    CATEGORY = "utils"  # Category for organization

    @staticmethod
    def rearrange_image_tensor_and_convert_to_pil(image_tensor):
        # Check if the tensor has a batch dimension
        if image_tensor.dim() == 4:
            # Permute the dimensions to the correct order and remove the batch dimension
            reorganized_tensor = image_tensor.permute(0, 3, 2, 1).squeeze(0)
        else:
            # If there's no batch dimension, assume the tensor is [height, width, num_channels]
            reorganized_tensor = image_tensor.permute(2, 1, 0)
        # Convert the tensor to a PIL image
        image_pil = transforms.ToPILImage()(reorganized_tensor)
        return image_pil

    @staticmethod
    def convert_and_add_batch_dimension(pil_image):
        # Convert the resulting PIL Image back to a tensor image
        tensor_image = transforms.ToTensor()(pil_image)
        # Reorganize the tensor dimensions
        reorganized_tensor = tensor_image.permute(2, 1, 0)
        # Add a batch dimension of size 1 at position 0
        reorganized_tensor_with_batch = reorganized_tensor.unsqueeze(0)
        return reorganized_tensor_with_batch

    @staticmethod
    def replace_background(image: torch.Tensor,
                           depth_map_feather_threshold,
                           depth_map_dilation_iterations,
                           depth_map_blur_radius):
        MEGAPIXELS = 1.0
        original = BackgroundReplacement.rearrange_image_tensor_and_convert_to_pil(image)
        options = {
            'depth_map_feather_threshold': depth_map_feather_threshold,
            'depth_map_dilation_iterations': depth_map_dilation_iterations,
            'depth_map_blur_radius': depth_map_blur_radius,
        }

        torch.cuda.empty_cache()
        print_green_text(f"Original size: {original.size}")
        print_green_text("Captioning...")
        caption = derive_caption(original)
        print_green_text(f"Caption: {caption}")

        torch.cuda.empty_cache()
        print_green_text(f"Ensuring resolution ({MEGAPIXELS}MP)...")
        resized = ensure_resolution(original, megapixels=MEGAPIXELS)
        print_green_text(f"Resized size: {resized.size}")

        torch.cuda.empty_cache()
        print_green_text("\033[92m" + "Segmenting..." + "\033[0m")
        [cropped, crop_mask] = segment(resized)

        torch.cuda.empty_cache()
        print("Depth mapping...")
        depth_map = get_depth_map(resized)

        torch.cuda.empty_cache()
        print_green_text("Feathering the depth map...")

        # Convert crop mask to grayscale and to numpy array
        crop_mask_np = np.array(crop_mask.convert('L'))

        # Convert to binary and dilate (grow) the edges
        # adjust threshold as needed
        crop_mask_binary = crop_mask_np > options.get(
            'depth_map_feather_threshold')
        # adjust iterations as needed
        dilated_mask = binary_dilation(
            crop_mask_binary, iterations=options.get('depth_map_dilation_iterations'))

        # Convert back to PIL Image
        dilated_mask = Image.fromarray((dilated_mask * 255).astype(np.uint8))

        # Apply Gaussian blur and normalize
        dilated_mask_blurred = dilated_mask.filter(
            ImageFilter.GaussianBlur(radius=options.get('depth_map_blur_radius')))
        dilated_mask_blurred_np = np.array(dilated_mask_blurred) / 255.0

        # Normalize depth map, apply blurred, dilated mask, and scale back
        depth_map_np = np.array(depth_map.convert('L')) / 255.0
        masked_depth_map_np = depth_map_np * dilated_mask_blurred_np
        masked_depth_map_np = (masked_depth_map_np * 255).astype(np.uint8)

        # Convert back to PIL Image
        masked_depth_map = Image.fromarray(masked_depth_map_np).convert('RGB')

        depth_image = BackgroundReplacement.convert_and_add_batch_dimension(masked_depth_map)
        cropped_image = BackgroundReplacement.convert_and_add_batch_dimension(cropped)
        return (depth_image, cropped_image,)


class ImageComposite:
    # Define the expected input types for the node
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
            },
        }

    @staticmethod
    def composite(overlay_image: torch.Tensor, background_image: torch.Tensor) -> torch.Tensor:
        print_green_text("Compositing...")
        # Extract the target height and width from the overlay tensor
        target_height, target_width = overlay_image.size(1), overlay_image.size(2)
        # Resize the background tensor to match the dimensions of the overlay tensor
        resized_background = torch.nn.functional.interpolate(background_image.permute(0, 3, 1, 2),
                                                size=(target_height, target_width), mode='bilinear').permute(0, 2, 3, 1)
        # Extract alpha channel from overlay
        alpha_channel = overlay_image[:, :, :, 3:4]
        # Invert alpha channel
        inverted_alpha = 1 - alpha_channel
        # Multiply overlay color by alpha and background color by inverted alpha
        result_color = (overlay_image[:, :, :, :3] * alpha_channel) + (resized_background[:, :, :, :3] * inverted_alpha)
        # Concatenate the result along the last dimension
        composited_image = torch.cat([result_color], dim=3)
        return (composited_image,)

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("composited_image",)
    FUNCTION = "composite"  # Function name
    CATEGORY = "utils"  # Category for organization


# Define a mapping of node class names to their respective classes
NODE_CLASS_MAPPINGS = {
    "BackgroundReplacement": BackgroundReplacement,
    "ImageComposite": ImageComposite,
}

# A dictionary that contains human-readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BackgroundReplacement": "Background Replacement",
    "ImageComposite": "Image Composite",
}
