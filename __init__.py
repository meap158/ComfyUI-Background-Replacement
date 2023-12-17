import warnings
import os
import shutil
import glob
import requests
from huggingface_hub import hf_hub_download
from .install import install as install_dependencies
from .background_replacement import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .background_replacement import background_replacement_path as base_path
from .captioner import init as init_captioner
from .upscaler import init as init_upscaler
from .segmenter import init as init_segmenter
from .depth_estimator import init as init_depth_estimator

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


def prepare_upscaler_environment():
    os.makedirs(os.path.join(base_path, "weights"), exist_ok=True)
    model_path = os.path.join(base_path, "weights", "RealESRGAN_x2plus.pth")

    if not os.path.exists(model_path):
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
        response = requests.get(url)
        with open(model_path, 'wb') as f:
            f.write(response.content)

    return model_path


def prepare_segmenter_environment():
    saved_models_path = os.path.join(base_path, "saved_models")
    git_path = os.path.join(base_path, "git")

    # Create directories if they don't exist
    os.makedirs(saved_models_path, exist_ok=True)
    os.makedirs(git_path, exist_ok=True)

    # Clone the DIS repository if not already present
    dis_repo_path = os.path.join(git_path, "xuebinqin", "DIS")
    if not os.path.exists(dis_repo_path):
        os.system(f"git clone https://github.com/xuebinqin/DIS.git {dis_repo_path}")

    # Download the model file using hf_hub_download
    hf_hub_download(
        repo_id="NimaBoscarino/IS-Net_DIS-general-use",
        filename="isnet-general-use.pth",
        local_dir=saved_models_path
    )

    # Remove the __pycache__ directory
    pycache_path = os.path.join(dis_repo_path, "IS-Net", "__pycache__")
    shutil.rmtree(pycache_path, ignore_errors=True)

    files_to_remove = [
        os.path.join(dis_repo_path, "IS-Net", "requirements.txt"),
        os.path.join(dis_repo_path, "IS-Net", "models", "__init__.py")
    ]  # Avoid import error
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)

    # Move all files and directories from IS-Net to the current directory
    source_path = os.path.join(dis_repo_path, "IS-Net", "*")
    files_to_move = glob.glob(source_path)

    for file_path in files_to_move:
        destination = os.path.join(base_path, os.path.basename(file_path))
        shutil.move(file_path, destination)

    # Delete the entire git directory
    shutil.rmtree(git_path, ignore_errors=True)

    return saved_models_path


def init():
    install_dependencies()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    init_captioner()

    model_path = prepare_upscaler_environment()
    init_upscaler(model_path)

    saved_models_path = prepare_segmenter_environment()
    init_segmenter(saved_models_path)

    init_depth_estimator()


init()
