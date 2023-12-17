import os
import shutil
from huggingface_hub import hf_hub_download


current_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
models_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(current_path)))), "models")

controlnet_path = os.path.join(models_path, "controlnet")
checkpoints_path = os.path.join(models_path, "checkpoints")

controlnet_model = "control-lora-depth-rank128.safetensors"
sd_xl_turbo_model = "sd_xl_turbo_1.0_fp16.safetensors"


if not os.path.exists(os.path.join(controlnet_path, controlnet_model)):
    subfolder = "control-LoRAs-rank128"
    hf_hub_download(repo_id="stabilityai/control-lora",
                    subfolder=subfolder,
                    filename=controlnet_model,
                    local_dir=controlnet_path,
                    local_dir_use_symlinks="auto")
    # After downloading, check if the file exists in the subfolder
    source_path = os.path.join(controlnet_path, subfolder, controlnet_model)
    if os.path.exists(source_path):
        # Define the target path
        target_path = os.path.join(controlnet_path, controlnet_model)
        # Move the file to the target directory
        shutil.move(source_path, target_path)
        print(f'File downloaded to: "{target_path}"')
        # Remove the subfolder
        shutil.rmtree(os.path.join(controlnet_path, subfolder))
else:
    print(f'"{controlnet_model}" already exists. Skipping download...')


if not os.path.exists(os.path.join(checkpoints_path, sd_xl_turbo_model)):
    hf_hub_download(repo_id="stabilityai/sdxl-turbo",
                    filename=sd_xl_turbo_model,
                    local_dir=checkpoints_path,
                    local_dir_use_symlinks="auto")
    target_path = os.path.join(checkpoints_path, sd_xl_turbo_model)
    print(f'File downloaded to: "{target_path}"')
else:
    print(f'"{sd_xl_turbo_model}" already exists. Skipping download...')
