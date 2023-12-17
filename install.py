import subprocess
import os
import sys
import pkg_resources
from packaging import version as pv
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def run_pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", *args])


def is_installed(
        package: str, version: str | None = None, strict: bool = True
):
    has_package = None
    try:
        has_package = pkg_resources.get_distribution(package)
        if has_package is not None:
            installed_version = has_package.version
            if (installed_version != version and strict is True) or (pv.parse(installed_version) < pv.parse(version) and strict is False):
                return False
            else:
                return True
        else:
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def install():
    flag_file = os.path.join(base_path, "requirements_installed.flag")
    if not os.path.exists(flag_file):
        print("### ComfyUI-Background-Replacement: Installing dependencies...")
        req_file = os.path.join(base_path, "requirements.txt")
        with open(req_file) as file:
            strict = True
            for package in file:
                package_version = None
                try:
                    package = package.strip()
                    if "==" in package:
                        package_version = package.split('==')[1]
                    elif ">=" in package:
                        package_version = package.split('>=')[1]
                        strict = False
                    if not is_installed(package, package_version, strict):
                        run_pip(package)
                except Exception as e:
                    print(e)
                    print(f"Warning: Failed to install {package}, ComfyUI-Background-Replacement will not work.")
                    raise e
        # Create a flag file to mark that installation is done
        with open(flag_file, "w") as flag_file:
            flag_file.write("Requirements installed")
        print("### ComfyUI-Background-Replacement: Requirements installed")
    else:
        print("### ComfyUI-Background-Replacement: Requirements already installed.")
