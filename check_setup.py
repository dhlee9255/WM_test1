"""Check that this computer is ready to run the world model, and say how to fix what isn't.

    python check_setup.py
"""
import importlib
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).parent
OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
problems = 0


def report(status, msg, fix=None):
    global problems
    print(f"{status} {msg}")
    if fix:
        print(f"       -> {fix}")
    if status == BAD:
        problems += 1


def nvidia_gpu():
    """(name, driver major version) from nvidia-smi, or None."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15).stdout.strip().splitlines()
        name, driver = [s.strip() for s in out[0].split(",")]
        return name, int(driver.split(".")[0])
    except Exception:
        return None


def torch_install_cmd(gpu):
    if gpu and "RTX 50" in gpu[0]:
        return "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128"
    return "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126"


def main():
    print(f"Python {sys.version.split()[0]}  ({sys.executable})\n")
    if sys.version_info < (3, 10):
        report(BAD, "Python 3.10 or newer is required", "install a newer Python and recreate the venv")

    gpu = nvidia_gpu()
    if gpu:
        report(OK, f"NVIDIA GPU: {gpu[0]} (driver {gpu[1]})")
        if "RTX 50" in gpu[0] and gpu[1] < 570:
            report(BAD, "RTX 50-series needs NVIDIA driver 570 or newer", "update the GPU driver from nvidia.com")
    else:
        report(WARN, "no NVIDIA GPU found (nvidia-smi missing) - the model will run on CPU, about 0.3 fps")

    try:
        import torch
    except ImportError:
        report(BAD, "PyTorch is not installed", torch_install_cmd(gpu))
        torch = None
    except OSError as e:  # DLL load failure on Windows
        report(BAD, f"PyTorch failed to load: {str(e).splitlines()[0][:120]}",
               "install the Microsoft Visual C++ Redistributable (x64), then " + torch_install_cmd(gpu))
        torch = None
    if torch is not None:
        build = torch.version.cuda or "CPU-only"
        print(f"       torch {torch.__version__}, built for CUDA {build}")
        if not torch.cuda.is_available():
            if gpu:
                report(BAD, "PyTorch cannot use the GPU (CPU-only build or driver too old)", torch_install_cmd(gpu))
            else:
                report(WARN, "PyTorch will run on CPU")
        else:
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"sm_{major}{minor}"
            if arch not in torch.cuda.get_arch_list():
                report(BAD, f"this PyTorch build has no kernels for {torch.cuda.get_device_name(0)} ({arch})",
                       torch_install_cmd(gpu))
            else:
                try:
                    x = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
                    torch.nn.functional.conv2d(x, torch.randn(8, 8, 3, 3, device="cuda", dtype=torch.float16))
                    torch.cuda.synchronize()
                    fast16 = (major, minor) >= (7, 0)
                    report(OK, f"CUDA works on {torch.cuda.get_device_name(0)} ({arch})"
                               + (", fp16 tensor cores" if fast16 else ", no fast fp16 (fp32 will be used)"))
                except RuntimeError as e:
                    report(BAD, f"CUDA test failed: {str(e).splitlines()[0]}", torch_install_cmd(gpu))

    # after torch: on Windows, importing panda3d first breaks torch's DLL loading
    for mod, pip_name in [("panda3d.core", "panda3d"), ("gltf", "panda3d-gltf"), ("simplepbr", "panda3d-simplepbr"),
                          ("pygame", "pygame"), ("numpy", "numpy"), ("PIL", "pillow")]:
        try:
            importlib.import_module(mod)
            report(OK, f"{pip_name}")
        except ImportError:
            report(BAD, f"{pip_name} is not installed", "pip install -r requirements.txt")

    assets = ROOT / "assets" / "processed"
    if (assets / "character-human.glb").exists() and (assets / "graveyard" / "road.glb").exists():
        report(OK, "game assets")
    else:
        report(BAD, "game assets are missing or outdated", "python setup_assets.py")

    ckpts = sorted((ROOT / "checkpoints").glob("*.pt"))
    if not ckpts:
        report(BAD, "no weights in checkpoints/", "download the weights (see README) into checkpoints/")
    elif torch is not None:
        for p in ckpts:
            try:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                report(OK, f"weights {p.name}: step {ck['step']}, {ck['config']['context']} context frames")
            except Exception as e:
                report(BAD, f"cannot read {p.name}: {e}")

    print()
    print("All good - run:  python play_model.py" if problems == 0 else f"{problems} problem(s) to fix (see -> above)")


if __name__ == "__main__":
    main()
