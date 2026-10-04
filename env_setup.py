"""
Environment detector + dependency installer for the Halden "Three Generative Recipes" project.

Supported targets:
  1. Google Colab with a T4 GPU (the graded run): detects the T4, never touches Colab's CUDA torch.
  2. Google Colab on a CPU runtime: detects it and prints how to switch to a T4.
  3. Local Windows / macOS / Linux (CPU; any CUDA GPU is used if torch can see it).

Usage
-----
Notebook / script (the Colab notebook has this file inlined into its first code cell):

    from env_setup import setup
    ENV = setup()            # installs only what is missing, then reports the device
    DEVICE = ENV["device"]   # "cuda" or "cpu"
    RUN_TAG = ENV["tag"]     # "t4gpu" | "gpu" | "cpu"; results are written with this suffix

Command line (the macOS/Windows installers call this inside the project's .venv):

    python env_setup.py              # detect, install missing packages, print a summary
    python env_setup.py --check      # detect and summarise only, install nothing
"""

import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# import name -> pip spec. Every third-party import in the assignment script must appear here.
PACKAGE_MAP = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "torch": "torch",
}
ASSIGNMENT_SCRIPT = Path("scripts") / "Assignment_04_Three_Generative_Recipes.py"
LOCAL_MODULES = {"env_setup", "update_readme"}
COLAB_ONLY_MODULES = {"google"}   # google.colab: provided by Colab, only imported behind a platform check
MIN_PYTHON = (3, 10)
T4_RUNTIME_HINT = (
    "In Colab: Runtime > Change runtime type > Hardware accelerator: T4 GPU > Save,\n"
    "then Runtime > Run all. (The runtime restarts, so re-run from the top.)"
)


# ── detection ────────────────────────────────────────────────────────────────
def is_colab() -> bool:
    if importlib.util.find_spec("google") is not None:
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            pass
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def query_nvidia_smi() -> dict | None:
    """GPU name / VRAM / driver from nvidia-smi: the torch-independent signal that a GPU is attached."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, mem_mib, driver = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
    return {"name": name, "vram_gb": round(float(mem_mib) / 1024, 1), "driver": driver}


def detect_environment() -> dict:
    colab = is_colab()
    system = platform.system()  # "Windows" | "Darwin" | "Linux"
    return {
        "platform": "colab" if colab else {"Windows": "windows", "Darwin": "macos"}.get(system, "linux"),
        "system": system,
        "python": platform.python_version(),
        "nvidia_smi": query_nvidia_smi(),
    }


# ── installation ─────────────────────────────────────────────────────────────
def required_imports(script: Path) -> list[str]:
    """Top-level third-party imports of the assignment script (stdlib and local modules excluded)."""
    import ast
    lines = script.read_text(encoding="utf-8").splitlines()
    source = "\n".join("" if ln.lstrip()[:1] in ("!", "%") else ln for ln in lines)  # drop notebook magics
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return sorted(found - set(sys.stdlib_module_names) - LOCAL_MODULES - COLAB_ONLY_MODULES)


def missing_packages(imports: list[str]) -> list[str]:
    unmapped = [i for i in imports if i not in PACKAGE_MAP]
    if unmapped:
        raise SystemExit(f"Unmapped third-party import(s): {unmapped}. Add them to PACKAGE_MAP in env_setup.py.")
    return [PACKAGE_MAP[i] for i in imports if importlib.util.find_spec(i) is None]


def project_root() -> Path:
    # __file__ is absent when this module is inlined into the Colab notebook
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def install_missing(env: dict, script: Path | None = None) -> list[str]:
    script = script or project_root() / ASSIGNMENT_SCRIPT
    imports = required_imports(script) if script.exists() else list(PACKAGE_MAP)
    todo = missing_packages(imports)
    if env["platform"] == "colab" and "torch" in todo:
        # Colab ships a CUDA build of torch; this only triggers if someone uninstalled it.
        print("[setup] torch is missing on Colab; installing the default (CUDA-enabled on Linux) build.")
    if todo:
        print(f"[setup] installing missing packages: {' '.join(todo)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *todo])
        importlib.invalidate_caches()
    else:
        print(f"[setup] all required packages already present: {' '.join(PACKAGE_MAP[i] for i in imports)}")
    return todo


# ── device resolution ────────────────────────────────────────────────────────
def resolve_device(env: dict) -> dict:
    torch = importlib.import_module("torch")
    info = {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "device": "cpu", "gpu_name": None, "vram_gb": None, "compute_capability": None,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(device="cuda", gpu_name=p.name, vram_gb=round(p.total_memory / 2**30, 1),
                    compute_capability=f"{p.major}.{p.minor}")
    info["is_t4"] = bool(info["gpu_name"] and "T4" in info["gpu_name"])
    info["tag"] = "t4gpu" if info["is_t4"] else ("gpu" if info["device"] == "cuda" else "cpu")
    return info


def _warn(msg: str) -> None:
    bar = "!" * 72
    print(f"\n{bar}\n{msg}\n{bar}\n")


def diagnose(env: dict) -> None:
    """Explain anything that would make the run slower or the results mislabelled."""
    smi = env["nvidia_smi"]
    if env["platform"] == "colab" and smi is None:
        _warn("Colab is on a CPU-only runtime. Everything still runs, but about 3-4x slower,\n"
              "and timings / peak VRAM will not be T4 numbers.\n" + T4_RUNTIME_HINT)
    elif smi and env["device"] == "cpu":
        _warn(f"nvidia-smi sees a GPU ({smi['name']}) but torch cannot use it "
              f"(torch {env['torch']}, CUDA build: {env['torch_cuda_build']}).\n"
              "Usually a CPU-only torch was pip-installed over the CUDA build (for example, from\n"
              "requirements.txt). On Colab: Runtime > Disconnect and delete runtime, then run again\n"
              "without installing torch. Locally: install the CUDA build from pytorch.org.")
    elif env["device"] == "cuda" and not env["is_t4"]:
        print(f"[setup] note: GPU is {env['gpu_name']}, not a T4. Results are tagged '{env['tag']}' "
              "so they are not confused with the graded T4 run.")
    if env["mps_available"] and env["device"] == "cpu":
        print("[setup] note: Apple MPS is available but deliberately not used: the Burgers solver needs "
              "float64 and the FNO uses complex FFT weights, which MPS does not fully support.")


def hardware_snapshot(env: dict) -> dict:
    """Everything a reader needs to interpret wall-clock and VRAM numbers."""
    snap = {k: env.get(k) for k in ("platform", "system", "python", "torch", "torch_cuda_build",
                                    "device", "gpu_name", "vram_gb", "compute_capability", "tag")}
    snap["cpu"] = platform.processor() or platform.machine()
    snap["cpu_count"] = os.cpu_count()
    snap["driver"] = (env.get("nvidia_smi") or {}).get("driver")
    for pkg in ("numpy", "pandas", "matplotlib"):
        try:
            snap[pkg] = importlib.import_module(pkg).__version__
        except ImportError:
            snap[pkg] = None
    return snap


def setup(install: bool = True, require_t4: bool = False) -> dict:
    """Detect the environment, install missing packages, resolve the device, and print a summary."""
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, found {platform.python_version()}")
    env = detect_environment()
    if install:
        install_missing(env)
    env.update(resolve_device(env))
    diagnose(env)
    if require_t4 and not env["is_t4"]:
        raise RuntimeError("require_t4=True but no Tesla T4 is visible.\n" + T4_RUNTIME_HINT)
    gpu = f"{env['gpu_name']} ({env['vram_gb']} GB, sm_{(env['compute_capability'] or '').replace('.', '')})" \
        if env["device"] == "cuda" else "none"
    print(f"[setup] platform={env['platform']}  python={env['python']}  torch={env['torch']}  "
          f"device={env['device']}  gpu={gpu}  results tag='{env['tag']}'")
    return env


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="detect and summarise only; install nothing")
    args = ap.parse_args()
    result = setup(install=not args.check)
    print(json.dumps(hardware_snapshot(result), indent=1))
