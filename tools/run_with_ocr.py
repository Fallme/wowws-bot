"""Prepare optional GPU wheels in a child process before importing the runner."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_component(command, *, timeout, **kwargs):
    """Keep component preparation responsive to the console's Stop button."""
    stop_file = os.environ.get("WOWS_STOP_FILE")
    if stop_file and Path(stop_file).exists():
        raise InterruptedError("OCR 组件准备已终止")
    process = subprocess.Popen(command, **kwargs)
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if stop_file and Path(stop_file).exists():
                raise InterruptedError("OCR 组件准备已终止")
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(command, timeout)
            time.sleep(.1)
        return subprocess.CompletedProcess(command, process.returncode)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def gpu_runtime_available() -> bool:
    """True when this ONNX Runtime build even exposes a CUDA provider.

    A missing provider means the GPU wheels are genuinely absent, while a
    present provider that fails inference is usually a transient condition
    (GPU busy with the game, driver hiccup) that must not trigger a
    multi-hundred-MB re-download.
    """
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def prepare_gpu():
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    probe = [sys.executable, str(ROOT / "tools/check_ocr_acceleration.py")]
    try:
        if run_component(probe, cwd=ROOT, timeout=90, creationflags=flags).returncode == 0:
            return True
        if sys.prefix == sys.base_prefix:
            print("[OCR] 不在项目虚拟环境，跳过自动安装，避免修改系统 Python", flush=True)
            return False
        if gpu_runtime_available():
            # Components are installed: the probe failure was transient (GPU
            # busy, driver hiccup).  Retry once, then fall back to CPU for
            # this run instead of re-downloading the GPU stack.
            print("[OCR] CUDA 组件已安装但探测失败，等待 3 秒后重试…", flush=True)
            time.sleep(3)
            if run_component(probe, cwd=ROOT, timeout=90, creationflags=flags).returncode == 0:
                return True
            print("[OCR] 重试仍失败，视为 GPU 暂不可用，本次回退 CPU，不重复下载", flush=True)
            return False
        cache = ROOT / "data" / "pip_cache"
        temp = ROOT / "data" / "install_tmp"
        cache.mkdir(parents=True, exist_ok=True)
        temp.mkdir(parents=True, exist_ok=True)
        install_env = dict(os.environ, PIP_CACHE_DIR=str(cache), TMP=str(temp), TEMP=str(temp))
        print("[OCR] 未检测到 CUDA 运行组件，自动安装项目锁定的 CUDA/cuDNN 组件", flush=True)
        result = run_component(
            [sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements-gpu.txt")],
            cwd=ROOT, timeout=900, creationflags=flags, env=install_env,
        )
        return result.returncode == 0 and run_component(
            probe, cwd=ROOT, timeout=90, creationflags=flags,
        ).returncode == 0
    except InterruptedError:
        raise
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[OCR] GPU 组件准备失败: {error}", flush=True)
        return False


if __name__ == "__main__":
    if os.environ.get("WOWS_OCR_DEVICE", "nvidia") == "nvidia" and not prepare_gpu():
        print("[OCR] NVIDIA OCR 不可用，本次回退 CPU；请检查 NVIDIA 驱动，未安装系统驱动", flush=True)
        os.environ["WOWS_OCR_DEVICE"] = "cpu"
    if os.environ.get("WOWS_STOP_FILE") and Path(os.environ["WOWS_STOP_FILE"]).exists():
        sys.exit(0)
    sys.path.insert(0, str(ROOT))
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
