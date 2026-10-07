"""llama.cpp tooling for the GGUF format: build llama-quantize (CPU), install llama-cpp-python (CUDA if possible),
convert HF checkpoints to GGUF Q4_K_M. Every step records what happened instead of failing silently.

Lessons encoded here (sanity follow-up runs):
  * llama.cpp's convert requirements pin a CPU-only torch -> install only gguf-py.
  * llama-quantize needs no CUDA; building it with CUDA failed on Kaggle (no driver stub) -> build it CPU-only.
  * llama-cpp-python's runtime deps (diskcache, ...) must be installed before importing it.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

LLAMA_CPP_PYTHON_VERSION = "0.3.36"
CUDA_WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"


def setup_tools(lc_dir: str) -> dict:
    """Clone llama.cpp, build llama-quantize CPU-only, install gguf-py. Returns paths + commit."""
    lc = Path(lc_dir)
    if not (lc / "convert_hf_to_gguf.py").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(lc)], check=True)
    quantize = lc / "build-cpu" / "bin" / "llama-quantize"
    if not quantize.exists():
        subprocess.run(["cmake", "-S", str(lc), "-B", str(lc / "build-cpu"), "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF",
                        "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
        subprocess.run(["cmake", "--build", str(lc / "build-cpu"), "--config", "Release", "-j", "4",
                        "--target", "llama-quantize"], check=True, capture_output=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", str(lc / "gguf-py"), "sentencepiece"], check=True)
    commit = subprocess.run(["git", "-C", str(lc), "log", "-1", "--format=%h %cd"], capture_output=True, text=True).stdout.strip()
    return {"llama_cpp_dir": str(lc), "quantize": str(quantize), "llama_cpp_commit": commit}


def _probe() -> tuple[bool, bool, str]:
    """(imports, gpu_offload, text) checked in a fresh process (the running kernel may hold a stale import)."""
    p = subprocess.run([sys.executable, "-c", "import llama_cpp; print(llama_cpp.__version__, "
                        "bool(llama_cpp.llama_supports_gpu_offload()))"], capture_output=True, text=True)
    out = (p.stdout + p.stderr).strip()
    return p.returncode == 0, p.returncode == 0 and p.stdout.strip().endswith("True"), out[-1500:]


def install_llama_cpp_python() -> dict:
    """Prebuilt CUDA wheel -> CUDA source build (sm_75) -> CPU build. Returns {'backend': ..., tier logs}."""
    log = {}
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diskcache", "jinja2", "typing-extensions"], check=True)

    def attempt(name, args, env=None):
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--force-reinstall",
                            "--no-deps"] + args, env=env, capture_output=True, text=True)
        imports, gpu, text = _probe() if r.returncode == 0 else (False, False, "")
        log[name] = {"pip_ok": r.returncode == 0, "imports": imports, "gpu_offload": gpu, "probe": text,
                     "stderr_tail": r.stderr[-3000:]}
        return imports, gpu

    pkg = f"llama-cpp-python=={LLAMA_CPP_PYTHON_VERSION}"
    _, gpu = attempt("prebuilt_cu124", [pkg, "--only-binary=:all:", "--extra-index-url", CUDA_WHEEL_INDEX])
    if gpu:
        log["backend"] = "cuda-prebuilt"
        return log
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    stubs = f"{root}/lib64/stubs"
    log["cuda_env"] = {"nvcc": nvcc, "cuda_root": root, "driver_stub": os.path.exists(f"{stubs}/libcuda.so")}
    args = (f"-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DCUDAToolkit_ROOT={root} "
            f"-DCMAKE_CUDA_COMPILER={nvcc} -DCMAKE_LIBRARY_PATH={stubs}")
    env = dict(os.environ, CMAKE_ARGS=args, FORCE_CMAKE="1", LIBRARY_PATH=stubs + ":" + os.environ.get("LIBRARY_PATH", ""))
    _, gpu = attempt("cuda_source_build", [pkg], env=env)
    if gpu:
        log["backend"] = "cuda-source"
        return log
    imports, _ = attempt("cpu_build", [pkg], env=dict(os.environ, CMAKE_ARGS="-DGGML_CUDA=off", FORCE_CMAKE="1"))
    log["backend"] = "cpu" if imports else None
    return log


def convert_q4km(hf_dir: str, out_path: str, tools: dict, outtype_tmp_dir: str = "/tmp") -> str:
    """HF checkpoint dir -> GGUF f16 (temporary) -> Q4_K_M. Returns out_path."""
    out = Path(out_path)
    if out.exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    f16 = Path(outtype_tmp_dir) / (out.stem + "-f16.gguf")
    r = subprocess.run([sys.executable, str(Path(tools["llama_cpp_dir"]) / "convert_hf_to_gguf.py"), str(hf_dir),
                        "--outtype", "f16", "--outfile", str(f16)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("convert failed: " + r.stderr[-1500:])
    r = subprocess.run([tools["quantize"], str(f16), str(out), "Q4_K_M"], capture_output=True, text=True)
    f16.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError("quantize failed: " + r.stderr[-1500:])
    return str(out)


def gpu_offload_smoke_test(tools: dict, work_dir: str) -> dict:
    """Gate 2: convert the tiny HF test model to GGUF f32 and run it with all layers offloaded.

    Checks that this llama-cpp-python build really executes on the GPU (not just that it imports)."""
    from huggingface_hub import snapshot_download
    work = Path(work_dir)
    hf_dir = work / "tiny-hf"
    hf_dir.mkdir(parents=True, exist_ok=True)
    for f in Path(snapshot_download("hf-internal-testing/tiny-random-LlamaForCausalLM")).iterdir():
        if f.is_file():
            shutil.copy(f, hf_dir / f.name)
    cfg = json.loads((hf_dir / "config.json").read_text())
    cfg["pad_token_id"] = None                     # the test model ships pad_token_id=-1 (rejected by the converter)
    (hf_dir / "config.json").write_text(json.dumps(cfg))
    gguf = work / "tiny-f32.gguf"
    r = subprocess.run([sys.executable, str(Path(tools["llama_cpp_dir"]) / "convert_hf_to_gguf.py"), str(hf_dir),
                        "--outtype", "f32", "--outfile", str(gguf)], capture_output=True, text=True)
    if r.returncode != 0:
        return {"ok": False, "error": "tiny convert failed: " + r.stderr[-800:]}
    code = ("import llama_cpp, json; from llama_cpp import Llama; "
            f"m = Llama(model_path={str(gguf)!r}, n_gpu_layers=-1, verbose=True); m.eval(m.tokenize(b'hello world')); "
            "print(json.dumps({'gpu_offload': bool(llama_cpp.llama_supports_gpu_offload())}))")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    text = p.stdout + p.stderr
    offloaded = any(s in text for s in ("offloaded", "CUDA0", "using device CUDA"))
    try:
        gpu = json.loads(p.stdout.strip().splitlines()[-1])["gpu_offload"]
    except (ValueError, IndexError, KeyError):
        gpu = False
    return {"ok": p.returncode == 0 and gpu and offloaded, "gpu_offload_flag": gpu, "log_mentions_gpu": offloaded,
            "log_tail": text[-2000:]}
