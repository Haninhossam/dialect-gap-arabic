"""GGUF feasibility cells for the sanity notebook (imported by make_sanity_notebook.py)."""


def gguf_cells(md, code):
    return [
md("""## GGUF Q4_K_M feasibility (edge format), Gemma-3-4B only
1. Build `llama-quantize` **CPU-only** (quantization is CPU work; this avoids Kaggle's CUDA driver-stub CMake error).
2. Install only the `gguf` package from llama.cpp. We do **not** install llama.cpp's convert requirements file: it pins a CPU-only torch that would replace Kaggle's CUDA torch.
3. Convert the HF checkpoint to GGUF f16, then quantize to Q4_K_M.
4. Build llama-cpp-python for the T4 (sm_75) with explicit CUDA paths; fall back to a CPU build if that fails. The backend actually used is recorded.
5. Score the same 24 prompts against the fp32 reference, and time 8 prompts CPU-only (edge-like latency)."""),
code("""%%time
import shutil, subprocess, sys
LC = '/kaggle/working/llama.cpp'
if not os.path.exists(LC):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/ggml-org/llama.cpp', LC], check=True)
subprocess.run(['cmake', '-S', LC, '-B', f'{LC}/build-cpu', '-DGGML_CUDA=OFF', '-DLLAMA_CURL=OFF',
                '-DCMAKE_BUILD_TYPE=Release'], check=True, capture_output=True)
subprocess.run(['cmake', '--build', f'{LC}/build-cpu', '--config', 'Release', '-j', '4', '--target', 'llama-quantize'],
               check=True, capture_output=True)
QUANTIZE = f'{LC}/build-cpu/bin/llama-quantize'
assert os.path.exists(QUANTIZE), 'llama-quantize was not built'
# gguf-py from the same checkout (matches the convert script); NOT the requirements file (it pins CPU torch)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'{LC}/gguf-py', 'sentencepiece', 'protobuf'], check=True)
print('llama-quantize OK | torch still', torch.__version__, '| CUDA available:', torch.cuda.is_available())"""),
code("""%%time
from huggingface_hub import snapshot_download
src = snapshot_download(LLMS['gemma3-4b'])
GG = '/kaggle/working/gguf'; os.makedirs(GG, exist_ok=True)
f16, q4 = f'{GG}/gemma3-4b-f16.gguf', f'{GG}/gemma3-4b-Q4_K_M.gguf'
if not os.path.exists(q4):
    r = subprocess.run([sys.executable, f'{LC}/convert_hf_to_gguf.py', src, '--outtype', 'f16', '--outfile', f16],
                       capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:]); r.check_returncode()
    r = subprocess.run([QUANTIZE, f16, q4, 'Q4_K_M'], capture_output=True, text=True)
    print(r.stderr[-800:]); r.check_returncode()
    os.remove(f16)  # free ~8 GB of /kaggle/working
print('GGUF sizes (GB):', {f: round(os.path.getsize(f'{GG}/{f}') / 1e9, 2) for f in os.listdir(GG)})"""),
code("""%%time
# GPU build for T4 only (sm_75 keeps compile time down), with explicit toolkit + driver-stub paths.
nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
stubs = f'{cuda_root}/lib64/stubs'
print('nvcc:', nvcc, '| CUDA root:', cuda_root, '| stubs exist:', os.path.exists(f'{stubs}/libcuda.so'))
cuda_args = (f'-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DCUDAToolkit_ROOT={cuda_root} '
             f'-DCMAKE_CUDA_COMPILER={nvcc} -DCMAKE_LIBRARY_PATH={stubs}')
pip_cmd = [sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--force-reinstall', '--no-deps', 'llama-cpp-python']
env = dict(os.environ, CMAKE_ARGS=cuda_args, FORCE_CMAKE='1',
           LIBRARY_PATH=stubs + ':' + os.environ.get('LIBRARY_PATH', ''))
r = subprocess.run(pip_cmd, env=env, capture_output=True, text=True)
GGUF_BACKEND = 'cuda' if r.returncode == 0 else None
if GGUF_BACKEND is None:
    print('CUDA build failed (tail):\\n', r.stderr[-2500:])
    r = subprocess.run(pip_cmd, env=dict(os.environ, CMAKE_ARGS='-DGGML_CUDA=off', FORCE_CMAKE='1'),
                       capture_output=True, text=True)
    GGUF_BACKEND = 'cpu' if r.returncode == 0 else None
    if GGUF_BACKEND is None: print('CPU build failed too:\\n', r.stderr[-2500:])
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diskcache', 'jinja2', 'typing-extensions'], check=False)
print('llama-cpp-python backend:', GGUF_BACKEND)"""),
code("""from dialectgap.gguf import load_gguf, score_mcq_gguf
from dialectgap.scoring import compare_runs
import llama_cpp
gpu_ok = bool(llama_cpp.llama_supports_gpu_offload())
print('llama.cpp GPU offload supported:', gpu_ok)
ref = from_jsonable(json.load(open(path('gemma3-4b', 'fp32'))))['single']
gg = {'build_backend': GGUF_BACKEND, 'gpu_offload_supported': gpu_ok, 'llama_cpp_python': llama_cpp.__version__}
runs = [('cpu_8prompts', 0, prompts[:8])] + ([('gpu_24prompts', -1, prompts)] if gpu_ok else [])
for name, ngl, ps in runs:
    try:
        llm = load_gguf(q4, n_gpu_layers=ngl, n_threads=os.cpu_count())
        r = score_mcq_gguf(llm, ps); del llm
        sub = {k: v[:len(ps)] for k, v in ref.items() if k != 'seconds'}
        gg[name] = {**compare_runs(sub, r), 'tok_per_s': float(r['n_tokens'].sum() / r['seconds']),
                    'sec_per_prompt': float(r['seconds'] / len(ps)), 'ok': True}
    except Exception as ex:
        gg[name] = {'ok': False, 'error': f'{type(ex).__name__}: {ex}'}
json.dump(gg, open(f'{OUT}/gguf_gemma3-4b_Q4_K_M.json', 'w'), indent=1); gg"""),
    ]
