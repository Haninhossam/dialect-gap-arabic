"""Part C (GGUF) cells for notebook 02 (imported by make_followup_notebook.py)."""


def gguf_cells(md, code):
    return [
md("""## C. GGUF Q4_K_M for all 3 models
`llama-quantize` is built CPU-only. Only `gguf-py` is installed: llama.cpp's convert requirements pin a CPU-only torch.
llama-cpp-python install order: (1) prebuilt CUDA 12.4 wheel, (2) CUDA source build for sm_75, (3) CPU build.
The tier used and the error from every failed tier are saved to `C_build_log.json`.

Scoring (`dialectgap/gguf.py`) feeds llama.cpp the **HF tokenizer's ids** and reads the last-token logits directly from the
context (run-1 bug: `llm.scores` stays all zeros when `logits_all=False`). A GGUF-f32 vs HF-f32 test shows max |Δ logprob| = 4e-6.
Each GGUF run gets the same validity check as the bitsandbytes formats."""),
code("""%%time
LC = '/kaggle/working/llama.cpp'
if not os.path.exists(LC):
    subprocess.run(['git', 'clone', '-q', '--depth', '1', 'https://github.com/ggml-org/llama.cpp', LC], check=True)
QUANTIZE = f'{LC}/build-cpu/bin/llama-quantize'
if not os.path.exists(QUANTIZE):
    subprocess.run(['cmake', '-S', LC, '-B', f'{LC}/build-cpu', '-DGGML_CUDA=OFF', '-DLLAMA_CURL=OFF',
                    '-DCMAKE_BUILD_TYPE=Release'], check=True, capture_output=True)
    subprocess.run(['cmake', '--build', f'{LC}/build-cpu', '--config', 'Release', '-j', '4', '--target', 'llama-quantize'],
                   check=True, capture_output=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', f'{LC}/gguf-py', 'sentencepiece'], check=True)
LLAMACPP_COMMIT = subprocess.run(['git', '-C', LC, 'log', '-1', '--format=%h %cd'], capture_output=True, text=True).stdout.strip()
print('llama-quantize:', os.path.exists(QUANTIZE), '| llama.cpp', LLAMACPP_COMMIT, '| torch still', torch.__version__, torch.cuda.is_available())"""),
code("""%%time
build_log = {}
def pip_try(name, args, env=None):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--force-reinstall', '--no-deps'] + args,
                       env=env, capture_output=True, text=True)
    build_log[name] = {'ok': r.returncode == 0, 'stderr_tail': r.stderr[-4000:]}
    if r.returncode != 0: return False
    probe = subprocess.run([sys.executable, '-c', 'import llama_cpp; print(llama_cpp.__version__, bool(llama_cpp.llama_supports_gpu_offload()))'],
                           capture_output=True, text=True)   # fresh process: the current kernel may have a stale import
    build_log[name]['probe'] = (probe.stdout + probe.stderr)[-1500:]
    return probe.returncode == 0 and probe.stdout.strip().endswith('True')

GGUF_BACKEND = None
# 1) prebuilt CUDA 12.4 wheel (same 0.3.36 release that loaded Qwen3.5 in run 1)
if pip_try('prebuilt_cu124', ['llama-cpp-python==0.3.36', '--only-binary=:all:',
                               '--extra-index-url', 'https://abetlen.github.io/llama-cpp-python/whl/cu124']):
    GGUF_BACKEND = 'cuda-prebuilt'
# 2) CUDA source build for the T4 only (sm_75), explicit toolkit and driver-stub paths
if GGUF_BACKEND is None:
    nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    stubs = f'{cuda_root}/lib64/stubs'
    build_log['cuda_env'] = {'nvcc': nvcc, 'cuda_root': cuda_root, 'driver_stub': os.path.exists(f'{stubs}/libcuda.so')}
    args = (f'-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DCUDAToolkit_ROOT={cuda_root} '
            f'-DCMAKE_CUDA_COMPILER={nvcc} -DCMAKE_LIBRARY_PATH={stubs}')
    env = dict(os.environ, CMAKE_ARGS=args, FORCE_CMAKE='1', LIBRARY_PATH=stubs + ':' + os.environ.get('LIBRARY_PATH', ''))
    if pip_try('cuda_source_build', ['llama-cpp-python==0.3.36'], env=env):
        GGUF_BACKEND = 'cuda-source'
# 3) CPU build (accuracy is still valid; only GPU speed is lost)
if GGUF_BACKEND is None:
    pip_try('cpu_build', ['llama-cpp-python==0.3.36'], env=dict(os.environ, CMAKE_ARGS='-DGGML_CUDA=off', FORCE_CMAKE='1'))
    GGUF_BACKEND = 'cpu' if build_log['cpu_build']['ok'] else None
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diskcache', 'jinja2'], check=False)
build_log['backend'] = GGUF_BACKEND
save('C_build_log', build_log)
print('llama-cpp-python backend:', GGUF_BACKEND)
for k, v in build_log.items():
    if isinstance(v, dict) and 'ok' in v: print(' ', k, 'ok' if v['ok'] else 'FAILED', v.get('probe', '')[:120])"""),
code("""import llama_cpp
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, scan_cache_dir
from dialectgap.gguf import load_gguf, score_mcq_gguf
from dialectgap.sanity import VALID_MIN_LETTER_MASS
GPU_OK = bool(llama_cpp.llama_supports_gpu_offload())
print('llama-cpp-python', llama_cpp.__version__, '| GPU offload:', GPU_OK)
GG = '/kaggle/working/gguf'; os.makedirs(GG, exist_ok=True)
for m, mid in LLMS.items():
    name = f'C_{m}__gguf_q4km'
    if done(name): print('skip', name); continue
    rec = {'model': mid, 'format': 'gguf_q4km', 'backend': GGUF_BACKEND, 'gpu_offload': GPU_OK,
           'llama_cpp_python': llama_cpp.__version__, 'llama_cpp_commit_converter': LLAMACPP_COMMIT}
    q4 = f'{GG}/{m}-Q4_K_M.gguf'
    try:
        tok = AutoTokenizer.from_pretrained(mid)   # ids for both HF and GGUF runs
        if not os.path.exists(q4):
            src = snapshot_download(mid)
            f16 = f'/tmp/{m}-f16.gguf'   # large intermediate: keep it out of /kaggle/working (20 GB limit)
            r = subprocess.run([sys.executable, f'{LC}/convert_hf_to_gguf.py', src, '--outtype', 'f16', '--outfile', f16],
                               capture_output=True, text=True)
            if r.returncode != 0: raise RuntimeError('convert failed: ' + r.stderr[-1500:])
            r = subprocess.run([QUANTIZE, f16, q4, 'Q4_K_M'], capture_output=True, text=True)
            if r.returncode != 0: raise RuntimeError('quantize failed: ' + r.stderr[-1500:])
            os.remove(f16)
            info = scan_cache_dir(); info.delete_revisions(*[rv.commit_hash for rp in info.repos if rp.repo_id == mid for rv in rp.revisions]).execute()
        rec['size_gb'] = os.path.getsize(q4) / 1e9
        for label, ngl, ps in [('gpu', -1, P24), ('cpu', 0, P24[:8])]:
            if label == 'gpu' and not GPU_OK: continue
            llm = load_gguf(q4, n_gpu_layers=ngl, n_threads=os.cpu_count())
            r = score_mcq_gguf(llm, tok, ps); del llm
            sub = {k: v[:len(ps)] for k, v in ref[m]['single'].items() if k != 'seconds'}
            mass = float(np.median(letter_mass(r['logprobs'])))
            rec[label] = {**compare_runs(sub, r), 'sec_per_prompt': r['seconds'] / len(ps),
                          'tok_per_s': float(r['n_tokens'].sum() / r['seconds']), 'median_letter_mass': mass,
                          'valid': bool(np.isfinite(r['logprobs']).all() and mass >= VALID_MIN_LETTER_MASS),
                          'acc': float((r['logprobs'].argmax(1) == G24[:len(ps)]).mean())}
        rec['ok'] = True
    except Exception as ex:
        rec.update(ok=False, error=f'{type(ex).__name__}: {ex}'[:3000])
    save(name, rec); print(name, json.dumps(rec, indent=1)[:1500])
!df -h /kaggle/working /tmp | tail -2"""),
    ]
