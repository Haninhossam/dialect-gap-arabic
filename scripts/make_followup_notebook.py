"""Generates notebooks/02_sanity_followup.ipynb."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
md("""# 02 — Sanity follow-up

Run 1 (`reports/sanity_findings.md`) showed that the Gemma-3 family returns NaN in fp16 on T4, including NF4/int8 with fp16 compute.
This notebook answers the open questions before the main runs:

- **A.** Gemma-3-4B and Nile-Chat-4B: int8 and NF4 with **fp32 compute**. Are they *valid*?
- **B.** Qwen3.5 thinking: does disabling thinking raise the probability mass on A–D without hurting accuracy?
  The Gemma family gets the same chat format, so we can decide whether one protocol fits all models.
- **C.** GGUF Q4_K_M for all 3 models (llama.cpp): conversion, validity, GPU speed, CPU-only latency.
- **D.** Qwen3.5 speed with the `flash-linear-attention` kernels (optional).

Validity rule for quantized formats (`dialectgap/sanity.py`): finite logits, median letter mass ≥ 0.05, non-empty greedy text.
fp32 references come from run 1 (`results/sanity/*__fp32.json`, committed in the repo).

**Settings:** GPU **T4 x2**, Internet **On**, Secret **`Kaggle_Token`** attached. Expected ≈ 1.5 h. Every result is saved as soon as
it is computed, and finished parts are skipped when the notebook is rerun."""),
code("""REPO_URL = "https://github.com/Haninhossam/dialect-gap-arabic.git"
!git clone -q $REPO_URL || (cd dialect-gap-arabic && git pull -q)
%cd dialect-gap-arabic
!pip install -q -U "transformers>=5.0" accelerate bitsandbytes && pip install -q -e ."""),
code("""# `pip install -e` is not visible to an already-running kernel; put src/ on the path directly.
import sys, os
sys.path.insert(0, os.path.abspath('src'))
import json, glob, shutil, subprocess, torch, transformers
import numpy as np, pandas as pd
from dialectgap.env import load_hf_token
load_hf_token()
print('torch', torch.__version__, '| transformers', transformers.__version__, '| GPUs', torch.cuda.device_count())
OUT = '/kaggle/working/followup' if os.path.exists('/kaggle') else 'results/followup'
os.makedirs(OUT, exist_ok=True)
REF = 'results/sanity'
def save(name, obj): json.dump(obj, open(f'{OUT}/{name}.json', 'w'), ensure_ascii=False, indent=1)
def done(name): return os.path.exists(f'{OUT}/{name}.json')"""),
md("""## Prompts
- **SANITY-24**: the same 24 prompts as run 1 (8 uids × eng/arb/arz), used to compare against the run-1 fp32 references.
- **FORMAT-DEV-300**: the first 100 uids (sorted) × eng/arb/arz. Used **only** to choose the prompt format. The choice rests
  mainly on letter mass, which needs no gold labels; accuracy is only a "does not hurt" check. These 100 uids are recorded,
  and the main analysis will also be reported without them as a robustness check."""),
code("""from dialectgap.data import load_belebele_aligned
from dialectgap.prompts import build_mcq_prompt
from dialectgap.models import LLMS
from dialectgap.sanity import run_precision, verdict, validity, to_jsonable, from_jsonable
from dialectgap.scoring import compare_runs, letter_mass
VARS = ['eng_Latn', 'arb_Arab', 'arz_Arab']
al = load_belebele_aligned(VARS)
def subset(k): return [it for v in VARS for it in al[v][:k]]
s24, s300 = subset(8), subset(100)
P24, G24 = [build_mcq_prompt(it) for it in s24], np.array([it['answer'] for it in s24])
P300, G300 = [build_mcq_prompt(it) for it in s300], np.array([it['answer'] for it in s300])
VAR300 = np.array([it['variety'] for it in s300])
GEN = [P24[16], P24[17]]
save('format_dev_uids', [it['uid'] for it in al['eng_Latn'][:100]])
ref = {m: from_jsonable(json.load(open(f'{REF}/{m}__fp32.json'))) for m in LLMS}
assert all(r['ok'] for r in ref.values()), 'missing fp32 references from run 1'
print(len(P24), len(P300))"""),
md("## A. Gemma family: int8 / NF4 with fp32 compute"),
code("""for m in ['gemma3-4b', 'nilechat-4b']:
    for prec in ['nf4', 'int8']:
        name = f'A_{m}__{prec}-c32'
        if done(name): print('skip', name); continue
        print('running', name, flush=True)
        rec = run_precision(LLMS[m], prec, P24, GEN, compute_dtype='fp32')
        rec['validity'] = validity(rec)
        if rec['ok']:
            rec['vs_fp32'] = compare_runs(ref[m]['single'], rec['single'])
            rec['acc'] = float((rec['single']['logprobs'].argmax(1) == G24).mean())
        save(name, to_jsonable(rec))
        print('  ', rec.get('validity'), rec.get('vs_fp32'), rec.get('error', '')[:200])"""),
md("""## B. Prompt format: does disabling thinking help?
Qwen3.5 at fp32 (reference) and fp16, in 3 formats; Gemma family at fp32 in raw + chat_nothink (one protocol for all?).
Letter tokens follow the format: `raw` scores " A", the other formats score "A" (see `dialectgap/prompts.py`)."""),
code("""FMT_RUNS = [('qwen3.5-4b', p, f) for p in ['fp32', 'fp16'] for f in ['raw', 'chat_nothink', 'raw_prefill_nothink']]
FMT_RUNS += [(m, 'fp32', f) for m in ['gemma3-4b', 'nilechat-4b'] for f in ['raw', 'chat_nothink']]
for m, prec, fmt in FMT_RUNS:
    name = f'B_{m}__{prec}__{fmt}'
    if done(name): print('skip', name); continue
    print('running', name, flush=True)
    rec = run_precision(LLMS[m], prec, P300, [GEN[0]], compute_dtype=prec, batch_size=8, fmt=fmt, check_batching=False)
    save(name, to_jsonable(rec))
    print('   ok' if rec['ok'] else '   FAILED ' + rec['error'][:300])"""),
code("""rows = []
for m, prec, fmt in FMT_RUNS:
    p = f'{OUT}/B_{m}__{prec}__{fmt}.json'
    if not os.path.exists(p): continue
    r = from_jsonable(json.load(open(p)))
    if not r['ok']: rows.append({'model': m, 'prec': prec, 'format': fmt, 'error': r['error'][:120]}); continue
    lp = r['single']['logprobs']; pred = lp.argmax(1); mass = letter_mass(lp)
    raw_p = f'{OUT}/B_{m}__{prec}__raw.json'
    agree_raw = float((pred == from_jsonable(json.load(open(raw_p)))['single']['logprobs'].argmax(1)).mean()) if os.path.exists(raw_p) else None
    row = {'model': m, 'prec': prec, 'format': fmt, 'mass_median': float(np.median(mass)), 'mass_p10': float(np.percentile(mass, 10)),
           'acc_all': float((pred == G300).mean()), 'agree_with_raw': agree_raw, 'tok_per_s': r['tokens_per_second_batched'],
           'gen': r['generations'][0][:80]}
    for v in VARS: row[f'acc_{v}'] = float((pred[VAR300 == v] == G300[VAR300 == v]).mean())
    rows.append(row)
fmt_summary = pd.DataFrame(rows); fmt_summary.to_csv(f'{OUT}/B_format_summary.csv', index=False)
pd.set_option('display.width', 250); fmt_summary"""),
md("""## C. GGUF Q4_K_M for all 3 models
`llama-quantize` is built CPU-only. Only `gguf-py` is installed: llama.cpp's convert requirements pin a CPU-only torch.
llama-cpp-python is built for the T4 (sm_75), with a CPU fallback; the backend actually used is recorded.
Scoring uses the **raw** format, so it is comparable to the run-1 fp32 references."""),
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
print('llama-quantize:', os.path.exists(QUANTIZE), '| torch still', torch.__version__, torch.cuda.is_available())
print(subprocess.run(['git', '-C', LC, 'log', '-1', '--format=%h %cd'], capture_output=True, text=True).stdout)"""),
code("""%%time
try:
    import llama_cpp; GGUF_BACKEND = 'preinstalled'
except ImportError:
    nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    stubs = f'{cuda_root}/lib64/stubs'
    print('nvcc:', nvcc, '| CUDA root:', cuda_root, '| driver stub present:', os.path.exists(f'{stubs}/libcuda.so'))
    cuda_args = (f'-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DCUDAToolkit_ROOT={cuda_root} '
                 f'-DCMAKE_CUDA_COMPILER={nvcc} -DCMAKE_LIBRARY_PATH={stubs}')
    pip_cmd = [sys.executable, '-m', 'pip', 'install', '-q', '--no-cache-dir', '--no-deps', 'llama-cpp-python']
    r = subprocess.run(pip_cmd, env=dict(os.environ, CMAKE_ARGS=cuda_args, FORCE_CMAKE='1',
                       LIBRARY_PATH=stubs + ':' + os.environ.get('LIBRARY_PATH', '')), capture_output=True, text=True)
    GGUF_BACKEND = 'cuda' if r.returncode == 0 else None
    if GGUF_BACKEND is None:
        print('CUDA build failed (tail):\\n', r.stderr[-2500:])
        r = subprocess.run(pip_cmd, env=dict(os.environ, CMAKE_ARGS='-DGGML_CUDA=off', FORCE_CMAKE='1'),
                           capture_output=True, text=True)
        GGUF_BACKEND = 'cpu' if r.returncode == 0 else None
        if GGUF_BACKEND is None: print('CPU build failed too:\\n', r.stderr[-2500:])
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'diskcache', 'jinja2'], check=False)
import llama_cpp
GPU_OK = bool(llama_cpp.llama_supports_gpu_offload())
print('llama-cpp-python', llama_cpp.__version__, '| build:', GGUF_BACKEND, '| GPU offload:', GPU_OK)"""),
code("""from huggingface_hub import snapshot_download, scan_cache_dir
from dialectgap.gguf import load_gguf, score_mcq_gguf
GG = '/kaggle/working/gguf'; os.makedirs(GG, exist_ok=True)
for m, mid in LLMS.items():
    name = f'C_{m}__gguf_q4km'
    if done(name): print('skip', name); continue
    rec = {'model': mid, 'format': 'gguf_q4km', 'build_backend': GGUF_BACKEND, 'gpu_offload': GPU_OK,
           'llama_cpp_python': llama_cpp.__version__}
    q4 = f'{GG}/{m}-Q4_K_M.gguf'
    try:
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
            r = score_mcq_gguf(llm, ps); del llm
            sub = {k: v[:len(ps)] for k, v in ref[m]['single'].items() if k != 'seconds'}
            rec[label] = {**compare_runs(sub, r), 'sec_per_prompt': r['seconds'] / len(ps),
                          'tok_per_s': float(r['n_tokens'].sum() / r['seconds']),
                          'median_letter_mass': float(np.median(letter_mass(r['logprobs']))),
                          'finite': bool(np.isfinite(r['logprobs']).all()),
                          'acc': float((r['logprobs'].argmax(1) == G24[:len(ps)]).mean())}
        rec['ok'] = True
    except Exception as ex:
        rec.update(ok=False, error=f'{type(ex).__name__}: {ex}'[:3000])
    save(name, rec); print(name, json.dumps(rec, indent=1)[:1500])
!df -h /kaggle/working /tmp | tail -2"""),
md("""## D. Qwen3.5 speed with flash-linear-attention (optional)
Run in a fresh Python process, because transformers checks for the kernels when it is imported. If the install fails, this is recorded and skipped."""),
code("""name = 'D_qwen_fla_speed'
if not done(name):
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'flash-linear-attention'], capture_output=True, text=True)
    script = '''
import sys, os, json; sys.path.insert(0, os.path.abspath("src"))
from dialectgap.sanity import run_precision
from dialectgap.models import LLMS
from dialectgap.data import load_belebele_aligned
from dialectgap.prompts import build_mcq_prompt
al = load_belebele_aligned(["eng_Latn","arb_Arab","arz_Arab"])
P = [build_mcq_prompt(it) for v in al for it in al[v][:100]]
rec = run_precision(LLMS["qwen3.5-4b"], "fp16", P, [], batch_size=8, check_batching=False)
print(json.dumps({"ok": rec["ok"], "tok_per_s": rec.get("tokens_per_second_batched"), "error": rec.get("error")}))
'''
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    res = {'pip_ok': r.returncode == 0, 'pip_err': r.stderr[-800:], 'stdout': out.stdout[-1500:], 'stderr': out.stderr[-1500:]}
    save(name, res)
print(json.load(open(f'{OUT}/{name}.json'))['stdout'][-500:])"""),
md("## Report (copy this output back if downloading the zip is inconvenient)"),
code("""import platform, bitsandbytes
print('ENV', platform.python_version(), 'torch', torch.__version__, 'transformers', transformers.__version__,
      'bitsandbytes', bitsandbytes.__version__, 'GPUs', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('\\n=== A: Gemma family quantized with fp32 compute ===')
for p in sorted(glob.glob(f'{OUT}/A_*.json')):
    r = json.load(open(p)); print(os.path.basename(p), r.get('validity'), r.get('vs_fp32'), 'acc', r.get('acc'),
                                  'tok/s', r.get('tokens_per_second_batched'), 'GB', r.get('peak_mem_gb'), r.get('error', '')[:200])
print('\\n=== B: prompt formats ==='); print(fmt_summary.to_csv(index=False))
print('=== C: GGUF ===')
for p in sorted(glob.glob(f'{OUT}/C_*.json')): print(json.dumps(json.load(open(p)))[:1500])
print('\\n=== D ==='); print(open(f'{OUT}/D_qwen_fla_speed.json').read()[-1200:] if done('D_qwen_fla_speed') else 'not run')"""),
md("## Save\nDownload `followup.zip` from the Output tab and unzip it into `results/followup/` in the repo."),
code("""!cd /kaggle/working && zip -qr followup.zip followup && ls -la followup.zip"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
dest = Path(__file__).resolve().parents[1] / "notebooks" / "02_sanity_followup.ipynb"
dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", dest, len(cells), "cells")
