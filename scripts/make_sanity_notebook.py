"""Generates notebooks/01_sanity_precision.ipynb (kept as code so the notebook is reviewable in diffs)."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
md("""# 01 — Precision sanity check (fp32 vs fp16 / bf16 / int8 / nf4, + GGUF feasibility)

**Why:** T4 has no native bf16, and Gemma-family models can overflow in fp16. Before any experiment we check, per model,
that each precision agrees with an **fp32 reference** on the same prompts.

**Stability rule (fixed before running, in `dialectgap/sanity.py`):** stable ⇔ no non-finite logits **and** argmax agreement
with fp32 ≥ 95% **and** mean 4-way KL < 0.01. We also check that batched (left-padded) scoring equals one-at-a-time scoring
for every model (matters for Qwen3.5's Gated-DeltaNet layers).

**Settings:** Accelerator **GPU T4 x2** (the fp32 reference of a ~4B model needs ~17 GB, so it is split across both GPUs),
Internet on, Kaggle secret **`Kaggle_Token`** holding your Hugging Face token (Gemma / FLORES+ terms accepted on huggingface.co).
Results are saved after each configuration, so a rerun resumes where it stopped."""),
code("""REPO_URL = "https://github.com/Haninhossam/dialect-gap-arabic.git"
!git clone -q $REPO_URL || (cd dialect-gap-arabic && git pull -q)
%cd dialect-gap-arabic
!pip install -q -U "transformers>=5.0" accelerate bitsandbytes sentence-transformers && pip install -q -e ."""),
code("""import os, json, glob, torch, transformers
import numpy as np, pandas as pd
from dialectgap.env import load_hf_token
load_hf_token()   # reads Kaggle secret 'Kaggle_Token' (or 'HF_TOKEN'), verifies it and Gemma access
print('torch', torch.__version__, '| transformers', transformers.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i); print(i, p.name, f'{p.total_memory/1e9:.1f} GB')
if torch.cuda.device_count() < 2:
    print('WARNING: fewer than 2 GPUs. Choose Accelerator = GPU T4 x2; otherwise the fp32 reference offloads to CPU (much slower).')
OUT = '/kaggle/working/sanity' if os.path.exists('/kaggle') else 'results/sanity'
os.makedirs(OUT, exist_ok=True)"""),
md("## Prompts: 8 Belebele questions (paired by uid) in English, MSA and Egyptian (same items → 24 prompts)"),
code("""from dialectgap.data import load_belebele, load_belebele_aligned
from dialectgap.prompts import build_mcq_prompt
VARS = ['eng_Latn', 'arb_Arab', 'arz_Arab']
# Row order differs between Belebele configs: always pair by uid (the loader checks items and gold answers)
aligned = load_belebele_aligned(VARS)
items = {v: aligned[v][:8] for v in VARS}
prompts = [build_mcq_prompt(it) for v in VARS for it in items[v]]
gen_prompts = [prompts[16], prompts[17]]  # two Egyptian prompts for a greedy-generation check
print(len(prompts), 'prompts'); print(prompts[16][:400])"""),
md("## LLMs × precisions"),
code("""from dialectgap.models import LLMS
from dialectgap.sanity import run_precision, verdict, to_jsonable, from_jsonable
PRECISIONS = ['fp32', 'fp16', 'bf16', 'int8', 'nf4']   # bf16 is emulated on T4: recorded for completeness

def path(m, p): return f'{OUT}/{m}__{p}.json'

def free_model_cache(model_id):
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    revs = [r.commit_hash for repo in info.repos if repo.repo_id == model_id for r in repo.revisions]
    if revs:
        info.delete_revisions(*revs).execute()

for short, mid in LLMS.items():
    for prec in PRECISIONS:
        if os.path.exists(path(short, prec)):
            print('skip (done)', short, prec); continue
        print('running', short, prec, flush=True)
        rec = run_precision(mid, prec, prompts, gen_prompts, compute_dtype='fp16')
        json.dump(to_jsonable(rec), open(path(short, prec), 'w'), ensure_ascii=False)
        print('  ok' if rec['ok'] else '  FAILED: ' + rec['error'][:300])
    if short != 'gemma3-4b':   # free disk (~8 GB per model); Gemma is reused below for the GGUF test
        free_model_cache(mid)
    !df -h /root | tail -1"""),
md("If a model is unstable in fp16, its int8/nf4 runs above also used fp16 compute. Re-run them with fp32 compute to separate the two effects:"),
code("""UNSTABLE_FP16 = []   # fill after reading the summary below, e.g. ['gemma3-4b']
for short in UNSTABLE_FP16:
    for prec in ['int8', 'nf4']:
        p = path(short, prec + '-c32')
        if not os.path.exists(p):
            rec = run_precision(LLMS[short], prec, prompts, gen_prompts, compute_dtype='fp32')
            json.dump(to_jsonable(rec), open(p, 'w'), ensure_ascii=False)"""),
md("## Summary"),
code("""rows = []
for short in LLMS:
    ref_p = path(short, 'fp32')
    if not os.path.exists(ref_p): continue
    ref = from_jsonable(json.load(open(ref_p)))
    for p in sorted(glob.glob(f'{OUT}/{short}__*.json')):
        prec = p.split('__')[1][:-5]
        rec = from_jsonable(json.load(open(p)))
        v = verdict(ref, rec) if prec != 'fp32' else {'stable': ref.get('ok')}
        bvs = rec.get('batch_vs_single') or {}
        rows.append({'model': short, 'precision': prec, 'ok': rec.get('ok'), 'stable_vs_fp32': v.get('stable'),
                     'argmax_agree': v.get('argmax_agreement'), 'mean_kl': v.get('mean_kl_4way'),
                     'max_abs_dlogp': v.get('max_abs_logprob_diff'), 'nonfinite': v.get('n_nonfinite_test'),
                     'batch==single': bvs.get('argmax_agreement'), 'batch_kl': bvs.get('mean_kl_4way'),
                     'tok_per_s': rec.get('tokens_per_second_batched'), 'peak_GB': rec.get('peak_mem_gb'),
                     'greedy_same': v.get('greedy_text_identical'), 'error': (rec.get('error') or '')[:120]})
summary = pd.DataFrame(rows); summary.to_csv(f'{OUT}/summary.csv', index=False)
pd.set_option('display.width', 250); summary"""),
code("""# Generations side by side: degraded text is a red flag even when the argmax agrees
for short in LLMS:
    for prec in PRECISIONS:
        p = path(short, prec)
        if os.path.exists(p):
            print(f'--- {short} {prec}:', json.load(open(p)).get('generations'))"""),
md("""## Embedding models: fp32 vs fp16 (int8 is checked in Phase 2)
Uses Belebele passages (these *are* FLORES sentences), so the FLORES+ gate is not needed here.
Each passage is identified by its source link, so English and Egyptian passages are aligned."""),
code("""from sentence_transformers import SentenceTransformer
EMB = {'labse': 'sentence-transformers/LaBSE', 'bge-m3': 'BAAI/bge-m3', 'qwen3-emb-0.6b': 'Qwen/Qwen3-Embedding-0.6B'}
eng_all = {it['uid'].split('#')[0]: it['passage'] for it in load_belebele('eng_Latn')}
arz_all = {it['uid'].split('#')[0]: it['passage'] for it in load_belebele('arz_Arab')}
keys = list(eng_all)[:100]; E = [eng_all[k] for k in keys]; A = [arz_all[k] for k in keys]
emb_rows = []
for short, mid in EMB.items():
    res = {}
    for dt in ['fp32', 'fp16']:
        try:
            m = SentenceTransformer(mid, device='cuda',
                                    model_kwargs={'torch_dtype': torch.float32 if dt == 'fp32' else torch.float16})
            e, a = m.encode(E, normalize_embeddings=True), m.encode(A, normalize_embeddings=True)
            res[dt] = a @ e.T; del m; torch.cuda.empty_cache()
        except Exception as ex:
            res[dt] = f'{type(ex).__name__}: {ex}'
    ok = all(isinstance(r, np.ndarray) for r in res.values())
    emb_rows.append({'model': short, 'ok': ok,
        'nonfinite_fp16': int((~np.isfinite(res['fp16'])).sum()) if ok else None,
        'max_abs_cos_diff': float(np.abs(res['fp32'] - res['fp16']).max()) if ok else None,
        'retrieval_argmax_agree': float((res['fp32'].argmax(1) == res['fp16'].argmax(1)).mean()) if ok else None,
        'R@1_fp32_arz->eng_100': float((res['fp32'].argmax(1) == np.arange(len(keys))).mean()) if ok else None,
        'error': '' if ok else str([r for r in res.values() if isinstance(r, str)])[:200]})
emb_summary = pd.DataFrame(emb_rows); emb_summary.to_csv(f'{OUT}/embedding_summary.csv', index=False); emb_summary"""),
md("""## GGUF Q4_K_M feasibility (edge format), Gemma-3-4B only
Builds llama.cpp with CUDA, converts the HF checkpoint to GGUF, quantizes to Q4_K_M, then scores the same 24 prompts with
llama-cpp-python on GPU, plus 8 prompts on CPU only (edge-like latency). Steps are timed to know the per-session overhead."""),
code("""%%time
!git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
!cmake -S /kaggle/working/llama.cpp -B /kaggle/working/llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF > /dev/null
!cmake --build /kaggle/working/llama.cpp/build --config Release -j 4 --target llama-quantize > /dev/null
!pip install -q -r /kaggle/working/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"""),
code("""%%time
from huggingface_hub import snapshot_download
src = snapshot_download(LLMS['gemma3-4b'])
GG = '/kaggle/working/gguf'; os.makedirs(GG, exist_ok=True)
!python /kaggle/working/llama.cpp/convert_hf_to_gguf.py {src} --outtype f16 --outfile {GG}/gemma3-4b-f16.gguf
!/kaggle/working/llama.cpp/build/bin/llama-quantize {GG}/gemma3-4b-f16.gguf {GG}/gemma3-4b-Q4_K_M.gguf Q4_K_M"""),
code("""%%time
!CMAKE_ARGS="-DGGML_CUDA=on" pip install -q llama-cpp-python"""),
code("""from dialectgap.gguf import load_gguf, score_mcq_gguf
from dialectgap.scoring import compare_runs
ref = from_jsonable(json.load(open(path('gemma3-4b', 'fp32'))))['single']
gg = {}
for name, ngl, ps in [('gpu', -1, prompts), ('cpu', 0, prompts[:8])]:
    try:
        llm = load_gguf(f'{GG}/gemma3-4b-Q4_K_M.gguf', n_gpu_layers=ngl, n_threads=os.cpu_count())
        r = score_mcq_gguf(llm, ps); del llm
        sub = {k: v[:len(ps)] for k, v in ref.items() if k != 'seconds'}
        gg[name] = {**compare_runs(sub, r), 'tok_per_s': float(r['n_tokens'].sum() / r['seconds']), 'ok': True}
    except Exception as ex:
        gg[name] = {'ok': False, 'error': f'{type(ex).__name__}: {ex}'}
json.dump(gg, open(f'{OUT}/gguf_gemma3-4b_Q4_K_M.json', 'w')); gg"""),
md("## Report (copy this cell's output back if downloading the zip is inconvenient)"),
code("""import platform, bitsandbytes, sentence_transformers
print('ENV', platform.python_version(), 'torch', torch.__version__, 'transformers', transformers.__version__,
      'bitsandbytes', bitsandbytes.__version__, 'sentence-transformers', sentence_transformers.__version__,
      'GPUs', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print('\\n=== LLM SUMMARY ===');       print(summary.to_csv(index=False))
print('=== EMBEDDING SUMMARY ===');     print(emb_summary.to_csv(index=False))
print('=== GGUF ===');                  print(json.dumps(gg, indent=1) if 'gg' in globals() else 'not run')"""),
md("## Save\nDownload `sanity.zip` from the Output tab and unzip it into `results/sanity/` in the repo."),
code("""!cd /kaggle/working && zip -qr sanity.zip sanity && ls -la sanity.zip"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
dest = Path(__file__).resolve().parents[1] / "notebooks" / "01_sanity_precision.ipynb"
dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", dest, len(cells), "cells")
