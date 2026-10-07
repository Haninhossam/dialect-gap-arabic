"""Generates notebooks/03_main_grid.ipynb."""
import json
from pathlib import Path


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
md("""# 03 — Main grid (RQ1 + RQ3)

Scores every (model × format × benchmark × variety × tier) **cell** and saves it on its own as soon as it finishes
(`grid/<model>/<fmt>/<bench>__<variety>__<tier>.json`, with item-level log-probabilities). Run order = blocks B1→B6 (see
`reports/grid_plan.md`). Cells that already exist with `ok=true`, in this session **or in the repo's `results/grid/`**,
are skipped. So each session continues where the previous one stopped, once its outputs have been committed to the repo.

- **Gate 1** (any HF block): Gemma-3-4B fp32 on the 300 FORMAT-DEV prompts. It must succeed and be valid, and it measures throughput.
- **Gate 2** (only for GGUF blocks): llama-cpp-python must really run on the GPU (tiny-model offload test). If not, GGUF cells are skipped.
- Each (model, format) job runs in its own process. The session stops starting new cells after `SESSION_BUDGET_HOURS`.

**Settings:** GPU **T4 x2**, Internet **On**, Secret **`Kaggle_Token`** attached → **Save Version → Save & Run All (Commit)**.
**After the run:** download `grid.zip` from the Output tab and unzip it into `results/grid/` in the repo."""),
code("""REPO_URL = "https://github.com/Haninhossam/dialect-gap-arabic.git"
!git clone -q $REPO_URL || (cd dialect-gap-arabic && git pull -q)
%cd dialect-gap-arabic
!pip install -q -U "transformers>=5.0" accelerate bitsandbytes && pip install -q -e ."""),
code("""# `pip install -e` is not visible to an already-running kernel; put src/ on the path directly.
import sys, os, time, glob, json, subprocess
sys.path.insert(0, os.path.abspath('src'))
import numpy as np, pandas as pd, torch, transformers
from dialectgap.env import load_hf_token
from dialectgap.isolate import mem_available_gb
load_hf_token()
print('torch', torch.__version__, '| transformers', transformers.__version__,
      '| GPUs', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
      f'| RAM available {mem_available_gb():.1f} GB')
print('repo commit:', subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip())"""),
md("## Configuration for THIS session"),
code("""RUN_BLOCKS = ['B1', 'B2', 'B3']   # session 1 = core. Later sessions: ['B4'], then ['B5'], then ['B6'] (see reports/grid_plan.md)
SESSION_BUDGET_HOURS = 10.5        # stop starting new cells after this (Kaggle's hard limit is 12 h per session)
BATCH_SIZE = 8

from dialectgap.env import on_kaggle
OUT = '/kaggle/working/grid' if on_kaggle() else 'results/grid_local'
os.makedirs(OUT, exist_ok=True)
DONE_DIRS = [OUT, 'results/grid'] + glob.glob('/kaggle/input/*/grid')
LOG = f'{OUT}/progress.log'
START = time.time(); DEADLINE = START + SESSION_BUDGET_HOURS * 3600
print('blocks:', RUN_BLOCKS, '| deadline in', SESSION_BUDGET_HOURS, 'h | resume dirs:', DONE_DIRS)"""),
code("""from dialectgap import paths
from dialectgap.grid import ItemStore, blocks, PROMPT_FORMAT, DAMMLU_SUBSET, FORMAT_DEV_N
from dialectgap.gridrun import is_done, make_spec, launch_job
from dialectgap.models import LLMS
store = ItemStore()
bele, dm = store.belebele(), store.dammlu()
meta = {'format_dev_uids': sorted(store.format_dev_uids()), 'dammlu_s50_uids': sorted(store._s50),
        'dammlu_subset': DAMMLU_SUBSET, 'prompt_format': PROMPT_FORMAT, 'format_dev_n': FORMAT_DEV_N}
json.dump(meta, open(f'{OUT}/grid_meta.json', 'w'), indent=1)
plan = [(bid, desc, cs) for bid, desc, cs in blocks() if bid in RUN_BLOCKS]
todo = {bid: [c for c in cs if not is_done(DONE_DIRS, c.key)] for bid, _, cs in plan}
for bid, desc, cs in plan:
    print(f'{bid}: {desc} | {len(cs)} cells, {len(todo[bid])} to run')"""),
md("## Gate 1: Gemma-3-4B fp32 on the 300 FORMAT-DEV prompts (GPU memory fix + throughput)"),
code("""from dialectgap.isolate import run_isolated
from dialectgap.prompts import build_mcq_prompt
from dialectgap.sanity import validity
HF_BLOCKS = any(c.fmt != 'gguf_q4km' for bid in todo for c in todo[bid])
if HF_BLOCKS:
    P300 = [build_mcq_prompt(it) for v in ['eng_Latn', 'arb_Arab', 'arz_Arab'] for it in bele[v][:FORMAT_DEV_N]]
    g1 = run_isolated(LLMS['gemma3-4b'], 'fp32', P300, [P300[0]], f'{OUT}/gate1_gemma_fp32.json',
                      compute_dtype='fp32', batch_size=BATCH_SIZE, check_batching=False, log_path=LOG)
    v1 = validity(g1) if g1['ok'] else {'valid': False}
    print('gate 1:', 'ok' if g1['ok'] else g1['error'], '| validity:', v1,
          '| tok/s:', round(g1.get('tokens_per_second_batched') or 0), '| GPU GB:', g1.get('peak_mem_gb'))
    if not (g1['ok'] and v1['valid']):
        raise SystemExit('GATE 1 FAILED: HF scoring is not safe to run. See gate1_gemma_fp32.json / progress.log')
else:
    print('no HF blocks this session: gate 1 skipped')"""),
md("## Gate 2: llama-cpp-python on the GPU (only when a GGUF block is scheduled)"),
code("""from dialectgap.llamacpp_setup import setup_tools, install_llama_cpp_python, gpu_offload_smoke_test, convert_q4km
GGUF_TODO = [c for bid in todo for c in todo[bid] if c.fmt == 'gguf_q4km']
GGUF_OK, TOOLS = False, None
if GGUF_TODO:
    TOOLS = setup_tools('/kaggle/working/llama.cpp')
    inst = install_llama_cpp_python()
    g2 = gpu_offload_smoke_test(TOOLS, '/kaggle/working/gate2')
    json.dump({'tools': TOOLS, 'install': inst, 'smoke': g2}, open(f'{OUT}/gate2_llamacpp.json', 'w'), indent=1)
    GGUF_OK = bool(g2['ok'])
    print('gate 2:', 'PASSED' if GGUF_OK else 'FAILED', '| backend:', inst.get('backend'), '|', {k: v for k, v in g2.items() if k != 'log_tail'})
    if not GGUF_OK:
        print('GGUF cells will be skipped this session (CPU-only llama.cpp is far too slow for the grid).')
else:
    print('no GGUF cells this session: gate 2 skipped')"""),
md("## Run the blocks"),
code("""from collections import OrderedDict
from huggingface_hub import snapshot_download, scan_cache_dir

def ensure_gguf(m):
    out = f'/kaggle/working/gguf/{m}-Q4_K_M.gguf'
    if not os.path.exists(out):
        src = snapshot_download(LLMS[m])
        convert_q4km(src, out, TOOLS)
        info = scan_cache_dir(); info.delete_revisions(*[r.commit_hash for rp in info.repos if rp.repo_id == LLMS[m] for r in rp.revisions]).execute()
    return out

stopped = False
for bid, desc, _ in plan:
    groups = OrderedDict()
    for c in todo[bid]:
        if c.fmt == 'gguf_q4km' and not GGUF_OK: continue
        groups.setdefault((c.model, c.fmt), []).append(c)
    print(f'=== {bid}: {desc} | {sum(len(v) for v in groups.values())} cells in {len(groups)} jobs', flush=True)
    for (m, f), cs in groups.items():
        cs = [c for c in cs if not is_done(DONE_DIRS, c.key)]
        if not cs: continue
        if time.time() > DEADLINE:
            stopped = True; break
        gguf_path = ensure_gguf(m) if f == 'gguf_q4km' else None
        spec = make_spec(LLMS[m], cs, store, OUT, PROMPT_FORMAT, gguf_path=gguf_path, deadline_epoch=DEADLINE, batch_size=BATCH_SIZE)
        res = launch_job(spec, f'{OUT}/_spec.json', timeout=int(DEADLINE - time.time()) + 3600, log_path=LOG)
        print(f'  {m} {f}: rc={res["returncode"]} done={len(res.get("done", []))} failed={len(res.get("failed", []))} '
              f'wall={res["wall_seconds"]/60:.1f} min', flush=True)
        if res['returncode'] != 0: print('    stderr:', res['stderr_tail'][-800:])
    if stopped:
        print('session budget reached: stopping; rerun next session to continue'); break
print(f'elapsed {(time.time() - START)/3600:.2f} h')"""),
md("## Quick look (full analysis is done offline from the saved item-level files)"),
code("""rows = []
fd = set(meta['format_dev_uids'])
for p in sorted(glob.glob(f'{OUT}/*/*/*.json')):
    r = json.load(open(p))
    if not r.get('ok'):
        rows.append({'key': r.get('key'), 'ok': False, 'error': r.get('error', '')[:100]}); continue
    lp, gold = np.array(r['logprobs']), np.array(r['gold'])
    row = {'key': r['key'], 'ok': True, 'valid': r['valid'], 'n': len(gold), 'acc': r['accuracy'],
           'mass': round(r['median_letter_mass'], 3), 'tok/s': round(r['tok_per_s'])}
    if r['cell']['bench'] == 'belebele':   # report on the 800 held-out items too (FORMAT-DEV excluded)
        keep = np.array([u not in fd for u in r['uids']])
        row['acc_heldout800'] = float((lp[keep].argmax(1) == gold[keep]).mean())
    rows.append(row)
quick = pd.DataFrame(rows); quick.to_csv(f'{OUT}/quick_summary.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400); quick"""),
code("""print(open(LOG).read() if os.path.exists(LOG) else 'no progress.log')"""),
md("## Save\nDownload `grid.zip` from the Output tab and unzip it into `results/grid/` in the repo (I then commit it, so the next session skips these cells)."),
code("""!cd /kaggle/working && rm -f grid/_spec.json && zip -qr grid.zip grid && ls -la grid.zip"""),
]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
dest = Path(__file__).resolve().parents[1] / "notebooks" / "03_main_grid.ipynb"
dest.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", dest, len(cells), "cells")
