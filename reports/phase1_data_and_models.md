# Phase 1: Datasets, models, and positioning (2026-10-06)

Read-only survey. Nothing was downloaded. Every fact below was checked on the linked dataset card, HF API, or the paper itself. "To verify" marks what could not be confirmed yet (usually because a repo is gated).

## 1. Benchmarks

| Dataset | Arabic varieties (exact codes) | Size | How made | License / access | Role |
|---|---|---|---|---|---|
| **Belebele** ([HF](https://huggingface.co/datasets/facebook/belebele), [Bandarkar et al., ACL 2024](https://arxiv.org/abs/2308.16884)) | `arb_Arab`, `arb_Latn`, `arz_Arab` (Egyptian), `apc_Arab` (N. Levantine), `ary_Arab` (Moroccan), `ars_Arab` (Najdi), `acm_Arab` (Mesopotamian). **No Tunisian.** Plus `eng_Latn` | 900 questions on 488 FLORES passages, 4 options, same items in every variety (paired) | English questions written by humans, then professionally translated; "created end-to-end without the use of machine translation" | CC BY-SA 4.0, not gated | **Main LLM benchmark** (RQ1–3) |
| **FLORES+** ([HF](https://huggingface.co/datasets/openlanguagedata/flores_plus), successor of FLORES-200) | `arb_Arab`, `arb_Latn`, `arz_Arab`, `apc_Arab`, `acm_Arab`, `acq_Arab`, `aeb_Arab`, `ars_Arab`, `ary_Arab`, `apd_Arab` (Sudanese). Whether `ajp_Arab` (S. Levantine) survives is **to verify** (repo is gated) | dev 997 / devtest 1,012 sentences per variety, fully parallel | Professional translation (NLLB) | CC BY-SA 4.0, **gated** (accept terms) | **Embedding benchmark**: bitext retrieval (RQ1, RQ3) |
| **DialectalArabicMMLU** ([HF](https://huggingface.co/datasets/MBZUAI/Dialectal-Arabic-MMLU), [Altakrori et al., LREC 2026](https://arxiv.org/abs/2510.27543)) | ENG, MSA, EGY, KSA, MAG (Moroccan), SYR, UAE | 3,135 Q × 7 = 21,945; shared question IDs (paired); 32 domains | Fully manual, by a language service provider (native translator → native reviewer → adjudicator). Native-speaker check: ~5% of translations had some inaccuracy | CC BY 4.0 | **Second LLM benchmark** (robustness across tasks; replication anchor) |
| **Dial2MSA-Verified** ([GitHub](https://github.com/khered20/Dial2MSA-Verified), [Khered et al., WACL-4 2025](https://aclanthology.org/2025.wacl-1.6.pdf)) | EGY, GLF, LEV, MGR tweets → MSA | EGY: 9,099 train / 200 dev / 2,000 test (3 MSA refs). GLF 6,575 / LEV 4,101 / MGR 3,312 train | Human translators produced the MSA versions; verification by human annotators | No LICENSE file; "for research purposes only" | **Candidate for RQ2c** (LoRA dialect→MSA rewriter) and for evaluating rewriting quality. Do not redistribute |

**Considered and not recommended**
- **MADAR** ([download page](https://camel.abudhabi.nyu.edu/madar-parallel-corpus)): human-made, 25 cities. The license grants internal research use only with **no modification or distribution** rights, and English is not included. LoRA training plus releasing weights is legally unclear. Skip.
- **AraDiCE** ([COLING 2025](https://aclanthology.org/2025.coling-main.283/)): built as **MT + human post-editing**, not human translation from scratch. Outside our "human-translated" rule. Exclude.
- **EgyptianBench / Nile-Chat benchmarks**: LLM-translated (Claude). Exclude.

## 2. Data-quality warning: FLORES/Belebele dialects may be weakly dialectal
The [FLORES-200 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) says that several languages "were translated from ... Modern Standard Arabic", not from English. NLLB describes Arabic languoids as "either translated directly from English or adapted from the Modern Standard Arabic dataset". So some dialect passages may sit close to MSA, which would *understate* the dialect gap.
**Mitigation (automatic, no manual review):** measure each variety's dialectness with the released ALDi model (Keleg et al. 2023; exact HF id and license to verify), plus word overlap with MSA. Report both per variety. This also answers the error-analysis question "does lexical overlap with MSA predict accuracy?".

## 3. Contamination and leakage
- **Contamination.** FLORES (2022) and Belebele (2023) are public and are probably in the pretraining data of 2025–2026 models. MMLU (English) is almost certainly contaminated. DialectalArabicMMLU (Oct 2025) is newer than Gemma-3 but older than Qwen3.5 (Mar 2026) and Gemma 4 (Apr 2026). Embedding models (LaBSE etc.) are trained on mined bitext that may overlap FLORES-like web text.
  *Handling:* our claims are about **paired gaps between varieties**, not absolute scores. We will report each model's release date against each benchmark's release date and state that contamination could inflate English/MSA more than dialects.
- **Leakage (hard rules).** We never train or tune on any FLORES split, because Belebele passages come from FLORES. Few-shot exemplars (RQ2b) come from items disjoint from the evaluated ones (cross-fold), never from the test items. LoRA data (Dial2MSA-Verified) gets an exact and near-duplicate check against all test sets before training.

## 4. Models (proposal)

**Constraint.** RQ3 needs an fp16 baseline on the same GPU. A T4 has 16 GB and no native bf16, so we cap LLMs at ~4–5B total parameters and run in fp16.

| LLM | Params | License | Why |
|---|---|---|---|
| `google/gemma-3-4b-it` | 4.3B | Gemma terms | Multilingual baseline. It is in the DialectalArabicMMLU table, which gives us a **replication anchor** for our scoring pipeline |
| `MBZUAI-Paris/Nile-Chat-4B` | ~4B (Gemma-3-4B base) | Gemma terms | Egyptian-adapted model on the **same base**, so we get a controlled comparison: does dialect adaptation shrink the gap, and does the gain survive quantization? |
| `Qwen/Qwen3.5-4B` (Mar 2026) | 4B | Apache 2.0 | Newest strong multilingual small model ("201 languages and dialects"). Caveats: loads via `AutoModelForMultimodalLM`, thinking mode must be disabled, and its hybrid Gated-DeltaNet layers need a bitsandbytes compatibility smoke test. Fallback: Qwen3.5-2B |
| ~~`tiiuae/Falcon-H1-Arabic-3B`~~ **dropped 2026-10-06** (gated repo, ID unconfirmed) | 3B | Falcon LLM License (permissive, not Apache) | Newest Arabic-centric small model. Hybrid Mamba, so it also needs a quantization smoke test. Fallback: Falcon-H1-3B-Instruct (also in the DialectalArabicMMLU table) |

Not chosen: Gemma 4 E4B (8B total parameters with per-layer embeddings, no fp16 fit). Jais-2-8B and ALLaM-7B (no fp16 fit). Gemma 4 E2B (5.1B total) is a possible swap-in.

| Embedding model | License | Why |
|---|---|---|
| `sentence-transformers/LaBSE` | Apache 2.0 | Classic bitext-mining model, the standard reference for FLORES retrieval |
| `BAAI/bge-m3` | MIT | Strong general multilingual retriever (XLM-R based) |
| `Qwen/Qwen3-Embedding-0.6B` | Apache 2.0 | Recent LLM-based embedder, instruction-aware |

## 5. Evaluation protocol (proposal)
- **Belebele and DA-MMLU scoring.** Zero-shot, log-likelihood scoring in the lm-eval-harness style that DA-MMLU uses. The prompt is passage + question + options A–D + "Answer:", and the prediction is the argmax of log P(" A"/" B"/" C"/" D"). The prompt template stays in English across all varieties, so only the *input variety* changes. This is deterministic with no answer-parsing failures. Chat template off, thinking off.
- **FLORES bitext retrieval.** On devtest, for each variety X: eng→X, X→eng, and MSA→X. Cosine nearest neighbour over all 1,012 candidates, reporting Recall@1 (= accuracy) both directions.
- **Statistics.** Item-level predictions are saved. Paired bootstrap (10k resamples) gives CIs on accuracy and on gaps. Exact McNemar tests cover key paired comparisons, with Holm correction.
- **RQ3.** fp16 vs 8-bit (bitsandbytes LLM.int8) vs 4-bit (NF4) on the same items. We report accuracy drop per variety, peak GPU memory, and latency (ms/item, fixed batch, warm-up excluded).

## 6. Positioning: what is already done
- **RQ1 is largely done for LLMs.** DialectalArabicMMLU (LREC 2026) evaluates 19 open models (1.6B–13B) on ENG / MSA / 5 dialects, fully human-translated and paired. Accuracy consistently drops from ENG → MSA → dialects. On its own, our RQ1 would be a replication.
- **Quantization × multilinguality** is studied: Marchisio et al., [EMNLP Findings 2024](https://arxiv.org/abs/2407.03211) found non-Latin scripts are hurt most. [Alshehhi et al. 2025](https://arxiv.org/abs/2507.19699) covers Arabic models with MSA benchmarks. [Soualhi 2026](https://arxiv.org/abs/2608.09941) covers 4-bit SLMs (Gemma 4, Qwen 3.5) on 8 languages; Arabic-dialect coverage is not stated in the abstract (to verify).
- **Not found: quantization × Arabic dialects**, i.e. whether compression *widens* the dialect gap. Also not found: cheap fixes compared on paired human benchmarks, or the embedding/bitext side of the dialect gap combined with dialectness.

**Recommended sharpening.** Make **RQ3 the headline**: "Does compression widen the Arabic dialect gap?" Treat RQ1 as the paired baseline, replicated on two benchmarks plus embeddings. Keep RQ2 as "which cheap fix survives quantization".
