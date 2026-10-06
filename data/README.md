# Data

Only existing, released, human-created or human-translated datasets are used. No collection, scraping, or annotation.
Raw/interim/processed files are not committed (see `.gitignore`); each is downloaded by a script from its official source.

| source | link | varieties used | size | license | used for |
|---|---|---|---|---|---|
| Belebele | https://huggingface.co/datasets/facebook/belebele | eng_Latn, arb_Arab, arz_Arab, apc_Arab, ary_Arab, ars_Arab, acm_Arab, arb_Latn | 900 Q per variety | CC BY-SA 4.0 | LLM eval (RQ1-3) |
| DialectalArabicMMLU | https://huggingface.co/datasets/MBZUAI/Dialectal-Arabic-MMLU | ENG, MSA, EGY, KSA, MAG, SYR, UAE | 3,135 Q per variety | CC BY 4.0 | LLM eval (RQ1-3) |
| FLORES+ | https://huggingface.co/datasets/openlanguagedata/flores_plus (gated) | eng + Arabic varieties (devtest) | 1,012 sentences | CC BY-SA 4.0 | embedding eval only, never training |
| Dial2MSA-Verified | https://github.com/khered20/Dial2MSA-Verified | EGY/GLF/LEV/MGR → MSA | EGY train 9,099 | research purposes only | LoRA (RQ2c) only; **not redistributed, fine-tuned weights not released** |

Pairing facts checked locally (2026-10-06):
- Belebele row order differs between configs; items are paired by `link#question_number` (900 identical uids, 0 gold-answer conflicts across the 8 varieties).
- DialectalArabicMMLU items are paired by `(domain, qid)`: 3,135 complete across all 7 varieties, 0 gold-answer conflicts, 32 domains with 68-100 items each.
