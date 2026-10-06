"""Load an LLM at a given precision. Precisions: fp32 | fp16 | bf16 | int8 | nf4."""
import gc

import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig

# Short names used in configs -> HF ids (3 LLMs; Falcon-H1-Arabic dropped 2026-10-06, see reports/phase1_data_and_models.md).
LLMS = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "nilechat-4b": "MBZUAI-Paris/Nile-Chat-4B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
}

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _quant_config(precision: str, compute_dtype: torch.dtype):
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if precision == "nf4":
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)
    return None


def _auto_classes():
    # Several current small models (Gemma 3, Qwen3.5) are multimodal checkpoints; try text-only first.
    names = ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"]
    return [getattr(transformers, n) for n in names if hasattr(transformers, n)]


def load_llm(model_id: str, precision: str, compute_dtype: str = "fp16", device_map="auto"):
    """Returns (model, tokenizer, loader_class_name). compute_dtype applies to int8/nf4 non-quantized parts."""
    cdtype = _DTYPES[compute_dtype]
    kwargs = {"device_map": device_map}
    qcfg = _quant_config(precision, cdtype)
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
        kwargs["dtype"] = cdtype
    else:
        kwargs["dtype"] = _DTYPES[precision]
    tok = AutoTokenizer.from_pretrained(model_id)
    errors = []
    for cls in _auto_classes():
        try:
            model = cls.from_pretrained(model_id, **kwargs)
            model.eval()
            return model, tok, cls.__name__
        except (ValueError, KeyError, TypeError) as e:  # unsupported config for this auto class
            errors.append(f"{cls.__name__}: {e}")
    raise RuntimeError(f"Could not load {model_id}:\n" + "\n".join(errors))


def free():
    """Call after `del model` in the caller to release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
