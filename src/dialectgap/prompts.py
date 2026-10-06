"""Prompt templates. The instruction stays in English for every variety, so only the input variety changes."""

LETTERS = ["A", "B", "C", "D"]

_MCQ_WITH_PASSAGE = (
    "The following is a multiple-choice reading comprehension question. "
    "Read the passage and choose the correct answer.\n\n"
    "Passage: {passage}\n\n"
    "Question: {question}\n"
    "{options}\n"
    "Answer:"
)

_MCQ_NO_PASSAGE = (
    "The following is a multiple-choice question. Choose the correct answer.\n\n"
    "Question: {question}\n"
    "{options}\n"
    "Answer:"
)


def build_mcq_prompt(item: dict) -> str:
    options = "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, item["options"]))
    template = _MCQ_WITH_PASSAGE if item.get("passage") else _MCQ_NO_PASSAGE
    return template.format(passage=item.get("passage", ""), question=item["question"], options=options)


# Prompt formats. Each fixes (a) how the MCQ prompt is wrapped and (b) which letter tokens are scored:
#   raw                  plain text ending in "Answer:"; next token is " A" (with space). lm-eval default.
#   chat_nothink         model chat template, generation prompt, thinking disabled (Qwen3.5 inserts an
#                        empty <think></think> block); the answer starts the assistant turn: "A" (no space).
#   raw_prefill_nothink  raw prompt + the empty think block Qwen3.5 emits on its own; next token "A".
FORMATS = {
    "raw": {"letter_prefix": " ", "add_special_tokens": True},
    "chat_nothink": {"letter_prefix": "", "add_special_tokens": False},
    "raw_prefill_nothink": {"letter_prefix": "", "add_special_tokens": True},
}

_EMPTY_THINK = "\n\n<think>\n\n</think>\n\n"


def format_prompt(prompt: str, tokenizer, fmt: str = "raw") -> str:
    if fmt == "raw":
        return prompt
    if fmt == "raw_prefill_nothink":
        return prompt + _EMPTY_THINK
    if fmt == "chat_nothink":
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    raise ValueError(f"unknown prompt format: {fmt}")
