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
