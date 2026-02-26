import re


def extract_reasoning(text: str) -> str:
    think_end_idx = text.find("</think>")

    if think_end_idx != -1:
        post_text = text[think_end_idx + len("</think>"):]
        match = re.search(r"<Rstart>(.*?)<Rend>", post_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "[NONE]"

    matches = list(re.finditer(r"<Rstart>(.*?)<Rend>", text, re.DOTALL))
    if matches:
        return matches[-1].group(1).strip()

    return "[NONE]"


def is_reasoning_valid(reasoning: str) -> bool:
    lines = [l.strip() for l in reasoning.split("\n") if l.strip()]
    if not lines:
        return False

    if not re.match(r"^R0:\s*\S", lines[0]):
        return False

    if not any(re.match(r"^R1:\s*\S", line) for line in lines[1:]):
        return False

    if not re.match(r"^Final Verdict:\s*\S", lines[-1]):
        return False

    return True