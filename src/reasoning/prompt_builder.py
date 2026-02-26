from pathlib import Path
from typing import Generator, Tuple, List, Dict, Any
import json


def load_base_prompt(prompt_dir: Path, n_shot: int) -> str:
    base_prompt_path = prompt_dir / f"{n_shot}-shot-prompt.txt"
    if not base_prompt_path.exists():
        raise FileNotFoundError(f"Base prompt file not found: {base_prompt_path}")

    return base_prompt_path.read_text(encoding="utf-8").strip()


def format_claim_block(
    claim: str,
    label: str,
    evidence_text: List[str],
) -> str:
    evidence_block = ",\n".join(f'      "{et}"' for et in evidence_text)

    return (
        f'Claim: "{claim}"\n'
        f'Label: "{label}"\n'
        f"Evidence Text: [\n"
        f"{evidence_block}\n"
        "]\n"
    )


def iter_prompts(
    dataset_path: Path,
    prompt_dir: Path,
    n_shot: int = 6,
) -> Generator[Tuple[str, str], None, None]:
    """
    Yields formatted prompts and corresponding claim IDs.
    """

    base_prompt = load_base_prompt(prompt_dir, n_shot)

    with dataset_path.open("r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    for entry in data:
        claim_id = entry.get("unique_claim_id", "")
        claim = entry.get("claim", "")
        label = entry.get("mapped_verdict", "")
        evidence_text = entry.get("evidence_text", [])

        claim_block = format_claim_block(claim, label, evidence_text)

        prompt = (
            base_prompt
            + "\n\n"
            + claim_block
            + "\nPlease provide the reasoning traces. "
              "Ensure that the reasoning output is between <Rstart> and <Rend>."
        )

        yield prompt, claim_id