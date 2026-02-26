import argparse
import json
from pathlib import Path
from tqdm import tqdm

from src.reasoning.prompt_builder import iter_prompts
from src.reasoning.generator import HFGenerator
from src.reasoning.extraction import extract_reasoning, is_reasoning_valid


def run(
    model_path: str,
    dataset_path: Path,
    prompt_dir: Path,
    output_dir: Path,
    n_shot: int,
    gpu_index: int | None,
    batch_size: int = 100,
    max_attempts: int = 5,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = HFGenerator(model_path=model_path, gpu_index=gpu_index)

    results = []
    file_counter = 1

    for prompt, claim_id in tqdm(
        iter_prompts(dataset_path, prompt_dir, n_shot=n_shot),
        desc="Generating reasoning traces",
    ):

        reasoning_trace = "[NONE]"
        attempts = 0

        while reasoning_trace == "[NONE]" and attempts < max_attempts:
            response = generator.generate(prompt)
            reasoning_trace = extract_reasoning(response)

            if not is_reasoning_valid(reasoning_trace):
                reasoning_trace = "[NONE]"

            attempts += 1

        if attempts >= max_attempts and reasoning_trace == "[NONE]":
            reasoning_trace = "[EXCEEDED MAX ATTEMPTS]"

        results.append(
            {
                "shots": n_shot,
                "claim_id": claim_id,
                "reasoning_trace": reasoning_trace,
            }
        )

        if len(results) == batch_size:
            output_path = output_dir / f"traces_{file_counter}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            results = []
            file_counter += 1

    if results:
        output_path = output_dir / f"traces_{file_counter}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Run reasoning generation.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-shot", type=int, default=2)
    parser.add_argument("--gpu-index", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run(
        model_path=args.model_path,
        dataset_path=args.dataset,
        prompt_dir=args.prompt_dir,
        output_dir=args.output_dir,
        n_shot=args.n_shot,
        gpu_index=args.gpu_index,
    )