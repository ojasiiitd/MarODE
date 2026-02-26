from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HFGenerator:
    def __init__(
        self,
        model_path: str,
        gpu_index: Optional[int] = None,
        torch_dtype=torch.float16,
    ):
        self.device = (
            torch.device(f"cuda:{gpu_index}")
            if gpu_index is not None and torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
        ).to(self.device)

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2000,
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
    ) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)