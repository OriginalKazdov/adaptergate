"""Qwen 2.5 Coder local backend.

Loads Qwen 2.5 Coder Instruct (4-bit quantized) via Unsloth or transformers
and runs inference on CUDA, MPS, or CPU. Supports per-tenant LoRA adapter
hot-swap via PEFT.

Usage:
    backend = QwenLocalBackend(model_name="Qwen/Qwen2.5-Coder-14B-Instruct")
    backend.load()
    text = backend.generate(system_prompt, user_prompt)

Memory: ~10GB at 4-bit for the 14B variant. Inference: ~30-100 tok/s on
consumer GPUs.

Requires the ``adaptergate[ml]`` extra:
    pip install "adaptergate[ml]"

On Apple Silicon, Unsloth's MPS support is rough — the loader falls back to
plain transformers + bitsandbytes 4-bit automatically.
"""

from __future__ import annotations

import torch
from pathlib import Path


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"


class QwenLocalBackend:
    """Local Qwen Coder inference backend."""

    def __init__(self, model_name: str = DEFAULT_MODEL, dtype: str = "auto"):
        self.model_name = model_name
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.device = self._pick_device()

    @staticmethod
    def _pick_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self, use_unsloth: bool = True, load_in_4bit: bool = True) -> None:
        """Load model. Try Unsloth first (faster), fall back to plain transformers."""
        if use_unsloth and self.device != "mps":
            # Unsloth's MPS path is still rough as of mid-2026; CUDA only by default
            try:
                from unsloth import FastLanguageModel
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=8192,
                    dtype=None,
                    load_in_4bit=load_in_4bit,
                )
                FastLanguageModel.for_inference(self.model)
                print(f"Loaded {self.model_name} via Unsloth on {self.device}")
                return
            except Exception as e:
                print(f"Unsloth load failed ({e}), falling back to transformers")

        # transformers fallback
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs = {"trust_remote_code": True}
        if self.device == "mps":
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = {"": "mps"}
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            # Qwen 2.5 (and most modern LLMs) use bfloat16 natively. Setting
            # compute_dtype=bf16 keeps Linear4bit outputs compatible with
            # the rest of the model (embeddings, layernorms, etc.).
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16
            kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self.model.eval()
        print(f"Loaded {self.model_name} via transformers on {self.device}")

    def generate(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate text given system + user prompts via the chat template."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def load_lora_adapter(self, adapter_path: Path, adapter_name: str = "default") -> None:
        """Hot-swap a LoRA adapter onto the loaded base model via PEFT."""
        from peft import PeftModel
        if not hasattr(self.model, "load_adapter"):
            # First adapter: wrap with PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, str(adapter_path), adapter_name=adapter_name
            )
        else:
            self.model.load_adapter(str(adapter_path), adapter_name=adapter_name)

    def set_active_adapter(self, adapter_name: str) -> None:
        """Switch the currently active LoRA adapter. No-op if no adapter loaded."""
        if hasattr(self.model, "set_adapter"):
            self.model.set_adapter(adapter_name)


if __name__ == "__main__":
    # Smoke check: instantiation only (no actual model load — 28GB)
    backend = QwenLocalBackend()
    print(f"backend constructed, device={backend.device}")
    print("To actually load, call backend.load() — will download ~28GB on first run")
