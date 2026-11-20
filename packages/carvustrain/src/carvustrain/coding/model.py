"""Offline causal language model backend."""
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory


def weight_size(path: Path) -> int:
    size = sum(p.stat().st_size for p in path.glob("*.safetensors"))
    if not size or not (path / "config.json").is_file():
        raise ValueError(f"Expected a local model with config.json and safetensors weights at {path}")
    return size


class CodingModel:
    def __init__(self, path: Path, device: str, max_tokens: int, training: bool = False):
        size = weight_size(path)
        try:
            import psutil
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ValueError("Install the coding extra with pip install '.[coding]'") from exc
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable. Select CPU with a smaller local model.")
        available = psutil.virtual_memory().available
        if device == "cuda":
            available = min(available, torch.cuda.mem_get_info()[0])
        # Conservative proxy for FP32 conversion, gradients, loading and activations.
        required = size * (10 if training else 5) + 256 * 1024**2
        if available < required:
            raise ValueError(
                f"Estimated memory requirement {required / 1024**3:.2f} GiB; "
                f"{available / 1024**3:.2f} GiB available. Try a smaller base model."
            )
        self.torch = torch
        self.device = device
        options = dict(local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(path, **options)
        if config.is_encoder_decoder or getattr(config, "quantization_config", None):
            raise ValueError("Select an unquantized decoder-only causal language model")
        self.tokenizer = AutoTokenizer.from_pretrained(path, **options)
        self.network, loading = AutoModelForCausalLM.from_pretrained(
            path, config=config, use_safetensors=True, dtype=torch.float32, output_loading_info=True, **options,
        )
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError("Selected checkpoint is missing compatible model weights")
        self.network.to(device)
        self.parameter_count = sum(parameter.numel() for parameter in self.network.parameters())
        self.max_tokens = min(max_tokens, getattr(self.network.config, "max_position_embeddings", max_tokens))

    def start_training(self, learning_rate: float):
        self.network.train()
        self.optimizer = self.torch.optim.SGD(self.network.parameters(), lr=learning_rate)

    def prompt_tokens(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(f"Question: {prompt}\nAnswer:")

    def train_step(self, row: dict[str, str]) -> float:
        prefix = self.prompt_tokens(row["prompt"])
        target = self.tokenizer.encode(" " + row["completion"], add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            target.append(self.tokenizer.eos_token_id)
        if not target or len(prefix) + len(target) > self.max_tokens:
            raise ValueError(f"Training example must fit in {self.max_tokens} tokens with a nonempty completion")
        ids = self.torch.tensor([prefix + target], device=self.device)
        labels = ids.clone()
        labels[:, :len(prefix)] = -100
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.network(
            input_ids=ids, attention_mask=self.torch.ones_like(ids), labels=labels, use_cache=False,
        ).loss
        if not self.torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        return loss.item()

    def save(self, destination: Path):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=destination.parent) as temporary:
            staged = Path(temporary) / "checkpoint"
            self.network.save_pretrained(staged, safe_serialization=True)
            self.tokenizer.save_pretrained(staged)
            staged.rename(destination)

    def answer(self, prompt: str) -> str:
        tokens = self.prompt_tokens(prompt)
        budget = min(128, self.max_tokens - len(tokens))
        if budget < 1:
            raise ValueError(f"Prompt and memory exceed the {self.max_tokens}-token context")
        ids = self.torch.tensor([tokens], device=self.device)
        self.network.eval()
        with self.torch.inference_mode():
            result = self.network.generate(
                ids, attention_mask=self.torch.ones_like(ids), max_new_tokens=budget,
                do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(result[0, len(tokens):], skip_special_tokens=True).strip()
