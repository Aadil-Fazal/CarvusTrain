"""Build a small, locally initialized model for verifying the coding workflow."""

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from .learn_brain import LearningBrain
from .module_call import create, read_dataset
from .schemas import TrainingConfig


def build_test_base(destination: Path) -> None:
    """Save deterministic initial weights and a tokenizer without network access."""
    try:
        import torch
        from tokenizers import Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    except ImportError as exc:
        raise ValueError("Install the coding runtime from Setup first.") from exc

    if destination.exists():
        raise FileExistsError(f"Test model already exists at {destination}")
    rows = read_dataset(destination.parent / "data.cl")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    tokenizer.train_from_iterator(
        (f"Question: {row['prompt']}\nAnswer: {row['completion']}" for row in rows),
        trainer=BpeTrainer(vocab_size=384, initial_alphabet=ByteLevel.alphabet(),
                           special_tokens=["[UNK]", "[EOS]"], show_progress=False),
    )
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staged = Path(temporary) / "base"
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]",
        ).save_pretrained(staged)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4)
            model = GPT2LMHeadModel(GPT2Config(
                vocab_size=tokenizer.get_vocab_size(), n_positions=64, n_embd=16, n_layer=1, n_head=2,
                eos_token_id=1, bos_token_id=1, pad_token_id=1,
            ))
        model.save_pretrained(staged, safe_serialization=True)
        staged.rename(destination)


def train_test_model(
    root: Path, config: TrainingConfig, *, on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Create a new CPU workspace and train a test checkpoint locally."""
    config.validate()
    base = root / "base"
    create(root, "CarvusTrain Test Model", base, max_tokens=64)
    build_test_base(base)
    return LearningBrain(root).train(config, on_progress=on_progress)
