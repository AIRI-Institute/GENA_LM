import itertools
from pathlib import Path
from typing import Optional
import click
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

DEFAULT_VOCAB_SIZE = 32_000


def create_bpe_tokenizer(
    input_dir: Path,
    output_dir: Path,
    tokenizer_name: str = "BPE",
    limit_files: Optional[int] = None,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> None:
    """
    Train basic BPE tokenizer without preprocessing with BERT-like special tokens

    Reads all .txt files in given directory, saves
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(vocab_size = vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    files = input_dir.glob("*.txt")
    if limit_files is not None or limit_files != 0:
        files = itertools.islice(files, 0, limit_files)

    tokenizer.train([str(file) for file in files], trainer)
    tokenizer.save(str((output_dir / f"{tokenizer_name}.json").resolve()))


@click.command()
@click.option("--input_dir", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--output_dir", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--limit-files", type=int, )
@click.option("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE, show_default=True)
@click.option("--tokenizer_name", default="BPE", show_default=True)
def cli(input_dir, output_dir, limit_files, vocab_size, tokenizer_name):
    if output_dir is None:
        output_dir = Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)
    create_bpe_tokenizer(
        input_dir=input_dir,
        output_dir=output_dir,
        limit_files=limit_files,
        vocab_size=vocab_size,
        tokenizer_name=tokenizer_name,
    )


if __name__ == "__main__":
    cli()
