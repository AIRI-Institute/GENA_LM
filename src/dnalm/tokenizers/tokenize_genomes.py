from pathlib import Path
import click
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


def create_bpe_tokenizer(
    input_dir: Path, output_dir: Path, tokenizer_name: str = "BPE"
) -> None:
    """
    Train basic BPE tokenizer without preprocessing with BERT-like special tokens

    Reads all .txt files in given directory, saves
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    files = input_dir.glob("*.txt")
    tokenizer.train([str(file) for file in files], trainer)
    tokenizer.save(str((output_dir / f"{tokenizer_name}.json").resolve()))


@click.command()
@click.option("--input_dir", type=click.Path(path_type=Path, dir_okay=True))
@click.option("--output_dir", type=click.Path(path_type=Path, dir_okay=True))
def cli(input_dir, output_dir):
    output_dir.mkdir(parents=True)
    # create_bpe_tokenizer(input_dir=input_dir, output_dir=output_dir)

if __name__ == "__main__":
    cli()