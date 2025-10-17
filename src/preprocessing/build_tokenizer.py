import os
import json
import yaml
import sentencepiece as spm
from tqdm import tqdm


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_tokenizer(config_path="configs/default.yaml"):
    config = load_config(config_path)
    tok_cfg = config["tokenizer"]
    data_cfg = config["data"]

    os.makedirs(tok_cfg["save_dir"], exist_ok=True)
    train_file = os.path.join(data_cfg["processed_path"], "train.json")

    # Load training text
    with open(train_file, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    corpus_path = os.path.join(tok_cfg["save_dir"], "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for sample in tqdm(train_data, desc="Building corpus"):
            f.write(sample["input"] + "\n")
            f.write(sample["target"] + "\n")

    model_prefix = os.path.join(tok_cfg["save_dir"], tok_cfg["model_prefix"])

    # ✅ DO NOT include <unk>, since SentencePiece adds it automatically
    user_defined_symbols = ["<pad>", "<bos>", "<eos>", "<sep>"]

    print("🚀 Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=tok_cfg["vocab_size"],
        character_coverage=tok_cfg.get("character_coverage", 1.0),
        model_type="bpe",
        user_defined_symbols=user_defined_symbols,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3
    )
    print(f"✅ Tokenizer saved to {tok_cfg['save_dir']}")


if __name__ == "__main__":
    build_tokenizer()
