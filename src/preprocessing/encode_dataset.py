import os
import json
import torch
import yaml
import sentencepiece as spm
from tqdm import tqdm


# -------------------------------------------------
# Load YAML Config
# -------------------------------------------------
def load_config(path="configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Encode individual text with special tokens
# -------------------------------------------------
def encode_text(sp, text, max_len, add_bos=True, add_eos=True):
    ids = sp.encode_as_ids(text)
    # Handle special tokens if they exist
    bos_id = sp.piece_to_id("<bos>") if add_bos and sp.piece_to_id("<bos>") != 0 else None
    eos_id = sp.piece_to_id("<eos>") if add_eos and sp.piece_to_id("<eos>") != 0 else None

    if bos_id is not None:
        ids = [bos_id] + ids
    if eos_id is not None:
        ids = ids + [eos_id]

    # truncate/pad if needed
    ids = ids[:max_len]
    return ids


# -------------------------------------------------
# Encode dataset split (train/val/test)
# -------------------------------------------------
def encode_dataset(config_path="configs/default.yaml"):
    config = load_config(config_path)
    data_cfg = config["data"]
    tok_cfg = config["tokenizer"]

    os.makedirs("data/encoded", exist_ok=True)

    model_path = os.path.join(tok_cfg["save_dir"], tok_cfg["model_prefix"] + ".model")
    sp = spm.SentencePieceProcessor(model_file=model_path)

    max_src = config["max_src_len"]
    max_tgt = config["max_tgt_len"]

    for split in ["train", "val", "test"]:
        file_path = os.path.join(data_cfg["processed_path"], f"{split}.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        enc_inputs, enc_targets = [], []

        print(f"\n🔤 Encoding {split} split ({len(data)} samples)...")
        for sample in tqdm(data, desc=f"Encoding {split}"):
            src_ids = encode_text(sp, sample["input"], max_src)
            tgt_ids = encode_text(sp, sample["target"], max_tgt)
            enc_inputs.append(src_ids)
            enc_targets.append(tgt_ids)

        # Save as torch tensors
        torch.save(
            {"inputs": enc_inputs, "targets": enc_targets},
            f"data/encoded/{split}_encoded.pt"
        )
        print(f"💾 Saved encoded {split} → data/encoded/{split}_encoded.pt")


if __name__ == "__main__":
    encode_dataset()
