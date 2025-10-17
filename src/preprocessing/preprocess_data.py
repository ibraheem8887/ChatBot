import os
import re
import json
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z0-9?.!,']+", " ", text)
    return text.strip()


def make_input(emotion, situation, utterance):
    return f"Emotion: {emotion} | Situation: {situation} | Customer: {utterance} Agent:"


def preprocess_dataset(config_path="configs/default.yaml"):
    config = load_config(config_path)
    data_cfg = config["data"]

    raw_dir = data_cfg["raw_path"]
    processed_dir = data_cfg["processed_path"]
    os.makedirs(processed_dir, exist_ok=True)

    csv_name = data_cfg.get("file_name", "emotion-emotion_69k.csv")
    raw_file = os.path.join(raw_dir, csv_name)
    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Dataset not found: {raw_file}")

    # Load CSV
    df = pd.read_csv(raw_file)
    print(f"✅ Loaded dataset '{csv_name}' with {len(df)} rows")
    print(f"📊 Columns found: {set(df.columns)}")

    # Rename columns to a consistent format
    rename_map = {
        "empathetic_dialogues": "prompt",  # customer utterance
        "Situation": "context",            # situation text
        "labels": "utterance",             # agent reply
        "emotion": "emotion"               # emotion label
    }
    df = df.rename(columns=rename_map)

    # Drop unnecessary columns
    keep_cols = ["emotion", "context", "prompt", "utterance"]
    df = df[keep_cols]

    print(f"✅ Using columns: {list(df.columns)}")

    # Normalize and build dataset
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        emotion = normalize_text(row["emotion"])
        situation = normalize_text(row["context"])
        utterance = normalize_text(row["prompt"])
        response = normalize_text(row["utterance"])

        records.append({
            "input": make_input(emotion, situation, utterance),
            "target": response
        })

    # Split dataset
    train_ratio = data_cfg.get("train_split", 0.8)
    val_ratio = data_cfg.get("val_split", 0.1)
    seed = config.get("seed", 42)

    train_data, temp_data = train_test_split(records, train_size=train_ratio, random_state=seed)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

    # Save JSONs
    for name, data in zip(["train", "val", "test"], [train_data, val_data, test_data]):
        path = os.path.join(processed_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 Saved {name} set ({len(data)} samples) → {path}")


if __name__ == "__main__":
    preprocess_dataset()
