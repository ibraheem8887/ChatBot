import os
import math
import yaml
import time
import torch
import random
import sentencepiece as spm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from collections import Counter

# adjust import path to your transformer implementation
from src.model.transformer import Transformer

# -----------------------------
# Utilities
# -----------------------------
def load_config(path="configs/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------
# Dataset wrapper
# -----------------------------
class EncodedDataset(Dataset):
    def __init__(self, encoded_pt_path):
        data = torch.load(encoded_pt_path)
        self.inputs = data["inputs"]
        self.targets = data["targets"]
        assert len(self.inputs) == len(self.targets)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.targets[idx], dtype=torch.long)

# -----------------------------
# Collate fn: pads, returns src, tgt_in, tgt_out, lengths
# -----------------------------
def collate_fn(batch, pad_idx, max_src_len=None, max_tgt_len=None):
    # batch: list of (src_tensor, tgt_tensor)
    srcs = [b[0][:max_src_len] if max_src_len else b[0] for b in batch]
    tgts = [b[1][:max_tgt_len] if max_tgt_len else b[1] for b in batch]

    # We prepare teacher forcing targets:
    # tgt_input = [bos] + tokens... (already included if tokenizer added bos)
    # For stability we will use the sequences as-is; shift inside training step
    src_pad = pad_sequence(srcs, batch_first=True, padding_value=pad_idx)  # (B, S)
    tgt_pad = pad_sequence(tgts, batch_first=True, padding_value=pad_idx)  # (B, T)

    return {
        "src": src_pad,
        "tgt": tgt_pad
    }

# -----------------------------
# Greedy generation (autoregressive)
# -----------------------------
@torch.no_grad()
def generate_greedy(model, sp, src_tensor, max_tgt_len, pad_idx, bos_id=None, eos_id=None, device="cpu"):
    """
    src_tensor: (1, src_len) tensor on device
    Returns token id list (without bos if present)
    """
    model.eval()
    src = src_tensor.to(device)
    # start with BOS if tokenizer has it else use first token of target if present
    if bos_id is None:
        try:
            bos_id = sp.piece_to_id("<bos>")
        except:
            bos_id = None
    if eos_id is None:
        try:
            eos_id = sp.piece_to_id("<eos>")
        except:
            eos_id = None

    # prepare initial tgt sequence: start with bos if available, else with pad (but better to require bos)
    if bos_id is not None:
        cur = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # (1,1)
    else:
        # fallback: take pad id then overwrite tokens
        cur = torch.tensor([[pad_idx]], dtype=torch.long, device=device)

    for _ in range(max_tgt_len):
        logits = model(src, cur)  # (1, cur_len, vocab)
        # get last time-step logits
        next_logits = logits[:, -1, :]  # (1, vocab)
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # (1,1)
        # append
        cur = torch.cat([cur, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break

    # remove initial BOS if present
    res = cur.squeeze(0).tolist()
    if bos_id is not None and len(res) and res[0] == bos_id:
        res = res[1:]
    # if eos present, cut at eos (exclusive)
    if eos_id is not None and eos_id in res:
        res = res[:res.index(eos_id)]
    return res

# -----------------------------
# Simple corpus-level BLEU (uses tokenized sequences)
# Fallback implementation: calculates n-gram precision with brevity penalty
# -----------------------------
def compute_corpus_bleu(references, hypotheses, max_n=4):
    """
    references: list of list of token ids (refs)
    hypotheses: list of list of token ids (hyps)
    returns: BLEU-like score (0..100)
    """
    weights = [1.0 / max_n] * max_n
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for ref, hyp in zip(references, hypotheses):
        ref_ngrams = [Counter(ngrams(ref, n)) for n in range(1, max_n + 1)]
        hyp_ngrams = [Counter(ngrams(hyp, n)) for n in range(1, max_n + 1)]
        for i in range(max_n):
            total = sum(hyp_ngrams[i].values())
            total_counts[i] += total
            if total == 0:
                continue
            # clipped count
            clip = 0
            for gram, cnt in hyp_ngrams[i].items():
                clip += min(cnt, ref_ngrams[i].get(gram, 0))
            clipped_counts[i] += clip

    precisions = []
    for i in range(max_n):
        if total_counts[i] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[i] / total_counts[i])

    # geometric mean
    if min(precisions) == 0:
        geo_mean = 0.0
    else:
        import math
        geo_mean = math.exp(sum((1.0 / max_n) * math.log(p) for p in precisions if p > 0))

    # brevity penalty
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    bleu = bp * geo_mean
    return bleu * 100.0

# small helper ngrams
def ngrams(seq, n):
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)] if len(seq) >= n else []

# -----------------------------
# Training / Validation loops
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, clip_norm, config):
    model.train()
    total_loss = 0.0
    it = 0
    for batch in tqdm(dataloader, desc="Train batches"):
        src = batch["src"].to(device)  # (B, S)
        tgt = batch["tgt"].to(device)  # (B, T)

        # prepare decoder input and target:
        # teacher forcing: input = tgt[:, :-1], target = tgt[:, 1:]
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        optimizer.zero_grad()
        logits = model(src, tgt_in)  # (B, T-1, V)
        V = logits.size(-1)
        # compute loss: flatten
        logits_flat = logits.reshape(-1, V)
        tgt_flat = tgt_out.reshape(-1)

        loss = criterion(logits_flat, tgt_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        total_loss += loss.item()
        it += 1
    return total_loss / max(1, it)


def validate(model, dataloader, criterion, device, sp, config, max_val_samples=200):
    model.eval()
    total_loss = 0.0
    it = 0
    refs = []
    hyps = []
    max_tgt_len = config["max_tgt_len"]
    pad_idx = config.get("pad_idx", 3)

    # Use the safe_id function to get bos_id and eos_id
    def safe_id(sp, token):
        idx = sp.piece_to_id(token)
        return idx if idx != -1 else None

    bos_id = safe_id(sp, "<bos>")
    eos_id = safe_id(sp, "<eos>")

    for i, batch in enumerate(tqdm(dataloader, desc="Validation batches")):
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.no_grad():
            logits = model(src, tgt_in)
            V = logits.size(-1)
            logits_flat = logits.reshape(-1, V)
            tgt_flat = tgt_out.reshape(-1)
            loss = criterion(logits_flat, tgt_flat)

        total_loss += loss.item()
        it += 1

        # greedy decode a small number for BLEU (to save time)
        if len(hyps) < max_val_samples:
            b = src.size(0)
            for bi in range(b):
                src_i = src[bi:bi+1]  # (1, S)
                hyp_ids = generate_greedy(model, sp, src_i, max_tgt_len, pad_idx,
                                          bos_id=bos_id,
                                          eos_id=eos_id,
                                          device=device)
                ref_ids = tgt[bi].tolist()
                # remove initial bos if present in ref
                if bos_id is not None and len(ref_ids) and ref_ids[0] == bos_id:
                    ref_ids = ref_ids[1:]
                # cut at eos if present
                if eos_id is not None and eos_id in ref_ids:
                    ref_ids = ref_ids[:ref_ids.index(eos_id)]
                # likewise, hyps already trimmed in generator
                refs.append(ref_ids)
                hyps.append(hyp_ids)

    avg_loss = total_loss / max(1, it)
    # compute BLEU using simple fallback function (token-ids)
    bleu = compute_corpus_bleu(refs, hyps) if len(hyps) > 0 else 0.0
    return avg_loss, bleu

# -----------------------------
# Main training orchestration
# -----------------------------
def main(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device","cpu") == "cuda" else "cpu")

    # tokenizer
    tok_dir = cfg["tokenizer"]["save_dir"]
    sp_model = os.path.join(tok_dir, cfg["tokenizer"]["model_prefix"] + ".model")
    sp = spm.SentencePieceProcessor(model_file=sp_model)

    # Safer token ID handling
    def safe_id(sp, token):
        idx = sp.piece_to_id(token)
        return idx if idx != -1 else None

    pad_idx = safe_id(sp, "<pad>") or cfg.get("pad_idx", 3)
    bos_id = safe_id(sp, "<bos>")
    eos_id = safe_id(sp, "<eos>")

    cfg["pad_idx"] = pad_idx

    # dataset paths
    encoded_dir = "data/encoded"
    train_ds = EncodedDataset(os.path.join(encoded_dir, "train_encoded.pt"))
    val_ds = EncodedDataset(os.path.join(encoded_dir, "val_encoded.pt"))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_idx, cfg.get("max_src_len"), cfg.get("max_tgt_len")))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_idx, cfg.get("max_src_len"), cfg.get("max_tgt_len")))

    # model
    model = Transformer(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        enc_layers=cfg["enc_layers"],
        dec_layers=cfg["dec_layers"],
        dropout=cfg["dropout"],
        pad_idx=pad_idx,
        max_len=max(cfg["max_src_len"], cfg["max_tgt_len"])
    ).to(device)

    # optimizer + scheduler (linear warmup)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), betas=(0.9, 0.98), eps=1e-9)
    total_steps = math.ceil(len(train_loader) * cfg["num_epochs"])
    warmup_steps = cfg.get("warmup_steps", 400)

    def lr_lambda(step):
        step = max(1, step)
        return (warmup_steps ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # loss (ignore pad)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # checkpoint dir
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    best_bleu = -1.0
    clip_norm = cfg.get("clip_norm", 1.0)

    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"\n=== Epoch {epoch}/{cfg['num_epochs']} ===")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, clip_norm, cfg)
        scheduler.step()
        val_loss, val_bleu = validate(model, val_loader, criterion, device, sp, cfg, max_val_samples=200)

        t1 = time.time()
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_bleu={val_bleu:.2f} | time={t1-t0:.1f}s")

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": cfg
        }
        torch.save(ckpt, os.path.join(ckpt_dir, f"ckpt_epoch{epoch}.pt"))
        # save best
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            print(f"Saved new best model (BLEU {best_bleu:.2f})")

    print("Training finished.")


if __name__ == "__main__":
    main()