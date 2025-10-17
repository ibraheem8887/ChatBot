import streamlit as st

# ==========================
# Streamlit Setup
# ==========================
st.set_page_config(page_title="Empathetic Chatbot", page_icon="💬", layout="centered")

st.title("🤖 Empathetic Chatbot")
st.write("Chat with your fine-tuned Transformer model!")

# Show loading spinner while imports happen
with st.spinner("Loading model and dependencies..."):
    import torch
    import torch.nn.functional as F
    import sentencepiece as spm
    import random
    import sys, os, yaml

    # Fix import paths
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.model.transformer import Transformer


# ==========================
# Load model & tokenizer (auto handles mismatches)
# ==========================
@st.cache_resource
def load_model_and_tokenizer():
    ckpt_path = "checkpoints/best.pt"
    sp_model_path = "data/tokenizer/empathetic.model"

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # --- Flexible checkpoint structure ---
    if "cfg" in ckpt:
        cfg = ckpt["cfg"]
        model_state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt.get("state_dict")
    else:
        # fallback config if checkpoint has no config
        try:
            with open("configs/default.yaml", "r") as f:
                cfg = yaml.safe_load(f)
        except Exception:
            cfg = {
                "tokenizer": {"vocab_size": 8000},
                "d_model": 256,
                "n_heads": 4,
                "d_ff": 512,
                "enc_layers": 2,
                "dec_layers": 2,
                "dropout": 0.1,
                "pad_idx": 3,
                "max_src_len": 128,
                "max_tgt_len": 50,
            }
        model_state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        st.warning("⚠️ Using default.yaml since checkpoint has no config.")

    # --- Auto detect d_model from weights if mismatch ---
    first_weight = next(iter(model_state.values()))
    if len(first_weight.shape) > 1:
        inferred_d_model = first_weight.shape[1]
        cfg["d_model"] = inferred_d_model

    # --- Load tokenizer ---
    sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    # --- Build model ---
    model = Transformer(
        vocab_size=cfg["tokenizer"]["vocab_size"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        enc_layers=cfg["enc_layers"],
        dec_layers=cfg["dec_layers"],
        dropout=cfg["dropout"],
        pad_idx=cfg.get("pad_idx", 3),
        max_len=max(cfg["max_src_len"], cfg["max_tgt_len"]),
    )

    # --- Safe load (ignores minor mismatches) ---
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    if missing:
        st.warning(f"⚠️ Missing {len(missing)} keys (some weights may differ)")
    if unexpected:
        st.info(f"ℹ️ Ignored {len(unexpected)} unexpected keys")

    model.eval()
    return model, sp, cfg


# ==========================
# Initialize model & device
# ==========================
model, sp, cfg = load_model_and_tokenizer()
device = torch.device(cfg.get("device", "cpu"))
model.to(device)


# ==========================
# Smart Decoding Function
# ==========================
@torch.no_grad()
def generate_reply(
    model,
    sp,
    input_text,
    max_len=50,
    device="cpu",
    temperature=0.8,
    top_k=20,
    use_beam=False,
    beam_width=3,
):
    pad_id = sp.piece_to_id("<pad>") if sp.piece_to_id("<pad>") != -1 else 0
    bos_id = sp.piece_to_id("<bos>") if sp.piece_to_id("<bos>") != -1 else None
    eos_id = sp.piece_to_id("<eos>") if sp.piece_to_id("<eos>") != -1 else None

    src = torch.tensor([sp.encode(input_text)], dtype=torch.long, device=device)

    # ===== Beam Search =====
    if use_beam:
        beams = [(torch.tensor([[bos_id or pad_id]], device=device), 0.0)]
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                logits = model(src, seq)
                logits = logits[:, -1, :] / temperature
                probs = F.log_softmax(logits, dim=-1)
                topk_probs, topk_idx = torch.topk(probs, beam_width)
                for i in range(beam_width):
                    next_id = topk_idx[0, i].item()
                    next_score = score + topk_probs[0, i].item()
                    new_seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)
                    new_beams.append((new_seq, next_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

            # stop if all finished
            if eos_id and all(eos_id in b[0][0].tolist() for b in beams):
                break

        best_seq = beams[0][0][0].tolist()
        if bos_id in best_seq:
            best_seq.remove(bos_id)
        if eos_id in best_seq:
            best_seq = best_seq[:best_seq.index(eos_id)]
        return sp.decode(best_seq)

    # ===== Top-K Sampling =====
    tgt = torch.tensor([[bos_id or pad_id]], dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(src, tgt)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, k=top_k)
        next_token = random.choices(topk_idx[0].tolist(), weights=topk_probs[0].tolist())[0]
        tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)
        if eos_id and next_token == eos_id:
            break

    output_ids = tgt[0].tolist()
    if bos_id in output_ids:
        output_ids.remove(bos_id)
    if eos_id in output_ids:
        output_ids = output_ids[:output_ids.index(eos_id)]
    return sp.decode(output_ids)


# ==========================
# Sidebar Controls
# ==========================
st.sidebar.header("⚙️ Generation Controls")
temperature = st.sidebar.slider("Temperature", 0.5, 1.5, 0.8, 0.05)
top_k = st.sidebar.slider("Top-K Sampling", 5, 50, 20, 1)
use_beam = st.sidebar.checkbox("Use Beam Search (deterministic)", value=False)
beam_width = st.sidebar.slider("Beam Width", 2, 8, 3, 1)


# ==========================
# Chat Interface
# ==========================
if "history" not in st.session_state:
    st.session_state.history = []

for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = generate_reply(
                model,
                sp,
                user_input,
                max_len=cfg["max_tgt_len"],
                device=device,
                temperature=temperature,
                top_k=top_k,
                use_beam=use_beam,
                beam_width=beam_width,
            )
        st.write(reply)

    st.session_state.history.append({"role": "assistant", "content": reply})
