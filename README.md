#  Empathetic Chatbot using Transformers  


> A Transformer-based chatbot built **from scratch** in PyTorch — designed to understand emotions and respond with empathy   

---

##  Introduction  

Most chatbots can answer questions — but they often *miss the emotion*.  
This project explores how **Generative AI** can make chatbots more *human-like and emotionally aware*.  

I implemented a **Transformer Encoder–Decoder** architecture from scratch, trained it on emotion-labeled dialogue data, and deployed it using **Streamlit** for real-time interaction.

---

##  Objective  

To design and train an empathetic chatbot capable of:  
- Understanding **emotion-tagged conversations**  
- Generating **context-aware and emotionally aligned replies**  
- Providing an **interactive chat interface** for real-time use  

---

##  Project Overview  

| Component | Description |
|------------|-------------|
| **Dataset Preprocessing** | Cleaned and structured the `emotion_emotion_69k` dataset |
| **Tokenizer** | Trained a custom **SentencePiece BPE** tokenizer (vocab = 8,000) |
| **Model** | Implemented **Transformer Encoder–Decoder** from scratch |
| **Training Pipeline** | Managed with **YAML configs** and **TensorBoard** tracking |
| **Web App** | Built with **Streamlit** for real-time chatting |
| **Advanced Decoding** | Implemented *temperature*, *top-k*, and *beam search* sampling |

---

##  Tech Stack  

 **PyTorch** – Transformer implementation  
 **SentencePiece** – Subword tokenization  
 **Pandas**, **NumPy**, **Matplotlib** – Data analysis & visualization  
 **scikit-learn** – Splitting and evaluation  
 **Streamlit** – Chat UI  
 **YAML** – Config management  
 **TensorBoard** – Model tracking  

---

## Project Phases  

### Phase 0: Environment Setup  
Installed dependencies:  
```bash
torch torchvision numpy pandas tqdm sentencepiece pyyaml matplotlib scikit-learn tensorboard streamlit pytest huggingface-hub

