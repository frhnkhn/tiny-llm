# 🧠 Tiny LLM From Scratch + Local Ollama Chatbot

A fully offline AI project built on macOS using:

* **A custom Tiny LLM** trained from scratch with PyTorch (educational model)
* **A beautiful Flask + HTML/CSS chat UI**
* **Ollama** running powerful local LLMs (LLaMA 3.2, Phi-3, etc.)
* **VS Code + virtual environment workflow**

This project has **two brains in one repo**:

1. **`tiny_llm_v2.py`** – A small character-level GPT-like model you train yourself.
2. **`cool_ollama_site.py`** – A real AI assistant powered by local Ollama models.

The frontend (website) is the same for both.

---

## 🚀 Features

### 🟦 Tiny LLM (Educational)

* Built completely from scratch in Python
* Character-level language model
* Context window with embeddings
* Training loop, loss function, sampling
* Checkpoint save + load
* Works entirely offline

### 🟩 Local Full LLM (Ollama)

* Uses **LLaMA**, **Phi-3**, **Gemma**, etc.
* Runs **fully offline** on macOS
* Replaces the tiny LLM for real conversations
* Uses same Flask + HTML chat UI

---

## 📂 Project Structure

```
llm-from-scratch/
│
├── tiny_llm_v2.py            # small PyTorch model
├── tiny_llm_v2_ckpt.pt       # checkpoint after training
│
├── cool_llm_site.py          # Flask site powered by Ollama
├── cool_llm_site.py          # Flask site powered by Tiny LLM
│
├── data.txt                  # training data for tiny LLM
│
├── templates/
│   └── index.html            # frontend UI
│
└── static/
    └── style.css             # UI styling
```

---

## 🛠️ Setup Instructions

### 1️⃣ Clone & open in VS Code

```bash
cd ~/Codes/llm-from-scratch
code .
```

### 2️⃣ Create virtual environment

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install torch flask requests
```

---

## 🔥 Running the Tiny LLM (Educational)

### Train:

```bash
python tiny_llm_v2.py
```

This generates:

```
tiny_llm_v2_ckpt.pt
```

### Run the tiny model website:

```bash
python cool_llm_site.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🤖 Running the Powerful Ollama Chatbot

### Install Ollama (macOS)

Download from:

```
https://ollama.com/download/mac
```

### Pull a local model

```bash
ollama pull llama3.2
```

### Start the chatbot site

```bash
python cool_ollama_site.py
```

Open the site:

```
http://127.0.0.1:5000
```

Now your website uses a **full local LLM**.

---

## 🎨 Frontend

* Clean_gradient background
* Glassy card UI
* Smooth chat bubbles
* Fully responsive

All frontend files are inside:

```
templates/index.html
static/style.css
```

---

## 📘 What You Learn from This Project

* How LLMs tokenize and generate text
* How to train a simple model from scratch
* How embeddings + context windows work
* How to build a Flask backend API
* How to make a modern frontend chat UI
* How to integrate Ollama into your own apps

Great for **college submissions, resumes, and GitHub portfolios**.

---

## 🧑‍💻 Credits

Built by **Farhan Khan**.

---

If you want, I can also:

* Add screenshots
* Add badges (Python, Flask, Ollama, PyTorch)
* Add a demo GIF
* Add installation notes for Linux/Windows

Just tell me!

---
## For Running locally

cd /Users/frhnkhn/Codes/llm-from-scratch
source .venv/bin/activate

python cool_ollama_site.py
