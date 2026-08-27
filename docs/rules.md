# Developer Rules & Prompts for AI Coding Assistant

This document contains instructions, rules, and system prompts to feed into your AI coding assistant (like Claude 3.5 Sonnet, GPT-4o, or Cursor) to guide the code generation for the **Multilingual Hate Speech Detection System**.

---

## 1. Role & Context
You are an expert AI/ML engineer specialising in Natural Language Processing (NLP), Transformer models, and lightweight production deployment. You are helping a developer build a moderate-to-advanced multilingual hate speech detector targeting **English, Hindi, and Hinglish (code-mixed)** using PyTorch, Hugging Face `transformers`, and a Streamlit frontend.

---

## 2. Core Constraints & Guarantees
Every script, model training step, and frontend component you generate must strictly adhere to these rules:

1.  **Frameworks:** Use `transformers` (Hugging Face) for models/tokenizers, `torch` for deep learning, `scikit-learn` for machine learning utilities, and `streamlit` for the frontend.
2.  **Base Architecture:** Utilise pre-trained multilingual transformers—ideally `bert-base-multilingual-cased` (mBERT) or `xlm-roberta-base`.
3.  **Clean Code Style:** All code must contain modular functions, docstrings, type hinting, and robust error handling. Do not generate large, unstructured scripts.
4.  **Low Latency Inference:** The frontend model loading must use caching (`@st.cache_resource`) so inference executes in under 2 seconds.

---

## 3. Step-by-Step Code Generation Instructions

### Rule 1: Text Preprocessing Pipeline
*   Write a Python cleaning function that accepts raw text.
*   **Rules:** 
    *   Convert all text to lowercase (to standardise Romanised Hinglish).
    *   Replace all user mentions (e.g., `@name`) with the uniform token `"username"`.
    *   Replace all hyperlinks (e.g., `https://...`) with the uniform token `"url"`.
    *   Clean excessive white spaces and strip emojis.

### Rule 2: TF-IDF Sample Selection (Computational Efficiency)
*   Because we have a strict deadline, implement a data-efficient sample selection technique before training.
*   **Rules:**
    *   Fit a `TfidfVectorizer` on the text.
    *   Calculate the sum of TF-IDF scores for each sentence in the dataset.
    *   Sort the dataset based on these scores.
    *   **Filter out the bottom 25%** of sentences (which represent low-information, redundant, or overly generic text).
    *   **Retain only the top 75%** of the most informative sentences for training. This will cut GPU training time in half while preserving or boosting classification performance.

### Rule 3: Tokenizer & Vocabulary Augmentation
*   Standard multilingual tokenizers struggle with local Hinglish internet slang and transliterated slurs.
*   **Rules:**
    *   Create a script that extracts out-of-vocabulary (OOV) terms and highly frequent colloquial swear words/slang from the training set.
    *   Explicitly append these unique tokens to the tokenizer using `tokenizer.add_tokens(new_tokens)`.
    *   Remember to resize the model's token embeddings layer using `model.resize_token_embeddings(len(tokenizer))` immediately after loading the base weights.

### Rule 4: Fine-Tuning Setup
*   Implement a standard sequence classification head over the multilingual base.
*   **Hyperparameters:**
    *   **Optimizer:** `AdamW` with a learning rate of `2e-5` to `5e-5`.
    *   **Epochs:** Limit to 3 epochs to prevent overfitting.
    *   **Sequence Length:** Set max padding/truncation length to 128 to save memory.
    *   **Train/Test Split:** Standard 80/20 split.

### Rule 5: Research-Grade Evaluation (MHC Testing)
*   Do not just evaluate with global accuracy. Implement a diagnostic script based on the **Multilingual HateCheck (MHC)** framework.
*   **Rules:**
    *   Test the final model specifically on functional categories: Negation (e.g., "I do not hate you" vs "I hate you"), spelling obfuscations (e.g., "h4te"), and counter-speech.
    *   Output a diagnostic report showing the Macro F1-Score, Precision, and Recall for each category.

### Rule 6: Streamlit UI Implementation
*   **Rules:**
    *   Design a clean, dual-tab layout (Tab 1: Predictor, Tab 2: Model Diagnostics & MHC Benchmarks).
    *   Load the fine-tuned model and tokenizer using Streamlit’s `@st.cache_resource` decorator to avoid reloading on every interaction.
    *   Provide a visual confidence slider or progress bar (0-100%) indicating how "hateful" or "neutral" the input text is.
    *   Enable preloaded Hinglish/Hindi examples from the benchmark data (like the IndoHateMix dataset) so users can immediately test the system's capabilities.
