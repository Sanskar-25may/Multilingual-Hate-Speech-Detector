# Project Implementation Plan: Multilingual Hate Speech Detector

This document outlines the structured, 4-day implementation timeline (August 26 – August 30) to build, fine-tune, and deploy a moderate-level Multilingual Hate Speech Detector. It provides step-by-step instructions, mathematical principles, data preparation frameworks, and model verification standards grounded in contemporary NLP research.

---

## 📅 Chronological Phase Breakdown

```
  August 26              August 27              August 28              August 29-30
 ┌───────────┐          ┌───────────┐          ┌───────────┐          ┌─────────────┐
 │  Phase 1  │ ────────>│  Phase 2  │ ────────>│  Phase 3  │ ────────>│   Phase 4   │
 │Data & Prep│          │Model & Voc│          │ Frontend  │          │Eval & Package│
 └───────────┘          └───────────┘          └───────────┘          └─────────────┘
```

---

## Phase 1: Data Acquisition, Preprocessing & TF-IDF Data Filtering (Day 1 - Aug 26)

### Objective
Procure Hindi-English code-mixed and multilingual datasets, clean the raw noisy text, and mathematically select the most informative samples to optimize training time on your tight deadline.

### 1. Dataset Selection
To ensure your project successfully targets Hindi, English, and Hinglish (code-mixed transliteration), you will utilize:
*   **HASOC 2021 Dataset (Primary):** The HASOC (Hate Speech and Offensive Content Identification in Indo-European Languages) 2021 Subtask-A Hinglish/Hindi dataset (curated from Twitter). We pivoted from our initial plan of using IndoHateMix to this HASOC 2021 Subtask-A mirror (`harjeet-blue/Hate-Speech-Detection-In-Social-Media`) because it shipped pre-labeled, high-quality, pre-split datasets ready for training, which fit our strict project timeline.
*   **Alternative/Supplementary Options:** The **INDOHATEMIX** dataset [182] and **Davidson et al.** dataset [34].

### 2. Standardized Text Preprocessing Pipeline
Social media text is highly unstructured. Write a Python preprocessing script to clean every text string before it is used for training or inference:
1.  **Tag Normalisation:** Replace all user mentions (e.g., `@amit_12`) with a uniform `[USER]` or `username` token [34, 45].
2.  **URL Neutralisation:** Replace web links and URLs with a uniform `[URL]` or `url` token [34, 45].
3.  **Spacing & Case Uniformity:** Convert all Latin-based characters (Hinglish/English) to lowercase and normalize multiple white spaces into a single space [34, 45, 149].
4.  **Emoji Transcription (Optional):** Retain emojis or transcribe them into text using standard libraries to preserve sentiment cues [236].

### 3. The "Extraordinary" Feature: TF-IDF Sample Filtering
To drastically reduce training time and eliminate redundancy without sacrificing model performance, you will implement a **TF-IDF-based sample selection mechanism** [34, 55].

#### Mathematical Workflow
1.  Represent your entire preprocessed dataset as a document collection $D$.
2.  Compute the Term Frequency-Inverse Document Frequency (TF-IDF) score for every word $t$ in each document $d$:
    $$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right)$$
3.  Calculate an **Aggregate TF-IDF Score** for each sentence/tweet by summing the individual TF-IDF scores of its constituent words [43, 64].
4.  Rank the dataset in descending order based on these aggregate scores.
5.  **Filter the Dataset:** Discard the bottom 25% of sentences (which consist of generic, repetitive, or low-information text) and **retain only the top 75% most informative sentences** [34, 37, 55, 58].

> **Research Grounding:** Fine-tuning BERT on only the top 75% of TF-IDF filtered training data maintains or slightly improves classification accuracy while reducing computational training time by nearly 50% [34, 39, 47, 48].

---

## Phase 2: Model Setup, Tokenizer Augmentation & Multilingual Fine-Tuning (Day 2 - Aug 27)

### Objective
Load a pre-trained multilingual transformer model, augment its native tokenizer dictionary to recognize Indian online slang, and fine-tune the architecture using an optimized parameter setup.

### 1. Base Architecture Selection
Instead of standard English BERT, you will load a pre-trained multilingual model capable of cross-lingual feature extraction:
*   **XLM-RoBERTa (`xlm-roberta-base`):** Pre-trained on 100 languages (including Hindi and English), showing exceptional contextual representation [19, 172, 196].
*   **mBERT (`bert-base-multilingual-cased`):** Pre-trained on 104 languages, providing robust transfer learning baselines [19, 172, 196].

### 2. Vocabulary Augmentation (Domain Adaptation)
Standard multilingual tokenizers rely on static vocabularies (e.g., mBERT's 110k WordPiece tokens) which frequently fail to capture informal code-mixed expressions, variant spellings, and obfuscated regional slurs [34, 36, 57].

#### Implementation Steps
1.  **Extract Out-of-Vocabulary (OOV) Terms:** Parse your HASOC training dataset to identify frequently occurring Hinglish profanities, regional slurs, and internet jargon not present in the transformer's default dictionary [44, 65].
2.  **Append to Tokenizer:** Use Hugging Face's API to add these custom tokens to your model's tokenizer:
    ```python
    tokenizer.add_tokens(["custom_hinglish_term1", "custom_hinglish_term2"])
    ```
3.  **Adjust Embedding Layers:** Call `model.resize_token_embeddings(len(tokenizer))` to dynamically update the input embedding layer [44, 65].

> **Research Grounding:** This lightweight domain-adaptation technique modifies only the input embedding layer without retraining the full base model, ensuring that the transformer can represent adversarial internet slang without losing pre-trained parameters [44, 50, 65, 71].

### 3. Model Training Configuration
Set up your PyTorch training loop or Hugging Face `Trainer` API with the following parameters:
*   **Train/Test Split:** Preserve an 80-20 stratified split to ensure unbiased assessment [43, 45, 170].
*   **Optimizer:** AdamW [19].
*   **Learning Rate:** Keep it small (e.g., `2e-5` to `5e-5` for standard fine-tuning) to prevent catastrophic forgetting [19, 38].
*   **Epochs:** 3 epochs (to prevent overfitting on code-mixed data).
*   **Sequence Length:** Set to 128 tokens (truncating longer strings) to optimize memory and speed [42, 63].

Once trained, save the model and tokenizer to a directory (e.g., `./saved_model/`) to be loaded by the frontend.

---

## Phase 3: Frontend Implementation & Integration (Day 3 - Aug 28)

### Objective
Develop a highly polished, responsive web application using Streamlit to accept multilingual text and perform low-latency real-time inference.

### 1. Core UI Elements (Layout Structure)
Refer to your `design.md` for full wireframe guidelines. The frontend must implement:
*   **Interactive Input Area:** A text area supporting dual Hindi (Devanagari) and Roman script (Hinglish/English) input.
*   **Dataset Examples Selector:** A drop-down menu containing preloaded sample Hinglish sentences from the HASOC 2021 benchmark to let users test predictions instantly [197, 222].
*   **Prediction Dashboard:** Colorful metric displays showing the final class label ("Safe" in green, "Hateful" in red) and confidence probabilities.

### 2. Low-Latency Pipeline Integration
To satisfy the strict performance constraint of **<2 seconds execution time**:
1.  **Single-Load Caching:** Decorate your model loading function with `@st.cache_resource` so that the heavy transformer weights are loaded into CPU/GPU memory only once upon startup.
2.  **Runtime Pipeline:** When the "Classify" button is pressed, pass the raw text through your standard cleaning function, tokenize it, and run forward inference using PyTorch's `with torch.no_grad():` block to maximize speed.

---

## Phase 4: Robust Diagnostic Evaluation & Packaging (Day 4 - Aug 29-30)

### Objective
Evaluate your model using traditional classification metrics, test its limits with specialized functional test suites, conduct an error analysis, and package the repository for submission.

### 1. Standard Metrics Evaluation
Calculate and compile the following metrics on your held-out 15% validation split:
*   **Macro F1-Score:** Crucial for evaluating hate speech due to class imbalance [143, 154].
*   **Precision and Recall:** Separately analyzed for the hateful and non-hateful classes to monitor false positives and false negatives [48, 69, 173].

### 2. Functional Testing: Multilingual HateCheck (MHC)
To make your project truly "extraordinary," evaluate your fine-tuned model against **Multilingual HateCheck (MHC)**, a diagnostic test suite containing hand-crafted, targeted test cases across multiple languages (including Hindi and English) [76, 79, 108, 111].

#### Key Test Classes to Run
*   **Contrastive Non-Hate:** Verify if your model can correctly identify non-abusive uses of profanity (e.g., "I had a f*cking great day") or counter-speech targeting hateful messages [80, 82, 86].
*   **Protected Group Identifiers:** Ensure the model does not misclassify simple sentences containing identity terms (e.g., "I am gay", "Muslim people") as hateful [80, 86, 127].
*   **Spelling Variations:** Test how resilient your model is to obfuscated spelling, missing characters, or chat-speak (e.g., romanized Hindi spellings) [80, 85].

### 3. Error Analysis & Documentation
Create a detailed markdown report documenting your model's blind spots [21, 197]. Show your instructor that you understand model behavior by analyzing:
*   **Implicit Hate:** Standard models often struggle with sarcasm, irony, and implicit slurs [24, 192, 197, 222].
*   **False Positives:** Analyze if the model is overly sensitive to specific curse words even when used in a friendly context [21, 30].

### 4. Dynamic Language Scaling Plan (Future Work)
Explain in your documentation how your architecture allows you to dynamically add new target languages (e.g., Bengali, Marathi) post-submission without starting from scratch:
*   **Cross-Lingual Transfer Learning (CLL):** Explain that the underlying transformer already understands 100+ languages [233, 248].
*   **Cascade Learning / Joint Learning (CL/JL+):** Detail how the model can be fine-tuned on a small set of the new target language while leveraging its previously learned hate speech features, achieving state-of-the-art transfer performance [233, 248].

---

## Summary of Deliverables

*   `preprocessing.py`: Script containing standard text normalization.
*   `tf_idf_filter.py`: Automated pipeline that ranks and cuts dataset size to 75%.
*   `train.py`: Vocabulary augmentation logic and fine-tuning script.
*   `app.py`: Streamlit-powered multilingual frontend.
*   `architecture.md` / `design.md` / `phases.md`: Completed technical blueprints.
