# Architecture Document: Multilingual Hate Speech Detection System

## 1. System Overview
This document outlines the technical architecture for a multilingual, transformer-based content moderation application designed to detect online hate speech. Initially targeting English, Hindi, and Hinglish (Hindi-English code-mixed) social media text, the system utilizes a fine-tuned multilingual transformer model (mBERT or XLM-RoBERTa). The design focuses on scalability, low latency, and dynamic extension, allowing the future integration of additional target languages via Cross-Lingual Transfer Learning (CLL).

---

## 2. High-Level Architecture
The system employs a modular, 4-tier pipeline structure to transition from raw social media inputs to highly accurate classification probabilities:

```
[ Raw User Input (Hinglish/Hindi/English) ]
                   │
                   ▼
     [ 1. Data Preprocessing Layer ]
      - Standardizes text (URL/mention maps)
      - TF-IDF Aggregation Filtering (Training-only)
                   │
                   ▼
 [ 2. Tokenization & Vocab Augmentation Layer ]
      - Text tokenization (max length 128)
      - Appending out-of-vocabulary (OOV) slang
                   │
                   ▼
    [ 3. Deep Learning Modeling Layer ]
      - Base Multilingual Transformer (mBERT/XLM-R)
      - Classification Head (Softmax/Sigmoid)
                   │
                   ▼
      [ 4. Evaluation & Diagnostic Layer ]
      - Benchmarking against Multilingual HateCheck (MHC)
```

---

## 3. Core Component Specifications

### 3.1. Data Preprocessing Layer
This component standardizes noisy social media text to reduce vocabulary variance and computational overhead:
*   **Text Cleaning:** Raw text is converted to lowercase. User mentions (e.g., `@user_123`) are replaced with a standardized `[USER]` token, and web URLs are mapped to a uniform `[URL]` token. Extra whitespace is removed.
*   **TF-IDF Sample Selection (Training-Phase Optimization):** To accelerate training under strict deadlines, the pipeline applies a TF-IDF mathematical filter. It computes TF-IDF word scores across the entire dataset, aggregates them per sentence, ranks all samples, and filters out the bottom 25% lowest-information entries. Retaining only the **top 75% most informative sentences** reduces model fine-tuning time by nearly 50% while maintaining or improving overall classification accuracy.

### 3.3. Tokenization & Vocabulary Augmentation Layer
Standard multilingual models suffer from "out-of-vocabulary" (OOV) issues when encountering localized slang, code-mixed text, or transliterated profanity.
*   **Base Tokenization:** Text is tokenized using the standard subword tokenizer of the selected transformer (e.g., WordPiece for mBERT) with a fixed maximum sequence length of 128.
*   **Vocabulary Augmentation:** Specific Hinglish slurs and dialectal expressions are extracted from the training corpus and manually appended to the tokenizer's dictionary using `.add_tokens()`. The model's embedding layer is then dynamically resized. This lightweight domain adaptation ensures the model processes regional curses as unified semantic units rather than breaking them into uninformative subword parts.

### 3.4. Deep Learning Modeling Layer
*   **Base Model Selection:** Fine-tuned on pre-trained multilingual architectures like `bert-base-multilingual-cased` (mBERT) or `xlm-roberta-base` (XLM-RoBERTa). XLM-RoBERTa is pre-trained on 100 languages, providing native structural knowledge of both English and Hindi.
*   **Code-Mixed Specialization:** The model is fine-tuned on specialized datasets like **IndoHateMix**, which capture real-world Hinglish syntax, transliterations, and informal social media structures.
*   **Classification Head:** A linear layer acts as a classification head on top of the transformer's pooled representation, outputting a probability distribution (Hate vs. Non-Hate).

### 3.5. Evaluation & Diagnostic Layer
*   **Functional Testing:** The system is evaluated not just on standard accuracy, but against the **Multilingual HateCheck (MHC)** framework. Using MHC's hand-crafted functional test suites, the model is checked for its resilience against linguistic challenges such as counter-speech, spelling obfuscations, and identity-term bias.
*   **Error Analysis:** Systematically documents model limitations on complex linguistic phenomena like irony, sarcasm, or implicit toxicity.

---

## 4. Dynamic Language Expansion
A key advantage of this architecture is its capacity for future-proofing:
*   **Cross-Lingual Transfer Learning (CLL):** Utilizing mBERT/XLM-R's pre-trained multilingual embedding space, the model can generalize hate speech concepts across languages.
*   **Cascade Learning (CL/JL+):** To add a new language later (e.g., Bengali), the model does not require full retraining. Instead, a smaller dataset of the new target language can be introduced through an additional joint or sequential fine-tuning stage (CL/JL+), transferring previously learned toxicity patterns to the new linguistic domain.
