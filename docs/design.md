# Frontend Design Specification: Multilingual Hate Speech Detector

This document outlines the user interface (UI) and user experience (UX) design for the Multilingual Hate Speech Detector. Built with **Streamlit (Python)**, this frontend is optimized for rapid, low-latency execution and high readability, making it ideal for demonstrating the model to both instructors and moderators.

---

## 1. Interface Overview & Wireframe Layout

The application utilizes a **split-screen layout** featuring a clean sidebar for configurations and a main dashboard for input, execution, and analytical visualization.

```
+-----------------------------------------------------------------------------------+
|                           MULTILINGUAL HATE SPEECH DETECTOR                       |
+-----------------------------------------------------------------------------------+
|  [Sidebar: Model Settings]     |  [Main Panel: Input & Predictions]               |
|                                |                                                   |
|  Language Filter Selection     |  Enter Text for Toxicity Evaluation               |
|  [X] English   [X] Hindi       |  +---------------------------------------------+  |
|  [X] Hinglish (Code-Mixed)     |  | Enter text (Devanagari or Roman Latin script)|  |
|                                |  +---------------------------------------------+  |
|  Model Selection               |  [ Analyze Text ]  [ Load Example Tweet ]        |
|  (o) XLM-RoBERTa               |                                                   |
|  ( ) mBERT                     |  +---------------------------------------------+  |
|                                |  | Prediction Banner: CLEAN / TOXIC             |  |
|  Toxicity Decision Threshold   |  +---------------------------------------------+  |
|  |=======o=========| 0.50      |                                                   |
|                                |  +---------------------------------------------+  |
|  System Status                 |  | Visual Metrics: Confidence Probability %     |  |
|  ● GPU Acceleration: Active    |  +---------------------------------------------+  |
|  ● Cache Status: Loaded        |                                                   |
|                                |  [ Diagnostics & Functional Benchmarks (Tab) ]   |
+-----------------------------------------------------------------------------------+
```

---

## 2. Key Interface Sections

### 2.1 Sidebar Control Panel
*   **Model Selector:** Radio button to toggle between fine-tuned **XLM-RoBERTa** (best for semantic representations) and **mBERT** (Multilingual BERT).
*   **Target Language Checkboxes:** Allows the user to toggle supported languages (English, Hindi, and Hinglish). This modifies the placeholder example text available on screen.
*   **Decision Threshold Slider:** A range slider from `0.0` to `1.0` (defaulting to `0.50`). Allows the instructor to manually change the model's tolerance level to see how predictions change dynamically.
*   **System Status Widget:** Displays backend load times, model memory consumption, and whether PyTorch is running on GPU (CUDA) or CPU.

### 2.2 Main Prediction Workspace
*   **Multilingual Text Area:** A clean text box accepting Unicode. It natively handles both standard Devanagari Hindi script (e.g., "दुनिया में सबसे ज़्यादा...") and Romanized transliterated Hinglish (e.g., "wo shudra insaan dikhte hai...").
*   **Action Row:**
    *   `Analyze Text` Button: Triggers the preprocessing, tokenization, and inference pipeline.
    *   `Load Example` Dropdown: Provides preloaded, vetted sample tweets from the **INDOHATEMIX** dataset to let the instructor test Hinglish code-mixing instantly.
*   **Dynamic Response Banner:**
    *   🚨 **Red Banner ("Hate Speech / Abusive Detected")**: Triggered if the model toxicity probability exceeds the threshold.
    *   ✅ **Green Banner ("Safe / Non-Hate Speech")**: Triggered if the text is classified as clean.
*   **Probability Metrics Dashboard:** 
    *   Utilizes a horizontal progress bar (or radial gauge) displaying the exact softmax probability distributions. For example: **Hate Speech Index: 89.2%** vs. **Safe Content: 10.8%**.

### 2.3 Advanced Diagnostic Dashboard (Extraordinary Factor)
To make your project stand out to your academic instructor, the main screen contains a secondary **"Model Diagnostics"** tab. This panel showcases the model's performance on standard benchmark datasets and functional testing suites:
*   **Linguistic Token Highlight:** Highlights specific words in the input text that contributed most to the classification (using simplified integrated gradients or attention-weight extraction).
*   **MHC (Multilingual HateCheck) Functional Tests:** A table detailing how your model handles tricky contrastive test categories (e.g., hate expressed using profanity, versus non-hateful uses of profanity; or counter-speech targeting slurs).
*   **Error Reporting Option:** A "Report Misclassification" button that logs false positives or negatives, simulating a human-in-the-loop framework for future training iterations.

---

## 3. Frontend-Backend Data Flow

```
+------------------+       User Input text       +----------------------------+
|  Streamlit UI    | --------------------------> | Text Preprocessing Layer   |
| (Input & Config) |                             | (URLs & Mentions Stripped) |
+------------------+                             +----------------------------+
         ^                                                     |
         |                                                     v
         | Displays Softmax                                    | Tokenized Vectors
         | Confidence Scores                                   v
+------------------+                             +----------------------------+
|  Output Display  | <-------------------------- |  BERT/XLM-R Classifier     |
| (Red/Green Alert)|   Classification Labels     |   (Softmax Inference)      |
+------------------+                             +----------------------------+
```

---

## 4. Performance Optimizations & Security

*   **Model Memory Caching:** The transformer backend utilizes Streamlit's `@st.cache_resource` decorator. The large mBERT/XLM-RoBERTa model weights (approx. 1.1GB) are loaded into system RAM only once when the server boots. Subsequent text inferences execute in **< 200 milliseconds**.
*   **Input Validation Constraints:** Text inputs are capped at 512 tokens (BERT's maximum context window). Inputs exceeding this limit display a warning banner and are automatically truncated to prevent model crashes.
*   **XSS & Injection Protection:** Streamlit natively escapes HTML code in standard input elements, preventing cross-site scripting (XSS) attacks or prompt injection during model evaluation.

---

## 5. Visual Styling Guidelines

*   **Primary Theme:** Dark Mode default to match developer-centric styling.
*   **Colour Palette:**
    *   *Background:* Charcoal Slate (`#1E1E24`)
    *   *Accents / Buttons:* Cool Electric Blue (`#3A86FF`)
    *   *Hate Alert:* Deep Crimson Rose (`#E63946`)
    *   *Safe Alert:* Emerald Teal (`#2A9D8F`)
*   **Typography:** Clean, high-legibility sans-serif fonts (Inter, Roboto, or standard system fonts) ensuring Devanagari Hindi characters render crisply alongside English text.
