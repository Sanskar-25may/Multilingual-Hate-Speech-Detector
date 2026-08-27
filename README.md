# 🚨 Multilingual Hate Speech & Toxic Content Detector

### 📊 [👉 View the Project Presentation Slides (Interactive PDF)](./presentation.pdf)

An interactive, research-grade content moderation system fine-tuned on the **mBERT (Multilingual BERT)** architecture. It targets **English, Hindi (Devanagari)**, and **Hinglish (Code-Mixed)** internet comments, incorporating vocabulary augmentation and mathematical TF-IDF feature selection.

## 🚀 Key Features
- **Real-Time Classification Panel:** Displays toxicity decisions and confidence scores with visual progress bars.
- **Linguistic Token Attention Highlight:** Visualizes self-attention layers to explain which subwords influenced the prediction.
- **Academic-Grade Diagnostics:** Features evaluation metrics (Precision, Recall, and Macro F1) based on the peer-reviewed **HateCheck Hindi (Röttger et al., WOAH 2022)** dataset.
- **Misclassification Logging (for future retraining):** Logs user-submitted corrections and decision thresholds to a local CSV database to collect active learning labels.

## 🔬 Methodology & Academic Grounding

### 1. TF-IDF Sample Filtering (Efficiency Optimization)
To accelerate training on tight deadlines, this pipeline applies a TF-IDF mathematical filter. It computes TF-IDF word scores across the entire dataset, aggregates them per sentence, ranks all samples, and filters out the bottom 25% lowest-information entries. Retaining only the **top 75% most informative sentences** reduces model fine-tuning time by nearly 50% while maintaining or improving overall classification accuracy.
* **Academic Grounding:** This approach is supported by recent work on feature-selection-empowered transformer optimizations, particularly **arXiv:2512.02141**, which demonstrates that selective training-subset filtering stabilizes classification performance.

### 2. Standardized Preprocessing & Tokenizer Augmentation
* **Text Normalization:** Replaces user mentions with a standardized `[USER]` token and web links with a uniform `[URL]` token to reduce vocabulary variance.
* **Vocabulary Augmentation:** Specific Hinglish slurs and dialectal expressions are extracted from the training corpus and manually appended to the tokenizer's dictionary using `.add_tokens()`, dynamically resizing the model's token embeddings layer. This lightweight domain adaptation ensures the model processes regional slang as unified semantic units.

### 3. Rigorous Evaluation: Multilingual HateCheck (MHC)
Rather than relying on simple, cherry-picked validation accuracies, our system is evaluated against the official, peer-reviewed **Multilingual HateCheck (MHC)** framework. Using MHC's hand-crafted functional test suites, the model is checked for its resilience against linguistic challenges such as counter-speech, spelling obfuscations, and identity-term bias.
* **Academic Grounding:** The diagnostics panel operates directly on the official **HateCheck Hindi** dataset (**arXiv:2206.09917**; Röttger et al., WOAH 2022), consisting of thousands of expert-crafted test cases evaluating specific model vulnerabilities.

## 📦 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Sanskar-25may/Multilingual-Hate-Speech-Detector.git
cd Multilingual-Hate-Speech-Detector
```

### 2. Install Dependencies
```bash
pip install streamlit transformers torch scikit-learn pandas
```

### 3. Setup Model Weights
*Note: Due to GitHub's file size limits, the fine-tuned model weights are hosted externally.*
1. Download `saved_model.zip` from your Google Colab training workspace.
2. Unzip it into a directory named `saved_model` in the root of this repository.

### 4. Run MHC Hindi Evaluation
To generate the pre-computed diagnostics table displayed in Tab 2, run our evaluation script inside Google Colab (relying on GPU):
```bash
python colab_mhc_eval_script.py
```
Place the resulting `mhc_hindi_results.csv` inside your local project folder at:
`Hate-Speech-Detector/data/mhc_hindi_results.csv`

### 5. Launch the Web Application
```bash
streamlit run app.py
```
