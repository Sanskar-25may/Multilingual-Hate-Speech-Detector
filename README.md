# 🚨 Multilingual Hate Speech & Toxic Content Detector

An interactive, research-grade content moderation system fine-tuned on the **mBERT (Multilingual BERT)** architecture. It targets **English, Hindi (Devanagari)**, and **Hinglish (Code-Mixed)** internet comments, incorporating vocabulary augmentation and mathematical TF-IDF feature selection.

## 🚀 Key Features
- **Real-Time Classification Panel:** Displays toxicity decisions and confidence scores with visual progress bars.
- **Linguistic Token Attention Highlight:** Visualizes self-attention layers to explain which subwords influenced the prediction.
- **Multilingual HateCheck (MHC) Diagnostics:** Category-wise evaluation metrics (Negation, Obfuscation, Bias) to audit model performance.
- **Active Learning Feedback Loop:** Logs misclassifications to a local database to collect retraining labels.

## 📦 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Sanskar-25may/Multilingual-Hate-Speech-Detector.git
cd Multilingual-Hate-Speech-Detector
2. Install Dependencies
pip install streamlit transformers torch scikit-learn pandas
3. Setup Model Weights
Note: Due to GitHub's file size limits, the fine-tuned model weights are hosted externally.
Download saved_model.zip from your Google Colab training workspace.
Unzip it into a directory named saved_model in the root of this repository.
4. Launch the Web Application
streamlit run app.py

---

### 🌐 Step 3: Create the Repository on GitHub

1. Go to your browser and open [github.com/new](https://github.com/new).
2. Log into your account (`Sanskar-25may`).
3. Set **Repository name** to: `Multilingual-Hate-Speech-Detector`
4. Leave it as **Public**.
5. **CRITICAL:** Do **NOT** check any boxes under "Initialize this repository with" (don't add a README, .gitignore, or License, as we have already created them locally).
6. Click **Create repository**.

---

### 💻 Step 4: Run Git Commands in Cursor Terminal

Open your local terminal inside Cursor and run these commands sequentially to push your code to your profile:

```bash
# 1. Initialize the local directory as a Git repository
git init

# 2. Add all local files (the .gitignore will automatically skip your heavy model folders)
git add .

# 3. Commit the files to your local repository
git commit -m "Initial commit: Complete Streamlit frontend with MHC diagnostics"

# 4. Rename the default branch to main
git branch -M main

# 5. Link your local repository to your new GitHub repository
git remote add origin https://github.com/Sanskar-25may/Multilingual-Hate-Speech-Detector.git

# 6. Push the code up to GitHub
git push -u origin main
(Note: If Git asks you to authenticate, sign in using your browser or your GitHub Personal Access Token.)