import os
import pandas as pd
import re
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Phase 1: Downloading & Preprocessing Datasets ---
print("\n--- Phase 1: Downloading & Preprocessing Datasets ---")
os.makedirs("data", exist_ok=True)

# 100% Public and verified HASOC 2021 English, Hindi, and Hinglish pre-merged mirrors
train_url = "https://raw.githubusercontent.com/harjeet-blue/Hate-Speech-Detection-In-Social-Media/main/train_dataset.csv"
test_url = "https://raw.githubusercontent.com/harjeet-blue/Hate-Speech-Detection-In-Social-Media/main/test_dataset.csv"

print("Downloading train dataset...")
df_train = pd.read_csv(train_url)
print("Downloading test dataset...")
df_test = pd.read_csv(test_url)

print(f"✅ Downloaded train set: {len(df_train)} rows")
print(f"✅ Downloaded test set: {len(df_test)} rows")

# Inspect columns
print("Columns in train dataset:", list(df_train.columns))
print("Columns in test dataset:", list(df_test.columns))

# Map columns dynamically
train_text_col = "text" if "text" in df_train.columns else df_train.columns[0]
train_label_col = "label" if "label" in df_train.columns else df_train.columns[1]

test_text_col = "text" if "text" in df_test.columns else df_test.columns[0]

print(f"Standardizing mapping: Train Text Column -> '{train_text_col}', Train Label Column -> '{train_label_col}'")

def clean_social_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df_train['cleaned_text'] = df_train[train_text_col].apply(clean_social_text)
df_test['cleaned_text'] = df_test[test_text_col].apply(clean_social_text)

# --- Phase 2: Applying the 75% TF-IDF Optimization Filter ---
print("\n--- Phase 2: TF-IDF Informative Sentence Filtering ---")
vectorizer = TfidfVectorizer(max_features=10000)
tfidf_matrix = vectorizer.fit_transform(df_train['cleaned_text'])
df_train['tfidf_score'] = tfidf_matrix.sum(axis=1).A1

df_train_sorted = df_train.sort_values(by='tfidf_score', ascending=False)
cutoff_idx = int(len(df_train_sorted) * 0.75)
df_train_filtered = df_train_sorted.iloc[:cutoff_idx].copy().drop(columns=['tfidf_score'])

print(f"Optimized training corpus from {len(df_train)} down to {len(df_train_filtered)} highly informative samples.")

# Map labels: Ensure labels are integers (NOT -> 0, HOF -> 1)
label_mapping = {'NOT': 0, 'HOF': 1, 0: 0, 1: 1}
df_train_filtered['label'] = df_train_filtered[train_label_col].map(label_mapping)
df_train_filtered = df_train_filtered.dropna(subset=['label', 'cleaned_text'])
df_train_filtered['label'] = df_train_filtered['label'].astype(int)

# --- Stratified Train-Validation Split (Crucial for Unlabeled Test Set) ---
print("\n--- Creating Stratified Train-Validation Split (85% Train, 15% Val) ---")
df_train_split, df_val_split = train_test_split(
    df_train_filtered,
    test_size=0.15,
    random_state=42,
    stratify=df_train_filtered['label']
)
print(f"Training subset: {len(df_train_split)} samples")
print(f"Validation subset: {len(df_val_split)} samples")

# --- Phase 3: Tokenization & Vocabulary Augmentation ---
print("\n--- Phase 3: Tokenization & Vocabulary Augmentation ---")
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# High-quality list of modern Hinglish slurs and social syntax tokens for vocabulary augmentation
custom_tokens = ["[USER]", "[URL]", "chutiya", "bhenchod", "saala", "kamina", "bakwas", "feku", "gandu", "gand", "harami", "madarchod"]
tokenizer.add_tokens(custom_tokens)
print(f"Augmented tokenizer vocabulary with {len(custom_tokens)} specialized Hinglish/Hindi slang tokens.")

# Prepare Hugging Face datasets
train_dataset = Dataset.from_pandas(df_train_split[['cleaned_text', 'label']])
val_dataset = Dataset.from_pandas(df_val_split[['cleaned_text', 'label']])

def tokenize_function(examples):
    return tokenizer(examples['cleaned_text'], padding='max_length', truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# --- Phase 4: Model Configuration & Fine-Tuning ---
print("\n--- Phase 4: Multilingual BERT Training ---")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.resize_token_embeddings(len(tokenizer)) # Sync model embeddings with augmented tokenizer
model = model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True if device == "cuda" else False, # Enable Mixed Precision on GPU
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
)

print("Starting mBERT fine-tuning...")
trainer.train()

# Save final model weights
print("\nSaving final model and tokenizer...")
model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")

# --- Optional Phase 5: Run Predictions on the Unlabeled Test Dataset ---
print("\n--- Optional Phase 5: Generating Predictions for the Unlabeled Test Set ---")
# Predict function
model.eval()
test_texts = df_test['cleaned_text'].tolist()
test_ids = df_test['id'].tolist() if 'id' in df_test.columns else list(range(len(df_test)))

predictions = []
batch_size = 32

print(f"Running inference on {len(test_texts)} unlabeled test samples...")
with torch.no_grad():
    for i in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
        predictions.extend(preds)

# Map labels back to string representations for analysis
reverse_mapping = {0: 'NOT', 1: 'HOF'}
df_test['predicted_label'] = [reverse_mapping[p] for p in predictions]

# Save prediction output
df_test.to_csv("test_predictions.csv", index=False)
print("Saved predictions to 'test_predictions.csv'!")

# Package model to download to local computer for Cursor integration
import shutil
shutil.make_archive("saved_model", "zip", "./saved_model")
print("Successfully saved and zipped model as 'saved_model.zip'! You can download this file from Colab.")
