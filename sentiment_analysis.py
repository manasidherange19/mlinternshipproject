# =============================================================
# SENTIMENT ANALYSIS WITH NLP & TRANSFORMERS
# AI Academia — Artificial Intelligence & Machine Learning Internship
# Name: [Your Name]
# Date: [Submission Date]
#
# This project builds and compares multiple sentiment analysis models
# on the IMDB Movie Reviews dataset — from classical ML baselines to
# transformer-based deep learning models using DistilBERT.
# =============================================================

# =============================================================
# SECTION 0 — IMPORTS
# =============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import pipeline
from collections import Counter
from scipy.sparse import csr_matrix

sns.set_style("whitegrid")

# =============================================================
# SECTION 1 — DATA LOADING
# =============================================================
print("\n" + "="*60)
print("SECTION 1 — DATA LOADING")
print("="*60)

dataset = load_dataset('stanfordnlp/imdb')
df_train = pd.DataFrame(dataset['train'])
df_test  = pd.DataFrame(dataset['test'])

print('Train shape:', df_train.shape)
print('Test shape: ', df_test.shape)
print(df_train.head())

print("\nLabel distribution (train):")
print(df_train['label'].value_counts())
print("0 = Negative, 1 = Positive")

# =============================================================
# SECTION 2 — TEXT CLEANING
# =============================================================
print("\n" + "="*60)
print("SECTION 2 — TEXT CLEANING")
print("="*60)

print('Label counts:')
print(df_train['label'].value_counts())
print('0 = Negative, 1 = Positive')

df_train['word_count'] = df_train['text'].apply(lambda x: len(x.split()))
print('Average word count:', df_train['word_count'].mean().round(1))
print('Max word count:    ', df_train['word_count'].max())
print('Min word count:    ', df_train['word_count'].min())

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)        # remove HTML tags
    text = text.lower()                       # lowercase
    text = re.sub(r'[^a-z\s]', '', text)     # remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # remove extra spaces
    return text

df_train['clean_text'] = df_train['text'].apply(clean_text)
df_test['clean_text']  = df_test['text'].apply(clean_text)

print('\nBefore:', df_train['text'].iloc[0][:200])
print('After: ', df_train['clean_text'].iloc[0][:200])

# =============================================================
# SECTION 3 — TF-IDF VECTORIZATION
# =============================================================
print("\n" + "="*60)
print("SECTION 3 — TF-IDF VECTORIZATION")
print("="*60)

tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(df_train['clean_text'])
X_test_tfidf  = tfidf.transform(df_test['clean_text'])

print('Train matrix shape:', X_train_tfidf.shape)
print('Test matrix shape: ', X_test_tfidf.shape)

feature_names = tfidf.get_feature_names_out()
print('Sample features:', feature_names[:20])

word_freq = np.asarray(X_train_tfidf.sum(axis=0)).flatten()
top_idx   = word_freq.argsort()[-20:][::-1]
top_words = [feature_names[i] for i in top_idx]
top_freq  = [word_freq[i] for i in top_idx]

plt.figure(figsize=(10, 4))
plt.bar(top_words, top_freq, color='steelblue')
plt.title('Top 20 Words by TF-IDF Score')
plt.xlabel('Word')
plt.ylabel('TF-IDF Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top_words.png')
plt.show()

# =============================================================
# SECTION 4 — EXPLORATORY DATA ANALYSIS
# =============================================================
print("\n" + "="*60)
print("SECTION 4 — EXPLORATORY DATA ANALYSIS")
print("="*60)

plt.figure(figsize=(10, 4))
sns.histplot(data=df_train, x='word_count', hue='label', bins=50,
             palette={0:'coral', 1:'steelblue'}, alpha=0.6)
plt.title('Review Length Distribution by Sentiment')
plt.xlabel('Word Count')
plt.ylabel('Number of Reviews')
plt.legend(['Negative', 'Positive'])
plt.tight_layout()
plt.savefig('length_distribution.png')
plt.show()

pos_text = ' '.join(df_train[df_train['label'] == 1]['clean_text'])
neg_text = ' '.join(df_train[df_train['label'] == 0]['clean_text'])

pos_words = Counter(pos_text.split()).most_common(15)
neg_words = Counter(neg_text.split()).most_common(15)

pos_df = pd.DataFrame(pos_words, columns=['word', 'count'])
neg_df = pd.DataFrame(neg_words, columns=['word', 'count'])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].barh(pos_df['word'], pos_df['count'], color='steelblue')
axes[0].set_title('Top Words in Positive Reviews')
axes[0].set_xlabel('Count')
axes[1].barh(neg_df['word'], neg_df['count'], color='coral')
axes[1].set_title('Top Words in Negative Reviews')
axes[1].set_xlabel('Count')
plt.tight_layout()
plt.savefig('top_words_by_sentiment.png')
plt.show()

# =============================================================
# SECTION 5 — LOGISTIC REGRESSION BASELINE
# =============================================================
print("\n" + "="*60)
print("SECTION 5 — LOGISTIC REGRESSION BASELINE")
print("="*60)

y_train = df_train['label'].values
y_test  = df_test['label'].values

lr_model = LogisticRegression(max_iter=1000, C=1.0)
lr_model.fit(X_train_tfidf, y_train)
y_pred_lr = lr_model.predict(X_test_tfidf)

print('Accuracy:', round(accuracy_score(y_test, y_pred_lr), 4))
print(classification_report(y_test, y_pred_lr, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred Neg', 'Pred Pos'],
            yticklabels=['Actual Neg', 'Actual Pos'])
plt.title('Logistic Regression — Confusion Matrix')
plt.tight_layout()
plt.savefig('lr_confusion_matrix.png')
plt.show()

coefs   = lr_model.coef_[0]
top_pos = np.argsort(coefs)[-10:]
top_neg = np.argsort(coefs)[:10]

print('Words most predictive of POSITIVE:')
print([feature_names[i] for i in top_pos])
print('Words most predictive of NEGATIVE:')
print([feature_names[i] for i in top_neg])

# =============================================================
# SECTION 6 — NAIVE BAYES
# =============================================================
print("\n" + "="*60)
print("SECTION 6 — NAIVE BAYES")
print("="*60)

X_train_nb = csr_matrix(X_train_tfidf.multiply(X_train_tfidf >= 0))
nb_model   = MultinomialNB()
nb_model.fit(X_train_nb, y_train)
y_pred_nb  = nb_model.predict(X_test_tfidf)

print('Naive Bayes Accuracy:', round(accuracy_score(y_test, y_pred_nb), 4))

results = pd.DataFrame({
    'Model':    ['Logistic Regression', 'Naive Bayes'],
    'Accuracy': [accuracy_score(y_test, y_pred_lr),
                 accuracy_score(y_test, y_pred_nb)]
})
print(results)

results.plot(kind='bar', x='Model', y='Accuracy',
             figsize=(7, 4), color=['steelblue', 'coral'], legend=False)
plt.title('Classical Models — Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# =============================================================
# SECTION 7 — NEURAL NETWORK (ANN)
# =============================================================
print("\n" + "="*60)
print("SECTION 7 — NEURAL NETWORK (ANN)")
print("="*60)

X_tr = torch.FloatTensor(X_train_tfidf.toarray())
X_te = torch.FloatTensor(X_test_tfidf.toarray())
y_tr = torch.LongTensor(y_train)
y_te = torch.LongTensor(y_test)

train_ds = TensorDataset(X_tr, y_tr)
train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)

class SentimentANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5000, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

model_ann = SentimentANN()
optimizer = torch.optim.Adam(model_ann.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

losses = []
for epoch in range(5):
    model_ann.train()
    total_loss = 0
    for xb, yb in train_dl:
        optimizer.zero_grad()
        loss = criterion(model_ann(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dl)
    losses.append(avg_loss)
    print(f'Epoch {epoch+1}/5 — Loss: {avg_loss:.4f}')

plt.figure(figsize=(7, 3))
plt.plot(range(1, 6), losses, marker='o', color='steelblue')
plt.title('ANN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('ann_loss.png')
plt.show()

model_ann.eval()
with torch.no_grad():
    preds = model_ann(X_te).argmax(dim=1).numpy()
print('ANN Accuracy:', round(accuracy_score(y_test, preds), 4))

# =============================================================
# SECTION 8 — TRANSFORMER (DISTILBERT)
# =============================================================
print("\n" + "="*60)
print("SECTION 8 — TRANSFORMER (DISTILBERT)")
print("="*60)

sentiment_model = pipeline(
    'sentiment-analysis',
    model='distilbert-base-uncased-finetuned-sst-2-english',
    truncation=True,
    max_length=512
)

sample_df    = df_test.sample(500, random_state=42).reset_index(drop=True)
sample_texts = sample_df['clean_text'].tolist()

raw_preds   = sentiment_model(sample_texts, batch_size=32)
pred_labels = [1 if p['label'] == 'POSITIVE' else 0 for p in raw_preds]
true_labels = sample_df['label'].tolist()

print('Transformer Accuracy (500 samples):',
      round(accuracy_score(true_labels, pred_labels), 4))
print(classification_report(true_labels, pred_labels,
      target_names=['Negative', 'Positive']))

my_reviews = [
    "This movie was absolutely brilliant, I loved every minute!",
    "Terrible film. Waste of time and money.",
    "It was okay, nothing special but not bad either."
]
results_custom = sentiment_model(my_reviews)
for review, result in zip(my_reviews, results_custom):
    print(f'Review: {review[:60]}...')
    print(f'Prediction: {result["label"]} (confidence: {result["score"]:.2f})')
    print()

# =============================================================
# SECTION 9 — MODEL COMPARISON & KEY FINDINGS
# =============================================================
print("\n" + "="*60)
print("SECTION 9 — MODEL COMPARISON & KEY FINDINGS")
print("="*60)

comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'ANN', 'DistilBERT (sample)'],
    'Accuracy': [
        round(accuracy_score(y_test, y_pred_lr), 4),
        round(accuracy_score(y_test, y_pred_nb), 4),
        round(accuracy_score(y_test, preds), 4),
        round(accuracy_score(true_labels, pred_labels), 4)
    ],
    'Type': ['Classical ML', 'Classical ML', 'Deep Learning', 'Transformer']
})
print(comparison)

colors = ['steelblue', 'steelblue', 'coral', 'green']
plt.figure(figsize=(9, 4))
bars = plt.bar(comparison['Model'], comparison['Accuracy'], color=colors)
plt.title('Model Accuracy Comparison — Sentiment Analysis')
plt.ylabel('Accuracy')
plt.ylim(0.75, 1.0)
plt.xticks(rotation=15)
best_idx = comparison['Accuracy'].idxmax()
bars[best_idx].set_edgecolor('gold')
bars[best_idx].set_linewidth(3)
plt.annotate('Best Model',
             xy=(best_idx, comparison['Accuracy'].max()),
             xytext=(best_idx + 0.3, comparison['Accuracy'].max() - 0.02),
             arrowprops=dict(arrowstyle='->', color='gold'), color='gold')
plt.tight_layout()
plt.savefig('final_comparison_annotated.png')
plt.show()

# ---- KEY FINDINGS ----
print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
lr_acc   = round(accuracy_score(y_test, y_pred_lr), 4)
nb_acc   = round(accuracy_score(y_test, y_pred_nb), 4)
ann_acc  = round(accuracy_score(y_test, preds), 4)
bert_acc = round(accuracy_score(true_labels, pred_labels), 4)

print(f"""
Finding 1: DistilBERT achieved {bert_acc*100:.1f}% accuracy vs {lr_acc*100:.1f}% for Logistic Regression.
Meaning:   Pre-trained transformers capture context and nuance that TF-IDF misses.
Implication: For production sentiment systems, transformers are worth the compute cost.

Finding 2: Logistic Regression hit {lr_acc*100:.1f}% on 25,000 test reviews.
Meaning:   TF-IDF features already capture strong signal for a fast, interpretable baseline.
Implication: Always benchmark transformers against this — if gain < 2%, use the cheaper model.

Finding 3: Naive Bayes scored {nb_acc*100:.1f}% — {abs(nb_acc-lr_acc)*100:.1f}% below Logistic Regression.
Meaning:   Naive Bayes' independence assumption limits accuracy on long reviews.
Implication: Use Naive Bayes only for quick prototyping, not production.

Finding 4: The ANN achieved {ann_acc*100:.1f}% after 5 epochs.
Meaning:   TF-IDF loss of word order caps ANN performance near classical models.
Implication: Sequence models (CNN/RNN/Transformer) preserve order and outperform feed-forward ANNs.

Finding 5: DistilBERT's confidence on custom reviews ranged {min(r['score'] for r in results_custom):.2f}–{max(r['score'] for r in results_custom):.2f}.
Meaning:   The model is well-calibrated on clear sentiment and uncertain on mixed reviews.
Implication: Expose confidence scores in production to route low-confidence cases to human review.
""")

print("""
LIMITATIONS
1. The ANN was trained on TF-IDF features which lose word order — a CNN, RNN, or
   Transformer would preserve sequence and likely improve accuracy.
2. DistilBERT was evaluated on only 500 samples (not the full 25,000 test set)
   due to compute limits — full evaluation may shift the reported accuracy.
""")