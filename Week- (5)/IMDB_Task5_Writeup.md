# IMDB Sentiment Analysis — Week 05

**Author:** Recently graduated Software Engineering student (Data Science foundation)  
**Date:** 2025-08-19 05:19

## Objective
Build a binary text classifier to predict whether an IMDB movie review is **positive (1)** or **negative (0)** using NLP.

## Approach
1. **EDA:** Removed duplicates, checked nulls, and visualized sentiment distribution.  
2. **Preprocessing:** HTML tag removal, non-letters stripping, lowercasing, and stopword removal. Encoded `sentiment` as 1/0.  
3. **Vectorization:** `TfidfVectorizer(max_features=5000)` on cleaned text.  
4. **Model:** `MultinomialNB` trained on the TF-IDF features.  
5. **Evaluation:** Accuracy, precision, recall, F1-score, and confusion matrix.  
6. **Interpretability:** Extracted top words associated with positive/negative predictions via log-probability differences.

## Results
- **Accuracy:** 0.8520

### Classification Report
```
              precision    recall  f1-score   support

    negative       0.85      0.85      0.85      4940
    positive       0.85      0.86      0.85      4977

    accuracy                           0.85      9917
   macro avg       0.85      0.85      0.85      9917
weighted avg       0.85      0.85      0.85      9917

```

### Confusion Matrix
Saved as `confusion_matrix.png`.

## Files Produced
- `sentiment_model.pkl` — trained Naive Bayes model  
- `tfidf_vectorizer.pkl` — trained TF-IDF vectorizer  
- `top_positive_words.csv`, `top_negative_words.csv` — most indicative tokens  
- `IMDB_Sentiment_Analysis_Naive_Bayes.ipynb` — full reproducible notebook  
- `sentiment_count.png`, `sentiment_pie.png`, `confusion_matrix.png` — EDA & evaluation charts

## Notes
- The pipeline is balanced and lightweight, making it ideal as a strong baseline.  
- Further improvements: hyperparameter tuning, n-grams, class balancing, or deep learning models (LSTM/CNN/Transformers).
