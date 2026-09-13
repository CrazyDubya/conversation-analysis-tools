# Ported from ai-conversation-analyzer/vector.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: main() now takes --input-dir (env ANALYZED_DIR); otherwise verbatim

import json
import os

import nltk
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')


def read_json_files(folder_path):
    keywords_list = []
    summaries_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                keywords = data.get('keywords', [])
                summary = data.get('summary', '')
                keywords_list.append(' '.join(keywords))
                summaries_list.append(summary)
    return keywords_list, summaries_list


def preprocess_text(text):
    tokens = nltk.word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words


def vectorize_ngrams(words, n, threshold=0.05):
    n_grams = [' '.join(ngram) for ngram in ngrams(words, n) if len(ngram) == n]
    vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = vectorizer.fit_transform(n_grams)
    scores = tfidf_matrix.toarray().flatten()

    n_gram_scores = {ngram: score for ngram, score in zip(n_grams, scores) if score > threshold}  # Adjusted threshold
    return n_gram_scores


def save_ngrams_to_csv(keywords, summaries, folder_path):
    combined_text = preprocess_text(' '.join(keywords + summaries))

    # Generate separate TF-IDF scores for each n-gram order and save to CSV
    for n in range(2, 7):
        ngram_scores = vectorize_ngrams(combined_text, n)
        df = pd.DataFrame(list(ngram_scores.items()), columns=['N-gram', 'TF-IDF Score'])
        df.to_csv(os.path.join(folder_path, f'ngram_{n}.csv'), index=False)
        print(f"Top {n}-grams saved to ngram_{n}.csv")


def main(folder_path=None):
    import argparse
    ap = argparse.ArgumentParser(description='TF-IDF n-gram analysis over analyzed_conversations JSONs.')
    ap.add_argument('--input-dir', default=os.environ.get('ANALYZED_DIR', 'analyzed_conversations'))
    folder_path = ap.parse_args().input_dir if folder_path is None else folder_path
    keywords, summaries = read_json_files(folder_path)

    # Save ngrams to different CSVs
    save_ngrams_to_csv(keywords, summaries, folder_path)


if __name__ == "__main__":
    main()
