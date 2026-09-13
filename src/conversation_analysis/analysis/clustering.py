# Ported from ai-conversation-analyzer/cluster.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: rewritten as functions; hardcoded /Users/puppuccino/... path is now a CLI arg/env var; Agg backend

"""KMeans clustering of conversation summaries (TF-IDF)."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def load_summaries(path):
    with open(path) as f:
        conversations_data = json.load(f)
    titles, summaries = [], []
    for conversation in conversations_data:
        titles.append(conversation.get('title', 'Untitled'))
        summary = ' '.join(
            part
            for node in conversation.get('mapping', {}).values()
            if node.get('message') and 'parts' in node['message'].get('content', {})
            for part in node['message']['content']['parts']
            if isinstance(part, str)
        )
        summaries.append(summary)
    return titles, summaries


def cluster_conversations(path, n_clusters=32, output='conversation_clusters.png'):
    titles, summaries = load_summaries(path)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(summaries)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    cluster_df = pd.DataFrame({'Title': titles, 'Cluster': labels})
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Cluster', data=cluster_df)
    plt.title('Conversation Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Conversations')
    plt.savefig(output)
    plt.close()
    return cluster_df


def main():
    ap = argparse.ArgumentParser(description='KMeans-cluster conversation summaries.')
    ap.add_argument('--input', default=os.environ.get('CONVERSATIONS_JSON', 'conversations.json'))
    ap.add_argument('--clusters', type=int, default=32)
    ap.add_argument('--output', default='conversation_clusters.png')
    args = ap.parse_args()
    df = cluster_conversations(args.input, args.clusters, args.output)
    print(df['Cluster'].value_counts())


if __name__ == '__main__':
    main()
