# Ported from ai-conversation-analyzer/anomaly.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: rewritten as functions; input path is now a CLI arg; Agg backend

"""Anomaly detection over conversations (IsolationForest on message counts)."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies(path, output='anomaly_detection.png', contamination=0.1):
    with open(path) as f:
        data = json.load(f)
    features = [len(conversation.get('mapping', {})) for conversation in data]
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(np.array(features).reshape(-1, 1))
    plt.scatter(range(len(features)), features, c=preds)
    plt.title('Anomaly Detection in Conversations')
    plt.xlabel('Conversation Index')
    plt.ylabel('Number of Messages')
    plt.savefig(output)
    plt.close()
    return preds


def main():
    ap = argparse.ArgumentParser(description='Detect anomalous conversations.')
    ap.add_argument('--input', default=os.environ.get('CONVERSATIONS_JSON', 'conversations.json'))
    ap.add_argument('--output', default='anomaly_detection.png')
    args = ap.parse_args()
    preds = detect_anomalies(args.input, args.output)
    print(f'{sum(1 for p in preds if p == -1)} anomalies / {len(preds)} conversations')


if __name__ == '__main__':
    main()
