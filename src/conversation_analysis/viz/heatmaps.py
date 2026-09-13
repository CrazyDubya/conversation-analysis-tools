# Ported from ai-conversation-analyzer/heater3.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: newer of the heater pair (adds TextBlob sentiment); hardcoded dir -> --input-dir/env CONV_ANALYSIS_INPUT_DIR

# heater.py

import json
import multiprocessing
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tqdm import tqdm

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Define the directory containing JSON files
DEFAULT_DIRECTORY = os.environ.get('CONV_ANALYSIS_INPUT_DIR', 'analyzed_conversations-2')


# Preprocess the text data
def preprocess(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)


# Function to calculate sentiment
def calculate_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity


# Function to calculate complexity
def calculate_complexity(text):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    total_words = len(words)
    noun_count = len([word for word, pos in pos_tags if pos.startswith('NN')])
    verb_count = len([word for word, pos in pos_tags if pos.startswith('VB')])
    if total_words > 0:
        complexity_score = (noun_count + verb_count) / total_words
    else:
        complexity_score = 0
    return complexity_score


# Analyze the data
def analyze_data(file_path, result_queue):
    with open(file_path, 'r') as file:
        conversation = json.load(file)

    user_message = ''
    assistant_response = ''
    for message in conversation.get('messages', []):
        content = ' '.join(message.get('content', []))
        if message.get('author') == 'user':
            user_message += content + ' '
        elif message.get('author') == 'assistant':
            assistant_response += content + ' '

    sentiment_score = calculate_sentiment(user_message)
    complexity_score = calculate_complexity(user_message)
    creativity_score = len(set(preprocess(user_message))) / len(preprocess(user_message)) if len(
        preprocess(user_message)) > 0 else 0
    conversation_length = len(preprocess(user_message)) + len(preprocess(assistant_response))

    result_queue.put((user_message.strip(), assistant_response.strip(), sentiment_score, complexity_score,
                      creativity_score, conversation_length))


# Create the heatmaps
def create_heatmaps(user_messages, assistant_responses, sentiment_scores, complexity_scores, creativity_scores,
                    conversation_lengths):
    # Sentiment vs. Complexity
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([sentiment_scores, complexity_scores]).T, cmap='coolwarm', cbar=True)
    plt.title('Sentiment vs. Complexity')
    plt.xlabel('Complexity')
    plt.ylabel('Sentiment')
    plt.savefig('sentiment_vs_complexity.png')
    plt.close()

    # Creativity vs. Conversation Length
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([creativity_scores, conversation_lengths]).T, cmap='coolwarm', cbar=True)
    plt.title('Creativity vs. Conversation Length')
    plt.xlabel('Conversation Length')
    plt.ylabel('Creativity')
    plt.savefig('creativity_vs_conversation_length.png')
    plt.close()

    # Sentiment vs. Creativity vs. Complexity
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sentiment_scores, creativity_scores, complexity_scores, c=conversation_lengths, cmap='coolwarm')
    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Creativity')
    ax.set_zlabel('Complexity')
    plt.title('Sentiment vs. Creativity vs. Complexity')
    plt.savefig('sentiment_vs_creativity_vs_complexity.png')
    plt.close()

    # Topic Similarity vs. Conversation Length
    vectorizer = TfidfVectorizer()
    user_tfidf_matrix = vectorizer.fit_transform(user_messages)
    assistant_tfidf_matrix = vectorizer.transform(assistant_responses)
    similarity_scores = cosine_similarity(user_tfidf_matrix, assistant_tfidf_matrix).diagonal()

    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([similarity_scores, conversation_lengths]).T, cmap='coolwarm', cbar=True)
    plt.title('Topic Similarity vs. Conversation Length')
    plt.xlabel('Conversation Length')
    plt.ylabel('Topic Similarity')
    plt.savefig('topic_similarity_vs_conversation_length.png')
    plt.close()

    # Creativity vs. Complexity
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array([creativity_scores, complexity_scores]).T, cmap='coolwarm', cbar=True)
    plt.title('Creativity vs. Complexity')
    plt.xlabel('Complexity')
    plt.ylabel('Creativity')
    plt.savefig('creativity_vs_complexity.png')
    plt.close()


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Sentiment/complexity/creativity heatmaps over analyzed conversations.')
    ap.add_argument('--input-dir', default=DEFAULT_DIRECTORY)
    ap.add_argument('--output-dir', default='.')
    _a = ap.parse_args()
    directory = _a.input_dir
    os.chdir(_a.output_dir) if _a.output_dir != '.' else None
    multiprocessing.freeze_support()

    # Create a shared queue using Manager
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()

    # Get the list of JSON files
    json_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.json')]

    # Create a process pool
    with multiprocessing.Pool() as pool:
        # Process the files in parallel
        pool.starmap(analyze_data, [(file, result_queue) for file in tqdm(json_files, desc="Processing files")])

    # Collect the results from the queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Combine the results
    user_messages = []
    assistant_responses = []
    sentiment_scores = []
    complexity_scores = []
    creativity_scores = []
    conversation_lengths = []

    for result in results:
        user_messages.append(result[0])
        assistant_responses.append(result[1])
        sentiment_scores.append(result[2])
        complexity_scores.append(result[3])
        creativity_scores.append(result[4])
        conversation_lengths.append(result[5])

    # Create the heatmaps
    create_heatmaps(user_messages, assistant_responses, sentiment_scores, complexity_scores, creativity_scores,
                    conversation_lengths)
