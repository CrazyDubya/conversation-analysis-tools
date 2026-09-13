# Ported from chatgpt-archive-clean/chatgptarchive-og/analyzeuserandasst.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: winner of twin diff: identical logic to the ai-conversation-analyzer copy except og uses relative 'conversations.json' path instead of a hardcoded /Users/puppuccino/... path

import json
import re
from collections import Counter

import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.util import ngrams
from textblob import TextBlob
from wordcloud import WordCloud

# Ensure the stopwords and other resources are downloaded
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Load the conversations.json file
file_path = 'conversations.json'
with open(file_path, 'r') as file:
    conversations_data = json.load(file)

# Extract messages from the 'mapping' key with a check for NoneType
user_messages = []
assistant_messages = []

for conversation in conversations_data:
    for node in conversation['mapping'].values():
        if node.get('message') and node['message'].get('content'):
            author_role = node['message']['author']['role']
            message_content = node['message']['content'].get('parts', [])
            if author_role == 'user':
                user_messages.extend(message_content)
            elif author_role == 'assistant':
                assistant_messages.extend(message_content)


# Function to perform text analysis
def analyze_text(messages, title):
    # Filter out non-text messages
    text_messages = [message for message in messages if isinstance(message, str)]

    # Join all messages into a single string
    all_text = ' '.join(text_messages)

    # Tokenize the text by words, converting to lowercase
    words = re.findall(r'\b\w+\b', all_text.lower())

    # Get the list of English stopwords
    stop_words = set(stopwords.words('english'))

    # Filter out stop words
    filtered_words = [word for word in words if word not in stop_words]

    # Count the frequency of each word
    word_counts = Counter(filtered_words)

    # Count the number of unique words
    unique_word_count = len(word_counts)

    # Get the 20 most common words
    common_words = word_counts.most_common(20)

    # Generate n-grams
    def get_ngrams(words, n):
        ngrams_list = ngrams(words, n)
        return Counter(ngrams_list).most_common(20)

    common_2grams = get_ngrams(filtered_words, 2)
    common_3grams = get_ngrams(filtered_words, 3)
    common_4grams = get_ngrams(filtered_words, 4)

    # Perform POS tagging
    pos_tags = pos_tag(filtered_words)
    pos_counts = Counter(tag for word, tag in pos_tags)
    common_pos_tags = pos_counts.most_common(20)

    # Create and display a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

    # Perform sentiment analysis on each text message
    sentiments = [TextBlob(message).sentiment.polarity for message in text_messages if message.strip()]

    # Calculate the overall sentiment
    overall_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    print(f"Overall sentiment for {title}: {overall_sentiment}")

    # Save the results to a file
    with open(f'analysis_results_{title}.txt', 'w') as result_file:
        result_file.write(f"Most common words for {title}:\n")
        for word, count in common_words:
            result_file.write(f"{word}: {count}\n")
        result_file.write(f"\nNumber of unique words: {unique_word_count}\n")
        result_file.write(f"\nMost common 2-grams:\n")
        for gram, count in common_2grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 3-grams:\n")
        for gram, count in common_3grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common 4-grams:\n")
        for gram, count in common_4grams:
            result_file.write(f"{' '.join(gram)}: {count}\n")
        result_file.write(f"\nMost common POS tags:\n")
        for tag, count in common_pos_tags:
            result_file.write(f"{tag}: {count}\n")
        result_file.write(f"\nOverall sentiment: {overall_sentiment}\n")


# Analyze user messages
analyze_text(user_messages, "User")

# Analyze assistant messages
analyze_text(assistant_messages, "Assistant")
