# Ported from ai-conversation-analyzer/conversation_analysis.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: winner of twin diff (summarizes to 3 sentences vs og 5); hardcoded path -> CONVERSATIONS_JSON env/relative default

import json
import os
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Load the conversations.json file
file_path = os.environ.get('CONVERSATIONS_JSON', 'conversations.json')
with open(file_path, 'r') as file:
    conversations_data = json.load(file)


# Function to extract keywords from text
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(10)]


# Function to generate a summary from text using sumy
def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join([str(sentence) for sentence in summary])


# Extract and analyze conversation data
def analyze_conversations(conversations):
    analyzed_data = []
    for conversation in conversations:
        conversation_id = conversation['id']
        title = conversation.get('title', 'Untitled')
        messages = []
        full_text = ''
        for node in conversation['mapping'].values():
            if node.get('message') and node['message']['content']:
                message_content = node['message']['content'].get('parts', [])
                author_role = node['message']['author']['role']
                text_parts = [part if isinstance(part, str) else part.get('text', '') for part in message_content]
                text = ' '.join(text_parts)
                full_text += ' ' + text
                messages.append({
                    'author': author_role,
                    'content': text
                })

        keywords = extract_keywords(full_text)
        summary = generate_summary(full_text)

        analyzed_data.append({
            'conversation_id': conversation_id,
            'title': title,
            'messages': messages,
            'keywords': keywords,
            'summary': summary
        })
    return analyzed_data


# Define the output directory
output_dir = './analyzed_conversations'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Analyze the conversations and save to new JSON files
analyzed_conversations = analyze_conversations(conversations_data)
for conversation in analyzed_conversations:
    output_file = os.path.join(output_dir, f"{conversation['title'][:10]}_{conversation['conversation_id']}.json")
    with open(output_file, 'w') as file:
        json.dump(conversation, file, indent=4)

print(f"Analyzed conversations have been saved to {output_dir}")
