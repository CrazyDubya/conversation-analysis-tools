# Ported from ai-conversation-analyzer/analyzed_conversations-duo-concu.py @ a616deaf (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: newer of the duo pair (threaded, tqdm); hardcoded path -> --input-dir/env CONV_ANALYSIS_INPUT_DIR; module-level execution wrapped in main(); outputs -> --output-dir

import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from tqdm import tqdm

# Download NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Define the directory containing individual JSON files
directory_path = os.environ.get('CONV_ANALYSIS_INPUT_DIR', 'gptlogs')


# Function to extract keywords from text
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(10)]


# Function to generate a summary from text using LexRank
def generate_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, 3)  # Summarize to 3 sentences
    return ' '.join([str(sentence) for sentence in summary])


# Extract and analyze conversation data
def analyze_conversation(file_path):
    with open(file_path, 'r') as file:
        conversation = json.load(file)

    conversation_id = conversation['conversation_id']
    title = conversation.get('title', 'Untitled')
    messages = []
    full_text = ''
    for node in conversation['messages']:
        if node.get('content'):
            message_content = node.get('content', [])
            author_role = node['author']
            text_parts = [part if isinstance(part, str) else part.get('text', '') for part in message_content]
            text = ' '.join(text_parts)
            full_text += ' ' + text
            messages.append({
                'author': author_role,
                'content': text
            })

    keywords = extract_keywords(full_text)
    summary = generate_summary(full_text)

    return {
        'conversation_id': conversation_id,
        'title': title,
        'messages': messages,
        'keywords': keywords,
        'summary': summary
    }




def main():
    import argparse
    ap = argparse.ArgumentParser(description='Keyword/summary analysis over per-conversation JSONs (threaded).')
    ap.add_argument('--input-dir', default=directory_path)
    ap.add_argument('--output-dir', default='./analyzed_conversations-3')
    args = ap.parse_args()
    directory_path_local = args.input_dir

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize progress bar
    json_files = [os.path.join(directory_path_local, filename) for filename in os.listdir(directory_path_local) if
                  filename.endswith('.json')]
    total_conversations = len(json_files)
    progress_bar = tqdm(total=total_conversations, desc='Analyzing conversations')

    # Analyze the conversations concurrently
    analyzed_conversations = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_conversation, file_path): file_path for file_path in json_files}
        for future in as_completed(futures):
            analyzed_conversation = future.result()
            analyzed_conversations.append(analyzed_conversation)
            output_file = os.path.join(output_dir,
                                       f"{analyzed_conversation['title'][:10]}_{analyzed_conversation['conversation_id']}.json")
            with open(output_file, 'w') as file:
                json.dump(analyzed_conversation, file, indent=4)
            progress_bar.update(1)
            elapsed_time = time.time() - start_time
            progress_bar.set_postfix(elapsed_time=f'{elapsed_time:.2f}s', refresh=False)

    progress_bar.close()

    # Save all analyzed data to a CSV file
    csv_output_file = os.path.join(args.output_dir, 'analyzed_conversations.csv')
    csv_headers = ['conversation_id', 'title', 'author', 'content', 'keywords', 'summary']

    with open(csv_output_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(csv_headers)
        for conversation in analyzed_conversations:
            for message in conversation['messages']:
                csv_writer.writerow([
                    conversation['conversation_id'],
                    conversation['title'],
                    message['author'],
                    message['content'],
                    ', '.join(conversation['keywords']),
                    conversation['summary']
                ])

    print(f"Analyzed conversations have been saved to {output_dir}")
    print(f"CSV file with all data saved as {csv_output_file}")


if __name__ == '__main__':
    main()
