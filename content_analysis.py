import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3
from collections import Counter
import re
import json

# Create output directory
output_dir = "content_analysis"
os.makedirs(output_dir, exist_ok=True)

# Set of common English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn',
    'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn'
}

def tokenize_text(text):
    """Simple function to tokenize text into words."""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase and split on non-alphanumeric characters
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    
    # Remove stopwords and short words
    return [word for word in words if word not in STOPWORDS and len(word) > 2]

# Function to execute SQL and return DataFrame
def run_query(query, conn):
    return pd.read_sql_query(query, conn)

# Connect to the database
try:
    conn = sqlite3.connect('/Users/pup/Desktop/Arch/conversations.db')
    print("Connected to database successfully")
except Exception as e:
    print(f"Error connecting to database: {e}")
    exit(1)

# Get assistant messages by platform (limit to 1000 per platform to keep processing manageable)
platforms = ['claude', 'chatgpt']
messages_by_platform = {}

print("Retrieving message data...")
for platform in platforms:
    query = f"""
        SELECT m.content
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE c.platform = '{platform}' AND m.sender = 'assistant'
        LIMIT 1000
    """
    try:
        messages_by_platform[platform] = run_query(query, conn)['content'].tolist()
        print(f"Retrieved {len(messages_by_platform[platform])} messages for {platform}")
    except Exception as e:
        print(f"Error retrieving data for {platform}: {e}")
        messages_by_platform[platform] = []

# Analyze word frequencies
print("Analyzing word frequencies...")
platform_word_freq = {}
for platform, messages in messages_by_platform.items():
    all_words = []
    
    for message in messages:
        words = tokenize_text(message)
        all_words.extend(words)
    
    # Get word frequencies
    word_freq = Counter(all_words)
    platform_word_freq[platform] = word_freq
    
    # Save top words
    top_words = pd.DataFrame(word_freq.most_common(100), columns=['word', 'frequency'])
    top_words.to_csv(os.path.join(output_dir, f'{platform}_top_words.csv'), index=False)
    print(f"Saved {platform}_top_words.csv")
    
    # Plot top 20 words
    plt.figure(figsize=(12, 6))
    top_20 = dict(word_freq.most_common(20))
    plt.bar(top_20.keys(), top_20.values())
    plt.title(f'Top 20 Words in {platform.capitalize()} Assistant Responses')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{platform}_top_words.png'))
    print(f"Saved {platform}_top_words.png")

# Compare vocabulary between platforms
print("Comparing vocabulary between platforms...")
if len(platform_word_freq) >= 2:
    common_words = set(word for word, freq in platform_word_freq['claude'].items() if freq > 5) & \
                set(word for word, freq in platform_word_freq['chatgpt'].items() if freq > 5)
    
    claude_unique = set(word for word, freq in platform_word_freq['claude'].items() if freq > 5) - \
                    set(word for word, freq in platform_word_freq['chatgpt'].items() if freq > 2)
    
    chatgpt_unique = set(word for word, freq in platform_word_freq['chatgpt'].items() if freq > 5) - \
                    set(word for word, freq in platform_word_freq['claude'].items() if freq > 2)
    
    # Save vocabulary comparison
    with open(os.path.join(output_dir, 'vocabulary_comparison.txt'), 'w') as f:
        f.write(f"Common words: {len(common_words)}\n")
        f.write(f"Claude unique words: {len(claude_unique)}\n")
        f.write(f"ChatGPT unique words: {len(chatgpt_unique)}\n\n")
        
        f.write("Sample of common words:\n")
        f.write(", ".join(list(common_words)[:100]) + "\n\n")
        
        f.write("Sample of Claude unique words:\n")
        f.write(", ".join(list(claude_unique)[:100]) + "\n\n")
        
        f.write("Sample of ChatGPT unique words:\n")
        f.write(", ".join(list(chatgpt_unique)[:100]) + "\n\n")
    
    print(f"Saved vocabulary_comparison.txt")

# Generate response style analysis
print("Analyzing response styles...")
style_query = """
    SELECT 
        c.platform,
        AVG(LENGTH(m.content)) as avg_length,
        AVG(LENGTH(m.content) - LENGTH(REPLACE(m.content, '.', ''))) as avg_sentences,
        AVG(LENGTH(m.content) - LENGTH(REPLACE(m.content, ' ', ''))) as avg_words
    FROM messages m
    JOIN conversations c ON m.conversation_id = c.id
    WHERE m.sender = 'assistant'
    GROUP BY c.platform
"""
try:
    style_data = run_query(style_query, conn)
    style_data.to_csv(os.path.join(output_dir, 'response_style_comparison.csv'), index=False)
    print(f"Saved response_style_comparison.csv")
    
    # Create a styled DataFrame for better visualization
    with open(os.path.join(output_dir, 'response_style_comparison.html'), 'w') as f:
        f.write(style_data.to_html())
    print(f"Saved response_style_comparison.html")
except Exception as e:
    print(f"Error analyzing response styles: {e}")

print(f"\nContent analysis complete. All results saved to {output_dir}/ directory")
