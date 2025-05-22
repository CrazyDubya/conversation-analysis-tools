import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import sqlite3
from datetime import datetime

# Create output directory for visualizations
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

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

# 1. Conversation Activity Over Time
print("Analyzing conversation activity over time...")
time_data_query = """
    SELECT 
        platform,
        strftime('%Y-%m', activity_date) AS month,
        SUM(conversations) AS total_conversations
    FROM time_activity
    GROUP BY platform, month
    ORDER BY month, platform;
"""
try:
    time_data = run_query(time_data_query, conn)
    
    # Plot time series
    plt.figure(figsize=(12, 6))
    platforms = time_data['platform'].unique()
    
    for platform in platforms:
        platform_data = time_data[time_data['platform'] == platform]
        plt.plot(platform_data['month'], platform_data['total_conversations'], marker='o', label=platform)
    
    plt.title('Conversation Activity Over Time')
    plt.xlabel('Month')
    plt.ylabel('Total Conversations')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activity_over_time.png'))
    print(f"Saved activity_over_time.png")
except Exception as e:
    print(f"Error processing time activity data: {e}")

# 2. Average Response Length Comparison
print("Analyzing message length comparison...")
length_data_query = """
    SELECT 
        c.platform,
        m.sender,
        AVG(LENGTH(m.content)) AS avg_length
    FROM messages m
    JOIN conversations c ON m.conversation_id = c.id
    GROUP BY c.platform, m.sender
    HAVING m.sender IN ('human', 'assistant')
"""
try:
    length_data = run_query(length_data_query, conn)
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    platforms = length_data['platform'].unique()
    senders = length_data['sender'].unique()
    
    x = range(len(platforms))
    width = 0.35
    
    for i, sender in enumerate(senders):
        sender_data = length_data[length_data['sender'] == sender]
        plt.bar([p + i * width for p in x], sender_data['avg_length'], width=width, label=sender)
    
    plt.title('Average Message Length by Platform and Sender')
    plt.xlabel('Platform')
    plt.ylabel('Average Length (characters)')
    plt.xticks([p + width/2 for p in x], platforms)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'message_length_comparison.png'))
    print(f"Saved message_length_comparison.png")
except Exception as e:
    print(f"Error processing message length data: {e}")

# 3. Model usage
print("Analyzing model usage...")
model_query = "SELECT * FROM model_usage"
try:
    model_data = run_query(model_query, conn)
    
    if not model_data.empty:
        plt.figure(figsize=(12, 8))
        platforms = model_data['platform'].unique()
        
        # Create a grouped bar chart
        models = model_data['model'].unique()
        model_groups = {}
        
        for platform in platforms:
            platform_data = model_data[model_data['platform'] == platform]
            model_groups[platform] = {model: 0 for model in models}
            
            for _, row in platform_data.iterrows():
                model_groups[platform][row['model']] = row['message_count']
        
        # Plot
        bar_width = 0.35
        index = range(len(models))
        
        for i, platform in enumerate(platforms):
            values = [model_groups[platform][model] for model in models]
            plt.bar([idx + i * bar_width for idx in index], values, bar_width, label=platform)
        
        plt.title('Message Count by Model')
        plt.xlabel('Model')
        plt.ylabel('Message Count')
        plt.xticks([idx + bar_width/2 for idx in index], models, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_usage.png'))
        print(f"Saved model_usage.png")
        
        # Save CSV
        model_data.to_csv(os.path.join(output_dir, 'model_usage.csv'), index=False)
        print(f"Saved model_usage.csv")
except Exception as e:
    print(f"Error processing model data: {e}")

# 4. Most Active Conversations
print("Finding most active conversations...")
active_query = """
    SELECT 
        conversation_id,
        title,
        platform,
        message_count,
        human_messages,
        assistant_messages,
        conversation_duration_minutes,
        ROUND(conversation_duration_minutes / 60.0, 2) AS conversation_hours
    FROM conversation_summary
    ORDER BY message_count DESC
    LIMIT 20
"""
try:
    active_conversations = run_query(active_query, conn)
    active_conversations.to_csv(os.path.join(output_dir, 'most_active_conversations.csv'), index=False)
    print(f"Saved most_active_conversations.csv")
    
    # Create a bar chart of most active conversations
    plt.figure(figsize=(14, 8))
    platforms = active_conversations['platform'].unique()
    colors = {'claude': 'blue', 'chatgpt': 'green'}
    
    plt.bar(active_conversations['conversation_id'], 
            active_conversations['message_count'],
            color=[colors.get(p, 'gray') for p in active_conversations['platform']])
    
    plt.title('Top 20 Most Active Conversations')
    plt.xlabel('Conversation ID')
    plt.ylabel('Message Count')
    plt.xticks(rotation=90)
    
    # Add a legend
    for platform, color in colors.items():
        plt.bar([], [], color=color, label=platform)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'most_active_conversations.png'))
    print(f"Saved most_active_conversations.png")
except Exception as e:
    print(f"Error processing active conversations data: {e}")

# 5. Content Pattern Analysis
print("Analyzing content patterns...")
pattern_query = """
    SELECT 
        c.id,
        c.title,
        c.platform,
        COUNT(m.id) AS matching_messages
    FROM conversations c
    JOIN messages m ON c.id = m.conversation_id
    WHERE m.content LIKE '%machine learning%' OR m.content LIKE '%AI%' OR m.content LIKE '%neural network%'
    GROUP BY c.id
    HAVING matching_messages > 1
    ORDER BY matching_messages DESC
"""
try:
    pattern_data = run_query(pattern_query, conn)
    pattern_data.to_csv(os.path.join(output_dir, 'ai_topic_conversations.csv'), index=False)
    print(f"Saved ai_topic_conversations.csv")
    
    # Create a bar chart for top 10 conversations with AI topics
    if not pattern_data.empty:
        top_10 = pattern_data.head(10)
        plt.figure(figsize=(14, 8))
        
        # Create a bar chart
        plt.bar(top_10['id'], top_10['matching_messages'], 
                color=[colors.get(p, 'gray') for p in top_10['platform']])
        
        plt.title('Top 10 Conversations Discussing AI Topics')
        plt.xlabel('Conversation ID')
        plt.ylabel('AI Topic Mentions')
        plt.xticks(rotation=90)
        
        # Add a legend
        for platform, color in colors.items():
            plt.bar([], [], color=color, label=platform)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ai_topic_conversations.png'))
        print(f"Saved ai_topic_conversations.png")
except Exception as e:
    print(f"Error processing content pattern data: {e}")

# 6. Conversation Duration Distribution
print("Analyzing conversation duration distribution...")
duration_query = """
    SELECT 
        platform,
        conversation_duration_minutes
    FROM conversation_summary
    WHERE conversation_duration_minutes > 0 AND conversation_duration_minutes < 1000
"""
try:
    duration_data = run_query(duration_query, conn)
    
    plt.figure(figsize=(12, 6))
    platforms = duration_data['platform'].unique()
    
    for platform in platforms:
        platform_data = duration_data[duration_data['platform'] == platform]
        plt.hist(platform_data['conversation_duration_minutes'], bins=30, 
                 alpha=0.5, label=platform)
    
    plt.title('Conversation Duration Distribution (minutes)')
    plt.xlabel('Duration (minutes)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'conversation_duration_distribution.png'))
    print(f"Saved conversation_duration_distribution.png")
except Exception as e:
    print(f"Error processing duration data: {e}")

# 7. Export full conversation summary for further analysis
print("Exporting conversation summary...")
try:
    summary_data = run_query("SELECT * FROM conversation_summary", conn)
    summary_data.to_csv(os.path.join(output_dir, 'conversation_summary.csv'), index=False)
    print(f"Saved conversation_summary.csv")
except Exception as e:
    print(f"Error exporting conversation summary: {e}")

print(f"\nAnalysis complete. All results saved to {output_dir}/ directory")
