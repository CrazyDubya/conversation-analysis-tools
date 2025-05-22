import os
import json
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
from uni_parse import ConversationParser
import numpy as np

# Create a parser instance
parser = ConversationParser("conversations.db")


def import_conversations():
    """Import conversations from both platforms."""
    # Process Claude conversations
    claude_path = os.path.join("Desktop", "Arch", "claude", "conversations.json")
    if os.path.exists(claude_path):
        print(f"Processing Claude conversations from {claude_path}...")
        parser.parse_file(claude_path, platform="claude")

    # Process ChatGPT conversations
    chatgpt_path = os.path.join("Desktop", "Arch", "chatgpt", "conversations.json")
    if os.path.exists(chatgpt_path):
        print(f"Processing ChatGPT conversations from {chatgpt_path}...")
        parser.parse_file(chatgpt_path, platform="chatgpt")


def analyze_conversations():
    """Analyze and compare conversations from both platforms."""
    conn = sqlite3.connect("conversations.db")

    # Get conversation counts by platform
    platform_counts = pd.read_sql(
        """
        SELECT platform, COUNT(*) as count 
        FROM conversations 
        GROUP BY platform
        """,
        conn,
    )

    # Get message counts by platform and role
    message_counts = pd.read_sql(
        """
        SELECT c.platform, m.role, COUNT(*) as count 
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        GROUP BY c.platform, m.role
        """,
        conn,
    )

    # Get average messages per conversation by platform
    avg_messages = pd.read_sql(
        """
        SELECT c.platform, 
               COUNT(m.id) as total_messages,
               COUNT(DISTINCT c.id) as total_conversations,
               CAST(COUNT(m.id) AS FLOAT) / COUNT(DISTINCT c.id) as avg_messages_per_conversation
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        GROUP BY c.platform
        """,
        conn,
    )

    # Get conversation creation distribution by month
    time_distribution = pd.read_sql(
        """
        SELECT platform, 
               strftime('%Y-%m', created_at) as month,
               COUNT(*) as count
        FROM conversations
        GROUP BY platform, month
        ORDER BY month
        """,
        conn,
    )

    # Get message length statistics
    message_lengths = pd.read_sql(
        """
        SELECT c.platform, m.role, 
               AVG(LENGTH(m.content)) as avg_length,
               MIN(LENGTH(m.content)) as min_length,
               MAX(LENGTH(m.content)) as max_length
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        GROUP BY c.platform, m.role
        """,
        conn,
    )

    conn.close()

    # Plotting the results

    # Platform conversation counts
    plt.figure(figsize=(10, 6))
    plt.bar(platform_counts["platform"], platform_counts["count"])
    plt.title("Number of Conversations by Platform")
    plt.ylabel("Count")
    plt.xlabel("Platform")
    plt.savefig("platform_conversation_counts.png")

    # Message counts by role and platform
    plt.figure(figsize=(12, 6))
    pivot_data = message_counts.pivot(index="platform", columns="role", values="count")
    pivot_data.plot(kind="bar")
    plt.title("Message Counts by Role and Platform")
    plt.ylabel("Count")
    plt.xlabel("Platform")
    plt.savefig("message_role_counts.png")

    # Average messages per conversation
    plt.figure(figsize=(10, 6))
    plt.bar(avg_messages["platform"], avg_messages["avg_messages_per_conversation"])
    plt.title("Average Messages per Conversation")
    plt.ylabel("Average Messages")
    plt.xlabel("Platform")
    plt.savefig("avg_messages_per_conversation.png")

    # Time distribution
    plt.figure(figsize=(14, 6))
    for platform in time_distribution["platform"].unique():
        platform_data = time_distribution[time_distribution["platform"] == platform]
        plt.plot(
            platform_data["month"], platform_data["count"], label=platform, marker="o"
        )
    plt.title("Conversation Creation by Month")
    plt.ylabel("Count")
    plt.xlabel("Month")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("conversation_time_distribution.png")

    # Message length by platform and role
    plt.figure(figsize=(12, 6))
    platforms = message_lengths["platform"].unique()
    roles = message_lengths["role"].unique()

    x = np.arange(len(platforms))
    width = 0.35 / len(roles)

    for i, role in enumerate(roles):
        role_data = message_lengths[message_lengths["role"] == role]
        plt.bar(x + i * width, role_data["avg_length"], width, label=role)

    plt.title("Average Message Length by Platform and Role")
    plt.ylabel("Average Characters")
    plt.xlabel("Platform")
    plt.xticks(x + width, platforms)
    plt.legend()
    plt.savefig("message_lengths.png")

    # Print summary statistics
    print("=== Conversation Analysis Results ===")
    print("\nPlatform Conversation Counts:")
    print(platform_counts)

    print("\nMessage Counts by Role and Platform:")
    print(message_counts)

    print("\nAverage Messages per Conversation:")
    print(avg_messages)

    print("\nMessage Length Statistics:")
    print(message_lengths)

    print("\nAnalysis complete! Check the generated charts for visualizations.")


def extract_sample_conversation():
    """Extract a sample conversation with all messages for demonstration."""
    conn = sqlite3.connect("conversations.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get a conversation with at least 5 messages
    cursor.execute(
        """
        SELECT c.id, c.title, c.platform, COUNT(m.id) as message_count
        FROM conversations c
        JOIN messages m ON c.id = m.conversation_id
        GROUP BY c.id
        HAVING COUNT(m.id) >= 5
        LIMIT 1
    """
    )

    conversation = cursor.fetchone()
    if not conversation:
        print("No conversations with enough messages found.")
        conn.close()
        return

    conv_id = conversation["id"]

    # Get the full conversation with messages
    cursor.execute(
        """
        SELECT c.*, json_group_array(
            json_object(
                'id', m.id,
                'sender', m.sender,
                'role', m.role,
                'content', m.content,
                'created_at', m.created_at,
                'order_index', m.order_index
            )
        ) as messages
        FROM conversations c
        JOIN messages m ON c.id = m.conversation_id
        WHERE c.id = ?
        GROUP BY c.id
    """,
        (conv_id,),
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        # Convert to dictionary and parse the JSON messages
        conv_dict = dict(result)
        try:
            conv_dict["messages"] = json.loads(conv_dict["messages"])

            # Save to a file
            with open("sample_conversation.json", "w") as f:
                json.dump(conv_dict, f, indent=2)

            print(f"Sample conversation saved to sample_conversation.json")
            print(f"Platform: {conv_dict['platform']}, Title: {conv_dict['title']}")
            print(f"Message count: {len(conv_dict['messages'])}")
        except json.JSONDecodeError:
            print("Error parsing messages JSON")
    else:
        print("Error retrieving conversation")


def find_xml_content():
    """Find conversations with XML-like content (common in Claude)."""
    conn = sqlite3.connect("conversations.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT c.id, c.title, c.platform, m.content
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE m.content LIKE '%<%>%'
        LIMIT 10
    """
    )

    results = cursor.fetchall()
    conn.close()

    print(f"Found {len(results)} messages with XML-like content")
    for result in results:
        print(f"Platform: {result[2]}, Conversation: {result[1]}")
        content_preview = result[3][:100] + "..." if len(result[3]) > 100 else result[3]
        print(f"Content preview: {content_preview}\n")


def main():
    """Main function to run all analyses."""
    # Create database tables
    print("Initializing database...")
    parser.initialize_database()

    # Import conversations (uncomment when ready)
    # import_conversations()

    # Run analyses
    print("Running analyses...")
    analyze_conversations()

    # Extract sample
    extract_sample_conversation()

    # Find XML content
    find_xml_content()


if __name__ == "__main__":
    main()
