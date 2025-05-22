# Conversation Database Analysis Tools

This repository contains scripts for analyzing conversation data from Claude and ChatGPT stored in a SQLite database.

## Overview

The database contains:
- `conversations` table: Metadata about each conversation
- `messages` table: Individual messages within conversations

## Database Views

The `create_views.sql` script creates the following views:

1. `message_pairs`: Pairs of human messages and assistant responses
2. `conversation_summary`: Overview statistics for each conversation
3. `message_length_stats`: Statistics about message lengths
4. `time_activity`: Time-based conversation activity data
5. `model_usage`: Usage statistics by model

## Analysis Scripts

1. `analyze_conversations.py`: Generates visualizations and exports CSV data
2. `content_analysis.py`: Performs text analysis on message content
3. `advanced_queries.sql`: Contains SQL queries for deeper data exploration

## Usage

1. Create the views:
   ```
   sqlite3 conversations.db < create_views.sql
   ```

2. Run the analysis scripts:
   ```
   python analyze_conversations.py
   python content_analysis.py
   ```

3. Or use the all-in-one script:
   ```
   ./run_analysis.sh
   ```

## Output

All visualizations and data exports will be saved to:
- `./visualizations/`: For general conversation analytics
- `./content_analysis/`: For text content analysis

## Advanced Analysis

To run specific advanced queries:
```
sqlite3 conversations.db < advanced_queries.sql
```

Or to run a specific query:
```
sqlite3 conversations.db "SELECT * FROM conversation_summary LIMIT 10;"
```
