#!/bin/bash

# Create views if they don't exist
echo "Creating database views..."
sqlite3 conversations.db < create_views.sql

# Run the conversation analysis script
echo "Running conversation analysis..."
python3 analyze_conversations.py

# Run the content analysis script
echo "Running content analysis..."
python3 content_analysis.py

echo "Analysis complete!"
echo "Results are available in ./visualizations/ and ./content_analysis/ directories"
