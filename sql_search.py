#!/usr/bin/env python3
"""
Advanced Conversation Search Engine

This script provides powerful search capabilities for the conversation database,
including keyword search, semantic search using embeddings, and advanced filtering.

Features:
- Full-text search with Boolean operators
- Semantic similarity search using embeddings
- Advanced filtering by platform, model, date, etc.
- Context-aware result presentation
- Search result visualization
- Conversation export capabilities
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import re
import os
import json
from datetime import datetime, timedelta
import time
from collections import Counter, defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.cm as cm

# Database connection
DB_PATH = "/Users/pup/Desktop/Arch/conversations.db"

# Output directories
SEARCH_RESULTS_DIR = "search_results"
EMBEDDINGS_DIR = "embeddings_cache"
os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Platform colors for visualization
PLATFORM_COLORS = {
    "claude": "#8C52FF",  # Purple
    "chatgpt": "#00A67E",  # Green
}


class ConversationSearchEngine:
    """Advanced search engine for conversation data"""

    def __init__(self, db_path=DB_PATH):
        """Initialize the search engine with database connection"""
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Cache some frequently used data
        self.platforms = self._get_platforms()
        self.models = self._get_models()
        self.date_range = self._get_date_range()

        # Initialize embedding vectors if needed
        self.embeddings = {}
        self.embedding_loaded = False

        # Check if views exist
        self._check_views()

        print(f"Connected to database with {self._count_conversations()} conversations")
        print(f"Date range: {self.date_range[0]} to {self.date_range[1]}")
        print(f"Platforms: {', '.join(self.platforms)}")

    def _check_views(self):
        """Check if required views exist, create them if not"""
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='view'")
        views = {row[0] for row in self.cursor.fetchall()}
        required_views = {
            "message_pairs",
            "conversation_summary",
            "message_length_stats",
            "time_activity",
            "model_usage",
        }

        if not required_views.issubset(views):
            print("Some required views are missing. Creating views...")

            # Read and execute SQL from create_views.sql
            try:
                with open("create_views.sql", "r") as f:
                    sql = f.read()
                    self.conn.executescript(sql)
                print("Views created successfully")
            except Exception as e:
                print(f"Error creating views: {e}")

    def _get_platforms(self):
        """Get list of unique platforms in the database"""
        self.cursor.execute("SELECT DISTINCT platform FROM conversations")
        return [row[0] for row in self.cursor.fetchall() if row[0]]

    def _get_models(self):
        """Get list of unique models in the database"""
        self.cursor.execute(
            "SELECT DISTINCT model FROM messages WHERE model IS NOT NULL AND model != ''"
        )
        return [row[0] for row in self.cursor.fetchall()]

    def _get_date_range(self):
        """Get min and max dates in the database"""
        self.cursor.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM conversations"
        )
        row = self.cursor.fetchone()
        return row[0][:10], row[1][:10]  # Extract just the date part

    def _count_conversations(self):
        """Count total conversations in the database"""
        self.cursor.execute("SELECT COUNT(*) FROM conversations")
        return self.cursor.fetchone()[0]

    def query_to_dataframe(self, query, params=()):
        """Execute SQL query and return results as a pandas DataFrame"""
        return pd.read_sql_query(query, self.conn, params=params)

    def run_search_interface(self):
        """Run the interactive search interface"""
        while True:
            self._show_search_menu()
            choice = input("\nEnter your choice (q to quit): ").strip()

            if choice.lower() == "q":
                print("Exiting search interface...")
                break

            try:
                choice = int(choice)
                self._handle_search_menu_choice(choice)
            except ValueError:
                print("Invalid choice. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")

    def _show_search_menu(self):
        """Display the main search menu options"""
        print("\n==== CONVERSATION SEARCH ENGINE ====")
        print("1. Keyword Search")
        print("2. Advanced Boolean Search")
        print("3. Semantic Similarity Search")
        print("4. Search by Date Range")
        print("5. Search by Platform/Model")
        print("6. Topic-based Search")
        print("7. Context Window Search")
        print("8. Conversation Export")
        print("9. Generate/Update Embeddings")
        print("10. Analyze Search Results")
        print("q. Quit")

    def _handle_search_menu_choice(self, choice):
        """Handle the user's search menu selection"""
        if choice == 1:
            self.keyword_search()
        elif choice == 2:
            self.boolean_search()
        elif choice == 3:
            self.semantic_search()
        elif choice == 4:
            self.date_range_search()
        elif choice == 5:
            self.platform_model_search()
        elif choice == 6:
            self.topic_search()
        elif choice == 7:
            self.context_window_search()
        elif choice == 8:
            self.export_conversations()
        elif choice == 9:
            self.generate_embeddings()
        elif choice == 10:
            self.analyze_search_results()
        else:
            print("Invalid choice")

    def keyword_search(self):
        """Perform a simple keyword search across messages"""
        print("\n==== Keyword Search ====")

        # Get search query
        query = input("Enter search keywords: ").strip()
        if not query:
            print("Search query cannot be empty.")
            return

        # Get filters
        sender_filter = (
            input("Filter by sender (human/assistant/both) [both]: ").strip().lower()
            or "both"
        )
        platform_filter = (
            input(f"Filter by platform ({'/'.join(self.platforms)}/all) [all]: ")
            .strip()
            .lower()
            or "all"
        )

        # Build SQL query
        sql_query = """
            SELECT 
                m.id AS message_id,
                m.conversation_id,
                c.title AS conversation_title,
                c.platform,
                m.sender,
                m.content,
                m.created_at,
                m.model
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.content LIKE ?
        """

        # Apply filters
        params = [f"%{query}%"]

        if sender_filter != "both":
            sql_query += " AND m.sender = ?"
            params.append(sender_filter)

        if platform_filter != "all":
            sql_query += " AND c.platform = ?"
            params.append(platform_filter)

        # Add limit and order
        sql_query += " ORDER BY m.created_at DESC LIMIT 100"

        # Execute query
        start_time = time.time()
        results = self.query_to_dataframe(sql_query, params)
        search_time = time.time() - start_time

        # Display results
        if results.empty:
            print(f"No results found for '{query}'")
            return

        print(f"\nFound {len(results)} results in {search_time:.2f} seconds:")

        # Extract and display snippets
        for i, row in enumerate(results.itertuples(), 1):
            content = row.content
            # Find the position of the match in content
            match_pos = content.lower().find(query.lower())

            # Extract snippet with context
            context_size = 60  # characters before and after the match
            start = max(0, match_pos - context_size)
            end = min(len(content), match_pos + len(query) + context_size)

            # Add ellipsis if we're not starting from the beginning
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(content) else ""

            snippet = prefix + content[start:end] + suffix

            # Truncate title if it's too long
            title = row.conversation_title if row.conversation_title else "Untitled"
            if len(title) > 40:
                title = title[:37] + "..."

            # Highlight the query term
            pattern = re.compile(re.escape(query), re.IGNORECASE)
            snippet = pattern.sub(f"\033[1;31m{query}\033[0m", snippet)

            # Print result with metadata
            print(f"\n{i}. {title} - {row.platform} ({row.created_at})")
            print(f"   Sender: {row.sender}")
            print(f"   {snippet}")

        # Ask if user wants more details or visualization
        self._ask_for_result_actions(results, query)

    def boolean_search(self):
        """Perform an advanced boolean search with AND, OR, NOT operators"""
        print("\n==== Advanced Boolean Search ====")
        print("Supported operators: AND, OR, NOT (must be uppercase)")
        print("Example: machine learning AND python NOT javascript")

        # Get search query
        query = input("Enter boolean search query: ").strip()
        if not query:
            print("Search query cannot be empty.")
            return

        # Parse boolean query
        terms = []
        include_terms = []
        exclude_terms = []

        parts = re.findall(r"(NOT\s+\S+|\S+)", query)

        for part in parts:
            if part.startswith("NOT "):
                exclude_terms.append(part[4:])
            elif part not in ("AND", "OR"):
                include_terms.append(part)

        # Build SQL query - this is a simple implementation
        # For a more complex parser, we'd need a proper boolean expression evaluator
        sql_query = """
            SELECT 
                m.id AS message_id,
                m.conversation_id,
                c.title AS conversation_title,
                c.platform,
                m.sender,
                m.content,
                m.created_at,
                m.model
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE 1=1
        """

        params = []

        # Add include terms
        if "OR" in parts:
            # OR search
            or_conditions = []
            for term in include_terms:
                or_conditions.append("m.content LIKE ?")
                params.append(f"%{term}%")

            if or_conditions:
                sql_query += f" AND ({' OR '.join(or_conditions)})"
        else:
            # AND search (default)
            for term in include_terms:
                sql_query += " AND m.content LIKE ?"
                params.append(f"%{term}%")

        # Add exclude terms
        for term in exclude_terms:
            sql_query += " AND m.content NOT LIKE ?"
            params.append(f"%{term}%")

        # Add limit and order
        sql_query += " ORDER BY m.created_at DESC LIMIT 100"

        # Execute query
        start_time = time.time()
        results = self.query_to_dataframe(sql_query, params)
        search_time = time.time() - start_time

        # Display results
        if results.empty:
            print(f"No results found for '{query}'")
            return

        print(f"\nFound {len(results)} results in {search_time:.2f} seconds:")

        # Extract and display snippets
        for i, row in enumerate(results.itertuples(), 1):
            content = row.content

            # Create a snippet with context
            if len(content) > 150:
                snippet = content[:150] + "..."
            else:
                snippet = content

            # Truncate title if it's too long
            title = row.conversation_title if row.conversation_title else "Untitled"
            if len(title) > 40:
                title = title[:37] + "..."

            # Highlight the terms
            for term in include_terms:
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                snippet = pattern.sub(f"\033[1;31m{term}\033[0m", snippet)

            # Print result with metadata
            print(f"\n{i}. {title} - {row.platform} ({row.created_at})")
            print(f"   Sender: {row.sender}")
            print(f"   {snippet}")

        # Ask if user wants more details or visualization
        self._ask_for_result_actions(results, query)

    def semantic_search(self):
        """Perform a semantic similarity search using text embeddings"""
        print("\n==== Semantic Similarity Search ====")

        # Check if embeddings are generated
        if not self._check_embeddings_exist():
            print(
                "Message embeddings not found. Please generate embeddings first (option 9)."
            )
            return

        # Load embeddings if not already loaded
        if not self.embedding_loaded:
            self._load_embeddings()

        # Get search query
        query = input("Enter your search query: ").strip()
        if not query:
            print("Search query cannot be empty.")
            return

        # Get limit
        try:
            limit = int(input("Maximum number of results [20]: ") or "20")
        except ValueError:
            limit = 20

        # Get similarity threshold
        try:
            threshold = float(input("Similarity threshold (0.0-1.0) [0.3]: ") or "0.3")
            threshold = max(0.0, min(1.0, threshold))
        except ValueError:
            threshold = 0.3

        # Create query embedding using TF-IDF
        vectorizer = TfidfVectorizer()

        # Fit on all message content we have embeddings for
        message_ids = list(self.embeddings.keys())
        messages = [
            self.message_content_cache[msg_id]
            for msg_id in message_ids
            if msg_id in self.message_content_cache
        ]

        try:
            vectorizer.fit(messages + [query])
            query_vector = vectorizer.transform([query]).toarray()[0]

            # Calculate similarity scores with all message embeddings
            similarities = []
            for msg_id, embedding in self.embeddings.items():
                similarity = self._cosine_similarity(query_vector, embedding)
                if similarity >= threshold:
                    similarities.append((msg_id, similarity))

            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Limit results
            similarities = similarities[:limit]

            if not similarities:
                print(f"No results found with similarity threshold of {threshold}.")
                return

            # Get message details
            message_ids = [msg_id for msg_id, _ in similarities]
            placeholders = ",".join("?" for _ in message_ids)

            query = f"""
                SELECT 
                    m.id AS message_id,
                    m.conversation_id,
                    c.title AS conversation_title,
                    c.platform,
                    m.sender,
                    m.content,
                    m.created_at,
                    m.model
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE m.id IN ({placeholders})
                ORDER BY CASE m.id {' '.join(f'WHEN ? THEN {i}' for i in range(len(message_ids)))} END
            """

            # Execute query with message IDs and their order
            results = self.query_to_dataframe(query, message_ids + message_ids)

            # Add similarity score to results
            similarity_dict = {msg_id: score for msg_id, score in similarities}
            results["similarity"] = results["message_id"].map(similarity_dict)

            print(f"\nFound {len(results)} semantic matches:")

            # Display results with similarity score
            for i, row in enumerate(results.itertuples(), 1):
                content = row.content

                # Create a snippet
                if len(content) > 150:
                    snippet = content[:150] + "..."
                else:
                    snippet = content

                # Truncate title if it's too long
                title = row.conversation_title if row.conversation_title else "Untitled"
                if len(title) > 40:
                    title = title[:37] + "..."

                # Print result with metadata and similarity
                print(f"\n{i}. {title} - {row.platform} ({row.created_at})")
                print(f"   Similarity: {row.similarity:.2f}")
                print(f"   Sender: {row.sender}")
                print(f"   {snippet}")

            # Ask if user wants more details or visualization
            self._ask_for_result_actions(results, query, semantic=True)

        except Exception as e:
            print(f"Error during semantic search: {e}")

    def date_range_search(self):
        """Search conversations by date range"""
        print("\n==== Date Range Search ====")
        print(
            f"Database contains conversations from {self.date_range[0]} to {self.date_range[1]}"
        )

        # Get date range
        start_date = (
            input(f"Start date (YYYY-MM-DD) [{self.date_range[0]}]: ").strip()
            or self.date_range[0]
        )
        end_date = (
            input(f"End date (YYYY-MM-DD) [{self.date_range[1]}]: ").strip()
            or self.date_range[1]
        )

        # Validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")
            return

        # Get additional filters
        platform_filter = (
            input(f"Filter by platform ({'/'.join(self.platforms)}/all) [all]: ")
            .strip()
            .lower()
            or "all"
        )
        keywords = input("Optional keywords to search for: ").strip()

        # Build SQL query
        sql_query = """
            SELECT 
                c.id AS conversation_id,
                c.title AS conversation_title,
                c.platform,
                c.created_at,
                c.updated_at,
                COUNT(m.id) AS message_count,
                SUM(CASE WHEN m.sender = 'human' THEN 1 ELSE 0 END) AS human_messages,
                SUM(CASE WHEN m.sender = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                MAX(m.model) AS last_model
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE DATE(c.created_at) BETWEEN ? AND ?
        """

        params = [start_date, end_date]

        # Apply platform filter
        if platform_filter != "all":
            sql_query += " AND c.platform = ?"
            params.append(platform_filter)

        # Apply keyword filter if provided
        if keywords:
            sql_query += " AND EXISTS (SELECT 1 FROM messages m2 WHERE m2.conversation_id = c.id AND m2.content LIKE ?)"
            params.append(f"%{keywords}%")

        # Group by conversation
        sql_query += """
            GROUP BY c.id, c.title, c.platform, c.created_at, c.updated_at
            ORDER BY c.created_at DESC
        """

        # Execute query
        start_time = time.time()
        results = self.query_to_dataframe(sql_query, params)
        search_time = time.time() - start_time

        # Display results
        if results.empty:
            print(
                f"No conversations found in the date range {start_date} to {end_date}"
            )
            return

        print(f"\nFound {len(results)} conversations in {search_time:.2f} seconds:")

        # Display conversations
        for i, row in enumerate(results.itertuples(), 1):
            # Truncate title if it's too long
            title = row.conversation_title if row.conversation_title else "Untitled"
            if len(title) > 50:
                title = title[:47] + "..."

            # Format date
            date = datetime.strptime(row.created_at, "%Y-%m-%d %H:%M:%S").strftime(
                "%b %d, %Y %H:%M"
            )

            # Print result with metadata
            print(f"\n{i}. {title}")
            print(f"   Platform: {row.platform} | Date: {date}")
            print(
                f"   Messages: {row.message_count} ({row.human_messages} human, {row.assistant_messages} assistant)"
            )
            if row.last_model:
                print(f"   Model: {row.last_model}")

        # Ask if user wants more details or visualization
        self._ask_for_conversation_actions(results)

    def platform_model_search(self):
        """Search conversations by platform and model"""
        print("\n==== Platform/Model Search ====")

        # Display available platforms and models
        print(f"Available platforms: {', '.join(self.platforms)}")
        print(f"Available models: {', '.join(self.models[:10])}...")

        # Get platform filter
        platform = (
            input(f"Select platform ({'/'.join(self.platforms)}/all) [all]: ")
            .strip()
            .lower()
            or "all"
        )

        if platform != "all" and platform not in self.platforms:
            print(f"Invalid platform. Available platforms: {', '.join(self.platforms)}")
            return

        # Get model filter with autocomplete
        print("Start typing a model name and press Enter to see matching models")
        model_prefix = input("Model prefix (or 'all'): ").strip()

        model = "all"
        if model_prefix.lower() != "all":
            # Find matching models
            matching_models = [
                m for m in self.models if model_prefix.lower() in m.lower()
            ]

            if not matching_models:
                print("No matching models found.")
                return

            print("Matching models:")
            for i, m in enumerate(matching_models[:10], 1):
                print(f"{i}. {m}")

            if len(matching_models) > 10:
                print(f"...and {len(matching_models) - 10} more")

            # Let user select a model
            model_idx = input(
                f"Select model (1-{min(10, len(matching_models))}) or press Enter for all: "
            ).strip()

            if model_idx:
                try:
                    idx = int(model_idx) - 1
                    if 0 <= idx < len(matching_models):
                        model = matching_models[idx]
                    else:
                        print("Invalid selection. Using all matching models.")
                except ValueError:
                    print("Invalid input. Using all matching models.")
            else:
                # Use all matching models
                model = "matching"

        # Additional filters
        date_filter = input("Filter by date range? (y/n) [n]: ").strip().lower() == "y"
        start_date = self.date_range[0]
        end_date = self.date_range[1]

        if date_filter:
            start_date = (
                input(f"Start date (YYYY-MM-DD) [{self.date_range[0]}]: ").strip()
                or self.date_range[0]
            )
            end_date = (
                input(f"End date (YYYY-MM-DD) [{self.date_range[1]}]: ").strip()
                or self.date_range[1]
            )

            # Validate dates
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                print("Invalid date format. Using full date range.")
                start_date = self.date_range[0]
                end_date = self.date_range[1]

        # Build SQL query
        sql_query = """
            SELECT 
                c.id AS conversation_id,
                c.title AS conversation_title,
                c.platform,
                c.created_at,
                COUNT(m.id) AS message_count,
                SUM(CASE WHEN m.sender = 'human' THEN 1 ELSE 0 END) AS human_messages,
                SUM(CASE WHEN m.sender = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
                GROUP_CONCAT(DISTINCT m.model) AS models_used
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE DATE(c.created_at) BETWEEN ? AND ?
        """

        params = [start_date, end_date]

        # Apply platform filter
        if platform != "all":
            sql_query += " AND c.platform = ?"
            params.append(platform)

        # Apply model filter
        if model != "all":
            if model == "matching":
                # Use all matching models
                model_conditions = []
                for m in matching_models:
                    model_conditions.append("m.model = ?")
                    params.append(m)

                if model_conditions:
                    sql_query += f" AND ({' OR '.join(model_conditions)})"
            else:
                # Use specific model
                sql_query += " AND m.model = ?"
                params.append(model)

        # Group by conversation
        sql_query += """
            GROUP BY c.id, c.title, c.platform, c.created_at
            ORDER BY c.created_at DESC
        """

        # Execute query
        start_time = time.time()
        results = self.query_to_dataframe(sql_query, params)
        search_time = time.time() - start_time

        # Display results
        if results.empty:
            print("No conversations found matching the criteria")
            return

        print(f"\nFound {len(results)} conversations in {search_time:.2f} seconds:")

        # Display conversations
        for i, row in enumerate(results.itertuples(), 1):
            # Truncate title if it's too long
            title = row.conversation_title if row.conversation_title else "Untitled"
            if len(title) > 50:
                title = title[:47] + "..."

            # Format date
            date = datetime.strptime(row.created_at, "%Y-%m-%d %H:%M:%S").strftime(
                "%b %d, %Y %H:%M"
            )

            # Print result with metadata
            print(f"\n{i}. {title}")
            print(f"   Platform: {row.platform} | Date: {date}")
            print(
                f"   Messages: {row.message_count} ({row.human_messages} human, {row.assistant_messages} assistant)"
            )
            print(f"   Models: {row.models_used}")

        # Ask if user wants more details or visualization
        self._ask_for_conversation_actions(results)

    def topic_search(self):
        """Search for conversations by topic using TF-IDF keywords"""
        print("\n==== Topic-based Search ====")

        # Analyze frequently used terms in the database
        topic_terms = self._extract_topic_terms()

        # Display common topics
        print("Common topics in the database:")
        for i, (topic, keywords) in enumerate(topic_terms.items(), 1):
            print(f"{i}. {topic}: {', '.join(keywords[:5])}")

        # Let user select a topic or enter custom keywords
        choice = (
            input("\nSelect a topic number or type 'custom' for custom keywords: ")
            .strip()
            .lower()
        )

        search_terms = []
        if choice == "custom":
            # Get custom keywords
            keywords = input("Enter keywords (comma separated): ").strip()
            search_terms = [k.strip() for k in keywords.split(",") if k.strip()]
        else:
            try:
                topic_idx = int(choice) - 1
                if 0 <= topic_idx < len(topic_terms):
                    topic = list(topic_terms.keys())[topic_idx]
                    search_terms = topic_terms[topic]
                    print(
                        f"Using keywords for topic '{topic}': {', '.join(search_terms)}"
                    )
                else:
                    print("Invalid topic number. Please enter a valid number.")
                    return
            except ValueError:
                print("Invalid input. Please enter a number or 'custom'.")
                return

        if not search_terms:
            print("No keywords specified.")
            return

        # Build SQL query - search for conversations with these keywords
        conditions = []
        params = []

        for term in search_terms:
            conditions.append("m.content LIKE ?")
            params.append(f"%{term}%")

        sql_query = f"""
            SELECT 
                c.id AS conversation_id,
                c.title AS conversation_title,
                c.platform,
                c.created_at,
                COUNT(DISTINCT m.id) AS message_count,
                COUNT(DISTINCT CASE WHEN m.content LIKE ? THEN m.id ELSE NULL END) AS relevant_messages,
                SUM(CASE WHEN m.sender = 'human' THEN 1 ELSE 0 END) AS human_messages,
                SUM(CASE WHEN m.sender = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            WHERE EXISTS (
                SELECT 1 FROM messages m2 
                WHERE m2.conversation_id = c.id
                AND ({' OR '.join(conditions)})
            )
            GROUP BY c.id, c.title, c.platform, c.created_at
            ORDER BY relevant_messages DESC
        """

        # Add the general topic term as first parameter
        topic_param = "%" + "%".join(search_terms) + "%"
        params = [topic_param] + params

        # Execute query
        start_time = time.time()
        results = self.query_to_dataframe(sql_query, params)
        search_time = time.time() - start_time

        # Display results
        if results.empty:
            print(
                f"No conversations found for the topic with keywords: {', '.join(search_terms)}"
            )
            return

        print(f"\nFound {len(results)} conversations in {search_time:.2f} seconds:")

        # Display conversations
        for i, row in enumerate(results.itertuples(), 1):
            # Truncate title if it's too long
            title = row.conversation_title if row.conversation_title else "Untitled"
            if len(title) > 50:
                title = title[:47] + "..."

            # Format date
            date = datetime.strptime(row.created_at, "%Y-%m-%d %H:%M:%S").strftime(
                "%b %d, %Y %H:%M"
            )

            # Calculate relevance percentage
            relevance = (row.relevant_messages / row.message_count) * 100

            # Print result with metadata
            print(f"\n{i}. {title}")
            print(f"   Platform: {row.platform} | Date: {date}")
            print(
                f"   Relevance: {relevance:.1f}% ({row.relevant_messages} of {row.message_count} messages)"
            )
            print(
                f"   Messages: {row.message_count} ({row.human_messages} human, {row.assistant_messages} assistant)"
            )

        # Ask if user wants more details or visualization
        self._ask_for_conversation_actions(results)

    def context_window_search(self):
        """Search for messages and display surrounding context"""
        print("\n==== Context Window Search ====")

        # Get search query
        query = input("Enter search keywords: ").strip()
        if not query:
            print("Search query cannot be empty.")
            return

        # Get context window size
        try:
            context_size = int(
                input("Number of messages before/after to include [2]: ") or "2"
            )
            context_size = max(0, min(10, context_size))
        except ValueError:
            context_size = 2

        # Build SQL query to find matching messages
        sql_query = """
            SELECT 
                m.id AS message_id,
                m.conversation_id,
                c.title AS conversation_title,
                c.platform,
                m.sender,
                m.content,
                m.created_at,
                m.order_index
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.content LIKE ?
            ORDER BY m.created_at DESC
            LIMIT 50
        """

        # Execute query
        start_time = time.time()
        results = self.query_to_dataframe(sql_query, [f"%{query}%"])

        if results.empty:
            print(f"No results found for '{query}'")
            return

        # Get context for each matching message
        context_results = []

        for _, row in results.iterrows():
            # Query to get surrounding messages
            context_query = """
                SELECT 
                    m.id AS message_id,
                    m.conversation_id,
                    m.sender,
                    m.content,
                    m.created_at,
                    m.order_index,
                    m.model,
                    CASE WHEN m.id = ? THEN 1 ELSE 0 END AS is_match
                FROM messages m
                WHERE m.conversation_id = ?
                AND m.order_index BETWEEN ? AND ?
                ORDER BY m.order_index
            """

            min_order = max(0, row["order_index"] - context_size)
            max_order = row["order_index"] + context_size

            context_messages = self.query_to_dataframe(
                context_query,
                [row["message_id"], row["conversation_id"], min_order, max_order],
            )

            # Add conversation title and platform
            context_messages["conversation_title"] = row["conversation_title"]
            context_messages["platform"] = row["platform"]

            context_results.append(context_messages)

        search_time = time.time() - start_time

        print(
            f"\nFound {len(results)} matches with context in {search_time:.2f} seconds:"
        )

        # Display results with context
        for i, context_df in enumerate(context_results, 1):
            # Get conversation info from the first row
            conv_title = context_df["conversation_title"].iloc[0]
            platform = context_df["platform"].iloc[0]

            # Truncate title if it's too long
            title = conv_title if conv_title else "Untitled"
            if len(title) > 50:
                title = title[:47] + "..."

            print(f"\n{i}. {title} - {platform}")

            # Print context messages
            for j, msg in enumerate(context_df.itertuples(), 1):
                # Format message with indentation
                sender_prefix = "H: " if msg.sender == "human" else "A: "

                # Truncate content if it's too long
                content = msg.content
                if len(content) > 100:
                    content = content[:97] + "..."

                # Highlight the query term in the matching message
                if msg.is_match:
                    pattern = re.compile(re.escape(query), re.IGNORECASE)
                    content = pattern.sub(f"\033[1;31m{query}\033[0m", content)
                    line_prefix = "→ "
                else:
                    line_prefix = "  "

                # Print the message
                print(f"{line_prefix}{sender_prefix}{content}")

        # Ask if user wants to view full conversation
        self._ask_for_context_actions(context_results)

    def export_conversations(self):
        """Export conversations to various formats"""
        print("\n==== Conversation Export ====")

        # Ask for export method
        print("Export methods:")
        print("1. Export conversation by ID")
        print("2. Export from previous search results")
        print("3. Export conversations by date range")

        choice = input("Select method (1-3): ").strip()

        conversation_ids = []

        if choice == "1":
            # Export by ID
            conv_id = input("Enter conversation ID: ").strip()
            conversation_ids = [conv_id]

        elif choice == "2":
            # Check if there are search results
            if (
                not hasattr(self, "last_search_results")
                or self.last_search_results.empty
            ):
                print("No search results available. Please perform a search first.")
                return

            # Get conversation IDs from last search
            if "conversation_id" in self.last_search_results.columns:
                conversation_ids = (
                    self.last_search_results["conversation_id"].unique().tolist()
                )

                # Limit the number of conversations to export
                if len(conversation_ids) > A:
                    print(
                        f"Found {len(conversation_ids)} conversations. This is a lot of data."
                    )
                    limit = (
                        input(
                            f"How many conversations to export (1-{len(conversation_ids)})? [10]: "
                        ).strip()
                        or "10"
                    )
                    try:
                        limit = int(limit)
                        conversation_ids = conversation_ids[:limit]
                    except ValueError:
                        conversation_ids = conversation_ids[:10]
            else:
                print("Last search results don't contain conversation IDs.")
                return

        elif choice == "3":
            # Export by date range
            start_date = (
                input(f"Start date (YYYY-MM-DD) [{self.date_range[0]}]: ").strip()
                or self.date_range[0]
            )
            end_date = (
                input(f"End date (YYYY-MM-DD) [{self.date_range[1]}]: ").strip()
                or self.date_range[1]
            )

            # Validate dates
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                print("Invalid date format. Please use YYYY-MM-DD.")
                return

            # Get conversation IDs in date range
            query = """
                SELECT id FROM conversations 
                WHERE DATE(created_at) BETWEEN ? AND ?
                ORDER BY created_at DESC
                LIMIT 100
            """

            df = self.query_to_dataframe(query, [start_date, end_date])

            if df.empty:
                print(
                    f"No conversations found in the date range {start_date} to {end_date}"
                )
                return

            conversation_ids = df["id"].tolist()

            # Limit the number of conversations to export
            if len(conversation_ids) > 10:
                print(
                    f"Found {len(conversation_ids)} conversations. This is a lot of data."
                )
                limit = (
                    input(
                        f"How many conversations to export (1-{len(conversation_ids)})? [10]: "
                    ).strip()
                    or "10"
                )
                try:
                    limit = int(limit)
                    conversation_ids = conversation_ids[:limit]
                except ValueError:
                    conversation_ids = conversation_ids[:10]

        else:
            print("Invalid choice.")
            return

        if not conversation_ids:
            print("No conversations to export.")
            return

        # Choose export format
        print("\nExport formats:")
        print("1. JSON")
        print("2. Markdown")
        print("3. HTML")
        print("4. CSV (messages only)")

        format_choice = input("Select format (1-4): ").strip()

        if format_choice not in ["1", "2", "3", "4"]:
            print("Invalid format choice.")
            return

        # Export the conversations
        self._export_conversations(conversation_ids, format_choice)

    def generate_embeddings(self):
        """Generate or update embeddings for semantic search"""
        print("\n==== Generate/Update Embeddings ====")

        # Check if embeddings already exist
        embedding_file = os.path.join(EMBEDDINGS_DIR, "embeddings.npz")
        content_file = os.path.join(EMBEDDINGS_DIR, "message_content.json")

        if os.path.exists(embedding_file) and os.path.exists(content_file):
            update = (
                input("Embeddings already exist. Update them? (y/n) [n]: ")
                .strip()
                .lower()
                == "y"
            )
            if not update:
                print("Using existing embeddings.")
                # Load embeddings if they're not already loaded
                if not self.embedding_loaded:
                    self._load_embeddings()
                return

        # Ask for limit
        try:
            limit = int(
                input("Maximum number of messages to embed [10000]: ") or "10000"
            )
        except ValueError:
            limit = 10000

        # Get message content for embedding
        print(f"Fetching up to {limit} messages for embedding...")

        query = f"""
            SELECT 
                m.id AS message_id,
                m.content
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE LENGTH(m.content) > 10  -- Skip very short messages
            ORDER BY RANDOM()  -- Randomize to get a diverse sample
            LIMIT {limit}
        """

        messages_df = self.query_to_dataframe(query)

        if messages_df.empty:
            print("No messages found to embed.")
            return

        print(f"Generating embeddings for {len(messages_df)} messages...")

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)

        try:
            # Fit and transform the messages
            message_contents = messages_df["content"].tolist()
            embeddings_matrix = vectorizer.fit_transform(message_contents)

            # Convert sparse matrix to dictionary of message_id: embedding
            embeddings = {}
            message_content = {}

            for i, row in enumerate(messages_df.itertuples()):
                embeddings[row.message_id] = embeddings_matrix[i].toarray()[0]
                message_content[row.message_id] = row.content

            # Save embeddings and content reference
            os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

            # Save embeddings using numpy's compressed format
            np.savez_compressed(embedding_file, **embeddings)

            # Save message content as JSON
            with open(content_file, "w") as f:
                json.dump(message_content, f)

            print(
                f"Saved embeddings for {len(embeddings)} messages to {embedding_file}"
            )

            # Update the loaded embeddings
            self.embeddings = embeddings
            self.message_content_cache = message_content
            self.embedding_loaded = True

        except Exception as e:
            print(f"Error generating embeddings: {e}")

    def analyze_search_results(self):
        """Analyze and visualize previous search results"""
        print("\n==== Analyze Search Results ====")

        # Check if there are search results
        if not hasattr(self, "last_search_results") or self.last_search_results.empty:
            print("No search results available. Please perform a search first.")
            return

        print(f"Analyzing {len(self.last_search_results)} search results...")

        # Create visualizations for search results
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        # 1. Results by platform
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_results_by_platform(ax1)

        # 2. Results over time
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_results_over_time(ax2)

        # 3. Word cloud or common terms (if content is available)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_result_keywords(ax3)

        # 4. Sender distribution
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_sender_distribution(ax4)

        plt.tight_layout()

        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SEARCH_RESULTS_DIR, f"search_analysis_{timestamp}.png")
        plt.savefig(filename)

        print(f"Analysis visualization saved to {filename}")
        plt.show()

    def _plot_results_by_platform(self, ax):
        """Plot search results distribution by platform"""
        if "platform" in self.last_search_results.columns:
            platform_counts = self.last_search_results["platform"].value_counts()

            bars = ax.bar(
                platform_counts.index,
                platform_counts.values,
                color=[PLATFORM_COLORS.get(p, "gray") for p in platform_counts.index],
            )

            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    str(int(height)),
                    ha="center",
                    va="bottom",
                )

            ax.set_title("Results by Platform")
            ax.set_xlabel("Platform")
            ax.set_ylabel("Number of Results")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(0.5, 0.5, "Platform data not available", ha="center", va="center")

    def _plot_results_over_time(self, ax):
        """Plot search results distribution over time"""
        if "created_at" in self.last_search_results.columns:
            # Convert to datetime
            self.last_search_results["date"] = pd.to_datetime(
                self.last_search_results["created_at"]
            )

            # Group by date and platform if available
            if "platform" in self.last_search_results.columns:
                # Group by date and platform
                date_platform = (
                    self.last_search_results.groupby(
                        [pd.Grouper(key="date", freq="D"), "platform"]
                    )
                    .size()
                    .unstack()
                    .fillna(0)
                )

                # Plot
                date_platform.plot(
                    kind="line",
                    marker="o",
                    ax=ax,
                    color=[
                        PLATFORM_COLORS.get(p, "gray") for p in date_platform.columns
                    ],
                )
            else:
                # Group by date only
                date_counts = self.last_search_results.groupby(
                    pd.Grouper(key="date", freq="D")
                ).size()

                # Plot
                date_counts.plot(kind="line", marker="o", ax=ax)

            ax.set_title("Results Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Results")
            ax.grid(True, alpha=0.3)

            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        else:
            ax.text(0.5, 0.5, "Timestamp data not available", ha="center", va="center")

    def _plot_result_keywords(self, ax):
        """Plot common keywords in search results"""
        if "content" in self.last_search_results.columns:
            # Extract common words
            all_text = " ".join(self.last_search_results["content"].fillna(""))

            # Tokenize and count
            words = re.findall(r"\b[a-zA-Z]{4,}\b", all_text.lower())

            # Remove common stopwords
            stopwords = {
                "that",
                "this",
                "with",
                "from",
                "have",
                "which",
                "would",
                "about",
                "there",
                "their",
                "these",
                "than",
                "also",
                "they",
                "your",
                "should",
            }
            words = [w for w in words if w not in stopwords]

            # Count word frequency
            word_counts = Counter(words)

            # Plot top words
            top_words = dict(word_counts.most_common(15))

            # Sort by frequency
            top_words = {
                k: v for k, v in sorted(top_words.items(), key=lambda item: item[1])
            }

            # Create horizontal bar chart
            ax.barh(
                list(top_words.keys()),
                list(top_words.values()),
                color=plt.cm.viridis(np.linspace(0, 1, len(top_words))),
            )

            ax.set_title("Common Terms in Search Results")
            ax.set_xlabel("Frequency")
            ax.invert_yaxis()  # Highest frequency at the top
            ax.grid(True, alpha=0.3, axis="x")
        else:
            ax.text(0.5, 0.5, "Content data not available", ha="center", va="center")

    def _plot_sender_distribution(self, ax):
        """Plot distribution of results by sender"""
        if "sender" in self.last_search_results.columns:
            # Get counts by sender
            sender_counts = self.last_search_results["sender"].value_counts()

            # Create pie chart
            ax.pie(
                sender_counts.values,
                labels=sender_counts.index,
                autopct="%1.1f%%",
                colors=["#3498DB", "#2ECC71"],
            )

            ax.set_title("Results by Sender")
        else:
            ax.text(0.5, 0.5, "Sender data not available", ha="center", va="center")

    def _ask_for_result_actions(self, results, query, semantic=False):
        """Ask the user what to do with search results"""
        # Save the results for later use
        self.last_search_results = results

        print("\nOptions:")
        print("1. View full message")
        print("2. View entire conversation")
        print("3. Visualize results")
        print("4. Export results")
        print("5. Return to search menu")

        choice = input("Select option (1-5): ").strip()

        if choice == "1":
            # View full message
            msg_num = input("Enter message number: ").strip()
            try:
                idx = int(msg_num) - 1
                if 0 <= idx < len(results):
                    row = results.iloc[idx]
                    print("\n" + "=" * 80)
                    print(f"Message from: {row['sender']} at {row['created_at']}")
                    print(f"Conversation: {row['conversation_title']}")
                    if semantic and "similarity" in row:
                        print(f"Similarity score: {row['similarity']:.2f}")
                    print("-" * 80)
                    print(row["content"])
                    print("=" * 80)
                else:
                    print("Invalid message number")
            except ValueError:
                print("Invalid input")

            # Re-display the options
            self._ask_for_result_actions(results, query, semantic)

        elif choice == "2":
            # View entire conversation
            msg_num = input("Enter message number: ").strip()
            try:
                idx = int(msg_num) - 1
                if 0 <= idx < len(results):
                    row = results.iloc[idx]
                    self._display_conversation(
                        row["conversation_id"], row["message_id"]
                    )
                else:
                    print("Invalid message number")
            except ValueError:
                print("Invalid input")

            # Re-display the options
            self._ask_for_result_actions(results, query, semantic)

        elif choice == "3":
            # Visualize results
            self.analyze_search_results()

            # Re-display the options
            self._ask_for_result_actions(results, query, semantic)

        elif choice == "4":
            # Export results
            self._export_search_results(results, query)

            # Re-display the options
            self._ask_for_result_actions(results, query, semantic)

        elif choice == "5":
            # Return to main menu
            return

        else:
            print("Invalid choice")
            self._ask_for_result_actions(results, query, semantic)

    def _ask_for_conversation_actions(self, results):
        """Ask the user what to do with conversation search results"""
        # Save the results for later use
        self.last_search_results = results

        print("\nOptions:")
        print("1. View conversation")
        print("2. Visualize results")
        print("3. Export conversations")
        print("4. Return to search menu")

        choice = input("Select option (1-4): ").strip()

        if choice == "1":
            # View conversation
            conv_num = input("Enter conversation number: ").strip()
            try:
                idx = int(conv_num) - 1
                if 0 <= idx < len(results):
                    row = results.iloc[idx]
                    self._display_conversation(row["conversation_id"])
                else:
                    print("Invalid conversation number")
            except ValueError:
                print("Invalid input")

            # Re-display the options
            self._ask_for_conversation_actions(results)

        elif choice == "2":
            # Visualize results
            self.analyze_search_results()

            # Re-display the options
            self._ask_for_conversation_actions(results)

        elif choice == "3":
            # Export conversations
            conversation_ids = results["conversation_id"].tolist()
            self._export_selected_conversations(conversation_ids)

            # Re-display the options
            self._ask_for_conversation_actions(results)

        elif choice == "4":
            # Return to main menu
            return

        else:
            print("Invalid choice")
            self._ask_for_conversation_actions(results)

    def _ask_for_context_actions(self, context_results):
        """Ask the user what to do with context search results"""
        print("\nOptions:")
        print("1. View full conversation")
        print("2. Export context results")
        print("3. Return to search menu")

        choice = input("Select option (1-3): ").strip()

        if choice == "1":
            # View full conversation
            result_num = input("Enter result number: ").strip()
            try:
                idx = int(result_num) - 1
                if 0 <= idx < len(context_results):
                    context_df = context_results[idx]
                    # Get the conversation ID and matching message ID
                    conversation_id = context_df["conversation_id"].iloc[0]
                    match_message_id = context_df.loc[
                        context_df["is_match"] == 1, "message_id"
                    ].iloc[0]

                    self._display_conversation(conversation_id, match_message_id)
                else:
                    print("Invalid result number")
            except ValueError:
                print("Invalid input")

            # Re-display the options
            self._ask_for_context_actions(context_results)

        elif choice == "2":
            # Export context results
            self._export_context_results(context_results)

            # Re-display the options
            self._ask_for_context_actions(context_results)

        elif choice == "3":
            # Return to main menu
            return

        else:
            print("Invalid choice")
            self._ask_for_context_actions(context_results)

    def _display_conversation(self, conversation_id, highlight_message_id=None):
        """Display a conversation with optional message highlighting"""
        # Query to get conversation details
        conv_query = """
            SELECT 
                c.id,
                c.title,
                c.platform,
                c.created_at
            FROM conversations c
            WHERE c.id = ?
        """

        conv_df = self.query_to_dataframe(conv_query, [conversation_id])

        if conv_df.empty:
            print(f"Conversation not found: {conversation_id}")
            return

        # Query to get messages
        msg_query = """
            SELECT 
                m.id,
                m.sender,
                m.content,
                m.created_at,
                m.model,
                m.order_index
            FROM messages m
            WHERE m.conversation_id = ?
            ORDER BY m.order_index
        """

        msg_df = self.query_to_dataframe(msg_query, [conversation_id])

        if msg_df.empty:
            print(f"No messages found for conversation: {conversation_id}")
            return

        # Display conversation header
        conv = conv_df.iloc[0]
        print("\n" + "=" * 80)
        print(f"Conversation: {conv['title'] or 'Untitled'}")
        print(f"Platform: {conv['platform']} | Created: {conv['created_at']}")
        print(f"Total messages: {len(msg_df)}")
        print("=" * 80)

        # Display messages
        for _, msg in msg_df.iterrows():
            # Format timestamp
            timestamp = datetime.strptime(msg["created_at"], "%Y-%m-%d %H:%M:%S")
            time_str = timestamp.strftime("%H:%M:%S")

            # Format sender
            sender = msg["sender"].capitalize()

            # Check if this is the highlighted message
            is_highlight = highlight_message_id and msg["id"] == highlight_message_id

            # Print message header
            if is_highlight:
                print(f"\n→ {sender} ({time_str}):")
            else:
                print(f"\n{sender} ({time_str}):")

            # Print model if available
            if msg["model"]:
                print(f"[Model: {msg['model']}]")

            # Print content with optional highlighting
            content = msg["content"]
            if is_highlight:
                print(f"\033[1;31m{content}\033[0m")
            else:
                print(content)

        print("\n" + "=" * 80)

    def _extract_topic_terms(self):
        """Extract common topics and their related terms from the database"""
        # Predefined topics and seed terms
        topics = {
            "Programming": [
                "code",
                "programming",
                "python",
                "javascript",
                "function",
                "class",
                "api",
            ],
            "AI/ML": [
                "machine learning",
                "ai",
                "neural network",
                "model",
                "training",
                "algorithm",
            ],
            "Education": [
                "learn",
                "teaching",
                "education",
                "student",
                "course",
                "study",
            ],
            "Business": [
                "business",
                "company",
                "product",
                "market",
                "customer",
                "strategy",
            ],
            "Data Science": [
                "data",
                "analysis",
                "statistics",
                "visualization",
                "graph",
                "chart",
            ],
            "Web Development": ["website", "html", "css", "web", "frontend", "backend"],
            "Creative Writing": [
                "writing",
                "story",
                "novel",
                "character",
                "plot",
                "narrative",
            ],
            "Health": [
                "health",
                "medicine",
                "disease",
                "treatment",
                "patient",
                "doctor",
            ],
        }

        # Option to dynamically expand topics based on co-occurrence
        # (simplified implementation for speed)

        return topics

    def _export_conversations(self, conversation_ids, format_choice):
        """Export conversations to the selected format"""
        if not conversation_ids:
            print("No conversations to export.")
            return

        # Create output directory
        os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)

        # Get conversations data
        conversations = []

        for conv_id in conversation_ids:
            # Get conversation details
            conv_query = """
                SELECT 
                    c.id,
                    c.title,
                    c.platform,
                    c.created_at,
                    c.updated_at
                FROM conversations c
                WHERE c.id = ?
            """

            conv_df = self.query_to_dataframe(conv_query, [conv_id])

            if conv_df.empty:
                print(f"Conversation not found: {conv_id}")
                continue

            # Get messages
            msg_query = """
                SELECT 
                    m.id,
                    m.sender,
                    m.content,
                    m.created_at,
                    m.model,
                    m.order_index
                FROM messages m
                WHERE m.conversation_id = ?
                ORDER BY m.order_index
            """

            msg_df = self.query_to_dataframe(msg_query, [conv_id])

            if msg_df.empty:
                print(f"No messages found for conversation: {conv_id}")
                continue

            # Add to conversations list
            conversations.append(
                {
                    "conversation": conv_df.iloc[0].to_dict(),
                    "messages": msg_df.to_dict("records"),
                }
            )

        if not conversations:
            print("No valid conversations found to export.")
            return

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export based on format choice
        if format_choice == "1":  # JSON
            # Export to JSON
            filename = os.path.join(
                SEARCH_RESULTS_DIR, f"conversations_export_{timestamp}.json"
            )

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)

            print(f"Exported {len(conversations)} conversations to {filename}")

        elif format_choice == "2":  # Markdown
            # Export to Markdown
            for i, conv in enumerate(conversations):
                conv_data = conv["conversation"]
                messages = conv["messages"]

                # Create safe filename from title or use ID
                title = conv_data.get("title") or f"conversation_{i + 1}"
                safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")

                filename = os.path.join(
                    SEARCH_RESULTS_DIR, f"{safe_title}_{timestamp}.md"
                )

                with open(filename, "w", encoding="utf-8") as f:
                    # Write header
                    f.write(
                        f"# {conv_data.get('title') or 'Untitled Conversation'}\n\n"
                    )
                    f.write(f"- **Platform:** {conv_data.get('platform')}\n")
                    f.write(f"- **Created:** {conv_data.get('created_at')}\n")
                    f.write(f"- **ID:** {conv_data.get('id')}\n\n")
                    f.write("---\n\n")

                    # Write messages
                    for msg in messages:
                        sender = msg.get("sender", "").capitalize()
                        content = msg.get("content", "")
                        timestamp = msg.get("created_at", "")
                        model = msg.get("model", "")

                        f.write(f"## {sender} ({timestamp})\n\n")

                        if model:
                            f.write(f"*Model: {model}*\n\n")

                        f.write(f"{content}\n\n")
                        f.write("---\n\n")

            print(
                f"Exported {len(conversations)} conversations to Markdown files in {SEARCH_RESULTS_DIR}"
            )

        elif format_choice == "3":  # HTML
            # Export to HTML
            for i, conv in enumerate(conversations):
                conv_data = conv["conversation"]
                messages = conv["messages"]

                # Create safe filename from title or use ID
                title = conv_data.get("title") or f"conversation_{i + 1}"
                safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")

                filename = os.path.join(
                    SEARCH_RESULTS_DIR, f"{safe_title}_{timestamp}.html"
                )

                with open(filename, "w", encoding="utf-8") as f:
                    # Write HTML header
                    f.write(
                        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{conv_data.get('title') or 'Untitled Conversation'}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .header {{ margin-bottom: 20px; padding-bottom: 10px; border-bottom: 1px solid #ccc; }}
        .message {{ margin-bottom: 20px; padding: 10px; border-radius: 5px; }}
        .human {{ background-color: #f0f0f0; }}
        .assistant {{ background-color: #e6f7ff; }}
        .metadata {{ font-size: 0.8em; color: #666; margin-bottom: 5px; }}
        pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{conv_data.get('title') or 'Untitled Conversation'}</h1>
        <p><strong>Platform:</strong> {conv_data.get('platform')}</p>
        <p><strong>Created:</strong> {conv_data.get('created_at')}</p>
        <p><strong>ID:</strong> {conv_data.get('id')}</p>
    </div>
"""
                    )

                    # Write messages
                    for msg in messages:
                        sender = msg.get("sender", "")
                        content = msg.get("content", "")
                        timestamp = msg.get("created_at", "")
                        model = msg.get("model", "")

                        # Convert code blocks for HTML
                        content = re.sub(
                            r"```(\w*)\n(.*?)\n```",
                            r"<pre><code>\2</code></pre>",
                            content,
                            flags=re.DOTALL,
                        )

                        # Handle line breaks
                        content = content.replace("\n", "<br>")

                        f.write(
                            f"""    <div class="message {sender}">
        <div class="metadata">
            <strong>{sender.capitalize()}</strong> ({timestamp})
            {f'<br><em>Model: {model}</em>' if model else ''}
        </div>
        <div class="content">
            {content}
        </div>
    </div>
"""
                        )

                    # Write HTML footer
                    f.write(
                        """</body>
</html>"""
                    )

            print(
                f"Exported {len(conversations)} conversations to HTML files in {SEARCH_RESULTS_DIR}"
            )

        elif format_choice == "4":  # CSV
            # Export to CSV
            filename = os.path.join(
                SEARCH_RESULTS_DIR, f"conversations_export_{timestamp}.csv"
            )

            # Prepare data for CSV
            rows = []

            for conv in conversations:
                conv_data = conv["conversation"]
                conv_id = conv_data.get("id")
                conv_title = conv_data.get("title")
                conv_platform = conv_data.get("platform")

                for msg in conv["messages"]:
                    rows.append(
                        {
                            "conversation_id": conv_id,
                            "conversation_title": conv_title,
                            "platform": conv_platform,
                            "message_id": msg.get("id"),
                            "sender": msg.get("sender"),
                            "content": msg.get("content"),
                            "created_at": msg.get("created_at"),
                            "model": msg.get("model"),
                            "order_index": msg.get("order_index"),
                        }
                    )

            # Convert to DataFrame and export
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)

            print(
                f"Exported {len(rows)} messages from {len(conversations)} conversations to {filename}"
            )

    def _export_search_results(self, results, query):
        """Export search results to various formats"""
        # Create output directory
        os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ask for export format
        print("\nExport formats:")
        print("1. CSV")
        print("2. JSON")

        format_choice = input("Select format (1-2): ").strip()

        if format_choice == "1":  # CSV
            # Export to CSV
            filename = os.path.join(
                SEARCH_RESULTS_DIR, f"search_results_{timestamp}.csv"
            )
            results.to_csv(filename, index=False)
            print(f"Exported search results to {filename}")

        elif format_choice == "2":  # JSON
            # Export to JSON
            filename = os.path.join(
                SEARCH_RESULTS_DIR, f"search_results_{timestamp}.json"
            )

            # Convert to dict for JSON serialization
            results_dict = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "result_count": len(results),
                "results": results.to_dict("records"),
            }

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_dict, f, ensure_ascii=False, indent=2)

            print(f"Exported search results to {filename}")

        else:
            print("Invalid format choice.")

    def _export_selected_conversations(self, conversation_ids):
        """Export selected conversations"""
        if not conversation_ids:
            print("No conversations to export.")
            return

        # Ask for export format
        print("\nExport formats:")
        print("1. JSON")
        print("2. Markdown")
        print("3. HTML")
        print("4. CSV (messages only)")

        format_choice = input("Select format (1-4): ").strip()

        if format_choice not in ["1", "2", "3", "4"]:
            print("Invalid format choice.")
            return

        # Export the conversations
        self._export_conversations(conversation_ids, format_choice)

    def _export_context_results(self, context_results):
        """Export context search results to various formats"""
        # Create output directory
        os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)

        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Ask for export format
        print("\nExport formats:")
        print("1. CSV")
        print("2. JSON")
        print("3. Markdown")

        format_choice = input("Select format (1-3): ").strip()

        if format_choice == "1":  # CSV
            # Export to CSV
            filename = os.path.join(
                SEARCH_RESULTS_DIR, f"context_results_{timestamp}.csv"
            )

            # Prepare data for CSV
            rows = []

            for i, context_df in enumerate(context_results):
                for _, row in context_df.iterrows():
                    rows.append(
                        {
                            "result_group": i + 1,
                            "conversation_id": row["conversation_id"],
                            "conversation_title": row.get("conversation_title", ""),
                            "platform": row.get("platform", ""),
                            "message_id": row["message_id"],
                            "sender": row["sender"],
                            "content": row["content"],
                            "created_at": row["created_at"],
                            "order_index": row["order_index"],
                            "is_match": row["is_match"],
                        }
                    )

            # Convert to DataFrame and export
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)

            print(f"Exported context results to {filename}")

        elif format_choice == "2":  # JSON
            # Export to JSON
            filename = os.path.join(
                SEARCH_RESULTS_DIR, f"context_results_{timestamp}.json"
            )

            # Convert to dict for JSON serialization
            results_list = []

            for i, context_df in enumerate(context_results):
                # Get conversation details
                conv_id = context_df["conversation_id"].iloc[0]
                conv_title = context_df.get(
                    "conversation_title", ["Unknown Title"]
                ).iloc[0]
                platform = context_df.get("platform", ["Unknown"]).iloc[0]

                results_list.append(
                    {
                        "result_group": i + 1,
                        "conversation_id": conv_id,
                        "conversation_title": conv_title,
                        "platform": platform,
                        "messages": context_df.to_dict("records"),
                    }
                )

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results_list, f, ensure_ascii=False, indent=2)

            print(f"Exported context results to {filename}")

        elif format_choice == "3":  # Markdown
            # Export to Markdown
            filename = os.path.join(
                SEARCH_RESULTS_DIR, f"context_results_{timestamp}.md"
            )

            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Context Search Results\n\n")
                f.write(
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )

                for i, context_df in enumerate(context_results):
                    # Get conversation details
                    conv_id = context_df["conversation_id"].iloc[0]
                    conv_title = context_df.get(
                        "conversation_title", ["Unknown Title"]
                    ).iloc[0]
                    platform = context_df.get("platform", ["Unknown"]).iloc[0]

                    f.write(f"## Result Group {i + 1}\n\n")
                    f.write(f"- **Conversation:** {conv_title}\n")
                    f.write(f"- **Platform:** {platform}\n")
                    f.write(f"- **ID:** {conv_id}\n\n")

                    f.write("### Messages\n\n")

                    for _, msg in context_df.iterrows():
                        sender = msg["sender"].capitalize()
                        is_match = msg["is_match"] == 1

                        if is_match:
                            f.write(f"* **{sender}:** {msg['content']} *(MATCH)*\n\n")
                        else:
                            f.write(f"* {sender}: {msg['content']}\n\n")

                    f.write("---\n\n")

            print(f"Exported context results to {filename}")

        else:
            print("Invalid format choice.")

    def _check_embeddings_exist(self):
        """Check if embeddings exist"""
        embedding_file = os.path.join(EMBEDDINGS_DIR, "embeddings.npz")
        content_file = os.path.join(EMBEDDINGS_DIR, "message_content.json")

        return os.path.exists(embedding_file) and os.path.exists(content_file)

    def _load_embeddings(self):
        """Load embeddings from file"""
        embedding_file = os.path.join(EMBEDDINGS_DIR, "embeddings.npz")
        content_file = os.path.join(EMBEDDINGS_DIR, "message_content.json")

        try:
            # Load embeddings
            data = np.load(embedding_file)
            self.embeddings = {k: data[k] for k in data.files}

            # Load message content cache
            with open(content_file, "r") as f:
                self.message_content_cache = json.load(f)

            self.embedding_loaded = True
            print(f"Loaded embeddings for {len(self.embeddings)} messages")

            return True
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            return False

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)


def main():
    """Main function to run the search engine"""
    search_engine = ConversationSearchEngine()
    search_engine.run_search_interface()


if __name__ == "__main__":
    main()
