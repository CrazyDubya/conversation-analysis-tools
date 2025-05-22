#!/usr/bin/env python3
"""
Conversation Search GUI

A graphical user interface for the Conversation Search Engine.
This GUI provides easy access to the search capabilities for exploring
conversation data from Claude and ChatGPT.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import re
import os
import json
from datetime import datetime, timedelta
import time
from threading import Thread
from collections import Counter, defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
    """Basic search engine functionality for the GUI"""

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

        return topics


class ConversationSearchGUI:
    """GUI wrapper for the Conversation Search Engine"""

    def __init__(self, root):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Conversation Search")

        # Set window size and make it resizable
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Initialize the search engine
        self.search_engine = ConversationSearchEngine(DB_PATH)

        # Cache some frequently used data
        self.platforms = self.search_engine.platforms
        self.models = self.search_engine.models
        self.date_range = self.search_engine.date_range

        # Set up the main frame
        self.setup_ui()

        # Track current search results
        self.current_results = None
        self.current_conversation = None

        # Status bar variables
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")

        # Progress bar variable
        self.progress_var = tk.DoubleVar()
        self.progress_var.set(0)

        # Print initial connection info
        self._update_status(
            f"Connected to database with {self.search_engine._count_conversations()} conversations"
        )

    def setup_ui(self):
        """Set up the main user interface"""
        # Create the main frame that will contain everything
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs for different search types
        self.keyword_tab = ttk.Frame(self.notebook)
        self.boolean_tab = ttk.Frame(self.notebook)
        self.semantic_tab = ttk.Frame(self.notebook)
        self.date_tab = ttk.Frame(self.notebook)
        self.platform_model_tab = ttk.Frame(self.notebook)
        self.topic_tab = ttk.Frame(self.notebook)
        self.context_tab = ttk.Frame(self.notebook)
        self.export_tab = ttk.Frame(self.notebook)
        self.settings_tab = ttk.Frame(self.notebook)

        # Add tabs to notebook
        self.notebook.add(self.keyword_tab, text="Keyword Search")
        self.notebook.add(self.boolean_tab, text="Boolean Search")
        self.notebook.add(self.semantic_tab, text="Semantic Search")
        self.notebook.add(self.date_tab, text="Date Search")
        self.notebook.add(self.platform_model_tab, text="Platform/Model")
        self.notebook.add(self.topic_tab, text="Topic Search")
        self.notebook.add(self.context_tab, text="Context Search")
        self.notebook.add(self.export_tab, text="Export")
        self.notebook.add(self.settings_tab, text="Settings")

        # Set up each tab
        self.setup_keyword_tab()
        self.setup_boolean_tab()
        self.setup_semantic_tab()
        self.setup_date_tab()
        self.setup_platform_model_tab()
        self.setup_topic_tab()
        self.setup_context_tab()
        self.setup_export_tab()
        self.setup_settings_tab()

        # Create status bar at the bottom
        self.status_bar = ttk.Frame(self.main_frame)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=(5, 0))

        # Left side: Status text
        self.status_label = ttk.Label(
            self.status_bar, textvariable=self.status_var, anchor=tk.W
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Right side: Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_bar,
            variable=self.progress_var,
            orient=tk.HORIZONTAL,
            length=200,
            mode="determinate",
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=(5, 0))

    def setup_keyword_tab(self):
        """Set up the keyword search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.keyword_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Keyword input
        ttk.Label(input_frame, text="Search Keywords:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.keyword_entry = ttk.Entry(input_frame, width=50)
        self.keyword_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.keyword_entry.bind("<Return>", lambda event: self.perform_keyword_search())

        # Filter options frame
        filter_frame = ttk.LabelFrame(input_frame, text="Filters")
        filter_frame.grid(
            row=1, column=0, columnspan=3, sticky=tk.W + tk.E, padx=5, pady=5
        )

        # Sender filter
        ttk.Label(filter_frame, text="Sender:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.sender_var = tk.StringVar(value="both")
        ttk.Radiobutton(
            filter_frame, text="Both", variable=self.sender_var, value="both"
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(
            filter_frame, text="Human", variable=self.sender_var, value="human"
        ).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(
            filter_frame, text="Assistant", variable=self.sender_var, value="assistant"
        ).grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Platform filter
        ttk.Label(filter_frame, text="Platform:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.platform_var = tk.StringVar(value="all")
        platform_options = ["all"] + self.platforms
        ttk.Combobox(
            filter_frame,
            textvariable=self.platform_var,
            values=platform_options,
            state="readonly",
            width=15,
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Search button
        ttk.Button(
            input_frame, text="Search", command=self.perform_keyword_search
        ).grid(row=0, column=2, padx=5, pady=5)

        # Create a PanedWindow for results and preview
        self.keyword_paned = ttk.PanedWindow(self.keyword_tab, orient=tk.HORIZONTAL)
        self.keyword_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results frame (left pane)
        results_frame = ttk.Frame(self.keyword_paned)

        # Results Treeview
        self.keyword_results_tree = ttk.Treeview(
            results_frame,
            columns=("platform", "sender", "date", "conversation"),
            show="headings",
        )

        # Set up columns
        self.keyword_results_tree.heading("platform", text="Platform")
        self.keyword_results_tree.heading("sender", text="Sender")
        self.keyword_results_tree.heading("date", text="Date")
        self.keyword_results_tree.heading("conversation", text="Conversation")

        # Set column widths
        self.keyword_results_tree.column("platform", width=80, anchor=tk.W)
        self.keyword_results_tree.column("sender", width=80, anchor=tk.W)
        self.keyword_results_tree.column("date", width=150, anchor=tk.W)
        self.keyword_results_tree.column("conversation", width=300, anchor=tk.W)

        # Create a scrollbar
        results_scroll = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self.keyword_results_tree.yview
        )
        self.keyword_results_tree.configure(yscrollcommand=results_scroll.set)

        # Pack the treeview and scrollbar
        self.keyword_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind select event
        self.keyword_results_tree.bind(
            "<<TreeviewSelect>>", self.on_keyword_result_select
        )

        # Preview frame (right pane)
        preview_frame = ttk.Frame(self.keyword_paned)

        # Preview text widget with scrollbar
        self.keyword_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD)
        self.keyword_preview.pack(fill=tk.BOTH, expand=True)

        # Add the panes to the PanedWindow
        self.keyword_paned.add(results_frame, weight=1)
        self.keyword_paned.add(preview_frame, weight=1)

        # Action buttons frame at the bottom
        action_frame = ttk.Frame(self.keyword_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            action_frame,
            text="View Full Conversation",
            command=self.view_selected_conversation,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            action_frame, text="Analyze Results", command=self.analyze_current_results
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            action_frame, text="Export Results", command=self.export_current_results
        ).pack(side=tk.LEFT, padx=5)

    def setup_boolean_tab(self):
        """Set up the boolean search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.boolean_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Boolean query input
        ttk.Label(input_frame, text="Boolean Query:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.boolean_entry = ttk.Entry(input_frame, width=50)
        self.boolean_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.boolean_entry.bind("<Return>", lambda event: self.perform_boolean_search())

        # Help text
        help_text = (
            "Examples: machine learning AND python\n"
            "          neural network OR deep learning\n"
            "          python NOT javascript"
        )
        help_label = ttk.Label(input_frame, text=help_text, foreground="gray")
        help_label.grid(row=1, column=1, sticky=tk.W, padx=5)

        # Search button
        ttk.Button(
            input_frame, text="Search", command=self.perform_boolean_search
        ).grid(row=0, column=2, padx=5, pady=5)

        # Create a PanedWindow for results and preview
        self.boolean_paned = ttk.PanedWindow(self.boolean_tab, orient=tk.HORIZONTAL)
        self.boolean_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Results frame (left pane)
        results_frame = ttk.Frame(self.boolean_paned)

        # Results Treeview
        self.boolean_results_tree = ttk.Treeview(
            results_frame,
            columns=("platform", "sender", "date", "conversation"),
            show="headings",
        )

        # Set up columns
        self.boolean_results_tree.heading("platform", text="Platform")
        self.boolean_results_tree.heading("sender", text="Sender")
        self.boolean_results_tree.heading("date", text="Date")
        self.boolean_results_tree.heading("conversation", text="Conversation")

        # Set column widths
        self.boolean_results_tree.column("platform", width=80, anchor=tk.W)
        self.boolean_results_tree.column("sender", width=80, anchor=tk.W)
        self.boolean_results_tree.column("date", width=150, anchor=tk.W)
        self.boolean_results_tree.column("conversation", width=300, anchor=tk.W)

        # Create a scrollbar
        results_scroll = ttk.Scrollbar(
            results_frame, orient=tk.VERTICAL, command=self.boolean_results_tree.yview
        )
        self.boolean_results_tree.configure(yscrollcommand=results_scroll.set)

        # Pack the treeview and scrollbar
        self.boolean_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind select event
        self.boolean_results_tree.bind(
            "<<TreeviewSelect>>", self.on_boolean_result_select
        )

        # Preview frame (right pane)
        preview_frame = ttk.Frame(self.boolean_paned)

        # Preview text widget with scrollbar
        self.boolean_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD)
        self.boolean_preview.pack(fill=tk.BOTH, expand=True)

        # Add the panes to the PanedWindow
        self.boolean_paned.add(results_frame, weight=1)
        self.boolean_paned.add(preview_frame, weight=1)

        # Action buttons frame at the bottom
        action_frame = ttk.Frame(self.boolean_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            action_frame,
            text="View Full Conversation",
            command=self.view_selected_conversation,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            action_frame, text="Analyze Results", command=self.analyze_current_results
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            action_frame, text="Export Results", command=self.export_current_results
        ).pack(side=tk.LEFT, padx=5)

    def setup_semantic_tab(self):
        """Set up the semantic search tab"""
        # Implementation details here...
        pass

    def setup_date_tab(self):
        """Set up the date search tab"""
        # Implementation details here...
        pass

    def setup_platform_model_tab(self):
        """Set up the platform/model search tab"""
        # Implementation details here...
        pass

    def setup_topic_tab(self):
        """Set up the topic search tab"""
        # Implementation details here...
        pass

    def setup_context_tab(self):
        """Set up the context window search tab"""
        # Implementation details here...
        pass

    def setup_export_tab(self):
        """Set up the export tab"""
        # Implementation details here...
        pass

    def setup_settings_tab(self):
        """Set up the settings tab"""
        # Implementation details here...
        pass

    def perform_keyword_search(self):
        """Perform a keyword search"""
        # Implementation details here...
        pass

    def perform_boolean_search(self):
        """Perform a boolean search"""
        # Implementation details here...
        pass

    def on_keyword_result_select(self, event):
        """Handle selection in keyword search results"""
        # Get selected item ID
        selected_items = self.keyword_results_tree.selection()
        if not selected_items:
            return

        item_id = selected_items[0]

        # Get message ID and conversation ID from tags
        tags = self.keyword_results_tree.item(item_id, "tags")
        message_id = tags[0]
        conversation_id = tags[1]

        # Retrieve the message content
        query = """
            SELECT m.content, m.sender, m.created_at, c.title
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.id = ?
        """
        result = self.search_engine.query_to_dataframe(query, [message_id])

        if not result.empty:
            # Clear previous preview
            self.keyword_preview.delete(1.0, tk.END)

            # Get message data
            content = result.iloc[0]["content"]
            sender = result.iloc[0]["sender"].capitalize()
            timestamp = result.iloc[0]["created_at"]
            title = result.iloc[0]["title"] or "Untitled"

            # Add message header
            header = f"From: {sender}\nDate: {timestamp}\nConversation: {title}\n\n"
            self.keyword_preview.insert(tk.END, header)

            # Insert message content
            self.keyword_preview.insert(tk.END, content)

            # Highlight search term if it's in the content
            search_term = self.keyword_entry.get().strip()
            if search_term:
                start_pos = "1.0"
                while True:
                    pos = self.keyword_preview.search(
                        search_term, start_pos, tk.END, nocase=True
                    )
                    if not pos:
                        break

                    end_pos = f"{pos}+{len(search_term)}c"
                    self.keyword_preview.tag_add("highlight", pos, end_pos)
                    self.keyword_preview.tag_config("highlight", background="yellow")

                    start_pos = end_pos

    def on_boolean_result_select(self, event):
        """Handle selection in boolean search results"""
        # Get selected item ID
        selected_items = self.boolean_results_tree.selection()
        if not selected_items:
            return

        item_id = selected_items[0]

        # Get message ID and conversation ID from tags
        tags = self.boolean_results_tree.item(item_id, "tags")
        message_id = tags[0]
        conversation_id = tags[1]

        # Retrieve the message content
        query = """
            SELECT m.content, m.sender, m.created_at, c.title
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.id = ?
        """
        result = self.search_engine.query_to_dataframe(query, [message_id])

        if not result.empty:
            # Clear previous preview
            self.boolean_preview.delete(1.0, tk.END)

            # Get message data
            content = result.iloc[0]["content"]
            sender = result.iloc[0]["sender"].capitalize()
            timestamp = result.iloc[0]["created_at"]
            title = result.iloc[0]["title"] or "Untitled"

            # Add message header
            header = f"From: {sender}\nDate: {timestamp}\nConversation: {title}\n\n"
            self.boolean_preview.insert(tk.END, header)

            # Insert message content
            self.boolean_preview.insert(tk.END, content)

            # Highlight boolean search terms
            search_query = self.boolean_entry.get().strip()
            terms = []

            # Extract terms from boolean query (simple parsing)
            parts = re.findall(r"(NOT\s+\S+|\S+)", search_query)
            for part in parts:
                if part.startswith("NOT "):
                    # Skip negated terms for highlighting
                    continue
                elif part not in ("AND", "OR"):
                    terms.append(part)

            # Highlight each term
            for term in terms:
                start_pos = "1.0"
                while True:
                    pos = self.boolean_preview.search(
                        term, start_pos, tk.END, nocase=True
                    )
                    if not pos:
                        break

                    end_pos = f"{pos}+{len(term)}c"
                    self.boolean_preview.tag_add("highlight", pos, end_pos)
                    self.boolean_preview.tag_config("highlight", background="yellow")

                    start_pos = end_pos

    def on_semantic_result_select(self, event):
        """Handle selection in semantic search results"""
        # Implementation details here...
        pass

    def on_date_result_select(self, event):
        """Handle double-click on date search results"""
        # Get selected item ID
        selected_items = self.date_results_tree.selection()
        if not selected_items:
            return

        # View the selected conversation
        self.view_date_conversation()

    def on_pm_result_select(self, event):
        """Handle double-click on platform/model search results"""
        # Get selected item ID
        selected_items = self.pm_results_tree.selection()
        if not selected_items:
            return

        # View the selected conversation
        self.view_pm_conversation()

    def on_topic_result_select(self, event):
        """Handle double-click on topic search results"""
        # Get selected item ID
        selected_items = self.topic_results_tree.selection()
        if not selected_items:
            return

        # View the selected conversation
        self.view_topic_conversation()

    def view_selected_conversation(self):
        """View the currently selected conversation in a new window"""
        # Determine which tab is active
        current_tab = self.notebook.select()

        if current_tab == str(self.keyword_tab):
            # Get selected item from keyword search
            selected_items = self.keyword_results_tree.selection()
            if not selected_items:
                messagebox.showinfo("No Selection", "Please select a message first.")
                return

            item_id = selected_items[0]
            tags = self.keyword_results_tree.item(item_id, "tags")
            message_id = tags[0]
            conversation_id = tags[1]

        elif current_tab == str(self.boolean_tab):
            # Get selected item from boolean search
            selected_items = self.boolean_results_tree.selection()
            if not selected_items:
                messagebox.showinfo("No Selection", "Please select a message first.")
                return

            item_id = selected_items[0]
            tags = self.boolean_results_tree.item(item_id, "tags")
            message_id = tags[0]
            conversation_id = tags[1]

        elif current_tab == str(self.semantic_tab):
            # Get selected item from semantic search
            selected_items = self.semantic_results_tree.selection()
            if not selected_items:
                messagebox.showinfo("No Selection", "Please select a message first.")
                return

            item_id = selected_items[0]
            tags = self.semantic_results_tree.item(item_id, "tags")
            message_id = tags[0]
            conversation_id = tags[1]

        else:
            messagebox.showinfo(
                "Not Applicable", "This action is not applicable to the current tab."
            )
            return

        # Open conversation viewer
        self.open_conversation_viewer(conversation_id, message_id)

    def view_date_conversation(self):
        """View the selected conversation from date search"""
        # Get selected item from date search
        selected_items = self.date_results_tree.selection()
        if not selected_items:
            messagebox.showinfo("No Selection", "Please select a conversation first.")
            return

        item_id = selected_items[0]
        tags = self.date_results_tree.item(item_id, "tags")
        conversation_id = tags[0]

        # Open conversation viewer
        self.open_conversation_viewer(conversation_id)

    def view_pm_conversation(self):
        """View the selected conversation from platform/model search"""
        # Get selected item from platform/model search
        selected_items = self.pm_results_tree.selection()
        if not selected_items:
            messagebox.showinfo("No Selection", "Please select a conversation first.")
            return

        item_id = selected_items[0]
        tags = self.pm_results_tree.item(item_id, "tags")
        conversation_id = tags[0]

        # Open conversation viewer
        self.open_conversation_viewer(conversation_id)

    def view_topic_conversation(self):
        """View the selected conversation from topic search"""
        # Get selected item from topic search
        selected_items = self.topic_results_tree.selection()
        if not selected_items:
            messagebox.showinfo("No Selection", "Please select a conversation first.")
            return

        item_id = selected_items[0]
        tags = self.topic_results_tree.item(item_id, "tags")
        conversation_id = tags[0]

        # Open conversation viewer
        self.open_conversation_viewer(conversation_id)

    def view_context_conversation(self):
        """View the full conversation from context search"""
        # Get currently selected tab in context results
        selected_tab = self.context_notebook.select()
        if not selected_tab:
            messagebox.showinfo("No Selection", "Please select a context result first.")
            return

        # Get the tab object
        tab = self.context_notebook.nametowidget(selected_tab)

        # Check if the tab has conversation_id attribute
        if not hasattr(tab, "conversation_id"):
            messagebox.showinfo(
                "No Conversation", "No conversation information available."
            )
            return

        conversation_id = tab.conversation_id
        message_id = getattr(tab, "match_message_id", None)

        # Open conversation viewer
        self.open_conversation_viewer(conversation_id, message_id)

    def open_conversation_viewer(self, conversation_id, highlight_message_id=None):
        """Open a new window to view the full conversation"""
        # Create a new top-level window
        viewer = tk.Toplevel(self.root)
        viewer.title("Conversation Viewer")
        viewer.geometry("900x700")

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

        conv_df = self.search_engine.query_to_dataframe(conv_query, [conversation_id])

        if conv_df.empty:
            messagebox.showerror("Error", f"Conversation not found: {conversation_id}")
            viewer.destroy()
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

        msg_df = self.search_engine.query_to_dataframe(msg_query, [conversation_id])

        if msg_df.empty:
            messagebox.showerror(
                "Error", f"No messages found for conversation: {conversation_id}"
            )
            viewer.destroy()
            return

        # Create header frame
        header_frame = ttk.Frame(viewer)
        header_frame.pack(fill=tk.X, padx=10, pady=5)

        # Display conversation info
        conv = conv_df.iloc[0]
        title = conv["title"] or "Untitled"
        platform = conv["platform"]
        date = datetime.strptime(conv["created_at"], "%Y-%m-%d %H:%M:%S").strftime(
            "%b %d, %Y %H:%M"
        )

        ttk.Label(
            header_frame, text=f"Title: {title}", font=("TkDefaultFont", 12, "bold")
        ).pack(anchor=tk.W)
        ttk.Label(header_frame, text=f"Platform: {platform} | Date: {date}").pack(
            anchor=tk.W
        )
        ttk.Label(header_frame, text=f"ID: {conversation_id}").pack(anchor=tk.W)
        ttk.Label(header_frame, text=f"Messages: {len(msg_df)}").pack(anchor=tk.W)

        ttk.Separator(viewer, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=5)

        # Create scrolled text for messages
        messages_frame = ttk.Frame(viewer)
        messages_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create a scrolled text widget
        conversation_text = scrolledtext.ScrolledText(messages_frame, wrap=tk.WORD)
        conversation_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for styling
        conversation_text.tag_configure("human", foreground="blue")
        conversation_text.tag_configure("assistant", foreground="green")
        conversation_text.tag_configure("highlight", background="yellow")
        conversation_text.tag_configure("model", foreground="gray")
        conversation_text.tag_configure("header", font=("TkDefaultFont", 10, "bold"))

        # Add messages to the text widget
        for _, msg in msg_df.iterrows():
            # Format timestamp
            timestamp = datetime.strptime(msg["created_at"], "%Y-%m-%d %H:%M:%S")
            time_str = timestamp.strftime("%H:%M:%S")

            # Format sender
            sender = msg["sender"].capitalize()

            # Determine if this is the highlighted message
            is_highlight = highlight_message_id and msg["id"] == highlight_message_id

            # Insert message header
            header = f"\n{sender} ({time_str}):\n"
            conversation_text.insert(tk.END, header, "header")

            # Insert model info if available
            if msg["model"]:
                model_text = f"[Model: {msg['model']}]\n"
                conversation_text.insert(tk.END, model_text, "model")

            # Insert message content
            if is_highlight:
                conversation_text.insert(tk.END, msg["content"], "highlight")
            else:
                conversation_text.insert(tk.END, msg["content"], msg["sender"])

            conversation_text.insert(tk.END, "\n")

        # Make text widget read-only
        conversation_text.config(state=tk.DISABLED)

        # If there's a highlighted message, scroll to it
        if highlight_message_id:
            # Find the position of the highlighted text
            highlight_ranges = conversation_text.tag_ranges("highlight")
            if highlight_ranges:
                # Scroll to the position of the highlight
                conversation_text.see(highlight_ranges[0])

        # Create bottom button frame
        button_frame = ttk.Frame(viewer)
        button_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(
            button_frame,
            text="Export Conversation",
            command=lambda: self.export_single_conversation(conversation_id),
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=viewer.destroy).pack(
            side=tk.RIGHT, padx=5
        )

        # Store reference to this conversation
        self.current_conversation = conversation_id

    def analyze_current_results(self):
        """Analyze and visualize current search results"""
        # Implementation details here...
        pass

    def export_current_results(self):
        """Export current search results"""
        # Implementation details here...
        pass

    def export_single_conversation(self, conversation_id):
        """Export a single conversation to file"""
        # Implementation details here...
        pass

    def export_selected_date_conversations(self):
        """Export selected conversations from date search"""
        # Implementation details here...
        pass

    def export_selected_pm_conversations(self):
        """Export selected conversations from platform/model search"""
        # Implementation details here...
        pass

    def export_selected_topic_conversations(self):
        """Export selected conversations from topic search"""
        # Implementation details here...
        pass

    def export_context_results(self):
        """Export context search results"""
        # Implementation details here...
        pass

    def export_conversations(self):
        """Export conversations based on selected method"""
        # Implementation details here...
        pass

    def _export_conversations_batch(self, conversation_ids):
        """Export multiple conversations in batch"""
        # Implementation details here...
        pass

    def generate_embeddings(self):
        """Generate or update embeddings for semantic search"""
        # Implementation details here...
        pass

    def _update_status(self, message):
        """Update the status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def _set_progress(self, value):
        """Set the progress bar value (0-100)"""
        self.progress_var.set(value)
        self.root.update_idletasks()

    def _reset_progress(self):
        """Reset the progress bar"""
        self.progress_var.set(0)
        self.root.update_idletasks()

    def open_directory(self, directory):
        """Open a directory in the file explorer"""
        try:
            import os
            import platform
            import subprocess

            # Make the path absolute
            abs_path = os.path.abspath(directory)

            # Open based on platform
            if platform.system() == "Windows":
                os.startfile(abs_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(["open", abs_path])
            else:  # Linux
                subprocess.call(["xdg-open", abs_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open directory: {e}")


# Main application code
def main():
    """Run the application"""
    root = tk.Tk()
    root.title("Conversation Search Engine")

    # Set application icon if available
    try:
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            # For Windows and Linux
            root.iconphoto(True, tk.PhotoImage(file=icon_path))
    except Exception:
        pass

    # Set theme - use standard theme if ttk.Style is available
    try:
        style = ttk.Style()

        # Check if 'clam' theme is available (more modern look)
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")

        # Configure style elements
        style.configure("TButton", padding=6)
        style.configure("TNotebook.Tab", padding=[12, 4])
    except Exception:
        pass

    # Initialize the application
    app = ConversationSearchGUI(root)

    # Start the main event loop
    root.mainloop()


if __name__ == "__main__":
    main()  # !/usr/bin/env python3
