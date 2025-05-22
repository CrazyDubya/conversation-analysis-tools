#!/usr/bin/env python3
"""
Conversation Search GUI

A graphical user interface for the Conversation Search Engine.
This GUI provides easy access to the search capabilities for exploring
conversation data from Claude and ChatGPT.

Features:
- Tabbed interface for different search types
- Results display with sortable columns
- Preview pane for quick content viewing
- Integrated visualization tools
- Export capabilities
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

# Import the search engine module
# We'll use its functionality but adapt it for GUI use
from conversation_search_engine import ConversationSearchEngine

# Database connection
DB_PATH = '/Users/pup/Desktop/Arch/conversations.db'

# Output directories
SEARCH_RESULTS_DIR = 'search_results'
EMBEDDINGS_DIR = 'embeddings_cache'
os.makedirs(SEARCH_RESULTS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Platform colors for visualization
PLATFORM_COLORS = {
    'claude': '#8C52FF',    # Purple
    'chatgpt': '#00A67E',   # Green
}

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
        self._update_status(f"Connected to database with {self.search_engine._count_conversations()} conversations")
    
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
        self.status_label = ttk.Label(self.status_bar, textvariable=self.status_var, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Right side: Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_bar, 
            variable=self.progress_var,
            orient=tk.HORIZONTAL,
            length=200,
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=(5, 0))
    
    def setup_keyword_tab(self):
        """Set up the keyword search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.keyword_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Keyword input
        ttk.Label(input_frame, text="Search Keywords:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.keyword_entry = ttk.Entry(input_frame, width=50)
        self.keyword_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.keyword_entry.bind("<Return>", lambda event: self.perform_keyword_search())
        
        # Filter options frame
        filter_frame = ttk.LabelFrame(input_frame, text="Filters")
        filter_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W+tk.E, padx=5, pady=5)
        
        # Sender filter
        ttk.Label(filter_frame, text="Sender:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.sender_var = tk.StringVar(value="both")
        ttk.Radiobutton(filter_frame, text="Both", variable=self.sender_var, value="both").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(filter_frame, text="Human", variable=self.sender_var, value="human").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        ttk.Radiobutton(filter_frame, text="Assistant", variable=self.sender_var, value="assistant").grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Platform filter
        ttk.Label(filter_frame, text="Platform:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.platform_var = tk.StringVar(value="all")
        platform_options = ["all"] + self.platforms
        ttk.Combobox(filter_frame, textvariable=self.platform_var, values=platform_options, state="readonly", width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Search button
        ttk.Button(input_frame, text="Search", command=self.perform_keyword_search).grid(row=0, column=2, padx=5, pady=5)
        
        # Create a PanedWindow for results and preview
        self.keyword_paned = ttk.PanedWindow(self.keyword_tab, orient=tk.HORIZONTAL)
        self.keyword_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results frame (left pane)
        results_frame = ttk.Frame(self.keyword_paned)
        
        # Results Treeview
        self.keyword_results_tree = ttk.Treeview(
            results_frame, 
            columns=("platform", "sender", "date", "conversation"),
            show="headings"
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
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.keyword_results_tree.yview)
        self.keyword_results_tree.configure(yscrollcommand=results_scroll.set)
        
        # Pack the treeview and scrollbar
        self.keyword_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind select event
        self.keyword_results_tree.bind("<<TreeviewSelect>>", self.on_keyword_result_select)
        
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
        
        ttk.Button(action_frame, text="View Full Conversation", command=self.view_selected_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Analyze Results", command=self.analyze_current_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Export Results", command=self.export_current_results).pack(side=tk.LEFT, padx=5)
    
    def setup_boolean_tab(self):
        """Set up the boolean search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.boolean_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Boolean query input
        ttk.Label(input_frame, text="Boolean Query:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.boolean_entry = ttk.Entry(input_frame, width=50)
        self.boolean_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.boolean_entry.bind("<Return>", lambda event: self.perform_boolean_search())
        
        # Help text
        help_text = "Examples: machine learning AND python\n" \
                    "          neural network OR deep learning\n" \
                    "          python NOT javascript"
        help_label = ttk.Label(input_frame, text=help_text, foreground="gray")
        help_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        
        # Search button
        ttk.Button(input_frame, text="Search", command=self.perform_boolean_search).grid(row=0, column=2, padx=5, pady=5)
        
        # Create a PanedWindow for results and preview
        self.boolean_paned = ttk.PanedWindow(self.boolean_tab, orient=tk.HORIZONTAL)
        self.boolean_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results frame (left pane)
        results_frame = ttk.Frame(self.boolean_paned)
        
        # Results Treeview
        self.boolean_results_tree = ttk.Treeview(
            results_frame, 
            columns=("platform", "sender", "date", "conversation"),
            show="headings"
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
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.boolean_results_tree.yview)
        self.boolean_results_tree.configure(yscrollcommand=results_scroll.set)
        
        # Pack the treeview and scrollbar
        self.boolean_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind select event
        self.boolean_results_tree.bind("<<TreeviewSelect>>", self.on_boolean_result_select)
        
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
        
        ttk.Button(action_frame, text="View Full Conversation", command=self.view_selected_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Analyze Results", command=self.analyze_current_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Export Results", command=self.export_current_results).pack(side=tk.LEFT, padx=5)
    
    def setup_semantic_tab(self):
        """Set up the semantic search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.semantic_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Semantic query input
        ttk.Label(input_frame, text="Semantic Query:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.semantic_entry = ttk.Entry(input_frame, width=50)
        self.semantic_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.semantic_entry.bind("<Return>", lambda event: self.perform_semantic_search())
        
        # Threshold slider
        ttk.Label(input_frame, text="Similarity Threshold:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.3)
        threshold_slider = ttk.Scale(input_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                                    variable=self.threshold_var, length=200)
        threshold_slider.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(input_frame, textvariable=tk.StringVar(value="0.0")).grid(row=1, column=1, sticky=tk.W, padx=(0, 0), pady=5)
        threshold_value_label = ttk.Label(input_frame, textvariable=self.threshold_var)
        threshold_value_label.grid(row=1, column=1, sticky=tk.E, padx=(0, 0), pady=5)
        ttk.Label(input_frame, textvariable=tk.StringVar(value="1.0")).grid(row=1, column=1, sticky=tk.E, padx=(200, 0), pady=5)
        
        # Limit input
        ttk.Label(input_frame, text="Result Limit:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.limit_var = tk.IntVar(value=20)
        limit_spinbox = ttk.Spinbox(input_frame, from_=1, to=100, textvariable=self.limit_var, width=5)
        limit_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Show embeddings status
        embeddings_status = "Embeddings: "
        if self.search_engine._check_embeddings_exist():
            embeddings_status += "Available"
            self.embeddings_available = True
        else:
            embeddings_status += "Not generated (See Settings tab)"
            self.embeddings_available = False
        
        ttk.Label(input_frame, text=embeddings_status).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Search button
        ttk.Button(input_frame, text="Search", command=self.perform_semantic_search).grid(row=0, column=2, padx=5, pady=5)
        
        # Create a PanedWindow for results and preview
        self.semantic_paned = ttk.PanedWindow(self.semantic_tab, orient=tk.HORIZONTAL)
        self.semantic_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results frame (left pane)
        results_frame = ttk.Frame(self.semantic_paned)
        
        # Results Treeview
        self.semantic_results_tree = ttk.Treeview(
            results_frame, 
            columns=("similarity", "platform", "sender", "date", "conversation"),
            show="headings"
        )
        
        # Set up columns
        self.semantic_results_tree.heading("similarity", text="Similarity")
        self.semantic_results_tree.heading("platform", text="Platform")
        self.semantic_results_tree.heading("sender", text="Sender")
        self.semantic_results_tree.heading("date", text="Date")
        self.semantic_results_tree.heading("conversation", text="Conversation")
        
        # Set column widths
        self.semantic_results_tree.column("similarity", width=70, anchor=tk.E)
        self.semantic_results_tree.column("platform", width=80, anchor=tk.W)
        self.semantic_results_tree.column("sender", width=80, anchor=tk.W)
        self.semantic_results_tree.column("date", width=150, anchor=tk.W)
        self.semantic_results_tree.column("conversation", width=300, anchor=tk.W)
        
        # Create a scrollbar
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.semantic_results_tree.yview)
        self.semantic_results_tree.configure(yscrollcommand=results_scroll.set)
        
        # Pack the treeview and scrollbar
        self.semantic_results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind select event
        self.semantic_results_tree.bind("<<TreeviewSelect>>", self.on_semantic_result_select)
        
        # Preview frame (right pane)
        preview_frame = ttk.Frame(self.semantic_paned)
        
        # Preview text widget with scrollbar
        self.semantic_preview = scrolledtext.ScrolledText(preview_frame, wrap=tk.WORD)
        self.semantic_preview.pack(fill=tk.BOTH, expand=True)
        
        # Add the panes to the PanedWindow
        self.semantic_paned.add(results_frame, weight=1)
        self.semantic_paned.add(preview_frame, weight=1)
        
        # Action buttons frame at the bottom
        action_frame = ttk.Frame(self.semantic_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="View Full Conversation", command=self.view_selected_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Analyze Results", command=self.analyze_current_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Export Results", command=self.export_current_results).pack(side=tk.LEFT, padx=5)
    
    def setup_date_tab(self):
        """Set up the date search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.date_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Date range inputs
        ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_date_entry = ttk.Entry(input_frame, width=15)
        self.start_date_entry.insert(0, self.date_range[0])
        self.start_date_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.end_date_entry = ttk.Entry(input_frame, width=15)
        self.end_date_entry.insert(0, self.date_range[1])
        self.end_date_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Platform filter
        ttk.Label(input_frame, text="Platform:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.date_platform_var = tk.StringVar(value="all")
        platform_options = ["all"] + self.platforms
        ttk.Combobox(input_frame, textvariable=self.date_platform_var, values=platform_options, state="readonly", width=15).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Optional keywords
        ttk.Label(input_frame, text="Optional Keywords:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.date_keywords_entry = ttk.Entry(input_frame, width=25)
        self.date_keywords_entry.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Search button
        ttk.Button(input_frame, text="Search", command=self.perform_date_search).grid(row=0, column=4, rowspan=2, padx=5, pady=5)
        
        # Create results treeview
        results_frame = ttk.Frame(self.date_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results Treeview
        self.date_results_tree = ttk.Treeview(
            results_frame, 
            columns=("platform", "date", "title", "messages", "human", "assistant"),
            show="headings"
        )
        
        # Set up columns
        self.date_results_tree.heading("platform", text="Platform")
        self.date_results_tree.heading("date", text="Date")
        self.date_results_tree.heading("title", text="Title")
        self.date_results_tree.heading("messages", text="Messages")
        self.date_results_tree.heading("human", text="Human")
        self.date_results_tree.heading("assistant", text="Assistant")
        
        # Set column widths
        self.date_results_tree.column("platform", width=80, anchor=tk.W)
        self.date_results_tree.column("date", width=150, anchor=tk.W)
        self.date_results_tree.column("title", width=300, anchor=tk.W)
        self.date_results_tree.column("messages", width=70, anchor=tk.E)
        self.date_results_tree.column("human", width=60, anchor=tk.E)
        self.date_results_tree.column("assistant", width=70, anchor=tk.E)
        
        # Create a scrollbar
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.date_results_tree.yview)
        self.date_results_tree.configure(yscrollcommand=results_scroll.set)
        
        # Create horizontal scrollbar
        h_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.date_results_tree.xview)
        self.date_results_tree.configure(xscrollcommand=h_scroll.set)
        
        # Pack the treeview and scrollbars
        self.date_results_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind double-click event to view conversation
        self.date_results_tree.bind("<Double-1>", self.on_date_result_select)
        
        # Action buttons frame at the bottom
        action_frame = ttk.Frame(self.date_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="View Selected Conversation", command=self.view_date_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Analyze Results", command=self.analyze_current_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Export Selected", command=self.export_selected_date_conversations).pack(side=tk.LEFT, padx=5)
    
    def setup_platform_model_tab(self):
        """Set up the platform/model search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.platform_model_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Platform filter
        ttk.Label(input_frame, text="Platform:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.pm_platform_var = tk.StringVar(value="all")
        platform_options = ["all"] + self.platforms
        ttk.Combobox(input_frame, textvariable=self.pm_platform_var, values=platform_options, state="readonly", width=15).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Model filter
        ttk.Label(input_frame, text="Model:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.pm_model_var = tk.StringVar(value="all")
        self.pm_model_combo = ttk.Combobox(input_frame, textvariable=self.pm_model_var, width=30)
        self.pm_model_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Populate model combobox with prefix matching
        self.pm_model_combo.bind('<KeyRelease>', self.filter_models)
        self.pm_model_combo.bind('<FocusIn>', self.get_models)
        
        # Date range checkbox and inputs
        self.date_filter_var = tk.BooleanVar(value=False)
        date_check = ttk.Checkbutton(input_frame, text="Filter by Date Range", variable=self.date_filter_var)
        date_check.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # Date range frame
        date_frame = ttk.Frame(input_frame)
        date_frame.grid(row=1, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(date_frame, text="Start:").pack(side=tk.LEFT, padx=(0, 2))
        self.pm_start_date_entry = ttk.Entry(date_frame, width=11)
        self.pm_start_date_entry.insert(0, self.date_range[0])
        self.pm_start_date_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(date_frame, text="End:").pack(side=tk.LEFT, padx=(5, 2))
        self.pm_end_date_entry = ttk.Entry(date_frame, width=11)
        self.pm_end_date_entry.insert(0, self.date_range[1])
        self.pm_end_date_entry.pack(side=tk.LEFT, padx=(0, 0))
        
        # Search button
        ttk.Button(input_frame, text="Search", command=self.perform_platform_model_search).grid(row=0, column=3, rowspan=2, padx=5, pady=5)
        
        # Create results treeview
        results_frame = ttk.Frame(self.platform_model_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results Treeview
        self.pm_results_tree = ttk.Treeview(
            results_frame, 
            columns=("platform", "date", "title", "messages", "models"),
            show="headings"
        )
        
        # Set up columns
        self.pm_results_tree.heading("platform", text="Platform")
        self.pm_results_tree.heading("date", text="Date")
        self.pm_results_tree.heading("title", text="Title")
        self.pm_results_tree.heading("messages", text="Messages")
        self.pm_results_tree.heading("models", text="Models Used")
        
        # Set column widths
        self.pm_results_tree.column("platform", width=80, anchor=tk.W)
        self.pm_results_tree.column("date", width=150, anchor=tk.W)
        self.pm_results_tree.column("title", width=300, anchor=tk.W)
        self.pm_results_tree.column("messages", width=70, anchor=tk.E)
        self.pm_results_tree.column("models", width=200, anchor=tk.W)
        
        # Create a scrollbar
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.pm_results_tree.yview)
        self.pm_results_tree.configure(yscrollcommand=results_scroll.set)
        
        # Create horizontal scrollbar
        h_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.pm_results_tree.xview)
        self.pm_results_tree.configure(xscrollcommand=h_scroll.set)
        
        # Pack the treeview and scrollbars
        self.pm_results_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind double-click event to view conversation
        self.pm_results_tree.bind("<Double-1>", self.on_pm_result_select)
        
        # Action buttons frame at the bottom
        action_frame = ttk.Frame(self.platform_model_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="View Selected Conversation", command=self.view_pm_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Analyze Results", command=self.analyze_current_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Export Selected", command=self.export_selected_pm_conversations).pack(side=tk.LEFT, padx=5)
    
    def setup_topic_tab(self):
        """Set up the topic search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.LabelFrame(self.topic_tab, text="Topic Selection")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Topic selection frame
        topic_frame = ttk.Frame(input_frame)
        topic_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Get common topics from search engine
        self.topics = self.search_engine._extract_topic_terms()
        
        # Create topic selection radiobuttons
        self.topic_var = tk.StringVar(value="custom")
        
        # Create a scrolled frame for topics
        topic_canvas = tk.Canvas(topic_frame, borderwidth=0)
        topic_scrollbar = ttk.Scrollbar(topic_frame, orient=tk.VERTICAL, command=topic_canvas.yview)
        topic_scrollable_frame = ttk.Frame(topic_canvas)
        
        topic_scrollable_frame.bind(
            "<Configure>",
            lambda e: topic_canvas.configure(scrollregion=topic_canvas.bbox("all"))
        )
        
        topic_canvas.create_window((0, 0), window=topic_scrollable_frame, anchor=tk.NW)
        topic_canvas.configure(yscrollcommand=topic_scrollbar.set)
        
        # Pack the canvas and scrollbar
        topic_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        topic_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add topics to the scrollable frame
        row = 0
        col = 0
        max_cols = 3
        
        for topic in self.topics.keys():
            ttk.Radiobutton(
                topic_scrollable_frame, 
                text=topic, 
                variable=self.topic_var, 
                value=topic,
                command=self.update_topic_keywords
            ).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Custom topic option
        ttk.Radiobutton(
            topic_scrollable_frame, 
            text="Custom Keywords", 
            variable=self.topic_var, 
            value="custom",
            command=self.update_topic_keywords
        ).grid(row=row + 1, column=0, columnspan=max_cols, sticky=tk.W, padx=5, pady=5)
        
        # Keywords frame
        keywords_frame = ttk.LabelFrame(input_frame, text="Keywords")
        keywords_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Keywords display/input
        self.topic_keywords_var = tk.StringVar()
        self.topic_keywords_entry = ttk.Entry(keywords_frame, textvariable=self.topic_keywords_var, width=60)
        self.topic_keywords_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        # Search button
        ttk.Button(keywords_frame, text="Search", command=self.perform_topic_search).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Create results treeview
        results_frame = ttk.Frame(self.topic_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results Treeview
        self.topic_results_tree = ttk.Treeview(
            results_frame, 
            columns=("platform", "date", "title", "relevance", "messages"),
            show="headings"
        )
        
        # Set up columns
        self.topic_results_tree.heading("platform", text="Platform")
        self.topic_results_tree.heading("date", text="Date")
        self.topic_results_tree.heading("title", text="Title")
        self.topic_results_tree.heading("relevance", text="Relevance")
        self.topic_results_tree.heading("messages", text="Messages")
        
        # Set column widths
        self.topic_results_tree.column("platform", width=80, anchor=tk.W)
        self.topic_results_tree.column("date", width=150, anchor=tk.W)
        self.topic_results_tree.column("title", width=300, anchor=tk.W)
        self.topic_results_tree.column("relevance", width=80, anchor=tk.E)
        self.topic_results_tree.column("messages", width=70, anchor=tk.E)
        
        # Create a scrollbar
        results_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.topic_results_tree.yview)
        self.topic_results_tree.configure(yscrollcommand=results_scroll.set)
        
        # Create horizontal scrollbar
        h_scroll = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.topic_results_tree.xview)
        self.topic_results_tree.configure(xscrollcommand=h_scroll.set)
        
        # Pack the treeview and scrollbars
        self.topic_results_tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind double-click event to view conversation
        self.topic_results_tree.bind("<Double-1>", self.on_topic_result_select)
        
        # Action buttons frame at the bottom
        action_frame = ttk.Frame(self.topic_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="View Selected Conversation", command=self.view_topic_conversation).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Analyze Results", command=self.analyze_current_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Export Selected", command=self.export_selected_topic_conversations).pack(side=tk.LEFT, padx=5)
    
    def setup_context_tab(self):
        """Set up the context window search tab"""
        # Split into top (input) and bottom (results) sections
        input_frame = ttk.Frame(self.context_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Context search input
        ttk.Label(input_frame, text="Search Keywords:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.context_entry = ttk.Entry(input_frame, width=50)
        self.context_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.context_entry.