#!/usr/bin/env python3
"""
Advanced Conversation Data Explorer

This script provides an interactive environment for exploring and visualizing
conversation data from the SQLite database containing both Claude and ChatGPT conversations.

Features:
- Dynamic query generation
- Real-time visualization
- Custom date range filtering
- Platform comparisons
- Topic and content analysis
- Time-series visualizations
- Interactive filtering and exploration
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime, timedelta
import numpy as np
import re
import json
import os
from collections import Counter, defaultdict
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
import textwrap
import calendar

# Set up plot style
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 12

# Database connection
DB_PATH = "/Users/pup/Desktop/Arch/conversations.db"

# Output directory for saving visualizations
OUTPUT_DIR = "advanced_visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color schemes
PLATFORM_COLORS = {
    "claude": "#8C52FF",  # Purple
    "chatgpt": "#00A67E",  # Green
}

# Custom colormap for heatmaps
cmap_colors = [
    (0.95, 0.95, 1),
    (0.8, 0.8, 1),
    (0.6, 0.6, 1),
    (0.4, 0.4, 0.9),
    (0.2, 0.2, 0.8),
    (0, 0, 0.7),
]
custom_cmap = LinearSegmentedColormap.from_list("custom_blues", cmap_colors)


class ConversationAnalyzer:
    """Class for analyzing and visualizing conversation data"""

    def __init__(self, db_path=DB_PATH):
        """Initialize the analyzer with database connection"""
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()

        # Check if views exist
        self._check_views()

        # Cache some frequently used data
        self.platforms = self._get_platforms()
        self.date_range = self._get_date_range()
        self.model_list = self._get_models()

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
                raise

    def _get_platforms(self):
        """Get list of unique platforms in the database"""
        self.cursor.execute("SELECT DISTINCT platform FROM conversations")
        return [row[0] for row in self.cursor.fetchall() if row[0]]

    def _get_date_range(self):
        """Get min and max dates in the database"""
        self.cursor.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM conversations"
        )
        row = self.cursor.fetchone()
        return row[0][:10], row[1][:10]  # Extract just the date part

    def _get_models(self):
        """Get list of unique models in the database"""
        self.cursor.execute(
            "SELECT DISTINCT model FROM messages WHERE model IS NOT NULL AND model != ''"
        )
        return [row[0] for row in self.cursor.fetchall()]

    def _count_conversations(self):
        """Count total conversations in the database"""
        self.cursor.execute("SELECT COUNT(*) FROM conversations")
        return self.cursor.fetchone()[0]

    def query_to_dataframe(self, query, params=()):
        """Execute SQL query and return results as a pandas DataFrame"""
        return pd.read_sql_query(query, self.conn, params=params)

    def run_analysis(self):
        """Run the interactive analysis session"""
        while True:
            self._show_menu()
            choice = input("\nEnter your choice (q to quit): ").strip()

            if choice.lower() == "q":
                print("Exiting...")
                break

            try:
                choice = int(choice)
                self._handle_menu_choice(choice)
            except ValueError:
                print("Invalid choice. Please enter a number.")
            except Exception as e:
                print(f"Error: {e}")

    def _show_menu(self):
        """Display the main menu options"""
        print("\n==== CONVERSATION DATA EXPLORER ====")
        print("1. Activity Dashboard")
        print("2. Platform Comparison")
        print("3. Message Content Analysis")
        print("4. Time Analysis")
        print("5. Model Usage Analysis")
        print("6. Response Time Analysis")
        print("7. Custom Query Visualization")
        print("8. Topic Analysis")
        print("9. User Engagement Patterns")
        print("10. Export All Visualizations")
        print("q. Quit")

    def _handle_menu_choice(self, choice):
        """Handle the user's menu selection"""
        if choice == 1:
            self.show_activity_dashboard()
        elif choice == 2:
            self.show_platform_comparison()
        elif choice == 3:
            self.analyze_message_content()
        elif choice == 4:
            self.analyze_time_patterns()
        elif choice == 5:
            self.analyze_model_usage()
        elif choice == 6:
            self.analyze_response_times()
        elif choice == 7:
            self.custom_query_visualization()
        elif choice == 8:
            self.topic_analysis()
        elif choice == 9:
            self.user_engagement_patterns()
        elif choice == 10:
            self.export_all_visualizations()
        else:
            print("Invalid choice")

    def show_activity_dashboard(self):
        """Show a comprehensive activity dashboard"""
        print("\nGenerating activity dashboard...")

        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Conversation activity over time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_conversation_activity(ax1)

        # 2. Message distribution by platform
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_message_distribution(ax2)

        # 3. Average message length
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_message_length(ax3)

        # 4. Conversation duration distribution
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_conversation_duration(ax4)

        # 5. Daily activity heatmap
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_daily_activity_heatmap(ax5)

        # Add a title for the dashboard
        fig.suptitle("Conversation Activity Dashboard", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save and show
        plt.savefig(os.path.join(OUTPUT_DIR, "activity_dashboard.png"))
        plt.show()

    def _plot_conversation_activity(self, ax):
        """Plot conversation activity over time"""
        query = """
            SELECT 
                platform,
                strftime('%Y-%m', activity_date) AS month,
                SUM(conversations) AS total_conversations
            FROM time_activity
            GROUP BY platform, month
            ORDER BY month
        """
        df = self.query_to_dataframe(query)

        # Convert month to datetime for better x-axis formatting
        df["month_dt"] = pd.to_datetime(df["month"] + "-01")

        # Plot for each platform
        for platform in self.platforms:
            platform_data = df[df["platform"] == platform]
            if not platform_data.empty:
                ax.plot(
                    platform_data["month_dt"],
                    platform_data["total_conversations"],
                    marker="o",
                    linewidth=2,
                    label=platform,
                    color=PLATFORM_COLORS.get(platform, "gray"),
                )

        # Format the x-axis to show months
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        ax.set_title("Conversation Activity Over Time")
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Conversations")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_message_distribution(self, ax):
        """Plot message distribution by platform and sender"""
        query = """
            SELECT 
                c.platform,
                m.sender,
                COUNT(*) as message_count
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender IN ('human', 'assistant')
            GROUP BY c.platform, m.sender
        """
        df = self.query_to_dataframe(query)

        # Pivot the data for plotting
        pivot_df = df.pivot(index="platform", columns="sender", values="message_count")

        # Plot stacked bars
        pivot_df.plot(kind="bar", stacked=True, ax=ax, color=["#2E86C1", "#28B463"])

        ax.set_title("Message Distribution by Platform and Sender")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Message Count")
        ax.grid(True, alpha=0.3, axis="y")

        # Add count labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", label_type="center")

    def _plot_message_length(self, ax):
        """Plot average message length by platform and sender"""
        query = """
            SELECT 
                c.platform,
                m.sender,
                AVG(LENGTH(m.content)) AS avg_length
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender IN ('human', 'assistant')
            GROUP BY c.platform, m.sender
        """
        df = self.query_to_dataframe(query)

        # Pivot the data for plotting
        pivot_df = df.pivot(index="platform", columns="sender", values="avg_length")

        # Plot grouped bars
        pivot_df.plot(kind="bar", ax=ax, color=["#2E86C1", "#28B463"])

        ax.set_title("Average Message Length by Platform and Sender")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Average Characters")
        ax.grid(True, alpha=0.3, axis="y")

        # Add count labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", label_type="edge")

    def _plot_conversation_duration(self, ax):
        """Plot conversation duration distribution"""
        query = """
            SELECT 
                platform,
                conversation_duration_minutes
            FROM conversation_summary
            WHERE conversation_duration_minutes > 0 
              AND conversation_duration_minutes < 500  -- Filter outliers
        """
        df = self.query_to_dataframe(query)

        # Plot histogram for each platform
        for platform in self.platforms:
            platform_data = df[df["platform"] == platform]
            if not platform_data.empty:
                ax.hist(
                    platform_data["conversation_duration_minutes"],
                    bins=20,
                    alpha=0.6,
                    label=platform,
                    color=PLATFORM_COLORS.get(platform, "gray"),
                )

        ax.set_title("Conversation Duration Distribution")
        ax.set_xlabel("Duration (minutes)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend()

    def _plot_daily_activity_heatmap(self, ax):
        """Plot daily activity heatmap (day of week vs hour of day)"""
        query = """
            SELECT 
                strftime('%w', created_at) AS day_of_week,
                strftime('%H', created_at) AS hour_of_day,
                COUNT(*) as message_count
            FROM messages
            GROUP BY day_of_week, hour_of_day
        """
        df = self.query_to_dataframe(query)

        # Convert to numeric
        df["day_of_week"] = pd.to_numeric(df["day_of_week"])
        df["hour_of_day"] = pd.to_numeric(df["hour_of_day"])

        # Create a pivot table for the heatmap
        pivot_df = df.pivot_table(
            index="day_of_week",
            columns="hour_of_day",
            values="message_count",
            fill_value=0,
        )

        # Reindex to ensure all hours and days are present
        pivot_df = pivot_df.reindex(index=range(7), columns=range(24), fill_value=0)

        # Create the heatmap
        im = ax.imshow(pivot_df, cmap=custom_cmap, aspect="auto")

        # Add colorbar
        plt.colorbar(im, ax=ax, label="Message Count")

        # Set labels
        days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        ax.set_yticks(range(7))
        ax.set_yticklabels(days)

        # Set hour labels for every 3 hours
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])

        ax.set_title("Message Activity by Day and Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")

    def show_platform_comparison(self):
        """Show comparative analysis between platforms"""
        print("\nGenerating platform comparison visualizations...")

        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Response length comparison
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_response_length_comparison(ax1)

        # 2. Response time comparison
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_response_time_comparison(ax2)

        # 3. Conversation count by day of week
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_conversation_dow_comparison(ax3)

        # 4. Message ratio comparison
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_message_ratio_comparison(ax4)

        # 5. Word usage comparison
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_word_usage_comparison(ax5)

        # Add a title for the dashboard
        fig.suptitle("Platform Comparison Analysis", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save and show
        plt.savefig(os.path.join(OUTPUT_DIR, "platform_comparison.png"))
        plt.show()

    def _plot_response_length_comparison(self, ax):
        """Plot response length comparison between platforms"""
        query = """
            SELECT 
                c.platform,
                AVG(LENGTH(m.content)) AS avg_human_length,
                AVG(LENGTH(a.content)) AS avg_assistant_length,
                AVG(LENGTH(a.content)) / AVG(LENGTH(m.content)) AS response_ratio
            FROM messages m
            JOIN messages a ON m.conversation_id = a.conversation_id
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender = 'human' AND a.sender = 'assistant'
              AND m.order_index + 1 = a.order_index
            GROUP BY c.platform
        """
        df = self.query_to_dataframe(query)

        # Create a grouped bar chart
        x = np.arange(len(df))
        width = 0.35

        ax.bar(x - width / 2, df["avg_human_length"], width, label="Human")
        ax.bar(x + width / 2, df["avg_assistant_length"], width, label="Assistant")

        # Add a line for the ratio
        ax2 = ax.twinx()
        ax2.plot(x, df["response_ratio"], "ro-", label="Ratio")
        ax2.set_ylabel("Response Ratio (Assistant/Human)")
        ax2.set_ylim(0, max(df["response_ratio"]) * 1.2)

        # Set the x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(df["platform"])

        # Add labels
        ax.set_title("Response Length Comparison")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Average Characters")

        # Add legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")

        # Add value labels
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2, height),
                ha="center",
                va="bottom",
            )

    def _plot_response_time_comparison(self, ax):
        """Plot response time comparison between platforms"""
        query = """
            SELECT
                c.platform,
                AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS avg_response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            GROUP BY c.platform
        """
        df = self.query_to_dataframe(query)

        # Create bar chart
        ax.bar(
            df["platform"],
            df["avg_response_time_minutes"],
            color=[PLATFORM_COLORS.get(p, "gray") for p in df["platform"]],
        )

        # Add labels
        ax.set_title("Average Response Time Comparison")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Average Response Time (minutes)")

        # Add value labels
        for i, v in enumerate(df["avg_response_time_minutes"]):
            ax.text(i, v + 0.1, f"{v:.1f}", ha="center")

        ax.grid(True, alpha=0.3, axis="y")

    def _plot_conversation_dow_comparison(self, ax):
        """Plot conversation count by day of week for each platform"""
        query = """
            SELECT 
                c.platform,
                strftime('%w', c.created_at) AS day_of_week,
                COUNT(*) as conversation_count
            FROM conversations c
            GROUP BY c.platform, day_of_week
        """
        df = self.query_to_dataframe(query)

        # Convert day_of_week to numeric and add day name
        df["day_of_week"] = pd.to_numeric(df["day_of_week"])
        days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        df["day_name"] = df["day_of_week"].apply(lambda x: days[int(x)])

        # Pivot the data for plotting
        pivot_df = df.pivot(
            index="day_name", columns="platform", values="conversation_count"
        )

        # Reorder days to start with Monday
        day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        pivot_df = pivot_df.reindex(day_order)

        # Plot
        pivot_df.plot(
            kind="bar",
            ax=ax,
            color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
        )

        ax.set_title("Conversation Count by Day of Week")
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Conversation Count")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="Platform")

    def _plot_message_ratio_comparison(self, ax):
        """Plot human to assistant message ratio comparison"""
        query = """
            SELECT 
                c.platform,
                SUM(CASE WHEN m.sender = 'human' THEN 1 ELSE 0 END) AS human_messages,
                SUM(CASE WHEN m.sender = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            GROUP BY c.platform
        """
        df = self.query_to_dataframe(query)

        # Calculate ratios
        df["messages_per_human"] = df["assistant_messages"] / df["human_messages"]
        df["human_pct"] = (
            df["human_messages"]
            / (df["human_messages"] + df["assistant_messages"])
            * 100
        )
        df["assistant_pct"] = (
            df["assistant_messages"]
            / (df["human_messages"] + df["assistant_messages"])
            * 100
        )

        # Create stacked percentage bars
        df_pct = df[["platform", "human_pct", "assistant_pct"]]
        df_pct.set_index("platform", inplace=True)

        df_pct.plot(kind="bar", stacked=True, ax=ax, color=["#3498DB", "#2ECC71"])

        # Add a text of the ratio
        for i, row in enumerate(df.itertuples()):
            ax.text(i, 102, f"Ratio: {row.messages_per_human:.2f}", ha="center")

        ax.set_title("Message Distribution Ratio")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Percentage (%)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(["Human", "Assistant"])

        # Set y-axis to go to 100%
        ax.set_ylim(0, 110)

        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="center")

    def _plot_word_usage_comparison(self, ax):
        """Plot word usage comparison between platforms"""
        # Sample common and platform-specific words based on simple regex
        query = """
            SELECT
                c.platform,
                m.content
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender = 'assistant'
            LIMIT 1000
        """
        df = self.query_to_dataframe(query)

        # Process the text to find common words
        platform_words = {}
        common_words = Counter()

        for platform in self.platforms:
            platform_content = " ".join(
                df[df["platform"] == platform]["content"].tolist()
            )
            # Simple tokenization - split on non-alphanumeric and convert to lowercase
            words = re.findall(r"\b[a-zA-Z]{4,}\b", platform_content.lower())

            # Remove common English stopwords
            stopwords = {
                "this",
                "that",
                "with",
                "from",
                "have",
                "which",
                "would",
                "about",
                "there",
                "their",
                "these",
            }
            words = [w for w in words if w not in stopwords]

            # Count words
            word_counts = Counter(words)
            platform_words[platform] = word_counts
            common_words.update(word_counts)

        # Get top common words
        top_common = [word for word, _ in common_words.most_common(10)]

        # Create a dataframe for plotting
        plot_data = []
        for word in top_common:
            row = {"word": word}
            for platform in self.platforms:
                # Get frequency per 1000 words
                total_words = sum(platform_words[platform].values())
                frequency = platform_words[platform].get(word, 0) / total_words * 1000
                row[platform] = frequency
            plot_data.append(row)

        plot_df = pd.DataFrame(plot_data)
        plot_df.set_index("word", inplace=True)

        # Plot
        plot_df.plot(
            kind="bar",
            ax=ax,
            color=[PLATFORM_COLORS.get(p, "gray") for p in self.platforms],
        )

        ax.set_title("Common Word Usage Frequency Comparison")
        ax.set_xlabel("Word")
        ax.set_ylabel("Frequency per 1000 words")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="Platform")

    def analyze_message_content(self):
        """Analyze and visualize message content patterns"""
        print("\nGenerating message content analysis...")

        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Top words by platform
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_top_words_by_platform(ax1)

        # 2. Code usage by platform
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_code_usage(ax2)

        # 3. Question frequency
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_question_frequency(ax3)

        # 4. Sentiment analysis
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_simple_sentiment(ax4)

        # 5. Message complexity
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_message_complexity(ax5)

        # Add a title for the dashboard
        fig.suptitle("Message Content Analysis", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save and show
        plt.savefig(os.path.join(OUTPUT_DIR, "message_content_analysis.png"))
        plt.show()

    def _plot_top_words_by_platform(self, ax):
        """Plot top words used by each platform"""
        query = """
            SELECT
                c.platform,
                m.content
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender = 'assistant'
            LIMIT 2000
        """
        df = self.query_to_dataframe(query)

        # Process the text to find common words
        platform_top_words = {}

        for platform in self.platforms:
            platform_content = " ".join(
                df[df["platform"] == platform]["content"].tolist()
            )
            # Simple tokenization - split on non-alphanumeric and convert to lowercase
            words = re.findall(r"\b[a-zA-Z]{4,}\b", platform_content.lower())

            # Remove common English stopwords
            stopwords = {
                "this",
                "that",
                "with",
                "from",
                "have",
                "which",
                "would",
                "about",
                "there",
                "their",
                "these",
                "your",
                "you",
                "can",
                "will",
                "and",
                "for",
                "are",
                "the",
                "not",
                "but",
                "what",
                "all",
                "use",
            }
            words = [w for w in words if w not in stopwords]

            # Count words and get top 10
            word_counts = Counter(words)
            platform_top_words[platform] = word_counts.most_common(10)

        # Set up subplots for each platform
        positions = range(10)
        width = 0.35

        offset = 0
        for platform in self.platforms:
            words, counts = (
                zip(*platform_top_words[platform])
                if platform_top_words[platform]
                else ([], [])
            )
            ax.barh(
                [p + offset for p in positions],
                counts,
                width,
                label=platform,
                color=PLATFORM_COLORS.get(platform, "gray"),
            )

            # Add word labels
            for i, (word, count) in enumerate(platform_top_words[platform]):
                ax.text(count + 5, i + offset, word, va="center")

            offset += width

        # Customize the plot
        ax.set_yticks([p + width / 2 for p in positions])
        ax.set_yticklabels(range(1, 11))  # 1-10 ranking
        ax.set_title("Top 10 Words by Platform")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Rank")
        ax.legend(title="Platform")
        ax.invert_yaxis()  # Invert y-axis to have rank 1 at the top

    def _plot_code_usage(self, ax):
        """Plot code block usage frequency by platform"""
        query = """
            SELECT
                c.platform,
                COUNT(*) as total_messages,
                SUM(CASE WHEN m.content LIKE '%```%' THEN 1 ELSE 0 END) as code_messages
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender = 'assistant'
            GROUP BY c.platform
        """
        df = self.query_to_dataframe(query)

        # Calculate percentage
        df["code_percentage"] = df["code_messages"] / df["total_messages"] * 100

        # Plot
        ax.bar(
            df["platform"],
            df["code_percentage"],
            color=[PLATFORM_COLORS.get(p, "gray") for p in df["platform"]],
        )

        # Add percentage labels
        for i, pct in enumerate(df["code_percentage"]):
            ax.text(i, pct + 0.5, f"{pct:.1f}%", ha="center")

        ax.set_title("Code Block Usage Frequency")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Percentage of Messages with Code Blocks")
        ax.grid(True, alpha=0.3, axis="y")

        # Set y-axis to go to 100%
        ax.set_ylim(0, max(df["code_percentage"]) * 1.2)

    def _plot_question_frequency(self, ax):
        """Plot question mark usage frequency by platform and sender"""
        query = """
            SELECT
                c.platform,
                m.sender,
                COUNT(*) as total_messages,
                SUM(CASE WHEN m.content LIKE '%?%' THEN 1 ELSE 0 END) as question_messages
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender IN ('human', 'assistant')
            GROUP BY c.platform, m.sender
        """
        df = self.query_to_dataframe(query)

        # Calculate percentage
        df["question_percentage"] = df["question_messages"] / df["total_messages"] * 100

        # Pivot the data for plotting
        pivot_df = df.pivot(
            index="platform", columns="sender", values="question_percentage"
        )

        # Plot
        pivot_df.plot(kind="bar", ax=ax)

        ax.set_title("Question Mark Usage Frequency")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Percentage of Messages with Question Marks")
        ax.grid(True, alpha=0.3, axis="y")

        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="edge")

    def _plot_simple_sentiment(self, ax):
        """Plot simple sentiment analysis based on positive/negative word frequency"""
        positive_words = [
            "great",
            "good",
            "excellent",
            "wonderful",
            "happy",
            "glad",
            "beautiful",
            "best",
            "better",
            "success",
            "successful",
            "easy",
            "helpful",
            "enjoy",
        ]
        negative_words = [
            "bad",
            "difficult",
            "hard",
            "problem",
            "issue",
            "error",
            "fail",
            "sorry",
            "unfortunately",
            "trouble",
            "wrong",
            "not",
            "cannot",
            "can't",
            "don't",
        ]

        query = """
            SELECT
                c.platform,
                m.content
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender = 'assistant'
            LIMIT 2000
        """
        df = self.query_to_dataframe(query)

        # Process the text for sentiment analysis
        platform_sentiment = {}

        for platform in self.platforms:
            platform_content = " ".join(
                df[df["platform"] == platform]["content"].tolist()
            ).lower()

            # Count positive and negative words
            pos_count = sum(
                platform_content.count(" " + word + " ") for word in positive_words
            )
            neg_count = sum(
                platform_content.count(" " + word + " ") for word in negative_words
            )

            # Calculate sentiment score (simple ratio)
            total = pos_count + neg_count
            pos_pct = (pos_count / total * 100) if total > 0 else 0
            neg_pct = (neg_count / total * 100) if total > 0 else 0

            platform_sentiment[platform] = {"positive": pos_pct, "negative": neg_pct}

        # Create dataframe for plotting
        sentiment_data = []
        for platform, scores in platform_sentiment.items():
            sentiment_data.append(
                {
                    "platform": platform,
                    "positive": scores["positive"],
                    "negative": scores["negative"],
                }
            )

        sentiment_df = pd.DataFrame(sentiment_data)
        sentiment_df.set_index("platform", inplace=True)

        # Plot
        sentiment_df.plot(kind="bar", ax=ax, color=["#2ECC71", "#E74C3C"])

        ax.set_title("Simple Sentiment Analysis")
        ax.set_xlabel("Platform")
        ax.set_ylabel("Percentage of Sentiment Words")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(["Positive", "Negative"])

        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="edge")

    def _plot_message_complexity(self, ax):
        """Plot message complexity by platform based on sentence length and word length"""
        query = """
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
        df = self.query_to_dataframe(query)

        # Calculate words per sentence
        df["words_per_sentence"] = df["avg_words"] / df["avg_sentences"]

        # Calculate average word length
        df["avg_word_length"] = (df["avg_length"] - df["avg_words"]) / df["avg_words"]

        # Create a scatter plot with bubble size proportional to average message length
        ax.scatter(
            df["words_per_sentence"],
            df["avg_word_length"],
            s=df["avg_length"] / 10,  # Scale bubble size
            alpha=0.7,
            c=[PLATFORM_COLORS.get(p, "gray") for p in df["platform"]],
        )

        # Add platform labels
        for i, row in df.iterrows():
            ax.annotate(
                row["platform"],
                (row["words_per_sentence"], row["avg_word_length"]),
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax.set_title("Message Complexity Comparison")
        ax.set_xlabel("Words per Sentence")
        ax.set_ylabel("Average Word Length (characters)")
        ax.grid(True, alpha=0.3)

        # Add a note about bubble size
        ax.text(
            0.05,
            0.05,
            "Bubble size represents average message length",
            transform=ax.transAxes,
            fontsize=10,
            alpha=0.7,
        )

    def analyze_time_patterns(self):
        """Analyze and visualize time-based patterns in conversation data"""
        print("\nAnalyzing time patterns...")

        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Activity by hour of day
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_hourly_activity(ax1)

        # 2. Activity by day of week
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_dow_activity(ax2)

        # 3. Activity by month
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_monthly_activity(ax3)

        # 4. Activity heatmap (hour x day)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_hour_day_heatmap(ax4)

        # 5. Conversation duration over time
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_duration_over_time(ax5)

        # Add a title for the dashboard
        fig.suptitle("Time Pattern Analysis", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save and show
        plt.savefig(os.path.join(OUTPUT_DIR, "time_pattern_analysis.png"))
        plt.show()

    def _plot_hourly_activity(self, ax):
        """Plot activity by hour of day"""
        query = """
            SELECT 
                c.platform,
                strftime('%H', m.created_at) AS hour,
                COUNT(*) as message_count
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            GROUP BY c.platform, hour
            ORDER BY hour
        """
        df = self.query_to_dataframe(query)

        # Convert hour to numeric
        df["hour"] = pd.to_numeric(df["hour"])

        # Pivot the data for plotting
        pivot_df = df.pivot(index="hour", columns="platform", values="message_count")

        # Fill missing values with 0
        pivot_df = pivot_df.fillna(0)

        # Plot
        pivot_df.plot(
            kind="bar",
            ax=ax,
            color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
        )

        ax.set_title("Conversation Duration Distribution")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Number of Conversations")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="Platform")

        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt="%d", label_type="edge", fontsize=8)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    def _plot_user_retention(self, ax):
        """Plot user retention over time"""
        query = """
            SELECT 
                strftime('%Y-%m', created_at) AS month,
                COUNT(DISTINCT account_id) as unique_users,
                COUNT(*) as total_conversations
            FROM conversations
            WHERE account_id IS NOT NULL
            GROUP BY month
            ORDER BY month
        """
        df = self.query_to_dataframe(query)

        # Convert month to datetime for better plotting
        df["month_dt"] = pd.to_datetime(df["month"] + "-01")

        # Create figure with two y-axes
        ax2 = ax.twinx()

        # Plot lines
        line1 = ax.plot(
            df["month_dt"], df["unique_users"], "b-", marker="o", label="Unique Users"
        )
        line2 = ax2.plot(
            df["month_dt"],
            df["total_conversations"],
            "r-",
            marker="s",
            label="Total Conversations",
        )

        # Format the x-axis to show months
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Set labels
        ax.set_title("User Retention and Conversation Volume")
        ax.set_xlabel("Month")
        ax.set_ylabel("Unique Users", color="b")
        ax2.set_ylabel("Total Conversations", color="r")

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left")

    def _plot_time_of_day_engagement(self, ax):
        """Plot time of day engagement patterns by platform"""
        query = """
            SELECT 
                c.platform,
                strftime('%H', m.created_at) AS hour,
                COUNT(*) as message_count
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            GROUP BY c.platform, hour
            ORDER BY hour
        """
        df = self.query_to_dataframe(query)

        # Convert hour to numeric
        df["hour"] = pd.to_numeric(df["hour"])

        # Normalize counts by platform
        for platform in df["platform"].unique():
            platform_total = df[df["platform"] == platform]["message_count"].sum()
            df.loc[df["platform"] == platform, "message_pct"] = (
                df.loc[df["platform"] == platform, "message_count"]
                / platform_total
                * 100
            )

        # Pivot for plotting
        pivot_df = df.pivot(index="hour", columns="platform", values="message_pct")

        # Fill missing values with 0
        pivot_df = pivot_df.fillna(0)

        # Plot
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)

        # Set up the radar chart
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)  # 24 hours
        angles = np.concatenate((angles, [angles[0]]))  # Close the loop

        # Draw one axis per hour and label them
        ax.set_thetagrids(range(0, 360, 15), [f"{h:02d}h" for h in range(0, 24)])

        # Plot each platform
        for platform in pivot_df.columns:
            values = pivot_df[platform].values
            values = np.concatenate((values, [values[0]]))  # Close the loop

            ax.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=platform,
                color=PLATFORM_COLORS.get(platform, "gray"),
            )
            ax.fill(
                angles, values, alpha=0.25, color=PLATFORM_COLORS.get(platform, "gray")
            )

        ax.set_title("Time of Day Engagement Pattern")
        ax.legend(loc="center", bbox_to_anchor=(0.5, -0.1), ncol=len(pivot_df.columns))

    def export_all_visualizations(self):
        """Generate and export all visualizations"""
        print("\nGenerating and exporting all visualizations...")

        # Run all analysis modules
        self.show_activity_dashboard()
        self.show_platform_comparison()
        self.analyze_message_content()
        self.analyze_time_patterns()
        self.analyze_model_usage()
        self.analyze_response_times()
        self.topic_analysis()
        self.user_engagement_patterns()

        # Also export some tables as CSV
        print("\nExporting data tables...")

        # 1. Monthly activity
        query = """
            SELECT 
                platform,
                strftime('%Y-%m', activity_date) AS month,
                SUM(conversations) AS total_conversations
            FROM time_activity
            GROUP BY platform, month
            ORDER BY month, platform
        """
        monthly_activity = self.query_to_dataframe(query)
        monthly_activity.to_csv(
            os.path.join(OUTPUT_DIR, "monthly_activity.csv"), index=False
        )
        print(f"Exported monthly_activity.csv")

        # 2. Response time comparison
        query = """
            SELECT
                c.platform,
                AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS avg_response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            GROUP BY c.platform
        """
        response_time = self.query_to_dataframe(query)
        response_time.to_csv(
            os.path.join(OUTPUT_DIR, "response_time_comparison.csv"), index=False
        )
        print(f"Exported response_time_comparison.csv")

        # 3. Model usage
        query = """
            SELECT * FROM model_usage
        """
        model_usage = self.query_to_dataframe(query)
        model_usage.to_csv(os.path.join(OUTPUT_DIR, "model_usage.csv"), index=False)
        print(f"Exported model_usage.csv")

        # 4. Message content stats
        query = """
            SELECT
                c.platform,
                AVG(LENGTH(m.content)) as avg_length,
                AVG(LENGTH(m.content) - LENGTH(REPLACE(m.content, '.', ''))) as avg_sentences,
                AVG(LENGTH(m.content) - LENGTH(REPLACE(m.content, ' ', ''))) as avg_words,
                AVG((LENGTH(m.content) - LENGTH(REPLACE(m.content, ' ', ''))) / 
                    CASE WHEN (LENGTH(m.content) - LENGTH(REPLACE(m.content, '.', ''))) = 0 THEN 1 
                    ELSE (LENGTH(m.content) - LENGTH(REPLACE(m.content, '.', ''))) END) as words_per_sentence
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE m.sender = 'assistant'
            GROUP BY c.platform
        """
        content_stats = self.query_to_dataframe(query)
        content_stats.to_csv(
            os.path.join(OUTPUT_DIR, "message_content_stats.csv"), index=False
        )
        print(f"Exported message_content_stats.csv")

        print(f"\nAll visualizations and data exported to {OUTPUT_DIR}/ directory")


def main():
    """Main function to run the conversation analyzer"""
    analyzer = ConversationAnalyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()

    # Plot
    pivot_df.plot(
        kind="line",
        marker="o",
        ax=ax,
        color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
    )

    ax.set_title("Message Activity by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Message Count")

    # Set x-axis ticks for each hour
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.grid(True, alpha=0.3)
    ax.legend(title="Platform")


def _plot_dow_activity(self, ax):
    """Plot activity by day of week"""
    query = """
            SELECT 
                c.platform,
                strftime('%w', m.created_at) AS day_of_week,
                COUNT(*) as message_count
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            GROUP BY c.platform, day_of_week
            ORDER BY day_of_week
        """
    df = self.query_to_dataframe(query)

    # Convert day_of_week to numeric and add day name
    df["day_of_week"] = pd.to_numeric(df["day_of_week"])
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    df["day_name"] = df["day_of_week"].apply(lambda x: days[int(x)])

    # Pivot the data for plotting
    pivot_df = df.pivot(index="day_name", columns="platform", values="message_count")

    # Reorder days to start with Monday
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot_df = pivot_df.reindex(day_order)

    # Plot
    pivot_df.plot(
        kind="bar",
        ax=ax,
        color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
    )

    ax.set_title("Message Activity by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Message Count")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Platform")


def _plot_monthly_activity(self, ax):
    """Plot activity by month"""
    query = """
            SELECT 
                c.platform,
                strftime('%Y-%m', m.created_at) AS month,
                COUNT(*) as message_count
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            GROUP BY c.platform, month
            ORDER BY month
        """
    df = self.query_to_dataframe(query)

    # Convert month to datetime for better plotting
    df["month_dt"] = pd.to_datetime(df["month"] + "-01")

    # Pivot the data for plotting
    pivot_df = df.pivot(index="month_dt", columns="platform", values="message_count")

    # Plot
    pivot_df.plot(
        kind="line",
        marker="o",
        ax=ax,
        color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
    )

    # Format the x-axis to show months
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_title("Monthly Message Activity")
    ax.set_xlabel("Month")
    ax.set_ylabel("Message Count")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Platform")


def _plot_hour_day_heatmap(self, ax):
    """Plot activity heatmap by hour and day"""
    query = """
            SELECT 
                strftime('%w', created_at) AS day_of_week,
                strftime('%H', created_at) AS hour_of_day,
                COUNT(*) as message_count
            FROM messages
            GROUP BY day_of_week, hour_of_day
        """
    df = self.query_to_dataframe(query)

    # Convert to numeric
    df["day_of_week"] = pd.to_numeric(df["day_of_week"])
    df["hour_of_day"] = pd.to_numeric(df["hour_of_day"])

    # Create a pivot table for the heatmap
    pivot_df = df.pivot_table(
        index="day_of_week", columns="hour_of_day", values="message_count", fill_value=0
    )

    # Reindex to ensure all hours and days are present
    pivot_df = pivot_df.reindex(index=range(7), columns=range(24), fill_value=0)

    # Create the heatmap
    im = ax.imshow(pivot_df, cmap=custom_cmap, aspect="auto")

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Message Count")

    # Set labels
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    ax.set_yticks(range(7))
    ax.set_yticklabels(days)

    # Set hour labels for every 3 hours
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])

    ax.set_title("Message Activity Heatmap")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week")

    # Add annotations with message counts
    for i in range(7):
        for j in range(24):
            count = pivot_df.iloc[i, j]
            if count > 0:
                ax.text(
                    j,
                    i,
                    str(int(count)),
                    ha="center",
                    va="center",
                    color="white" if count > pivot_df.max().max() / 3 else "black",
                    fontsize=8,
                )


def _plot_duration_over_time(self, ax):
    """Plot average conversation duration over time"""
    query = """
            SELECT 
                c.platform,
                strftime('%Y-%m', c.created_at) AS month,
                AVG(conversation_duration_minutes) as avg_duration
            FROM conversation_summary s
            JOIN conversations c ON s.conversation_id = c.id
            WHERE conversation_duration_minutes > 0
              AND conversation_duration_minutes < 500  -- Filter outliers
            GROUP BY c.platform, month
            ORDER BY month
        """
    df = self.query_to_dataframe(query)

    # Convert month to datetime for better plotting
    df["month_dt"] = pd.to_datetime(df["month"] + "-01")

    # Plot for each platform
    for platform in self.platforms:
        platform_data = df[df["platform"] == platform]
        if not platform_data.empty:
            ax.plot(
                platform_data["month_dt"],
                platform_data["avg_duration"],
                marker="o",
                linewidth=2,
                label=platform,
                color=PLATFORM_COLORS.get(platform, "gray"),
            )

    # Format the x-axis to show months
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_title("Average Conversation Duration Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Duration (minutes)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Platform")


def analyze_model_usage(self):
    """Analyze and visualize model usage patterns"""
    print("\nAnalyzing model usage patterns...")

    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Model usage by platform
    ax1 = fig.add_subplot(gs[0, :])
    self._plot_model_usage_by_platform(ax1)

    # 2. Model usage over time
    ax2 = fig.add_subplot(gs[1, :])
    self._plot_model_usage_over_time(ax2)

    # 3. Average message length by model
    ax3 = fig.add_subplot(gs[2, 0])
    self._plot_avg_length_by_model(ax3)

    # 4. Model distribution by platform
    ax4 = fig.add_subplot(gs[2, 1])
    self._plot_model_distribution(ax4)

    # Add a title for the dashboard
    fig.suptitle("Model Usage Analysis", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and show
    plt.savefig(os.path.join(OUTPUT_DIR, "model_usage_analysis.png"))
    plt.show()


def _plot_model_usage_by_platform(self, ax):
    """Plot model usage by platform"""
    query = """
            SELECT * FROM model_usage
        """
    df = self.query_to_dataframe(query)

    # Sort by message count
    df = df.sort_values(by="message_count", ascending=False)

    # Plot
    platforms = df["platform"].unique()

    # Get top models for each platform
    platform_data = {}
    for platform in platforms:
        platform_df = df[df["platform"] == platform].head(
            5
        )  # Top 5 models per platform
        platform_data[platform] = platform_df

    # Set up positions for grouped bars
    bar_width = 0.35
    positions = {}

    for i, platform in enumerate(platforms):
        positions[platform] = np.arange(len(platform_data[platform])) + i * bar_width

    # Plot bars for each platform
    for platform in platforms:
        ax.bar(
            positions[platform],
            platform_data[platform]["message_count"],
            width=bar_width,
            label=platform,
            color=PLATFORM_COLORS.get(platform, "gray"),
        )

        # Add model labels
        for i, model in enumerate(platform_data[platform]["model"]):
            ax.text(
                positions[platform][i],
                0,
                model,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("Top Models by Platform")
    ax.set_xlabel("Model")
    ax.set_ylabel("Message Count")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Platform")

    # Remove x-tick labels as we have text labels
    ax.set_xticks([])


def _plot_model_usage_over_time(self, ax):
    """Plot model usage over time"""
    query = """
            SELECT 
                model,
                strftime('%Y-%m', created_at) AS month,
                COUNT(*) as message_count
            FROM messages
            WHERE model IS NOT NULL AND model != ''
            GROUP BY model, month
            ORDER BY month
        """
    df = self.query_to_dataframe(query)

    # Convert month to datetime for better plotting
    df["month_dt"] = pd.to_datetime(df["month"] + "-01")

    # Get top 5 models by total usage
    model_totals = (
        df.groupby("model")["message_count"].sum().sort_values(ascending=False)
    )
    top_models = model_totals.head(5).index.tolist()

    # Filter for top models
    df_top = df[df["model"].isin(top_models)]

    # Pivot the data for plotting
    pivot_df = df_top.pivot(index="month_dt", columns="model", values="message_count")

    # Fill missing values with 0
    pivot_df = pivot_df.fillna(0)

    # Plot
    pivot_df.plot(kind="line", marker="o", ax=ax)

    # Format the x-axis to show months
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_title("Model Usage Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Message Count")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model")


def _plot_avg_length_by_model(self, ax):
    """Plot average message length by model"""
    query = """
            SELECT 
                model,
                AVG(LENGTH(content)) as avg_length,
                COUNT(*) as message_count
            FROM messages
            WHERE model IS NOT NULL AND model != ''
            GROUP BY model
            HAVING message_count > 100  -- Filter out models with few messages
            ORDER BY avg_length DESC
        """
    df = self.query_to_dataframe(query)

    # Plot
    bars = ax.barh(
        df["model"], df["avg_length"], color=plt.cm.viridis(np.linspace(0, 1, len(df)))
    )

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 10,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
        )

    ax.set_title("Average Message Length by Model")
    ax.set_xlabel("Average Characters")
    ax.set_ylabel("Model")
    ax.grid(True, alpha=0.3, axis="x")

    # Sort from top to bottom for better readability
    ax.invert_yaxis()


def _plot_model_distribution(self, ax):
    """Plot model distribution by platform"""
    query = """
            SELECT * FROM model_usage
        """
    df = self.query_to_dataframe(query)

    # Calculate percentage within each platform
    total_by_platform = df.groupby("platform")["message_count"].sum().reset_index()
    df = df.merge(total_by_platform, on="platform", suffixes=("", "_total"))
    df["percentage"] = df["message_count"] / df["message_count_total"] * 100

    # Get top 3 models for each platform
    top_models_by_platform = {}
    for platform in df["platform"].unique():
        platform_df = df[df["platform"] == platform].sort_values(
            "percentage", ascending=False
        )
        top_models = platform_df.head(3)

        # Add "Other" category
        other_pct = 100 - top_models["percentage"].sum()
        if other_pct > 0:
            other_row = pd.DataFrame(
                {"platform": [platform], "model": ["Other"], "percentage": [other_pct]}
            )
            top_models = pd.concat([top_models, other_row])

        top_models_by_platform[platform] = top_models

    # Plot pie charts for each platform
    num_platforms = len(top_models_by_platform)
    cols = min(3, num_platforms)
    rows = (num_platforms + cols - 1) // cols

    # Set up a grid within this axis for multiple pie charts
    gridspec = GridSpec(
        rows, cols, wspace=0.3, hspace=0.3, left=0.1, right=0.9, top=0.9, bottom=0.1
    )

    # Remove the current axis
    ax.remove()

    for i, platform in enumerate(sorted(top_models_by_platform.keys())):
        row, col = i // cols, i % cols
        sub_ax = fig.add_subplot(gridspec[row, col])

        data = top_models_by_platform[platform]
        wedges, texts, autotexts = sub_ax.pie(
            data["percentage"],
            labels=data["model"],
            autopct="%1.1f%%",
            textprops={"fontsize": 9},
            colors=plt.cm.tab10(np.linspace(0, 1, len(data))),
        )

        # Customize text
        for text in texts:
            text.set_fontsize(8)
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_color("white")

        sub_ax.set_title(f"{platform} Models", fontsize=12)


def analyze_response_times(self):
    """Analyze and visualize response time patterns"""
    print("\nAnalyzing response time patterns...")

    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Average response time by platform
    ax1 = fig.add_subplot(gs[0, 0])
    self._plot_avg_response_time_by_platform(ax1)

    # 2. Response time distribution
    ax2 = fig.add_subplot(gs[0, 1])
    self._plot_response_time_distribution(ax2)

    # 3. Response time by hour of day
    ax3 = fig.add_subplot(gs[1, 0])
    self._plot_response_time_by_hour(ax3)

    # 4. Response time by day of week
    ax4 = fig.add_subplot(gs[1, 1])
    self._plot_response_time_by_dow(ax4)

    # 5. Response time over time (time series)
    ax5 = fig.add_subplot(gs[2, :])
    self._plot_response_time_over_time(ax5)

    # Add a title for the dashboard
    fig.suptitle("Response Time Analysis", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and show
    plt.savefig(os.path.join(OUTPUT_DIR, "response_time_analysis.png"))
    plt.show()


def _plot_avg_response_time_by_platform(self, ax):
    """Plot average response time by platform"""
    query = """
            SELECT
                c.platform,
                AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS avg_response_time_minutes,
                STDDEV((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS stddev_response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            GROUP BY c.platform
        """
    # Fallback for SQLite which doesn't have STDDEV
    query = """
            SELECT
                c.platform,
                AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS avg_response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            GROUP BY c.platform
        """
    df = self.query_to_dataframe(query)

    # Plot
    bars = ax.bar(
        df["platform"],
        df["avg_response_time_minutes"],
        color=[PLATFORM_COLORS.get(p, "gray") for p in df["platform"]],
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    ax.set_title("Average Response Time by Platform")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Average Response Time (minutes)")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_response_time_distribution(self, ax):
    """Plot response time distribution"""
    query = """
            SELECT
                c.platform,
                (julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60 AS response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            WHERE (julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60 < 10  -- Filter extreme outliers
        """
    df = self.query_to_dataframe(query)

    # Plot histograms
    for platform in self.platforms:
        platform_data = df[df["platform"] == platform]
        if not platform_data.empty:
            ax.hist(
                platform_data["response_time_minutes"],
                bins=30,
                alpha=0.6,
                label=platform,
                color=PLATFORM_COLORS.get(platform, "gray"),
            )

    ax.set_title("Response Time Distribution")
    ax.set_xlabel("Response Time (minutes)")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_response_time_by_hour(self, ax):
    """Plot response time by hour of day"""
    query = """
            SELECT
                c.platform,
                strftime('%H', mp.human_timestamp) AS hour,
                AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS avg_response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            GROUP BY c.platform, hour
            ORDER BY hour
        """
    df = self.query_to_dataframe(query)

    # Convert hour to numeric
    df["hour"] = pd.to_numeric(df["hour"])

    # Pivot the data for plotting
    pivot_df = df.pivot(
        index="hour", columns="platform", values="avg_response_time_minutes"
    )

    # Fill missing values with 0
    pivot_df = pivot_df.fillna(0)

    # Plot
    pivot_df.plot(
        kind="line",
        marker="o",
        ax=ax,
        color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
    )

    ax.set_title("Average Response Time by Hour of Day")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Response Time (minutes)")

    # Set x-axis ticks for each hour
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.grid(True, alpha=0.3)
    ax.legend(title="Platform")


def _plot_response_time_by_dow(self, ax):
    """Plot response time by day of week"""
    query = """
            SELECT
                c.platform,
                strftime('%w', mp.human_timestamp) AS day_of_week,
                AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS avg_response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            GROUP BY c.platform, day_of_week
            ORDER BY day_of_week
        """
    df = self.query_to_dataframe(query)

    # Convert day_of_week to numeric and add day name
    df["day_of_week"] = pd.to_numeric(df["day_of_week"])
    days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    df["day_name"] = df["day_of_week"].apply(lambda x: days[int(x)])

    # Pivot the data for plotting
    pivot_df = df.pivot(
        index="day_name", columns="platform", values="avg_response_time_minutes"
    )

    # Reorder days to start with Monday
    day_order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pivot_df = pivot_df.reindex(day_order)

    # Plot
    pivot_df.plot(
        kind="bar",
        ax=ax,
        color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
    )

    ax.set_title("Average Response Time by Day of Week")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Average Response Time (minutes)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Platform")


def _plot_response_time_over_time(self, ax):
    """Plot response time trends over time"""
    query = """
            SELECT
                c.platform,
                strftime('%Y-%m', mp.human_timestamp) AS month,
                AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60) AS avg_response_time_minutes
            FROM message_pairs mp
            JOIN conversations c ON mp.conversation_id = c.id
            GROUP BY c.platform, month
            ORDER BY month
        """
    df = self.query_to_dataframe(query)

    # Convert month to datetime for better plotting
    df["month_dt"] = pd.to_datetime(df["month"] + "-01")

    # Plot for each platform
    for platform in self.platforms:
        platform_data = df[df["platform"] == platform]
        if not platform_data.empty:
            ax.plot(
                platform_data["month_dt"],
                platform_data["avg_response_time_minutes"],
                marker="o",
                linewidth=2,
                label=platform,
                color=PLATFORM_COLORS.get(platform, "gray"),
            )

    # Format the x-axis to show months
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    ax.set_title("Average Response Time Over Time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average Response Time (minutes)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Platform")


def custom_query_visualization(self):
    """Allow the user to enter a custom SQL query and visualize the results"""
    print("\nCustom Query Visualization")
    print(
        "Enter a SQL query to visualize. The query should return data suitable for plotting."
    )
    print(
        "Example: SELECT platform, COUNT(*) as count FROM conversations GROUP BY platform"
    )

    query = input("\nEnter your SQL query: ")

    try:
        df = self.query_to_dataframe(query)

        if df.empty:
            print("Query returned no data.")
            return

        print("\nQuery returned data:")
        print(df.head())

        # Determine the type of visualization based on the data
        print("\nSelect visualization type:")
        print("1. Bar Chart")
        print("2. Line Chart")
        print("3. Pie Chart")
        print("4. Scatter Plot")
        print("5. Table (no visualization)")

        vis_type = input("Enter your choice (1-5): ")

        plt.figure(figsize=(12, 8))

        if vis_type == "1":  # Bar Chart
            x_col = input("Enter column name for x-axis: ")
            y_col = input("Enter column name for y-axis: ")

            if x_col not in df.columns or y_col not in df.columns:
                print("Invalid column names.")
                return

            df.plot(kind="bar", x=x_col, y=y_col)
            plt.title(f"{y_col} by {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)

        elif vis_type == "2":  # Line Chart
            x_col = input("Enter column name for x-axis: ")
            y_col = input("Enter column name for y-axis: ")

            if x_col not in df.columns or y_col not in df.columns:
                print("Invalid column names.")
                return

            df.plot(kind="line", x=x_col, y=y_col, marker="o")
            plt.title(f"{y_col} by {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)

        elif vis_type == "3":  # Pie Chart
            value_col = input("Enter column name for values: ")
            label_col = input("Enter column name for labels: ")

            if value_col not in df.columns or label_col not in df.columns:
                print("Invalid column names.")
                return

            df.plot(kind="pie", y=value_col, labels=df[label_col], autopct="%1.1f%%")
            plt.title(f"Distribution of {value_col}")
            plt.ylabel("")

        elif vis_type == "4":  # Scatter Plot
            x_col = input("Enter column name for x-axis: ")
            y_col = input("Enter column name for y-axis: ")

            if x_col not in df.columns or y_col not in df.columns:
                print("Invalid column names.")
                return

            df.plot(kind="scatter", x=x_col, y=y_col)
            plt.title(f"{y_col} vs {x_col}")
            plt.xlabel(x_col)
            plt.ylabel(y_col)

        elif vis_type == "5":  # Table
            print("\nFull results:")
            pd.set_option("display.max_rows", None)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.width", None)
            print(df)
            pd.reset_option("display.max_rows")
            pd.reset_option("display.max_columns")
            pd.reset_option("display.width")

            # Export to CSV if desired
            export = input("Export to CSV? (y/n): ")
            if export.lower() == "y":
                filename = (
                    input("Enter filename (default: query_result.csv): ")
                    or "query_result.csv"
                )
                df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
                print(f"Exported to {os.path.join(OUTPUT_DIR, filename)}")

            return

        else:
            print("Invalid choice.")
            return

        plt.tight_layout()

        # Save if desired
        save = input("Save visualization? (y/n): ")
        if save.lower() == "y":
            filename = (
                input("Enter filename (default: custom_visualization.png): ")
                or "custom_visualization.png"
            )
            plt.savefig(os.path.join(OUTPUT_DIR, filename))
            print(f"Saved to {os.path.join(OUTPUT_DIR, filename)}")

        plt.show()

    except Exception as e:
        print(f"Error: {e}")


def topic_analysis(self):
    """Analyze conversation topics based on content"""
    print("\nPerforming topic analysis...")

    # Define topic keywords
    topics = {
        "AI/ML": [
            "machine learning",
            "deep learning",
            "neural network",
            "ai",
            "artificial intelligence",
            "model",
            "training",
            "algorithm",
            "prediction",
            "classification",
        ],
        "Programming": [
            "code",
            "programming",
            "python",
            "javascript",
            "java",
            "function",
            "class",
            "variable",
            "object",
            "method",
            "compiler",
            "developer",
        ],
        "Education": [
            "learn",
            "teach",
            "education",
            "student",
            "teacher",
            "school",
            "university",
            "course",
            "study",
            "academic",
            "research",
            "knowledge",
        ],
        "Business": [
            "business",
            "company",
            "startup",
            "product",
            "market",
            "customer",
            "revenue",
            "profit",
            "strategy",
            "management",
            "entrepreneur",
        ],
        "Data": [
            "data",
            "database",
            "sql",
            "analytics",
            "visualization",
            "statistic",
            "analysis",
            "report",
            "chart",
            "metric",
            "insight",
        ],
        "Web": [
            "web",
            "website",
            "html",
            "css",
            "frontend",
            "backend",
            "server",
            "client",
            "browser",
            "api",
            "request",
            "response",
            "http",
        ],
        "Mobile": [
            "mobile",
            "app",
            "ios",
            "android",
            "smartphone",
            "tablet",
            "application",
            "notification",
            "interface",
            "touch",
            "gesture",
        ],
        "Creative": [
            "creative",
            "design",
            "art",
            "write",
            "story",
            "content",
            "creative writing",
            "poetry",
            "novel",
            "fiction",
            "character",
            "plot",
        ],
    }

    # Create SQL query with LIKE conditions for each topic
    topic_conditions = []
    for topic, keywords in topics.items():
        like_conditions = []
        for keyword in keywords[
            :5
        ]:  # Limit to first 5 keywords to avoid query size limits
            like_conditions.append(f"m.content LIKE '%{keyword}%'")
        topic_conditions.append(
            f"SUM(CASE WHEN {' OR '.join(like_conditions)} THEN 1 ELSE 0 END) AS {topic.replace('/', '_')}"
        )

    query = f"""
            SELECT
                c.platform,
                {', '.join(topic_conditions)},
                COUNT(*) as total_messages
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            GROUP BY c.platform
        """

    try:
        df = self.query_to_dataframe(query)

        # Calculate percentages
        for topic in [t.replace("/", "_") for t in topics.keys()]:
            df[f"{topic}_pct"] = df[topic] / df["total_messages"] * 100

        # Create a figure with subplots
        fig = plt.figure(figsize=(16, 16))
        gs = GridSpec(3, 2, figure=fig)

        # 1. Topic distribution by platform
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_topic_distribution(ax1, df, topics)

        # 2. Top topics by platform
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_top_topics(ax2, df, topics)

        # 3. Topic comparison between platforms
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_topic_comparison(ax3, df, topics)

        # 4. Topic correlations - find commonly co-occurring topics by analyzing conversations
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_topic_correlations(ax4, topics)

        # Add a title for the dashboard
        fig.suptitle("Topic Analysis", fontsize=20, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save and show
        plt.savefig(os.path.join(OUTPUT_DIR, "topic_analysis.png"))
        plt.show()

    except Exception as e:
        print(f"Error in topic analysis: {e}")


def _plot_topic_distribution(self, ax, df, topics):
    """Plot the distribution of topics across platforms"""
    # Create a DataFrame for plotting
    plot_data = []

    for platform in df["platform"]:
        for topic in [t.replace("/", "_") for t in topics.keys()]:
            plot_data.append(
                {
                    "Platform": platform,
                    "Topic": topic.replace("_", "/"),
                    "Percentage": df[df["platform"] == platform][f"{topic}_pct"].values[
                        0
                    ],
                }
            )

    plot_df = pd.DataFrame(plot_data)

    # Pivot for plotting
    pivot_df = plot_df.pivot(index="Topic", columns="Platform", values="Percentage")

    # Plot
    pivot_df.plot(
        kind="bar",
        ax=ax,
        color=[PLATFORM_COLORS.get(p, "gray") for p in pivot_df.columns],
    )

    ax.set_title("Topic Distribution by Platform")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Percentage of Messages")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(title="Platform")

    # Add percentage labels
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=8)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")


def _plot_top_topics(self, ax, df, topics):
    """Plot the top topics for each platform"""
    # Create a subplot for each platform
    num_platforms = len(df)
    if num_platforms > 1:
        fig, axes = plt.subplots(1, num_platforms, figsize=(12, 5))

        for i, (_, row) in enumerate(df.iterrows()):
            platform = row["platform"]
            topic_data = {}

            for topic in [t.replace("/", "_") for t in topics.keys()]:
                topic_data[topic.replace("_", "/")] = row[f"{topic}_pct"]

            # Sort topics by percentage
            topic_data = {
                k: v
                for k, v in sorted(
                    topic_data.items(), key=lambda item: item[1], reverse=True
                )
            }

            # Plot
            axes[i].barh(
                list(topic_data.keys()),
                list(topic_data.values()),
                color=PLATFORM_COLORS.get(platform, "gray"),
            )

            axes[i].set_title(f"Top Topics for {platform}")
            axes[i].set_xlabel("Percentage")

            # Add percentage labels
            for j, v in enumerate(topic_data.values()):
                axes[i].text(v + 0.5, j, f"{v:.1f}%", va="center")

            # Reverse y-axis to have highest percentage on top
            axes[i].invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "top_topics_by_platform.png"))

        # Remove this axis as we created a new figure
        ax.remove()
        return

    # If only one platform, plot directly on the given axis
    platform = df["platform"].iloc[0]
    topic_data = {}

    for topic in [t.replace("/", "_") for t in topics.keys()]:
        topic_data[topic.replace("_", "/")] = df[f"{topic}_pct"].iloc[0]

    # Sort topics by percentage
    topic_data = {
        k: v
        for k, v in sorted(topic_data.items(), key=lambda item: item[1], reverse=True)
    }

    # Plot
    ax.barh(
        list(topic_data.keys()),
        list(topic_data.values()),
        color=PLATFORM_COLORS.get(platform, "gray"),
    )

    ax.set_title(f"Top Topics for {platform}")
    ax.set_xlabel("Percentage")

    # Add percentage labels
    for i, v in enumerate(topic_data.values()):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")

    # Reverse y-axis to have highest percentage on top
    ax.invert_yaxis()


def _plot_topic_comparison(self, ax, df, topics):
    """Plot a direct comparison of topic percentages between platforms"""
    if len(df) < 2:
        ax.text(
            0.5,
            0.5,
            "Need at least 2 platforms for comparison",
            ha="center",
            va="center",
            fontsize=12,
        )
        return

    # Create data for radar chart
    topic_names = [t for t in topics.keys()]
    num_topics = len(topic_names)

    # Set up the angles for each topic
    angles = np.linspace(0, 2 * np.pi, num_topics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Set up plot
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_thetagrids(np.degrees(angles[:-1]), topic_names)

    # Draw y-axis labels
    ax.set_rlabel_position(0)
    ax.set_rticks([10, 20, 30, 40, 50])
    ax.set_rlim(0, 50)

    # Plot each platform
    for _, row in df.iterrows():
        platform = row["platform"]

        # Get topic percentages
        values = [row[f'{t.replace("/", "_")}_pct'] for t in topic_names]
        values += values[:1]  # Close the loop

        # Plot
        ax.plot(
            angles,
            values,
            linewidth=2,
            label=platform,
            color=PLATFORM_COLORS.get(platform, "gray"),
        )
        ax.fill(angles, values, alpha=0.25, color=PLATFORM_COLORS.get(platform, "gray"))

    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    ax.set_title("Topic Comparison Between Platforms")


def _plot_topic_correlations(self, ax, topics):
    """Plot correlations between topics"""
    # Query to get topic co-occurrence in conversations
    topic_pairs = []

    for topic1 in topics:
        for topic2 in topics:
            if topic1 < topic2:  # Avoid duplicates and self-pairs
                # Create SQL conditions
                topic1_conditions = " OR ".join(
                    [f"m.content LIKE '%{kw}%'" for kw in topics[topic1][:5]]
                )
                topic2_conditions = " OR ".join(
                    [f"m.content LIKE '%{kw}%'" for kw in topics[topic2][:5]]
                )

                query = f"""
                        SELECT 
                            COUNT(DISTINCT m.conversation_id) as count
                        FROM messages m
                        WHERE 
                            (SELECT COUNT(*) FROM messages m2 
                             WHERE m2.conversation_id = m.conversation_id
                             AND ({topic1_conditions})) > 0
                        AND
                            (SELECT COUNT(*) FROM messages m3 
                             WHERE m3.conversation_id = m.conversation_id
                             AND ({topic2_conditions})) > 0
                    """

                try:
                    result = self.query_to_dataframe(query)
                    count = result["count"].iloc[0]

                    if count > 0:
                        topic_pairs.append((topic1, topic2, count))
                except Exception as e:
                    print(f"Error in topic correlation: {e}")

    # Create correlation matrix
    correlation_matrix = np.zeros((len(topics), len(topics)))
    topic_list = list(topics.keys())

    # Fill the matrix
    for topic1, topic2, count in topic_pairs:
        i = topic_list.index(topic1)
        j = topic_list.index(topic2)
        correlation_matrix[i, j] = count
        correlation_matrix[j, i] = count  # Mirror

    # Normalize
    max_count = correlation_matrix.max()
    if max_count > 0:
        correlation_matrix = correlation_matrix / max_count

    # Plot heatmap
    im = ax.imshow(correlation_matrix, cmap="YlOrRd")

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Normalized Co-occurrence")

    # Set labels
    ax.set_xticks(range(len(topics)))
    ax.set_yticks(range(len(topics)))
    ax.set_xticklabels(topics.keys(), rotation=45, ha="right")
    ax.set_yticklabels(topics.keys())

    # Add annotations
    for i in range(len(topics)):
        for j in range(len(topics)):
            text = ax.text(
                j,
                i,
                f"{correlation_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if correlation_matrix[i, j] < 0.5 else "white",
            )

    ax.set_title("Topic Co-occurrence Correlation")


def user_engagement_patterns(self):
    """Analyze user engagement patterns"""
    print("\nAnalyzing user engagement patterns...")

    # Create a figure with subplots
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(3, 2, figure=fig)

    # 1. Conversation frequency distribution
    ax1 = fig.add_subplot(gs[0, 0])
    self._plot_conversation_freq_distribution(ax1)

    # 2. Message count per conversation
    ax2 = fig.add_subplot(gs[0, 1])
    self._plot_message_per_conversation(ax2)

    # the "conversation_summary" view gives us duration, message counts, etc.

    # 3. Engagement duration
    ax3 = fig.add_subplot(gs[1, 0])
    self._plot_engagement_duration(ax3)

    # 4. User retention - conversations over time
    ax4 = fig.add_subplot(gs[1, 1])
    self._plot_user_retention(ax4)

    # 5. Time of day engagement pattern
    ax5 = fig.add_subplot(gs[2, :])
    self._plot_time_of_day_engagement(ax5)

    # Add a title for the dashboard
    fig.suptitle("User Engagement Analysis", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and show
    plt.savefig(os.path.join(OUTPUT_DIR, "user_engagement_analysis.png"))
    plt.show()


def _plot_conversation_freq_distribution(self, ax):
    """Plot conversation frequency distribution by account"""
    query = """
            SELECT 
                platform,
                account_id, 
                COUNT(*) as conversation_count
            FROM conversations
            WHERE account_id IS NOT NULL
            GROUP BY platform, account_id
        """
    df = self.query_to_dataframe(query)

    # Plot histograms
    for platform in self.platforms:
        platform_data = df[df["platform"] == platform]
        if not platform_data.empty:
            ax.hist(
                platform_data["conversation_count"],
                bins=20,
                alpha=0.6,
                label=platform,
                color=PLATFORM_COLORS.get(platform, "gray"),
            )

    ax.set_title("Conversation Count per User")
    ax.set_xlabel("Number of Conversations")
    ax.set_ylabel("Number of Users")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Log scale for y-axis if data is highly skewed
    if df["conversation_count"].max() > 10 * df["conversation_count"].median():
        ax.set_yscale("log")
        ax.set_ylabel("Number of Users (log scale)")


def _plot_message_per_conversation(self, ax):
    """Plot message count per conversation"""
    query = """
            SELECT 
                c.platform,
                COUNT(m.id) as message_count
            FROM conversations c
            JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id, c.platform
        """
    df = self.query_to_dataframe(query)

    # Plot histograms
    for platform in self.platforms:
        platform_data = df[df["platform"] == platform]
        if not platform_data.empty:
            ax.hist(
                platform_data["message_count"],
                bins=20,
                alpha=0.6,
                label=platform,
                color=PLATFORM_COLORS.get(platform, "gray"),
                range=(0, platform_data["message_count"].quantile(0.95)),
            )  # Exclude outliers

    ax.set_title("Messages per Conversation")
    ax.set_xlabel("Number of Messages")
    ax.set_ylabel("Number of Conversations")
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_engagement_duration(self, ax):
    """Plot conversation duration distribution"""
    query = """
            SELECT 
                platform,
                conversation_duration_minutes
            FROM conversation_summary
            WHERE conversation_duration_minutes > 0 
              AND conversation_duration_minutes < 500  -- Filter outliers
        """
    df = self.query_to_dataframe(query)

    # Create duration buckets
    df["duration_bucket"] = pd.cut(
        df["conversation_duration_minutes"],
        bins=[0, 1, 5, 15, 30, 60, 120, 240, 500],
        labels=[
            "<1 min",
            "1-5 min",
            "5-15 min",
            "15-30 min",
            "30-60 min",
            "1-2 hrs",
            "2-4 hrs",
            "4+ hrs",
        ],
    )

    # Group by platform and duration bucket
    grouped = (
        df.groupby(["platform", "duration_bucket"]).size().reset_index(name="count")
    )

    # Pivot for plotting
    pivot_df = grouped.pivot(
        index="duration_bucket", columns="platform", values="count"
    )

    # Fill missing values with 0
    pivot_df = pivot_df.fillna(0)
