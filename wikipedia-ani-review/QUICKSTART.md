# Quick Start Guide

This guide will help you get started with the Wikipedia ANI Review system.

## Prerequisites

- Python 3.8 or higher
- Internet connection (for scraping Wikipedia)
- OpenAI API key (optional, for LLM analysis)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CrazyDubya/WikipediaANIReview.git
   cd WikipediaANIReview
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system (optional but recommended):**
   ```bash
   cp config.example.env .env
   ```
   
   Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

## Basic Usage

### Step 1: Initialize the Database

```bash
python ani_review.py init
```

This creates the SQLite database schema.

### Step 2: Scrape Wikipedia ANI Data

**Option A: Quick test (current page only)**
```bash
python ani_review.py scrape
```

**Option B: Full scrape (includes archives - takes longer)**
```bash
python ani_review.py scrape --archives --archive-limit 50
```

Note: The scraper respects Wikipedia's rate limits with a 1-second delay between requests.

### Step 3: Analyze Threads

```bash
python ani_review.py analyze
```

This uses LLM (OpenAI GPT) to categorize incidents and identify requested outcomes. If you don't have an API key, it will use keyword-based mock analysis.

### Step 4: Generate Report

```bash
python ani_review.py report
```

This displays:
- Database statistics
- Top thread creators
- Top responders
- Category distribution
- Requested outcomes
- Potentially overactive users

### Step 5: Look Up Specific Users

```bash
python ani_review.py user "Username"
```

## Advanced Usage

### Custom Thresholds

Identify overactive users with custom thresholds:

```bash
python ani_review.py report --thread-threshold 5 --post-threshold 20
```

### Limited Analysis

Analyze only a specific number of threads:

```bash
python ani_review.py analyze --limit 100
```

### Custom Database Location

Use a different database file:

```bash
python ani_review.py init --database-url sqlite:///custom_db.db
python ani_review.py scrape --database-url sqlite:///custom_db.db
```

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Initialize
python ani_review.py init

# 2. Scrape current ANI page
python ani_review.py scrape

# 3. Scrape 10 archive pages (this takes about 10-15 minutes)
python ani_review.py scrape --archives --archive-limit 10

# 4. Analyze all collected threads
python ani_review.py analyze

# 5. Generate comprehensive report
python ani_review.py report

# 6. Look up a specific user
python ani_review.py user "AdminUsername"

# 7. Generate report with lower thresholds to see more results
python ani_review.py report --thread-threshold 3 --post-threshold 10
```

## Understanding the Results

### Thread Creators
Users who create many ANI threads may be:
- Active patrollers legitimately reporting issues
- Users who frequently encounter problematic editors
- Potentially overusing the noticeboard

### Responders
Users who respond frequently may be:
- Administrators handling reports
- Experienced editors providing input
- Community members engaged in discussions

### Categories
The system categorizes incidents into types like:
- Edit warring
- Personal attacks
- Vandalism
- Sockpuppetry
- etc.

### Overactive User Detection
The "overactivity score" is calculated as:
```
score = (threads_created × 2.0) + (posts_made × 0.5)
```

Higher scores indicate users who are very active in ANI discussions. This doesn't necessarily indicate abuse, but users with very high scores may warrant closer examination.

## Troubleshooting

### "No module named 'dotenv'"
Run: `pip install -r requirements.txt`

### "No OpenAI API key found"
The system will work with keyword-based analysis. To use LLM analysis, add your API key to `.env`.

### "Failed to resolve 'en.wikipedia.org'"
Check your internet connection or firewall settings.

### Database is empty after scraping
Check your network connection to Wikipedia. The scraper requires internet access.

## Next Steps

- Examine the database directly with SQLite tools
- Modify analysis categories in `analyzer.py`
- Adjust scraping behavior in `scraper.py`
- Create custom queries using the database models

## Getting Help

For issues or questions:
1. Check the main README.md
2. Review the code comments
3. Open an issue on GitHub
