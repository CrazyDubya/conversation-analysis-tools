# Wikipedia ANI Review

A comprehensive system for analyzing Wikipedia Administrator Noticeboard (ANI) discussions to identify user activity patterns, categorize incidents, detect voting patterns, and analyze outcomes.

## Overview

This tool scrapes, stores, and analyzes Wikipedia ANI discussions to:
- **Track thread creation and filing patterns** by users
- **Monitor response patterns** and comment structure in discussions
- **Categorize incidents** using LLM-powered analysis
- **Extract targets and evidence** from complaints (diffs, policy violations, etc.)
- **Parse closure outcomes** and admin decisions
- **Detect and analyze voting patterns** in proposals and consensus discussions
- **Identify overactive users** who may be weaponizing Wikipedia rules
- **Generate comprehensive statistical reports** on ANI activity
- **Track filer success rates** and outcome patterns

## Features

### Data Collection
- 🔍 **MediaWiki API Integration**: Reliable structured data extraction (recommended)
- ⚡ **Async/Await Support**: 3-10x faster concurrent scraping with rate limiting
- 🌐 **HTML Scraper Fallback**: Backup scraping method
- 📁 **Archive Support**: Scrapes current and historical ANI archives
- 💾 **Raw Wikitext Storage**: Preserves original text for re-parsing

### Analysis & Tracking
- 🤖 **LLM Analysis**: Uses OpenAI GPT to categorize incidents and outcomes
- 🚀 **Concurrent Analysis**: Async LLM calls for 3-5x faster processing
- 🎯 **Target Identification**: Extracts reported users from {{userlinks}} and {{user}} templates
- 🔗 **Evidence Extraction**: Captures diffs, policy shortcuts, noticeboard references
- 📋 **Closure Parsing**: Detects {{atop}} templates and outcomes (blocked/warned/no-action/etc.)
- 🗳️ **Vote Detection**: Identifies and analyzes voting structures (Support/Oppose/Neutral)
- 📊 **Statistical Analysis**: Identifies patterns and overactive users
- 🖥️ **CLI Interface**: Easy-to-use command-line tools (sync + async)
- 📈 **Performance Profiling**: Benchmarking and NO GIL opportunity analysis

### Database Schema
- **Enhanced Thread Model**: 20+ fields including filer, targets, evidence, outcomes, closures
- **Post Structure Tracking**: Indent depth, comment kind, raw wikitext
- **Vote Records**: Complete voting data with stance, strength modifiers, timestamps
- **Filing Outcomes**: Success rate tracking and outcome analysis
- **User Statistics**: Activity patterns, voting behavior, filing success rates

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/CrazyDubya/WikipediaANIReview.git
cd WikipediaANIReview
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment (optional):
```bash
cp config.example.env .env
# Edit .env with your settings (especially OpenAI API key for LLM analysis)
```

## Usage

### Initialize Database

First, create the database schema:

```bash
python ani_review.py init
```

### Scrape ANI Data

**Recommended: Use Async API scraper (3-10x faster)**

Scrape the current ANI page using async:

```bash
python ani_review_async.py async-scrape
```

Scrape current page and archives concurrently:

```bash
python ani_review_async.py async-scrape --archives --archive-limit 20
```

Custom rate limiting:

```bash
python ani_review_async.py async-scrape \
    --max-concurrent 5 \
    --request-delay 1.0 \
    --archives
```

**Alternative: Use synchronous API scraper**

```bash
python ani_review.py scrape --use-api
```

Scrape current page and archives:

```bash
python ani_review.py scrape --use-api --archives --archive-limit 20
```

**Alternative: Use HTML scraping**

```bash
python ani_review.py scrape --use-html
```

**What gets extracted:**
- Thread titles and section anchors
- Filer (thread creator) and timestamp
- All comments with indent depth and structure
- Reported users from {{userlinks}} and {{user}} templates
- Diff links, policy shortcuts (WP:XYZ), noticeboard references
- Closure outcomes from {{atop}} templates
- Votes in consensus discussions (Support/Oppose/Neutral)
- Raw wikitext for future re-parsing

### Analyze Threads

**Recommended: Use async analyzer (3-5x faster)**

Run concurrent LLM analysis on collected threads:

```bash
python ani_review_async.py async-analyze
```

Analyze with custom concurrency:

```bash
python ani_review_async.py async-analyze --limit 50 --max-concurrent 3
```

**Alternative: Use synchronous analyzer**

Run LLM analysis on collected threads:

```bash
python ani_review.py analyze
```

Analyze a limited number of threads:

```bash
python ani_review.py analyze --limit 50
```

**Note**: Analysis requires an OpenAI API key. Without one, the system will use keyword-based mock analysis.

### Generate Reports

View comprehensive analysis report:

```bash
python ani_review.py report
```

Show threads with identified targets:

```bash
python ani_review.py report --show-targets
```

Customize overactive user thresholds:

```bash
python ani_review.py report --thread-threshold 15 --post-threshold 100
```

**Report includes:**
- Database statistics (threads, users, analyzed threads)
- Top thread creators and responders
- Category and outcome distributions
- Closure outcome statistics
- Routing suggestions (misfiled threads)
- Overactive user detection
- Threads with identified targets

### View Voting Threads

See threads with voting/consensus data:

```bash
python ani_review.py votes --limit 30
```

### User Lookup

Get detailed information about a specific user (including voting stats):

```bash
python ani_review.py user "Username"
```

## Database Schema

The system uses SQLite with the following tables:

### Core Tables
- **users**: Wikipedia user accounts with activity statistics
- **threads**: ANI discussion threads with extensive metadata (20+ fields)
  - Thread identification: title, URL, section_id, thread_anchor
  - Filing info: filer_user, filer_timestamp
  - Targets: JSON array of reported users
  - Evidence: diffs, policy_shortcuts, noticeboard_refs, content_area
  - Outcomes: outcome_status, outcome_text, closing_admin, closure_timestamp
  - Categorization: category_routing_suggested
  - Raw data: raw_wikitext for re-parsing
- **posts**: Individual posts within threads
  - Structure: indent_depth, comment_kind
  - Content: content, raw_wikitext
  - Metadata: author, posted_at, position
- **thread_analyses**: LLM-generated categorizations
- **user_activities**: Activity tracking metrics

### Advanced Tables
- **votes**: Voting records in consensus discussions
  - Vote details: proposal_id, voter, vote_timestamp
  - Stance: support/oppose/neutral/comment
  - Analysis: strength_modifier, matched_final
- **filing_outcomes**: Success rate tracking
  - Actions: requested_action, actual_action
  - Timing: time_to_close
  - Success: is_successful flag

## Analysis Categories

### Incident Types
- Edit warring
- Personal attacks
- Disruptive editing
- Copyright violations
- Sockpuppetry
- Vandalism
- Harassment
- NPOV violations
- Canvassing
- Gaming the system
- Other conduct issues

### Requested Outcomes
- Block
- Topic ban
- Interaction ban
- Warning
- Arbitration
- Mediation
- Community discussion
- No action
- Other

### Topic Areas
- Biography
- Politics
- Religion
- Science
- History
- Entertainment
- Sports
- Geography
- Technology
- Current events
- Wikipedia policy
- Other

## Performance and Benchmarking

### Async Performance Benefits

The async implementation provides significant speedup for I/O-bound operations:

- **Scraping:** 3-10x faster with concurrent requests
- **Analysis:** 3-5x faster with concurrent LLM calls
- **Archives:** 5-8x faster with parallel processing

### Benchmark Commands

**Compare sync vs async performance:**
```bash
python ani_review_async.py benchmark --num-threads 20
```

**Test rate limiting behavior:**
```bash
python ani_review_async.py test-rate-limiting --max-concurrent 5 --request-delay 1.0
```

**Generate NO GIL opportunity report:**
```bash
python ani_review_async.py nogil-report > NOGIL_ANALYSIS.txt
```

### Rate Limiting Guidelines

Safe rate limiting configurations:
- **Default:** 5 concurrent requests, 1.0s delay (recommended)
- **Conservative:** 3 concurrent, 1.5s delay (for shared IPs)
- **Aggressive:** 10 concurrent, 0.5s delay (use with caution)

The system uses both semaphore (concurrent limit) and delay (rate limit) to ensure Wikipedia servers are not overwhelmed.

### NO GIL (PEP 703) Opportunities

The codebase has been analyzed for CPU-bound sections that would benefit from NO GIL:

**High benefit sections:**
- Wikitext parsing (regex-heavy): 1.5-2x speedup expected
- Vote parsing: 1.3-1.8x speedup expected

**I/O-bound sections (async sufficient):**
- HTTP requests: async provides 5-10x speedup
- LLM API calls: async provides 3-5x speedup

**Total expected performance:**
- Async alone: 3-10x faster than sync
- Async + NO GIL: 4-15x faster than sync (when NO GIL available)

See `ASYNC_REFACTOR.md` for detailed analysis and implementation strategies.

## Configuration

Configure the system using environment variables (`.env` file):

```bash
# Database
DATABASE_URL=sqlite:///ani_review.db

# OpenAI API (for LLM analysis)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Scraping
USER_AGENT=WikipediaANIReview/1.0 (Research Project)
REQUEST_DELAY=1.0

# Analysis thresholds
OVERACTIVE_THREAD_THRESHOLD=10
OVERACTIVE_RESPONSE_THRESHOLD=50
```

## Overactive User Detection

The system identifies potentially overactive users using:

1. **Thread Creation Frequency**: Users who create excessive ANI threads
2. **Response Volume**: Users who post excessively in discussions
3. **Overactivity Score**: Calculated as `(threads_created * 2) + (posts_made * 0.5)`

Users exceeding configurable thresholds are flagged for review to determine if they may be weaponizing Wikipedia rules.

## Example Workflow

**Standard workflow (sync):**
```bash
# 1. Initialize the database (creates enhanced schema)
python ani_review.py init

# 2. Scrape current ANI page using MediaWiki API
python ani_review.py scrape --use-api

# 3. Scrape some archives (this can take time)
python ani_review.py scrape --use-api --archives --archive-limit 10

# 4. Analyze threads with LLM
python ani_review.py analyze

# 5. Generate comprehensive report
python ani_review.py report --show-targets

# 6. View threads with voting
python ani_review.py votes

# 7. Look up specific user (with voting stats)
python ani_review.py user "ExampleUser"
```

**High-performance workflow (async):**
```bash
# 1. Initialize the database
python ani_review.py init

# 2. Async scrape (3-10x faster)
python ani_review_async.py async-scrape --archives --archive-limit 10

# 3. Async analyze (3-5x faster)
python ani_review_async.py async-analyze --limit 100

# 4. Benchmark performance
python ani_review_async.py benchmark --num-threads 20 --test-analysis

# 5. Generate reports (use existing commands)
python ani_review.py report --show-targets
python ani_review.py votes

# 6. Analyze NO GIL opportunities
python ani_review_async.py nogil-report > NOGIL_ANALYSIS.txt
```

## Key Features

### MediaWiki API Integration
- **Reliable**: Uses Wikipedia's official API for structured data
- **Complete**: Extracts section metadata, timestamps, and wikitext
- **Efficient**: Respects rate limits with configurable delays
- **Concurrent**: Async implementation for 3-10x speedup
- **Fallback**: HTML scraper available as backup

### Comprehensive Data Extraction
- **Thread Metadata**: Title, anchor, filer, timestamp, section ID
- **Targets**: Reported users from templates ({{userlinks}}, {{user}})
- **Evidence**: Diff links, policy shortcuts (WP:XYZ), noticeboard references
- **Structure**: Indent depth, comment kinds, reply chains
- **Outcomes**: Closure templates ({{atop}}, {{atopg}}), admin decisions
- **Votes**: Support/Oppose/Neutral with strength modifiers
- **Raw Data**: Original wikitext preserved for re-parsing

### Advanced Analytics
- **Filer Success Rates**: Track what happens to reports
- **Voting Patterns**: Analyze consensus discussions
- **Routing Detection**: Identify misfiled threads (should go to BLP, SPI, etc.)
- **Overactivity Detection**: Flag potential rule weaponization
- **Outcome Statistics**: What happens in ANI threads
- **Performance Profiling**: Benchmark sync vs async, identify NO GIL opportunities

## Ethical Considerations

This tool is designed for research and transparency purposes. When using this system:

- Respect Wikipedia's rate limits and robots.txt
- Use appropriate user agent identification
- Don't harass or target individual users
- Consider privacy implications when sharing findings
- Remember that high activity doesn't necessarily indicate abuse

## Documentation

- **README.md** - This file, comprehensive usage guide
- **ASYNC_REFACTOR.md** - Asyncio implementation and NO GIL analysis
- **FEATURES.md** - Detailed implementation status and feature documentation
- **IMPLEMENTATION.md** - High-level implementation summary
- **QUICKSTART.md** - Quick getting started guide
- **COMPLETION.md** - Implementation completion status

## License

This project is intended for research and educational purposes.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Disclaimer

This is a research tool. Results should be interpreted carefully and not used to make accusations without proper context and investigation. Wikipedia has its own established processes for dealing with problematic behavior.
