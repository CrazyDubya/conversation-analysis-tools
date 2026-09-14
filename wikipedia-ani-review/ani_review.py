#!/usr/bin/env python3
"""Main CLI script for Wikipedia ANI Review system."""

import os
import sys
import click
from dotenv import load_dotenv

from models import init_db, get_session
from scraper import ANIScraper
from analyzer import ThreadAnalyzer
from database import DatabaseManager


# Load environment variables
load_dotenv()


@click.group()
def cli():
    """Wikipedia ANI Review - Analyze Administrator Noticeboard discussions."""
    pass


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
def init(database_url):
    """Initialize the database."""
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    click.echo(f"Initializing database at {db_url}...")
    engine = init_db(db_url)
    click.echo("Database initialized successfully!")


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
@click.option('--archives/--no-archives', default=False, help='Scrape archives (slow)')
@click.option('--archive-limit', default=10, help='Number of archives to scrape')
@click.option('--use-api/--use-html', default=True, help='Use MediaWiki API (default) or HTML scraping')
def scrape(database_url, archives, archive_limit, use_api):
    """Scrape ANI pages and store in database."""
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    # Initialize database
    engine = init_db(db_url)
    session = get_session(engine)
    db_manager = DatabaseManager(session)
    
    user_agent = os.getenv('USER_AGENT', 'WikipediaANIReview/1.0')
    request_delay = float(os.getenv('REQUEST_DELAY', '1.0'))
    
    # Choose scraper based on option
    if use_api:
        from api_scraper import MediaWikiAPIScraper
        scraper = MediaWikiAPIScraper(user_agent=user_agent, request_delay=request_delay)
        click.echo("Using MediaWiki API scraper (recommended)")
    else:
        scraper = ANIScraper(user_agent=user_agent, request_delay=request_delay)
        click.echo("Using HTML scraper")
    
    # Scrape current ANI page
    click.echo("Scraping current ANI page...")
    threads = scraper.scrape_current_ani()
    click.echo(f"Found {len(threads)} threads on current page")
    
    for thread_data in threads:
        try:
            thread = db_manager.create_thread(thread_data)
            click.echo(f"  Stored: {thread.title[:60]}...")
        except Exception as e:
            click.echo(f"  Error storing thread: {e}", err=True)
    
    # Scrape archives if requested
    if archives:
        click.echo(f"\nScraping up to {archive_limit} archive pages...")
        
        if use_api and hasattr(scraper, 'get_archive_list'):
            # Use API to get actual archive list
            archive_numbers = scraper.get_archive_list()
            if archive_numbers:
                click.echo(f"Found {len(archive_numbers)} archives in index")
                archive_numbers = archive_numbers[:archive_limit]
            else:
                # Fallback to sequential numbering
                archive_numbers = list(range(1, archive_limit + 1))
            
            for i, archive_num in enumerate(archive_numbers, 1):
                click.echo(f"Scraping archive {i}/{len(archive_numbers)}: IncidentArchive{archive_num}")
                
                try:
                    archive_threads = scraper.scrape_archive(archive_num)
                    click.echo(f"  Found {len(archive_threads)} threads")
                    
                    for thread_data in archive_threads:
                        try:
                            thread = db_manager.create_thread(thread_data)
                        except Exception as e:
                            # Silently skip duplicates or errors
                            pass
                except Exception as e:
                    click.echo(f"  Error scraping archive: {e}", err=True)
        else:
            # HTML scraper - use old method
            archive_urls = scraper.find_archive_pages(limit=archive_limit)
            
            for i, archive_url in enumerate(archive_urls, 1):
                click.echo(f"Scraping archive {i}/{len(archive_urls)}: {archive_url}")
                
                try:
                    archive_threads = scraper.scrape_archive(archive_url)
                    click.echo(f"  Found {len(archive_threads)} threads")
                    
                    for thread_data in archive_threads:
                        try:
                            thread = db_manager.create_thread(thread_data)
                        except Exception as e:
                            # Silently skip duplicates or errors
                            pass
                except Exception as e:
                    click.echo(f"  Error scraping archive: {e}", err=True)
    
    click.echo(f"\nTotal threads in database: {db_manager.get_thread_count()}")
    click.echo(f"Total users in database: {db_manager.get_user_count()}")


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
@click.option('--limit', default=None, type=int, help='Limit number of threads to analyze')
def analyze(database_url, limit):
    """Analyze threads using LLM."""
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    # Initialize database and analyzer
    engine = init_db(db_url)
    session = get_session(engine)
    db_manager = DatabaseManager(session)
    
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    analyzer = ThreadAnalyzer(api_key=api_key, model=model)
    
    # Get threads that haven't been analyzed
    from models import Thread, ThreadAnalysis
    query = session.query(Thread).outerjoin(ThreadAnalysis).filter(
        ThreadAnalysis.id == None
    )
    
    if limit:
        query = query.limit(limit)
    
    threads = query.all()
    
    if not threads:
        click.echo("No unanalyzed threads found.")
        return
    
    click.echo(f"Analyzing {len(threads)} threads...")
    
    if not analyzer.client:
        click.echo("Warning: No OpenAI API key found. Using mock analysis.", err=True)
    
    for i, thread in enumerate(threads, 1):
        try:
            click.echo(f"[{i}/{len(threads)}] Analyzing: {thread.title[:60]}...")
            
            # Get posts for this thread
            posts = [{
                'author': p.author.username if p.author else 'Unknown',
                'content': p.content
            } for p in thread.posts]
            
            # Analyze
            analysis = analyzer.analyze_thread(thread.title, posts)
            
            # Store analysis
            db_manager.store_thread_analysis(thread, analysis)
            
            click.echo(f"  Category: {analysis['primary_category']}")
            click.echo(f"  Outcome: {analysis['requested_outcome']}")
            
        except Exception as e:
            click.echo(f"  Error: {e}", err=True)
    
    click.echo(f"\nTotal analyzed threads: {db_manager.get_analyzed_thread_count()}")


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
@click.option('--thread-threshold', default=10, help='Thread creation threshold')
@click.option('--post-threshold', default=50, help='Post creation threshold')
@click.option('--show-targets/--no-show-targets', default=False, help='Show threads with identified targets')
def report(database_url, thread_threshold, post_threshold, show_targets):
    """Generate analysis report."""
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    # Initialize database
    engine = init_db(db_url)
    session = get_session(engine)
    db_manager = DatabaseManager(session)
    
    click.echo("=" * 70)
    click.echo("WIKIPEDIA ANI REVIEW - ANALYSIS REPORT")
    click.echo("=" * 70)
    
    # Database statistics
    click.echo("\n📊 DATABASE STATISTICS")
    click.echo(f"Total threads: {db_manager.get_thread_count()}")
    click.echo(f"Total users: {db_manager.get_user_count()}")
    click.echo(f"Analyzed threads: {db_manager.get_analyzed_thread_count()}")
    click.echo(f"Threads with diff evidence: {db_manager.get_threads_with_diffs()}")
    
    # Top thread creators
    click.echo("\n👤 TOP THREAD CREATORS")
    creators = db_manager.get_top_thread_creators(20)
    for i, (username, count) in enumerate(creators, 1):
        click.echo(f"{i:2d}. {username:30s} - {count:3d} threads")
    
    # Top responders
    click.echo("\n💬 TOP RESPONDERS")
    responders = db_manager.get_top_responders(20)
    for i, (username, count) in enumerate(responders, 1):
        click.echo(f"{i:2d}. {username:30s} - {count:4d} posts")
    
    # Category distribution
    click.echo("\n📁 CATEGORY DISTRIBUTION")
    categories = db_manager.get_category_distribution()
    for category, count in categories:
        if category:
            click.echo(f"  {category:30s} - {count:3d} threads")
    
    # Requested outcome distribution
    click.echo("\n⚖️  REQUESTED OUTCOME DISTRIBUTION")
    outcomes = db_manager.get_outcome_distribution()
    for outcome, count in outcomes:
        if outcome:
            click.echo(f"  {outcome:30s} - {count:3d} threads")
    
    # Closure statistics
    click.echo("\n🔒 CLOSURE OUTCOME STATISTICS")
    closures = db_manager.get_closure_statistics()
    if closures:
        for outcome, count in closures:
            click.echo(f"  {outcome:30s} - {count:3d} threads")
    else:
        click.echo("  No closure data available")
    
    # Routing statistics
    click.echo("\n🔀 ROUTING SUGGESTIONS")
    routing = db_manager.get_routing_statistics()
    if routing:
        for category, count in routing:
            click.echo(f"  {category:30s} - {count:3d} threads")
    else:
        click.echo("  No routing data available")
    
    # Overactive users
    click.echo("\n⚠️  POTENTIALLY OVERACTIVE USERS")
    click.echo(f"(Thresholds: {thread_threshold} threads or {post_threshold} posts)")
    overactive = db_manager.identify_overactive_users(thread_threshold, post_threshold)
    
    if overactive:
        for i, user in enumerate(overactive[:20], 1):
            click.echo(f"{i:2d}. {user['username']:30s}")
            click.echo(f"    Threads: {user['threads_created']:3d} | Posts: {user['posts_made']:4d} | Score: {user['overactive_score']:.1f}")
    else:
        click.echo("  No overactive users identified with current thresholds")
    
    # Show threads with targets if requested
    if show_targets:
        click.echo("\n🎯 THREADS WITH IDENTIFIED TARGETS")
        targets_data = db_manager.get_threads_with_targets(10)
        if targets_data:
            for i, thread_info in enumerate(targets_data, 1):
                click.echo(f"\n{i}. {thread_info['title'][:60]}")
                click.echo(f"   Filer: {thread_info['filer'] or 'Unknown'}")
                click.echo(f"   Targets: {', '.join(thread_info['targets'])}")
                click.echo(f"   Outcome: {thread_info['outcome'] or 'Pending'}")
        else:
            click.echo("  No threads with identified targets")
    
    click.echo("\n" + "=" * 70)


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
@click.argument('username')
def user(database_url, username):
    """Get detailed information about a specific user."""
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    engine = init_db(db_url)
    session = get_session(engine)
    db_manager = DatabaseManager(session)
    
    stats = db_manager.get_user_statistics(username)
    
    if not stats:
        click.echo(f"User '{username}' not found in database.")
        return
    
    user_data = stats[0]
    
    click.echo(f"\n👤 User: {user_data['username']}")
    click.echo(f"Threads created: {user_data['threads_created']}")
    click.echo(f"Posts made: {user_data['posts_made']}")
    click.echo(f"First seen: {user_data['first_seen']}")
    click.echo(f"Last seen: {user_data['last_seen']}")
    
    # Check for voting statistics
    voter_stats = db_manager.get_voter_statistics(username)
    if voter_stats:
        click.echo(f"\n📊 Voting Statistics:")
        vs = voter_stats[0]
        click.echo(f"Total votes: {vs['total_votes']}")
        click.echo(f"  Support: {vs['support']}")
        click.echo(f"  Oppose: {vs['oppose']}")
        click.echo(f"  Neutral: {vs['neutral']}")
        click.echo(f"  Comment: {vs['comment']}")


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
@click.option('--limit', default=20, help='Number of threads to show')
def votes(database_url, limit):
    """Show threads with voting data."""
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    engine = init_db(db_url)
    session = get_session(engine)
    db_manager = DatabaseManager(session)
    
    click.echo("=" * 70)
    click.echo("THREADS WITH VOTING DATA")
    click.echo("=" * 70)
    
    threads = db_manager.get_threads_with_votes(limit)
    
    if not threads:
        click.echo("\nNo threads with voting data found.")
        return
    
    for i, thread in enumerate(threads, 1):
        click.echo(f"\n{i}. {thread['title'][:60]}")
        click.echo(f"   Votes: {thread['vote_count']}")
        click.echo(f"   Outcome: {thread['outcome'] or 'Pending'}")
        click.echo(f"   URL: {thread['url']}")
    
    click.echo("\n" + "=" * 70)


if __name__ == '__main__':
    cli()
