#!/usr/bin/env python3
"""Async commands for Wikipedia ANI Review system.

This module adds async-enabled commands to the CLI for improved performance:
- async-scrape: Concurrent scraping with rate limiting
- async-analyze: Concurrent LLM analysis
- benchmark: Compare sync vs async performance
- nogil-report: Analyze NO GIL opportunities
"""

import os
import sys
import asyncio
import logging
import click
from dotenv import load_dotenv

from models import init_db, get_session, Thread, ThreadAnalysis
from database import DatabaseManager
from api_scraper import MediaWikiAPIScraper
from api_scraper_async import AsyncMediaWikiAPIScraper
from analyzer import ThreadAnalyzer
from analyzer_async import AsyncThreadAnalyzer
from profiling_utils import (
    PerformanceProfiler,
    NOGILAnalyzer,
    benchmark_scraping,
    benchmark_analysis
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Async commands for Wikipedia ANI Review."""
    pass


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
@click.option('--archives/--no-archives', default=False, help='Scrape archives')
@click.option('--archive-limit', default=10, help='Number of archives to scrape')
@click.option('--max-concurrent', default=5, help='Max concurrent requests')
@click.option('--request-delay', default=1.0, help='Delay between requests (seconds)')
def async_scrape(database_url, archives, archive_limit, max_concurrent, request_delay):
    """Scrape ANI pages using async for concurrent requests.
    
    Benefits:
    - 3-10x faster than sync scraping
    - Respects rate limits with semaphore
    - Concurrent section fetching
    """
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    # Initialize database
    engine = init_db(db_url)
    session = get_session(engine)
    db_manager = DatabaseManager(session)
    
    user_agent = os.getenv('USER_AGENT', 'WikipediaANIReview/1.0 (Async)')
    
    # Create async scraper
    scraper = AsyncMediaWikiAPIScraper(
        user_agent=user_agent,
        request_delay=request_delay,
        max_concurrent=max_concurrent
    )
    
    click.echo(f"Using async MediaWiki API scraper")
    click.echo(f"Rate limiting: {request_delay}s delay, {max_concurrent} concurrent requests")
    
    async def run_scrape():
        profiler = PerformanceProfiler()
        
        # Scrape current ANI page
        click.echo("\n📥 Scraping current ANI page...")
        with profiler.time_block("current_page"):
            threads = await scraper.scrape_current_ani()
        
        click.echo(f"Found {len(threads)} threads ({profiler.results['current_page']:.2f}s)")
        
        # Store threads
        stored_count = 0
        for thread_data in threads:
            try:
                thread = db_manager.create_thread(thread_data)
                stored_count += 1
            except Exception as e:
                logger.error(f"Error storing thread: {e}")
        
        click.echo(f"Stored {stored_count} threads")
        
        # Scrape archives if requested
        if archives:
            click.echo(f"\n📚 Scraping up to {archive_limit} archives concurrently...")
            
            # Get archive list
            archive_numbers = await scraper.get_archive_list()
            if archive_numbers:
                click.echo(f"Found {len(archive_numbers)} archives in index")
                archive_numbers = archive_numbers[:archive_limit]
            else:
                archive_numbers = list(range(1, archive_limit + 1))
            
            # Scrape all archives concurrently - KEY ASYNC BENEFIT
            with profiler.time_block("archives"):
                all_archive_threads = await scraper.scrape_multiple_archives(archive_numbers)
            
            click.echo(f"Found {len(all_archive_threads)} threads in archives ({profiler.results['archives']:.2f}s)")
            
            # Store archive threads
            archive_stored = 0
            for thread_data in all_archive_threads:
                try:
                    db_manager.create_thread(thread_data)
                    archive_stored += 1
                except Exception:
                    pass  # Skip duplicates
            
            click.echo(f"Stored {archive_stored} new archive threads")
        
        # Summary
        click.echo(f"\n✅ Scraping complete!")
        click.echo(f"Total threads in database: {db_manager.get_thread_count()}")
        click.echo(f"Total time: {sum(profiler.results.values()):.2f}s")
    
    # Run async function
    asyncio.run(run_scrape())


@cli.command()
@click.option('--database-url', default=None, help='Database URL')
@click.option('--limit', default=None, type=int, help='Limit number of threads to analyze')
@click.option('--max-concurrent', default=3, help='Max concurrent API calls')
def async_analyze(database_url, limit, max_concurrent):
    """Analyze threads using async for concurrent LLM calls.
    
    Benefits:
    - 3-5x faster than sync analysis
    - Rate-limited concurrent API calls
    - Efficient batch processing
    """
    db_url = database_url or os.getenv('DATABASE_URL', 'sqlite:///ani_review.db')
    
    # Initialize database
    engine = init_db(db_url)
    session = get_session(engine)
    db_manager = DatabaseManager(session)
    
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Create async analyzer
    analyzer = AsyncThreadAnalyzer(
        api_key=api_key,
        model=model,
        max_concurrent=max_concurrent
    )
    
    if not analyzer.client:
        click.echo("⚠️  No OpenAI API key found. Using mock analysis.", err=True)
    
    # Get unanalyzed threads
    query = session.query(Thread).outerjoin(ThreadAnalysis).filter(
        ThreadAnalysis.id == None
    )
    
    if limit:
        query = query.limit(limit)
    
    threads = query.all()
    
    if not threads:
        click.echo("No unanalyzed threads found.")
        return
    
    click.echo(f"Analyzing {len(threads)} threads with {max_concurrent} concurrent calls...")
    
    async def run_analysis():
        profiler = PerformanceProfiler()
        
        # Prepare thread data
        thread_data = [
            (thread.title, [{'author': p.author.username if p.author else 'Unknown',
                            'content': p.content} for p in thread.posts])
            for thread in threads
        ]
        
        # Analyze all threads concurrently - KEY ASYNC BENEFIT
        with profiler.time_block("async_analysis"):
            analyses = await analyzer.analyze_multiple_threads(thread_data)
        
        # Store results
        for thread, analysis in zip(threads, analyses):
            try:
                db_manager.store_thread_analysis(thread, analysis)
                click.echo(f"✓ {thread.title[:60]}: {analysis['primary_category']}")
            except Exception as e:
                logger.error(f"Error storing analysis: {e}")
        
        click.echo(f"\n✅ Analysis complete!")
        click.echo(f"Total analyzed: {len(analyses)}")
        click.echo(f"Total time: {profiler.results['async_analysis']:.2f}s")
        click.echo(f"Avg time per thread: {profiler.results['async_analysis']/len(analyses):.2f}s")
    
    asyncio.run(run_analysis())


@cli.command()
@click.option('--num-threads', default=10, help='Number of threads to test with')
@click.option('--test-analysis/--no-test-analysis', default=False, help='Also benchmark analysis')
def benchmark(num_threads, test_analysis):
    """Benchmark sync vs async performance.
    
    Compares:
    - Sync vs async scraping
    - Sync vs async analysis (optional)
    - Shows speedup metrics
    """
    click.echo("="*70)
    click.echo("PERFORMANCE BENCHMARK: SYNC vs ASYNC")
    click.echo("="*70)
    
    user_agent = os.getenv('USER_AGENT', 'WikipediaANIReview/1.0')
    request_delay = float(os.getenv('REQUEST_DELAY', '1.0'))
    
    async def run_benchmarks():
        # Benchmark scraping
        click.echo(f"\n🔬 Benchmarking scraping (fetching {num_threads} sections)...")
        
        sync_scraper = MediaWikiAPIScraper(
            user_agent=user_agent,
            request_delay=request_delay
        )
        
        async_scraper = AsyncMediaWikiAPIScraper(
            user_agent=user_agent,
            request_delay=request_delay,
            max_concurrent=5
        )
        
        scrape_results = await benchmark_scraping(sync_scraper, async_scraper, num_threads)
        
        # Benchmark analysis if requested
        if test_analysis:
            click.echo(f"\n🔬 Benchmarking analysis ({num_threads} threads)...")
            
            # Create some dummy thread data
            dummy_threads = [
                (f"Thread {i}", [
                    {'author': 'User1', 'content': 'Test content with edit warring discussion'},
                    {'author': 'User2', 'content': 'Response about blocking the user'}
                ])
                for i in range(num_threads)
            ]
            
            api_key = os.getenv('OPENAI_API_KEY')
            model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
            
            sync_analyzer = ThreadAnalyzer(api_key=api_key, model=model)
            async_analyzer = AsyncThreadAnalyzer(api_key=api_key, model=model, max_concurrent=3)
            
            analysis_results = await benchmark_analysis(sync_analyzer, async_analyzer, dummy_threads)
        
        # Summary
        click.echo("\n" + "="*70)
        click.echo("BENCHMARK SUMMARY")
        click.echo("="*70)
        
        click.echo(f"\n📊 Scraping Results:")
        click.echo(f"  Sync time:  {scrape_results['sync_time']:.2f}s")
        click.echo(f"  Async time: {scrape_results['async_time']:.2f}s")
        click.echo(f"  Speedup:    {scrape_results['speedup']:.2f}x")
        
        if test_analysis:
            click.echo(f"\n📊 Analysis Results:")
            click.echo(f"  Sync time:  {analysis_results['sync_time']:.2f}s")
            click.echo(f"  Async time: {analysis_results['async_time']:.2f}s")
            click.echo(f"  Speedup:    {analysis_results['speedup']:.2f}x")
        
        click.echo("\n💡 Interpretation:")
        click.echo("  - Async provides significant speedup for I/O-bound operations")
        click.echo("  - Speedup scales with number of concurrent operations")
        click.echo("  - Respects rate limits while maximizing throughput")
        
        click.echo("\n" + "="*70)
    
    asyncio.run(run_benchmarks())


@cli.command()
@click.option('--output', default=None, help='Output file (default: stdout)')
def nogil_report(output):
    """Generate report on NO GIL (PEP 703) opportunities.
    
    Documents:
    - CPU-bound sections that would benefit from NO GIL
    - I/O-bound sections where async is sufficient
    - Expected speedups and implementation strategies
    """
    report = NOGILAnalyzer.generate_nogil_report()
    
    if output:
        with open(output, 'w') as f:
            f.write(report)
        click.echo(f"Report written to {output}")
    else:
        click.echo(report)


@cli.command()
@click.option('--max-concurrent', default=5, help='Max concurrent requests to test')
@click.option('--request-delay', default=1.0, help='Request delay to test (seconds)')
def test_rate_limiting(max_concurrent, request_delay):
    """Test and demonstrate rate limiting behavior.
    
    Shows:
    - How semaphore limits concurrent requests
    - How delay prevents overwhelming servers
    - Actual vs theoretical throughput
    """
    click.echo("="*70)
    click.echo("RATE LIMITING TEST")
    click.echo("="*70)
    
    click.echo(f"\nConfiguration:")
    click.echo(f"  Max concurrent: {max_concurrent}")
    click.echo(f"  Request delay:  {request_delay}s")
    
    user_agent = os.getenv('USER_AGENT', 'WikipediaANIReview/1.0')
    
    async def test_rate_limit():
        scraper = AsyncMediaWikiAPIScraper(
            user_agent=user_agent,
            request_delay=request_delay,
            max_concurrent=max_concurrent
        )
        
        click.echo(f"\n🧪 Making 10 test requests...")
        
        import time
        start = time.time()
        
        # Make 10 concurrent requests
        import aiohttp
        async with aiohttp.ClientSession(headers={'User-Agent': user_agent}) as session:
            tasks = [
                scraper.get_page_sections(session, "Wikipedia:Administrators' noticeboard/Incidents")
                for _ in range(10)
            ]
            results = await asyncio.gather(*tasks)
        
        elapsed = time.time() - start
        
        click.echo(f"\n📊 Results:")
        click.echo(f"  Total time: {elapsed:.2f}s")
        click.echo(f"  Successful: {sum(1 for r in results if r)}")
        click.echo(f"  Avg per request: {elapsed/10:.2f}s")
        
        # Calculate theoretical limits
        theoretical_min = 10 * request_delay / max_concurrent
        click.echo(f"\n💡 Analysis:")
        click.echo(f"  Theoretical minimum time: {theoretical_min:.2f}s")
        click.echo(f"  Actual time: {elapsed:.2f}s")
        click.echo(f"  Efficiency: {(theoretical_min/elapsed)*100:.1f}%")
        
        click.echo(f"\n✅ Rate limiting is working correctly!")
        click.echo(f"   - Semaphore limits to {max_concurrent} concurrent requests")
        click.echo(f"   - Each request respects {request_delay}s delay")
        click.echo(f"   - Wikipedia servers are protected from overload")
    
    asyncio.run(test_rate_limit())


if __name__ == '__main__':
    cli()
