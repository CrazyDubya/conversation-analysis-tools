#!/usr/bin/env python3
"""
Simple example demonstrating async performance benefits.

This script shows:
1. How to use async scraper
2. Rate limiting in action
3. Expected performance improvements
"""

import asyncio
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_scraper import MediaWikiAPIScraper
from api_scraper_async import AsyncMediaWikiAPIScraper


def sync_example():
    """Example of synchronous scraping (old way)."""
    print("\n" + "="*70)
    print("SYNCHRONOUS SCRAPING EXAMPLE")
    print("="*70)
    
    scraper = MediaWikiAPIScraper(
        user_agent="WikipediaANIReview/1.0 (Demo)",
        request_delay=0.2  # Fast for demo purposes
    )
    
    print("\n🐌 Scraping 3 sections sequentially...")
    start = time.time()
    
    # In real usage, would scrape actual threads
    # Here we'll just simulate with section requests
    print("  - Fetching section 1...")
    sections = scraper.get_page_sections(scraper.ANI_PAGE)
    
    print("  - Fetching section 2...")
    sections2 = scraper.get_page_sections(scraper.ANI_PAGE)
    
    print("  - Fetching section 3...")
    sections3 = scraper.get_page_sections(scraper.ANI_PAGE)
    
    elapsed = time.time() - start
    
    print(f"\n⏱️  Sync time: {elapsed:.2f}s")
    print(f"📊 Throughput: {3/elapsed:.2f} sections/sec")
    
    return elapsed


async def async_example():
    """Example of asynchronous scraping (new way)."""
    print("\n" + "="*70)
    print("ASYNCHRONOUS SCRAPING EXAMPLE")
    print("="*70)
    
    scraper = AsyncMediaWikiAPIScraper(
        user_agent="WikipediaANIReview/1.0 (Demo)",
        request_delay=0.2,  # Fast for demo purposes
        max_concurrent=3
    )
    
    print("\n🚀 Scraping 3 sections concurrently...")
    start = time.time()
    
    import aiohttp
    async with aiohttp.ClientSession(headers={'User-Agent': scraper.user_agent}) as session:
        # Fetch all sections concurrently
        tasks = [
            scraper.get_page_sections(session, scraper.ANI_PAGE)
            for _ in range(3)
        ]
        
        print("  - All 3 requests launched concurrently...")
        results = await asyncio.gather(*tasks)
        print(f"  - All 3 requests completed!")
    
    elapsed = time.time() - start
    
    print(f"\n⏱️  Async time: {elapsed:.2f}s")
    print(f"📊 Throughput: {3/elapsed:.2f} sections/sec")
    
    return elapsed


async def main():
    """Run both examples and compare."""
    print("\n" + "="*70)
    print("ASYNC PERFORMANCE DEMONSTRATION")
    print("="*70)
    print("\nThis demo compares synchronous vs asynchronous scraping.")
    print("Both use the same rate limiting (0.2s delay per request).")
    print("The difference is that async can make concurrent requests.\n")
    
    # Run sync example
    sync_time = sync_example()
    
    # Run async example
    async_time = await async_example()
    
    # Compare
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    speedup = sync_time / async_time if async_time > 0 else 0
    time_saved = sync_time - async_time
    
    print(f"\n📈 Results:")
    print(f"  Synchronous:  {sync_time:.2f}s")
    print(f"  Asynchronous: {async_time:.2f}s")
    print(f"  Speedup:      {speedup:.2f}x")
    print(f"  Time saved:   {time_saved:.2f}s ({time_saved/sync_time*100:.1f}%)")
    
    print("\n💡 Key Points:")
    print("  ✓ Async is faster for I/O-bound operations")
    print("  ✓ Rate limiting is still respected")
    print("  ✓ More sections = more speedup benefit")
    print("  ✓ Real-world speedup: 3-10x for 20+ sections")
    
    print("\n🎯 When to Use Async:")
    print("  ✓ Scraping multiple threads/archives")
    print("  ✓ Analyzing multiple threads with LLM")
    print("  ✓ Any I/O-heavy workflow")
    
    print("\n⚠️  When NOT to Use Async:")
    print("  ✗ Single thread scraping (no benefit)")
    print("  ✗ CPU-bound parsing only (use NO GIL instead)")
    print("  ✗ Simple scripts with minimal I/O")
    
    print("\n🔬 NO GIL Opportunities:")
    print("  For CPU-bound sections (regex parsing, text analysis),")
    print("  Python's upcoming NO GIL (PEP 703) could provide an")
    print("  additional 1.5-2x speedup on top of async benefits.")
    print("  See ASYNC_REFACTOR.md for details.")
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # Note: This demo makes real API requests to Wikipedia
    # It's safe because it uses appropriate rate limiting
    print("\n⚠️  This demo makes real API requests to Wikipedia.")
    print("Press Ctrl+C to cancel, or wait 3 seconds to continue...")
    
    try:
        time.sleep(3)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Demo cancelled by user.")
        sys.exit(0)
