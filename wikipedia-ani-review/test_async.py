"""Simple test to verify async implementation works correctly.

This test file demonstrates:
1. Async scraper initialization and rate limiting
2. Basic async operations
3. Mock testing without hitting Wikipedia servers
"""

import asyncio
import aiohttp
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_scraper_async import AsyncMediaWikiAPIScraper
from analyzer_async import AsyncThreadAnalyzer


async def test_async_scraper_initialization():
    """Test that async scraper initializes correctly with rate limiting."""
    print("🧪 Testing async scraper initialization...")
    
    scraper = AsyncMediaWikiAPIScraper(
        user_agent="Test/1.0",
        request_delay=0.1,
        max_concurrent=3
    )
    
    assert scraper.request_delay == 0.1
    assert scraper.max_concurrent == 3
    assert scraper.semaphore._value == 3
    
    print("✅ Async scraper initialized correctly")


async def test_async_analyzer_initialization():
    """Test that async analyzer initializes correctly."""
    print("\n🧪 Testing async analyzer initialization...")
    
    analyzer = AsyncThreadAnalyzer(
        api_key=None,  # No API key for testing
        model="gpt-3.5-turbo",
        max_concurrent=2
    )
    
    assert analyzer.model == "gpt-3.5-turbo"
    assert analyzer.max_concurrent == 2
    assert analyzer.semaphore._value == 2
    assert analyzer.client is None  # No API key
    
    print("✅ Async analyzer initialized correctly")


async def test_mock_analysis():
    """Test mock analysis works without API key."""
    print("\n🧪 Testing mock analysis...")
    
    analyzer = AsyncThreadAnalyzer(api_key=None)
    
    result = await analyzer.analyze_thread(
        "User editing without sources",
        [
            {"author": "User1", "content": "This user keeps adding unsourced content"},
            {"author": "User2", "content": "I recommend a block for vandalism"}
        ]
    )
    
    assert "primary_category" in result
    assert "requested_outcome" in result
    assert result["confidence_score"] == 0.3  # Mock analysis has low confidence
    
    print(f"✅ Mock analysis completed: {result['primary_category']} / {result['requested_outcome']}")


async def test_multiple_analyses():
    """Test concurrent analysis of multiple threads."""
    print("\n🧪 Testing concurrent analysis...")
    
    analyzer = AsyncThreadAnalyzer(api_key=None, max_concurrent=3)
    
    threads = [
        (f"Thread {i}", [{"author": "User", "content": "Edit warring discussion"}])
        for i in range(5)
    ]
    
    results = await analyzer.analyze_multiple_threads(threads)
    
    assert len(results) == 5
    assert all("primary_category" in r for r in results)
    
    print(f"✅ Analyzed {len(results)} threads concurrently")


async def test_rate_limiting():
    """Test that rate limiting works correctly."""
    print("\n🧪 Testing rate limiting...")
    
    scraper = AsyncMediaWikiAPIScraper(
        user_agent="Test/1.0",
        request_delay=0.05,  # 50ms delay
        max_concurrent=2
    )
    
    import time
    start = time.time()
    
    # Test rate limiter directly
    tasks = []
    for _ in range(6):
        # Use semaphore and rate limiting
        async def limited_operation():
            async with scraper.semaphore:
                await scraper._rate_limited_request()
                await asyncio.sleep(0.01)  # Simulate some work
        
        tasks.append(limited_operation())
    
    await asyncio.gather(*tasks)
    
    elapsed = time.time() - start
    
    # With max_concurrent=2 and delay=0.05, minimum time should be:
    # 6 requests / 2 concurrent * 0.05s = 0.15s
    expected_min = 0.15
    
    print(f"✅ Rate limiting working: {elapsed:.3f}s (expected >{expected_min:.3f}s)")
    assert elapsed >= expected_min * 0.8  # Allow some tolerance


async def test_wikitext_parsing():
    """Test CPU-bound wikitext parsing (NO GIL opportunity)."""
    print("\n🧪 Testing wikitext parsing (CPU-bound)...")
    
    scraper = AsyncMediaWikiAPIScraper()
    
    # Sample wikitext
    wikitext = """
This is a test thread about [[User:TestUser|TestUser]].
The user has been edit warring, see [[Special:Diff/12345]].
This violates [[WP:3RR]] and [[WP:NPOV]].
The user should be blocked.
    """
    
    result = scraper._parse_wikitext(wikitext)
    
    assert "posts" in result
    assert "diffs" in result
    assert "policy_shortcuts" in result
    assert len(result["diffs"]) > 0
    assert len(result["policy_shortcuts"]) >= 2  # Should find 3RR and NPOV
    
    print(f"✅ Wikitext parsing completed: {len(result['diffs'])} diffs, {len(result['policy_shortcuts'])} policy shortcuts")


async def main():
    """Run all tests."""
    print("="*70)
    print("ASYNC IMPLEMENTATION TEST SUITE")
    print("="*70)
    
    try:
        await test_async_scraper_initialization()
        await test_async_analyzer_initialization()
        await test_mock_analysis()
        await test_multiple_analyses()
        await test_rate_limiting()
        await test_wikitext_parsing()
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED")
        print("="*70)
        
        print("\n💡 Key Findings:")
        print("  - Async scraper and analyzer initialize correctly")
        print("  - Rate limiting works with semaphore + delay")
        print("  - Mock analysis works without API key")
        print("  - Concurrent analysis processes multiple threads")
        print("  - Wikitext parsing (CPU-bound) marked for NO GIL benefit")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
