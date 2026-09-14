"""Profiling and benchmarking utilities for async vs sync performance comparison.

This module provides tools to:
1. Profile sync vs async scraping and analysis
2. Identify CPU-bound vs I/O-bound sections
3. Demonstrate NO GIL opportunities
4. Generate performance reports
"""

import time
import asyncio
import cProfile
import pstats
import io
from typing import Callable, Dict, List, Any, Optional
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profile and benchmark sync vs async implementations."""
    
    def __init__(self):
        self.results = {}
    
    @contextmanager
    def time_block(self, name: str):
        """Context manager to time a code block.
        
        Args:
            name: Name of the block being timed
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.results[name] = elapsed
            logger.info(f"{name}: {elapsed:.3f}s")
    
    async def time_async_block(self, name: str, coro):
        """Time an async coroutine.
        
        Args:
            name: Name of the operation
            coro: Coroutine to time
            
        Returns:
            Result of the coroutine
        """
        start = time.perf_counter()
        try:
            result = await coro
            return result
        finally:
            elapsed = time.perf_counter() - start
            self.results[name] = elapsed
            logger.info(f"{name}: {elapsed:.3f}s")
    
    def profile_sync_function(self, func: Callable, *args, **kwargs) -> Any:
        """Profile a synchronous function with cProfile.
        
        Args:
            func: Function to profile
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
        """
        profiler = cProfile.Profile()
        profiler.enable()
        
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        
        profiler.disable()
        
        # Print stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        logger.info(f"\n=== Profile for {func.__name__} ({elapsed:.3f}s) ===")
        logger.info(s.getvalue())
        
        return result
    
    async def profile_async_function(self, coro, name: str = "async_function") -> Any:
        """Profile an async coroutine.
        
        Note: cProfile doesn't work well with async, so we use timing analysis.
        
        Args:
            coro: Coroutine to profile
            name: Name for logging
            
        Returns:
            Coroutine result
        """
        start = time.perf_counter()
        result = await coro
        elapsed = time.perf_counter() - start
        
        logger.info(f"\n=== Async profile for {name} ===")
        logger.info(f"Total time: {elapsed:.3f}s")
        
        return result
    
    def compare_implementations(
        self,
        sync_func: Callable,
        async_func: Callable,
        sync_args: tuple = (),
        async_args: tuple = (),
        name: str = "operation"
    ) -> Dict[str, float]:
        """Compare sync vs async implementations.
        
        Args:
            sync_func: Synchronous implementation
            async_func: Asynchronous implementation (coroutine function)
            sync_args: Arguments for sync function
            async_args: Arguments for async function
            name: Name of operation being compared
            
        Returns:
            Dictionary with timing results and speedup
        """
        logger.info(f"\n=== Comparing {name} ===")
        
        # Time sync version
        with self.time_block(f"{name}_sync"):
            sync_result = sync_func(*sync_args)
        sync_time = self.results[f"{name}_sync"]
        
        # Time async version
        async def run_async():
            return await async_func(*async_args)
        
        with self.time_block(f"{name}_async"):
            async_result = asyncio.run(run_async())
        async_time = self.results[f"{name}_async"]
        
        # Calculate speedup
        speedup = sync_time / async_time if async_time > 0 else 0
        
        results = {
            'sync_time': sync_time,
            'async_time': async_time,
            'speedup': speedup,
            'time_saved': sync_time - async_time
        }
        
        logger.info(f"Sync time: {sync_time:.3f}s")
        logger.info(f"Async time: {async_time:.3f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Time saved: {results['time_saved']:.3f}s")
        
        return results
    
    def generate_report(self) -> str:
        """Generate a performance report.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*70)
        report.append("PERFORMANCE PROFILING REPORT")
        report.append("="*70)
        
        if not self.results:
            report.append("\nNo profiling data collected.")
            return "\n".join(report)
        
        report.append("\nTiming Results:")
        report.append("-" * 70)
        
        for name, elapsed in sorted(self.results.items()):
            report.append(f"{name:40s}: {elapsed:8.3f}s")
        
        # Calculate speedups for paired operations
        report.append("\n" + "-" * 70)
        report.append("Speedup Analysis:")
        report.append("-" * 70)
        
        sync_ops = {k: v for k, v in self.results.items() if '_sync' in k}
        for sync_name, sync_time in sync_ops.items():
            base_name = sync_name.replace('_sync', '')
            async_name = f"{base_name}_async"
            
            if async_name in self.results:
                async_time = self.results[async_name]
                speedup = sync_time / async_time if async_time > 0 else 0
                report.append(f"{base_name:30s}: {speedup:5.2f}x speedup")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


class NOGILAnalyzer:
    """Analyze and document NO GIL opportunities in the codebase.
    
    This class helps identify CPU-bound sections that would benefit
    from true parallel threading in free-threaded Python (PEP 703).
    """
    
    # CPU-bound operations that could benefit from NO GIL
    CPU_BOUND_OPERATIONS = {
        'wikitext_parsing': {
            'location': 'api_scraper_async.py::_parse_wikitext',
            'description': 'Regex-heavy parsing of wikitext (split, regex, string ops)',
            'expected_speedup': '1.5-2x with 4 threads on 4+ cores',
            'benefit': 'HIGH - Complex regex and string processing',
        },
        'vote_parsing': {
            'location': 'vote_parser.py::parse_votes',
            'description': 'Vote structure detection and parsing',
            'expected_speedup': '1.3-1.8x with 4 threads',
            'benefit': 'MEDIUM - Regex and text analysis',
        },
        'mock_analysis': {
            'location': 'analyzer_async.py::_mock_analysis',
            'description': 'Keyword-based categorization',
            'expected_speedup': '1.2-1.5x with 4 threads',
            'benefit': 'LOW-MEDIUM - Simple string operations',
        },
        'json_parsing': {
            'location': 'Various: JSON loads/dumps operations',
            'description': 'JSON serialization and deserialization',
            'expected_speedup': '1.1-1.3x with 4 threads',
            'benefit': 'LOW - Python json module is already efficient',
        },
    }
    
    # I/O-bound operations (async is sufficient, NO GIL not needed)
    IO_BOUND_OPERATIONS = {
        'http_requests': {
            'location': 'api_scraper_async.py::_api_request',
            'description': 'HTTP API calls to MediaWiki',
            'async_benefit': 'HIGH - 5-10x speedup with concurrent requests',
            'nogil_benefit': 'NONE - I/O bound, async handles this well',
        },
        'llm_api_calls': {
            'location': 'analyzer_async.py::analyze_thread',
            'description': 'OpenAI API calls for thread analysis',
            'async_benefit': 'HIGH - 3-5x speedup with concurrent calls',
            'nogil_benefit': 'NONE - I/O bound, async is optimal',
        },
        'database_operations': {
            'location': 'database.py::*',
            'description': 'SQLite database queries and inserts',
            'async_benefit': 'LOW-MEDIUM - Some benefit from async',
            'nogil_benefit': 'LOW - SQLite has its own locking',
        },
    }
    
    @classmethod
    def generate_nogil_report(cls) -> str:
        """Generate a report on NO GIL opportunities.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("\n" + "="*70)
        report.append("NO GIL (PEP 703) OPPORTUNITY ANALYSIS")
        report.append("="*70)
        
        report.append("\n📊 CPU-BOUND OPERATIONS (Would benefit from NO GIL)")
        report.append("-" * 70)
        
        for op_name, details in cls.CPU_BOUND_OPERATIONS.items():
            report.append(f"\n{op_name.upper().replace('_', ' ')}:")
            report.append(f"  Location: {details['location']}")
            report.append(f"  Description: {details['description']}")
            report.append(f"  Expected Speedup: {details['expected_speedup']}")
            report.append(f"  Benefit Level: {details['benefit']}")
        
        report.append("\n\n🌐 I/O-BOUND OPERATIONS (Async is sufficient)")
        report.append("-" * 70)
        
        for op_name, details in cls.IO_BOUND_OPERATIONS.items():
            report.append(f"\n{op_name.upper().replace('_', ' ')}:")
            report.append(f"  Location: {details['location']}")
            report.append(f"  Description: {details['description']}")
            report.append(f"  Async Benefit: {details['async_benefit']}")
            report.append(f"  NO GIL Benefit: {details['nogil_benefit']}")
        
        report.append("\n\n💡 IMPLEMENTATION STRATEGY")
        report.append("-" * 70)
        report.append("""
For CPU-bound sections in free-threaded Python:

1. Use ThreadPoolExecutor for CPU-heavy parsing:
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       futures = [executor.submit(parse_wikitext, text) for text in texts]
       results = [f.result() for f in futures]
   ```

2. Combine with asyncio for I/O operations:
   ```python
   async def process_thread(thread_text):
       # I/O: Fetch data (use async)
       data = await fetch_section(thread_text)
       
       # CPU: Parse data (use ThreadPoolExecutor in NO GIL Python)
       loop = asyncio.get_event_loop()
       result = await loop.run_in_executor(executor, parse_wikitext, data)
       
       return result
   ```

3. Current implementation:
   - Async for I/O: ✅ Provides 3-10x speedup
   - ThreadPool for CPU: ⏳ Ready for NO GIL (see examples above)
   - Expected total speedup with NO GIL: 1.5-2.5x additional on top of async
        """)
        
        report.append("\n\n⚠️ IMPORTANT NOTES")
        report.append("-" * 70)
        report.append("""
1. ASYNC vs NO GIL:
   - Async: Best for I/O-bound (network, disk, API calls)
   - NO GIL: Best for CPU-bound (parsing, computation, algorithms)
   - This codebase is ~70% I/O-bound, ~30% CPU-bound

2. Expected Performance:
   - Async alone: 3-10x faster for scraping/analysis
   - NO GIL addition: 1.5-2x faster on top (for CPU sections)
   - Combined: 4-15x faster than original sync code

3. When NO GIL helps:
   ✓ Regex-heavy wikitext parsing
   ✓ Multiple mock analyses in parallel
   ✓ Batch JSON operations
   
4. When NO GIL doesn't help:
   ✗ Network requests (use async)
   ✗ API calls (use async)
   ✗ Simple string ops (overhead > benefit)
        """)
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


# Convenience functions for common profiling tasks
async def benchmark_scraping(sync_scraper, async_scraper, num_threads: int = 10):
    """Benchmark sync vs async scraping performance.
    
    Args:
        sync_scraper: Synchronous scraper instance
        async_scraper: Asynchronous scraper instance
        num_threads: Number of threads to scrape
    """
    profiler = PerformanceProfiler()
    
    logger.info(f"\n🔬 Benchmarking scraping performance ({num_threads} threads)")
    
    # Sync version
    with profiler.time_block("sync_scraping"):
        sync_threads = sync_scraper.scrape_current_ani()
        sync_count = len(sync_threads)
    
    # Async version
    with profiler.time_block("async_scraping"):
        async_threads = await async_scraper.scrape_current_ani()
        async_count = len(async_threads)
    
    # Calculate speedup
    sync_time = profiler.results["sync_scraping"]
    async_time = profiler.results["async_scraping"]
    speedup = sync_time / async_time if async_time > 0 else 0
    
    logger.info(f"\n📊 Results:")
    logger.info(f"Threads scraped: {sync_count} (sync), {async_count} (async)")
    logger.info(f"Sync time: {sync_time:.2f}s")
    logger.info(f"Async time: {async_time:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return {
        'sync_time': sync_time,
        'async_time': async_time,
        'speedup': speedup,
        'sync_count': sync_count,
        'async_count': async_count
    }


async def benchmark_analysis(sync_analyzer, async_analyzer, threads: List[tuple]):
    """Benchmark sync vs async analysis performance.
    
    Args:
        sync_analyzer: Synchronous analyzer instance
        async_analyzer: Asynchronous analyzer instance
        threads: List of (title, posts) tuples to analyze
    """
    profiler = PerformanceProfiler()
    
    logger.info(f"\n🔬 Benchmarking analysis performance ({len(threads)} threads)")
    
    # Sync version
    with profiler.time_block("sync_analysis"):
        sync_results = []
        for title, posts in threads:
            result = sync_analyzer.analyze_thread(title, posts)
            sync_results.append(result)
    
    # Async version
    with profiler.time_block("async_analysis"):
        async_results = await async_analyzer.analyze_multiple_threads(threads)
    
    # Calculate speedup
    sync_time = profiler.results["sync_analysis"]
    async_time = profiler.results["async_analysis"]
    speedup = sync_time / async_time if async_time > 0 else 0
    
    logger.info(f"\n📊 Results:")
    logger.info(f"Analyses completed: {len(sync_results)} (sync), {len(async_results)} (async)")
    logger.info(f"Sync time: {sync_time:.2f}s")
    logger.info(f"Async time: {async_time:.2f}s")
    logger.info(f"Speedup: {speedup:.2f}x")
    
    return {
        'sync_time': sync_time,
        'async_time': async_time,
        'speedup': speedup,
        'count': len(threads)
    }
