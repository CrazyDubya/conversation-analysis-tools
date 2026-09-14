"""Async MediaWiki API-based scraper for Wikipedia ANI.

This module provides asynchronous scraping capabilities using aiohttp
for concurrent HTTP requests while respecting rate limits.

Performance Benefits:
- I/O-bound: Async provides significant speedup for network requests
- Can fetch multiple sections/threads concurrently
- NO GIL: Not needed here - async handles I/O concurrency efficiently

Rate Limiting:
- Configurable semaphore limits concurrent requests
- Delay between requests to respect Wikipedia's rate limits
- Default: 5 concurrent requests, 1.0s delay per request
"""

import re
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import aiohttp
from vote_parser import VoteParser

# Configure logging
logger = logging.getLogger(__name__)


class AsyncMediaWikiAPIScraper:
    """Async scraper using MediaWiki API for reliable data extraction."""
    
    API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
    ANI_PAGE = "Wikipedia:Administrators' noticeboard/Incidents"
    ANI_ARCHIVE_INDEX = "Wikipedia:Administrators' noticeboard/IncidentArchives"
    
    def __init__(
        self,
        user_agent: str = "WikipediaANIReview/1.0",
        request_delay: float = 1.0,
        max_concurrent: int = 5
    ):
        """Initialize the async API scraper.
        
        Args:
            user_agent: User agent string for requests
            request_delay: Delay between requests in seconds (rate limiting)
            max_concurrent: Maximum concurrent requests (default: 5)
        """
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.max_concurrent = max_concurrent
        
        # Semaphore for controlling concurrent requests
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Rate limiting: track last request time
        self.last_request_time = 0
        self.rate_limit_lock = asyncio.Lock()
    
    async def _rate_limited_request(self):
        """Ensure rate limiting between requests."""
        async with self.rate_limit_lock:
            # Calculate time since last request
            now = asyncio.get_event_loop().time()
            time_since_last = now - self.last_request_time
            
            # If not enough time has passed, wait
            if time_since_last < self.request_delay:
                wait_time = self.request_delay - time_since_last
                await asyncio.sleep(wait_time)
            
            # Update last request time
            self.last_request_time = asyncio.get_event_loop().time()
    
    async def _api_request(
        self,
        session: aiohttp.ClientSession,
        params: Dict
    ) -> Optional[Dict]:
        """Make an async API request to MediaWiki.
        
        Args:
            session: aiohttp client session
            params: Request parameters
            
        Returns:
            JSON response or None if failed
        """
        # Use semaphore to limit concurrent requests
        async with self.semaphore:
            # Apply rate limiting
            await self._rate_limited_request()
            
            try:
                params['format'] = 'json'
                async with session.get(
                    self.API_ENDPOINT,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except Exception as e:
                logger.error(f"API request error: {e}")
                return None
    
    async def get_page_wikitext(
        self,
        session: aiohttp.ClientSession,
        page_title: str
    ) -> Optional[str]:
        """Get raw wikitext for a page.
        
        Args:
            session: aiohttp client session
            page_title: Title of the Wikipedia page
            
        Returns:
            Wikitext content or None
        """
        params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': page_title,
            'rvprop': 'content',
            'rvslots': 'main'
        }
        
        data = await self._api_request(session, params)
        if not data:
            return None
        
        try:
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            if page_id == '-1':
                return None
            
            content = pages[page_id]['revisions'][0]['slots']['main']['*']
            return content
        except (KeyError, IndexError):
            return None
    
    async def get_page_sections(
        self,
        session: aiohttp.ClientSession,
        page_title: str
    ) -> List[Dict]:
        """Get section structure of a page.
        
        Args:
            session: aiohttp client session
            page_title: Title of the Wikipedia page
            
        Returns:
            List of section dictionaries
        """
        params = {
            'action': 'parse',
            'page': page_title,
            'prop': 'sections'
        }
        
        data = await self._api_request(session, params)
        if not data or 'parse' not in data:
            return []
        
        return data['parse'].get('sections', [])
    
    async def get_section_wikitext(
        self,
        session: aiohttp.ClientSession,
        page_title: str,
        section_index: int
    ) -> Optional[str]:
        """Get wikitext for a specific section.
        
        Args:
            session: aiohttp client session
            page_title: Title of the Wikipedia page
            section_index: Section index number
            
        Returns:
            Section wikitext or None
        """
        params = {
            'action': 'query',
            'prop': 'revisions',
            'titles': page_title,
            'rvprop': 'content',
            'rvsection': str(section_index),
            'rvslots': 'main'
        }
        
        data = await self._api_request(session, params)
        if not data:
            return None
        
        try:
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            content = pages[page_id]['revisions'][0]['slots']['main']['*']
            return content
        except (KeyError, IndexError):
            return None
    
    async def scrape_current_ani(self) -> List[Dict]:
        """Scrape current ANI page using async API.
        
        This method demonstrates async benefits:
        - Fetches section list first
        - Then fetches all section content concurrently
        - Significant speedup vs sequential (N sections -> ~N/5 time with 5 concurrent)
        
        Returns:
            List of thread dictionaries
        """
        async with aiohttp.ClientSession(
            headers={'User-Agent': self.user_agent}
        ) as session:
            # Get sections
            sections = await self.get_page_sections(session, self.ANI_PAGE)
            
            # Filter level 2 sections (threads)
            thread_sections = [s for s in sections if s.get('toclevel') == 2]
            
            # Parse all sections concurrently - KEY ASYNC BENEFIT
            tasks = [
                self._parse_section(session, self.ANI_PAGE, section, archived=False)
                for section in thread_sections
            ]
            
            # Gather results concurrently
            thread_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out None and exceptions
            threads = [
                t for t in thread_results
                if t is not None and not isinstance(t, Exception)
            ]
            
            return threads
    
    async def scrape_archive(self, archive_number: int) -> List[Dict]:
        """Scrape an ANI archive using async API.
        
        Args:
            archive_number: Archive number
            
        Returns:
            List of thread dictionaries
        """
        archive_page = f"Wikipedia:Administrators' noticeboard/IncidentArchive{archive_number}"
        
        async with aiohttp.ClientSession(
            headers={'User-Agent': self.user_agent}
        ) as session:
            # Get sections
            sections = await self.get_page_sections(session, archive_page)
            
            # Filter level 2 sections
            thread_sections = [s for s in sections if s.get('toclevel') == 2]
            
            # Parse all sections concurrently
            tasks = [
                self._parse_section(session, archive_page, section, archived=True)
                for section in thread_sections
            ]
            
            thread_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            threads = [
                t for t in thread_results
                if t is not None and not isinstance(t, Exception)
            ]
            
            return threads
    
    async def scrape_multiple_archives(
        self,
        archive_numbers: List[int]
    ) -> List[Dict]:
        """Scrape multiple archives concurrently.
        
        This demonstrates major async benefit:
        - N archives scraped in parallel (up to max_concurrent limit)
        - Much faster than sequential scraping
        
        Args:
            archive_numbers: List of archive numbers to scrape
            
        Returns:
            Combined list of thread dictionaries from all archives
        """
        tasks = [
            self.scrape_archive(archive_num)
            for archive_num in archive_numbers
        ]
        
        # Gather all archive results
        archive_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_threads = []
        for result in archive_results:
            if isinstance(result, list):
                all_threads.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Archive scraping error: {result}")
        
        return all_threads
    
    async def _parse_section(
        self,
        session: aiohttp.ClientSession,
        page_title: str,
        section: Dict,
        archived: bool
    ) -> Optional[Dict]:
        """Parse a section into thread data.
        
        Args:
            session: aiohttp client session
            page_title: Page title
            section: Section dictionary from API
            archived: Whether this is from archive
            
        Returns:
            Thread data dictionary or None
        """
        # Get section wikitext
        wikitext = await self.get_section_wikitext(
            session,
            page_title,
            section['index']
        )
        if not wikitext:
            return None
        
        title = section.get('line', '')
        anchor = section.get('anchor', '')
        
        # Parse the wikitext - CPU-bound section
        # NO GIL OPPORTUNITY: This parsing could benefit from true parallelism
        # if moved to separate thread pool for CPU-bound work
        parsed_data = self._parse_wikitext(wikitext)
        
        # Build thread URL
        base_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        thread_url = f"{base_url}#{anchor}"
        
        thread_data = {
            'title': title,
            'url': thread_url,
            'section_id': anchor,
            'thread_anchor': anchor,
            'archived': archived,
            'archive_name': page_title.split('/')[-1] if archived else None,
            'raw_wikitext': wikitext,
            'posts': parsed_data['posts'],
            'created_at': parsed_data.get('filer_timestamp'),
            'filer_user': parsed_data.get('filer_user'),
            'filer_timestamp': parsed_data.get('filer_timestamp'),
            'targets': json.dumps(parsed_data.get('targets', [])),
            'diffs': json.dumps(parsed_data.get('diffs', [])),
            'policy_shortcuts': json.dumps(parsed_data.get('policy_shortcuts', [])),
            'noticeboard_refs': json.dumps(parsed_data.get('noticeboard_refs', [])),
            'content_area': json.dumps(parsed_data.get('content_area', [])),
            'outcome_status': parsed_data.get('outcome_status'),
            'outcome_text': parsed_data.get('outcome_text'),
            'closing_admin': parsed_data.get('closing_admin'),
            'closure_timestamp': parsed_data.get('closure_timestamp'),
            'category_routing_suggested': parsed_data.get('category_routing_suggested'),
            'votes': parsed_data.get('votes', []),
            'has_voting': parsed_data.get('has_voting', False),
        }
        
        return thread_data
    
    def _parse_wikitext(self, wikitext: str) -> Dict:
        """Parse wikitext to extract structured data.
        
        NOTE: This is CPU-bound parsing work (regex, string processing).
        NO GIL OPPORTUNITY: If running on free-threaded Python (PEP 703),
        this could run in true parallel threads for better performance.
        Currently, asyncio doesn't help here - it's not I/O bound.
        
        Args:
            wikitext: Raw wikitext content
            
        Returns:
            Dictionary with parsed data
        """
        result = {
            'posts': [],
            'targets': [],
            'diffs': [],
            'policy_shortcuts': [],
            'noticeboard_refs': [],
            'content_area': [],
            'votes': [],
            'has_voting': False,
        }
        
        # Check for voting structure
        result['has_voting'] = VoteParser.has_voting_structure(wikitext)
        
        # Parse votes if present
        if result['has_voting']:
            result['votes'] = VoteParser.parse_votes(wikitext)
        
        # Split into lines for parsing
        lines = wikitext.split('\n')
        
        position = 0
        for line in lines:
            if not line.strip():
                continue
            
            # Parse indent depth
            indent_depth = self._count_indent(line)
            
            # Determine comment kind
            comment_kind = self._determine_comment_kind(line)
            
            # Extract signatures from this line
            signatures = self._extract_signatures_from_line(line)
            
            # Extract targets from userlinks templates
            targets = self._extract_targets(line)
            result['targets'].extend(targets)
            
            # Extract diffs
            diffs = self._extract_diffs(line)
            result['diffs'].extend(diffs)
            
            # Extract policy shortcuts
            shortcuts = self._extract_policy_shortcuts(line)
            result['policy_shortcuts'].extend(shortcuts)
            
            # Extract noticeboard references
            nb_refs = self._extract_noticeboard_refs(line)
            result['noticeboard_refs'].extend(nb_refs)
            
            # Extract content area links
            content_links = self._extract_content_links(line)
            result['content_area'].extend(content_links)
            
            # Create post for each signature found
            for sig in signatures:
                post = {
                    'content': line,
                    'author': sig['username'],
                    'timestamp': sig['timestamp'],
                    'position': position,
                    'indent_depth': indent_depth,
                    'comment_kind': comment_kind,
                    'raw_wikitext': line
                }
                result['posts'].append(post)
                position += 1
                
                # First post is the filer
                if position == 1:
                    result['filer_user'] = sig['username']
                    result['filer_timestamp'] = sig['timestamp']
        
        # Check for unsigned attribution
        if not result['posts']:
            unsigned_user = self._extract_unsigned_attribution(wikitext)
            if unsigned_user:
                result['filer_user'] = unsigned_user
        
        # Extract closure information
        closure_info = self._extract_closure_info(wikitext)
        result.update(closure_info)
        
        # Determine routing category
        result['category_routing_suggested'] = self._determine_routing_category(wikitext)
        
        # Deduplicate lists
        result['targets'] = list(set(result['targets']))
        result['diffs'] = list(set(result['diffs']))
        result['policy_shortcuts'] = list(set(result['policy_shortcuts']))
        result['noticeboard_refs'] = list(set(result['noticeboard_refs']))
        result['content_area'] = list(set(result['content_area']))
        
        return result
    
    def _count_indent(self, line: str) -> int:
        """Count leading colons for indent depth."""
        count = 0
        for char in line:
            if char == ':':
                count += 1
            elif char not in [' ', '\t']:
                break
        return count
    
    def _determine_comment_kind(self, line: str) -> str:
        """Determine the kind of comment."""
        stripped = line.lstrip(':').lstrip()
        
        if stripped.startswith('*'):
            return 'bullet'
        elif stripped.startswith('#'):
            return 'numbered'
        elif stripped.startswith('{{'):
            return 'template-led'
        else:
            return 'plain'
    
    def _extract_signatures_from_line(self, line: str) -> List[Dict]:
        """Extract user signatures from a line."""
        signatures = []
        
        user_pattern = r'\[\[User(?:[ _]talk)?:([^\]|]+)(?:\|[^\]]+)?\]\]'
        timestamp_pattern = r'(\d{2}:\d{2},\s+\d{1,2}\s+\w+\s+\d{4}\s+\(UTC\))'
        
        user_matches = list(re.finditer(user_pattern, line))
        timestamp_matches = list(re.finditer(timestamp_pattern, line))
        
        for user_match in user_matches:
            username = user_match.group(1).replace('_', ' ')
            timestamp = None
            
            for ts_match in timestamp_matches:
                if ts_match.start() > user_match.end():
                    timestamp_str = ts_match.group(1)
                    timestamp = self._parse_timestamp(timestamp_str)
                    break
            
            if username:
                signatures.append({
                    'username': username,
                    'timestamp': timestamp
                })
        
        return signatures
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse Wikipedia timestamp to datetime."""
        try:
            clean_str = timestamp_str.replace('(UTC)', '').strip()
            dt = datetime.strptime(clean_str, '%H:%M, %d %B %Y')
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return None
    
    def _extract_targets(self, line: str) -> List[str]:
        """Extract target users from userlinks/user templates."""
        targets = []
        
        patterns = [
            r'\{\{userlinks\|([^\}|]+)',
            r'\{\{user\|([^\}|]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            targets.extend([m.strip() for m in matches])
        
        return targets
    
    def _extract_diffs(self, line: str) -> List[str]:
        """Extract diff URLs from line."""
        pattern = r'Special:Diff/(\d+)'
        matches = re.findall(pattern, line, re.IGNORECASE)
        return [f"Special:Diff/{m}" for m in matches]
    
    def _extract_policy_shortcuts(self, line: str) -> List[str]:
        """Extract Wikipedia policy shortcuts (WP:XYZ)."""
        pattern = r'\b(?:WP|Wikipedia):([A-Z0-9]+)\b'
        matches = re.findall(pattern, line)
        return [f"WP:{m}" for m in matches]
    
    def _extract_noticeboard_refs(self, line: str) -> List[str]:
        """Extract references to other noticeboards."""
        noticeboards = ['AN3', 'AIV', 'RFPP', 'SPI', 'AE', 'BLP', 'DR', 'AAR', 'UAA']
        
        found = []
        for nb in noticeboards:
            if nb in line.upper():
                found.append(nb)
        
        return found
    
    def _extract_content_links(self, line: str) -> List[str]:
        """Extract wikilinks to content pages."""
        pattern = r'\[\[([^\]|:]+)(?:\|[^\]]+)?\]\]'
        matches = re.findall(pattern, line)
        
        filtered = []
        for match in matches:
            if not any(match.startswith(prefix) for prefix in ['User:', 'User talk:', 'Wikipedia:', 'Special:', 'File:', 'Category:']):
                filtered.append(match.strip())
        
        return filtered
    
    def _extract_unsigned_attribution(self, wikitext: str) -> Optional[str]:
        """Extract user from unsigned comment attribution."""
        pattern = r'\{\{unsigned[^}]*\|([^\}|]+)'
        match = re.search(pattern, wikitext, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        alt_pattern = r'Preceding unsigned comment added by.*?\[\[User:([^\]|]+)'
        match = re.search(alt_pattern, wikitext, re.IGNORECASE)
        
        if match:
            return match.group(1).strip().replace('_', ' ')
        
        return None
    
    def _extract_closure_info(self, wikitext: str) -> Dict:
        """Extract closure template information."""
        result = {}
        
        atop_pattern = r'\{\{atop\s*\|[^}]*status\s*=\s*([^|}]+)'
        match = re.search(atop_pattern, wikitext, re.IGNORECASE)
        if match:
            result['outcome_status'] = match.group(1).strip()
        
        result_pattern = r'\{\{atop[^}]*result\s*=\s*([^|}]+)'
        match = re.search(result_pattern, wikitext, re.IGNORECASE)
        if match:
            result['outcome_text'] = match.group(1).strip()
        
        atopg_pattern = r'\{\{atopg\s*\|\s*status\s*=\s*([^|}]+)'
        match = re.search(atopg_pattern, wikitext, re.IGNORECASE)
        if match and 'outcome_status' not in result:
            result['outcome_status'] = match.group(1).strip()
        
        admin_pattern = r'(?:closed|closing)\s+(?:by|admin).*?\[\[User:([^\]|]+)'
        match = re.search(admin_pattern, wikitext, re.IGNORECASE)
        if match:
            result['closing_admin'] = match.group(1).strip().replace('_', ' ')
        
        return result
    
    def _determine_routing_category(self, wikitext: str) -> Optional[str]:
        """Determine if thread should be routed elsewhere."""
        text_lower = wikitext.lower()
        
        routing_map = {
            'BLP': ['wp:blp', 'blp noticeboard', 'living person'],
            'AN3': ['3rr', 'three revert', 'edit war'],
            'AIV': ['vandalism', 'aiv'],
            'RFPP': ['protection', 'page protection', 'rfpp'],
            'SPI': ['sock', 'sockpuppet', 'meatpuppet', 'spi'],
            'AE': ['arbitration enforcement', 'ae'],
            'DR': ['dispute resolution', 'content dispute'],
            'UAA': ['username', 'uaa']
        }
        
        for category, keywords in routing_map.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return None
    
    async def get_archive_list(self) -> List[int]:
        """Get list of available archive numbers from the archive index.
        
        Returns:
            List of archive numbers
        """
        async with aiohttp.ClientSession(
            headers={'User-Agent': self.user_agent}
        ) as session:
            wikitext = await self.get_page_wikitext(session, self.ANI_ARCHIVE_INDEX)
            if not wikitext:
                return []
            
            pattern = r'IncidentArchive(\d+)'
            matches = re.findall(pattern, wikitext)
            
            archive_numbers = sorted([int(m) for m in matches], reverse=True)
            return archive_numbers
