"""MediaWiki API-based scraper for Wikipedia ANI."""

import re
import json
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import requests
from vote_parser import VoteParser

# Configure logging
logger = logging.getLogger(__name__)


class MediaWikiAPIScraper:
    """Scraper using MediaWiki API for reliable data extraction."""
    
    API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
    ANI_PAGE = "Wikipedia:Administrators' noticeboard/Incidents"
    ANI_ARCHIVE_INDEX = "Wikipedia:Administrators' noticeboard/IncidentArchives"
    
    def __init__(self, user_agent: str = "WikipediaANIReview/1.0", request_delay: float = 1.0):
        """Initialize the API scraper.
        
        Args:
            user_agent: User agent string for requests
            request_delay: Delay between requests in seconds
        """
        self.user_agent = user_agent
        self.request_delay = request_delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent
        })
    
    def _api_request(self, params: Dict) -> Optional[Dict]:
        """Make an API request to MediaWiki.
        
        Args:
            params: Request parameters
            
        Returns:
            JSON response or None if failed
        """
        try:
            time.sleep(self.request_delay)
            params['format'] = 'json'
            response = self.session.get(self.API_ENDPOINT, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request error: {e}")
            return None
    
    def get_page_wikitext(self, page_title: str) -> Optional[str]:
        """Get raw wikitext for a page.
        
        Args:
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
        
        data = self._api_request(params)
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
    
    def get_page_sections(self, page_title: str) -> List[Dict]:
        """Get section structure of a page.
        
        Args:
            page_title: Title of the Wikipedia page
            
        Returns:
            List of section dictionaries
        """
        params = {
            'action': 'parse',
            'page': page_title,
            'prop': 'sections'
        }
        
        data = self._api_request(params)
        if not data or 'parse' not in data:
            return []
        
        return data['parse'].get('sections', [])
    
    def get_section_wikitext(self, page_title: str, section_index: int) -> Optional[str]:
        """Get wikitext for a specific section.
        
        Args:
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
        
        data = self._api_request(params)
        if not data:
            return None
        
        try:
            pages = data['query']['pages']
            page_id = list(pages.keys())[0]
            content = pages[page_id]['revisions'][0]['slots']['main']['*']
            return content
        except (KeyError, IndexError):
            return None
    
    def scrape_current_ani(self) -> List[Dict]:
        """Scrape current ANI page using API.
        
        Returns:
            List of thread dictionaries
        """
        # Get sections
        sections = self.get_page_sections(self.ANI_PAGE)
        
        # Filter level 2 sections (threads)
        thread_sections = [s for s in sections if s.get('toclevel') == 2]
        
        threads = []
        for section in thread_sections:
            thread_data = self._parse_section(self.ANI_PAGE, section, archived=False)
            if thread_data:
                threads.append(thread_data)
        
        return threads
    
    def scrape_archive(self, archive_number: int) -> List[Dict]:
        """Scrape an ANI archive using API.
        
        Args:
            archive_number: Archive number
            
        Returns:
            List of thread dictionaries
        """
        archive_page = f"Wikipedia:Administrators' noticeboard/IncidentArchive{archive_number}"
        
        # Get sections
        sections = self.get_page_sections(archive_page)
        
        # Filter level 2 sections
        thread_sections = [s for s in sections if s.get('toclevel') == 2]
        
        threads = []
        for section in thread_sections:
            thread_data = self._parse_section(archive_page, section, archived=True)
            if thread_data:
                threads.append(thread_data)
        
        return threads
    
    def _parse_section(self, page_title: str, section: Dict, archived: bool) -> Optional[Dict]:
        """Parse a section into thread data.
        
        Args:
            page_title: Page title
            section: Section dictionary from API
            archived: Whether this is from archive
            
        Returns:
            Thread data dictionary or None
        """
        # Get section wikitext
        wikitext = self.get_section_wikitext(page_title, section['index'])
        if not wikitext:
            return None
        
        title = section.get('line', '')
        anchor = section.get('anchor', '')
        
        # Parse the wikitext
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
            'votes': parsed_data.get('votes', []),  # Pass votes along
            'has_voting': parsed_data.get('has_voting', False),
        }
        
        return thread_data
    
    def _parse_wikitext(self, wikitext: str) -> Dict:
        """Parse wikitext to extract structured data.
        
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
        """Count leading colons for indent depth.
        
        Args:
            line: Wikitext line
            
        Returns:
            Indent depth (number of leading colons)
        """
        count = 0
        for char in line:
            if char == ':':
                count += 1
            elif char not in [' ', '\t']:
                break
        return count
    
    def _determine_comment_kind(self, line: str) -> str:
        """Determine the kind of comment.
        
        Args:
            line: Wikitext line
            
        Returns:
            Comment kind (plain/bullet/numbered/template-led)
        """
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
        """Extract user signatures from a line.
        
        Args:
            line: Wikitext line
            
        Returns:
            List of signature dictionaries
        """
        signatures = []
        
        # Pattern for standard Wikipedia signatures
        # Format: [[User:Username|...]] or [[User talk:Username|...]] followed by timestamp
        user_pattern = r'\[\[User(?:[ _]talk)?:([^\]|]+)(?:\|[^\]]+)?\]\]'
        timestamp_pattern = r'(\d{2}:\d{2},\s+\d{1,2}\s+\w+\s+\d{4}\s+\(UTC\))'
        
        # Find all user mentions
        user_matches = list(re.finditer(user_pattern, line))
        timestamp_matches = list(re.finditer(timestamp_pattern, line))
        
        # Try to pair users with timestamps
        for user_match in user_matches:
            username = user_match.group(1).replace('_', ' ')
            timestamp = None
            
            # Look for timestamp after this user mention
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
        """Parse Wikipedia timestamp to datetime.
        
        Args:
            timestamp_str: Timestamp string like "12:34, 1 January 2024 (UTC)"
            
        Returns:
            datetime object or None
        """
        try:
            # Remove (UTC) and parse
            clean_str = timestamp_str.replace('(UTC)', '').strip()
            dt = datetime.strptime(clean_str, '%H:%M, %d %B %Y')
            # Make timezone-aware (UTC)
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            return None
    
    def _extract_targets(self, line: str) -> List[str]:
        """Extract target users from userlinks/user templates.
        
        Args:
            line: Wikitext line
            
        Returns:
            List of target usernames
        """
        targets = []
        
        # {{userlinks|Username}} or {{user|Username}}
        patterns = [
            r'\{\{userlinks\|([^\}|]+)',
            r'\{\{user\|([^\}|]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, line, re.IGNORECASE)
            targets.extend([m.strip() for m in matches])
        
        return targets
    
    def _extract_diffs(self, line: str) -> List[str]:
        """Extract diff URLs from line.
        
        Args:
            line: Wikitext line
            
        Returns:
            List of diff URLs
        """
        # Pattern for Special:Diff URLs
        pattern = r'Special:Diff/(\d+)'
        matches = re.findall(pattern, line, re.IGNORECASE)
        return [f"Special:Diff/{m}" for m in matches]
    
    def _extract_policy_shortcuts(self, line: str) -> List[str]:
        """Extract Wikipedia policy shortcuts (WP:XYZ).
        
        Args:
            line: Wikitext line
            
        Returns:
            List of policy shortcuts
        """
        # Pattern for WP:SHORTCUT or Wikipedia:SHORTCUT
        pattern = r'\b(?:WP|Wikipedia):([A-Z0-9]+)\b'
        matches = re.findall(pattern, line)
        return [f"WP:{m}" for m in matches]
    
    def _extract_noticeboard_refs(self, line: str) -> List[str]:
        """Extract references to other noticeboards.
        
        Args:
            line: Wikitext line
            
        Returns:
            List of noticeboard names
        """
        # Common noticeboards
        noticeboards = ['AN3', 'AIV', 'RFPP', 'SPI', 'AE', 'BLP', 'DR', 'AAR', 'UAA']
        
        found = []
        for nb in noticeboards:
            if nb in line.upper():
                found.append(nb)
        
        return found
    
    def _extract_content_links(self, line: str) -> List[str]:
        """Extract wikilinks to content pages.
        
        Args:
            line: Wikitext line
            
        Returns:
            List of page titles
        """
        # Pattern for [[Page title]] or [[Page title|Display text]]
        pattern = r'\[\[([^\]|:]+)(?:\|[^\]]+)?\]\]'
        matches = re.findall(pattern, line)
        
        # Filter out special pages
        filtered = []
        for match in matches:
            if not any(match.startswith(prefix) for prefix in ['User:', 'User talk:', 'Wikipedia:', 'Special:', 'File:', 'Category:']):
                filtered.append(match.strip())
        
        return filtered
    
    def _extract_unsigned_attribution(self, wikitext: str) -> Optional[str]:
        """Extract user from unsigned comment attribution.
        
        Args:
            wikitext: Full wikitext
            
        Returns:
            Username or None
        """
        # Pattern: {{unsigned|Username}} or similar
        pattern = r'\{\{unsigned[^}]*\|([^\}|]+)'
        match = re.search(pattern, wikitext, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        # Alternative: "—Preceding unsigned comment added by [[User:Username|..."
        alt_pattern = r'Preceding unsigned comment added by.*?\[\[User:([^\]|]+)'
        match = re.search(alt_pattern, wikitext, re.IGNORECASE)
        
        if match:
            return match.group(1).strip().replace('_', ' ')
        
        return None
    
    def _extract_closure_info(self, wikitext: str) -> Dict:
        """Extract closure template information.
        
        Args:
            wikitext: Full wikitext
            
        Returns:
            Dictionary with closure info
        """
        result = {}
        
        # Pattern for {{atop|status=...|result=...}}
        atop_pattern = r'\{\{atop\s*\|[^}]*status\s*=\s*([^|}]+)'
        match = re.search(atop_pattern, wikitext, re.IGNORECASE)
        if match:
            result['outcome_status'] = match.group(1).strip()
        
        # Pattern for result text
        result_pattern = r'\{\{atop[^}]*result\s*=\s*([^|}]+)'
        match = re.search(result_pattern, wikitext, re.IGNORECASE)
        if match:
            result['outcome_text'] = match.group(1).strip()
        
        # Try {{atopg}} variant
        atopg_pattern = r'\{\{atopg\s*\|\s*status\s*=\s*([^|}]+)'
        match = re.search(atopg_pattern, wikitext, re.IGNORECASE)
        if match and 'outcome_status' not in result:
            result['outcome_status'] = match.group(1).strip()
        
        # Extract closing admin
        # Look for "closed by [[User:Admin|Admin]]" or similar
        admin_pattern = r'(?:closed|closing)\s+(?:by|admin).*?\[\[User:([^\]|]+)'
        match = re.search(admin_pattern, wikitext, re.IGNORECASE)
        if match:
            result['closing_admin'] = match.group(1).strip().replace('_', ' ')
        
        return result
    
    def _determine_routing_category(self, wikitext: str) -> Optional[str]:
        """Determine if thread should be routed elsewhere.
        
        Args:
            wikitext: Full wikitext
            
        Returns:
            Routing category or None
        """
        text_lower = wikitext.lower()
        
        # Check for common routing keywords
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
    
    def get_archive_list(self) -> List[int]:
        """Get list of available archive numbers from the archive index.
        
        Returns:
            List of archive numbers
        """
        # Get the archive index page
        wikitext = self.get_page_wikitext(self.ANI_ARCHIVE_INDEX)
        if not wikitext:
            return []
        
        # Extract archive numbers
        # Pattern: IncidentArchive123
        pattern = r'IncidentArchive(\d+)'
        matches = re.findall(pattern, wikitext)
        
        # Convert to integers and return sorted
        archive_numbers = sorted([int(m) for m in matches], reverse=True)
        return archive_numbers
