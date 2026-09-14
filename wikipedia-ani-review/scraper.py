"""Wikipedia ANI scraper module."""

import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
import requests
from bs4 import BeautifulSoup


class ANIScraper:
    """Scraper for Wikipedia Administrator Noticeboard (Incidents)."""
    
    # Main ANI page
    ANI_BASE_URL = "https://en.wikipedia.org"
    ANI_PAGE = "/wiki/Wikipedia:Administrators%27_noticeboard/Incidents"
    
    # Archive pages
    ANI_ARCHIVE_BASE = "/wiki/Wikipedia:Administrators%27_noticeboard/IncidentArchive"
    
    def __init__(self, user_agent: str = "WikipediaANIReview/1.0", request_delay: float = 1.0):
        """Initialize the scraper.
        
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
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """Fetch and parse a Wikipedia page.
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            time.sleep(self.request_delay)
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'lxml')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def scrape_current_ani(self) -> List[Dict]:
        """Scrape the current ANI page for active threads.
        
        Returns:
            List of thread dictionaries
        """
        url = urljoin(self.ANI_BASE_URL, self.ANI_PAGE)
        soup = self._fetch_page(url)
        
        if not soup:
            return []
        
        threads = []
        content_div = soup.find('div', {'id': 'mw-content-text'})
        
        if not content_div:
            return []
        
        # Find all level 2 and 3 headings (thread titles)
        headings = content_div.find_all(['h2', 'h3'])
        
        for heading in headings:
            # Skip special sections
            headline = heading.find('span', {'class': 'mw-headline'})
            if not headline:
                continue
            
            title = headline.get_text(strip=True)
            
            # Skip navigation sections
            if title.lower() in ['contents', 'references', 'external links', 'see also']:
                continue
            
            section_id = headline.get('id', '')
            
            # Get the section URL
            thread_url = f"{url}#{section_id}"
            
            # Extract posts from this section
            posts = self._extract_posts_from_section(heading)
            
            thread_data = {
                'title': title,
                'url': thread_url,
                'section_id': section_id,
                'archived': False,
                'archive_name': None,
                'posts': posts,
                'created_at': self._extract_earliest_timestamp(posts)
            }
            
            threads.append(thread_data)
        
        return threads
    
    def _extract_posts_from_section(self, heading) -> List[Dict]:
        """Extract all posts from a section.
        
        Args:
            heading: BeautifulSoup heading element
            
        Returns:
            List of post dictionaries
        """
        posts = []
        position = 0
        
        # Get all sibling elements until the next heading
        current = heading.find_next_sibling()
        
        while current and current.name not in ['h2', 'h3']:
            # Look for signature patterns
            if current.name in ['p', 'dl', 'ul', 'ol', 'div']:
                # Extract text content with proper formatting
                content = current.get_text(separator=' ', strip=True)
                
                # Find signatures (timestamps and usernames)
                signatures = self._extract_signatures(current)
                
                if content.strip() and signatures:
                    for sig in signatures:
                        post = {
                            'content': content,
                            'author': sig.get('username'),
                            'timestamp': sig.get('timestamp'),
                            'position': position
                        }
                        posts.append(post)
                        position += 1
            
            current = current.find_next_sibling()
        
        return posts
    
    def _extract_signatures(self, element) -> List[Dict]:
        """Extract user signatures from an element.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            List of signature dictionaries
        """
        signatures = []
        
        # Look for user links
        user_links = element.find_all('a', href=re.compile(r'/wiki/User:|/wiki/User_talk:'))
        
        for link in user_links:
            username = None
            timestamp = None
            
            # Extract username from URL
            href = link.get('href', '')
            if '/wiki/User:' in href or '/wiki/User_talk:' in href:
                username = href.split(':', 1)[1].split('/')[0].replace('_', ' ')
            
            # Look for timestamp near the user link
            # Timestamps typically look like: 12:34, 1 January 2024 (UTC)
            parent_text = link.parent.get_text() if link.parent else ''
            timestamp_match = re.search(
                r'(\d{1,2}:\d{2},\s+\d{1,2}\s+\w+\s+\d{4}\s+\(UTC\))',
                parent_text
            )
            
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = self._parse_timestamp(timestamp_str)
            
            if username:
                signatures.append({
                    'username': username,
                    'timestamp': timestamp
                })
        
        return signatures
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Parse Wikipedia timestamp string to datetime.
        
        Args:
            timestamp_str: Timestamp string like "12:34, 1 January 2024 (UTC)"
            
        Returns:
            datetime object or None
        """
        try:
            # Remove (UTC) and parse
            clean_str = timestamp_str.replace('(UTC)', '').strip()
            return datetime.strptime(clean_str, '%H:%M, %d %B %Y')
        except Exception:
            return None
    
    def _extract_earliest_timestamp(self, posts: List[Dict]) -> Optional[datetime]:
        """Get the earliest timestamp from a list of posts.
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Earliest datetime or None
        """
        timestamps = [p['timestamp'] for p in posts if p.get('timestamp')]
        return min(timestamps) if timestamps else None
    
    def find_archive_pages(self, limit: int = 100) -> List[str]:
        """Find all ANI archive page URLs.
        
        Args:
            limit: Maximum number of archive URLs to return
        
        Returns:
            List of archive URLs
        """
        # Wikipedia ANI archives are typically numbered
        # We'll check a range and see which ones exist
        archives = []
        
        # Check archives from 1 to limit
        # In production, you might want to scrape the archive index
        for i in range(1, limit + 1):
            archive_url = f"{self.ANI_BASE_URL}{self.ANI_ARCHIVE_BASE}{i}"
            archives.append(archive_url)
        
        return archives
    
    def scrape_archive(self, archive_url: str) -> List[Dict]:
        """Scrape an ANI archive page.
        
        Args:
            archive_url: URL of the archive page
            
        Returns:
            List of thread dictionaries
        """
        soup = self._fetch_page(archive_url)
        
        if not soup:
            return []
        
        # Extract archive name from URL
        archive_name = archive_url.split('/')[-1]
        
        threads = []
        content_div = soup.find('div', {'id': 'mw-content-text'})
        
        if not content_div:
            return []
        
        # Find all level 2 and 3 headings
        headings = content_div.find_all(['h2', 'h3'])
        
        for heading in headings:
            headline = heading.find('span', {'class': 'mw-headline'})
            if not headline:
                continue
            
            title = headline.get_text(strip=True)
            
            # Skip navigation sections
            if title.lower() in ['contents', 'references', 'external links', 'see also']:
                continue
            
            section_id = headline.get('id', '')
            thread_url = f"{archive_url}#{section_id}"
            
            posts = self._extract_posts_from_section(heading)
            
            thread_data = {
                'title': title,
                'url': thread_url,
                'section_id': section_id,
                'archived': True,
                'archive_name': archive_name,
                'posts': posts,
                'created_at': self._extract_earliest_timestamp(posts)
            }
            
            threads.append(thread_data)
        
        return threads
