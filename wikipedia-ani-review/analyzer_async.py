"""Async LLM-based analysis module for categorizing ANI threads.

This module provides asynchronous analysis capabilities for concurrent
LLM API calls with rate limiting.

Performance Benefits:
- I/O-bound: API calls benefit greatly from async concurrency
- Can analyze multiple threads concurrently
- NO GIL: Not needed for API I/O - async is sufficient

CPU-bound sections (NO GIL opportunities):
- Text preprocessing and cleanup
- JSON parsing and validation
- Keyword-based mock analysis
These could benefit from true threading in free-threaded Python.
"""

import json
import os
import asyncio
from typing import Dict, List, Optional
from openai import AsyncOpenAI

import logging
logger = logging.getLogger(__name__)


class AsyncThreadAnalyzer:
    """Async analyzer for ANI threads using LLM."""
    
    # Configuration constants
    MAX_POST_LENGTH = 500
    
    # Category definitions
    CATEGORIES = [
        "Edit warring",
        "Personal attacks",
        "Disruptive editing",
        "Copyright violations",
        "Sockpuppetry",
        "Vandalism",
        "Harassment",
        "NPOV violations",
        "Canvassing",
        "Gaming the system",
        "Other conduct issues"
    ]
    
    REQUESTED_OUTCOMES = [
        "Block",
        "Topic ban",
        "Interaction ban",
        "Warning",
        "Arbitration",
        "Mediation",
        "Community discussion",
        "No action",
        "Other"
    ]
    
    TOPIC_AREAS = [
        "Biography",
        "Politics",
        "Religion",
        "Science",
        "History",
        "Entertainment",
        "Sports",
        "Geography",
        "Technology",
        "Current events",
        "Wikipedia policy",
        "Other"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        max_concurrent: int = 3
    ):
        """Initialize the async analyzer.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: OpenAI model to use
            max_concurrent: Max concurrent API calls (rate limiting)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        self.max_concurrent = max_concurrent
        
        # Semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        if self.api_key and self.api_key != 'your_api_key_here':
            self.client = AsyncOpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    async def analyze_thread(
        self,
        thread_title: str,
        posts: List[Dict]
    ) -> Dict:
        """Analyze a thread and categorize it.
        
        Args:
            thread_title: Title of the thread
            posts: List of post dictionaries with 'content' and 'author'
            
        Returns:
            Analysis dictionary with categories and outcomes
        """
        if not self.client:
            # Return mock analysis if no API key
            return self._mock_analysis(thread_title, posts)
        
        # Use semaphore for rate limiting
        async with self.semaphore:
            # Prepare the content for analysis
            # NO GIL OPPORTUNITY: Text preprocessing is CPU-bound
            thread_content = self._prepare_thread_content(thread_title, posts)
            
            # Create the prompt
            prompt = self._create_analysis_prompt(thread_content)
            
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing Wikipedia Administrator Noticeboard discussions. "
                                       "You categorize incidents and identify requested outcomes."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                
                # Parse the response
                # NO GIL OPPORTUNITY: JSON parsing is CPU-bound
                analysis_text = response.choices[0].message.content
                return self._parse_analysis_response(analysis_text)
                
            except Exception as e:
                logger.error(f"Error during LLM analysis: {e}")
                return self._mock_analysis(thread_title, posts)
    
    async def analyze_multiple_threads(
        self,
        threads: List[tuple]
    ) -> List[Dict]:
        """Analyze multiple threads concurrently.
        
        This demonstrates the key async benefit for LLM analysis:
        - Can make multiple API calls in parallel
        - Significant speedup vs sequential analysis
        - Respects rate limits via semaphore
        
        Args:
            threads: List of (thread_title, posts) tuples
            
        Returns:
            List of analysis dictionaries
        """
        tasks = [
            self.analyze_thread(title, posts)
            for title, posts in threads
        ]
        
        # Gather all analyses concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        analyses = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Analysis error: {result}")
                # Add empty analysis for failed threads
                analyses.append(self._empty_analysis())
            else:
                analyses.append(result)
        
        return analyses
    
    def _prepare_thread_content(
        self,
        title: str,
        posts: List[Dict],
        max_posts: int = 10
    ) -> str:
        """Prepare thread content for analysis.
        
        NO GIL OPPORTUNITY: String concatenation and slicing is CPU-bound.
        In free-threaded Python, this could run in parallel with other prep work.
        """
        content = f"Thread Title: {title}\n\n"
        
        for i, post in enumerate(posts[:max_posts]):
            author = post.get('author', 'Unknown')
            text = post.get('content', '')[:self.MAX_POST_LENGTH]
            content += f"Post {i+1} by {author}:\n{text}\n\n"
        
        if len(posts) > max_posts:
            content += f"... ({len(posts) - max_posts} more posts)\n"
        
        return content
    
    def _create_analysis_prompt(self, thread_content: str) -> str:
        """Create the analysis prompt."""
        categories_str = ", ".join(self.CATEGORIES)
        outcomes_str = ", ".join(self.REQUESTED_OUTCOMES)
        topics_str = ", ".join(self.TOPIC_AREAS)
        
        prompt = f"""Analyze the following Wikipedia ANI (Administrator Noticeboard/Incidents) discussion:

{thread_content}

Please provide:
1. Primary Category: Choose ONE from [{categories_str}]
2. Secondary Categories: List any additional applicable categories (comma-separated)
3. Requested Outcome: Choose ONE from [{outcomes_str}]
4. Topic Area: Choose ONE from [{topics_str}]
5. Confidence Score: Rate your confidence in this analysis from 0.0 to 1.0

Respond in JSON format:
{{
    "primary_category": "...",
    "secondary_categories": ["...", "..."],
    "requested_outcome": "...",
    "topic_area": "...",
    "confidence_score": 0.0
}}"""
        
        return prompt
    
    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse the LLM response.
        
        NO GIL OPPORTUNITY: JSON parsing and string manipulation is CPU-bound.
        """
        try:
            # Try to extract JSON from the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
                
                return {
                    'primary_category': analysis.get('primary_category', 'Other conduct issues'),
                    'secondary_categories': analysis.get('secondary_categories', []),
                    'requested_outcome': analysis.get('requested_outcome', 'Other'),
                    'topic_area': analysis.get('topic_area', 'Other'),
                    'confidence_score': analysis.get('confidence_score', 0.5),
                    'raw_analysis': response_text
                }
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
        
        # Fallback to basic analysis
        return {
            'primary_category': 'Other conduct issues',
            'secondary_categories': [],
            'requested_outcome': 'Other',
            'topic_area': 'Other',
            'confidence_score': 0.0,
            'raw_analysis': response_text
        }
    
    def _mock_analysis(self, title: str, posts: List[Dict]) -> Dict:
        """Generate mock analysis when LLM is not available.
        
        NO GIL OPPORTUNITY: Keyword matching and string processing is CPU-bound.
        In free-threaded Python, multiple mock analyses could run in true parallel.
        """
        # Simple keyword-based categorization
        title_lower = title.lower()
        content_lower = " ".join([p.get('content', '')[:200] for p in posts[:5]]).lower()
        combined = title_lower + " " + content_lower
        
        # Determine primary category
        if any(word in combined for word in ['edit war', 'revert', '3rr']):
            primary = 'Edit warring'
        elif any(word in combined for word in ['attack', 'harassment', 'abuse']):
            primary = 'Personal attacks'
        elif any(word in combined for word in ['vandal', 'vandalism']):
            primary = 'Vandalism'
        elif any(word in combined for word in ['sock', 'puppet', 'meatpuppet']):
            primary = 'Sockpuppetry'
        elif any(word in combined for word in ['disrupt', 'disruptive']):
            primary = 'Disruptive editing'
        else:
            primary = 'Other conduct issues'
        
        # Determine requested outcome
        if any(word in combined for word in ['block', 'ban', 'indef']):
            outcome = 'Block'
        elif 'topic ban' in combined:
            outcome = 'Topic ban'
        elif 'interaction ban' in combined:
            outcome = 'Interaction ban'
        elif any(word in combined for word in ['warn', 'warning']):
            outcome = 'Warning'
        else:
            outcome = 'Other'
        
        return {
            'primary_category': primary,
            'secondary_categories': [],
            'requested_outcome': outcome,
            'topic_area': 'Other',
            'confidence_score': 0.3,
            'raw_analysis': 'Mock analysis based on keywords'
        }
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis for failed threads."""
        return {
            'primary_category': 'Other conduct issues',
            'secondary_categories': [],
            'requested_outcome': 'Other',
            'topic_area': 'Other',
            'confidence_score': 0.0,
            'raw_analysis': 'Analysis failed'
        }
