"""LLM-based analysis module for categorizing ANI threads."""

import json
import os
from typing import Dict, List, Optional
from openai import OpenAI


class ThreadAnalyzer:
    """Analyze ANI threads using LLM to categorize topics and outcomes."""
    
    # Configuration constants
    MAX_POST_LENGTH = 500  # Maximum characters per post to include in analysis
    
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
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize the analyzer.
        
        Args:
            api_key: OpenAI API key (if None, reads from environment)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        
        if self.api_key and self.api_key != 'your_api_key_here':
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def analyze_thread(self, thread_title: str, posts: List[Dict]) -> Dict:
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
        
        # Prepare the content for analysis
        thread_content = self._prepare_thread_content(thread_title, posts)
        
        # Create the prompt
        prompt = self._create_analysis_prompt(thread_content)
        
        try:
            response = self.client.chat.completions.create(
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
            analysis_text = response.choices[0].message.content
            return self._parse_analysis_response(analysis_text)
            
        except Exception as e:
            print(f"Error during LLM analysis: {e}")
            return self._mock_analysis(thread_title, posts)
    
    def _prepare_thread_content(self, title: str, posts: List[Dict], max_posts: int = 10) -> str:
        """Prepare thread content for analysis.
        
        Args:
            title: Thread title
            posts: List of posts
            max_posts: Maximum number of posts to include
            
        Returns:
            Formatted thread content string
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
        """Create the analysis prompt.
        
        Args:
            thread_content: Formatted thread content
            
        Returns:
            Prompt string
        """
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
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed analysis dictionary
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
            print(f"Error parsing LLM response: {e}")
        
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
        
        Args:
            title: Thread title
            posts: List of posts
            
        Returns:
            Mock analysis dictionary
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
