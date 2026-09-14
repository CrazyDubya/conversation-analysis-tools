"""Database operations for storing and retrieving ANI data."""

import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from models import User, Thread, Post, ThreadAnalysis, UserActivity, Vote, FilingOutcome


def utc_now():
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class DatabaseManager:
    """Manager for database operations."""
    
    def __init__(self, session: Session):
        """Initialize the database manager.
        
        Args:
            session: SQLAlchemy session
        """
        self.session = session
    
    def get_or_create_user(self, username: str) -> User:
        """Get existing user or create new one.
        
        Args:
            username: Wikipedia username
            
        Returns:
            User object
        """
        user = self.session.query(User).filter_by(username=username).first()
        
        if not user:
            user = User(username=username)
            self.session.add(user)
            self.session.commit()
        
        return user
    
    def create_thread(self, thread_data: Dict) -> Thread:
        """Create a new thread in the database.
        
        Args:
            thread_data: Dictionary with thread information
            
        Returns:
            Thread object
        """
        # Check if thread already exists
        existing = self.session.query(Thread).filter_by(url=thread_data['url']).first()
        if existing:
            return existing
        
        # Get or create creator
        creator = None
        if thread_data.get('posts') and len(thread_data['posts']) > 0:
            first_post = thread_data['posts'][0]
            if first_post.get('author'):
                creator = self.get_or_create_user(first_post['author'])
                creator.threads_created += 1
        
        # Get or create filer if specified
        filer_user = thread_data.get('filer_user')
        if filer_user and (not creator or creator.username != filer_user):
            creator = self.get_or_create_user(filer_user)
            creator.threads_created += 1
        
        # Create thread
        thread = Thread(
            title=thread_data['title'],
            url=thread_data['url'],
            creator=creator,
            created_at=thread_data.get('created_at'),
            archived=1 if thread_data.get('archived') else 0,
            archive_name=thread_data.get('archive_name'),
            section_id=thread_data.get('section_id'),
            thread_anchor=thread_data.get('thread_anchor'),
            filer_user=thread_data.get('filer_user'),
            filer_timestamp=thread_data.get('filer_timestamp'),
            targets=thread_data.get('targets'),
            diffs=thread_data.get('diffs'),
            policy_shortcuts=thread_data.get('policy_shortcuts'),
            noticeboard_refs=thread_data.get('noticeboard_refs'),
            content_area=thread_data.get('content_area'),
            outcome_status=thread_data.get('outcome_status'),
            outcome_text=thread_data.get('outcome_text'),
            closing_admin=thread_data.get('closing_admin'),
            closure_timestamp=thread_data.get('closure_timestamp'),
            category_routing_suggested=thread_data.get('category_routing_suggested'),
            raw_wikitext=thread_data.get('raw_wikitext'),
            num_posts=len(thread_data.get('posts', []))
        )
        
        self.session.add(thread)
        self.session.commit()
        
        # Add posts
        participants = set()
        for post_data in thread_data.get('posts', []):
            author = None
            if post_data.get('author'):
                author = self.get_or_create_user(post_data['author'])
                author.posts_made += 1
                author.last_seen = utc_now()
                participants.add(author.username)
            
            post = Post(
                thread=thread,
                author=author,
                content=post_data.get('content', ''),
                posted_at=post_data.get('timestamp'),
                position=post_data.get('position', 0),
                indent_depth=post_data.get('indent_depth', 0),
                comment_kind=post_data.get('comment_kind'),
                raw_wikitext=post_data.get('raw_wikitext')
            )
            self.session.add(post)
        
        thread.num_participants = len(participants)
        self.session.commit()
        
        # Store votes if present
        votes_data = thread_data.get('votes', [])
        if votes_data:
            self._store_votes(thread, votes_data)
        
        return thread
    
    def _store_votes(self, thread: Thread, votes_data: List[Dict]):
        """Store votes for a thread.
        
        Args:
            thread: Thread object
            votes_data: List of vote dictionaries
        """
        for vote_data in votes_data:
            # Get or create voter
            voter_username = vote_data.get('voter_user')
            if not voter_username:
                continue
            
            voter = self.get_or_create_user(voter_username)
            
            # Create vote record
            vote = Vote(
                thread=thread,
                proposal_id=vote_data.get('proposal_id'),
                voter=voter,
                vote_timestamp=vote_data.get('vote_timestamp'),
                stance=vote_data.get('stance'),
                strength_modifier=vote_data.get('strength_modifier')
            )
            self.session.add(vote)
        
        self.session.commit()
    
    def store_thread_analysis(self, thread: Thread, analysis: Dict) -> ThreadAnalysis:
        """Store LLM analysis for a thread.
        
        Args:
            thread: Thread object
            analysis: Analysis dictionary
            
        Returns:
            ThreadAnalysis object
        """
        # Check if analysis exists
        existing = self.session.query(ThreadAnalysis).filter_by(thread_id=thread.id).first()
        
        if existing:
            # Update existing
            existing.primary_category = analysis.get('primary_category')
            existing.secondary_categories = ','.join(analysis.get('secondary_categories', []))
            existing.requested_outcome = analysis.get('requested_outcome')
            existing.topic_area = analysis.get('topic_area')
            existing.confidence_score = analysis.get('confidence_score')
            existing.raw_analysis = analysis.get('raw_analysis')
            existing.analyzed_at = utc_now()
            self.session.commit()
            return existing
        else:
            # Create new
            thread_analysis = ThreadAnalysis(
                thread=thread,
                primary_category=analysis.get('primary_category'),
                secondary_categories=','.join(analysis.get('secondary_categories', [])),
                requested_outcome=analysis.get('requested_outcome'),
                topic_area=analysis.get('topic_area'),
                confidence_score=analysis.get('confidence_score'),
                raw_analysis=analysis.get('raw_analysis')
            )
            self.session.add(thread_analysis)
            self.session.commit()
            return thread_analysis
    
    def get_user_statistics(self, username: str = None) -> List[Dict]:
        """Get user activity statistics.
        
        Args:
            username: Specific username (if None, returns all users)
            
        Returns:
            List of user statistics dictionaries
        """
        query = self.session.query(User)
        
        if username:
            query = query.filter_by(username=username)
        
        users = query.all()
        
        stats = []
        for user in users:
            stats.append({
                'username': user.username,
                'threads_created': user.threads_created,
                'posts_made': user.posts_made,
                'first_seen': user.first_seen,
                'last_seen': user.last_seen
            })
        
        return stats
    
    def get_top_thread_creators(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get users who created the most threads.
        
        Args:
            limit: Number of users to return
            
        Returns:
            List of (username, thread_count) tuples
        """
        results = self.session.query(
            User.username,
            User.threads_created
        ).filter(
            User.threads_created > 0
        ).order_by(
            desc(User.threads_created)
        ).limit(limit).all()
        
        return results
    
    def get_top_responders(self, limit: int = 20) -> List[Tuple[str, int]]:
        """Get users who made the most posts.
        
        Args:
            limit: Number of users to return
            
        Returns:
            List of (username, post_count) tuples
        """
        results = self.session.query(
            User.username,
            User.posts_made
        ).filter(
            User.posts_made > 0
        ).order_by(
            desc(User.posts_made)
        ).limit(limit).all()
        
        return results
    
    def get_category_distribution(self) -> List[Tuple[str, int]]:
        """Get distribution of thread categories.
        
        Returns:
            List of (category, count) tuples
        """
        results = self.session.query(
            ThreadAnalysis.primary_category,
            func.count(ThreadAnalysis.id)
        ).group_by(
            ThreadAnalysis.primary_category
        ).order_by(
            desc(func.count(ThreadAnalysis.id))
        ).all()
        
        return results
    
    def get_outcome_distribution(self) -> List[Tuple[str, int]]:
        """Get distribution of requested outcomes.
        
        Returns:
            List of (outcome, count) tuples
        """
        results = self.session.query(
            ThreadAnalysis.requested_outcome,
            func.count(ThreadAnalysis.id)
        ).group_by(
            ThreadAnalysis.requested_outcome
        ).order_by(
            desc(func.count(ThreadAnalysis.id))
        ).all()
        
        return results
    
    def identify_overactive_users(
        self, 
        thread_threshold: int = 10,
        post_threshold: int = 50
    ) -> List[Dict]:
        """Identify potentially overactive users.
        
        Args:
            thread_threshold: Minimum threads created to be considered overactive
            post_threshold: Minimum posts made to be considered overactive
            
        Returns:
            List of user dictionaries with activity metrics
        """
        users = self.session.query(User).filter(
            (User.threads_created >= thread_threshold) | 
            (User.posts_made >= post_threshold)
        ).all()
        
        overactive = []
        for user in users:
            # Calculate an "overactivity score"
            # Higher score = more potentially problematic
            score = (user.threads_created * 2.0) + (user.posts_made * 0.5)
            
            overactive.append({
                'username': user.username,
                'threads_created': user.threads_created,
                'posts_made': user.posts_made,
                'overactive_score': score,
                'first_seen': user.first_seen,
                'last_seen': user.last_seen
            })
        
        # Sort by score
        overactive.sort(key=lambda x: x['overactive_score'], reverse=True)
        
        return overactive
    
    def get_thread_count(self) -> int:
        """Get total number of threads in database.
        
        Returns:
            Thread count
        """
        return self.session.query(Thread).count()
    
    def get_user_count(self) -> int:
        """Get total number of users in database.
        
        Returns:
            User count
        """
        return self.session.query(User).count()
    
    def get_analyzed_thread_count(self) -> int:
        """Get number of analyzed threads.
        
        Returns:
            Analyzed thread count
        """
        return self.session.query(ThreadAnalysis).count()
    
    def get_threads_with_targets(self, limit: int = 20) -> List[Dict]:
        """Get threads with identified target users.
        
        Args:
            limit: Number of threads to return
            
        Returns:
            List of thread dictionaries with targets
        """
        threads = self.session.query(Thread).filter(
            Thread.targets != None,
            Thread.targets != '[]'
        ).limit(limit).all()
        
        result = []
        for thread in threads:
            targets = json.loads(thread.targets) if thread.targets else []
            result.append({
                'title': thread.title,
                'targets': targets,
                'filer': thread.filer_user,
                'outcome': thread.outcome_status,
                'url': thread.url
            })
        
        return result
    
    def get_closure_statistics(self) -> List[Tuple[str, int]]:
        """Get distribution of closure outcomes.
        
        Returns:
            List of (outcome_status, count) tuples
        """
        results = self.session.query(
            Thread.outcome_status,
            func.count(Thread.id)
        ).filter(
            Thread.outcome_status != None
        ).group_by(
            Thread.outcome_status
        ).order_by(
            desc(func.count(Thread.id))
        ).all()
        
        return results
    
    def get_routing_statistics(self) -> List[Tuple[str, int]]:
        """Get distribution of routing categories.
        
        Returns:
            List of (category, count) tuples
        """
        results = self.session.query(
            Thread.category_routing_suggested,
            func.count(Thread.id)
        ).filter(
            Thread.category_routing_suggested != None
        ).group_by(
            Thread.category_routing_suggested
        ).order_by(
            desc(func.count(Thread.id))
        ).all()
        
        return results
    
    def get_threads_with_diffs(self) -> int:
        """Get count of threads with diff evidence.
        
        Returns:
            Count of threads with diffs
        """
        return self.session.query(Thread).filter(
            Thread.diffs != None,
            Thread.diffs != '[]'
        ).count()
    
    def get_filer_success_stats(self, username: str = None) -> List[Dict]:
        """Get filer success statistics.
        
        Args:
            username: Specific filer username (if None, returns all)
            
        Returns:
            List of filer statistics
        """
        query = self.session.query(Thread).filter(
            Thread.filer_user != None,
            Thread.outcome_status != None
        )
        
        if username:
            query = query.filter(Thread.filer_user == username)
        
        threads = query.all()
        
        # Group by filer
        filer_stats = {}
        for thread in threads:
            filer = thread.filer_user
            if filer not in filer_stats:
                filer_stats[filer] = {
                    'username': filer,
                    'total_filings': 0,
                    'outcomes': {}
                }
            
            filer_stats[filer]['total_filings'] += 1
            outcome = thread.outcome_status or 'unknown'
            filer_stats[filer]['outcomes'][outcome] = filer_stats[filer]['outcomes'].get(outcome, 0) + 1
        
        return list(filer_stats.values())
    
    def get_threads_with_votes(self, limit: int = 20) -> List[Dict]:
        """Get threads that have voting data.
        
        Args:
            limit: Number of threads to return
            
        Returns:
            List of thread dictionaries with vote counts
        """
        from sqlalchemy import func
        
        results = self.session.query(
            Thread.id,
            Thread.title,
            Thread.url,
            func.count(Vote.id).label('vote_count')
        ).join(
            Vote, Thread.id == Vote.thread_id
        ).group_by(
            Thread.id
        ).order_by(
            desc(func.count(Vote.id))
        ).limit(limit).all()
        
        threads_data = []
        for thread_id, title, url, vote_count in results:
            thread = self.session.query(Thread).get(thread_id)
            threads_data.append({
                'id': thread_id,
                'title': title,
                'url': url,
                'vote_count': vote_count,
                'outcome': thread.outcome_status
            })
        
        return threads_data
    
    def get_voter_statistics(self, username: str = None) -> List[Dict]:
        """Get voter statistics.
        
        Args:
            username: Specific voter username (if None, returns all)
            
        Returns:
            List of voter statistics
        """
        query = self.session.query(Vote)
        
        if username:
            query = query.join(User).filter(User.username == username)
        
        votes = query.all()
        
        # Group by voter
        voter_stats = {}
        for vote in votes:
            voter_name = vote.voter.username
            if voter_name not in voter_stats:
                voter_stats[voter_name] = {
                    'username': voter_name,
                    'total_votes': 0,
                    'support': 0,
                    'oppose': 0,
                    'neutral': 0,
                    'comment': 0
                }
            
            voter_stats[voter_name]['total_votes'] += 1
            stance = vote.stance or 'comment'
            if stance in voter_stats[voter_name]:
                voter_stats[voter_name][stance] += 1
        
        return list(voter_stats.values())
