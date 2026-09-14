"""Vote parsing and analysis module."""

import re
from typing import List, Dict, Optional
from datetime import datetime


class VoteParser:
    """Parser for ANI voting/consensus discussions."""
    
    # Vote stance keywords
    STANCE_PATTERNS = {
        'support': r'\*\s*\'*\**(Strong\s+)?Support\**(.*?)(?=\n\*|\Z)',
        'oppose': r'\*\s*\'*\**(Strong\s+)?Oppose\**(.*?)(?=\n\*|\Z)',
        'neutral': r'\*\s*\'*\**Neutral\**(.*?)(?=\n\*|\Z)',
        'comment': r'\*\s*\'*\**Comment\**(.*?)(?=\n\*|\Z)',
    }
    
    # Strength modifiers
    STRENGTH_MODIFIERS = {
        'strong': r'\b(?:strong|strongly)\b',
        'weak': r'\b(?:weak|weakly|tentative)\b',
        'conditional': r'\b(?:conditional|if|provided that|as long as)\b'
    }
    
    # Vote trigger patterns
    VOTE_TRIGGERS = [
        r'===\s*Proposal:',
        r'===\s*Motion:',
        r'===\s*Topic ban:',
        r'===\s*Interaction ban:',
        r'===\s*Vote:',
        r'===\s*Community consensus:',
    ]
    
    @staticmethod
    def has_voting_structure(wikitext: str) -> bool:
        """Check if wikitext contains voting structure.
        
        Args:
            wikitext: Raw wikitext content
            
        Returns:
            True if voting structure is detected
        """
        text_lower = wikitext.lower()
        
        # Check for vote trigger markers
        for pattern in VoteParser.VOTE_TRIGGERS:
            if re.search(pattern, wikitext, re.IGNORECASE):
                return True
        
        # Check for multiple stance keywords
        stance_count = 0
        for stance in ['support', 'oppose', 'neutral']:
            # Escape stance to prevent regex injection
            escaped_stance = re.escape(stance)
            if re.search(rf'\*\s*\'*\**{escaped_stance}\**', wikitext, re.IGNORECASE):
                stance_count += 1
        
        # If we have at least 2 different stance types, likely a vote
        return stance_count >= 2
    
    @staticmethod
    def extract_proposals(wikitext: str) -> List[Dict]:
        """Extract proposal sections from wikitext.
        
        Args:
            wikitext: Raw wikitext content
            
        Returns:
            List of proposal dictionaries
        """
        proposals = []
        
        # Split by level-3 headings
        sections = re.split(r'\n===\s*(.+?)\s*===', wikitext)
        
        current_proposal = None
        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break
            
            heading = sections[i]
            content = sections[i + 1]
            
            # Check if this is a proposal/vote section
            is_proposal = any(
                keyword in heading.lower() 
                for keyword in ['proposal', 'motion', 'topic ban', 'vote', 'interaction ban']
            )
            
            if is_proposal:
                proposal_id = VoteParser._generate_proposal_id(heading)
                proposals.append({
                    'proposal_id': proposal_id,
                    'heading': heading,
                    'content': content,
                    'votes': []
                })
        
        return proposals
    
    @staticmethod
    def parse_votes(wikitext: str, proposal_id: str = None) -> List[Dict]:
        """Parse votes from wikitext.
        
        Args:
            wikitext: Raw wikitext content
            proposal_id: Optional proposal identifier
            
        Returns:
            List of vote dictionaries
        """
        votes = []
        
        # Split into lines for parsing
        lines = wikitext.split('\n')
        
        for line in lines:
            # Try each stance pattern
            for stance, pattern in VoteParser.STANCE_PATTERNS.items():
                match = re.match(pattern, line, re.IGNORECASE | re.DOTALL)
                if match:
                    vote = VoteParser._parse_vote_line(line, stance, proposal_id)
                    if vote:
                        votes.append(vote)
                    break
        
        return votes
    
    @staticmethod
    def _parse_vote_line(line: str, stance: str, proposal_id: str = None) -> Optional[Dict]:
        """Parse a single vote line.
        
        Args:
            line: Wikitext line containing vote
            stance: Detected stance (support/oppose/etc.)
            proposal_id: Optional proposal identifier
            
        Returns:
            Vote dictionary or None
        """
        # Extract voter username
        user_pattern = r'\[\[User(?:[ _]talk)?:([^\]|]+)(?:\|[^\]]+)?\]\]'
        user_match = re.search(user_pattern, line)
        
        if not user_match:
            return None
        
        username = user_match.group(1).replace('_', ' ')
        
        # Extract timestamp
        timestamp_pattern = r'(\d{2}:\d{2},\s+\d{1,2}\s+\w+\s+\d{4}\s+\(UTC\))'
        timestamp_match = re.search(timestamp_pattern, line)
        timestamp = None
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            try:
                clean_str = timestamp_str.replace('(UTC)', '').strip()
                timestamp = datetime.strptime(clean_str, '%H:%M, %d %B %Y')
            except (ValueError, AttributeError):
                # Invalid timestamp format, leave as None
                pass
        
        # Detect strength modifiers
        strength_modifier = None
        for modifier, pattern in VoteParser.STRENGTH_MODIFIERS.items():
            if re.search(pattern, line, re.IGNORECASE):
                strength_modifier = modifier
                break
        
        return {
            'proposal_id': proposal_id,
            'voter_user': username,
            'vote_timestamp': timestamp,
            'stance': stance,
            'strength_modifier': strength_modifier,
            'raw_line': line
        }
    
    @staticmethod
    def _generate_proposal_id(heading: str) -> str:
        """Generate a proposal ID from heading.
        
        Args:
            heading: Proposal heading text
            
        Returns:
            Proposal identifier
        """
        # Normalize to lowercase, remove special chars, replace spaces with underscores
        normalized = re.sub(r'[^\w\s]', '', heading.lower())
        normalized = re.sub(r'\s+', '_', normalized.strip())
        return normalized[:50]  # Limit length
    
    @staticmethod
    def detect_herd_direction(votes: List[Dict]) -> Optional[str]:
        """Determine the majority direction from votes.
        
        Args:
            votes: List of vote dictionaries
            
        Returns:
            Majority stance or None
        """
        if not votes:
            return None
        
        stance_counts = {}
        for vote in votes:
            stance = vote.get('stance', 'comment')
            if stance in ['support', 'oppose']:  # Only count support/oppose for herd
                stance_counts[stance] = stance_counts.get(stance, 0) + 1
        
        if not stance_counts:
            return None
        
        # Return the stance with most votes
        return max(stance_counts, key=stance_counts.get)
    
    @staticmethod
    def calculate_contrarian_rate(voter_votes: List[Dict], all_threads: List[Dict]) -> float:
        """Calculate how often a voter goes against the herd.
        
        Args:
            voter_votes: List of votes by a specific voter
            all_threads: List of all threads with vote data
            
        Returns:
            Contrarian rate (0.0 to 1.0)
        """
        if not voter_votes:
            return 0.0
        
        contrarian_count = 0
        total_counted = 0
        
        for vote in voter_votes:
            # Find the thread this vote belongs to
            thread_votes = [v for v in all_threads if v.get('proposal_id') == vote.get('proposal_id')]
            
            if len(thread_votes) < 3:  # Need at least 3 votes to determine herd
                continue
            
            herd_direction = VoteParser.detect_herd_direction(thread_votes)
            if herd_direction and vote.get('stance') in ['support', 'oppose']:
                total_counted += 1
                if vote['stance'] != herd_direction:
                    contrarian_count += 1
        
        if total_counted == 0:
            return 0.0
        
        return contrarian_count / total_counted
    
    @staticmethod
    def extract_tally_language(wikitext: str) -> List[str]:
        """Extract tally or consensus language from closures.
        
        Args:
            wikitext: Raw wikitext content
            
        Returns:
            List of tally phrases found
        """
        tally_patterns = [
            r'by community consensus',
            r'consensus (?:is|was|to)',
            r'(?:clear|rough|general) consensus',
            r'\d+\s*to\s*\d+',  # Vote counts like "5 to 2"
            r'majority (?:support|oppose)',
            r'no consensus',
            r'withdrawn',
            r'moot',
        ]
        
        found = []
        for pattern in tally_patterns:
            matches = re.findall(pattern, wikitext, re.IGNORECASE)
            found.extend(matches)
        
        return found
