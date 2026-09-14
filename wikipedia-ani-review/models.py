"""Database models for Wikipedia ANI Review system."""

from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


def utc_now():
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class User(Base):
    """Model for Wikipedia users."""
    
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    first_seen = Column(DateTime, default=utc_now)
    last_seen = Column(DateTime, default=utc_now, onupdate=utc_now)
    
    # Statistics
    threads_created = Column(Integer, default=0)
    posts_made = Column(Integer, default=0)
    
    # Relationships
    threads = relationship("Thread", back_populates="creator", foreign_keys="Thread.creator_id")
    posts = relationship("Post", back_populates="author")
    
    def __repr__(self):
        return f"<User(username='{self.username}')>"


class Thread(Base):
    """Model for ANI discussion threads."""
    
    __tablename__ = 'threads'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), unique=True, nullable=False)
    creator_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    created_at = Column(DateTime, nullable=True)
    archived = Column(Integer, default=0)  # Boolean: 0=active, 1=archived
    
    # Metadata
    archive_name = Column(String(500), nullable=True)
    section_id = Column(String(255), nullable=True)
    thread_anchor = Column(String(255), nullable=True)
    
    # Filer information (who started the thread)
    filer_user = Column(String(255), nullable=True)
    filer_timestamp = Column(DateTime, nullable=True)
    
    # Target users (reported/involved editors) - JSON string
    targets = Column(Text, nullable=True)  # JSON array of usernames
    
    # Evidence and references - JSON strings
    diffs = Column(Text, nullable=True)  # JSON array of diff URLs
    policy_shortcuts = Column(Text, nullable=True)  # JSON array of WP:XYZ shortcuts
    noticeboard_refs = Column(Text, nullable=True)  # JSON array of noticeboard links
    content_area = Column(Text, nullable=True)  # JSON array of referenced pages
    
    # Closure information
    outcome_status = Column(String(100), nullable=True)  # blocked/warned/no-action/etc.
    outcome_text = Column(Text, nullable=True)  # Freeform result text
    closing_admin = Column(String(255), nullable=True)
    closure_timestamp = Column(DateTime, nullable=True)
    
    # Routing/categorization
    category_routing_suggested = Column(String(100), nullable=True)  # BLP, AN3, AIV, etc.
    
    # Raw wikitext for re-parsing
    raw_wikitext = Column(Text, nullable=True)
    
    # Statistics
    num_posts = Column(Integer, default=0)
    num_participants = Column(Integer, default=0)
    
    # Relationships
    creator = relationship("User", back_populates="threads", foreign_keys=[creator_id])
    posts = relationship("Post", back_populates="thread", cascade="all, delete-orphan")
    analysis = relationship("ThreadAnalysis", back_populates="thread", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Thread(title='{self.title[:50]}...')>"


class Post(Base):
    """Model for individual posts within threads."""
    
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('threads.id'), nullable=False)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    content = Column(Text, nullable=False)
    posted_at = Column(DateTime, nullable=True)
    position = Column(Integer, nullable=True)  # Position in thread
    
    # Response structure
    indent_depth = Column(Integer, default=0)  # Count of leading : colons
    comment_kind = Column(String(50), nullable=True)  # plain/bullet/numbered/template-led
    
    # Raw wikitext for re-parsing
    raw_wikitext = Column(Text, nullable=True)
    
    # Relationships
    thread = relationship("Thread", back_populates="posts")
    author = relationship("User", back_populates="posts")
    
    def __repr__(self):
        return f"<Post(id={self.id}, thread_id={self.thread_id})>"


class ThreadAnalysis(Base):
    """Model for LLM-based analysis of threads."""
    
    __tablename__ = 'thread_analyses'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('threads.id'), unique=True, nullable=False)
    
    # Categories
    primary_category = Column(String(100), nullable=True)
    secondary_categories = Column(String(500), nullable=True)  # JSON string of categories
    
    # Requested outcomes
    requested_outcome = Column(String(100), nullable=True)
    
    # Topic areas
    topic_area = Column(String(100), nullable=True)
    
    # Analysis metadata
    analyzed_at = Column(DateTime, default=utc_now)
    confidence_score = Column(Float, nullable=True)
    
    # Raw analysis
    raw_analysis = Column(Text, nullable=True)
    
    # Relationships
    thread = relationship("Thread", back_populates="analysis")
    
    def __repr__(self):
        return f"<ThreadAnalysis(thread_id={self.thread_id}, category='{self.primary_category}')>"


class UserActivity(Base):
    """Model for tracking user activity patterns."""
    
    __tablename__ = 'user_activities'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    
    # Activity metrics
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    threads_created = Column(Integer, default=0)
    posts_made = Column(Integer, default=0)
    unique_threads_participated = Column(Integer, default=0)
    
    # Flags
    is_overactive = Column(Integer, default=0)  # Boolean
    overactive_score = Column(Float, default=0.0)
    
    def __repr__(self):
        return f"<UserActivity(user_id={self.user_id}, period={self.period_start})>"


class Vote(Base):
    """Model for tracking votes in ANI discussions."""
    
    __tablename__ = 'votes'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('threads.id'), nullable=False)
    proposal_id = Column(String(255), nullable=True)  # Sub-proposal within thread
    voter_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    vote_timestamp = Column(DateTime, nullable=True)
    
    # Vote details
    stance = Column(String(50), nullable=True)  # support/oppose/neutral/comment/other
    strength_modifier = Column(String(50), nullable=True)  # strong/weak/conditional
    
    # Analysis
    matched_final = Column(Integer, nullable=True)  # Boolean: did vote match final outcome
    
    # Relationships
    thread = relationship("Thread")
    voter = relationship("User")
    
    def __repr__(self):
        return f"<Vote(thread_id={self.thread_id}, voter_id={self.voter_id}, stance='{self.stance}')>"


class FilingOutcome(Base):
    """Model for tracking filer success rates."""
    
    __tablename__ = 'filing_outcomes'
    
    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('threads.id'), unique=True, nullable=False)
    filer_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Requested vs actual
    requested_action = Column(String(100), nullable=True)  # What filer asked for
    actual_action = Column(String(100), nullable=True)  # What actually happened
    
    # Timing
    time_to_close = Column(Float, nullable=True)  # Hours from filing to closure
    
    # Success flag
    is_successful = Column(Integer, nullable=True)  # Boolean: did filer get what they wanted
    
    # Relationships
    thread = relationship("Thread")
    filer = relationship("User")
    
    def __repr__(self):
        return f"<FilingOutcome(thread_id={self.thread_id}, is_successful={self.is_successful})>"


def init_db(database_url: str):
    """Initialize the database and create all tables."""
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    """Create and return a database session."""
    Session = sessionmaker(bind=engine)
    return Session()
