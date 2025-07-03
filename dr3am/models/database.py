"""Database models for dr3am using SQLAlchemy"""

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, JSON, Float, 
    Boolean, Text, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, Any, Optional

Base = declarative_base()


class User(Base):
    """User model for multi-user support"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    preferences = Column(JSON, default=dict)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    interests = relationship("UserInterest", back_populates="user", cascade="all, delete-orphan")
    research_results = relationship("ResearchResultDB", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(user_id='{self.user_id}', username='{self.username}')>"


class Conversation(Base):
    """Conversation storage model"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey('users.user_id'), nullable=False, index=True)
    conversation_id = Column(String(255), nullable=False, index=True)
    content = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    timestamp = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_conversations_user_timestamp', 'user_id', 'timestamp'),
        Index('ix_conversations_user_conversation', 'user_id', 'conversation_id'),
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id='{self.user_id}', role='{self.role}')>"


class UserInterest(Base):
    """User interest tracking model"""
    __tablename__ = 'user_interests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey('users.user_id'), nullable=False, index=True)
    topic = Column(String(500), nullable=False)
    interest_type = Column(String(100), nullable=False)  # 'hobby', 'learning', etc.
    engagement_level = Column(Integer, nullable=False)  # 1-4 scale
    confidence_score = Column(Float, nullable=False)
    first_mentioned = Column(DateTime, nullable=False)
    last_mentioned = Column(DateTime, nullable=False)
    mention_count = Column(Integer, default=1, nullable=False)
    key_phrases = Column(JSON, default=list)
    context_summary = Column(Text)
    research_potential = Column(Float, nullable=False)
    related_topics = Column(JSON, default=list)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="interests")
    research_opportunities = relationship("ResearchOpportunityDB", back_populates="interest", cascade="all, delete-orphan")
    
    # Constraints and indexes
    __table_args__ = (
        Index('ix_user_interests_user_topic', 'user_id', 'topic'),
        Index('ix_user_interests_user_type', 'user_id', 'interest_type'),
        Index('ix_user_interests_confidence', 'confidence_score'),
        Index('ix_user_interests_last_mentioned', 'last_mentioned'),
        UniqueConstraint('user_id', 'topic', name='uq_user_interest_topic'),
    )
    
    def __repr__(self):
        return f"<UserInterest(id={self.id}, user_id='{self.user_id}', topic='{self.topic}')>"


class ResearchOpportunityDB(Base):
    """Research opportunity storage model"""
    __tablename__ = 'research_opportunities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    interest_id = Column(Integer, ForeignKey('user_interests.id'), nullable=False, index=True)
    opportunity_id = Column(String(255), unique=True, nullable=False, index=True)
    research_type = Column(String(100), nullable=False)  # 'find_resources', 'deep_dive', etc.
    priority_score = Column(Float, nullable=False)
    research_query = Column(Text, nullable=False)
    expected_value = Column(String(50), nullable=False)  # 'high', 'medium', 'low'
    urgency = Column(String(50), nullable=False)  # 'urgent', 'high', 'medium', 'low'
    research_modules = Column(JSON, default=list)
    context = Column(JSON, default=dict)
    status = Column(String(50), default='pending', nullable=False)  # 'pending', 'in_progress', 'completed', 'failed'
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    scheduled_at = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    interest = relationship("UserInterest", back_populates="research_opportunities")
    research_results = relationship("ResearchResultDB", back_populates="opportunity", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_research_opportunities_status', 'status'),
        Index('ix_research_opportunities_priority', 'priority_score'),
        Index('ix_research_opportunities_scheduled', 'scheduled_at'),
    )
    
    def __repr__(self):
        return f"<ResearchOpportunityDB(id={self.id}, opportunity_id='{self.opportunity_id}', status='{self.status}')>"


class ResearchResultDB(Base):
    """Research result storage model"""
    __tablename__ = 'research_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey('users.user_id'), nullable=False, index=True)
    opportunity_id = Column(Integer, ForeignKey('research_opportunities.id'), nullable=True, index=True)
    research_id = Column(String(255), unique=True, nullable=False, index=True)
    topic = Column(String(500), nullable=False)
    research_type = Column(String(100), nullable=False)
    findings = Column(JSON, default=list)
    summary = Column(Text, nullable=False)
    insights = Column(JSON, default=list)
    sources = Column(JSON, default=list)
    confidence_score = Column(Float, nullable=False)
    research_timestamp = Column(DateTime, nullable=False)
    expiry_time = Column(DateTime, nullable=True)
    metadata = Column(JSON, default=dict)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    is_archived = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="research_results")
    opportunity = relationship("ResearchOpportunityDB", back_populates="research_results")
    actionable_insights = relationship("ActionableInsightDB", back_populates="research_result", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_research_results_user_topic', 'user_id', 'topic'),
        Index('ix_research_results_confidence', 'confidence_score'),
        Index('ix_research_results_timestamp', 'research_timestamp'),
        Index('ix_research_results_expiry', 'expiry_time'),
    )
    
    def __repr__(self):
        return f"<ResearchResultDB(id={self.id}, research_id='{self.research_id}', topic='{self.topic}')>"


class ActionableInsightDB(Base):
    """Actionable insight storage model"""
    __tablename__ = 'actionable_insights'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    research_result_id = Column(Integer, ForeignKey('research_results.id'), nullable=False, index=True)
    insight_id = Column(String(255), unique=True, nullable=False, index=True)
    action_type = Column(String(100), nullable=False)  # 'purchase', 'read', 'visit', etc.
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    url = Column(Text, nullable=True)
    urgency = Column(String(50), default='medium', nullable=False)
    estimated_value = Column(String(200), nullable=True)
    deadline = Column(DateTime, nullable=True)
    status = Column(String(50), default='pending', nullable=False)  # 'pending', 'in_progress', 'completed', 'dismissed'
    user_feedback = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    acted_upon_at = Column(DateTime, nullable=True)
    
    # Relationships
    research_result = relationship("ResearchResultDB", back_populates="actionable_insights")
    
    # Indexes for performance
    __table_args__ = (
        Index('ix_actionable_insights_action_type', 'action_type'),
        Index('ix_actionable_insights_urgency', 'urgency'),
        Index('ix_actionable_insights_status', 'status'),
        Index('ix_actionable_insights_deadline', 'deadline'),
    )
    
    def __repr__(self):
        return f"<ActionableInsightDB(id={self.id}, insight_id='{self.insight_id}', action_type='{self.action_type}')>"


class APIUsageLog(Base):
    """API usage tracking model"""
    __tablename__ = 'api_usage_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(255), ForeignKey('users.user_id'), nullable=True, index=True)
    api_provider = Column(String(100), nullable=False)  # 'google_search', 'news_api', etc.
    endpoint = Column(String(200), nullable=False)
    request_type = Column(String(50), nullable=False)  # 'search', 'news', 'shopping', etc.
    cost = Column(Float, nullable=True)
    response_time = Column(Float, nullable=True)  # in seconds
    success = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    request_timestamp = Column(DateTime, default=func.now(), nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Indexes for performance and analytics
    __table_args__ = (
        Index('ix_api_usage_provider_timestamp', 'api_provider', 'request_timestamp'),
        Index('ix_api_usage_user_timestamp', 'user_id', 'request_timestamp'),
        Index('ix_api_usage_success', 'success'),
        Index('ix_api_usage_cost', 'cost'),
    )
    
    def __repr__(self):
        return f"<APIUsageLog(id={self.id}, api_provider='{self.api_provider}', success={self.success})>"


class SystemMetrics(Base):
    """System performance metrics model"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_unit = Column(String(50), nullable=True)
    metric_category = Column(String(100), nullable=False)  # 'performance', 'usage', 'error', etc.
    timestamp = Column(DateTime, default=func.now(), nullable=False)
    metadata = Column(JSON, default=dict)
    
    # Indexes for time-series queries
    __table_args__ = (
        Index('ix_system_metrics_name_timestamp', 'metric_name', 'timestamp'),
        Index('ix_system_metrics_category_timestamp', 'metric_category', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, metric_name='{self.metric_name}', value={self.metric_value})>"


# Database connection and session management
class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self, database_url: str, echo: bool = False):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=echo)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        
    def drop_tables(self):
        """Drop all database tables"""
        Base.metadata.drop_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session"""
        session = self.SessionLocal()
        try:
            return session
        except Exception:
            session.close()
            raise
            
    def close_session(self, session):
        """Close a database session"""
        session.close()


# Database utility functions
def get_database_manager(database_url: str, echo: bool = False) -> DatabaseManager:
    """Get a database manager instance"""
    return DatabaseManager(database_url, echo)


def init_database(database_url: str, echo: bool = False) -> DatabaseManager:
    """Initialize database with all tables"""
    db_manager = DatabaseManager(database_url, echo)
    db_manager.create_tables()
    return db_manager