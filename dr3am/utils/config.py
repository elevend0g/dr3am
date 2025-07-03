"""Configuration management using Pydantic"""

from pydantic import BaseSettings, Field, validator
from typing import Optional, List, Dict, Any
import os
from pathlib import Path


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    url: str = Field(default="sqlite:///dr3am.db", env="DATABASE_URL")
    echo: bool = Field(default=False, env="DATABASE_ECHO")
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    
    @validator('url')
    def validate_database_url(cls, v):
        if not v:
            raise ValueError("Database URL cannot be empty")
        return v


class APISettings(BaseSettings):
    """API integration configuration"""
    # Google Search API
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    google_search_engine_id: Optional[str] = Field(default=None, env="GOOGLE_SEARCH_ENGINE_ID")
    
    # News API
    news_api_key: Optional[str] = Field(default=None, env="NEWS_API_KEY")
    
    # Shopping APIs
    amazon_api_key: Optional[str] = Field(default=None, env="AMAZON_API_KEY")
    ebay_api_key: Optional[str] = Field(default=None, env="EBAY_API_KEY")
    
    # Rate limiting
    api_rate_limit_per_minute: int = Field(default=60, env="API_RATE_LIMIT_PER_MINUTE")
    api_rate_limit_per_day: int = Field(default=1000, env="API_RATE_LIMIT_PER_DAY")
    
    # Budget management
    daily_api_budget: float = Field(default=10.0, env="DAILY_API_BUDGET")
    monthly_api_budget: float = Field(default=300.0, env="MONTHLY_API_BUDGET")
    
    @validator('daily_api_budget', 'monthly_api_budget')
    def validate_budget(cls, v):
        if v < 0:
            raise ValueError("Budget cannot be negative")
        return v


class SemanticAnalysisSettings(BaseSettings):
    """Semantic analysis configuration"""
    analysis_window_days: int = Field(default=30, env="ANALYSIS_WINDOW_DAYS")
    min_mentions_for_interest: int = Field(default=2, env="MIN_MENTIONS_FOR_INTEREST")
    confidence_threshold: float = Field(default=0.6, env="CONFIDENCE_THRESHOLD")
    max_interests_per_analysis: int = Field(default=10, env="MAX_INTERESTS_PER_ANALYSIS")
    
    # LLM configuration
    llm_model: str = Field(default="gpt-3.5-turbo", env="LLM_MODEL")
    llm_api_key: Optional[str] = Field(default=None, env="LLM_API_KEY")
    llm_base_url: Optional[str] = Field(default=None, env="LLM_BASE_URL")
    llm_temperature: float = Field(default=0.3, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    
    @validator('analysis_window_days', 'min_mentions_for_interest', 'max_interests_per_analysis')
    def validate_positive_int(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @validator('confidence_threshold', 'llm_temperature')
    def validate_float_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0.0 and 1.0")
        return v


class ResearchExecutionSettings(BaseSettings):
    """Research execution configuration"""
    auto_research_enabled: bool = Field(default=True, env="AUTO_RESEARCH_ENABLED")
    research_cooldown_hours: int = Field(default=24, env="RESEARCH_COOLDOWN_HOURS")
    max_concurrent_research: int = Field(default=3, env="MAX_CONCURRENT_RESEARCH")
    research_timeout_seconds: int = Field(default=300, env="RESEARCH_TIMEOUT_SECONDS")
    retry_attempts: int = Field(default=3, env="RETRY_ATTEMPTS")
    
    # Cache settings
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    cache_ttl_hours: int = Field(default=24, env="CACHE_TTL_HOURS")
    cache_max_size: int = Field(default=1000, env="CACHE_MAX_SIZE")
    
    @validator('research_cooldown_hours', 'max_concurrent_research', 'research_timeout_seconds', 'retry_attempts')
    def validate_positive_int(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class SecuritySettings(BaseSettings):
    """Security and authentication configuration"""
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=1440, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 24 hours
    refresh_token_expire_days: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_credentials: bool = Field(default=True, env="CORS_CREDENTIALS")
    cors_methods: List[str] = Field(default=["*"], env="CORS_METHODS")
    cors_headers: List[str] = Field(default=["*"], env="CORS_HEADERS")
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if v == "your-secret-key-change-in-production":
            import warnings
            warnings.warn("Using default secret key. Change this in production!")
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v


class LoggingSettings(BaseSettings):
    """Logging configuration"""
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")  # "json" or "text"
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_max_bytes: int = Field(default=10*1024*1024, env="LOG_MAX_BYTES")  # 10MB
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")
    
    # Structured logging
    enable_structured_logging: bool = Field(default=True, env="ENABLE_STRUCTURED_LOGGING")
    log_to_console: bool = Field(default=True, env="LOG_TO_CONSOLE")
    log_to_file: bool = Field(default=False, env="LOG_TO_FILE")
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class MonitoringSettings(BaseSettings):
    """Monitoring and metrics configuration"""
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_port: int = Field(default=9090, env="METRICS_PORT")
    enable_health_checks: bool = Field(default=True, env="ENABLE_HEALTH_CHECKS")
    health_check_interval: int = Field(default=30, env="HEALTH_CHECK_INTERVAL")
    
    # Performance monitoring
    enable_performance_monitoring: bool = Field(default=True, env="ENABLE_PERFORMANCE_MONITORING")
    slow_query_threshold: float = Field(default=1.0, env="SLOW_QUERY_THRESHOLD")
    enable_request_tracing: bool = Field(default=False, env="ENABLE_REQUEST_TRACING")


class ServerSettings(BaseSettings):
    """Server configuration"""
    host: str = Field(default="0.0.0.0", env="DR3AM_HOST")
    port: int = Field(default=8000, env="DR3AM_PORT")
    workers: int = Field(default=1, env="DR3AM_WORKERS")
    reload: bool = Field(default=False, env="DR3AM_RELOAD")
    
    # Request handling
    max_request_size: int = Field(default=16*1024*1024, env="MAX_REQUEST_SIZE")  # 16MB
    request_timeout: int = Field(default=60, env="REQUEST_TIMEOUT")
    
    @validator('port')
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class RedisSettings(BaseSettings):
    """Redis configuration for caching and sessions"""
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    redis_socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    
    # Session management
    session_expire_seconds: int = Field(default=3600, env="SESSION_EXPIRE_SECONDS")
    
    @validator('redis_max_connections', 'redis_socket_timeout', 'session_expire_seconds')
    def validate_positive_int(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class Settings(BaseSettings):
    """Main application settings"""
    # Application info
    app_name: str = Field(default="dr3am", env="APP_NAME")
    app_version: str = Field(default="1.0.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    api: APISettings = APISettings()
    semantic_analysis: SemanticAnalysisSettings = SemanticAnalysisSettings()
    research_execution: ResearchExecutionSettings = ResearchExecutionSettings()
    security: SecuritySettings = SecuritySettings()
    logging: LoggingSettings = LoggingSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    server: ServerSettings = ServerSettings()
    redis: RedisSettings = RedisSettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    @validator('environment')
    def validate_environment(cls, v):
        valid_environments = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_environments:
            raise ValueError(f"Environment must be one of {valid_environments}")
        return v.lower()
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.environment == "testing"
    
    def get_database_url(self) -> str:
        """Get the configured database URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get the configured Redis URL"""
        return self.redis.redis_url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.dict()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment"""
    global _settings
    _settings = Settings()
    return _settings


def create_settings_from_dict(config_dict: Dict[str, Any]) -> Settings:
    """Create settings from dictionary"""
    return Settings(**config_dict)


def create_test_settings() -> Settings:
    """Create settings for testing"""
    return Settings(
        environment="testing",
        debug=True,
        database=DatabaseSettings(url="sqlite:///:memory:", echo=False),
        logging=LoggingSettings(log_level="DEBUG", log_to_console=False),
        security=SecuritySettings(secret_key="test-secret-key-32-characters-long"),
        redis=RedisSettings(redis_url="redis://localhost:6379/1")
    )