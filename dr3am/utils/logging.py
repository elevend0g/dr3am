"""Structured logging configuration for dr3am"""

import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import structlog
from pythonjsonlogger import jsonlogger

from .config import get_settings, LoggingSettings


class JSONFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp in ISO format
        log_record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        
        # Add service info
        log_record['service'] = 'dr3am'
        log_record['version'] = '1.0.0'
        
        # Add level as string
        log_record['level'] = record.levelname
        
        # Add logger name
        log_record['logger'] = record.name
        
        # Add process and thread info
        log_record['process_id'] = record.process
        log_record['thread_id'] = record.thread
        
        # Add filename and line number
        log_record['filename'] = record.filename
        log_record['line_number'] = record.lineno
        
        # Add function name
        log_record['function'] = record.funcName
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }


class TextFormatter(logging.Formatter):
    """Custom text formatter for human-readable logs"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class ContextFilter(logging.Filter):
    """Filter to add contextual information to log records"""
    
    def __init__(self):
        super().__init__()
        self.context = {}
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add context to the record
        for key, value in self.context.items():
            setattr(record, key, value)
        return True
    
    def set_context(self, **kwargs):
        """Set context variables"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context variables"""
        self.context.clear()


class LoggerManager:
    """Centralized logger management"""
    
    def __init__(self, settings: Optional[LoggingSettings] = None):
        self.settings = settings or get_settings().logging
        self.context_filter = ContextFilter()
        self._configured = False
    
    def setup_logging(self):
        """Set up logging configuration"""
        if self._configured:
            return
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.settings.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatters
        if self.settings.log_format == "json":
            formatter = JSONFormatter()
        else:
            formatter = TextFormatter()
        
        # Console handler
        if self.settings.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(self.context_filter)
            root_logger.addHandler(console_handler)
        
        # File handler
        if self.settings.log_to_file and self.settings.log_file:
            log_file_path = Path(self.settings.log_file)
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=self.settings.log_file,
                maxBytes=self.settings.log_max_bytes,
                backupCount=self.settings.log_backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(self.context_filter)
            root_logger.addHandler(file_handler)
        
        # Configure specific loggers
        self._configure_specific_loggers()
        
        # Set up structured logging if enabled
        if self.settings.enable_structured_logging:
            self._setup_structlog()
        
        self._configured = True
    
    def _configure_specific_loggers(self):
        """Configure specific loggers with appropriate levels"""
        # Suppress noisy third-party loggers
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('aiohttp').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        
        # Configure dr3am loggers
        logging.getLogger('dr3am').setLevel(getattr(logging, self.settings.log_level))
        logging.getLogger('dr3am.semantic_analyzer').setLevel(logging.INFO)
        logging.getLogger('dr3am.research_engine').setLevel(logging.INFO)
        logging.getLogger('dr3am.mcp_server').setLevel(logging.INFO)
        logging.getLogger('dr3am.auth').setLevel(logging.INFO)
    
    def _setup_structlog(self):
        """Set up structlog for structured logging"""
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.add_logger_name,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.dev.ConsoleRenderer() if self.settings.log_format == "text" else structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, self.settings.log_level)
            ),
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        if not self._configured:
            self.setup_logging()
        return logging.getLogger(name)
    
    def get_structured_logger(self, name: str):
        """Get a structured logger"""
        if not self._configured:
            self.setup_logging()
        return structlog.get_logger(name)
    
    def set_context(self, **kwargs):
        """Set logging context"""
        self.context_filter.set_context(**kwargs)
    
    def clear_context(self):
        """Clear logging context"""
        self.context_filter.clear_context()


# Global logger manager instance
_logger_manager: Optional[LoggerManager] = None


def get_logger_manager() -> LoggerManager:
    """Get the global logger manager"""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    return _logger_manager


def setup_logging(settings: Optional[LoggingSettings] = None):
    """Set up logging configuration"""
    manager = LoggerManager(settings)
    manager.setup_logging()
    
    global _logger_manager
    _logger_manager = manager


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name"""
    return get_logger_manager().get_logger(name)


def get_structured_logger(name: str):
    """Get a structured logger"""
    return get_logger_manager().get_structured_logger(name)


def set_logging_context(**kwargs):
    """Set logging context"""
    get_logger_manager().set_context(**kwargs)


def clear_logging_context():
    """Clear logging context"""
    get_logger_manager().clear_context()


class RequestLogger:
    """Request logging middleware"""
    
    def __init__(self, logger_name: str = "dr3am.requests"):
        self.logger = get_logger(logger_name)
    
    async def log_request(self, request, response, duration: float):
        """Log HTTP request details"""
        self.logger.info(
            "HTTP Request",
            extra={
                'request': {
                    'method': request.method,
                    'url': str(request.url),
                    'headers': dict(request.headers),
                    'query_params': dict(request.query_params),
                    'path_params': getattr(request, 'path_params', {}),
                },
                'response': {
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                },
                'duration_ms': round(duration * 1000, 2),
                'user_agent': request.headers.get('user-agent'),
                'remote_addr': request.client.host if request.client else None,
            }
        )


class APILogger:
    """API call logging for external services"""
    
    def __init__(self, logger_name: str = "dr3am.api_calls"):
        self.logger = get_logger(logger_name)
    
    def log_api_call(
        self,
        service: str,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        cost: Optional[float] = None,
        error: Optional[str] = None,
        **kwargs
    ):
        """Log external API call"""
        log_data = {
            'api_call': {
                'service': service,
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'duration_ms': round(duration * 1000, 2),
                'success': 200 <= status_code < 300,
            }
        }
        
        if cost is not None:
            log_data['api_call']['cost'] = cost
        
        if error:
            log_data['api_call']['error'] = error
        
        # Add any additional context
        log_data['api_call'].update(kwargs)
        
        if error:
            self.logger.error(f"API call failed: {service}", extra=log_data)
        else:
            self.logger.info(f"API call: {service}", extra=log_data)


class PerformanceLogger:
    """Performance logging for monitoring"""
    
    def __init__(self, logger_name: str = "dr3am.performance"):
        self.logger = get_logger(logger_name)
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "ms",
        category: str = "performance",
        **context
    ):
        """Log performance metric"""
        self.logger.info(
            f"Performance metric: {metric_name}",
            extra={
                'performance': {
                    'metric_name': metric_name,
                    'value': value,
                    'unit': unit,
                    'category': category,
                    'context': context,
                }
            }
        )
    
    def log_slow_operation(
        self,
        operation: str,
        duration: float,
        threshold: float,
        **context
    ):
        """Log slow operation warning"""
        self.logger.warning(
            f"Slow operation detected: {operation}",
            extra={
                'slow_operation': {
                    'operation': operation,
                    'duration_ms': round(duration * 1000, 2),
                    'threshold_ms': round(threshold * 1000, 2),
                    'context': context,
                }
            }
        )


class SecurityLogger:
    """Security event logging"""
    
    def __init__(self, logger_name: str = "dr3am.security"):
        self.logger = get_logger(logger_name)
    
    def log_auth_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        **context
    ):
        """Log authentication event"""
        log_data = {
            'security_event': {
                'event_type': event_type,
                'success': success,
                'user_id': user_id,
                'details': details or {},
                'context': context,
            }
        }
        
        if success:
            self.logger.info(f"Auth event: {event_type}", extra=log_data)
        else:
            self.logger.warning(f"Auth event failed: {event_type}", extra=log_data)
    
    def log_security_violation(
        self,
        violation_type: str,
        severity: str = "medium",
        details: Optional[Dict[str, Any]] = None,
        **context
    ):
        """Log security violation"""
        log_data = {
            'security_violation': {
                'violation_type': violation_type,
                'severity': severity,
                'details': details or {},
                'context': context,
            }
        }
        
        self.logger.error(f"Security violation: {violation_type}", extra=log_data)


# Pre-configured logger instances
request_logger = RequestLogger()
api_logger = APILogger()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()