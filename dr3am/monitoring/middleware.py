"""Monitoring middleware for FastAPI"""

import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from dr3am.utils.logging import request_logger, performance_logger
from .metrics import get_metrics_collector


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Add request ID to context
        request_id = request.headers.get("x-request-id", f"req_{int(start_time * 1000)}")
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log the request
            await request_logger.log_request(request, response, duration)
            
            # Log slow requests
            slow_threshold = 1.0  # 1 second
            if duration > slow_threshold:
                performance_logger.log_slow_operation(
                    operation=f"{request.method} {request.url.path}",
                    duration=duration,
                    threshold=slow_threshold,
                    request_id=request_id,
                    user_agent=request.headers.get('user-agent'),
                    remote_addr=request.client.host if request.client else None,
                )
            
            # Add request ID to response headers
            response.headers["x-request-id"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log the error
            request_logger.logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra={
                    'request': {
                        'method': request.method,
                        'url': str(request.url),
                        'duration_ms': round(duration * 1000, 2),
                        'request_id': request_id,
                    },
                    'error': str(e),
                },
                exc_info=True
            )
            
            raise


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting HTTP metrics"""
    
    def __init__(self, app, metrics_collector=None):
        super().__init__(app)
        self.metrics = metrics_collector or get_metrics_collector()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record metrics
            self.metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code,
                duration=duration
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error metrics
            self.metrics.record_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=500,
                duration=duration
            )
            
            self.metrics.record_error(
                method=request.method,
                endpoint=request.url.path,
                error_type=type(e).__name__
            )
            
            raise