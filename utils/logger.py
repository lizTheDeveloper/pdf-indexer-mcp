"""
Centralized logging module for PDF Research Paper Indexing MCP Server.

Provides consistent logging configuration across all components with:
- Structured logging format
- Log level configuration
- File and console handlers
- Performance metrics logging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class PDFIndexerLogger:
    """Centralized logger for the PDF Research Paper Indexing MCP Server."""
    
    _instance: Optional['PDFIndexerLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if PDFIndexerLogger._initialized:
            return
        
        self.logger = logging.getLogger("pdf_indexer")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers if already initialized
        if self.logger.handlers:
            PDFIndexerLogger._initialized = True
            return
        
        # Create logs directory if it doesn't exist (relative to this package)
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create file handler with timestamped log file
        log_file = log_dir / f"pdf_indexer_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        PDFIndexerLogger._initialized = True
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


def get_logger() -> logging.Logger:
    """Get the centralized logger instance."""
    return PDFIndexerLogger().get_logger()


def log_performance_metric(operation: str, duration_seconds: float, **kwargs):
    """Log performance metrics for operations.
    
    Args:
        operation: Name of the operation
        duration_seconds: Duration in seconds
        **kwargs: Additional metrics to log
    """
    logger = get_logger()
    metrics_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.info(f"PERF: {operation} - {duration_seconds:.3f}s" + (f" - {metrics_str}" if metrics_str else ""))


def log_error_with_context(error: Exception, context: dict):
    """Log errors with full context.
    
    Args:
        error: The exception that occurred
        context: Additional context dictionary
    """
    logger = get_logger()
    context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
    logger.error(f"ERROR: {type(error).__name__}: {str(error)} - Context: {context_str}", exc_info=True)

