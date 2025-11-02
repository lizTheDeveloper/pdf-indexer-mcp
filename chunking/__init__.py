"""
Document chunking module with multiple strategies.
"""

from .header_based import chunk_by_headers
from .s2_chunking import chunk_by_s2

__all__ = [
    'chunk_by_headers',
    'chunk_by_s2',
]

