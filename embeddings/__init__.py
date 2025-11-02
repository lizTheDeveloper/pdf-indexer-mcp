"""
Embeddings module for generating text embeddings using MLX.

This module provides functionality to generate embeddings for text chunks
using Apple Silicon-optimized MLX models, and manage FAISS vector indices.
"""

from .generator import EmbeddingGenerator, get_embedding_generator
from .faiss_index import FAISSIndexManager

__all__ = [
    "EmbeddingGenerator",
    "get_embedding_generator",
    "FAISSIndexManager"
]

