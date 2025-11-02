"""
FAISS index manager for vector similarity search.

This module provides the FAISSIndexManager class that manages FAISS indices
for efficient similarity search over embeddings.
"""
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import faiss

from ..utils.logger import get_logger, log_performance_metric, log_error_with_context

logger = get_logger()

# Directory for storing FAISS indices
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
INDEXES_DIR = SCRIPT_DIR / "indexes"
INDEXES_DIR.mkdir(exist_ok=True)


class FAISSIndexManager:
    """
    Manages FAISS vector indices for similarity search.
    
    This class handles creation, loading, and persistence of FAISS indices,
    as well as adding vectors and performing similarity searches.
    """
    
    def __init__(self, index_name: str = "research_papers", embedding_dim: int = 384):
        """
        Initialize the FAISS index manager.
        
        Args:
            index_name: Name of the index (used for file persistence)
            embedding_dim: Dimension of the embeddings
        """
        self.index_name = index_name
        self.embedding_dim = embedding_dim
        self.index_path = INDEXES_DIR / f"{index_name}.faiss"
        self.mapping_path = INDEXES_DIR / f"{index_name}_mapping.npy"
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        
        # Mapping from FAISS index position to chunk_id
        self.embedding_to_chunk: List[int] = []
        
        logger.info(f"Initialized FAISSIndexManager: {index_name} (dim={embedding_dim})")
    
    def create_index(self) -> None:
        """
        Create a new FAISS index.
        
        Uses IndexFlatL2 for exact L2 distance search, which is suitable
        for smaller datasets (< 1M vectors).
        """
        try:
            logger.info(f"Creating new FAISS index with dimension {self.embedding_dim}")
            
            # Use L2 distance (Euclidean) - suitable for normalized embeddings
            # For better results with cosine similarity, use IndexFlatIP after normalizing
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Alternative: Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
            # self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            self.embedding_to_chunk = []
            
            logger.info(f"Created FAISS index: {self.index.ntotal} vectors")
            
        except Exception as e:
            log_error_with_context(e, {
                "operation": "create_faiss_index",
                "embedding_dim": self.embedding_dim
            })
            raise
    
    def load_index(self) -> bool:
        """
        Load an existing FAISS index from disk.
        
        Returns:
            True if successfully loaded, False if index doesn't exist
        """
        try:
            if not self.index_path.exists():
                logger.info(f"Index file not found: {self.index_path}")
                return False
            
            logger.info(f"Loading FAISS index from {self.index_path}")
            start_time = time.time()
            
            self.index = faiss.read_index(str(self.index_path))
            
            # Load mapping
            if self.mapping_path.exists():
                self.embedding_to_chunk = np.load(str(self.mapping_path)).tolist()
            else:
                logger.warning(f"Mapping file not found: {self.mapping_path}")
                self.embedding_to_chunk = []
            
            duration = time.time() - start_time
            
            log_performance_metric(
                "load_faiss_index",
                duration,
                index_name=self.index_name,
                num_vectors=self.index.ntotal
            )
            
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors "
                       f"with {len(self.embedding_to_chunk)} mappings")
            
            return True
            
        except Exception as e:
            log_error_with_context(e, {
                "operation": "load_faiss_index",
                "index_path": str(self.index_path)
            })
            return False
    
    def save_index(self) -> bool:
        """
        Save the FAISS index to disk.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            if self.index is None:
                logger.warning("Cannot save: index is None")
                return False
            
            logger.info(f"Saving FAISS index to {self.index_path}")
            start_time = time.time()
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            
            # Save mapping
            np.save(str(self.mapping_path), np.array(self.embedding_to_chunk))
            
            duration = time.time() - start_time
            
            log_performance_metric(
                "save_faiss_index",
                duration,
                index_name=self.index_name,
                num_vectors=self.index.ntotal
            )
            
            logger.info(f"Saved FAISS index: {self.index.ntotal} vectors")
            
            return True
            
        except Exception as e:
            log_error_with_context(e, {
                "operation": "save_faiss_index",
                "index_path": str(self.index_path)
            })
            return False
    
    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[int]) -> bool:
        """
        Add embeddings to the FAISS index.
        
        Args:
            embeddings: numpy array of embeddings (n, embedding_dim)
            chunk_ids: List of chunk_id values corresponding to each embedding
            
        Returns:
            True if successfully added, False otherwise
        """
        try:
            if self.index is None:
                logger.error("Cannot add embeddings: index is None")
                return False
            
            if len(embeddings) != len(chunk_ids):
                logger.error(f"Mismatch: {len(embeddings)} embeddings but {len(chunk_ids)} chunk_ids")
                return False
            
            if embeddings.shape[1] != self.embedding_dim:
                logger.error(f"Dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}")
                return False
            
            logger.info(f"Adding {len(embeddings)} embeddings to FAISS index")
            start_time = time.time()
            
            # Ensure embeddings are float32 (FAISS requirement)
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            # Normalize embeddings for better cosine similarity results
            # (L2 distance on normalized vectors equals cosine similarity)
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.index.add(embeddings)
            
            # Update mapping
            self.embedding_to_chunk.extend(chunk_ids)
            
            duration = time.time() - start_time
            
            log_performance_metric(
                "add_embeddings_to_faiss",
                duration,
                num_embeddings=len(embeddings),
                total_vectors=self.index.ntotal
            )
            
            logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
            
            return True
            
        except Exception as e:
            log_error_with_context(e, {
                "operation": "add_embeddings_to_faiss",
                "num_embeddings": len(embeddings),
                "num_chunk_ids": len(chunk_ids)
            })
            return False
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[int], List[float]]:
        """
        Search for nearest neighbors to a query embedding.
        
        Args:
            query_embedding: Query embedding vector (1D or 2D array)
            k: Number of nearest neighbors to return
            
        Returns:
            Tuple of (chunk_ids, distances) for the k nearest neighbors
        """
        try:
            if self.index is None:
                logger.error("Cannot search: index is None")
                return [], []
            
            if self.index.ntotal == 0:
                logger.warning("Cannot search: index is empty")
                return [], []
            
            # Ensure query is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            # Ensure float32
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)
            
            # Normalize query
            faiss.normalize_L2(query_embedding)
            
            # Limit k to available vectors
            k = min(k, self.index.ntotal)
            
            # Search
            distances, indices = self.index.search(query_embedding, k)
            
            # Convert FAISS indices to chunk_ids
            chunk_ids = []
            valid_distances = []
            
            for idx, dist in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.embedding_to_chunk):
                    chunk_ids.append(self.embedding_to_chunk[idx])
                    valid_distances.append(float(dist))
            
            logger.info(f"Search found {len(chunk_ids)} results")
            
            return chunk_ids, valid_distances
            
        except Exception as e:
            log_error_with_context(e, {
                "operation": "faiss_search",
                "k": k
            })
            return [], []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index.
        
        Returns:
            Dictionary with index statistics
        """
        if self.index is None:
            return {
                "initialized": False,
                "num_vectors": 0,
                "embedding_dim": self.embedding_dim,
                "index_name": self.index_name
            }
        
        return {
            "initialized": True,
            "num_vectors": self.index.ntotal,
            "num_mappings": len(self.embedding_to_chunk),
            "embedding_dim": self.embedding_dim,
            "index_name": self.index_name,
            "index_type": type(self.index).__name__
        }

