"""
S2 Chunking: Hybrid Spatial-Semantic document chunking.

Based on the paper "S2 Chunking: A Hybrid Framework for Document Segmentation 
Through Integrated Spatial and Semantic Analysis"

Algorithm:
1. Build graph where nodes are document elements (segments)
2. Calculate combined weights using spatial proximity and semantic similarity
3. Apply spectral clustering to partition into cohesive chunks
4. Enforce token length constraints
"""
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.cluster import SpectralClustering
from sentence_transformers import SentenceTransformer

from ..utils.logger import get_logger

logger = get_logger()

# Global embedding model (lazy loaded)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading sentence transformer model...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
    return _embedding_model


def calculate_spatial_distance(bbox1: tuple, bbox2: tuple) -> float:
    """
    Calculate Euclidean distance between centroids of two bounding boxes.
    
    Args:
        bbox1: (x0, y0, x1, y1) coordinates
        bbox2: (x0, y0, x1, y1) coordinates
        
    Returns:
        Euclidean distance between centroids
    """
    # Calculate centroids
    centroid1_x = (bbox1[0] + bbox1[2]) / 2
    centroid1_y = (bbox1[1] + bbox1[3]) / 2
    
    centroid2_x = (bbox2[0] + bbox2[2]) / 2
    centroid2_y = (bbox2[1] + bbox2[3]) / 2
    
    # Euclidean distance
    distance = np.sqrt((centroid1_x - centroid2_x)**2 + (centroid1_y - centroid2_y)**2)
    return distance


def calculate_spatial_weight(distance: float) -> float:
    """
    Convert spatial distance to weight (inverse relationship).
    
    Args:
        distance: Euclidean distance between elements
        
    Returns:
        Spatial weight (higher = closer together)
    """
    return 1.0 / (1.0 + distance)


def calculate_semantic_similarity(text1: str, text2: str, embeddings: np.ndarray, 
                                  idx1: int, idx2: int) -> float:
    """
    Calculate cosine similarity between text embeddings.
    
    Args:
        text1: First text (unused, embeddings already computed)
        text2: Second text (unused, embeddings already computed)
        embeddings: Pre-computed embeddings matrix
        idx1: Index of first element
        idx2: Index of second element
        
    Returns:
        Cosine similarity score [0, 1]
    """
    # Embeddings are already normalized by sentence-transformers
    # Cosine similarity is just the dot product
    similarity = np.dot(embeddings[idx1], embeddings[idx2])
    return max(0.0, similarity)  # Clamp to [0, 1]


def calculate_combined_weights(segments: List[Dict[str, Any]]) -> np.ndarray:
    """
    Calculate combined spatial and semantic weights for all segment pairs.
    
    Args:
        segments: List of text segments with bbox and text
        
    Returns:
        Affinity matrix (n_segments x n_segments) with combined weights
    """
    num_segments = len(segments)
    
    # Generate embeddings for all segments
    logger.info(f"Generating embeddings for {num_segments} segments...")
    model = get_embedding_model()
    texts = [seg["text"] for seg in segments]
    embeddings = model.encode(texts, normalize_embeddings=True)
    logger.info("Embeddings generated")
    
    # Initialize affinity matrix
    affinity_matrix = np.zeros((num_segments, num_segments))
    
    # Calculate weights for each pair
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            # Spatial weight
            spatial_dist = calculate_spatial_distance(
                segments[i]["bbox"], 
                segments[j]["bbox"]
            )
            spatial_weight = calculate_spatial_weight(spatial_dist)
            
            # Semantic weight
            semantic_weight = calculate_semantic_similarity(
                segments[i]["text"],
                segments[j]["text"],
                embeddings,
                i, j
            )
            
            # Combined weight (average)
            combined_weight = (spatial_weight + semantic_weight) / 2.0
            
            # Symmetric matrix
            affinity_matrix[i, j] = combined_weight
            affinity_matrix[j, i] = combined_weight
    
    # Diagonal elements (self-similarity)
    np.fill_diagonal(affinity_matrix, 1.0)
    
    return affinity_matrix


def estimate_num_clusters(segments: List[Dict[str, Any]], 
                          max_token_length: int,
                          avg_tokens_per_char: float = 0.25) -> int:
    """
    Estimate the number of clusters needed based on token length constraints.
    
    Args:
        segments: List of text segments
        max_token_length: Maximum tokens per chunk
        avg_tokens_per_char: Rough estimate of tokens per character
        
    Returns:
        Estimated number of clusters
    """
    total_chars = sum(len(seg["text"]) for seg in segments)
    estimated_tokens = total_chars * avg_tokens_per_char
    num_clusters = max(1, int(np.ceil(estimated_tokens / max_token_length)))
    
    # Ensure we don't have more clusters than segments
    num_clusters = min(num_clusters, len(segments))
    
    logger.info(f"Estimated {num_clusters} clusters for {estimated_tokens:.0f} tokens")
    return num_clusters


def split_by_token_length(chunks: List[Dict[str, Any]], 
                          max_token_length: int) -> List[Dict[str, Any]]:
    """
    Split chunks that exceed token length constraints.
    
    Args:
        chunks: List of chunks to potentially split
        max_token_length: Maximum tokens per chunk
        
    Returns:
        List of chunks with token length enforced
    """
    # Simple character-based approximation (1 token ≈ 4 characters)
    max_chars = max_token_length * 4
    
    final_chunks = []
    chunk_index = 0
    
    for chunk in chunks:
        text = chunk["text"]
        
        if len(text) <= max_chars:
            # Chunk is within limits
            chunk["chunk_index"] = chunk_index
            final_chunks.append(chunk)
            chunk_index += 1
        else:
            # Split chunk into smaller pieces
            # Try to split at paragraph boundaries
            paragraphs = text.split("\n\n")
            current_text = ""
            
            for para in paragraphs:
                if len(current_text) + len(para) + 2 <= max_chars:
                    if current_text:
                        current_text += "\n\n" + para
                    else:
                        current_text = para
                else:
                    # Save current chunk
                    if current_text:
                        final_chunks.append({
                            "chunk_index": chunk_index,
                            "text": current_text,
                            "header_path": chunk.get("header_path", ""),
                            "page_start": chunk["page_start"],
                            "page_end": chunk["page_end"],
                            "header_level": chunk.get("header_level", 0)
                        })
                        chunk_index += 1
                    
                    # Start new chunk with current paragraph
                    current_text = para
            
            # Save remaining text
            if current_text:
                final_chunks.append({
                    "chunk_index": chunk_index,
                    "text": current_text,
                    "header_path": chunk.get("header_path", ""),
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "header_level": chunk.get("header_level", 0)
                })
                chunk_index += 1
    
    return final_chunks


def chunk_by_s2(segments: List[Dict[str, Any]], 
                max_token_length: int = 512) -> List[Dict[str, Any]]:
    """
    Apply S2 chunking algorithm to segment list.
    
    Args:
        segments: List of text segments with bbox and text metadata
        max_token_length: Maximum tokens per chunk
        
    Returns:
        List of chunks with metadata:
        - chunk_index: int (0-based)
        - text: str (full chunk text)
        - header_path: str (if available)
        - page_start: int (first page of chunk)
        - page_end: int (last page of chunk)
        - header_level: int (if available)
    """
    if not segments:
        return []
    
    if len(segments) == 1:
        # Single segment - return as single chunk
        return [{
            "chunk_index": 0,
            "text": segments[0]["text"],
            "header_path": "",
            "page_start": segments[0]["page"],
            "page_end": segments[0]["page"],
            "header_level": 0
        }]
    
    logger.info(f"Starting S2 chunking for {len(segments)} segments...")
    
    # Step 1: Calculate combined weights (spatial + semantic)
    affinity_matrix = calculate_combined_weights(segments)
    
    # Step 2: Estimate number of clusters
    num_clusters = estimate_num_clusters(segments, max_token_length)
    
    # Step 3: Apply spectral clustering
    logger.info(f"Applying spectral clustering with {num_clusters} clusters...")
    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    labels = clustering.fit_predict(affinity_matrix)
    
    # Step 4: Group segments by cluster
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(segments[idx])
    
    # Step 5: Convert clusters to chunks
    chunks = []
    for cluster_id in sorted(clusters.keys()):
        cluster_segments = clusters[cluster_id]
        
        # Combine text
        text = "\n\n".join([seg["text"] for seg in cluster_segments])
        
        # Determine page range
        pages = [seg["page"] for seg in cluster_segments]
        page_start = min(pages)
        page_end = max(pages)
        
        chunks.append({
            "chunk_index": cluster_id,
            "text": text,
            "header_path": "",  # S2 chunking doesn't use headers
            "page_start": page_start,
            "page_end": page_end,
            "header_level": 0
        })
    
    # Step 6: Split chunks that exceed token length
    final_chunks = split_by_token_length(chunks, max_token_length)
    
    logger.info(f"S2 chunking complete: {len(final_chunks)} final chunks")
    
    return final_chunks

