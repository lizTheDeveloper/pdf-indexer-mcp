"""
Embedding generator using MLX for efficient embedding generation on Apple Silicon.

This module provides the EmbeddingGenerator class that uses MLX-Embeddings
to generate text embeddings efficiently on Apple Silicon hardware.
"""
import time
from typing import List, Dict, Any, Optional
import numpy as np
import mlx.core as mx

from ..utils.logger import get_logger, log_performance_metric, log_error_with_context

logger = get_logger()

# Global singleton instance
_embedding_generator = None


class EmbeddingGenerator:
    """
    Generates text embeddings using MLX models optimized for Apple Silicon.
    
    This class uses MLX-Embeddings to generate embeddings efficiently,
    with automatic batching and conversion to numpy arrays for FAISS compatibility.
    """
    
    def __init__(self, model_name: str = "mlx-community/Qwen3-Embedding-0.6B"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Hugging Face model identifier for the embedding model
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        
        logger.info(f"Initializing EmbeddingGenerator with model: {model_name}")
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model and tokenizer."""
        try:
            from mlx_embeddings.utils import load
            import os
            
            start_time = time.time()
            
            # Check if we need HuggingFace authentication
            # Try to load the model - if it fails with ModelNotFoundError, 
            # it might be an authentication issue or the model path is wrong
            try:
                self.model, self.tokenizer = load(self.model_name)
            except Exception as load_error:
                # Check if it's a ModelNotFoundError - might need authentication
                error_msg = str(load_error).lower()
                if "model not found" in error_msg or "not found" in error_msg:
                    # Try to provide helpful error message
                    logger.warning(
                        f"Model loading failed. This might be due to: "
                        f"1) HuggingFace authentication required (run: huggingface-cli login), "
                        f"2) Network connectivity issues, "
                        f"3) Model name mismatch. "
                        f"Attempting to load: {self.model_name}"
                    )
                    # Check if HF token is set
                    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
                    if not hf_token:
                        logger.warning(
                            "No HuggingFace token found in environment. "
                            "Public models should work without auth, but some may require it."
                        )
                raise  # Re-raise the original error
            
            # Get embedding dimension by generating a test embedding
            test_texts = ["test"]
            test_input = self.tokenizer.batch_encode_plus(
                test_texts,
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=512
            )
            test_output = self.model(
                test_input["input_ids"],
                attention_mask=test_input.get("attention_mask")
            )
            
            # Get embedding dimension from output
            if hasattr(test_output, 'text_embeds'):
                self.embedding_dim = test_output.text_embeds.shape[-1]
            elif hasattr(test_output, 'last_hidden_state'):
                self.embedding_dim = test_output.last_hidden_state.shape[-1]
            else:
                raise ValueError("Cannot determine embedding dimension from model output")
            
            duration = time.time() - start_time
            
            log_performance_metric(
                "load_embedding_model",
                duration,
                model_name=self.model_name,
                embedding_dim=self.embedding_dim
            )
            
            logger.info(f"Loaded embedding model: {self.model_name} (dim={self.embedding_dim})")
            
        except Exception as e:
            log_error_with_context(e, {
                "operation": "load_embedding_model",
                "model_name": self.model_name
            })
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
        start_time = time.time()
        
        all_embeddings = []
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self._generate_batch(batch_texts)
                all_embeddings.append(batch_embeddings)
                
                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Processed {i + len(batch_texts)}/{len(texts)} texts")
            
            # Concatenate all batches
            embeddings = np.vstack(all_embeddings)
            
            duration = time.time() - start_time
            
            log_performance_metric(
                "generate_embeddings",
                duration,
                num_texts=len(texts),
                batch_size=batch_size,
                embeddings_per_sec=len(texts) / duration if duration > 0 else 0
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings in {duration:.2f}s "
                       f"({len(texts)/duration:.1f} texts/sec)")
            
            return embeddings
            
        except Exception as e:
            log_error_with_context(e, {
                "operation": "generate_embeddings",
                "num_texts": len(texts),
                "batch_size": batch_size
            })
            raise
    
    def _generate_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings (batch)
            
        Returns:
            numpy array of embeddings for the batch
        """
        # Tokenize batch
        inputs = self.tokenizer.batch_encode_plus(
            texts,
            return_tensors="mlx",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate embeddings
        outputs = self.model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )
        
        # Extract embeddings (normalized and pooled)
        if hasattr(outputs, 'text_embeds'):
            embeddings_mlx = outputs.text_embeds
        elif hasattr(outputs, 'last_hidden_state'):
            # Mean pooling if text_embeds not available
            embeddings_mlx = self._mean_pooling(
                outputs.last_hidden_state,
                inputs.get("attention_mask")
            )
            # L2 normalize
            embeddings_mlx = embeddings_mlx / mx.linalg.norm(
                embeddings_mlx, axis=1, keepdims=True
            )
        else:
            raise ValueError("Cannot extract embeddings from model output")
        
        # Convert MLX array to numpy with explicit dtype
        # MLX arrays have buffer compatibility issues with numpy, so we need to be careful
        # First, ensure the array is evaluated (not lazy)
        embeddings_mlx = mx.array(embeddings_mlx)
        
        # Convert to list first, then to numpy array to avoid buffer format issues
        # This is slower but more reliable for MLX->numpy conversion
        try:
            # Try direct conversion first
            embeddings_np = np.array(embeddings_mlx, dtype=np.float32, copy=True)
        except (ValueError, TypeError, RuntimeError) as e:
            # Fallback: convert via tolist() if direct conversion fails
            # This happens when MLX uses a dtype that numpy can't interpret directly
            logger.debug(f"Direct MLX->numpy conversion failed ({type(e).__name__}), using tolist() fallback")
            embeddings_list = embeddings_mlx.tolist()
            embeddings_np = np.array(embeddings_list, dtype=np.float32)
        
        # Ensure contiguous array for FAISS
        if not embeddings_np.flags['C_CONTIGUOUS']:
            embeddings_np = np.ascontiguousarray(embeddings_np, dtype=np.float32)
        
        return embeddings_np
    
    def _mean_pooling(self, token_embeddings: mx.array, 
                     attention_mask: Optional[mx.array] = None) -> mx.array:
        """
        Apply mean pooling to token embeddings.
        
        Args:
            token_embeddings: Token-level embeddings (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Pooled embeddings (batch_size, hidden_dim)
        """
        if attention_mask is not None:
            # Expand attention mask to match embedding dimensions
            mask_expanded = attention_mask.unsqueeze(-1).astype(token_embeddings.dtype)
            # Sum embeddings, weighted by mask
            sum_embeddings = mx.sum(token_embeddings * mask_expanded, axis=1)
            # Divide by number of non-masked tokens
            sum_mask = mx.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            return sum_embeddings / sum_mask
        else:
            # Simple mean if no mask
            return mx.mean(token_embeddings, axis=1)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim


def get_embedding_generator(model_name: str = "mlx-community/Qwen3-Embedding-0.6B") -> EmbeddingGenerator:
    """
    Get or create a singleton EmbeddingGenerator instance.
    
    This function ensures that only one embedding model is loaded at a time,
    conserving memory.
    
    Args:
        model_name: Hugging Face model identifier
        
    Returns:
        EmbeddingGenerator instance
    """
    global _embedding_generator
    
    if _embedding_generator is None or _embedding_generator.model_name != model_name:
        _embedding_generator = EmbeddingGenerator(model_name)
    
    return _embedding_generator

