"""
Header-based semantic chunking for documents.
"""
from typing import List, Dict, Any
import re


def build_header_path(segments: List[Dict[str, Any]], current_index: int) -> str:
    """
    Build the header path string for a given segment based on preceding headers.
    
    Args:
        segments: List of all segments
        current_index: Index of the current segment
        
    Returns:
        Header path string (e.g., "1. Introduction > 1.1. Background")
    """
    path_parts = []
    header_stack = []  # Stack of (level, text) pairs
    
    for i in range(current_index + 1):
        seg = segments[i]
        if seg["is_header"]:
            level = seg["header_level"]
            text = seg["text"].strip()
            
            # Remove header level indicators if present
            text = re.sub(r'^\d+\.\s*', '', text)
            
            # Maintain header hierarchy stack
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            
            header_stack.append((level, text))
    
    # Build path from stack
    if header_stack:
        path_parts = [text for _, text in header_stack]
    
    return " > ".join(path_parts) if path_parts else ""


def chunk_by_headers(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Chunk PDF text by headers, grouping content under each header.
    
    Args:
        segments: List of text segments with metadata
        
    Returns:
        List of chunks with metadata:
        - chunk_index: int (0-based)
        - text: str (full chunk text)
        - header_path: str (full header hierarchy)
        - page_start: int (first page of chunk)
        - page_end: int (last page of chunk)
        - header_level: int (level of the main header)
    """
    chunks = []
    current_chunk_segments = []
    current_header_path = ""
    current_header_level = 0
    current_page_start = 0
    current_page_end = 0
    chunk_index = 0
    
    for i, segment in enumerate(segments):
        is_header = segment["is_header"]
        header_level = segment["header_level"]
        page = segment["page"]
        
        # If we hit a header of same or higher level (going up hierarchy), start new chunk
        if is_header and header_level > 0:
            # Save previous chunk if it has content
            if current_chunk_segments:
                chunk_text = "\n\n".join([seg["text"] for seg in current_chunk_segments])
                if chunk_text.strip():
                    chunks.append({
                        "chunk_index": chunk_index,
                        "text": chunk_text.strip(),
                        "header_path": current_header_path,
                        "page_start": current_page_start,
                        "page_end": current_page_end,
                        "header_level": current_header_level
                    })
                    chunk_index += 1
            
            # Start new chunk
            current_chunk_segments = [segment]
            current_header_path = build_header_path(segments, i)
            current_header_level = header_level
            current_page_start = page
            current_page_end = page
        else:
            # Add to current chunk
            if not current_chunk_segments:
                # Start chunk with first non-header if no header yet
                current_page_start = page
            current_chunk_segments.append(segment)
            current_page_end = page
    
    # Save final chunk
    if current_chunk_segments:
        chunk_text = "\n\n".join([seg["text"] for seg in current_chunk_segments])
        if chunk_text.strip():
            chunks.append({
                "chunk_index": chunk_index,
                "text": chunk_text.strip(),
                "header_path": current_header_path,
                "page_start": current_page_start,
                "page_end": current_page_end,
                "header_level": current_header_level
            })
    
    return chunks

