"""
Database operations module using raw SQL for performance and control.

This module provides database operations using raw SQL queries via SQLAlchemy's
connection interface, following the user's preference for raw SQL over ORM operations.
"""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .models import Base
from ..utils.logger import get_logger, log_performance_metric, log_error_with_context

logger = get_logger()

# Database path - use absolute path relative to this module
MODULE_DIR = Path(__file__).parent.parent.absolute()
DB_PATH = MODULE_DIR / "indexes" / "research_papers.db"


def get_engine() -> Engine:
    """
    Get or create SQLAlchemy engine for the database.
    
    Returns:
        SQLAlchemy Engine object
    """
    db_url = f"sqlite:///{DB_PATH}"
    engine = create_engine(db_url, echo=False)
    return engine


def initialize_database() -> bool:
    """
    Initialize the database by creating all tables if they don't exist.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Initializing database at {DB_PATH}")
        
        # Ensure indexes directory exists
        DB_PATH.parent.mkdir(exist_ok=True)
        
        # Create all tables using SQLAlchemy models
        engine = get_engine()
        Base.metadata.create_all(engine)
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        log_error_with_context(e, {"operation": "initialize_database"})
        return False


def insert_paper(url: str, filename: str, title: Optional[str] = None,
                 num_pages: int = 0, chunking_method: Optional[str] = None) -> Optional[int]:
    """
    Insert a new paper into the database.
    
    Args:
        url: Original download URL
        filename: Filename in ./papers/ directory
        title: Extracted title (optional)
        num_pages: Total number of pages
        chunking_method: Method used for chunking (header or s2)
        
    Returns:
        paper_id if successful, None otherwise
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Check if paper already exists
            result = conn.execute(
                text("SELECT paper_id FROM papers WHERE filename = :filename"),
                {"filename": filename}
            )
            existing = result.fetchone()
            
            if existing:
                logger.info(f"Paper already exists with filename: {filename}, paper_id: {existing[0]}")
                return existing[0]
            
            # Insert new paper
            result = conn.execute(
                text("""
                    INSERT INTO papers (url, filename, title, num_pages, num_chunks, chunking_method)
                    VALUES (:url, :filename, :title, :num_pages, 0, :chunking_method)
                """),
                {
                    "url": url,
                    "filename": filename,
                    "title": title,
                    "num_pages": num_pages,
                    "chunking_method": chunking_method
                }
            )
            conn.commit()
            
            paper_id = result.lastrowid
            logger.info(f"Inserted paper: {filename}, paper_id: {paper_id}")
            return paper_id
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "insert_paper",
            "filename": filename,
            "url": url
        })
        return None


def insert_chunks(paper_id: int, chunks: List[Dict[str, Any]]) -> bool:
    """
    Insert chunks for a paper with navigation indices.
    
    Args:
        paper_id: ID of the paper
        chunks: List of chunk dictionaries with keys:
            - chunk_index: Sequential index (0-based)
            - text: Full text content
            - header_path: Header hierarchy path
            - header_level: Header level (0-3)
            - page_start: First page number
            - page_end: Last page number
            
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        num_chunks = len(chunks)
        
        with engine.connect() as conn:
            # Calculate prev/next indices and insert chunks
            for i, chunk in enumerate(chunks):
                prev_idx = i - 1 if i > 0 else None
                next_idx = i + 1 if i < num_chunks - 1 else None
                
                conn.execute(
                    text("""
                        INSERT INTO chunks (
                            paper_id, chunk_index, text, header_path, header_level,
                            page_start, page_end, prev_chunk_index, next_chunk_index
                        ) VALUES (
                            :paper_id, :chunk_index, :text, :header_path, :header_level,
                            :page_start, :page_end, :prev_chunk_index, :next_chunk_index
                        )
                    """),
                    {
                        "paper_id": paper_id,
                        "chunk_index": chunk["chunk_index"],
                        "text": chunk["text"],
                        "header_path": chunk.get("header_path", ""),
                        "header_level": chunk.get("header_level", 0),
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "prev_chunk_index": prev_idx,
                        "next_chunk_index": next_idx
                    }
                )
            
            # Update paper's num_chunks
            conn.execute(
                text("UPDATE papers SET num_chunks = :num_chunks WHERE paper_id = :paper_id"),
                {"num_chunks": num_chunks, "paper_id": paper_id}
            )
            
            conn.commit()
            
        logger.info(f"Inserted {num_chunks} chunks for paper_id: {paper_id}")
        return True
        
    except Exception as e:
        log_error_with_context(e, {
            "operation": "insert_chunks",
            "paper_id": paper_id,
            "num_chunks": len(chunks)
        })
        return False


def insert_sections(paper_id: int, chunks: List[Dict[str, Any]]) -> bool:
    """
    Analyze chunks and insert sections based on header paths.
    
    Args:
        paper_id: ID of the paper
        chunks: List of chunk dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        
        # Group chunks by header path
        sections_map: Dict[str, Dict[str, Any]] = {}
        
        for chunk in chunks:
            header_path = chunk.get("header_path", "")
            if not header_path:
                continue
                
            if header_path not in sections_map:
                sections_map[header_path] = {
                    "header_path": header_path,
                    "header_level": chunk.get("header_level", 0),
                    "start_chunk_index": chunk["chunk_index"],
                    "end_chunk_index": chunk["chunk_index"],
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"]
                }
            else:
                # Update end indices
                section = sections_map[header_path]
                section["end_chunk_index"] = chunk["chunk_index"]
                section["page_end"] = max(section["page_end"], chunk["page_end"])
        
        # Insert sections
        with engine.connect() as conn:
            for section_data in sections_map.values():
                result = conn.execute(
                    text("""
                        INSERT INTO sections (
                            paper_id, header_path, header_level,
                            start_chunk_index, end_chunk_index,
                            page_start, page_end
                        ) VALUES (
                            :paper_id, :header_path, :header_level,
                            :start_chunk_index, :end_chunk_index,
                            :page_start, :page_end
                        )
                    """),
                    {
                        "paper_id": paper_id,
                        **section_data
                    }
                )
                
                section_id = result.lastrowid
                
                # Update chunks with section_id
                conn.execute(
                    text("""
                        UPDATE chunks
                        SET section_id = :section_id
                        WHERE paper_id = :paper_id
                        AND chunk_index BETWEEN :start_idx AND :end_idx
                    """),
                    {
                        "section_id": section_id,
                        "paper_id": paper_id,
                        "start_idx": section_data["start_chunk_index"],
                        "end_idx": section_data["end_chunk_index"]
                    }
                )
            
            conn.commit()
        
        logger.info(f"Inserted {len(sections_map)} sections for paper_id: {paper_id}")
        return True
        
    except Exception as e:
        log_error_with_context(e, {
            "operation": "insert_sections",
            "paper_id": paper_id
        })
        return False


def get_paper_by_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve paper metadata by filename.
    
    Args:
        filename: Filename in ./papers/ directory
        
    Returns:
        Dictionary with paper metadata or None if not found
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT paper_id, url, filename, download_date, title,
                           num_chunks, num_pages, chunking_method
                    FROM papers
                    WHERE filename = :filename
                """),
                {"filename": filename}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "paper_id": row[0],
                    "url": row[1],
                    "filename": row[2],
                    "download_date": row[3],
                    "title": row[4],
                    "num_chunks": row[5],
                    "num_pages": row[6],
                    "chunking_method": row[7]
                }
            return None
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_paper_by_filename",
            "filename": filename
        })
        return None


def get_chunk(paper_id: int, chunk_index: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific chunk by paper_id and chunk_index.
    
    Args:
        paper_id: ID of the paper
        chunk_index: Index of the chunk within the paper
        
    Returns:
        Dictionary with chunk data or None if not found
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT chunk_id, chunk_index, text, header_path, header_level,
                           page_start, page_end, prev_chunk_index, next_chunk_index,
                           section_id, embedding_index
                    FROM chunks
                    WHERE paper_id = :paper_id AND chunk_index = :chunk_index
                """),
                {"paper_id": paper_id, "chunk_index": chunk_index}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "chunk_id": row[0],
                    "paper_id": paper_id,
                    "chunk_index": row[1],
                    "text": row[2],
                    "header_path": row[3],
                    "header_level": row[4],
                    "page_start": row[5],
                    "page_end": row[6],
                    "prev_chunk_index": row[7],
                    "next_chunk_index": row[8],
                    "section_id": row[9],
                    "embedding_index": row[10]
                }
            return None
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_chunk",
            "paper_id": paper_id,
            "chunk_index": chunk_index
        })
        return None


def get_chunks_range(paper_id: int, start_index: int, end_index: int) -> List[Dict[str, Any]]:
    """
    Retrieve a range of chunks for a paper.
    
    Args:
        paper_id: ID of the paper
        start_index: Starting chunk index (inclusive)
        end_index: Ending chunk index (inclusive)
        
    Returns:
        List of chunk dictionaries
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT chunk_id, chunk_index, text, header_path, header_level,
                           page_start, page_end, prev_chunk_index, next_chunk_index,
                           section_id, embedding_index
                    FROM chunks
                    WHERE paper_id = :paper_id
                    AND chunk_index BETWEEN :start_idx AND :end_idx
                    ORDER BY chunk_index ASC
                """),
                {"paper_id": paper_id, "start_idx": start_index, "end_idx": end_index}
            )
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "paper_id": paper_id,
                    "chunk_index": row[1],
                    "text": row[2],
                    "header_path": row[3],
                    "header_level": row[4],
                    "page_start": row[5],
                    "page_end": row[6],
                    "prev_chunk_index": row[7],
                    "next_chunk_index": row[8],
                    "section_id": row[9],
                    "embedding_index": row[10]
                })
            
            return chunks
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_chunks_range",
            "paper_id": paper_id,
            "start_index": start_index,
            "end_index": end_index
        })
        return []


def get_section(paper_id: int, header_path: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve section metadata by header path.
    
    Args:
        paper_id: ID of the paper
        header_path: Header path to search for
        
    Returns:
        Dictionary with section metadata or None if not found
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT section_id, header_path, header_level,
                           start_chunk_index, end_chunk_index,
                           page_start, page_end
                    FROM sections
                    WHERE paper_id = :paper_id AND header_path = :header_path
                """),
                {"paper_id": paper_id, "header_path": header_path}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "section_id": row[0],
                    "paper_id": paper_id,
                    "header_path": row[1],
                    "header_level": row[2],
                    "start_chunk_index": row[3],
                    "end_chunk_index": row[4],
                    "page_start": row[5],
                    "page_end": row[6]
                }
            return None
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_section",
            "paper_id": paper_id,
            "header_path": header_path
        })
        return None


def list_all_papers() -> List[Dict[str, Any]]:
    """
    List all papers in the database.
    
    Returns:
        List of paper dictionaries
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT paper_id, url, filename, download_date, title,
                           num_chunks, num_pages, chunking_method
                    FROM papers
                    ORDER BY download_date DESC
                """)
            )
            
            papers = []
            for row in result:
                papers.append({
                    "paper_id": row[0],
                    "url": row[1],
                    "filename": row[2],
                    "download_date": row[3],
                    "title": row[4],
                    "num_chunks": row[5],
                    "num_pages": row[6],
                    "chunking_method": row[7]
                })
            
            return papers
            
    except Exception as e:
        log_error_with_context(e, {"operation": "list_all_papers"})
        return []


def get_paper_structure(paper_id: int) -> Dict[str, Any]:
    """
    Get the complete structure of a paper (sections and chunk ranges).
    
    Args:
        paper_id: ID of the paper
        
    Returns:
        Dictionary with paper structure including sections
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            # Get paper metadata
            paper_result = conn.execute(
                text("""
                    SELECT paper_id, url, filename, title, num_chunks, num_pages, chunking_method
                    FROM papers
                    WHERE paper_id = :paper_id
                """),
                {"paper_id": paper_id}
            )
            paper_row = paper_result.fetchone()
            
            if not paper_row:
                return {}
            
            # Get all sections
            sections_result = conn.execute(
                text("""
                    SELECT section_id, header_path, header_level,
                           start_chunk_index, end_chunk_index,
                           page_start, page_end
                    FROM sections
                    WHERE paper_id = :paper_id
                    ORDER BY start_chunk_index ASC
                """),
                {"paper_id": paper_id}
            )
            
            sections = []
            for row in sections_result:
                sections.append({
                    "section_id": row[0],
                    "header_path": row[1],
                    "header_level": row[2],
                    "start_chunk_index": row[3],
                    "end_chunk_index": row[4],
                    "page_start": row[5],
                    "page_end": row[6]
                })
            
            return {
                "paper_id": paper_row[0],
                "url": paper_row[1],
                "filename": paper_row[2],
                "title": paper_row[3],
                "num_chunks": paper_row[4],
                "num_pages": paper_row[5],
                "chunking_method": paper_row[6],
                "sections": sections
            }
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_paper_structure",
            "paper_id": paper_id
        })
        return {}


def get_all_chunks_for_paper(paper_id: int) -> List[Dict[str, Any]]:
    """
    Get all chunks for a paper, including their database IDs.
    
    This is used for embedding generation.
    
    Args:
        paper_id: ID of the paper
        
    Returns:
        List of chunk dictionaries with all metadata
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT chunk_id, chunk_index, text, header_path, header_level,
                           page_start, page_end, embedding_index
                    FROM chunks
                    WHERE paper_id = :paper_id
                    ORDER BY chunk_index ASC
                """),
                {"paper_id": paper_id}
            )
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "paper_id": paper_id,
                    "chunk_index": row[1],
                    "text": row[2],
                    "header_path": row[3],
                    "header_level": row[4],
                    "page_start": row[5],
                    "page_end": row[6],
                    "embedding_index": row[7]
                })
            
            return chunks
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_all_chunks_for_paper",
            "paper_id": paper_id
        })
        return []


def update_chunk_embedding_index(chunk_id: int, embedding_index: int) -> bool:
    """
    Update the embedding_index for a chunk.
    
    Args:
        chunk_id: Database ID of the chunk
        embedding_index: Index in the FAISS vector store
        
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE chunks
                    SET embedding_index = :embedding_index
                    WHERE chunk_id = :chunk_id
                """),
                {"chunk_id": chunk_id, "embedding_index": embedding_index}
            )
            conn.commit()
        
        return True
        
    except Exception as e:
        log_error_with_context(e, {
            "operation": "update_chunk_embedding_index",
            "chunk_id": chunk_id,
            "embedding_index": embedding_index
        })
        return False


def update_chunks_embedding_indices(updates: List[Tuple[int, int]]) -> bool:
    """
    Update embedding_index for multiple chunks in a batch.
    
    Args:
        updates: List of (chunk_id, embedding_index) tuples
        
    Returns:
        True if successful, False otherwise
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            for chunk_id, embedding_index in updates:
                conn.execute(
                    text("""
                        UPDATE chunks
                        SET embedding_index = :embedding_index
                        WHERE chunk_id = :chunk_id
                    """),
                    {"chunk_id": chunk_id, "embedding_index": embedding_index}
                )
            conn.commit()
        
        logger.info(f"Updated {len(updates)} chunk embedding indices")
        return True
        
    except Exception as e:
        log_error_with_context(e, {
            "operation": "update_chunks_embedding_indices",
            "num_updates": len(updates)
        })
        return False


def get_chunks_by_ids(chunk_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Retrieve multiple chunks by their chunk_id values.
    
    Args:
        chunk_ids: List of chunk_id values
        
    Returns:
        List of chunk dictionaries
    """
    try:
        if not chunk_ids:
            return []
        
        engine = get_engine()
        
        with engine.connect() as conn:
            # Create placeholders for IN clause
            placeholders = ",".join([f":id{i}" for i in range(len(chunk_ids))])
            params = {f"id{i}": chunk_id for i, chunk_id in enumerate(chunk_ids)}
            
            result = conn.execute(
                text(f"""
                    SELECT c.chunk_id, c.paper_id, c.chunk_index, c.text, 
                           c.header_path, c.header_level, c.page_start, c.page_end,
                           c.prev_chunk_index, c.next_chunk_index, c.section_id, 
                           c.embedding_index, p.filename, p.title
                    FROM chunks c
                    JOIN papers p ON c.paper_id = p.paper_id
                    WHERE c.chunk_id IN ({placeholders})
                    ORDER BY c.paper_id, c.chunk_index
                """),
                params
            )
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "paper_id": row[1],
                    "chunk_index": row[2],
                    "text": row[3],
                    "header_path": row[4],
                    "header_level": row[5],
                    "page_start": row[6],
                    "page_end": row[7],
                    "prev_chunk_index": row[8],
                    "next_chunk_index": row[9],
                    "section_id": row[10],
                    "embedding_index": row[11],
                    "filename": row[12],
                    "title": row[13]
                })
            
            return chunks
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_chunks_by_ids",
            "num_chunk_ids": len(chunk_ids)
        })
        return []


def get_chunks_by_page_range(paper_id: int, page_start: int, page_end: int) -> List[Dict[str, Any]]:
    """
    Retrieve chunks that overlap with a page range.
    
    Args:
        paper_id: ID of the paper
        page_start: Starting page number (0-indexed)
        page_end: Ending page number (0-indexed, inclusive)
        
    Returns:
        List of chunk dictionaries that overlap the page range
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT chunk_id, chunk_index, text, header_path, header_level,
                           page_start, page_end, prev_chunk_index, next_chunk_index,
                           section_id, embedding_index
                    FROM chunks
                    WHERE paper_id = :paper_id
                    AND NOT (page_end < :page_start OR page_start > :page_end)
                    ORDER BY chunk_index ASC
                """),
                {"paper_id": paper_id, "page_start": page_start, "page_end": page_end}
            )
            
            chunks = []
            for row in result:
                chunks.append({
                    "chunk_id": row[0],
                    "paper_id": paper_id,
                    "chunk_index": row[1],
                    "text": row[2],
                    "header_path": row[3],
                    "header_level": row[4],
                    "page_start": row[5],
                    "page_end": row[6],
                    "prev_chunk_index": row[7],
                    "next_chunk_index": row[8],
                    "section_id": row[9],
                    "embedding_index": row[10]
                })
            
            return chunks
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_chunks_by_page_range",
            "paper_id": paper_id,
            "page_start": page_start,
            "page_end": page_end
        })
        return []


def get_chunk_by_embedding_index(embedding_index: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a chunk by its embedding_index.
    
    Args:
        embedding_index: Index in the FAISS vector store
        
    Returns:
        Dictionary with chunk data or None if not found
    """
    try:
        engine = get_engine()
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT chunk_id, paper_id, chunk_index, text, header_path, header_level,
                           page_start, page_end, prev_chunk_index, next_chunk_index,
                           section_id, embedding_index
                    FROM chunks
                    WHERE embedding_index = :embedding_index
                """),
                {"embedding_index": embedding_index}
            )
            row = result.fetchone()
            
            if row:
                return {
                    "chunk_id": row[0],
                    "paper_id": row[1],
                    "chunk_index": row[2],
                    "text": row[3],
                    "header_path": row[4],
                    "header_level": row[5],
                    "page_start": row[6],
                    "page_end": row[7],
                    "prev_chunk_index": row[8],
                    "next_chunk_index": row[9],
                    "section_id": row[10],
                    "embedding_index": row[11]
                }
            return None
            
    except Exception as e:
        log_error_with_context(e, {
            "operation": "get_chunk_by_embedding_index",
            "embedding_index": embedding_index
        })
        return None

