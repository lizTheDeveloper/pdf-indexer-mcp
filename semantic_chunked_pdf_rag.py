"""
PDF Research Paper Indexing MCP Server

An MCP server that enables AI agents to download PDF research papers,
chunk them semantically using multiple strategies (header-based and S2),
and perform semantic search using embeddings.
"""

import os
import re
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import httpx
import fitz  # PyMuPDF
from fastmcp import FastMCP

# Ensure the package root is in the Python path for standalone execution
PACKAGE_ROOT = Path(__file__).parent.absolute()
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.logger import get_logger, log_performance_metric, log_error_with_context
from pdf_processing import extract_text_with_headers, validate_pdf_content
from chunking import chunk_by_headers, chunk_by_s2
from database.operations import (
    initialize_database,
    insert_paper,
    insert_chunks,
    insert_sections,
    get_paper_by_filename,
    list_all_papers,
    get_paper_structure,
    get_all_chunks_for_paper,
    update_chunks_embedding_indices,
    get_chunks_by_ids,
    get_chunks_range,
    get_chunks_by_page_range,
    get_section,
    get_chunk
)
from embeddings import get_embedding_generator, FAISSIndexManager

# Initialize logger
logger = get_logger()

# Initialize FastMCP server
mcp = FastMCP("PDF Research Paper Indexer")

# Directory configuration - use absolute paths relative to this script
SCRIPT_DIR = Path(__file__).parent.absolute()
PAPERS_DIR = SCRIPT_DIR / "papers"
INDEXES_DIR = SCRIPT_DIR / "indexes"

# Ensure directories exist
PAPERS_DIR.mkdir(exist_ok=True)
INDEXES_DIR.mkdir(exist_ok=True)

# Initialize database
if not initialize_database():
    logger.error("Failed to initialize database")
else:
    logger.info("Database initialized successfully")


def extract_filename_from_url(url: str) -> str:
    """
    Extract a filename from a URL.
    
    Args:
        url: The URL to extract filename from
        
    Returns:
        A sanitized filename derived from the URL
    """
    parsed = urlparse(url)
    
    # Try to get filename from path
    path_parts = parsed.path.strip('/').split('/')
    if path_parts and path_parts[-1]:
        filename = path_parts[-1]
        # Remove query parameters from filename
        if '?' in filename:
            filename = filename.split('?')[0]
    else:
        # Generate filename from domain and path
        domain = parsed.netloc.replace('.', '_')
        path_hash = str(abs(hash(parsed.path)))[:8]
        filename = f"{domain}_{path_hash}.pdf"
    
    # Ensure it ends with .pdf
    if not filename.lower().endswith('.pdf'):
        filename += '.pdf'
    
    # Sanitize filename (remove invalid characters)
    filename = re.sub(r'[<>:"|?*]', '_', filename)
    # Limit filename length
    if len(filename) > 200:
        name, ext = os.path.splitext(filename)
        filename = name[:190] + ext
    
    return filename


@mcp.tool
def download_pdf(url: str) -> dict:
    """
    Download a PDF research paper from a URL and save it to the papers directory.
    
    Use this tool when:
    - User asks to download a paper from a URL (arXiv, direct PDF links, etc.)
    - User provides a URL to a PDF document
    - You need to save a PDF locally before indexing it
    
    The downloaded PDF will be saved in the papers/ directory with a sanitized filename.
    If the PDF already exists, it will not be re-downloaded.
    
    Args:
        url: The URL of the PDF document to download (must include http:// or https://)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if download was successful
        - filename: str with the saved filename (if successful, e.g., "1706.03762.pdf")
        - filepath: str with the absolute path to the saved file (if successful)
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    Example:
        download_pdf("https://arxiv.org/pdf/1706.03762.pdf")
        → Returns: {"success": True, "filename": "1706.03762.pdf", ...}
    """
    logger.info(f"Starting PDF download from URL: {url}")
    
    # Validate URL format
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            error_msg = f"Invalid URL format: {url}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "invalid_url",
                "message": error_msg
            }
    except Exception as e:
        log_error_with_context(e, {"url": url, "operation": "url_validation"})
        return {
            "success": False,
            "error": "url_validation_error",
            "message": f"Failed to validate URL: {str(e)}"
        }
    
    # Extract filename from URL
    filename = extract_filename_from_url(url)
    filepath = PAPERS_DIR / filename
    
    # Check if file already exists
    if filepath.exists():
        logger.info(f"PDF already exists at {filepath}, skipping download")
        return {
            "success": True,
            "filename": filename,
            "filepath": str(filepath.absolute()),
            "message": f"PDF already exists at {filepath}"
        }
    
    # Download PDF
    try:
        import time
        start_time = time.time()
        
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            
            # Validate content type if available
            content_type = response.headers.get('content-type', '').lower()
            if content_type and 'application/pdf' not in content_type and 'application/octet-stream' not in content_type:
                logger.warning(f"Unexpected content type: {content_type} for URL: {url}")
            
            # Validate PDF content
            pdf_content = response.content
            if not validate_pdf_content(pdf_content):
                error_msg = f"Downloaded content is not a valid PDF file (URL: {url})"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": "invalid_pdf",
                    "message": error_msg
                }
            
            # Save to file
            with open(filepath, 'wb') as f:
                f.write(pdf_content)
            
            duration = time.time() - start_time
            file_size_mb = len(pdf_content) / (1024 * 1024)
            
            log_performance_metric(
                "download_pdf",
                duration,
                url=url,
                filename=filename,
                size_mb=round(file_size_mb, 2)
            )
            
            logger.info(f"Successfully downloaded PDF: {filename} ({file_size_mb:.2f} MB) from {url}")
            
            return {
                "success": True,
                "filename": filename,
                "filepath": str(filepath.absolute()),
                "message": f"Successfully downloaded PDF: {filename}"
            }
    
    except httpx.TimeoutException as e:
        error_msg = f"Timeout while downloading PDF from {url}"
        log_error_with_context(e, {"url": url, "operation": "download_pdf", "error_type": "timeout"})
        return {
            "success": False,
            "error": "timeout",
            "message": error_msg
        }
    
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error {e.response.status_code} while downloading PDF from {url}"
        log_error_with_context(e, {"url": url, "status_code": e.response.status_code, "operation": "download_pdf"})
        return {
            "success": False,
            "error": "http_error",
            "message": error_msg
        }
    
    except httpx.RequestError as e:
        error_msg = f"Network error while downloading PDF from {url}: {str(e)}"
        log_error_with_context(e, {"url": url, "operation": "download_pdf", "error_type": "network"})
        return {
            "success": False,
            "error": "network_error",
            "message": error_msg
        }
    
    except IOError as e:
        error_msg = f"Failed to save PDF file to {filepath}: {str(e)}"
        log_error_with_context(e, {"url": url, "filepath": str(filepath), "operation": "save_file"})
        return {
            "success": False,
            "error": "file_write_error",
            "message": error_msg
        }
    
    except Exception as e:
        error_msg = f"Unexpected error while downloading PDF from {url}: {str(e)}"
        log_error_with_context(e, {"url": url, "operation": "download_pdf", "error_type": "unexpected"})
        return {
            "success": False,
            "error": "unexpected_error",
            "message": error_msg
        }


@mcp.tool
def chunk_pdf(filename: str, method: str = "header") -> dict:
    """
    Extract text from a PDF and divide it into semantic chunks using header-based or S2 chunking.
    
    Use this tool when:
    - You need to analyze the structure of a PDF before indexing
    - User asks how a PDF is organized or wants to see chunks
    - You need to preview chunks before indexing
    
    Chunking methods:
    - "header": Preserves document structure, groups content under headers. Best for academic papers with clear sections.
    - "s2": Spatial-semantic hybrid approach. Uses layout analysis and semantic similarity. Best for unstructured documents.
    
    Args:
        filename: The filename of the PDF in ./papers/ directory (e.g., "1706.03762.pdf")
        method: Chunking method to use - "header" (default, preserves structure) or "s2" (spatial-semantic)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if chunking was successful
        - num_chunks: int with number of chunks created (if successful)
        - chunks: list of chunk dictionaries with preview text and metadata
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    Note: This only chunks the PDF - it does NOT store chunks in the database.
    Use index_pdf() to both chunk and store in the database.
    """
    logger.info(f"Starting PDF chunking for file: {filename} using method: {method}")
    
    # Validate method
    if method not in ["header", "s2"]:
        error_msg = f"Invalid chunking method: {method}. Must be 'header' or 's2'"
        logger.error(error_msg)
        return {
            "success": False,
            "error": "invalid_method",
            "message": error_msg
        }
    
    filepath = PAPERS_DIR / filename
    
    # Check if file exists
    if not filepath.exists():
        error_msg = f"PDF file not found: {filepath}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": "file_not_found",
            "message": error_msg
        }
    
    # Validate it's a PDF
    if not filename.lower().endswith('.pdf'):
        error_msg = f"File is not a PDF: {filename}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": "not_a_pdf",
            "message": error_msg
        }
    
    try:
        import time
        start_time = time.time()
        
        # Open PDF with PyMuPDF
        doc = fitz.open(filepath)
        logger.info(f"Opened PDF: {filename}, pages: {len(doc)}")
        
        # Extract text with header detection
        segments = extract_text_with_headers(doc)
        logger.info(f"Extracted {len(segments)} text segments")
        
        # Apply selected chunking method
        if method == "header":
            chunks = chunk_by_headers(segments)
        else:  # method == "s2"
            chunks = chunk_by_s2(segments, max_token_length=512)
        
        logger.info(f"Created {len(chunks)} chunks using {method} method")
        
        # Convert chunks to serializable format
        chunk_list = []
        for chunk in chunks:
            chunk_list.append({
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"][:1000] + "..." if len(chunk["text"]) > 1000 else chunk["text"],
                "full_text_length": len(chunk["text"]),
                "header_path": chunk.get("header_path", ""),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "header_level": chunk.get("header_level", 0)
            })
        
        duration = time.time() - start_time
        
        log_performance_metric(
            f"chunk_pdf_{method}",
            duration,
            filename=filename,
            num_chunks=len(chunks),
            num_segments=len(segments),
            method=method
        )
        
        logger.info(f"Successfully chunked PDF: {filename} into {len(chunks)} chunks using {method}")
        
        doc.close()
        
        return {
            "success": True,
            "method": method,
            "num_chunks": len(chunks),
            "chunks": chunk_list,
            "message": f"Successfully chunked PDF into {len(chunks)} chunks using {method} method"
        }
    
    except fitz.FileDataError as e:
        error_msg = f"Invalid or corrupted PDF file: {filename}"
        log_error_with_context(e, {"filename": filename, "operation": "chunk_pdf", "error_type": "file_data"})
        return {
            "success": False,
            "error": "invalid_pdf",
            "message": error_msg
        }
    
    except Exception as e:
        error_msg = f"Unexpected error while chunking PDF {filename}: {str(e)}"
        log_error_with_context(e, {"filename": filename, "operation": "chunk_pdf", "error_type": "unexpected", "method": method})
        return {
            "success": False,
            "error": "unexpected_error",
            "message": error_msg
        }


@mcp.tool
def index_pdf(filename: str, url: str = "", method: str = "header") -> dict:
    """
    Index a PDF by extracting chunks and storing them in the database for search and retrieval.
    
    This is the PRIMARY tool for making papers searchable. It performs a complete indexing workflow:
    1. Extracts and chunks the PDF text
    2. Stores chunks in the SQLite database with navigation indices
    3. Creates section mappings for header-based navigation (if using header method)
    
    Use this tool when:
    - User asks to index a paper or make it searchable
    - User wants to add a paper to the knowledge base
    - You need to prepare a paper for semantic search
    
    After indexing, use generate_embeddings() to make the paper searchable via semantic search.
    
    Args:
        filename: The filename of the PDF in ./papers/ directory (must already be downloaded)
        url: Original URL where the PDF was downloaded from (optional, for metadata)
        method: Chunking method - "header" (default, best for academic papers) or "s2" (spatial-semantic)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if indexing was successful
        - paper_id: int with database ID of the paper (if successful, use this for future queries)
        - num_chunks: int with number of chunks created (if successful)
        - num_sections: int with number of sections created (header method only)
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    Example workflow:
        1. download_pdf(url) → gets filename
        2. index_pdf(filename, url=url, method="header") → stores in database
        3. generate_embeddings(filename) → makes it searchable
    """
    logger.info(f"Starting PDF indexing for file: {filename} using method: {method}")
    
    # Validate method
    if method not in ["header", "s2"]:
        error_msg = f"Invalid chunking method: {method}. Must be 'header' or 's2'"
        logger.error(error_msg)
        return {
            "success": False,
            "error": "invalid_method",
            "message": error_msg
        }
    
    filepath = PAPERS_DIR / filename
    
    # Check if file exists
    if not filepath.exists():
        error_msg = f"PDF file not found: {filepath}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": "file_not_found",
            "message": error_msg
        }
    
    # Check if paper is already indexed
    existing_paper = get_paper_by_filename(filename)
    if existing_paper:
        logger.info(f"Paper already indexed: {filename}, paper_id: {existing_paper['paper_id']}")
        return {
            "success": True,
            "paper_id": existing_paper["paper_id"],
            "num_chunks": existing_paper["num_chunks"],
            "message": f"Paper already indexed (paper_id: {existing_paper['paper_id']})"
        }
    
    try:
        import time
        start_time = time.time()
        
        # Open PDF with PyMuPDF
        doc = fitz.open(filepath)
        num_pages = len(doc)
        logger.info(f"Opened PDF: {filename}, pages: {num_pages}")
        
        # Extract text with header detection
        segments = extract_text_with_headers(doc)
        logger.info(f"Extracted {len(segments)} text segments")
        
        # Apply selected chunking method
        if method == "header":
            chunks = chunk_by_headers(segments)
        else:  # method == "s2"
            chunks = chunk_by_s2(segments, max_token_length=512)
        
        logger.info(f"Created {len(chunks)} chunks using {method} method")
        doc.close()
        
        # Extract title from first chunk if available
        title = None
        if chunks and chunks[0].get("header_path"):
            # Use first header as title
            first_header = chunks[0]["header_path"].split(" > ")[0]
            title = first_header if first_header else None
        
        # Insert paper into database
        paper_id = insert_paper(
            url=url if url else f"file://{filepath.absolute()}",
            filename=filename,
            title=title,
            num_pages=num_pages,
            chunking_method=method
        )
        
        if not paper_id:
            return {
                "success": False,
                "error": "database_error",
                "message": "Failed to insert paper into database"
            }
        
        # Insert chunks into database
        if not insert_chunks(paper_id, chunks):
            return {
                "success": False,
                "error": "database_error",
                "message": "Failed to insert chunks into database"
            }
        
        # Insert sections (only for header-based chunking)
        num_sections = 0
        if method == "header":
            if insert_sections(paper_id, chunks):
                # Count unique header paths
                unique_headers = set(c.get("header_path", "") for c in chunks if c.get("header_path"))
                num_sections = len(unique_headers)
            else:
                logger.warning("Failed to insert sections, but continuing")
        
        duration = time.time() - start_time
        
        log_performance_metric(
            f"index_pdf_{method}",
            duration,
            filename=filename,
            paper_id=paper_id,
            num_chunks=len(chunks),
            num_sections=num_sections,
            method=method
        )
        
        logger.info(f"Successfully indexed PDF: {filename}, paper_id: {paper_id}, {len(chunks)} chunks, {num_sections} sections")
        
        result = {
            "success": True,
            "paper_id": paper_id,
            "num_chunks": len(chunks),
            "message": f"Successfully indexed PDF: {len(chunks)} chunks stored in database (paper_id: {paper_id})"
        }
        
        if num_sections > 0:
            result["num_sections"] = num_sections
        
        return result
    
    except fitz.FileDataError as e:
        error_msg = f"Invalid or corrupted PDF file: {filename}"
        log_error_with_context(e, {"filename": filename, "operation": "index_pdf", "error_type": "file_data"})
        return {
            "success": False,
            "error": "invalid_pdf",
            "message": error_msg
        }
    
    except Exception as e:
        error_msg = f"Unexpected error while indexing PDF {filename}: {str(e)}"
        log_error_with_context(e, {"filename": filename, "operation": "index_pdf", "error_type": "unexpected", "method": method})
        return {
            "success": False,
            "error": "unexpected_error",
            "message": error_msg
        }


@mcp.tool
def list_indexed_papers() -> dict:
    """
    List all papers that have been indexed in the database.
    
    Use this tool when:
    - User asks "what papers do you have?" or "show me all papers"
    - You need to check what papers are available before searching
    - User wants to see the papers in the knowledge base
    
    Returns:
        Dictionary containing:
        - success: bool indicating if query was successful
        - papers: list of paper dictionaries with metadata (filename, title, num_chunks, etc.)
        - count: int with number of papers (if successful)
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    Each paper in the list includes: filename, title, paper_id, num_chunks, and download_date.
    """
    try:
        papers = list_all_papers()
        
        # Convert datetime objects to strings for serialization
        for paper in papers:
            if paper.get("download_date"):
                paper["download_date"] = str(paper["download_date"])
        
        return {
            "success": True,
            "count": len(papers),
            "papers": papers,
            "message": f"Found {len(papers)} indexed papers"
        }
        
    except Exception as e:
        error_msg = f"Error listing papers: {str(e)}"
        log_error_with_context(e, {"operation": "list_indexed_papers"})
        return {
            "success": False,
            "error": "database_error",
            "message": error_msg
        }


@mcp.tool
def get_document_structure(filename: str) -> dict:
    """
    Get the structure of an indexed document including sections, headers, and chunk ranges.
    
    Use this tool when:
    - User asks about a paper's structure or sections
    - User wants to know what sections a paper contains
    - You need to understand a paper's organization before retrieving specific sections
    
    This shows the hierarchical structure of the paper (headers, sections, chunk ranges).
    Use get_document_section() to retrieve the actual content of specific sections.
    
    Args:
        filename: The filename of the PDF in ./papers/ directory (must be indexed)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if query was successful
        - structure: dict with paper metadata and sections list (if successful)
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    The structure includes paper metadata and a list of sections with their header paths and chunk ranges.
    """
    try:
        # Get paper by filename
        paper = get_paper_by_filename(filename)
        
        if not paper:
            return {
                "success": False,
                "error": "paper_not_found",
                "message": f"Paper not found in database: {filename}"
            }
        
        # Get paper structure
        structure = get_paper_structure(paper["paper_id"])
        
        if not structure:
            return {
                "success": False,
                "error": "database_error",
                "message": f"Failed to retrieve structure for paper: {filename}"
            }
        
        # Convert datetime to string
        if structure.get("download_date"):
            structure["download_date"] = str(structure["download_date"])
        
        return {
            "success": True,
            "structure": structure,
            "message": f"Retrieved structure for {filename}: {structure.get('num_chunks', 0)} chunks, {len(structure.get('sections', []))} sections"
        }
        
    except Exception as e:
        error_msg = f"Error retrieving document structure: {str(e)}"
        log_error_with_context(e, {"operation": "get_document_structure", "filename": filename})
        return {
            "success": False,
            "error": "database_error",
            "message": error_msg
        }


@mcp.tool
def search_research_papers(
    query: str,
    k: int = 5,
    context_window: int = 1,
    model_name: str = "mlx-community/Qwen3-Embedding-0.6B"
) -> dict:
    """
    Search indexed research papers using semantic similarity (meaning-based search, not keyword matching).
    
    This is the PRIMARY tool for finding relevant content. It performs semantic search across ALL indexed papers.
    
    Use this tool when:
    - User asks questions like "find papers about X" or "search for information on Y"
    - User wants to discover what papers discuss a topic
    - You need to find relevant chunks across the entire knowledge base
    - User asks "what do the papers say about X?"
    
    How it works:
    1. Generates a semantic embedding for the query text
    2. Searches the FAISS vector index for k most similar chunks
    3. Retrieves matched chunks with surrounding context (previous/next chunks)
    4. Returns chunks ranked by similarity with full metadata
    
    The search understands meaning, not just keywords. "attention mechanisms" will find chunks discussing attention even if they don't contain the exact words.
    
    Args:
        query: Search query text (describe what you're looking for in natural language)
        k: Number of top results to return (default: 5, increase for more results)
        context_window: Number of neighboring chunks to include before/after each match (default: 1, increase for more context)
        model_name: Embedding model identifier (default: Qwen3-Embedding-0.6B, 1024 dimensions)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if search was successful
        - query: str with the original query
        - num_results: int with number of results found
        - results: list of result dictionaries ranked by similarity, each containing:
            - chunk_id: Database ID of the chunk
            - paper_id: Database ID of the paper
            - filename: Filename of the paper (e.g., "1706.03762.pdf")
            - title: Title of the paper (if available)
            - chunk_index: Index of the chunk within the paper
            - text: Full text of the chunk (the relevant content)
            - header_path: Header hierarchy path (e.g., "Introduction", "Methods.Experimental Setup")
            - header_level: Header level (0-3)
            - page_start: First page number (0-indexed)
            - page_end: Last page number (0-indexed)
            - distance: Similarity distance (lower is more similar, 0.0 = identical)
            - is_context: bool indicating if this is a context chunk (not a direct match)
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    Example:
        search_research_papers("transformer attention mechanisms", k=5)
        → Returns top 5 chunks discussing attention mechanisms across all papers
    
    Note: Papers must be indexed (index_pdf) and have embeddings generated (generate_embeddings) before they can be searched.
    """
    logger.info(f"Starting semantic search for query: '{query[:100]}...'")
    
    try:
        import time
        start_time = time.time()
        
        # Initialize embedding generator
        embedding_gen = get_embedding_generator(model_name)
        embedding_dim = embedding_gen.get_embedding_dimension()
        
        # Generate query embedding
        logger.info(f"Generating query embedding...")
        query_embeddings = embedding_gen.generate_embeddings([query], batch_size=1)
        query_embedding = query_embeddings[0]
        
        # Initialize FAISS index manager
        faiss_manager = FAISSIndexManager(
            index_name="research_papers",
            embedding_dim=embedding_dim
        )
        
        # Load existing index
        if not faiss_manager.load_index():
            error_msg = "FAISS index not found. Please generate embeddings first using generate_embeddings tool."
            logger.error(error_msg)
            return {
                "success": False,
                "error": "index_not_found",
                "message": error_msg
            }
        
        if faiss_manager.index.ntotal == 0:
            error_msg = "FAISS index is empty. Please generate embeddings first."
            logger.error(error_msg)
            return {
                "success": False,
                "error": "empty_index",
                "message": error_msg
            }
        
        # Search FAISS index
        logger.info(f"Searching FAISS index for top {k} matches...")
        chunk_ids, distances = faiss_manager.search(query_embedding, k=k)
        
        if not chunk_ids:
            logger.info("No results found")
            return {
                "success": True,
                "query": query,
                "num_results": 0,
                "results": [],
                "message": "No results found for query"
            }
        
        logger.info(f"Found {len(chunk_ids)} matches")
        
        # Retrieve matched chunks with metadata
        matched_chunks = get_chunks_by_ids(chunk_ids)
        
        # Build result list with distances
        chunk_id_to_distance = {cid: dist for cid, dist in zip(chunk_ids, distances)}
        
        results = []
        added_chunk_ids = set()
        
        # Process each matched chunk
        for chunk in matched_chunks:
            chunk_id = chunk["chunk_id"]
            paper_id = chunk["paper_id"]
            chunk_index = chunk["chunk_index"]
            
            # Add the matched chunk
            if chunk_id not in added_chunk_ids:
                results.append({
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "filename": chunk.get("filename", ""),
                    "title": chunk.get("title", ""),
                    "chunk_index": chunk_index,
                    "text": chunk["text"],
                    "header_path": chunk.get("header_path", ""),
                    "header_level": chunk.get("header_level", 0),
                    "page_start": chunk["page_start"],
                    "page_end": chunk["page_end"],
                    "distance": chunk_id_to_distance.get(chunk_id, 0.0),
                    "is_context": False
                })
                added_chunk_ids.add(chunk_id)
            
            # Add context chunks if requested
            if context_window > 0:
                # Get context range
                start_idx = max(0, chunk_index - context_window)
                end_idx = chunk_index + context_window
                
                context_chunks = get_chunks_range(paper_id, start_idx, end_idx)
                
                for ctx_chunk in context_chunks:
                    ctx_chunk_id = ctx_chunk["chunk_id"]
                    
                    # Skip if already added (including the matched chunk itself)
                    if ctx_chunk_id in added_chunk_ids:
                        continue
                    
                    results.append({
                        "chunk_id": ctx_chunk_id,
                        "paper_id": paper_id,
                        "filename": chunk.get("filename", ""),
                        "title": chunk.get("title", ""),
                        "chunk_index": ctx_chunk["chunk_index"],
                        "text": ctx_chunk["text"],
                        "header_path": ctx_chunk.get("header_path", ""),
                        "header_level": ctx_chunk.get("header_level", 0),
                        "page_start": ctx_chunk["page_start"],
                        "page_end": ctx_chunk["page_end"],
                        "distance": None,  # Context chunks don't have distances
                        "is_context": True
                    })
                    added_chunk_ids.add(ctx_chunk_id)
        
        duration = time.time() - start_time
        
        log_performance_metric(
            "search_research_papers",
            duration,
            query_length=len(query),
            k=k,
            context_window=context_window,
            num_results=len(results)
        )
        
        logger.info(f"Search completed: {len(results)} total chunks (including context) in {duration:.2f}s")
        
        return {
            "success": True,
            "query": query,
            "num_results": len(results),
            "results": results,
            "message": f"Found {len(chunk_ids)} matches with {len(results) - len(chunk_ids)} context chunks"
        }
    
    except Exception as e:
        error_msg = f"Unexpected error during search: {str(e)}"
        log_error_with_context(e, {
            "operation": "search_research_papers",
            "query": query[:100],
            "k": k,
            "context_window": context_window
        })
        return {
            "success": False,
            "error": "unexpected_error",
            "message": error_msg
        }


@mcp.tool
def get_document_section(
    filename: str,
    chunk_index: Optional[int] = None,
    header_path: Optional[str] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None
) -> dict:
    """
    Retrieve a specific section or range of chunks from an indexed document.
    
    Use this tool when:
    - User asks "show me the Introduction section" or "get the Methods section"
    - User wants to see a specific page range (e.g., "pages 5-10")
    - You need to retrieve a specific chunk by index
    - User asks for detailed content from a specific part of a paper
    
    This tool retrieves the actual content, unlike get_document_structure() which only shows the structure.
    
    You can retrieve sections in three ways:
    1. By header_path: "Introduction", "Methods.Experimental Setup", etc.
    2. By page range: page_start and page_end (0-indexed)
    3. By chunk_index: specific chunk number (0-indexed)
    
    Args:
        filename: The filename of the PDF in ./papers/ directory (must be indexed)
        chunk_index: (Optional) Index of a specific chunk to retrieve (0-based)
        header_path: (Optional) Header path to retrieve (e.g., "Introduction", "Methods.Results")
        page_start: (Optional) Starting page number (0-indexed)
        page_end: (Optional) Ending page number (0-indexed)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if retrieval was successful
        - paper_id: int with database ID of the paper (if successful)
        - filename: str with filename
        - num_chunks: int with number of chunks returned (if successful)
        - chunks: list of chunk dictionaries with full text and metadata (if successful)
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    Each chunk includes: chunk_index, text (full content), header_path, header_level, page_start, page_end.
    
    Note: Use get_document_structure() first to see available sections and header paths.
    """
    logger.info(f"Getting document section for: {filename}")
    
    # Validate that at least one query parameter is provided
    if chunk_index is None and header_path is None and page_start is None:
        error_msg = "Must provide at least one of: chunk_index, header_path, or page_start/page_end"
        logger.error(error_msg)
        return {
            "success": False,
            "error": "missing_parameters",
            "message": error_msg
        }
    
    try:
        # Get paper from database
        paper = get_paper_by_filename(filename)
        if not paper:
            error_msg = f"Paper not found in database: {filename}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "paper_not_found",
                "message": error_msg
            }
        
        paper_id = paper["paper_id"]
        chunks = []
        
        # Query by chunk_index
        if chunk_index is not None:
            logger.info(f"Querying by chunk_index: {chunk_index}")
            chunk = get_chunk(paper_id, chunk_index)
            if chunk:
                chunks = [chunk]
            else:
                logger.warning(f"Chunk not found: paper_id={paper_id}, chunk_index={chunk_index}")
        
        # Query by header_path
        elif header_path is not None:
            logger.info(f"Querying by header_path: {header_path}")
            section = get_section(paper_id, header_path)
            if section:
                chunks = get_chunks_range(
                    paper_id,
                    section["start_chunk_index"],
                    section["end_chunk_index"]
                )
            else:
                logger.warning(f"Section not found: paper_id={paper_id}, header_path={header_path}")
        
        # Query by page range
        elif page_start is not None:
            if page_end is None:
                page_end = page_start  # Single page query
            logger.info(f"Querying by page range: {page_start}-{page_end}")
            chunks = get_chunks_by_page_range(paper_id, page_start, page_end)
        
        # Convert chunks to serializable format
        chunk_list = []
        for chunk in chunks:
            chunk_list.append({
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "header_path": chunk.get("header_path", ""),
                "header_level": chunk.get("header_level", 0),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "prev_chunk_index": chunk.get("prev_chunk_index"),
                "next_chunk_index": chunk.get("next_chunk_index"),
                "section_id": chunk.get("section_id"),
                "embedding_index": chunk.get("embedding_index")
            })
        
        logger.info(f"Retrieved {len(chunk_list)} chunks")
        
        return {
            "success": True,
            "paper_id": paper_id,
            "filename": filename,
            "num_chunks": len(chunk_list),
            "chunks": chunk_list,
            "message": f"Retrieved {len(chunk_list)} chunks from {filename}"
        }
    
    except Exception as e:
        error_msg = f"Unexpected error retrieving document section: {str(e)}"
        log_error_with_context(e, {
            "operation": "get_document_section",
            "filename": filename,
            "chunk_index": chunk_index,
            "header_path": header_path,
            "page_start": page_start,
            "page_end": page_end
        })
        return {
            "success": False,
            "error": "unexpected_error",
            "message": error_msg
        }


@mcp.tool
def generate_embeddings(filename: str, model_name: str = "mlx-community/Qwen3-Embedding-0.6B") -> dict:
    """
    Generate semantic embeddings for all chunks in a paper and add them to the FAISS vector index for search.
    
    This tool is REQUIRED to make a paper searchable via semantic search. After indexing a paper (index_pdf),
    you must generate embeddings before the paper can be found by search_research_papers().
    
    Use this tool when:
    - User asks to make a paper searchable
    - You've just indexed a paper and need to enable semantic search
    - A paper is indexed but search_research_papers() returns no results
    
    How it works:
    1. Loads all chunks from the database for the specified paper
    2. Generates semantic embeddings using MLX (optimized for Apple Silicon, ~35 embeddings/second)
    3. Adds embeddings to the FAISS vector index for fast similarity search
    4. Updates database with embedding indices
    5. Persists the FAISS index to disk (survives server restarts)
    
    Embeddings capture semantic meaning - similar concepts have similar embeddings even if words differ.
    
    Args:
        filename: The filename of the PDF in ./papers/ directory (must be indexed first)
        model_name: Hugging Face model identifier for embeddings (default: Qwen3-Embedding-0.6B, 1024 dimensions)
        
    Returns:
        Dictionary containing:
        - success: bool indicating if embedding generation was successful
        - paper_id: int with database ID of the paper (if successful)
        - num_embeddings: int with number of embeddings generated (if successful)
        - embedding_dim: int with dimension of embeddings (if successful, 1024 for Qwen3-Embedding-0.6B)
        - error: str with error message (if unsuccessful)
        - message: str with descriptive message
    
    Example workflow:
        1. download_pdf(url) → downloads paper
        2. index_pdf(filename) → indexes paper (stores chunks in database)
        3. generate_embeddings(filename) → makes it searchable
        4. search_research_papers("query") → can now find content in this paper
    
    Performance: ~35 embeddings/second on Apple Silicon. First run downloads model automatically.
    """
    logger.info(f"Starting embedding generation for: {filename}")
    
    try:
        import time
        start_time = time.time()
        
        # Get paper from database
        paper = get_paper_by_filename(filename)
        if not paper:
            error_msg = f"Paper not found in database: {filename}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "paper_not_found",
                "message": error_msg
            }
        
        paper_id = paper["paper_id"]
        logger.info(f"Found paper: {filename}, paper_id: {paper_id}")
        
        # Get all chunks for the paper
        chunks = get_all_chunks_for_paper(paper_id)
        if not chunks:
            error_msg = f"No chunks found for paper: {filename}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "no_chunks",
                "message": error_msg
            }
        
        logger.info(f"Found {len(chunks)} chunks for paper_id: {paper_id}")
        
        # Check if embeddings already exist
        if any(chunk.get("embedding_index") is not None for chunk in chunks):
            logger.warning(f"Some chunks already have embeddings for paper: {filename}")
            # We'll regenerate them anyway
        
        # Initialize embedding generator
        embedding_gen = get_embedding_generator(model_name)
        embedding_dim = embedding_gen.get_embedding_dimension()
        
        # Extract text from chunks
        texts = [chunk["text"] for chunk in chunks]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = embedding_gen.generate_embeddings(texts, batch_size=32)
        
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {embedding_dim}")
        
        # Initialize FAISS index manager
        faiss_manager = FAISSIndexManager(
            index_name="research_papers",
            embedding_dim=embedding_dim
        )
        
        # Try to load existing index, or create new one
        if not faiss_manager.load_index():
            logger.info("Creating new FAISS index")
            faiss_manager.create_index()
        
        # Get current index size to calculate embedding indices
        start_embedding_index = faiss_manager.index.ntotal
        
        # Add embeddings to FAISS index
        if not faiss_manager.add_embeddings(embeddings, chunk_ids):
            error_msg = "Failed to add embeddings to FAISS index"
            logger.error(error_msg)
            return {
                "success": False,
                "error": "faiss_add_error",
                "message": error_msg
            }
        
        # Update database with embedding indices
        updates = [
            (chunk_ids[i], start_embedding_index + i)
            for i in range(len(chunk_ids))
        ]
        
        if not update_chunks_embedding_indices(updates):
            error_msg = "Failed to update database with embedding indices"
            logger.error(error_msg)
            # Continue anyway - embeddings are in FAISS
            logger.warning("Embeddings are in FAISS but database not updated")
        
        # Save FAISS index to disk
        if not faiss_manager.save_index():
            error_msg = "Failed to save FAISS index to disk"
            logger.error(error_msg)
            # This is not fatal - index is in memory
        
        duration = time.time() - start_time
        
        log_performance_metric(
            "generate_embeddings",
            duration,
            filename=filename,
            paper_id=paper_id,
            num_embeddings=len(embeddings),
            embedding_dim=embedding_dim,
            embeddings_per_sec=len(embeddings) / duration if duration > 0 else 0
        )
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings for {filename} "
                   f"in {duration:.2f}s")
        
        return {
            "success": True,
            "paper_id": paper_id,
            "filename": filename,
            "num_embeddings": len(embeddings),
            "embedding_dim": embedding_dim,
            "model_name": model_name,
            "faiss_total_vectors": faiss_manager.index.ntotal,
            "message": f"Successfully generated {len(embeddings)} embeddings for {filename}"
        }
    
    except Exception as e:
        error_msg = f"Unexpected error while generating embeddings for {filename}: {str(e)}"
        log_error_with_context(e, {
            "operation": "generate_embeddings",
            "filename": filename,
            "model_name": model_name
        })
        return {
            "success": False,
            "error": "unexpected_error",
            "message": error_msg
        }


def main():
    """Main entry point for the MCP server."""
    logger.info("Starting PDF Research Paper Indexing MCP Server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
