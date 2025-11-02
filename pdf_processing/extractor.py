"""
PDF text extraction with layout analysis.
"""
from typing import List, Dict, Any, Tuple
from collections import Counter
import re
import fitz

from ..utils.logger import get_logger

logger = get_logger()


def calculate_common_font_size(doc: fitz.Document) -> float:
    """
    Calculate the most common font size in the document (body text size).
    
    Args:
        doc: PyMuPDF document object
        
    Returns:
        Most common font size
    """
    font_sizes = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        font_sizes.append(span.get("size", 0))
    
    if not font_sizes:
        return 12.0  # Default
    
    # Return the most common font size
    size_counts = Counter(font_sizes)
    return size_counts.most_common(1)[0][0]


def is_likely_header(text: str, font_size: float, font_name: str, flags: int, 
                     common_font_size: float, x_position: float) -> Tuple[bool, int]:
    """
    Determine if a text span is likely a header based on heuristics.
    
    Args:
        text: The text content
        font_size: Font size of the text
        font_name: Font name (e.g., "CMBX12" for bold, "CMR17" for regular)
        flags: Font flags (20 = bold, 4 = regular)
        common_font_size: Most common font size in the document
        x_position: X position of the text on the page
        
    Returns:
        Tuple of (is_header: bool, header_level: int)
        header_level: 1 = main section, 2 = subsection, 3 = subsubsection
    """
    text_stripped = text.strip()
    
    # Empty or very short text is not a header
    if len(text_stripped) < 3:
        return False, 0
    
    # Check for academic paper section patterns
    section_patterns = [
        r'^(\d+\.?\s+)?(Abstract|Introduction|Related Work|Background|Methodology|Methods|Results|Discussion|Conclusion|References|Bibliography)',
        r'^(\d+\.?\s+)?[A-Z][a-z]+\s+',  # Title case followed by lowercase
        r'^\d+\.\s+[A-Z]',  # Numbered section (e.g., "1. Introduction")
        r'^[A-Z][A-Z\s]+$',  # All caps (likely a header)
    ]
    
    is_section_like = any(re.match(pattern, text_stripped, re.IGNORECASE) for pattern in section_patterns)
    
    # Font size analysis - headers are typically larger
    size_ratio = font_size / common_font_size if common_font_size > 0 else 1.0
    
    # Bold text is more likely to be a header
    is_bold = (flags & 16) == 16 or 'BX' in font_name.upper() or 'Bold' in font_name
    
    # Left-aligned text is more likely to be a header (x_position < 100 typically)
    is_left_aligned = x_position < 100
    
    # Determine header level
    header_level = 0
    if is_section_like:
        if size_ratio > 1.5 or (size_ratio > 1.2 and is_bold):
            header_level = 1  # Main section
        elif size_ratio > 1.1 or is_bold:
            header_level = 2  # Subsection
        elif size_ratio >= 1.0 or (is_bold and is_left_aligned):
            header_level = 3  # Subsubsection
    
    # Check if it looks like a header
    is_header = (
        (size_ratio > 1.1 and is_bold) or
        (is_section_like and (size_ratio > 1.05 or is_bold)) or
        (is_bold and is_left_aligned and len(text_stripped) < 100)
    )
    
    return is_header, header_level


def extract_text_with_headers(doc: fitz.Document) -> List[Dict[str, Any]]:
    """
    Extract text from PDF with header detection and structure.
    
    Args:
        doc: PyMuPDF document object
        
    Returns:
        List of text segments with metadata:
        - text: str
        - page: int (0-indexed)
        - bbox: tuple (x0, y0, x1, y1)
        - is_header: bool
        - header_level: int (0 = not header, 1-3 = header levels)
        - font_size: float
        - font_name: str
    """
    segments = []
    common_font_size = calculate_common_font_size(doc)
    
    logger.info(f"Calculated common font size: {common_font_size:.2f}")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                block_text_parts = []
                block_is_header = False
                block_header_level = 0
                block_font_size = common_font_size
                block_font_name = ""
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                
                for line in block.get("lines", []):
                    line_text_parts = []
                    
                    for span in line.get("spans", []):
                        span_text = span.get("text", "").strip()
                        if not span_text:
                            continue
                        
                        font_size = span.get("size", common_font_size)
                        font_name = span.get("font", "")
                        flags = span.get("flags", 0)
                        x_position = span.get("bbox", [0, 0, 0, 0])[0]
                        
                        is_header, header_level = is_likely_header(
                            span_text, font_size, font_name, flags, 
                            common_font_size, x_position
                        )
                        
                        line_text_parts.append(span_text)
                        
                        # Track header status for the block
                        if is_header:
                            block_is_header = is_header
                            block_header_level = max(block_header_level, header_level)
                            block_font_size = font_size
                            block_font_name = font_name
                    
                    if line_text_parts:
                        line_text = " ".join(line_text_parts)
                        block_text_parts.append(line_text)
                
                if block_text_parts:
                    block_text = "\n".join(block_text_parts).strip()
                    if block_text:
                        segments.append({
                            "text": block_text,
                            "page": page_num,
                            "bbox": block_bbox,
                            "is_header": block_is_header,
                            "header_level": block_header_level,
                            "font_size": block_font_size,
                            "font_name": block_font_name
                        })
    
    return segments

