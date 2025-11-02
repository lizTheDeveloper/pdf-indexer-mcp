"""
PDF content validation utilities.
"""


def validate_pdf_content(content: bytes) -> bool:
    """
    Validate that content is actually a PDF by checking the PDF header.
    
    PDF files start with "%PDF-" followed by a version number (e.g., "%PDF-1.4").
    We check for the first 4 bytes "%PDF" as a minimal validation.
    
    Args:
        content: The content bytes to validate
        
    Returns:
        True if content appears to be a valid PDF, False otherwise
    """
    if len(content) < 5:
        return False
    
    # PDF files start with %PDF- followed by version number (e.g., %PDF-1.4)
    # Check for the first 5 bytes to verify the proper PDF header format
    pdf_header = content[:5]
    if pdf_header == b'%PDF-':
        return True
    
    # Fallback: check first 4 bytes (some valid PDFs might have slightly different formatting)
    return content[:4] == b'%PDF'

