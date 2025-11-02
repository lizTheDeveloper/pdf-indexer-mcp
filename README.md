# PDF Research Paper Indexing MCP Server

A standalone Model Context Protocol (MCP) server for indexing and searching research papers in PDF format. This server enables AI agents to download, chunk, index, and semantically search PDF research papers using MLX-optimized embeddings.

## Features

- **PDF Download**: Download research papers from URLs
- **Multiple Chunking Methods**: 
  - Header-based chunking (for structured academic papers)
  - S2 chunking (spatial-semantic hybrid approach)
- **Database Indexing**: Store papers and chunks in SQLite database
- **Semantic Search**: Search papers using MLX-optimized embeddings (Qwen3-Embedding-0.6B)
- **FAISS Vector Index**: Fast similarity search using FAISS

## Installation

1. **Clone or copy this directory**

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### MCP Server Setup

#### For Cursor IDE

Edit your MCP configuration file:
- **macOS**: `~/Library/Application Support/Cursor/User/globalStorage/mcp.json`
- **Windows**: `%APPDATA%\Cursor\User\globalStorage\mcp.json`
- **Linux**: `~/.config/Cursor/User/globalStorage/mcp.json`

Add the following configuration:

```json
{
  "mcpServers": {
    "pdf-indexer": {
      "command": "/path/to/pdf_indexer_mcp/venv/bin/python3",
      "args": [
        "/path/to/pdf_indexer_mcp/semantic_chunked_pdf_rag.py"
      ],
      "env": {}
    }
  }
}
```

**Note**: Replace `/path/to/pdf_indexer_mcp` with the absolute path to this directory.

#### For Claude Desktop

Edit your MCP configuration file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

Add the same configuration as above.

## Directory Structure

```
pdf_indexer_mcp/
├── semantic_chunked_pdf_rag.py  # Main MCP server
├── utils/                       # Logging utilities
├── pdf_processing/             # PDF text extraction
├── chunking/                   # Chunking algorithms
├── database/                    # Database models and operations
├── embeddings/                 # MLX embedding generation and FAISS
├── papers/                     # Downloaded PDFs (created automatically)
├── indexes/                    # Database and FAISS indices (created automatically)
├── logs/                       # Log files (created automatically)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## Available MCP Tools

1. **`download_pdf(url: str)`** - Download a PDF from URL
2. **`chunk_pdf(filename: str, method: str = "header")`** - Extract and chunk a PDF
3. **`index_pdf(filename: str, url: str = "", method: str = "header")`** - Index a PDF in the database
4. **`list_indexed_papers()`** - List all indexed papers
5. **`get_document_structure(filename: str)`** - Get paper structure (sections)
6. **`get_document_section(filename: str, ...)`** - Get a specific section
7. **`generate_embeddings(filename: str, model_name: str = "mlx-community/Qwen3-Embedding-0.6B")`** - Generate embeddings for a paper
8. **`search_research_papers(query: str, k: int = 5, context_window: int = 1, model_name: str = "mlx-community/Qwen3-Embedding-0.6B")`** - Search indexed papers semantically

## Usage Workflow

1. **Download a paper**:
   ```python
   result = download_pdf("https://arxiv.org/pdf/1706.03762.pdf")
   ```

2. **Index the paper**:
   ```python
   result = index_pdf("1706.03762.pdf", method="header")
   ```

3. **Generate embeddings**:
   ```python
   result = generate_embeddings("1706.03762.pdf")
   ```

4. **Search papers**:
   ```python
   results = search_research_papers("attention mechanism", k=5)
   ```

## Requirements

- Python 3.8+
- macOS (for MLX support) or Linux/Windows (with CPU fallback)
- ~500MB RAM for embeddings
- ~1GB disk space for model downloads

## Model Information

- **Embedding Model**: `mlx-community/Qwen3-Embedding-0.6B` (1024 dimensions)
- **Optimized for**: Apple Silicon (MLX framework)
- **Embedding Generation**: ~35 embeddings/second on Apple Silicon

## Logging

Logs are written to `logs/pdf_indexer_YYYYMMDD.log` with detailed information including:
- Operations performed
- Performance metrics
- Errors with full context

## License

This is a standalone MCP server package extracted from a larger project. Use as needed.

## Troubleshooting

### MCP Server Not Starting

1. Check virtual environment is activated
2. Verify dependencies installed: `pip list | grep fastmcp`
3. Check logs in `logs/` directory
4. Test server manually: `python semantic_chunked_pdf_rag.py`

### Embedding Generation Fails

1. Verify MLX installed: `python -c "import mlx.core as mx; print('OK')"`
2. Check available RAM (needs ~500MB)
3. First run downloads model automatically

### Search Returns No Results

1. Verify papers are indexed using `list_indexed_papers()`
2. Generate embeddings using `generate_embeddings()`
3. Check database and FAISS index exist in `indexes/` directory

