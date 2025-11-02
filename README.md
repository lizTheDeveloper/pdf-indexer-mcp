# PDF Indexer MCP Server

A **Model Context Protocol (MCP) server** that enables AI agents to download, index, and semantically search PDF research papers. This server provides 8 tools that AI agents can discover and use autonomously to build research paper knowledge bases and answer questions.

## What is MCP?

**Model Context Protocol (MCP)** is a standardized protocol that allows AI agents to discover and use tools. Instead of being limited to text generation, agents become **action-capable systems** that can:

1. **Discover Tools**: Agents automatically discover available tools from connected MCP servers
2. **Understand Capabilities**: Agents read tool descriptions and parameters to understand what each tool can do
3. **Execute Tasks**: Agents call tools with appropriate parameters to accomplish goals
4. **Compose Workflows**: Agents can combine multiple tools from different servers to solve complex problems

### How MCP Works

```
AI Agent → MCP Protocol → Tool Server → Execution → Results → Agent
```

When you connect this MCP server to an AI agent (like in Cursor, Claude Desktop, or via OpenAI Agents framework), the agent automatically:

- Discovers all 8 tools available from this server
- Understands what each tool does from their descriptions
- Uses the tools when they're needed to complete tasks
- Can combine tools in complex workflows

## Features

- 📥 **PDF Download**: Download research papers from URLs
- 📄 **Intelligent Chunking**: Two chunking strategies:
  - **Header-based**: Preserves document structure (ideal for academic papers)
  - **S2 chunking**: Spatial-semantic hybrid approach for optimal semantic chunks
- 🗄️ **Database Indexing**: Store papers and chunks in SQLite with navigation indices
- 🔍 **Semantic Search**: Search papers using MLX-optimized embeddings (Qwen3-Embedding-0.6B)
- ⚡ **FAISS Vector Index**: Fast similarity search even for thousands of chunks
- 🧠 **Context-Aware**: Retrieve surrounding chunks for better context

## Quick Start: Suggested Prompt

Once the MCP server is configured, you can use prompts like:

```
"I have this research paper URL: [URL]. Please download it, index it, 
make it searchable, and then search for information about [topic]."
```

Or more simply:
```
"Download and index this paper: [URL], then search it for information about [topic]."
```

The agent will automatically:
1. Download the PDF
2. Index it into the database
3. Generate embeddings for semantic search
4. Search for relevant content
5. Present the results

## Installation

### Option 1: Install from GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/lizTheDeveloper/pdf-indexer-mcp.git
cd pdf-indexer-mcp

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Install via pip (After Publishing)

```bash
pip install pdf-indexer-mcp
```

## MCP Server Setup

### For Cursor IDE

1. **Locate MCP configuration file**:
   - **macOS**: `~/Library/Application Support/Cursor/User/globalStorage/mcp.json`
   - **Windows**: `%APPDATA%\Cursor\User\globalStorage\mcp.json`
   - **Linux**: `~/.config/Cursor/User/globalStorage/mcp.json`

2. **Add configuration** (create file if it doesn't exist):
   ```json
   {
     "mcpServers": {
       "pdf-indexer": {
         "command": "/absolute/path/to/pdf_indexer_mcp/venv/bin/python3",
         "args": [
           "/absolute/path/to/pdf_indexer_mcp/semantic_chunked_pdf_rag.py"
         ],
         "env": {}
       }
     }
   }
   ```

3. **Restart Cursor** completely (not just reload)

4. **Verify**: After restart, you should see 8 tools available:
   - `mcp_pdf-indexer_download_pdf`
   - `mcp_pdf-indexer_chunk_pdf`
   - `mcp_pdf-indexer_index_pdf`
   - `mcp_pdf-indexer_list_indexed_papers`
   - `mcp_pdf-indexer_get_document_structure`
   - `mcp_pdf-indexer_get_document_section`
   - `mcp_pdf-indexer_generate_embeddings`
   - `mcp_pdf-indexer_search_research_papers`

### For Claude Desktop

1. **Locate MCP configuration file**:
   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
   - **Linux**: `~/.config/Claude/claude_desktop_config.json`

2. **Add the same configuration** as above

3. **Restart Claude Desktop** completely

### For OpenAI Agents Framework

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="PDF Indexer",
    params={
        "command": "/path/to/pdf_indexer_mcp/venv/bin/python3",
        "args": ["/path/to/pdf_indexer_mcp/semantic_chunked_pdf_rag.py"],
    },
) as pdf_indexer_server:
    agent = Agent(
        name="Research Assistant",
        instructions="Help users search and analyze research papers",
        mcp_servers=[pdf_indexer_server],
        model="gpt-4"
    )
    
    result = await Runner.run(
        agent,
        "Download and index this paper: https://arxiv.org/pdf/1706.03762.pdf"
    )
```

## Available MCP Tools

This server exposes 8 tools that AI agents can use. Tools are automatically discovered by agents when the server is connected.

### 1. `download_pdf(url: str)`

**What it does**: Downloads a PDF research paper from a URL and saves it locally.

**When agents use it**: When you ask to download a paper, the agent automatically discovers and uses this tool.

**Example agent workflow**:
```
User: "Download the attention paper from arxiv"
Agent: 
  1. Discovers download_pdf tool
  2. Calls: download_pdf("https://arxiv.org/pdf/1706.03762.pdf")
  3. Returns: Downloaded paper saved as "1706.03762.pdf"
```

**Returns**:
- `success`: bool
- `filename`: str (e.g., "1706.03762.pdf")
- `filepath`: str (absolute path)
- `message`: str

### 2. `chunk_pdf(filename: str, method: str = "header")`

**What it does**: Extracts text from a PDF and chunks it using header-based or S2 chunking.

**Parameters**:
- `filename`: PDF filename (must be in `papers/` directory)
- `method`: `"header"` (default) or `"s2"` for spatial-semantic chunking

**When agents use it**: When asked to analyze or process a PDF's structure.

**Returns**:
- `success`: bool
- `num_chunks`: int
- `chunks`: list of chunk dictionaries with preview text
- `method`: str (chunking method used)

### 3. `index_pdf(filename: str, url: str = "", method: str = "header")`

**What it does**: Complete indexing workflow - downloads (if needed), chunks, and stores in database.

**When agents use it**: The most common tool agents use - it handles the full pipeline.

**Example agent workflow**:
```
User: "Index this paper and make it searchable"
Agent:
  1. Discovers index_pdf tool
  2. Calls: index_pdf("paper.pdf", url="https://...", method="header")
  3. Paper is now in database and searchable
```

**Returns**:
- `success`: bool
- `paper_id`: int (database ID)
- `num_chunks`: int
- `num_sections`: int (for header method)
- `message`: str

### 4. `list_indexed_papers()`

**What it does**: Lists all papers currently indexed in the database.

**When agents use it**: When asked "what papers do you have?" or "show me all papers".

**Returns**:
- `success`: bool
- `count`: int
- `papers`: list of paper metadata

### 5. `get_document_structure(filename: str)`

**What it does**: Gets the complete structure of a paper (sections, headers, chunk ranges).

**When agents use it**: When you ask about a paper's structure or sections.

**Returns**:
- `success`: bool
- `structure`: dict with paper metadata and sections list

### 6. `get_document_section(filename: str, ...)`

**What it does**: Retrieves a specific section of a document.

**Parameters** (use one of):
- `chunk_index`: int - Get specific chunk by index
- `header_path`: str - Get section by header path (e.g., "Introduction")
- `page_start` / `page_end`: int - Get chunks in page range

**When agents use it**: When asked "show me the Introduction section" or "get page 5-10".

**Returns**:
- `success`: bool
- `paper_id`: int
- `num_chunks`: int
- `chunks`: list of full chunk content

### 7. `generate_embeddings(filename: str, model_name: str = "mlx-community/Qwen3-Embedding-0.6B")`

**What it does**: Generates semantic embeddings for all chunks in a paper and adds them to the FAISS vector index.

**When agents use it**: Agents automatically use this before semantic search.

**Example agent workflow**:
```
User: "Make this paper searchable"
Agent:
  1. Calls index_pdf() - indexes the paper
  2. Calls generate_embeddings() - makes it searchable
  3. Paper is now ready for semantic search
```

**Returns**:
- `success`: bool
- `paper_id`: int
- `num_embeddings`: int
- `embedding_dim`: int (1024 for Qwen3-Embedding-0.6B)
- `model_name`: str

### 8. `search_research_papers(query: str, k: int = 5, context_window: int = 1, model_name: str = "mlx-community/Qwen3-Embedding-0.6B")`

**What it does**: Semantically searches all indexed papers using embeddings and returns the most relevant chunks.

**Parameters**:
- `query`: Search query text
- `k`: Number of top results (default: 5)
- `context_window`: Number of neighboring chunks to include (default: 1)
- `model_name`: Embedding model (default: Qwen3-Embedding-0.6B)

**When agents use it**: When asked questions like "find papers about transformers" or "search for attention mechanisms".

**Example agent workflow**:
```
User: "What papers discuss attention mechanisms?"
Agent:
  1. Discovers search_research_papers tool
  2. Calls: search_research_papers("attention mechanisms", k=5)
  3. Gets relevant chunks with context
  4. Synthesizes answer from results
```

**Returns**:
- `success`: bool
- `query`: str (original query)
- `num_results`: int
- `results`: list of result dictionaries with:
  - `chunk_id`, `paper_id`, `filename`, `title`
  - `text`: Full chunk text
  - `header_path`: Section location
  - `page_start`, `page_end`: Page numbers
  - `distance`: Similarity score (lower = more similar)
  - `is_context`: bool (true if context chunk, not direct match)

## How Agents Use These Tools

### Autonomous Tool Discovery

When you connect this MCP server, agents automatically discover all 8 tools. Each tool has:
- **Name**: What the tool is called
- **Description**: What the tool does (agents read this!)
- **Parameters**: What inputs the tool needs
- **Return Type**: What the tool returns

Agents use these descriptions to understand when to use each tool.

### Typical Agent Workflow

```
User: "Find papers about transformers and summarize the key findings"

Agent workflow:
1. Discovers list_indexed_papers() → checks what's available
2. Discovers search_research_papers() → searches for "transformers"
3. Discovers get_document_section() → gets more context for top results
4. Synthesizes findings into summary

All tool calls happen autonomously!
```

### Multi-Tool Composition

Agents can combine tools in sophisticated ways:

```python
# Example: Agent decides to do a complete research workflow
1. download_pdf("https://arxiv.org/pdf/...") 
   → Downloads paper
2. index_pdf("paper.pdf", url="...", method="header")
   → Indexes paper in database
3. generate_embeddings("paper.pdf")
   → Makes it searchable
4. search_research_papers("related topic", k=5)
   → Finds related papers
5. get_document_section(filename, header_path="Introduction")
   → Gets specific sections for context
```

## Complete Usage Example

### Via Cursor/Claude Desktop

Once the MCP server is configured and restarted, you can simply ask:

```
You: "Download and index this paper about transformers"
Agent: [Automatically uses download_pdf and index_pdf tools]

You: "Search for papers about attention mechanisms"
Agent: [Automatically uses search_research_papers tool]

You: "Show me the Introduction section of the transformer paper"
Agent: [Automatically uses get_document_section tool]
```

### Via OpenAI Agents Framework

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async with MCPServerStdio(
    name="PDF Indexer",
    params={
        "command": "/path/to/venv/bin/python3",
        "args": ["/path/to/semantic_chunked_pdf_rag.py"],
    },
) as pdf_server:
    agent = Agent(
        name="Research Assistant",
        instructions="""
        You help users manage and search research papers.
        You can download, index, and search papers using the available tools.
        """,
        mcp_servers=[pdf_server],
        model="gpt-4"
    )
    
    # Agent autonomously uses tools
    result = await Runner.run(
        agent,
        "Download this paper, index it, make it searchable, and then search for related work on attention"
    )
    print(result.final_output)
```

## Package Structure

```
pdf_indexer_mcp/
├── semantic_chunked_pdf_rag.py  # Main MCP server (exposes tools)
├── utils/                       # Logging utilities
├── pdf_processing/             # PDF text extraction
├── chunking/                   # Chunking algorithms
├── database/                    # Database models and operations
├── embeddings/                 # MLX embedding generation and FAISS
├── papers/                     # Downloaded PDFs (created automatically)
├── indexes/                    # Database and FAISS indices (created automatically)
├── logs/                       # Log files (created automatically)
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Package metadata for pip
├── LICENSE                     # GPL-3.0 copyleft license
└── README.md                   # This file
```

## Learning RAG (Retrieval-Augmented Generation)

This MCP server demonstrates a complete **RAG (Retrieval-Augmented Generation)** pipeline for research papers. Understanding RAG is essential for building effective AI systems that can access and use external knowledge.

### What is RAG?

**RAG** combines information retrieval with language generation, allowing LLMs to:
1. **Retrieve** relevant information from external sources (here: research papers)
2. **Augment** the LLM's context with retrieved information
3. **Generate** responses grounded in retrieved content

Instead of relying solely on pre-trained knowledge, RAG enables systems to answer questions using up-to-date, domain-specific information.

### How This Server Implements RAG

This MCP server provides a complete RAG implementation:

#### 1. **Document Ingestion** (Retrieval Setup)
- **`download_pdf()`**: Fetch papers from URLs
- **`index_pdf()`**: Extract and chunk text, store in database
- Creates a searchable knowledge base

#### 2. **Semantic Indexing** (Vector Search)
- **`generate_embeddings()`**: Convert text chunks into semantic vectors
- Uses MLX-optimized embeddings (Qwen3-Embedding-0.6B, 1024 dimensions)
- Stores vectors in FAISS for fast similarity search

#### 3. **Retrieval** (Finding Relevant Content)
- **`search_research_papers()`**: Semantic search across all papers
- Finds relevant chunks based on meaning, not just keywords
- Returns ranked results with context

#### 4. **Augmentation** (Context Enhancement)
- **`get_document_section()`**: Retrieve full context from specific sections
- Includes surrounding chunks for better understanding
- Provides metadata (section, page, headers)

#### 5. **Generation** (LLM Response)
- Agent receives retrieved chunks
- Uses them as context to generate grounded responses
- Responses are based on actual paper content, not just training data

### RAG Pipeline Flow

```
User Query
    ↓
Semantic Search (search_research_papers)
    ↓
Find Relevant Chunks (FAISS vector search)
    ↓
Retrieve Context (get_document_section if needed)
    ↓
Augment LLM Context (pass chunks to LLM)
    ↓
Generate Response (grounded in retrieved content)
```

### Key RAG Concepts Demonstrated

1. **Chunking Strategy**: Two approaches shown:
   - **Header-based**: Preserves structure, ideal for academic papers
   - **S2 chunking**: Spatial-semantic hybrid for unstructured documents

2. **Semantic Search**: Uses embeddings to find meaning, not just keywords
   - "attention mechanisms" finds related concepts even without exact words
   - Better than traditional keyword search

3. **Vector Database**: FAISS for fast similarity search
   - Scales to thousands of chunks
   - Sub-millisecond search times

4. **Incremental Indexing**: Add papers without rebuilding entire index
   - Each paper can be indexed independently
   - Embeddings added incrementally

5. **Context Windows**: Retrieve surrounding chunks for better context
   - Helps maintain narrative flow
   - Provides background for understanding

### Why RAG Matters

**Without RAG**: LLMs can only use pre-trained knowledge, which may be:
- Outdated (training data cutoff)
- Generic (not domain-specific)
- Limited (no access to private/publications)

**With RAG**: LLMs can:
- Access current information (newly published papers)
- Use domain-specific knowledge (research papers)
- Ground responses in verifiable sources
- Answer questions about documents not in training data

### RAG Best Practices (This Implementation)

1. **Effective Chunking**: Balance chunk size - too small loses context, too large dilutes relevance
2. **Semantic Embeddings**: Use models optimized for your domain (here: research papers)
3. **Vector Search**: Fast retrieval is essential (FAISS provides sub-millisecond search)
4. **Metadata Preservation**: Keep headers, pages, sections for navigation
5. **Context Retrieval**: Include surrounding chunks for better understanding

### Further Learning

To understand RAG better:
- Experiment with different chunking methods (header vs S2)
- Try different embedding models
- Adjust context_window in search_research_papers()
- Explore the database structure to see how chunks are stored
- Check logs to see performance metrics

This implementation provides a production-ready RAG system you can study and extend.

## Requirements

- **Python**: 3.9+ (required by dependencies like numpy 2.2.6)
- **Platform**: macOS (for MLX optimization), Linux/Windows (with CPU fallback)
- **RAM**: ~500MB for embeddings
- **Disk**: ~1GB for model downloads (first run)

## Technical Details

### Embedding Model

- **Model**: `mlx-community/Qwen3-Embedding-0.6B`
- **Dimensions**: 1024
- **Framework**: MLX (optimized for Apple Silicon)
- **Speed**: ~35 embeddings/second on Apple Silicon

### Chunking Methods

**Header-based** (`method="header"`):
- Best for academic papers with clear structure
- Preserves document hierarchy
- Groups content under headers

**S2 Chunking** (`method="s2"`):
- Hybrid spatial-semantic approach
- Combines layout analysis with semantic similarity
- Optimal for unstructured documents

### Storage

- **Database**: SQLite (`indexes/research_papers.db`)
- **Vector Index**: FAISS (`indexes/research_papers.faiss`)
- **Mapping**: NumPy array (`indexes/research_papers_mapping.npy`)

## Troubleshooting

### MCP Server Not Starting

1. **Verify virtual environment**:
   ```bash
   which python3  # Should show path in venv/bin/python3
   ```

2. **Check dependencies**:
   ```bash
   pip list | grep fastmcp
   ```

3. **Test server manually**:
   ```bash
   python semantic_chunked_pdf_rag.py
   ```
   If it starts without errors, press Ctrl+C to stop.

4. **Check logs**:
   ```bash
   tail -f logs/pdf_indexer_*.log
   ```

### Tools Not Appearing in Agent

1. **Restart completely** (not just reload)
2. **Check configuration path** is absolute (not relative)
3. **Verify Python path** points to virtual environment
4. **Check MCP logs** for connection errors

### Embedding Generation Fails

1. **Verify MLX installed**:
   ```bash
   python -c "import mlx.core as mx; print('OK')"
   ```

2. **Check available RAM** (needs ~500MB)

3. **First run** downloads model automatically (may take time)

### Search Returns No Results

1. **Verify papers indexed**:
   ```bash
   # Agents should discover list_indexed_papers() tool
   ```

2. **Generate embeddings**:
   ```bash
   # Agents should discover generate_embeddings() tool
   ```

3. **Check FAISS index exists** in `indexes/` directory

## Contributing

Contributions welcome! This project uses GPL-3.0 copyleft licensing.

## License

GNU General Public License v3.0 (GPL-3.0) - Copyleft License

Copyright (C) 2025 Liz Howard (@lizTheDeveloper)

See [LICENSE](LICENSE) file for full license text.

## Links

- **Repository**: https://github.com/lizTheDeveloper/pdf-indexer-mcp
- **Issues**: https://github.com/lizTheDeveloper/pdf-indexer-mcp/issues
- **Author**: Liz Howard (@lizTheDeveloper)
