# Student Initiatives: PDF Indexer MCP Server Improvements

**For The Multiverse School Students**  
**Updated**: November 3, 2025

Welcome! This document breaks down the PDF Indexer MCP Server into bite-sized initiatives perfect for students working with AI agents, Cursor, and modern development practices. Whether you're new to coding or a vibe coder, there's something here for you.

## 🎯 How to Use This Document

1. **Pick an initiative** that interests you or matches your skill level
2. **Ask Cursor or your AI agent** to help you understand the codebase
3. **Start small** - even tiny improvements are valuable!
4. **Ask questions** - the community is here to help
5. **Share your work** - submit PRs and celebrate wins!

---

## 🚀 Level 1: First Steps (Great for Beginners!)

These initiatives are perfect if you're new to the project or want to build confidence.

### Initiative 1: Add Your First Test
**Goal**: Write a simple test for one of the MCP tools  
**Why it matters**: Tests help us know if code works, even after changes  
**Skills you'll learn**: Python testing basics, understanding tool functions  
**Estimated time**: 2-4 hours

**What to do**:
- Pick a simple tool (like `list_indexed_papers()`)
- Write a test that checks if it returns the right structure
- Use pytest (it's friendly!)
- Ask Cursor: "Help me write a test for the list_indexed_papers function"

**Success looks like**: Running `pytest` and seeing your test pass ✅

---

### Initiative 2: Improve Error Messages
**Goal**: Make error messages more helpful and user-friendly  
**Why it matters**: Better errors = less frustration for everyone  
**Skills you'll learn**: Error handling, user experience thinking  
**Estimated time**: 3-5 hours

**What to do**:
- Find places where errors happen (look for `logger.error` or exceptions)
- Make error messages clearer and more helpful
- Add suggestions for fixing common problems
- Example: Instead of "File not found", say "The PDF file 'paper.pdf' wasn't found. Make sure it's in the papers/ directory."

**Success looks like**: Error messages that actually help users fix problems

---

### Initiative 3: Add Progress Indicators
**Goal**: Show progress for long operations (like indexing papers)  
**Why it matters**: Users want to know things are working!  
**Skills you'll learn**: Progress reporting, user feedback  
**Estimated time**: 4-6 hours

**What to do**:
- Find operations that take time (embedding generation, PDF chunking)
- Add progress callbacks or status messages
- Show "Processing chunk 5 of 20..." or similar
- Use Cursor to help you understand how the operations work

**Success looks like**: Users can see progress instead of wondering if it's frozen

---

### Initiative 4: Write Better Documentation
**Goal**: Improve code comments and docstrings  
**Why it matters**: Good docs help everyone understand the code  
**Skills you'll learn**: Technical writing, code comprehension  
**Estimated time**: 2-3 hours per file

**What to do**:
- Pick a file that needs better documentation
- Add clear comments explaining what functions do
- Improve docstrings (the triple-quoted strings in functions)
- Use plain language - explain it like you're talking to a friend

**Success looks like**: New contributors can understand the code easily

---

## 🎨 Level 2: Feature Additions (Intermediate)

Ready to add new capabilities? These initiatives let you build real features.

### Initiative 5: Add a Delete Paper Tool
**Goal**: Create a new MCP tool to delete papers from the database  
**Why it matters**: Users need to manage their paper collection  
**Skills you'll learn**: Database operations, creating new MCP tools  
**Estimated time**: 4-6 hours

**What to do**:
- Study how `index_pdf` works (it's a good template!)
- Create `delete_paper()` tool
- Make sure it removes the paper, chunks, and embeddings
- Handle edge cases (what if paper doesn't exist?)
- Ask Cursor: "Show me how to delete a row from the database using SQLAlchemy"

**Success looks like**: Users can clean up their paper collection

---

### Initiative 6: Add Batch Operations
**Goal**: Let users index multiple papers at once  
**Why it matters**: Efficiency! No one wants to index 10 papers one by one  
**Skills you'll learn**: Batch processing, loops, progress tracking  
**Estimated time**: 6-8 hours

**What to do**:
- Create `batch_index_pdfs()` tool that takes a list of URLs
- Process them in a loop (or in parallel if you're feeling fancy!)
- Show progress: "Indexing paper 3 of 10..."
- Handle errors gracefully (don't stop if one fails)
- Use Cursor to help with the batch processing logic

**Success looks like**: Users can add multiple papers with one command

---

### Initiative 7: Add Metadata Extraction
**Goal**: Extract paper title, authors, and publication date from PDFs  
**Why it matters**: Makes papers more searchable and organized  
**Skills you'll learn**: PDF parsing, data extraction, pattern matching  
**Estimated time**: 6-10 hours

**What to do**:
- Research libraries that extract PDF metadata (PyPDF2, pdfplumber)
- Extract title, authors, publication date, DOI if available
- Store in database (may need to update the schema!)
- Show metadata when listing papers
- Ask Cursor: "How do I extract metadata from a PDF file in Python?"

**Success looks like**: Papers have rich metadata instead of just filenames

---

### Initiative 8: Add Duplicate Detection
**Goal**: Detect if a PDF is already in the database  
**Why it matters**: Prevents duplicate papers and wasted storage  
**Skills you'll learn**: File hashing, database queries, comparison logic  
**Estimated time**: 4-6 hours

**What to do**:
- Calculate a hash of PDF content (like a fingerprint)
- Check if hash exists in database before indexing
- Show helpful message: "This paper is already indexed!"
- Optionally: show when it was added and by which method
- Use Cursor to help you understand hashing in Python

**Success looks like**: System prevents duplicate papers automatically

---

## 🔧 Level 3: Performance & Polish (Intermediate-Advanced)

Ready to make things faster and better? These initiatives improve the system.

### Initiative 9: Add Caching for Embeddings
**Goal**: Cache embeddings so we don't regenerate them unnecessarily  
**Why it matters**: Faster searches, less computation  
**Skills you'll learn**: Caching strategies, performance optimization  
**Estimated time**: 6-8 hours

**What to do**:
- Understand how embeddings are generated (check `generate_embeddings` function)
- Add caching layer (maybe use a simple file-based cache or Redis if you're adventurous)
- Check cache before generating new embeddings
- Invalidate cache when paper is updated
- Ask Cursor: "How do I implement caching in Python?"

**Success looks like**: Generating embeddings for the same paper is instant

---

### Initiative 10: Improve Search with Hybrid Search
**Goal**: Combine semantic search with keyword search for better results  
**Why it matters**: Best of both worlds - meaning AND keywords  
**Skills you'll learn**: Search algorithms, ranking, combining results  
**Estimated time**: 8-12 hours

**What to do**:
- Research hybrid search (semantic + keyword/BM25)
- Implement keyword search component
- Combine results from both methods (Reciprocal Rank Fusion is cool!)
- Make results better than either method alone
- Use Cursor to help you understand search algorithms

**Success looks like**: Search finds relevant papers even when exact words don't match

---

### Initiative 11: Add Database Indexing
**Goal**: Make database queries faster with proper indexes  
**Why it matters**: Faster searches, better performance with lots of papers  
**Skills you'll learn**: Database optimization, SQL indexes  
**Estimated time**: 4-6 hours

**What to do**:
- Identify frequently queried columns (filename, paper_id, etc.)
- Add database indexes for those columns
- Test query performance before and after
- Document which indexes were added and why
- Ask Cursor: "How do I add indexes to a SQLite database using SQLAlchemy?"

**Success looks like**: Queries are noticeably faster with 100+ papers

---

### Initiative 12: Add Progress Reporting
**Goal**: Show real-time progress for long operations  
**Why it matters**: Better user experience, no more wondering if it's working  
**Skills you'll learn**: Progress tracking, callbacks, user feedback  
**Estimated time**: 6-8 hours

**What to do**:
- Add progress callbacks to long operations
- Show percentage complete or "X of Y" progress
- Estimate time remaining (optional but cool!)
- Update progress as work happens
- Use Cursor to help you implement progress tracking

**Success looks like**: Users see progress bars or status updates during long operations

---

## 🎓 Level 4: Advanced Features (For Experienced Students)

Ready to tackle bigger challenges? These initiatives add sophisticated capabilities.

### Initiative 13: Add Multiple Embedding Models
**Goal**: Let users choose different embedding models  
**Why it matters**: Different models work better for different use cases  
**Skills you'll learn**: Model management, configuration, abstraction  
**Estimated time**: 10-14 hours

**What to do**:
- Research different embedding models (Qwen, BGE, etc.)
- Create abstraction layer for model switching
- Add configuration option to choose model
- Handle model loading and switching
- Test that different models work correctly
- Ask Cursor: "How do I load different MLX embedding models dynamically?"

**Success looks like**: Users can choose the best model for their needs

---

### Initiative 14: Add GraphRAG Support
**Goal**: Extract entities and relationships to build knowledge graphs  
**Why it matters**: Enables relationship-based search and discovery  
**Skills you'll learn**: Entity extraction, graph databases, relationships  
**Estimated time**: 12-16 hours

**What to do**:
- Research GraphRAG (Microsoft's approach)
- Extract entities (people, concepts, methods) from papers
- Extract relationships between entities
- Build a graph structure
- Enable graph-based search queries
- Use Cursor to help you understand entity extraction

**Success looks like**: Users can find papers by relationships, not just keywords

---

### Initiative 15: Add Multimodal RAG
**Goal**: Extract and search images, charts, and figures from PDFs  
**Why it matters**: Papers have important visual content too!  
**Skills you'll learn**: Image processing, multimodal embeddings, OCR  
**Estimated time**: 14-18 hours

**What to do**:
- Extract images from PDFs
- Generate embeddings for images (use vision models)
- Store image embeddings in database
- Enable image search: "Find papers with architecture diagrams"
- Handle both text and image search
- Ask Cursor: "How do I extract images from PDFs and generate embeddings?"

**Success looks like**: Users can search for visual content in papers

---

### Initiative 16: Add Citation Network
**Goal**: Build a network of paper citations  
**Why it matters**: Find related papers through citation relationships  
**Skills you'll learn**: Citation parsing, graph networks, network analysis  
**Estimated time**: 10-14 hours

**What to do**:
- Extract citations from papers (bibliography sections)
- Parse citation formats (APA, MLA, etc.)
- Build citation graph (which papers cite which)
- Add "Related Papers" feature based on citations
- Visualize citation network (optional but cool!)
- Use Cursor to help you parse citation formats

**Success looks like**: Users can discover papers through citation relationships

---

## 🛠️ Level 5: Infrastructure & DevOps (Advanced)

These initiatives improve how the system runs and is deployed.

### Initiative 17: Add Docker Support
**Goal**: Create Docker container for easy deployment  
**Why it matters**: Makes it easy to run anywhere, consistent environment  
**Skills you'll learn**: Docker, containerization, deployment  
**Estimated time**: 6-8 hours

**What to do**:
- Create Dockerfile for the MCP server
- Handle dependencies and Python environment
- Create docker-compose.yml for easy setup
- Test that it runs in container
- Document how to use it
- Ask Cursor: "How do I create a Dockerfile for a Python MCP server?"

**Success looks like**: Users can run the server with `docker-compose up`

---

### Initiative 18: Add HTTP Transport
**Goal**: Support HTTP transport in addition to stdio  
**Why it matters**: More deployment options, web integration  
**Skills you'll learn**: HTTP servers, REST APIs, transport protocols  
**Estimated time**: 10-14 hours

**What to do**:
- Research FastMCP HTTP transport support
- Add HTTP server capability
- Create REST API endpoints for tools
- Handle authentication (API keys?)
- Test HTTP transport works
- Document HTTP API
- Ask Cursor: "How do I add HTTP transport to a FastMCP server?"

**Success looks like**: Server can run as HTTP service

---

### Initiative 19: Add Monitoring & Metrics
**Goal**: Track server usage and performance  
**Why it matters**: Understand how it's being used, find bottlenecks  
**Skills you'll learn**: Metrics, monitoring, observability  
**Estimated time**: 8-10 hours

**What to do**:
- Add metrics collection (operation counts, durations, errors)
- Export metrics in Prometheus format (optional)
- Create dashboard or status endpoint
- Track usage patterns
- Log important events
- Ask Cursor: "How do I add metrics tracking to a Python application?"

**Success looks like**: We can see how the server is being used

---

### Initiative 20: Add CI/CD Pipeline
**Goal**: Automate testing and deployment  
**Why it matters**: Catch bugs early, deploy confidently  
**Skills you'll learn**: CI/CD, GitHub Actions, automation  
**Estimated time**: 6-8 hours

**What to do**:
- Set up GitHub Actions workflow
- Run tests automatically on PRs
- Check code quality (linting, formatting)
- Auto-deploy to PyPI on releases (optional)
- Add badges to README
- Ask Cursor: "How do I set up GitHub Actions for a Python project?"

**Success looks like**: Tests run automatically, code quality is maintained

---

## 🎯 Bonus Initiatives (Creative & Fun!)

These are open-ended and let you be creative!

### Initiative 21: Add a Web UI
**Goal**: Create a web interface for browsing and searching papers  
**Why it matters**: Not everyone loves command line!  
**Skills you'll learn**: Web development, frontend, APIs  
**Estimated time**: 15-20 hours

**What to do**:
- Build a simple web interface (React, Vue, or plain HTML+JS)
- Connect to MCP server via HTTP (if you did Initiative 18!)
- Show paper browser, search interface, results
- Make it look nice!
- Ask Cursor: "Help me build a web interface for an MCP server"

**Success looks like**: Users can browse papers in a web browser

---

### Initiative 22: Add Export Features
**Goal**: Export search results to various formats  
**Why it matters**: Users want to share and save results  
**Skills you'll learn**: File formats, data export, user experience  
**Estimated time**: 6-8 hours

**What to do**:
- Add export to Markdown
- Add export to CSV
- Add export to BibTeX (for citations)
- Add export to JSON
- Make it easy to use
- Ask Cursor: "How do I export Python data to different file formats?"

**Success looks like**: Users can export results in their preferred format

---

### Initiative 23: Add Paper Recommendations
**Goal**: Suggest related papers based on what user has indexed  
**Why it matters**: Help users discover new relevant papers  
**Skills you'll learn**: Recommendation algorithms, similarity, ML  
**Estimated time**: 10-12 hours

**What to do**:
- Analyze what papers user has
- Find similar papers (using embeddings)
- Suggest papers based on topics
- Show why each paper was recommended
- Ask Cursor: "How do I implement a recommendation system?"

**Success looks like**: System suggests relevant papers automatically

---

### Initiative 24: Add Visualization Dashboard
**Goal**: Visualize paper collection with charts and graphs  
**Why it matters**: Visual understanding of your research collection  
**Skills you'll learn**: Data visualization, charts, dashboards  
**Estimated time**: 8-12 hours

**What to do**:
- Show paper count over time
- Visualize topics (word clouds?)
- Show citation network graph
- Create interactive charts
- Use libraries like Plotly or D3.js
- Ask Cursor: "How do I create data visualizations in Python/web?"

**Success looks like**: Beautiful visualizations of the paper collection

---

## 📝 How to Get Started

1. **Read the README** - Understand what the project does
2. **Explore the codebase** - Use Cursor to navigate and understand
3. **Pick an initiative** - Start with Level 1 if you're new!
4. **Ask questions** - Use Cursor, ask in Discord, check docs
5. **Make changes** - Start small, test often
6. **Submit a PR** - Share your work!

## 🎓 Learning Resources

- **FastMCP Docs**: https://github.com/jlowin/fastmcp
- **MCP Protocol**: https://modelcontextprotocol.io
- **RAG Concepts**: Check the README.md in this repo!
- **Python Testing**: pytest documentation
- **SQLAlchemy**: Official docs for database work

## 💡 Tips for Success

- **Start small**: Even tiny improvements are valuable
- **Use AI agents**: Cursor is your friend! Ask it questions
- **Test your changes**: Make sure things still work
- **Read existing code**: Learn from what's already there
- **Ask for help**: No question is too basic
- **Celebrate wins**: Every contribution matters!

## 🤝 Contributing

When you're ready to submit your work:

1. Fork the repository
2. Create a branch: `git checkout -b your-initiative-name`
3. Make your changes
4. Test thoroughly
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request
8. Describe what you did and why

## 📊 Initiative Tracking

Want to see what's being worked on? Check GitHub Issues labeled with initiative numbers!

---

**Remember**: This is a learning project. Every contribution teaches something new, and every improvement makes the tool better for everyone. Have fun, ask questions, and build something awesome! 🚀

**Last Updated**: November 3, 2025  
**Maintainer**: Liz Howard (@lizTheDeveloper)  
**School**: The Multiverse School

