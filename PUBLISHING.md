# Publishing Guide

## Current Status

✅ **Package built successfully**
- Location: `dist/` directory
- Files:
  - `pdf_indexer_mcp-1.0.0-py3-none-any.whl` (51KB)
  - `pdf_indexer_mcp-1.0.0.tar.gz` (55KB)
- ✅ Validation passed with `twine check`
- ✅ GitHub repository: https://github.com/lizTheDeveloper/pdf-indexer-mcp
- ✅ All changes committed and pushed

---

## 1. Publishing to PyPI

### Step 1: Create PyPI Account (if needed)
1. Visit: https://pypi.org/account/register/
2. Use email: `lizthedeveloper@gmail.com`
3. Complete registration

### Step 2: Get API Token
1. Visit: https://pypi.org/manage/account/token/
2. Create a new API token
3. Give it a name like "pdf-indexer-mcp"
4. Scope: Select "Entire account" or just "pdf-indexer-mcp" project
5. Copy the token (starts with `pypi-...`)

### Step 3: Upload to PyPI
```bash
cd /Users/annhoward/openai_agents_10_23_25/pdf_indexer_mcp
source build_env/bin/activate
python3 -m twine upload dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: `pypi-...` (your API token)

Or set environment variables:
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-your-token-here
python3 -m twine upload dist/*
```

### After Publishing
- Package will be available at: https://pypi.org/project/pdf-indexer-mcp/
- Users can install with: `pip install pdf-indexer-mcp`

---

## 2. Publishing to Smithery

### Option A: Via Web Interface (Recommended)
1. Visit: https://smithery.ai/new
2. Sign up/Sign in with:
   - Email: `lizthedeveloper@gmail.com` or `liz@themultiverse.school`
   - Or connect with GitHub (already logged in as @lizTheDeveloper)
3. Choose "Publish via URL" or "Continue with GitHub"
4. Enter repository URL: `https://github.com/lizTheDeveloper/pdf-indexer-mcp`
5. Follow their publishing workflow:
   - Add metadata (name, description, tags)
   - Configure MCP server settings
   - Set up deployment configuration
   - Publish

### Option B: Via Smithery CLI (if available)
```bash
# Install Smithery CLI
npm install -g @smithery/cli

# Authenticate
npx @smithery/cli auth

# Publish (check their docs for exact command)
npx @smithery/cli publish --repo https://github.com/lizTheDeveloper/pdf-indexer-mcp
```

### What to Include in Smithery
- **Server Name**: PDF Indexer MCP
- **Description**: A Model Context Protocol (MCP) server for indexing and semantically searching PDF research papers
- **Repository**: https://github.com/lizTheDeveloper/pdf-indexer-mcp
- **Transport**: stdio (currently)
- **Installation**: `pip install pdf-indexer-mcp`
- **Configuration**: See README.md for MCP setup instructions

---

## 3. Verification

### After PyPI Publication
```bash
pip search pdf-indexer-mcp  # Search for package
pip install pdf-indexer-mcp  # Test installation
```

### After Smithery Publication
- Visit your server page on Smithery
- Verify it appears in the registry
- Test installation via Smithery CLI or web interface

---

## Notes

- **PyPI**: First publication creates the project. Subsequent versions use the same command.
- **Smithery**: May require approval or review process depending on their policies.
- **Version Updates**: When updating, change version in `pyproject.toml`, rebuild, and re-upload.

---

## Troubleshooting

### PyPI Upload Fails
- Check token has correct scope
- Ensure package name isn't taken
- Verify all required fields in pyproject.toml

### Smithery Publishing Issues
- Check repository is public
- Verify MCP server structure is correct
- Contact Smithery support if needed

---

**Last Updated**: 2025-01-31

