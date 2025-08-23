# Changelog

All notable changes to Claude Semantic Session Manager will be documented in this file.

## [3.8.0] - 2025-08-23

### Added
- **Complete Conversation Saving**: Removed 10KB limitation, now saves 100% of conversation history
- **3-Level Summary System**: Implemented Quick (500 tokens), Normal (2000 tokens), and Deep (10000 tokens) restore modes
- **Token Optimization**: Achieves 96% token reduction compared to full context loading
- **Enhanced Context Extraction**: New `extract_conversation_context()` method with level-based summarization
- **Improved Memory Efficiency**: Process line-by-line reading for large JSONL files

### Changed
- Updated `ConversationMonitor._monitor_loop()` to read from file beginning instead of last 10KB
- Enhanced `restore_state()` to display context summaries based on restore level
- Version bumped from 3.6.0 to 3.8.0

### Fixed
- Missing `defaultdict` import that caused restore failures
- Incomplete conversation history preservation issue

## [1.1.0] - 2025-08-23

### Added
- **Integrated Session Manager** (`session_manager_integrated.py`)
  - Interactive session selection with number, keyword, or "latest"
  - Session list with detailed metadata display
  - Context-aware restoration showing previous work
  
- **Context Extraction** (`session_context_extractor.py`)
  - Extract important information from conversation history
  - Generate structured context summaries
  - Automatic context file generation (JSON and Markdown)
  
- **Enhanced Slash Commands**
  - Unified `/session` command for all session operations
  - Simplified `/save` command with auto-features
  - Consolidated redundant commands

### Changed
- Improved README with v1.1.0 features documentation
- Updated slash command examples for simplified workflow
- Enhanced session restoration with automatic context display

### Fixed
- Excluded `_context.json` files from session listing
- Windows encoding issues in context extraction
- Session file filtering to avoid duplicates

## [1.0.0] - 2025-08-22

### Initial Release
- Core session management functionality
- Semantic search without vector database
- Conversation history tracking
- Git status integration
- Claude Code CLI integration
- Automatic embedding generation
- Multi-level restoration (quick, normal, deep)

---
*For detailed feature documentation, see the [README](README.md)*
