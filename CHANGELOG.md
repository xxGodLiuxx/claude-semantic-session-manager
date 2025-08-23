# Changelog

All notable changes to this project will be documented in this file.

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

## [3.6.0] - 2025-08-22

### Added
- Universal embedding generation for all sessions
- Persistent semantic search across all historical sessions
- Complete searchability without external vector databases

## [3.5.0] - 2025-08-22

### Added
- Full automation of session management
- Smart restore functionality
- Intelligent description generation

## [3.4.0] - 2025-08-22

### Added
- Semantic search functionality without vector database
- Embedding-based session similarity matching

## [3.3.0] - 2025-08-22

### Changed
- Extended archive period from 7 to 180 days

## [3.2.0] - 2025-08-22

### Added
- In-project archiving system
- 1GB size threshold notifications

## [1.1.0] - 2025-08-22

### Added
- Context extraction functionality
- Integrated session management
- Slash command examples

### Changed
- Simplified command structure from 10 to 3 commands

## [1.0.0] - 2025-08-22

### Added
- Initial release with semantic session management
- Basic save and restore functionality
- Git integration support