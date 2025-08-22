# Claude Semantic Session Manager

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A session state manager for Claude Code CLI with semantic search capabilities. Save and restore your coding sessions with natural language search.

## What This Tool Does

When using Claude Code CLI for development, this tool helps you:
- **Save session state** automatically (git status, files, conversation)
- **Search past sessions** using natural language queries
- **Restore context** from previous sessions
- **Track conversation history** from Claude CLI

All data stays local on your machine.

## Key Features

### Semantic Search
```bash
# Find sessions using natural language
python session_manager.py search "authentication bug with JWT tokens"

# Results with similarity scores
1. SESSION_20250821_143242 (similarity: 0.892)
   Description: Fixed JWT refresh token expiration bug
2. SESSION_20250819_091523 (similarity: 0.743)  
   Description: Implemented OAuth2 authentication flow
```

### Automatic Session Tracking
- Git branch, commits, and diff capture
- Modified and untracked files tracking
- Conversation history from Claude CLI
- Intelligent description generation

### Context Restoration
```bash
# Restore with semantic query
python session_manager.py restore --semantic "working on user dashboard"

# Or restore by session ID
python session_manager.py restore SESSION_20250822_143242
```

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/xxGodLiuxx/claude-semantic-session-manager.git
cd claude-semantic-session-manager
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **First-time setup** (optional - for existing sessions)
```bash
python regenerate_embeddings.py
```

## Usage

### Basic Commands

#### Save Current Session
```bash
# Auto-generated description
python session_manager.py save

# With custom description
python session_manager.py save --description "Implemented user authentication"
```

#### Search Sessions
```bash
# Semantic search
python session_manager.py search "database migrations"
```

#### Restore Session
```bash
# Restore latest session
python session_manager.py restore

# Restore specific session
python session_manager.py restore SESSION_20250822_143242

# Different detail levels
python session_manager.py restore --level quick    # Summary only
python session_manager.py restore --level normal   # Summary + context (default)
python session_manager.py restore --level deep --load-conversation  # Everything
```

#### List Recent Sessions
```bash
python session_manager.py list --limit 10
```

## Claude Code CLI Integration

### Setting Up Slash Commands

Create custom slash commands for seamless integration:

1. **Create command file**: `~/.claude/commands/save-session.md`
```markdown
---
description: Save current session state
allowed-tools: ["Bash"]
---

!cd /your/project/path && python session_manager.py save --description "$DESCRIPTION"
```

2. **Use in Claude Code**:
```
/save-session "Implemented semantic search feature"
```

### Example Slash Commands

The `examples/slash_commands/` directory contains:
- `/save-session` - Save current session
- `/smart-restore` - Restore with semantic search  
- `/session-search` - Search sessions

See `examples/slash_commands/README.md` for setup instructions.

## How It Works

### Technical Implementation
- Uses `sentence-transformers` for text embeddings
- Stores embeddings locally using pickle
- Performs similarity search with NumPy
- Monitors Claude CLI conversation files
- Captures git state and file changes

### Data Storage
- Sessions stored as JSON files in `session_states/`
- Embeddings cached in `~/.claude/session_embeddings.pkl`
- Old sessions archived to `session_archives/` after 180 days

### Performance
- Handles thousands of sessions efficiently
- Search completes in under 200ms typically
- Embeddings generated automatically when saving

## Programmatic Usage

```python
from session_manager import SessionStateManager

# Initialize manager
manager = SessionStateManager(project_root="/path/to/project")

# Save state
session_id = manager.save_state(description="Feature complete")

# Search sessions
results = manager.semantic_search_sessions(
    query="authentication bug",
    top_k=10
)

# Smart restore
manager.smart_restore(semantic_query="working on API endpoints")
```

## Configuration

### Archive Settings
Sessions older than 180 days are automatically archived:
```bash
python session_manager.py cleanup --days 180
```

### Embedding Regeneration
If embeddings get out of sync:
```bash
python regenerate_embeddings.py --force
```

## Troubleshooting

**Installation Issues**
```bash
pip install sentence-transformers numpy
```

**Embeddings Out of Sync**
```bash
python regenerate_embeddings.py --verify
```

**Performance with Many Sessions**
- Tool handles 5000+ sessions efficiently
- Search time scales linearly with session count
- Consider archiving very old sessions

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built for the [Claude Code](https://claude.ai/code) community
- Uses [sentence-transformers](https://www.sbert.net/) for embeddings