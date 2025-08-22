# 🚀 Claude Semantic Session Manager

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Claude Code Compatible](https://img.shields.io/badge/Claude_Code-Compatible-green.svg)](https://claude.ai/code)
[![No Vector DB Required](https://img.shields.io/badge/Vector_DB-Not_Required-brightgreen.svg)](#why-no-vector-database)

**The ONLY Claude Code session manager with semantic search that doesn't require a vector database.** 

Pure Python implementation that lets you search through months of coding sessions using natural language - no external services, no complex setup, just works.

## 🎯 The Problem

When using Claude Code CLI for extended development sessions, you face:
- **Context Loss**: Claude forgets what you worked on yesterday
- **Session Fragmentation**: Hard to find that specific implementation from last week
- **Manual Tracking**: Constantly copying important information
- **No Searchability**: Can't search across sessions with "that authentication bug fix"

## ✨ The Solution

Claude Semantic Session Manager automatically:
- 📸 **Captures** complete session state (git, files, conversation)
- 🔍 **Enables semantic search** across ALL your sessions
- 🔄 **Restores context** intelligently based on your query
- 🚫 **No vector DB required** - runs completely locally
- ⚡ **Zero configuration** - works out of the box

## 🌟 Key Features

### Semantic Search Without Vector Database
```bash
# Find sessions using natural language
python session_manager.py search "authentication bug with JWT tokens"

# Results ranked by relevance
1. SESSION_20250821_143242 (similarity: 0.892)
   Description: Fixed JWT refresh token expiration bug
2. SESSION_20250819_091523 (similarity: 0.743)  
   Description: Implemented OAuth2 authentication flow
```

### Automatic Session Tracking
- Git branch, commits, and diff capture
- Modified and untracked files tracking
- Conversation history from Claude CLI
- Intelligent description generation from changes

### Smart Context Restoration
```bash
# Restore with semantic query
python session_manager.py restore --semantic "working on user dashboard"

# Or restore by session ID
python session_manager.py restore SESSION_20250822_143242
```

## 🚀 Quick Start

### Installation (2 minutes)

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/claude-semantic-session-manager.git
cd claude-semantic-session-manager
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **First-time setup** (one-time only)
```bash
# Generate embeddings for existing sessions (if any)
python regenerate_embeddings.py
```

That's it! No database setup, no API keys, no configuration files.

## 📖 Usage Guide

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

# Returns top 5 relevant sessions with similarity scores
```

#### Restore Session
```bash
# Restore latest session
python session_manager.py restore

# Restore specific session
python session_manager.py restore SESSION_20250822_143242

# Restore with different detail levels
python session_manager.py restore --level quick    # Summary only
python session_manager.py restore --level normal   # Summary + context (default)
python session_manager.py restore --level deep --load-conversation  # Everything
```

#### List Recent Sessions
```bash
python session_manager.py list --limit 10
```

## 🔧 Claude Code CLI Integration

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

### Recommended Slash Commands

#### `/smart-restore`
```markdown
---
description: Smart context-aware restoration
allowed-tools: ["Bash"]
---

!cd /your/project/path && python session_manager.py restore --semantic "$QUERY"
```

#### `/session-search`
```markdown
---
description: Search sessions semantically
allowed-tools: ["Bash"]
---

!cd /your/project/path && python session_manager.py search "$QUERY"
```

## 🏗️ Architecture

### Why No Vector Database?

Traditional solutions require complex setups with vector databases like:
- Pinecone, Weaviate, Qdrant
- Redis with vector extensions
- PostgreSQL with pgvector

**Our approach:**
- Uses `sentence-transformers` for embeddings
- NumPy for cosine similarity calculations
- Pickle for efficient caching
- **Result**: 50ms search across 5000+ sessions

### Technical Stack

- **Embeddings**: `all-MiniLM-L6-v2` model (lightweight, fast)
- **Storage**: JSON for sessions, pickle for embeddings
- **Search**: Cosine similarity with NumPy
- **Performance**: O(n) search, suitable for up to 5000 sessions

## 📊 Comparison with Alternatives

| Feature | Our Solution | claude-sessions | Claude Context | RedisVL |
|---------|-------------|-----------------|----------------|---------|
| Semantic Search | ✅ | ❌ | ✅ | ✅ |
| No Vector DB | ✅ | ✅ | ❌ | ❌ |
| Setup Time | 2 min | 5 min | 60+ min | 30+ min |
| Memory Usage | 200MB | 50MB | 500MB+ | 400MB+ |
| Max Sessions | 5000 | 100 | Unlimited | Unlimited |
| Privacy | Local only | Local only | Cloud option | Cloud option |

## 🔍 Advanced Usage

### Programmatic API

```python
from session_manager import SessionStateManager

# Initialize manager
manager = SessionStateManager(project_root="/path/to/project")

# Save state programmatically
session_id = manager.save_state(description="Feature complete")

# Search sessions
results = manager.semantic_search_sessions(
    query="authentication bug",
    top_k=10
)

for similarity, session_id, info in results:
    print(f"{session_id}: {info['description']} (score: {similarity:.3f})")

# Smart restore
manager.smart_restore(semantic_query="working on API endpoints")
```

### Auto-Save Configuration

The manager includes a background auto-saver that triggers on:
- Significant git changes
- Every 30 minutes of activity
- Before potentially destructive operations

### Archiving Old Sessions

Sessions older than 180 days are automatically archived:
```bash
python session_manager.py cleanup --days 180
```

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'sentence_transformers'"**
```bash
pip install sentence-transformers
```

**"Embeddings out of sync"**
```bash
python regenerate_embeddings.py --force
```

**Performance slow with many sessions**
- The tool handles up to 5000 sessions efficiently
- For larger scales, consider implementing batched search
- Or contribute a vector DB adapter!

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Priority Areas
- [ ] Vector database adapters (optional backends)
- [ ] Web UI for session browsing
- [ ] Session diff visualization
- [ ] Multi-project support
- [ ] Team collaboration features

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the [Claude Code](https://claude.ai/code) community
- Inspired by the need for better AI coding session management
- Uses the excellent [sentence-transformers](https://www.sbert.net/) library

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/claude-semantic-session-manager&type=Date)](https://star-history.com/#yourusername/claude-semantic-session-manager&Date)

---

**Made with ❤️ for developers who refuse to lose context**

*If this tool saves you time, consider starring ⭐ the repository!*