---
explanation: Save session - unified command for all save operations
requires_approval: false
---

# Session Save

Save current work state with automatic conversation history and context generation.

```bash
cd path/to/project && python session_manager.py save
```

## Usage

```
/save                   # Auto-generated description
/save "description"     # Custom description
```

## Saved Information

- 📝 Working context (directory, Python environment)
- 🔀 Git status (branch, modified files)
- 💬 Conversation history (automatic JSONL)
- 📋 Context information (auto-generated)
- 🔍 Semantic search embeddings

## Features

- **Auto-description**: Intelligently generated from work content
- **Conversation tracking**: v3.0.0+ automatic
- **Context extraction**: Automatic context file generation
- **Searchable**: Semantic search enabled

## Related Commands

- `/session`: Restore saved sessions
- `/session list`: View all sessions

---
*Session Manager v1.1.0*