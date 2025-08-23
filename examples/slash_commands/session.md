---
explanation: Integrated session management - list, select, and restore with context
requires_approval: false
---

# Integrated Session Management

Complete session management with interactive selection and context restoration.

```bash
cd path/to/project && python session_manager_integrated.py interactive
```

## Features

### 📋 Session List (Latest 20)
- Description and timestamp
- Conversation history lines
- Git branch and file changes
- Context availability

### 🎯 Interactive Selection
- **Number input**: Select from list
- **Keyword search**: Search descriptions
- **'latest'**: Select most recent
- **'q'/'quit'**: Exit

### 📄 Context Restoration
- Session information
- Previous work summary
- Commands and file operations
- Tool usage statistics

## Usage

```
/session              # Interactive mode (recommended)
/session list         # List only
/session restore ID   # Direct restoration
```

## Workflow

1. Execute `/session`
2. View session list
3. Select by number or keyword
4. Context is displayed
5. Continue work

---
*Integrated Session Manager v1.1.0*