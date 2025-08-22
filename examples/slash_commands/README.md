# Claude Code Slash Commands Setup

This directory contains example slash commands for seamless integration with Claude Code CLI.

## Installation

1. **Copy commands to Claude Code directory**:
```bash
# Linux/Mac
cp *.md ~/.claude/commands/

# Windows
copy *.md %USERPROFILE%\.claude\commands\
```

2. **Update paths in command files**:
Edit each `.md` file and replace `/path/to/your/project` with your actual project path.

Example for `save-session.md`:
```markdown
!cd /home/user/my-project && python session_manager.py save --description "${DESCRIPTION:-Auto-saved session}"
```

## Available Commands

### `/save-session`
Save current session state with optional description.
```
/save-session "Implemented user authentication"
```

### `/smart-restore`
Restore session using semantic search.
```
/smart-restore "working on database issues"
```

### `/session-search`
Search sessions semantically.
```
/session-search "authentication bug fixes"
```

## Creating Custom Commands

1. **Create new `.md` file** in `~/.claude/commands/`
2. **Use this template**:
```markdown
---
description: Your command description
allowed-tools: ["Bash"]
---

!cd /path/to/project && python session_manager.py [command] [args]
```

3. **Test the command** in Claude Code:
```
/your-command-name
```

## Advanced Usage

### Dynamic Descriptions
Use Claude Code variables for dynamic content:
```markdown
!cd /path/to/project && python session_manager.py save --description "Session: ${CURRENT_TIME}"
```

### Conditional Commands
Check for specific conditions:
```markdown
!cd /path/to/project && if [ -f session_manager.py ]; then python session_manager.py list; else echo "Session manager not found"; fi
```

### Error Handling
Add error checking:
```markdown
!cd /path/to/project && python session_manager.py save --description "${DESCRIPTION}" || echo "Failed to save session"
```

## Troubleshooting

**Command not found**:
- Ensure `.md` file is in `~/.claude/commands/`
- Restart Claude Code CLI
- Check file permissions

**Path errors**:
- Use absolute paths in commands
- Verify session_manager.py location
- Test paths manually first

**Permission errors**:
- Ensure Python script is executable
- Check directory write permissions
- Run with appropriate user permissions