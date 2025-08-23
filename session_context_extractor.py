#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session Context Extractor - Extract context from conversation history
Makes past context understandable for Claude Code CLI
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

class SessionContextExtractor:
    """Extract important context from conversation history"""
    
    def __init__(self, session_id: str = None):
        self.project_root = Path.cwd()
        self.states_dir = self.project_root / "session_states"
        self.session_id = session_id or self._get_latest_session()
        
    def _get_latest_session(self) -> str:
        """Get the latest session ID"""
        session_files = sorted(self.states_dir.glob("SESSION_*.json"))
        if not session_files:
            return None
        return session_files[-1].stem
    
    def extract_context(self) -> Dict[str, Any]:
        """Extract context information from conversation history"""
        
        conv_file = self.states_dir / f"{self.session_id}_conv.jsonl"
        if not conv_file.exists():
            print(f"[ERROR] Conversation file not found: {conv_file}")
            return None
        
        context = {
            "session_id": self.session_id,
            "extracted_at": datetime.now().isoformat(),
            "tools_used": defaultdict(int),
            "files_created": set(),
            "files_modified": set(),
            "commands_executed": [],
            "key_decisions": [],
            "errors_encountered": [],
            "user_requests": [],
            "conversation_flow": []
        }
        
        # Parse conversation history
        with open(conv_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    self._process_message(data, context)
                except json.JSONDecodeError:
                    continue
        
        # Convert sets to lists for JSON serialization
        context["files_created"] = list(context["files_created"])
        context["files_modified"] = list(context["files_modified"])
        
        return context
    
    def _process_message(self, data: Dict, context: Dict):
        """Process message and extract context information"""
        
        msg = data.get('message', {})
        role = msg.get('role', '')
        content = msg.get('content', [])
        
        # Extract user requests
        if role == 'user' and not data.get('isMeta'):
            user_text = self._extract_text_content(content)
            if user_text and len(user_text) > 10:
                context["user_requests"].append({
                    "request": user_text[:200],
                    "timestamp": data.get('timestamp', '')
                })
                context["conversation_flow"].append(f"USER: {user_text[:100]}")
        
        # Process assistant responses
        elif role == 'assistant':
            # Track tool usage
            for item in content if isinstance(content, list) else []:
                if isinstance(item, dict) and item.get('type') == 'tool_use':
                    tool_name = item.get('name', '')
                    context["tools_used"][tool_name] += 1
                    self._process_tool_use(item, context)
            
            # Extract important text information
            assistant_text = self._extract_text_content(content)
            if assistant_text:
                self._extract_key_info(assistant_text, context)
                if len(assistant_text) > 50:
                    context["conversation_flow"].append(f"ASSISTANT: {assistant_text[:100]}")
        
        # Process tool results
        if 'toolUseResult' in data:
            result = data['toolUseResult']
            self._process_tool_result(result, context)
    
    def _process_tool_use(self, tool_use: Dict, context: Dict):
        """Process tool usage information"""
        
        tool_name = tool_use.get('name', '')
        tool_input = tool_use.get('input', {})
        
        if tool_name == 'Write':
            file_path = tool_input.get('file_path', '')
            if file_path:
                context["files_created"].add(file_path)
        
        elif tool_name in ['Edit', 'MultiEdit']:
            file_path = tool_input.get('file_path', '')
            if file_path:
                context["files_modified"].add(file_path)
        
        elif tool_name == 'Bash':
            command = tool_input.get('command', '')
            if command and not command.startswith('cd '):
                context["commands_executed"].append(command[:100])
    
    def _process_tool_result(self, result: Dict, context: Dict):
        """Process tool execution results"""
        
        stdout = result.get('stdout', '')
        stderr = result.get('stderr', '')
        
        # Record errors
        if stderr and len(stderr) > 10:
            context["errors_encountered"].append({
                "error": stderr[:200],
                "type": "tool_execution"
            })
    
    def _extract_text_content(self, content) -> str:
        """Extract text from content"""
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    return item.get('text', '')
        
        return ""
    
    def _extract_key_info(self, text: str, context: Dict):
        """Extract key information from text"""
        
        # Look for completion markers
        completion_markers = ['completed', 'implemented', 'created', 'fixed', '完了', '実装']
        for marker in completion_markers:
            if marker in text.lower():
                context["key_decisions"].append(text[:200])
                break
        
        # Extract error information
        if 'error' in text.lower():
            context["errors_encountered"].append({
                "error": text[:200],
                "type": "mentioned"
            })
    
    def generate_context_summary(self, context: Dict) -> str:
        """Generate a markdown summary of context"""
        
        summary = f"""# Session Context: {context['session_id']}

## Previous Work Summary

### Main Operations
"""
        
        # Tool usage statistics
        if context["tools_used"]:
            summary += "\n#### Tools Used:\n"
            for tool, count in sorted(context["tools_used"].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                summary += f"- {tool}: {count} times\n"
        
        # File operations
        if context["files_created"] or context["files_modified"]:
            summary += "\n### File Operations\n"
            if context["files_created"]:
                summary += "\n#### Files Created:\n"
                for f in list(context["files_created"])[:10]:
                    summary += f"- `{f}`\n"
            
            if context["files_modified"]:
                summary += "\n#### Files Modified:\n"
                for f in list(context["files_modified"])[:10]:
                    summary += f"- `{f}`\n"
        
        # Commands executed
        if context["commands_executed"]:
            summary += "\n### Commands Executed\n```bash\n"
            for cmd in context["commands_executed"][:10]:
                summary += f"{cmd}\n"
            summary += "```\n"
        
        # User requests
        if context["user_requests"]:
            summary += "\n### Main User Requests\n"
            for req in context["user_requests"][:5]:
                summary += f"- {req['request']}\n"
        
        # Errors encountered
        if context["errors_encountered"]:
            summary += "\n### Errors Encountered\n"
            for err in context["errors_encountered"][:5]:
                summary += f"- [{err['type']}] {err['error'][:100]}...\n"
        
        # Conversation flow
        if context["conversation_flow"]:
            summary += "\n### Conversation Flow (Latest)\n"
            for msg in context["conversation_flow"][-10:]:
                summary += f"- {msg}\n"
        
        summary += f"\n---\n*Extracted at: {context['extracted_at']}*\n"
        
        return summary
    
    def save_context_file(self, context: Dict):
        """Save context information to files"""
        
        # Save as JSON
        context_file = self.states_dir / f"{self.session_id}_context.json"
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context, f, ensure_ascii=False, indent=2, default=str)
        
        # Save as Markdown
        summary = self.generate_context_summary(context)
        summary_file = self.states_dir / f"{self.session_id}_context.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        return context_file, summary_file

def main():
    """Main processing"""
    
    # Windows encoding fix
    if sys.platform == 'win32':
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    
    # Get session ID from argument or use latest
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Extract context
    extractor = SessionContextExtractor(session_id)
    context = extractor.extract_context()
    
    if not context:
        print("[ERROR] Failed to extract context")
        return 1
    
    # Save files
    json_file, md_file = extractor.save_context_file(context)
    
    # Console output
    summary = extractor.generate_context_summary(context)
    print(summary)
    
    print(f"\n[OK] Context saved to:")
    print(f"  - JSON: {json_file}")
    print(f"  - Markdown: {md_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())