#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Session Manager - Comprehensive session management with context
Provides interactive selection, context extraction, and seamless restoration
"""

import os
import sys
import json
import codecs
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    os.environ['PYTHONIOENCODING'] = 'utf-8'

class IntegratedSessionManager:
    """Integrated session management class"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.states_dir = self.project_root / "session_states"
        self.states_dir.mkdir(exist_ok=True)
        self.current_selection = None
        
    def list_sessions_with_details(self, limit: int = 20) -> List[Dict]:
        """
        Get session list with detailed information
        
        Args:
            limit: Number of sessions to display
            
        Returns:
            List of session information dictionaries
        """
        sessions = []
        # Exclude _context.json and _conv.json files
        session_files = sorted(
            [f for f in self.states_dir.glob("SESSION_*.json") 
             if not (f.stem.endswith("_context") or f.stem.endswith("_conv"))],
            reverse=True
        )
        
        for i, session_file in enumerate(session_files[:limit]):
            try:
                # Load session information
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for conversation history
                conv_file = session_file.parent / f"{session_file.stem}_conv.jsonl"
                has_conversation = conv_file.exists()
                conv_lines = 0
                if has_conversation:
                    conv_lines = len(conv_file.read_text(encoding='utf-8', errors='ignore').splitlines())
                
                # Check for context file
                context_file = session_file.parent / f"{session_file.stem}_context.md"
                has_context = context_file.exists()
                
                session_info = {
                    "index": i + 1,
                    "session_id": session_file.stem,
                    "description": data.get('description', 'No description'),
                    "timestamp": data.get('timestamp', ''),
                    "has_conversation": has_conversation,
                    "conversation_lines": conv_lines,
                    "has_context": has_context,
                    "git_branch": data.get('git_status', {}).get('branch', 'unknown'),
                    "files_modified": len(data.get('git_status', {}).get('modified_files', [])),
                    "files_created": len(data.get('git_status', {}).get('untracked_files', []))
                }
                sessions.append(session_info)
                
            except Exception as e:
                print(f"[WARNING] Failed to load session {session_file.stem}: {e}")
                
        return sessions
    
    def display_session_list(self, sessions: List[Dict]):
        """Display session list in formatted way"""
        
        print("\n" + "="*80)
        print("📋 Session History (Latest 20)")
        print("="*80)
        
        for session in sessions:
            # Index and basic info
            print(f"\n{session['index']:2}. [{session['session_id']}]")
            
            # Description (truncate if too long)
            desc = session['description']
            if len(desc) > 60:
                desc = desc[:57] + "..."
            print(f"    📝 {desc}")
            
            # Meta information
            meta_parts = []
            
            # Conversation history
            if session['has_conversation']:
                meta_parts.append(f"💬 {session['conversation_lines']} lines")
            
            # Context info
            if session['has_context']:
                meta_parts.append("📄 Context")
            
            # Git info
            if session['git_branch'] != 'unknown':
                meta_parts.append(f"🔀 {session['git_branch']}")
            
            # File changes
            if session['files_modified'] > 0 or session['files_created'] > 0:
                meta_parts.append(f"📁 M:{session['files_modified']} N:{session['files_created']}")
            
            # Timestamp
            if session['timestamp']:
                time_str = session['timestamp'][:19].replace('T', ' ')
                meta_parts.append(f"🕐 {time_str}")
            
            if meta_parts:
                print(f"    {' | '.join(meta_parts)}")
        
        print("\n" + "="*80)
    
    def select_session_interactive(self, sessions: List[Dict]) -> Optional[str]:
        """
        Interactively select a session
        
        Returns:
            Selected session ID or None
        """
        print("\n💡 Usage:")
        print("  • Enter number: Select session by number")
        print("  • Enter keyword: Search by description")
        print("  • 'latest': Select most recent session")
        print("  • 'q'/'quit': Exit")
        
        while True:
            print("\nSelect> ", end="", flush=True)
            choice = input().strip()
            
            # Exit
            if choice.lower() in ['q', 'quit']:
                return None
            
            # Latest session
            if choice.lower() in ['latest', 'l']:
                if sessions:
                    return sessions[0]['session_id']
                else:
                    print("[ERROR] No sessions found")
                    continue
            
            # Select by number
            try:
                index = int(choice)
                if 1 <= index <= len(sessions):
                    return sessions[index - 1]['session_id']
                else:
                    print(f"[ERROR] Please enter a number between 1-{len(sessions)}")
                    continue
            except ValueError:
                pass
            
            # Search by keyword
            if len(choice) > 2:
                print(f"[INFO] Searching for '{choice}'...")
                result = self._search_session_by_keyword(choice, sessions)
                if result:
                    return result
            
            print("[ERROR] Invalid input")
    
    def _search_session_by_keyword(self, keyword: str, sessions: List[Dict]) -> Optional[str]:
        """Search session by keyword"""
        
        keyword_lower = keyword.lower()
        
        # Search in descriptions
        for session in sessions:
            if keyword_lower in session['description'].lower():
                print(f"[FOUND] Match: {session['description'][:50]}...")
                return session['session_id']
        
        # Search in branch names
        for session in sessions:
            if keyword_lower in session['git_branch'].lower():
                print(f"[FOUND] Branch match: {session['git_branch']}")
                return session['session_id']
        
        print(f"[WARNING] No session found matching '{keyword}'")
        return None
    
    def extract_context(self, session_id: str) -> Dict[str, Any]:
        """Extract context from conversation history"""
        
        conv_file = self.states_dir / f"{session_id}_conv.jsonl"
        if not conv_file.exists():
            return None
        
        context = {
            "session_id": session_id,
            "tools_used": defaultdict(int),
            "files_created": set(),
            "files_modified": set(),
            "commands_executed": [],
            "user_requests": [],
            "key_outputs": [],
            "errors": []
        }
        
        # Parse conversation history
        with open(conv_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    msg = data.get('message', {})
                    role = msg.get('role', '')
                    content = msg.get('content', [])
                    
                    # User requests
                    if role == 'user' and not data.get('isMeta'):
                        text = self._extract_text(content)
                        if text and len(text) > 10:
                            context["user_requests"].append(text[:100])
                    
                    # Tool usage
                    elif role == 'assistant':
                        for item in content if isinstance(content, list) else []:
                            if isinstance(item, dict) and item.get('type') == 'tool_use':
                                tool_name = item.get('name', '')
                                context["tools_used"][tool_name] += 1
                                self._process_tool(item, context)
                    
                except json.JSONDecodeError:
                    continue
        
        # Convert sets to lists
        context["files_created"] = list(context["files_created"])
        context["files_modified"] = list(context["files_modified"])
        
        return context
    
    def _extract_text(self, content) -> str:
        """Extract text from content"""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    return item.get('text', '')
        return ""
    
    def _process_tool(self, tool_use: Dict, context: Dict):
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
            if command:
                context["commands_executed"].append(command[:50])
    
    def restore_with_context(self, session_id: str):
        """Restore session with context information"""
        
        # Load session information
        session_file = self.states_dir / f"{session_id}.json"
        if not session_file.exists():
            print(f"[ERROR] Session not found: {session_id}")
            return False
        
        with open(session_file, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        # Extract context
        context = self.extract_context(session_id)
        
        # Display restoration information
        print("\n" + "="*80)
        print(f"🔄 Session Restoration: {session_id}")
        print("="*80)
        
        # Basic information
        print(f"\n📝 Description: {session_data.get('description', 'No description')}")
        print(f"🕐 Saved at: {session_data.get('timestamp', 'Unknown')[:19]}")
        
        # Working context
        working = session_data.get('working_context', {})
        if working:
            print(f"\n📂 Working Directory: {working.get('directory', 'Unknown')}")
            print(f"🐍 Python: {working.get('python_version', 'Unknown')}")
        
        # Git status
        git = session_data.get('git_status', {})
        if git:
            print(f"\n🔀 Git Branch: {git.get('branch', 'Unknown')}")
            print(f"📝 Modified Files: {len(git.get('modified_files', []))}")
            print(f"✨ New Files: {len(git.get('untracked_files', []))}")
        
        # Context information
        if context:
            print("\n" + "-"*40)
            print("📋 Previous Work Context")
            print("-"*40)
            
            # Main requests
            if context["user_requests"]:
                print("\n🎯 Main Requests:")
                for req in context["user_requests"][:3]:
                    print(f"  • {req}")
            
            # File operations
            if context["files_created"] or context["files_modified"]:
                print(f"\n📁 File Operations:")
                if context["files_created"]:
                    print(f"  Created: {', '.join(context['files_created'][:3])}")
                if context["files_modified"]:
                    print(f"  Modified: {', '.join(context['files_modified'][:3])}")
            
            # Tool usage
            if context["tools_used"]:
                print(f"\n🔧 Tools Used:")
                for tool, count in list(context["tools_used"].items())[:5]:
                    print(f"  • {tool}: {count} times")
            
            # Commands executed
            if context["commands_executed"]:
                print(f"\n💻 Commands Executed:")
                for cmd in context["commands_executed"][:5]:
                    print(f"  $ {cmd}...")
        
        print("\n" + "="*80)
        print("✅ Session Restored Successfully")
        print("="*80)
        
        # Show conversation history file path
        conv_file = self.states_dir / f"{session_id}_conv.jsonl"
        if conv_file.exists():
            print(f"\n💬 Conversation History: {conv_file}")
            print("   → Can be loaded in Claude Code CLI")
        
        return True
    
    def run_interactive_flow(self):
        """Run interactive session selection and restoration flow"""
        
        print("\n🚀 Integrated Session Management System")
        print("="*50)
        
        # Get session list
        sessions = self.list_sessions_with_details(limit=20)
        
        if not sessions:
            print("[WARNING] No sessions found")
            return
        
        # Display list
        self.display_session_list(sessions)
        
        # Select session
        selected_id = self.select_session_interactive(sessions)
        
        if not selected_id:
            print("\n[INFO] Session selection cancelled")
            return
        
        # Restore with context
        print(f"\n[INFO] Restoring session {selected_id}...")
        success = self.restore_with_context(selected_id)
        
        if success:
            print("\n💡 You can now continue your work")
        else:
            print("\n[ERROR] Failed to restore session")

def main():
    """Main processing"""
    
    manager = IntegratedSessionManager()
    
    # Command line argument processing
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            # List only
            sessions = manager.list_sessions_with_details()
            manager.display_session_list(sessions)
            
        elif command == "restore":
            # Restore specific session
            if len(sys.argv) > 2:
                session_id = sys.argv[2]
                manager.restore_with_context(session_id)
            else:
                print("[ERROR] Please specify session ID")
                
        elif command == "interactive":
            # Interactive mode
            manager.run_interactive_flow()
            
        else:
            print(f"[ERROR] Unknown command: {command}")
            print("Usage: python session_manager_integrated.py [list|restore|interactive]")
    else:
        # Default is interactive mode
        manager.run_interactive_flow()

if __name__ == "__main__":
    main()