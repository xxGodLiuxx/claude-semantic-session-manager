#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claude Semantic Session Manager v1.0.0
A lightweight session state manager with semantic search capabilities for Claude Code CLI.
No vector database required - pure Python implementation with sentence-transformers.

Features:
- Automatic session state capture and restoration
- Semantic search across all sessions
- Git integration for context preservation
- Conversation history tracking
- Smart context-aware restoration
- Zero configuration setup

Author: Open Source Community
License: MIT
"""

import os
import sys
import json
import subprocess
import hashlib
import traceback
import threading
import time
import shutil
import zipfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
from glob import glob
import codecs
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    except:
        pass
    os.environ['PYTHONIOENCODING'] = 'utf-8'

VERSION = "1.0.0"
LAST_UPDATED = "2025-08-22"

# Archive configuration
ARCHIVE_DAYS = 180  # Keep sessions for 6 months

class ConversationMonitor:
    """Monitor and save conversation history from Claude CLI JSONL files"""
    
    def __init__(self, session_id: str, output_dir: Path):
        self.session_id = session_id
        self.output_file = output_dir / f"{session_id}_conv.jsonl"
        self.monitoring = False
        self.monitor_thread = None
        self.processed_messages = set()
        
    def start_monitoring(self):
        """Start background JSONL monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _find_latest_jsonl(self) -> Optional[Path]:
        """Find the latest Claude CLI JSONL file"""
        # Look for Claude CLI project files
        home = Path.home()
        patterns = [
            home / ".claude" / "projects" / "**" / "*.jsonl",
        ]
        
        all_files = []
        for pattern in patterns:
            files = list(pattern.parent.glob(pattern.name))
            all_files.extend(files)
        
        if not all_files:
            return None
        
        return max(all_files, key=lambda x: x.stat().st_mtime)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        last_size = 0
        jsonl_file = None
        
        while self.monitoring:
            try:
                # Find JSONL file
                if not jsonl_file or not jsonl_file.exists():
                    jsonl_file = self._find_latest_jsonl()
                    if not jsonl_file:
                        time.sleep(2)
                        continue
                
                # Check for new content
                current_size = jsonl_file.stat().st_size
                if current_size > last_size:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        f.seek(last_size)
                        new_lines = f.read()
                        
                        # Save to output file
                        with open(self.output_file, 'a', encoding='utf-8') as out:
                            out.write(new_lines)
                    
                    last_size = current_size
                
            except Exception:
                pass
            
            time.sleep(1)

class SemanticSearch:
    """Lightweight semantic search without vector database"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[Path] = None):
        """Initialize semantic search engine
        
        Args:
            model_name: Sentence transformer model name
            cache_dir: Directory for embedding cache
        """
        self.model = None
        self.model_name = model_name
        self.cache_dir = cache_dir or (Path.home() / ".claude")
        self.cache_file = self.cache_dir / "session_embeddings.pkl"
        self.embeddings_cache = {}
        
        try:
            self.model = SentenceTransformer(model_name)
            self._load_cache()
        except Exception as e:
            print(f"[WARNING] Failed to initialize semantic search: {e}")
    
    def _load_cache(self):
        """Load embeddings cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
            except Exception:
                self.embeddings_cache = {}
    
    def _save_cache(self):
        """Save embeddings cache to disk"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            print(f"[WARNING] Failed to save embeddings cache: {e}")
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.model:
            return None
        
        try:
            # Check cache
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.embeddings_cache:
                return self.embeddings_cache[text_hash]
            
            # Generate new embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Cache it
            self.embeddings_cache[text_hash] = embedding
            
            return embedding
        except Exception:
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, documents: List[Tuple[str, str]], top_k: int = 5) -> List[Tuple[float, str, str]]:
        """Search for similar documents
        
        Args:
            query: Search query
            documents: List of (id, text) tuples
            top_k: Number of results to return
            
        Returns:
            List of (similarity, id, text) tuples
        """
        if not self.model or not documents:
            return []
        
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        results = []
        for doc_id, doc_text in documents:
            doc_embedding = self.get_embedding(doc_text)
            if doc_embedding is not None:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                results.append((similarity, doc_id, doc_text))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def save_embeddings(self):
        """Persist embeddings cache to disk"""
        self._save_cache()

class BackgroundAutoSaver:
    """Automatic background saving with intelligent triggers"""
    
    def __init__(self, manager):
        self.manager = manager
        self.last_save_time = datetime.now()
        self.last_git_hash = None
        self.save_thread = None
        self.running = False
        
    def start(self):
        """Start background auto-save monitoring"""
        self.running = True
        self.save_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.save_thread.start()
        print("[INFO] Background auto-saver started")
    
    def stop(self):
        """Stop background monitoring"""
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=5)
    
    def _get_git_hash(self) -> Optional[str]:
        """Get current git state hash"""
        try:
            result = subprocess.run(
                ["git", "diff", "HEAD"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                return hashlib.md5(result.stdout.encode()).hexdigest()
        except:
            pass
        return None
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                # Check every 5 minutes
                time.sleep(300)
                
                current_hash = self._get_git_hash()
                time_since_save = (datetime.now() - self.last_save_time).seconds
                
                # Auto-save conditions
                should_save = False
                reason = ""
                
                # 1. Significant git changes
                if current_hash and current_hash != self.last_git_hash:
                    should_save = True
                    reason = "git_changes"
                    self.last_git_hash = current_hash
                
                # 2. Time-based (every 30 minutes)
                elif time_since_save > 1800:
                    should_save = True
                    reason = "scheduled"
                
                if should_save:
                    print(f"\n[AUTO-SAVE] Triggering auto-save (reason: {reason})")
                    self.manager.save_state(
                        description=f"Auto-save ({reason})",
                        auto_save=True
                    )
                    self.last_save_time = datetime.now()
                    
            except Exception as e:
                print(f"[WARNING] Auto-save error: {e}")
            
            time.sleep(60)  # Check every minute

class SessionStateManager:
    def __init__(self, project_root: Optional[str] = None, quiet_mode: bool = False, enable_semantic: bool = True):
        """Initialize Session State Manager
        
        Args:
            project_root: Project root directory
            quiet_mode: Suppress output
            enable_semantic: Enable semantic search features
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.session_dir = self.project_root / "session_states"
        self.archive_dir = self.project_root / "session_archives"
        self.metadata_file = self.session_dir / "metadata.json"
        self.quiet = quiet_mode
        self.semantic_search = None
        self.auto_saver = None
        
        # Initialize semantic search if enabled
        if enable_semantic:
            try:
                self.semantic_search = SemanticSearch()
                if not self.quiet:
                    print("[INFO] Semantic search engine initialized")
            except Exception as e:
                if not self.quiet:
                    print(f"[WARNING] Semantic search unavailable: {e}")
        
        self._ensure_directories()
        
        # Start auto-saver
        self.auto_saver = BackgroundAutoSaver(self)
    
    def _ensure_directories(self):
        """Ensure necessary directories exist"""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_git_info(self) -> Dict[str, Any]:
        """Get current git repository information"""
        git_info = {
            "branch": "unknown",
            "commit": None,
            "modified_files": [],
            "untracked_files": [],
            "diff_stats": {}
        }
        
        try:
            # Get branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
            
            # Get modified files
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if line.startswith(" M") or line.startswith("M "):
                        git_info["modified_files"].append(line[3:])
                    elif line.startswith("??"):
                        git_info["untracked_files"].append(line[3:])
            
            # Get diff stats
            result = subprocess.run(
                ["git", "diff", "--stat"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0 and result.stdout.strip():
                git_info["diff_stats"]["summary"] = result.stdout.strip().split('\n')[-1]
        
        except Exception:
            pass
        
        return git_info
    
    def _generate_smart_description(self, conversation_file: Optional[Path] = None) -> str:
        """Generate intelligent description from git diff and conversation history"""
        components = []
        
        # Analyze git changes
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0 and result.stdout.strip():
                files = result.stdout.strip().split('\n')
                
                # Categorize changes
                categories = {
                    'session_manager': False,
                    'claude_md': False,
                    'docs': False,
                    'scripts': False,
                    'configs': False
                }
                
                for file in files:
                    if 'session' in file.lower() and 'manager' in file.lower():
                        categories['session_manager'] = True
                    elif 'CLAUDE.md' in file:
                        categories['claude_md'] = True
                    elif file.startswith('docs/'):
                        categories['docs'] = True
                    elif file.startswith('scripts/'):
                        categories['scripts'] = True
                    elif any(ext in file for ext in ['.json', '.yaml', '.yml', '.ini']):
                        categories['configs'] = True
                
                # Build description
                if categories['session_manager']:
                    components.append("Session Manager updates")
                if categories['claude_md']:
                    components.append("Configuration updates")
                if categories['docs']:
                    components.append("Documentation changes")
                if categories['scripts']:
                    components.append("Script modifications")
                if categories['configs']:
                    components.append("Config updates")
        except:
            pass
        
        # Analyze conversation if available
        if conversation_file and conversation_file.exists():
            try:
                # Look for key implementation markers
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Search for version mentions
                    version_pattern = r'v\d+\.\d+\.\d+'
                    versions = re.findall(version_pattern, content)
                    if versions:
                        latest_version = max(versions)
                        components.insert(0, f"Implementation {latest_version}")
                    
                    # Look for feature keywords
                    if 'semantic' in content.lower() and 'search' in content.lower():
                        components.append("Semantic search")
                    if 'embedding' in content.lower():
                        components.append("Embeddings")
                    if 'automat' in content.lower():
                        components.append("Automation")
            except:
                pass
        
        if components:
            return " - ".join(components[:3])  # Limit to 3 main components
        
        return f"Development session {datetime.now().strftime('%H:%M')}"
    
    def save_state(self, description: Optional[str] = None, auto_save: bool = False) -> str:
        """Save current session state
        
        Args:
            description: Optional description
            auto_save: Whether this is an auto-save
            
        Returns:
            Session ID
        """
        timestamp = datetime.now()
        session_id = f"SESSION_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Use smart description if not provided
        conversation_file = self.session_dir / f"{session_id}_conv.jsonl"
        if not description:
            description = self._generate_smart_description(conversation_file)
        
        # Collect session data
        session_data = {
            "session_id": session_id,
            "timestamp": timestamp.isoformat(),
            "description": description,
            "auto_save": auto_save,
            "working_directory": str(self.project_root),
            "python_version": sys.version.split()[0],
            "git_info": self._get_git_info(),
            "environment": {
                "platform": sys.platform,
                "encoding": sys.getdefaultencoding()
            }
        }
        
        # Save session file
        session_file = self.session_dir / f"{session_id}.json"
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        # Generate summary
        self._create_summary(session_id, session_data)
        
        # Update metadata
        self._update_metadata(session_id, description)
        
        # Generate embedding if semantic search is enabled
        if self.semantic_search:
            self._generate_embedding(session_id, session_data)
        
        if not self.quiet:
            print(f"[SUCCESS] Session saved: {session_id}")
            print(f"  Description: {description}")
        
        return session_id
    
    def _generate_embedding(self, session_id: str, session_data: Dict[str, Any]):
        """Generate and cache embedding for session"""
        if not self.semantic_search:
            return
        
        # Create searchable text from session data
        git_info = session_data.get('git_info', {})
        
        text_components = [
            session_data.get('description', ''),
            session_data.get('working_directory', ''),
            session_data.get('python_version', ''),
            git_info.get('branch', ''),
            ' '.join(git_info.get('modified_files', [])),
            ' '.join(git_info.get('untracked_files', []))
        ]
        
        session_text = ' '.join(filter(None, text_components))
        
        # Generate embedding
        embedding = self.semantic_search.get_embedding(session_text)
        if embedding is not None:
            # Save to cache
            self.semantic_search.embeddings_cache[session_id] = embedding
            self.semantic_search.save_embeddings()
    
    def semantic_search_sessions(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict]]:
        """Search sessions using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (similarity, session_id, session_info) tuples
        """
        if not self.semantic_search:
            print("[ERROR] Semantic search not available")
            return []
        
        # Prepare documents
        documents = []
        session_info_map = {}
        
        for session_file in sorted(self.session_dir.glob("SESSION_*.json")):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session_id = data['session_id']
                    
                    # Create searchable text
                    git_info = data.get('git_info', {})
                    text = ' '.join(filter(None, [
                        data.get('description', ''),
                        data.get('working_directory', ''),
                        git_info.get('branch', ''),
                        ' '.join(git_info.get('modified_files', []))
                    ]))
                    
                    documents.append((session_id, text))
                    session_info_map[session_id] = {
                        'description': data.get('description', 'No description'),
                        'timestamp': data.get('timestamp', 'Unknown')
                    }
            except Exception:
                continue
        
        # Search
        results = self.semantic_search.search(query, documents, top_k)
        
        # Format results
        formatted_results = []
        for similarity, session_id, _ in results:
            if session_id in session_info_map:
                formatted_results.append((similarity, session_id, session_info_map[session_id]))
        
        return formatted_results
    
    def regenerate_all_embeddings(self, force: bool = False) -> int:
        """Regenerate embeddings for all sessions
        
        Args:
            force: Force regeneration even if embeddings exist
            
        Returns:
            Number of sessions processed
        """
        if not self.semantic_search:
            print("[ERROR] Semantic search not initialized")
            return 0
        
        sessions = list(self.session_dir.glob("SESSION_*.json"))
        total = len(sessions)
        processed = 0
        skipped = 0
        errors = 0
        
        print(f"[INFO] Found {total} sessions to process")
        
        for i, session_file in enumerate(sessions, 1):
            try:
                session_id = session_file.stem
                
                # Check if already cached (unless forced)
                if not force and session_id in self.semantic_search.embeddings_cache:
                    skipped += 1
                    continue
                
                # Load session data
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Generate embedding
                self._generate_embedding(session_id, data)
                processed += 1
                
                # Progress update
                if i % 10 == 0:
                    print(f"[PROGRESS] Processed {i}/{total} sessions...")
                    
            except Exception as e:
                errors += 1
                print(f"[ERROR] Failed to process {session_file.name}: {e}")
        
        # Save all embeddings
        self.semantic_search.save_embeddings()
        
        print(f"\n[COMPLETE] Embedding regeneration finished:")
        print(f"  - Total sessions: {total}")
        print(f"  - Newly processed: {processed}")
        print(f"  - Skipped (cached): {skipped}")
        print(f"  - Errors: {errors}")
        print(f"  - Cache size: {len(self.semantic_search.embeddings_cache)} embeddings")
        
        return processed
    
    def verify_embeddings(self) -> Dict[str, int]:
        """Verify embedding coverage status
        
        Returns:
            Statistics dictionary
        """
        if not self.semantic_search:
            print("[ERROR] Semantic search not initialized")
            return {}
        
        sessions = list(self.session_dir.glob("SESSION_*.json"))
        total_sessions = len(sessions)
        cached_embeddings = len(self.semantic_search.embeddings_cache)
        
        coverage = (cached_embeddings / total_sessions * 100) if total_sessions > 0 else 0
        
        stats = {
            'total_sessions': total_sessions,
            'cached_embeddings': cached_embeddings,
            'coverage_percent': coverage,
            'cache_file': str(self.semantic_search.cache_file),
            'cache_exists': self.semantic_search.cache_file.exists()
        }
        
        print(f"\n[INFO] Embedding Status:")
        print(f"  - Total sessions: {total_sessions}")
        print(f"  - Cached embeddings: {cached_embeddings}")
        print(f"  - Coverage: {coverage:.1f}%")
        print(f"  - Cache file: {stats['cache_file']}")
        print(f"  - Cache exists: {stats['cache_exists']}")
        
        return stats
    
    def smart_restore(self, semantic_query: Optional[str] = None) -> Optional[str]:
        """Smart context-aware session restoration
        
        Args:
            semantic_query: Natural language query for finding session
            
        Returns:
            Session ID if restored, None otherwise
        """
        # Try semantic search first if query provided
        if semantic_query and self.semantic_search:
            results = self.semantic_search_sessions(semantic_query, top_k=1)
            if results and results[0][0] > 0.7:  # Similarity threshold
                session_id = results[0][1]
                print(f"[SMART] Found matching session: {session_id}")
                print(f"  Similarity: {results[0][0]:.3f}")
                print(f"  Description: {results[0][2]['description']}")
                return self.restore_state(session_id, level='normal')
        
        # Fallback to latest session
        sessions = self._list_sessions()
        if not sessions:
            print("[INFO] No sessions found")
            return None
        
        latest = sessions[0]
        print(f"[SMART] Restoring latest session: {latest['session_id']}")
        return self.restore_state(latest['session_id'], level='normal')
    
    def restore_state(self, session_id: Optional[str] = None, level: str = 'normal', load_conversation: bool = False) -> Optional[str]:
        """Restore a session state
        
        Args:
            session_id: Session to restore (latest if None)
            level: Restoration level ('quick', 'normal', 'deep')
            load_conversation: Load conversation history
            
        Returns:
            Session ID if restored
        """
        if not session_id:
            # Get latest session
            sessions = self._list_sessions()
            if not sessions:
                print("[ERROR] No sessions found")
                return None
            session_id = sessions[0]['session_id']
        
        session_file = self.session_dir / f"{session_id}.json"
        if not session_file.exists():
            print(f"[ERROR] Session not found: {session_id}")
            return None
        
        # Load session data
        with open(session_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"RESTORING SESSION: {session_id}")
        print(f"{'='*60}")
        
        # Quick level - just summary
        if level == 'quick':
            summary_file = self.session_dir / f"{session_id}_summary.md"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    print(f.read())
        
        # Normal level - summary + context
        elif level == 'normal':
            summary_file = self.session_dir / f"{session_id}_summary.md"
            if summary_file.exists():
                with open(summary_file, 'r', encoding='utf-8') as f:
                    print(f.read())
            
            print(f"\n[GIT STATE]")
            git_info = data.get('git_info', {})
            print(f"Branch: {git_info.get('branch', 'unknown')}")
            print(f"Modified files: {len(git_info.get('modified_files', []))}")
            print(f"Untracked files: {len(git_info.get('untracked_files', []))}")
        
        # Deep level - everything
        elif level == 'deep':
            print(json.dumps(data, indent=2))
            
            if load_conversation:
                conv_file = self.session_dir / f"{session_id}_conv.jsonl"
                if conv_file.exists():
                    print(f"\n[CONVERSATION HISTORY]")
                    with open(conv_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            print(line.strip())
        
        print(f"\n[RESTORED] Session {session_id}")
        return session_id
    
    def _list_sessions(self) -> List[Dict]:
        """List all sessions"""
        sessions = []
        for session_file in sorted(self.session_dir.glob("SESSION_*.json"), reverse=True):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sessions.append({
                        'session_id': data['session_id'],
                        'timestamp': data['timestamp'],
                        'description': data.get('description', 'No description')
                    })
            except:
                continue
        return sessions
    
    def _create_summary(self, session_id: str, session_data: Dict):
        """Create a summary file for the session"""
        summary_file = self.session_dir / f"{session_id}_summary.md"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Session State: {session_id}\n\n")
            f.write(f"## Basic Information\n")
            f.write(f"- **Saved at**: {session_data['timestamp']}\n")
            f.write(f"- **Mode**: {'auto' if session_data.get('auto_save') else 'manual'}\n")
            f.write(f"- **Description**: {session_data['description']}\n")
            f.write(f"- **Conversation history**: Saved (v3.0.0)\n\n")
            
            f.write(f"## Working Environment\n")
            f.write(f"- **Working directory**: {session_data['working_directory']}\n")
            f.write(f"- **Python**: {session_data['python_version']}\n\n")
            
            f.write(f"## Git Status\n")
            git_info = session_data['git_info']
            f.write(f"- **Branch**: {git_info['branch']}\n")
            f.write(f"- **Modified files**: {len(git_info['modified_files'])}\n")
            f.write(f"- **Untracked files**: {len(git_info['untracked_files'])}\n\n")
            
            f.write(f"---\n")
            f.write(f"*Claude Semantic Session Manager v{VERSION}*\n")
    
    def _update_metadata(self, session_id: str, description: str):
        """Update metadata file"""
        metadata = {}
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except:
                metadata = {}
        
        metadata[session_id] = {
            'timestamp': datetime.now().isoformat(),
            'description': description
        }
        
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def cleanup_old_sessions(self, days: int = ARCHIVE_DAYS):
        """Archive old sessions
        
        Args:
            days: Sessions older than this will be archived
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        archived_count = 0
        
        for session_file in self.session_dir.glob("SESSION_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                session_date = datetime.fromisoformat(data['timestamp'])
                
                if session_date < cutoff_date:
                    # Create archive
                    archive_name = f"{session_file.stem}.zip"
                    archive_path = self.archive_dir / archive_name
                    
                    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                        # Add all related files
                        for pattern in [
                            f"{session_file.stem}.json",
                            f"{session_file.stem}_summary.md",
                            f"{session_file.stem}_conv.jsonl"
                        ]:
                            file_path = self.session_dir / pattern
                            if file_path.exists():
                                zf.write(file_path, pattern)
                                file_path.unlink()
                    
                    archived_count += 1
            except Exception as e:
                print(f"[WARNING] Failed to archive {session_file.name}: {e}")
        
        if archived_count > 0:
            print(f"[INFO] Archived {archived_count} old sessions")
        
        # Check archive size
        self._check_archive_size()
    
    def _check_archive_size(self):
        """Check and report archive directory size"""
        total_size = sum(f.stat().st_size for f in self.archive_dir.glob("*.zip"))
        
        if total_size > 1024 * 1024 * 1024:  # 1GB
            size_gb = total_size / (1024 * 1024 * 1024)
            print(f"[WARNING] Archive size is {size_gb:.2f}GB")
            
            # Show largest files
            files = [(f, f.stat().st_size) for f in self.archive_dir.glob("*.zip")]
            files.sort(key=lambda x: x[1], reverse=True)
            
            print("[INFO] Largest archive files:")
            for f, size in files[:5]:
                size_mb = size / (1024 * 1024)
                print(f"  - {f.name}: {size_mb:.2f}MB")


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"Claude Semantic Session Manager v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save current session state')
    save_parser.add_argument('--description', '-d', help='Session description')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore session state')
    restore_parser.add_argument('session_id', nargs='?', help='Session ID to restore')
    restore_parser.add_argument('--level', choices=['quick', 'normal', 'deep'], default='normal',
                               help='Restoration level')
    restore_parser.add_argument('--load-conversation', action='store_true',
                               help='Load conversation history (deep level only)')
    restore_parser.add_argument('--semantic', help='Search query for semantic restore')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List saved sessions')
    list_parser.add_argument('--limit', type=int, default=10, help='Number of sessions to show')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Archive old sessions')
    cleanup_parser.add_argument('--days', type=int, default=ARCHIVE_DAYS,
                                help=f'Archive sessions older than N days (default: {ARCHIVE_DAYS})')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a session')
    delete_parser.add_argument('session_id', help='Session ID to delete')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Semantic search for sessions')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = SessionStateManager()
    
    if args.command == 'save':
        session_id = manager.save_state(description=args.description)
        
    elif args.command == 'restore':
        if args.semantic:
            manager.smart_restore(semantic_query=args.semantic)
        else:
            manager.restore_state(
                session_id=args.session_id,
                level=args.level,
                load_conversation=args.load_conversation
            )
    
    elif args.command == 'list':
        sessions = manager._list_sessions()[:args.limit]
        
        if not sessions:
            print("[INFO] No sessions found")
        else:
            print(f"\n[INFO] Found {len(sessions)} recent sessions:\n")
            for i, session in enumerate(sessions, 1):
                print(f"{i}. {session['session_id']}")
                print(f"   Saved: {session['timestamp']}")
                print(f"   Description: {session['description']}\n")
    
    elif args.command == 'cleanup':
        manager.cleanup_old_sessions(days=args.days)
    
    elif args.command == 'delete':
        session_file = manager.session_dir / f"{args.session_id}.json"
        
        if not session_file.exists():
            print(f"[ERROR] Session not found: {args.session_id}")
            return
        
        if not args.force:
            confirm = input(f"Delete session {args.session_id}? (y/N): ")
            if confirm.lower() != 'y':
                print("[INFO] Deletion cancelled")
                return
        
        # Delete all related files
        for pattern in [
            f"{args.session_id}.json",
            f"{args.session_id}_summary.md", 
            f"{args.session_id}_conv.jsonl"
        ]:
            file_path = manager.session_dir / pattern
            if file_path.exists():
                file_path.unlink()
        
        print(f"[SUCCESS] Session deleted: {args.session_id}")
    
    elif args.command == 'search':
        print(f"[INFO] Searching for: {args.query}")
        print("-" * 60)
        
        results = manager.semantic_search_sessions(args.query, top_k=args.top_k)
        
        if not results:
            print("[INFO] No matching sessions found")
        else:
            print(f"[INFO] Found {len(results)} matching sessions:\n")
            for i, (similarity, session_id, info) in enumerate(results, 1):
                print(f"{i}. {session_id}")
                print(f"   Similarity: {similarity:.3f}")
                print(f"   Description: {info['description']}")
                print(f"   Saved at: {info.get('timestamp', 'Unknown')}\n")


if __name__ == "__main__":
    main()