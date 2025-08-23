#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session State Manager v3.6.0 - Universal Embedding & Complete Searchability
全セッションのエンベディング化と完全検索性を実現
過去・現在・未来の全てのセッションが永続的に検索可能

Features:
- v3.5.0の全機能を継承（完全自動化）
- 全過去セッションのエンベディング再生成
- 永続的なセマンティック検索
- ベクトルDB不要（完全ローカル）
- 初回のみの一括エンベディング生成
- 以降は全て自動

Author: Claude Code CLI for JaH
Created: 2025-08-22
Based on: v3.5.0 (2025-08-22)
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
from collections import defaultdict
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

VERSION = "3.8.0"
LAST_UPDATED = "2025-08-23"

# Archive configuration
ARCHIVE_DAYS = 180  # Extended from 7 to 180 days (6 months)

class ConversationMonitor:
    """JSONL監視による会話履歴保存"""
    
    def __init__(self, session_id: str, output_dir: Path):
        self.session_id = session_id
        self.output_file = output_dir / f"{session_id}_conv.jsonl"
        self.monitoring = False
        self.monitor_thread = None
        self.processed_messages = set()
        
    def start_monitoring(self):
        """バックグラウンドでJSONL監視開始"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def _find_latest_jsonl(self) -> Optional[Path]:
        """最新のClaude CLIのJSONLファイルを検索"""
        patterns = [
            r"C:\Users\liuco\.claude\projects\**\*.jsonl",
            r"C:\Users\*\.claude\projects\**\*.jsonl"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob(pattern, recursive=True)
            all_files.extend(files)
        
        if not all_files:
            return None
        
        return Path(max(all_files, key=lambda x: os.path.getmtime(x)))
    
    def _generate_message_id(self, data: Dict) -> str:
        """メッセージの一意ID生成"""
        key_parts = []
        
        if 'message' in data:
            msg = data['message']
            content_str = str(msg.get('content', ''))[:100]
            key_parts.append(content_str)
            key_parts.append(msg.get('role', ''))
        
        if 'timestamp' in data:
            key_parts.append(str(data['timestamp']))
        
        id_str = '_'.join(filter(None, key_parts))
        return hashlib.md5(id_str.encode()).hexdigest()[:16]
    
    def _process_line(self, line: str):
        """行を処理して保存（v3.8.0新規）"""
        try:
            data = json.loads(line)
            msg_id = self._generate_message_id(data)
            
            # 重複チェック
            if msg_id not in self.processed_messages:
                self.processed_messages.add(msg_id)
                
                # 会話履歴として保存
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False)
                    f.write('\n')
        except json.JSONDecodeError:
            pass
    
    def _monitor_loop(self):
        """監視ループ（v3.8.0: 完全保存版）"""
        current_file = None
        current_handle = None
        
        try:
            while self.monitoring:
                # 最新ファイル検索
                latest_file = self._find_latest_jsonl()
                
                if latest_file and latest_file != current_file:
                    # ファイル切り替え
                    if current_handle:
                        current_handle.close()
                    
                    current_file = latest_file
                    current_handle = open(current_file, 'r', encoding='utf-8', errors='ignore')
                    
                    # v3.8.0: 完全読み込み（10KB制限を削除）
                    current_handle.seek(0)  # ファイルの先頭から読む
                    
                    # 既存メッセージを全て処理
                    for line in current_handle:
                        if line.strip():
                            self._process_line(line)
                
                # 新規メッセージの監視継続
                if current_handle:
                    line = current_handle.readline()
                    if line:
                        self._process_line(line)
                
                time.sleep(0.1)  # CPU負荷軽減
                
        except Exception as e:
            print(f"[WARNING] Conversation monitor error: {e}")
        finally:
            if current_handle:
                current_handle.close()


class SemanticSearch:
    """軽量セマンティック検索機能（ベクトルDBなし）"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        セマンティック検索エンジンの初期化
        
        Args:
            model_name: 使用するSentenceTransformerモデル名
        """
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        self.cache_file = Path.home() / '.claude' / 'session_embeddings.pkl'
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # 既存のキャッシュを読み込み
        self.load_cache()
    
    def _init_model(self):
        """モデルを遅延初期化（初回使用時のみ）"""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                print(f"[Warning] Failed to load model: {e}")
                self.model = None
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        テキストをベクトル化（キャッシュ付き）
        
        Args:
            text: 埋め込みを生成するテキスト
            
        Returns:
            埋め込みベクトル（numpy配列）またはNone
        """
        if not text:
            return None
        
        # キャッシュチェック
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        # モデル初期化
        self._init_model()
        if self.model is None:
            return None
        
        try:
            # 埋め込み生成
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # キャッシュに保存
            self.embeddings_cache[text_hash] = embedding
            
            # 定期的にディスクに保存（100件ごと）
            if len(self.embeddings_cache) % 100 == 0:
                self.save_cache()
            
            return embedding
            
        except Exception as e:
            print(f"[Warning] Failed to generate embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        コサイン類似度を計算
        
        Args:
            vec1: ベクトル1
            vec2: ベクトル2
            
        Returns:
            類似度（0.0～1.0）
        """
        if vec1 is None or vec2 is None:
            return 0.0
        
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
            
        except Exception:
            return 0.0
    
    def search(self, query: str, documents: List[Tuple[str, str]], top_k: int = 5) -> List[Tuple[float, str, str]]:
        """
        セマンティック検索を実行
        
        Args:
            query: 検索クエリ
            documents: (ID, テキスト)のタプルリスト
            top_k: 上位何件を返すか
            
        Returns:
            [(類似度, ID, テキスト), ...] のリスト
        """
        if not query or not documents:
            return []
        
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        results = []
        for doc_id, doc_text in documents:
            if not doc_text:
                continue
            
            doc_embedding = self.get_embedding(doc_text)
            if doc_embedding is None:
                continue
            
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            results.append((similarity, doc_id, doc_text))
        
        # 類似度でソート
        results.sort(reverse=True, key=lambda x: x[0])
        return results[:top_k]
    
    def save_cache(self):
        """埋め込みキャッシュをディスクに保存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
        except Exception as e:
            print(f"[Warning] Failed to save embedding cache: {e}")
    
    def load_cache(self):
        """埋め込みキャッシュを読み込み"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
            except Exception:
                # キャッシュ読み込み失敗は無視
                self.embeddings_cache = {}
    
    def clear_cache(self):
        """キャッシュをクリア"""
        self.embeddings_cache = {}
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
            except:
                pass


class BackgroundAutoSaver:
    """バックグラウンド自動保存機能"""
    
    def __init__(self, manager: 'SessionStateManager'):
        """
        自動保存マネージャーの初期化
        
        Args:
            manager: SessionStateManagerインスタンス
        """
        self.manager = manager
        self.last_save = datetime.now()
        self.last_activity = datetime.now()
        self.auto_save_interval = 30 * 60  # 30分
        self.change_threshold = 10  # 10ファイル以上の変更で自動保存
        self.monitoring = False
        self.thread = None
        
    def start(self):
        """バックグラウンド監視開始"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
    def stop(self):
        """バックグラウンド監視停止"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=2)
    
    def _monitor_loop(self):
        """監視ループ"""
        while self.monitoring:
            try:
                time.sleep(60)  # 1分ごとにチェック
                
                if self._should_auto_save():
                    self._perform_auto_save()
                    
            except Exception as e:
                self.manager._log("WARNING", f"Auto-save monitor error: {e}")
    
    def _should_auto_save(self) -> bool:
        """自動保存すべきか判定"""
        # 時間経過チェック
        time_since_save = (datetime.now() - self.last_save).total_seconds()
        if time_since_save > self.auto_save_interval:
            return True
        
        # 変更量チェック
        git_status = self._get_modified_count()
        if git_status > self.change_threshold:
            return True
        
        return False
    
    def _get_modified_count(self) -> int:
        """変更ファイル数を取得"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                return len([l for l in lines if l])
        except:
            pass
        return 0
    
    def _perform_auto_save(self):
        """自動保存実行"""
        try:
            # 自動description生成
            description = self._generate_auto_description()
            
            # サイレント保存
            session_id = self.manager.save_state(
                description=description,
                auto_generated=True
            )
            
            if session_id:
                self.last_save = datetime.now()
                self.manager._log("INFO", f"Auto-saved session: {session_id}")
                
        except Exception as e:
            self.manager._log("ERROR", f"Auto-save failed: {e}")
    
    def _generate_auto_description(self) -> str:
        """自動description生成"""
        parts = []
        
        # 時刻
        time_str = datetime.now().strftime("%H:%M")
        parts.append(f"Auto-save at {time_str}")
        
        # Git情報
        try:
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            if branch_result.returncode == 0:
                branch = branch_result.stdout.strip()
                parts.append(f"on {branch}")
        except:
            pass
        
        # 変更ファイル数
        modified = self._get_modified_count()
        if modified > 0:
            parts.append(f"{modified} files changed")
        
        return " - ".join(parts)


class SessionStateManager:
    """セッション状態管理クラス v3.5.0 - 完全自動化機能付き"""
    
    def __init__(self, quiet_mode: bool = False, enable_semantic: bool = True, 
                 enable_auto_save: bool = False):
        """
        Args:
            quiet_mode: True の場合、エラーを静かに処理
            enable_semantic: セマンティック検索機能を有効化
            enable_auto_save: バックグラウンド自動保存を有効化
        """
        self.quiet_mode = quiet_mode
        self.project_root = Path(__file__).parent.parent
        self.states_dir = self.project_root / "session_states"
        self.states_dir.mkdir(exist_ok=True)
        self.metadata_file = self.states_dir / "metadata.json"
        self.log_file = self.states_dir / "session_manager.log"
        self.conversation_monitor = None
        
        # セマンティック検索エンジン
        self.semantic_search = None
        if enable_semantic:
            try:
                self.semantic_search = SemanticSearch()
                self._log("INFO", "Semantic search engine initialized")
            except Exception as e:
                self._log("WARNING", f"Failed to initialize semantic search: {e}")
                self.semantic_search = None
        
        # バックグラウンド自動保存（v3.5.0新機能）
        self.auto_saver = None
        if enable_auto_save:
            self.auto_saver = BackgroundAutoSaver(self)
            self.auto_saver.start()
            self._log("INFO", "Background auto-save enabled")
        
        # Claude Code CLI検出（v3.5.0新機能）
        self.is_claude_code = os.environ.get('CLAUDE_CODE_CLI') is not None
        
        # Initialize metadata if needed
        if not self.metadata_file.exists():
            self._init_metadata()
    
    def _log(self, level: str, message: str):
        """ログ記録（作業を妨げないよう別ファイルに）"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}\n"
            
            # ファイルログのみ（コンソール出力は最小限）
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
            
            # エラーと重要情報のみコンソール出力
            if not self.quiet_mode and level in ["ERROR", "INFO"]:
                print(f"[{level}] {message}")
        except:
            pass
    
    def _init_metadata(self):
        """メタデータ初期化"""
        metadata = {
            "version": VERSION,
            "sessions": [],
            "last_updated": datetime.now().isoformat(),
            "stats": {
                "total_sessions": 0,
                "total_restores": 0,
                "auto_saves": 0,
                "conversations_saved": 0
            }
        }
        self._save_json(self.metadata_file, metadata)
    
    def _save_json(self, filepath: Path, data: Dict, indent: int = 2):
        """JSON保存（エラー時も作業継続）"""
        try:
            # 一時ファイルに書き込んでから移動（安全性）
            temp_file = filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent, default=str)
            temp_file.replace(filepath)
            return True
        except Exception as e:
            self._log("ERROR", f"JSON save failed: {e}")
            return False
    
    def _load_json(self, filepath: Path) -> Optional[Dict]:
        """JSON読み込み（エラー時None）"""
        try:
            if not filepath.exists():
                return None
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self._log("ERROR", f"JSON load failed: {e}")
            return None
    
    def _get_git_status(self) -> Dict[str, Any]:
        """Git状態を安全に取得（非破壊的）"""
        git_info = {
            "branch": "unknown",
            "modified_files": [],
            "untracked_files": [],
            "staged_files": [],
            "stash_list": [],
            "last_commit": "",
            "submodules": {},
            "remote_status": "unknown"
        }
        
        try:
            # 現在のブランチ
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=5, cwd=self.project_root
            )
            if result.returncode == 0:
                git_info["branch"] = result.stdout.strip()
            
            # Git status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=5, cwd=self.project_root
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    status = line[:2]
                    filename = line[3:].strip()
                    
                    if 'M' in status:
                        git_info["modified_files"].append(filename)
                    if '?' in status:
                        git_info["untracked_files"].append(filename)
                    if 'A' in status or status[0] in 'MADRC':
                        git_info["staged_files"].append(filename)
            
            # 最後のコミット
            result = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=5, cwd=self.project_root
            )
            if result.returncode == 0:
                git_info["last_commit"] = result.stdout.strip()
            
            # Stash list
            result = subprocess.run(
                ["git", "stash", "list"],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=5, cwd=self.project_root
            )
            if result.returncode == 0 and result.stdout.strip():
                git_info["stash_list"] = result.stdout.strip().split('\n')[:5]
            
            # サブモジュール状態
            submodule_dirs = ["Mneme-Personal-Memory-Lighthouse", "Pharosophia_Ch2"]
            for submodule in submodule_dirs:
                submodule_path = self.project_root / submodule
                if submodule_path.exists():
                    sub_info = {"exists": True, "modified": 0, "untracked": 0}
                    
                    result = subprocess.run(
                        ["git", "status", "--porcelain"],
                        capture_output=True, text=True, encoding='utf-8', errors='replace',
                        timeout=5, cwd=submodule_path
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        for line in lines:
                            if line.startswith(' M') or line.startswith('M'):
                                sub_info["modified"] += 1
                            elif line.startswith('??'):
                                sub_info["untracked"] += 1
                    
                    git_info["submodules"][submodule] = sub_info
            
            # リモートとの差分
            result = subprocess.run(
                ["git", "status", "-uno"],
                capture_output=True, text=True, encoding='utf-8', errors='replace',
                timeout=5, cwd=self.project_root
            )
            if result.returncode == 0:
                if "Your branch is ahead" in result.stdout:
                    git_info["remote_status"] = "ahead"
                elif "Your branch is behind" in result.stdout:
                    git_info["remote_status"] = "behind"
                elif "have diverged" in result.stdout:
                    git_info["remote_status"] = "diverged"
                else:
                    git_info["remote_status"] = "up-to-date"
        
        except Exception as e:
            self._log("WARNING", f"Git status collection failed: {e}")
        
        return git_info
    
    def _get_working_context(self) -> Dict[str, Any]:
        """作業コンテキスト収集"""
        context = {
            "session_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "working_directory": str(self.project_root),
            "current_directory": str(Path.cwd()),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "encoding": sys.getdefaultencoding(),
            "recent_files": [],
            "environment_vars": {}
        }
        
        try:
            # 最近編集されたファイル（トップ20）
            all_files = []
            for ext in ['*.py', '*.md', '*.json', '*.txt', '*.yaml', '*.yml']:
                for file_path in self.project_root.rglob(ext):
                    if not any(part.startswith('.') for part in file_path.parts):
                        try:
                            stat = file_path.stat()
                            all_files.append({
                                "path": str(file_path.relative_to(self.project_root)),
                                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                                "size": stat.st_size
                            })
                        except:
                            pass
            
            # 最新順にソート
            all_files.sort(key=lambda x: x["modified"], reverse=True)
            context["recent_files"] = all_files[:20]
            
            # 重要な環境変数
            important_vars = [
                "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "NOTION_API_KEY",
                "PYTHONIOENCODING", "PATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"
            ]
            for var in important_vars:
                value = os.environ.get(var)
                if value:
                    # APIキーなどは一部マスキング
                    if "KEY" in var or "TOKEN" in var or "SECRET" in var:
                        value = "***HIDDEN***"
                    context["environment_vars"][var] = value
        
        except Exception as e:
            self._log("WARNING", f"Context collection failed: {e}")
        
        return context
    
    def _get_claude_context(self) -> Dict[str, Any]:
        """Claude固有のコンテキスト情報"""
        claude_context = {
            "think_hard_status": "unknown",
            "active_commands": [],
            "session_files": [],
            "project_settings": {}
        }
        
        try:
            # セッション関連ファイル
            session_files = list(self.project_root.glob("SESSION_*.md"))
            for sf in session_files[:10]:  # 最新10件
                stat = sf.stat()
                claude_context["session_files"].append({
                    "name": sf.name,
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "size_kb": stat.st_size // 1024
                })
            
            # プロジェクト設定ファイル
            settings_files = [
                self.project_root / ".claude" / "settings.json",
                self.project_root / ".claude" / "settings.local.json"
            ]
            
            for settings_file in settings_files:
                if settings_file.exists():
                    settings = self._load_json(settings_file)
                    if settings:
                        # プリプロンプト関連の設定を取得
                        if "preprompt" in settings:
                            claude_context["project_settings"]["preprompt"] = settings["preprompt"][:100] + "..."
                        if "modelOverrides" in settings:
                            claude_context["project_settings"]["modelOverrides"] = settings["modelOverrides"]
                        if "experimentalFeatures" in settings:
                            claude_context["project_settings"]["experimentalFeatures"] = settings["experimentalFeatures"]
        
        except Exception as e:
            self._log("WARNING", f"Claude context collection failed: {e}")
        
        return claude_context
    
    def _check_archive_size(self):
        """アーカイブディレクトリのサイズをチェックし、1GB以上の場合は通知"""
        try:
            archive_dir = self.project_root / "session_archives"
            if not archive_dir.exists():
                return
            
            # ディレクトリサイズを計算
            total_size = 0
            for item in archive_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
            
            # GB単位に変換
            size_gb = total_size / (1024 ** 3)
            
            if size_gb >= 1.0:
                print(f"\n[WARNING] Archive size: {size_gb:.2f} GB")
                print(f"[INFO] Large archive detected in: {archive_dir}")
                print(f"[INFO] Consider cleaning old archives with 'cleanup' command")
                
                # 個別のアーカイブファイルをリスト表示（上位5件）
                archive_files = sorted(
                    [f for f in archive_dir.glob('*.zip')],
                    key=lambda x: x.stat().st_size,
                    reverse=True
                )
                
                if archive_files:
                    print("\n[INFO] Largest archive files:")
                    for f in archive_files[:5]:
                        file_size_mb = f.stat().st_size / (1024 ** 2)
                        print(f"  - {f.name}: {file_size_mb:.2f} MB")
            
            return size_gb
            
        except Exception as e:
            self._log("WARNING", f"Archive size check failed: {e}")
            return 0
    
    def _archive_old_sessions(self):
        """7日経過したセッションをプロジェクト内専用アーカイブディレクトリに保存"""
        try:
            # プロジェクト内のアーカイブディレクトリ
            archive_dir = self.project_root / "session_archives"
            archive_dir.mkdir(exist_ok=True)
            
            cutoff_date = datetime.now() - timedelta(days=ARCHIVE_DAYS)  # 180 days
            files_to_archive = []
            
            # 180日以上前のセッションファイルを検索（半年保持）
            for session_file in self.states_dir.glob("SESSION_*.json"):
                # ファイル名から日付を抽出
                try:
                    date_str = session_file.stem.split('_')[1]  # SESSION_YYYYMMDD_...
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        files_to_archive.append(session_file)
                        # 関連ファイルも含める
                        related_files = [
                            session_file.with_suffix('.json'),
                            session_file.with_name(f"{session_file.stem}_summary.md"),
                            session_file.with_name(f"{session_file.stem}_conv.jsonl")
                        ]
                        for rf in related_files:
                            if rf.exists() and rf not in files_to_archive:
                                files_to_archive.append(rf)
                except:
                    continue
            
            if files_to_archive:
                # アーカイブ作成
                archive_name = f"sessions_archive_{datetime.now().strftime('%Y%m%d')}.zip"
                archive_path = archive_dir / archive_name
                
                with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for file_path in files_to_archive:
                        zf.write(file_path, file_path.name)
                        file_path.unlink()  # アーカイブ後に削除
                
                self._log("INFO", f"Archived {len(files_to_archive)} files to {archive_path}")
        
        except Exception as e:
            self._log("WARNING", f"Archive operation failed: {e}")
    
    def save_state(self,
                   description: str = "",
                   include_git: bool = True,
                   create_stash: bool = False,
                   save_conversation: bool = True,
                   auto_generated: bool = False) -> Optional[str]:
        """
        作業状態を保存（会話履歴保存機能付き）
        
        Args:
            description: セッションの説明
            include_git: Git情報を含めるか
            create_stash: Git stashを作成するか（デフォルトFalse - 非破壊的）
            save_conversation: 会話履歴を保存するか（v3.0.0新機能）
        
        Returns:
            セッションID or None
        """
        try:
            # アーカイブサイズチェック（v3.2.0新機能）
            self._check_archive_size()
            
            # セッションID生成
            timestamp = datetime.now()
            session_id = f"SESSION_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}"
            
            # 会話履歴監視開始（v3.0.0新機能）
            if save_conversation:
                self.conversation_monitor = ConversationMonitor(session_id, self.states_dir)
                self.conversation_monitor.start_monitoring()
                self._log("INFO", "Conversation monitoring started")
            
            # セッションデータ構築
            # v3.5.0: スマートdescription生成
            if not description:
                description = self._generate_smart_description(include_git)
                if not auto_generated:
                    print(f"[AUTO] Generated description: {description}")
            
            session_data = {
                "session_id": session_id,
                "version": VERSION,
                "mode": "auto" if auto_generated else "manual",
                "description": description,
                "timestamp": timestamp.isoformat(),
                "working_context": self._get_working_context(),
                "conversation_saved": save_conversation,  # v3.0.0
                "auto_generated": auto_generated  # v3.5.0
            }
            
            # Git情報追加
            if include_git:
                session_data["git_status"] = self._get_git_status()
                
                # Stash作成（オプション）
                if create_stash:
                    stash_result = subprocess.run(
                        ["git", "stash", "push", "-m", f"auto-save-{session_id}"],
                        capture_output=True, text=True, encoding='utf-8', errors='replace',
                        cwd=self.project_root
                    )
                    if stash_result.returncode == 0:
                        session_data["git_stash_created"] = True
                        self._log("INFO", "Git stash created")
            
            # Claude固有情報
            session_data["claude_context"] = self._get_claude_context()
            
            # TODO統合プレースホルダー
            session_data["todo_context"] = {
                "integration_status": "pending",
                "placeholder": True
            }
            
            # セッションファイル保存
            session_file = self.states_dir / f"{session_id}.json"
            if not self._save_json(session_file, session_data):
                return None
            
            # サマリーファイル作成
            summary_file = self.states_dir / f"{session_id}_summary.md"
            self._create_summary(session_data, summary_file)
            
            # メタデータ更新
            self._update_metadata(session_id, description)
            
            # セマンティック検索用の埋め込み生成（v3.4.0新機能）
            if self.semantic_search and description:
                try:
                    # セッション情報をテキスト化
                    session_text = f"{description} {working_context.get('directory', '')} {working_context.get('python_version', '')}"
                    if git_status:
                        session_text += f" {git_status.get('branch', '')} {' '.join(git_status.get('modified', []))}"
                    
                    # 埋め込み生成とキャッシュ保存
                    embedding = self.semantic_search.get_embedding(session_text)
                    if embedding is not None:
                        self.semantic_search.save_cache()
                        self._log("INFO", f"Generated embedding for session {session_id}")
                except Exception as e:
                    self._log("WARNING", f"Failed to generate embedding: {e}")
            
            # 古いセッションのアーカイブ（v3.0.0新機能）
            self._archive_old_sessions()
            
            self._log("INFO", f"Saving session state: {session_id}")
            print(f"[OK] Session saved: {session_id}")
            
            if not self.quiet_mode:
                print(f"To restore: python {__file__} restore --session-id {session_id}")
            
            return session_id
            
        except Exception as e:
            self._log("ERROR", f"Save failed: {e}")
            if not self.quiet_mode:
                print(f"[ERROR] Session save failed: {e}")
            return None
    
    def _generate_smart_description(self, include_git: bool = True) -> str:
        """
        インテリジェントなdescription生成（v3.5.0強化版）
        
        Returns:
            自動生成されたdescription
        """
        parts = []
        
        # 会話履歴から主要トピック抽出
        try:
            conv_file = self._find_latest_conversation_file()
            if conv_file and conv_file.exists():
                topics = self._extract_conversation_topics(conv_file)
                if topics:
                    parts.append(f"Topics: {', '.join(topics[:3])}")
        except:
            pass
        
        # Git差分から作業内容を分析
        if include_git:
            git_info = self._analyze_git_changes()
            if git_info:
                parts.append(git_info)
        
        # 変更ファイルの拡張子から作業種別を推測
        work_type = self._classify_work_type()
        if work_type:
            parts.insert(0, work_type)
        
        # 時刻を追加（自動保存の場合）
        if not parts:
            parts.append(f"Work session at {datetime.now().strftime('%H:%M')}")
        
        return " - ".join(parts)
    
    def _extract_conversation_topics(self, conv_file: Path, max_topics: int = 3) -> List[str]:
        """会話履歴から主要トピックを抽出"""
        topics = []
        keywords = []
        
        try:
            with open(conv_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'message' in data and isinstance(data['message'], dict):
                            content = str(data['message'].get('content', ''))
                            # 重要キーワード抽出（簡易版）
                            if 'implement' in content.lower():
                                keywords.append('implementation')
                            if 'error' in content.lower() or 'fix' in content.lower():
                                keywords.append('bugfix')
                            if 'test' in content.lower():
                                keywords.append('testing')
                            if 'refactor' in content.lower():
                                keywords.append('refactoring')
                    except:
                        continue
        except:
            pass
        
        # キーワードから重複を除いて返す
        return list(set(keywords))[:max_topics]
    
    def _analyze_git_changes(self) -> str:
        """Git変更を分析して説明を生成"""
        try:
            # 変更ファイル取得
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0 and result.stdout:
                files = result.stdout.strip().split('\n')
                file_count = len(files)
                
                # ファイル種別を分析
                py_files = [f for f in files if f.endswith('.py')]
                js_files = [f for f in files if f.endswith(('.js', '.ts'))]
                doc_files = [f for f in files if f.endswith('.md')]
                
                parts = []
                if py_files:
                    parts.append(f"{len(py_files)} Python")
                if js_files:
                    parts.append(f"{len(js_files)} JS/TS")
                if doc_files:
                    parts.append(f"{len(doc_files)} docs")
                
                if parts:
                    return f"Modified: {', '.join(parts)}"
                else:
                    return f"{file_count} files modified"
        except:
            pass
        
        return ""
    
    def _classify_work_type(self) -> str:
        """作業種別を推測"""
        try:
            # 最近のコミットメッセージから推測
            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                capture_output=True,
                text=True,
                encoding='utf-8'
            )
            
            if result.returncode == 0 and result.stdout:
                commit_msg = result.stdout.strip().lower()
                if 'feat' in commit_msg or 'add' in commit_msg:
                    return "Feature"
                elif 'fix' in commit_msg or 'bug' in commit_msg:
                    return "Bugfix"
                elif 'refactor' in commit_msg:
                    return "Refactor"
                elif 'test' in commit_msg:
                    return "Testing"
                elif 'doc' in commit_msg:
                    return "Documentation"
        except:
            pass
        
        return "Development"
    
    def _find_latest_conversation_file(self) -> Optional[Path]:
        """最新の会話履歴ファイルを検索"""
        patterns = [
            r"C:\Users\liuco\.claude\projects\**\*.jsonl",
            r"C:\Users\*\.claude\projects\**\*.jsonl"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob(pattern, recursive=True)
            all_files.extend(files)
        
        if all_files:
            return Path(max(all_files, key=lambda x: os.path.getmtime(x)))
        
        return None
    
    def _generate_auto_description(self, include_git: bool) -> str:
        """自動説明生成"""
        parts = []
        
        if include_git:
            git_status = self._get_git_status()
            if git_status["branch"] != "unknown":
                parts.append(f"branch:{git_status['branch']}")
            
            # ファイル変更状況
            counts = []
            if git_status["modified_files"]:
                counts.append(f"{len(git_status['modified_files'])} files modified")
            if git_status["staged_files"]:
                counts.append(f"{len(git_status['staged_files'])} staged")
            if git_status["untracked_files"]:
                counts.append(f"{len(git_status['untracked_files'])} untracked")
            
            if counts:
                parts.append(", ".join(counts))
        
        if not parts:
            parts.append("Working session")
        
        return "Session on " + ", ".join(parts)
    
    def _create_summary(self, session_data: Dict, summary_file: Path):
        """セッションサマリー作成"""
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# Session State: {session_data['session_id']}\n\n")
                f.write(f"## 基本情報\n")
                f.write(f"- **保存時刻**: {session_data['timestamp']}\n")
                f.write(f"- **モード**: {session_data['mode']}\n")
                f.write(f"- **説明**: {session_data['description']}\n")
                if session_data.get('conversation_saved'):
                    f.write(f"- **会話履歴**: 保存済み (v3.0.0)\n")
                f.write("\n")
                
                f.write(f"## 作業環境\n")
                ctx = session_data['working_context']
                f.write(f"- **作業ディレクトリ**: {ctx['working_directory']}\n")
                f.write(f"- **Python**: {ctx['python_version']}\n")
                f.write("\n")
                
                if 'git_status' in session_data:
                    git = session_data['git_status']
                    f.write(f"## Git状態\n")
                    f.write(f"- **ブランチ**: {git['branch']}\n")
                    f.write(f"- **変更ファイル**: {len(git['modified_files'])}件\n")
                    f.write(f"- **未追跡ファイル**: {len(git['untracked_files'])}件\n")
                    f.write("\n")
                
                if 'claude_context' in session_data:
                    claude = session_data['claude_context']
                    f.write(f"## Claude設定\n")
                    f.write(f"- **Think Hard**: {claude.get('think_hard_status', 'unknown')}\n")
                    f.write("\n")
                
                # 最近のファイル
                if ctx['recent_files']:
                    f.write(f"## 最近の作業ファイル（上位5件）\n")
                    for file_info in ctx['recent_files'][:5]:
                        f.write(f"- {file_info['path']} ({file_info['modified']})\n")
                    f.write("\n")
                
                f.write("---\n")
                f.write(f"*Session State Manager v{VERSION}*\n")
        
        except Exception as e:
            self._log("WARNING", f"Summary creation failed: {e}")
    
    def _update_metadata(self, session_id: str, description: str):
        """メタデータ更新"""
        try:
            metadata = self._load_json(self.metadata_file) or self._init_metadata()
            
            # セッション追加
            metadata["sessions"].insert(0, {
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "description": description
            })
            
            # 最新20件のみ保持
            metadata["sessions"] = metadata["sessions"][:20]
            
            # 統計更新
            metadata["stats"]["total_sessions"] += 1
            metadata["last_updated"] = datetime.now().isoformat()
            
            self._save_json(self.metadata_file, metadata)
        
        except Exception as e:
            self._log("WARNING", f"Metadata update failed: {e}")
    
    def extract_conversation_context(self, session_id: str, level: str = "normal") -> Dict[str, Any]:
        """
        会話履歴から文脈情報を抽出（v3.8.0新機能）
        
        Args:
            session_id: セッションID
            level: 抽出レベル ("quick", "normal", "deep")
        
        Returns:
            レベルに応じた文脈情報
        """
        conv_file = self.states_dir / f"{session_id}_conv.jsonl"
        if not conv_file.exists():
            return None
        
        context = {
            "session_id": session_id,
            "level": level,
            "summary": "",
            "key_points": [],
            "files_changed": [],
            "commands_executed": [],
            "errors_encountered": [],
            "tools_used": defaultdict(int),
            "user_requests": [],
            "recent_messages": []
        }
        
        # 会話履歴を解析
        messages = []
        with open(conv_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    messages.append(data)
                    self._analyze_message(data, context)
                except json.JSONDecodeError:
                    continue
        
        # レベルに応じた要約生成
        if level == "quick":
            # Quick: 最小限の要約（500トークン相当）
            context["summary"] = self._generate_quick_summary(context)
            context["token_estimate"] = 500
        elif level == "normal":
            # Normal: 要約＋主要文脈（2000トークン相当）
            context["summary"] = self._generate_normal_summary(context)
            context["key_points"] = context["user_requests"][:5]
            context["files_changed"] = list(set(context["files_changed"]))[:10]
            context["token_estimate"] = 2000
        elif level == "deep":
            # Deep: 要約＋文脈＋最新メッセージ（10000トークン相当）
            context["summary"] = self._generate_deep_summary(context)
            context["key_points"] = context["user_requests"][:10]
            context["files_changed"] = list(set(context["files_changed"]))
            context["recent_messages"] = messages[-50:] if len(messages) > 50 else messages
            context["token_estimate"] = 10000
        
        return context
    
    def _analyze_message(self, data: Dict, context: Dict):
        """メッセージを分析して文脈情報を抽出（v3.8.0）"""
        msg = data.get('message', {})
        role = msg.get('role', '')
        content = msg.get('content', [])
        
        # ユーザーリクエストの抽出
        if role == 'user' and not data.get('isMeta'):
            user_text = self._extract_text_content(content)
            if user_text and len(user_text) > 10:
                context["user_requests"].append(user_text[:200])
        
        # ツール使用の記録
        elif role == 'assistant':
            for item in content if isinstance(content, list) else []:
                if isinstance(item, dict) and item.get('type') == 'tool_use':
                    tool_name = item.get('name', '')
                    context["tools_used"][tool_name] += 1
                    
                    # ファイル操作の記録
                    tool_input = item.get('input', {})
                    if tool_name in ['Write', 'Edit', 'MultiEdit']:
                        file_path = tool_input.get('file_path', '')
                        if file_path:
                            context["files_changed"].append(file_path)
                    elif tool_name == 'Bash':
                        command = tool_input.get('command', '')
                        if command:
                            context["commands_executed"].append(command[:100])
    
    def _extract_text_content(self, content) -> str:
        """コンテンツからテキストを抽出（v3.8.0）"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'text':
                    texts.append(item.get('text', ''))
                elif isinstance(item, str):
                    texts.append(item)
            return ' '.join(texts)
        return ''
    
    def _generate_quick_summary(self, context: Dict) -> str:
        """クイック要約生成（v3.8.0）"""
        requests = context["user_requests"]
        if requests:
            main_request = requests[0] if requests else "No specific request"
            files_count = len(set(context["files_changed"]))
            tools_count = sum(context["tools_used"].values())
            return f"Main: {main_request[:100]}... | Files: {files_count} | Tools: {tools_count} calls"
        return "Session with minimal activity"
    
    def _generate_normal_summary(self, context: Dict) -> str:
        """通常要約生成（v3.8.0）"""
        summary_parts = []
        
        # 主要リクエスト
        if context["user_requests"]:
            summary_parts.append(f"Requests: {len(context['user_requests'])} tasks")
            summary_parts.append(f"Main: {context['user_requests'][0][:150]}")
        
        # ファイル操作
        if context["files_changed"]:
            summary_parts.append(f"Files modified: {len(set(context['files_changed']))}")
        
        # ツール使用統計
        if context["tools_used"]:
            top_tools = sorted(context["tools_used"].items(), key=lambda x: x[1], reverse=True)[:3]
            tools_str = ", ".join([f"{t[0]}({t[1]})" for t in top_tools])
            summary_parts.append(f"Tools: {tools_str}")
        
        return " | ".join(summary_parts)
    
    def _generate_deep_summary(self, context: Dict) -> str:
        """詳細要約生成（v3.8.0）"""
        summary = self._generate_normal_summary(context)
        
        # エラー情報追加
        if context["errors_encountered"]:
            summary += f" | Errors: {len(context['errors_encountered'])}"
        
        # コマンド実行情報
        if context["commands_executed"]:
            summary += f" | Commands: {len(context['commands_executed'])}"
        
        return summary
    
    def restore_state(self, session_id: Optional[str] = None, 
                     apply_changes: bool = False,
                     restore_level: str = "normal",
                     load_conversation: bool = False,
                     semantic_query: Optional[str] = None) -> bool:
        """
        セッション復元（非対話型 v3.1.0 + セマンティック検索 v3.4.0）
        
        Args:
            session_id: 復元するセッションID（Noneで最新）
            apply_changes: Trueの場合、Git stashなどを実際に適用
            restore_level: 復元レベル ("quick", "normal", "deep")
            load_conversation: 会話履歴を読み込むか（deepレベル時のみ有効）
            semantic_query: セマンティック検索クエリ（v3.4.0新機能）
        
        Returns:
            成功/失敗
        """
        try:
            # セマンティック検索によるセッション選択（v3.4.0新機能）
            if not session_id and semantic_query and self.semantic_search:
                print(f"[INFO] Searching sessions semantically: {semantic_query}")
                results = self.semantic_search_sessions(semantic_query, top_k=1)
                if results:
                    similarity, found_id, info = results[0]
                    if similarity > 0.5:  # 類似度の闾値
                        session_id = found_id
                        print(f"[INFO] Found matching session with similarity {similarity:.3f}: {found_id}")
                        print(f"[INFO] Description: {info['description']}")
                    else:
                        print(f"[WARNING] No highly similar session found (best match: {similarity:.3f})")
                        return False
                else:
                    print(f"[WARNING] No session found for query: {semantic_query}")
                    return False
            
            # アーカイブサイズチェック（v3.2.0新機能）
            self._check_archive_size()
            
            # セッションファイル特定
            if session_id:
                session_file = self.states_dir / f"{session_id}.json"
            else:
                # 最新のセッションを検索
                session_files = sorted(
                    self.states_dir.glob("SESSION_*.json"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                if not session_files:
                    print("[ERROR] No saved sessions found")
                    return False
                session_file = session_files[0]
                session_id = session_file.stem
            
            if not session_file.exists():
                print(f"[ERROR] Session not found: {session_id}")
                return False
            
            # 復元レベルの検証（v3.1.0）
            if restore_level not in ["quick", "normal", "deep"]:
                print(f"[WARNING] Invalid restore level '{restore_level}', using 'normal'")
                restore_level = "normal"
            
            # セッションデータ読み込み
            session_data = self._load_json(session_file)
            if not session_data:
                print(f"[ERROR] Failed to load session data")
                return False
            
            print(f"\n[INFO] Restoring session: {session_id}")
            print(f"[INFO] Restore level: {restore_level}")
            print(f"[INFO] Description: {session_data.get('description', 'No description')}")
            print(f"[INFO] Saved at: {session_data.get('timestamp', 'Unknown')}")
            
            # レベルに応じた復元
            if restore_level in ["normal", "deep"]:
                # 作業コンテキスト表示
                if 'working_context' in session_data:
                    ctx = session_data['working_context']
                    print(f"\n## Working Context")
                    print(f"- Directory: {ctx.get('working_directory', 'Unknown')}")
                    print(f"- Python: {ctx.get('python_version', 'Unknown')}")
                    
                    if ctx.get('recent_files'):
                        print(f"\n## Recent Files (Top 5)")
                        for file_info in ctx['recent_files'][:5]:
                            print(f"  - {file_info['path']} ({file_info['modified']})")
            
            # v3.8.0: 復元レベルごとの文脈要約表示
            conv_file = self.states_dir / f"{session_id}_conv.jsonl"
            if conv_file.exists():
                # 文脈情報を抽出
                context = self.extract_conversation_context(session_id, level=restore_level)
                if context:
                    print(f"\n## Session Context (v3.8.0 - {restore_level} mode)")
                    print(f"Token usage: ~{context['token_estimate']} tokens")
                    print(f"Summary: {context['summary']}")
                    
                    # レベルごとの追加情報
                    if restore_level in ["normal", "deep"]:
                        if context["key_points"]:
                            print(f"\n### Key Tasks:")
                            for i, point in enumerate(context["key_points"][:5], 1):
                                print(f"  {i}. {point[:100]}...")
                        
                        if context["files_changed"]:
                            print(f"\n### Files Modified ({len(context['files_changed'])} total):")
                            for file in context["files_changed"][:10]:
                                print(f"  - {file}")
                    
                    if restore_level == "deep":
                        if context["commands_executed"]:
                            print(f"\n### Commands Executed ({len(context['commands_executed'])} total):")
                            for cmd in context["commands_executed"][:5]:
                                print(f"  $ {cmd}")
                        
                        # 会話履歴の統計
                        line_count = sum(1 for _ in open(conv_file, 'r', encoding='utf-8'))
                        print(f"\n### Full History Stats:")
                        print(f"  - Total messages: {line_count}")
                        print(f"  - Recent messages loaded: {len(context.get('recent_messages', []))}")
                        
                        if load_conversation:
                            print(f"[INFO] Full conversation context loaded ({line_count} messages)")
                        else:
                            print(f"[INFO] Using summarized context only")
            
            # Git状態表示
            if 'git_status' in session_data:
                git = session_data['git_status']
                print(f"\n## Git Status")
                print(f"- Branch: {git.get('branch', 'Unknown')}")
                print(f"- Modified: {len(git.get('modified_files', []))} files")
                print(f"- Untracked: {len(git.get('untracked_files', []))} files")
                
                if git.get('stash_list'):
                    print(f"\n## Available Stashes")
                    for stash in git['stash_list'][:3]:
                        print(f"  - {stash}")
                
                # Stash自動適用（非対話型 v3.1.0）
                if apply_changes and session_data.get('git_stash_created'):
                    print("[INFO] Applying Git stash...")
                    result = subprocess.run(
                        ["git", "stash", "pop"],
                        capture_output=True, text=True, encoding='utf-8', errors='replace',
                        cwd=self.project_root
                    )
                    if result.returncode == 0:
                        print("[OK] Git stash applied")
                    else:
                        print(f"[WARNING] Stash apply failed: {result.stderr}")
            
            print(f"\n[OK] Session restored: {session_id}")
            return True
            
        except Exception as e:
            self._log("ERROR", f"Restore failed: {e}")
            print(f"[ERROR] Session restore failed: {e}")
            return False
    
    def smart_restore(self, auto_detect: bool = True) -> bool:
        """
        コンテキスト認識型の自動復元（v3.5.0新機能）
        
        Args:
            auto_detect: 現在のコンテキストから自動検出
            
        Returns:
            復元成功/失敗
        """
        if not self.semantic_search:
            # セマンティック検索が無効な場合は最新セッションを復元
            return self.restore_state(restore_level="quick")
        
        try:
            # 現在のコンテキストから検索クエリ生成
            context_parts = []
            
            # 現在のディレクトリ
            cwd = os.path.basename(os.getcwd())
            context_parts.append(cwd)
            
            # Gitブランチ
            try:
                result = subprocess.run(
                    ["git", "branch", "--show-current"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                if result.returncode == 0:
                    branch = result.stdout.strip()
                    context_parts.append(branch)
            except:
                pass
            
            # 最近変更されたファイル
            try:
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                if result.returncode == 0 and result.stdout:
                    files = result.stdout.strip().split('\n')[:5]
                    for f in files:
                        if f:
                            filename = os.path.basename(f.split()[-1])
                            context_parts.append(filename)
            except:
                pass
            
            # コンテキストクエリ生成
            context_query = " ".join(context_parts)
            
            if not context_query:
                # コンテキストが取得できない場合は最新セッションを復元
                return self.restore_state(restore_level="quick")
            
            # セマンティック検索
            results = self.semantic_search_sessions(context_query, top_k=3)
            
            if results:
                # 最も類似度の高いセッションを選択
                best_match = results[0]
                similarity = best_match[0]
                session_id = best_match[1]
                info = best_match[2]
                
                # 類似度が閾値以上の場合のみ復元
                if similarity > 0.6:
                    print(f"[AUTO] Found related session (similarity: {similarity:.3f})")
                    print(f"[AUTO] Session: {session_id}")
                    print(f"[AUTO] Description: {info['description']}")
                    
                    # 自動復元
                    return self.restore_state(
                        session_id=session_id,
                        restore_level="quick"
                    )
                else:
                    print(f"[AUTO] No highly similar session found (best: {similarity:.3f})")
                    
                    # 候補を表示
                    if results:
                        print("[AUTO] Available sessions:")
                        for i, (sim, sid, info) in enumerate(results[:3], 1):
                            print(f"  {i}. {info['description'][:50]}... (similarity: {sim:.3f})")
                    
                    return False
            else:
                print("[AUTO] No matching sessions found")
                return False
                
        except Exception as e:
            self._log("ERROR", f"Smart restore failed: {e}")
            # エラー時は通常の復元にフォールバック
            return self.restore_state(restore_level="quick")
    
    def regenerate_all_embeddings(self, force: bool = False) -> int:
        """
        全セッションのエンベディングを再生成（v3.6.0新機能）
        
        Args:
            force: 既存のエンベディングも再生成するか
            
        Returns:
            処理したセッション数
        """
        if not self.semantic_search:
            print("[ERROR] Semantic search is not enabled")
            return 0
        
        print("[INFO] Starting embedding regeneration for all sessions...")
        processed_count = 0
        skipped_count = 0
        error_count = 0
        
        # 全セッションファイルを取得
        session_files = sorted(self.states_dir.glob("SESSION_*.json"))
        total = len(session_files)
        
        if total == 0:
            print("[INFO] No sessions found")
            return 0
        
        print(f"[INFO] Found {total} sessions to process")
        
        for i, session_file in enumerate(session_files, 1):
            try:
                # JSONLファイルはスキップ
                if session_file.stem.endswith("_conv"):
                    continue
                
                # セッションデータを読み込み
                session_data = self._load_json(session_file)
                if not session_data:
                    error_count += 1
                    continue
                
                # セッションテキストを構築
                session_text_parts = []
                
                # Description
                if desc := session_data.get('description'):
                    session_text_parts.append(desc)
                
                # Working context
                if context := session_data.get('working_context', {}):
                    if directory := context.get('directory'):
                        session_text_parts.append(directory)
                    if python_ver := context.get('python_version'):
                        session_text_parts.append(python_ver)
                
                # Git status
                if git_status := session_data.get('git_status', {}):
                    if branch := git_status.get('branch'):
                        session_text_parts.append(branch)
                    if modified := git_status.get('modified'):
                        session_text_parts.extend(modified[:5])  # 最初の5ファイル
                
                # テキストを結合
                session_text = " ".join(session_text_parts)
                
                if not session_text:
                    skipped_count += 1
                    continue
                
                # キャッシュチェック（force=Falseの場合）
                text_hash = hashlib.md5(session_text.encode('utf-8')).hexdigest()
                if not force and text_hash in self.semantic_search.embeddings_cache:
                    skipped_count += 1
                    if i % 10 == 0:
                        print(f"[INFO] Progress: {i}/{total} (skipped: {skipped_count}, processed: {processed_count})")
                    continue
                
                # エンベディング生成
                embedding = self.semantic_search.get_embedding(session_text)
                if embedding is not None:
                    processed_count += 1
                    
                    # 進捗表示（10件ごと）
                    if i % 10 == 0:
                        print(f"[INFO] Progress: {i}/{total} (processed: {processed_count}, skipped: {skipped_count})")
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                self._log("WARNING", f"Failed to process {session_file.name}: {e}")
        
        # キャッシュを保存
        if processed_count > 0:
            self.semantic_search.save_cache()
            print(f"[INFO] Saved embedding cache to disk")
        
        # 結果サマリー
        print(f"\n[COMPLETE] Embedding regeneration finished:")
        print(f"  - Total sessions: {total}")
        print(f"  - Newly processed: {processed_count}")
        print(f"  - Skipped (cached): {skipped_count}")
        print(f"  - Errors: {error_count}")
        print(f"  - Cache size: {len(self.semantic_search.embeddings_cache)} embeddings")
        
        return processed_count
    
    def verify_embeddings(self) -> Dict[str, int]:
        """
        エンベディングの状態を確認（v3.6.0新機能）
        
        Returns:
            統計情報のDict
        """
        if not self.semantic_search:
            return {"error": "Semantic search not enabled"}
        
        # セッション数をカウント
        session_files = list(self.states_dir.glob("SESSION_*.json"))
        session_count = len([f for f in session_files if not f.stem.endswith("_conv")])
        
        # キャッシュ数
        cache_count = len(self.semantic_search.embeddings_cache)
        
        # カバレッジ計算
        coverage = (cache_count / session_count * 100) if session_count > 0 else 0
        
        stats = {
            "total_sessions": session_count,
            "cached_embeddings": cache_count,
            "coverage_percent": round(coverage, 1),
            "cache_file": str(self.semantic_search.cache_file),
            "cache_file_exists": self.semantic_search.cache_file.exists()
        }
        
        print(f"\n[INFO] Embedding Status:")
        print(f"  - Total sessions: {stats['total_sessions']}")
        print(f"  - Cached embeddings: {stats['cached_embeddings']}")
        print(f"  - Coverage: {stats['coverage_percent']}%")
        print(f"  - Cache file: {stats['cache_file']}")
        print(f"  - Cache exists: {stats['cache_file_exists']}")
        
        return stats
    
    def semantic_search_sessions(self, query: str, top_k: int = 5) -> List[Tuple[float, str, Dict]]:
        """
        セマンティック検索でセッションを検索
        
        Args:
            query: 検索クエリ
            top_k: 上位何件を返すか
            
        Returns:
            [(類似度, セッションID, セッション情報), ...] のリスト
        """
        if not self.semantic_search:
            print("[WARNING] Semantic search is not available")
            return []
        
        # 全セッションファイルを取得
        sessions = []
        for session_file in sorted(self.states_dir.glob("SESSION_*.json"), reverse=True):
            if session_file.stem.endswith("_conv"):
                continue
            
            # 要約ファイルが存在する場合は読み込み
            summary_file = session_file.with_suffix('.md')
            session_text = ""
            
            if summary_file.exists():
                try:
                    session_text = summary_file.read_text(encoding='utf-8')[:1000]
                except:
                    pass
            
            # セッションデータも読み込み
            if not session_text:
                try:
                    session_data = self._load_json(session_file)
                    if session_data:
                        # descriptionとworking_contextから情報を取得
                        desc = session_data.get('description', '')
                        context = session_data.get('working_context', {})
                        session_text = f"{desc} {context.get('directory', '')} {context.get('python_version', '')}"
                except:
                    continue
            
            if session_text:
                sessions.append((session_file.stem, session_text))
        
        if not sessions:
            print("[INFO] No sessions found for semantic search")
            return []
        
        # セマンティック検索実行
        results = self.semantic_search.search(query, sessions, top_k)
        
        # 結果を整形
        formatted_results = []
        for similarity, session_id, text in results:
            session_file = self.states_dir / f"{session_id}.json"
            session_data = self._load_json(session_file)
            
            if session_data:
                session_info = {
                    'id': session_id,
                    'description': session_data.get('description', 'No description'),
                    'saved_at': session_data.get('saved_at', 'Unknown'),
                    'similarity': similarity
                }
                formatted_results.append((similarity, session_id, session_info))
        
        return formatted_results
    
    def list_sessions(self, limit: int = 10):
        """保存されたセッション一覧表示"""
        try:
            # アーカイブサイズチェック（v3.2.0新機能）
            self._check_archive_size()
            
            session_files = sorted(
                self.states_dir.glob("SESSION_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not session_files:
                print("No saved sessions found")
                return
            
            print(f"\n## Recent Sessions (Latest {limit})")
            print("-" * 80)
            
            for i, session_file in enumerate(session_files[:limit], 1):
                session_data = self._load_json(session_file)
                if session_data:
                    session_id = session_data.get('session_id', session_file.stem)
                    timestamp = session_data.get('timestamp', 'Unknown')
                    description = session_data.get('description', 'No description')
                    has_conversation = session_data.get('conversation_saved', False)
                    
                    # 会話履歴の有無を表示（v3.0.0）
                    conv_marker = " [CONV]" if has_conversation else ""
                    
                    print(f"{i:2}. {session_id}{conv_marker}")
                    print(f"    Time: {timestamp}")
                    print(f"    Desc: {description}")
                    
                    if 'git_status' in session_data:
                        branch = session_data['git_status'].get('branch', 'unknown')
                        print(f"    Git:  {branch}")
                    print()
            
            print("-" * 80)
            print("[CONV] = 会話履歴あり (v3.0.0)")
            
        except Exception as e:
            print(f"[ERROR] Failed to list sessions: {e}")
    
    def cleanup_old_sessions(self, days: int = 180):
        """古いセッションファイルをクリーンアップ（デフォルト: 180日以上前）"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            removed_count = 0
            
            for session_file in self.states_dir.glob("SESSION_*.json"):
                try:
                    # ファイル名から日付を抽出
                    date_str = session_file.stem.split('_')[1]
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        # 関連ファイルも削除
                        related_files = [
                            session_file,
                            session_file.with_name(f"{session_file.stem}_summary.md"),
                            session_file.with_name(f"{session_file.stem}_conv.jsonl")  # v3.0.0
                        ]
                        
                        for rf in related_files:
                            if rf.exists():
                                rf.unlink()
                                removed_count += 1
                except:
                    continue
            
            print(f"[OK] Removed {removed_count} old session files (>{days} days)")
            
        except Exception as e:
            print(f"[ERROR] Cleanup failed: {e}")


def main():
    """コマンドライン実行用エントリーポイント"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"Session State Manager v{VERSION} - Non-Interactive Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Save current state with conversation:
    python session_state_manager.py save --description "Feature implementation"
  
  Restore latest session interactively:
    python session_state_manager.py restore
  
  Restore specific session:
    python session_state_manager.py restore --session-id SESSION_20250822_123456
  
  List recent sessions:
    python session_state_manager.py list
  
  Clean old sessions:
    python session_state_manager.py cleanup --days 30
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save current session state')
    save_parser.add_argument('--description', '-d', type=str, default="",
                            help='Session description')
    save_parser.add_argument('--no-git', action='store_true',
                            help='Skip Git information')
    save_parser.add_argument('--create-stash', action='store_true',
                            help='Create Git stash')
    save_parser.add_argument('--no-conversation', action='store_true',
                            help='Skip conversation history (v3.0.0)')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore session state')
    restore_parser.add_argument('--session-id', '-s', type=str,
                               help='Session ID to restore (latest if not specified)')
    restore_parser.add_argument('--apply', action='store_true',
                               help='Apply Git changes (stash pop)')
    restore_parser.add_argument('--level', '-l', type=str, 
                               choices=['quick', 'normal', 'deep'],
                               default='normal',
                               help='Restore level: quick(summary only), normal(default), deep(with conversation)')
    restore_parser.add_argument('--load-conversation', action='store_true',
                               help='Load conversation history (only for deep level)')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List saved sessions')
    list_parser.add_argument('--limit', '-l', type=int, default=10,
                            help='Number of sessions to show')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean old sessions')
    cleanup_parser.add_argument('--days', '-d', type=int, default=30,
                               help='Remove sessions older than N days')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete specific session')
    delete_parser.add_argument('session_id', help='Session ID to delete')
    
    # Search command (v3.4.0)
    search_parser = subparsers.add_parser('search', help='Semantic search for sessions')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', '-k', type=int, default=5,
                              help='Number of top results to show')
    
    args = parser.parse_args()
    
    # Manager instance
    manager = SessionStateManager(quiet_mode=False)
    
    if args.command == 'save':
        session_id = manager.save_state(
            description=args.description,
            include_git=not args.no_git,
            create_stash=args.create_stash,
            save_conversation=not args.no_conversation
        )
        if session_id:
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.command == 'restore':
        success = manager.restore_state(
            session_id=args.session_id,
            apply_changes=args.apply,
            restore_level=args.level,
            load_conversation=args.load_conversation
        )
        sys.exit(0 if success else 1)
    
    elif args.command == 'list':
        manager.list_sessions(limit=args.limit)
        sys.exit(0)
    
    elif args.command == 'cleanup':
        manager.cleanup_old_sessions(days=args.days)
        sys.exit(0)
    
    elif args.command == 'delete':
        # 特定セッション削除
        session_file = manager.states_dir / f"{args.session_id}.json"
        if session_file.exists():
            related_files = [
                session_file,
                session_file.with_name(f"{args.session_id}_summary.md"),
                session_file.with_name(f"{args.session_id}_conv.jsonl")
            ]
            removed = 0
            for rf in related_files:
                if rf.exists():
                    rf.unlink()
                    removed += 1
            print(f"[OK] Deleted {removed} files for session: {args.session_id}")
        else:
            print(f"[ERROR] Session not found: {args.session_id}")
            sys.exit(1)
    
    elif args.command == 'search':
        # セマンティック検索（v3.4.0新機能）
        print(f"[INFO] Searching for: {args.query}")
        print("-" * 60)
        
        results = manager.semantic_search_sessions(args.query, args.top_k)
        
        if not results:
            print("[INFO] No matching sessions found")
        else:
            print(f"[INFO] Found {len(results)} matching sessions:\n")
            
            for idx, (similarity, session_id, info) in enumerate(results, 1):
                print(f"{idx}. {session_id}")
                print(f"   Similarity: {similarity:.3f}")
                print(f"   Description: {info['description']}")
                print(f"   Saved at: {info['saved_at']}")
                print()
        
        sys.exit(0)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()