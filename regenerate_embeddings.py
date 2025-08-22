#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate embeddings for all sessions
Use this script for initial setup or to rebuild the embedding cache

Usage:
    python regenerate_embeddings.py          # Process new sessions only
    python regenerate_embeddings.py --force  # Regenerate all embeddings
    python regenerate_embeddings.py --verify # Check embedding status
"""

import sys
import argparse
from pathlib import Path

# Import the session manager
from session_manager import SessionStateManager

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate embeddings for all sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--force', 
        action='store_true',
        help='Force regenerate all embeddings (ignore cache)'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify embedding status only'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Project root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Initialize Session State Manager
    print("[INFO] Initializing Session State Manager...")
    manager = SessionStateManager(
        project_root=args.project_root,
        quiet_mode=True, 
        enable_semantic=True
    )
    
    if not manager.semantic_search:
        print("[ERROR] Failed to initialize semantic search engine")
        print("[INFO] Please install required dependencies:")
        print("  pip install sentence-transformers")
        sys.exit(1)
    
    if args.verify:
        # Check status only
        stats = manager.verify_embeddings()
        
        if stats.get('coverage_percent', 0) < 100:
            print(f"\n[RECOMMENDATION] Run this script without --verify to generate missing embeddings")
            print(f"  python {__file__}")
        else:
            print(f"\n[OK] All sessions have embeddings!")
        
    else:
        # Regenerate embeddings
        print("\n" + "="*60)
        print("EMBEDDING REGENERATION FOR ALL SESSIONS")
        print("="*60)
        
        if args.force:
            print("[MODE] Force regeneration (all embeddings will be recreated)")
        else:
            print("[MODE] Smart regeneration (skip existing embeddings)")
        
        # Execute
        processed = manager.regenerate_all_embeddings(force=args.force)
        
        if processed > 0:
            print(f"\n[SUCCESS] Generated embeddings for {processed} sessions")
            print("[INFO] All sessions are now searchable with semantic search!")
            
            # Test search
            print("\n[TEST] Testing semantic search...")
            test_query = "session manager"
            results = manager.semantic_search_sessions(test_query, top_k=3)
            
            if results:
                print(f"[OK] Search for '{test_query}' returned {len(results)} results")
                for i, (similarity, session_id, info) in enumerate(results, 1):
                    desc = info['description'][:50] + "..." if len(info['description']) > 50 else info['description']
                    print(f"  {i}. {desc} (similarity: {similarity:.3f})")
            else:
                print(f"[WARNING] No results found for test query")
        else:
            print(f"\n[INFO] No new embeddings needed (all sessions already have embeddings)")
        
        # Final status check
        print("\n" + "-"*60)
        manager.verify_embeddings()
    
    print("\n[COMPLETE] Embedding regeneration script finished")

if __name__ == "__main__":
    main()