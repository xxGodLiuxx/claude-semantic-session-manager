#!/usr/bin/env python3
"""
Basic usage examples for Claude Semantic Session Manager

This script demonstrates the core functionality and API usage.
Run this to verify your installation is working correctly.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import session_manager
sys.path.insert(0, str(Path(__file__).parent.parent))

from session_manager import SessionStateManager

def demo_basic_functionality():
    """Demonstrate basic session management functionality"""
    print("🚀 Claude Semantic Session Manager - Basic Usage Demo")
    print("=" * 60)
    
    # Initialize manager
    print("\n1. Initializing Session Manager...")
    manager = SessionStateManager(quiet_mode=False, enable_semantic=True)
    
    if not manager.semantic_search:
        print("❌ Semantic search not available. Please install dependencies:")
        print("   pip install sentence-transformers")
        return False
    
    print("✅ Session manager initialized successfully!")
    
    # Create some demo sessions
    print("\n2. Creating demo sessions...")
    
    demo_sessions = [
        "Implemented user authentication with JWT tokens",
        "Fixed database connection pooling issues",
        "Added semantic search functionality to the app",
        "Refactored API endpoints for better performance",
        "Setup CI/CD pipeline with GitHub Actions"
    ]
    
    session_ids = []
    for desc in demo_sessions:
        session_id = manager.save_state(description=desc)
        session_ids.append(session_id)
        print(f"   📝 Created: {desc}")
    
    # Test semantic search
    print(f"\n3. Testing semantic search...")
    
    test_queries = [
        "authentication and security",
        "database performance",
        "search functionality"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        results = manager.semantic_search_sessions(query, top_k=2)
        
        if results:
            for i, (similarity, session_id, info) in enumerate(results, 1):
                print(f"   {i}. {info['description']}")
                print(f"      Similarity: {similarity:.3f}")
        else:
            print("   No results found")
    
    # Test session listing
    print(f"\n4. Listing recent sessions...")
    sessions = manager._list_sessions()
    
    print(f"Found {len(sessions)} sessions:")
    for i, session in enumerate(sessions[:3], 1):
        print(f"   {i}. {session['session_id']}")
        print(f"      Description: {session['description']}")
        print(f"      Saved: {session['timestamp']}")
    
    # Test restoration
    if sessions:
        print(f"\n5. Testing session restoration...")
        latest_session = sessions[0]
        print(f"Restoring: {latest_session['session_id']}")
        manager.restore_state(latest_session['session_id'], level='quick')
    
    # Cleanup demo sessions
    print(f"\n6. Cleaning up demo sessions...")
    for session_id in session_ids:
        session_file = manager.session_dir / f"{session_id}.json"
        related_files = [
            manager.session_dir / f"{session_id}.json",
            manager.session_dir / f"{session_id}_summary.md",
            manager.session_dir / f"{session_id}_conv.jsonl"
        ]
        
        for file_path in related_files:
            if file_path.exists():
                file_path.unlink()
    
    print("✅ Demo completed successfully!")
    print("\n💡 Try these commands:")
    print("   python session_manager.py save --description 'My first session'")
    print("   python session_manager.py search 'your search query'")
    print("   python session_manager.py list")
    
    return True

def check_installation():
    """Check if all dependencies are properly installed"""
    print("🔧 Checking Installation...")
    print("-" * 30)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check required packages
    required_packages = [
        'sentence_transformers',
        'numpy',
        'transformers',
        'torch'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install sentence-transformers")
        return False
    
    print("\n✅ All dependencies installed!")
    return True

if __name__ == "__main__":
    print("Claude Semantic Session Manager - Installation Check & Demo")
    print("=" * 70)
    
    # Check installation first
    if not check_installation():
        print("\n🚨 Please install missing dependencies and try again.")
        sys.exit(1)
    
    print("\n")
    
    # Run demo
    success = demo_basic_functionality()
    
    if success:
        print("\n🎉 Everything is working correctly!")
        print("You're ready to use Claude Semantic Session Manager!")
    else:
        print("\n🚨 Demo failed. Check error messages above.")
        sys.exit(1)