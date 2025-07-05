#!/usr/bin/env python3
"""
ChromaDB Cleanup Utility for StreamlyAI
This script helps clean up old ChromaDB collections that may have embedding dimension mismatches.
"""

import os
import shutil
import json
from pathlib import Path

def cleanup_chroma_collections():
    """Clean up all ChromaDB collections to resolve dimension mismatches"""
    
    print("🧹 StreamlyAI ChromaDB Cleanup Utility")
    print("=" * 50)
    print()
    
    chroma_base_path = "./Chroma"
    
    if not os.path.exists(chroma_base_path):
        print("✅ No ChromaDB directory found. Nothing to clean up.")
        return
    
    # Find all job directories in Chroma
    job_dirs = []
    for item in os.listdir(chroma_base_path):
        item_path = os.path.join(chroma_base_path, item)
        if os.path.isdir(item_path) and len(item) == 36:  # UUID length
            job_dirs.append(item)
    
    if not job_dirs:
        print("✅ No ChromaDB collections found. Nothing to clean up.")
        return
    
    print(f"Found {len(job_dirs)} ChromaDB collections:")
    for job_id in job_dirs:
        print(f"  - {job_id}")
    
    print()
    response = input("Do you want to remove all ChromaDB collections? This will require re-embedding. (y/N): ")
    
    if response.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Backup job data first
    jobs_dir = "./jobs"
    if os.path.exists(jobs_dir):
        print("\n📋 Backing up job data...")
        backup_dir = "./jobs_backup"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Copy JSON files (job metadata)
        for filename in os.listdir(jobs_dir):
            if filename.endswith('.json'):
                src = os.path.join(jobs_dir, filename)
                dst = os.path.join(backup_dir, filename)
                shutil.copy2(src, dst)
                print(f"  ✓ Backed up {filename}")
    
    # Remove ChromaDB collections
    print("\n🗑️  Removing ChromaDB collections...")
    removed_count = 0
    
    for job_id in job_dirs:
        try:
            job_path = os.path.join(chroma_base_path, job_id)
            shutil.rmtree(job_path)
            print(f"  ✓ Removed {job_id}")
            removed_count += 1
        except Exception as e:
            print(f"  ✗ Failed to remove {job_id}: {e}")
    
    print(f"\n✅ Cleanup completed! Removed {removed_count} collections.")
    print("\nNext steps:")
    print("1. Restart your StreamlyAI server")
    print("2. Re-process any videos you want to query")
    print("3. The system will create new ChromaDB collections with the correct embedding dimensions")
    
    if os.path.exists(backup_dir):
        print(f"\n📁 Job metadata backed up to: {backup_dir}")
        print("You can restore job information if needed.")

def check_chroma_status():
    """Check the status of ChromaDB collections"""
    
    print("🔍 ChromaDB Status Check")
    print("=" * 30)
    print()
    
    chroma_base_path = "./Chroma"
    
    if not os.path.exists(chroma_base_path):
        print("❌ ChromaDB directory not found")
        return
    
    job_dirs = []
    for item in os.listdir(chroma_base_path):
        item_path = os.path.join(chroma_base_path, item)
        if os.path.isdir(item_path) and len(item) == 36:
            job_dirs.append(item)
    
    if not job_dirs:
        print("✅ No ChromaDB collections found")
        return
    
    print(f"Found {len(job_dirs)} ChromaDB collections:")
    print()
    
    for job_id in job_dirs:
        job_path = os.path.join(chroma_base_path, job_id)
        size = sum(f.stat().st_size for f in Path(job_path).rglob('*') if f.is_file())
        size_mb = size / (1024 * 1024)
        
        # Check if job metadata exists
        job_file = f"./jobs/{job_id}.json"
        has_metadata = os.path.exists(job_file)
        
        status = "✅" if has_metadata else "⚠️"
        print(f"{status} {job_id} ({size_mb:.1f} MB) - Metadata: {'Yes' if has_metadata else 'No'}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "check":
            check_chroma_status()
        elif sys.argv[1] == "cleanup":
            cleanup_chroma_collections()
        else:
            print("Usage:")
            print("  python cleanup_chroma.py check    - Check ChromaDB status")
            print("  python cleanup_chroma.py cleanup  - Clean up all collections")
    else:
        print("ChromaDB Cleanup Utility")
        print()
        print("Commands:")
        print("  check    - Check ChromaDB status")
        print("  cleanup  - Clean up all collections")
        print()
        print("Example: python cleanup_chroma.py check") 