#!/usr/bin/env python3
"""
Upload test results to HuggingFace in a separate folder
Does NOT overwrite existing checkpoints
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: huggingface_hub not found. Install with: pip install huggingface-hub")
    sys.exit(1)


def upload_to_hf(
    results_dir: str,
    hf_repo: str,
    hf_token: str,
    target_folder: str = "test_results",
    description: str = None,
    dry_run: bool = False
):
    """
    Upload test results to HuggingFace in a separate folder
    
    Args:
        results_dir: Local path to test results directory
        hf_repo: HuggingFace repo identifier (e.g., "user/repo")
        hf_token: HuggingFace API token
        target_folder: Folder name in HF repo to upload to (default: "test_results")
        description: Optional description for commit
        dry_run: If True, show what would be uploaded without actually uploading
    """
    
    if not os.path.exists(results_dir):
        print(f"❌ Error: Results directory not found: {results_dir}")
        return False
    
    # Initialize API
    api = HfApi()
    
    # Create timestamp-based folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_folder = f"{target_folder}/{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"Prepare Upload Test Results to HuggingFace")
    print(f"{'='*70}")
    print(f"Repository: {hf_repo}")
    print(f"Target folder: {upload_folder}")
    print(f"Local directory: {results_dir}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*70}\n")
    
    # Find all files first
    files_to_upload = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, results_dir)
            files_to_upload.append((file_path, rel_path))
    
    print(f"Found {len(files_to_upload)} files to upload:\n")
    for file_path, rel_path in sorted(files_to_upload):
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  {rel_path} ({file_size:.1f} KB)")
    
    if not files_to_upload:
        print(f"\n⚠ No files found in {results_dir}")
        print(f"Make sure test_inference.py completed and created output files!")
        return False
    
    if dry_run:
        print(f"\n{'='*70}")
        print(f"DRY RUN: Would upload {len(files_to_upload)} files")
        print(f"{'='*70}\n")
        return True
    
    # Authenticate
    try:
        print(f"\n{'='*70}")
        print(f"Authenticating with HuggingFace...")
        login(token=hf_token, add_to_git_credential=False)
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Uploading {len(files_to_upload)} files...")
    print(f"{'='*70}\n")
    
    # Upload files
    uploaded_count = 0
    failed_files = []
    
    for file_path, rel_path in sorted(files_to_upload):
        # Create target path in HF repo
        hf_path = f"{upload_folder}/{rel_path}".replace("\\", "/")
        
        try:
            print(f"[{uploaded_count+1}/{len(files_to_upload)}] {rel_path}...", end=" ", flush=True)
            
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=hf_path,
                repo_id=hf_repo,
                token=hf_token,
                commit_message=description or f"Upload test results {timestamp}"
            )
            
            print("✓")
            uploaded_count += 1
            
        except Exception as e:
            print(f"✗ ({str(e)[:50]}...)")
            failed_files.append((rel_path, str(e)))
    
    print(f"\n{'='*70}")
    print(f"Upload Summary")
    print(f"{'='*70}")
    print(f"Total files: {len(files_to_upload)}")
    print(f"Successfully uploaded: {uploaded_count}")
    print(f"Failed: {len(failed_files)}")
    print(f"HuggingFace folder: {hf_repo}/{upload_folder}")
    
    if failed_files:
        print(f"\nFailed files:")
        for file_path, error in failed_files:
            print(f"  ✗ {file_path}: {error[:60]}")
    
    print(f"{'='*70}\n")
    
    if uploaded_count == len(files_to_upload):
        print(f"✓ All files uploaded successfully!")
        return True
    else:
        print(f"⚠ Some files failed to upload ({len(failed_files)} errors)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload test results to HuggingFace"
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./test_results",
        help="Path to test results directory (default: ./test_results)"
    )
    
    parser.add_argument(
        "--hf_repo",
        type=str,
        default="23LebronJames23/MM-UNet",
        help="HuggingFace repo (e.g., user/repo)"
    )
    
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    
    parser.add_argument(
        "--target_folder",
        type=str,
        default="test_results",
        help="Folder name in HF repo to upload to (default: test_results)"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional commit message description"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be uploaded without actually uploading"
    )
    
    args = parser.parse_args()
    
    # Get token from args or environment
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("❌ Error: HuggingFace token not provided!")
        print("   Either:")
        print("   1. Pass --hf_token argument")
        print("   2. Set HF_TOKEN environment variable")
        sys.exit(1)
    
    # Upload
    success = upload_to_hf(
        results_dir=args.results_dir,
        hf_repo=args.hf_repo,
        hf_token=hf_token,
        target_folder=args.target_folder,
        description=args.description,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
