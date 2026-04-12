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
    description: str = None
):
    """
    Upload test results to HuggingFace in a separate folder
    
    Args:
        results_dir: Local path to test results directory
        hf_repo: HuggingFace repo identifier (e.g., "user/repo")
        hf_token: HuggingFace API token
        target_folder: Folder name in HF repo to upload to (default: "test_results")
        description: Optional description for commit
    """
    
    if not os.path.exists(results_dir):
        print(f"❌ Error: Results directory not found: {results_dir}")
        return False
    
    # Initialize API
    api = HfApi()
    
    try:
        # Authenticate
        print(f"Authenticating with HuggingFace...")
        login(token=hf_token, add_to_git_credential=False)
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Create timestamp-based folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_folder = f"{target_folder}/{timestamp}"
    
    print(f"\n{'='*70}")
    print(f"Uploading test results to HuggingFace")
    print(f"{'='*70}")
    print(f"Repository: {hf_repo}")
    print(f"Target folder: {upload_folder}")
    print(f"Local directory: {results_dir}")
    print(f"{'='*70}\n")
    
    # Walk through results directory
    uploaded_count = 0
    total_files = 0
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            total_files += 1
            file_path = os.path.join(root, file)
            
            # Calculate relative path
            rel_path = os.path.relpath(file_path, results_dir)
            
            # Create target path in HF repo
            hf_path = f"{upload_folder}/{rel_path}".replace("\\", "/")
            
            try:
                print(f"Uploading: {rel_path}...", end=" ")
                
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
                print(f"✗ Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"Upload Summary")
    print(f"{'='*70}")
    print(f"Total files: {total_files}")
    print(f"Successfully uploaded: {uploaded_count}")
    print(f"HuggingFace folder: {hf_repo}/{upload_folder}")
    print(f"{'='*70}\n")
    
    if uploaded_count == total_files:
        print(f"✓ All files uploaded successfully!")
        return True
    else:
        print(f"⚠ Some files failed to upload ({total_files - uploaded_count} errors)")
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
        description=args.description
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
