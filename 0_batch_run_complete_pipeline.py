#!/usr/bin/env python3
# 0_batch_run_complete_pipeline.py
# Complete pipeline runner for podcast transcription and knowledge extraction
# 
# Runs the complete pipeline:
# 1. Extract MP3 from video files (1_batch_extract_mp3_from_video.py)
# 2. Transcribe MP3 to text (2_batch_whisper_split_then_transcribe_mp3.py)  
# 3. Extract entities with parallel processing (4b_batch_parallel_extract_entities_from_augmented_txts.py)
#
# Usage:
#   python 0_batch_run_complete_pipeline.py                    # Use default directories
#   python 0_batch_run_complete_pipeline.py --skip-augmentation # Skip step 3 (augmentation)
#   python 0_batch_run_complete_pipeline.py --help             # Show help

import sys
import subprocess
import pathlib
import os

def print_help():
    """Print usage help."""
    print("""
Complete Podcast Processing Pipeline

Usage:
  python 0_batch_run_complete_pipeline.py [OPTIONS]

Options:
  --skip-augmentation    Skip the augmentation step (3_batch_augment_transcripts.py)
                        Useful if you only want transcription without AI metadata
  --help, -h            Show this help message

Pipeline Steps:
  1. Extract MP3 from videos    (1_batch_extract_mp3_from_video.py)
  2. Transcribe MP3 to text     (2_batch_whisper_split_then_transcribe_mp3.py)
  3. Augment with AI metadata   (3_batch_augment_transcripts.py) [optional]
  4. Extract entities parallel  (4b_batch_parallel_extract_entities_from_augmented_txts.py)

Default Directories:
  data/1_video_source/     -> data/2_mp3_sound_source/
  data/2_mp3_sound_source/ -> data/3_txt_transcribed/
  data/3_txt_transcribed/  -> data/4_augmented/
  data/4_augmented/        -> data/5_entities/

Requirements:
  - ffmpeg on PATH
  - OPENAI_API_KEY environment variable set
  - Python packages: openai, tqdm, scikit-learn, numpy, tenacity

Example:
  export OPENAI_API_KEY="sk-your-key-here"
  python 0_batch_run_complete_pipeline.py
    """)

def check_requirements():
    """Check if required tools, dependencies and environment variables are available."""
    errors = []
    warnings = []
    
    # Check for OPENAI_API_KEY
    if not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY environment variable not set")
        errors.append("   Fix: export OPENAI_API_KEY=\"sk-your-api-key-here\"")
    
    # Check for ffmpeg and ffprobe
    for tool in ["ffmpeg", "ffprobe"]:
        try:
            subprocess.run([tool, "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            errors.append(f"{tool} not found on PATH")
            errors.append(f"   Fix (macOS): brew install ffmpeg")
            errors.append(f"   Fix (Ubuntu/Debian): sudo apt update && sudo apt install ffmpeg")
            errors.append(f"   Fix (Windows): Download from https://ffmpeg.org/download.html and add to PATH")
    
    # Check for required Python packages
    required_packages = [
        ("openai", "OpenAI API client"),
        ("tqdm", "Progress bars"),
        ("sklearn", "Scikit-learn for embeddings"), 
        ("numpy", "Numerical computations"),
        ("tenacity", "Retry logic for API calls")
    ]
    
    missing_packages = []
    for package_name, description in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(f"{package_name} ({description})")
    
    if missing_packages:
        errors.append("Missing Python packages:")
        for package in missing_packages:
            errors.append(f"   - {package}")
        errors.append("   Fix: pip install openai tqdm scikit-learn numpy tenacity")
        errors.append("   Or: pip install -r requirements.txt (if available)")
    
    # Check for required Python scripts
    required_scripts = [
        "1_batch_extract_mp3_from_video.py",
        "2_batch_whisper_split_then_transcribe_mp3.py", 
        "3_batch_augment_transcripts.py",
        "4b_batch_parallel_extract_entities_from_augmented_txts.py"
    ]
    
    for script in required_scripts:
        if not pathlib.Path(script).exists():
            errors.append(f"Required script not found: {script}")
    
    # Check Python version (optional warning)
    import sys
    if sys.version_info < (3, 8):
        warnings.append(f"Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.8+ recommended.")
    
    # Display results
    if warnings:
        print("⚠️  Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
        print()
    
    if errors:
        print("❌ Requirements check failed:")
        for error in errors:
            print(f"   {error}")
        print()
        print("💡 Installation Summary:")
        print("   1. Install ffmpeg: brew install ffmpeg  (macOS) or sudo apt install ffmpeg  (Linux)")
        print("   2. Install Python packages: pip install openai tqdm scikit-learn numpy tenacity")  
        print("   3. Set API key: export OPENAI_API_KEY=\"sk-your-key-here\"")
        sys.exit(1)
    
    print("✅ Requirements check passed")

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print(f"\n🚀 Running: {description}")
    print(f"   Script: {script_name}")
    
    try:
        # Run script with default directories (no arguments)
        result = subprocess.run([sys.executable, script_name], check=True, capture_output=False)
        print(f"✅ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"   Exit code: {e.returncode}")
        return False

def main():
    # Parse arguments
    skip_augmentation = False
    
    for arg in sys.argv[1:]:
        if arg in ["--help", "-h"]:
            print_help()
            sys.exit(0)
        elif arg == "--skip-augmentation":
            skip_augmentation = True
        else:
            print(f"Unknown argument: {arg}")
            print("Use --help for usage information")
            sys.exit(1)
    
    print("🎬 Starting Complete Podcast Processing Pipeline")
    print(f"   Augmentation: {'SKIPPED' if skip_augmentation else 'ENABLED'}")
    
    # Check requirements
    check_requirements()
    
    # Step 1: Extract MP3 from videos
    success = run_script(
        "1_batch_extract_mp3_from_video.py",
        "Extract MP3 audio from video files"
    )
    if not success:
        print("\n❌ Pipeline stopped due to error in step 1")
        sys.exit(1)
    
    # Step 2: Transcribe MP3 to text
    success = run_script(
        "2_batch_whisper_split_then_transcribe_mp3.py", 
        "Transcribe MP3 files to text using Whisper API"
    )
    if not success:
        print("\n❌ Pipeline stopped due to error in step 2")
        sys.exit(1)
    
    # Step 3: Augment with AI metadata (optional)
    if not skip_augmentation:
        success = run_script(
            "3_batch_augment_transcripts.py",
            "Augment transcripts with AI-generated metadata"
        )
        if not success:
            print("\n❌ Pipeline stopped due to error in step 3")
            sys.exit(1)
        
        # Step 4: Extract entities from augmented texts
        success = run_script(
            "4b_batch_parallel_extract_entities_from_augmented_txts.py",
            "Extract entities and relations with parallel processing"
        )
    else:
        print("\n⏭️  Skipping augmentation step as requested")
        print("   Note: Entity extraction requires augmented texts")
        print("   Pipeline will stop here unless augmented files already exist")
        
        # Check if augmented files exist
        augmented_dir = pathlib.Path("data/4_augmented")
        if augmented_dir.exists() and any(augmented_dir.glob("*.augmented.txt")):
            print("   Found existing augmented files, proceeding with entity extraction")
            success = run_script(
                "4b_batch_parallel_extract_entities_from_augmented_txts.py",
                "Extract entities and relations with parallel processing"
            )
        else:
            print("   No augmented files found, skipping entity extraction")
            success = True
    
    if not success:
        print(f"\n❌ Pipeline stopped due to error in final step")
        sys.exit(1)
    
    print("\n🎉 Complete pipeline finished successfully!")
    print("\nOutput locations:")
    print("   MP3 files:        data/2_mp3_sound_source/")
    print("   Transcripts:      data/3_txt_transcribed/")
    if not skip_augmentation:
        print("   Augmented files:  data/4_augmented/")
        print("   Entities:         data/5_entities/")

if __name__ == "__main__":
    main()