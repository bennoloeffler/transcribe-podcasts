#!/usr/bin/env python3
# batch_extract_mp3_from_video.py
# Extract MP3 audio tracks from video files in source_dir into target_dir.
# Usage:
#   python batch_extract_mp3_from_video.py <source_dir> <target_dir>

import sys, pathlib, subprocess, shutil, logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mp3_extraction_debug.log'),
        logging.StreamHandler()
    ]
)

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v"}

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.stderr.write("ERROR: ffmpeg not found on PATH.\n")
        sys.exit(3)

def run_ffmpeg_to_mp3(src: pathlib.Path, dst: pathlib.Path):
    logging.info(f"Creating target directory: {dst.parent}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -vn = drop video; encode stable mp3 (CBR 192k, 44.1kHz, stereo)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-i", str(src),
        "-vn", "-acodec", "libmp3lame", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(dst),
    ]
    subprocess.check_call(cmd)

def main():
    start_time = datetime.now()
    logging.info(f"=== MP3 Extraction Started at {start_time} ===")
    
    if len(sys.argv) == 1:
        # Use default directories
        source_dir = pathlib.Path("data/1_video_source").expanduser()
        target_dir = pathlib.Path("data/2_mp3_sound_source").expanduser()
        print(f"Using default directories: {source_dir} -> {target_dir}")
    elif len(sys.argv) == 3:
        source_dir = pathlib.Path(sys.argv[1]).expanduser()
        target_dir = pathlib.Path(sys.argv[2]).expanduser()
    else:
        print("Usage: python 1_batch_extract_mp3_from_video.py [<source_dir> <target_dir>]")
        print("  No arguments: uses default directories data/1_video_source/ -> data/2_mp3_sound_source/")
        print("  Two arguments: uses provided directories")
        sys.exit(1)

    if not source_dir.exists():
        logging.error(f"Source directory not found: {source_dir}")
        print(f"Source not found: {source_dir}")
        sys.exit(2)

    # Check target directory and log existing files
    logging.info(f"Source directory: {source_dir}")
    logging.info(f"Target directory: {target_dir}")
    
    if target_dir.exists():
        existing_files = list(target_dir.glob("*"))
        logging.info(f"Target directory exists with {len(existing_files)} existing files:")
        for file in existing_files[:10]:  # Log first 10 files to avoid spam
            logging.info(f"  Existing: {file.name} ({file.stat().st_size} bytes)")
        if len(existing_files) > 10:
            logging.info(f"  ... and {len(existing_files) - 10} more files")
    else:
        logging.info(f"Target directory does not exist, will be created during processing")

    ensure_ffmpeg()

    files = sorted([p for p in source_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

    if not files:
        print(f"No video files found in {source_dir}. Supported: {', '.join(sorted(VIDEO_EXTS))}")
        sys.exit(0)

    for src in tqdm(files, desc="Extracting MP3", unit="file"):
        dst = target_dir / (src.stem + ".mp3")
        logging.info(f"Processing: {src.name} -> {dst.name}")
        try:
            run_ffmpeg_to_mp3(src, dst)
            if dst.exists():
                size_mb = dst.stat().st_size / (1024 * 1024)
                logging.info(f"Successfully created MP3: {dst.name} ({size_mb:.2f} MB)")
            else:
                logging.error(f"MP3 file was not created: {dst}")
        except subprocess.CalledProcessError as e:
            error_file = target_dir / (src.stem + ".error.txt")
            error_file.write_text(f"ffmpeg error: {e}\n")
            logging.error(f"FFmpeg error for {src.name}: {e}")

    # Log final results
    if target_dir.exists():
        final_files = list(target_dir.glob("*.mp3"))
        error_files = list(target_dir.glob("*.error.txt"))
        logging.info(f"MP3 extraction complete. Created {len(final_files)} MP3 files, {len(error_files)} errors")
        for mp3 in final_files:
            size_mb = mp3.stat().st_size / (1024 * 1024)
            logging.info(f"  Created: {mp3.name} ({size_mb:.2f} MB)")
    
    print(f"Done. MP3s in: {target_dir}")
    logging.info(f"MP3 extraction process completed at {datetime.now()}")

if __name__ == "__main__":
    main()
