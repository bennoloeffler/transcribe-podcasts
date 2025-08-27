#!/usr/bin/env python3
# batch_extract_mp3_from_video.py
# Extract MP3 audio tracks from video files in source_dir into target_dir.
# Usage:
#   python batch_extract_mp3_from_video.py <source_dir> <target_dir>

import sys, pathlib, subprocess, shutil
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".m4v"}

def ensure_ffmpeg():
    if shutil.which("ffmpeg") is None:
        sys.stderr.write("ERROR: ffmpeg not found on PATH.\n")
        sys.exit(3)

def run_ffmpeg_to_mp3(src: pathlib.Path, dst: pathlib.Path):
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
        print(f"Source not found: {source_dir}")
        sys.exit(2)

    ensure_ffmpeg()

    files = sorted([p for p in source_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS])

    if not files:
        print(f"No video files found in {source_dir}. Supported: {', '.join(sorted(VIDEO_EXTS))}")
        sys.exit(0)

    for src in tqdm(files, desc="Extracting MP3", unit="file"):
        dst = target_dir / (src.stem + ".mp3")
        try:
            run_ffmpeg_to_mp3(src, dst)
        except subprocess.CalledProcessError as e:
            (target_dir / (src.stem + ".error.txt")).write_text(f"ffmpeg error: {e}\n")

    print(f"Done. MP3s in: {target_dir}")

if __name__ == "__main__":
    main()
