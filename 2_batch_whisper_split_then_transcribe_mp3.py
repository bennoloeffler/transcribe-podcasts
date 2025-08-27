# batch_whisper_split_then_transcribe_mp3.py
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   python batch_whisper_split_then_transcribe_mp3.py /path/to/source_dir /path/to/target_dir
#
# python batch_whisper_split_then_transcribe_mp3.py "/Users/benno/VundS Dropbox/Externe Dateifreigabe/video_projekte_vunds/2025-08-06-ANL-Downloads-Podcast-PdW-ALLE" transcribed_podcasts
#
# Installs:
#   pip install openai tqdm
#   ffmpeg + ffprobe must be available on PATH

import os, sys, pathlib, shutil, json, subprocess, math, tempfile, logging
from tqdm import tqdm
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcription_debug.log'),
        logging.StreamHandler()
    ]
)

MAX_BYTES = 24 * 1024 * 1024   # keep a buffer under 25MB limit
MIN_CHUNK_SEC = 120            # don't create ultra-short chunks
MAX_CHUNK_SEC = 900            # cap chunk length to ~15min for stability

def ffprobe_bitrate_kbps(path: pathlib.Path) -> float:
    """Return average audio bitrate in kbps using ffprobe."""
    logging.info(f"Getting bitrate for {path}")
    cmd = [
        "ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=bit_rate",
        "-of","json", str(path)
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
        data = json.loads(out)
        br = data["streams"][0].get("bit_rate")
        bitrate = float(br)/1000.0 if br else 192.0
        logging.info(f"Bitrate: {bitrate} kbps")
        return bitrate
    except Exception as e:
        logging.error(f"Error getting bitrate: {e}")
        return 192.0  # fallback 192 kbps if unknown

def get_duration_seconds(path: pathlib.Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path)
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        duration = float(out)
        logging.info(f"File duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        return duration
    except Exception as e:
        logging.error(f"Error getting duration: {e}")
        return 0.0

def compute_chunk_seconds(path: pathlib.Path) -> int:
    """Compute safe segment_time (s) to keep segments < MAX_BYTES, based on bitrate and duration."""
    kbps = ffprobe_bitrate_kbps(path)                   # kb/s
    bytes_per_sec = (kbps * 1000) / 8.0                 # B/s
    duration = get_duration_seconds(path)
    
    # target seconds so that size ~= MAX_BYTES * 0.95
    sec = int(math.floor((MAX_BYTES * 0.95) / bytes_per_sec))
    
    # Ensure we create multiple chunks if the file exceeds size limits
    # If the computed chunk size is >= half the file duration, reduce it
    if duration > 0 and sec >= duration / 2:
        # Create at least 2-3 chunks for files that need splitting
        sec = max(MIN_CHUNK_SEC, int(duration / 3))
        logging.info(f"Adjusted chunk size based on duration: {sec} seconds")
    
    final_sec = max(MIN_CHUNK_SEC, min(sec, MAX_CHUNK_SEC))
    logging.info(f"Computed chunk seconds: {final_sec} (raw: {sec}, bytes_per_sec: {bytes_per_sec}, duration: {duration:.2f}s)")
    return final_sec

def segment_audio(src: pathlib.Path, workdir: pathlib.Path) -> list[pathlib.Path]:
    """Split MP3 into N chunks using ffmpeg -f segment. Returns list of chunk paths."""
    logging.info(f"Segmenting audio file: {src}")
    workdir.mkdir(parents=True, exist_ok=True)
    seg_dur = compute_chunk_seconds(src)
    outpat = workdir / (src.stem + "_%03d.mp3")
    cmd = [
        "ffmpeg","-v","error","-y","-i", str(src),
        "-f","segment","-segment_time", str(seg_dur),
        "-c","copy","-reset_timestamps","1",
        str(outpat)
    ]
    logging.info(f"Running ffmpeg command: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        # Use a more robust approach to find chunks - glob all mp3 files and filter
        all_mp3s = list(workdir.glob("*.mp3"))
        # Filter to only files that match our expected pattern
        stem_pattern = src.stem + "_"
        chunks = [f for f in all_mp3s if f.stem.startswith(stem_pattern)]
        chunks = sorted(chunks)
        
        logging.info(f"Created {len(chunks)} chunks: {[c.name for c in chunks]}")
        for chunk in chunks:
            size_mb = chunk.stat().st_size / (1024 * 1024)
            logging.info(f"Chunk {chunk.name}: {size_mb:.2f} MB")
        return chunks
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg segmentation failed: {e}")
        raise

def transcribe_file(client: OpenAI, mp3: pathlib.Path) -> str:
    size_mb = mp3.stat().st_size / (1024 * 1024)
    logging.info(f"Transcribing file: {mp3.name} ({size_mb:.2f} MB)")
    try:
        with open(mp3, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="de",
                response_format="text"
            )
            logging.info(f"Transcription successful for {mp3.name}, length: {len(result)} chars")
            if not result or not result.strip():
                logging.warning(f"Empty transcription result for {mp3.name}")
            return result
    except Exception as e:
        logging.error(f"Transcription failed for {mp3.name}: {e}")
        raise

def process_one(client: OpenAI, src: pathlib.Path, target_dir: pathlib.Path):
    size_mb = src.stat().st_size / (1024 * 1024)
    logging.info(f"Processing file: {src.name} ({size_mb:.2f} MB)")
    logging.info(f"File size check: {src.stat().st_size} bytes vs MAX_BYTES {MAX_BYTES}")
    
    # If file already < limit, try direct; else split
    if src.stat().st_size <= MAX_BYTES:
        logging.info(f"File {src.name} is under size limit, transcribing directly")
        text = transcribe_file(client, src)
        output_file = target_dir / (src.stem + ".txt")
        output_file.write_text(text, encoding="utf-8")
        logging.info(f"Direct transcription complete, output written to {output_file}")
        return

    logging.info(f"File {src.name} exceeds size limit, will split into chunks")
    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        logging.info(f"Using temporary directory: {td_path}")
        chunks = segment_audio(src, td_path)
        texts = []
        for ch in tqdm(chunks, desc=f"  chunks for {src.name}", leave=False, unit="chunk"):
            try:
                chunk_text = transcribe_file(client, ch).strip()
                texts.append(chunk_text)
                logging.info(f"Chunk {ch.name} transcribed successfully, {len(chunk_text)} chars")
            except Exception as e:
                error_msg = f"[Chunk {ch.name} ERROR] {e}"
                texts.append(error_msg)
                logging.error(f"Chunk {ch.name} failed: {e}")
        
        # Merge chunk texts with separators
        merged = []
        for i, t in enumerate(texts, 1):
            merged.append(f"[Teil {i}]\n{t}\n")
        final_text = "\n".join(merged)
        output_file = target_dir / (src.stem + ".txt")
        output_file.write_text(final_text, encoding="utf-8")
        logging.info(f"Chunked transcription complete, output written to {output_file} ({len(final_text)} chars total)")

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_whisper_split_then_transcribe.py <source_dir> <target_dir>")
        sys.exit(1)

    source_dir = pathlib.Path(sys.argv[1]).expanduser()
    target_dir = pathlib.Path(sys.argv[2]).expanduser()

    if not source_dir.exists():
        print(f"Source not found: {source_dir}")
        sys.exit(2)

    # delete & recreate target
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI()
    mp3s = sorted(source_dir.glob("*.mp3"))
    print(f"Found {len(mp3s)} mp3 files in {source_dir}")

    for mp3 in tqdm(mp3s, desc="Transcribing files", unit="file"):
        try:
            process_one(client, mp3, target_dir)
        except subprocess.CalledProcessError as e:
            (target_dir / (mp3.stem + ".error.txt")).write_text(f"ffmpeg error: {e}", encoding="utf-8")
        except Exception as e:
            (target_dir / (mp3.stem + ".error.txt")).write_text(str(e), encoding="utf-8")

if __name__ == "__main__":
    main()
