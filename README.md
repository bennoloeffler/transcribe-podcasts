# Podcast Transcription and Knowledge Extraction Pipeline
(c) Benno Löffler

A comprehensive Python pipeline for processing German podcasts and videos, from download to intelligent knowledge extraction. Transform audio/video content into structured entity-relationship knowledge graphs using OpenAI's APIs.

## 🚀 Features

- **Audio Extraction**: Extract high-quality MP3 from video files
- **AI Transcription**: OpenAI Whisper API with smart chunking for large files  
- **Content Augmentation**: AI-generated metadata (keywords, segments, summaries)
- **Entity Extraction**: Intelligent identification of people, concepts, methods, tools, and relationships
- **Knowledge Graphs**: Automated creation of structured entity-relationship networks
- **Advanced Deduplication**: Multi-level embedding analysis with clustering for high-quality results

## 📋 Prerequisites

### Required Software
- **Python 3.8+**
- **ffmpeg** and **ffprobe** (must be available on PATH)
- **OpenAI API Key** with access to Whisper and GPT-4o

### Installation

1. **Install Python Dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install openai tqdm scikit-learn numpy tenacity pymupdf python-docx python-pptx markdown
   ```

2. **Install ffmpeg**
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html and add to PATH
   ```

3. **Set OpenAI API Key**
   ```bash
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```

## 📁 Directory Structure

The pipeline processes files through 6 stages with clear directory organization:

```
project/
├── data/
│   ├── 0_pdf_office_etc_source/ # Input: PDF/Office documents (.pdf, .docx, .pptx, .txt, .md)
│   ├── 1_video_source/          # Input: Video files (.mp4, .webm, .mov, etc.)
│   ├── 2_mp3_sound_source/      # Stage 1: AUDIO INPUT from PODCAST + Extracted MP3 audio files from Video
│   ├── 3_txt_transcribed/       # Stage 2: Text from transcription and document extraction
│   ├── 4_augmented/            # Stage 3: AI-augmented files with metadata
│   └── 5_entities/             # Stage 4: Entity files and knowledge graphs
└── scripts...
```

## 🎥 Downloading YouTube Videos

For YouTube content, use `yt-dlp` to download videos:

```bash
# Install yt-dlp
pip install yt-dlp

# Download single video
yt-dlp -o "data/1_video_source/%(title)s [%(id)s].%(ext)s" "https://youtube.com/watch?v=VIDEO_ID"

# Download playlist
yt-dlp -o "data/1_video_source/%(playlist_index)s. %(title)s [%(id)s].%(ext)s" "https://youtube.com/playlist?list=PLAYLIST_ID"

# Download audio-only (high quality)
yt-dlp -f "bestaudio[ext=m4a]" -o "data/2_mp3_sound_source/%(title)s [%(id)s].%(ext)s" "https://youtube.com/watch?v=VIDEO_ID"
```

## 🔧 Usage Guide

### Stage 0: Extract Text from PDF and Office Documents
Extract text from documents and convert to markdown format:

```bash
# Using default directories:
python 0_batch_extract_txt_from_pdf_office_etc.py

# Using custom directories:
python 0_batch_extract_txt_from_pdf_office_etc.py documents/ extracted_texts/
```

**Supported formats**: .pdf, .docx, .pptx, .txt, .md  
**Output**: 
- PDF/DOCX/PPTX: Extracted text as `.pdf.md.txt`, `.docx.md.txt`, `.pptx.md.txt`
- MD files: Copied to `.md.txt`  
- TXT files: Copied directly as `.txt` (collision backup as `.old`)  
**Dependencies**: `pip install pymupdf python-docx python-pptx markdown`

### Stage 1: Extract Audio from Videos
Extract MP3 audio tracks from video files using ffmpeg:

```bash
# Using default directories:
python 1_batch_extract_mp3_from_video.py

# Using custom directories:
python 1_batch_extract_mp3_from_video.py data/1_video_source/ data/2_mp3_sound_source/
```

**Supported formats**: .mp4, .mkv, .webm, .mov, .avi, .flv, .m4v
**Output**: High-quality MP3 files (CBR 192kbps, 44.1kHz, stereo)

### Stage 2: Transcribe Audio to Text
Transcribe MP3 files using OpenAI's Whisper API:

```bash
export OPENAI_API_KEY="sk-your-api-key"

# Using default directories:
python 2_batch_whisper_split_then_transcribe_mp3.py

# Using custom directories:
python 2_batch_whisper_split_then_transcribe_mp3.py data/2_mp3_sound_source/ data/3_txt_transcribed/
```

**Features**:
- Automatic chunking for files >24MB (API limit)
- Duration-aware segmentation (2-15 minute chunks)
- German language optimization
- Smart part merging with `[Teil N]` markers
- Comprehensive error logging

### Stage 3: Augment with AI Metadata
Add AI-generated metadata and clean formatting:

```bash
export OPENAI_API_KEY="sk-your-api-key"

# Using default directories:
python 3_batch_augment_transcripts.py

# Using custom directories:
python 3_batch_augment_transcripts.py data/3_txt_transcribed/ data/4_augmented/
```

**Generated metadata**:
- Keywords and key phrases
- Content segments and structure
- Practical examples mentioned
- Executive summary
- 80-character formatted text

### Stage 4a: Basic Entity Extraction
Extract entities and relationships with standard approach:

```bash
export OPENAI_API_KEY="sk-your-api-key"

# Using default directories:
python 4a_batch_extract_entities_from_augmented_txts.py

# Using custom directories:
python 4a_batch_extract_entities_from_augmented_txts.py data/4_augmented/ data/5_entities/
```

### Stage 4b: Advanced Parallel Extraction (Recommended)
Extract entities using advanced parallel processing and multi-level analysis:

```bash
export OPENAI_API_KEY="sk-your-api-key"

# Default extraction: data/4_augmented/ -> data/5_entities/
python 4b_batch_parallel_extract_entities_from_augmented_txts.py

# Custom extraction: <source> -> <target>
python 4b_batch_parallel_extract_entities_from_augmented_txts.py custom_source/ custom_target/
```

**Advanced features**:
- 3 parallel LLM calls per text chunk
- Multi-level embedding analysis (names, descriptions, combined)
- Graph-based entity clustering
- LLM-assisted intelligent merging
- 9 entity types: PERSON, ORGANIZATION, CONCEPT, METHOD, ZUSAMMENHANG, HYPOTHESE, FRAGE, INTERVENTION, WERKZEUG, UNKNOWN

## 📊 Expected Results

### Individual Files
Each processed file generates corresponding outputs:

```
input: "document.pdf" or "podcast-episode.mp4"
├── data/3_txt_transcribed/document.pdf.md.txt (from PDF)
├── data/2_mp3_sound_source/podcast-episode.mp3 (from video)
├── data/3_txt_transcribed/podcast-episode.txt (from audio transcription)
├── data/4_augmented/document.pdf.md.augmented.txt
└── data/5_entities/document.pdf.md.augmented.entities.txt
```

### Consolidated Knowledge Graph
Final consolidated files containing merged entities across all processed content:

- **4a output**: `__ALL_ENTITIES_ALL_RELATIONS.txt` 
- **4b output**: `__ALL_ENTITIES_ALL_RELATIONS_PARALLEL.txt`

### Entity File Format
```
ENTITIES
Benno: Moderator des Podcasts über Organisationsentwicklung. FOLGEN: 001, 002, 015
Selbstorganisation: Organisationsmethode bei der Teams autonom arbeiten. FOLGEN: 008, 017, 031

RELATIONS  
Benno --> Selbstorganisation: SHORT: Diskutiert LONG: Benno diskutiert regelmäßig die Vor- und Nachteile von Selbstorganisation
```

## 🎯 Processing Modes

The 4b script offers three simple usage modes:

```bash
# Mode 1: Full extraction (default directories)
python 4b_batch_parallel_extract_entities_from_augmented_txts.py

# Mode 2: Full extraction (custom directories)
python 4b_batch_parallel_extract_entities_from_augmented_txts.py source_folder/ target_folder/

# Mode 3: Consolidate only - merge existing *.entities.txt files
python 4b_batch_parallel_extract_entities_from_augmented_txts.py data/5_entities/
```

**Mode details:**
- **0 arguments**: Extract from `data/4_augmented/` → `data/5_entities/`
- **2 arguments**: Extract from `<source>` → `<target>` 
- **1 argument**: Consolidate existing `*.entities.txt` files in `<target>` folder only

## 🚀 Complete Pipeline Automation

Run the entire pipeline with a single command:

```bash
export OPENAI_API_KEY="sk-your-key-here"

# Complete processing (PDFs + Videos + Transcription + AI Analysis)
python 0__batch_run_complete_pipeline.py
```

**Pipeline includes:**
- Dependency checking with installation guidance
- Automatic error handling and progress reporting  
- Complete workflow from source documents to knowledge graphs

## ⚠️ Important Notes

### API Costs
- **Whisper API**: ~$0.006 per minute of audio
- **GPT-4o**: ~$2.50-$10.00 per 1M tokens (varies by content complexity)
- **Embeddings**: ~$0.13 per 1M tokens
- **Stage 4b uses significantly more API calls** due to parallel processing and embedding analysis

### File Size Limits  
- Max file size for direct transcription: 24MB
- Files >24MB are automatically chunked
- Max tokens per AI request: 12,000 (auto-split if larger)
- Minimum chunk duration: 2 minutes
- Maximum chunk duration: 15 minutes

### Language Support
- **Optimized for German content** (podcast transcripts)
- Whisper API configured with `language="de"`
- Entity extraction prompts in German
- Can be adapted for other languages by modifying prompts

## 🔧 Troubleshooting

### Common Issues

**"ffmpeg not found"**
- Install ffmpeg and ensure it's on your PATH
- Test with: `ffmpeg -version`

**"Empty transcription files"**
- Check your OpenAI API key is valid
- Verify audio file isn't corrupted
- Check `transcription_debug.log` for details

**"API rate limit exceeded"**
- The scripts include automatic retry logic
- For heavy usage, consider API rate limit increases

**"Entity extraction produces duplicates"**  
- Use 4b (parallel) script for better deduplication
- Check entity similarity thresholds in the code
- Review `parallel_entity_extraction.log` for details

### Log Files
- `transcription_debug.log` - Audio transcription issues
- `augmentation_debug.log` - Metadata generation issues  
- `entity_extraction_debug.log` - Entity extraction issues (4a)
- `parallel_entity_extraction.log` - Parallel extraction issues (4b)

## 📈 Performance Tips

1. **Use 4b script** for highest quality entity extraction
2. **Process in batches** to monitor API costs
3. **Check logs regularly** for processing issues
4. **Use consolidate-only mode** to re-merge existing entities without re-extraction
5. **Monitor token usage** in OpenAI dashboard

## 🤝 Contributing

This pipeline was developed for German organizational development podcast content but can be adapted for other domains and languages by:

- Modifying entity type patterns in the scripts
- Adapting prompts for your content domain  
- Adjusting similarity thresholds for your use case
- Adding new entity types as needed

## 📄 License

do What The Fuck you want to Public License

Version 1.0, March 2000
Copyright (C) 2000 Banlu Kemiyatorn (]d).
136 Nives 7 Jangwattana 14 Laksi Bangkok
Everyone is permitted to copy and distribute verbatim copies
of this license document, but changing it is not allowed.

Ok, the purpose of this license is simple
and you just

DO WHAT THE FUCK YOU WANT TO.