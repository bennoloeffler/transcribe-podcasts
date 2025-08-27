# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a German podcast/video transcription and knowledge extraction pipeline with five main Python scripts:

- `1_batch_extract_mp3_from_video.py`: Extracts MP3 audio tracks from video files using ffmpeg
- `2_batch_whisper_split_then_transcribe_mp3.py`: Transcribes MP3 files using OpenAI's Whisper API with intelligent chunking for large files
- `3_batch_augment_transcripts.py`: Augments transcript files with AI-generated JSON metadata (keywords, segments, examples, summary)
- `4a_batch_extract_entities_from_augmented_txts.py`: Extracts entities and relations from transcripts with AI-powered similarity merging
- `4b_batch_parallel_extract_entities_from_augmented_txts.py`: Advanced parallel entity extraction using 3 concurrent LLM calls with hybrid deduplication

## Key Dependencies

- **ffmpeg/ffprobe**: Required on PATH for audio/video processing
- **OpenAI Python SDK** (`openai`): For Whisper transcription API calls
- **tqdm**: Progress bars during processing
- **OPENAI_API_KEY**: Environment variable required for transcription

## Directory Structure

The pipeline follows a 3-stage processing flow with optional 4th stage:

- `data_01_video_source/`: Source video files (.webm, .mp4, etc.)
- `data_02_mp3_sound_source/`: Extracted MP3 audio files and podcast sources  
- `data_03_txt_transcribed/`: Final transcription text files
- `data_04_augmented/`: (Optional) Augmented files with JSON metadata and cleaned text
- `data_05_entities/`: (Optional) Entity/relation files and consolidated knowledge graph

## Common Commands

### Extract MP3 from video files:
```bash
python 1_batch_extract_mp3_from_video.py data_01_video_source/ data_02_mp3_sound_source/
```

### Transcribe MP3 files:
```bash
export OPENAI_API_KEY="sk-..."
python 2_batch_whisper_split_then_transcribe_mp3.py data_02_mp3_sound_source/ data_03_txt_transcribed/
```

### Augment transcripts with AI metadata:
```bash
export OPENAI_API_KEY="sk-..."
python 3_batch_augment_transcripts.py data_03_txt_transcribed/ data_04_augmented/
```

### Extract entities and relations (standard approach):
```bash
export OPENAI_API_KEY="sk-..."
python 4a_batch_extract_entities_from_augmented_txts.py data_04_augmented/ data_05_entities/
```

### Extract entities and relations (parallel approach - recommended):
```bash
export OPENAI_API_KEY="sk-..."
python 4b_batch_parallel_extract_entities_from_augmented_txts.py data_04_augmented/ data_05_entities/
```

### Consolidate existing entity files only:
```bash
# For 4a (standard):
echo "c" | python 4a_batch_extract_entities_from_augmented_txts.py data_04_augmented/ data_05_entities/

# For 4b (parallel):
echo "c" | python 4b_batch_parallel_extract_entities_from_augmented_txts.py data_04_augmented/ data_05_entities/
```

## Architecture Notes

### Audio Processing (Steps 1-2):
- **Smart chunking**: Large MP3 files are automatically split based on bitrate analysis to stay under OpenAI's 25MB API limit
- **Duration-aware segmentation**: Files are split considering both size and duration to ensure effective chunking
- **Robust filename handling**: Special characters in filenames (brackets, spaces) are handled correctly
- **Error handling**: Failed operations create `.error.txt` files with diagnostic information
- **German language**: Transcription is configured for German language content (`language="de"`)
- **Chunk management**: Files larger than API limits are segmented with ffmpeg, transcribed in parts, then merged with part markers (`[Teil N]`)

### Text Processing (Steps 3-4):
- **AI-powered metadata extraction**: Generates keywords, segments, examples, and summaries in German
- **Text cleaning and formatting**: Removes chunking markers, formats with 80-character line breaks
- **Token-aware processing**: Splits large texts to stay within OpenAI API limits
- **JSON validation**: Ensures proper JSON structure with error handling and fallbacks

### Entity Extraction (Step 4a - Standard):
- **Entity extraction & relation mapping**: Creates knowledge graphs from transcript content
- **Vector similarity analysis**: Uses embeddings to detect and merge duplicate entities
- **AI-powered merge decisions**: GPT-4o determines whether similar entities should be merged
- **Multi-phase merging**: Name-based grouping → exact matches → similarity analysis → AI decisions
- **FOLGEN reference tracking**: Smart formatting based on source count (full names ≤10, numbers 11-50, count >50)

### Entity Extraction (Step 4b - Parallel, Recommended):
- **3-parallel LLM calls**: Each text chunk processed by 3 concurrent LLM instances using `asyncio.gather()`
- **Multi-level embedding analysis**: Creates separate embeddings for entity names, descriptions, and name+description combinations
- **Advanced clustering**: Groups similar entities using graph-based clustering from similarity relationships
- **Type-aware similarity thresholds**: Different similarity thresholds for different entity types (PERSON: 0.92, CONCEPT: 0.78, etc.)
- **LLM-assisted cluster merging**: Intelligent merging of entity clusters with comprehensive descriptions
- **Extended entity types**: PERSON, ORGANIZATION, CONCEPT, METHOD, ZUSAMMENHANG, HYPOTHESE, FRAGE, INTERVENTION, WERKZEUG, UNKNOWN
- **Improved completeness**: Multiple runs catch entities that single runs might miss
- **Quality validation**: Entities appearing in 2-3 runs are more reliable
- **Exception handling**: Individual extraction failures don't crash the entire process

### General:
- **Progress tracking**: All scripts use tqdm for real-time progress monitoring
- **Comprehensive logging**: Debug logging available via multiple `.log` files
- **Retry logic with exponential backoff**: Handles temporary API failures gracefully
- **Interactive mode selection**: Both entity extraction scripts offer full extraction vs consolidate-only modes

## File Processing Limits

- Max file size for direct transcription: 24MB (safety buffer under 25MB API limit)
- Minimum chunk duration: 2 minutes (120 seconds)
- Maximum chunk duration: 15 minutes (900 seconds)
- Default audio settings: CBR 192kbps, 44.1kHz, stereo
- Augmented file format: JSON metadata + "----" separator + cleaned text
- Max tokens per AI request: 12,000 (texts are automatically split if larger)
- Entity similarity thresholds (4b): Type-aware (PERSON: 0.92, ORGANIZATION: 0.88, CONCEPT: 0.78, METHOD: 0.83, ZUSAMMENHANG: 0.80, HYPOTHESE: 0.75, FRAGE: 0.82, INTERVENTION: 0.85, WERKZEUG: 0.87)
- Retry attempts: 3 retries with exponential backoff (4-10 seconds)
- Parallel extractions: 3 concurrent runs per text chunk (4b only)
- Multi-level embeddings: Name, description, and name+description vectors (4b only)
- Entity types: PERSON, ORGANIZATION, CONCEPT, METHOD, ZUSAMMENHANG, HYPOTHESE, FRAGE, INTERVENTION, WERKZEUG, UNKNOWN

## Output File Formats

### Entity Files (.entities.txt):
```
ENTITIES
EntityName: Description with details. FOLGEN: source1, source2
AnotherEntity: Another description. FOLGEN: sourceN

RELATIONS
EntityA --> EntityB: SHORT: Brief relation LONG: Detailed explanation
EntityC --> EntityA: SHORT: Another relation LONG: More details
```

### Consolidated Files:
- `__ALL_ENTITIES_ALL_RELATIONS.txt` (4a - standard approach)
- `__ALL_ENTITIES_ALL_RELATIONS_PARALLEL.txt` (4b - parallel approach)

## Known Issues Fixed

- **Text chunking bug**: Large texts showing "chunk 1/1" instead of proper splitting - fixed chunk size calculation
- **Glob pattern failures** with special characters in filenames - resolved with robust file discovery
- **Oversized chunks** for long files - resolved with duration-aware chunk calculation
- **Silent failures** - resolved with comprehensive logging and error reporting
- **Inconsistent entity formatting** - resolved with improved prompts and dash-removal parsing
- **API failure handling** - resolved with retry logic and detailed error diagnostics
- **Entity parsing failures** - resolved with robust parsing logic and debugging output
- **Critical entity merging failure**: Identical entities (e.g., "Benno", "Fabian") not merging - fixed normalization and name-based merging
- **Entity deduplication within single LLM response**: Same entity extracted multiple times per response - added explicit deduplication instructions to prompts