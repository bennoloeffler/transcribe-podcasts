#!/usr/bin/env python3
# batch_augment_transcripts.py
# Augments transcript files with AI-generated JSON metadata and cleaned text
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   python batch_augment_transcripts.py <source_folder> <target_folder>

import os
import sys
import json
import pathlib
import re
import textwrap
import logging
from typing import Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('augmentation_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
MAX_TOKENS_PER_REQUEST = 12000  # Conservative limit for GPT models
LINE_WIDTH = 80  # Target line width for text formatting

EXTRACTION_PROMPT = """Bitte fasse den folgenden Podcast-Transkript-Text in diesem JSON-Format zusammen:

{
  "keywords": ["Schlüsselwort1", "Schlüsselwort2", "..."],
  "segments": [
    {"Titel des ersten Abschnitts": "Zusammenfassung des Hauptinhalts"},
    {"Titel des zweiten Abschnitts": "Zusammenfassung des Hauptinhalts"},
    "..."
  ],
  "examples_herding": ["Konkrete Beispiele oder Erklärungen von Fabian über Herding", "Weitere Beispiele", "..."],
  "summary": "Eine kurze Zusammenfassung des gesamten Dokuments. Stichpunkte und kurze Einsichten für den Leser."
}

WICHTIGE HINWEISE:
- Verwende NUR DEUTSCHE SPRACHE für alle Ergebnisse
- Dies ist ein Podcast-Transkript von Benno und Fabian
- Der Podcast heißt "Primat Der Wertschöpfung"
- Fabian ist CTO bei der Firma Herding in Amberg
- Fabian und Benno sprechen über "Organisations-Entwicklung" und "moderne Soziologie zur Beschreibung und dem Verständnis von Mustern und Mechanismen in Organisationen"
- IGNORIERE die Segmentierung [Teil 1], [Teil 2], etc. - das sind KEINE logischen Absätze, sondern kommen vom Chunking der Transkription

Extrahiere das JSON aus diesem Text:

"""

def clean_text(text: str) -> str:
    """Remove chunking markers and clean up the text."""
    logger.info("Cleaning text: removing chunk markers and formatting")
    
    # Remove [Teil X] markers
    text = re.sub(r'\[Teil \d+\]\s*', '', text)
    
    # Remove excessive whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Replace multiple newlines with double
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
    text = text.strip()
    
    return text

def format_text_with_line_breaks(text: str, width: int = LINE_WIDTH) -> str:
    """Format text with natural line breaks at specified width."""
    logger.info(f"Formatting text with line breaks at {width} characters")
    
    paragraphs = text.split('\n\n')
    formatted_paragraphs = []
    
    for paragraph in paragraphs:
        if paragraph.strip():
            # Wrap each paragraph
            wrapped = textwrap.fill(
                paragraph.strip(),
                width=width,
                break_long_words=False,
                break_on_hyphens=False
            )
            formatted_paragraphs.append(wrapped)
    
    return '\n\n'.join(formatted_paragraphs)

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count (1 token ≈ 4 characters)."""
    return len(text) // 4

def split_text_for_processing(text: str, max_tokens: int = MAX_TOKENS_PER_REQUEST) -> List[str]:
    """Split text into chunks that fit within token limits."""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    logger.info(f"Text too large ({estimate_tokens(text)} tokens), splitting into chunks")
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        test_chunk = current_chunk + '\n\n' + paragraph if current_chunk else paragraph
        
        if estimate_tokens(test_chunk) <= max_tokens:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                # Single paragraph is too large, split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                for sentence in sentences:
                    test_chunk = current_chunk + ' ' + sentence if current_chunk else sentence
                    if estimate_tokens(test_chunk) <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def extract_json_from_text(client: OpenAI, text: str) -> Optional[Dict]:
    """Extract JSON metadata from text using OpenAI API."""
    try:
        logger.info(f"Extracting JSON metadata from text ({estimate_tokens(text)} tokens)")
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": EXTRACTION_PROMPT + text
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        json_str = response.choices[0].message.content
        logger.info(f"Received JSON response: {len(json_str)} characters")
        
        # Parse and validate JSON
        result = json.loads(json_str)
        
        # Ensure required keys exist
        required_keys = ["keywords", "segments", "examples_herding", "summary"]
        for key in required_keys:
            if key not in result:
                logger.warning(f"Missing required key: {key}")
                result[key] = [] if key != "summary" else ""
        
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {e}")
        return None
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return None

def merge_json_results(json_results: List[Dict]) -> Dict:
    """Merge multiple JSON results from text chunks."""
    if not json_results:
        return {
            "keywords": [],
            "segments": [],
            "examples_herding": [],
            "summary": ""
        }
    
    if len(json_results) == 1:
        return json_results[0]
    
    logger.info(f"Merging {len(json_results)} JSON results")
    
    merged = {
        "keywords": [],
        "segments": [],
        "examples_herding": [],
        "summary": ""
    }
    
    # Merge keywords (deduplicate)
    all_keywords = []
    for result in json_results:
        all_keywords.extend(result.get("keywords", []))
    merged["keywords"] = list(dict.fromkeys(all_keywords))  # Remove duplicates, preserve order
    
    # Merge segments
    for result in json_results:
        merged["segments"].extend(result.get("segments", []))
    
    # Merge examples
    for result in json_results:
        merged["examples_herding"].extend(result.get("examples_herding", []))
    
    # Combine summaries
    summaries = [result.get("summary", "") for result in json_results if result.get("summary")]
    if summaries:
        merged["summary"] = " ".join(summaries)
    
    return merged

def process_file(client: OpenAI, src_file: pathlib.Path, target_dir: pathlib.Path):
    """Process a single transcript file."""
    logger.info(f"Processing file: {src_file.name}")
    
    try:
        # Read source file
        with open(src_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        
        if not original_text.strip():
            logger.warning(f"File {src_file.name} is empty, skipping")
            return
        
        # Clean the text
        cleaned_text = clean_text(original_text)
        
        # Split text if necessary
        text_chunks = split_text_for_processing(cleaned_text)
        
        # Extract JSON from each chunk
        json_results = []
        for i, chunk in enumerate(text_chunks, 1):
            logger.info(f"Processing chunk {i}/{len(text_chunks)} for {src_file.name}")
            json_result = extract_json_from_text(client, chunk)
            if json_result:
                json_results.append(json_result)
        
        # Merge JSON results if multiple chunks
        final_json = merge_json_results(json_results)
        
        if not final_json or not any(final_json.values()):
            logger.error(f"Failed to extract valid JSON for {src_file.name}")
            # Create error file
            error_file = target_dir / (src_file.stem + ".error.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"JSON extraction failed for {src_file.name}\n")
                f.write(f"Original text length: {len(original_text)} characters\n")
            return
        
        # Format the cleaned text with line breaks
        formatted_text = format_text_with_line_breaks(cleaned_text)
        
        # Create augmented file
        target_file = target_dir / (src_file.stem + ".augmented.txt")
        
        with open(target_file, 'w', encoding='utf-8') as f:
            # Write JSON metadata
            json.dump(final_json, f, ensure_ascii=False, indent=2)
            f.write('\n\n')
            f.write('-' * 60)
            f.write('\n\n')
            # Write formatted text
            f.write(formatted_text)
        
        logger.info(f"Successfully created augmented file: {target_file.name}")
        logger.info(f"Keywords: {len(final_json.get('keywords', []))}, "
                   f"Segments: {len(final_json.get('segments', []))}, "
                   f"Examples: {len(final_json.get('examples_herding', []))}")
        
    except Exception as e:
        logger.error(f"Error processing {src_file.name}: {e}")
        # Create error file
        error_file = target_dir / (src_file.stem + ".error.txt")
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"Processing error for {src_file.name}: {str(e)}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_augment_transcripts.py <source_folder> <target_folder>")
        sys.exit(1)
    
    source_dir = pathlib.Path(sys.argv[1]).expanduser()
    target_dir = pathlib.Path(sys.argv[2]).expanduser()
    
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        sys.exit(2)
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")
        sys.exit(3)
    
    # Find all txt files
    txt_files = sorted([f for f in source_dir.glob("*.txt") if f.is_file()])
    
    if not txt_files:
        print(f"No .txt files found in {source_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(txt_files)} text files to process")
    
    # Process each file
    for txt_file in tqdm(txt_files, desc="Augmenting transcripts", unit="file"):
        process_file(client, txt_file, target_dir)
    
    logger.info(f"Processing complete. Augmented files in: {target_dir}")
    print(f"Done. Augmented files created in: {target_dir}")

if __name__ == "__main__":
    main()