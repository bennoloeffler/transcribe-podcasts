#!/usr/bin/env python3
# batch_extract_entities_from_augmented_txts.py
# Extracts entities and relations from augmented transcript files, performs similarity-based merging
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   python batch_extract_entities_from_augmented_txts.py <source_folder> <dest_folder>

import os
import sys
import json
import pathlib
import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('entity_extraction_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
ANALYSIS_MODEL = "gpt-4o"

# Entity type classification patterns
ENTITY_TYPE_PATTERNS = {
    'PERSON': [
        'benno', 'fabian', 'gerhard', 'fritze', 'simon', 'sprenger', 'sophie', 'scholl',
        'wohland', 'lea', 'lotta', 'peter', 'tobias', 'paul'
    ],
    'ORGANIZATION': [
        'herding', 'fraunhofer', 'intrinsify', 'amberg', 'unternehmen', 'firma', 'konzern',
        'organisation', 'institut', 'gmbh', 'ag'
    ],
    'CONCEPT': [
        'illusion', 'kultur', 'vertrauen', 'macht', 'führung', 'kommunikation', 'mindset',
        'motivation', 'innovation', 'wertschöpfung', 'selbstorganisation', 'agilität',
        'transformation', 'entwicklung', 'management', 'leadership', 'sicherheit'
    ],
    'METHOD': [
        'scrum', 'ccpm', 'okr', 'kata', 'lean', 'agile', 'kanban', 'dynamics',
        'iso', 'zertifizierung', 'framework', 'methode', 'prozess', 'verfahren'
    ]
}

# Type-specific similarity thresholds
SIMILARITY_THRESHOLDS = {
    'PERSON': 0.92,
    'ORGANIZATION': 0.88, 
    'CONCEPT': 0.78,
    'METHOD': 0.83,
    'UNKNOWN': 0.85
}

def normalize_entity_name(name: str) -> str:
    """Normalize entity name by removing formatting artifacts."""
    if not name:
        return ""
    
    original = name
    # Remove leading dashes and bullet points
    normalized = re.sub(r'^[-•*]\s*', '', name.strip())
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    result = normalized.lower()
    
    # Debug log normalization
    if original != result:
        logger.debug(f"Normalized '{original}' -> '{result}'")
    
    return result

def extract_folgen_name(source_file: str) -> str:
    """Extract Folgen identifier from source file name."""
    try:
        # Remove path and extension: "path/004 Vertrauen - Classic.augmented.entities.txt" -> "004 Vertrauen - Classic"
        filename = pathlib.Path(source_file).stem
        
        # Remove .augmented suffix if present
        if filename.endswith('.augmented'):
            filename = filename[:-10]  # Remove ".augmented"
        
        # Remove .entities suffix if present  
        if filename.endswith('.entities'):
            filename = filename[:-9]  # Remove ".entities"
            
        return filename.strip()
        
    except Exception as e:
        logger.warning(f"Could not extract Folgen name from {source_file}: {e}")
        return ""

def format_folgen_reference(folgen_sources: Set[str]) -> str:
    """Format FOLGEN reference based on count and content."""
    if not folgen_sources:
        return ""
        
    count = len(folgen_sources)
    
    if count > 50:
        return f"FOLGEN: in {count} Folgen erwähnt"
    elif count > 10:
        # Extract just the numbers for compact format
        numbers = []
        for folgen in sorted(folgen_sources):
            # Extract leading number from "004 Vertrauen - Classic" -> "004"
            match = re.match(r'^(\d+)', folgen)
            if match:
                numbers.append(match.group(1))
            else:
                numbers.append(folgen[:10])  # Fallback to first 10 chars
        return f"FOLGEN: {', '.join(numbers)}"
    else:
        # Full names for smaller sets
        sorted_folgen = sorted(folgen_sources)
        return f"FOLGEN: {', '.join(sorted_folgen)}"

def classify_entity_type(entity_name: str, entity_description: str) -> str:
    """Classify entity type based on name and description patterns."""
    name_lower = entity_name.lower()
    desc_lower = entity_description.lower()
    combined_text = f"{name_lower} {desc_lower}"
    
    # Check each type's patterns
    for entity_type, patterns in ENTITY_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern in combined_text:
                return entity_type
    
    # Additional heuristics
    if any(word in desc_lower for word in ['person', 'moderator', 'cto', 'autor', 'experte']):
        return 'PERSON'
    elif any(word in desc_lower for word in ['unternehmen', 'firma', 'organisation', 'institut']):
        return 'ORGANIZATION'
    elif any(word in desc_lower for word in ['konzept', 'prinzip', 'ansatz', 'philosophie', 'denkweise']):
        return 'CONCEPT'
    elif any(word in desc_lower for word in ['methode', 'framework', 'tool', 'werkzeug', 'system', 'modell']):
        return 'METHOD'
    
    return 'UNKNOWN'

@dataclass
class Entity:
    name: str
    description: str
    source_file: str
    normalized_name: str = None
    entity_type: str = "UNKNOWN"
    folgen_sources: Set[str] = None
    
    def __post_init__(self):
        if self.normalized_name is None:
            self.normalized_name = normalize_entity_name(self.name)
        if self.entity_type == "UNKNOWN":
            self.entity_type = classify_entity_type(self.name, self.description)
        if self.folgen_sources is None:
            self.folgen_sources = set()
            # Extract Folgen name from source file
            folgen_name = extract_folgen_name(self.source_file)
            if folgen_name:
                self.folgen_sources.add(folgen_name)
    
    def __str__(self):
        folgen_ref = format_folgen_reference(self.folgen_sources)
        if folgen_ref:
            return f"{self.name}: {self.description} {folgen_ref}"
        return f"{self.name}: {self.description}"

@dataclass
class Relation:
    source: str
    target: str
    short_desc: str
    long_desc: str
    source_file: str
    
    def __str__(self):
        return f"{self.source} --> {self.target}: SHORT: {self.short_desc} LONG: {self.long_desc}"

ENTITY_EXTRACTION_PROMPT = """
Analysiere den folgenden Podcast-Transkript-Text und extrahiere ALLE wichtigen Entitäten und deren Beziehungen.

WICHTIGE HINWEISE:
- Dies ist ein Podcast-Transkript von Benno und Fabian über Organisationsentwicklung
- Fabian ist CTO bei der Firma Herding in Amberg
- Extrahiere ALLE relevanten Entitäten: Personen, Unternehmen, Konzepte, Methoden, abstrakte Ideen
- Verwende prägnante, eindeutige Namen für Entitäten
- Beschreibungen sollen informativ aber knapp sein (max. 2-3 Sätze)
- Beziehungen sollen spezifisch und aussagekräftig sein

AUSGABEFORMAT (WICHTIG: Keine Aufzählungszeichen verwenden!):

ENTITIES
EntitätName1: Beschreibung mit Attributen oder Definition und relevanten Details
EntitätName2: Weitere Beschreibung mit wichtigen Informationen
EntitätName3: Noch eine Beschreibung ohne Bindestriche oder Aufzählungszeichen

RELATIONS
EntitätName1 --> EntitätName2: SHORT: Kurzbeschreibung (verursacht, inspiriert, erschwert, ist Teil von, etc.) LONG: Ausführlichere Erklärung der Beziehung
EntitätName3 --> EntitätName1: SHORT: Weitere Beziehung LONG: Detaillierte Beschreibung
EntitätName2 --> EntitätName3: SHORT: Noch eine Beziehung LONG: Ausführliche Beschreibung ohne Bindestriche

WICHTIG: Verwende NIEMALS Bindestriche (-) oder andere Aufzählungszeichen vor den Entitäts- oder Relationsnamen. Beginne jede Zeile direkt mit dem Namen.

DEDUPLICATION (SEHR WICHTIG):
  - Wenn du eine Entität mehrfach im Text erkennst, erstelle NUR EINEN Eintrag
  - Kombiniere alle relevanten Informationen in einer einzigen, umfassenden Beschreibung
  - Entitäten mit identischen Namen (z.B. "Benno", "Fabian", "Herding") sind DIESELBE Entität
  - Längere, informativere Beschreibungen sind besser als mehrere kurze
  - Prüfe deine finale Liste: Jeder Entitätsname darf nur EINMAL vorkommen

Analysiere folgenden Text:

"""

ENTITY_MERGE_PROMPT = """Du bist Experte für Organisationsentwicklung und sollst entscheiden, ob zwei ähnliche Entitäten zusammengeführt werden sollen.

KONTEXT: Dies sind Entitäten aus Podcast-Transkripten von Benno und Fabian über Organisationsentwicklung und moderne Soziologie in Organisationen.

AUFGABE: Entscheide, ob die folgenden zwei Entitäten DIESELBE Sache beschreiben und zusammengeführt werden sollten.

ENTITÄT 1:
{entity1}

ENTITÄT 2:
{entity2}

ANWEISUNGEN:
- Wenn die Entitäten DASSELBE Konzept/Person/Unternehmen/Methode beschreiben → MERGE
- Wenn sie ähnlich, aber UNTERSCHIEDLICHE Aspekte darstellen → SEPARATE
- Bei MERGE: Erstelle eine verbesserte, kombinierte Beschreibung

AUSGABEFORMAT (JSON):
{{
  "decision": "MERGE" oder "SEPARATE",
  "merged_name": "OptimalerName" (nur bei MERGE),
  "merged_description": "Kombinierte Beschreibung" (nur bei MERGE),
  "reasoning": "Kurze Begründung der Entscheidung"
}}
"""

def extract_content_from_augmented_file(file_path: pathlib.Path) -> str:
    """Extract the main text content from an augmented file (skip JSON metadata)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the separator line and extract text after it
        separator_match = re.search(r'-{20,}', content)
        if separator_match:
            text_content = content[separator_match.end():].strip()
            return text_content
        else:
            logger.warning(f"No separator found in {file_path.name}, using full content")
            return content
            
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return ""

def parse_entities_and_relations(text: str, source_file: str) -> Tuple[List[Entity], List[Relation]]:
    """Parse entities and relations from the OpenAI response text."""
    entities = []
    relations = []
    
    try:
        logger.info(f"Parsing response text: {len(text)} characters")
        logger.debug(f"Response text preview: {text[:500]}...")
        
        # Split by sections
        sections = text.split('\n\n')
        current_section = None
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if section.upper().startswith('ENTITIES'):
                current_section = 'entities'
                logger.info("Found ENTITIES section")
                # Process entities in this section
                lines = section.split('\n')[1:]  # Skip the header line
                for line in lines:
                    line = line.strip()
                    if ':' in line and not line.startswith('RELATIONS'):
                        name_desc = line.split(':', 1)
                        if len(name_desc) == 2:
                            name = name_desc[0].strip()
                            # Remove leading dash if present
                            if name.startswith('- '):
                                name = name[2:].strip()
                            elif name.startswith('-'):
                                name = name[1:].strip()
                            desc = name_desc[1].strip()
                            entities.append(Entity(name, desc, source_file))
                            logger.debug(f"Parsed entity: {name}")
            
            elif section.upper().startswith('RELATIONS'):
                current_section = 'relations'
                logger.info("Found RELATIONS section")
                # Process relations in this section  
                lines = section.split('\n')[1:]  # Skip the header line
                for line in lines:
                    line = line.strip()
                    if '-->' in line:
                        # Parse: Source --> Target: SHORT: short_desc LONG: long_desc
                        relation_match = re.match(r'(.+?)\s*-->\s*(.+?):\s*SHORT:\s*(.+?)\s*LONG:\s*(.+)', line)
                        if relation_match:
                            source = relation_match.group(1).strip()
                            target = relation_match.group(2).strip()
                            short_desc = relation_match.group(3).strip()
                            long_desc = relation_match.group(4).strip()
                            relations.append(Relation(source, target, short_desc, long_desc, source_file))
                            logger.debug(f"Parsed relation: {source} --> {target}")
                        else:
                            # Fallback: simpler parsing
                            arrow_parts = line.split('-->', 1)
                            if len(arrow_parts) == 2:
                                source = arrow_parts[0].strip()
                                rest = arrow_parts[1]
                                colon_parts = rest.split(':', 1)
                                if len(colon_parts) == 2:
                                    target = colon_parts[0].strip()
                                    desc_part = colon_parts[1].strip()
                                    # Try to extract SHORT and LONG
                                    if 'SHORT:' in desc_part and 'LONG:' in desc_part:
                                        short_match = re.search(r'SHORT:\s*(.+?)\s*LONG:', desc_part)
                                        long_match = re.search(r'LONG:\s*(.+)', desc_part)
                                        short_desc = short_match.group(1).strip() if short_match else desc_part
                                        long_desc = long_match.group(1).strip() if long_match else ""
                                    else:
                                        short_desc = desc_part
                                        long_desc = ""
                                    relations.append(Relation(source, target, short_desc, long_desc, source_file))
                                    logger.debug(f"Parsed relation (fallback): {source} --> {target}")
            
            elif current_section == 'entities' and ':' in section:
                # Continue parsing entities
                for line in section.split('\n'):
                    line = line.strip()
                    if ':' in line and not line.startswith('RELATIONS'):
                        name_desc = line.split(':', 1)
                        if len(name_desc) == 2:
                            name = name_desc[0].strip()
                            # Remove leading dash if present
                            if name.startswith('- '):
                                name = name[2:].strip()
                            elif name.startswith('-'):
                                name = name[1:].strip()
                            desc = name_desc[1].strip()
                            entities.append(Entity(name, desc, source_file))
                            logger.debug(f"Parsed entity (continued): {name}")
            
            elif current_section == 'relations' and '-->' in section:
                # Continue parsing relations
                for line in section.split('\n'):
                    line = line.strip()
                    if '-->' in line:
                        relation_match = re.match(r'(.+?)\s*-->\s*(.+?):\s*SHORT:\s*(.+?)\s*LONG:\s*(.+)', line)
                        if relation_match:
                            source = relation_match.group(1).strip()
                            target = relation_match.group(2).strip()
                            short_desc = relation_match.group(3).strip()
                            long_desc = relation_match.group(4).strip()
                            relations.append(Relation(source, target, short_desc, long_desc, source_file))
                            logger.debug(f"Parsed relation (continued): {source} --> {target}")
        
        # Log if parsing failed
        if not entities and not relations:
            logger.warning(f"No entities or relations parsed from response. Full text: {text}")
    
    except Exception as e:
        logger.error(f"Error parsing entities and relations: {e}")
        logger.error(f"Full response text: {text}")
    
    return entities, relations

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def call_openai_with_retry(client: OpenAI, prompt: str, file_name: str) -> str:
    """Call OpenAI API with retry logic."""
    try:
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content
        logger.info(f"OpenAI API success for {file_name}: {len(response_text)} characters")
        return response_text
        
    except Exception as e:
        logger.error(f"OpenAI API error for {file_name}: {e}")
        raise

def extract_entities_from_file(client: OpenAI, file_path: pathlib.Path) -> Tuple[List[Entity], List[Relation]]:
    """Extract entities and relations from a single augmented transcript file."""
    logger.info(f"Extracting entities from: {file_path.name}")
    
    try:
        text_content = extract_content_from_augmented_file(file_path)
        if not text_content.strip():
            logger.warning(f"No content found in {file_path.name}")
            return [], []
        
        logger.info(f"Content length: {len(text_content)} characters")
        
        # Estimate tokens and split if necessary
        estimated_tokens = len(text_content) // 4
        max_tokens = 12000
        
        if estimated_tokens > max_tokens:
            logger.info(f"Text too large ({estimated_tokens} tokens), splitting")
            # Split into chunks
            words = text_content.split()
            # Calculate chunk size in words (tokens / 4 * 0.8 for safety margin)
            chunk_size_words = int(max_tokens / 4 * 0.8)  # ~2400 words per chunk
            chunks = [' '.join(words[i:i+chunk_size_words]) for i in range(0, len(words), chunk_size_words)]
            logger.info(f"Created {len(chunks)} chunks from {len(words)} words (chunk size: {chunk_size_words} words)")
        else:
            chunks = [text_content]
        
        all_entities = []
        all_relations = []
        
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)} for {file_path.name}")
            
            try:
                # Use retry-enabled API call
                full_prompt = ENTITY_EXTRACTION_PROMPT + chunk
                response_text = call_openai_with_retry(client, full_prompt, file_path.name)
                
                logger.info(f"Received entity extraction response: {len(response_text)} characters")
                logger.debug(f"Response preview: {response_text[:200]}...")
                
                entities, relations = parse_entities_and_relations(response_text, file_path.name)
                all_entities.extend(entities)
                all_relations.extend(relations)
                
                logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations from chunk {i}")
                
                # If no entities/relations were extracted, log the full response for debugging
                if not entities and not relations:
                    logger.error(f"PARSING FAILED for {file_path.name}, chunk {i}")
                    logger.error(f"Full OpenAI response: {response_text}")
                    
            except Exception as e:
                logger.error(f"Failed to process chunk {i} for {file_path.name}: {e}")
                # Continue with other chunks
                continue
        
        return all_entities, all_relations
        
    except Exception as e:
        logger.error(f"Error extracting entities from {file_path}: {e}")
        return [], []

def create_embeddings(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    try:
        logger.info(f"Creating embeddings for {len(texts)} texts")
        
        # Process in batches to avoid API limits
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            logger.info(f"Created embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return np.array(all_embeddings)
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        return np.array([])

def find_exact_matches(entities: List[Entity]) -> List[Tuple[int, int]]:
    """Find entities with identical normalized names."""
    logger.info("Phase 1: Finding exact matches")
    exact_matches = []
    name_to_indices = defaultdict(list)
    
    # Group entities by normalized name
    for i, entity in enumerate(entities):
        logger.debug(f"Entity {i}: name='{entity.name}' -> normalized='{entity.normalized_name}'")
        name_to_indices[entity.normalized_name].append(i)
    
    # Debug: Show all normalized names and their counts
    for normalized_name, indices in name_to_indices.items():
        if len(indices) > 1:
            logger.info(f"Found {len(indices)} entities with normalized name '{normalized_name}': {[entities[i].name for i in indices]}")
    
    # Find groups with multiple entities (exact matches)
    for normalized_name, indices in name_to_indices.items():
        if len(indices) > 1:
            # For each group of identical entities, create one representative and mark others for merging
            # This ensures all identical entities get merged into one, not just pairs
            primary_idx = indices[0]  # First entity becomes the primary
            for secondary_idx in indices[1:]:
                exact_matches.append((primary_idx, secondary_idx))
                logger.info(f"Exact match found: {entities[primary_idx].name} <-> {entities[secondary_idx].name}")
    
    return exact_matches

def find_similar_entities_by_type(entities: List[Entity], client: OpenAI) -> List[Tuple[int, int, float]]:
    """Find pairs of similar entities using type-aware thresholds and embeddings."""
    logger.info(f"Phase 2: Finding similar entities among {len(entities)} entities using type-aware approach")
    
    # Create text representations for embedding
    entity_texts = [f"{entity.name}: {entity.description}" for entity in entities]
    
    # Create embeddings
    embeddings = create_embeddings(client, entity_texts)
    
    if len(embeddings) == 0:
        logger.error("Failed to create embeddings")
        return []
    
    # Calculate cosine similarities
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find similar pairs using type-specific thresholds
    similar_pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            entity_i = entities[i]
            entity_j = entities[j]
            
            # Use the more restrictive threshold of the two entities
            threshold_i = SIMILARITY_THRESHOLDS.get(entity_i.entity_type, SIMILARITY_THRESHOLDS['UNKNOWN'])
            threshold_j = SIMILARITY_THRESHOLDS.get(entity_j.entity_type, SIMILARITY_THRESHOLDS['UNKNOWN'])
            threshold = max(threshold_i, threshold_j)
            
            similarity = similarity_matrix[i][j]
            if similarity >= threshold:
                similar_pairs.append((i, j, similarity))
                logger.info(f"Similar entities found: {entity_i.name} ({entity_i.entity_type}) <-> {entity_j.name} ({entity_j.entity_type}) (similarity: {similarity:.3f}, threshold: {threshold:.3f})")
    
    # Sort by similarity (highest first)
    similar_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return similar_pairs

def should_merge_entities(client: OpenAI, entity1: Entity, entity2: Entity) -> Optional[Entity]:
    """Use OpenAI to decide if two entities should be merged, and return merged entity if yes."""
    try:
        prompt = ENTITY_MERGE_PROMPT.format(
            entity1=f"{entity1.name}: {entity1.description}",
            entity2=f"{entity2.name}: {entity2.description}"
        )
        
        response = client.chat.completions.create(
            model=ANALYSIS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        
        if result.get("decision") == "MERGE":
            merged_name = result.get("merged_name", entity1.name)
            merged_desc = result.get("merged_description", entity1.description)
            logger.info(f"Merging entities: {entity1.name} + {entity2.name} -> {merged_name}")
            logger.info(f"Reasoning: {result.get('reasoning', 'No reasoning provided')}")
            
            # Create merged entity with combined source info and FOLGEN sources
            sources = set([entity1.source_file, entity2.source_file])
            merged_source = ", ".join(sorted(sources))
            
            # Combine FOLGEN sources
            combined_folgen = set(entity1.folgen_sources) | set(entity2.folgen_sources)
            
            # Determine entity type (prefer the more specific one)
            merged_type = entity1.entity_type if entity1.entity_type != "UNKNOWN" else entity2.entity_type
            
            return Entity(
                name=merged_name,
                description=merged_desc, 
                source_file=merged_source,
                normalized_name=normalize_entity_name(merged_name),
                entity_type=merged_type,
                folgen_sources=combined_folgen
            )
        else:
            logger.info(f"Keeping entities separate: {entity1.name} and {entity2.name}")
            logger.info(f"Reasoning: {result.get('reasoning', 'No reasoning provided')}")
            return None
            
    except Exception as e:
        logger.error(f"Error in merge decision for {entity1.name} and {entity2.name}: {e}")
        return None

def merge_entities_by_name_first(entities: List[Entity]) -> List[Entity]:
    """Merge all entities with identical names (case-insensitive) into single entities with combined descriptions."""
    logger.info("Phase 0: Merging entities with identical names")
    
    # Group entities by normalized name (using actual name, not pre-normalized)
    name_groups = defaultdict(list)
    for entity in entities:
        normalized_name = entity.name.lower().strip()
        name_groups[normalized_name].append(entity)
    
    merged_entities = []
    for normalized_name, group in name_groups.items():
        if len(group) == 1:
            # Single entity, keep as is
            merged_entities.append(group[0])
        else:
            # Multiple entities with same name - merge them
            logger.info(f"Merging {len(group)} entities with name '{group[0].name}': {[e.source_file for e in group]}")
            
            # Use the first entity as base
            primary = group[0]
            
            # Combine all descriptions, removing duplicates while preserving order
            combined_descriptions = []
            seen_descriptions = set()
            
            for entity in group:
                desc = entity.description.strip()
                if desc and desc not in seen_descriptions:
                    combined_descriptions.append(desc)
                    seen_descriptions.add(desc)
            
            # Join descriptions with sentences/periods properly
            final_description = ". ".join(combined_descriptions).strip()
            if final_description and not final_description.endswith('.'):
                final_description += "."
            
            # Combine all FOLGEN sources
            all_folgen = set()
            for entity in group:
                if entity.folgen_sources:
                    all_folgen.update(entity.folgen_sources)
            
            # Create merged entity
            merged_entity = Entity(
                name=primary.name,
                description=final_description,
                source_file=", ".join([e.source_file for e in group]),
                normalized_name=primary.normalized_name,
                entity_type=primary.entity_type,
                folgen_sources=all_folgen
            )
            
            merged_entities.append(merged_entity)
            logger.info(f"Created merged entity '{merged_entity.name}' with {len(all_folgen)} FOLGEN sources and combined description: {final_description[:100]}...")
    
    logger.info(f"Name-based merging complete: {len(entities)} -> {len(merged_entities)} entities")
    return merged_entities

def merge_entities_advanced(entities: List[Entity], relations: List[Relation], client: OpenAI) -> Tuple[List[Entity], List[Relation]]:
    """Advanced multi-stage entity merging with exact matches and type-aware similarity."""
    logger.info("Starting advanced entity merging process")
    
    # Phase 0: Merge entities with identical names first (critical for relation integrity)
    entities = merge_entities_by_name_first(entities)
    
    # Phase 1: Find exact matches (no API calls needed)
    exact_matches = find_exact_matches(entities)
    logger.info(f"Found {len(exact_matches)} exact match pairs")
    
    # Phase 2: Find similar entities with type-aware thresholds
    similar_pairs = []
    remaining_entities = list(entities)  # Work with remaining entities after exact merges
    
    if len(remaining_entities) > 1:
        similar_pairs = find_similar_entities_by_type(remaining_entities, client)
        logger.info(f"Found {len(similar_pairs)} similar entity pairs")
    
    # Combine all pairs for processing
    all_pairs = []
    
    # Add exact matches with perfect similarity
    for i, j in exact_matches:
        all_pairs.append((i, j, 1.0))
    
    # Add similar pairs
    all_pairs.extend(similar_pairs)
    
    if not all_pairs:
        logger.info("No similar entities found")
        return entities, relations
    
    # Track merges: old_name -> new_name
    merge_map = {}
    entities_to_remove = set()
    entities_to_add = []
    
    # Process all pairs (exact matches and similar pairs)
    for i, j, similarity in all_pairs:
        entity1 = entities[i]
        entity2 = entities[j]
        
        # Skip if either entity was already merged
        if i in entities_to_remove or j in entities_to_remove:
            continue
        
        # For exact matches (similarity = 1.0), merge automatically
        # For similar matches, ask OpenAI to decide
        if similarity >= 0.99:  # Exact match
            logger.info(f"Auto-merging exact match: {entity1.name} + {entity2.name}")
            merged_entity = create_merged_entity(entity1, entity2, prefer_first=True)
        else:
            # Ask OpenAI to decide on merge
            merged_entity = should_merge_entities(client, entity1, entity2)
        
        if merged_entity:
            # Record the merge
            merge_map[entity1.name] = merged_entity.name
            merge_map[entity2.name] = merged_entity.name
            
            # Mark original entities for removal
            entities_to_remove.add(i)
            entities_to_remove.add(j)
            
            # Add merged entity
            entities_to_add.append(merged_entity)
    
    # Create new entity list
    new_entities = []
    for i, entity in enumerate(entities):
        if i not in entities_to_remove:
            new_entities.append(entity)
    new_entities.extend(entities_to_add)
    
    # Update relations based on merges
    new_relations = []
    for relation in relations:
        new_source = merge_map.get(relation.source, relation.source)
        new_target = merge_map.get(relation.target, relation.target)
        
        # Skip self-relations that might be created by merging
        if new_source != new_target:
            new_relations.append(Relation(
                new_source, new_target, 
                relation.short_desc, relation.long_desc, 
                relation.source_file
            ))
    
    logger.info(f"Advanced merging complete: {len(entities)} -> {len(new_entities)} entities, {len(relations)} -> {len(new_relations)} relations")
    
    return new_entities, new_relations

def create_merged_entity(entity1: Entity, entity2: Entity, prefer_first: bool = True) -> Entity:
    """Create a merged entity combining information from two entities."""
    if prefer_first:
        primary, secondary = entity1, entity2
    else:
        primary, secondary = entity2, entity1
    
    # Combine FOLGEN sources
    combined_folgen = set(primary.folgen_sources) | set(secondary.folgen_sources)
    
    # Use primary entity's name and description, but could be enhanced with AI
    merged_entity = Entity(
        name=primary.name,
        description=primary.description,
        source_file=f"{primary.source_file}, {secondary.source_file}",
        normalized_name=primary.normalized_name,
        entity_type=primary.entity_type,
        folgen_sources=combined_folgen
    )
    
    logger.info(f"Created merged entity: {merged_entity.name} with {len(combined_folgen)} FOLGEN sources")
    
    return merged_entity

def write_entities_file(entities: List[Entity], relations: List[Relation], output_file: pathlib.Path):
    """Write entities and relations to a file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("ENTITIES\n")
            for entity in sorted(entities, key=lambda e: e.name):
                # Use the Entity's __str__ method which includes FOLGEN references
                f.write(f"{str(entity)}\n")
            
            f.write("\nRELATIONS\n")
            for relation in sorted(relations, key=lambda r: (r.source, r.target)):
                f.write(f"{relation.source} --> {relation.target}: SHORT: {relation.short_desc} LONG: {relation.long_desc}\n")
        
        logger.info(f"Written {len(entities)} entities and {len(relations)} relations to {output_file}")
        
    except Exception as e:
        logger.error(f"Error writing entities file {output_file}: {e}")

def consolidate_only_mode(dest_dir: pathlib.Path):
    """Consolidate existing entity files from target folder without re-extraction."""
    logger.info("Running in consolidate-only mode")
    
    if not dest_dir.exists():
        print(f"Destination directory not found: {dest_dir}")
        sys.exit(2)
    
    # Find existing entity files
    entity_files = list(dest_dir.glob("*.entities.txt"))
    if not entity_files:
        print(f"No entity files found in {dest_dir}")
        sys.exit(3)
    
    print(f"Found {len(entity_files)} entity files to consolidate")
    logger.info(f"Found {len(entity_files)} entity files to consolidate")
    
    # Read all entities and relations from existing files
    all_entities = []
    all_relations = []
    
    for file_path in entity_files:
        entities, relations = read_entities_from_file(file_path)
        all_entities.extend(entities)
        all_relations.extend(relations)
        logger.info(f"Loaded {len(entities)} entities and {len(relations)} relations from {file_path.name}")
    
    logger.info(f"Loaded total: {len(all_entities)} entities and {len(all_relations)} relations")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)
    
    # Merge entities
    print("Starting entity merging process...")
    merged_entities, merged_relations = merge_entities_advanced(all_entities, all_relations, client)
    
    # Write consolidated output
    consolidated_path = dest_dir / "__ALL_ENTITIES_ALL_RELATIONS.txt"
    write_entities_file(merged_entities, merged_relations, consolidated_path)
    
    print(f"\nConsolidation complete!")
    print(f"Files processed: {len(entity_files)}")
    print(f"Total entities: {len(merged_entities)} (after merging)")
    print(f"Total relations: {len(merged_relations)}")
    print(f"Consolidated file: {consolidated_path}")

def read_entities_from_file(file_path: pathlib.Path) -> Tuple[List[Entity], List[Relation]]:
    """Read entities and relations from an existing entity file."""
    entities = []
    relations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse similar to the original parsing logic
        sections = content.split('\n\n')
        current_section = None
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            if section.upper().startswith('ENTITIES'):
                current_section = 'entities'
                # Process entities
                lines = section.split('\n')[1:]  # Skip header
                for line in lines:
                    line = line.strip()
                    if ':' in line and 'FOLGEN:' in line:
                        # Split on first colon to get name and rest
                        name_rest = line.split(':', 1)
                        if len(name_rest) == 2:
                            name = name_rest[0].strip()
                            rest = name_rest[1].strip()
                            
                            # Find FOLGEN part
                            if 'FOLGEN:' in rest:
                                desc_folgen = rest.split('FOLGEN:', 1)
                                desc = desc_folgen[0].strip()
                                folgen_str = desc_folgen[1].strip()
                                
                                # Parse FOLGEN sources
                                folgen_sources = set()
                                if folgen_str:
                                    # Handle different FOLGEN formats
                                    if ',' in folgen_str:
                                        for source in folgen_str.split(','):
                                            folgen_sources.add(source.strip())
                                    else:
                                        folgen_sources.add(folgen_str)
                                
                                entities.append(Entity(name, desc, str(file_path), folgen_sources=folgen_sources))
                            
            elif section.upper().startswith('RELATIONS'):
                current_section = 'relations'
                # Process relations
                lines = section.split('\n')[1:]  # Skip header
                for line in lines:
                    line = line.strip()
                    if '-->' in line:
                        parts = line.split('-->', 1)
                        if len(parts) == 2:
                            from_entity = parts[0].strip()
                            to_rest = parts[1].strip()
                            
                            if ':' in to_rest:
                                to_desc = to_rest.split(':', 1)
                                to_entity = to_desc[0].strip()
                                description = to_desc[1].strip()
                                
                                # Parse SHORT and LONG descriptions
                                short_desc = ""
                                long_desc = description
                                
                                if 'SHORT:' in description and 'LONG:' in description:
                                    parts = description.split('LONG:', 1)
                                    if len(parts) == 2:
                                        short_part = parts[0].replace('SHORT:', '').strip()
                                        long_part = parts[1].strip()
                                        short_desc = short_part
                                        long_desc = long_part
                                
                                relations.append(Relation(from_entity, to_entity, short_desc, long_desc, str(file_path)))
        
        return entities, relations
        
    except Exception as e:
        logger.error(f"Error reading entities from {file_path}: {e}")
        return [], []

def full_extraction_mode(source_dir: pathlib.Path, dest_dir: pathlib.Path):
    """Full extraction mode: extract entities from source files AND consolidate."""
    logger.info("Running in full extraction mode")
    
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}")
        sys.exit(2)
    
    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI client
    try:
        client = OpenAI()
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")
        sys.exit(3)
    
    # Find augmented files
    augmented_files = sorted([f for f in source_dir.glob("*.augmented.txt") if f.is_file()])
    
    if not augmented_files:
        print(f"No .augmented.txt files found in {source_dir}")
        sys.exit(0)
    
    logger.info(f"Found {len(augmented_files)} augmented files to process")
    
    # Phase 1: Extract entities from each file
    all_entities = []
    all_relations = []
    
    for file_path in tqdm(augmented_files, desc="Extracting entities", unit="file"):
        entities, relations = extract_entities_from_file(client, file_path)
        
        if entities or relations:
            # Write individual file results
            output_file = dest_dir / (file_path.stem + ".entities.txt")
            write_entities_file(entities, relations, output_file)
            
            # Add to global collections
            all_entities.extend(entities)
            all_relations.extend(relations)
        else:
            # Create detailed error file with diagnostics
            error_file = dest_dir / (file_path.stem + ".error.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Failed to extract entities from {file_path.name}\n")
                f.write(f"Timestamp: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n")
                f.write(f"File size: {file_path.stat().st_size} bytes\n")
                
                try:
                    content = extract_content_from_augmented_file(file_path)
                    f.write(f"Content length: {len(content)} characters\n")
                    f.write(f"Content preview: {content[:500]}...\n")
                    
                    # Try a simple test extraction to see what's happening
                    test_prompt = ENTITY_EXTRACTION_PROMPT + content[:2000]  # First 2000 chars
                    f.write(f"Test prompt length: {len(test_prompt)} characters\n")
                except Exception as inner_e:
                    f.write(f"Error reading content: {inner_e}\n")
    
    logger.info(f"Phase 1 complete: Extracted {len(all_entities)} entities and {len(all_relations)} relations total")
    
    if not all_entities:
        logger.warning("No entities extracted, exiting")
        return
    
    # Phase 2: Advanced entity merging with multi-stage approach  
    logger.info("Starting Phase 2: Advanced entity similarity analysis and merging")
    final_entities, final_relations = merge_entities_advanced(all_entities, all_relations, client)
    
    # Write consolidated file
    consolidated_file = dest_dir / "__ALL_ENTITIES_ALL_RELATIONS.txt"
    write_entities_file(final_entities, final_relations, consolidated_file)
    
    logger.info("Processing complete!")
    print(f"Entity extraction complete!")
    print(f"Individual files: {len(augmented_files)} processed")
    print(f"Total entities: {len(final_entities)} (after merging)")
    print(f"Total relations: {len(final_relations)}")
    print(f"Consolidated file: {consolidated_file}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_extract_entities_from_augmented_txts.py <source_folder> <dest_folder>")
        sys.exit(1)
    
    source_dir = pathlib.Path(sys.argv[1]).expanduser()
    dest_dir = pathlib.Path(sys.argv[2]).expanduser()
    
    # Ask user for processing mode
    print("\nChoose processing mode:")
    print("1. Extract entities to single files AND consolidate to one txt file (default)")
    print("2. Consolidate only - read existing entity files from target folder and merge")
    choice = input("Press ENTER for default, or 'c' for consolidate-only: ").strip().lower()
    
    if choice == 'c':
        return consolidate_only_mode(dest_dir)
    else:
        return full_extraction_mode(source_dir, dest_dir)

if __name__ == "__main__":
    main()