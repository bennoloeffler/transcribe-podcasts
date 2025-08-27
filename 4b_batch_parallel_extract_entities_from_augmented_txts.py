#!/usr/bin/env python3
# batch_parallel_extract_entities_from_augmented_txts.py
# Extracts entities and relations from augmented transcript files using 3-parallel LLM calls with hybrid deduplication
# Usage:
#   export OPENAI_API_KEY="sk-..."
#   python batch_parallel_extract_entities_from_augmented_txts.py <source_folder> <dest_folder>

import os
import sys
import json
import pathlib
import re
import logging
import asyncio
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
        logging.FileHandler('parallel_entity_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL = "text-embedding-3-small"
ANALYSIS_MODEL = "gpt-4o"
PARALLEL_RUNS = 3  # Number of parallel extraction runs

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
    ],
    'ZUSAMMENHANG': [
        'beziehung', 'verbindung', 'zusammenhang', 'relation', 'korrelation', 'abhängigkeit',
        'wechselwirkung', 'synergie', 'interdependenz', 'kausalität', 'kopplung'
    ],
    'HYPOTHESE': [
        'hypothese', 'annahme', 'behauptung', 'these', 'vermutung', 'spekulation',
        'theorie', 'postulat', 'prämisse', 'axiom', 'grundannahme'
    ],
    'FRAGE': [
        'frage', 'fragestellung', 'problem', 'herausforderung', 'dilemma', 'paradox',
        'rätsel', 'unbekannte', 'unsicherheit', 'klärungsbedarf'
    ],
    'INTERVENTION': [
        'intervention', 'eingriff', 'maßnahme', 'aktion', 'initiative', 'program',
        'kampagne', 'projekt', 'reform', 'veränderung', 'umgestaltung'
    ],
    'WERKZEUG': [
        'werkzeug', 'tool', 'instrument', 'hilfsmittel', 'technik', 'software',
        'plattform', 'system', 'anwendung', 'utility', 'ressource'
    ]
}

# Type-specific similarity thresholds for multi-level embedding analysis
SIMILARITY_THRESHOLDS = {
    'PERSON': 0.92,
    'ORGANIZATION': 0.88, 
    'CONCEPT': 0.78,
    'METHOD': 0.83,
    'ZUSAMMENHANG': 0.80,
    'HYPOTHESE': 0.75,
    'FRAGE': 0.82,
    'INTERVENTION': 0.85,
    'WERKZEUG': 0.87,
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
    
    # Additional heuristics for extended types
    if any(word in desc_lower for word in ['person', 'moderator', 'cto', 'autor', 'experte', 'kollege', 'mitarbeiter']):
        return 'PERSON'
    elif any(word in desc_lower for word in ['unternehmen', 'firma', 'organisation', 'institut']):
        return 'ORGANIZATION'
    elif any(word in desc_lower for word in ['konzept', 'prinzip', 'ansatz', 'philosophie', 'denkweise', 'theorie']):
        return 'CONCEPT'
    elif any(word in desc_lower for word in ['methode', 'framework', 'vorgehen', 'prozess', 'verfahren']):
        return 'METHOD'
    elif any(word in desc_lower for word in ['verbindung', 'beziehung', 'zusammenhang', 'relation', 'wechselwirkung']):
        return 'ZUSAMMENHANG'
    elif any(word in desc_lower for word in ['hypothese', 'annahme', 'behauptung', 'these', 'vermutung', 'theorie']):
        return 'HYPOTHESE'
    elif any(word in desc_lower for word in ['frage', 'fragestellung', 'problem', 'herausforderung', 'dilemma']):
        return 'FRAGE'
    elif any(word in desc_lower for word in ['intervention', 'eingriff', 'maßnahme', 'aktion', 'initiative']):
        return 'INTERVENTION'
    elif any(word in desc_lower for word in ['werkzeug', 'tool', 'instrument', 'hilfsmittel', 'technik', 'software']):
        return 'WERKZEUG'
    
    return 'UNKNOWN'

@dataclass
class Entity:
    name: str
    description: str
    source_file: str
    normalized_name: str = None
    entity_type: str = "UNKNOWN"
    folgen_sources: Set[str] = None
    run_id: str = ""  # Track which parallel run this came from
    
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
    run_id: str = ""  # Track which parallel run this came from
    
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

SYNTHESIS_PROMPT = """
Erstelle EINE umfassende Beschreibung für die Entität "{entity_name}" basierend auf diesen Varianten aus verschiedenen Extraktionen:

{descriptions}

AUFGABE:
- Kombiniere die wichtigsten Informationen zu einer präzisen, informativen Beschreibung
- Bewahre alle wertvollen Details auf, aber vermeide Redundanzen
- Verwende einen klaren, informativen Schreibstil
- Die finale Beschreibung kann länger sein als die Einzelbeschreibungen, wenn dadurch mehr Wert entsteht

Antworte nur mit der finalen Beschreibung (ohne "Version" oder andere Präfixe).
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

def parse_entities_and_relations(text: str, source_file: str, run_id: str) -> Tuple[List[Entity], List[Relation]]:
    """Parse entities and relations from the OpenAI response text."""
    entities = []
    relations = []
    
    try:
        logger.info(f"Parsing response text for {run_id}: {len(text)} characters")
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
                logger.info(f"Found ENTITIES section in {run_id}")
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
                            entity = Entity(name, desc, source_file)
                            entity.run_id = run_id
                            entities.append(entity)
                            logger.debug(f"Parsed entity in {run_id}: {name}")
            
            elif section.upper().startswith('RELATIONS'):
                current_section = 'relations'
                logger.info(f"Found RELATIONS section in {run_id}")
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
                            relation = Relation(source, target, short_desc, long_desc, source_file)
                            relation.run_id = run_id
                            relations.append(relation)
                            logger.debug(f"Parsed relation in {run_id}: {source} --> {target}")
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
                                    relation = Relation(source, target, short_desc, long_desc, source_file)
                                    relation.run_id = run_id
                                    relations.append(relation)
                                    logger.debug(f"Parsed relation (fallback) in {run_id}: {source} --> {target}")
    
    except Exception as e:
        logger.error(f"Error parsing entities and relations for {run_id}: {e}")
        logger.error(f"Full response text: {text}")
    
    return entities, relations

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,))
)
def call_openai_with_retry(client: OpenAI, prompt: str, context: str) -> str:
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
        logger.info(f"OpenAI API success for {context}: {len(response_text)} characters")
        return response_text
        
    except Exception as e:
        logger.error(f"OpenAI API error for {context}: {e}")
        raise

async def extract_single_chunk(client: OpenAI, chunk: str, context: str) -> Tuple[List[Entity], List[Relation]]:
    """Extract entities from a single chunk of text."""
    try:
        full_prompt = ENTITY_EXTRACTION_PROMPT + chunk
        response_text = call_openai_with_retry(client, full_prompt, context)
        
        entities, relations = parse_entities_and_relations(response_text, context.split('_run')[0], context)
        
        logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations from {context}")
        return entities, relations
        
    except Exception as e:
        logger.error(f"Failed to extract from chunk {context}: {e}")
        return [], []

async def extract_entities_parallel(client: OpenAI, text_chunks: List[str], file_path: pathlib.Path) -> Tuple[List[Entity], List[Relation]]:
    """Run 3 parallel extractions and merge results."""
    logger.info(f"Starting parallel extraction for {file_path.name} with {len(text_chunks)} chunks")
    
    # 1. Run 3 parallel extractions for each chunk
    tasks = []
    for run_id in range(PARALLEL_RUNS):
        for chunk_idx, chunk in enumerate(text_chunks):
            context = f"{file_path.name}_run{run_id+1}_chunk{chunk_idx+1}"
            task = extract_single_chunk(client, chunk, context)
            tasks.append(task)
    
    # Execute all extractions in parallel
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 2. Collect all entities and relations, filtering out exceptions
    all_entities = []
    all_relations = []
    successful_extractions = 0
    
    for result in all_results:
        if isinstance(result, Exception):
            logger.error(f"Parallel extraction failed: {result}")
            continue
            
        entities, relations = result
        all_entities.extend(entities)
        all_relations.extend(relations)
        successful_extractions += 1
    
    logger.info(f"Parallel extraction complete: {successful_extractions}/{len(tasks)} successful extractions")
    logger.info(f"Total extracted: {len(all_entities)} entities, {len(all_relations)} relations before deduplication")
    
    # 3. Deduplicate using advanced multi-level approach
    merged_entities = deduplicate_entities_advanced_multilevel(client, all_entities, file_path)
    merged_relations = deduplicate_relations(all_relations)
    
    logger.info(f"After deduplication: {len(merged_entities)} entities, {len(merged_relations)} relations")
    
    return merged_entities, merged_relations

def create_embeddings_multilevel(client: OpenAI, entities: List[Entity]) -> Dict[str, np.ndarray]:
    """Create multi-level embeddings: name, description, and name+description."""
    if not entities:
        return {'names': np.array([]), 'descriptions': np.array([]), 'combined': np.array([])}
    
    logger.info(f"Creating multi-level embeddings for {len(entities)} entities")
    
    # Prepare text lists
    names = [entity.name for entity in entities]
    descriptions = [entity.description for entity in entities]
    combined = [f"{entity.name}: {entity.description}" for entity in entities]
    
    # Create embeddings for each level
    embeddings = {}
    
    try:
        # Name embeddings
        logger.info("Creating name embeddings")
        embeddings['names'] = create_embeddings_batch(client, names, "names")
        
        # Description embeddings  
        logger.info("Creating description embeddings")
        embeddings['descriptions'] = create_embeddings_batch(client, descriptions, "descriptions")
        
        # Combined embeddings
        logger.info("Creating combined name+description embeddings")
        embeddings['combined'] = create_embeddings_batch(client, combined, "combined")
        
        logger.info("Multi-level embeddings creation complete")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating multi-level embeddings: {e}")
        return {'names': np.array([]), 'descriptions': np.array([]), 'combined': np.array([])}

def create_embeddings_batch(client: OpenAI, texts: List[str], level_name: str) -> np.ndarray:
    """Create embeddings for a batch of texts."""
    try:
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
            
            logger.debug(f"Created {level_name} embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return np.array(all_embeddings)
        
    except Exception as e:
        logger.error(f"Error creating {level_name} embeddings: {e}")
        return np.array([])

def find_similar_entities_multilevel(entities: List[Entity], embeddings: Dict[str, np.ndarray]) -> List[Tuple[int, int, float, str]]:
    """Find similar entities using multi-level embedding analysis."""
    logger.info(f"Starting multi-level similarity analysis for {len(entities)} entities")
    
    if len(entities) < 2:
        return []
    
    similar_pairs = []
    
    # Check each embedding level
    levels = ['names', 'descriptions', 'combined']
    
    for level in levels:
        if level not in embeddings or len(embeddings[level]) == 0:
            logger.warning(f"No embeddings available for level: {level}")
            continue
            
        logger.info(f"Analyzing {level} similarities")
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings[level])
        
        # Find similar pairs using type-specific thresholds
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
                    similar_pairs.append((i, j, similarity, level))
                    logger.info(f"Similar entities found via {level}: {entity_i.name} ({entity_i.entity_type}) <-> {entity_j.name} ({entity_j.entity_type}) (similarity: {similarity:.3f}, threshold: {threshold:.3f})")
    
    # Remove duplicates and sort by similarity (highest first)
    unique_pairs = {}
    for i, j, similarity, level in similar_pairs:
        key = (min(i, j), max(i, j))  # Normalize order
        if key not in unique_pairs or similarity > unique_pairs[key][2]:
            unique_pairs[key] = (i, j, similarity, level)
    
    result = list(unique_pairs.values())
    result.sort(key=lambda x: x[2], reverse=True)
    
    logger.info(f"Multi-level analysis complete: found {len(result)} unique similar pairs")
    return result

def group_similar_entities_by_clusters(entities: List[Entity], similar_pairs: List[Tuple[int, int, float, str]]) -> List[List[int]]:
    """Group entities into clusters based on similarity relationships."""
    logger.info(f"Grouping {len(entities)} entities into clusters based on {len(similar_pairs)} similarity pairs")
    
    # Create adjacency list
    graph = defaultdict(set)
    for i, j, similarity, level in similar_pairs:
        graph[i].add(j)
        graph[j].add(i)
    
    # Find connected components (clusters)
    visited = set()
    clusters = []
    
    for i in range(len(entities)):
        if i not in visited:
            # BFS to find all connected entities
            cluster = []
            queue = [i]
            while queue:
                current = queue.pop(0)
                if current not in visited:
                    visited.add(current)
                    cluster.append(current)
                    # Add all connected entities to queue
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
            
            clusters.append(cluster)
    
    # Filter out single-entity clusters (no merging needed)
    multi_entity_clusters = [cluster for cluster in clusters if len(cluster) > 1]
    
    logger.info(f"Found {len(multi_entity_clusters)} clusters with multiple entities requiring merging")
    for i, cluster in enumerate(multi_entity_clusters):
        entity_names = [entities[idx].name for idx in cluster]
        logger.info(f"Cluster {i+1}: {entity_names}")
    
    return multi_entity_clusters

def llm_merge_entity_cluster(client: OpenAI, entities: List[Entity], cluster_indices: List[int], file_path: pathlib.Path) -> List[Entity]:
    """Use LLM to intelligently merge a cluster of similar entities."""
    cluster_entities = [entities[i] for i in cluster_indices]
    
    if len(cluster_entities) == 1:
        return cluster_entities
    
    logger.info(f"LLM merging cluster of {len(cluster_entities)} entities: {[e.name for e in cluster_entities]}")
    
    # Prepare cluster information for LLM
    entity_descriptions = []
    for i, entity in enumerate(cluster_entities):
        entity_descriptions.append(f"Entity {i+1}: {entity.name} ({entity.entity_type})\nDescription: {entity.description}\nFOLGEN: {', '.join(sorted(entity.folgen_sources)) if entity.folgen_sources else 'None'}")
    
    prompt = f"""Du bist Experte für Organisationsentwicklung und sollst entscheiden, wie ähnliche Entitäten zusammengeführt werden sollen.

KONTEXT: Dies sind Entitäten aus Podcast-Transkripten von Benno und Fabian über Organisationsentwicklung und moderne Soziologie in Organisationen.

AUFGABE: Analysiere diese {len(cluster_entities)} ähnlichen Entitäten und erstelle finale konsolidierte Entitäten:

{chr(10).join(entity_descriptions)}

ANWEISUNGEN:
- Entscheide welche Entitäten wirklich identisch sind und zusammengeführt werden sollen
- Erstelle finale konsolidierte Entitäten mit optimalen Namen und umfassenden Beschreibungen
- Verschiedene Aspekte derselben Sache sollen zusammengeführt werden
- Wirklich unterschiedliche Konzepte sollen getrennt bleiben

AUSGABEFORMAT:
MERGED_ENTITIES:
EntityName (EntityType): Comprehensive description combining relevant information

MAPPING:
[finale_entität] <- [original_entity_1, original_entity_2, ...]

Antworte nur im angegebenen Format."""

    try:
        response = call_openai_with_retry(client, prompt, f"cluster_merge_{file_path.name}")
        
        # Parse the response to extract merged entities
        merged_entities = parse_llm_cluster_response(response, cluster_entities, file_path)
        
        logger.info(f"LLM cluster merge complete: {len(cluster_entities)} -> {len(merged_entities)} entities")
        return merged_entities
        
    except Exception as e:
        logger.error(f"Error in LLM cluster merging: {e}")
        # Fallback: return the entity with the longest description
        longest_entity = max(cluster_entities, key=lambda e: len(e.description))
        
        # Combine FOLGEN sources
        combined_folgen = set()
        for entity in cluster_entities:
            if entity.folgen_sources:
                combined_folgen.update(entity.folgen_sources)
        
        longest_entity.folgen_sources = combined_folgen
        longest_entity.run_id = f"fallback_cluster_of_{len(cluster_entities)}"
        return [longest_entity]

def parse_llm_cluster_response(response: str, original_entities: List[Entity], file_path: pathlib.Path) -> List[Entity]:
    """Parse LLM response for cluster merging."""
    try:
        lines = response.strip().split('\n')
        merged_entities = []
        
        in_entities_section = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('MERGED_ENTITIES:'):
                in_entities_section = True
                continue
            elif line.startswith('MAPPING:'):
                in_entities_section = False
                continue
            
            if in_entities_section and ':' in line:
                # Parse entity line: "EntityName (EntityType): Description"
                if '(' in line and ')' in line:
                    # Extract name, type, and description
                    paren_start = line.find('(')
                    paren_end = line.find(')')
                    colon_pos = line.find(':', paren_end)
                    
                    if paren_start > 0 and paren_end > paren_start and colon_pos > paren_end:
                        name = line[:paren_start].strip()
                        entity_type = line[paren_start+1:paren_end].strip()
                        description = line[colon_pos+1:].strip()
                        
                        # Combine all FOLGEN sources from original entities
                        combined_folgen = set()
                        for entity in original_entities:
                            if entity.folgen_sources:
                                combined_folgen.update(entity.folgen_sources)
                        
                        merged_entity = Entity(
                            name=name,
                            description=description,
                            source_file=file_path.stem,
                            entity_type=entity_type,
                            folgen_sources=combined_folgen,
                            run_id=f"llm_cluster_merged"
                        )
                        merged_entities.append(merged_entity)
        
        if merged_entities:
            return merged_entities
        else:
            # Fallback: couldn't parse, use original logic
            raise ValueError("Could not parse LLM response")
            
    except Exception as e:
        logger.error(f"Error parsing LLM cluster response: {e}")
        # Fallback: return longest entity with combined sources
        longest_entity = max(original_entities, key=lambda e: len(e.description))
        combined_folgen = set()
        for entity in original_entities:
            if entity.folgen_sources:
                combined_folgen.update(entity.folgen_sources)
        
        longest_entity.folgen_sources = combined_folgen
        return [longest_entity]

def deduplicate_entities_advanced_multilevel(client: OpenAI, entities: List[Entity], file_path: pathlib.Path) -> List[Entity]:
    """Advanced multi-level entity deduplication using embeddings and clustering."""
    logger.info(f"Starting advanced multi-level deduplication for {len(entities)} entities")
    
    if len(entities) < 2:
        return entities
    
    # Phase 1: Rule-based name grouping (as before)
    name_groups = defaultdict(list)
    for entity in entities:
        normalized = normalize_entity_name(entity.name)
        name_groups[normalized].append(entity)
    
    # Merge entities with identical names first
    pre_merged_entities = []
    for normalized_name, group in name_groups.items():
        if len(group) == 1:
            pre_merged_entities.append(group[0])
        else:
            # Merge entities with same name
            logger.info(f"Name-based merging {len(group)} entities: '{group[0].name}'")
            merged = synthesize_entity_descriptions(client, group, file_path)
            pre_merged_entities.append(merged)
    
    logger.info(f"After name-based merging: {len(entities)} -> {len(pre_merged_entities)} entities")
    
    if len(pre_merged_entities) < 2:
        return pre_merged_entities
    
    # Phase 2: Multi-level embedding analysis
    embeddings = create_embeddings_multilevel(client, pre_merged_entities)
    
    # Phase 3: Find similar entities across all levels
    similar_pairs = find_similar_entities_multilevel(pre_merged_entities, embeddings)
    
    if not similar_pairs:
        logger.info("No similar entities found via multi-level analysis")
        return pre_merged_entities
    
    # Phase 4: Group entities into clusters
    clusters = group_similar_entities_by_clusters(pre_merged_entities, similar_pairs)
    
    if not clusters:
        logger.info("No clusters found for merging")
        return pre_merged_entities
    
    # Phase 5: LLM-assisted cluster merging
    final_entities = []
    merged_indices = set()
    
    for cluster in clusters:
        merged_entities = llm_merge_entity_cluster(client, pre_merged_entities, cluster, file_path)
        final_entities.extend(merged_entities)
        merged_indices.update(cluster)
    
    # Add entities that weren't part of any cluster
    for i, entity in enumerate(pre_merged_entities):
        if i not in merged_indices:
            final_entities.append(entity)
    
    logger.info(f"Advanced multi-level deduplication complete: {len(entities)} -> {len(final_entities)} entities")
    return final_entities

def deduplicate_entities_hybrid(client: OpenAI, entities: List[Entity], file_path: pathlib.Path) -> List[Entity]:
    """Hybrid deduplication: rule-based grouping + LLM synthesis."""
    logger.info(f"Starting hybrid deduplication for {len(entities)} entities")
    
    # Phase 1: Group by normalized names (rule-based)
    name_groups = defaultdict(list)
    for entity in entities:
        normalized = normalize_entity_name(entity.name)
        name_groups[normalized].append(entity)
    
    logger.info(f"Grouped into {len(name_groups)} name groups")
    
    merged_entities = []
    
    for normalized_name, group in name_groups.items():
        if len(group) == 1:
            # Single entity, no merging needed
            merged_entities.append(group[0])
        else:
            # Multiple descriptions - use LLM to synthesize
            logger.info(f"Synthesizing {len(group)} variants of '{group[0].name}' from runs: {[e.run_id for e in group]}")
            merged_entity = synthesize_entity_descriptions(client, group, file_path)
            merged_entities.append(merged_entity)
    
    logger.info(f"Hybrid deduplication complete: {len(entities)} -> {len(merged_entities)} entities")
    return merged_entities

def synthesize_entity_descriptions(client: OpenAI, entities: List[Entity], file_path: pathlib.Path) -> Entity:
    """Use LLM to create one comprehensive description from multiple variants."""
    descriptions = [f"Version {i+1} (from {e.run_id}): {e.description}" for i, e in enumerate(entities)]
    
    prompt = SYNTHESIS_PROMPT.format(
        entity_name=entities[0].name,
        descriptions='\n'.join(descriptions)
    )
    
    try:
        response = call_openai_with_retry(client, prompt, f"synthesis_{entities[0].name}_{file_path.name}")
        
        # Combine FOLGEN sources from all variants
        combined_sources = set()
        for entity in entities:
            if entity.folgen_sources:
                combined_sources.update(entity.folgen_sources)
        
        # Use the most common entity type
        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity.entity_type] += 1
        most_common_type = max(type_counts, key=type_counts.get)
        
        logger.info(f"Synthesized description for '{entities[0].name}' from {len(entities)} variants")
        
        return Entity(
            name=entities[0].name,
            description=response.strip(),
            source_file=file_path.stem,
            folgen_sources=combined_sources,
            entity_type=most_common_type,
            run_id=f"synthesized_from_{len(entities)}_runs"
        )
        
    except Exception as e:
        logger.error(f"Failed to synthesize descriptions for '{entities[0].name}': {e}")
        # Fallback: use the longest description
        longest_entity = max(entities, key=lambda e: len(e.description))
        
        # Still combine FOLGEN sources
        combined_sources = set()
        for entity in entities:
            if entity.folgen_sources:
                combined_sources.update(entity.folgen_sources)
        
        longest_entity.folgen_sources = combined_sources
        longest_entity.run_id = f"fallback_from_{len(entities)}_runs"
        return longest_entity

def deduplicate_relations(relations: List[Relation]) -> List[Relation]:
    """Deduplicate relations using rule-based approach."""
    logger.info(f"Deduplicating {len(relations)} relations")
    
    # Group relations by (source, target) pair
    relation_groups = defaultdict(list)
    for relation in relations:
        key = (relation.source.lower().strip(), relation.target.lower().strip())
        relation_groups[key].append(relation)
    
    deduplicated_relations = []
    
    for (source, target), group in relation_groups.items():
        if len(group) == 1:
            deduplicated_relations.append(group[0])
        else:
            # Multiple relations between same entities - use the one with longest description
            best_relation = max(group, key=lambda r: len(r.long_desc))
            logger.info(f"Merged {len(group)} relations: {best_relation.source} --> {best_relation.target}")
            deduplicated_relations.append(best_relation)
    
    logger.info(f"Relation deduplication complete: {len(relations)} -> {len(deduplicated_relations)} relations")
    return deduplicated_relations

def extract_entities_from_file_parallel(client: OpenAI, file_path: pathlib.Path) -> Tuple[List[Entity], List[Relation]]:
    """Extract entities and relations from a single file using parallel approach."""
    logger.info(f"Starting parallel extraction for: {file_path.name}")
    
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
        
        # Run parallel extraction
        return asyncio.run(extract_entities_parallel(client, chunks, file_path))
        
    except Exception as e:
        logger.error(f"Error extracting entities from {file_path}: {e}")
        return [], []

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
    
    # Apply advanced multi-level deduplication 
    print("Starting advanced multi-level deduplication process...")
    merged_entities = deduplicate_entities_advanced_multilevel(client, all_entities, pathlib.Path("consolidated"))
    merged_relations = deduplicate_relations(all_relations)
    
    # Write consolidated output
    consolidated_path = dest_dir / "__ALL_ENTITIES_ALL_RELATIONS_PARALLEL.txt"
    write_entities_file(merged_entities, merged_relations, consolidated_path)
    
    print(f"\nConsolidation complete!")
    print(f"Files processed: {len(entity_files)}")
    print(f"Total entities: {len(merged_entities)} (after advanced multi-level deduplication)")
    print(f"Total relations: {len(merged_relations)}")
    print(f"Consolidated file: {consolidated_path}")

def full_extraction_mode(source_dir: pathlib.Path, dest_dir: pathlib.Path):
    """Full extraction mode: extract entities from source files using parallel approach AND consolidate."""
    logger.info("Running in full parallel extraction mode")
    
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
    
    logger.info(f"Found {len(augmented_files)} augmented files to process with {PARALLEL_RUNS}-parallel extraction")
    
    # Phase 1: Extract entities from each file using parallel approach
    all_entities = []
    all_relations = []
    
    for file_path in tqdm(augmented_files, desc="Parallel extraction", unit="file"):
        entities, relations = extract_entities_from_file_parallel(client, file_path)
        
        if entities or relations:
            # Write individual file results
            output_file = dest_dir / (file_path.stem + ".parallel.entities.txt")
            write_entities_file(entities, relations, output_file)
            
            # Add to global collections
            all_entities.extend(entities)
            all_relations.extend(relations)
        else:
            # Create detailed error file with diagnostics
            error_file = dest_dir / (file_path.stem + ".parallel.error.txt")
            with open(error_file, 'w', encoding='utf-8') as f:
                f.write(f"Failed to extract entities from {file_path.name} using parallel approach\n")
                f.write(f"Timestamp: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', (), None))}\n")
                f.write(f"Parallel runs: {PARALLEL_RUNS}\n")
                f.write(f"File size: {file_path.stat().st_size} bytes\n")
                
                try:
                    content = extract_content_from_augmented_file(file_path)
                    f.write(f"Content length: {len(content)} characters\n")
                    f.write(f"Content preview: {content[:500]}...\n")
                except Exception as inner_e:
                    f.write(f"Error reading content: {inner_e}\n")
    
    logger.info(f"Phase 1 complete: Extracted {len(all_entities)} entities and {len(all_relations)} relations total")
    
    if not all_entities:
        logger.warning("No entities extracted, exiting")
        return
    
    # Phase 2: Global advanced multi-level deduplication across all files
    logger.info("Starting Phase 2: Global advanced multi-level deduplication across all files")
    final_entities = deduplicate_entities_advanced_multilevel(client, all_entities, pathlib.Path("global_consolidation"))
    final_relations = deduplicate_relations(all_relations)
    
    # Write consolidated file
    consolidated_file = dest_dir / "__ALL_ENTITIES_ALL_RELATIONS_PARALLEL.txt"
    write_entities_file(final_entities, final_relations, consolidated_file)
    
    logger.info("Parallel processing complete!")
    print(f"Parallel entity extraction complete!")
    print(f"Individual files: {len(augmented_files)} processed with {PARALLEL_RUNS}-parallel runs each")
    print(f"Total entities: {len(final_entities)} (after global advanced multi-level deduplication)")
    print(f"Total relations: {len(final_relations)}")
    print(f"Consolidated file: {consolidated_file}")
    print(f"Approach: {PARALLEL_RUNS} parallel LLM calls per chunk with multi-level embedding analysis and cluster merging")

def main():
    # Parse arguments
    args = sys.argv[1:]
    
    # Handle --help
    if "--help" in args or "-h" in args:
        print("Usage: python 4b_batch_parallel_extract_entities_from_augmented_txts.py [ARGUMENTS]")
        print()
        print("Modes:")
        print("  No arguments:")
        print("    Extract entities from data/4_augmented/ -> data/5_entities/")
        print("    Source folder must exist with *.augmented.txt files")
        print()
        print("  Two arguments <source> <target>:")
        print("    Extract entities from <source> -> <target>")
        print("    Source folder must exist with *.augmented.txt files")
        print("    Target folder will be created if it doesn't exist")
        print()
        print("  One argument <target>:")
        print("    Consolidate-only mode: merge existing *.entities.txt files in <target>")
        print("    Target folder must exist and contain *.entities.txt files")
        print("    Creates __ALL_ENTITIES_ALL_RELATIONS_PARALLEL.txt")
        print()
        print("Examples:")
        print("  python 4b_batch_parallel_extract_entities_from_augmented_txts.py")
        print("  python 4b_batch_parallel_extract_entities_from_augmented_txts.py custom_src/ custom_target/")
        print("  python 4b_batch_parallel_extract_entities_from_augmented_txts.py data/5_entities/")
        sys.exit(0)
    
    # Parse arguments based on count
    if len(args) == 0:
        # Mode 1: Default directories - full extraction
        source_dir = pathlib.Path("data/4_augmented").expanduser()
        dest_dir = pathlib.Path("data/5_entities").expanduser()
        
        if not source_dir.exists():
            print(f"❌ Error: Source directory not found: {source_dir}")
            print("Expected: data/4_augmented/ with *.augmented.txt files")
            print("Use --help for usage information")
            sys.exit(1)
        
        print(f"Mode 1: Full extraction using default directories")
        print(f"Source: {source_dir} -> Target: {dest_dir}")
        return full_extraction_mode(source_dir, dest_dir)
    
    elif len(args) == 1:
        # Mode 3: Consolidate-only mode
        dest_dir = pathlib.Path(args[0]).expanduser()
        
        if not dest_dir.exists():
            print(f"❌ Error: Target directory not found: {dest_dir}")
            print("For consolidate-only mode, target directory must exist with *.entities.txt files")
            print("Use --help for usage information")
            sys.exit(1)
        
        # Check for existing entity files
        entity_files = list(dest_dir.glob("*.entities.txt"))
        if not entity_files:
            print(f"❌ Error: No *.entities.txt files found in: {dest_dir}")
            print("For consolidate-only mode, target directory must contain *.entities.txt files")
            print("Use --help for usage information")
            sys.exit(1)
        
        print(f"Mode 3: Consolidate-only mode")
        print(f"Target: {dest_dir} (found {len(entity_files)} *.entities.txt files)")
        return consolidate_only_mode(dest_dir)
    
    elif len(args) == 2:
        # Mode 2: Custom source and target - full extraction
        source_dir = pathlib.Path(args[0]).expanduser()
        dest_dir = pathlib.Path(args[1]).expanduser()
        
        if not source_dir.exists():
            print(f"❌ Error: Source directory not found: {source_dir}")
            print("Source directory must exist with *.augmented.txt files")
            print("Use --help for usage information")
            sys.exit(1)
        
        # Check for augmented files
        augmented_files = list(source_dir.glob("*.augmented.txt"))
        if not augmented_files:
            print(f"❌ Error: No *.augmented.txt files found in: {source_dir}")
            print("Source directory must contain *.augmented.txt files")
            print("Use --help for usage information")
            sys.exit(1)
        
        print(f"Mode 2: Full extraction with custom directories")
        print(f"Source: {source_dir} -> Target: {dest_dir} (found {len(augmented_files)} *.augmented.txt files)")
        return full_extraction_mode(source_dir, dest_dir)
    
    else:
        print("❌ Error: Invalid number of arguments")
        print()
        print("Valid usage:")
        print("  python 4b_batch_parallel_extract_entities_from_augmented_txts.py                    # Default extraction")
        print("  python 4b_batch_parallel_extract_entities_from_augmented_txts.py <source> <target>  # Custom extraction")
        print("  python 4b_batch_parallel_extract_entities_from_augmented_txts.py <target>           # Consolidate only")
        print()
        print("Use --help for detailed usage information")
        sys.exit(1)

if __name__ == "__main__":
    main()