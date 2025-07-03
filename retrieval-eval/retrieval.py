import re
import random

random.seed(42)

from tqdm import tqdm
from qid_mapping import (
    load_entity_maps, 
    get_qid_for_entity
)
from elasticsearch import Elasticsearch
from typing import List, Optional, Dict, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queries import (
    build_two_entity_alias_search_query,
    build_entity_alias_search_query,
    build_two_entity_search_query,
    build_entity_search_query
)

from constants import (
    CASE_SENSITIVE_INDEX_NAME,
    CASE_INSENSITIVE_INDEX_NAME,
    DEFAULT_METADATA_THRESHOLDS
)

def get_entity_chunk_ids_by_string_search(
    entity: Union[str, Tuple[str, str]],
    index: str = CASE_SENSITIVE_INDEX_NAME,
    es: Optional[Elasticsearch] = None,
    batch_size: int = 10000,
    base_query_func: Optional[callable] = None,
    filter_func: Optional[callable] = None,
) -> List[int]:
    """
    Get chunk IDs for an entity or pair of entities
    
    Args:
        entity: Entity name (str) or entity pair (tuple of two strings) to search for
        index: Name of the Elasticsearch index to search
        es: Optional Elasticsearch client instance
        batch_size: Number of documents to fetch per request
        base_query_func: Query function to use (if not specified, will be determined by entity type and search_with_aliases)
    
    Returns:
        List of chunk IDs where the entity/entities appear
    """
    # Use provided ES client or initialize from global
    es = es or globals().get('es')
    if es is None:
        raise ValueError("Elasticsearch client not provided")
    
    # Set up query parameters
    base_query = base_query_func(entity, filter_func)
    # TODO(Daniela): check if we need to set source.
    base_query["_source"] = ["chunk_id"]  # Only return chunk_id field
    base_query["size"] = batch_size
    
    chunk_ids_with_scores = []
    chunk_to_matching_spans = {}

    def process_hits(hits):
        for hit in hits:
            spans = []
            chunk_id = hit["_source"]["chunk_id"]

            chunk_ids_with_scores.append((chunk_id, hit["_score"]))
            
            # Keep track of position shifts due to tag removal
            offset_shift = 0  # Track how much shorter the text becomes after each removal
            text = hit["highlight"]["text"][0]
            pattern = re.compile(r'<em>(.*?)</em>')
            for match in pattern.finditer(text):
                raw_start, _ = match.span()
                phrase = match.group(1)

                # Compute new start and end after previous shifts
                adjusted_start = raw_start - offset_shift
                adjusted_end = adjusted_start + len(phrase)
                
                # Each <em>...</em> contributes 9 characters removed ("<em>" + "</em>")
                offset_shift += len(match.group(0)) - len(phrase)

                spans.append({
                    "char_start": adjusted_start, 
                    "char_end": adjusted_end, 
                    "text_mention": phrase, 
                })
            
            chunk_to_matching_spans[chunk_id] = spans

    # Initial scroll request
    scroll_response = es.search(
        index=index,
        body=base_query,
        scroll="1000m"
    )
    
    # Process initial batch
    scroll_id = scroll_response['_scroll_id']
    hits = scroll_response['hits']['hits']
    process_hits(hits)
    
    # Continue fetching results until no more hits
    while hits:
        scroll_response = es.scroll(
            scroll_id=scroll_id, 
            scroll='1000m',
        )
        
        hits = scroll_response['hits']['hits']
        if not hits:
            break
        
        process_hits(hits)
    
    # Clean up scroll
    es.clear_scroll(scroll_id=scroll_id)
    
    chunk_ids_with_scores = sorted(chunk_ids_with_scores, key=lambda x: x[1], reverse=True)
    
    # Extract only the chunk_id from each tuple, as the scores where only needed for sorting
    chunk_ids = [x[0] for x in chunk_ids_with_scores]

    return {"chunks": chunk_ids, "spans": chunk_to_matching_spans}

def search_entities_parallel_by_string_search(
    entity_names: List[Union[str, Tuple[str, str]]], 
    es_client: Optional[Elasticsearch] = None,
    index: str = CASE_SENSITIVE_INDEX_NAME,
    batch_size: int = 10000,
    max_workers: int = 8,
    search_with_aliases: bool = True,
    show_progress: bool = False,
    filter_func: Optional[callable] = None,
) -> Dict[Union[str, Tuple[str, str]], List[int]]:
    """
    Search for multiple entities in parallel using ThreadPoolExecutor
    
    Args:
        entity_names: List of entity names (strings) or entity pairs (tuples)
        index: Name of the Elasticsearch index to search
        batch_size: Number of documents to fetch per request
        max_workers: Maximum number of parallel threads
        search_with_aliases: Whether to search with aliases
        
    Returns:
        Dictionary mapping entity names or pairs to chunk_ids
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use a list to track futures and their corresponding entities
        futures = []
        
        for entity in entity_names:
            base_query_func = build_entity_alias_search_query if search_with_aliases else build_entity_search_query
            
            # Submit the search task and store the future with its entity
            future = executor.submit(
                get_entity_chunk_ids_by_string_search, 
                entity, 
                index, 
                es_client,  # Use default ES client
                batch_size,
                base_query_func,
                filter_func,
            )
            futures.append((future, entity))
        
        # Map futures to their entity keys for lookup
        future_to_entity = {future: entity for future, entity in futures}

        # Choose iterator: tqdm-wrapped or plain
        iterator = as_completed(future_to_entity)
        if show_progress:
            try:
                iterator = tqdm(iterator, total=len(futures), desc="String search")
            except ImportError:
                print("tqdm not installed, proceeding without progress bar.")

        for future in iterator:
            entity_key = future_to_entity[future]
            try:
                chunk_ids = future.result()
                results[entity_key] = chunk_ids
            except Exception as e:
                print(f"Error processing {entity_key}: {e}")
                results[entity_key] = []
    
    return results

def find_entities_chunk_ids_by_string_search(
    entity_names: Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]], 
    es_client: Optional[Elasticsearch] = None,
    index: str = CASE_SENSITIVE_INDEX_NAME,
    batch_size: int = 10000,
    max_workers: int = 8,
    search_with_aliases: bool = True,
    show_progress: bool = False,
    top_k: Optional[List[int]] = None,
    sample_k: Optional[int] = None,
    filter_func: Optional[callable] = None,
):
    """
    Search for multiple entity names or entity pairs in parallel and display results
    
    Args:
        entity_names: A single entity name, list of entity names, or list of entity pairs (tuples)
        batch_size: Number of documents to fetch per request
        max_workers: Maximum number of parallel threads
        search_with_aliases: Whether to search with aliases
        top_k: Limit results to top k chunks per entity
    """
    # Convert single entity to list
    if isinstance(entity_names, (str, tuple)):
        entity_names = [entity_names]
    
    # Execute parallel search
    results = search_entities_parallel_by_string_search(
        entity_names=entity_names,
        es_client=es_client,
        index=index,
        batch_size=batch_size,
        max_workers=max_workers,
        search_with_aliases=search_with_aliases,
        show_progress=show_progress,
        filter_func=filter_func,
    )
    
    for entity, result in results.items():
        chunks_by_k = {}
        ranked_chunks = [(rank, item) for rank, item in enumerate(result["chunks"])]
        
        if sample_k is not None and top_k is not None:
            for k in top_k:
                if k > len(ranked_chunks):
                    continue
                if sample_k >= k:
                    chunks_by_k[k] = ranked_chunks[:k]
                else:
                    chunks_by_k[k] = random.sample(ranked_chunks[:k], sample_k)
            results[entity]["chunks"] = chunks_by_k

    return results

def find_entities_chunk_ids_by_metadata(
    entity_names: Union[str, Tuple[str, str], List[Union[str, Tuple[str, str]]]],
    index: str = CASE_SENSITIVE_INDEX_NAME,
    es_client: Optional[Elasticsearch] = None,
    thresholds: Dict = {},
    show_progress: bool = False,
    top_k_metadata: Optional[List[int]] = None, 
    top_k_sample: Optional[int] = None,
    batch_size: int = 1000,
) -> Dict[Union[str, Tuple[str, str]], Union[List[int], Dict[str, List[int]]]]:
    """
    Find chunk indices where specific entities appear based on metadata criteria.
    Uses only the explicitly provided non-None thresholds.
    """
    es_client = es_client or globals().get('es')
    if es_client is None:
        raise ValueError("Elasticsearch client not provided.")

    load_entity_maps() 

    if isinstance(entity_names, str):
        entities_to_process = [entity_names]
    elif isinstance(entity_names, tuple) and len(entity_names) == 2 and isinstance(entity_names[0], str):
        entities_to_process = [entity_names] 
    elif isinstance(entity_names, list):
        entities_to_process = entity_names
    else:
        raise ValueError("entity_names must be str, tuple of two str, or list of these.")

    def process_entity(entity_input):
        # Handles both single entities and pairs
        if isinstance(entity_input, tuple) and len(entity_input) == 2:
            entity1_label, entity2_label = entity_input
            res1 = find_entities_chunk_ids_by_metadata(
                entity1_label, index, es_client, 
                thresholds=thresholds,
                top_k_metadata=None, 
                batch_size=batch_size
            )
            res2 = find_entities_chunk_ids_by_metadata(
                entity2_label, index, es_client, 
                thresholds=thresholds,
                top_k_metadata=None, 
                batch_size=batch_size
            )
            entity1_chunks = set(res1.get(entity1_label, []))
            entity2_chunks = set(res2.get(entity2_label, []))
            intersection_chunks = sorted(list(entity1_chunks.intersection(entity2_chunks)))
            return entity_input, intersection_chunks

        entity_label = entity_input
        qid = get_qid_for_entity(entity_label)
        if not qid:
            print(f"Warning: QID not found for entity label '{entity_label}'. Skipping metadata search.")
            return entity_label, []

        # Add all relevant score fields to _source
        should_clauses = []
        for source, threshold in thresholds.items():
            clause = {
                "range": {
                    f"entities.candidates.scores_by_source.{source}": {
                        "gte": threshold
                    }
                }
            }
            should_clauses.append(clause)

        query_body = {
            "query": {
                "nested": {
                "path": "entities",
                "query": {
                    "nested": {
                    "path": "entities.candidates",
                    "query": {
                        "bool": {
                        "must": [
                            { "term": { "entities.candidates.qid": qid } },
                            {
                            "bool": {
                                "should": should_clauses,
                                "minimum_should_match": 1
                            }
                            }
                        ]
                        }
                    }
                    }
                },
                "inner_hits": {
                    "name": "matching_entities",
                    "_source": [
                    "entities.char_start",
                    "entities.char_end",
                    "entities.text_mention",
                    "entities.candidates.qid",
                    "entities.candidates.aggregated_score",
                    "entities.candidates.scores_by_source.hyperlinks",
                    "entities.candidates.scores_by_source.entity_linking",
                    "entities.candidates.scores_by_source.coref",
                    "entities.candidates.scores_by_source.coref_cluster"
                    ],
                    "size": 100
                }
                }
            },
            "size": 10000,
            "_source": [
                "chunk_id"
            ]
        }

        chunk_ids_with_scores = {}
        chunk_to_matching_spans = {}

        scroll_resp = es_client.search(index=index, body=query_body, scroll="1000m")
        scroll_id = scroll_resp.get('_scroll_id')
        
        while True:
            hits = scroll_resp['hits']['hits']
            if not hits:
                break
            for hit in hits:
                chunk_id = hit['_source']['chunk_id']
                # Find the candidate for this qid
                spans = []
                if "inner_hits" in hit and "matching_entities" in hit["inner_hits"] and \
                    "hits" in hit["inner_hits"]["matching_entities"] and \
                    "hits" in hit["inner_hits"]["matching_entities"]["hits"]:
                    for span in [h["_source"] for h in hit["inner_hits"]["matching_entities"]["hits"]["hits"]]:
                        spans.append({
                            "char_start": span["char_start"], 
                            "char_end": span["char_end"], 
                            "text_mention": span["text_mention"], 
                            "candidates": [c for c in span["candidates"] if c["qid"] == qid]
                        })
                
                chunk_to_matching_spans[chunk_id] = spans
                
                for candidate in [c for s in spans for c in s["candidates"]]:
                    # Associiate the maximim mention score with the chunk
                    candidate_agg_score = candidate.get("aggregated_score", 0.0)
                    chunk_ids_with_scores[chunk_id] = max(chunk_ids_with_scores.get(chunk_id, 0.0), candidate_agg_score)
            
            if not scroll_id: break
            
            try:
                scroll_resp = es_client.scroll(scroll_id=scroll_id, scroll="2m")
            except Exception as e:
                print(f"Error during scroll for {entity_label}, scroll_id {scroll_id}: {e}")
                break
        
        if scroll_id:
            try:
                es_client.clear_scroll(scroll_id=scroll_id)
            except Exception as e:
                print(f"Error clearing scroll_id '{scroll_id}': {e}")

        sorted_chunks = sorted(chunk_ids_with_scores.items(), key=lambda item: (-item[1], item[0]))
        sorted_chunks = [item[0] for item in sorted_chunks]
        ranked_chunks = [(rank, item) for rank, item in enumerate(sorted_chunks)]
        chunks_by_k = {}
        for k in top_k_metadata:
            if k > len(ranked_chunks):
                continue
            if top_k_sample >= k:
                chunks_by_k[k] = ranked_chunks[:k]
            else:
                chunks_by_k[k] = random.sample(ranked_chunks[:k], top_k_sample)
        
        return entity_label, chunks_by_k, chunk_to_matching_spans

    results = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_entity, entity) for entity in entities_to_process]
        iterator = as_completed(futures)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(futures), desc="Metadata search")
            except ImportError:
                print("tqdm not installed, proceeding without progress bar.")
        for future in iterator:
            entity_label, chunks, spans = future.result()
            results[entity_label] = {"chunks": chunks, "spans": spans}

    return results
