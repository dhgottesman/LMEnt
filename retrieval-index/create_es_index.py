import sys
import random
import time
import torch
import concurrent.futures
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pathlib import Path
from tqdm import tqdm
import threading
from queue import Queue, Empty
from threading import Thread

# Make sure to set http.max_content_length: 1GB in elasticsearch.yml.

# Add the OLMo-core path to system path
sys.path.append("/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/src")

from examples.kas.train import build_config, seed_all, set_random_seeds
from transformers import AutoTokenizer
from olmo_core.data import TokenizerConfig, NumpyDatasetConfig, NumpyDatasetType
from olmo_core.data.numpy_dataset import VSLCurriculumType, VSLCurriculumConfig

# --- Process-local globals for dataset and tokenizer ---
DATASET = None
TOKENIZER = None

# Mapping for the case-insensitive index
CASE_INSENSITIVE_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index": {
            "similarity": {
                "default": {
                    "type": "BM25"
                }
            }
        }
    },
    "mappings": {
        "dynamic": False,
        "properties": {
            "chunk_id": {"type": "integer"},
            "article_id": {"type": "integer"},
            "title": {"type": "text"},
            "metadata_source": {"type": "text"},
            "text": {
                "type": "text",
                "fields": {
                    "raw": {"type": "keyword"}
                }
            },
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "entities": {
                "type": "nested",
                "properties": {
                    "char_start": {"type": "integer"},
                    "char_end": {"type": "integer"},
                    "text_mention": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword"}
                        }
                    },
                    "candidates": {
                        "type": "nested",
                        "properties": {
                            "qid": {"type": "keyword"},
                            "name": {
                                "type": "text",
                                "fields": {
                                    "raw": {"type": "keyword"}
                                }
                            },
                            "aggregated_score": {"type": "float"},
                            "scores_by_source": {
                                "type": "object",
                                "properties": {
                                    "hyperlinks": {"type": "float"},
                                    "entity_linking": {"type": "float"},
                                    "coref": {"type": "float"},
                                    "coref_cluster": {"type": "float"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

# MAPPING for the case sensitive index
CASE_SENSITIVE_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index": {
            "similarity": {
                "default": {
                    "type": "BM25"
                }
            }
        },
        "analysis": {
            "analyzer": {
                "default": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": []
                }
            }
        }
    },
    "mappings": {
        "dynamic": False,
        "properties": {
            "chunk_id": {"type": "integer"},
            "article_id": {"type": "integer"},
            "title": {"type": "text"},
            "metadata_source": {"type": "text"},
            "text": {"type": "text"},
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "entities": {
                "type": "nested",
                "properties": {
                    "char_start": {"type": "integer"},
                    "char_end": {"type": "integer"},
                    "text_mention": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword"}
                        }
                    },
                    "candidates": {
                        "type": "nested",
                        "properties": {
                            "qid": {"type": "keyword"},
                            "name": {
                                "type": "text",
                                "fields": {
                                    "raw": {"type": "keyword"}
                                }
                            },
                            "aggregated_score": {"type": "float"},
                            "scores_by_source": {
                                "type": "object",
                                "properties": {
                                    "hyperlinks": {"type": "float"},
                                    "entity_linking": {"type": "float"},
                                    "coref": {"type": "float"},
                                    "coref_cluster": {"type": "float"}
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


def get_esclient(scheme="https", host="localhost", port=9200):
    host = "132.67.130.202"
    return Elasticsearch(
        f"{scheme}://{host}:{port}", 
        basic_auth=("elastic", "G*+2PQqsqZ2NCn5aCSoA"), 
        request_timeout=300, 
        max_retries=10, 
        retry_on_timeout=True,
        verify_certs=False,
        ssl_show_warn=False
    )

def process_init(dataset_config_dict):
    global DATASET, TOKENIZER
    from olmo_eval import HFTokenizer

    tokenizer_config = TokenizerConfig.dolma2()
    try:
        dataset_config = NumpyDatasetConfig(**dataset_config_dict)
        DATASET = dataset_config.build()
        TOKENIZER = HFTokenizer(
                tokenizer_config.identifier,
                pad_token_id=tokenizer_config.pad_token_id,
                eos_token_id=tokenizer_config.eos_token_id,
                bos_token_id=tokenizer_config.bos_token_id,
            )
    except Exception as e:
        print(f"Error in worker process_init: {e}", flush=True)
        import traceback
        traceback.print_exc()

def validate_and_normalize_entity_data(entity_list_from_metadata: list) -> list:
    """
    Validates and normalizes the entity data structure before indexing.
    Ensures all required fields are present and have correct types.
    Assumes 'name' is always present in candidate data.
    Ensures scores_by_source has float values or are omitted.
    """
    validated_entities = []
    if not isinstance(entity_list_from_metadata, list):
        return []

    for mention_data in entity_list_from_metadata:
        if not isinstance(mention_data, dict): continue

        char_start = mention_data.get("char_start")
        char_end = mention_data.get("char_end")
        text_mention_val = mention_data.get("text_mention")

        if not (isinstance(char_start, int) and isinstance(char_end, int) and isinstance(text_mention_val, str)):
            print(f"Skipping invalid mention data (missing/wrong type for char_start/end/text_mention): {mention_data}", flush=True)
            continue
        
        validated_mention = {
            "char_start": char_start,
            "char_end": char_end,
            "text_mention": text_mention_val,
            "candidates": []
        }

        raw_candidates = mention_data.get("candidates", [])
        if not isinstance(raw_candidates, list):
            print(f"Skipping mention with invalid candidates format: {mention_data}", flush=True)
            continue

        for cand_data in raw_candidates:
            if not isinstance(cand_data, dict):
                print(f"Skipping candidate with invalid format: {cand_data}", flush=True)
                continue

            qid = cand_data.get("qid")
            name = cand_data.get("name")

            if not qid and isinstance(qid, str):
                print(f"Skipping candidate with missing/invalid QID: {cand_data}", flush=True)
                continue # QID is mandatory for a candidate

            agg_score = cand_data.get("aggregated_score", 0.0)
            try:
                agg_score = float(agg_score)
            except (ValueError, TypeError):
                agg_score = 0.0

            scores_by_source_validated = {}
            raw_scores_by_source = cand_data.get("scores_by_source", {})
            if isinstance(raw_scores_by_source, dict):
                for source_key, score_val in raw_scores_by_source.items():
                    try:
                        scores_by_source_validated[source_key] = float(score_val)
                    except (ValueError, TypeError):
                        print(f"Invalid score for source '{source_key}' in candidate {qid}: {score_val}. Skipping this score.", flush=True)
                        # Optionally set to 0.0 or just skip if conversion fails
                        # scores_by_source_validated[source_key] = 0.0
                        pass 

            validated_candidate = {
                "qid": qid,
                "name": name,
                "aggregated_score": agg_score,
                "scores_by_source": scores_by_source_validated
            }
            validated_mention["candidates"].append(validated_candidate)
        
        if validated_mention["candidates"]:
            validated_entities.append(validated_mention)
            
    return validated_entities

def fetch_and_prepare(idx):
    global DATASET, TOKENIZER
    if DATASET is None or TOKENIZER is None:
        print(f"Error: DATASET or TOKENIZER not initialized in worker for chunk {idx}. Skipping.", flush=True)
        return idx, None
    try:
        chunk_data_from_dataset = DATASET[idx]
        entity_list = chunk_data_from_dataset["metadata"].get("entities", [])
        entities_structured_for_index = validate_and_normalize_entity_data(entity_list)

        body = {
            'chunk_id': idx,
            'article_id': chunk_data_from_dataset["metadata"].get("id"),
            'title': chunk_data_from_dataset["metadata"].get("title"),
            'metadata_source': chunk_data_from_dataset["metadata"].get("src"),
            'text': TOKENIZER.decode(chunk_data_from_dataset["input_ids"].tolist()),
            'start': chunk_data_from_dataset["metadata"].get("start"),
            'end': chunk_data_from_dataset["metadata"].get("end"),
            'entities': entities_structured_for_index,
        }
        return idx, body
    except Exception as e:
        print(f"Error in fetch_and_prepare for chunk {idx}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        return idx, None

def main():
    print("Starting indexing process...", flush=True)
    work_dir = Path("/home/morg/students/gottesman3/knowledge-analysis-suite/OLMo-core/hp_final/dataset-cache")
    print(f"Using work directory: {work_dir}", flush=True)

    tokenizer_config_obj = TokenizerConfig.dolma2()
    include_instance_metadata = True
    dataset_config = NumpyDatasetConfig.glob(
        "/home/morg/students/gottesman3/knowledge-analysis-suite/dolma/python/final_tokenizations_with_offsets/no_special/*.npy",
        name=NumpyDatasetType.kas_vsl,
        max_sequence_length=2048,
        min_sequence_length=64,
        vsl_curriculum=VSLCurriculumConfig(name=VSLCurriculumType.grow_p2, num_cycles=8, balanced=False),
        tokenizer=tokenizer_config_obj,
        work_dir=str(work_dir),
        include_instance_metadata=include_instance_metadata,
    )

    dataset = dataset_config.build()
    
    es = get_esclient()
    index_name = 'enwiki_case_sensitive' # or enwiki
    print(f"Target index: {index_name}", flush=True)

    MAPPING = CASE_SENSITIVE_MAPPING if index_name == 'enwiki_case_sensitive' else CASE_INSENSITIVE_MAPPING

    if es.indices.exists(index=index_name):
        print(f"Index: {index_name} already exists. For a clean run, please delete it first or use a new index name.", flush=True)
        return
    else:
        print(f"Creating index: {index_name} with new mapping...", flush=True)
        es.indices.create(index=index_name, body=MAPPING)

    # Update the index settings nested docs
    new_limit = 100000000
    if es.indices.put_settings(index=index_name, body={"index.mapping.nested_objects.limit": new_limit}).get("acknowledged"):
        print(f"Updated nested object limit to {new_limit} for index '{index_name}'", flush=True)
    else:
        print("Failed to update nested object limit.", flush=True)

    dataset_config_dict = dataset_config.__dict__
    tokenizer_identifier = tokenizer_config_obj.identifier
    
    # === Bulk workers and queue setup ===
    bulk_queue = Queue()
    BULK_WORKERS = 4
    BULK_BATCH_SIZE = 250
    MAX_BULK_RETRIES = 5

    def bulk_worker(es_client):
        while True:
            try:
                actions = bulk_queue.get(timeout=10)
            except Empty:
                continue
            if actions is None:
                break
            for attempt in range(MAX_BULK_RETRIES):
                try:
                    _, errors = bulk(es_client, actions, raise_on_error=False, request_timeout=300)
                    if errors:
                        print(f"Bulk error batch of size {len(errors)} on attempt {attempt+1}")
                    break
                except Exception as e:
                    wait_time = 2 ** attempt
                    print(f"[bulk_worker] Bulk attempt {attempt+1} failed: {e}. Retrying in {wait_time}s", flush=True)
                    time.sleep(wait_time)
            else:
                print("[bulk_worker] Giving up on a batch after max retries.", flush=True)
            bulk_queue.task_done()

    bulk_threads = []
    for _ in range(BULK_WORKERS):
        t = Thread(target=bulk_worker, args=(es,), daemon=True)
        t.start()
        bulk_threads.append(t)

    # === Processing loop ===
    start = 0
    total_chunks = len(dataset)
    print(f"Total chunks in dataset: {total_chunks}", flush=True)

    start_time = time.time()
    processed = 0
    bulk_actions = []
    max_workers = 128

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=process_init,
        initargs=(dataset_config_dict)
    ) as executor:
        future_to_idx = {
            executor.submit(fetch_and_prepare, idx + start): idx + start
            for idx in range(min(total_chunks, max_workers * 2))
        }

        next_idx = max(future_to_idx.values()) + 1

        while future_to_idx:
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                del future_to_idx[future]

                try:
                    idx_processed, body = future.result()
                    if body is not None:
                        if not es.exists(index=index_name, id=idx_processed):
                            bulk_actions.append({
                                "_index": index_name,
                                "_id": idx_processed,
                                "_source": body
                            })

                    processed += 1
                    if len(bulk_actions) >= BULK_BATCH_SIZE:
                        bulk_queue.put(bulk_actions)
                        bulk_actions = []

                    if processed % BULK_BATCH_SIZE == 0:
                        print(f"Processed {processed} chunks in {time.time() - start_time:.2f}s", flush=True)

                except Exception as e:
                    print(f"Future failed for chunk {idx}: {e}", flush=True)

                # When a chunk is processed, move on to processing the next one. 
                if next_idx < total_chunks:
                    future_to_idx[executor.submit(fetch_and_prepare, next_idx)] = next_idx
                    next_idx += 1

        if bulk_actions:
            bulk_queue.put(bulk_actions)

    # Signal and join bulk threads
    for _ in bulk_threads:
        bulk_queue.put(None)
    bulk_queue.join()
    for t in bulk_threads:
        t.join()

    print(f"Indexing complete for {processed} chunks. Total time: {time.time() - start_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()