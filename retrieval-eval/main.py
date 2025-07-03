import re
import sys
import csv
import datetime
import pandas as pd
from google import genai
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Optional, Dict, Tuple, Union
from elasticsearch import Elasticsearch
from es_client import get_esclient
from retrieval import find_entities_chunk_ids_by_metadata, find_entities_chunk_ids_by_string_search
from llm_eval import generate
from qid_mapping import get_qid_for_entity, load_aliases, generate_suffix_variations
from cache import (
    initialize_cache, 
    initialize_overall_status_cache, 
    get_overall_status_from_cache, 
    save_overall_status_to_cache,
    get_cached_text
)
from queries import filter_short_aliases

from constants import (
    CASE_SENSITIVE_INDEX_NAME,
    CASE_INSENSITIVE_INDEX_NAME,
    GEMINI_API_KEY,
    DEFAULT_METADATA_THRESHOLDS
)

def get_window_range(entity_start, entity_end, window_size, chunk_text):
    """
    Adjusts the start and end indices to maintain a window size around a given range.

    Parameters:
        entity_start (int): The starting index of the entity's appearance in the chunk_text.
        entity_end (int): The ending index of the entity's appearance in the chunk_text.
        window_size (int): The desired window size.
        chunk_text (str): The text chunk to adjust within.

    Returns:
        tuple: A tuple containing the adjusted (text_start, text_end).
    """
    text_start = max(0, entity_start - window_size // 2)
    text_end = min(len(chunk_text), entity_end + window_size // 2)
    
    # Adjust to maintain the window size if the start or end gets cropped
    actual_window_size = text_end - text_start
    if actual_window_size < window_size:
        if text_start == 0:
            text_end = min(len(chunk_text), text_end + (window_size - actual_window_size))
        elif text_end == len(chunk_text):
            text_start = max(0, text_start - (window_size - actual_window_size))
    
    return text_start, text_end

# Assume get_chunk_text_by_id is defined (e.g., from an earlier cell)
def get_chunk_text_by_id(chunk_id: int, index: str = CASE_SENSITIVE_INDEX_NAME, es_client: Optional[Elasticsearch] = None) -> Optional[str]:
    es_client_to_use = es_client or globals().get('es')
    if es_client_to_use is None:
        print("Error: ES client not available for get_chunk_text_by_id")
        return None
    query = {"query": {"term": {"chunk_id": chunk_id}}, "_source": ["text"]}
    try:
        response = es_client_to_use.search(index=index, body=query)
        hits = response.get('hits', {}).get('hits', [])
        if not hits: return None
        return hits[0]['_source']['text']
    except Exception as e:
        print(f"Error fetching chunk text for {chunk_id}: {e}")
        return None

def _perform_llm_eval_for_chunk(
    chunk_id: int,
    rank: int,
    entity_label: str,
    eval_type: str, 
    triggering_method_name: str, 
    mentions_in_chunk_for_eval: List[Dict[str, Any]], 
    window_size: int,
    max_llm_calls_per_type: int,
    index_for_text_retrieval: str, 
    es_client_for_text_retrieval: Optional[Elasticsearch],
) -> Tuple[str, List]:
    chunk_text = get_chunk_text_by_id(chunk_id, index=index_for_text_retrieval, es_client=es_client_for_text_retrieval)
    if chunk_text is None:
        status_to_save = f"{eval_type.upper()}_TEXT_RETRIEVAL_ERROR"
        save_overall_status_to_cache(chunk_id, entity_label, eval_type, status_to_save)
        return status_to_save, []
    
    # Not cleaning the whole chunk_text here to preserve original offsets for mentions.
    # Cleaning will be done on the extracted window.
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    llm_detailed_log_rows_list: List[Dict] = []
    final_status = ""
    highest_implicit_score_seen = 0 # 0 means no 1 or 2 seen yet, or error
    llm_call_was_made = False
    last_llm_response = "LLM_ERROR" 

    for idx, mention in enumerate(mentions_in_chunk_for_eval[:max_llm_calls_per_type]):
        char_start = mention.get('char_start')
        char_end = mention.get('char_end')

        # Use your get_window_range with original_chunk_text
        text_start, text_end = get_window_range(char_start, char_end, window_size, chunk_text)
        window_text = chunk_text[text_start: text_end]
        
        # Clean the extracted window before sending to LLM
        cleaned_window = re.sub(r'\s+', ' ', window_text).strip()
        
        raw_response = generate(gemini_client, entity_label, cleaned_window, is_implicit=(eval_type == 'implicit'))
        llm_call_was_made = True
        last_llm_response = raw_response 

        llm_detailed_log_rows_list.append({
            "entity_label": entity_label, "method_name_context": triggering_method_name, 
            "chunk_id": chunk_id, 
            "rank_in_method_context": rank,
            "mention_attempt_in_chunk": idx + 1, 
            "mention_char_start": char_start, "mention_char_end": char_end,
            "mention_text_from_source": mention.get('text_mention', ''), 
            "mention_agg_score_from_source": mention.get('aggregated_score'),
            "window_text_sent_to_llm": cleaned_window,
            "eval_type_for_llm": eval_type, 
            "raw_llm_response": raw_response
        })

        if eval_type == 'explicit' and "yes" in raw_response.lower(): 
            # Break on the first positive explicit decision.
            final_status = "YES_EXPLICIT"
            break 

        if eval_type == 'implicit':
            score = raw_response.strip()
            if score == "3":
                # Break on the first positive implicit decision.
                final_status = "SCORE_3_IMPLICIT"
                break
            elif score == "2":
                highest_implicit_score_seen = max(highest_implicit_score_seen, 2)
            elif score == "1":
                highest_implicit_score_seen = max(highest_implicit_score_seen, 1)
    
    # There was some issue such that final_status is not set. 
    if not final_status: 
        if not mentions_in_chunk_for_eval:
            # No mentions found.
            final_status = f"{eval_type.upper()}_NO_MENTIONS_TO_EVAL"
        elif not llm_call_was_made: 
            # LLM call was not made
            final_status = f"{eval_type.upper()}_MENTION_PROCESSING_ERROR"
        elif eval_type == 'explicit':
            # If last_llm_response == "LLM_ERROR", it means no valid LLM response was captured, likely due to an error in the LLM call.
            final_status = "EXPLICIT_LLM_ERROR" if last_llm_response == "LLM_ERROR" else "NO_EXPLICIT"
        elif eval_type == 'implicit':
            # We didn't break out early in the implicit eval, so we need to set the final_status.
            if highest_implicit_score_seen == 2:
                final_status = "SCORE_2_IMPLICIT"
            elif highest_implicit_score_seen == 1:
                final_status = "SCORE_1_IMPLICIT"
            elif last_llm_response == "LLM_ERROR":
                final_status = "IMPLICIT_LLM_ERROR"
            else:
                final_status = "IMPLICIT_NON_123_RESPONSE"

    if not final_status:
        print(f"Critical Error: No final status determined for {chunk_id}, {entity_label}, {eval_type}.")
        final_status = f"{eval_type.upper()}_UNHANDLED_LOGIC_ERROR"
    
    save_overall_status_to_cache(chunk_id, entity_label, eval_type, final_status)
    
    return final_status, llm_detailed_log_rows_list

def evaluate_entity_llm_metrics(
    entity_names_input: Union[str, List[str]],
    k_values_for_precision_table: List[int],
    window_size: int = 130,
    max_llm_calls_per_mention_type_per_chunk: int = 3,
    es_client_instance: Optional[Elasticsearch] = None,
    verbose: bool = True,
    save_csv_files: bool = True,
):
    if isinstance(entity_names_input, str):
        list_of_entity_labels = [entity_names_input]
    else:
        list_of_entity_labels = entity_names_input

    es_client = es_client_instance or globals().get('es')
    if es_client is None:
        raise ValueError("Elasticsearch client not provided for LLM evaluation.")

    retrieval_method_configs = {
        "M1:H": {"func": find_entities_chunk_ids_by_metadata, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "thresholds": {"hyperlinks": DEFAULT_METADATA_THRESHOLDS["hyperlinks"]}, "top_k_metadata": k_values_for_precision_table, "top_k_sample": 100}, "is_string_method": False, "locate_search_with_aliases": True},
        "M2:EL": {"func": find_entities_chunk_ids_by_metadata, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "thresholds": {"entity_linking": DEFAULT_METADATA_THRESHOLDS["entity_linking"]}, "top_k_metadata": k_values_for_precision_table, "top_k_sample": 100}, "is_string_method": False, "locate_search_with_aliases": True},
        "M3:H+EL": {"func": find_entities_chunk_ids_by_metadata, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "thresholds": {"hyperlinks": DEFAULT_METADATA_THRESHOLDS["hyperlinks"], "entity_linking": DEFAULT_METADATA_THRESHOLDS["entity_linking"]}, "top_k_metadata": k_values_for_precision_table, "top_k_sample": 100}, "is_string_method": False, "locate_search_with_aliases": True},
        "M4:H+EL+Co+CoC": {"func": find_entities_chunk_ids_by_metadata, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "thresholds": {"hyperlinks": DEFAULT_METADATA_THRESHOLDS["hyperlinks"], "entity_linking": DEFAULT_METADATA_THRESHOLDS["entity_linking"], "coref": DEFAULT_METADATA_THRESHOLDS["coref"], "coref_cluster": DEFAULT_METADATA_THRESHOLDS["coref_cluster"]}, "top_k_metadata": k_values_for_precision_table, "top_k_sample": 100}, "is_string_method": False, "locate_search_with_aliases": True},
        "S1:C-SS-CS": {"func": find_entities_chunk_ids_by_string_search, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "search_with_aliases": False, "top_k": k_values_for_precision_table, "sample_k": 100}, "is_string_method": True, "locate_search_with_aliases": False},
        "S2:CA-SS-CS": {"func": find_entities_chunk_ids_by_string_search, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "search_with_aliases": True, "top_k": k_values_for_precision_table, "sample_k": 100}, "is_string_method": True, "locate_search_with_aliases": True},
        "S1:C-SS-CS-F": {"func": find_entities_chunk_ids_by_string_search, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "search_with_aliases": False, "top_k": k_values_for_precision_table, "sample_k": 100, "filter_func": filter_short_aliases}, "is_string_method": True, "locate_search_with_aliases": False},
        "S2:CA-SS-CS-F": {"func": find_entities_chunk_ids_by_string_search, "params": {"index": CASE_SENSITIVE_INDEX_NAME, "search_with_aliases": True, "top_k": k_values_for_precision_table, "sample_k": 100, "filter_func": filter_short_aliases}, "is_string_method": True, "locate_search_with_aliases": True},
        "S1:C-SS-CI-F": {"func": find_entities_chunk_ids_by_string_search, "params": {"index": CASE_INSENSITIVE_INDEX_NAME, "search_with_aliases": False, "top_k": k_values_for_precision_table, "sample_k": 100, "filter_func": filter_short_aliases}, "is_string_method": True, "locate_search_with_aliases": False},
        "S2:CA-SS-CI-F": {"func": find_entities_chunk_ids_by_string_search, "params": {"index": CASE_INSENSITIVE_INDEX_NAME, "search_with_aliases": True, "top_k": k_values_for_precision_table, "sample_k": 100, "filter_func": filter_short_aliases}, "is_string_method": True, "locate_search_with_aliases": True},
    }

    retrieved_chunks_store: Dict[str, Dict[str, List[int]]] = {label: {} for label in list_of_entity_labels}
    llm_detailed_interaction_log: List[Dict] = []
    result_log: List[Dict] = []

    precision_results_agg: Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, int]]]]] = defaultdict(  # entity_label level
        lambda: defaultdict(              # method_name level
            lambda: {
                "explicit": defaultdict(lambda: {"confirmed": 0, "errors": 0}),
                "implicit": defaultdict(lambda: {"confirmed": 0, "errors": 0})
            }
        )
    )

    # Phase 1: Retrieve all chunks
    if verbose: print(f"Phase 1: Retrieving chunks for all methods...")
    retrieval_jobs = []
    for entity_label in list_of_entity_labels:
        for method_name, config in retrieval_method_configs.items():
            retrieval_jobs.append((entity_label, method_name, config))

    with ThreadPoolExecutor(max_workers=min(len(retrieval_jobs), 128)) as executor:
        future_to_job = {
            executor.submit(
                config["func"],
                entity_label,
                es_client=es_client,
                **config["params"]
            ): (entity_label, method_name)
            for entity_label, method_name, config in retrieval_jobs
        }

        iterator = as_completed(future_to_job)

        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=len(future_to_job), desc="Chunk Retrieval")
            except ImportError:
                pass

        for future in iterator:
            entity_label, method_name = future_to_job[future]
            try:
                results_dict = future.result()
                retrieved_chunks_store[entity_label][method_name] = results_dict.get(entity_label, [])
            except Exception as e:
                print(f"Error retrieving chunks for {entity_label}, {method_name}: {e}")
    
    # Phase 2: LLM Evaluation
    if verbose: print("\nPhase 2: LLM Evaluation...")
    for entity_label in list_of_entity_labels:
        if verbose: print(f"  LLM Eval for entity: {entity_label}")

        # This helper is called by the main eval functions below
        def get_prepared_mentions_for_eval(spans: Dict, is_string_method_type: bool):
            mentions_list = []
            if is_string_method_type:
                for span in sorted(spans, key=lambda s: (s["char_end"] - s["char_start"]), reverse=True): # Sort by length within the chunk - this is a heuristic to prioritize longer mentions as they might have a better chance of being correct
                    span["aggregated_score"] = 0.0 # Dummy score
                    mentions_list.append(span) # Dummy score
            else:
                # Metadata matches are: (char_start, char_end, text_mention, agg_score, qid, cand_name)
                for span in sorted(spans, key=lambda s: max([c["aggregated_score"] for c in s["candidates"]]), reverse=True):
                    mention = {}
                    for k in ["char_start", "char_end", "text_mention"]:
                        mention[k] = span[k]
                    mention["aggregated_score"] = max([c["aggregated_score"] for c in span["candidates"]]) # should only have one value
                    mentions_list.append(mention)
            return mentions_list

        # --- Step 2: Use Retrieval Methods to perform primary LLM calls and populate cache ---
        for method_name in retrieval_method_configs:
            config = retrieval_method_configs[method_name]
            if verbose: print(f"    Processing {method_name}...")

            # There will be duplicates for multiple k values, but it's okay because we save them in the cache and will not send the same chunk + entity.
            chunk_ids = [item for v in retrieved_chunks_store[entity_label].get(method_name, {}).get("chunks", {}).values() for item in v]
            for rank, chunk_id in tqdm(chunk_ids, desc=method_name):
                if get_overall_status_from_cache(chunk_id, entity_label, 'explicit'):
                    continue # Result already exists in cache, skip.

                if get_overall_status_from_cache(chunk_id, entity_label, 'implicit'):
                    continue # Result already exists in cache, skip.

                spans = retrieved_chunks_store[entity_label][method_name]["spans"][chunk_id]
                mentions = get_prepared_mentions_for_eval(spans, config["is_string_method"])
                
                # --- Explicit evaluation for the current chunk ---
                expl_status, interactions = _perform_llm_eval_for_chunk(chunk_id, rank, entity_label, 'explicit', method_name, mentions,
                                                        window_size, max_llm_calls_per_mention_type_per_chunk,
                                                        config["params"]["index"], es_client)
                llm_detailed_interaction_log = llm_detailed_interaction_log + interactions
                # --- Implicit evaluation for the current chunk ---
                if expl_status == "YES_EXPLICIT":
                    save_overall_status_to_cache(chunk_id, entity_label, 'implicit', "SCORE_3_IMPLICIT_FROM_EXPLICIT_YES")
                else:
                    _, interactions = _perform_llm_eval_for_chunk(chunk_id, rank, entity_label, 'implicit', method_name, mentions,
                                                window_size, max_llm_calls_per_mention_type_per_chunk,
                                                config["params"]["index"], es_client)
                    llm_detailed_interaction_log = llm_detailed_interaction_log + interactions

        if verbose: print(f"    Finished cache-centric LLM evaluations for {entity_label}.")
        
    # Phase 3: Calculate Precision Table
    if verbose: print("\nPhase 3: Calculating Precision Table data...")
    error_statuses = ["EXPLICIT_LLM_ERROR", "IMPLICIT_LLM_ERROR", 
                      "EXPLICIT_LOCATE_ERROR", "IMPLICIT_LOCATE_ERROR", # Should not happen if locate is done before _perform call
                      "EXPLICIT_TEXT_RETRIEVAL_ERROR", "IMPLICIT_TEXT_RETRIEVAL_ERROR",
                      "EXPLICIT_MENTION_PROCESSING_ERROR", "IMPLICIT_MENTION_PROCESSING_ERROR",
                      "EXPLICIT_NO_MENTIONS_TO_EVAL", "IMPLICIT_NO_MENTIONS_TO_EVAL", # If locate returns no mentions
                      "IMPLICIT_NON_123_RESPONSE", "EXPLICIT_UNHANDLED_LOGIC_ERROR", "IMPLICIT_UNHANDLED_LOGIC_ERROR"]

    for entity_label in tqdm(list_of_entity_labels, desc="Entities"):
        for method_name in tqdm(retrieval_method_configs, "Retrieval Method"):
            retrieved_ids = retrieved_chunks_store[entity_label].get(method_name, []).get("chunks", [])
            for k_val, cids in retrieved_ids.items():             
                conf_expl, err_expl = 0, 0
                for rank, cid in cids:
                    status = get_overall_status_from_cache(cid, entity_label, 'explicit')
                    result_log.append({
                        "entity_label": entity_label,
                        "method_name_context": method_name,
                        "chunk_id": cid,
                        "eval_type": "explicit",
                        "rank_in_method_context": rank + 1,
                        "k": k_val,
                        "status": status,
                    })
                    if status == "YES_EXPLICIT": conf_expl += 1
                    elif status in error_statuses or status is None : err_expl += 1 # Count None as error for precision
                precision_results_agg[entity_label][method_name]['explicit'][k_val]['confirmed'] = conf_expl
                precision_results_agg[entity_label][method_name]['explicit'][k_val]['errors'] = err_expl
                
                conf_impl, err_impl = 0, 0
                for rank, cid in cids:
                    status = get_overall_status_from_cache(cid, entity_label, 'implicit')
                    result_log.append({
                        "entity_label": entity_label,
                        "method_name_context": method_name,
                        "chunk_id": cid,
                        "eval_type": "implicit",
                        "rank_in_method_context": rank + 1,
                        "k": k_val,
                        "status": status,
                    })
                    if status == "SCORE_3_IMPLICIT" or status == "SCORE_3_IMPLICIT_FROM_EXPLICIT_YES": conf_impl += 1
                    elif status in error_statuses or status is None: err_impl += 1
                precision_results_agg[entity_label][method_name]['implicit'][k_val]['confirmed'] = conf_impl
                precision_results_agg[entity_label][method_name]['implicit'][k_val]['errors'] = err_impl

    # Phase 4: Save CSVs
    if save_csv_files:
        if verbose: print("\nPhase 4: Saving CSV files...")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        detailed_log_filename = f"llm_eval_detailed_log_{ts}.csv"
        detailed_log_fields = [
            "entity_label", "method_name_context", "chunk_id", "rank_in_method_context", 
            "mention_attempt_in_chunk", "mention_char_start", "mention_char_end", 
            "mention_text_from_source", "mention_agg_score_from_source",
            "window_text_sent_to_llm", "eval_type_for_llm", "raw_llm_response"
        ]
        try:
            with open(detailed_log_filename, "w", newline="", encoding="utf-8") as f_detailed:
                writer = csv.DictWriter(f_detailed, fieldnames=detailed_log_fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(llm_detailed_interaction_log)
            if verbose: print(f"  Saved detailed LLM interactions to: {detailed_log_filename}")
        except Exception as e: print(f"Error saving detailed log CSV: {e}")

        detailed_result_filename = f"llm_eval_detailed_results_{ts}.csv"
        detailed_result_fields = [
            "entity_label", "method_name_context", "chunk_id", "eval_type", "rank_in_method_context", 
            "k", "status"
        ]
        try:
            with open(detailed_result_filename, "w", newline="", encoding="utf-8") as f_result:
                writer = csv.DictWriter(f_result, fieldnames=detailed_result_fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(result_log)
            if verbose: print(f"  Saved detailed LLM interactions to: {detailed_result_filename}")
        except Exception as e: print(f"Error saving detailed log CSV: {e}")

        summary_log_filename = f"llm_eval_precision_summary_{ts}.csv"
        summary_fields = ["entity_label", "retrieval_method", "evaluation_type", 
                          "k_value", "precision_at_k", "confirmed_count", "error_chunks_at_k"]
        try:
            with open(summary_log_filename, "w", newline="", encoding="utf-8") as f_summary:
                writer = csv.DictWriter(f_summary, fieldnames=summary_fields)
                writer.writeheader()
                for ent_l, m_data in precision_results_agg.items():
                    for meth_n, t_data in m_data.items():
                        for eval_t, k_map in t_data.items():
                            for k_v, counts in k_map.items():
                                prec = counts['confirmed'] / k_v
                                writer.writerow({
                                    "entity_label": ent_l, "retrieval_method": meth_n, "evaluation_type": eval_t,
                                    "k_value": k_v, "precision_at_k": f"{prec:.4f}",
                                    "confirmed_count": counts['confirmed'], 
                                    "error_chunks_at_k": counts.get('errors', 0)
                                })
            if verbose: print(f"  Saved precision summary table to: {summary_log_filename}")
        except Exception as e: print(f"Error saving summary log CSV: {e}")

    if verbose: print("\nLLM Evaluation Process Completed Successfully.")
    return precision_results_agg


if __name__ == "__main__":
    es = get_esclient()

    dev_df = pd.read_csv("popqa_entities/dev_entities.csv")
    entity_names = dev_df["entity"].tolist()

    initialize_cache()
    initialize_overall_status_cache()

    evaluate_entity_llm_metrics(entity_names, [1, 5, 10, 100, 1000, 10000, 100000], es_client_instance=es)