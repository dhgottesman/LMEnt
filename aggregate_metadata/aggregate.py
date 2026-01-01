import os
import ujson as json
from tqdm import tqdm

import sys
import csv
import glob
import argparse

from utils import *

from concurrent.futures import ProcessPoolExecutor, as_completed


def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_processed_ids(output_path):
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        return set()
    processed = set()
    for row in stream_jsonl(output_path):
        if "id" in row:
            processed.add(row["id"])
    return processed

def process_doc(doc, skip_coref=False):
    hyperlinks = normalize_hyperlinks(doc.get("hyperlinks_clean", []))
    coref = doc.get("coref", [])
    entity_linking = normalize_entity_linking(doc.get("entity_linking", []))

    if not skip_coref:
        enriched_clusters = enrich_coref_clusters(coref, entity_linking + hyperlinks)
        enriched_clusters = {
            cid: spans for cid, spans in enriched_clusters.items()
            if any(span.get("entities") or span.get("links") for span in spans)
        }

        entity_scores, enriched_clusters = score_entities_by_subject_likelihood(enriched_clusters)
        filtered_enriched_clusters = {}
        for cid, spans in enriched_clusters.items():
            scores = [v for span in spans for v in span.get("score", {}).values()]
            if scores and max(scores) >= 0.3:
                filtered_enriched_clusters[cid] = spans
    else:
        print(f"Skipping coref for doc {doc['id']}")
        filtered_enriched_clusters, entity_scores = {}, {}

    return aggregate_mentions(hyperlinks, entity_linking, filtered_enriched_clusters, entity_scores)

def aggregate_entity_mentions(docs_path, output_path):
    processed_ids = load_processed_ids(output_path)
    with open(output_path, "a", encoding="utf-8") as out_f:
        for doc in tqdm(stream_jsonl(docs_path)):
            doc_id = doc.get("id")

            if doc_id in processed_ids:
                continue

            if doc_id in {"44471088", "62180015", "67215817"}:
                continue

            entities = process_doc(doc)
            doc["entities"] = entities

            doc.pop("hyperlinks_clean")
            doc.pop("hyperlinks")
            doc.pop("coref")
            doc.pop("entity_linking")

            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            processed_ids.add(doc_id)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple script with one index argument.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="A file created by maverick_coref/run_maverick.py"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The output file to write the aggregated mentions to. This will be ingested by the dolma tokenizer."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    ndjson_file = args.input_file # "/home/morg/dataset/maverick/maverick_0.json"
    output_file = args.output_dir + f"entity_mentions_{args.index}.json"

    aggregate_entity_mentions(ndjson_file, output_file)

