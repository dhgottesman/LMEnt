from collections import defaultdict
import re
from refined_lmdb import LmdbImmutableDict

import numpy as np
import copy
from collections import defaultdict, Counter

qcode_to_wiki = LmdbImmutableDict(path="/home/morg/dataset/refined/organised_data_dir/wikidata_data/qcode_to_wiki.lmdb", write_mode=False)


def is_pronoun(text):
    return re.fullmatch(r"\b(?:he|she|it|they|we|i|you|him|her|us|them|me|my|your|his|their|our|its|mine|yours|hers|ours|theirs)\b", text.lower()) is not None

def strip_leading_article(text: str) -> str:
    pattern = r'^(a|an|the)\b\s*'
    return re.sub(pattern, '', text, count=1, flags=re.I)

def normalize_span_text(text: str) -> str:
    """
    Normalizes a text span by stripping a leading article and a trailing
    possessive ('s or ').
    
    Examples:
        >>> normalize_span_text("the boy's toy")  # No change, possessive is not at the end
        "the boy's toy"
        >>> normalize_span_text("the boy's")
        'boy'
        >>> normalize_span_text("the boys'")
        'boys'
        >>> normalize_span_text("James'")
        'James'
        >>> normalize_span_text("an apple")
        'apple'
    """
    # First, strip the leading article
    text_no_article = strip_leading_article(text)
    
    # Next, strip a trailing 's or ' using a regex
    # ('s|') : Matches 's' OR just '. The pipe order is important to match 's' first.
    # $       : Asserts this pattern must be at the end of the string.
    possessive_pattern = r"('s|')$"
    normalized_text = re.sub(possessive_pattern, '', text_no_article)
    
    return normalized_text

def normalize_entity_linking(entity_linking):
    normalized = []
    for mention in entity_linking:
        if entity_id := mention.get("predicted_entity", {}).get("wikidata_entity_id"):
            name = mention.get("predicted_entity").get("human_readable_name", None)
            normalized.append({
                "type": "entity",
                "start": mention["start"],
                "end": mention["start"] + mention["ln"],
                "text": mention["text"],
                "id": entity_id,
                "name": name or qcode_to_wiki.get(entity_id) or "",
                "confidence": mention.get("entity_linking_model_confidence_score"),
            })
    return normalized

def normalize_hyperlinks(hyperlinks):
    normalized = []            
    for link in hyperlinks:
        if entity_id := link.get("qcode"):
            normalized.append({
                "type": "link",
                "start": link["start"],
                "end": link["end"],
                "text": link["surface_form"],
                "name": link["uri"].replace("_", " "),
                "id": entity_id,
                "confidence": 1.0,
            })
    return normalized

def longest_common_substring(s1, s2):
    s1 = s1.lower()
    s2 = s2.lower()
    m = len(s1)
    n = len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                longest = max(longest, dp[i + 1][j + 1])

    return longest

def lcs_overlap_ratio(unlinked_text, anchor_text):
    unlinked_text = normalize_span_text(unlinked_text)
    anchor_text = normalize_span_text(anchor_text)
    lcs_len = longest_common_substring(unlinked_text, anchor_text)

    len_anchor = len(anchor_text)
    len_unlinked = len(unlinked_text)
    
    if not len_anchor or not len_unlinked:
        return 0.0
    
    a = lcs_len / len_anchor
    b = lcs_len / len_unlinked

    if a + b == 0:
        return 0.0

    return 2 * a * b / (a + b)

def best_entity_or_link_match(span_text, anchor_spans):
    best = {
        "match_type": None,
        "anchor": None,
        "item": None,
        "score": 0.0,
    }

    for anchor in anchor_spans:
        for link in anchor.get("links", []):
            score = lcs_overlap_ratio(span_text, link["surface_form"]) * 1.0
            if score > best["score"]:
                best = {
                    "match_type": "link",
                    "anchor": anchor,
                    "item": link,
                    "score": score,
                }

        for entity in anchor.get("entities", []):
            score = lcs_overlap_ratio(span_text, entity["el_text"]) * entity.get("confidence")

            if score > best["score"]:
                best = {
                    "match_type": "entity",
                    "anchor": anchor,
                    "item": entity,
                    "score": score,
                }

    return best if best["score"] > 0.3 else None

def calculate_normalized_coverage(coref_span, candidate_span):
    # Normalize the text and calculate the length of the stripped article
    norm_coref_text = normalize_span_text(coref_span['text'])
    norm_candidate_text = normalize_span_text(candidate_span['text'])

    coref_start_offset = coref_span['text'].find(norm_coref_text) if norm_coref_text else 0
    candidate_start_offset = candidate_span['text'].find(norm_candidate_text) if norm_candidate_text else 0

    adj_coref_start = coref_span['start'] + coref_start_offset
    adj_candidate_start = candidate_span['start'] + candidate_start_offset

    overlap_start = max(adj_coref_start, adj_candidate_start)
    overlap_end = min(coref_span['end'], candidate_span['end'])
    overlap_length = max(0, overlap_end - overlap_start) # Use max(0,...) to prevent negative lengths.

    norm_span_length = len(norm_coref_text)
    
    # Avoid division by zero if the coref span had no content (e.g., was just "the").
    if norm_span_length == 0:
        return 0.0

    return overlap_length / norm_span_length

def enrich_coref_clusters(coref, candidates):
    enriched_clusters = defaultdict(list)

    # Iterate over the coref clusters.
    for cluster_id, cluster in enumerate(coref["clusters_char_offsets"]):
        # Iterate over each mention span in a given cluster.
        for i, span_offsets in enumerate(cluster):
            span_start, span_end = span_offsets
            
            span_entry = {
                "start": span_start,
                "end": span_end,
                "coref_text": coref["clusters_char_text"][cluster_id][i],
                "entities": [],
                "links": [],
            }

            for candidate in candidates:
                # Basic overlap check
                if not (candidate["end"] <= span_start or candidate["start"] >= span_end):
                    
                    # Calculate coverage ratio using the dedicated helper function
                    coverage_ratio = calculate_normalized_coverage(
                        {'start': span_start, 'end': span_end, 'text': span_entry['coref_text']},
                        {'start': candidate['start'], 'end': candidate['end'], 'text': candidate['text']}
                    )

                    # Append to the correct list based on the candidate's type
                    if candidate["type"] == "entity":
                        span_entry["entities"].append({
                            "id": candidate["id"],
                            "el_text": candidate["text"],
                            "start": candidate["start"],
                            "end": candidate["end"],
                            "coverage_ratio": coverage_ratio,
                            "confidence": candidate["confidence"]
                        })
                    elif candidate["type"] == "link":
                        span_entry["links"].append({
                            "id": candidate["id"],
                            "surface_form": candidate["text"],
                            "start": candidate["start"],
                            "end": candidate["end"],
                            "coverage_ratio": coverage_ratio,
                            "confidence": candidate["confidence"]
                        })

            enriched_clusters[cluster_id].append(span_entry)

    return dict(enriched_clusters)

def score_entities_by_subject_likelihood(enriched_clusters):
    new_enriched_clusters = copy.deepcopy(enriched_clusters)
    cluster_entity_scores = {}

    for cluster_id, spans in new_enriched_clusters.items():
        entity_counts = defaultdict(float)
        total_contribution = 0.0
        seen_mentions = set()

        anchor_spans = []
        unlinked_spans = []

        for span in spans:
            if is_pronoun(span["coref_text"]):
                continue

            has_entities = bool(span.get("entities"))
            has_links = bool(span.get("links"))

            if has_entities or has_links:
                weights = defaultdict(float)

                for link in span.get("links", []):
                    key = (link["id"], link["start"], link["end"])
                    weight = 1.0 * link["coverage_ratio"]
                    weights[link["id"]] = max(weight, weights[link["id"]])

                    if key in seen_mentions:
                        continue
                    seen_mentions.add(key)

                    entity_counts[link["id"]] += weight
                    total_contribution += weight

                for entity in span.get("entities", []):
                    key = (entity["id"], entity["start"], entity["end"])
                    weight = entity["confidence"] * entity["coverage_ratio"]
                    weights[entity["id"]] = max(weight, weights[entity["id"]])
                    
                    if key in seen_mentions:
                        continue
                    seen_mentions.add(key)

                    entity_counts[entity["id"]] += weight
                    total_contribution += weight

                span["score"] = dict(weights)
                anchor_spans.append(span)

            else:
                unlinked_spans.append(span)

        # Infer contribution for unlinked spans via char-level overlap with a hyperlink or entity-linking mention..
        for span in unlinked_spans:
            result = best_entity_or_link_match(span["coref_text"], anchor_spans)
            # print(result)

            if result:
                entity = result["item"]
                weights = {entity["id"]: result["score"]}
                entity_id = entity["id"]

                span["score"] = weights
                span["id"] = entity_id
                entity_counts[entity_id] += weights[entity_id]
                total_contribution += weights[entity_id]

            else:
                span["score"] = {}
                span["id"] = None            

        total_contribution = max(1.0, total_contribution)

        if entity_counts:
            counts = np.array([count for count in entity_counts.values()])
            # Optionally, add a temperature parameter to soften/harden (T=1.0 is standard)
            temperature = 1.0
            exp_counts = np.exp(counts / temperature)
            softmax_scores = exp_counts / np.sum(exp_counts)
            normalized_scores = {
                entity_id: softmax_score
                for entity_id, softmax_score in zip(entity_counts.keys(), softmax_scores)
            }
        else:
            normalized_scores = {}

        cluster_entity_scores[cluster_id] = normalized_scores

    return cluster_entity_scores, new_enriched_clusters

def aggregate_mention_entity_scores(
    mentions,
    hyperlink_weight=1.0,
    entity_linking_weight=1.0,
    coref_weight=0.75,
    coref_cluster_weight=0.5,
):
    results = {}

    for mention_key, mention in mentions.items():
        entity_scores = {}
        mention_result = {}

        # Collect all possible entity IDs in this mention
        entity_ids = set()
        for field in ["hyperlinks", "entity_linking", "coref", "coref-cluster"]:
            if field in mention:
                entity_ids.update(mention[field].keys())
                # Also copy the raw field (once per mention, not per entity)
                if field not in mention_result and mention[field]:
                    mention_result[field] = mention[field]

        # For each entity, aggregate weighted scores for the fields present
        for entity_id in entity_ids:
            weighted_sum = 0.0
            weight_sum = hyperlink_weight + entity_linking_weight + coref_weight + coref_cluster_weight

            if "hyperlinks" in mention and entity_id in mention["hyperlinks"]:
                weighted_sum += hyperlink_weight * mention["hyperlinks"][entity_id]
            if "entity_linking" in mention and entity_id in mention["entity_linking"]:
                weighted_sum += entity_linking_weight * mention["entity_linking"][entity_id]
            if "coref" in mention and entity_id in mention["coref"]:
                weighted_sum += coref_weight * mention["coref"][entity_id]
            if "coref-cluster" in mention and entity_id in mention["coref-cluster"]:
                weighted_sum += coref_cluster_weight * mention["coref-cluster"][entity_id]

            if weight_sum > 0:
                agg_score = weighted_sum / weight_sum
                agg_score = round(max(0.0, min(1.0, agg_score)), 2)
                if agg_score > 0:
                    entity_scores[entity_id] = agg_score

        if entity_scores:
            mention_result["aggregated"] = entity_scores

        if mention_result:
            results[mention_key] = mention_result

    return results

def aggregate_mentions(
    hyperlinks,
    entity_linking,
    enriched_clusters,
    entity_scores,
    hyperlink_weight=4.0,
    entity_linking_weight=3.0,
    coref_weight=2.0,
    coref_cluster_weight=1.0,
):
    mentions = {}

    def add_score(source, start, end, text, qid, score, name=None):
        key = (start, end)
        if score == 0:
            return
        if key not in mentions:
            mentions[key] = {"text": text, "name": defaultdict(list)}
        if source not in mentions[key]:
            mentions[key][source] = {}
        mentions[key][source][qid] = round(score, 2)
        if name:
            mentions[key]["name"][qid].append(name)

    # Hyperlink evidence
    for m in hyperlinks:
        add_score("hyperlinks", m["start"], m["end"], m["text"], m["id"], m["confidence"], m.get("name"))

    # Entity linking evidence
    for m in entity_linking:
        add_score("entity_linking", m["start"], m["end"], m["text"], m["id"], m["confidence"], m.get("name"))

    # Coref and coref-cluster evidence
    for cluster_id, cluster in enriched_clusters.items():
        for m in cluster:
            start, end = m["start"], m["end"]
            text = m["coref_text"]
            for qid, score in m.get("score", {}).items():
                add_score("coref", start, end, text, qid, score, qcode_to_wiki.get(qid) or "")
            for qid, score in entity_scores.get(cluster_id, {}).items():
                add_score("coref_cluster", start, end, text, qid, score, qcode_to_wiki.get(qid) or "")

    # Final aggregation
    final_output = []
    for (start, end), data in mentions.items():
        all_qids = set()
        for source in ["hyperlinks", "entity_linking", "coref", "coref_cluster"]:
            all_qids.update(data.get(source, {}).keys())

        candidates = []
        for qid in all_qids:
            scores = {
                "hyperlinks": data.get("hyperlinks", {}).get(qid, 0.0),
                "entity_linking": data.get("entity_linking", {}).get(qid, 0.0),
                "coref": data.get("coref", {}).get(qid, 0.0),
                "coref_cluster": data.get("coref_cluster", {}).get(qid, 0.0),
            }

            weighted_sum = (
                scores["hyperlinks"] * hyperlink_weight +
                scores["entity_linking"] * entity_linking_weight +
                scores["coref"] * coref_weight +
                scores["coref_cluster"] * coref_cluster_weight
            )
            total_weight = hyperlink_weight + entity_linking_weight + coref_weight + coref_cluster_weight

            aggregated_score = round(max(0.0, min(1.0, weighted_sum / total_weight)), 2)
            name = data["name"].get(qid, [])  # Try to get from hyperlinks or entity_linking

            candidates.append({
                "qid": qid,
                "name": "" if len(name) == 0 else name[0],
                "scores_by_source": scores,
                "aggregated_score": aggregated_score
            })

        if candidates:
            final_output.append({
                "char_start": start,
                "char_end": end,
                "text_mention": data["text"],
                "candidates": candidates
            })

    return final_output

