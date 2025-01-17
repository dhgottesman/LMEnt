import copy
import json
import re
from fastcoref import FCoref
import wiki_dump_reader as wiki
from json_writer import JsonWriter
from typing import Dict, Generator, List, Tuple
import argparse
import torch
from tokenizers import Tokenizer
import mwparserfromhell
from mwparserfromhell.nodes import *
from mwparserfromhell.wikicode import Wikicode

import mwxml


def _bad_text(text):
    bad_prefixes = ("file:", "image:", "category:")
    return len(str(text).strip()) == 0 or str(text).strip().lower().startswith(bad_prefixes)

def _bad_link(link):
    bad_prefixes = ("file:", "image:", "category:")
    return str(link.title).strip().lower().startswith(bad_prefixes)

def _bad_types(node):
    return not(isinstance(node, Tag) or isinstance(node, Heading) or isinstance(node, Wikilink) or isinstance(node, Text))

# Inspo taken from: https://github.com/earwig/earwigbot/blob/dfae10cf12e46d6ad3c33312fdc7bd4649be47c3/src/earwigbot/wiki/copyvios/parsers.py#L154
def strip_code(wikicode):
    for node in wikicode.filter(matches=_bad_types, recursive=False):
        wikicode.remove(node)
    for node in wikicode.filter_wikilinks(matches=_bad_link, recursive=False):
        wikicode.remove(node)
    for node in wikicode.filter_text(matches=_bad_text, recursive=False):
        wikicode.remove(node)
    for node in wikicode.filter_tags(recursive=False):
        bad_tags = ("ref", "table", "gallery")
        if node.tag in bad_tags:
            wikicode.remove(node)
        elif len(node._contents.nodes) == 0:
            wikicode.remove(node)
        else:
            # Need to make sure not stripping links.
            idx = wikicode.index(node)
            new_contents = strip_code(copy.deepcopy(node._contents))
            if len(new_contents.nodes) == 0:
                wikicode.remove(node)
            elif len(new_contents.nodes) == 1:
                wikicode.set(idx, new_contents.nodes[0])
            else:
                # Remove Tag node and bring all it's children to the top-level.
                # This happens when the Tag contains Text and Wikilinks.
                wikicode.remove(node)
                for offset, child in enumerate(new_contents.nodes):
                    wikicode.insert(idx + offset, child)
    return wikicode

def get_sections(wikicode, section_spans):
    sections = wikicode.get_sections(flat=True, include_lead=True, levels=[2, 3, 4, 5, 6])
    flat_list = []

    for section in sections:
        headings = section.filter_headings()
        if headings:
            section_title = headings[0].title.strip()
            heading_level = headings[0].level
            section_start = section_spans[section_title]["begin"]
            section_end = section_start
        else:
            # This is the lead (no heading)
            section_title = "Lead"
            heading_level = 1
            section_start = 0
            section_end = 0

        # Extract the immediate content of this section (excluding sub-sections)
        section_text = str(section)
        section_end += len(section_text)

        flat_list.append({
            "title": section_title,
            "level": heading_level,
            "content": section_text,
            "begin": section_start,
            "end": section_end
        })

    return flat_list

def parse_wikipedia_dump(dump_file: str) -> Generator[Tuple[str, str, Wikicode, List[Dict], List[Dict], bool], None, None]:
    for page in mwxml.Dump.from_file(dump_file):
        id, title, redirect = page.id, page.title, page.redirect
        if redirect:
            yield id, title, None, [], None, True
        else:
            last_revision = None
            for revision in page:
                last_revision = revision
            text = last_revision.text
            wikicode = mwparserfromhell.parse(text)
            wikicode = strip_code(wikicode)

            wikicode, links, header_spans = find_wikilinks_spans(wikicode) # convert spans to links
            
            sections = get_sections(copy.deepcopy(wikicode), header_spans)

            for link in links:
                assert(str(wikicode)[link["begin"]:link["end"]] == link["text"])

            for section in sections:
                assert(str(wikicode)[section["begin"]:section["end"]] == section["content"])

            yield id, title, wikicode, sections, links, False

def parse_link(link: dict) -> Tuple[int, int, str, str]:
    return int(link['begin']), int(link['end']), link['link'], link['text']

def is_coref_cluster_linked(link_start: int, link_end: int, coref_entity_cluster: Tuple[Tuple[int, int],...]) -> bool:
    for entity_start, entity_end in coref_entity_cluster:
        if link_start >= entity_start and link_end <= entity_end:
            return True

    return False

def get_coref_clusters(coref: FCoref, texts: List[str], as_strings=False) -> List[List[Tuple[Tuple[int, int],...]]]:
    return [x.get_clusters(as_strings=as_strings) for x in coref.predict(texts, max_tokens_in_batch=100000)]

def get_all_linked_entities(coref_clusters: List[Tuple[Tuple[int, int],...]], links: List[Dict]) -> Generator[Tuple[int, int, str], None, None]:
    for link in links:
        found = False
        link_start, link_end, entity_name, link_text = parse_link(link)
        for entity_cluster in coref_clusters:
            if is_coref_cluster_linked(link_start, link_end, entity_cluster):
                # We have a link that's contained within a span of a coref cluster, meaning the whole cluster is related to that link
                for entity_start, entity_end in entity_cluster:
                    yield (entity_start, entity_end, entity_name)

                # Remove the cluster from the list so we don't double count
                coref_clusters.remove(entity_cluster)

                found = True
                break

        if not found:
            yield (link_start, link_end, entity_name)

def find_wikilinks_spans(wikicode):
    entity_spans = []
    header_spans = {}
    current_pos = 0  # Tracks the aggregated position in the text

    for i, node in enumerate(wikicode.nodes):
        # Process Wikilinks
        if isinstance(node, Wikilink):
            node_text = node.__strip__()
            node_length = len(node_text)
            entity_spans.append({
                'link': str(node.title).strip(),
                'text': node_text,
                'begin': current_pos,
                'end': current_pos + node_length
            })
            wikicode.set(i, Text(node_text))  # Replace Wikilink with plain text

        # Process Headings and embedded Wikilinks
        elif isinstance(node, Heading):
            heading_pos = 0
            heading_nodes = copy.deepcopy(node.title.nodes)

            for j, heading_node in enumerate(node.title.nodes):
                heading_text = heading_node.__strip__() if isinstance(heading_node, Wikilink) else str(heading_node)
                heading_length = len(heading_text)

                if isinstance(heading_node, Wikilink):
                    entity_spans.append({
                        'link': str(heading_node.title).strip(),
                        'text': heading_text,
                        'begin': current_pos + heading_pos + node.level,
                        'end': current_pos + heading_pos + node.level + heading_length
                    })
                    heading_nodes[j] = Text(heading_text)

                heading_pos += heading_length

            node.title.nodes = heading_nodes
            wikicode.set(i, node)

            # Set the header span after updating Wikilinks.
            node_text = str(node)
            node_length = len(node_text)
            header_spans[node.title.strip()] = {'begin': current_pos}

        else:
            node_text = str(node)
            node_length = len(node_text)

        current_pos += node_length  # Move position forward

    return wikicode, entity_spans, header_spans

def write(dump_file_path: str, out_base_path: str, device: str):
    coref = FCoref(device=device, enable_progress_bar=True)
    writer = JsonWriter(out_base_path, verbose=True)
    tokenizer = Tokenizer.from_file("/home/morg/students/gottesman3/OLMo/olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json")
    metrics = {'text_length': 0,
               'token_count': 0, 
               'word_count': 0,
               'article_count': 0,
               'redirect_count': 0,
               'entity_count': 0
              }
    for id, title, wikicode, sections, links, is_redirect in parse_wikipedia_dump(dump_file_path):
        entities = []
        text = str(wikicode)
        metrics['redirect_count'] += is_redirect

        if not is_redirect:
            # Only normal pages have entities.
            coref_clusters = get_coref_clusters(coref, [text])[0]
            for entity_start, entity_end, entity_name in get_all_linked_entities(coref_clusters, links):
                entities.append({"entity_start": entity_start, "entity_end": entity_end, "entity_name": entity_name})
            del coref_clusters    
            torch.cuda.empty_cache()
            
            metrics['text_length'] += len(text)
            metrics['token_count'] += len(tokenizer.encode(text))
            metrics['word_count'] += len(text.split(' '))
            metrics['article_count'] += 1
            metrics['entity_count'] += len(entities)
        
        writer.write({"src_file": dump_file_path, "id": id, "text": text, "title": title, "entities": entities, 'is_redirect': is_redirect, 'sections': sections})
    
    writer.close()
    with open(writer.out_base_path + '_metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    write(args.dump_file, args.output, args.device)
