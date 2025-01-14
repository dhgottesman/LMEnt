import re
from fastcoref import FCoref
import wiki_dump_reader as wiki
from json_writer import JsonWriter
from typing import Dict, Generator, List, Tuple
import argparse
import torch
from tokenizers import Tokenizer

REDIRECT_REGEX = re.compile('^redirect \\[\\[([a-z\\s_,]+)\\]\\]$')
REDIRECT_REGEX_2 = re.compile('^redirect ([^\\n]+)(\\n|.)*$')

def my_build_links(text: str) -> Tuple[str, List[Dict]]:
    # We don't want to just remove the links, we want to replace them with the link text so that the returned text
    # is parseable. We also want to make sure the links indices are consistent with the new text.
    out = ""

    links = []
    last_index = 0
    offset = 0

    for match in re.finditer(r"\[\[(.*?)(?:\|(.*?))?\]\]", text):
        entity_name, link_text = match.groups()
        link_text = link_text or entity_name

        out += text[last_index:match.start()] + link_text
        last_index = match.end()

        # Calculate new begin after removing the link text
        begin = match.start() - offset
        end = begin + len(link_text)

        # Update offset - for each link we remove, the text gets shorter by the length of the link text
        offset += match.end() - match.start() - len(link_text)

        if not (entity_name.startswith("Category:") or entity_name.startswith("File:")):
            links.append({"begin": begin, "end": end, "link": entity_name, "text": link_text})

    return out, links


def parse_wikipedia_dump(dump_file: str) -> Generator[Tuple[str, str, str, List[Dict], bool], None, None]:
    cleaner = wiki.cleaner.Cleaner()
    for id, title, text in wiki.iterate(dump_file):
        text = cleaner.clean_text(text)
        assert id is not None, f"ID is None for title: {title}"
        assert title is not None, f"Title is None for ID: {id}"
        
        # Check for redirection page
        lower_text = text.lower().strip()
        match = REDIRECT_REGEX.match(lower_text)
        if match is None:
            match = REDIRECT_REGEX_2.match(lower_text)
        if match is not None:
            # Is a redirect
            redirect_name = match.group(1)
            yield id, title, redirect_name, None, True
        else:
            # Is a normal page
            # Their version is very complicated and buggy, so I just implemented it using regex
            # cleaned_text, links = cleaner.build_links(text)
            cleaned_text, links = my_build_links(text)
            
            match = REDIRECT_REGEX.match(cleaned_text)
            if match is None:
                match = REDIRECT_REGEX_2.match(cleaned_text)
            if match is not None:
                print(cleaned_text)
                yield id, title, match.group(1), None, True
                continue
            yield id, title, cleaned_text, links, False

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


def parse_text_into_sections(title, text):
    # Regular expression to match headers marked by "=" signs
    section_pattern = re.compile(r"=+\s*([^=\n]+)\s*=+")
    # Split text into sections using the headers
    matches = list(section_pattern.finditer(text))

    if len(matches) == 0:
        print(title)
        print('---')
        print(text)
        print('---')
        print(text.strip().startswith('Category:'))
        return []

    starting_section = {'header': title, 'content': text[0:matches[0].start()], 'section_id': 0, 'start_index': 0, 'end_index': matches[0].start()}
    
    # Create a list of dictionaries with headers and their corresponding content
    parsed_sections = [starting_section]
    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start_position = match.end()  # The content starts after the header
        end_position = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start_position:end_position].strip()
        parsed_sections.append({
            "header": header,
            "content": content,
            'section_id': i + 1,
            'start_index': start_position,
            'end_index': end_position,
        })
    
    return parsed_sections


def parse(dump_file_path: str, device: str) -> Generator[Tuple[str, str, int, int, str], None, None]:
    coref = FCoref(device=device, enable_progress_bar=False)
    for id, title, text, links, is_redirect in parse_wikipedia_dump(dump_file_path):
        if is_redirect:
            yield id, title, text, links, is_redirect 
        else:
            coref_clusters = get_coref_clusters(coref, [text])[0]
            for entity_start, entity_end, entity_name in get_all_linked_entities(coref_clusters, links):
                yield id, title, entity_start, entity_end, entity_name

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
    for id, title, text, links, is_redirect in parse_wikipedia_dump(dump_file_path):
        ents = []
        sections = []
        if not is_redirect:
            # Only normal pages have entities.
            coref_clusters = get_coref_clusters(coref, [text])[0]
            ents = []
            for entity_start, entity_end, entity_name in get_all_linked_entities(coref_clusters, links):
                ents.append({"entity_start": entity_start, "entity_end": entity_end, "entity_name": entity_name})
            del coref_clusters    
            torch.cuda.empty_cache()
            
            sections = parse_text_into_sections(title, text)

        metrics['text_length'] += len(text)
        metrics['token_count'] += len(tokenizer.encode(text))
        metrics['word_count'] += len(text.split(' '))
        metrics['article_count'] += 1
        metrics['redirect_count'] += is_redirect
        metrics['entity_count'] += len(ents)
        writer.write({"src_file": dump_file_path, "id": id, "text": text, "title": title, "entities": ents, 'is_redirect': is_redirect, 'sections': sections})

    writer.close()
    with open(writer.out_base_path + '_metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True)
    parser.add_argument("--output", type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if args.output:
        write(args.dump_file, args.output, args.device)
    else:
        parse(args.dump_file, args.device)
