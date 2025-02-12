import time
from dataclasses import dataclass, field
from copy import deepcopy
from json_writer import JsonWriter
from typing import Dict, Generator, List, Tuple, NamedTuple, Optional
import argparse
import torch
from tokenizers import Tokenizer
import mwparserfromhell
from mwparserfromhell.nodes import *
import mwxml
import wandb
import os
import sys
import traceback
import signal
from collections import defaultdict
import re

BAD_PREFIXES = ("file:", "image:", "category:")
BAD_TEXTS = ("thumb",)
BAD_TAGS = ("ref", "table", "gallery", "img", "figure", "figcaption") 
BAD_TEMPLATES = ("reflist", "notelist", "notelist-ua", "notelist-lr", "notelist-ur", "notelist-lg")
BAD_SECTIONS = ("references", "external links", "sources", "further reading",
                "see also", "citations", "note", "notes", "explanatory notes", "general and cited sources", 
                "primary sources", "secondary sources", "tertiary sources", "bibliography", "other",
                "subnotes", "other notes", "general bibliography", "works cited", "notes and references"
            )

WIKIPROJECT_PAGES = ("WikiProject", )
IMAGE_PAGES = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp", ".tiff", ".ico")
DISAMBIGUATION_PAGES = ("(Disambiguation)", "(disambiguation)")
DISAMBIGUATION_PHRASES = (" may refer to", " may mean", " can refer to", " also refer to")
OTHER_PAGES = ("Wikipedia", "Wikipedia talk:", "File:", "File talk:", "MediaWiki:", 
                            "MediaWiki talk:", "Template:", "Template talk:", "Help:", "Help talk:",
                            "Category:", "Category talk:", "Portal:", "Portal talk:", "Draft:", "Draft talk:", "MOS:", "MOS talk:", "TimedText:",
                            "TimedText talk:", "Module:", "Module talk:", "List of"
                        )


ARTICLE_TEMPLATE = (r"{{Short description|",)
DISAMBIGUATION_TEMPLATE = (r"{{Disambiguation|", r"{{Wiktionary|")
TEMPLATE = (r"{{tl|", )
CATEGORY_TEMPLATE = (r"{{c|", r"{{cl|", r"{{cls|", r"{{lc|", r"{{lcs|", r"{{cconly|")

REDIRECT_NAMESPACE = "Redirect"
ARTICLE_NAMESPACE = "Article"
WIKIPROJECT_NAMESPACE = "WikiProject"
DISAMBIGUATION_NAMESPACE = "Disambiguation"
IMAGE_NAMESPACE = "Image"
TEMPLATE_NAMESPACE = "Template"
CATEGORY_NAMESPACE = "Category"
GENERAL_NOT_ARTICLE_NAMESPACE = "Not Article"
UNKNOWN_NAMESPACE = "Unknown" # for debugging

def get_namespace(title, sections, wikicode):
    # Convert templates to lowercase for efficient checking
    templates = [str(template).lower() for template in wikicode.filter_templates(recursive=False)]

    # Check for article templates indicating article pages
    if any(good.lower() in tmpl for good in ARTICLE_TEMPLATE for tmpl in templates):
        return ARTICLE_NAMESPACE

    # Check for other non-article pages title suffixes and prefixes
    title_lower = title.lower()
    for pattern in DISAMBIGUATION_PAGES:
        if title_lower.endswith(pattern.lower()):
            return DISAMBIGUATION_NAMESPACE
    for pattern in IMAGE_PAGES:
        if title_lower.endswith(pattern.lower()):
            return IMAGE_NAMESPACE  
    for pattern in WIKIPROJECT_PAGES:
        if pattern.lower() in title_lower:
            return WIKIPROJECT_NAMESPACE   
    for prefix in OTHER_PAGES:
        if title_lower.startswith(prefix.lower()):
            return prefix.rstrip(":")

    # Captures many types of non-article pages.
    match = re.compile(r'^([^/]+)/[^/]+(?:/[^/]+)*$').match(title_lower)
    if match:
        return GENERAL_NOT_ARTICLE_NAMESPACE
    
    match = re.compile(r'^.+-stub$').match(title_lower)
    if match:
        return TEMPLATE_NAMESPACE
    
    # Check for other pages (non-article) templates
    for pattern in DISAMBIGUATION_TEMPLATE:
        if any(pattern.lower() in tmpl for tmpl in templates):
            return DISAMBIGUATION_NAMESPACE
    for pattern in TEMPLATE:
        if any(pattern.lower() in tmpl for tmpl in templates):
            return TEMPLATE_NAMESPACE
    for pattern in CATEGORY_TEMPLATE:
        if any(pattern.lower() in tmpl for tmpl in templates):
            return CATEGORY_NAMESPACE
    
    if any(pattern.lower() in sections[0]["content"] for pattern in DISAMBIGUATION_PHRASES):
        return DISAMBIGUATION_NAMESPACE
    
    if len(sections) == 1:
        return GENERAL_NOT_ARTICLE_NAMESPACE

    return ARTICLE_NAMESPACE

def clean_wikicode(wikicode):
    """
    When we process the parsed data, we should make sure we remove the following sections... 
    https://arxiv.org/pdf/2112.04426
        We first parse articles using mwparserfromhell5. We then remove sections with the following
        titles: "references", "external links", "sources", "further reading", "see also", "citations", and "note". In
        the remaining sections, we remove Wikilinks and remove the following templates: "reflist", "notelist",
        "notelist-ua", "notelist-lr", "notelist-ur", and "notelist-lg". We also exclude objects with the "ref" or
        "table" tag and clean the remaining text with the strip_code function.
    """
    tags = wikicode.filter_tags(matches=lambda n: any(bad in n.tag for bad in BAD_TAGS), recursive=False)
    for tag in tags:
        wikicode.remove(tag)

    templates = wikicode.filter_templates(matches=lambda n: any(str(n.name).lower().strip().startswith(bad) for bad in BAD_TEMPLATES), recursive=False)
    for template in templates:
        wikicode.remove(template)

    wikilinks = wikicode.filter_wikilinks(matches=lambda n: any(n.title.strip_code().lower().startswith(bad) for bad in BAD_PREFIXES), recursive=False)
    for link in wikilinks:
        wikicode.remove(link)

    return wikicode

def parse(wikicode):
    normalize, collapse, keep_template_params = True, False, False
    kwargs = {
        "normalize": normalize,
        "collapse": collapse,
        "keep_template_params": keep_template_params,
    }
    entities = []
    start_index = 0
    nodes = []
    for node in wikicode.nodes:
        stripped = node.__strip__(**kwargs)
        if stripped:
            text = str(stripped)
            nodes.append(text)
            if isinstance(node, Wikilink):
                entities.append({
                        "entity_start": start_index,
                        "entity_end": start_index + len(text),
                        "entity_text": text,
                        "entity_name": node.title.strip_code()
                    }
                )
            start_index += len(text)
    return "".join(nodes), entities

def get_sections(wikicode, cleaned_text):
    sections = wikicode.get_sections(flat=True, include_lead=True, levels=[1, 2, 3, 4, 5, 6])
    flat_list = []
    for section in sections:
        headings = section.filter_headings()
        # Extract the immediate content of this section (excluding sub-sections)
        if headings:
            section_title = headings[0].title.strip_code().strip()
            heading_level = headings[0].level
            
            section_text = section.strip_code(collapse=False)
            section_start = cleaned_text.find(section_text)
            section_end = section_start + len(section_text)

            flat_list.append({
                "title": section_title,
                "level": heading_level,
                "content": section_text,
                "begin": section_start,
                "end": section_end
            })

    return flat_list

@dataclass
class WikipediaPageOutput:
    id: str                               # Page ID
    title: str                            # Page title
    text: str = ""                        # Cleaned article text
    sections: List[Dict] = field(default_factory=list)  # List of section dictionaries
    entities: List[Dict] = field(default_factory=list)  # List of entity dictionaries (can be None)
    namespace: str = ""                   # Namespace for non-article pages (disambiguation, etc.)
    start_time: float = 0.0               # Timestamp when processing started
    error: str = ""                       # Error message if an error occurred

def parse_wikipedia_dump(dump_file: str, start_id: id) -> Generator[WikipediaPageOutput, None, None]:
    print(f"Starting after {start_id}")
    skip = True if start_id > -1 else False
    
    for page in mwxml.Dump.from_file(dump_file):
        if page.id == start_id:
            skip = False
            continue
        if skip:
            continue
        
        start_time = time.time()
        
        # Don't parse redirect pages.
        if page.redirect:
            yield WikipediaPageOutput(id=page.id, title=page.title, namespace=REDIRECT_NAMESPACE, start_time=start_time)
            continue

        try:
            last_revision = next(revision for revision in page if revision)
            text = last_revision.text
            original_wikicode = mwparserfromhell.parse(text)
            wikicode = deepcopy(original_wikicode)

            # Insert a title Heading
            wikicode.nodes.insert(0, Text(" "))
            wikicode.nodes.insert(0, Heading(title=page.title, level=1))
            
            wikicode = clean_wikicode(wikicode)
            cleaned_text, entities = parse(wikicode)
            sections = get_sections(wikicode, cleaned_text)

            # Make small adjustment to namespace.
            namespace = get_namespace(page.title, sections, original_wikicode)

            for link in entities:
                assert cleaned_text[link["entity_start"]:link["entity_end"]] == link["entity_text"], f"Entity indexes don't align: {link}"

            for section in sections:
                assert cleaned_text[section["begin"]:section["end"]] == section["content"], f"Section indexes don't align: {section}"

            yield WikipediaPageOutput(id=page.id, title=page.title, text=cleaned_text, sections=sections, entities=entities, namespace=namespace, start_time=start_time)
        except Exception as e:
            # We propagate the error so we can record the problematic pages and continue.
            yield WikipediaPageOutput(id=page.id, title=page.title, start_time=start_time, error=traceback.format_exc())


def write(dump_file_path: str, writer: JsonWriter):
    print(f"Writing to {writer.file.name}")
    tokenizer = Tokenizer.from_file("/home/morg/students/gottesman3/OLMo/olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json")
    metrics = defaultdict(int)
    
    start_id = writer.last_processed_doc

    for output in parse_wikipedia_dump(dump_file_path, start_id):
        if len(output.error) > 0:
            print(f"FAILED TO PARSE FILE -- page: {output.id}, title: {output.title}, error: {output.error}")
            metrics['error_count'] += 1
            writer.write({"src_file": dump_file_path, "id": output.id, "title": output.title, "error": output.error})
        elif output.namespace == REDIRECT_NAMESPACE:
            metrics['redirect_count'] += 1
            writer.write({"src_file": dump_file_path, "id": output.id, "title": output.title, 'namespace': output.namespace})
        else:
            namespace = "_".join(output.namespace.split(" ")).lower()
            metrics[f'{namespace}_count'] += 1
            metrics[f'{namespace}_text_length'] += len(output.text)
            metrics[f'{namespace}_word_count'] += len(output.text.split(' '))
            metrics[f'{namespace}_entity_count'] += len(output.entities)
            
            warning = ""
            try:
                tokens = tokenizer.encode(output.text)
            except Exception as e:
                output.text = output.text.encode('utf-8','replace').decode('utf-8')
                tokens = tokenizer.encode(output.text)
                warning = f"WARNING -- removed tokens that aren't compliant with utf-8"
            metrics[f'{namespace}_token_count'] += len(tokens)
            
            record = {
                "src_file": dump_file_path, 
                "id": output.id, 
                "namespace": output.namespace, 
                "text": output.text, 
                "title": output.title, 
                "entities": output.entities, 
                "sections": output.sections
            }
            
            if len(warning) > 0:
                record["warning"] = warning
            
            writer.write(record)

        end_time = time.time()
        wandb.log({**metrics, 'iteration_time': end_time - output.start_time})
    
    writer.close()    

def global_exception_handler(writer):
    def _global_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, torch.cuda.OutOfMemoryError):
            print("Global Handler: Caught CUDA Out of Memory Error.")
            print("Exception details:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            writer.cleanup()
        else:
            # Optionally handle other exceptions or pass them to the default handler
            print("Global Handler: Caught an exception.")
            traceback.print_exception(exc_type, exc_value, exc_traceback)
            writer.cleanup()
        # Call the default excepthook to ensure standard behavior
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    return _global_exception_handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_file", type=str, required=True)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    
    wandb.login()
    run = wandb.init(project="process_wiki_simple_dump", name=os.path.basename(args.dump_file))
    
    writer = JsonWriter(args.output, verbose=True)

    handler = global_exception_handler(writer)
    # Register the exception handler
    sys.excepthook = handler
    # Register the signal handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, writer.cleanup)
    signal.signal(signal.SIGTERM, writer.cleanup)

    try:
        write(args.dump_file, writer)
    except Exception as e:
        stack_trace = traceback.format_exc()
        print(stack_trace)
        writer.cleanup()

