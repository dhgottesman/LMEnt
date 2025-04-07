import os
import re
from refined.inference.processor import Refined
import bz2
import xml.etree.ElementTree as ET

def strip_ns(elem):
    """Recursively remove namespace from tags."""
    elem.tag = elem.tag.split("}")[-1]  # Remove namespace
    for child in elem:
        strip_ns(child)
        
def extract_sample(input_file, output_file, num_pages):
    """
    Extracts a sample of Wikipedia pages from a bz2-compressed XML dump file.
    """
    with bz2.open(input_file, "rt", encoding="utf-8") as f:
        context = ET.iterparse(f, events=("start", "end"))
        root = None
        count = 0
        for event, elem in context:
            if event == "start" and elem.tag.endswith("mediawiki"):
                root = elem  # Capture the root element
            if event == "end" and elem.tag.endswith("page"):
                count += 1
                if count > num_pages:
                    break  
            
        if root is not None:
            strip_ns(root)
            tree = ET.ElementTree(root)
            tree.write(output_file, encoding="utf-8", xml_declaration=False)


def get_refined_model():
    """
    Initialize the Refined model for entity linking.
    This function is called only once to avoid re-initializing the model multiple times.
    """
    return Refined.from_pretrained(model_name='wikipedia_model', # model name has several options, look at the documentation.
                                   entity_set="wikipedia") # Entity set is between "wikipedia" and "wikidata", for us wikipedia is relevant.


def parse_wikipedia_article(input_file, article_title):
    """
    Parses a Wikipedia article text file, cleans it from markup, and extracts link information.

    Args:
        input_file (str): Path to the input text file.
        article_title (str): The title of the article being parsed.

    Returns:
        tuple: A tuple containing:
            - cleaned_text (str): The text of the article with Wikipedia markup removed.
            - article_to_links (dict): Dictionary mapping the article title to a list of links.
            - link_to_articles (dict): Dictionary mapping link titles to a list of articles where they appear.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    cleaned_text = ""
    article_to_links = {article_title: []}
    link_to_articles = {}
    current_cleaned_index = 0

    # Regular expression to find links in the format [[Link Title]] or [[Link Title|Display Text]]
    link_regex = r'\[\[([^\]]+?)\]\]'

    for match in re.finditer(link_regex, text):
        full_match = match.group(0)
        link_content = match.group(1)
        start_index = match.start()
        end_index = match.end()

        parts = link_content.split('|')
        link_title = parts[0].strip()
        if len(parts) > 1:
            display_text = parts[1].strip()
        else:
            display_text = link_title

        # Append the text before the link to the cleaned text
        cleaned_text += text[current_cleaned_index:start_index]

        # Calculate the span of the display text in the cleaned text
        span_start = len(cleaned_text)
        cleaned_text += display_text
        span_end = len(cleaned_text)
        span = (span_start, span_end)

        # Update article_to_links
        article_to_links[article_title].append({
            'link_title': link_title,
            'span': span
        })

        # Update link_to_articles
        if link_title not in link_to_articles:
            link_to_articles[link_title] = []
        link_to_articles[link_title].append({
            'article_title': article_title,
            'span': span
        })

        current_cleaned_index = end_index

    # Append any remaining text after the last link
    cleaned_text += text[current_cleaned_index:]

    return cleaned_text, article_to_links, link_to_articles


def process_article(input_file, article_title, refined_model):
    """
    Processes a single Wikipedia article using both parsing tools and calculates statistics.

    Args:
        input_file (str): Path to the input text file.
        article_title (str): The title of the article being processed.

    Returns:
        tuple: A tuple containing:
            - article_entity_mentions (list): List of dictionaries containing span and entity title for this article.
            - article_links (list): List of unique entity titles found in this article by both tools.
            - tool1_links (set): Set of (entity_title, span) found by tool 1.
            - tool2_links (set): Set of (entity_title, span) found by tool 2.
    """
    cleaned_text, tool1_article_to_links, _ = parse_wikipedia_article(input_file, article_title)

    refined_links = refined_model.process_text(cleaned_text)

    article_entity_mentions = []
    tool2_links_set = set()
    for link in refined_links:
        if link.predicted_entity:
            entity_title = link.predicted_entity.wikipedia_entity_title
            span = (link.start, link.start + link.ln)
            article_entity_mentions.append({'span': span, 'entity_title': entity_title})
            tool2_links_set.add((entity_title, span))

    tool1_links_set = set()
    if article_title in tool1_article_to_links:
        for link_info in tool1_article_to_links[article_title]:
            tool1_links_set.add((link_info['link_title'], link_info['span']))

    # Combine links from both tools for the article
    all_links_in_article = set()
    for title, span in tool1_links_set:
        all_links_in_article.add(title)
    for entity in article_entity_mentions:
        all_links_in_article.add(entity['entity_title'])

    return article_entity_mentions, list(all_links_in_article), tool1_links_set, tool2_links_set

def process_directory(directory_path):
    """
    Processes all articles in the given directory, aggregates results, and calculates statistics.

    Args:
        directory_path (str): Path to the directory containing the article files.

    Returns:
        tuple: A tuple containing:
            - all_entity_mentions (dict): Dictionary of entity mentions per article.
            - all_article_links (dict): Dictionary of links per article.
            - statistics (dict): Dictionary containing the overlap statistics.
    """
    all_entity_mentions = {}
    all_article_links = {}
    total_tool1_links = 0
    total_tool2_links = 0
    common_links = 0

    article_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Initialize the refined model
    refined_model = get_refined_model()

    for filename in article_files:
        if not filename.endswith(".txt"): # Assuming text files
            continue
        article_title = filename[:-4]  # Remove .txt extension
        input_file_path = os.path.join(directory_path, filename)

        mentions, links, tool1_set, tool2_set = process_article(input_file_path, article_title, refined_model)

        all_entity_mentions[article_title] = mentions
        all_article_links[article_title] = links

        total_tool1_links += len(tool1_set)
        total_tool2_links += len(tool2_set)
        common_links += len(tool1_set.intersection(tool2_set))

    statistics = {}
    if total_tool1_links > 0:
        statistics['recall_tool2'] = common_links / total_tool1_links
    else:
        statistics['recall_tool2'] = 0.0

    if total_tool2_links > 0:
        statistics['precision_tool2'] = common_links / total_tool2_links
    else:
        statistics['precision_tool2'] = 0.0

    statistics['total_links_tool1'] = total_tool1_links
    statistics['total_links_tool2'] = total_tool2_links
    statistics['common_links'] = common_links

    return all_entity_mentions, all_article_links, statistics

