import csv

ENTITY_ALIAS_FILE = '/home/morg/students/gilaiedotan2/subjects_aliases_popQA_descriptions.csv'

entity_to_qid_map = {}
qid_to_entity_map = {}
entity_label_to_description_map = {} # For descriptions, keyed by sub_label

def load_entity_maps():
    """Loads mappings from entity label to QID, QID to label, and label to description."""
    global entity_to_qid_map, qid_to_entity_map, entity_label_to_description_map
    if entity_to_qid_map: # Already loaded
        return

    with open(ENTITY_ALIAS_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sub_label = row['sub_label']
            sub_uri = row['sub_uri']
            entity_to_qid_map[sub_label] = sub_uri
            qid_to_entity_map[sub_uri] = sub_label
            entity_label_to_description_map[sub_label] = f'{sub_label} - {row["sub_description_text"]}'
    print(f"Loaded QID and description maps for {len(entity_to_qid_map)} entities.")

# Call this once to populate the maps
# load_entity_maps() # Will be called later explicitly or when es client is setup

def get_qid_for_entity(entity_name):
    load_entity_maps() # Ensure maps are loaded
    return entity_to_qid_map.get(entity_name)

def get_entity_for_qid(qid):
    load_entity_maps() # Ensure maps are loaded
    return qid_to_entity_map.get(qid)

def get_description_for_entity_label(entity_label):
    load_entity_maps() # Ensure maps are loaded
    return entity_label_to_description_map.get(entity_label, entity_label)


def generate_suffix_variations(entities):
    """
    Generate all possible suffix variations for a given entity or list of entities.

    Args:
        entities (str or list): The base entity name or a list of entity names.

    Returns:
        list: A list containing all variations with common suffixes for the given entities.
    """
    if isinstance(entities, str):
        entities = [entities]
    
    suffixes = ["", "'s", "’s"]
    variations = [f"{entity}{suffix}" for entity in entities for suffix in suffixes]
    return variations

# Read the CSV file to extract aliases
def load_aliases(filter_func=None):
    load_entity_maps() # Ensure maps are loaded as this file is the source
    aliases_dict = {}
    with open(ENTITY_ALIAS_FILE, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            aliases = row['sub_aliases'].split(';')
            # Filter out empty strings that might result from trailing semicolons or empty alias fields
            aliases = [alias.strip() for alias in aliases if alias.strip()]
            if filter_func:
                aliases = filter_func(aliases)
            aliases_with_suffixes = generate_suffix_variations(aliases) 
            aliases_dict[row['sub_label']] = aliases_with_suffixes
    return aliases_dict

# Read the CSV file to extract entities with their descriptions
def load_entity_description_dict():
    load_entity_maps() # Ensures entity_label_to_description_map is populated
    return entity_label_to_description_map.copy() # Return a copy

