from qid_mapping import (
    load_aliases,
    generate_suffix_variations,
)


def filter_short_aliases(aliases):
    exceptions = {"US", "USA", "UK", "UAE"}
    return [alias for alias in aliases if len(alias) > 3 or alias in exceptions]

def build_alias_query(aliases):
    """
    Build an Elasticsearch query with 'should' match_phrase clauses for aliases
    and a highlight block that wraps all matches with <mark>...</mark> and returns the full text as one fragment.

    Args:
        aliases (list of str): List of alias strings to match.
        fragment_size (int): Size of the highlight fragment. Default is large to return full field.

    Returns:
        dict: Full Elasticsearch query with query and highlight blocks.
    """
    if not aliases:
        return {"query": {"match_none": {}}}  # Safe fallback for empty alias list

    should_clauses = [{"match_phrase": {"text": alias}} for alias in aliases if alias]

    return {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        },
        "highlight": {
            "fields": {
                "text": {}
            },
            "number_of_fragments": 0,
        }
    }

def build_entity_alias_search_query(entity_name, filter_func=None):
    """
    Build an Elasticsearch query that matches entity aliases using properly
    formatted regex patterns and match_phrase queries
    
    Args:
        entity_name: Name of the entity to search for
        
    Returns:
        Elasticsearch query dict
    """
    aliases_dict = load_aliases(filter_func)
    aliases = aliases_dict.get(entity_name, [entity_name])
    if entity_name not in aliases:
        aliases.append(entity_name)

    return build_alias_query(aliases)
    
def build_entity_search_query(entity_name, filter_func=None):
    """
    Build a query for exact matches of an entity name in the Elasticsearch index.

    Args:
        entity_name (str): The entity name to search for (exact match).

    Returns:
        Elasticsearch query dict
    """
    entity_name_variations = generate_suffix_variations([entity_name])

    if not entity_name_variations:
        return {"query": {"match_none": {}}}  # Safe fallback if no variations generated

    should_clauses = [{"match_phrase": {"text": name_var}} for name_var in entity_name_variations if name_var]

    return {
        "query": {
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        },
        "highlight": {
            "fields": {
                "text": {}
            },
            "number_of_fragments": 0
        }
    }
    
# Both functions below are necessary because the scores for searching two entities together in a single query
# might differ from the scores when searching for them separately. This is due to Elasticsearch's scoring
# mechanisms, which consider query structure, term relationships, and field norms.
def build_two_entity_search_query(entity1, entity2):
    """
    Build a query for exact matches of two entity names on text.

    Args:
        entity1 (str): The first entity name to search for.
        entity2 (str): The second entity name to search for.

    Returns:
        dict: Elasticsearch query dict.
    """
    raise Exception("Unimplemented")
    # entity1_variations = generate_suffix_variations([entity1])
    # entity2_variations = generate_suffix_variations([entity2])

    # return {
    #     "query": {
    #         "bool": {
    #             "must": [
    #                 {"bool": {"should": [{"match_phrase": {"text": name_var}} for name_var in entity1_variations], "minimum_should_match": 1}},
    #                 {"bool": {"should": [{"match_phrase": {"text": name_var}} for name_var in entity2_variations], "minimum_should_match": 1}}
    #             ]
    #         }
    #     }
    # }

def build_two_entity_alias_search_query(entity1, entity2, filter_func=None):
    """
    Build an Elasticsearch query that matches aliases of two entities using properly
    formatted regex patterns and match_phrase queries.

    Args:
        entity1 (str): The first entity name to search for.
        entity2 (str): The second entity name to search for.

    Returns:
        dict: Elasticsearch query dict.
    """
    raise Exception("Unimplemented")
    # aliases_dict = load_aliases(filter_func)
    # aliases1 = aliases_dict.get(entity1, [entity1])
    # aliases2 = aliases_dict.get(entity2, [entity2])

    # if entity1 not in aliases1:
    #     aliases1.append(entity1)
    # if entity2 not in aliases2:
    #     aliases2.append(entity2)

    # return {
    #     "query": {
    #         "bool": {
    #             "must": [
    #                 build_alias_query(aliases1),
    #                 build_alias_query(aliases2)
    #             ]
    #         }
    #     }
    # }