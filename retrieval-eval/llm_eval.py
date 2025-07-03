import os
from google.genai import types
from cache import (
    save_to_cache,
    get_cached_response
)
from qid_mapping import get_description_for_entity_label
from constants import GEMINI_API_KEY

def create_explicit_mention_prompt(entity_name, text):
    """
    Create a prompt for explicit mention detection of an entity in a given text.

    Args:
        entity_name (str): The name of the entity.
        text (str): The text to analyze.

    Returns:
        str: The explicit mention prompt.
    """
    entity_description = get_description_for_entity_label(entity_name)
    return f"""Determine if the provided text **directly mentions or discusses** the entity `{entity_description}`. Do not infer or assume the entity's presence based on related individuals or contexts. Answer only with 'Yes' or 'No'.

Example 1:
Entity: Albert Einstein
Text: theory of relativity transformed our understanding of physics. Einstein's famous equation E=mc² became a cornerstone of modern science. His work on
Answer: Yes

Example 2:
Entity: Donald Trump
Text: renegotiate her prenuptial agreement. Her stepdaughter Ivanka Trump fulfilled some of the first lady's traditional duties. She kept
Answer: No

Task:
Entity: `{entity_description}`
Text: {text}
Answer:"""

def create_implicit_mention_prompt(entity_name, text):
    """
    Create a prompt for implicit mention detection of an entity in a given text.

    Args:
        entity_name (str): The name of the entity.
        text (str): The text to analyze.

    Returns:
        str: The implicit mention prompt.
    """
    entity_description = get_description_for_entity_label(entity_name)
    return f"""You are a precise linguistic judge. Your task is to determine if a specific word or phrase *within* the `TEXT` directly **acts as a substitute for** or **refers back to** the specific `ENTITY`. Evaluate based **both on the text and your general knowledge**, but apply **MAXIMUM critical scrutiny** regarding certainty. Your primary goal is to avoid false positives (wrongly assigning Score 3).
This means a pronoun ('it', 'he', 'they'), definite description, or alias is used *instead of* the `ENTITY`'s name.
**CRITICAL RULE 1:** Associated concepts (products, etc.) are NOT substitutes for the entity itself (Score 1 or 2, not 3).
**CRITICAL RULE 2:** Conflicting details (wrong dates, names, facts) indicate certain absence (Score 1). Doubtful details prevent Score 3.
**CRITICAL RULE 3 (The Exclusivity Test - MANDATORY for Pronoun/Description cases):** When assessing a pronoun ('He/She/It/They') linked to a description:
   1. Use knowledge to identify potential matches, including the provided `ENTITY`.
   2. **Perform a Negative Check:** Ask 'Could this description plausibly fit **ANY OTHER** prominent entity of the same type (competitor, alternative, similar figure)?' Be thorough; consider common knowledge associations (e.g., Who is most famously associated with X?).
   3. **Assign Score 3 ONLY IF:** The description is an **OBJECTIVELY UNIQUE IDENTIFIER** (e.g., 'world's highest peak', '44th US President') **AND** the Negative Check (Step 2) confirms **NO OTHER** plausible prominent entity fits. **Mere 'best fit' is insufficient.**
   4. **Assign Score 2 IF:** The description fits the `ENTITY` but ALSO plausibly fits others (Negative Check fails), OR if the description is subjective ('greatest'), or describes general characteristics/performance/history that are not objectively unique.
Your response MUST be ONLY a single integer: 1, 2, or 3.

Score Definitions:
1: **Certain Absence.** No substitute phrase exists. OR, text details *directly contradict* the `ENTITY` (Rule 2).
2: **Uncertain / Needs More Context (DEFAULT for ambiguity).** A phrase *might* substitute, but **exclusivity is NOT proven** within the snippet. This is the **correct score** for descriptions fitting the `ENTITY` but also potentially fitting competitors/alternatives (Rule 3, Negative Check fails), or for subjective/generic descriptions, or when minor doubts exist. Assume ambiguity unless exclusivity is undeniable.
3: **Certain Presence (Requires Undeniable Proof).** A phrase acts as an **OBJECTIVELY and EXCLUSIVELY** unique substitute known to refer only to the `ENTITY` (passing Rule 3's Negative Check rigorously), OR has an unambiguous antecedent *in the text*. Must have **ZERO** conflicting/doubtful details (Rule 2).

**MAXIMIZE SKEPTICISM.** Avoid Score 3 unless the text provides *irrefutable proof* of exclusive reference. When evaluating pronouns+descriptions, Score 2 is the standard outcome unless the description is objectively unique AND survives the negative check. Do NOT 'guess' or choose the 'most likely' fit for Score 3.

--- EXAMPLES ---

ENTITY: Microsoft (Company)
TEXT: Many users rely on Windows operating systems daily. Software development continues to evolve.
SCORE: 1 (Mentions associated product, no substitute phrase. Rule 1.)

ENTITY: Elizabeth II (UK Queen)
TEXT: is also a patron of the Queen Elisabeth Music Competition, an international competition founded in 1937 as an initiative of Queen Elisabeth and...
SCORE: 1 (Conflicting details - spelling, date - prove it's not the target entity. Rule 2.)

ENTITY: Buzz Aldrin (Astronaut)
TEXT: After the mission, his wife realized he had fallen into a depression, something she had not seen before.
SCORE: 2 (Description strongly associated with Aldrin but not objectively unique to ONLY him *within this snippet*. Negative check might find other astronauts with issues. Rule 3 demands skepticism -> Score 2.)

ENTITY: World Wildlife Fund (Organization)
TEXT: It is an international non-governmental organization founded in the 20th century that works in the field of wilderness preservation...
SCORE: 2 (Descriptive summary, but other large international conservation NGOs exist. This could fit IUCN, Conservation Int'l, etc.? Yes, plausibly. Not demonstrably exclusive -> Score 2.)

ENTITY: Mount Everest (Mountain)
TEXT: Climbing the world's highest peak requires immense preparation. That colossal mountain tests endurance.
SCORE: 3 ('world's highest peak' is objectively unique; negative check confirms no other mountain fits. Rule 3 unique.)

ENTITY: Barack Obama
TEXT: As the 44th US President, he signed the Affordable Care Act into law.
SCORE: 3 ('the 44th US President' is objectively unique; negative check confirms no other person fits. Rule 3 unique.)

--- TASK ---

ENTITY: `{entity_description}`
TEXT: {text}
SCORE:"""

def generate(client, entity_name, text_window, is_implicit=False):
    global GEMINI_API_KEY
    # Check cache first
    cached_response = get_cached_response(entity_name, text_window, is_implicit)
    if cached_response:
        return cached_response
    if is_implicit:
        input_text = create_implicit_mention_prompt(entity_name, text_window)
    else:
        input_text = create_explicit_mention_prompt(entity_name, text_window)

    model = "gemini-2.5-flash-lite-preview-06-17" #'gemini-2.0-flash' (db_2) #"gemini-2.5-flash-preview-04-17" (db) "gemini-2.5-flash-lite-preview-06-17" (db_3)
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_text),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
        thinking_config = types.ThinkingConfig(include_thoughts=False, thinking_budget=0)
    )

    response = ""
    prompt_token_count, candidates_token_count = 0, 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        # Handle the text part
        text = getattr(chunk, "text", None)
        if text is not None:
            response += text
            if hasattr(chunk, "usage_metadata"):
                prompt_token_count = chunk.usage_metadata.prompt_token_count
                candidates_token_count = chunk.usage_metadata.candidates_token_count
        else:
            finish_reason = getattr(chunk.candidates[0], "finish_reason", None) if getattr(chunk, "candidates", []) else None
            if finish_reason == "STOP":
                # Normal final chunk, no need to warn
                pass
            else:
                print(f"Warning: Received chunk with None text: {chunk}")
        
    # Save the result to the cache (can add validation that the response is formatted correctly here if needed, but from the tests that I've done, it seems to be consistent)
    save_to_cache(entity_name, text_window, is_implicit, response, prompt_token_count, candidates_token_count)
    
    return response