import hashlib
import sqlite3 
from typing import Optional

from constants import (
    CACHE_DB,
    OVERALL_STATUS_CACHE_DB
)


def initialize_overall_status_cache():
    global OVERALL_STATUS_CACHE_DB

    conn = sqlite3.connect(OVERALL_STATUS_CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS overall_status_cache (
            chunk_id INTEGER,
            entity_label TEXT,
            eval_type TEXT, 
            status TEXT,    
            PRIMARY KEY (chunk_id, entity_label, eval_type)
        )
    """)
    conn.commit()
    conn.close()

def get_overall_status_from_cache(chunk_id: int, entity_label: str, eval_type: str) -> Optional[str]:
    global OVERALL_STATUS_CACHE_DB

    conn = sqlite3.connect(OVERALL_STATUS_CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM overall_status_cache WHERE chunk_id = ? AND entity_label = ? AND eval_type = ?",
                   (chunk_id, entity_label, eval_type))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def save_overall_status_to_cache(chunk_id: int, entity_label: str, eval_type: str, status: str):
    global OVERALL_STATUS_CACHE_DB

    conn = sqlite3.connect(OVERALL_STATUS_CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO overall_status_cache (chunk_id, entity_label, eval_type, status) VALUES (?, ?, ?, ?)",
                   (chunk_id, entity_label, eval_type, status))
    conn.commit()
    conn.close()

def initialize_cache():
    global CACHE_DB

    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id TEXT PRIMARY KEY,
            entity_name TEXT,
            text TEXT,
            response TEXT,
            input_token_count INTEGER,
            output_token_count INTEGER
        )
    """)
    conn.commit()
    conn.close()

def get_cached_response(entity_name, text, is_implicit):
    global CACHE_DB

    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    query_id = hashlib.sha256(f"{entity_name}:{text}:{is_implicit}".encode()).hexdigest()
    cursor.execute("SELECT response FROM cache WHERE id = ?", (query_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_cached_text(entity_name, cid, is_implicit):
    global CACHE_DB

    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    query_id = hashlib.sha256(f"{entity_name}:{text}:{is_implicit}".encode()).hexdigest()
    cursor.execute("SELECT text FROM cache WHERE id = ?", (query_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def save_to_cache(entity_name, text, is_implicit, response, prompt_token_count, candidates_token_count):
    global CACHE_DB
    
    conn = sqlite3.connect(CACHE_DB)
    cursor = conn.cursor()
    query_id = hashlib.sha256(f"{entity_name}:{text}:{is_implicit}".encode()).hexdigest()
    cursor.execute("INSERT OR REPLACE INTO cache (id, entity_name, text, response, input_token_count, output_token_count) VALUES (?, ?, ?, ?, ?, ?)",
                   (query_id, entity_name, text, response, prompt_token_count, candidates_token_count))
    conn.commit()
    conn.close()