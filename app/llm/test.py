from pathlib import Path
import json
from functools import lru_cache
from typing import Set

@lru_cache
def load_banned_words(path: str | Path = "locals/ban_kwds.json") -> Set[str]:
    """Return a set of banned words loaded from a JSON file."""
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {w.lower() for w in data.get("banned_words", [])}
    except Exception as exc:
        raise RuntimeError(f"Could not load banned words: {exc}") from exc


def contains_offensive_language(text: str, kwds_path: str | Path = "locals/ban_kwds.json") -> bool:
    """Return True if *text* contains any banned keyword (case-insensitive)."""
    prompt = text.lower()
    banned = load_banned_words(kwds_path)
    return any(word in prompt for word in banned)
        
result = contains_offensive_language('¿qué me recomiendas para conjuntar esta mierda de traje?', 'locals/ban_kwds.json')
print(result)
        
    
    
    






    
