"""
Utility functions for the analytics module.
Add general-purpose helpers here to avoid duplication and keep other modules clean.
"""

# Example utility function (replace or expand as needed)

from typing import Union

def safe_divide(a: Union[int, float], b: Union[int, float]) -> float:
    """Safely divide a by b, returning 0 if b is zero."""
    return float(a) / float(b) if b else 0.0
