import pytest
import inspect

from src.analytics import visualization

def test_discover_visualization_functions():
    """Discover and print all available functions in the visualization module."""
    # Get all functions/classes from the module
    module_items = inspect.getmembers(visualization)
    
    # Filter to only public functions (not starting with _)
    public_functions = [name for name, obj in module_items 
                        if inspect.isfunction(obj) and not name.startswith('_')]
    
    # Print available functions for inspection
    print("\nAvailable visualization functions:")
    for func_name in public_functions:
        func = getattr(visualization, func_name)
        sig = inspect.signature(func)
        print(f"- {func_name}{sig}")
    
    # This will always pass, it's just for discovery
    assert True