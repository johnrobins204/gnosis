import sys
from typing import Dict, Any, List, Optional

from src.analytics import run_from_config

def run(argv=None):
    """
    Run analytics from command line arguments.
    
    Arguments:
    - --input: Path to input CSV file
    - --output: Path to output CSV file
    - --group-by: Column to group by (comma-separated)
    - --metrics: Metrics to calculate (comma-separated)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analytics")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--group-by", required=True, help="Column(s) to group by (comma-separated)")
    parser.add_argument("--metrics", help="Metrics to calculate (comma-separated)")
    
    args = parser.parse_args(argv)
    
    config = {
        "input_csv": args.input,
        "output_csv": args.output,
        "group_by": args.group_by.split(",")
    }
    
    if args.metrics:
        config["metrics"] = args.metrics.split(",")
    
    result = run_from_config(config)
    if not result["success"]:
        print(f"ERROR: {result.get('error')}")
        sys.exit(1)
    
    print(f"Analytics complete, output written to {args.output}")
    return result

if __name__ == "__main__":
    run()