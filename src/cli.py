
import argparse
import sys
from typing import List
from orchestrator import orchestrate
from logging_config import get_logger

_logger = get_logger("cli")



from typing import Optional

def run(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="sop-orchestrator")
    parser.add_argument("--config", required=True, help="Path to run YAML config")
    args = parser.parse_args(argv)

    res = orchestrate(args.config)
    if res.get("success"):
        return 0
    else:
        for e in res.get("errors", []):
            _logger.error("%s", e)
        return 5


if __name__ == "__main__":
    raise SystemExit(run())