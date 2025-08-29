from typing import Dict, Any, Optional, List
import json
import time

from gnosis_io import load_csv, write_dataframe
from models import get_model_instance
from gnosis_types import ModelResponse
from logging_config import get_logger

_logger = get_logger("inference")


def _row_to_model_id(row: Dict[str, Any], default_model: str) -> str:
    # prefer explicit column 'model_id' then 'model' then default
    return row.get("model_id") or row.get("model") or default_model


def run_from_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _logger.info("run_from_config start for inference, keys=%s", list(cfg.keys()))
    """
    cfg keys:
        - input_csv: path to CSV with at least 'prompt' column
        - output_csv: path to write outputs
        - default_model: (optional) model identifier to use when rows don't provide one
        - model_config: (optional) dict passed to get_model_instance
        - api_params: (optional) dict passed to get_model_instance
        - temperature: (optional) float, sampling temperature
        - top_p: (optional) float, nucleus sampling probability
        - top_k: (optional) int, top-k sampling
        - seed: (optional) int, random seed
        - stop: (optional) list or str, stop sequences
        - batch_size: (optional) int, batch size for generation
        - output_format: (optional) str, output format
        - max_tokens: (optional) int, max tokens to generate
        - log_level: (optional) str, logging level
        - ... (other advanced options)
    Returns {"success": bool, "artifacts": [output_csv], "error": optional}
    """
    required = ["input_csv", "output_csv"]
    for k in required:
        if k not in cfg:
            _logger.error("missing config key: %s", k)
            return {"success": False, "error": f"missing config key: {k}", "artifacts": []}

    input_csv = cfg["input_csv"]
    output_csv = cfg["output_csv"]
    default_model = cfg.get("default_model", "google:default")
    model_config = cfg.get("model_config")
    api_params = cfg.get("api_params")

    # Advanced inference options (all optional)
    inference_kwargs = {}
    for key in [
        "temperature", "top_p", "top_k", "seed", "stop", "batch_size",
        "output_format", "max_tokens", "log_level", "n", "presence_penalty", "frequency_penalty"
    ]:
        if key in cfg:
            inference_kwargs[key] = cfg[key]

    try:
        df = load_csv(input_csv)
    except Exception as e:
        _logger.exception("failed to load input_csv: %s", e)
        return {"success": False, "error": f"failed to load input_csv: {e}", "artifacts": []}

    if "prompt" not in df.columns:
        _logger.error("input CSV missing required 'prompt' column")
        return {"success": False, "error": "input CSV missing required 'prompt' column", "artifacts": []}

    rows = []
    start_ts = time.time()
    for _, r in df.iterrows():
        prompt = r.get("prompt", "")
        model_id = _row_to_model_id(r.to_dict(), default_model)
        model = get_model_instance(model_id, config=model_config, api_params=api_params)
        try:
            # Allow row-level override for advanced options if present in row
            row_kwargs = inference_kwargs.copy()
            for k in row_kwargs.keys():
                if k in r:
                    row_kwargs[k] = r[k]
            resp: ModelResponse = model.generate(prompt, **row_kwargs)
            rows.append({
                "model": resp.model,
                "prompt": resp.prompt,
                "completion": resp.completion,
                "score": resp.score,
                "metadata": json.dumps(resp.metadata or {}),
            })
            _logger.info("generated for model=%s prompt_len=%d", resp.model, len(resp.prompt or ""))
        except Exception as e:
            _logger.exception("generation failed for model=%s: %s", model_id, e)
            rows.append({
                "model": model_id,
                "prompt": prompt,
                "completion": "",
                "score": None,
                "metadata": json.dumps({"error": str(e)}),
            })

    duration = time.time() - start_ts
    out_df = __import__("pandas").DataFrame(rows)
    try:
        write_dataframe(out_df, output_csv, index=False)
        _logger.info("run_from_config inference finished, wrote %s (rows=%d) duration=%.2fs", output_csv, len(out_df), duration)
        return {"success": True, "artifacts": [output_csv]}
    except Exception as e:
        _logger.exception("failed to write output_csv: %s", e)
        return {"success": False, "error": f"failed to write output_csv: {e}", "artifacts": []}


def run(argv: Optional[List[str]] = None) -> int:
    """
    CLI wrapper: --input, --output, [--default-model]
    Returns 0 on success, non-zero on failure.
    """
    import argparse
    parser = argparse.ArgumentParser(prog="inference")
    parser.add_argument("--input", required=True, help="Input CSV path (must include 'prompt')")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--default-model", default="google:default", help="Default model identifier")
    args = parser.parse_args(argv)
    cfg = {"input_csv": args.input, "output_csv": args.output, "default_model": args.default_model}
    res = run_from_config(cfg)
    if res.get("success"):
        return 0
    else:
        _logger.error("%s", res.get("error", "Unknown error"))
        return 3

# Main guard for direct execution
if __name__ == "__main__":
    import sys
    sys.exit(run())