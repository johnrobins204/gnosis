import importlib
import yaml
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional

from gnosis_io import write_provenance
from analytics.experiment_tracker import ExperimentTracker
from analytics.api import AnalyticsAPI

_COMPONENT_MAP = {
    "inference": "src.inference",
    "judge": "src.judge",
    "analyst": "src.analyst",
}


class Orchestrator:
    """Class for orchestrating experiments and analytics."""

    def __init__(self) -> None:
        self.tracker = ExperimentTracker()
        self.analytics = AnalyticsAPI()

    def run_experiment(self, config: Any, data: Any) -> Any:
        """Run an experiment and track its results."""
        import logging
        logger = logging.getLogger("orchestrator")
        fingerprint = self.tracker.generate_fingerprint(config)
        logger.info(f"Experiment fingerprint: {fingerprint}")

        # Check if results already exist
        try:
            results = self.tracker.load_results(fingerprint)
            logger.info("Results already exist. Loading from cache.")
        except FileNotFoundError:
            logger.info("Running experiment...")
            results = self.analytics.calculate_metrics(data)
            self.tracker.save_results(fingerprint, results)

        return results

    def compare_experiments(self, fingerprints: List[str]) -> Dict[str, Any]:
        """Compare results from multiple experiments."""
        results = [self.tracker.load_results(fp) for fp in fingerprints]
        # Example comparison logic (extend as needed)
        comparison = {fp: res for fp, res in zip(fingerprints, results)}
        return comparison


from typing import Callable, Optional

def orchestrate(config_path: str | Path, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, Any]:
    """
    Load YAML config and execute listed steps sequentially.

    YAML expected shape:
      steps:
        - name: step1
          component: inference    # or explicit module: src.inference
          config: {...}
        - name: step2
          component: judge
          config: {...}

    Returns dict: {"success": bool, "artifacts": [...], "errors": [...]}
    """
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    if not config_path.exists():
        return {"success": False, "artifacts": [], "errors": [f"config not found: {config_path}"]}

    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    steps = cfg.get("steps", [])
    artifacts: List[str] = []
    errors: List[str] = []

    total_steps = len(steps)
    for idx, step in enumerate(steps):
        name = step.get("name") or step.get("component") or "<unnamed>"
        comp = step.get("component")
        module_path = step.get("module") or _COMPONENT_MAP.get(comp)
        step_cfg = step.get("config", {})

        # Progress callback before each step
        if progress_callback:
            progress_callback(idx, total_steps, f"Starting step {idx+1}/{total_steps}: {name}")

        if not module_path:
            errors.append(f"{name}: unknown component and no module provided")
            break

        try:
            module = importlib.import_module(module_path)
        except Exception as e:
            errors.append(f"{name}: failed to import module {module_path}: {e}")
            break

        if not hasattr(module, "run_from_config"):
            errors.append(f"{name}: module {module_path} missing run_from_config")
            break

        try:
            result = module.run_from_config(step_cfg)
        except Exception as e:
            errors.append(f"{name}: exception during run_from_config: {e}")
            break

        if not result.get("success"):
            errors.append(f"{name}: step failed: {result.get('error')}")
            break

        step_artifacts = result.get("artifacts", [])
        for art in step_artifacts:
            try:
                meta = write_provenance(art, {"step": name, "component": comp or module_path, "config": step_cfg})
            except Exception:
                meta = None
            artifacts.append(art)
            if meta:
                artifacts.append(meta)
        # Progress callback after each step (optional)
        if progress_callback:
            progress_callback(idx + 1, total_steps, f"Finished step {idx+1}/{total_steps}: {name}")

    return {"success": len(errors) == 0, "artifacts": artifacts, "errors": errors}